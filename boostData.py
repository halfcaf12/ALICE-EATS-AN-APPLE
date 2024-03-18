import xgboost as xgb
#import scipy as sp
import numpy as np
import os, sys
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from visualizeData import loadFromNPZ, timeIt
from visualizeData import graphCylindricalPoints, getFieldPoints, plotRTheta
import argparse
from typing import Dict

PLOT_SIZE = (12,6)

def fit_and_score(estimator, X_train, X_test, y_train, y_test):
    """Fit the estimator on the train set and score it on both sets"""
    estimator.fit(X_train, y_train, eval_set=[(X_test, y_test)])
    train_score = estimator.score(X_train, y_train)
    test_score = estimator.score(X_test, y_test)
    return estimator, train_score, test_score

def train_valid_test(arr: np.ndarray, test_percentage, valid_percentage=0.0, random_state=None):
    n = arr.shape[0]
    test_n = int(test_percentage * n)
    valid_n = int(valid_percentage * n)
    train_n = n - test_n - valid_n
    if train_n <= 0:
        print('you have left no space for training!')
        sys.exit()
    indices = np.linspace(0,n,n,dtype=int,endpoint=False)
    np.random.shuffle(indices)
    shuffle_array = arr[indices]
    test = shuffle_array[:test_n]
    valid = shuffle_array[test_n:test_n+valid_n]
    train = shuffle_array[test_n+valid_n:]
    print("train:",train_n,train.shape,"valid",valid_n,valid.shape,"test",test_n,test.shape)
    return (train, valid, test)

def within_eps(act, pred, eps):
    return np.logical_and(pred <= (act + eps), pred >= (act - eps))
def outside_eps(act, pred, eps):
    return np.logical_or(pred > (act + eps), pred < (act - eps))

def printAccuracy(actual, predicted, eps=0):  # predicted are floats for fSubdetId
    """ string of accuracy for each label from int predictions, 3 decimal points after percent """
    if eps:
        func = lambda act, pred: within_eps(act, pred, eps)
    else:
        func = lambda act, pred: act == pred
    ret = ["tot:%.3f" % (100*np.sum(func(actual, predicted))/actual.shape[0])]
    labels = np.unique(actual)
    for label in labels:
        indices = actual == label
        ac = actual[indices]
        pr = predicted[indices]
        perc = 100*np.sum(func(ac,pr))/ac.shape[0]
        if perc:
            perc = "%.3f" % perc
            if perc[-3:] == "000":
                perc = perc[:-4] # exclude decimal
            ret.append("%d:%s" % (label, perc))
        else:  # nothing predicted
            continue
            ret.append("~%d" % label)
    return ', '.join(ret)

def eighteenOGonLabels(R,T):
    ''' inputs: XY is dim (size,2), Z is dim (size) for labels 
        doesn't label any empty sectors of data '''
    assert(T.size == R.size)
    Rs = [0,5,10,17,35,40,45,80,134,250,292.5,375,450] # last is j super big
    sectors = np.empty(T.size)
    rings = np.empty(T.size)
    angle = 2*np.pi/18
    rlabel = 0; slabel = 0
    for r in range(len(Rs)-1):  # bound radius
        layer_mask = R > Rs[r]
        layer_mask = np.logical_and(layer_mask, R < Rs[r+1])
        for t in range(18):     # bound angle
            angle_mask = T > t*angle
            angle_mask = np.logical_and(angle_mask, T < (t+1)*angle)
            mask = np.logical_and(angle_mask, layer_mask)
            sectors[mask] = slabel
            if sectors[mask].size:
                slabel += 1
        rings[layer_mask] = rlabel
        if rings[layer_mask].size: rlabel += 1
    return rings, sectors

# ---- GET AND FORMAT DATA ---- #
random_state = 19
np.random.seed(random_state)
# maybe later filter by events??

# events ordered by size: index of event, event, number of clusters
events_ordered = loadFromNPZ("events_increasing_size")

print("navigating event orderings, largest to smallest")
i = events_ordered.shape[0] - 1
while 1:
    print(events_ordered[i])
    t = input("type e to exit:").lower().rstrip()
    if t == 'e':
        break
    i -= 1
    if i < 0: break

# ---- FORMAT: cylindrical coordinates, sector, ring, fSubdetId ---- #
@timeIt
def getCoords(eventidxs=[], maxnumevents= 10000):
    clusters = loadFromNPZ("../clusters")
    #events = np.unique(clusters["event"])
    points, events_used = getFieldPoints(clusters, "event", eventidxs, maxnumevents, printinfo=0)
    # ----- WE GOING CYLINDRICAL IN THIS BIH ------ #
    R = np.sqrt(points["fV.fY"]**2 + points["fV.fX"]**2)
    Xs = np.where(points["fV.fX"], points["fV.fX"], 1e-32)  # dont divide by zero
    arctans = np.arctan(points["fV.fY"]/Xs)
    # transcribe arctan to actual angles
    T = np.where(points["fV.fY"] > 0, np.where(Xs > 0, arctans, arctans+np.pi), np.where(Xs > 0, 2*np.pi + arctans, arctans + np.pi))
    Z = points["fV.fZ"]
    rings, sectors = eighteenOGonLabels(R,T)
    subid = points["fSubdetId"]
    # can cast to object type to preserve int-ness of the labels, but 
    # instead we will just change the types of the labels accordingly 
    # when we use them later
    coords = np.stack((R,T,Z,rings,sectors,subid),axis=-1)
    print("coords have shape",coords.shape,"and of type",coords.dtype)
    print("\tusing events",', '.join([str(ev) for ev in events_used]))
    return coords, events_used

def classifyModel(fieldidx: 0|1|2, x_dim: 1|2|3, tvt_split=0.125, ev_idxs=[], maxnumevents=10000, 
                  save_model_name="", makePlotly=True, makeLinear=False, showPyplots=False, plotPolar=True):
    '''
    - fieldidx indexes whether sector, ring, or fsubdetid are being classified
    - tvt_split: partitions valid and test sets as tvt_split percentage of data
    - x_dim is whether to use just r, r + theta, or r + theta + z as input features
    - makePlotly: whether to graph labels in plotly(recommend). will make pyplot regardless
    '''
    if fieldidx < 0 or fieldidx > 2: fieldidx = 0
    if x_dim < 1: x_dim = 2
    fieldsMode = x_dim > 2   # predict one field from the other two fields?

    fIds = ("Ring","Sector","fSubdetId")
    fId = fIds[fieldidx]
    idx = fieldidx + 3
    print(f"now classifying {fId}")
    
    # assumes you do not specify same event twice
    ev_idxs_to_make = []; numused = 0  # events_ordered has (index, event, size)
    for evidx in ev_idxs:
        i = 0
        if evidx.isdigit():
            ev_idxs_to_make.append(int(evidx))
            for ordered_ev in events_ordered:
                if ordered_ev[0] == evidx:
                    numused += ordered_ev[2]
                    break
        elif evidx[0] == "l":
            num_l = 0
            if len(evidx) > 1:
                num_l = evidx[1:]
                if num_l.isdigit():
                    num_l = int(num_l)
            for ordered_ev in events_ordered[::-1]:
                if ordered_ev[2] + numused <= maxnumevents:
                    if i == num_l:
                        ev_idxs_to_make.append(ordered_ev[0])
                        numused += ordered_ev[2]
                        break
                    i += 1
        elif evidx[0] == "s":
            num_include = 1
            if len(ev_idxs[0]) > 1:
                num_include = ev_idxs[0][1:]
                if num_include.isdigit():
                    num_include = int(num_include)
            for ordered_ev in events_ordered:
                if i >= num_include or ordered_ev[2] + numused >= maxnumevents: break
                ev_idxs_to_make.append(ordered_ev[0])
                numused += ordered_ev[2]
                i += 1
    print("making events:",ev_idxs_to_make)
    coords, events_used = getCoords(ev_idxs_to_make, maxnumevents)  # coords: float array, events_used: int list
    eventnames = ', '.join([str(ev) for ev in events_used])
    linearstring = ("(Linear) " if makeLinear else "")
    if makePlotly: graphCylindricalPoints(coords, idx, title="All "+str(coords.shape[0])+" Points: "+eventnames, markersize=6)
    if x_dim == 2:  # graph R, Theta
        plotRTheta(coords, idx, "Initial points", radius_bounds=(0,0), showplot=showPyplots, polar=plotPolar)

    X = coords[:,:x_dim]
    Y = coords[:,idx].astype(int)

    print("x shape:",X.shape,"y shape:",Y.shape)
    train, valid, test = train_valid_test(coords, tvt_split, tvt_split, random_state)

    # predict for field idx from x spatial things
    X_train = train[:,:x_dim]
    X_valid = valid[:,:x_dim]
    X_test  = test[:,:x_dim]
    Y_train = train[:,idx].astype(int)
    Y_valid = valid[:,idx].astype(int)
    Y_test  = test[:,idx].astype(int)

    print("train",X_train.shape,Y_train.shape)
    print("valid",X_valid.shape,Y_valid.shape)
    print("test",X_test.shape,Y_test.shape)

    kClasses = np.unique(Y).size

    M = xgb.DMatrix(X, Y)
    M_train = xgb.DMatrix(X_train, Y_train)
    M_valid = xgb.DMatrix(X_valid, Y_valid)

    # -------- PPOTENTIAL TREE PARAMETERS
    param_linear = {"objective": "multi:softmax", 
                    "booster": "gblinear",
                    "alpha": 0.0,   # L1 regular ization
                    "lambda": 0.001, # L2 regularization
                    "updater": "shotgun",  # or coord_descent
                    #"feature_selector": "thrifty",
                    "num_class": kClasses
                    }
    kRounds = 100  # number of rounds to make leaves??? maybe??
    maxDepth = kClasses

    param_tree = {
        "max_depth": maxDepth,
        "min_child_weight": 0.1,
        "max_leaves": 0,  # no max
        "max_bin": 512,  # max bins for continuous features
        "objective": "multi:softmax",  
        # multi:softmax, #  only outputs one probability of that group, this does prob for all groups, might b interesting
        "num_class": kClasses, # for mult-class
        "eval_metric": "mlogloss", # or merror, auc
        #"eval_metric": "merror",
        "lambda": 0.0001,      # L2 regularization term
        "tree_method": "hist", #fastest greedy algo
        "min_split_loss": 0,
        "learning_rate": 0.3,
    }

    # switch out of multi-class classification for fSubdetId
    if fieldidx == 2:
        param_tree["objective"] = "reg:squarederror"
        param_tree["eval_metric"] = "rmse"
        del param_tree["num_class"]
        param_linear["objective"] = "reg:squarederror"
        del param_linear["num_class"]
        eps = 5
    else:
        eps = 0

    # verbosity 0 (silent), 1 (warning), 2 (info), and 3 (debug)
    # use_rmm uses RAPIDS Memory Manager (RMM) to allocate GPU memory
    xgb.config_context(verbosity=2,use_rmm=False)
    watchlist = [(M_train, "train"),(M_valid, "valid")]
    # early-stopping rounds determines when validation set has stopped improving
    @timeIt
    def trainingXGB(makeLinear):
        if makeLinear:
            return xgb.train(param_linear, M_train, kRounds, evals=watchlist, early_stopping_rounds=10)
        return xgb.train(param_tree, M_train, kRounds, evals=watchlist, early_stopping_rounds=10)
    
    bst = trainingXGB(makeLinear)
    all_pred   = bst.predict(xgb.DMatrix(X), iteration_range=(0, bst.best_iteration + 1))
    test_pred  = bst.predict(xgb.DMatrix(X_test),  iteration_range=(0, bst.best_iteration + 1))
    train_pred = bst.predict(xgb.DMatrix(X_train), iteration_range=(0, bst.best_iteration + 1))
    valid_pred = bst.predict(xgb.DMatrix(X_valid), iteration_range=(0, bst.best_iteration + 1))

    print("accuracies:\n- train; ",printAccuracy(Y_train,train_pred,2),
          "\n- valid; ",printAccuracy(Y_valid,valid_pred,2),
          "\n- test; ",printAccuracy(Y_test,test_pred,2))

    all_differences = all_pred - coords[:,idx]

    print(f'avg:{np.mean(all_differences)},std:{np.std(all_differences)},max:{np.max(all_differences)},min:{np.min(all_differences)}')
    for eps in np.linspace(0.001,10,100):
        n = np.sum(within_eps(coords[:,idx], all_pred,eps))
        print("%d within %.3f: %.1f" % (n,eps,100*n/coords.shape[0])+"%")

    if save_model_name:
        print("saving model...")
        bst.save_model(save_model_name+".ubj")
        # dump model and featuremap
        for ending in ('_raw.txt','_featmap.txt'):
            save_model_path = save_model_name+ending
            if not os.path.isfile(save_model_path):
                f = open(save_model_path, 'w')
                f.close()
        #bst.dump_model(save_model_name+'_raw.txt', save_model_name+'_featmap.txt')
    # can later load model again
    #bst_new = xgb.Booster({'nthread': 4})  # init model
    #bst_new.load_model('0001.model')  # load data
    
    if not makeLinear and maxnumevents < 100000:
        fig, axs = plt.subplots(2,1,figsize=(12,8))
        xgb.plot_importance(bst, ax=axs[0])
        xgb.plot_tree(bst, num_trees=2, ax=axs[1])
        fig.savefig("xgb_fig.pdf")
        if showPyplots: fig.show()

    # ---- PLOT TEST ---- #
    print("plotting...")
    plotTestDetail = False
    if plotTestDetail:
        new_test = np.copy(test)
        new_test[:,idx] = test_pred
    indices = within_eps(coords[:,idx], all_pred, eps)
    new_all = coords[indices]
    print("creating to predict...")
    diffs = all_pred[indices] - new_all[:,idx]
    mostunder = min(diffs); mostover = max(diffs)
    print(f"underpredicted by up to {mostunder} & overpredicted up to {mostover}")
    new_all[:,idx] = diffs
    if x_dim == 2:  # plot points in R, Theta
        plotRTheta(new_all, idx, "Predicted Points", radius_bounds=(0,0), showplot=showPyplots, polar=plotPolar)
        if plotTestDetail:
            plotRTheta(test, idx, "Test-Real", showplot=showPyplots)
            plotRTheta(new_test, idx, "Test-Predicted")
            plotRTheta(new_test, idx, "Test-Predicted Small Radius", radius_bounds=(0,45), showplot=showPyplots, polar=plotPolar)
            plotRTheta(new_test, idx, "Test-Predicted Large Radius", radius_bounds=(80,0), showplot=showPyplots, polar=plotPolar)
        else: # detail all points
            plotRTheta(new_all, idx, "Predicted Small Radius", radius_bounds=(0,45), showplot=showPyplots, polar=plotPolar)
            plotRTheta(new_all, idx, "Predicted Large Radius", radius_bounds=(80,0), showplot=showPyplots, polar=plotPolar)
    if makePlotly:
        print(f"plotting diffs... {new_all.shape[0]}")
        tit = linearstring+"Difference + %.2f: " % mostunder+eventnames
        graphCylindricalPoints(new_all, idx, markersize=6, title=tit)
        if plotTestDetail:
            graphCylindricalPoints(test, idx, markersize=6, title="Test Real: "+eventnames)
            graphCylindricalPoints(new_test, idx, markersize=6,  title=linearstring+"Test Predicted: "+eventnames)


def quantile_loss(args: argparse.Namespace, X, Y) -> None:
    """Train a quantile regression model."""
    # Train on 0.05 and 0.95 quantiles. The model is similar to multi-class and
    # multi-target models.
    alpha = np.array([0.05, 0.5, 0.95])
    evals_result: Dict[str, Dict] = {}

    X_train, X_test, y_train, y_test = train_test_split(X, Y, random_state=random_state)
    # We will be using the `hist` tree method, quantile DMatrix can be used to preserve
    # memory.
    # Do not use the `exact` tree method for quantile regression, otherwise the
    # performance might drop.
    XY = xgb.QuantileDMatrix(X, Y)
    # use Xy as a reference
    XY_train = xgb.QuantileDMatrix(X_train, Y_train, ref=XY)
    XY_test = xgb.QuantileDMatrix(X_test, y_test, ref=XY)

    booster = xgb.train(
        {
            # Use the quantile objective function.
            "objective": "reg:quantileerror",
            "tree_method": "hist",
            "quantile_alpha": alpha,
            # Let's try not to overfit.
            "learning_rate": 0.04,
            "max_depth": 5,
        },
        XY,
        num_boost_round=32,
        early_stopping_rounds=2,
        # The evaluation result is a weighted average across multiple quantiles.
        evals=[(XY, "Train"), (XY_test, "Test")],
        evals_result=evals_result,
    )
    xx = np.atleast_2d(np.linspace(0, 10, 1000)).T
    scores = booster.inplace_predict(xx)
    # dim 1 is the quantiles
    assert scores.shape[0] == xx.shape[0]
    assert scores.shape[1] == alpha.shape[0]

    y_lower = scores[:, 0]  # alpha=0.05
    y_med   = scores[:, 1]  # alpha=0.5, median
    y_upper = scores[:, 2]  # alpha=0.95

    # Train a mse model for comparison
    booster = xgb.train(
        {
            "objective": "reg:squarederror",
            "tree_method": "hist",
            # Let's try not to overfit.
            "learning_rate": 0.04,
            "max_depth": 5,
        },
        XY,
        num_boost_round=32,
        early_stopping_rounds=2,
        evals=[(XY_train, "Train"), (XY_test, "Test")],
        evals_result=evals_result,
    )
    xx = np.atleast_2d(np.linspace(0, 10, 1000)).T
    y_pred = booster.inplace_predict(xx)

    if args.plot:
        fig = plt.figure(figsize=(10, 10))
        plt.plot(xx, f(xx), "g:", linewidth=3, label=r"$f(x) = x\,\sin(x)$")
        plt.plot(X_test, y_test, "b.", markersize=10, label="Test observations")
        plt.plot(xx, y_med, "r-", label="Predicted median")
        plt.plot(xx, y_pred, "m-", label="Predicted mean")
        plt.plot(xx, y_upper, "k-")
        plt.plot(xx, y_lower, "k-")
        plt.fill_between(
            xx.ravel(), y_lower, y_upper, alpha=0.4, label="Predicted 90% interval"
        )
        plt.xlabel("$x$")
        plt.ylabel("$f(x)$")
        plt.ylim(-10, 25)
        plt.legend(loc="upper left")
        plt.show()

# custom multi-class objective function
def softmax(x):
    '''Softmax function with x as input vector.'''
    e = np.exp(x)
    return e / np.sum(e)

def softprob_obj(predt: np.ndarray, data: xgb.DMatrix):
    '''Loss function.  Computing the gradient and approximated hessian (diagonal).
    Reimplements the `multi:softprob` inside XGBoost.

    '''
    labels = data.get_label()
    if data.get_weight().size == 0:
        # Use 1 as weight if we don't have custom weight.
        weights = np.ones((kRows, 1), dtype=float)
    else:
        weights = data.get_weight()

    # The prediction is of shape (rows, classes), each element in a row
    # represents a raw prediction (leaf weight, hasn't gone through softmax
    # yet).  In XGBoost 1.0.0, the prediction is transformed by a softmax
    # function, fixed in later versions.
    assert predt.shape == (kRows, kClasses)

    grad = np.zeros((kRows, kClasses), dtype=float)
    hess = np.zeros((kRows, kClasses), dtype=float)

    eps = 1e-6

    # compute the gradient and hessian, slow iterations in Python, only
    # suitable for demo.  Also the one in native XGBoost core is more robust to
    # numeric overflow as we don't do anything to mitigate the `exp` in
    # `softmax` here.
    for r in range(predt.shape[0]):
        target = labels[r]
        p = softmax(predt[r, :])
        for c in range(predt.shape[1]):
            assert target >= 0 or target <= kClasses
            g = p[c] - 1.0 if c == target else p[c]
            g = g * weights[r]
            h = max((2.0 * p[c] * (1.0 - p[c]) * weights[r]).item(), eps)
            grad[r, c] = g
            hess[r, c] = h

    # Right now (XGBoost 1.0.0), reshaping is necessary
    grad = grad.reshape((kRows * kClasses, 1))
    hess = hess.reshape((kRows * kClasses, 1))
    return grad, hess

def predict(booster: xgb.Booster, X):
    '''A customized prediction function that converts raw prediction to
    target class.

    '''
    # Output margin means we want to obtain the raw prediction obtained from
    # tree leaf weight.
    predt = booster.predict(X, output_margin=True)
    out = np.zeros(kRows)
    for r in range(predt.shape[0]):
        # the class with maximum prob (not strictly prob as it haven't gone
        # through softmax yet so it doesn't sum to 1, but result is the same
        # for argmax).
        i = np.argmax(predt[r])
        out[r] = i
    return out

def merror(predt: np.ndarray, dtrain: xgb.DMatrix):
    y = dtrain.get_label()
    # Like custom objective, the predt is untransformed leaf weight when custom objective
    # is provided.

    # With the use of `custom_metric` parameter in train function, custom metric receives
    # raw input only when custom objective is also being used.  Otherwise custom metric
    # will receive transformed prediction.
    assert predt.shape == (kRows, kClasses)
    out = np.zeros(kRows)
    for r in range(predt.shape[0]):
        i = np.argmax(predt[r])
        out[r] = i

    assert y.shape == out.shape

    errors = np.zeros(kRows)
    errors[y != out] = 1.0
    return 'PyMError', np.sum(errors) / kRows

def plot_history(custom_results, native_results):
    fig, axs = plt.subplots(2, 1)
    ax0 = axs[0]
    ax1 = axs[1]

    pymerror = custom_results['train']['PyMError']
    merror = native_results['train']['merror']

    x = np.arange(0, kRounds, 1)
    ax0.plot(x, pymerror, label='Custom objective')
    ax0.legend()
    ax1.plot(x, merror, label='multi:softmax')
    ax1.legend()

    plt.show()

def custom_objective(args):
    custom_results = {}
    # Use our custom objective function
    booster_custom = xgb.train({'num_class': kClasses,
                                'disable_default_eval_metric': True},
                               M_train,
                               num_boost_round=kRounds,
                               obj=softprob_obj,
                               custom_metric=merror,
                               evals_result=custom_results,
                               evals=[(M_train, 'train'),(M_valid,'valid')])

    predt_custom = predict(booster_custom, M_test)

    native_results = {}
    # Use the same objective function defined in XGBoost.
    booster_native = xgb.train({'num_class': kClasses,
                                "objective": "multi:softmax",
                                'eval_metric': 'merror'},
                               M_train,
                               num_boost_round=kRounds,
                               evals_result=native_results,
                               evals=[(M_train, 'train'),(M_valid,'valid')])
    predt_native = booster_native.predict(M_test)

    # We are reimplementing the loss function in XGBoost, so it should
    # be the same for normal cases.
    assert np.all(predt_custom == predt_native)
    np.testing.assert_allclose(custom_results['train']['PyMError'],
                               native_results['train']['merror'])

    if args.plot != 0:
        plot_history(custom_results, native_results)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Arguments classifying XGBoost on cluster data')
    parser.add_argument('--np', dest="np", default=True, 
                        action="store_false", help='turn off plotly attempt')
    parser.add_argument('--nps', '--nopyplot', '--nompl', dest="nps", default=True, 
                        action="store_false", help='turn off matplotlib showing plots')
    parser.add_argument('--field', "--f", dest="f", type=int, default=0, 
                        help='index of field to fit in (sector#, ring#, id#)')
    parser.add_argument('--dim', dest="xdim", type=int, default=2, 
                        help='dimensions of x to use out of (radius, theta, z) if 4,5,6 will index into a fields instead')
    parser.add_argument('--tvt', "--split", dest="split", type=float, default=0.125, 
                        help='percentage to partition testing and validation sets.')
    parser.add_argument('--ev', "--events", dest="evs", default=[], nargs = "+",
                        help='event indices to use (0-317, in increasing number of events). specs of "l3" and "s2" will add 3rd largest w.r.t maxnevs, 2nd smallest event resp.')
    parser.add_argument('--maxn', "--maxnevs", "--maxevs", dest="maxnevs", type=int, default=50000,
                        help='maximum number of events to pull')
    parser.add_argument('--save', "--savename", "--name", dest="sname", default="",
                        help='save created model in a file format with given name')
    parser.add_argument('--linear', "--l", dest="makelinear", default=False, action="store_true",
                        help='Make linear booster instead of tree')
    args = parser.parse_args()
    #custom_objective(args)
    #quantile_loss(args)
    classifyModel(args.f, args.xdim, tvt_split=args.split, ev_idxs=args.evs, maxnumevents=args.maxnevs, 
                  save_model_name=args.sname, makePlotly=args.np, makeLinear=args.makelinear, 
                  showPyplots=args.nps, plotPolar=False)