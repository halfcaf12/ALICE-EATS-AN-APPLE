#import xgboost as xgb
import scipy as sp
import numpy as np
import os, sys, time
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from visualizeData import loadFromNPZ, graphCylindricalPoints, getFieldPoints
import argparse
from typing import Dict

def fit_and_score(estimator, X_train, X_test, y_train, y_test):
    """Fit the estimator on the train set and score it on both sets"""
    estimator.fit(X_train, y_train, eval_set=[(X_test, y_test)])
    train_score = estimator.score(X_train, y_train)
    test_score = estimator.score(X_test, y_test)
    return estimator, train_score, test_score

def train_valid_test(arr: np.ndarray, test_percentage, valid_percentage=0.0, random_state=None):
    n = arr.size
    test_n = int(test_percentage * n)
    valid_n = int(valid_percentage * n)
    train_n = n - test_n - valid_n
    if train_n <= 0:
        print('you have left no space for training!')
        sys.exit()
    random_indices = np.random.shuffle(np.linspace(0,n,n))
    shuffle_array = arr[random_indices]
    test = shuffle_array[:test_n]
    valid = shuffle_array[test_n:test_n+valid_n]
    train = shuffle_array[test_n+valid_n:]
    print(train_n,train.size)
    return train, valid, test

random_state = 19
np.random.seed(random_state)
clusters = loadFromNPZ("../clusters")
events = np.unique(clusters["event"])
# maybe later filter by events??
events_increasing_size = [148, 530, 90, 568, 569, 561, 676, 25, 20, 606, 577, 579, 115, 620, 73, 583, 501, 675, 
                          517, 10, 507, 640, 616, 67, 644, 92, 612, 503, 509, 98, 596, 601, 499, 599, 651, 627, 
                          672, 545, 663, 37, 531, 474, 622, 547, 526, 14, 516, 45, 633, 653, 562, 652, 27, 108, 
                          623, 657, 59, 563, 42, 72, 53, 532, 512, 152, 505, 61, 588, 100, 39, 578, 589, 536, 112, 
                          26, 626, 47, 542, 555, 210, 641, 126, 654, 525, 574, 619, 680, 673, 590, 678, 89, 68, 
                          213, 634, 261, 551, 77, 557, 677, 510, 9, 635, 2, 566, 564, 550, 522, 661, 54, 95, 584, 
                          670, 70, 107, 600, 639, 65, 597, 581, 475, 617, 567, 50, 43, 32, 506, 97, 607, 502, 8, 
                          614, 71, 149, 81, 225, 76, 624, 660, 615, 34, 481, 575, 17, 548, 63, 31, 647, 658, 62, 
                          637, 508, 645, 605, 66, 570, 16, 593, 576, 648, 74, 610, 674, 80, 587, 36, 87, 56, 104, 
                          611, 78, 411, 422, 64, 485, 94, 669, 111, 185, 29, 450, 7, 642, 649, 537, 631, 656, 543, 
                          135, 198, 101, 33, 99, 665, 250, 582, 500, 636, 1, 487, 44, 15, 357, 498, 613, 538, 3, 679, 
                          540, 18, 113, 28, 6, 655, 351, 594, 529, 621, 592, 533, 524, 585, 424, 19, 527, 671, 349, 
                          49, 513, 48, 188, 549, 57, 598, 343, 377, 11, 110, 13, 630, 363, 88, 103, 309, 650, 229, 
                          270, 136, 666, 558, 300, 51, 560, 5, 515, 155, 38, 556, 389, 316, 511, 52, 520, 310, 379, 
                          432, 493, 384, 553, 274, 84, 477, 386, 221, 303, 296, 668, 618, 0, 216, 535, 466, 82, 572, 
                          209, 625, 323, 489, 337, 399, 378, 319, 234, 4, 262, 197, 292, 350, 140, 159, 211, 308, 
                          24, 60, 179, 141, 286, 109, 268, 194, 174, 105, 85, 265, 282, 118, 21, 147, 235, 236]
eventidxs = range(0,5)
points, events_used = getFieldPoints(clusters, "event", eventidxs, maxnumevents = 10000, printinfo=2)

# ----- WE GOING CYLINDRICAL IN THIS BIH ------ #
def eighteenOGonLabels(R,T):
    ''' inputs: XY is dim (size,2), Z is dim (size) for labels '''
    assert(T.size == R.size)
    Rs = [0,5,10,17,35,40,45,80,134,250,292.5,375,450] # last is j super big
    labels = np.empty(T.size)
    layers = np.empty(T.size)
    angle = 2*np.pi/18
    rlabel = 0
    for r in range(len(Rs)-1):  # bound radius
        layer_mask = R > Rs[r]
        layer_mask = np.logical_and(layer_mask, R < Rs[r+1])
        for t in range(18):     # bound angle
            angle_mask = T > t*angle
            angle_mask = np.logical_and(angle_mask, T < (t+1)*angle)
            mask = np.logical_and(angle_mask, layer_mask)
            labels[mask] = t + rlabel*18
        layers[layer_mask] = rlabel
        if not layers[layer_mask].size:
            print("layer",r,"is empty!")
        else:
            rlabel += 1
    return labels, layers

R = np.sqrt(points["fV.fY"]**2 + points["fV.fX"]**2)
Xs = np.where(points["fV.fX"], points["fV.fX"], 1e-32)  # dont divide by zero
arctans = np.arctan(points["fV.fY"]/Xs)
# transcribe arctan to actual angles
T0 = np.where(points["fV.fY"] > 0, np.where(Xs > 0, arctans, arctans+np.pi), np.where(Xs > 0, 2*np.pi + arctans, arctans + np.pi))
T = T0
Z = points["fV.fZ"]
labels, layers = eighteenOGonLabels(R,T)
subid = points["fSubdetId"]
# ---- FORMAT: cylindrical coordinates, sector, ring, fSubdetId ---- #
fIds = ("Sector","Ring","fSubdetId")
coords = np.stack((R,T,Z,labels,layers,subid),axis=-1)

idx = 4
print(coords[:10,idx],coords[-10:,idx])
graphCylindricalPoints(coords, idx, markersize=2)

train, valid, test = train_valid_test(coords, 0.125, 0.125, random_state)

X_train = train[:,:3]
X_valid = valid[:,:3]
X_test  = test[:,:3]
Y_train = train[:,idx]
Y_valid = valid[:,idx]
Y_test  = test[:,idx]


sys.exit()


param_linear = {
    "objective": "binary:logistic",
    "booster": "gblinear",
    "alpha": 0.0001,
    "lambda": 1,
}

param_tree = [
    ("max_depth", 3),
    ("objective", "binary:logistic"),
    ("eval_metric", "logloss"),
    ("eval_metric", "error"),
]

# for multi-class regression, use multi:softprob
# verbosity 0 (silent), 1 (warning), 2 (info), and 3 (debug)
# use_rmm uses RAPIDS Memory Manager (RMM) to allocate GPU memory
xgb.config_context(verbosity=2,use_rmm=False)
watchlist = [(test, "eval"), (train, "train")]
num_round = 4
bst = xgb.train(param_tree, train, num_round, watchlist)
preds = bst.predict(test)
labels = test.get_label()

clf = xgb.XGBClassifier(tree_method="hist", early_stopping_rounds=3)


results = {}

for train, test in cv.split(X, y):
    X_train = X[train]
    X_test = X[test]
    y_train = y[train]
    y_test = y[test]
    est, train_score, test_score = fit_and_score(
        clf, X_train, X_test, y_train, y_test
    )
    results[est] = (train_score, test_score)


def f(x: np.ndarray) -> np.ndarray:
    """The function to predict."""
    return x * np.sin(x)


def quantile_loss(args: argparse.Namespace) -> None:
    """Train a quantile regression model."""
    rng = np.random.RandomState(1994)
    # Generate a synthetic dataset for demo, the generate process is from the sklearn
    # example.
    X = np.atleast_2d(rng.uniform(0, 10.0, size=1000)).T
    expected_y = f(X).ravel()

    sigma = 0.5 + X.ravel() / 10.0
    noise = rng.lognormal(sigma=sigma) - np.exp(sigma**2.0 / 2.0)
    y = expected_y + noise

    # Train on 0.05 and 0.95 quantiles. The model is similar to multi-class and
    # multi-target models.
    alpha = np.array([0.05, 0.5, 0.95])
    evals_result: Dict[str, Dict] = {}

    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=rng)
    # We will be using the `hist` tree method, quantile DMatrix can be used to preserve
    # memory.
    # Do not use the `exact` tree method for quantile regression, otherwise the
    # performance might drop.
    Xy = xgb.QuantileDMatrix(X, y)
    # use Xy as a reference
    Xy_test = xgb.QuantileDMatrix(X_test, y_test, ref=Xy)

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
        Xy,
        num_boost_round=32,
        early_stopping_rounds=2,
        # The evaluation result is a weighted average across multiple quantiles.
        evals=[(Xy, "Train"), (Xy_test, "Test")],
        evals_result=evals_result,
    )
    xx = np.atleast_2d(np.linspace(0, 10, 1000)).T
    scores = booster.inplace_predict(xx)
    # dim 1 is the quantiles
    assert scores.shape[0] == xx.shape[0]
    assert scores.shape[1] == alpha.shape[0]

    y_lower = scores[:, 0]  # alpha=0.05
    y_med = scores[:, 1]  # alpha=0.5, median
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
        Xy,
        num_boost_round=32,
        early_stopping_rounds=2,
        evals=[(Xy, "Train"), (Xy_test, "Test")],
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Specify it to enable plotting the outputs.",
    )
    args = parser.parse_args()
    quantile_loss(args)