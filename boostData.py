#import xgboost as xgb
import scipy as sp
import numpy as np
import os, sys, time
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from visualizeData import loadFromNPZ
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
eventidx = 0
points = clusters[clusters["event"] == events[eventidx]]

# ----- WE GOING CYLINDRICAL IN THIS BIH ------ #
# care about x-y points
# want to model 
T = np.arctan(points["fV.fY"]/points["fV.fX"])
R = np.sqrt(points["fV.fY"]**2 + points["fV.fX"]**2)
Z = points["fV.fZ"]
coords = np.vstack((T,R,Z),axis=-1)
train, valid, test = train_valid_test(XY, 0.125, 0.125, random_state)

#TODO: split inner points via ring...
#TODO: 6 inner rings of fDetId==0
# RELABEL fSubdetId
# relabel to 0,1,2,3,4,5
# 2 rings of fDetId==1, 
# relabel to 10,11
# 1 ring of fDetId==2, fLabel[3]>0 indicates energy level??
# 1 ring of fDetId==3
fIds = ("fDetId","fSubdetId","fLabel[3]")
Z = points["fDetId"]

# 18 ring sections
#half angle point 
radius1 = (131.75 + 135.12)/2
radius1 = 85.21
pt1 = np.array((13.38,85.14))
pt2 = np.array((16.86,84.58))
midpt = (pt1+pt2)/2

hypotenuse = np.linalg.norm(midpt)
halfangle = np.arccos(radius1/hypotenuse)
print(radius1, hypotenuse, halfangle*2, 2*np.pi/18)

def eighteenOGonLabel(XY,Z):
    ''' inputs: XY is dim (size,2), Z is dim (size) for labels '''
    angle = 2*np.pi/18
    thetas = np.arctan(XY[:,1] / XY[:,0])




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