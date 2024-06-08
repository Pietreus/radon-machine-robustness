import itertools
import time

import numpy as np
import sklearn
from joblib import Parallel, delayed
from numpy.random import random

from radon_machine import svm_radon_machine
from radon_machine.svm_radon_machine import RadonMachineSVM

"""

"""
if __name__ == "__main__":


    arr = np.load("datasets/SUSY.npy")

    folds = sklearn.model_selection.KFold(n_splits=5, shuffle=True, random_state=11905150)


    # print(arr)
    X, y = arr[:950000, 1:], arr[:950000, 0][:]
    radon_machine = RadonMachineSVM(100, 3, n_jobs=10)
    print(arr.shape)
    start = time.time()

    radon_machine.fit(X, y)
    print(radon_machine.score(X, y))
    print(radon_machine.n_estimators)
    print(time.time()-start)
    exit()
    # print(separability_test_perceptron.score(X_test, y_test))

    # noisy_radon_machine = RadonMachineSVM(Perceptron(random_state=11905150), 2, 100, n_jobs=1, sigma=0.1)
    # extreme_radon_machine = RadonMachine(Perceptron(random_state=11905150), 3, 100, n_jobs=1)
    # perceptron = RadonMachine(Perceptron(random_state=11905150), 0, 100, n_jobs=1, maximum_height=0)

    split_sets = []
    for i, (train_index, test_index) in enumerate(folds.split(X)):
        X_train = X[train_index]
        y_train = y[train_index]
        for outliers in outlier_list:
            X_train_o = X_train.copy()
            y_train_o = y_train.copy()
            if type == "move":
                if outliers > 0:
                    X_train_o[0:outliers] = X_train[0:outliers] + \
                                            100 * ((1 - 2 * y_train[0:outliers].T) * separability_test_perceptron.coef_.T).T
            if type == "flip" or type == "flip_close":
                y_train_o[0:outliers] = 1 - y_train_o[0:outliers]
            split_sets.append((X_train_o, y_train_o, X[test_index], y[test_index], outliers, i))

    with Parallel(15) as parallel:
        x = parallel(delayed(repeated_holdout_validation)(
            clone(learner),
            train_set, y_train, X_test, y_test, outliers, i) for
                     learner, (train_set, y_train, X_test, y_test, outliers, i) in
                     itertools.product([radon_machine],
                                       split_sets))

    bag_res = pd.DataFrame([subitem for item in x for subitem in item],
                           columns=["k", "sigma", "n_estimators", "height", "accuracy", "failed_perceptrons",
                                    "duration", "outliers", "fold"])
    bag_res.to_csv(f"processed_data/susy_radon_machine_experiment_{type}.csv",
                   header=True, index=False)
    print("DONE")


