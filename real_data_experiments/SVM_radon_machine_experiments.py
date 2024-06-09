import csv
import itertools
import time

import numpy as np
import sklearn
from joblib import Parallel, delayed
from numpy.random import random
from sklearn.metrics import roc_auc_score

from radon_machine import svm_radon_machine
from radon_machine.svm_radon_machine import RadonMachineSVM

"""

"""
if __name__ == "__main__":

    arr = np.load("datasets/SUSY.npy")

    folds = sklearn.model_selection.KFold(n_splits=5, shuffle=True, random_state=11905150)

    # print(arr)
    X, y = arr[:, 1:], arr[:, 0][:]
    radon_machine = RadonMachineSVM(100, 3, n_jobs=1)
    print(arr.shape)
    start = time.time()

    radon_machine.fit(X, y)

    original_estimators = radon_machine.get_estimators()
    orthogonal_vector = lambda v: v - v @ (np.random.rand(v.size)) * np.random.rand(v.size) / np.linalg.norm(v)
    with open("results/susy_svm_outliers_robustness_noise.csv", mode='w', newline='') as file:
        writer = csv.writer(file)

    # Write the header
        writer.writerow(['outliers', 'auc', 'trials', 'sigma'])

        for sigma in [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 0.2, 0.5, 1, 2]:
            attacked_estimators = original_estimators.copy()
            attacked_estimators[0, :] *= -1#attacked_estimators[0, :]
            radon_machine.sigma = sigma
            # print(attacked_estimators)
            for i in [0,25,50,75,100,125,150,175,200,225,250,275,300,325,350,375,400,425,450,475,500]:  #range(min(len(original_estimators), 50)):
                attacked_estimators[0:i, :] = attacked_estimators[0, :]
                for trial in range(10):
                    radon_machine.set_estimators(attacked_estimators, False)
                    auc = roc_auc_score(y, radon_machine.decision_function(X))
                    print(f"outliers {i}, accuracy {auc}, trials {trial}, sigma {radon_machine.sigma}")
                    writer.writerow([i, auc, trial, sigma])
    # print(attacked_estimators)
    print(radon_machine.n_estimators)
    print(time.time() - start)
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
                                            100 * ((1 - 2 * y_train[
                                                            0:outliers].T) * separability_test_perceptron.coef_.T).T
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
