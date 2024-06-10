import csv
import itertools
import time

import numpy as np
import sklearn
from joblib import Parallel, delayed
from numpy.random import random
from sklearn.metrics import roc_auc_score
from sklearn.svm import LinearSVC

from radon_machine import svm_radon_machine
from radon_machine.svm_radon_machine import RadonMachineSVM

"""

"""


def orthogonal_vector(v): v - v @ (np.random.rand(v.size)) * np.random.rand(v.size) / np.linalg.norm(v)


def run_hypothesis_flipping_experiment(X_train, y_train, X_test, y_test, split_idx,
                                       num_trials=10,
                                       log_file=None,
                                       max_height=10,
                                       sigmas=None,
                                       outliers=None,
                                       shuffle=True,
                                       base_estimator=LinearSVC, **kwargs):
    if outliers is None:#is expected to be non-decreasing
        outliers = [0, 25, 50, 75, 100, 125, 150, 175, 200, 225, 250, 275, 300, 325, 350, 375, 400, 425, 450, 475,
                    500, 600, 700, 800, 900, 1000]
    if sigmas is None:
        sigmas = [0, 1e-10, 1e-9, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 0.2, 0.5, 1, 2]

    radon_machine = RadonMachineSVM(base_estimator,100, max_height, n_jobs=1, random_state=11905150, **kwargs)
    radon_machine.fit(X_train, y_train)
    original_estimators = radon_machine.get_estimators()
    attacked_estimators = original_estimators.copy()
    attacked_estimator = -attacked_estimators[0, :]  #attacked_estimators[0, :]

    with open(log_file + f".{split_idx}", mode="a", newline='') as file:
        writer = csv.writer(file)
        if split_idx == 0:
            writer.writerow(['split', 'trial', 'sigma', 'outliers', 'height', 'auc', 'params', 'max_condition_number'])
        for sigma in sigmas:
            radon_machine.sigma = sigma
            for num_outliers in outliers:
                attacked_estimators[:num_outliers, :] = attacked_estimator + np.random.randn(*attacked_estimators[:num_outliers, :].shape) * (1e-10 * original_estimators.std(axis=0))
                for trial in range(num_trials):
                    radon_machine.set_estimators(attacked_estimators, shuffle)
                    auc = roc_auc_score(y_test, radon_machine.decision_function(X_test))
                    writer.writerow([split_idx, trial, sigma, num_outliers, radon_machine.height, auc, radon_machine.est_params, max(radon_machine.condition_numbers)])


if __name__ == "__main__":

    arr = np.load("datasets/SUSY.npy")
    X, y = arr[:, 1:], arr[:, 0][:]
    log_file = (f"results/hypothesis_flipping_under_noise_"
                f"{time.strftime("%Y-%m-%d_%H:%M:%S", time.gmtime())}.csv")
    folds = sklearn.model_selection.KFold(n_splits=10, shuffle=True, random_state=11905150)


    with Parallel(10) as parallel:
        x = parallel(delayed(run_hypothesis_flipping_experiment)(
            X[train_idx, :], y[train_idx], X[test_idx, :], y[test_idx], i, log_file=log_file)
                     for (i, (train_idx, test_idx)) in enumerate(folds.split(X)))

    # print(arr)
    exit()

    radon_machine = RadonMachineSVM(100, 3, n_jobs=1)
    print(arr.shape)
    start = time.time()

    radon_machine.fit(X, y)

    original_estimators = radon_machine.get_estimators()

    with open("results/susy_svm_outliers_robustness_noise.csv", mode='w', newline='') as file:
        writer = csv.writer(file)

        # Write the header

        for sigma in [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 0.2, 0.5, 1, 2]:
            attacked_estimators = original_estimators.copy()
            attacked_estimators[0, :] *= -1  #attacked_estimators[0, :]
            radon_machine.sigma = sigma
            # print(attacked_estimators)
            for i in [0, 25, 50, 75, 100, 125, 150, 175, 200, 225, 250, 275, 300, 325, 350, 375, 400, 425, 450, 475,
                      500]:  #range(min(len(original_estimators), 50)):
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
