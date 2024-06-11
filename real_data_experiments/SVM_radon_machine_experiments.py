import csv
import itertools
import time
import re
import numpy as np
import sklearn
from joblib import Parallel, delayed
from numpy.random import random
from scipy.io import arff
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.svm import LinearSVC

from radon_machine import svm_radon_machine
from radon_machine.svm_radon_machine import RadonMachineSVM

"""

"""


def orthogonal_vector(v):
    v[0:2, ] = v[1::-1, ]
    v[0] *= -1
    return v


def orthogonal_big_vector(v):
    res = v - v @ (np.random.rand(v.size)) * np.random.rand(v.size) / np.linalg.norm(v)

    return res * 10


def run_hypothesis_flipping_experiment(X_train, y_train, X_test, y_test, split_idx,
                                       num_trials=10,
                                       log_file=None,
                                       max_height=10,
                                       sigmas=None,
                                       outliers=None,
                                       shuffle=True,
                                       base_estimator=LinearSVC, **kwargs):
    if outliers is None:  #is expected to be non-decreasing
        outliers = [0, 25, 50, 75, 100, 125, 150, 175, 200, 225, 250, 275, 300, 325, 350, 375, 400, 425, 450, 475,
                    500, 600, 700, 800, 900, 1000]
    if sigmas is None:
        sigmas = [0, 1e-10, 1e-9, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 0.2, 0.5, 1, 2]

    radon_machine = RadonMachineSVM(base_estimator, 100, max_height, n_jobs=1, random_state=11905150, **kwargs)
    radon_machine.fit(X_train, y_train)
    original_estimators = radon_machine.get_estimators()
    attacked_estimators = original_estimators.copy()
    attacked_estimator = -attacked_estimators[0, :]  #attacked_estimators[0, :]

    with open(log_file + f".{split_idx}", mode="a", newline='') as file:
        writer = csv.writer(file)
        if split_idx == 0:
            writer.writerow(['split', 'trial', 'sigma', 'outliers', 'height', 'auc', 'params', 'max_condition_number'])
        for sigma in sigmas:
            attacked_estimators = original_estimators.copy()
            radon_machine.sigma = sigma
            for num_outliers in outliers:
                attacked_estimators[:num_outliers, :] = attacked_estimator + np.random.randn(
                    *attacked_estimators[:num_outliers, :].shape) * (1e-10 * original_estimators.std(axis=0))
                for trial in range(num_trials):
                    radon_machine.set_estimators(attacked_estimators, shuffle)
                    auc = roc_auc_score(y_test, radon_machine.decision_function(X_test))
                    writer.writerow(
                        [split_idx, trial, sigma, num_outliers, radon_machine.height, auc, radon_machine.est_params,
                         max(radon_machine.condition_numbers)])


def run_hypothesis_90deg_experiment(X_train, y_train, X_test, y_test, split_idx,
                                    num_trials=10,
                                    log_file=None,
                                    max_height=10,
                                    sigmas=None,
                                    outliers=None,
                                    shuffle=True,
                                    base_estimator=LinearSVC, **kwargs):
    if outliers is None:  #is expected to be non-decreasing
        outliers = [0, 25, 50, 75, 100, 125, 150, 175, 200, 225, 250, 275, 300, 325, 350, 375, 400, 425, 450, 475,
                    500, 600, 700, 800, 900, 1000]
    if sigmas is None:
        sigmas = [0, 1e-10, 1e-9, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 0.2, 0.5, 1, 2]

    radon_machine = RadonMachineSVM(base_estimator, 100, max_height, n_jobs=1, random_state=11905150, **kwargs)
    radon_machine.fit(X_train, y_train)
    original_estimators = radon_machine.get_estimators()
    attacked_estimators = original_estimators.copy()
    attacked_estimator = orthogonal_vector(attacked_estimators[0, :])

    with open(log_file + f".{split_idx}", mode="a", newline='') as file:
        writer = csv.writer(file)
        if split_idx == 0:
            writer.writerow(['split', 'trial', 'sigma', 'outliers', 'height', 'auc', 'params', 'max_condition_number'])
        for sigma in sigmas:
            attacked_estimators = original_estimators.copy()
            radon_machine.sigma = sigma
            for num_outliers in outliers:
                attacked_estimators[:num_outliers, :] = attacked_estimator + np.random.randn(
                    *attacked_estimators[:num_outliers, :].shape) * (1e-10 * original_estimators.std(axis=0))
                for trial in range(num_trials):
                    radon_machine.set_estimators(attacked_estimators, shuffle)
                    auc = roc_auc_score(y_test, radon_machine.decision_function(X_test))
                    writer.writerow(
                        [split_idx, trial, sigma, num_outliers, radon_machine.height, auc, radon_machine.est_params,
                         max(radon_machine.condition_numbers)])


def run_hypothesis_orthogonal_big_vec_experiment(X_train, y_train, X_test, y_test, split_idx,
                                                 num_trials=10,
                                                 log_file=None,
                                                 max_height=10,
                                                 sigmas=None,
                                                 outliers=None,
                                                 shuffle=True,
                                                 base_estimator=LinearSVC, **kwargs):
    if outliers is None:  #is expected to be non-decreasing
        outliers = [0, 25, 50, 75, 100, 125, 150, 175, 200, 225, 250, 275, 300, 325, 350, 375, 400, 425, 450, 475,
                    500, 600, 700, 800, 900, 1000]
    if sigmas is None:
        sigmas = [0, 1e-10, 1e-9, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 0.2, 0.5, 1, 2]

    radon_machine = RadonMachineSVM(base_estimator, 100, max_height, n_jobs=1, random_state=11905150, **kwargs)
    radon_machine.fit(X_train, y_train)
    original_estimators = radon_machine.get_estimators()
    attacked_estimators = original_estimators.copy()
    attacked_estimator = orthogonal_big_vector(attacked_estimators[0, :])

    with open(log_file + f".{split_idx}", mode="a", newline='') as file:
        writer = csv.writer(file)
        if split_idx == 0:
            writer.writerow(['split', 'trial', 'sigma', 'outliers', 'height', 'auc', 'params', 'max_condition_number'])
        for sigma in sigmas:
            attacked_estimators = original_estimators.copy()
            radon_machine.sigma = sigma
            for num_outliers in outliers:
                attacked_estimators[:num_outliers, :] = attacked_estimator + np.random.randn(
                    *attacked_estimators[:num_outliers, :].shape) * (1e-10 * original_estimators.std(axis=0))
                for trial in range(num_trials):
                    radon_machine.set_estimators(attacked_estimators, shuffle)
                    auc = roc_auc_score(y_test, radon_machine.decision_function(X_test))
                    writer.writerow(
                        [split_idx, trial, sigma, num_outliers, radon_machine.height, auc, radon_machine.est_params,
                         max(radon_machine.condition_numbers)])



def run_hypothesis_flipping_big_vec_experiment(X_train, y_train, X_test, y_test, split_idx,
                                                 num_trials=10,
                                                 log_file=None,
                                                 max_height=10,
                                                 sigmas=None,
                                                 outliers=None,
                                                 shuffle=True,
                                                 base_estimator=LinearSVC, **kwargs):
    if outliers is None:  #is expected to be non-decreasing
        outliers = [0, 25, 50, 75, 100, 125, 150, 175, 200, 225, 250, 275, 300, 325, 350, 375, 400, 425, 450, 475,
                    500, 600, 700, 800, 900, 1000]
    if sigmas is None:
        sigmas = [0, 1e-10, 1e-9, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 0.2, 0.5, 1, 2]

    radon_machine = RadonMachineSVM(base_estimator, 100, max_height, n_jobs=1, random_state=11905150, **kwargs)
    radon_machine.fit(X_train, y_train)
    original_estimators = radon_machine.get_estimators()
    attacked_estimators = original_estimators.copy()
    attacked_estimator = -attacked_estimators[0, :] * 10

    with open(log_file + f".{split_idx}", mode="a", newline='') as file:
        writer = csv.writer(file)
        if split_idx == 0:
            writer.writerow(['split', 'trial', 'sigma', 'outliers', 'height', 'auc', 'params', 'max_condition_number'])
        for sigma in sigmas:
            attacked_estimators = original_estimators.copy()
            radon_machine.sigma = sigma
            for num_outliers in outliers:
                attacked_estimators[:num_outliers, :] = attacked_estimator + np.random.randn(
                    *attacked_estimators[:num_outliers, :].shape) * (1e-10 * original_estimators.std(axis=0))
                for trial in range(num_trials):
                    radon_machine.set_estimators(attacked_estimators, shuffle)
                    auc = roc_auc_score(y_test, radon_machine.decision_function(X_test))
                    writer.writerow(
                        [split_idx, trial, sigma, num_outliers, radon_machine.height, auc, radon_machine.est_params,
                         max(radon_machine.condition_numbers)])



def call_parallel(function,X,y,log_name,learner):
    log_file = (f"results/{log_name}_"
                f"{time.strftime("%Y-%m-%d_%H:%M:%S", time.gmtime())}.csv")
    folds = sklearn.model_selection.KFold(n_splits=10, shuffle=True, random_state=11905150)

    with Parallel(10) as parallel:
        x = parallel(delayed(function)(
            X[train_idx, :], y[train_idx], X[test_idx, :], y[test_idx], i, log_file=log_file,
            base_estimator=learner)
                     for (i, (train_idx, test_idx)) in enumerate(folds.split(X)))


def preprocess_arff(arff_file):
    with open(arff_file, 'r') as f:
        lines = f.readlines()

    data_lines = lines[13:]
    # Remove {, }, and numbering
    data_lines = [re.sub(r'\{|}|[0-9] ', '', line) for line in data_lines]

    # Write preprocessed data to a new file
    with open('datasets/preprocessed_codrnaNorm.arff', 'w') as f:
        f.writelines(lines[:13])
        f.writelines(data_lines)


if __name__ == "__main__":

    # preprocess_arff("datasets/codrnaNorm.arff")
    # exit()

    data, meta = arff.loadarff('datasets/preprocessed_codrnaNorm.arff')
    arr = [(x1,x2,x3,x4,x5,x6,x7,x8, y == b'1') for (x1,x2,x3,x4,x5,x6,x7,x8,y) in data]
    arr = np.array(arr, dtype=np.float64)
    X, y = arr[:, ::-1], arr[:, -1]

    call_parallel(run_hypothesis_flipping_experiment,X,y,"codrnaNorm_SVM_flip",LinearSVC)
    call_parallel(run_hypothesis_flipping_big_vec_experiment,X,y,"codrnaNorm_SVM_big_flip",LinearSVC)
    call_parallel(run_hypothesis_flipping_experiment,X,y,"codrnaNorm_LogReg_flip",LogisticRegression)
    call_parallel(run_hypothesis_flipping_big_vec_experiment,X,y,"codrnaNorm_LogReg_big_flip",LogisticRegression)


    data, meta = arff.loadarff('datasets/SEA_50.arff')
    arr = [(a, b, c, d == b'groupA') for (a, b, c, d) in data]
    arr = np.array(arr, dtype=np.float64)
    X, y = arr[:, ::-1], arr[:, -1]

    call_parallel(run_hypothesis_flipping_experiment,X,y,"sea50_SVM_flip",LinearSVC)
    call_parallel(run_hypothesis_flipping_big_vec_experiment,X,y,"sea50_SVM_big_flip",LinearSVC)
    call_parallel(run_hypothesis_flipping_experiment,X,y,"sea50_LogReg_flip",LogisticRegression)
    call_parallel(run_hypothesis_flipping_big_vec_experiment,X,y,"sea50_LogReg_big_flip",LogisticRegression)


    arr = np.load("datasets/SUSY.npy")
    X, y = arr[:, 1:], arr[:, 0][:]

    call_parallel(run_hypothesis_flipping_experiment,X,y,"SUSY_SVM_flip",LinearSVC)
    call_parallel(run_hypothesis_flipping_big_vec_experiment,X,y,"SUSY_SVM_big_flip",LinearSVC)
    call_parallel(run_hypothesis_flipping_experiment,X,y,"SUSY_LogReg_flip",LogisticRegression)
    call_parallel(run_hypothesis_flipping_big_vec_experiment,X,y,"SUSY_LogReg_big_flip",LogisticRegression)


    log_file = (f"results/sea50_linearSVC_hypothesis_flipping_under_noise_"
                f"{time.strftime("%Y-%m-%d_%H:%M:%S", time.gmtime())}.csv")
    folds = sklearn.model_selection.KFold(n_splits=10, shuffle=True, random_state=11905150)

    with Parallel(10) as parallel:
        x = parallel(delayed(run_hypothesis_flipping_experiment)(
            X[train_idx, :], y[train_idx], X[test_idx, :], y[test_idx], i, log_file=log_file,
            base_estimator=LinearSVC)
                     for (i, (train_idx, test_idx)) in enumerate(folds.split(X)))

        log_file = (f"results/sea50_linearSVC_hypothesis_big_vec_flipping_under_noise_"
                    f"{time.strftime("%Y-%m-%d_%H:%M:%S", time.gmtime())}.csv")

    with Parallel(10) as parallel:
        x = parallel(delayed(run_hypothesis_flipping_big_vec_experiment)(
            X[train_idx, :], y[train_idx], X[test_idx, :], y[test_idx], i, log_file=log_file,
            base_estimator=LinearSVC)
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
        #
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
