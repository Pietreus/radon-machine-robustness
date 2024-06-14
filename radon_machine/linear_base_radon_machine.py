import itertools
import math
import random
import signal

import numpy as np
from joblib import Parallel, delayed
from sklearn.datasets import make_classification
from sklearn.linear_model._base import LinearClassifierMixin
from sklearn.svm import LinearSVC

from radon_machine.radon_point.iterated_radon_point import iterated_radon_point


def initializer():
    signal.signal(signal.SIGINT, signal.SIG_IGN)


class RadonMachineLinearBase(LinearClassifierMixin):
    """Radon Machine for using some linear SKLearn models as base learners.

    Upon fitting, internally splits the data and fits multiple linear model instances to each split.
    These are then aggregated using iterated radon point computation.
    """

    def __init__(self, Base_estimator=LinearSVC, min_samples: int = 100, maximum_height: int = 10,
                 n_jobs=2, sigma=1e-8, random_state: int = 11905150, **kwargs):
        """
        :param min_samples:
        :param maximum_height:
        :param n_jobs:
        :param pre_trained: parameter representation of pretrained estimators
        """
        self.Base_estimator = Base_estimator
        self.min_samples = min_samples
        self.maximum_height = maximum_height
        self.n_jobs = n_jobs
        self.sigma = sigma
        self.random_state = random_state


        self.svc_kwargs = kwargs

        self.n_estimators = None
        self.estimators = None
        self._n = None
        self._radon_number = None
        self._fit_estimator = None
        self.height = None
        self._X = None
        self._y = None

    def set_height(self):
        """
        Sets the radon machine to the maximum height, depending on the given limit and the size of the dataset given
        :return:
        """
        self.height = min(self.maximum_height,
                          math.floor(math.log(self._n / self.min_samples) / math.log(self._radon_number)))

    def _train_single_base_estimator(self, data, sample_weight=None):
        if len(data) == 1:
            data = data[0]
        X, y = data
        _learner = self.Base_estimator(random_state=self.random_state, **self.svc_kwargs).fit(X, y, sample_weight)

        return np.append(_learner.coef_, _learner.intercept_)

    def parse_params(self, params):
        estimator = self.Base_estimator(random_state=self.random_state, **self.svc_kwargs)
        estimator.coef_ = np.array([params[:-1]])
        estimator.intercept_ = np.array(params[-1])
        estimator.classes_ = [0, 1]
        return estimator

    def set_estimators(self, estimators, shuffle=True):
        self.estimators = estimators.copy()
        self.aggregate_estimators(shuffle=shuffle)

    def get_estimators(self) -> np.matrix:
        return self.estimators

    def aggregate_estimators(self, shuffle=True):
        # aggregate
        if shuffle:
            np.random.shuffle(self.estimators)
        estimator, self.condition_numbers = iterated_radon_point(self.estimators, self._radon_number, self.height, self.sigma)
        self._fit_estimator = self.parse_params(estimator)
        self.est_params = estimator

    def fit(self, X, y, sample_weight=None):
        """

        :param X: numerical training data
        :param y: binary label variable
        :param sample_weight: will be unused for now
        :return: self after fit
        """
        self._n, self._radon_number = np.shape(X)
        # TODO assumption that parameters are d+1 does not have to be true, but works for linear models
        self._radon_number += 1 + 2

        np.random.seed(self.random_state)
        self.set_height()

        folds = self._radon_number ** self.height
        self.n_estimators = folds

        if self.height == 0:  # fallback in case training data very small
            estimator = self._train_single_base_estimator((X, y), sample_weight=sample_weight)
            self._fit_estimator = self.parse_params(estimator)
            return

        # train base estimators
        # stratified split to ensure each svm learns a reasonable function
        vector = np.arange(len(X))
        np.random.shuffle(vector)
        fold_sizes = np.full(folds, len(X) // folds)
        fold_sizes[:len(X) % folds] += 1
        split_indices = np.cumsum(fold_sizes)[:-1]
        idx = np.split(vector, split_indices)
        with Parallel(self.n_jobs) as parallel:
            x = parallel(delayed(self._train_single_base_estimator)(
                data) for data in itertools.product([(X[split], y[split]) for split in idx]))
        estimators = np.array(x)
        self.estimators = estimators
        self.aggregate_estimators()
        return self

    def predict(self, X):
        # here just use the only estimator that is left from the radon fitting process
        return self._fit_estimator.predict(X)

    def score(self, X, y):
        # self.est_avg_acc = np.mean([self.parse_params(est).score(X, y) for est in self.estimators])
        return self._fit_estimator.score(X, y)

    def set_random_state(self, random_state):
        random.seed(random_state)
        self.random_state = random_state

    def decision_function(self, X):
        return self._fit_estimator.decision_function(X)


if __name__ == "__main__":
    X, y = make_classification(n_features=4, random_state=0)
    classifier = RadonMachineLinearBase()
    classifier.fit(X, y)
    classifier.predict(X)
    pass
