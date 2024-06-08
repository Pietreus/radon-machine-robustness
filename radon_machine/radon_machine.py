import itertools
import math
import random
import signal

import numpy as np
from joblib import Parallel, delayed
from sklearn.base import clone
from sklearn.linear_model._stochastic_gradient import BaseSGDClassifier, BaseSGD
from sklearn.model_selection import StratifiedKFold

from radon_machine.radon_point.fast_radon_points import radon_point3
from radon_machine.radon_point.iterated_radon_point import iterated_radon_point, iterated_radon3_point


def radon_selection(pts, rad):
    med = np.median(pts, axis=0)
    mini = -1
    mindist = np.inf
    for i in range(len(rad)):
        dist = np.linalg.norm(rad[i] - med)
        if mindist > dist:
            mindist = dist
            mini = i
    return rad[mini]


def radon_aggregate(pts, r):
    radons = []
    # TODO jitter to "assure" general position
    pts += np.random.randn(len(pts), len(pts.T)) * 1e-5
    for i in range(0, len(pts), r):
        rad = radon_point3(pts[i:(i + r)])
        radons.append(radon_selection(pts[i:(i + r)], rad))
    return np.array(radons)


def par_radon_aggregate(pts, r, n_jobs=1):
    with Parallel(n_jobs=n_jobs) as parallel:
        radons = parallel(delayed(radon_selection)(
            pts[i:(i + r)],
            radon_point3(pts[i:(i + r)])
        ) for i in range(0, len(pts), r))
    return np.array(radons)


def _train_single_learner(learner, data):
    X, y = data
    _learner = learner
    _learner = _learner.fit(X, y)

    return np.append(_learner.coef_, _learner.intercept_), _learner.n_iter_ == _learner.max_iter


def initializer():
    signal.signal(signal.SIGINT, signal.SIG_IGN)


class RadonMachine(BaseSGD):
    def __init__(self, base_learner: BaseSGDClassifier, k: int, min_samples: int = 100, maximum_height: int = 100,
                 n_jobs=2, pre_trained=None, sigma=1e-5):
        """

        :param base_learner:
        :param k:
        :param min_samples:
        :param maximum_height:
        :param n_jobs:
        :param pre_trained: parameter representation of pretrained estimators
        """
        self.n_estimators = None
        self.min_samples = min_samples
        self._n = None
        self._d = None
        self._fit_estimator = None
        self.k = k
        self.height = None
        self.maximum_height = maximum_height
        self.base_learner = base_learner
        self.n_jobs = n_jobs
        self._X = None
        self._y = None
        self.pre_trained = pre_trained
        self.sigma = sigma
        self.random_state = 11905150

    def parse_params(self, params):
        estimator = self.base_learner
        estimator.coef_ = np.array([params[:-1]])
        estimator.intercept_ = np.array(params[-1])
        estimator.classes_ = [0, 1]
        return estimator

    def fit(self, X, y):
        # TODO assumption that parameters are d+1 does not have to be true
        self._n, self._d = np.shape(X)
        self._d += 1

        np.random.seed(self.random_state)
        # self._X = X + self._d
        # self._y = y

        self.height = min(self.maximum_height,
                          math.floor(math.log(self._n / self.min_samples) / math.log(self._d + self.k)))

        if self.pre_trained is not None:
            folds = (self._d + self.k) ** self.height - len(self.pre_trained)
        else:
            folds = (self._d + self.k) ** self.height
        self.n_estimators = folds
        if self.height == 0:
            estimator, self.est_conv_fail = _train_single_learner(self.base_learner, (X, y))
            self._fit_estimator = self.parse_params(estimator)
            return

        skf = StratifiedKFold(folds, shuffle=True, random_state=self.random_state)
        with Parallel(self.n_jobs) as parallel:
            x = parallel(delayed(_train_single_learner)(
                clone(learner),
                data) for learner, data in
                                  itertools.product([self.base_learner],
                                                    [(X[test], y[test]) for _, test in skf.split(X, y)]))
        estimators, fails = zip(*x)
        self.est_conv_fail = sum(fails)
        estimators = np.array(estimators)
        if self.pre_trained is not None:
            estimators = np.vstack((estimators, self.pre_trained))
        np.random.shuffle(estimators)

        if self.k == 2:
            estimator = iterated_radon_point(estimators, self._d + self.k, self.height, self.sigma)
        else:
            estimator = iterated_radon3_point(estimators, self._d + self.k, self.height, self.sigma)
        self._fit_estimator = self.parse_params(estimator)



    def predict(self, X):
        # here just use the only estimator that is left from the radon fitting process
        return self._fit_estimator.predict(X)

    def score(self, X, y):
        # self.est_avg_acc = np.mean([self.parse_params(est).score(X, y) for est in self.estimators])
        return self._fit_estimator.score(X, y)

    def set_random_state(self, random_state):
        random.seed(random_state)
        self.random_state = random_state

