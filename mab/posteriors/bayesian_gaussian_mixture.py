import logging
import random

import numpy as np
from sklearn.mixture import BayesianGaussianMixture
import pickle

from mab.posteriors import Posterior, SinglePosteriors


class GaussianMixturePosterior(Posterior):
    """A gaussian mixture posterior of a bandit's arms"""
    def __init__(self, k, tol, max_iter, seed=None):
        # Super call
        super(GaussianMixturePosterior, self).__init__(seed)
        # The internal mixture distribution
        self._mixture = BayesianGaussianMixture(n_components=k, covariance_type='full', reg_covar=1e-06,
                                                tol=tol, max_iter=max_iter, n_init=3, init_params='random',
                                                weight_concentration_prior_type='dirichlet_process',
                                                weight_concentration_prior=None, mean_precision_prior=None,
                                                mean_prior=None, degrees_of_freedom_prior=None, covariance_prior=None,
                                                random_state=seed,
                                                warm_start=False,
                                                verbose=0, verbose_interval=10)

    @staticmethod
    def new(seed, k, tol, max_iter):
        return GaussianMixturePosterior(k, tol, max_iter, seed)

    def update(self, reward, t):
        self.rewards.append(reward)
        X = np.array(self.rewards)
        # Data contains a single feature (reward)
        X = X.reshape(-1, 1)
        # sklearn requires at least as many samples as mixture components
        if len(X) >= self._mixture.n_components:
            self._mixture.fit(X)
        else:
            logging.warning(f"Less rewards gathered than number of components for BGM fitting: {X}. "
                            f"Using it duplicated.")
            X = [X[0]] * self._mixture.n_components
            self._mixture.fit(X)

    def sample(self, t):
        """Sample the means"""

        samples = []
        weights = []
        sample = 0
        # Sample each mixture
        for m in range(self._mixture.n_components):
            # Sample the mixture's mean
            mu = self._mixture.means_[m][0]
            # The precision of each components on the mean distribution (Gaussian).
            mu_precision = self._mixture.mean_precision_[m]
            mu_variance = 1 / mu_precision
            mu_std_dev = mu_variance ** (1/2)
            # Sample the mean distribution
            m_sample = self.rng.normal(loc=mu, scale=mu_std_dev)
            # Adapt the mean according to its weight
            w = self._mixture.weights_[m]
            # Sample the weight distribution?
            # w_concentration = self._mixture.weight_concentration_[m]
            # w_sample = self.rng.dirichlet()

            weights.append(w)
            samples.append(m_sample)

            # print(mu, mu_precision, mu_std_dev, m_sample, w, w_concentration)

        # Choose a mean according to the weights
        sample = random.choices(samples, weights, k=1)[0]

        # sample = self.sample_old(t)
        # print("sample:", sample)

        return sample

    def sample_old(self, t):
        """Sample the reward distribution"""
        X, y = self._mixture.sample()
        X = X[0][0]
        return X

    def save(self, path):
        with open(path, 'wb') as file:
            pickle.dump(self._mixture, file)

    def load(self, path):
        with open(path, 'rb') as file:
            self._mixture = pickle.load(file)

    def get_mixture(self):
        return self._mixture


class BGMPosteriors(SinglePosteriors):
    """A Bayesian Gaussian Mixture Posterior for a given number of bandit arms."""
    def __init__(self, nr_arms, seed=None, k=2, tol=0.001, max_iter=100):
        self.k = k
        self.tol = tol
        self.max_iter = max_iter
        super(BGMPosteriors, self).__init__(nr_arms, GaussianMixturePosterior, seed)

    def _create_posteriors(self, seed, posterior_type):
        return [posterior_type.new(seed + i, self.k, self.tol, self.max_iter) for i in range(self.nr_arms)]
