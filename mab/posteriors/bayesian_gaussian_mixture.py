import logging
import random
from pathlib import Path

import numpy as np
from sklearn.mixture import BayesianGaussianMixture
import pickle

from mab.posteriors import Posterior, SinglePosteriors


class GaussianMixturePosterior(Posterior):
    """A gaussian mixture posterior of a bandit's arms"""
    def __init__(self, k, tol, max_iter, num=0, seed=None, log_dir="./rl-tmp"):
        # Super call
        super(GaussianMixturePosterior, self).__init__(seed)
        path = log_dir
        self._output_path = Path(f"{path}/Posteriors/posterior_{num}/").absolute()

        self.mean_ = 0
        self.var_ = 0
        self.std_ = 0
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
    def new(seed, k, tol, max_iter, num=0, log_dir="/Users/alexandracimpean/Documents/VUB/PhD/COVID19/Code/rl-tmp/"):
        return GaussianMixturePosterior(k, tol, max_iter, num, seed, log_dir)

    def update(self, reward, t):
        self.rewards.append(reward)
        X = np.array(self.rewards)
        # Data contains a single feature (reward)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        # sklearn requires at least as many samples as mixture components
        if len(X) >= max(2, self._mixture.n_components):
            self._mixture.fit(X)
        else:
            logging.warning(f"Less rewards gathered than number of components for BGM fitting: {X}. "
                            f"Using it duplicated.")
            while len(X) < max(2, self._mixture.n_components):
                X = [*X] + [*X]
            # X = [X[0]] * max(2, self._mixture.n_components)
            self._mixture.fit(X)
        # Update statistics
        self.mean_ = self.mixture_mean()
        self.var_ = self.mixture_mean_variance()
        self.std_ = self.get_std_dev()

    def sample(self, t):
        """Sample the means"""
        samples = []
        weights = []
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
            weights.append(w)
            samples.append(m_sample)

        # Choose a mean according to the weights
        sample = random.choices(samples, weights, k=1)[0]
        return sample

    def sample_old(self, t):
        """Sample the reward distribution"""
        X, y = self._mixture.sample()
        X = X[0][0]
        return X

    def save(self, path):
        with open(path, 'wb') as file:
            data = (self.rng, self._mixture, self.rewards, self.mean_, self.var_, self.std_)
            pickle.dump(data, file)
            # pickle.dump(self._mixture, file)

    def load(self, path):
        with open(path, 'rb') as file:
            self.rng, self._mixture, self.rewards, self.mean_, self.var_, self.std_ = pickle.load(file)
            # self._mixture = pickle.load(file)

    def get_mixture(self):
        return self._mixture

    def mixture_mean(self):
        """The mean of the gaussian mixture distribution."""
        mean = np.sum(self._mixture.means_.reshape(1, -1)[0] * self._mixture.weights_)
        return mean

    def mixture_mean_variance(self):
        pi = self._mixture.weights_
        # precision = self._mixture.mean_precision_
        # var_k = 1 / precision
        var_k = self._mixture.covariances_
        var_k = var_k.reshape(1, -1)[0]
        # std_dev = var_k ** (1 / 2)
        # mu = self.mixture_mean()

        mu_k = self._mixture.means_
        mu_k = mu_k.reshape(1, -1)[0]
        total_variance = np.sum(pi * var_k) + np.sum(pi * mu_k ** 2) - (np.sum(mu_k * pi) ** 2)

        # print(pi, mu_k, var_k, total_variance)
        # print("total_variance", total_variance)

        return total_variance

    def get_std_dev(self):
        """Get std dev from mean precision on each component"""
        pi = self._mixture.weights_
        # prec = self._mixture.mean_precision_
        # prec = np.sum(pi * prec)
        # var = 1 / prec
        var = self.mixture_mean_variance()
        std = np.sqrt(var)
        # print(std, pi, self._mixture.mean_precision_)
        return std


class BGMPosteriors(SinglePosteriors):
    """A Bayesian Gaussian Mixture Posterior for a given number of bandit arms."""
    def __init__(self, nr_arms, seed=None, k=2, tol=0.001, max_iter=100, log_dir="./"):
        self.k = k
        self.tol = tol
        self.max_iter = max_iter
        self.log_dir = log_dir
        super(BGMPosteriors, self).__init__(nr_arms, GaussianMixturePosterior, seed)

    def _create_posteriors(self, seed, posterior_type):
        return [posterior_type.new(seed + i, self.k, self.tol, self.max_iter, i, self.log_dir) for i in range(self.nr_arms)]
