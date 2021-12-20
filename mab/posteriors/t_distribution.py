from pathlib import Path

import numpy as np
import pickle

from mab.posteriors import Posterior, SinglePosteriors


class TDistributionPosterior(Posterior):
    """A t-distribution posteriors of a bandit's arms"""
    def __init__(self, seed=None, alpha=0.5):
        super(TDistributionPosterior, self).__init__(seed)
        self.alpha = alpha
        self._mean = -1
        self._sos = -1

    @staticmethod
    def new(seed=None, alpha=0.5):
        return TDistributionPosterior(seed, alpha)

    def update(self, reward, t):
        self.rewards.append(reward)
        self._mean = sum(self.rewards) / len(self.rewards)
        self._sos = sum([(r - self._mean) ** 2 for r in self.rewards])

    def sample(self, t):
        n = len(self.rewards)
        t_sample = self.rng.standard_t(self._freedom(n))
        return self._mean + (t_sample * self.sigma(n))

    def _freedom(self, n):
        return n + int(2 * self.alpha) - 1

    def sigma(self, n=None):
        if n is None:
            n = len(self.rewards)
        return np.sqrt(self._sos / (n * self._freedom(n)))

    def save(self, path: Path):
        with open(path, mode="wb") as file:
            data = [self.alpha, self._mean, self._sos, self.rng, self.rewards]
            pickle.dump(data, file)

    def load(self, path: Path):
        with open(path, mode="rb") as file:
            data = pickle.load(file)
            self.alpha, self._mean, self._sos, self.rng, self.rewards = data


class TDistributionPosteriors(SinglePosteriors):
    """Gaussian posteriors for a given number of bandit arms."""
    def __init__(self, nr_arms, seed=None, alpha=0.5):
        self.alpha = alpha
        super(TDistributionPosteriors, self).__init__(nr_arms, TDistributionPosterior, seed)

    def _create_posteriors(self, seed, posterior_type):
        return [posterior_type.new(seed + i, self.alpha) for i in range(self.nr_arms)]
