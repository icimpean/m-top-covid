from pathlib import Path

import numpy as np
import pickle

from mab.posteriors import Posterior, SinglePosteriors


class GaussianPosterior(Posterior):
    """A gaussian posteriors of a bandit's arms"""
    def __init__(self, seed=None, var=1.0, var0=1.0, mu0=0.0):
        super(GaussianPosterior, self).__init__(seed)
        self._var = var
        self._var0 = var0
        self._mu0 = mu0
        self.rng = np.random.default_rng(seed=seed)

    @staticmethod
    def new(seed=None, var=1.0, var0=1.0, mu0=0.0):
        return GaussianPosterior(seed, var, var0, mu0)

    def update(self, reward, t):
        self.rewards.append(reward)

    def sample(self, t):
        n = len(self.rewards)
        denom = (1.0 / self._var0 + n / self._var)
        u = (self._mu0 / self._var0 + sum(self.rewards) / self._var) / denom
        var = 1.0 / denom
        stddev = np.sqrt(var)
        return self.rng.normal(u, stddev)

    def save(self, path: Path):
        with open(path, mode="wb") as file:
            data = [self._var, self._var0, self._mu0, self.rng, self.rewards]
            pickle.dump(data, file)

    def load(self, path: Path):
        with open(path, mode="rb") as file:
            data = pickle.load(file)
            self._var, self._var0, self._mu0, self.rng, self.rewards = data


class GaussianPosteriors(SinglePosteriors):
    """Gaussian posteriors for a given number of bandit arms."""
    def __init__(self, nr_arms, seed=None, var=1.0, var0=1.0, mu0=0.0):
        self.var = var
        self.var0 = var0
        self.mu0 = mu0
        super(GaussianPosteriors, self).__init__(nr_arms, GaussianPosterior, seed)

    def _create_posteriors(self, seed, posterior_type):
        return [posterior_type.new(seed + i, self.var, self.var0, self.mu0) for i in range(self.nr_arms)]
