import math
from pathlib import Path

import numpy as np
import pickle

from mab.posteriors import Posterior, SinglePosteriors


class TDistributionPosterior(Posterior):
    """A t-distribution posteriors of a bandit's arms"""
    def __init__(self, seed=None, alpha=0.5):
        super(TDistributionPosterior, self).__init__(seed)
        self.alpha = alpha
        self.times_to_init_ = max(2, 3 - math.ceil(2 * self.alpha))
        self.mean_ = 0
        self.var_ = 0
        self.std_ = 0

    @staticmethod
    def new(seed=None, alpha=0.5):
        return TDistributionPosterior(seed, alpha)

    def update(self, reward, t):
        self.rewards.append(reward)
        self.mean_ = self.mean(t)
        self.var_ = self.var()
        self.std_ = self.sigma()

    def sample(self, t):
        n = len(self.rewards)
        freedom = self.freedom(n)
        sigma = np.sqrt(np.var(self.rewards) / freedom)
        mean = np.mean(self.rewards)
        t_sample = self.rng.standard_t(freedom)
        return mean + (t_sample * sigma)

    def freedom(self, n):
        return n + int(2 * self.alpha) - 1

    def sigma(self, n=None):
        return np.sqrt(self.var())

    def var(self):
        n = len(self.rewards)
        if n <= 2:
            return np.var(self.rewards)
        freedom = self.freedom(n)
        return np.var(self.rewards) / (freedom - 2)

    def save(self, path: Path):
        with open(path, mode="wb") as file:
            data = [self.alpha, self.rng, self.rewards, self.mean_, self.var_, self.std_]
            pickle.dump(data, file)

    def load(self, path: Path):
        with open(path, mode="rb") as file:
            data = pickle.load(file)
            self.alpha, self.rng, self.rewards, self.mean_, self.var_, self.std_ = data


class TDistributionPosteriors(SinglePosteriors):
    """Gaussian posteriors for a given number of bandit arms."""
    def __init__(self, nr_arms, seed=None, alpha=0.5):
        self.alpha = alpha
        super(TDistributionPosteriors, self).__init__(nr_arms, TDistributionPosterior, seed)

    def _create_posteriors(self, seed, posterior_type):
        return [posterior_type.new(seed + i, self.alpha) for i in range(self.nr_arms)]
