import numpy as np

from mab.posteriors import Posterior


class GaussianPosterior(Posterior):
    """A gaussian posteriors of a bandit's arms"""
    def __init__(self, seed=None, alpha=0.5):
        super(GaussianPosterior, self).__init__(seed)
        self.alpha = alpha
        self._mean = -1
        self._sos = -1

    @staticmethod
    def new(seed=None, alpha=0.5):
        return GaussianPosterior(seed, alpha)

    def _freedom(self, n):
        return n + int(2 * self.alpha) - 1

    def sigma(self, n=None):
        if n is None:
            n = len(self.rewards)
        return np.sqrt(self._sos / (n * self._freedom(n)))

    def update(self, reward, t):
        self.rewards.append(reward)
        self._mean = sum(self.rewards) / len(self.rewards)
        self._sos = sum([(r - self._mean) ** 2 for r in self.rewards])

    def sample(self, t):
        n = len(self.rewards)
        t_sample = self.rng.standard_t(self._freedom(n))
        return self._mean + (t_sample * self.sigma(n))
