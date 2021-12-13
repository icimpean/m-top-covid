import numpy as np

from mab.posteriors import Posteriors
from mab.sampling import Sampling


class RandomSampling(Sampling):
    """A random sampling method for the bandit."""
    def __init__(self, posteriors: Posteriors, seed):
        # Super call
        super(RandomSampling, self).__init__(posteriors, seed)
        # self.posteriors = posteriors
        self._nr_arms = len(posteriors)
        self.rng = np.random.default_rng(seed=seed)

    def best_arm(self, t):
        """Randomly sample an arm"""
        return self.sample_arm(t)

    def sample_arm(self, t):
        """Sample an arm based on the sampling method."""
        return self.rng.integers(0, self._nr_arms)
