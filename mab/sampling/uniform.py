import numpy as np

from mab.posteriors import Posteriors
from mab.sampling import Sampling


class UniformSampling(Sampling):
    """A random sampling method for the bandit."""
    def __init__(self, posteriors: Posteriors, top_m, seed):
        # Super call
        super(UniformSampling, self).__init__(posteriors, seed)
        # self.posteriors = posteriors
        self._nr_arms = len(posteriors)
        self.m = top_m
        self.rng = np.random.default_rng(seed=seed)

        self.has_ranking = True
        self.sample_ordering = None
        self.current_ranking = None

    @staticmethod
    def new(top_m):
        """Workaround to add a given top_m arms to the sampling method"""
        return lambda posteriors, seed: UniformSampling(posteriors, top_m, seed)

    def best_arm(self, t):
        """Randomly sample an arm"""
        return self.sample_arm(t)

    def sample_arm(self, t):
        """Sample an arm based on the sampling method."""
        least_sampled = self.least_sampled_indices()
        arm = np.random.choice(least_sampled)

        theta = self.posteriors.sample_all(t)
        order = np.argsort(-np.array(theta))
        self.sample_ordering = order
        self.current_ranking = self.top_m(t)

        return arm

    def least_sampled_indices(self):
        count_per_arm = list(map(lambda p: len(p.rewards), self.posteriors))
        return np.where(count_per_arm == np.min(count_per_arm))[0]

    def top_m(self, t):
        means = self.posteriors.means_per_arm(t)
        if isinstance(means, list):
            means = np.array(means)
        return np.argsort(-means)[0:self.m]
