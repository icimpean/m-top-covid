import numpy as np

from mab.posteriors import Posteriors


class Sampling(object):
    """A sampling method for the bandit."""
    def __init__(self, posteriors: Posteriors, seed):
        self.posteriors = posteriors
        self.seed = seed
        np.random.seed(self.seed)

    def best_arm(self, t):
        """Get the index with the highest sampled posterior reward."""
        # Sample the posteriors
        samples = self.posteriors.sample_all(t)
        best_arm = np.argmax(samples)
        return best_arm

    def sample_arm(self, t):
        """Sample an arm based on the sampling method."""
        raise NotImplementedError
