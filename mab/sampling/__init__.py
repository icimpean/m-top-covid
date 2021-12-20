import pickle
from pathlib import Path

import numpy as np

from mab.posteriors import Posteriors


class Sampling(object):
    """A sampling method for the bandit."""
    def __init__(self, posteriors: Posteriors, seed):
        self.posteriors = posteriors
        self.seed = seed
        self.rng = np.random.default_rng(seed=self.seed)
        self.has_ranking = False

    def best_arm(self, t):
        """Get the index with the highest sampled posterior reward."""
        # Sample the posteriors
        samples = self.posteriors.sample_all(t)
        best_arm = np.argmax(samples)
        return best_arm

    def sample_arm(self, t):
        """Sample an arm based on the sampling method."""
        raise NotImplementedError

    def update(self, arm, reward, t):
        """Update the posteriors"""
        self.posteriors.update(arm, reward, t)

    def compute_posteriors(self, t):
        """Compute posteriors. Executed before sampling/updating rewards."""
        self.posteriors.compute_posteriors(t)

    def save(self, path: Path):
        """Save the sampling method to the given file path"""
        with open(path, mode="wb") as file:
            data = [self.seed, self.rng]
            pickle.dump(data, file)

    def load(self, path: Path):
        """Load the sampling method from the given file path"""
        with open(path, mode="rb") as file:
            data = pickle.load(file)
            self.seed, self.rng = data
