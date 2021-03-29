import numpy as np

from mab.posteriors import Posteriors
from mab.sampling import Sampling


class BFTS(Sampling):
    """Boundary Focused Thompson Sampling"""
    def __int__(self, posteriors: Posteriors, top_m, seed):
        # Super call
        super(BFTS, self).__int__(posteriors, seed)
        #
        self.m = top_m

    def sample_arm(self, t):
        """Sample an arm based on the sampling method."""
        # Sample all arms and order them
        theta = self.posteriors.sample_all(t)
        order = np.argsort(-theta)
        # Choose an arm from the boundary (top_m boundary)
        arm_i = order[self.m - 1 + np.random.choice([0, 1])]
        return arm_i

    def top_m(self, t):
        """Get the top m arms at timestep t"""
        # Get the means per arm
        means = self.posteriors.means_per_arm(t)
        return np.argsort(-means)[0:self.m]
