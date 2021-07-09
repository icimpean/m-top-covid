import numpy as np

from mab.posteriors import Posteriors
from mab.sampling import Sampling


class BFTS(Sampling):
    """Boundary Focused Thompson Sampling

    From: https://github.com/plibin-vub/bfts

    Attributes:
        posteriors: The Posteriors to apply the sampling method to
        top_m: The number of posteriors to provide as the top m best posteriors.
        seed: The seed for initialisation.
    """
    def __init__(self, posteriors: Posteriors, top_m, seed):
        # Super call
        super(BFTS, self).__init__(posteriors, seed)
        #
        self.m = top_m

    @staticmethod
    def new(top_m):
        """Workaround to add a given top_m arms to the sampling method"""
        return lambda posteriors, seed: BFTS(posteriors, top_m, seed)

    def sample_arm(self, t):
        """Sample an arm based on the sampling method."""
        # Sample all arms and order them
        theta = self.posteriors.sample_all(t)

        # TODO: remove
        print(f"Arm samples: {theta}")

        order = np.argsort(-np.array(theta))
        # Choose an arm from the boundary (top_m boundary)
        arm_i = order[self.m - 1 + np.random.choice([0, 1])]

        # print(f"=== TOP_M arms at timestep {t}: {self.top_m(t)} ===")

        return arm_i

    def top_m(self, t):
        """Get the top m arms at timestep t"""
        # Get the means per arm
        means = self.posteriors.means_per_arm(t)
        if isinstance(means, list):
            means = np.array(means)
        return np.argsort(-means)[0:self.m]
