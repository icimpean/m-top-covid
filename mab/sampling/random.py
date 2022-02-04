from mab.sampling import Sampling


class RandomSampling(Sampling):
    """A random sampling method for the bandit."""
    def __init__(self, nr_arms, seed):
        # Super call
        super(RandomSampling, self).__init__(nr_arms, seed)

    def best_arm(self, t):
        """Randomly sample an arm"""
        return self.sample_arm(t)

    def sample_arm(self, t):
        """Sample an arm based on the sampling method."""
        return self.rng.integers(0, self.nr_arms)
