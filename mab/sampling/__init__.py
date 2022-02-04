import pickle
from pathlib import Path

import numpy as np


class Sampling(object):
    """A sampling method for the bandit."""
    def __init__(self, nr_arms, seed):
        self.nr_arms = nr_arms
        self.seed = seed
        self.rng = np.random.default_rng(seed=self.seed)
        self.has_ranking = False

        self.rewards_per_arm = [[] for i in range(self.nr_arms)]
        self.mean_per_arm = np.zeros(self.nr_arms)
        self.var_per_arm = np.zeros(self.nr_arms)
        self.std_per_arm = np.zeros(self.nr_arms)

    def best_arm(self, t):
        """Get the best arm"""
        best_arm = np.argmax(self.mean_per_arm)
        return best_arm

    def sample_arm(self, t):
        """Sample an arm based on the sampling method."""
        raise NotImplementedError

    def update(self, arm, reward, t):
        """Update the posteriors"""
        self.rewards_per_arm[arm].append(reward)
        rewards = self.rewards_per_arm[arm]
        self.mean_per_arm[arm] = np.mean(rewards)
        self.var_per_arm[arm] = np.var(rewards)
        self.std_per_arm[arm] = np.std(rewards)

    def compute_posteriors(self, t):
        """Compute posteriors. Executed before sampling/updating rewards."""
        pass

    def save(self, t, path: Path):
        """Save the sampling method to the given file path"""
        with open(path, mode="wb") as file:
            data = [self.nr_arms, self.seed, self.rng,
                    self.rewards_per_arm, self.mean_per_arm, self.var_per_arm, self.std_per_arm]
            pickle.dump(data, file)

    def load(self, t, path: Path):
        """Load the sampling method from the given file path"""
        with open(path, mode="rb") as file:
            data = pickle.load(file)
            self.nr_arms, self.seed, self.rng, self.rewards_per_arm, self.mean_per_arm, self.var_per_arm, \
                self.std_per_arm = data
