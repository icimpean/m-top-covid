import logging
import pickle
from pathlib import Path

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
        super(BFTS, self).__init__(posteriors.nr_arms, seed)
        #
        self.posteriors = posteriors
        self.m = top_m
        self.has_ranking = True
        self.sample_ordering = None
        self.current_ranking = None

    @staticmethod
    def new(top_m):
        """Workaround to add a given top_m arms to the sampling method"""
        return lambda posteriors, seed: BFTS(posteriors, top_m, seed)

    def update(self, arm, reward, t):
        """Update the posteriors"""
        self.rewards_per_arm[arm].append(reward)
        self.posteriors.update(arm, reward, t)
        self.mean_per_arm[arm] = self.posteriors[arm].mean_
        self.var_per_arm[arm] = self.posteriors[arm].var_
        self.std_per_arm[arm] = self.posteriors[arm].std_

    def sample_arm(self, t):
        """Sample an arm based on the sampling method."""
        # Sample all arms and order them
        theta = self.posteriors.sample_all(t)
        order = np.argsort(-np.array(theta))
        # Choose an arm from the boundary (top_m boundary)
        arm_i = order[self.m - 1 + self.rng.choice([0, 1])]

        self.sample_ordering = order
        self.current_ranking = self.top_m(t)
        logging.info(f"=== TOP-M arms at timestep {t}: {self.current_ranking} ===")
        return arm_i

    def top_m(self, t):
        """Get the top m arms at timestep t"""
        # Get the means per arm
        means = self.posteriors.means_per_arm(t)
        if isinstance(means, list):
            means = np.array(means)
        return np.argsort(-means)[0:self.m]

    def compute_posteriors(self, t):
        """Compute posteriors. Executed before sampling/updating rewards."""
        self.posteriors.compute_posteriors(t)

    def save(self, t, path: Path):
        """Save the sampling method to the given file path"""
        # Sampling method
        sampling_dir = path / "Sampling"
        sampling_dir.mkdir(exist_ok=True)
        sampling_path = sampling_dir / f"t{t}.sampling"
        with open(sampling_path, mode="wb") as file:
            data = [self.seed, self.rng, self.m, self.has_ranking, self.sample_ordering, self.current_ranking,
                    self.rewards_per_arm, self.mean_per_arm, self.var_per_arm, self.std_per_arm]
            pickle.dump(data, file)
        # Posteriors
        posterior_dir = path / "Posteriors"
        posterior_dir.mkdir(exist_ok=True)
        posterior_dir /= f"t{t}-"
        self.posteriors.save(posterior_dir)

    def load(self, t, path: Path):
        """Load the sampling method from the given file path"""
        # Sampling method
        sampling_path = path / "Sampling" / f"t{t}.sampling"
        with open(sampling_path, mode="rb") as file:
            data = pickle.load(file)
            self.seed, self.rng, self.m, self.has_ranking, self.sample_ordering, self.current_ranking, \
                self.rewards_per_arm, self.mean_per_arm, self.var_per_arm, self.std_per_arm = data
        # Posteriors
        posterior_dir = path / "Posteriors" / f"t{t}-"
        self.posteriors.load(posterior_dir)
