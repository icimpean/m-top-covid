from pathlib import Path
from typing import List

import numpy as np


class Posterior(object):
    """A posterior distribution for a single arm."""
    def __init__(self, seed=None):
        self.seed = seed
        self.rng = np.random.default_rng(seed=seed)
        self.rewards = []

    @staticmethod
    def new(seed=None):
        raise NotImplementedError

    def update(self, reward, t):
        raise NotImplementedError

    def sample(self, t):
        raise NotImplementedError

    def mean(self, t):
        return np.mean(self.rewards)

    def save(self, path: Path):
        """Save the posterior to the given file path"""
        raise NotImplementedError

    def load(self, path: Path):
        """Load the posterior from the given file path"""
        raise NotImplementedError


class Posteriors(object):
    """The posteriors of a bandit's arms."""
    def __init__(self, nr_arms, seed):
        # Store the arguments
        self.nr_arms = nr_arms
        self.seed = seed
        #
        self._len = 0

    def update(self, arm, reward, t):
        """Update the given arm with the received reward at timestep t."""
        raise NotImplementedError

    def sample_all(self, t):
        """Sample a reward from all posteriors."""
        raise NotImplementedError

    def sample_best_arm(self, t):
        """Sample the best arm"""
        raise NotImplementedError

    def means_per_arm(self, t):
        """Get the mean reward per arm"""
        raise NotImplementedError

    def compute_posteriors(self, t):
        """Compute the posterior distributions, before sampling an arm for timestep t."""
        raise NotImplementedError

    def save(self, path: Path):
        """Save the posteriors to the given directory path"""
        raise NotImplementedError

    def load(self, path: Path):
        """Load the posteriors from the given directory path"""
        raise NotImplementedError


class SinglePosteriors(Posteriors):
    """The posteriors of a bandit's arms, built with Posterior instances."""
    def __init__(self, nr_arms, posterior_type, seed, **args):
        # Super call
        super(SinglePosteriors, self).__init__(nr_arms, seed)
        # Initialise the posteriors
        _seed = np.random.randint(0, 1000) if seed is None else seed
        self.posteriors: List[Posterior] = self._create_posteriors(_seed, posterior_type)
        self._len = len(self.posteriors)
        self.arm_means = [0 for _ in self.posteriors]

    def __getitem__(self, item):
        return self.posteriors[item]

    def __len__(self):
        return len(self.posteriors)

    def __iter__(self):
        self._n = 0
        return self

    def __next__(self):
        if self._n < self._len:
            n = self._n
            self._n += 1
            return self[n]
        else:
            self._n = 0
            raise StopIteration

    def _create_posteriors(self, seed, posterior_type):
        return [posterior_type.new(seed + i) for i in range(self.nr_arms)]

    def update(self, arm, reward, t):
        # Update the posterior
        self[arm].update(reward, t)
        # Update the mean of the arm to update
        self.arm_means[arm] = self[arm].mean(t)

    def sample_all(self, t):
        return [p.sample(t) for p in self.posteriors]

    def sample_best_arm(self, t):
        # Default gets best action
        samples = self.sample_all(t)
        return np.argmax(samples)

    def means_per_arm(self, t):
        return [p.mean(t) for p in self.posteriors]

    def compute_posteriors(self, t):
        pass

    def save(self, path: Path):
        for i in range(self.nr_arms):
            p_path = path.with_name(f"{path.name}arm_{i}.posterior")
            self.posteriors[i].save(p_path)

    def load(self, path: Path):
        for i in range(self.nr_arms):
            p_path = path.with_name(f"{path.name}arm_{i}.posterior")
            self.posteriors[i].load(p_path)


class GroupPosterior(Posteriors):
    """The posteriors of a bandit's arms, containing all posteriors."""
    def __init__(self, nr_arms, seed):
        # Super call
        super(GroupPosterior, self).__init__(nr_arms, seed)
        #
        self._len = 0

    def update(self, arm, reward, t):
        raise NotImplementedError

    def sample_all(self, t):
        raise NotImplementedError

    def sample_best_arm(self, t):
        raise NotImplementedError

    def means_per_arm(self, t):
        raise NotImplementedError

    def compute_posteriors(self, t):
        raise NotImplementedError

    def save(self, path: Path):
        raise NotImplementedError

    def load(self, path: Path):
        raise NotImplementedError
