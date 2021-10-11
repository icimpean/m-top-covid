import random

import numpy as np


class Env(object):
    """An environment an algorithm can interact with"""
    def __init__(self, seed=None):
        # Set the seed
        self.seed = seed
        np.random.seed(self.seed)
        random.seed(self.seed)

        # The observation and action space
        self.observation_space = None
        self.nr_actions = 1
        self.action_space = (self.nr_actions,)

        # The internal state and timestep
        self._state = None
        self._timestep = 0

    def reset(self, **args):
        """Reset the environment and return the state.

        By default, there is no state (None).
        """
        return None

    def step(self, action):
        """Execute the given action and return the next state (if any),
        the reward and whether or not the episode is over"""
        state = None
        reward = None
        done = False
        info = {}
        # Execute the action
        # (To be done by the
        return state, reward, done, info

    def print_info(self):
        """Print information about the environment.

        Useful for randomised environments"""
        pass

    def close(self):
        pass


class GaussianEnv(Env):
    """A Gaussian environment for testing.

    Represents an environment with gaussian rewards per arm.
    """
    def __init__(self, nr_actions=10, seed=None, normalised=False):
        # Super call
        super(GaussianEnv, self).__init__(seed)

        # The observation and action space
        self.observation_space = None  # MAB setting
        self.nr_actions = nr_actions
        self.action_space = (self.nr_actions,)
        self.normalised = normalised

        # The rewards follow a gaussian distribution (chosen randomly)
        self.rng = np.random.default_rng(seed=seed)
        if self.normalised:
            self.means = self.rng.random(size=self.nr_actions)
            self.std_dev = self.rng.random(size=self.nr_actions)
        else:
            self.means = self.rng.uniform(0, 20, size=self.nr_actions) * 0.5
            self.std_dev = self.rng.random(size=self.nr_actions) * 3 + 0.5

    def step(self, action):
        state = None
        done = False
        info = {}
        # Sample reward from distribution of the given action
        mean = self.means[action]
        std_dev = self.std_dev[action]
        reward = self.rng.normal(loc=mean, scale=std_dev)
        if self.normalised:
            reward = np.clip(reward, 0, 1)
        #
        return state, reward, done, info

    def print_info(self):
        stripes = "----------------------------"
        print(f"--- Gaussian Environment ---")
        for arm, (mean, std_dev) in enumerate(zip(self.means, self.std_dev)):
            print(f"arm {arm}: mean {mean}, std. dev. {std_dev}")
        print(stripes)


class GaussianMixtureEnv(Env):
    """A Gaussian mixture environment.

    Represents an environment with rewards drawn from a gaussian mixture distribution per arm.
    """
    def __init__(self, nr_actions=10, mixtures_per_arm=2, seed=None, normalised=False):
        # Super call
        super(GaussianMixtureEnv, self).__init__(seed)

        # The observation and action space
        self.observation_space = None  # MAB setting
        self.nr_actions = nr_actions
        self.action_space = (self.nr_actions,)
        self.normalised = normalised

        # The rewards follow a gaussian mixture distribution (randomly initialised)
        self.rng = np.random.default_rng(seed=seed)

        # Different mixtures per action, with different proportions
        self.k = mixtures_per_arm
        self._mixtures = [i for i in range(self.k)]
        self.pi = self.rng.random(size=(self.nr_actions, self.k))
        self.pi /= self.pi.sum(axis=1, keepdims=True)
        # The means and standard deviations, per mixture per action
        if self.normalised:
            self.means = self.rng.random(size=(self.nr_actions, self.k))
            self.std_dev = self.rng.random(size=(self.nr_actions, self.k))
        else:
            self.means = self.rng.uniform(0, 20, size=(self.nr_actions, self.k)) * 0.5
            self.std_dev = self.rng.random(size=(self.nr_actions, self.k)) * 3 + 0.5

    def _sample_reward(self, action):
        # Choose a mixture for the given action, given their probabilities
        k = self.rng.multinomial(1, pvals=self.pi[action]).argmax()
        # Draw a sample from the gaussian of the given mixture
        mean = self.means[action, k]
        std_dev = self.std_dev[action, k]
        reward = self.rng.normal(loc=mean, scale=std_dev)
        if self.normalised:
            reward = np.clip(reward, 0, 1)
        return reward

    def step(self, action):
        state = None
        done = False
        info = {}
        # Sample reward from distribution of the given action
        reward = self._sample_reward(action)
        #
        return state, reward, done, info

    def print_info(self):
        stripes = "----------------------------"
        print(f"--- Gaussian Mixture Environment with {self.k} mixtures per action ---")
        for arm in range(self.nr_actions):
            print(f"Action {arm}:")
            for k, (pi, mean, std_dev) in enumerate(zip(self.pi[arm], self.means[arm], self.std_dev[arm])):
                print(f"\t[mixture {k}] pi: {pi}, mean: {mean}, std. dev. {std_dev}")
            print(f"\tmean: {sum(self.pi[arm] * self.means[arm])}")
        print(stripes)


class TestGaussianMixtureEnv(GaussianMixtureEnv):
    """A Gaussian mixture environment for testing"""
    def __init__(self, seed=None):
        # Super call
        super(TestGaussianMixtureEnv, self).__init__(seed)

        # Fixed environment
        arms = [
            [(1, 10, 2), (2, 11, 1.4)],
            [(2, 8, 1.2), (3, 9, 1)],
            [(3, 9, 2.1), (4, 11, 1.7)],
            [(7, 10, 2), (5, 12, 1)],
            [(1, 10, 1.4), (2, 13, 1)],
            [(1, 8, 2), (1, 10, 1.1)],
            [(2, 13, 0.7), (7, 8, 1)],
            [(1, 10, 0.9), (2, 15, 1.3)],
        ]

        self.k = 2
        self.nr_actions = len(arms)

        self.pi = np.zeros(shape=(self.nr_actions, self.k))
        self.means = np.zeros_like(self.pi)
        self.std_dev = np.zeros_like(self.pi)

        for arm, values in enumerate(arms):
            for k, (pi, mean, std_dev) in enumerate(values):
                self.pi[arm][k] = pi
                self.means[arm][k] = mean
                self.std_dev[arm][k] = std_dev
        self.pi /= self.pi.sum(axis=1, keepdims=True)


class Test2GaussianMixtureEnv(GaussianMixtureEnv):
    """A Gaussian mixture environment for testing"""
    def __init__(self, seed=None):
        # Super call
        super(Test2GaussianMixtureEnv, self).__init__(seed)

        # Fixed environment
        arms = [
            [(1, 10, 2), (2, 11, 1.4), (4, 9, 0.8)],
            [(2, 8, 1.2), (3, 9, 1), (1, 9, 2)],
            [(3, 9, 2.1), (4, 11, 1.7), (4, 8, 1.2)],
            [(7, 10, 2), (5, 12, 1), (1, 8, 1.4)],
            [(1, 10, 1.4), (2, 13, 1), (1, 12, 2)],
            [(1, 8, 2), (1, 10, 1.1), (1, 12, 1.7)],
            [(2, 13, 0.7), (7, 8, 1), (3, 11, 1.1)],
            [(1, 10, 0.9), (2, 15, 1.3), (1, 12, 1)],
        ]

        self.k = 3
        self.nr_actions = len(arms)

        self.pi = np.zeros(shape=(self.nr_actions, self.k))
        self.means = np.zeros_like(self.pi)
        self.std_dev = np.zeros_like(self.pi)

        for arm, values in enumerate(arms):
            for k, (pi, mean, std_dev) in enumerate(values):
                self.pi[arm][k] = pi
                self.means[arm][k] = mean
                self.std_dev[arm][k] = std_dev
        self.pi /= self.pi.sum(axis=1, keepdims=True)
