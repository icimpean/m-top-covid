import random

import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats


class Env(object):
    """An environment an algorithm can interact with"""
    def __init__(self, seed=None):
        # Set the seed
        self.seed = seed
        np.random.seed(self.seed)
        random.seed(self.seed)

        # The observation and action space
        self.observation_space = None
        self.action_space = (1,)

        # The internal state and timestep
        self._state = None
        self._timestep = 0

    def reset(self):
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


class GaussianEnv(Env):
    """A Gaussian environment for testing.

    Represents an environment with gaussian rewards per arm.
    """
    def __init__(self, nr_actions=10, seed=None):
        # Super call
        super(GaussianEnv, self).__init__(seed)

        # The observation and action space
        self.observation_space = None  # MAB setting
        self.nr_actions = nr_actions
        self.action_space = (self.nr_actions,)

        # The rewards follow a gaussian distribution (chosen randomly)
        self.rng = np.random.default_rng(seed=seed)
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
        #
        return state, reward, done, info

    def print_info(self):
        stripes = "----------------------------"
        print(f"--- Gaussian Environment ---")
        for arm, (mean, std_dev) in enumerate(zip(self.means, self.std_dev)):
            print(f"arm {arm}: mean {mean}, std. dev. {std_dev}")
        print(stripes)

    def plot_rewards(self, show=False):
        plt.title(f"Rewards for gaussian environment with {self.nr_actions} actions")
        for arm, (mean, std_dev) in enumerate(zip(self.means, self.std_dev)):
            x = np.linspace(mean - 3*std_dev**2, mean + 3*std_dev**2, 100)
            y = stats.norm.pdf(x, mean, std_dev**2)
            plt.plot(x, y, label=str(arm))
        plt.legend()
        plt.savefig("gaussian rewards environment.png")
        if show:
            plt.show()
        plt.close()


class GaussianMixtureEnv(Env):
    """A Gaussian mixture environment for testing.

    Represents an environment with rewards drawn from a gaussian mixture distribution per arm.
    """
    def __init__(self, nr_actions=10, mixtures_per_arm=2, seed=None):
        # Super call
        super(GaussianMixtureEnv, self).__init__(seed)

        # The observation and action space
        self.observation_space = None  # MAB setting
        self.nr_actions = nr_actions
        self.action_space = (self.nr_actions,)

        # The rewards follow a gaussian mixture distribution (randomly initialised)
        self.rng = np.random.default_rng(seed=seed)

        # Different mixtures per action, with different proportions
        self.k = mixtures_per_arm
        self._mixtures = [i for i in range(self.k)]
        self.pi = self.rng.random(size=(self.nr_actions, self.k))
        self.pi /= self.pi.sum(axis=1, keepdims=True)
        # The means and standard deviations, per mixture per action
        self.means = self.rng.uniform(0, 20, size=(self.nr_actions, self.k)) * 0.5
        self.std_dev = self.rng.random(size=(self.nr_actions, self.k)) * 3 + 0.5

    def _sample_reward(self, action):
        # Choose a mixture for the given action, given their probabilities
        k = self.rng.multinomial(1, pvals=self.pi[action]).argmax()
        # Draw a sample from the gaussian of the given mixture
        mean = self.means[action, k]
        std_dev = self.std_dev[action, k]
        reward = self.rng.normal(loc=mean, scale=std_dev)
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
        print(stripes)

    def plot_rewards(self, show=False):
        plt.title(f"Rewards for a Gaussian Mixture Environment with {self.nr_actions} actions")
        for arm in range(self.nr_actions):
            min_x = np.inf
            max_x = -np.inf
            for k, (pi, mean, std_dev) in enumerate(zip(self.pi[arm], self.means[arm], self.std_dev[arm])):
                x1 = mean - 3*std_dev**2
                x2 = mean + 3*std_dev**2
                min_x = min(min_x, x1)
                max_x = max(max_x, x2)
            new_x = np.linspace(min_x, max_x, 1000)
            new_y = np.zeros_like(new_x)
            for k, (pi, mean, std_dev) in enumerate(zip(self.pi[arm], self.means[arm], self.std_dev[arm])):
                y = stats.norm.pdf(new_x, mean, std_dev**2)
                new_y += y * pi
            plt.plot(new_x, new_y, label=str(arm))
        plt.legend()
        plt.savefig("gaussian mixture rewards environment.png")
        if show:
            plt.show()
        plt.close()
