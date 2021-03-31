import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats


class Env(object):
    """An environment an algorithm can interact with

    TODO: documentation
    """
    def __init__(self, seed=None):
        # Set the seed
        self.seed = seed
        np.random.seed(self.seed)

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
        plt.title(f"Rewards for {self.nr_actions}-armed Bandit environment")
        for arm, (mean, std_dev) in enumerate(zip(self.means, self.std_dev)):
            x = np.linspace(mean - 3*std_dev**2, mean + 3*std_dev**2, 100)
            y = stats.norm.pdf(x, mean, std_dev**2)
            plt.plot(x, y, label=str(arm))
        plt.legend()
        plt.savefig("gaussian rewards environment.png")
        if show:
            plt.show()
        plt.close()
