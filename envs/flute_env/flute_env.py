import csv
import numpy as np
import random

from envs.env import Env
from envs.flute_env.action_wrapper import ActionWrapper


class FluteMDPEnv(Env):
    """The environment for simulating influenza from sampling the real distributions.

    Attributes:
        seed: (Optional) The random seed to use for initialising the random generators
            and the simulator.
        R0: (Optional) The R0 value to indicate which distributions to lead in.
        config_file: (Optional) The configuration file containing the arm distributions.
    """
    def __init__(self, seed=0, R0=2.0):
        # Super call
        super(FluteMDPEnv, self).__init__(seed)
        # Set the seed
        self.seed = seed
        random.seed(self.seed)
        np.random.seed(self.seed)
        self.rng = np.random.default_rng(seed=seed)

        # The distributions file
        self.R0 = R0
        self.config_file = f"../envs/flute_env/real-distributions/seattle-{self.R0}.csv"
        self._distributions = {}
        self._load_distributions()
        self.nr_actions = len(self._distributions)
        self.action_wrapper = ActionWrapper()

    def _load_distributions(self):
        with open(self.config_file, mode="r") as csv_file:
            reader = csv.reader(csv_file)
            for line in reader:
                # Arm
                arm_code = line[0]
                arm = int(arm_code[1:], base=2)
                rewards = [float(r) for r in line[1:]]
                std_dev = np.std(rewards)
                mean = np.mean(rewards)
                self._distributions[arm] = (mean, std_dev)

    def reset(self, **args):
        """Reset the environment and return the initial state (or None if no states are used)."""
        return None

    def step(self, action):
        """Perform a step of step_size in the simulation.

        Args:
            action: The action corresponding to a vaccine strategy.

        Returns:
            state, reward, done, info - feedback from the interaction with the environment.
        """
        print(f"Chosen action {action}")
        state = None
        done = True
        info = {}
        # Sample a reward from the given action
        mean, std_dev = self._distributions[action]
        reward = 1 - self.rng.normal(mean, std_dev)  # minimise epidemic size
        # Give feedback
        return state, reward, done, info

    def random_action(self):
        """Get a random action"""
        return self.rng.uniform(0, self.nr_actions)
