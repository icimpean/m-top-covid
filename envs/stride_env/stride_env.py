# noinspection PyUnresolvedReferences
import pylibstride as stride

import numpy as np
import random

from envs.env import Env
from envs.stride_env.action_wrapper import ActionWrapper


class StrideMDPEnv(Env):
    """The python wrapper for the STRIDE simulator for vaccine strategies (vaccine branch).

    Implements an environment interface for python algorithms.
    https://github.com/lwillem/stride/tree/vaccine (original)
    https://github.com/icimpean/stride/tree/vaccine (fork)

    Attributes:
        states: (Boolean) Indicating whether or not to use states as part of the
            agent-environment interaction. Default: False.
        seed: (Int) The random seed to use for initialising the random generators
            and the simulator. Default: 0.
        episode_duration: (Int) The length in days of a single simulation run.
            Default: 6 * 30 = 6 months.
        step_size: (Int) The number of days to follow a certain action before
            selecting a new action. Default: 2 * 30 = 2 months.
        config_file: The XML configuration file for the STRIDE simulator.
            Defaults to provided file in this directory.
        vaccine_availability: The CSV file with the number of available vaccines
            per vaccine type. Defaults to None.
        reward_type: How to process the reward signal before providing it to the agent.
            Accepted values:
                - 'neg': returns the negative of the reward signal
                - 'norm': normalise the reward based on the entire population
                - None: leaves the reward unchanged
    """
    def __init__(self, states=False, seed=0, episode_duration=6 * 30, step_size=2 * 30,
                 config_file="./run_default.xml", vaccine_availability=None,
                 reward_type=None):
        # Super call
        super(StrideMDPEnv, self).__init__(seed)
        # Set the seed
        self.seed = seed
        np.random.seed(self.seed)

        # The configuration file for STRIDE
        self.config_file = config_file
        # The MDP interface to the STRIDE simulator
        self._mdp = stride.MDP()

        # Episode length
        self.episode_duration = episode_duration
        # The step size (in days) of how often you can interact with the environment
        self.step_size = step_size
        # The number of timesteps per episode where the agent can select an action
        self.steps_per_episode = self.episode_duration // self.step_size

        # Action space for an action
        self._action_space = len(stride.AllVaccineTypes) ** len(stride.AllAgeGroups)
        self.action_wrapper = ActionWrapper(1, episode_duration, vaccine_availability)

        # Reward
        self.reward_type = reward_type
        self._population_size = 0

        # The internal state and timestep
        self.states = states
        self._state = None
        self._timestep = 0

    def reset(self):
        """Reset the environment and return the initial state (or None if no states are used)."""
        if self._timestep != 0:
            self._timestep = 0
            self._mdp.End()
        self._mdp.Create(self.config_file)
        self._population_size = self._mdp.GetPopulationSize()
        return None

    def close(self):
        """Signal the simulator to end the simulation and its own processes."""
        self._mdp.End()

    def step(self, action):
        """Perform a step of step_size in the simulation.

        Args:
            action: The action given to vaccinate the age groups with a vaccine type.
                e.g. [VaccineType.noVaccine VaccineType.noVaccine VaccineType.adeno VaccineType.mRNA VaccineType.adeno]

        Returns:
            state, reward, done, info - feedback from the interaction with the environment.
        """
        # Each arm (action) is a collection of actions per age group
        combined_action = self.action_wrapper.get_combined_action(action)
        print(f"Chosen action {action}")
        state = infected = None
        info = {}
        # Execute the action to vaccinate and simulate for as many days as required
        for _ in range(self.step_size):
            self._vaccinate(combined_action)
            infected = self._mdp.SimulateDay()
        # Transform the reward as requested
        reward = self._transform_reward(infected)
        # Another timestep has passed
        self._timestep += 1
        # The episode is done once we reach the episode duration
        done = self._timestep >= self.episode_duration
        # Give feedback
        return state, reward, done, info

    def _vaccinate(self, combined_action):
        # TODO parallelize in STRIDE?
        #   multiple age groups could be vaccinated at the same time since they don't overlap
        for action in combined_action:
            print(f"Vaccinating...", action)
            self._mdp.Vaccinate(*action)

    @staticmethod
    def random_action(available_vaccines=None):
        """Get a random action

        Args:
            available_vaccines: The number of vaccines to administer. If None, a random number is chosen.

        Returns:
            availableVaccines, ageGroup, vaccineType
        """
        if available_vaccines is None:
            available_vaccines = random.randint(0, 600) * 100
        group = random.choice(stride.AllAgeGroups)
        v_type = random.choice(stride.AllVaccineTypes[1:])
        return available_vaccines, group, v_type

    def _transform_reward(self, infected):
        if self.reward_type == "neg":
            return -infected
        elif self.reward_type == "norm":
            return (self._population_size - infected) / self._population_size
        else:
            return infected


class BanditStrideMDPEnv(StrideMDPEnv):
    """Wrapper for bandit algorithm

    Attributes:
        states: (Boolean) Indicating whether or not to use states as part of the
            agent-environment interaction. Default: False.
        seed: (Int) The random seed to use for initialising the random generators
            and the simulator. Default: 0.
        episode_duration: (Int) The length in days of a single simulation run.
            Default: 6 * 30 = 6 months.
        step_size: (Int) The number of days to follow a certain action before
            selecting a new action. Default: 2 * 30 = 2 months.
        config_file: The XML configuration file for the STRIDE simulator.
            Defaults to provided file in this directory.
        vaccine_availability: The CSV file with the number of available vaccines
            per vaccine type. Defaults to None.
        reward_type: How to process the reward signal before providing it to the agent.
            Accepted values:
                - 'neg': returns the negative of the reward signal
                - 'norm': normalise the reward based on the entire population
                - None: leaves the reward unchanged
    """
    def __init__(self, states=False, seed=0, episode_duration=6 * 30, step_size=2 * 30,
                 config_file="./run_default.xml", vaccine_availability=None,
                 reward_type=None):
        # Super call
        super(BanditStrideMDPEnv, self).__init__(states, seed, episode_duration, step_size, config_file,
                                                 vaccine_availability, reward_type)
        # Action is defined as a combination of actions over entire episode
        self.action_wrapper = ActionWrapper(step_size, episode_duration, vaccine_availability)
