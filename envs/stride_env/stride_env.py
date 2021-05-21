# noinspection PyUnresolvedReferences
import pylibstride as stride

import gc
import numpy as np
import random
import resource

from envs.env import Env
from envs.stride_env.action_wrapper import ActionWrapper
from resources.vaccine_supply import VaccineSupply, ConstantVaccineSupply


class StrideMDPEnv(Env):
    """The python wrapper for the STRIDE simulator for vaccine strategies (vaccine branch).

    Implements an environment interface for python algorithms.
    https://github.com/lwillem/stride/tree/vaccine (original)
    https://github.com/icimpean/stride/tree/vaccine (fork)

    Attributes:
        states: (Optional) Boolean indicating whether or not to use states as part of the
            agent-environment interaction.
        seed: (Optional) The random seed to use for initialising the random generators
            and the simulator.
        episode_duration: (Optional) The length in days of a single simulation run.
        step_size: (Optional) The number of days to follow a certain action before
            selecting a new action.
        config_file: (Optional) The XML configuration file for the STRIDE simulator.
            Defaults to provided file in this directory.
        available_vaccines: (Optional) The VaccineSupply for the simulation.
        reward_type: (Optional) How to process the reward signal before providing it to the agent.
            Accepted values:
                - 'neg':  returns the negative of the reward signal
                - 'norm': normalise the reward based on the entire population
                -  None:  leaves the reward unchanged
    """
    def __init__(self, states=False, seed=0, episode_duration=6 * 30, step_size=2 * 30,
                 config_file="./run_default.xml", available_vaccines: VaccineSupply = ConstantVaccineSupply(),
                 reward_type=None):
        # Super call
        super(StrideMDPEnv, self).__init__(seed)
        # Set the seed
        self.seed = seed
        random.seed(self.seed)
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
        self.action_wrapper = ActionWrapper(available_vaccines)

        # Reward
        self.reward_type = reward_type
        self._population_size = 0

        # The internal state and timestep
        self.states = states
        self._state = None
        self._timestep = 0
        self._e = 0

        self._x = []

    def reset(self, seed=None, output_dir=None, output_prefix=None):
        """Reset the environment and return the initial state (or None if no states are used)."""
        if self._timestep != 0:
            self._timestep = 0
            self._mdp.End()
            self._e += 1

            # del self._mdp
            # gc.collect()
            # self._mdp = stride.MDP()

            self._mdp.ClearSimulation()

        # Measure memory usage from python
        x = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        self._x.append(x)

        output_dir = output_dir if output_dir is not None else ""
        output_prefix = output_prefix if output_prefix is not None else ""
        seed = seed if seed is not None else self.seed
        self._mdp.Create(self.config_file, seed, output_dir, output_prefix)
        self._population_size = self._mdp.GetPopulationSize()
        return None

    def close_x(self):
        for i, x in enumerate(self._x):
            print(f"Resources: {x} bytes ({round(x / 1024, 2)} kB, {round(x / 1024 ** 2, 2)} MB, {round(x / 1024 ** 3, 2)} GB)")
        # print("---")
        # for i, x in enumerate(self._x):
        #     print(x / 1024 ** 2)

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
        days = range(self._timestep * self.step_size, (self._timestep * self.step_size) + self.step_size)
        combined_action = self.action_wrapper.get_combined_action(action, days)
        print(f"Chosen action {action}")
        state = infected = None
        info = {}
        # Execute the action to vaccinate and simulate for as many days as required
        for t in range(self.step_size):
            self._vaccinate(combined_action[t])
            infected = self._mdp.SimulateDay()

            print(f"infected: {self._mdp.CountInfectedCases()}, exposed: {self._mdp.CountExposedCases()}, "
                  f"infectious: {self._mdp.CountInfectiousCases()}, symptomatic: {self._mdp.CountSymptomaticCases()}")

        # Transform the reward as requested
        reward = self._transform_reward(infected)
        # Another timestep has passed
        self._timestep += 1
        # The episode is done once we reach the episode duration
        done = self._timestep >= self.steps_per_episode
        # Give feedback
        return state, reward, done, info

    def _vaccinate(self, combined_action):
        # Could be parallelized in STRIDE:
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
