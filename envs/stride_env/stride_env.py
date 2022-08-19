# noinspection PyUnresolvedReferences
import os

import pylibstride as stride

from enum import Enum, auto
import numpy as np
import random
import resource
import logging
from typing import Type
import csv

from envs.env import Env
from envs.stride_env.action_wrapper import ActionWrapper, NoWasteActionWrapper
from resources.vaccine_supply import VaccineSupply, ConstantVaccineSupply


class Reward(Enum):
    """The enumeration for the type of reward to use"""
    # Counts
    infected = auto()
    exposed = auto()
    infectious = auto()
    symptomatic = auto()
    hospitalised = auto()

    # Cumulative counts
    total_infected = auto()
    total_hospitalised = auto()
    total_at_risk = auto()
    total_at_risk_hosp = auto()


# The mapping of reward types and the corresponding method call
_reward_mapping = {
    Reward.infected: lambda mdp: mdp.CountInfectedCases(),
    Reward.exposed: lambda mdp: mdp.CountExposedCases(),
    Reward.infectious: lambda mdp: mdp.CountInfectiousCases(),
    Reward.symptomatic: lambda mdp: mdp.CountSymptomaticCases(),
    Reward.hospitalised: lambda mdp: mdp.CountHospitalisedCases(),
    #
    Reward.total_infected: lambda mdp: mdp.GetTotalInfected(),
    Reward.total_hospitalised: lambda mdp: mdp.GetTotalHospitalised(),
    Reward.total_at_risk: lambda mdp: mdp.GetTotalInfected(),
    Reward.total_at_risk_hosp: lambda mdp: mdp.GetTotalHospitalised(),
}


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
        reward: (Optional) Which metric from stride to use as the reward, default is
            the total number of infections.
        reward_type: (Optional) How to process the reward signal before providing it to the agent.
            Accepted values:
                - 'neg':  returns the negative of the reward signal
                - 'norm': normalise the reward based on the entire population
                -  None:  leaves the reward unchanged
        mRNA_properties: The properties of the mRNA vaccines
        adeno_properties: The properties of the Adeno vaccines
        action_wrapper: (Optional) The action wrapper to use to translate arms into vaccination requests.
    """
    def __init__(self, states=False, seed=0, episode_duration=6 * 30, step_size=2 * 30,
                 config_file="./config/run_default.xml", available_vaccines: VaccineSupply = ConstantVaccineSupply(),
                 reward=Reward.total_infected, reward_type=None, reward_factor=1,
                 mRNA_properties: stride.VaccineProperties = stride.LinearVaccineProperties("mRNA vaccine", 0.95, 0.95, 1.00, 42),
                 adeno_properties: stride.VaccineProperties = stride.LinearVaccineProperties("Adeno vaccine", 0.67, 0.67, 1.00, 42),
                 action_wrapper: Type[ActionWrapper] = ActionWrapper):
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

        # Vaccine properties
        self.mRNA_properties = mRNA_properties
        self.adeno_properties = adeno_properties
        # Action space for an action
        self.action_wrapper = action_wrapper(available_vaccines)
        self.nr_arms = len(stride.AllVaccineTypes) ** len(stride.AllAgeGroups)

        # Reward
        self.reward = reward
        self.reward_type = reward_type
        self.reward_factor = reward_factor
        self._population_size = 0
        self._age_groups_sizes = {}
        self._at_risk = 0

        # The internal state and timestep
        self.states = states
        self._state = None
        self._timestep = 0
        self._e = 0

        self._x = []

        self.first = True

    def reset(self, seed=None, output_dir=None, output_prefix=None):
        """Reset the environment and return the initial state (or None if no states are used)."""
        if self._timestep != 0:
            self._timestep = 0
            self._mdp.End()
            self._e += 1
            self._mdp.ClearSimulation()

        # Measure memory usage from python
        x = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        self._x.append(x)

        output_dir = output_dir if output_dir is not None else ""
        output_prefix = output_prefix if output_prefix is not None else ""
        seed = seed if seed is not None else self.seed
        # Create a new simulation
        self._mdp.Create(self.config_file, self.mRNA_properties, self.adeno_properties,
                         seed, output_dir, output_prefix)
        self._population_size = self._mdp.GetPopulationSize()
        self._age_groups_sizes = self._mdp.GetAgeGroupSizes()
        self._at_risk = self._mdp.GetAtRisk()
        print("at risk:", self._at_risk)
        print(self._age_groups_sizes)
        return None

    def close_x(self):
        for i, x in enumerate(self._x):
            print(f"Resources: {x} bytes ({round(x / 1024, 2)} kB, {round(x / 1024 ** 2, 2)} MB, {round(x / 1024 ** 3, 2)} GB)")

    def close(self):
        """Signal the simulator to end the simulation and its own processes."""
        self._mdp.End()

    def step(self, action=None):
        """Perform a step of step_size in the simulation.

        Args:
            action: The action given to vaccinate the age groups with a vaccine type.
                e.g. [VaccineType.noVaccine VaccineType.noVaccine VaccineType.adeno VaccineType.mRNA VaccineType.adeno]

        Returns:
            state, reward, done, info - feedback from the interaction with the environment.
        """
        print(f"Chosen action {action}")
        # print(f"Population size: {self._population_size}")
        # print(f"Age groups {self._mdp.GetAgeGroupSizes()}")
        state = reward = None
        info = {}
        # Execute the action to vaccinate and simulate for as many days as required
        for t in range(self.step_size):
            # If an action is given (!= None), vaccinate
            if action is not None:
                combined_action = self.action_wrapper.get_combined_action(action, self._timestep + t,
                                                                          self._population_size,
                                                                          self._mdp.GetAgeGroupSizes())
                self._vaccinate(t, action, combined_action)
            # Simulate the day and get the reward
            self._mdp.SimulateDay()
            reward = self.get_reward()

            # print(f"infected: {self._mdp.CountInfectedCases()}, exposed: {self._mdp.CountExposedCases()}, "
            #       f"infectious: {self._mdp.CountInfectiousCases()}, symptomatic: {self._mdp.CountSymptomaticCases()}, "
            #       f"hospitalised: {self._mdp.CountHospitalisedCases()}, total hosp.: {self._mdp.GetTotalHospitalised()}",
            #       f"at risk at start: {self._at_risk}, (temp) reward: {self._transform_reward(reward)}")
            # print(f"Unvaccinated age groups:", self._mdp.GetAgeGroupSizes())
            # print(f"Vaccinated age groups:", self._mdp.GetVaccinatedAgeGroups())

        # Transform the reward as requested
        reward = self._transform_reward(reward)
        # Another timestep has passed
        self._timestep += 1
        # The episode is done once we reach the episode duration
        done = self._timestep >= self.steps_per_episode
        # Give feedback
        return state, reward, done, info

    def _vaccinate(self, t, action, combined_action):
        fn = f"vaccine_distributions_{action}.csv"
        vts = [v.name for v in stride.AllVaccineTypes]
        ags = [g.name for g in stride.AllAgeGroups]
        combos = []
        for g in ags:
            for v in vts:
                combos.append(f"{g} - {v}")
        fields = ["day", *combos]

        with open(fn, mode="a") as file:
            w = csv.DictWriter(file, fields)
            if self.first:
                w.writeheader()
                self.first = False

            days_actions = {"day": t}
            for action in combined_action:
                self._mdp.Vaccinate(*action)

                n, g, v = action
                days_actions[f"{g.name} - {v.name}"] = n
            w.writerow(days_actions)

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

    def _transform_reward(self, num):
        # Consider entire population or only susceptible/not immune
        size = self._at_risk if self.reward in [Reward.total_at_risk, Reward.total_at_risk_hosp] \
            else self._population_size
        if self.reward_type == "neg":
            return -num * self.reward_factor
        elif self.reward_type == "norm":
            return (size - num) / size * self.reward_factor
        else:
            return num * self.reward_factor

    def get_reward(self):
        """Get the reward from the MDP"""
        return _reward_mapping[self.reward](self._mdp)


class StrideGroundTruthEnv(Env):
    def __init__(self, use_inf=True, reward_type="norm", reward_factor=1, seed=0):
        # Super call
        super(StrideGroundTruthEnv, self).__init__(seed)
        # Set the seed
        self.use_inf = use_inf
        self.reward_factor = reward_factor
        self.reward_type = reward_type
        self.seed = seed
        random.seed(self.seed)
        np.random.seed(self.seed)
        self.rng = np.random.default_rng(seed=seed)

        # The distributions file
        self.config_file = f"./envs/stride_env/real-distributions/{'inf' if self.use_inf else 'hosp'}.csv"
        if not os.path.isfile(self.config_file):
            self.config_file = "../" + self.config_file
        self.rewards = {}
        self._distributions = {}
        self._load_distributions()
        self.action_wrapper = NoWasteActionWrapper()
        self.nr_actions = self.action_wrapper.num_actions

    def _load_distributions(self):
        with open(self.config_file, mode="r") as csv_file:
            reader = csv.reader(csv_file)
            for line in reader:
                # Arm
                arm = int(line[0])
                rewards = [float(r) for r in line[1:]]
                if self.reward_type == "neg":
                    at_risk_pop = 8626594
                    rewards = [int(-(1 - r) * at_risk_pop) for r in rewards]
                rewards = [r * self.reward_factor for r in rewards]

                self.rewards[arm] = rewards
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
        state = None
        done = True
        info = {}
        # Sample a reward from the given action
        # rewards = self.rewards[action]
        # reward = self.rng.choice(rewards)
        mean, std_dev = self._distributions[action]
        reward = self.rng.normal(mean, std_dev) * self.reward_factor
        # Give feedback
        return state, reward, done, info
