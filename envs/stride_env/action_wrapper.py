# noinspection PyUnresolvedReferences
import pylibstride as stride

import numpy as np
import random
from itertools import product


class ActionWrapper(object):
    """An action wrapper for STRIDE, translating an agent's action to a collection of actions."""
    def __init__(self, step_size, episode_duration, vaccine_availability=None):
        # Store the arguments
        self.step_size = step_size
        self.episode_duration = episode_duration
        self.vaccine_availability = vaccine_availability
        # TODO: vaccine availability loaded from file, per day
        if self.vaccine_availability is None:
            self.vaccine_availability = {
                v_type: 1000 for v_type in stride.AllVaccineTypes if v_type != stride.VaccineType.noVaccine
            }
        # Action space
        self._vaccine_options = stride.AllVaccineTypes
        self._age_groups = stride.AllAgeGroups
        self.decisions = self.episode_duration // self.step_size
        self._all_actions = self._create_action_array()

    def get_raw_action(self, arm):
        """Get the raw action, corresponding to a vaccine type per age group"""
        return self._all_actions[arm]

    def get_combined_action(self, arm):
        """Get the combined action, consisting of one action per age group that needs vaccinating.

        Args:
            arm: (Int) The arm to translate into a combined action.

        Returns:
            A list of (availableVaccines, ageGroup, vaccineType) tuples, each representing a call to vaccinate.
        """
        # TODO: 3 decisions
        vaccine_per_group = self.get_raw_action(arm)
        return self._divide_vaccines(vaccine_per_group, self.vaccine_availability)

    @staticmethod
    def _divide_vaccines(vaccine_per_group, available_vaccines):
        # Collect the age groups per vaccine type
        occurrences = {
            v_type: [] for v_type in stride.AllVaccineTypes if v_type != stride.VaccineType.noVaccine
        }
        for idx, v_type in enumerate(vaccine_per_group):
            # Ignore no vaccines (=> no vaccination needs to happen)
            if v_type == stride.VaccineType.noVaccine:
                continue
            # Assign this group's index to the occurrences
            occurrences[v_type].append(idx)

        # For each of the vaccines, divide the available vaccines per age group that uses them
        divided_vaccines = {
            v_type: [] for v_type in stride.AllVaccineTypes if v_type != stride.VaccineType.noVaccine
        }
        for v_type, groups in occurrences.items():
            # No group gets a v_type vaccine
            if len(groups) == 0:
                continue
            # Get the number of available vaccines for the given vaccine type
            # TODO: varying availability per day per vaccine type
            available = available_vaccines[v_type]
            # Only one group gets a vaccine
            if len(groups) == 1:
                divided_vaccines[v_type].append(available)
            # Two or more groups share a vaccine
            else:
                # Divide the available vaccines into as many age groups that require them
                count = len(occurrences[v_type])
                div, rem = divmod(available, count)
                per_group = [div] * count
                # For uneven division, randomly assign remainder
                if rem != 0:
                    sampled = random.sample(range(count), rem)  # TODO random.seed consistency
                    for s in sampled:
                        per_group[s] += rem
                divided_vaccines[v_type] = per_group

        # The vaccine types, age groups and number of vaccines per group need to be translated into actions
        # An action to vaccinate for stride is characterised by: availableVaccines, ageGroup, vaccineType
        actions = []
        for v_type, groups in occurrences.items():
            divided = divided_vaccines[v_type]
            for group, div in zip(groups, divided):
                action = (div, stride.AllAgeGroups[group], v_type)  # availableVaccines, ageGroup, vaccineType
                actions.append(action)
        # Return the actions
        return actions

    def _create_action_array(self):
        # Vaccine types (+ no vaccine) per age group
        vaccines_per_age_group = np.array(list(product(stride.AllVaccineTypes, repeat=len(stride.AllAgeGroups))))
        # Total # TODO: 3 decisions
        total_vaccines = np.full(shape=(self.decisions, *vaccines_per_age_group.shape),
                                 fill_value=vaccines_per_age_group)

        print(total_vaccines)

        return vaccines_per_age_group


if __name__ == '__main__':  # TODO: remove
    print(3 ** 5, 3 * 3 ** 5)

    aw = ActionWrapper(step_size=1, episode_duration=3)
    a = 23#230

    print("Raw action:", aw.get_raw_action(a))
    print("Combined action:", aw.get_combined_action(a))
