# noinspection PyUnresolvedReferences
import pylibstride as stride

import numpy as np
import random
from itertools import product

from resources.vaccine_supply import ConstantVaccineSupply, VaccineSupply


class ActionWrapper(object):
    """An action wrapper for STRIDE, translating an agent's action to a collection of actions."""
    def __init__(self, available_vaccines: VaccineSupply = ConstantVaccineSupply()):
        # Store the arguments
        self.available_vaccines = available_vaccines
        # Action space
        self._all_actions = self._create_action_array()

    def get_raw_action(self, arm):
        """Get the raw action, corresponding to a vaccine type per age group"""
        return self._all_actions[arm]

    def get_combined_action(self, arm, days):
        """Get the combined action, consisting of one action per age group that needs vaccinating.

        Args:
            arm: (Int) The arm to translate into a combined action.
            days: (List[Int]) The days of the simulation to get the available vaccines for.

        Returns:
            A list of (availableVaccines, ageGroup, vaccineType) tuples, each representing a call to vaccinate.
        """
        vaccine_per_group = self.get_raw_action(arm)
        return self._divide_vaccines_days(vaccine_per_group, days)

    def _divide_vaccines_days(self, vaccine_per_group, days):
        available_vaccines = self.available_vaccines.get_available_vaccines(days)
        actions = [self._divide_vaccines(vaccine_per_group, availability) for availability in available_vaccines]
        return actions

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

    @staticmethod
    def _create_action_array():
        # Vaccine types (+ no vaccine) per age group
        vaccines_per_age_group = np.array(list(product(stride.AllVaccineTypes, repeat=len(stride.AllAgeGroups))))
        return vaccines_per_age_group


if __name__ == '__main__':  # TODO: remove

    aw = ActionWrapper()
    a = 230
    some_arms = [0, 1, 11, 23, 123, 230]

    # print("Raw action:", aw.get_raw_action(a))
    # print("Combined action:", aw.get_combined_action(a, [0]))

    for a in some_arms:
        print(aw.get_raw_action(a))
