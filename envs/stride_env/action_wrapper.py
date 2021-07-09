# noinspection PyUnresolvedReferences
import pylibstride as stride

import numpy as np
import random
from itertools import product

from resources.vaccine_supply import ConstantVaccineSupply, VaccineSupply


class ActionWrapper(object):
    """An action wrapper for STRIDE, translating an agent's action to a collection of actions.

    Attributes:
        available_vaccines: (Optional) A VaccineSupply schedule to retrieve the
            available vaccines per day from. Defaults to: ConstantVaccineSupply().
    """
    # All vaccine types, except noVaccine
    _v_types = stride.AllVaccineTypes[1:]

    def __init__(self, available_vaccines: VaccineSupply = ConstantVaccineSupply()):
        # Store the arguments
        self.available_vaccines = available_vaccines
        # Action space
        self._all_actions = self._create_action_array()

    def get_raw_action(self, arm):
        """Get the raw action, corresponding to a vaccine type per age group.

        Args:
            arm: (Int) The arm to translate into a combined action.

        Returns:
            An array of VaccineTypes per age group.
        """
        return self._all_actions[arm]

    def get_pretty_raw_action(self, arm):
        """Get the action for printing"""
        action = self._all_actions[arm]
        new_action = [f"{g.name}: {a.name}" for a, g in zip(action, stride.AllAgeGroups)]
        return new_action

    def get_combined_action(self, arm, days, pop_size, group_sizes):
        """Get the combined action, consisting of one action per age group that needs vaccinating.

        Args:
            arm: (Int) The arm to translate into a combined action.
            days: (List[Int]) The days of the simulation to get the available vaccines for.
            pop_size: The population size to get vaccines for.
            group_sizes: The number of people per group.

        Returns:
            A list of (availableVaccines, ageGroup, vaccineType) tuples,
                each representing a call to vaccinate.
        """
        vaccine_per_group = self.get_raw_action(arm)
        return self._divide_vaccines_days(vaccine_per_group, days, pop_size, group_sizes)

    def _divide_vaccines_days(self, vaccine_per_group, days, pop_size, group_sizes):
        available_vaccines = self.available_vaccines.get_available_vaccines(days, pop_size)
        actions = [self._divide_vaccines(vaccine_per_group, availability, group_sizes)
                   for availability in available_vaccines]
        return actions

    @staticmethod
    def _divide_vaccines(vaccine_per_group, available_vaccines, group_sizes):
        # Collect the age groups per vaccine type
        occurrences = {v_type: [] for v_type in ActionWrapper._v_types}
        for idx, v_type in enumerate(vaccine_per_group):
            # Ignore no vaccines (=> no vaccination needs to happen)
            if v_type == stride.VaccineType.noVaccine:
                continue
            # Assign this group's index to the occurrences
            occurrences[v_type].append(idx)

        # For each of the vaccines, divide the available vaccines per age group that uses them
        divided_vaccines = {v_type: [] for v_type in ActionWrapper._v_types}
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
                counts = ActionWrapper._divide(available, groups, group_sizes)
                divided_vaccines[v_type] = counts

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
        return np.array(list(product(stride.AllVaccineTypes, repeat=len(stride.AllAgeGroups))))

    @staticmethod
    def _divide(count, groups, group_sizes):
        """Divide the vaccines proportionally over the given groups"""
        total_size = sum([group_sizes[g] for g in groups])
        remaining = count
        new_counts = []
        # Decide the available vaccines based on the group size
        for group in groups:
            group_count = round(count * group_sizes[group] / total_size)
            # print(f"Assigning {group_count} vaccines to group {group}. Fraction: {group_sizes[group] / total_size}")
            new_counts.append(group_count)
            remaining -= group_count
        # Divide the remainder evenly, & randomly among the groups
        # print(f"count: {count}, groups: {groups}, group_sizes: {group_sizes}, remaining: {remaining}")
        c = -1 if remaining < 0 else 1
        rem_groups = list(range(len(groups)))
        random.shuffle(rem_groups)
        rem_groups = rem_groups[:abs(remaining)]
        for i in rem_groups:
            new_counts[i] += c
        #
        return new_counts

    @staticmethod
    def _divide_old(count, divisor):
        """Divide equally, regardless of the group sizes"""
        d, m = divmod(count, divisor)
        new_counts = [d for _ in range(divisor)]
        for i in range(m):
            new_counts[i] += 1
        # Random shuffle
        random.shuffle(new_counts)
        return new_counts


def print_arm(arm):
    """Print the arm as the vaccine types per age group."""
    return ActionWrapper().get_pretty_raw_action(arm)


if __name__ == '__main__':
    aw = ActionWrapper()
    print(aw.get_pretty_raw_action(2))

