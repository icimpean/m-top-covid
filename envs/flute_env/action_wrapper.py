from enum import Enum
from itertools import product, groupby

import numpy as np


class AgeGroup(Enum):
    preSchoolChildren = 0  # 0-4 years old
    schoolAgeChildren = 1  # 5-18 years old
    youngAdults = 2  # 19-29 years old
    adults = 3  # 30-64 years old
    elderly = 4  # 65+ years old


class VaccineType(Enum):
    noVaccine = 0
    vaccine = 1


class ActionWrapper(object):
    """An action wrapper for Flute environment, translating an agent's action to a collection of actions."""
    def __init__(self):
        # Action space
        self._all_actions = self._create_action_array()
        self.num_actions = len(self._all_actions)

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
        new_action = [f"{g.name}: {a.name}" for a, g in zip(action, AgeGroup)]
        return new_action

    def get_vaccine_names(self, arm):
        action = self._all_actions[arm]
        new_action = [f"NA" if a == VaccineType.noVaccine else f"{a.name}" for a, g in
                      zip(action, AgeGroup)]
        return new_action

    @staticmethod
    def _create_action_array():
        # Vaccine types (+ no vaccine) per age group
        return np.array(list(product(VaccineType, repeat=len(AgeGroup))))

    def group_actions_v_type(self):
        """Group actions into smaller groups which have similar vaccine strategies"""
        groups_per_type = []

        action_numbers = [(arm, self._nums(self._all_actions[arm])) for arm in range(self.num_actions)]
        action_numbers = sorted(action_numbers, key=lambda an: an[1])
        groups = groupby(action_numbers, key=lambda an: an[1])

        for n, g in enumerate(groups):
            count, group = g
            groups_per_type.append(list(group))
        return groups_per_type

    @staticmethod
    def _nums(action):
        nums = [0 for v_type in VaccineType]
        for g in action:
            nums[g] += 1
        return nums

    def group_actions_age(self):
        """Group actions into smaller groups which vaccinate the same groups"""
        groups_per_type = []

        action_numbers = [(arm, self._nums2(self._all_actions[arm])) for arm in range(self.num_actions)]
        action_numbers = sorted(action_numbers, key=lambda an: an[1])
        groups = groupby(action_numbers, key=lambda an: an[1])

        for n, g in enumerate(groups):
            count, group = g
            groups_per_type.append(list(group))
        return groups_per_type

    @staticmethod
    def _nums2(action):
        nums = ["0" for age in AgeGroup]
        for i, v in enumerate(action):
            if v != VaccineType.noVaccine:
                nums[i] = "1"
        return "".join(nums)


def print_arm(aw, arm):
    """Print the arm as the vaccine types per age group."""
    return aw.get_pretty_raw_action(arm)


def latex_table_arms(aw, arms):
    """Print the rows of a LaTeX table to display the underlying arms' allocations."""
    for a in arms:
        print(f"{a} & " + " & ".join(aw.get_vaccine_names(a)), "\\\\\\hline")


if __name__ == '__main__':
    top = [8, 24,  9,  4, 28, 12, 25, 31, 13, 7]
    aw = ActionWrapper()

    res = [aw.get_pretty_raw_action(a) for a in top]
    for r in res:
        print(r)
