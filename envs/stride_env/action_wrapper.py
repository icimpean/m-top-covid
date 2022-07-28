# noinspection PyUnresolvedReferences
import pylibstride as stride

import csv
import numpy as np
import random
from itertools import product, groupby

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
        new_action = [f"{g.name}: {a.name}" for a, g in zip(action, stride.AllAgeGroups)]
        return new_action

    def get_vaccine_names(self, arm):
        action = self._all_actions[arm]
        new_action = [f"NA" if a == stride.VaccineType.noVaccine else f"{a.name}" for a, g in
                      zip(action, stride.AllAgeGroups)]
        return new_action

    def get_combined_action(self, arm, day, pop_size, group_sizes):
        """Get the combined action, consisting of one action per age group that needs vaccinating.

        Args:
            arm: (Int) The arm to translate into a combined action.
            day: (List[Int]) The day of the simulation to get the available vaccines for.
            pop_size: The population size to get vaccines for.
            group_sizes: The number of people per group.

        Returns:
            A list of (availableVaccines, ageGroup, vaccineType) tuples,
                each representing a call to vaccinate.
        """
        vaccine_per_group = self.get_raw_action(arm)
        return self._divide_vaccines_day(vaccine_per_group, day, pop_size, group_sizes)

    def _divide_vaccines_day(self, vaccine_per_group, day, pop_size, group_sizes):
        available_vaccines = self.available_vaccines.get_available_vaccines([day], pop_size)[0]
        actions = self._divide_vaccines(vaccine_per_group, available_vaccines, group_sizes)
        return actions

    def get_combined_action_old(self, arm, days, pop_size, group_sizes):
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
        sum_total = sum(group_sizes.values())
        if sum_total == 0:
            # There is nobody left to vaccinate at all
            print("EVERYONE IS VACCINATED")
            return []

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
        remainder_actions = []
        for v_type, groups in occurrences.items():
            # No group gets a v_type vaccine
            if len(groups) == 0:
                continue
            # Get the number of available vaccines for the given vaccine type
            available = available_vaccines[v_type]

            # If all groups for the given vaccine type have been vaccinated
            sizes = [group_sizes[g] for g in groups]
            if sum(sizes) == 0:
                print(f"NO INDIVIDUALS left in the groups {groups} to vaccinate:")
                # => give vaccines to other groups
                remainder_actions = ActionWrapper._divide_remainder(available, v_type, vaccine_per_group, group_sizes)
                print(f"\tsplitting remainder ({available}): {remainder_actions}")

            # Only one group gets a vaccine
            elif len(groups) == 1:
                using_vaccines = min(available, group_sizes[groups[0]])
                divided_vaccines[v_type].append(using_vaccines)
                overflow = available - using_vaccines
                if using_vaccines < available:
                    # => give vaccines to other groups
                    remainder_actions = ActionWrapper._divide_remainder(overflow, v_type,
                                                                        vaccine_per_group, group_sizes)
                    print(f"AGE GROUP {groups[0]}")
                    print(f"\tsplitting overflow 1 ({overflow}): {remainder_actions}")

            # Two or more groups share a vaccine
            else:
                # Divide the available vaccines into as many age groups that require them
                counts, overflow = ActionWrapper._divide(available, groups, group_sizes)
                divided_vaccines[v_type] = counts
                if overflow > 0:
                    # => give vaccines to other groups
                    remainder_actions = ActionWrapper._divide_remainder(overflow, v_type,
                                                                        vaccine_per_group, group_sizes)
                    print(f"AGE GROUPS {groups}")
                    print(f"\tsplitting overflow 2 ({overflow}): {remainder_actions}")

        # The vaccine types, age groups and number of vaccines per group need to be translated into actions
        # An action to vaccinate for stride is characterised by: availableVaccines, ageGroup, vaccineType
        actions = []
        for v_type, groups in occurrences.items():
            divided = divided_vaccines[v_type]
            for group, div in zip(groups, divided):
                action = (div, stride.AllAgeGroups[group], v_type)  # availableVaccines, ageGroup, vaccineType
                actions.append(action)
        # Add any remainder actions at the end => priority to following arm before dividing remainder
        actions.extend(remainder_actions)

        # Avoid 0 or negative -1 vaccine quantities due to rounding in divisions
        for action in actions.copy():
            div, group, v_type = action
            if div <= 0:
                actions.remove(action)
                print("Removed invalid action", action)

        # Return the actions
        return actions

    @staticmethod
    def _divide_remainder(available, vaccine_type, vaccine_per_group, group_sizes):
        # Priority
        # 1. Give vaccine to requested age group (handled by _divide_vaccines)
        # 2. Give vaccines to other groups using the same vaccine type (handled by _divide_vaccines)
        # 3. Give vaccine to the other age groups with no vaccines
        # 4. Give to any age groups with non-vaccinated individuals

        # 3. Get all other age groups that don't receive any vaccine
        remainder_groups = []
        for idx, v_type in enumerate(vaccine_per_group):
            group = stride.AllAgeGroups[idx]
            # No vaccine groups get remainder
            if v_type == stride.VaccineType.noVaccine and group_sizes[group] != 0:
                remainder_groups.append(idx)
        # 4. any group with not vaccinated individuals
        if len(remainder_groups) == 0:
            for idx, v_type in enumerate(vaccine_per_group):
                group = stride.AllAgeGroups[idx]
                # Not groups with the current vaccine type
                if v_type != vaccine_type and group_sizes[group] != 0:
                    remainder_groups.append(idx)

        # Only one group gets a vaccine
        if len(remainder_groups) == 1:
            # Only one group gets a vaccine
            divided_vaccines = [available]
        # Divide over all remainder groups
        else:
            counts, overflow = ActionWrapper._divide(available, remainder_groups, group_sizes)
            divided_vaccines = counts

        # Create actions
        actions = []
        for group, div in zip(remainder_groups, divided_vaccines):
            action = (div, stride.AllAgeGroups[group], vaccine_type)  # availableVaccines, ageGroup, vaccineType
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
        # Can only give out as many vaccines as there are people
        remaining = min(count, total_size)
        overflow = count - remaining
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
        return new_counts, overflow

    def export_csv(self, file_path):
        """Export the arms from the action wrapper as a CVS file"""
        age_groups = [g.name for g in stride.AllAgeGroups]
        fields = ["Arm", *age_groups]
        with open(file_path, mode="w") as file:
            writer = csv.writer(file)
            writer.writerow(fields)
            for arm in range(self.num_actions):
                row = [arm, *self.get_vaccine_names(arm)]
                writer.writerow(row)

    def group_actions_v_type(self):
        """Group actions into smaller groups which have similar vaccine strategies"""
        groups_per_type = []

        action_numbers = [(arm, self._nums(self._all_actions[arm])) for arm in range(self.num_actions)]
        action_numbers = sorted(action_numbers, key=lambda an: an[1])
        groups = groupby(action_numbers, key=lambda an: an[1])

        for n, g in enumerate(groups):
            # print("group", n)
            count, group = g
            groups_per_type.append(list(group))
            # print(list(group))
        return groups_per_type

    @staticmethod
    def _nums(action):
        nums = [0 for v_type in stride.AllVaccineTypes]
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
        nums = ["0" for age in stride.AllAgeGroups]
        for i, v in enumerate(action):
            if v != stride.VaccineType.noVaccine:
                nums[i] = "1"
        return "".join(nums)


class NoWasteActionWrapper(ActionWrapper):
    """Action wrapper for not wasting any vaccine types.

    Each arm uses both the mRNA and Adeno vaccine type."""

    @staticmethod
    def _create_action_array():
        full_actions = ActionWrapper._create_action_array()
        # Disregard all actions that don't contain both vaccine types
        new_actions = [action for action in full_actions if NoWasteActionWrapper._no_waste(action)]
        # Return the actions
        return np.array(new_actions)

    @staticmethod
    def _no_waste(action):
        found = {v_type: False for v_type in NoWasteActionWrapper._v_types}
        for v in action:
            if v != stride.VaccineType.noVaccine:
                found[v] = True
        return all(found.values())


def print_arm(aw, arm):
    """Print the arm as the vaccine types per age group."""
    return aw.get_pretty_raw_action(arm)


def latex_table_arms(aw, arms):
    """Print the rows of a LaTeX table to display the underlying arms' allocations."""
    for a in arms:
        print(f"{a} & " + " & ".join(aw.get_vaccine_names(a)), "\\\\\\hline")


def latex_all_arms(aw):
    # new_action = [f"{g.name}: {a.name}" for a, g in zip(action, stride.AllAgeGroups)]
    groups = [g.name for g in stride.AllAgeGroups]
    print(f"Arm & {' & '.join(groups)}\\\\\\hline")

    for a in range(aw.num_actions):
        print(f"{a} & " + " & ".join(aw.get_vaccine_names(a)), "\\\\\\hline")
