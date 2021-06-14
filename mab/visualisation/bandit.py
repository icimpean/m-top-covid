import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from envs.stride_env.action_wrapper import ActionWrapper
from loggers.bandit_logger import BanditLogger
from mab.visualisation import Visualisation
from mab.visualisation.stride import StrideVisualisation


class BanditVisualisation(Visualisation):
    """The visualisation for a bandit experiment."""

    def __init__(self):
        # Super call
        super(BanditVisualisation, self).__init__()
        #
        self.arms = None
        self.rewards_per_arm = None
        self.episodes_per_arm = None
        #
        self.stride_vis = StrideVisualisation()

    def load_file(self, bandit_file, requested_arms=None, sort=True):
        with open(bandit_file, "r") as file:
            rewards = {}
            episodes = {}
            arms = set()
            logger = BanditLogger()

            entry_fields = logger.entry_fields
            reader = csv.DictReader(file, fieldnames=entry_fields)
            skip = True
            for entry in reader:
                if skip:
                    skip = False
                    continue

                # Get the arm, episode and given reward
                arm = int(entry[logger.arm])
                reward = float(entry[logger.reward])
                episode = int(entry[logger.episode])

                arms.add(arm)
                if rewards.get(arm) is None:
                    rewards[arm] = [reward]
                else:
                    rewards[arm].append(reward)
                if episodes.get(arm) is None:
                    episodes[arm] = [episode]
                else:
                    episodes[arm].append(episode)

            if sort:
                arms = sorted(arms)

            # Calculate the averages per arm
            min_reward = np.inf
            max_reward = -np.inf
            for arm, arm_rewards in rewards.items():
                rewards[arm] = np.mean(arm_rewards)
                if requested_arms is None or arm in requested_arms:
                    min_reward = min(min_reward, rewards[arm])
                    max_reward = max(max_reward, rewards[arm])

        # Set the data
        self.arms = arms
        self.rewards_per_arm = (rewards, min_reward, max_reward)
        self.episodes_per_arm = episodes

    def plot_average_reward_per_arm(self, requested_arms=None, show=True, save_file=None):
        # Get the data
        rewards, min_reward, max_reward = self.rewards_per_arm
        plot_arms = self.arms if requested_arms is None else requested_arms
        x_ticks = (range(len(plot_arms)), plot_arms)

        # Set up the plot
        self._plot_text(title="Average reward per arm", x_label="Arm", y_label="Reward", legend=None, x_ticks=x_ticks)
        # Center the graph around the y_values to plot
        plt.ylim(self._center_y_lim(min_reward, max_reward))

        # Plot the data
        self._bar_plot(plot_arms, get_y=lambda x: rewards[x], colors=self._default_color)
        # Show, save and close the plot
        self._show_save_close(show, save_file)

    def plot_top(self, top_m=None, best=True, action_wrapper: ActionWrapper = None, show=True, save_file=None):
        # Get the data
        rewards, min_reward, max_reward = self.rewards_per_arm
        if top_m is None:
            top_m = len(self.arms)

        # Get the top best/worst
        sorted_rewards = sorted(rewards.items(), key=lambda ar: ar[1], reverse=best)
        sorted_rewards = sorted_rewards[:top_m]
        min_reward = sorted_rewards[-1 if best else 0][1]
        max_reward = sorted_rewards[0 if best else -1][1]
        arms = [a for a, r in sorted_rewards]
        x_ticks = (range(top_m), arms)

        # Set up the plot
        self._plot_text(title=f"Top {top_m} {'best' if best else 'worst'} arms",
                        x_label="Arm", y_label="Average Reward", legend=None, x_ticks=x_ticks)
        # Center the graph around the y_values to plot
        plt.ylim(self._center_y_lim(min_reward, max_reward))

        # Plot the data
        self._bar_plot(sorted_rewards, get_y=lambda x: x[1], colors=self._default_color)
        # Show, save and close the plot
        self._show_save_close(show, save_file)

        if action_wrapper is not None:
            self._print_top_arms(sorted_rewards, best, action_wrapper)

    @staticmethod
    def _print_top_arms(top_arms, best, action_wrapper=None):
        print(f"Top {len(top_arms)} {'best' if best else 'worst'} arms")
        for arm, reward in top_arms:
            line = f"Arm {arm}"
            if action_wrapper is not None:
                line += " " + str(action_wrapper.get_pretty_raw_action(arm))
            line += f": reward {reward}"
            print(line)

    def plot_stride_values(self, stride_csv_directory, file_name, requested_arms=None,
                           show=True, save_file=None):
        # Plot the requested arms
        plot_arms = self.arms if requested_arms is None else requested_arms
        name = self.stride_vis.files_names[file_name]

        # Set up the plot
        self._plot_text(title=f"Number of {name}", x_label="Days", y_label=name, legend=None)

        # Get the data for each of the files:
        min_reward = np.inf
        max_reward = -np.inf
        # Plot each arm
        for arm in plot_arms:
            # Average per arm
            average = []
            # For each episode
            for episode in self.episodes_per_arm[arm]:
                stride_csv_file = Path(stride_csv_directory) / str(episode) / file_name
                y_values, min_y, max_y = self.stride_vis.load_file(stride_csv_file)
                average.append(y_values)
                min_reward = min(min_reward, min_y)
                max_reward = max(max_reward, max_y)
            # Plot the average
            average = np.mean(average, axis=0)
            plt.plot(range(len(average)), average, label=arm)

        plt.legend()
        # Show, save and close the plot
        self._show_save_close(show, save_file)

    def plot_single_arm(self, arm, stride_csv_directory, file_name=None, plot_average=False, show=True, save_file=None):

        # Plot the requested arm
        file_names = [file_name] if file_name is not None else [self.stride_vis.files_names.keys()]
        names = [self.stride_vis.files_names[fn] for fn in file_names]
        name = self.stride_vis.files_names[file_name] if file_name is not None else "Individuals"
        # Set up the plot
        self._plot_text(title=f"Number of {name} for arm {arm}", x_label="Days", y_label=name, legend=None)

        lw = 0.5 if plot_average else 1.0
        alpha = 0.4 if plot_average else 1.0
        colors = ["blue", "red", "green", "black", "orange", "purple"]

        for idx, n in enumerate(names):
            # Average for arm
            average = []
            # For each episode
            for episode in self.episodes_per_arm[arm]:
                stride_csv_file = Path(stride_csv_directory) / str(episode) / file_names[idx]
                y_values, min_reward, max_reward = self.stride_vis.load_file(stride_csv_file)
                average.append(y_values)
                plt.plot(range(len(y_values)), y_values, lw=lw, color=colors[idx], alpha=alpha)
            # Plot the average
            if plot_average:
                average = np.mean(average, axis=0)
                plt.plot(range(len(average)), average, lw=lw * 3, color=colors[idx], label=n)

            # Show, save and close the plot
            self._show_save_close(show, save_file)

    def plot_arm_frequency(self, requested_arms=None, best=True, show=True, save_file=None):
        # Get the data
        arms = self.arms if requested_arms is None else requested_arms
        episodes = [len(self.episodes_per_arm[arm]) for arm in arms]

        # Get the top best/worst
        sorted_episodes = sorted(episodes, reverse=best)
        min_count = sorted_episodes[-1 if best else 0]
        max_count = sorted_episodes[0 if best else -1]

        x_ticks = (range(len(arms)), arms)

        # Set up the plot
        self._plot_text(title=f"Arm counts for {'most' if best else 'least'} pulled arms",
                        x_label="Arm", y_label="Episodes", legend=None, x_ticks=x_ticks)
        # Center the graph around the y_values to plot
        plt.ylim(self._center_y_lim(min_count, max_count))

        # Plot the data
        self._bar_plot(sorted_episodes, colors=self._default_color)
        # Show, save and close the plot
        self._show_save_close(show, save_file)

    def get_top_arms(self, top_rewards=True, best=True):
        # Top based on rewards
        data = self.rewards_per_arm[0] if top_rewards else self.episodes_per_arm
        if not top_rewards:
            data = {a: len(e) for a, e in data.items()}

        # Get the top best/worst
        sorted_top = sorted(data.items(), key=lambda x: x[1], reverse=best)
        min_top = sorted_top[-1 if best else 0][1]
        max_top = sorted_top[0 if best else -1][1]

        return sorted_top, min_top, max_top
