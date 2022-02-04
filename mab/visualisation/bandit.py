import csv
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from envs.stride_env.action_wrapper import ActionWrapper, NoWasteActionWrapper
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
        self.times_per_arm = None
        #
        self.stride_vis = StrideVisualisation()

    def load_file(self, bandit_file, requested_arms=None, sort=True):
        self.load_files([bandit_file], requested_arms, sort)

    def load_files(self, bandit_files, requested_arms=None, sort=True, combine_episodes=True):
        """Combine multiple bandit file results"""
        rewards = {}
        episodes = {}
        times = {}
        arms = set()
        logger = BanditLogger()
        entry_fields = logger.entry_fields
        last_episode = 0

        for bandit_file in bandit_files:
            with open(bandit_file, "r") as file:
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
                    arm_time = float(entry[logger.time])
                    # Update the episode number if merging multiple files
                    if combine_episodes:
                        episode += last_episode
                    if episode >= 100:  # TODO: temp skip
                        break

                    arms.add(arm)
                    if rewards.get(arm) is None:
                        rewards[arm] = [reward]
                        episodes[arm] = [episode]
                        times[arm] = [arm_time]
                    else:
                        rewards[arm].append(reward)
                        episodes[arm].append(episode)
                        times[arm].append(arm_time)
            last_episode = episode + 1

        if sort:
            arms = sorted(arms)
        # Calculate the averages per arm
        min_reward = np.inf
        max_reward = -np.inf
        min_time = np.inf
        max_time = -np.inf
        for arm, arm_rewards in rewards.items():
            rewards[arm] = arm_rewards
            if requested_arms is None or arm in requested_arms:
                min_reward = min(min_reward, *rewards[arm])
                max_reward = max(max_reward, *rewards[arm])
                min_time = min(min_time, *times[arm])
                max_time = max(max_time, *times[arm])

        # Set the data
        self.arms = arms
        self.rewards_per_arm = (rewards, min_reward, max_reward)
        self.episodes_per_arm = episodes
        self.times_per_arm = (times, min_time, max_time)

    def load_files_checkpoints(self, bandit_files, checkpoints, requested_arms=None, sort=True, combine_episodes=True):
        """Combine multiple bandit file results"""
        rewards = {}
        episodes = {}
        times = {}
        arms = set()
        logger = BanditLogger()
        entry_fields = logger.entry_fields
        last_episode = 0
        last_checkpoints = {}

        for bandit_file in (bandit_files + checkpoints):
            with open(bandit_file, "r") as file:
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
                    arm_time = float(entry[logger.time])
                    # Update the episode number if merging multiple files
                    if combine_episodes:
                        episode += last_episode

                    last_chpnt = last_checkpoints.get(arm)
                    if last_chpnt is not None and episode < last_chpnt:
                        continue
                    elif last_chpnt is not None and episode == last_chpnt:
                        last_checkpoints[arm] = episode
                        arms.add(arm)
                        rewards[arm][-1] = reward
                        episodes[arm][-1] = episode
                        times[arm][-1] = arm_time
                    else:
                        last_checkpoints[arm] = episode
                        arms.add(arm)
                        if rewards.get(arm) is None:
                            rewards[arm] = [reward]
                            episodes[arm] = [episode]
                            times[arm] = [arm_time]
                        else:
                            rewards[arm].append(reward)
                            episodes[arm].append(episode)
                            times[arm].append(arm_time)
            last_episode = episode + 1

        if sort:
            arms = sorted(arms)

        # Calculate the averages per arm
        min_reward = np.inf
        max_reward = -np.inf
        min_time = np.inf
        max_time = -np.inf
        for arm, arm_rewards in rewards.items():
            rewards[arm] = arm_rewards
            if requested_arms is None or arm in requested_arms:
                min_reward = min(min_reward, *rewards[arm])
                max_reward = max(max_reward, *rewards[arm])
                min_time = min(min_time, *times[arm])
                max_time = max(max_time, *times[arm])

        # Set the data
        self.arms = arms
        self.rewards_per_arm = (rewards, min_reward, max_reward)
        self.episodes_per_arm = episodes
        self.times_per_arm = (times, min_time, max_time)

    def rewards_to_cluster(self, save_file, requested_arms=None):  # TODO
        # noinspection PyUnresolvedReferences
        import pylibstride as stride
        from sklearn.preprocessing import OneHotEncoder
        from sklearn.cluster import AgglomerativeClustering
        from sklearn.decomposition import PCA

        rewards = self.rewards_per_arm[0]
        aw = NoWasteActionWrapper()
        # X = arm strategy
        # y = reward
        arms = []
        X = []
        y = []
        X_original = []
        # All vaccine types per age group

        all_features = [[v_type for v_type in stride.AllVaccineTypes] for age in stride.AllAgeGroups]
        encoder = OneHotEncoder(categories=all_features)
        fit_samples = [[v_type for age in stride.AllAgeGroups] for v_type in stride.AllVaccineTypes]
        encoder.fit(fit_samples)

        for arm, arm_rewards in rewards.items():
            strategy = aw.get_raw_action(arm)
            new_strategy = encoder.transform([strategy]).toarray()[0]
            arm_mean = np.mean(arm_rewards)
            arms.append(arm)
            X.append(new_strategy)
            y.append(arm_mean)
            X_original.append(strategy)

        linkage = "ward"
        # for linkage in ("ward", "average", "complete", "single"):
        clustering = AgglomerativeClustering(linkage=linkage,
                                             n_clusters=10, distance_threshold=None,
                                             # n_clusters=None, distance_threshold=5,
                                             )
        clustering.fit(X)

        unique_labels = []
        for c in clustering.labels_:
            if c not in unique_labels:
                unique_labels.append(c)
        print("nr clusters: ", clustering.n_clusters_)
        for arm, strategy, cluster in zip(arms, y, clustering.labels_):
            print(f"Arm: {arm}, cluster: {cluster}")

        X_pca = PCA(2).fit_transform(X)
        done = {}
        for (x, y), c in zip(X_pca, clustering.labels_):
            if done.get(c):
                color = done[c]
                l = None
                plt.scatter(x, y, c=color, label=l)
            else:
                l = f"cluster {c}"
                p = plt.scatter(x, y, label=l)
                plt.draw()
                col = p.get_facecolors()[-1].tolist()
                done[c] = col
        # plt.scatter(X_pca[:, 0], X_pca[:, 1], c=clustering.labels_)
        plt.title("Clusters")
        print("Labels:", clustering.labels_, unique_labels)
        # plt.legend([f"cluster {l}" for l in unique_labels])
        plt.legend()
        # plt.show()
        plt.savefig("temp_2d.png")

        X_pca = PCA(3).fit_transform(X_original)
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        done = {}
        for (x, y, z), c in zip(X_pca, clustering.labels_):
            if done.get(c):
                color = done[c]
                l = None
                ax.scatter(x, y, z, c=color, label=l)
            else:
                l = f"cluster {c}"
                p = ax.scatter(x, y, z, c=c, label=l)
                plt.draw()
                col = p.get_facecolors()[-1].tolist()
                done[c] = col
        # ax.scatter(X_pca[:, 0], X_pca[:, 1], X_pca[:, 2], c=clustering.labels_)
        plt.title("Clusters")
        plt.legend()
        # plt.show()
        plt.savefig("temp_3d.png")

    def rewards_to_cluster2(self, save_file, requested_arms=None):  # TODO
        # noinspection PyUnresolvedReferences
        import pylibstride as stride
        from sklearn.preprocessing import OneHotEncoder
        from sklearn.cluster import AgglomerativeClustering
        from sklearn.decomposition import PCA

        rewards = self.rewards_per_arm[0]
        aw = NoWasteActionWrapper()
        # X = arm strategy, y = reward
        arms = []
        X = []
        Y = []
        X_original = []
        #
        all_features = [[v_type for v_type in stride.AllVaccineTypes] for age in stride.AllAgeGroups]
        encoder = OneHotEncoder(categories=all_features)
        fit_samples = [[v_type for age in stride.AllAgeGroups] for v_type in stride.AllVaccineTypes]
        encoder.fit(fit_samples)

        for arm, arm_rewards in rewards.items():
            strategy = aw.get_raw_action(arm)
            arm_mean = np.mean(arm_rewards)
            arms.append(arm)

            new_strategy = encoder.transform([strategy]).toarray()[0]
            new_strategy = new_strategy.tolist() + [arm_mean]

            X.append(new_strategy)
            Y.append(arm_mean)
            X_original.append(strategy)

        linkage = "ward"
        # for linkage in ("ward", "average", "complete", "single"):
        clustering = AgglomerativeClustering(linkage=linkage,
                                             n_clusters=10, distance_threshold=None,
                                             # n_clusters=None, distance_threshold=5,
                                             )
        clusters = clustering.fit_predict(X)

        colors = plt.cm.tab20(clusters / clustering.n_clusters_)

        X_pca = PCA(2).fit_transform(X)

        fig = plt.figure()
        ax = fig.add_subplot()
        for clust in range(clustering.n_clusters_):
            x_pca = X_pca[clusters == clust]
            cols = colors[clusters == clust]
            c_arms = np.array(arms)[clusters == clust]
            done = False
            for arm, x, c in zip(c_arms, x_pca, cols):
                if arm > 179:
                    print("ERROR!", c_arms)
                label = None
                if not done:
                    label = f"Cluster {clust}"
                    done = True
                ax.plot(*x, marker=f"${arm:03d}$", c=c, label=label, ms=15)
        # ax.legend()

        box = ax.get_position()
        ax.set_position([box.x0, box.y0 + box.height * 0.1, box.width, box.height * 0.9])
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), fancybox=True, shadow=True, ncol=5)

        plt.savefig(f"clusters_2d.png")
        plt.show()

        # exit()

        X_pca = PCA(3).fit_transform(X_original)
        fig = plt.figure(figsize=(6, 6))
        ax = fig.add_subplot(projection='3d')
        for clust in range(clustering.n_clusters_):
            x_pca = X_pca[clusters == clust]
            cols = colors[clusters == clust]
            c_arms = np.array(arms)[clusters == clust]
            done = False
            for arm, x, c in zip(c_arms, x_pca, cols):
                if arm > 179:
                    print("ERROR!", c_arms)
                label = None
                if not done:
                    label = f"Cluster {clust}"
                    done = True
                ax.plot(*x, marker=f"${arm:03d}$", c=c, label=label, ms=15)
        # plt.legend()

        box = ax.get_position()
        ax.set_position([box.x0, box.y0 + box.height * 0.1, box.width, box.height * 0.9])
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), fancybox=True, shadow=True, ncol=3)

        plt.savefig(f"clusters_3d.png", dpi=300)
        plt.show()

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
        self.show_save_close(show, save_file)

    def plot_top(self, top_m=None, best=True, action_wrapper: ActionWrapper = None, show=True, save_file=None):
        # Get the data
        rewards, min_reward, max_reward = self.rewards_per_arm
        if top_m is None:
            top_m = len(self.arms)

        # Get the top best/worst
        sorted_rewards = sorted(rewards.items(), key=lambda ar: np.mean(ar[1]), reverse=best)
        sorted_rewards = sorted_rewards[:top_m]
        min_reward = sorted_rewards[-1 if best else 0][1]
        max_reward = sorted_rewards[0 if best else -1][1]
        arms = [a for a, r in sorted_rewards]
        x_ticks = (range(top_m), arms)

        # Set up the plot
        self._plot_text(title=f"Top {top_m} {'best' if best else 'worst'} arms",
                        x_label="Arm", y_label="Average Reward", legend=None, x_ticks=x_ticks)
        # Center the graph around the y_values to plot
        print(min_reward, max_reward)

        plt.ylim(self._center_y_lim(min_reward, max_reward))

        # Plot the data
        self._bar_plot(sorted_rewards, get_y=lambda x: x[1], colors=self._default_color)
        # Show, save and close the plot
        self.show_save_close(show, save_file)

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
        self.show_save_close(show, save_file)

    def plot_single_arm(self, arm, stride_csv_directory, file_name=None, plot_average=False, plot_cumulative=False,
                        show=True, save_file=None):

        # Plot the requested arm
        names = self.stride_vis.files_names
        if file_name is not None:
            file_names = [file_name]
        elif plot_cumulative:
            file_names = list(self.stride_vis.cumulative_file_names.keys())
            names = self.stride_vis.cumulative_file_names
        else:
            file_names = list(self.stride_vis.files_names.keys())

        if file_name is None:
            names = [names[fn] for fn in file_names]
        else:
            try:
                names = [names[fn] for fn in file_names]
            except KeyError:
                names = [self.stride_vis.cumulative_file_names[fn] for fn in file_names]
        # print("names:", names)
        try:
            name = names[file_name] if file_name is not None else "Individuals"
        except TypeError:
            name = names[0]
        # Set up the plot
        self._plot_text(title=f"Number of {name} for arm {arm}", x_label="Days", y_label=name, legend=None)

        lw = 0.5 if plot_average else 1.0
        alpha = 0.4 if plot_average else 1.0
        colors = ["blue", "red", "green", "black", "orange", "purple", "pink", "yellow", "grey"]

        p = Path(save_file).parent / f"statistics{'_cumul' if plot_cumulative else ''}.txt"
        with open(p, mode="w") as file:
            for idx, n in enumerate(names):
                # Average for arm
                average = []
                # For each episode
                for episode in self.episodes_per_arm[arm]:
                    stride_csv_file = Path(stride_csv_directory) / str(episode) / file_names[idx]
                    if not os.path.exists(stride_csv_file):  # TODO: abstract
                        stride_csv_file = Path(stride_csv_directory.replace("config_0", "config_0_checkpoints")) / str(
                            episode) / file_names[idx]
                    try:
                        y_values, min_reward, max_reward = self.stride_vis.load_file(stride_csv_file)
                        average.append(y_values)
                        plt.plot(range(len(y_values)), y_values, lw=lw, color=colors[idx], alpha=alpha,
                                 label=None if plot_average else n)
                    except FileNotFoundError:
                        print(f"FileNotFoundError for arm {arm}: {stride_csv_file}")
                        continue
                # Plot the average
                average = np.mean(average, axis=0)
                print(f"Average max {n}: {np.max(average)}")
                file.write(f"Average max {n}: {np.max(average)}\n")
                if plot_average:
                    plt.plot(range(len(average)), average, lw=lw * 3, color=colors[idx], label=n)

        plt.legend()
        # Show, save and close the plot
        self.show_save_close(show, save_file)

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
        self.show_save_close(show, save_file)

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

    def plot_violin(self, requested_arms=None, sorted_arms=False, best=True,
                    new_title=None,
                    show=True, save_file=None):
        """Create a violin plot for the requested arms."""
        arms = self.arms if requested_arms is None else requested_arms
        rewards, min_reward, max_reward = self.rewards_per_arm
        # Sort if requested
        if sorted_arms:
            # Get the top best/worst
            rewards = sorted(rewards.items(), key=lambda ar: np.mean(ar[1]), reverse=best)
            min_reward = rewards[-1 if best else 0][1]
            max_reward = rewards[0 if best else -1][1]
            arms = [a for a, r in rewards]
        x_ticks = (range(len(arms)), arms)

        if True:  # len(arms) > 10:  # TODO
            plt.rcParams["figure.figsize"] = (len(arms) * 0.5, plt.rcParamsDefault["figure.figsize"][1])
        # Set up the plot
        self._plot_text(title=f"Density of outcome distributions" if new_title is None else new_title,
                        x_label="Arm", y_label="reward", legend=None, x_ticks=x_ticks)
        # Center the graph around the y_values to plot
        # plt.ylim(self._center_y_lim(0.9978, 0.9999))  # TODO: min_reward, max_reward
        # plt.ylim(self._center_y_lim(0.724, 1.0))  # TODO: min_reward, max_reward
        plt.ylim(self._center_y_lim(0.475, 0.925))  # TODO: min_reward, max_reward

        # rewards = [rewards[arm] for arm in arms]
        new_rewards = []
        for arm in arms:
            try:
                r = rewards[arm]
            except KeyError:
                r = [0]
            new_rewards.append(r)
        rewards = new_rewards

        plt.violinplot(rewards, x_ticks[0],
                       points=200, vert=True, widths=1.0,
                       # showmeans=True, showextrema=True, showmedians=True,  # TODO: separate
                       showmeans=True, showextrema=True, showmedians=False,  # TODO: separate
                       bw_method=0.5
                       )
        # Show, save and close the plot
        self.show_save_close(show, save_file)
        plt.rcParams["figure.figsize"] = plt.rcParamsDefault["figure.figsize"]

    def plot_violin_hosp(self, stride_csv_directory, requested_arms=None, new_title=None,
                         show=True, save_file=None):
        """Create a violin plot for the requested arms."""
        arms = self.arms if requested_arms is None else requested_arms

        file_name = self.stride_vis.cases_hospitalised

        y = []
        min_y = np.inf
        max_y = -np.inf

        # For each episode
        for arm in arms:
            min_a = np.inf
            max_a = -np.inf
            arm_hosp = []
            for episode in range(100):
                stride_csv_file = Path(stride_csv_directory) / f"arm_{arm}" / str(episode) / file_name
                if not os.path.exists(stride_csv_file):  # TODO: abstract
                    stride_csv_file = Path(
                        stride_csv_directory.replace("config_0", "config_0_checkpoints")) / f"arm_{arm}" / str(
                        episode) / file_name
                try:
                    y_values, _, _ = self.stride_vis.load_file(stride_csv_file)
                    hosp_reward = y_values[-1]
                    arm_hosp.append(hosp_reward)
                    min_y = min(min_y, hosp_reward)
                    max_y = max(max_y, hosp_reward)

                    min_a = min(min_a, hosp_reward)
                    max_a = max(max_a, hosp_reward)
                    arm_hosp.append(hosp_reward)
                except FileNotFoundError:
                    print(f"FileNotFoundError for arm {arm}: {stride_csv_file}")
                    continue
            y.append(arm_hosp)
            print("Arm", arm)
            print("\tmin_reward", min_a, "max_reward", max_a, "avg", np.mean(arm_hosp))

        rewards, min_reward, max_reward = y, min_y, max_y
        x_ticks = (range(len(arms)), arms)

        print(min_y, max_y)

        if True:  # len(arms) > 10:  # TODO
            plt.rcParams["figure.figsize"] = (len(arms) * 0.5, plt.rcParamsDefault["figure.figsize"][1])
        # Set up the plot
        self._plot_text(title=f"Density of outcome distributions" if new_title is None else new_title,
                        x_label="Arm", y_label="reward", legend=None, x_ticks=x_ticks)
        # Center the graph around the y_values to plot
        plt.ylim(self._center_y_lim(-10, 80000))  # TODO: min_reward, max_reward

        plt.violinplot(rewards, x_ticks[0],
                       points=200, vert=True, widths=1.0,
                       # showmeans=True, showextrema=True, showmedians=True,  # TODO: separate
                       showmeans=True, showextrema=True, showmedians=False,  # TODO: separate
                       bw_method=0.5
                       )
        # Show, save and close the plot
        self.show_save_close(show, save_file)
        plt.rcParams["figure.figsize"] = plt.rcParamsDefault["figure.figsize"]

    def plot_time(self, requested_arms=None, sorted_arms=False, best=True,
                  new_title=None,
                  show=True, save_file=None):
        """Create a violin plot for the requested arms."""
        arms = self.arms if requested_arms is None else requested_arms
        rewards, min_reward, max_reward = self.times_per_arm
        # Sort if requested
        if sorted_arms:
            # Get the top best/worst
            rewards = sorted(rewards.items(), key=lambda ar: np.mean(ar[1]), reverse=best)
            min_reward = rewards[-1 if best else 0][1]
            max_reward = rewards[0 if best else -1][1]
            arms = [a for a, r in rewards]
        x_ticks = (range(len(arms)), arms)

        # print(rewards)
        means = []
        std = []
        rs = []


        for a, r in rewards.items():
            print("Arm", a)
            print("\tmean:", np.mean(r), "min", np.min(r), "max", np.max(r), "std", np.std(r), "mean + std", np.mean(r) + np.std(r))
            
            new_r = [i for i in r if i < 2900]
            new_r = np.array(new_r)
            means.append(np.mean(new_r))
            std.append(np.std(new_r))
            rs.append(np.max(new_r))

        print("overall mean:", np.mean(means))
        print("overall std:", np.std(std))
        print("rs: ", np.max(rs))

            # print("Arm", a)
            # print("\tmean:", np.mean(r), "min", np.min(r), "max", np.max(r), "std", np.std(r), "mean + std", np.mean(r) + np.std(r))
        # print("Total min: ", min_reward, "total max: ", max_reward)

        if True:  # len(arms) > 10:  # TODO
            plt.rcParams["figure.figsize"] = (len(arms) * 0.5, plt.rcParamsDefault["figure.figsize"][1])
        # Set up the plot
        self._plot_text(title=f"Density of outcome distributions" if new_title is None else new_title,
                        x_label="Arm", y_label="time", legend=None, x_ticks=x_ticks)
        # Center the graph around the y_values to plot
        plt.ylim(self._center_y_lim(min_reward, max_reward))  # TODO: min_reward, max_reward

        # rewards = [rewards[arm] for arm in arms]
        new_rewards = []
        for arm in arms:
            try:
                r = rewards[arm]
            except KeyError:
                r = [0]
            new_rewards.append(r)
        rewards = new_rewards

        plt.violinplot(rewards, x_ticks[0],
                       points=200, vert=True, widths=1.0,
                       # showmeans=True, showextrema=True, showmedians=True,  # TODO: separate
                       showmeans=True, showextrema=True, showmedians=False,  # TODO: separate
                       bw_method=0.5
                       )
        # Show, save and close the plot
        self.show_save_close(show, save_file)
        plt.rcParams["figure.figsize"] = plt.rcParamsDefault["figure.figsize"]
