import csv

import matplotlib.pyplot as plt
import numpy as np

from envs.flute_env.flute_env import FluteMDPEnv
from loggers.top_m_logger import TopMLogger
from mab.visualisation import Visualisation


class FluteVisualisation(Visualisation):
    """The visualisation for a flute experiment."""

    def __init__(self):
        # Super call
        super(FluteVisualisation, self).__init__()
        #
        self.arms = range(32)
        self.rankings = {}
        self.means = {}
        self.variances = {}
        self.sigmas = {}
        self.samples = {}
        self.min_max = {}
        self.colours = ["blue", "green", "red", "orange", "purple", "teal", ]

    def load_file(self, bandit_file, requested_arms=None, sort=True):
        self.load_files([bandit_file], requested_arms, sort)

    def load_files(self, alg_name, bandit_files, combine_episodes=True):
        """Combine multiple bandit file results"""
        rankings = []
        means = []
        variances = []
        sigmas = []
        logger = TopMLogger()
        entry_fields = logger.entry_fields
        last_episode = 0

        def read_list(string):
            split = string[1:-1].split(", ")
            if len(split) == 1:
                split = string[1:-1].split(" ")
            # Some arrays/lists get aligned if they have less digits in one element => skip empty strings
            new_list = [np.float64(s) for s in split if s != ""]
            return new_list

        for bandit_file in bandit_files:
            with open(bandit_file, "r") as file:
                reader = csv.DictReader(file, fieldnames=entry_fields)
                skip = True

                run_rankings = []
                run_means = []
                run_var = []
                run_sigma = []
                for entry in reader:
                    if skip:
                        skip = False
                        continue
                    # Get the arm, episode and ranking
                    arm = int(entry[logger.arm])
                    episode = int(entry[logger.episode])
                    ranking = entry[logger.ranking]
                    ranking_means = entry[logger.means]
                    ranking_variances = entry[logger.variances]
                    ranking_sigma = entry[logger.std_dev]

                    if ranking == "":
                        ranking = []
                    else:
                        split = ranking[1:-1].split(" ")
                        ranking = [int(s) for s in split if s != ""]
                    # Means, Variance, Std_dev
                    mean = read_list(ranking_means)
                    var = read_list(ranking_variances)
                    sigma = read_list(ranking_sigma)
                    # Update the episode number if merging multiple files
                    if combine_episodes:
                        episode += last_episode

                    run_rankings.append(ranking)
                    run_means.append(mean)
                    run_var.append(var)
                    run_sigma.append(sigma)
            last_episode = episode + 1
            rankings.append(run_rankings)
            means.append(run_means)
            variances.append(run_var)
            sigmas.append(run_sigma)

        # Set the data
        self.rankings[alg_name] = rankings
        self.means[alg_name] = means
        self.variances[alg_name] = variances
        self.sigmas[alg_name] = sigmas

    def plot_real_violin(self, R0, show=True, save_file=None):
        # Set up the plot
        self._plot_text(title=f"Influenza (R0={R0})", x_label="reward sample",
                        y_label="count", legend=None, x_ticks=None)
        # Get the data
        arms = self.arms
        f_env = FluteMDPEnv(R0=R0)
        thr = f_env.threshold
        x_ticks = (range(len(arms)), arms)
        rewards = [f_env.unthr_samples[arm] for arm in arms]

        # Set up the plot
        self._plot_text(title=f"Density of outcome distributions",
                        x_label="Arm", y_label="reward", legend=None, x_ticks=x_ticks)
        plt.violinplot(rewards, x_ticks[0], showmeans=True)
        plt.hlines((1-thr), 0, len(arms), color="black")
        # plt.ylim(0, 0.01)
        # Show, save and close the plot
        self.show_save_close(show, save_file)

    def plot_proportions(self, R0, top_m, show=True, save_file=None):
        # Get the data
        f_env = FluteMDPEnv(R0=R0)
        true_ranking = set(f_env.get_true_ranking(top_m))

        # Set up the plot
        self._plot_text(title=f"Top-{top_m} Influenza (R0={R0})", x_label="time step",
                        y_label="proportion", legend=None, x_ticks=None)
        for n, (algo_name, rankings) in enumerate(self.rankings.items()):
            colour = self.colours[n]
            proportions = []
            for run_ranking in rankings:
                prop = [len(true_ranking.intersection(rr)) / top_m for rr in run_ranking]
                proportions.append(prop)
            proportions = np.array(proportions)
            mean_proportions = np.mean(proportions, axis=0)
            std_proportions = np.var(proportions, axis=0)

            plt.plot(mean_proportions, color=colour, lw=1, label=algo_name)
            plt.fill_between(range(len(mean_proportions)),
                             mean_proportions-std_proportions, mean_proportions+std_proportions,
                             color=colour, alpha=0.5)
        plt.legend()

        # Show, save and close the plot
        self.show_save_close(show, save_file)

    def plot_sum_means(self, R0, top_m, clip=False, show=True, save_file=None):
        # Get the data
        f_env = FluteMDPEnv(R0=R0)
        true_means = []
        for arm in self.arms:
            mu, _ = f_env.get_distribution(arm)
            true_means.append(1-mu)  # reward!
        true_means = np.array(true_means)

        # Set up the plot
        self._plot_text(title=f"Top-{top_m} Influenza (R0={R0})", x_label="time step",
                        y_label="sum", legend=None, x_ticks=None)
        for n, (algo_name, means) in enumerate(self.means.items()):
            rankings = self.rankings[algo_name]
            colour = self.colours[n]
            sums = []
            for m, r in zip(means, rankings):
                m_sum = []
                for mm, rr in zip(m, r):
                    if len(rr) == 0:
                        s = 0
                    else:
                        new_m = np.take(true_means, rr)
                        s = np.nansum(new_m)
                    m_sum.append(s)
                sums.append(m_sum)
            sums = np.array(sums)

            start = 32 * 2 if clip else 0
            x = range(start, len(sums[0]))
            sums = sums[:, start:]

            mean_sums = np.mean(sums, axis=0)
            var_sums = np.std(sums, axis=0)

            plt.plot(x, mean_sums, color=colour, lw=1, label=algo_name)
            plt.fill_between(x, mean_sums - var_sums,
                             mean_sums + var_sums, color=colour, alpha=0.5)
        plt.legend()

        # Show, save and close the plot
        self.show_save_close(show, save_file)

    def plot_posterior(self, arm, R0, top_m, clip=True, show=True, save_file=None):
        # Set up the plot
        self._plot_text(title=f"Arm {arm} posterior | Top-{top_m} Influenza (R0={R0})", x_label="time step",
                        y_label="mean", legend=None, x_ticks=None)

        for n, algo_name in enumerate(self.rankings.keys()):
            means = np.array(self.means[algo_name])
            sigma = np.array(self.sigmas[algo_name])
            colour = self.colours[n]
            start = len(self.arms) * 2 if clip else 0

            x = range(start, len(means[0]))
            arm_means = means[:, start:, arm]
            arm_sigmas = sigma[:, start:, arm]
            # Remove np.inf values from plots, if present
            arm_sigmas = np.where(arm_sigmas == np.inf, 0, arm_sigmas)

            mean_means = np.mean(arm_means, axis=0)
            sigma_means = np.mean(arm_sigmas, axis=0)

            plt.plot(x, mean_means, color=colour, lw=1, label=algo_name)
            plt.fill_between(x, mean_means - sigma_means, mean_means + sigma_means, color=colour, alpha=0.5)

        plt.legend()
        # Show, save and close the plot
        self.show_save_close(show, save_file)

    def plot_posterior_run(self, arm, episode, R0, top_m, clip=True, show=True, save_file=None):
        # Set up the plot
        self._plot_text(title=f"Arm {arm} posterior | Top-{top_m} Influenza (R0={R0})", x_label="time step",
                        y_label="mean", legend=None, x_ticks=None)

        for n, algo_name in enumerate(self.rankings.keys()):
            means = np.array(self.means[algo_name])
            sigma = np.array(self.sigmas[algo_name])
            colour = self.colours[n]
            start = len(self.arms) * 2 if clip else 0

            x = range(start, len(means[0]))
            arm_means = means[episode, start:, arm]
            arm_sigmas = sigma[episode, start:, arm]
            # Remove np.inf values from plots, if present
            arm_sigmas = np.where(arm_sigmas == np.inf, 0, arm_sigmas)

            plt.plot(x, arm_means, color=colour, lw=1, label=algo_name)
            plt.fill_between(x, arm_means - arm_sigmas, arm_means + arm_sigmas, color=colour, alpha=0.5)

        plt.legend()
        # Show, save and close the plot
        self.show_save_close(show, save_file)

    def plot_posteriors(self, algo_name, arms, R0, top_m, clip=True, show=True, save_file=None):
        # Set up the plot
        self._plot_text(title=f"{algo_name} Posteriors | Top-{top_m} Influenza (R0={R0})", x_label="time step",
                        y_label="mean", legend=None, x_ticks=None)

        for n, arm in enumerate(arms):
            means = np.array(self.means[algo_name])
            sigmas = np.array(self.sigmas[algo_name])

            start = 32 * 2 if clip else 0
            x = range(start, len(means[0]))
            arm_means = means[:, start:, arm]
            arm_sigmas = sigmas[:, start:, arm]
            # Remove np.inf values from plots, if present
            arm_sigmas = np.where(arm_sigmas == np.inf, 0, arm_sigmas)

            mean_means = np.mean(arm_means, axis=0)
            sigma_means = np.mean(arm_sigmas, axis=0)

            p = plt.plot(x, mean_means, lw=1, label=arm)
            plt.fill_between(x, mean_means - sigma_means, mean_means + sigma_means, color=p[0].get_color(), alpha=0.5)

        plt.legend()
        # # Show, save and close the plot
        self.show_save_close(show, save_file)

    def plot_posteriors_run(self, run, algo_name, arms, R0, top_m, clip=True, show=True, save_file=None):
        # Set up the plot
        self._plot_text(title=f"{algo_name} Posteriors | Top-{top_m} Influenza (R0={R0})", x_label="time step",
                        y_label="mean", legend=None, x_ticks=None)

        for n, arm in enumerate(arms):
            means = np.array(self.means[algo_name])
            sigmas = np.array(self.sigmas[algo_name])

            start = 32 * 2 if clip else 0
            x = range(start, len(means[0]))
            arm_means = means[run, start:, arm]
            arm_sigmas = sigmas[run, start:, arm]
            # Remove np.inf values from plots, if present
            arm_sigmas = np.where(arm_sigmas == np.inf, 0, arm_sigmas)

            p = plt.plot(x, arm_means, lw=1, label=arm)
            plt.fill_between(x, arm_means - arm_sigmas, arm_means + arm_sigmas, color=p[0].get_color(), alpha=0.5)

        plt.legend()
        # # Show, save and close the plot
        self.show_save_close(show, save_file)

    def plot_percentages_top(self, R0, top_m, show=True, save_file=None):
        w, h = plt.rcParams.get('figure.figsize')
        fig, ax = plt.subplots(figsize=(max(w, 0.25 * len(self.arms)), h * 0.75))
        ax.set_title(f"Top-{top_m} Influenza (R0={R0})")
        ax.set_xlabel("arm")
        ax.set_ylabel("count")
        ax.tick_params(axis='both', which='major')
        ax.tick_params(axis='both', which='minor')

        factor = 1
        label_locations = np.arange(len(self.arms)) * factor
        num_ys = len(self.rankings.keys())
        bar_width = (0.9 * factor / num_ys)

        ax.set_xticks(label_locations)
        ax.set_xticklabels(self.arms)

        def get_x(bar_idx):
            return label_locations + (bar_idx - (num_ys - 1) / 2) * bar_width

        percentages = {algo_name: {arm: 0 for arm in self.arms} for algo_name in self.rankings.keys()}

        for n, (algo_name, rankings) in enumerate(self.rankings.items()):
            colour = self.colours[n]
            for r in rankings:
                for rr in r:
                    for a in rr:
                        percentages[algo_name][a] += 1
            ax.bar(get_x(n), percentages[algo_name].values(), width=bar_width, label=algo_name, color=colour)
        plt.legend()

        # # Show, save and close the plot
        self.show_save_close(show, save_file)

    @staticmethod
    def _print_top_arms(top_arms, best, action_wrapper=None):
        print(f"Top {len(top_arms)} {'best' if best else 'worst'} arms")
        for arm, reward in top_arms:
            line = f"Arm {arm}"
            if action_wrapper is not None:
                line += " " + str(action_wrapper.get_pretty_raw_action(arm))
            line += f": reward {reward}"
            print(line)
