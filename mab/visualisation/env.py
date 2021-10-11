from pathlib import Path

import matplotlib.pyplot as plt
import scipy.stats as stats

from envs.env import *
from mab.visualisation import Visualisation


class EnvVisualisation(Visualisation):
    """An environment an algorithm can interact with"""
    def __init__(self):
        # Super class
        super(EnvVisualisation, self).__init__()

    def plot_env(self, env: Env, plot_mean=False, show=False, save_file=None):
        # Plot all arms together
        self._plot_text(title=f"Environment reward distributions", x_label="reward", y_label=None)
        for arm in range(env.nr_actions):
            self.plot_env_arm(env, arm, plot_mean, plot_mixtures=False, coloured=None)
        plt.legend()
        self.show_save_close(show, save_file)

    def plot_arms(self, env: Env, show=False, save_dir=None, plot_mean=False, plot_mixtures=True):
        # Plot each arm separately
        for arm in range(env.nr_actions):
            self._plot_text(title=f"Reward distribution Arm {arm}", x_label="reward", y_label=None)
            self.plot_env_arm(env, arm, plot_mean, plot_mixtures, coloured="black")
            plt.legend()
            save_file = None if save_dir is None else Path(save_dir) / f"env_arm_{arm}.png"
            self.show_save_close(show, save_file)

    def plot_env_arm(self, env: Env, arm, plot_mean, plot_mixtures, coloured="black",
                     prefix_mixture="mixture", prefix_arm="Arm"):
        # Gaussian arm
        if isinstance(env, GaussianEnv):
            mean = env.means[arm]
            std_dev = env.std_dev[arm]
            x1 = mean - 3 * std_dev ** 2
            x2 = mean + 3 * std_dev ** 2
            x = np.linspace(x1, x2, 100)
            y = stats.norm.pdf(x, mean, std_dev ** 2)
            p = plt.plot(x, y, label=str(arm))
            if plot_mean:
                plt.axvline(mean, color=p[-1].get_color(), ls="--")
            return x1, x2

        # Gaussian mixture
        elif isinstance(env, GaussianMixtureEnv):
            min_x = np.inf
            max_x = -np.inf

            print(env.pi, env.means, env.std_dev)

            for k, (pi, mean, std_dev) in enumerate(zip(env.pi[arm], env.means[arm], env.std_dev[arm])):
                x1 = mean - 3 * std_dev ** 2
                x2 = mean + 3 * std_dev ** 2
                min_x = min(min_x, x1)
                max_x = max(max_x, x2)
            new_x = np.linspace(min_x, max_x, 1000)
            new_y = np.zeros_like(new_x)
            arm_mean = 0
            for k, (pi, mean, std_dev) in enumerate(zip(env.pi[arm], env.means[arm], env.std_dev[arm])):
                y = stats.norm.pdf(new_x, mean, std_dev ** 2)
                new_y += y * pi
                arm_mean += mean * pi

                if plot_mixtures:
                    p = plt.plot(new_x, y * pi, label=f"{prefix_mixture} {k}", lw=0.7)
                    if plot_mean:
                        self.mean_line(mean, p, lw=0.7)
            p = plt.plot(new_x, new_y, label=f"{prefix_arm} {arm}", color=coloured)
            if plot_mean:
                self.mean_line(arm_mean, p)
            return min_x, max_x

        else:
            raise ValueError


if __name__ == '__main__':

    e = Test2GaussianMixtureEnv(seed=0)

    v = EnvVisualisation()
    v.plot_env(e, show=True, save_file=None, plot_mean=True)
    v.plot_arms(e, show=True, save_dir=None, plot_mean=True, plot_mixtures=True)
