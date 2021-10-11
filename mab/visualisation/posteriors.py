import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats

from mab.bandits.bandit import Bandit, load_rewards
from mab.posteriors import Posterior
from mab.posteriors.bayesian_gaussian_mixture import GaussianMixturePosterior
from mab.posteriors.bnpy_gaussian_mixture import BNPYGaussianMixturePosterior
from mab.visualisation import Visualisation
from mab.visualisation.env import EnvVisualisation


class PosteriorVisualisation(Visualisation):
    """The visualisation for a bandit's posteriors."""

    def __init__(self):
        # Super call
        super(PosteriorVisualisation, self).__init__()
        #

    def plot_posterior(self, bandit: Bandit, arm, t="_end", n_samples=2000, separate_mixtures=True,
                       show=True, save_file=None):
        """Plot the posterior distribution for the given arm of the bandit."""
        self._posteriors(bandit, [arm], t, n_samples, separate_mixtures, show, save_file)

    def plot_posteriors(self, bandit: Bandit, requested_arms=None, t="_end", n_samples=2000,
                        show=True, save_file=None):
        # Get the arms to plot
        arms = range(bandit.nr_arms) if requested_arms is None else requested_arms
        separate_mixtures = False
        self._posteriors(bandit, arms, t, n_samples, separate_mixtures, show, save_file)

    def _posteriors(self, bandit: Bandit, arms, t="_end", n_samples=2000, separate_mixtures=True,
                    show=True, save_file=None):
        # Load the posterior(s) for the given timestep
        bandit.load(t)
        # Set up the plot
        title = f"PDF Posterior for arm {arms[0]}" if len(arms) == 0 else f"PDF Posteriors for bandit"
        self._plot_text(title=title, x_label="reward", y_label="frequency")
        # Plot the posterior for the given arm
        x = self._get_lin_space(bandit, arms, n_samples)
        for arm in arms:
            self.posterior(bandit.posteriors[arm], x, separate_mixtures, label=f"arm {arm}")
        # Only legend if more than 1 posterior, or if separate mixtures requested
        # TODO: keep restriction on max labels for large number of posteriors?
        if (len(arms) > 1 or separate_mixtures) and len(arms) <= 20:
            plt.legend()
        # Show, save and close the plot
        self.show_save_close(show, save_file)

    def _mean_precision(self, posterior: Posterior, x, plot_mixtures=False, plot_mean=True, n_samples=2000):
        # Gaussian Mixture Posterior
        if isinstance(posterior, GaussianMixturePosterior):
            mixture = posterior.get_mixture()
            m_weights = mixture.weights_
            m_means = mixture.means_
            m_precisions = mixture.mean_precision_
        elif isinstance(posterior, BNPYGaussianMixturePosterior):
            m_weights = posterior.get_weights()
            m_means = posterior.get_means()
            m_precisions = posterior.mean_precisions()
        # No plotting method implemented
        else:
            raise NotImplementedError(f"No plotting method defined for {posterior}")

        # Plot the mean uncertainty
        comb_y = np.zeros_like(x)
        p_mean = 0
        for n, (weight, mean, prec) in enumerate(zip(m_weights, m_means, m_precisions)):
            # precision = inverse of variance
            std_dev = (1 / prec) ** (1 / 2)
            y = stats.norm.pdf(x, mean, std_dev)
            y *= weight / n_samples
            comb_y += y
            p_mean += mean * weight
            if plot_mixtures:
                p = plt.plot(x, y, label="mixture precision")
                if plot_mean:
                    self.mean_line(mean, p)

        comb_y /= np.sum(comb_y)
        # Plot the mean uncertainty
        p = plt.plot(x, comb_y, label="mean precision", color="red")
        if plot_mean:
            self.mean_line(p_mean, p)

    @staticmethod
    def _get_lin_space(bandit: Bandit, arms, n_samples, distance=4):
        """Get the linear space to plot the posteriors of the given arms"""
        # TODO: distance=3?
        min_x = np.inf
        max_x = -np.inf
        for arm in arms:
            posterior = bandit.posteriors[arm]
            # Gaussian Mixture Posterior
            if isinstance(posterior, GaussianMixturePosterior):
                mixture = posterior.get_mixture()
                # Decide on the x-axis domain to plot
                for mean, cov in zip(mixture.means_, mixture.covariances_):
                    mean = mean[0]
                    cov = cov[0][0]
                    x1 = mean - distance * cov ** (1 / 2)
                    x2 = mean + distance * cov ** (1 / 2)
                    min_x = min(min_x, x1)
                    max_x = max(max_x, x2)
        # Create the linear space to plot for
        x = np.linspace(min_x, max_x, n_samples)
        return x

    def posterior(self, posterior: Posterior, x, separate_mixtures, label="total", plot_mean=False):
        # Gaussian Mixture Posterior
        if isinstance(posterior, GaussianMixturePosterior):
            mixture = posterior.get_mixture()
            m_weights = mixture.weights_
            m_means = mixture.means_
            m_covars = mixture.covariances_
        elif isinstance(posterior, BNPYGaussianMixturePosterior):
            m_weights = posterior.get_weights()
            m_means = posterior.get_means()
            # m_covars = posterior.get_cov()
            m_covars = [posterior.get_cov(k) for k in range(posterior.get_K())]
        # No plotting method implemented
        else:
            raise NotImplementedError(f"No plotting method defined for {posterior}")

        comb_y = np.zeros_like(x)
        p_mean = 0
        # Plot the multivariate gaussian for each of the mixtures
        for n, (weight, mean, cov) in enumerate(zip(m_weights, m_means, m_covars)):
            print(n)
            print(weight)
            print(mean)
            print(cov)

            y = stats.multivariate_normal.pdf(x, mean, cov[0])
            # y *= weight / n_samples
            comb_y += y
            p_mean += mean * weight
            if separate_mixtures:
                p = plt.plot(x, y, label=f"m$_{n}$", lw=0.7)
                if plot_mean:
                    self.mean_line(mean, p, lw=0.7)  #, label=f"$\mu_{n}$")
        # Plot the combined distribution
        p = plt.plot(x, comb_y, label=label, color="blue" if separate_mixtures else None)
        if plot_mean:
            self.mean_line(p_mean, p)

    def plot_env_posterior(self, bandit: Bandit, arm, n_samples=2000, separate_mixtures=True,
                           plot_mean=True, title_idx=None,
                           show=True, save_file=None):
        """Plot the posterior distribution for the given arm of the bandit."""
        env = bandit.env
        # Set up the plot
        title = f"Posterior distribution for arm {arm}{'' if title_idx is None else f' (t={title_idx})'}"
        self._plot_text(title=title, x_label="reward", y_label=None)

        # Plot env distribution
        v = EnvVisualisation()
        min_x, max_x = v.plot_env_arm(env, arm, plot_mean=plot_mean, plot_mixtures=separate_mixtures, coloured="black",
                                      prefix_mixture="mixture", prefix_arm="Arm")
        x = np.linspace(min_x, max_x, n_samples)
        self.posterior(bandit.posteriors[arm], x, separate_mixtures, label="Posterior", plot_mean=plot_mean)
        plt.legend()

        # Show, save and close the plot
        self.show_save_close(show, save_file)

    def plot_posterior_precision(self, bandit: Bandit, arm, plot_mixtures=True, n_samples=2000, title_idx=None,
                                 show=True, save_file=None):
        """Plot the mean precision of the posterior against the environment's mean."""
        env = bandit.env

        # Set up the plot
        title = f"Mean precision for arm {arm}{'' if title_idx is None else f' (t={title_idx})'}"
        self._plot_text(title=title, x_label="reward", y_label=None)

        # Plot env distribution
        v = EnvVisualisation()
        min_x, max_x = v.plot_env_arm(env, arm, plot_mean=True, plot_mixtures=plot_mixtures, coloured="black",
                                      prefix_mixture="mixture", prefix_arm="Arm")
        x = np.linspace(min_x, max_x, n_samples)
        # Get the posterior for the given arm
        self._mean_precision(bandit.posteriors[arm], x, plot_mixtures, plot_mean=True, n_samples=n_samples)

        plt.legend()

        # Show, save and close the plot
        self.show_save_close(show, save_file)

    def plot_posterior_precision2(self, bandit: Bandit, arm, plot_mixtures=False, n_samples=2000, title_idx=None,
                                  show=True, save_file=None):
        """Plot the mean precision of the posterior against the environment's mean."""
        env = bandit.env

        # Set up the plot
        title = f"Mean precision for arm {arm}{'' if title_idx is None else f' (t={title_idx})'}"
        self._plot_text(title=title, x_label="reward", y_label=None)

        # Plot env distribution
        v = EnvVisualisation()
        min_x, max_x = v.plot_env_arm(env, arm, plot_mean=True, plot_mixtures=plot_mixtures, coloured="black",
                                      prefix_mixture="mixture", prefix_arm="Arm")
        x = np.linspace(min_x, max_x, n_samples)

        # Plot the sampled rewards
        bins = 30
        rewards = load_rewards(bandit.log_file.parent)
        plt.hist(rewards, density=True, bins=bins, label="samples", color="grey", alpha=0.7)

        # Get the posterior for the given arm
        self._mean_precision(bandit.posteriors[arm], x, plot_mixtures, plot_mean=True, n_samples=n_samples)

        plt.legend()

        # Show, save and close the plot
        self.show_save_close(show, save_file)
