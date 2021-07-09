import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats

from mab.bandits.bandit import Bandit
from mab.posteriors import Posterior
from mab.posteriors.bayesian_gaussian_mixture import GaussianMixturePosterior
from mab.visualisation import Visualisation


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
            self._posterior(bandit.posteriors[arm], x, n_samples, separate_mixtures, label=f"arm {arm}")
        # Only legend if more than 1 posterior, or if separate mixtures requested
        # TODO: keep restriction on max labels for large number of posteriors?
        if (len(arms) > 1 or separate_mixtures) and len(arms) <= 20:
            plt.legend()
        # Show, save and close the plot
        self._show_save_close(show, save_file)

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

    def _posterior(self, posterior: Posterior, x, n_samples, separate_mixtures, label="total"):
        # Gaussian Mixture Posterior
        if isinstance(posterior, GaussianMixturePosterior):
            mixture = posterior.get_mixture()
            comb_y = np.zeros_like(x)
            # Plot the multivariate gaussian for each of the mixtures
            for n, (weight, mean, cov) in enumerate(zip(mixture.weights_, mixture.means_, mixture.covariances_)):
                y = stats.multivariate_normal.pdf(x, mean, cov)
                y *= weight / n_samples
                comb_y += y
                if separate_mixtures:
                    plt.plot(x, y, label=f"mixture {n}")
            # Plot the combined distribution
            plt.plot(x, comb_y, label=label, lw=2.5 if separate_mixtures else None,
                     color="black" if separate_mixtures else None)
        # No plotting method implemented
        else:
            raise NotImplementedError(f"No plotting method defined for {posterior}")
