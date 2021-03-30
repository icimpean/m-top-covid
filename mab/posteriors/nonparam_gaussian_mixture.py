import copy

import numpy as np
import scipy.stats as stats

from mab.posteriors import GroupPosterior
from mab.posteriors.variational_posterior import VariationalPosterior


class NGMPosterior(GroupPosterior, VariationalPosterior):
    """A Linear Gaussian Mixture Posterior, with Normal Inverse Gamma conjugate prior.
    Uses variational updates.

    TODO: documentation
    """
    def __init__(self, nr_arms, k, prior_k, d_context, t_max,
                 pi, theta, sigma, variational_max_iter, variational_lb_eps, seed=None):
        # Super call
        super(GroupPosterior, self).__init__(nr_arms, seed)
        # Seed the random generators
        np.random.seed(self.seed)
        self.rng = np.random.default_rng(seed=seed)

        # Number of time steps
        self.t_max = t_max
        # Context (or None if not used/available)
        self.d_context = d_context
        self.context = None
        self.context = np.ones(shape=(self.d_context, self.t_max))  # Static context
        # Number of arms
        self.nr_arms = nr_arms
        self._len = self.nr_arms

        # Reward function
        self.reward_function_dist = stats.norm
        self.prior_nig = stats.invgamma  # inverse gamma prior

        # Variational parameters
        self.variational_max_iter = variational_max_iter
        self.variational_lb_eps = variational_lb_eps

        # Arm priors and posteriors
        self.prior_K = prior_k
        # Normal-inverse Gamma (NIG) for linear Gaussian
        self.prior_alpha = np.ones((self.nr_arms, self.prior_K))
        self.prior_beta = np.ones_like(self.prior_alpha)
        # Dirichlet for mixture weights
        self.prior_gamma = np.ones_like(self.prior_alpha)
        # Posteriors
        self.k = k
        self.alpha = copy.deepcopy(self.prior_alpha)
        self.beta = copy.deepcopy(self.prior_beta)
        self.gamma = copy.deepcopy(self.prior_gamma)
        # Initialise the remaining priors and posteriors
        # Mixture proportions
        self.pi = self._init_mixture_proportions(pi)
        # Thetas (prior + posterior)
        self.prior_theta, self.theta = self._init_regression_parameters(theta)
        # Sigmas (prior + posterior)
        self.prior_sigma, self.sigma = self._init_emission_variance(sigma)

        # The actions and rewards used
        self.actions = np.zeros((self.nr_arms, self.t_max))
        self.rewards = np.zeros_like(self.actions)
        self.rewards_expected = np.zeros_like(self.rewards)

        # The predicted distributions of the arms
        self.means = np.zeros((self.nr_arms, self.t_max))
        self.vars = np.zeros((self.nr_arms, self.t_max))

    def compute_posteriors(self, t):  # compute_arm_predictive_density
        """Update the posteriors based on the information gathered so far."""
        # Sample reward's parameters, given updated Variational parameters
        rewards_samples = np.zeros((self.nr_arms, 1))
        # For each arm
        for a in range(self.nr_arms):
            # Data for each mixture
            rewards_per_mixture_samples = np.zeros((self.prior_K, 1))

            # Compute for each mixture
            for k in range(self.prior_K):
                # First sample variance from inverse gamma for each mixture
                sigma_samples = stats.invgamma.rvs(self.alpha[a, k], scale=self.beta[a, k], size=(1, 1))
                # Then multivariate Gaussian parameters
                theta_samples = self.theta[a, k, :][:, None] + np.sqrt(sigma_samples) * (
                    stats.multivariate_normal.rvs(cov=self.sigma[a, k, :, :], size=1).reshape(1, self.d_context).T)
                # Draw per mixture rewards given sampled parameters
                rewards_per_mixture_samples[k, :] = self.reward_function_dist.rvs(
                    loc=np.einsum('d,dm->m', self.context[:, t], theta_samples), scale=np.sqrt(sigma_samples))

            # How to compute rewards over mixtures
            sampling_method = 'z_sampling'

            # Sample Z
            if sampling_method == 'z_sampling':
                # Draw Z from mixture proportions as determined by Dirichlet multinomial
                pi = self.gamma[a] / (self.gamma[a].sum())
                z_samples = np.random.multinomial(1, pi, size=1).T
                # Draw rewards for each of the picked mixture
                # Note: transposed used due to python indexing
                rewards_samples[a, :] = rewards_per_mixture_samples.T[z_samples.T == 1]
            # Sample pi
            elif sampling_method == 'pi_sampling':
                # Draw mixture proportions as determined by Dirichlet multinomial
                pi_samples = stats.dirichlet.rvs(self.gamma[a], size=1).T
                # Draw rewards given sampled parameters
                rewards_samples[a, :] = np.einsum('km,km->m', pi_samples, rewards_per_mixture_samples)
            # Expected pi
            elif sampling_method == 'pi_expected':
                # Computed expected mixture proportions as determined by Dirichlet multinomial
                pi = self.gamma[a] / (self.gamma[a].sum())
                # Draw rewards, by averaging over expected mixture proportions
                rewards_samples[a, :] = np.einsum('k,km->m', pi, rewards_per_mixture_samples)

        # Monte Carlo integration over reward samples
        max_ = (rewards_samples.argmax(axis=0)[None, :] == np.arange(self.nr_arms)[:, None]).astype(int)
        # Mean times reward is maximum
        self.means[:, t] = max_.mean(axis=1)
        # Variance of times reward is maximum
        self.vars[:, t] = max_.var(axis=1)
        # Also, compute expected rewards
        self.rewards_expected[:, t] = rewards_samples.mean(axis=1)

    def update(self, arm, reward, t):
        # Add the received reward
        self.actions[arm, t] = 1
        self.rewards[arm, t] = reward
        # Update the posteriors
        self.update_posterior(t)

    def sample_all(self, t):
        # Sample each arm
        samples = self.rng.multinomial(1, self.means_per_arm(t), size=1)[0]
        # self.actions[self.rng.multinomial(1, self.means[:, t], size=1).sum(axis=0).argmax(), t] = 1
        # action = np.where(self.actions[:, t] == 1)[0][0]
        return samples

    def sample_best_arm(self, t):
        means = self.means_per_arm(t)
        return np.argmax(means)

    def means_per_arm(self, t):
        # Get the means per arm at timestep t
        return self.means[:, t]
