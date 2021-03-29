import logging

import numpy as np
import scipy.special as special


class VariationalPosterior(object):
    """A Linear Gaussian Mixture Posterior, with Normal Inverse Gamma conjugate prior.
    A variational update

    TODO: documentation
    """

    #####################
    # Update Posteriors #
    #####################
    def update_posterior(self, t):
        # Variational update
        # Requires responsibilities of mixtures per arm
        self.r = np.zeros((self.nr_arms, self.prior_K, self.rewards.shape[1]))
        # Lower bound
        lower_bound = np.zeros(self.variational_max_iter + 1)
        lower_bound[0] = np.finfo(float).min
        # First iteration
        n_iter = 1
        self._update_variational_resp()
        self._update_variational_params()
        lower_bound[n_iter] = self._update_variational_lowerbound()
        print(f't={t}, n_iter={n_iter} with lower bound={lower_bound[n_iter]}')

        # Iterate while not converged or not max iterations
        while (n_iter < self.variational_max_iter and abs(
                lower_bound[n_iter] - lower_bound[n_iter - 1]) >= (
                       self.variational_lb_eps * abs(lower_bound[n_iter - 1]))):
            n_iter += 1
            self._update_variational_resp()
            self._update_variational_params()
            lower_bound[n_iter] = self._update_variational_lowerbound()
            print(f't={t}, n_iter={n_iter} with lower bound={lower_bound[n_iter]}')

    def _update_variational_resp(self):
        # Variational responsibilities for linear Gaussian Mixture with Normal Inverse Gamma conjugate prior
        # Indicators for picked arms
        tIdx, aIdx = np.where(self.actions.T)
        # Compute rho
        rho = (-0.5 * (np.log(self.beta) - special.digamma(self.alpha))[:, :, None]
               -0.5 * (np.einsum('it,akij,jt->akt', self.context[:, tIdx], self.sigma, self.context[:, tIdx])
                        + np.power(self.rewards[:, tIdx].sum(axis=0)
                                   - np.einsum('it,aki->akt', self.context[:, tIdx], self.theta), 2)
                        * self.alpha[:, :, None] / self.beta[:, :, None])
               + (special.digamma(self.gamma) - special.digamma(self.gamma.sum(axis=1, keepdims=True)))[:, :, None]
               )
        # Exponentiate (and try to avoid numerical errors)
        r = np.exp(rho - rho.max(axis=1, keepdims=True))
        # And normalize
        self.r[:, :, tIdx] = r / (r.sum(axis=1, keepdims=True))

    def _update_variational_params(self):
        # Variational parameters for linear Gaussian Mixture with Normal Inverse Gamma conjugate prior
        # For each arm
        for a in range(self.nr_arms):
            # Pick times when arm was played
            this_a = self.actions[a, :] == 1
            # For each mixture component
            for k in np.arange(self.prior_K):
                # Its responsibilities
                R_ak = np.diag(self.r[a, k, this_a])
                # Equation 8 from paper
                # Update Gamma
                self.gamma[a, k] = self.prior_gamma[a, k] + self.r[a, k, this_a].sum()
                # Update Sigma
                self.sigma[a, k, :, :] = np.linalg.inv(np.linalg.inv(self.prior_sigma[a, k, :, :])
                                                       + np.dot(np.dot(self.context[:, this_a], R_ak),
                                                                self.context[:, this_a].T))
                # Update and append Theta
                self.theta[a, k, :] = np.dot(self.sigma[a, k, :, :], np.dot(np.linalg.inv(self.prior_sigma[a, k, :, :]),
                                                                            self.prior_theta[a, k, :])
                                             + np.dot(np.dot(self.context[:, this_a], R_ak), self.rewards[a, this_a].T))
                # Update and append alpha
                self.alpha[a, k] = self.prior_alpha[a, k] + (self.r[a, k, this_a].sum()) / 2
                # Update and append beta
                self.beta[a, k] = self.prior_beta[a, k] + 1 / 2 * (
                        np.dot(np.dot(self.rewards[a, this_a], R_ak), self.rewards[a, this_a].T) +
                        np.dot(self.prior_theta[a, k, :].T, np.dot(np.linalg.inv(self.prior_sigma[a, k, :, :]),
                                                                   self.prior_theta[a, k, :])) -
                        np.dot(self.theta[a, k, :].T,
                               np.dot(np.linalg.inv(self.sigma[a, k, :, :]), self.theta[a, k, :]))
                )

    def _update_variational_lowerbound(self):  # TODO: used?
        # Indicators for picked arms
        tIdx, aIdx = np.where(self.actions.T)
        atIdx = np.arange(aIdx.size)
        # E{ln f(Y|Z,\theta,\sigma)}
        tmp = ((np.log(2 * np.pi) + np.log(self.beta) - special.digamma(self.alpha))[:, :, None]
               + np.einsum('it,akij,jt->akt', self.context[:, tIdx], self.sigma, self.context[:, tIdx])
               + np.power(self.rewards[:, tIdx].sum(axis=0)
                          - np.einsum('it,aki->akt', self.context[:, tIdx], self.theta), 2)
               * self.alpha[:, :, None] / self.beta[:, :, None])
        # Sum over mixture and time indicators
        E_ln_fY = -0.5 * (self.r[aIdx, :, tIdx] * tmp[aIdx, :, atIdx]).sum(axis=(1, 0))

        # E{ln p(Z|\pi}
        tmp = self.r[:, :, tIdx] * (special.digamma(self.gamma)
                                    - special.digamma(self.gamma.sum(axis=1, keepdims=True)))[:, :, None]
        # Sum over mixture and arm-time indicators
        E_ln_pZ = tmp[aIdx, :, atIdx].sum(axis=(1, 0))

        # E{ln f(\pi}
        tmp = (special.gammaln(self.prior_gamma.sum(axis=1, keepdims=True))
               - special.gammaln(self.prior_gamma).sum(axis=1, keepdims=True)
               + ((self.prior_gamma - 1) * (special.digamma(self.gamma) - special.digamma(
                    self.gamma.sum(axis=1, keepdims=True)))).sum(axis=1, keepdims=True))
        # Sum over all arms
        E_ln_fpi = tmp.sum(axis=0)

        # E{ln f(\theta,\sigma}
        tmp = (self.prior_alpha * np.log(self.prior_beta)
               - special.gammaln(self.prior_alpha) - self.d_context / 2 * np.log(2 * np.pi)
               - 0.5 * np.linalg.det(self.prior_sigma)
               - (self.d_context / 2 + self.prior_alpha + 1) * (np.log(self.beta) - special.digamma(self.alpha))
               - self.prior_beta * self.alpha / self.beta
               - 0.5 * (np.einsum('akij->ak', np.einsum('akij,akjl->akil', np.linalg.inv(self.prior_sigma), self.sigma))
                        + np.einsum('aki,akij,akj->ak', (self.theta - self.prior_theta),
                                    np.linalg.inv(self.prior_sigma),
                                    (self.theta - self.prior_theta)) * self.alpha / self.beta))
        # Sum over mixtures and arms
        E_ln_fthetasigma = tmp.sum(axis=(1, 0))

        # E{ln q(Z|\pi}
        tmp = self.r[:, :, tIdx] * np.log(self.r[:, :, tIdx])
        # Sum over mixture and arm-time indicators
        E_ln_qZ = tmp[aIdx, :, atIdx].sum(axis=(1, 0))

        # E{ln q(\pi}
        tmp = (special.gammaln(self.gamma.sum(axis=1, keepdims=True))
               - special.gammaln(self.gamma).sum(axis=1, keepdims=True)
               + ((self.gamma - 1) * (special.digamma(self.gamma) - special.digamma(
                    self.gamma.sum(axis=1, keepdims=True)))).sum(axis=1, keepdims=True))
        # Sum over all arms
        E_ln_qpi = tmp.sum(axis=0)

        # E{ln q(\theta,\sigma}
        tmp = (self.alpha * np.log(self.beta)
               - special.gammaln(self.alpha) - self.d_context / 2 * np.log(2 * np.pi) - 0.5 * np.linalg.det(self.sigma)
               - (self.d_context / 2 + self.alpha + 1) * (np.log(self.beta) - special.digamma(self.alpha))
               - self.alpha - self.d_context / 2)
        # Sum over mixtures and arms
        E_ln_qthetasigma = tmp.sum(axis=(1, 0))
        # Return lower bound
        return E_ln_fY + E_ln_pZ + E_ln_fpi + E_ln_fthetasigma - E_ln_qZ - E_ln_qpi - E_ln_qthetasigma

    ##################
    # Initialisation #
    ##################
    def _init_mixture_proportions(self, pi=None):
        """Initialise the mixture proportions, per arm per mixture.

        Args:
            pi: The mixture proportions per arm per mixture.
                If None, then random probabilities are assigned.

        Returns:
            None.
        """
        shape = (self.nr_arms, self.k)
        # Random proportions
        if pi is None:
            logging.debug(f"Using random mixture proportions.")
            new_pi = self.rng.random(size=shape)
            # Probabilities must sum up to 1 per arm
            new_pi /= new_pi.sum(axis=1)
        # Use given proportions
        else:
            logging.debug(f"Using given mixture proportions.")
            # Enough values should be given
            assert len(pi) == self.nr_arms * self.k
            new_pi = np.reshape(pi, newshape=shape)
            # Probabilities must sum up to 1 per arm
            assert np.all(new_pi.sum(axis=1) == np.ones(self.nr_arms))
        # Return the mixture proportions
        return new_pi

    def _init_regression_parameters(self, theta=None):
        """Initialise the regression parameter theta.

        Args:
            theta: The theta per arm, mixture and context dimension.
                If None, then random parameters are assigned.

        Returns:
            None.
        """
        shape = (self.nr_arms, self.k, self.d_context)
        # The initial theta for the prior
        prior_theta = np.ones(shape=shape)
        for k in range(self.prior_K):
            prior_theta[:, k] = k
        # Random parameters
        if theta is None:
            logging.debug(f"Using random regression parameters.")
            new_theta = self.rng.standard_normal(size=shape)
        # Use given proportions
        else:
            logging.debug(f"Using given regression parameters.")
            # Enough values should be given
            assert len(theta) == self.nr_arms * self.k * self.d_context
            new_theta = np.reshape(theta, newshape=shape)
        # Return the regression parameters
        return prior_theta, new_theta

    def _init_emission_variance(self, sigma=None):
        """Initialise the emission variance sigma.

        Args:
            sigma: The sigma per arm and mixture.
                If None, then a unit variance is used.

        Returns:
            None.
        """
        shape = (self.nr_arms, self.k)
        prior_shape = (self.nr_arms, self.k, self.d_context, self.d_context)
        # The initial sigma for the prior
        prior_sigma = np.zeros(shape=prior_shape)
        for arm in range(self.nr_arms):
            for k in range(self.prior_K):
                prior_sigma[arm, k] = np.eye(self.d_context)
        # Unit variance
        if sigma is None:
            logging.debug(f"Using a unit variance.")
            new_sigma = np.ones(size=shape)
        # Use given proportions
        else:
            logging.debug(f"Using given emission variance.")
            # Enough values should be given
            assert len(sigma) == self.nr_arms * self.k
            new_sigma = np.reshape(sigma, newshape=shape)
        # Return the regression parameters
        return prior_sigma, new_sigma