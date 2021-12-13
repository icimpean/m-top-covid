import sys

import numpy as np

from mab.posteriors import Posteriors
from mab.sampling import Sampling


class AT_LUCB(Sampling):
    """AnyTime Lower and Upper Confidence Bound Sampling

    From: https://github.com/plibin-vub/bfts

    Attributes:
        posteriors: The Posteriors to apply the sampling method to
        top_m: The number of posteriors to provide as the top m best posteriors.
        sigma1: The

        seed: The seed for initialisation.
    """
    def __init__(self, posteriors: Posteriors, top_m,
                 sigma1=0.5, alpha=0.99, epsilon=0,
                 seed=None):
        # Super call
        super(AT_LUCB, self).__init__(posteriors, seed)
        #
        self.m = top_m
        self.sigma1 = sigma1
        self.alpha = alpha
        self.epsilon = epsilon

        self.Jt = np.full(self.m, -1)
        self.S = [1]
        self.nr_arms = self.posteriors.nr_arms
        self.pull_lowest = True
        self._t = 1

        self.has_ranking = True
        self.sample_ordering = None
        self.current_ranking = None

    @staticmethod
    def new(top_m, sigma1=0.5, alpha=0.99, epsilon=0):
        """Workaround to add a given top_m arms to the sampling method"""
        return lambda posteriors, seed: AT_LUCB(posteriors, top_m, sigma1, alpha, epsilon, seed)

    def sample_arm(self, t):
        """Sample an arm based on the sampling method."""
        if t == 0:
            self._t = 1
        elif self.pull_lowest:
            self._t += 1
        t = self._t
        if self.term(t, self.sigma(self.S[t - 1]), self.epsilon):
            s = self.S[t - 1]
            while self.term(t, self.sigma(s), self.epsilon):
                s = s + 1
            self.S.append(s)
            self.Jt = self.top_m(t)
        else:
            self.S.append(self.S[t - 1])
            if self.S[t] == 1:
                self.Jt = self.top_m(t)
        # Original pulls both lowest and highest arm each timestep
        #   => divide over two steps and consider each two steps as 1 within AT-LUCB
        h_or_l = self.l if self.pull_lowest else self.h
        # Sample the arm
        arm = h_or_l(t, self.sigma(self.S[t - 1]))
        # print("l" if self.pull_lowest else "h", t, arm)

        theta = self.posteriors.sample_all(t)
        order = np.argsort(-np.array(theta))
        self.sample_ordering = order
        self.current_ranking = self.top_m(t)

        # Next time pull the opposite
        self.pull_lowest = not self.pull_lowest
        # Return the highest/lowest sampled arm
        assert arm >= 0, f"Arm sampled was {arm}, should be >= 0"
        return arm

    def top_m(self, t):
        """Get the top m arms at timestep t"""
        # Get the means per arm
        means = np.array(self.posteriors.means_per_arm(t))
        return np.argsort(-means)[0:self.m]

    def sigma(self, s):
        return self.sigma1 * self.alpha ** (s - 1)

    def beta(self, u, t, sigma1):
        k1 = 1.25
        n = self.nr_arms
        return ((np.log(n * k1 * (t ** 4) / sigma1)) / (2 * u)) ** 0.5

    def term(self, t, sigma, epsilon):
        h = self.h(t, sigma)  # max
        l = self.l(t, sigma)  # min
        U = self.U(t, l, sigma)
        L = self.L(t, h, sigma)
        return U - L < epsilon

    def L(self, t, a, sigma):
        mu = self._get_mean(a, t)
        if mu == 0.0:
            return float("-inf")
        else:
            return mu - self.beta(len(self.posteriors[a].rewards), t, sigma)

    def U(self, t, a, sigma):
        mu = self._get_mean(a, t)
        if mu == 0.0:
            return float("inf")
        else:
            return mu + self.beta(len(self.posteriors[a].rewards), t, sigma)

    def h(self, t, sigma):
        min_ = sys.float_info.max
        min_index = -1
        for j in self.Jt:
            L = self.L(t, j, sigma)
            if L < min_:
                min_ = L
                min_index = j

        return int(min_index)

    def l(self, t, sigma):
        max_ = sys.float_info.min
        max_index = -1
        for j in range(self.nr_arms):
            if j in self.Jt:
                pass
            else:
                U = self.U(t, j, sigma)
                if U > max_:
                    max_ = U
                    max_index = j

        return int(max_index)

    def _get_mean(self, arm, t):
        # Protect against NaN mean at the start due to no rewards for a given arm
        mu = self.posteriors[arm].mean(t)
        if np.isnan(mu):
            mu = 0
        return mu
