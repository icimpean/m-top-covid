import math
from pathlib import Path

import numpy as np
import pickle
from scipy.special import gammaln
from scipy.stats import t, uniform

from mab.posteriors import SinglePosteriors
from mab.posteriors.t_distribution import TDistributionPosterior


class TruncatedTDistributionPosterior(TDistributionPosterior):
    """A t-distribution posteriors of a bandit's arms"""
    def __init__(self, seed=None, alpha=0.5, a=0.0, b=1.0):
        super(TruncatedTDistributionPosterior, self).__init__(seed)
        self.alpha = alpha
        self.a = a
        self.b = b
        self.times_to_init_ = max(2, 3 - math.ceil(2 * self.alpha))
        self.mean_ = 0
        self.var_ = 0
        self.std_ = 0
        self.t = t
        self.uniform = uniform
        self.t.random_state = self.rng
        self.uniform.random_state = self.rng

    @staticmethod
    def new(seed=None, alpha=0.5, a=0.0, b=1.0):
        return TruncatedTDistributionPosterior(seed, alpha)

    def sample(self, timestep):
        n = len(self.rewards)
        freedom = self.freedom(n)
        sigma = np.sqrt(np.var(self.rewards) / freedom)
        # Special case where all rewards == mu leads to error, due to 0 sigma
        if sigma == 0.0:
            sigma += 1e-15
        mean = np.mean(self.rewards)

        t_dist = self.t(df=freedom, loc=mean, scale=sigma)
        Fa = t_dist.cdf(self.a)
        Fb = t_dist.cdf(self.b)
        u = self.uniform.rvs(Fa, Fb - Fa, size=1)[0]
        return t_dist.ppf(u)

    def truncated_t_mean(self, rewards, mu, sigma):
        n = len(rewards)
        v = self.freedom(n)
        def F(x):
            constant_n = gammaln((v + 1) / 2)
            constant_d = gammaln(v / 2)
            constant = np.exp(constant_n - constant_d) / math.sqrt(v*math.pi)
            return constant * (v/(1-v) * (1 + x*x/v)**((1-v)/2))
        a_ = (self.a - mu)/sigma
        b_ = (self.b - mu)/sigma
        return (F(b_) - F(a_))/(self.t.cdf(b_, v)-self.t.cdf(a_, v))

    def mean(self, t):
        n = len(self.rewards)
        mu = np.mean(self.rewards)
        if n < self.times_to_init_:
            return mu
        else:
            sigma = np.sqrt(np.var(self.rewards)/self.freedom(n))
            # Special case where all rewards == mu leads to error, due to 0 sigma
            if sigma == 0.0:
                sigma += 1e-15
            m = self.truncated_t_mean(self.rewards, mu, sigma)
            return m*sigma + mu

    def save(self, path: Path):
        with open(path, mode="wb") as file:
            data = [self.alpha, self.a, self.b, self.rng, self.rewards, self.mean_, self.var_, self.std_]
            pickle.dump(data, file)

    def load(self, path: Path):
        with open(path, mode="rb") as file:
            data = pickle.load(file)
            self.alpha, self.a, self.b, self.rng, self.rewards, self.mean_, self.var_, self.std_ = data
            self.t.random_state = self.rng
            self.uniform.random_state = self.rng


class TruncatedTDistributionPosteriors(SinglePosteriors):
    """Gaussian posteriors for a given number of bandit arms."""
    def __init__(self, nr_arms, seed=None, alpha=0.5, a=0.0, b=1.0):
        self.alpha = alpha
        self.a = a
        self.b = b
        super(TruncatedTDistributionPosteriors, self).__init__(nr_arms, TruncatedTDistributionPosterior, seed)

    def _create_posteriors(self, seed, posterior_type):
        return [posterior_type.new(seed + i, self.alpha) for i in range(self.nr_arms)]
