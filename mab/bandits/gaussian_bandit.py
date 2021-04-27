from typing import Type

from mab.bandits.bandit import Bandit
from mab.posteriors.gaussian import GaussianPosteriors
from mab.sampling import Sampling


class GaussianBandit(Bandit):
    """The class implementing a gaussian bandit with Thompson sampling"""
    def __init__(self, nr_arms, env, sampling_method: Type[Sampling], seed=None, log_file="./bandit_log.csv"):
        # Create the posteriors and sampling method
        posteriors = GaussianPosteriors(nr_arms, seed)
        sampling = sampling_method(posteriors, seed)
        # Super call
        super(GaussianBandit, self).__init__(nr_arms, env, sampling, seed, log_file)
