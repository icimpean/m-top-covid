from typing import Type

from mab.bandits.bandit import Bandit
from mab.posteriors.bayesian_gaussian_mixture import BGMPosteriors
from mab.sampling import Sampling
from mab.sampling.thompson_sampling import ThompsonSampling


class BayesianGaussianMixtureBandit(Bandit):
    """The class implementing a gaussian mixture bandit"""
    def __init__(self, nr_arms, env, sampling_method: Type[Sampling] = ThompsonSampling,
                 k=2, variational_max_iter=100, variational_tol=0.001, seed=None):
        # Create the posteriors and sampling method
        posteriors = BGMPosteriors(nr_arms, seed, k, variational_tol, variational_max_iter)
        sampling = sampling_method(posteriors, seed)
        # Super call
        super(BayesianGaussianMixtureBandit, self).__init__(nr_arms, env, sampling, seed)
