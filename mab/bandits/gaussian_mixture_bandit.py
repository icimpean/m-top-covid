from mab.bandits.bandit import Bandit
from mab.posteriors.nonparam_gaussian_mixture import NGMPosterior
from mab.sampling.thompson_sampling import ThompsonSampling


class GaussianMixtureBandit(Bandit):
    """The class implementing a nonparametric gaussian mixture bandit with Thompson sampling"""
    def __init__(self, nr_arms, env, t_max, sampling_method=ThompsonSampling,
                 k=2, prior_k=2, d_context=2, pi=None, theta=None, sigma=None,
                 variational_max_iter=100, variational_lb_eps=0.001, seed=None):
        # Create the posteriors and sampling method
        posteriors = NGMPosterior(nr_arms, k, prior_k, d_context, t_max, pi, theta, sigma,
                                  variational_max_iter, variational_lb_eps, seed)
        sampling = sampling_method(posteriors, seed)
        # Super call
        super(GaussianMixtureBandit, self).__init__(nr_arms, env, sampling, seed)
