from pathlib import Path
from typing import Type

from mab.bandits.bandit import Bandit
from mab.posteriors.bayesian_gaussian_mixture import BGMPosteriors
from mab.sampling import Sampling
from mab.sampling.thompson_sampling import ThompsonSampling


class BayesianGaussianMixtureBandit(Bandit):
    """The class implementing a bayesian gaussian mixture bandit"""
    def __init__(self, nr_arms, env, sampling_method: Type[Sampling] = ThompsonSampling,
                 k=2, variational_max_iter=100, variational_tol=0.001, seed=None, log_dir="./test_results",
                 save_interval=10):
        # Create the posteriors and sampling method
        posteriors = BGMPosteriors(nr_arms, seed, k, variational_tol, variational_max_iter)
        sampling = sampling_method(posteriors, seed)
        # Super call
        super(BayesianGaussianMixtureBandit, self).__init__(nr_arms, env, sampling, seed, log_dir, save_interval)

    def save(self, t):
        """Save the bandit's posteriors"""
        path = Path(self._log_dir) / "Posteriors"
        path.mkdir(exist_ok=True)
        path /= f"t{t}-"
        self.sampling.posteriors.save(path)

    def load(self, t):
        """Load the bandit's posteriors"""
        path = Path(self._log_dir) / "Posteriors" / f"t{t}-"
        self.sampling.posteriors.load(path)
