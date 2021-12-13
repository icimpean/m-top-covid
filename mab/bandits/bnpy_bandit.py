from pathlib import Path
from typing import Type

from mab.bandits.bandit import Bandit
from mab.posteriors.bnpy_gaussian_mixture import BNPYBGMPosteriors
from mab.sampling import Sampling
from mab.sampling.thompson_sampling import ThompsonSampling


class BNPYBayesianGaussianMixtureBandit(Bandit):
    """The class implementing a bayesian gaussian mixture bandit with the BNPY implementation."""

    def __init__(self, nr_arms, env, sampling_method: Type[Sampling] = ThompsonSampling,
                 k=2, variational_max_iter=100, variational_tol=0.001, seed=None, log_dir="./test_results",
                 save_interval=10):
        # Create the posteriors and sampling method
        posteriors = BNPYBGMPosteriors(nr_arms, seed, k, variational_tol, variational_max_iter, log_dir)
        sampling = sampling_method(posteriors, seed)
        # Super call
        super(BNPYBayesianGaussianMixtureBandit, self).__init__(nr_arms, env, sampling, seed, log_dir, save_interval)

    def save(self, t):
        """Save the bandit's posteriors"""
        # TODO
        path = Path(self._log_dir) / "Posteriors"
        path.mkdir(exist_ok=True)
        path /= f"t{t}-"
        self.sampling.posteriors.save(path)

    def load(self, t):
        """Load the bandit's posteriors"""
        # Playing from given checkpoint
        self._from_checkpoint = True
        # TODO
        path = Path(self._log_dir) / "Posteriors" / f"t{t}-"
        self.sampling.posteriors.load(path)
