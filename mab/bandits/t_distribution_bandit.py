from pathlib import Path
from typing import Type

from mab.bandits.bandit import Bandit
from mab.posteriors.t_distribution import TDistributionPosteriors
from mab.sampling import Sampling


class TDistributionBandit(Bandit):
    """The class implementing a t-distribution bandit with Thompson sampling"""
    def __init__(self, nr_arms, env, sampling_method: Type[Sampling], seed=None, log_dir="./test_results",
                 save_interval=1):
        # Create the posteriors and sampling method
        posteriors = TDistributionPosteriors(nr_arms, seed)
        sampling = sampling_method(posteriors, seed)
        # Super call
        super(TDistributionBandit, self).__init__(nr_arms, env, sampling, seed, log_dir, save_interval)

    def save(self, t):
        """Save the bandit's posteriors"""
        path = Path(self._log_dir) / "Posteriors"
        path.mkdir(exist_ok=True)
        path /= f"t{t}-"
        self.sampling.posteriors.save(path)

    def load(self, t):
        """Load the bandit's posteriors"""
        # Playing from given checkpoint
        self._from_checkpoint = True
        path = Path(self._log_dir) / "Posteriors" / f"t{t}-"
        self.sampling.posteriors.load(path)
