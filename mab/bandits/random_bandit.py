import os
import time
from typing import Type

from mab.bandits.bandit import Bandit
from mab.sampling import Sampling


class RandomBandit(Bandit):
    """A multi-armed bandit with random sampling and no posteriors.

    Attributes:
        nr_arms: The number of arms the bandit has.
        env: The Env instance for the bandit to interact with.
        sampling_method: The sampling method for sampling arms and computing the posteriors.
        seed: The seed to use for random generations.
        log_dir: The directory where to store the log files for both the bandit and environment (if applicable).
        save_interval: After how many episodes to save the bandit.
    """
    def __init__(self, nr_arms, env, sampling_method: Type[Sampling], seed=None, log_dir="./test_results", save_interval=1):
        # Create the posteriors and sampling method
        posteriors = [None] * nr_arms
        sampling = sampling_method(posteriors, seed)
        # Super call
        super(RandomBandit, self).__init__(nr_arms, env, sampling, seed, log_dir, save_interval)

    def best_arm(self, t):
        """Select the best arm based on the current posteriors."""
        arm = self.sampling.best_arm(t)
        return arm

    def _play(self, t, select_arm):
        """Play an arm selected by the given select_arm function"""
        time_start = time.time()
        # Reset the simulation
        output_prefix = self._log_dir / f"{t}"
        os.makedirs(output_prefix, exist_ok=True)
        state = self.env.reset(seed=t, output_dir=str(self._log_dir), output_prefix=str(output_prefix))

        # Select the arm with the given selection method
        arm = select_arm(t)

        # Play the arm
        next_state, reward, done, info = self.env.step(arm)
        self._print_step(t, arm, reward)

        time_end = time.time()

        # Log the data
        entry = self.logger.create_entry(t, arm, reward, time_end - time_start)
        self.logger.write_data(entry, self.log_file)
        # Save the bandit if necessary
        # if t % self.save_interval == 0:
        #     self.save(t)

    def save(self, t):
        """Save the bandit's weights/posteriors"""
        pass

    def load(self, t):
        """Load the bandit's weights/posteriors"""
        pass