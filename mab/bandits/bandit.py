import os
import time
from pathlib import Path

from loggers.bandit_logger import BanditLogger
from mab.sampling import Sampling


class Bandit(object):
    """A multi-armed bandit.

    Attributes:
        nr_arms: The number of arms the bandit has.
        env: The Env instance for the bandit to interact with.
        sampling_method: The sampling method for sampling arms and computing the posteriors.
        seed: The seed to use for random generations.
        log_dir: The directory where to store the log files for both the bandit and environment (if applicable).
        save_interval: After how many episodes to save the bandit.
    """
    def __init__(self, nr_arms, env, sampling_method: Sampling, seed=None, log_dir="./test_results", save_interval=1):
        self.nr_arms = nr_arms
        self.env = env
        self.sampling = sampling_method
        self.posteriors = self.sampling.posteriors
        self.seed = seed
        self.save_interval = save_interval
        self.logger = BanditLogger()
        self._log_dir = Path(log_dir)
        self._log_file = self._log_dir / "bandit_log.csv"
        os.makedirs(self._log_dir, exist_ok=True)

    def best_arm(self, t):
        """Select the best arm based on the current posteriors."""
        arm = self.posteriors.sample_best_arm(t)
        return arm

    def play_bandit(self, episodes, initialise_arms=0):
        """Run the bandit for the given number of steps.

        Args:
            episodes: The number of (fixed length) episodes to let the bandit play.
            initialise_arms: The number of times to play each arm to initialise the posteriors at the start.
                Defaults to 0 (no initialisation).

        Returns:
            None.
        """
        self.logger.create_file(self._log_file)

        t = 0
        # Play each arm initialise_arms times
        for arm in range(self.nr_arms):
            for _ in range(initialise_arms):
                self._play(t, lambda _: arm)
                t += 1
        start_t = self.nr_arms * initialise_arms

        for t in range(start_t, episodes):
            self._play(t, self.sampling.sample_arm)

    def _play(self, t, select_arm):
        """Play an arm selected by the given select_arm function"""
        time_start = time.time()
        # Reset the simulation
        output_prefix = self._log_dir / f"{t}"
        os.makedirs(output_prefix, exist_ok=True)
        state = self.env.reset(seed=t, output_dir=str(self._log_dir), output_prefix=str(output_prefix))

        # Compute the posteriors of the bandit
        self.posteriors.compute_posteriors(t)
        # Select the arm with the given selection method
        arm = select_arm(t)

        # Play the arm
        next_state, reward, done, info = self.env.step(arm)
        self._print_step(t, arm, reward)
        # Update the posteriors
        self.posteriors.update(arm, reward, t)

        time_end = time.time()

        # Log the data
        entry = self.logger.create_entry(t, arm, reward, time_end - time_start)
        self.logger.write_data(entry, self._log_file)
        # Save the bandit if necessary
        if t % self.save_interval == 0:
            self.save(t)

    @staticmethod
    def _print_step(t, arm, reward):
        print(f"step {t}: Arm {arm}, reward {reward}")

    def test_bandit(self):
        """A small testing case for stride bandit, playing one arm for 0 to 5 age groups vaccinated"""
        self.logger.create_file(self._log_file)
        t = 0
        some_arms = [0, 1, 11, 23, 123, 230]
        for arm in some_arms:
            self._play(t, lambda _: arm)
            t += 1

        # TODO: remove
        # self.env.close_x()
        # import resource
        # x = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        # print(f"End Resources: {x} bytes ({round(x / 1024, 2)} kB, {round(x / 1024 ** 2, 2)} MB, {round(x / 1024 ** 3, 2)} GB)")

        #
        self.env.close()

    def save(self, t):
        """Save the bandit's weights/posteriors"""
        raise NotImplementedError

    def load(self, t):
        """Load the bandit's weights/posteriors"""
        raise NotImplementedError
