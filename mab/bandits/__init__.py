import csv
import logging
import os
import time
from pathlib import Path

from envs.stride_env.stride_env import StrideMDPEnv
from loggers.bandit_logger import BanditLogger
from loggers.top_m_logger import TopMLogger
from mab.sampling import Sampling
from mab.sampling.at_lucb import AT_LUCB


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
        self.seed = seed
        self.save_interval = save_interval
        self.logger = BanditLogger()
        self.sample_logger = TopMLogger()
        self._log_dir = Path(log_dir)
        self.log_file = self._log_dir / "bandit_log.csv"
        self.sample_log_file = self._log_dir / "sampling_log.csv"
        os.makedirs(self._log_dir, exist_ok=True)
        self._from_checkpoint = False
        self._time_created = time.time()

    def best_arm(self, t):
        """Select the best arm based on the current posteriors."""
        arm = self.sampling.best_arm(t)
        return arm

    def play_bandit(self, episodes, initialise_arms=0, timestep=0, stop_condition=lambda _: False,
                    time_limit=None, limit_min=20 * 60):
        """Run the bandit for the given number of steps.

        Args:
            episodes: The number of (fixed length) episodes to let the bandit play.
            initialise_arms: The number of times to play each arm to initialise the posteriors at the start.
                Defaults to 0 (no initialisation).
            timestep: The timestep to start playing at. Useful when starting from a checkpoint.
            time_limit: The limit in seconds the bandit can play for (account for time limit on cluster & save on time)
            limit_min: The number of seconds minimum required to continue playing, else save and exit.
        Returns:
            None.
        """
        self.logger.create_file(self.log_file, from_checkpoint=self._from_checkpoint)
        self.sample_logger.create_file(self.sample_log_file, from_checkpoint=self._from_checkpoint)

        t = timestep
        start_t = t
        if not self._from_checkpoint:
            # Play each arm initialise_arms times
            for _ in range(initialise_arms):
                for arm in range(self.nr_arms):
                    self.play_(t, lambda _: arm)
                    t += 1
            start_t = self.nr_arms * initialise_arms

        for t in range(start_t, episodes):
            self.play_(t, self.sampling.sample_arm)

            # Early stop-condition
            if stop_condition(t):
                self._print_stop(t)
                break

            # Save in time if running on cluster
            if self._time_limit(t, time_limit, limit_min):
                break

    def play_(self, t, select_arm):
        """Play an arm selected by the given select_arm function"""
        time_start = time.time()
        # Reset the simulation
        output_prefix = self._log_dir / f"{t}"
        # Create output directories for environments that require it
        if isinstance(self.env, StrideMDPEnv):
            os.makedirs(output_prefix, exist_ok=True)
        state = self.env.reset(seed=t, output_dir=str(self._log_dir), output_prefix=str(output_prefix))

        r = 1
        if isinstance(self.sampling, AT_LUCB):  # plays twice at each step
            r = 2

        for _ in range(r):
            # Compute the posteriors of the bandit
            self.sampling.compute_posteriors(t)
            # Select the arm with the given selection method
            arm = select_arm(t)

            # Play the arm
            next_state, reward, done, info = self.env.step(arm)
            self._print_step(t, arm, reward)
            # Update the posteriors
            self.sampling.update(arm, reward, t)

        time_end = time.time()

        # Log the data
        entry = self.logger.create_entry(t, arm, reward, time_end - time_start)
        self.logger.write_data(entry, self.log_file)

        if self.sampling.has_ranking:
            ranking = self.sampling.current_ranking
            ranking_means = self.sampling.mean_per_arm
            variances = self.sampling.var_per_arm
            std_devs = self.sampling.std_per_arm
        else:
            ranking = None
            ranking_means = None
            variances = None
            std_devs = None
        entry = self.sample_logger.create_entry(t, arm, ranking, ranking_means, variances, std_devs)
        self.sample_logger.write_data(entry, self.sample_log_file)

        # Save the bandit if necessary
        if t % self.save_interval == 0:
            self.save(t)

    def _time_limit(self, t, time_limit, limit_min):
        # Save in time if running on cluster
        if time_limit is not None:
            time_remaining = time_limit - (time.time() - self._time_created)
            if time_remaining < limit_min:
                self._print_limit(t, limit_min, time_remaining)
                self.save("_time")
                return True
            # Save 2 steps before time is up, just in case
            elif time_remaining < (2 * limit_min):
                self.save("_pre_time")
        return False

    @staticmethod
    def _print_step(t, arm, reward):
        logging.info(f"step {t}: Arm {arm}, reward {reward}")

    @staticmethod
    def _print_stop(t):
        logging.info(f"Stopped bandit early due to stopping condition at timestep {t}")

    @staticmethod
    def _print_limit(t, limit_min, time_remaining):
        logging.info(f"Stopped bandit early after timestep {t} due to time limit. "
                     f"Less than {limit_min} seconds remaining ({time_remaining})")

    def test_bandit(self):
        """A small testing case for stride bandit"""
        self.logger.create_file(self.log_file, from_checkpoint=self._from_checkpoint)
        self.sample_logger.create_file(self.sample_log_file, from_checkpoint=self._from_checkpoint)
        t = 0
        some_arms = [0, 1, 5, 10, 20, 21, 25, 30, 45, 50, 51, 55, 56, 67, 68, 100, 101, 120, 151, 160]
        for arm in some_arms:
            self.play_(t, lambda _: arm)
            t += 1
        self.env.close()

    def play_arms(self, arms, timestep=0, time_limit=None, limit_min=20 * 60, callbacks=None):
        self.logger.create_file(self.log_file, from_checkpoint=self._from_checkpoint)
        self.sample_logger.create_file(self.sample_log_file, from_checkpoint=self._from_checkpoint)

        t = timestep
        for i, arm in enumerate(arms):
            self.play_(i + t, lambda _: arm)

            if callbacks is not None:
                for callback in callbacks:
                    callback(i + t, arm)

            if self._time_limit(t, time_limit, limit_min):
                break

    def save(self, t):
        """Save the bandit's sampling method & posteriors"""
        path = Path(self._log_dir)
        path.mkdir(exist_ok=True)
        self.sampling.save(t, path)

    def load(self, t):
        """Load the bandit's sampling method & posteriors"""
        # Playing from given checkpoint
        self._from_checkpoint = True
        path = Path(self._log_dir)
        self.sampling.load(t, path)


def load_rewards(save_directory):
    with open(Path(save_directory) / "bandit_log.csv", mode="r") as file:
        reader = csv.DictReader(file)
        skip_first = True
        rewards = []
        for line in reader:
            if skip_first:
                skip_first = False
                continue
            rewards.append(float(line["Reward"]))
        return rewards
