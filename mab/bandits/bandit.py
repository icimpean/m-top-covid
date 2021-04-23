from loggers.bandit_logger import BanditLogger
from mab.sampling import Sampling


class Bandit(object):
    """"""
    def __init__(self, nr_arms, env, sampling_method: Sampling, seed=None):
        self.nr_arms = nr_arms
        self.env = env
        self.sampling = sampling_method
        self.posteriors = self.sampling.posteriors
        self.seed = seed
        self.logger = BanditLogger()
        self._log_file = "./bandit_log.csv"

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
        """
        self.logger.create_file(self._log_file)

        t = 0
        # Play each arm initialise_arms times
        for arm in range(self.nr_arms):
            for _ in range(initialise_arms):
                state = self.env.reset()

                # Compute the posteriors of the bandit
                self.posteriors.compute_posteriors(t)

                # Play the arm
                next_state, reward, done, info = self.env.step(arm)
                self._print_step(t, arm, reward)
                # Update the posteriors
                self.posteriors.update(arm, reward, t)

                entry = self.logger.create_entry(t, arm, reward)
                self.logger.write_data(entry, self._log_file)

                t += 1
        start_t = self.nr_arms * initialise_arms

        for t in range(start_t, episodes):
            state = self.env.reset()

            # Compute the posteriors of the bandit
            self.posteriors.compute_posteriors(t)

            # Sample and arm
            arm = self.sampling.sample_arm(t)  # TODO: use state
            # Play the arm
            next_state, reward, done, info = self.env.step(arm)
            self._print_step(t, arm, reward)

            # Update the posteriors
            self.posteriors.update(arm, reward, t)

            entry = self.logger.create_entry(t, arm, reward)
            self.logger.write_data(entry, self._log_file)

            t += 1

    @staticmethod
    def _print_step(t, arm, reward):
        print(f"step {t}: Arm {arm}, reward {reward}")
