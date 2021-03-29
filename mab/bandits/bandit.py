from mab.sampling import Sampling


class Bandit(object):
    """"""
    def __init__(self, nr_arms, env, sampling_method: Sampling, seed=None):
        self.nr_arms = nr_arms
        self.env = env
        self.sampling = sampling_method
        self.posteriors = self.sampling.posteriors

    def best_arm(self, t):
        """Select the best arm based on the current posteriors."""
        arm = self.posteriors.sample_best_arm(t)
        return arm

    def play_bandit(self, steps):
        """Run the bandit for the given number of steps"""
        # TODO: initialise the environment context
        state = None

        for t in range(steps):
            # Compute the posteriors of the bandit
            self.posteriors.compute_posteriors(t)

            # Sample and arm
            arm = self.sampling.sample_arm(t)  # TODO: use state
            # Play the arm
            next_state, reward, done, info = self.env.step(arm)

            # Update the posteriors
            self.posteriors.update(arm, reward, t)
            # next_state becomes current state
            state = next_state

        # TODO: true rewards + regret?

    # def plot_posteriors(self):  # TODO remove/replace: depends on posterior distributions, not bandit class
    #     plt.title(f"Posterior rewards for {self.nr_arms}-armed Bandit")
    #     for arm, posterior in enumerate(self.posteriors):
    #         mean = posterior.mean()
    #         std_dev = posterior.sigma()
    #         x = np.linspace(mean - 3*std_dev**2, mean + 3*std_dev**2, 100)
    #         y = stats.norm.pdf(x, mean, std_dev**2)
    #         plt.plot(x, y, label=str(arm))
    #         print(f"arm {arm}: mean {mean}, std. dev. {std_dev} (variance {std_dev**2})")
    #     plt.legend()
    #     plt.savefig("posteriors.png")
    #     # plt.show()
    #     plt.close()
