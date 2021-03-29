from mab.posteriors import Posteriors
from mab.sampling import Sampling


class ThompsonSampling(Sampling):
    """"""
    def __init__(self, posteriors: Posteriors, seed):
        super(ThompsonSampling, self).__init__(posteriors, seed)

    def sample_arm(self, t):
        """Select the arm with the best sampled reward."""
        return self.best_arm(t)


# class TopTwoThompsonSampling(ThompsonSampling):
#     """"""
#     def __init__(self, seed):
#         super().__init__(seed)
#
#     def _sample_bernoulli_arm(self, posteriors: Posteriors, beta):
#         # Sample the best arm
#         arm = self.best_arm(posteriors)
#         # Sample bernoulli
#         b = bernoulli.rsv(b=beta, size=1)
#         if b != 0:
#             # Return the best arm
#             return arm
#         else:
#             # Return the second best arm
#             def resample(i, max_tries):
#                 if i == max_tries:
#                     # Stop sampling and sample randomly
#                     print("Max tries reached")
#                     random_arm = self.rng.uniform(0, len(posteriors))
#                     return random_arm
#                 else:
#                     # Resample the "best" arm
#                     resampled_arm = self.best_arm(posteriors)
#                     # If the resampled arm differs, return it => second best
#                     if arm != resampled_arm:
#                         return resampled_arm
#                     # Retry
#                     else:
#                         return resample(i + 1, max_tries)
#             top_two_arm = resample(0, max_tries=1000)
#             return top_two_arm
