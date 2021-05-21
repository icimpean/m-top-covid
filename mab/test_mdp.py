# noinspection PyUnresolvedReferences
import pylibstride as stride

from envs.stride_env.stride_env import StrideMDPEnv
from mab.bandits.bgm_bandit import BayesianGaussianMixtureBandit
from mab.bandits.random_bandit import RandomBandit
from mab.sampling.random import RandomSampling
from mab.sampling.thompson_sampling import ThompsonSampling
from resources.vaccine_supply import ConstantVaccineSupply


if __name__ == '__main__':

    s = 0  # Integer (or None for random seed configuration)
    n_arms = 3 ** 5
    steps = 10

    episode_duration = 2 * 30
    # Bandit decides policy once at the start
    step_size = episode_duration

    vaccine_supply = ConstantVaccineSupply(
        vaccine_type_counts={
            stride.VaccineType.mRNA:  60000,
            stride.VaccineType.adeno: 40000,
        }
    )

    # The type of environment
    env = StrideMDPEnv(states=False, seed=s, episode_duration=episode_duration, step_size=step_size,
                       config_file="../envs/stride_env/run_default_11M.xml",
                       # config_file="../envs/stride_env/run_default.xml",
                       available_vaccines=vaccine_supply,
                       reward_type='norm'
                       )

    # The sampling method
    # sampling_method = ThompsonSampling
    # sampling_method = BFTS.new(top_m=2)
    sampling_method = RandomSampling

    # Gaussian posterior bandit
    # bandit = GaussianBandit(n_arms, env, sampling_method, seed=s)
    # Sklearn-based implementation (Nonparametric Gaussian Mixture posterior bandit)
    bandit = BayesianGaussianMixtureBandit(n_arms, env, sampling_method, k=2, seed=s,
                                           log_dir="../../Data/test_results",
                                           save_interval=10,
                                           )
    # Random bandit
    # bandit = RandomBandit(n_arms, env, sampling_method, seed=s, save_interval=10,
    #                       log_dir="../../Data/test_results"
    #                       )

    # Let the bandit run for the given number of episodes
    # bandit.play_bandit(episodes=3, initialise_arms=3)
    bandit.test_bandit()
    bandit.save("_end")
