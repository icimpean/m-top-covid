from envs.stride_env.stride_env import BanditStrideMDPEnv
from mab.bandits.bgm_bandit import BayesianGaussianMixtureBandit
from mab.sampling.thompson_sampling import ThompsonSampling

if __name__ == '__main__':


    # import pylibstride as stride
    # print(stride.AllVaccineTypes, stride.AllAgeGroups)
    #
    # exit()


    s = 0  # Integer (or None for random seed configuration)
    # n_arms = 3 * 3 ** 5  # TODO: 3 decisions
    n_arms = 3 ** 5  # TODO: 3 decisions
    steps = 10

    episode_duration = 1 * 30
    step_size = 1 * 30

    # The type of environment
    env = BanditStrideMDPEnv(states=False, seed=0, episode_duration=episode_duration, step_size=step_size,
                             config_file="../envs/stride_env/run_default.xml", vaccine_availability=None)

    # The sampling method
    sampling_method = ThompsonSampling
    # sampling_method = BFTS.new(top_m=2)

    # Gaussian posterior bandit
    # bandit = GaussianBandit(n_arms, env, sampling_method, seed=s)
    # Sklearn-based implementation (Nonparametric Gaussian Mixture posterior bandit)
    bandit = BayesianGaussianMixtureBandit(n_arms, env, sampling_method, k=2, seed=s)

    # Let the bandit run for the given number of steps
    # bandit.play_bandit(episodes=3, initialise_arms=3)  # TODO init = 3
    bandit.play_bandit(episodes=3, initialise_arms=1)
