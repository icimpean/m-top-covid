import os

from envs.env import GaussianMixtureEnv
from mab.bandits.bgm_bandit import BayesianGaussianMixtureBandit
from mab.bandits.random_bandit import RandomBandit
from mab.sampling.bfts import BFTS
from mab.sampling.random import RandomSampling
from mab.sampling.thompson_sampling import ThompsonSampling
from mab.visualisation.bandit import BanditVisualisation
from mab.visualisation.posteriors import PosteriorVisualisation


if __name__ == '__main__':

    import time
    t_start = time.time()

    s = 0  # Integer (or None for random seed configuration)
    n_arms = 20  # TODO 3 ** 5
    top_m = n_arms // 2  # TODO
    k = 2

    # TEST
    log_dir = "../../Data/test_run/bgm_small/sample_rewards/"

    # The maximum number of episodes to run
    episodes = 3000
    # The type of environment
    # env = GaussianEnv(n_arms, seed=s, normalised=True)
    env = GaussianMixtureEnv(n_arms, mixtures_per_arm=2, seed=s, normalised=False)
    # env.print_info()
    # env.plot_rewards(show=True)

    # The sampling method
    # sampling_method = ThompsonSampling
    sampling_method = BFTS.new(top_m=top_m)
    # sampling_method = RandomSampling

    # Gaussian posterior bandit
    # bandit = GaussianBandit(n_arms, env, sampling_method, seed=s)
    # Sklearn-based implementation (Nonparametric Gaussian Mixture posterior bandit)
    bandit = BayesianGaussianMixtureBandit(n_arms, env, sampling_method, k=2,
                                           variational_max_iter=100, variational_tol=0.001,
                                           seed=s, log_dir=log_dir, save_interval=10)
    # Random bandit
    # bandit = RandomBandit(n_arms, env, sampling_method, seed=s, sa
    # ve_interval=100, log_dir=log_dir)

    # Let the bandit run for the given number of episodes
    try:
        bandit.play_bandit(episodes=episodes, initialise_arms=3)
    except KeyboardInterrupt:
        print("stopping early...")

    t_end = time.time()
    print(f"Experiment took {t_end-t_start} seconds")
    bandit.save("_end")

    arms = range(n_arms)
    pv = PosteriorVisualisation()
    p_dir = f"{log_dir}/_vis/"
    os.makedirs(p_dir, exist_ok=True)

    pv.plot_posteriors(bandit, requested_arms=None, t="_end", n_samples=2000, show=False, save_file=f"{p_dir}/posteriors.png")

    bandit.posteriors.print_posteriors()
    env.print_info()
    env.plot_rewards(save_file=f"{p_dir}/env_rewards.png")
