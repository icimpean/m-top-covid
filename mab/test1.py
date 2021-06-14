from envs.env import GaussianEnv, GaussianMixtureEnv
from mab.bandits.bgm_bandit import BayesianGaussianMixtureBandit
from mab.bandits.gaussian_bandit import GaussianBandit
from mab.bandits.gaussian_mixture_bandit import GaussianMixtureBandit
from mab.sampling.at_lucb import AT_LUCB
from mab.sampling.bfts import BFTS
from mab.sampling.thompson_sampling import ThompsonSampling

import numpy as np

if __name__ == '__main__':

    import time
    t_start = time.time()

    s = 0  # Integer (or None for random seed configuration)
    n_arms = 3 ** 5
    top_m = 100
    k = 3

    steps = 100000
    # The type of environment
    # env = GaussianEnv(n_arms, seed=s, normalised=True)
    env = GaussianMixtureEnv(n_arms, mixtures_per_arm=2, seed=s, normalised=True)
    env.print_info()
    env.plot_rewards()

    # The sampling method
    # sampling_method = ThompsonSampling
    sampling_method = BFTS.new(top_m=top_m)
    # sampling_method = AT_LUCB.new(top_m=3)

    # Gaussian posterior bandit
    # bandit = GaussianBandit(n_arms, env, sampling_method, seed=s)
    # Nonparametric Gaussian Mixture posterior bandit
    # bandit = GaussianMixtureBandit(n_arms, env, k=2, sampling_method=sampling_method, t_max=steps, seed=s)
    # Sklearn-based implementation
    bandit = BayesianGaussianMixtureBandit(n_arms, env, sampling_method, k=k, seed=s,
                                           log_dir="../../Data/run2/GM_env",
                                           save_interval=1000,
                                           )

    # Stop if top-m doesn't change in more than 10 episodes
    top = None
    t_top = 0
    t_max = 25

    def stop_condition(t):
        global top, t_top
        b_top = bandit.sampling.top_m(t)
        if (b_top == top).all():
            t_top += 1
            if t_top >= t_max:
                return True
        else:
            t_top = 0
            top = b_top
        return False

    # Let the bandit run for the given number of steps
    bandit.play_bandit(episodes=steps, initialise_arms=3, stop_condition=stop_condition)

    t_end = time.time()

    print(f"Experiment took {t_end - t_start} seconds")
    bandit.save("_end")

    print(bandit.sampling.top_m(steps))
