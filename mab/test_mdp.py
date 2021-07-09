# noinspection PyUnresolvedReferences
import pylibstride as stride

import os

from envs.stride_env.calendar import create_calendar
from envs.stride_env.stride_env import StrideMDPEnv, Reward
from mab.bandits.bgm_bandit import BayesianGaussianMixtureBandit
from mab.bandits.random_bandit import RandomBandit
from mab.sampling.bfts import BFTS
from mab.sampling.random import RandomSampling
from mab.sampling.thompson_sampling import ThompsonSampling
from mab.visualisation.bandit import BanditVisualisation
from mab.visualisation.posteriors import PosteriorVisualisation
from resources.vaccine_supply import ConstantVaccineSupply


if __name__ == '__main__':

    import time
    t_start = time.time()

    s = 0  # Integer (or None for random seed configuration)
    n_arms = 3 ** 5
    top_m = 50
    k = 2

    # TEST
    log_dir = "../../Data/test_run/600k/"
    config_file = "../envs/stride_env/config1_600k.xml"
    create_calendar(config_file, "../envs/stride_env/600k_calendar.csv")

    episode_duration = 2 * 30
    # Bandit decides policy once at the start
    step_size = episode_duration
    # The maximum number of episodes to run
    episodes = 2000
    # Stop if top-m doesn't change in more than t_max episodes
    t_max = 25

    # The vaccine supply
    vaccine_supply = ConstantVaccineSupply(
        vaccine_type_counts={
            stride.VaccineType.mRNA:  60000,
            stride.VaccineType.adeno: 40000,
        }, population_size=11000000
    )

    # The type of environment
    env = StrideMDPEnv(states=False, seed=s, episode_duration=episode_duration, step_size=step_size,
                       config_file=config_file, available_vaccines=vaccine_supply,
                       reward=Reward.total_infected, reward_type='norm',
                       mRNA_properties=stride.LinearVaccineProperties("mRNA vaccine", 0.95, 0.95, 1.00, 42),
                       adeno_properties=stride.LinearVaccineProperties("Adeno vaccine", 0.67, 0.67, 1.00, 42),
                       )

    # The sampling method
    # sampling_method = ThompsonSampling
    sampling_method = BFTS.new(top_m=top_m)
    # sampling_method = RandomSampling

    # Gaussian posterior bandit
    # bandit = GaussianBandit(n_arms, env, sampling_method, seed=s)
    # Sklearn-based implementation (Nonparametric Gaussian Mixture posterior bandit)  TODO
    bandit = BayesianGaussianMixtureBandit(n_arms, env, sampling_method, k=2, seed=s, log_dir=log_dir, save_interval=10)
    # Random bandit
    # bandit = RandomBandit(n_arms, env, sampling_method, seed=s, save_interval=100, log_dir=log_dir)

    # Stop if top-m doesn't change in more than t_max episodes
    top = None
    t_top = 0

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

    # Let the bandit run for the given number of episodes
    try:
        bandit.play_bandit(episodes=episodes, initialise_arms=3, stop_condition=stop_condition)
        # bandit.test_bandit()
    except KeyboardInterrupt:
        print("stopping early...")

    t_end = time.time()
    print(f"Experiment took {t_end-t_start} seconds")
    bandit.save("_end")

    # Plot the top_m best arms
    v = BanditVisualisation()
    stride_dir = log_dir
    path = f"{stride_dir}/bandit_log.csv"
    v.load_file(path)
    v.plot_top(top_m=top_m)

    p_dir = f"{log_dir}/_vis/"
    os.makedirs(p_dir, exist_ok=True)
    for arm in range(n_arms):
        v.plot_single_arm(arm=arm, stride_csv_directory=stride_dir, file_name=None, plot_average=True,
                          plot_cumulative=False, show=True, save_file=f"{p_dir}/arm_{arm}.png")
        v.plot_single_arm(arm=arm, stride_csv_directory=stride_dir, file_name=None, plot_average=True,
                          plot_cumulative=True, show=True, save_file=f"{p_dir}/arm_{arm}_cumul.png")
