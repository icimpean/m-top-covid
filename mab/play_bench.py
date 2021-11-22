# noinspection PyUnresolvedReferences
import pylibstride as stride

import argparse
import time

import sys
sys.path.append("./")  # for command-line execution to find the other packages (e.g. envs)

from envs.stride_env.action_wrapper import NoWasteActionWrapper
from envs.stride_env.stride_env import StrideMDPEnv, Reward
from args import parser as general_parser
from bandits.random_bandit import RandomBandit
from sampling.random import RandomSampling
from resources.vaccine_supply import ObservedVaccineSupply


# Play a single arm multiple times
parser = argparse.ArgumentParser(description="=============== MDP STRIDE ===============",
                                 epilog="example:\n\tTODO",
                                 formatter_class=argparse.RawDescriptionHelpFormatter,
                                 parents=[general_parser])


def run_arm(parser_args):
    t_start = time.time()

    # The arm(s) to run
    arms = [0, 79, 115] * 3
    # Bandit decides policy once at the start
    episode_duration = 120
    step_size = episode_duration

    seed = 123

    # The vaccine supply
    # Weekly deliveries, based on https://covid-vaccinatie.be/en
    vaccine_supply = ObservedVaccineSupply(starting_date="2021-01-01", days=episode_duration,
                                           population_size=11000000, seed=seed)

    # The type of environment
    env = StrideMDPEnv(states=False, seed=seed, episode_duration=episode_duration,
                       step_size=step_size,
                       config_file=parser_args.config, available_vaccines=vaccine_supply,
                       reward=Reward.total_infected, reward_type='norm',
                       mRNA_properties=stride.LinearVaccineProperties("mRNA vaccine", 0.95, 0.95, 1.00, 42),
                       adeno_properties=stride.LinearVaccineProperties("Adeno vaccine", 0.67, 0.67, 1.00, 42),
                       action_wrapper=NoWasteActionWrapper
                       )

    # The sampling method
    sampling_method = RandomSampling
    # Random bandit (random bandit stores no posteriors, only used to play the arms requested by the commandline)
    bandit = RandomBandit(env.nr_arms, env, sampling_method, seed=seed,
                          save_interval=100, log_dir=parser_args.save_dir)

    # Let the bandit run for the given number of episodes
    try:
        bandit.play_arms(arms)
    except KeyboardInterrupt:
        print("Stopping early...")

    t_end = time.time()
    print(f"Experiment took {t_end - t_start} seconds")
    bandit.save("_end")


if __name__ == '__main__':

    args = parser.parse_args()
    run_arm(args)
