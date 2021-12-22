# noinspection PyUnresolvedReferences
import pylibstride as stride
from pathlib import Path

import argparse
import time

import sys
sys.path.append("./")  # for command-line execution to find the other packages (e.g. envs)

from mab.bandits.random_bandit import RandomBandit
from mab.sampling.random import RandomSampling
from envs.stride_env.action_wrapper import ActionWrapper, NoWasteActionWrapper
from envs.stride_env.stride_env import StrideMDPEnv, Reward
from args import parser as general_parser, load_checkpoint
from resources.vaccine_supply import ObservedVaccineSupply


# Play a single arm multiple times
parser = argparse.ArgumentParser(description="=============== MDP STRIDE ===============",
                                 epilog="example:\n\tpython3 mab/play_bandit.py envs/stride_env/config/run_default.xml ../runs/test_run0/ 60 500",
                                 formatter_class=argparse.RawDescriptionHelpFormatter,
                                 parents=[general_parser])
parser.add_argument("arm", type=int, help="The arm to play")
parser.add_argument("episodes", type=int, help="The number of times to play the given arm")
parser.add_argument("-l", type=int, default=24*60*60, help="Time limit in seconds for running the experiment")
parser.add_argument("-m", type=int, default=20*60, help="Minimum time in seconds required to play another episode")


def run_arm(parser_args):
    t_start = time.time()

    # Bandit decides policy once at the start
    step_size = parser_args.episode_duration

    # The vaccine supply
    # Weekly deliveries, based on https://covid-vaccinatie.be/en
    vaccine_supply = ObservedVaccineSupply(starting_date="2021-01-01", days=parser_args.episode_duration,
                                           population_size=11000000, seed=parser_args.seed)

    # The type of environment
    env = StrideMDPEnv(states=False, seed=parser_args.seed, episode_duration=parser_args.episode_duration,
                       step_size=step_size,
                       config_file=parser_args.config, available_vaccines=vaccine_supply,
                       reward=Reward.total_at_risk, reward_type='norm',  # TODO: add in XML config
                       # TODO: add in XML config
                       mRNA_properties=stride.LinearVaccineProperties("mRNA vaccine", 0.95, 0.95, 1.00, 42),
                       adeno_properties=stride.LinearVaccineProperties("Adeno vaccine", 0.67, 0.67, 1.00, 42),
                       action_wrapper=NoWasteActionWrapper
                       )

    # The sampling method
    sampling_method = RandomSampling
    # Random bandit (random bandit stores no posteriors, only used to play the arms requested by the commandline)
    bandit = RandomBandit(env.nr_arms, env, sampling_method, seed=parser_args.seed,
                          save_interval=10, log_dir=parser_args.save_dir)

    # Start from checkpoint if given
    timestep = load_checkpoint(parser_args, bandit)

    # Let the bandit run for the given number of episodes
    try:
        bandit.play_arms([parser_args.arm] * parser_args.episodes, timestep=timestep,
                         time_limit=parser_args.l, limit_min=parser_args.m)
    except KeyboardInterrupt:
        print("Stopping early...")

    t_end = time.time()
    print(f"Experiment took {t_end - t_start} seconds")
    bandit.save("_end")


if __name__ == '__main__':

    args = parser.parse_args()
    # args = parser.parse_args([
    #     "../envs/stride_env/config/run_default_cmd.xml", "../../Data/run_debug/test_load/", "3", "--episode_duration", "10", "-c", "_end", "-t", "9"
    # ])
    # args = parser.parse_args([
    #     "../envs/stride_env/config/run_default_cmd.xml", "../../Data/run_debug/test_load7/", "5", "50", "-l", "30", "--episode_duration", "10", "-m", "5", #"-c", "_end", "-t", "5"
    # ])

    run_arm(args)
