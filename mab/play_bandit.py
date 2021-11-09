# noinspection PyUnresolvedReferences
from pathlib import Path

import pylibstride as stride

import argparse
import time

import sys
sys.path.append("./")  # for command-line execution to find the other packages (e.g. envs)

from envs.stride_env.action_wrapper import ActionWrapper, NoWasteActionWrapper
from envs.stride_env.stride_env import StrideMDPEnv, Reward
from args import parser as general_parser
from bandits.bnpy_bandit import BNPYBayesianGaussianMixtureBandit
from sampling.bfts import BFTS
from resources.vaccine_supply import ObservedVaccineSupply


# Play a single arm multiple times
parser = argparse.ArgumentParser(description="=============== MDP STRIDE ===============",
                                 epilog="example:\n\tpython3 mab/play_bandit.py envs/stride_env/config/run_default.xml ../runs/test_run0/ 60 500",
                                 formatter_class=argparse.RawDescriptionHelpFormatter,
                                 parents=[general_parser])
parser.add_argument("episodes", type=int, help="The number of episodes to play")


def run_arm(parser_args):
    t_start = time.time()

    # Bandit decides policy once at the start
    step_size = parser_args.episode_duration

    # The vaccine supply
    # vaccine_supply = ConstantVaccineSupply(  # TODO: add in XML config
    #     vaccine_type_counts={
    #         stride.VaccineType.mRNA: 60000,
    #         stride.VaccineType.adeno: 40000,
    #     }, population_size=11000000
    # )
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
    # from envs.env import TestGaussianMixtureEnv, Test2GaussianMixtureEnv
    # env = Test2GaussianMixtureEnv(seed=parser_args.seed)

    # The sampling method
    sampling_method = BFTS.new(top_m=10)  # TODO: abstract top_m to command line
    # Random bandit (random bandit stores no posteriors, only used to play the arms requested by the commandline)
    bandit = BNPYBayesianGaussianMixtureBandit(env.action_wrapper.num_actions, env, sampling_method, k=2,  # TODO: abstract args to command line
                                               variational_max_iter=10, variational_tol=0.001,
                                               seed=parser_args.seed, log_dir=Path(parser_args.save_dir).absolute(),
                                               save_interval=10)
    # bandit = RandomBandit(env.nr_arms, env, sampling_method, seed=parser_args.seed,
    #                       save_interval=100, log_dir=parser_args.save_dir)

    # Let the bandit run for the given number of episodes
    try:
        bandit.play_bandit(parser_args.episodes, initialise_arms=0)  # TODO
    except KeyboardInterrupt:
        print("Stopping early...")

    t_end = time.time()
    print(f"Experiment took {t_end - t_start} seconds")
    bandit.save("_end")


if __name__ == '__main__':

    args = parser.parse_args()
    # args = parser.parse_args([
    #     "envs/stride_env/config/run_default.xml", "../../Data/run_debug/test0/", "50", "--episode_duration", "20"
    # ])

    run_arm(args)
