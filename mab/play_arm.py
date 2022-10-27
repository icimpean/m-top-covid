# noinspection PyUnresolvedReferences
import pylibstride as stride

from pathlib import Path
import argparse
import time

import sys
sys.path.append("./")  # for command-line execution to find the other packages (e.g. envs)

from mab.bandits.random_bandit import RandomBandit
from envs.stride_env.stride_env import Reward
from args import parser as general_parser, load_checkpoint, create_stride_env

# Play a single arm multiple times
parser = argparse.ArgumentParser(description="=============== MDP STRIDE ===============",
                                 epilog="example:\n\tpython3 mab/play_arm.py ./envs/stride_env/config/conf_0/config.xml ./results/top_10/TT/ --arm 0 --episodes 2000 --episode_duration 120 --seed 123",
                                 formatter_class=argparse.RawDescriptionHelpFormatter,
                                 parents=[general_parser])
parser.add_argument("--arm", type=str, required=True, help="The arm to play")


def run_arm(parser_args):
    t_start = time.time()

    parser_args.save_dir = Path(parser_args.save_dir).absolute()
    # No arm was given => no action
    arm = parser_args.arm
    arm = int(arm) if arm.isdigit() else None

    # The type of environment
    env = create_stride_env(parser_args, reward=Reward.total_at_risk)

    # Random bandit (random bandit stores no posteriors, only used to play the arms requested by the commandline)
    bandit = RandomBandit(env.nr_arms, env, seed=parser_args.seed, save_interval=10, log_dir=parser_args.save_dir)
    # Start from checkpoint if given
    timestep = load_checkpoint(parser_args, bandit)

    # Let the bandit run for the given number of episodes
    try:
        bandit.play_arms([arm] * parser_args.episodes, timestep=timestep,
                         time_limit=parser_args.l, limit_min=parser_args.m)
    except KeyboardInterrupt:
        print("Stopping early...")

    t_end = time.time()
    print(f"Experiment took {t_end - t_start} seconds")
    bandit.save("_end")


if __name__ == '__main__':

    import logging
    logging.getLogger().setLevel("INFO")

    args = parser.parse_args()
    # a = "179" #"None"
    # args = parser.parse_args([
    #     "../envs/stride_env/config/conf_0/config_bandit1_600k.xml", f"../../Data/debug/arms_{a}/",
    #     "--episodes", "20", "--arm", a, "--episode_duration", "121",
    #     # "-l", "30", "-m", "5",
    #     # "-c", "_time", "-t", "8",
    # ])

    run_arm(args)
