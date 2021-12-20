from pathlib import Path

import argparse
import time

import sys

sys.path.append("./")  # for command-line execution to find the other packages (e.g. envs)

from args import parser as general_parser
from bandits.t_distribution_bandit import TDistributionBandit
from envs.flute_env.flute_env import FluteMDPEnv
from sampling.bfts import BFTS
from sampling.at_lucb import AT_LUCB
from sampling.uniform import UniformSampling

# Play a single arm multiple times
parser = argparse.ArgumentParser(description="=============== MDP STRIDE ===============",
                                 epilog="example:\n\tpython3 mab/play_bandit.py envs/stride_env/config/run_default.xml ../runs/test_run0/ 60 500",
                                 formatter_class=argparse.RawDescriptionHelpFormatter,
                                 parents=[general_parser])
parser.add_argument("episodes", type=int, help="The number of episodes to play")
parser.add_argument("R0", type=float, help="The R0 for the environment to use")


def run_arm(parser_args):  # , sampling_method):
    t_start = time.time()

    # Bandit decides policy once at the start
    step_size = parser_args.episode_duration

    # The type of environment
    env = FluteMDPEnv(seed=parser_args.seed, R0=parser_args.R0)

    # The sampling method
    sampling_method = BFTS.new(top_m=5)  # TODO: abstract sampling method and top_m to command line
    bandit = TDistributionBandit(env.action_wrapper.num_actions, env, sampling_method,
                                 seed=parser_args.seed, log_dir=Path(parser_args.save_dir).absolute(),
                                 save_interval=1000)

    # Let the bandit run for the given number of episodes
    try:
        bandit.play_bandit(parser_args.episodes, initialise_arms=2)
    except KeyboardInterrupt:
        print("Stopping early...")

    t_end = time.time()
    print(f"Experiment took {t_end - t_start} seconds")
    bandit.save("_end")


if __name__ == '__main__':
    args = parser.parse_args()
    # args = parser.parse_args([
    #     "x", "../../Data/run_debug/flute_test2/", "500", "1.4"
    # ])
    run_arm(args)

    # t_st = time.time()
    # r0 = [1.4, 1.6, 1.8, 2.0, 2.2, 2.4]
    # num_sims = 100
    # for m in [3, 5]:
    #     samp = {
    #         "BFTS": BFTS.new(top_m=m),
    #         "AT_LUCB": AT_LUCB.new(top_m=m),
    #         "Uniform": UniformSampling.new(top_m=m)
    #     }
    #     for name, sm in samp.items():
    #         for r in r0:
    #             for n in range(num_sims):
    #                 args = parser.parse_args([
    #                     "x", f"../../Data/influenza/top-{m}/{name}/R0_{r}/{n}/", "500", f"{r}",
    #                 ])
    #                 run_arm(args, sm)
    # t_nd = time.time()
    # print(f"Experiment took {t_nd - t_st} seconds")
