# noinspection PyUnresolvedReferences
import pylibstride as stride

import argparse
import time
from pathlib import Path

import sys

from mab.sampling.at_lucb import AT_LUCB
from mab.sampling.uniform import UniformSampling

sys.path.append("./")  # for command-line execution to find the other packages (e.g. envs)

from envs.stride_env.stride_env import Reward, StrideGroundTruthEnv
from args import parser as general_parser, create_stride_env, load_checkpoint
from bandits import Bandit
from mab.posteriors.bnpy_gaussian_mixture import BNPYBGMPosteriors
from mab.posteriors.t_distribution import TDistributionPosteriors
from mab.posteriors.truncated_t_distribution import TruncatedTDistributionPosteriors
from sampling.bfts import BFTS


# Play a single arm multiple times
parser = argparse.ArgumentParser(description="=============== MDP STRIDE ===============",
                                 epilog="example:\n\tpython3 mab/play_bandit.py envs/stride_env/config/run_default.xml ../runs/test_run0/ 60 500",
                                 formatter_class=argparse.RawDescriptionHelpFormatter,
                                 parents=[general_parser])
parser.add_argument("--algorithm", type=str, default="BFTS", choices=["BFTS", "AT-LUCB", "Uniform"],)
parser.add_argument("--posterior", type=str, default="T", choices=["T", "TT", "BGM"],
                    help="The type of posterior to use")
parser.add_argument("--top_m", type=int, default=3, help="The m-top to use")
parser.add_argument("--reward", type=str, default="inf", choices=["inf", "hosp", "hosp_neg"], help="The reward to use")
parser.add_argument("--reward_type", type=str, default="norm", choices=["norm", "neg"], help="The reward type to use")


def get_reward(parser_args):
    # Infected
    if parser_args.reward == "inf":
        reward = Reward.total_at_risk
    elif parser_args.reward == "hosp":
        reward = Reward.total_at_risk_hosp
    elif parser_args.reward == "hosp_neg":
        reward = Reward.total_hospitalised
    else:
        raise ValueError(f"Unsupported reward: {parser_args.reward}")
    print("Reward set to", reward.name)
    return reward


def create_posteriors(parser_args, nr_arms):
    initialise_arms = 0
    # t-distribution
    if parser_args.posterior == "T":
        print("Using t-distribution...")
        posteriors = TDistributionPosteriors(nr_arms, parser_args.seed)
        initialise_arms = 2
    # Truncated t-distribution
    elif parser_args.posterior == "TT":
        print("Using truncated t-distribution...")
        posteriors = TruncatedTDistributionPosteriors(nr_arms, parser_args.seed, a=0.0, b=0.1)
        initialise_arms = 2
    # Bayesian Gaussian Mixture
    elif parser_args.posterior == "BGM":
        print("Using Bayesian Gaussian Mixture...")
        posteriors = BNPYBGMPosteriors(nr_arms, parser_args.seed, k=10, log_dir=Path(parser_args.save_dir).absolute())
        initialise_arms = 1
    # Unsupported
    else:
        raise ValueError(f"Unsupported posterior type: {parser_args.posterior}")
    return posteriors, initialise_arms


def create_stride_env_gt(parser_args, reward=Reward.total_at_risk, reward_type='norm'):
    # The type of environment
    use_inf = reward == Reward.total_at_risk
    env = StrideGroundTruthEnv(use_inf, seed=parser_args.seed)
    return env


def run_arm(parser_args):
    t_start = time.time()

    # The type of environment
    reward = get_reward(parser_args)
    env = create_stride_env_gt(parser_args, reward=reward, reward_type=parser_args.reward_type)
    nr_arms = env.action_wrapper.num_actions

    # The sampling method
    log_dir = Path(parser_args.save_dir).absolute()
    initialise_arms = 0
    if parser_args.algorithm == "Uniform":
        sampling_method = UniformSampling(nr_arms, parser_args.top_m, parser_args.seed)
    elif parser_args.algorithm == "AT-LUCB":
        sampling_method = AT_LUCB(nr_arms, parser_args.top_m, seed=parser_args.seed)
    elif parser_args.algorithm == "BFTS":
        posteriors, initialise_arms = create_posteriors(parser_args, nr_arms)
        sampling_method = BFTS(posteriors, parser_args.top_m, seed=parser_args.seed)
    else:
        raise ValueError(parser_args.algorithm)
    # The bandit
    bandit = Bandit(nr_arms, env, sampling_method, seed=parser_args.seed, log_dir=log_dir, save_interval=100)

    # Start from checkpoint if given
    timestep = load_checkpoint(parser_args, bandit)

    # Let the bandit run for the given number of episodes
    try:
        bandit.play_bandit(parser_args.episodes, timestep=timestep, initialise_arms=initialise_arms,
                           time_limit=parser_args.l, limit_min=parser_args.m)
    except KeyboardInterrupt:
        print("Stopping early...")

    t_end = time.time()
    print(f"Experiment took {t_end - t_start} seconds")
    bandit.save("_end")


if __name__ == '__main__':

    import logging
    logging.getLogger().setLevel("INFO")

    # algo = "AT-LUCB"
    # post = "BGM"
    # m = "10"
    # episodes = "10000"
    #
    # step = 1  # TODO 20
    # start_sim = 0 * step
    # rw = "inf"
    # seed = 0
    #
    # args = parser.parse_args([
    #     "no_config.xml", f"../../Data/ground_truth/{rw}/{algo}/{seed}/",
    #     "--episodes", episodes, "--episode_duration", "120", "--reward", rw, "--reward_type", "norm",
    #     "--posterior", post, "--top_m", m, "--seed", f"{seed}", "--algorithm", algo
    # ])
    run_arm(args)
