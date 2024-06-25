# noinspection PyUnresolvedReferences
import pylibstride as stride

import argparse
import time
from pathlib import Path

import sys
sys.path.append("./")  # for command-line execution to find the other packages (e.g. envs)

from envs.stride_env.stride_env import Reward, StrideGroundTruthEnv
from args import parser as general_parser, create_stride_env, load_checkpoint
from bandits import Bandit
from mab.posteriors.bayesian_gaussian_mixture import BGMPosteriors
from mab.posteriors.bnpy_gaussian_mixture import BNPYBGMPosteriors
# from mab.posteriors.gamma import GammaPosteriors
from mab.posteriors.t_distribution import TDistributionPosteriors
from mab.posteriors.truncated_t_distribution import TruncatedTDistributionPosteriors
from sampling.bfts import BFTS
from mab.sampling.at_lucb import AT_LUCB
from mab.sampling.uniform import UniformSampling


# Play a single arm multiple times
parser = argparse.ArgumentParser(description="=============== MDP STRIDE ===============",
                                 epilog="example:\n\tpython3 mab/play_bandit_gt.py ./envs/stride_env/config/conf_0/config.xml ./results/top_10/TT/ --episodes 2000 --episode_duration 120 --reward inf --reward_type norm --posterior TT --top_m 10 --seed 123",
                                 formatter_class=argparse.RawDescriptionHelpFormatter,
                                 parents=[general_parser])
parser.add_argument("--algorithm", type=str, default="BFTS", choices=["BFTS", "AT-LUCB", "Uniform"],)
parser.add_argument("--posterior", type=str, default="T", choices=["T", "TT", "BGM", "GM", "Gamma"],
                    help="The type of posterior to use")
parser.add_argument("--top_m", type=int, default=3, help="The m-top to use")
parser.add_argument("--reward", type=str, default="inf", choices=["inf", "hosp", "hosp_neg"], help="The reward to use")
parser.add_argument("--reward_type", type=str, default="norm", choices=["norm", "neg"], help="The reward type to use")
parser.add_argument("--reward_factor", type=int, default=1, help="The reward factor to multiply the reward with")
parser.add_argument("--sF", type=float, default=0.1, help="The scaling factor for the gaussian mixture prior")


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
        tol = 0.001
        sF = parser_args.sF
        posteriors = BNPYBGMPosteriors(nr_arms, parser_args.seed, k=10, tol=tol, sF=sF,
                                       log_dir=Path(parser_args.save_dir).absolute())
        initialise_arms = 1
    # Sklearn BGM
    elif parser_args.posterior == "GM":
        print("Using Sklearn Bayesian Gaussian Mixture...")
        posteriors = BGMPosteriors(nr_arms, parser_args.seed, k=10, log_dir=Path(parser_args.save_dir).absolute())
        initialise_arms = 1
    # # Gamma
    # elif parser_args.posterior == "Gamma":
    #     print("Using Gamma distribution...")
    #     posteriors = GammaPosteriors(nr_arms, parser_args.seed, alpha=0.5, beta=0)
    #     initialise_arms = 1
    # Unsupported
    else:
        raise ValueError(f"Unsupported posterior type: {parser_args.posterior}")
    return posteriors, initialise_arms


def create_stride_env_gt(parser_args, reward=Reward.total_at_risk, reward_type='norm', reward_factor=1):
    # The type of environment
    use_inf = reward == Reward.total_at_risk
    env = StrideGroundTruthEnv(use_inf, reward_type=reward_type, reward_factor=reward_factor, seed=parser_args.seed)
    return env


def run_arm(parser_args):
    t_start = time.time()

    # The type of environment
    reward = get_reward(parser_args)
    env = create_stride_env_gt(parser_args, reward=reward, reward_type=parser_args.reward_type,
                               reward_factor=parser_args.reward_factor)
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
    bandit = Bandit(nr_arms, env, sampling_method, seed=parser_args.seed, log_dir=log_dir, save_interval=1000000)

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
    # bandit.save("_end")


if __name__ == '__main__':

    import logging
    # logging.getLogger().setLevel("INFO")
    logging.getLogger().setLevel("WARNING")

    args = parser.parse_args()
    s_start = args.seed
    s_total = 20

    # algo = "BFTS"
    # post = "TT"
    # m = "10"
    # episodes = "10000"
    # rw = "inf"
    # r_type = "norm"
    # rf = "1"
    # sF = "0.001"
    # seed = 0

    import time
    tstart = time.time()
    for seed in range(s_start, s_start + s_total):
        print(f"Simulation {seed}")
        t0 = time.time()
        # args = parser.parse_args([
        #     "no_config.xml", f"../../../Data/ground_truth2024/{rw}{'-' + r_type if r_type != 'norm' else ''}/"
        #                      f"{algo}{'-' + post}{'-' + rf if float(rf) != 1 else ''}-{sF}/{seed}/",
        #     "--episodes", episodes, "--episode_duration", "120", "--reward", rw, "--reward_type", r_type,
        #     "--reward_factor", rf, "--sF", sF,
        #     "--posterior", post, "--top_m", m, "--seed", f"{seed}", "--algorithm", algo,
        # ])
        args.seed = seed
        args.save_dir = "".join(args.save_dir.split("/")[:-1]) + f"/{seed}/"
        run_arm(args)
        t1 = time.time()
        print(f"Simulation {seed}, {t1-t0} seconds")
    print(f"{time.time()-tstart} seconds total")

    # args = parser.parse_args()
    # run_arm(args)
