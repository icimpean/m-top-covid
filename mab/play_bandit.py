# noinspection PyUnresolvedReferences
import pylibstride as stride

import argparse
import time
from pathlib import Path

import sys
sys.path.append("./")  # for command-line execution to find the other packages (e.g. envs)

from envs.stride_env.action_wrapper import NoChildrenNoWasteActionWrapper, NoWasteActionWrapper
from envs.stride_env.stride_env import Reward
from args import parser as general_parser, create_stride_env, load_checkpoint
from bandits import Bandit
from mab.posteriors.bnpy_gaussian_mixture import BNPYBGMPosteriors
from mab.posteriors.t_distribution import TDistributionPosteriors
from mab.posteriors.truncated_t_distribution import TruncatedTDistributionPosteriors
from sampling.bfts import BFTS


# Play a single arm multiple times
parser = argparse.ArgumentParser(description="=============== MDP STRIDE ===============",
                                 epilog="example:\n\tpython3 mab/play_bandit.py ../envs/stride_env/config/conf_0/config.xml ./results/top_10/TT/ --episodes 2000 --episode_duration 120 --reward inf --reward_type norm --posterior TT --top_m 10 --seed 123",
                                 formatter_class=argparse.RawDescriptionHelpFormatter,
                                 parents=[general_parser])
parser.add_argument("--posterior", type=str, default="T", choices=["T", "TT", "BGM"],
                    help="The type of posterior to use")
parser.add_argument("--top_m", type=int, default=3, help="The m-top to use")
parser.add_argument("--reward", type=str, default="inf", choices=["inf", "hosp", "hosp_neg"], help="The reward to use")
parser.add_argument("--reward_type", type=str, default="norm", choices=["norm", "neg"], help="The reward type to use")
parser.add_argument("--reward_factor", type=int, default=1, help="The reward factor to multiply the reward with")
parser.add_argument("--sF", type=float, default=0.1, help="The scaling factor for the gaussian mixture prior")
parser.add_argument('--childless', dest='childless', action='store_true', help="Exclude children from vaccines")
parser.add_argument("--uptake", type=float, default=1.0, help="The vaccine uptake in the population")


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
    # Unsupported
    else:
        raise ValueError(f"Unsupported posterior type: {parser_args.posterior}")
    return posteriors, initialise_arms


def run_bandit(parser_args):
    t_start = time.time()

    # The type of environment
    reward = get_reward(parser_args)
    env = create_stride_env(parser_args, reward=reward, reward_type=parser_args.reward_type,
                            is_childless=parser_args.childless, uptake=parser_args.uptake,
                            action_wrapper=NoChildrenNoWasteActionWrapper if parser_args.childless else NoWasteActionWrapper,
                            two_doses=True, second_dose_day=28)  # TODO: abstract
    nr_arms = env.action_wrapper.num_actions

    # The sampling method
    log_dir = Path(parser_args.save_dir).absolute()
    posteriors, initialise_arms = create_posteriors(parser_args, nr_arms)
    sampling_method = BFTS(posteriors, parser_args.top_m, seed=parser_args.seed)
    # The bandit
    bandit = Bandit(nr_arms, env, sampling_method, seed=parser_args.seed, log_dir=log_dir, save_interval=10)

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

    args = parser.parse_args()
    # post = "TT"
    # m = "10"
    # uptake = 0.9
    # args = parser.parse_args([
    #     "./envs/stride_env/config/conf_1/config.xml", f"/Users/alexandracimpean/Desktop/VSC_stride/SC1_Rhosp_U0.9/seed_0/",
    #     "--episodes", "2000", "--episode_duration", "120", "--reward", "hosp", "--reward_type", "norm",
    #     "--posterior", post, "--top_m", m, "--seed", "0",
    #     "--uptake", f"{uptake}",
    #     # "--childless"
    #     # "-l", "30", "-m", "5",
    #     # "-c", "350", "-t", "351",
    # ])

    run_bandit(args)
