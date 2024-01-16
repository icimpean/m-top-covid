# noinspection PyUnresolvedReferences
import pylibstride as stride

import argparse

from envs.stride_env.action_wrapper import NoWasteActionWrapper
from envs.stride_env.stride_env import StrideMDPEnv, Reward
from resources.vaccine_supply import ObservedVaccineSupply


# General parser for stride experiments
parser = argparse.ArgumentParser(description="MDP STRIDE",
                                 epilog="example:\n\tpython3 TODO",
                                 formatter_class=argparse.RawDescriptionHelpFormatter,
                                 add_help=False)
parser.add_argument("config", type=str, help="The configuration XML file for which to create a calendar")
parser.add_argument("save_dir", type=str, help="The directory to save the results to")
parser.add_argument("--episodes", type=int, required=True, help="The number of times to play the bandit for")

parser.add_argument("--seed", type=int, default=0, help="The seed for the random generators (default: 0)")
parser.add_argument("--episode_duration", type=int, default=121,
                    help="The length of a simulation, in days (default: 121)")

# Checkpoints
parser.add_argument("-l", type=int, default=24*60*60, help="Time limit in seconds for running the experiment")
parser.add_argument("-m", type=int, default=20*60, help="Minimum time in seconds required to play another episode")
parser.add_argument("-c", type=str, help="Restart from given checkpoint")
parser.add_argument("-t", type=int, help="Restart with given timestep (if -c is a name and not a timestep number)")


def load_checkpoint(parser_args, bandit):
    timestep = 0
    # Start from checkpoint if given
    checkpoint = parser_args.c
    if checkpoint is not None:
        print("LOADING BANDIT FROM CHECKPOINT", checkpoint)
        bandit.load(checkpoint)
        if checkpoint.isnumeric():
            timestep = parser_args.c
        else:
            timestep = parser_args.episodes
        if parser_args.t is not None:
            timestep = parser_args.t
            print("SET CHECKPOINT TIMESTEP", timestep)
    # Return new timestep to start with
    return timestep


def create_stride_env(parser_args, reward=Reward.total_at_risk, reward_type='norm',
                      is_childless=False, action_wrapper=NoWasteActionWrapper, uptake=1,
                      two_doses=False, second_dose_day=28):
    # The vaccine supply
    # Weekly deliveries, based on https://covid-vaccinatie.be/en
    vaccine_supply = ObservedVaccineSupply(starting_date="2021-01-01", days=parser_args.episode_duration,
                                           population_size=11000000, seed=parser_args.seed)
    # The type of environment
    env = StrideMDPEnv(states=False, seed=parser_args.seed, episode_duration=parser_args.episode_duration,
                       step_size=parser_args.episode_duration,  # Bandit decides policy once at the start
                       config_file=parser_args.config,
                       # TODO: add in XML config or parser_args
                       available_vaccines=vaccine_supply,
                       reward=reward, reward_type=reward_type,
                       mRNA_properties=stride.LinearVaccineProperties("mRNA vaccine", 0.95, 0.95, 1.00, 42),
                       adeno_properties=stride.LinearVaccineProperties("Adeno vaccine", 0.67, 0.67, 1.00, 42),
                       action_wrapper=action_wrapper, is_childless=is_childless, uptake=uptake,
                       two_doses=two_doses, second_dose_day=second_dose_day)
    return env
