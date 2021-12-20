import argparse

# General parser for stride experiments
parser = argparse.ArgumentParser(description="MDP STRIDE",
                                 epilog="example:\n\tpython3 TODO",
                                 formatter_class=argparse.RawDescriptionHelpFormatter,
                                 add_help=False)
parser.add_argument("config", type=str, help="The configuration XML file for which to create a calendar")
parser.add_argument("save_dir", type=str, help="The directory to save the results to")

parser.add_argument("--seed", type=int, default=0, help="The seed for the random generators (default: 0)")
parser.add_argument("--episode_duration", type=int, default=60,
                    help="The length of a simulation, in days (default: 60)")
parser.add_argument("--max_episode", type=int, default=2000,
                    help="The maximum number of episodes to run (default: 2000)")

parser.add_argument("-c", type=str, help="Restart from given checkpoint")
parser.add_argument("-t", type=int, help="Restart with given timestep")


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
