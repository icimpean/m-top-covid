import argparse

parser = argparse.ArgumentParser(description="Calendar creator for STRIDE",
                                 epilog="example:\n\tpython3 envs/stride_env/calendar.py envs/stride_env/config/config0_11M.xml envs/stride_env/calendars/test.csv",
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("config", type=str, help="The configuration XML file for which to create a calendar")
parser.add_argument("calendar", type=str, help="The file path for the CSV calendar to create")
parser.add_argument("--no_holidays", action="store_true", help="Skip holidays")
