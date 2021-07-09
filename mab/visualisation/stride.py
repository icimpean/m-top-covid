from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from mab.visualisation import Visualisation


class StrideVisualisation(Visualisation):
    """The visualisation for a stride experiment."""

    def __init__(self):
        # Super call
        super(StrideVisualisation, self).__init__()

        # File names used by stride
        self.exposed = "exposed.csv"
        self.infected = "infected.csv"
        self.infectious = "infectious.csv"
        self.symptomatic = "symptomatic.csv"
        self.hospitalised = "hospitalised.csv"
        #
        self.cases = "cases.csv"
        self.cases_hospitalised = "cases_hospitalised.csv"
        # self.cases_exposed = "cases_exposed.csv"

        # The CSV files generated by stride
        self.files_names = {
            self.exposed: "Exposed",
            self.infected: "Infected",
            self.infectious: "Infectious",
            self.symptomatic: "Symptomatic",
            self.hospitalised: "Hospitalised",
        }
        #
        self.cumulative_file_names = {
            self.cases: "Total Infected Cases",
            self.cases_hospitalised: "Total Hospitalised",
            # self.cases_exposed: "Total Exposed",
        }

    @staticmethod
    def load_file(stride_csv_file):
        with open(stride_csv_file, "r") as file:
            y_values = file.readline()
            y_values = [int(val) for val in y_values.split(sep=",")]
            min_value = min(y_values)
            max_value = max(y_values)
            return y_values, min_value, max_value

    def plot_run(self, stride_csv_directory, episode, show=True, save_file=None):
        # Set up the plot
        self._plot_text(title="Cases for a single run", x_label="Days", y_label="# Individuals", legend=None)

        # Get the data for each of the files:
        min_value = np.inf
        max_value = -np.inf
        for file_name, name in self.files_names.items():
            stride_csv_file = Path(stride_csv_directory) / str(episode) / file_name
            y_values, min_y, max_y = self.load_file(stride_csv_file)
            min_value = min(min_value, min_y)
            max_value = max(max_value, max_y)

            plt.plot(range(len(y_values)), y_values, label=name)

        # Center the graph around the y_values to plot
        plt.ylim(self._center_y_lim(min_value, max_value))
        plt.legend()

        # Show, save and close the plot
        self._show_save_close(show, save_file)