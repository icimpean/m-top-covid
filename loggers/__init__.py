import csv


class LogEntry(object):
    """A log entry containing data for experiments.

    Attributes:
        entry_fields: The names of the (column) fields used for each entry.
    """
    def __init__(self):
        self.entry_fields = []

    def create_entry(self, *args):
        """Method to create an entry for the log."""
        raise NotImplementedError

    def create_file(self, path):
        """Create the CSV file"""
        with open(path, "w", newline="") as file_writer:
            writer = csv.DictWriter(file_writer, fieldnames=self.entry_fields)
            writer.writeheader()

    def write_data(self, data, path):
        """Write a list of entries to a given file."""
        with open(path, "a", newline="") as file_writer:
            writer = csv.DictWriter(file_writer, fieldnames=self.entry_fields)
            writer.writerow(data)
