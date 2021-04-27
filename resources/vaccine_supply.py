# noinspection PyUnresolvedReferences
import pylibstride as stride

import csv
import urllib.request as request


class VaccineSupply(object):
    """A vaccine supply to get the available vaccines per day from."""
    # All vaccine types, except noVaccine
    _v_types = stride.AllVaccineTypes[1:]

    def get_available_vaccines(self, days):
        """Get the available vaccines for the given days.

        Args:
            days: The list of days to get the available vaccines for.

        Returns:
            available_vaccines, the available vaccines for those days, per vaccine type.
        """
        raise NotImplementedError


class ConstantVaccineSupply(VaccineSupply):
    """A constant vaccine supply per day

    Attributes:
        vaccine_type_counts:
            If an integer => constant for all vaccine types.
            If a dictionary => The number of available vaccines, per type.
    """
    def __init__(self, vaccine_type_counts=None):
        self._vaccine_counts = {}
        # None, chosen constants
        if vaccine_type_counts is None:
            self._vaccine_counts = {stride.VaccineType.mRNA: 20000, stride.VaccineType.adeno: 12000}
        # A constant was given
        elif isinstance(vaccine_type_counts, int):
            self._vaccine_counts = {v_type: vaccine_type_counts for v_type in self._v_types}
        # A dictionary was given
        elif isinstance(vaccine_type_counts, dict):
            self._vaccine_counts = vaccine_type_counts
        # Unknown
        else:
            raise ValueError(f"Unsupported vaccine_type_counts: {vaccine_type_counts}."
                             f"\n\tExpecting one of None, int or a dictionary of (stride.VaccineType: int) pairs")

    def get_available_vaccines(self, days):
        # Counts are meant to be read only, so providing a list with the same pointer should not be an issue
        one_day = self._vaccine_counts
        available_vaccines = [one_day] * len(days)
        return available_vaccines


class ObservedVaccineSupply(VaccineSupply):
    """Vaccine supply based on observations of the current vaccinations in Belgium via
        https://covid-vaccinatie.be/en

    Attributes:
        data_directory: (Optional) The data directory where the CSV files are stored to load in.
            If None, the data is retrieved via a get request.
    """
    # The urls to the data
    last_updated_url = "https://covid-vaccinatie.be/api/v1/last-updated.csv"
    doses_administered_url = "https://covid-vaccinatie.be/api/v1/administered.csv"
    doses_administered_per_vaccine_url = "https://covid-vaccinatie.be/api/v1/administered-by-vaccine-type.csv"
    doses_delivered_url = "https://covid-vaccinatie.be/api/v1/delivered.csv"

    def __init__(self, data_directory=None):
        self._last_update = None
        self._vaccine_counts = {}

    def get_last_updated(self):
        """Check when the last update happened"""
        stream = request.urlopen(self.last_updated_url)
        data = stream.read().decode('utf-8')
        last_updated = data.split("\"")[1]
        # TODO: only if outdated, retrieve new data
        if last_updated != self._last_update:
            # TODO: new data
            #
            self._last_update = last_updated

    def get_available_vaccines(self, days):
        available_vaccines = [self._vaccine_counts[day] for day in days]
        return available_vaccines

    def load_administered(self, starting_date="2021-01-05"):
        """Load data from the administered vaccines.

        Args:
            starting_date: The starting date from where to gather data.

        Returns:
            None.
        """
        stream = request.urlopen(self.doses_administered_per_vaccine_url)
        data = stream.read().decode('utf-8').split("\n")
        reader = csv.reader(data)

        available_vaccines = []
        current_date = None
        current_available = {v_type: 0 for v_type in self._v_types}

        # First row is a header ['date', 'region', 'type', 'first_dose', 'second_dose']
        skip_header = True
        for row in reader:
            # Skip the header
            if skip_header:
                skip_header = False
                continue
            # Skip empty rows
            elif len(row) == 0:
                continue
            # Skip all entries with a date smaller than the starting date
            elif row[0] < starting_date:
                continue

            # Check if the date still matches up with the current one
            if current_date != row[0] and current_date is not None:
                available_vaccines.append(current_available)
                current_available = {v_type: 0 for v_type in self._v_types}
            current_date = row[0]

            # Extract the vaccine type and counts if a vaccine name is provided
            if row[2] != "":
                v_type = self._get_vaccine_type(row[2])
                # Second dose considered as first dose => 1 extra person getting vaccinated that day
                count = int(row[3]) + int(row[4])
                current_available[v_type] += count

        self._vaccine_counts = available_vaccines

    def load_delivered(self, starting_date="2020-12-28"):
        """Load data from the delivered vaccines. # TODO implement

        Args:
            starting_date: The starting date from where to gather data.

        Returns:
            None.
        """
        stream = request.urlopen(self.doses_delivered_url)
        data = stream.read().decode('utf-8').split("\n")
        reader = csv.reader(data)

        available_vaccines = []
        current_date = None
        current_available = {v_type: 0 for v_type in self._v_types}

        # First row is a header ['date', 'amount', 'manufacturer']
        skip_header = True
        for row in reader:
            # Skip the header
            if skip_header:
                skip_header = False
                continue
            # Skip empty rows
            elif len(row) == 0:
                continue
            # Skip all entries with a date smaller than the starting date
            elif row[0] < starting_date:
                continue
            # TODO: remove
            print(row)

            # Check if the date still matches up with the current one
            if current_date != row[0] and current_date is not None:
                available_vaccines.append(current_available)
                current_available = {v_type: 0 for v_type in self._v_types}
            current_date = row[0]

            # Extract the vaccine type and counts if a vaccine name is provided
            if row[2] != "":
                v_type = self._get_vaccine_type(row[2])
                # Second dose considered as first dose => 1 extra person getting vaccinated that day
                count = int(row[3]) + int(row[4])
                current_available[v_type] += count

        self._vaccine_counts = available_vaccines

    @staticmethod
    def _get_vaccine_type(name):
        mRNA = ["Pfizer/BioNTech", "Moderna"]
        adeno = ["AstraZeneca/Oxford", "Johnson&Johnson"]
        # mRNA
        if name in mRNA:
            return stride.VaccineType.mRNA
        # adeno
        elif name in adeno:
            return stride.VaccineType.adeno
        # Unknown vaccine type
        else:
            raise RuntimeError(f"Unknown vaccine name: {name}")


if __name__ == '__main__':
    a = 0

    counts = {
        stride.VaccineType.mRNA: 20000,
        stride.VaccineType.adeno: 12000,
    }

    vs = ConstantVaccineSupply(vaccine_type_counts=counts)
    # vs = ObservedVaccineSupply()
    # vs.load_administered()

    cs = vs.get_available_vaccines(range(5, 25))
    print(cs)


