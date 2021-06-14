# noinspection PyUnresolvedReferences
import pylibstride as stride

import csv
import datetime
import urllib.request as request


class VaccineSupply(object):
    """A vaccine supply to get the available vaccines per day from."""
    # All vaccine types, except noVaccine
    _v_types = stride.AllVaccineTypes[1:]

    def get_available_vaccines(self, days, pop_size=None):
        """Get the available vaccines for the given days.

        Args:
            days: The list of days to get the available vaccines for.
            pop_size: The population size to get vaccines for.

        Returns:
            available_vaccines, the available vaccines for those days, per vaccine type.
        """
        raise NotImplementedError


class ConstantVaccineSupply(VaccineSupply):
    """A constant vaccine supply per day

    Attributes:
        vaccine_type_counts: (Optional)
            If an integer => constant for all vaccine types.
            If a dictionary => The number of available vaccines, per type.
        population_size: (Optional) The population size the counts are meant for.
            Used to provide a vaccine count based on the current population.
    """
    def __init__(self, vaccine_type_counts=None, population_size=11000000):
        self._vaccine_counts = {}
        self.population_size = population_size
        # None, chosen constants
        if vaccine_type_counts is None:
            self._vaccine_counts = {stride.VaccineType.mRNA: 60000, stride.VaccineType.adeno: 40000}
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

    def get_available_vaccines(self, days, pop_size=None):
        # Counts are meant to be read only, so providing a list with the same pointer should not be an issue
        one_day = self._vaccine_counts.copy()
        # If a population size is given, calculate the vaccine counts
        if pop_size is not None:
            for v_type in one_day:
                one_day[v_type] = round(one_day[v_type] * pop_size / self.population_size)
        available_vaccines = [one_day] * len(days)
        return available_vaccines


class ObservedVaccineSupply(VaccineSupply):
    """Vaccine supply based on observations of the current vaccinations in Belgium via
        https://covid-vaccinatie.be/en

    Note: For use_administered=False, we assume that the date for the next vaccine delivery of a certain type
     is known to divide the already supplied vaccines over the intermediate days.

    Attributes:
        population_size: (Optional) The population size the counts are meant for.
            Used to provide a vaccine count based on the current population.
        data_directory: (Optional) The data directory where the CSV files are stored to load in.
            If None, the data is retrieved via a get request.
    """
    # The urls to the data
    last_updated_url = "https://covid-vaccinatie.be/api/v1/last-updated.csv"
    doses_administered_url = "https://covid-vaccinatie.be/api/v1/administered.csv"
    doses_administered_per_vaccine_url = "https://covid-vaccinatie.be/api/v1/administered-by-vaccine-type.csv"
    doses_delivered_url = "https://covid-vaccinatie.be/api/v1/delivered.csv"

    def __init__(self, use_administered, starting_date="", population_size=11000000,
                 data_directory=None):
        self.use_administered = use_administered
        self.starting_date = starting_date
        self.population_size = population_size
        #
        self._last_update = None
        self._vaccine_counts = []
        if use_administered:
            self.load_administered(starting_date)
        else:
            self.load_delivered(starting_date)

    def get_last_updated(self):  # TODO: only if outdated, retrieve new data
        """Check when the last update happened"""
        stream = request.urlopen(self.last_updated_url)
        data = stream.read().decode('utf-8')
        last_updated = data.split("\"")[1]
        if last_updated != self._last_update:
            self._last_update = last_updated

    def get_available_vaccines(self, days, pop_size=None):
        # Avoid indexing errors for missing days: use the last day as counts
        get_index = lambda d: d if d < len(self._vaccine_counts) else -1
        # If a population size is given, calculate the vaccine counts
        if pop_size is None:
            available_vaccines = [self._vaccine_counts[get_index(day)] for day in days]
        else:
            available_vaccines = [self._vaccine_counts[get_index(day)].copy() for day in days]
            for one_day in available_vaccines:
                for v_type in one_day:
                    one_day[v_type] = round(one_day[v_type] * pop_size / self.population_size)
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
        """Load data from the delivered vaccines.

        Args:
            starting_date: The starting date from where to gather data.

        Returns:
            None.
        """
        stream = request.urlopen(self.doses_delivered_url)
        data = stream.read().decode('utf-8').split("\n")
        reader = csv.reader(data)

        first_dates = {v_type: None for v_type in self._v_types}
        last_dates = {v_type: None for v_type in self._v_types}
        counts_per_delivery = {v_type: [] for v_type in self._v_types}

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

            # Extract the vaccine type
            v_type = self._get_vaccine_type(row[2])
            date = datetime.date.fromisoformat(row[0])
            count = int(row[1])

            # Keep track of the first and last day, per vaccine type
            if first_dates[v_type] is None:
                first_dates[v_type] = date
            if last_dates[v_type] is None or last_dates[v_type] < date:
                last_dates[v_type] = date

            # Update counts for the vaccine type
            if len(counts_per_delivery[v_type]) != 0 and counts_per_delivery[v_type][-1][0] == date:
                counts_per_delivery[v_type][-1][1] += count
            else:
                counts_per_delivery[v_type].append([date, count])

        # Supplies must start from first date: not supplied vaccines have count 0
        supplies = {v_type: [] for v_type in self._v_types}
        first_date = min(first_dates.values())
        last_date = max(last_dates.values())
        for v_type, date_counts in counts_per_delivery.items():
            # Supply counts must start from the first date, even if it doesn't match up
            first = date_counts[0][0]
            # Supply doesn't start on the first available date: empty
            if first_date < first:
                diff = (first - first_date).days
                for _ in range(diff):
                    supplies[v_type].append(0)

            # The current counts last until the next date
            for i in range(len(date_counts)-1):
                current_counts = date_counts[i]
                next_counts = date_counts[i + 1]
                diff = (next_counts[0] - current_counts[0]).days
                counts = self.divide(current_counts[1], diff)
                supplies[v_type].extend(counts)

            # The last count lasts for a week more than the last delivery
            last_diff = (last_date - date_counts[-1][0]).days + 7
            counts = self.divide(date_counts[-1][1], last_diff)
            supplies[v_type].extend(counts)

        # Store the counts grouped per day
        self._vaccine_counts = []
        for v_counts in zip(*supplies.values()):
            available_vaccines = {v_type: v_count for v_type, v_count in zip(self._v_types, v_counts)}
            self._vaccine_counts.append(available_vaccines)

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

    @staticmethod
    def divide(count, divisor):
        d, m = divmod(count, divisor)
        new_counts = [d for _ in range(divisor)]
        for i in range(m):
            new_counts[i] += 1
        return new_counts
