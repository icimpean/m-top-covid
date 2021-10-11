# noinspection PyUnresolvedReferences
import pylibstride as stride

import csv
import datetime
import random
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

    def __init__(self, starting_date="2021-01-01", days=181, population_size=11000000,
                 data_directory=None, seed=0):
        self.starting_date = starting_date
        self.population_size = population_size
        random.seed(seed)
        #
        self._last_update = None
        self._vaccine_counts = []
        self.load_delivered(starting_date, days)

        # TODO: remove
        self.counts = self._vaccine_counts

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

    def load_delivered(self, starting_date, days):
        """Load data from the delivered vaccines.

        Args:
            starting_date: The starting date of the simulations.
            days: The number of days the simulation runs for.

        Returns:
            None.
        """
        start_date = Date.fromisoformat(starting_date)
        end_date = start_date + datetime.timedelta(days=days)
        counts_per_week = {}

        with open("./resources/weekly_delivered.csv", mode="r") as file:
            reader = csv.reader(file)
            # First row is a header
            # ['Week', 'Pfizer/BioNTech', 'Moderna', 'AstraZeneca/Oxford', 'Johnson&Johnson' ,'Total']
            skip_header = True
            for row in reader:
                # Skip the header
                if skip_header:
                    skip_header = False
                    continue
                # Skip empty rows
                elif len(row) == 0:
                    continue

                # Get the week from the row
                week_date = Date.fromisoformat(row[0])
                # Skip all entries longer than a week before the starting date
                if (start_date - week_date).days >= 7:
                    # print(f"Skipping {week_date}, is more than a week before {start_date}")
                    continue
                # Skip all deliveries after simulation ends
                if week_date > end_date:
                    break

                # Extract vaccine counts
                count_mRNA = sum([int(c) for c in row[1:3]])
                count_adeno = sum([int(c) for c in row[3:5]])
                # Divide counts over the week
                count_mRNA = self._divide(count_mRNA)
                count_adeno = self._divide(count_adeno)

                counts_per_week[week_date] = {
                    stride.VaccineType.mRNA: count_mRNA,
                    stride.VaccineType.adeno: count_adeno,
                }

        self._vaccine_counts = []
        for week_date, counts in counts_per_week.items():
            # print("Weekdate", week_date)
            for weekday in range(7):
                date = week_date + datetime.timedelta(days=weekday)
                # Skip dates before/after the start/end dates
                if start_date > date:
                    # print("day", date, "before", start_date)
                    continue
                elif date > end_date:
                    # print("day", date, "later than end", end_date)
                    break
                week_counts = counts_per_week[week_date]
                v_counts = {v_type: c[weekday] for v_type, c in week_counts.items()}
                self._vaccine_counts.append(v_counts)

    @staticmethod
    def _divide(count):
        q, r = divmod(count, 7)
        new_counts = [q] * 7
        for i in range(r):
            new_counts[i] += 1
        random.shuffle(new_counts)
        return new_counts

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


class Date(datetime.date):
    def isoweek(self):
        return self.isocalendar()[1]


if __name__ == '__main__':
    vs = ObservedVaccineSupply(starting_date="2021-01-01", days=120, seed=0)

    for day, c in enumerate(vs.counts):
        print(f"day {day}: {c}")
