import csv
import xml.etree.ElementTree as ET
from datetime import date, timedelta

NA = "NA"
general = "general"
boolean = "boolean"
double = "double"


def _fill(writer, distancing, d_0, d_compliance, d_exit, d_end, no_distancing, cnt_reduction, cnt_reduction_exit,
          ages=(NA,), fill_holidays=()):
    range_before = [d for d in _date_range(d_0, d_compliance) if d not in fill_holidays]
    range_compliance = [d for d in _date_range(d_compliance, d_exit) if d not in fill_holidays]
    range_exit = [d for d in _date_range(d_exit, d_end, inclusive=True) if d not in fill_holidays]
    # Only add days that are not holidays
    for age in ages:
        # Days before reduction
        for d in range_before:
            writer.writerow([distancing, d, no_distancing, double, age])
        # Days of contact reduction
        for d in range_compliance:
            writer.writerow([distancing, d, cnt_reduction, double, age])
        # Exit strategy
        for d in range_exit:
            writer.writerow([distancing, d, cnt_reduction_exit, double, age])


def _fill_(writer, distancing, d_start, d_end, value, ages=(NA,)):
    for age in ages:
        for d in _date_range(d_start, d_end, inclusive=True):
            writer.writerow([distancing, d, value, double, age])


def _find(e_tree, match, second_match=None, fun=lambda text: text, default_value=None):
    value = e_tree.find(match)
    if value is not None:
        return fun(value.text)
    if second_match is not None:
        value = e_tree.find(second_match)
        if value is not None:
            return fun(value.text)
    return default_value


def _to_date(string):
    values = [int(d) for d in string.split("-")]
    return date(*values)


def _date_to_string(d):
    return str(d)


def _date_range(start, end, inclusive=False):
    """Create a range of dates from start to end."""
    if not isinstance(start, date):
        start = _to_date(start)
    if not isinstance(end, date):
        end = _to_date(end)
    diff = (end - start).days
    if inclusive:
        diff += 1
    dates = [start + timedelta(days=d) for d in range(diff)]
    return dates


# The public holidays
holidays = [
    # 2019
    "2019-01-01", "2019-04-22", "2019-05-01", "2019-05-30", "2019-06-10", "2019-07-21", "2019-08-15",
    "2019-11-01", "2019-11-11", "2019-12-25",
    # 2020
    "2020-01-01", "2020-04-13", "2020-05-01", "2020-05-21", "2020-06-01", "2020-07-21", "2020-08-15",
    "2020-11-01", "2020-11-11", "2020-12-25",
    # 2021
    "2021-01-01", "2021-04-05", "2021-05-01", "2021-05-13", "2021-06-24", "2021-07-21", "2021-08-15",
    "2021-11-01", "2021-11-11", "2021-12-25",
]

# The school holidays
# Primary + secondary
# noinspection PyTypeChecker
school_holidays = [
    # 2021
    *_date_range("2021-01-01", "2021-01-03", inclusive=True),
    *_date_range("2021-02-15", "2021-02-21", inclusive=True),
    *_date_range("2021-04-05", "2021-04-18", inclusive=True),
    *_date_range("2021-07-01", "2021-08-31", inclusive=True),
    *_date_range("2021-11-01", "2021-11-07", inclusive=True),
    *_date_range("2021-12-27", "2021-12-31", inclusive=True),
]
# College
# noinspection PyTypeChecker
college_holidays = [
    # 2021
    *_date_range("2021-01-01", "2021-01-03", inclusive=True),
    *_date_range("2021-02-15", "2021-02-21", inclusive=True),
    *_date_range("2021-04-05", "2021-04-18", inclusive=True),
    *_date_range("2021-07-01", "2021-09-19", inclusive=True),
    *_date_range("2021-11-01", "2021-11-07", inclusive=True),
    *_date_range("2021-12-27", "2021-12-31", inclusive=True),
]


def create_calendar(xml_config_file, calendar_file, with_holidays=False):
    """Create a csv calendar file for the given stride configuration file."""
    # Read the file
    tree = ET.parse(xml_config_file)
    root = tree.getroot()

    start_date = _find(root, "start_date", fun=_to_date, default_value=date(2021, 1, 1))
    compliance_delay_workplace = _find(root, "compliance_delay_workplace", fun=int, default_value=0)
    compliance_delay_other = _find(root, "compliance_delay_other", fun=int, default_value=0)
    compliance_delay_collectivity = _find(root, "compliance_delay_collectivity", fun=int, default_value=0)

    date_end = date(2021, 12, 31)
    # TODO: fixed dates?
    date_t0 = date(2021, 1, 1)
    date_exit_wp = date(2021, 12, 31)
    date_exit_other = date(2021, 12, 31)

    # Create the calendar file
    with open(calendar_file, mode='w') as file:
        writer = csv.writer(file)
        field_names = ["category", "date", "value", "type", "age"]
        # Add the header
        writer.writerow(field_names)

        # Categories
        schools_closed = "schools_closed"
        workplace_distancing = "workplace_distancing"
        community_distancing = "community_distancing"
        collectivity_distancing = "collectivity_distancing"

        # Write the holidays
        if with_holidays:
            # Public holidays
            for holiday in holidays:
                writer.writerow([general, holiday, 1, boolean, NA])
            # School holidays
            for holiday in school_holidays:
                for age in range(18):
                    writer.writerow([schools_closed, holiday, 1, boolean, age])
            # College holidays
            for holiday in college_holidays:
                for age in range(18, 26):
                    writer.writerow([general, holiday, 1, boolean, NA])

        # Contact reductions
        no_distancing = 0

        # Schools
        date_compliance_school = date_t0
        date_exit_school = date_end
        # Contact reduction values
        cnt_reduction_school = _find(root, "cnt_reduction_school", fun=float, default_value=0)
        cnt_reduction_school_exit = _find(root, "cnt_reduction_school_exit", fun=float, default_value=0)
        # Fill in the dates
        # _fill(writer, schools_closed, date_t0, date_compliance_school, date_exit_school, date_end,
        #       no_distancing, cnt_reduction_school, cnt_reduction_school_exit, ages=range(26))

        # Primary
        _fill(writer, schools_closed, date_t0, date_compliance_school, date_exit_school, date_end,
              no_distancing, cnt_reduction_school, cnt_reduction_school_exit, ages=range(12),
              fill_holidays=school_holidays)
        # Secondary
        cnt_reduction_school_secondary = _find(root, "cnt_reduction_school_secondary",
                                               second_match="cnt_reduction_school", fun=float, default_value=0)
        cnt_reduction_school_exit_secondary = _find(root, "cnt_reduction_school_exit_secondary",
                                                    second_match="cnt_reduction_school_exit", fun=float, default_value=0)
        _fill(writer, schools_closed, date_t0, date_compliance_school, date_exit_school, date_end,
              no_distancing, cnt_reduction_school_secondary, cnt_reduction_school_exit_secondary, ages=range(12, 18),
              fill_holidays=school_holidays)
        # Tertiary
        cnt_reduction_school_tertiary = _find(root, "cnt_reduction_school_tertiary",
                                              second_match="cnt_reduction_school", fun=float, default_value=0)
        cnt_reduction_school_exit_tertiary = _find(root, "cnt_reduction_school_exit_tertiary",
                                                   second_match="cnt_reduction_school_exit", fun=float, default_value=0)
        _fill(writer, schools_closed, date_t0, date_compliance_school, date_exit_school, date_end,
              no_distancing, cnt_reduction_school_tertiary, cnt_reduction_school_exit_tertiary, ages=range(18, 26),
              fill_holidays=college_holidays)

        # Workplace
        date_compliance_wp = date_t0 + timedelta(days=compliance_delay_workplace)
        # Contact reduction values
        cnt_reduction_workplace = _find(root, "cnt_reduction_workplace", fun=float, default_value=0)
        cnt_reduction_workplace_exit = _find(root, "cnt_reduction_workplace_exit", fun=float, default_value=0)
        # Fill in the dates
        _fill(writer, workplace_distancing, date_t0, date_compliance_wp, date_exit_wp, date_end,
              no_distancing, cnt_reduction_workplace, cnt_reduction_workplace_exit)

        # Community
        date_compliance_other = date_t0 + timedelta(days=compliance_delay_other)
        # Contact reduction values
        cnt_reduction_other = _find(root, "cnt_reduction_other", fun=float, default_value=0)
        cnt_reduction_other_exit = _find(root, "cnt_reduction_other_exit", fun=float, default_value=0)
        # Fill in the dates
        _fill(writer, community_distancing, date_t0, date_compliance_other, date_exit_other, date_end,
              no_distancing, cnt_reduction_other, cnt_reduction_other_exit)

        # Collectivity
        date_compliance_collectivity = date_t0 + timedelta(days=compliance_delay_collectivity)
        # Contact reduction values
        cnt_baseline_collectivity = _find(root, "cnt_baseline_collectivity", fun=float, default_value=0)
        cnt_reduction_collectivity = _find(root, "cnt_reduction_collectivity", fun=float, default_value=0)
        # Fill in the dates
        _fill(writer, collectivity_distancing, date_t0, start_date, date_compliance_collectivity, date_end,
              cnt_baseline_collectivity, cnt_baseline_collectivity, cnt_reduction_collectivity)

        # Contact Tracing
        contact_tracing = "contact_tracing"
        _fill_(writer, contact_tracing, start_date, date_end, value=1)

        # Household Clustering
        household_clustering = "household_clustering"
        _fill_(writer, household_clustering, start_date, date_end, value=0)

        # Imported Cases
        imported_cases = "imported_cases"
        _fill_(writer, imported_cases, start_date, date_end, value=0)

        # Universal Testing
        universal_testing = "universal_testing"
        _fill_(writer, universal_testing, start_date, date_end, value=0)


if __name__ == '__main__':
    create_calendar("./config0.xml", "./test_calendar.csv", with_holidays=True)

    # Agent-based HH config
    # create_calendar("./run_HH.xml", "./HH_calendar.csv")
