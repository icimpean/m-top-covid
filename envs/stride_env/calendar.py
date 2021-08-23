import csv
import xml.etree.ElementTree as ET
from datetime import date, timedelta

from envs.stride_env.calendar_args import parser


NA = "NA"
general = "general"
boolean = "boolean"
double = "double"


def _fill(writer, category, d_start, d_end, cnt_reduction, value_type=double, ages=(NA,), fill_holidays=()):
    # Skip holidays
    date_range = [d for d in _date_range(d_start, d_end, inclusive=True) if d not in fill_holidays]
    for age in ages:
        for d in date_range:
            writer.writerow([category, d, cnt_reduction, value_type, age])


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

    # Update the xml file to point to the new calendar file
    c_file = root.find("holidays_file")
    c_file.text = calendar_file
    tree.write(xml_config_file, encoding='unicode')

    # Start and end date of the calendar
    start_date = _find(root, "start_date", fun=_to_date, default_value=date(2021, 1, 1))
    end_date = date(2021, 12, 31)  # TODO optional argument?

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
        contact_tracing = "contact_tracing"
        household_clustering = "household_clustering"
        imported_cases = "imported_cases"
        universal_testing = "universal_testing"

        # Write the holidays
        if with_holidays:
            # Public holidays
            for holiday in holidays:
                writer.writerow([general, holiday, 1, boolean, NA])
            # School holidays
            for holiday in school_holidays:
                for age in range(0, 18):
                    writer.writerow([schools_closed, holiday, 1.0, double, age])
            # College holidays
            for holiday in college_holidays:
                for age in range(18, 26):
                    writer.writerow([schools_closed, holiday, 1.0, double, age])

        # Schools
        cnt_reduction_school = _find(root, "cnt_reduction_school", fun=float, default_value=0)
        cnt_reduction_school_secondary = _find(root, "cnt_reduction_school_secondary", fun=float,
                                               default_value=cnt_reduction_school)
        cnt_reduction_school_tertiary = _find(root, "cnt_reduction_school_tertiary", fun=float,
                                              default_value=cnt_reduction_school)
        # Add holidays if requested
        holidays_k12 = school_holidays if with_holidays else []
        holidays_col = college_holidays if with_holidays else []
        # Primary
        _fill(writer, schools_closed, start_date, end_date, cnt_reduction_school, value_type=double,
              ages=range(0, 12), fill_holidays=holidays_k12)
        # Secondary
        _fill(writer, schools_closed, start_date, end_date, cnt_reduction_school_secondary, value_type=double,
              ages=range(12, 18), fill_holidays=holidays_k12)
        # Tertiary
        _fill(writer, schools_closed, start_date, end_date, cnt_reduction_school_tertiary, value_type=double,
              ages=range(18, 26), fill_holidays=holidays_col)

        # Workplace
        cnt_reduction_workplace = _find(root, "cnt_reduction_workplace", fun=float, default_value=0)
        _fill(writer, workplace_distancing, start_date, end_date, cnt_reduction_workplace, value_type=double)
        # Community
        cnt_reduction_other = _find(root, "cnt_reduction_other", fun=float, default_value=0)
        _fill(writer, community_distancing, start_date, end_date, cnt_reduction_other, value_type=double)
        # Collectivity
        cnt_reduction_collectivity = _find(root, "cnt_reduction_collectivity", fun=float, default_value=0)
        _fill(writer, collectivity_distancing, start_date, end_date, cnt_reduction_collectivity, value_type=double)

        # Contact Tracing
        cnt_tracing = 1  # TODO: from config
        _fill(writer, contact_tracing, start_date, end_date, cnt_tracing, value_type=boolean)
        # Household Clustering
        cnt_household_clustering = 0  # TODO: from config
        _fill(writer, household_clustering, start_date, end_date, cnt_household_clustering, value_type=boolean)
        # Imported Cases
        imported = 0  # TODO: from config
        _fill(writer, imported_cases, start_date, end_date, imported, value_type=boolean)
        # Universal Testing
        cnt_universal_testing = 0  # TODO: from config
        _fill(writer, universal_testing, start_date, end_date, cnt_universal_testing, value_type=boolean)


def create_contact_vectors(sim_days, cnt_workplace, cnt_community, cnt_collectivity):
    workplace_distancing = [cnt_workplace for day in range(sim_days)]
    community_distancing = [cnt_community for day in range(sim_days)]
    collectivity_distancing = [cnt_collectivity for day in range(sim_days)]
    return workplace_distancing, community_distancing, collectivity_distancing


def get_contact_reduction(xml_config_file, sim_days):
    """Create contact reduction vectors from the given configuration and number of days."""
    # Read the file
    tree = ET.parse(xml_config_file)
    root = tree.getroot()
    # Get the values
    cnt_reduction_workplace = _find(root, "cnt_reduction_workplace", fun=float, default_value=0)
    cnt_reduction_community = _find(root, "cnt_reduction_other", fun=float, default_value=0)
    cnt_reduction_collectivity = _find(root, "cnt_reduction_collectivity", fun=float, default_value=0)

    return create_contact_vectors(sim_days, cnt_reduction_workplace, cnt_reduction_community, cnt_reduction_collectivity)


if __name__ == '__main__':

    args = parser.parse_args()
    create_calendar(args.config, args.calendar, with_holidays=not args.no_holidays)

    # include_holidays = True
    # config_file = "config/config0_11M.xml"
    # calendar_name = "calendars/calendar_0.csv"  # if include_holidays else "calendars/calendar_no_holidays_0.csv"
    # create_calendar(config_file, calendar_name, with_holidays=include_holidays)
