"""Read the hand-kept national generic ballot crosstabs into model observations.

data/crosstabs.csv has one row per (poll, group) with columns: pollster, start_date, end_date,
population (lv/rv/v/a), partisan (D, R, or blank), dimension, group, dem, rep (percent), and n
(respondents in the group). Valid dimension/group pairs are listed in demographics.GROUPS.
"""

import pandas as pd

from forecast.demographics import GROUP_KEYS

COLUMNS = [
    "pollster",
    "start_date",
    "end_date",
    "population",
    "partisan",
    "dimension",
    "group",
    "dem",
    "rep",
    "n",
]
# Subgroups of under this many respondents are too noisy to be worth the weighting error.
_MIN_GROUP_N = 30


def crosstab_observations(table: pd.DataFrame) -> pd.DataFrame:
    """Validate crosstab rows and convert them to two-party observations.

    Args:
        table: Rows of data/crosstabs.csv.

    Returns:
        One row per usable (poll, group) with group (a GROUP_KEYS entry), pollster, population,
        partisan_sign, dem, rep, n_two_party and midpoint date.

    Raises:
        ValueError: If a column is missing or a row names an unknown group.
    """
    missing = set(COLUMNS) - set(table.columns)
    if missing:
        msg = f"crosstabs.csv is missing columns: {sorted(missing)}"
        raise ValueError(msg)
    obs = table.assign(group=table["dimension"].str.strip() + ":" + table["group"].str.strip())
    unknown = sorted(set(obs["group"]) - set(GROUP_KEYS))
    if unknown:
        msg = f"Unknown crosstab groups {unknown}; expected one of {list(GROUP_KEYS)}"
        raise ValueError(msg)
    obs = obs[(obs["n"] >= _MIN_GROUP_N) & (obs["dem"] > 0) & (obs["rep"] > 0)].copy()
    start = pd.to_datetime(obs["start_date"])
    obs["date"] = start + (pd.to_datetime(obs["end_date"]) - start) / 2
    obs["n_two_party"] = obs["n"] * (obs["dem"] + obs["rep"]) / 100
    obs["partisan_sign"] = obs["partisan"].map({"D": 1, "R": -1}).fillna(0).astype(int)
    obs["population"] = obs["population"].fillna("a")
    return obs.drop(columns="dimension").reset_index(drop=True)
