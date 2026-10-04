"""Demographic groups, their 2024 vote, and each state's and district's group composition.

Crosstabs measure how each group has swung since 2024. A race moves with that swing in
proportion to how its group mix differs from the nation's, so the model needs (a) each group's
2024 vote as the reference point and (b) the group mix of every state and district.
"""

from dataclasses import dataclass
from io import StringIO

import numpy as np
import pandas as pd

# 2024 presidential two-party Democratic share nationally (Harris 48.3%, Trump 49.8%).
NATIONAL_2024_SHARE = 0.4924


@dataclass(frozen=True)
class Group:
    """One demographic group and how it voted in 2024.

    Attributes:
        dimension: The crosstab it belongs to ("race", "education", "age").
        name: Group name as written in data/crosstabs.csv.
        harris: Percent voting for Harris.
        trump: Percent voting for Trump.
        voter_share: Percent of 2024 voters in this group.
    """

    dimension: str
    name: str
    harris: float
    trump: float
    voter_share: float

    @property
    def key(self) -> str:
        """Identifier such as "race:white"."""
        return f"{self.dimension}:{self.name}"

    @property
    def offset(self) -> float:
        """Logit of the group's 2024 two-party D share, relative to the national result."""
        share = self.harris / (self.harris + self.trump)
        return float(
            np.log(share / (1 - share)) - np.log(NATIONAL_2024_SHARE / (1 - NATIONAL_2024_SHARE))
        )


# Pew Research Center 2024 validated voters ("Behind Trump's 2024 Victory", June 2025).
GROUPS = (
    Group("race", "white", 43, 55, 71),
    Group("race", "black", 83, 15, 10),
    Group("race", "hispanic", 51, 48, 10),
    Group("race", "asian", 57, 40, 4),
    Group("race", "other", 42, 55, 4),
    Group("education", "college", 57, 41, 41),
    Group("education", "noncollege", 42, 56, 59),
    Group("age", "18-29", 58, 39, 15),
    Group("age", "30-44", 47, 50, 22),
    Group("age", "45-64", 45, 53, 35),
    Group("age", "65+", 48, 51, 28),
)
GROUP_KEYS = tuple(g.key for g in GROUPS)
DIMENSIONS = tuple(dict.fromkeys(g.dimension for g in GROUPS))

STATE_FIPS = {
    "01": "AL", "02": "AK", "04": "AZ", "05": "AR", "06": "CA", "08": "CO", "09": "CT",
    "10": "DE", "12": "FL", "13": "GA", "15": "HI", "16": "ID", "17": "IL", "18": "IN",
    "19": "IA", "20": "KS", "21": "KY", "22": "LA", "23": "ME", "24": "MD", "25": "MA",
    "26": "MI", "27": "MN", "28": "MS", "29": "MO", "30": "MT", "31": "NE", "32": "NV",
    "33": "NH", "34": "NJ", "35": "NM", "36": "NY", "37": "NC", "38": "ND", "39": "OH",
    "40": "OK", "41": "OR", "42": "PA", "44": "RI", "45": "SC", "46": "SD", "47": "TN",
    "48": "TX", "49": "UT", "50": "VT", "51": "VA", "53": "WA", "54": "WV", "55": "WI",
    "56": "WY",
}  # fmt: skip

# ACS 1-year detailed tables, by GEO_ID prefix: nation, states, and 119th Congress districts.
ACS_TABLES = ("b05003", "b05003h", "b05003b", "b05003i", "b05003d", "b15003", "b01001")
_NATION, _STATE, _DISTRICT = "0100000US", "0400000US", "5001900US"


def _citizens_18_plus(table: pd.DataFrame, code: str) -> pd.Series:
    """Adult citizens (native-born plus naturalized, men and women) from a B05003 table."""
    return sum(table[f"{code}_E{line:03d}"] for line in (9, 11, 20, 22))


def _sum(table: pd.DataFrame, code: str, lines) -> pd.Series:
    return sum(table[f"{code}_E{line:03d}"] for line in lines)


def _geo(geo_id: str) -> str | None:
    """Map an ACS GEO_ID to "US", a state ("TX"), or a district ("TX-07"); None otherwise."""
    if geo_id == _NATION:
        return "US"
    if geo_id.startswith(_STATE):
        return STATE_FIPS.get(geo_id.removeprefix(_STATE))
    if geo_id.startswith(_DISTRICT):
        code = geo_id.removeprefix(_DISTRICT)
        state, district = STATE_FIPS.get(code[:2]), int(code[2:])
        # At-large seats are district 00 in the ACS and district 1 in the races table.
        return f"{state}-{max(district, 1):02d}" if state else None
    return None


def composition(tables: dict[str, str]) -> pd.DataFrame:
    """Group shares of the adult population for the nation, every state and every district.

    Race uses adult citizens (the closest ACS proxy for eligible voters); education uses adults
    25 and over; age uses all adults.

    Args:
        tables: Raw pipe-delimited ACS summary files keyed by table name (see ACS_TABLES).

    Returns:
        Long table with columns geo, group (a GROUP_KEYS entry) and share. Shares sum to 1
        within each geo and dimension.
    """
    t = {}
    for name, text in tables.items():
        frame = pd.read_csv(StringIO(text), sep="|", dtype={"GEO_ID": str})
        frame["geo"] = frame["GEO_ID"].map(_geo)
        t[name] = frame.dropna(subset=["geo"]).set_index("geo")
    adults = _citizens_18_plus(t["b05003"], "B05003")
    race = pd.DataFrame(
        {
            "white": _citizens_18_plus(t["b05003h"], "B05003H"),
            "black": _citizens_18_plus(t["b05003b"], "B05003B"),
            "hispanic": _citizens_18_plus(t["b05003i"], "B05003I"),
            "asian": _citizens_18_plus(t["b05003d"], "B05003D"),
        }
    ).fillna(0)  # The ACS suppresses small groups; their members fall into "other".
    race["other"] = (adults - race.sum(axis=1)).clip(lower=0)
    college = _sum(t["b15003"], "B15003", range(22, 26))
    education = pd.DataFrame(
        {"college": college, "noncollege": t["b15003"]["B15003_E001"] - college}
    )
    age_lines = {
        "18-29": [*range(7, 12), *range(31, 36)],
        "30-44": [*range(12, 15), *range(36, 39)],
        "45-64": [*range(15, 20), *range(39, 44)],
        "65+": [*range(20, 26), *range(44, 50)],
    }
    age = pd.DataFrame({g: _sum(t["b01001"], "B01001", lines) for g, lines in age_lines.items()})
    parts = []
    for dimension, counts in (("race", race), ("education", education), ("age", age)):
        shares = counts.div(counts.sum(axis=1), axis=0)
        shares.columns = [f"{dimension}:{c}" for c in shares.columns]
        parts.append(shares)
    wide = pd.concat(parts, axis=1)
    return (
        wide.rename_axis("geo")
        .reset_index()
        .melt(id_vars="geo", var_name="group", value_name="share")
        .round({"share": 4})
        .sort_values(["geo", "group"], ignore_index=True)
    )


def race_composition(races: pd.DataFrame, shares: pd.DataFrame) -> np.ndarray:
    """How each race's group mix differs from the nation's, averaged over dimensions.

    Districts redrawn for 2026 use their state's mix, since the ACS still reports the old map.
    Races with no composition data get zeros (no demographic effect).

    Args:
        races: Modeled races with state, district, office and redistricted columns.
        shares: Output of `composition`.

    Returns:
        Array (n_races, len(GROUPS)). Multiplying it by a vector of group swings gives each
        race's shift relative to the national swing.
    """
    wide = shares.pivot_table(index="geo", columns="group", values="share").reindex(
        columns=list(GROUP_KEYS)
    )
    redrawn = races.get("redistricted", pd.Series(False, index=races.index)).fillna(False)
    geo = np.where(
        (races["office"] == "house") & ~redrawn.astype(bool),
        races["state"] + "-" + races["district"].astype(int).map("{:02d}".format),
        races["state"],
    )
    deviation = wide.reindex(geo).to_numpy() - wide.loc["US"].to_numpy()
    # Each dimension gives its own estimate of the race's shift; average them, not add them.
    return np.nan_to_num(deviation) / len(DIMENSIONS)
