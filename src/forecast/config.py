"""Static settings for the 2026 forecast: dates, seat math, priors, and file locations."""

from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

ELECTION_DATE = date(2026, 11, 3)
# Polls earlier than this are ignored; the national random walk starts here.
START_DATE = date(2025, 1, 6)

OFFICES = ("house", "senate", "governor")

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
SITE_DATA_DIR = ROOT / "site" / "data"


@dataclass(frozen=True)
class Chamber:
    """Seat arithmetic for one office.

    Attributes:
        total: Total seats in the chamber (or governorships).
        not_up: Seats not on the ballot this cycle, by party.
        majority: Seats each party needs for control, or None when control is not meaningful.
    """

    total: int
    not_up: dict[str, int] = field(default_factory=dict)
    majority: dict[str, int] | None = None


# Independents King and Sanders caucus with Democrats and are counted as D.
# The Vice President is a Republican, so Democrats need 51 Senate seats and Republicans 50.
CHAMBERS = {
    "house": Chamber(total=435, majority={"D": 218, "R": 218}),
    "senate": Chamber(total=100, not_up={"D": 34, "R": 31}, majority={"D": 51, "R": 50}),
    "governor": Chamber(total=50, not_up={"D": 6, "R": 8}),
}


@dataclass(frozen=True)
class Priors:
    """Prior scales for the model, on the logit scale of the Democratic two-party share.

    A logit shift of 0.02 is about one point of margin near a 50-50 race.
    """

    walk_weekly: float = 0.02
    house_effect_scale: float = 0.05
    poll_extra_noise: float = 0.05


@dataclass(frozen=True)
class History:
    """What past elections say, measured by scripts/calibrate_history.py, in margin points.

    Attributes:
        incumbency: Edge of an incumbent running again, from 2022 and 2024 House races.
        race_spread: Per office, how far 2022 and 2024 results landed from PVI + national vote +
            incumbency (an SD, set so that 80% of races fell within 1.28 of it).
        national_miss: Typical (root-mean-square) miss shared by every race's late polls in one
            election, over `poll_elections`.
        largest_national_miss: The largest such miss (+ means polls overstated Democrats).
        largest_national_miss_year: The election it happened in.
        state_miss: Typical further miss shared by every poll in one state.
        race_miss: Typical further miss shared by every poll of one race.
        pvi_elections: Elections behind incumbency and race_spread.
        poll_elections: Elections behind the polling misses (FiveThirtyEight's poll archive).
    """

    incumbency: float = 1.7
    race_spread: dict[str, float] = field(
        default_factory=lambda: {"house": 6.0, "senate": 6.4, "governor": 11.5}
    )
    national_miss: float = 3.0
    largest_national_miss: float = 6.5
    largest_national_miss_year: int = 2020
    state_miss: float = 2.7
    race_miss: float = 3.3
    pvi_elections: str = "2022 and 2024"
    poll_elections: str = "13 elections from 1998 to 2022"


PRIORS = Priors()
HISTORY = History()
# Logit units per point of margin near a 50-50 race, to turn HISTORY into model units.
POINT = 0.02

# Poll sample size assumed when a pollster does not report one.
DEFAULT_SAMPLE_SIZE = 600
# Cap on sample size so very large online panels do not swamp everything else.
MAX_SAMPLE_SIZE = 3000
