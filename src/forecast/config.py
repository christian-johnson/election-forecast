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
    """Prior scales for the model, all on the logit scale of the Democratic two-party share.

    A logit shift of 0.04 is about one point of two-party share near 50%.
    """

    walk_weekly: float = 0.02
    office_shift: float = 0.10
    incumbency_mean: float = 0.05
    incumbency_sd: float = 0.04
    race_scale: float = 0.15
    house_effect_scale: float = 0.05
    population_effect: float = 0.03
    partisan_sponsor_mean: float = 0.03
    partisan_sponsor_sd: float = 0.03
    poll_extra_noise: float = 0.05
    group_swing: float = 0.2


@dataclass(frozen=True)
class ElectionDayError:
    """Polling error that the polls cannot reveal, added when simulating the election.

    Same logit units as Priors. The national term moves every race together; the state term
    moves every race in one state together. The group term is a miss within one demographic
    group (e.g. non-college voters), which moves races according to their group mix.
    """

    national: float = 0.06
    state: float = 0.05
    race: float = 0.03
    group: float = 0.08


PRIORS = Priors()
ELECTION_DAY_ERROR = ElectionDayError()

# Poll sample size assumed when a pollster does not report one.
DEFAULT_SAMPLE_SIZE = 600
# Cap on sample size so very large online panels do not swamp everything else.
MAX_SAMPLE_SIZE = 3000
