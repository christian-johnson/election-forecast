"""Flatten raw VoteHub poll records into a tidy table with one row per answer."""

import pandas as pd

from forecast.config import START_DATE
from forecast.races import STATES

POLL_TYPES = {
    "generic-ballot": "generic",
    "us-senator": "senate",
    "governor": "governor",
    "us-representative": "house",
}
PARTISAN_SPONSOR = {"DEM": "D", "REP": "R"}
COLUMNS = [
    "poll_id",
    "office",
    "state",
    "district",
    "pollster",
    "start_date",
    "end_date",
    "sample_size",
    "population",
    "partisan",
    "choice",
    "pct",
]


def _place(office: str, poll: dict) -> tuple[str, int] | None:
    """Return (state, district) for a poll; generic ballot polls are ("US", 0)."""
    subject = poll["subject"] or ""
    if not subject.startswith("2026"):
        return None
    place = subject.removeprefix("2026").strip()
    if office == "generic":
        return "US", 0
    if office == "house":
        seat = poll["seat_name"] or place
        state, _, district = seat.partition("-")
        return (state, int(district)) if district.isdigit() else None
    # Primary polls carry a party suffix ("2026 Texas Democratic") and are not state names.
    return (STATES[place], 0) if place in STATES else None


def flatten_polls(raw: list[dict]) -> pd.DataFrame:
    """Keep 2026 general election and generic ballot polls and flatten them to answer rows.

    Args:
        raw: Poll records from fetch.fetch_polls.

    Returns:
        DataFrame with one row per (poll, answer) and the columns in COLUMNS.
    """
    rows = []
    for poll in raw:
        office = POLL_TYPES.get(poll["poll_type"])
        if office is None or poll["end_date"] < START_DATE.isoformat():
            continue
        place = _place(office, poll)
        if place is None:
            continue
        rows.extend(
            {
                "poll_id": poll["id"],
                "office": office,
                "state": place[0],
                "district": place[1],
                "pollster": poll["pollster"],
                "start_date": poll["start_date"],
                "end_date": poll["end_date"],
                "sample_size": poll["sample_size"],
                "population": poll["population"] or "a",
                "partisan": PARTISAN_SPONSOR.get(poll["partisan"]),
                "choice": answer["choice"],
                "pct": answer["pct"],
            }
            for answer in poll["answers"]
            if answer["pct"] is not None
        )
    return pd.DataFrame(rows, columns=COLUMNS)
