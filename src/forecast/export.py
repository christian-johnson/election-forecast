"""Summarize simulated elections into the JSON file the website reads."""

from datetime import UTC, datetime

import numpy as np
import pandas as pd

from forecast.config import CHAMBERS, ELECTION_DATE, OFFICES
from forecast.model import election_week, week_index
from forecast.races import STATES
from forecast.simulate import WIN_SHARE, winners

_STATE_NAMES = {abbr: name for name, abbr in STATES.items()}
_INTERVAL = (10, 90)
_DIST_PERCENTILES = (0.5, 99.5)


def _interval(values: np.ndarray, digits: int = 1) -> dict[str, float]:
    lo, hi = np.percentile(values, _INTERVAL)
    return {
        "mean": round(float(values.mean()), digits),
        "lo": round(float(lo), digits),
        "hi": round(float(hi), digits),
    }


def _margin(share: np.ndarray) -> np.ndarray:
    """D-side minus R-side margin in points."""
    return (2 * share - 1) * 100


def race_label(row: pd.Series) -> str:
    """Short display name: "OH-01" for House districts, the state name otherwise."""
    if row["office"] == "house":
        return f"{row['state']}-{row['district']:02d}"
    return _STATE_NAMES[row["state"]]


def _seat_dist(seats: np.ndarray) -> dict:
    """Share of draws (in thousandths) at each seat count over the middle 99% of draws."""
    lo, hi = np.percentile(seats, _DIST_PERCENTILES).round().astype(int)
    counts = np.bincount(seats[(seats >= lo) & (seats <= hi)] - lo, minlength=hi - lo + 1)
    return {"start": int(lo), "counts": np.round(1000 * counts / len(seats)).astype(int).tolist()}


def _seat_summary(office: str, wins: np.ndarray) -> dict:
    chamber = CHAMBERS[office]
    parties = sorted({str(p) for p in np.unique(wins)} | {"D", "R"})
    seats = {p: (wins == p).sum(axis=1) + chamber.not_up.get(p, 0) for p in parties}
    out = {
        "total": chamber.total,
        "not_up": chamber.not_up,
        "seats": {p: _interval(s.astype(float)) for p, s in seats.items()},
        "dem_seat_dist": _seat_dist(seats["D"]),
    }
    if chamber.majority:
        out["majority"] = chamber.majority
        out["p_control"] = {
            p: round(float((seats[p] >= need).mean()), 3) for p, need in chamber.majority.items()
        }
    return out


def _race_records(races: pd.DataFrame, shares: np.ndarray, n_polls: pd.Series) -> list[dict]:
    modeled = races.index[races["rule"] == "model"]
    column = {race_idx: i for i, race_idx in enumerate(modeled)}
    records = []
    for idx, row in races.iterrows():
        record = {
            "id": row["race_id"],
            "label": race_label(row),
            "state": row["state"],
            "district": int(row["district"]),
            "d_name": row["d_name"],
            "r_name": row["r_name"],
            "d_party": row["d_party"],
            "r_party": row["r_party"],
            "incumbent": row["incumbent"],
            "incumbent_party": row["incumbent_party"],
            "incumbent_running": pd.notna(row["incumbent_side"]),
            "pvi": row["pvi"],
            "n_polls": int(n_polls.get(row["race_id"], 0)),
            "rule": row["rule"],
        }
        if idx in column:
            share = shares[:, column[idx]]
            record["p_d"] = round(float((share > WIN_SHARE).mean()), 3)
            record["margin"] = _interval(_margin(share))
        else:
            record["p_d"] = 1.0 if row["rule"] == "D" else 0.0
            record["margin"] = None
        records.append(record)
    return records


def build_forecast(
    races: pd.DataFrame,
    race_obs: pd.DataFrame,
    posterior: dict[str, np.ndarray],
    shares: np.ndarray,
    n_polls: dict[str, int],
) -> dict:
    """Assemble the site's forecast document.

    Args:
        races: Full races table; modeled rows in the same order as the columns of shares.
        race_obs: Race poll observations used in the fit.
        posterior: Output of model.fit.
        shares: Output of simulate.simulate_shares.
        n_polls: Number of observations of each kind ("race", "generic").

    Returns:
        JSON-serializable dict with national, per-office seat, and per-race summaries.
    """
    nat = posterior["nat"]
    wins = winners(races, shares)
    today = pd.Series([pd.Timestamp.now(tz=UTC).tz_localize(None)])
    week_now = int(week_index(today)[0])
    polls_per_race = race_obs["race_id"].value_counts()
    offices = {}
    for office in OFFICES:
        mask = (races["office"] == office).to_numpy()
        subset = races[mask]
        modeled_cols = (races.loc[races["rule"] == "model", "office"] == office).to_numpy()
        offices[office] = {
            **_seat_summary(office, wins[:, mask]),
            "races": _race_records(subset, shares[:, modeled_cols], polls_per_race),
        }
    return {
        "updated": datetime.now(UTC).isoformat(timespec="minutes"),
        "election_date": ELECTION_DATE.isoformat(),
        "n_polls": n_polls,
        "generic_ballot": {
            "today": _interval(_margin(1 / (1 + np.exp(-nat[:, week_now])))),
            "election_day": _interval(_margin(1 / (1 + np.exp(-nat[:, election_week()])))),
        },
        "offices": offices,
    }
