"""Data behind the About page's walk through the model, from the national polls to the full fit.

The page shows the generic ballot polls and the national mood fitted to them, the extra spread
and house effects the fit learned, a handful of well-polled races with their polls, and a sample
of the fit's draws, so it can replay how every draw moves all of them, and the House, at once.
"""

from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from itertools import chain, zip_longest

import numpy as np
import pandas as pd

from forecast.breakdown import optional, poll_records, weekly
from forecast.config import CHAMBERS, ELECTION_DATE, HISTORY, OFFICES, START_DATE
from forecast.export import race_label
from forecast.model import RACE_SPREAD, RaceSet, pollster_names

# Races shown with their polls (the most polled of each office) and draws replayed in the final fit.
SHOWN_PER_OFFICE = 4
SHOWN_DRAWS = 200
# Draws are stored as logits in thousandths, so the file stays small.
_LOGIT_SCALE = 1000


def _margin(logit: np.ndarray) -> np.ndarray:
    return (2 / (1 + np.exp(-logit)) - 1) * 100


def _interval(margins: np.ndarray) -> dict[str, float]:
    lo, hi = np.percentile(margins, [10, 90])
    return {
        "mean": round(float(margins.mean()), 1),
        "lo": round(float(lo), 1),
        "hi": round(float(hi), 1),
    }


def _thousandths(logit: np.ndarray) -> list:
    return np.round(logit * _LOGIT_SCALE).astype(int).tolist()


@dataclass(frozen=True)
class Elections:
    """The simulated elections behind the published forecast, one per posterior draw.

    Attributes:
        miss: National polling miss of each draw, from simulate.national_miss (logit).
        house_d: Democratic House seats in each draw.
    """

    miss: np.ndarray
    house_d: np.ndarray


def _shown_races(race_obs: pd.DataFrame, office: dict[str, str]) -> list[str]:
    """The most-polled races of each office, alternating offices, with the most polled first."""
    counts = race_obs["race_id"].value_counts()
    offices = counts.index.map(office)
    ranked = [counts.index[offices == o][:SHOWN_PER_OFFICE] for o in dict.fromkeys(offices)]
    return [race_id for race_id in chain(*zip_longest(*ranked)) if race_id is not None]


def build_likelihood(
    modeled_races: pd.DataFrame,
    race_obs: pd.DataFrame,
    generic_obs: pd.DataFrame,
    posterior: dict[str, np.ndarray],
    elections: Elections,
) -> dict:
    """Assemble the About page's walkthrough of the model.

    Args:
        modeled_races: Races passed to model.build_inputs, in the same order.
        race_obs: Race poll observations (polls of unmodeled races are skipped).
        generic_obs: Generic ballot poll observations.
        posterior: Output of model.fit.
        elections: The simulated elections behind the published forecast.

    Returns:
        JSON-serializable dict. "national" holds the generic ballot polls and the weekly mood
        ({mean, sd} on the logit scale); "races" the most-polled races, each with its polls,
        its lean before its polls ("prior": [mean, sd], logit) and its weekly estimate; "draws"
        a sample of the fit's draws in chain order ("nat" by week, "lean" by shown race and the
        national polling "miss", as logits in thousandths, and the House seats each draw gives
        Democrats). Also "walk" (the
        mood's weekly step), "noise" (each poll type's extra spread) and "pollsters" (house
        effects), in margin points; "history", the past-election numbers in config.HISTORY; and
        the number of modeled races, race polls and draws in the full fit.
    """
    column = {race_id: i for i, race_id in enumerate(modeled_races["race_id"])}
    race_obs = race_obs[race_obs["race_id"].isin(column)]
    shown = _shown_races(
        race_obs, dict(zip(modeled_races["race_id"], modeled_races["office"], strict=True))
    )
    columns = [column[race_id] for race_id in shown]
    race_set = RaceSet.from_races(modeled_races)
    rows = modeled_races.set_index("race_id")
    nat, lean = posterior["nat"], posterior["lean"]
    races = [
        {
            "id": race_id,
            "label": race_label(rows.loc[race_id]),
            "office": rows.loc[race_id, "office"],
            "pvi": rows.loc[race_id, "pvi"],
            "incumbent_side": optional(rows.loc[race_id, "incumbent_side"]),
            "prior": [
                round(float(race_set.expected[i]), 4),
                round(float(RACE_SPREAD[race_set.office[i]]), 4),
            ],
            "weekly": weekly(nat + lean[:, i, None], posterior),
            "polls": poll_records(race_obs[race_obs["race_id"] == race_id]),
        }
        for race_id, i in zip(shown, columns, strict=True)
    ]
    picks = np.linspace(0, len(lean) - 1, SHOWN_DRAWS).round().astype(int)
    house = _margin(posterior["house_sd"][:, None] * posterior["house_z"])
    poll_counts = pd.concat([generic_obs["pollster"], race_obs["pollster"]]).value_counts()
    majority = CHAMBERS["house"].majority["D"]
    return {
        "as_of": datetime.now(UTC).date().isoformat(),
        "start": START_DATE.isoformat(),
        "election_date": ELECTION_DATE.isoformat(),
        "history": asdict(HISTORY),
        "n_races": len(modeled_races),
        "n_race_polls": len(race_obs),
        "n_draws": len(lean),
        "walk": round(float(_margin(posterior["walk_sd"]).mean()), 2),
        "noise": {
            name: round(float(_margin(posterior[f"{name}_noise"]).mean()), 1)
            for name in ("generic", *OFFICES)
            if f"{name}_noise" in posterior
        },
        "pollsters": [
            {"name": name, "n_polls": int(poll_counts.get(name, 0)), **_interval(house[:, j])}
            for j, name in enumerate(pollster_names(race_obs, generic_obs))
        ],
        "national": {"polls": poll_records(generic_obs), "weekly": weekly(nat, posterior)},
        "races": races,
        "draws": {
            "nat": _thousandths(nat[picks]),
            "lean": _thousandths(lean[picks][:, columns]),
            "miss": _thousandths(elections.miss[picks, 0]),
            "house_d": elections.house_d[picks].astype(int).tolist(),
        },
        "house": {
            "total": CHAMBERS["house"].total,
            "majority": majority,
            "p_d": round(float((elections.house_d >= majority).mean()), 3),
        },
    }
