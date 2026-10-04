"""Show how each piece of evidence builds up a race's forecast, step by step.

Every modeled race's election-day logit is a sum of terms: the national mood, a shared polling
miss, the race's PVI baseline, an office and incumbency term, a demographic term, the race's own
effect (from its polls), and local polling misses. Each step adds one more term to the running
sum, so the last step is exactly the published forecast.
"""

import numpy as np
import pandas as pd

from forecast.model import election_week

STEPS = (
    ("polls", "National polls"),
    ("miss", "Shared polling miss"),
    ("lean", "Partisan lean"),
    ("office", "Office and incumbency"),
    ("groups", "Demographics"),
    ("race", "This race's polls"),
    ("local", "Local surprises"),
)
_BINS = 40
_RANGE_PERCENTILES = (0.5, 99.5)


def _terms(posterior: dict[str, np.ndarray], errors: dict[str, np.ndarray]) -> list[np.ndarray]:
    """Each step's logit term as an array (draws, n_races), in STEPS order."""
    nat = posterior["nat"][:, election_week()]
    parts = ("lean_office", "lean_groups", "lean_race")
    baseline = posterior["lean"] - sum(posterior[p] for p in parts)
    return [
        np.broadcast_to(nat[:, None], baseline.shape),
        errors["national"],
        baseline,
        *(posterior[p] for p in parts),
        errors["state"] + errors["group"] + errors["race"],
    ]


def _margin(logit: np.ndarray) -> np.ndarray:
    return (2 / (1 + np.exp(-logit)) - 1) * 100


def _interval(margins: np.ndarray) -> dict[str, float]:
    lo, hi = np.percentile(margins, [10, 90])
    return {
        "mean": round(float(margins.mean()), 1),
        "lo": round(float(lo), 1),
        "hi": round(float(hi), 1),
    }


def _summary(margins: np.ndarray, start: float, width: float) -> dict:
    # The outermost 1% of draws fall outside the bins and are left off the chart.
    counts, _ = np.histogram(margins, bins=_BINS, range=(start, start + width * _BINS))
    return {
        "p_d": round(float((margins > 0).mean()), 3),
        **_interval(margins),
        # Share of draws per bin, in thousandths, so the file stays small.
        "hist": np.round(1000 * counts / len(margins)).astype(int).tolist(),
    }


def build_steps(
    modeled_races: pd.DataFrame,
    posterior: dict[str, np.ndarray],
    errors: dict[str, np.ndarray],
) -> dict:
    """Build every modeled race's step-by-step forecast for the site.

    Args:
        modeled_races: Races passed to model.build_inputs, in the same order.
        posterior: Output of model.fit.
        errors: Output of simulate.election_day_errors, as used for the published forecast.

    Returns:
        {"steps": [{key, label}], "national": interval, "races": {race_id: {start, width,
        steps: [{p_d, mean, lo, hi, hist}]}}}, with margins in D-side points.
    """
    margins = _margin(np.cumsum(np.stack(_terms(posterior, errors)), axis=0))
    # One set of bins per race, wide enough for the middle 99% of every step.
    lo = np.percentile(margins, _RANGE_PERCENTILES[0], axis=1).min(axis=0)
    hi = np.percentile(margins, _RANGE_PERCENTILES[1], axis=1).max(axis=0)
    start = np.floor(lo)
    width = np.round(np.maximum((hi - start) / _BINS, 0.25), 3)
    races = {
        race_id: {
            "start": float(start[i]),
            "width": float(width[i]),
            "steps": [_summary(margins[k, :, i], start[i], width[i]) for k in range(len(STEPS))],
        }
        for i, race_id in enumerate(modeled_races["race_id"])
    }
    return {
        "steps": [{"key": key, "label": label} for key, label in STEPS],
        "national": _interval(_margin(posterior["nat"][:, election_week()])),
        "races": races,
    }
