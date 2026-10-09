"""Show how a race's forecast builds up from the national picture to the local one.

Four steps, each a set of simulated election-day margins: the national mood from the polls; plus
a national polling miss; then the race before its own polls (PVI and incumbency, with the spread
past races showed around them); then after its polls, which is exactly the published forecast.
The timeline behind each step's chart (the estimate week by week) is built here too.
"""

from dataclasses import asdict
from datetime import UTC

import numpy as np
import pandas as pd

from forecast.config import HISTORY, OFFICES, START_DATE
from forecast.model import RACE_SPREAD, RaceSet, election_week, week_index

STEPS = (
    ("national", "National polls"),
    ("national_miss", "National polling miss"),
    ("prior", "Before its polls"),
    ("polls", "After its polls"),
)
_BINS = 40
_RANGE_PERCENTILES = (0.5, 99.5)


def _steps_logit(
    races: RaceSet, posterior: dict[str, np.ndarray], miss: np.ndarray, rng: np.random.Generator
) -> np.ndarray:
    """Each step's election-day logit, shape (len(STEPS), draws, n_races)."""
    shape = posterior["lean"].shape
    nat = np.broadcast_to(posterior["nat"][:, election_week(), None], shape)
    national = nat + miss
    prior = races.expected + RACE_SPREAD[races.office] * rng.standard_normal(shape)
    return np.stack([nat, national, national + prior, national + posterior["lean"]])


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
    miss: np.ndarray,
    rng: np.random.Generator,
) -> dict:
    """Build every modeled race's step-by-step forecast for the site.

    Args:
        modeled_races: Races passed to model.build_inputs, in the same order.
        posterior: Output of model.fit.
        miss: Output of simulate.national_miss, as used for the published forecast.
        rng: Random generator for the before-polls draws.

    Returns:
        {"steps": [{key, label}], "national": interval, "history": config.HISTORY,
        "noise": {office: extra spread of its polls}, "races": {race_id: {expected, start, width,
        steps: [{p_d, mean, lo, hi, hist}]}}}, with margins in D-side points. "expected" is the
        race's margin on election day from the national mood, PVI and incumbency alone.
    """
    races = RaceSet.from_races(modeled_races)
    margins = _margin(_steps_logit(races, posterior, miss, rng))
    # One set of bins per race, wide enough for the middle 99% of every step.
    lo = np.percentile(margins, _RANGE_PERCENTILES[0], axis=1).min(axis=0)
    hi = np.percentile(margins, _RANGE_PERCENTILES[1], axis=1).max(axis=0)
    start = np.floor(lo)
    width = np.round(np.maximum((hi - start) / _BINS, 0.25), 3)
    expected = _margin(posterior["nat"][:, election_week()].mean() + races.expected)
    by_race = {
        race_id: {
            "expected": round(float(expected[i]), 1),
            "start": float(start[i]),
            "width": float(width[i]),
            "steps": [_summary(margins[k, :, i], start[i], width[i]) for k in range(len(STEPS))],
        }
        for i, race_id in enumerate(modeled_races["race_id"])
    }
    return {
        "steps": [{"key": key, "label": label} for key, label in STEPS],
        "national": _interval(_margin(posterior["nat"][:, election_week()])),
        "history": asdict(HISTORY),
        "noise": {
            office: round(float(_margin(posterior[f"{office}_noise"]).mean()), 1)
            for office in OFFICES
            if f"{office}_noise" in posterior
        },
        "races": by_race,
    }


def weekly(logit: np.ndarray, posterior: dict[str, np.ndarray]) -> dict[str, list[float]]:
    """Mean and SD of a weekly logit path; past this week they follow the random walk exactly.

    The walk has no drift, so the expected mood stays at this week's value and its variance
    grows by one step's variance per week. Averaging draws instead adds sampling wiggle.

    Args:
        logit: Draws of a path, shape (draws, n_weeks).
        posterior: Output of model.fit, for the walk's weekly step.

    Returns:
        {"mean": [...], "sd": [...]}, one value per week, on the logit scale.
    """
    now = min(
        int(week_index(pd.Series([pd.Timestamp.now(tz=UTC).tz_localize(None)]))[0]),
        logit.shape[1] - 1,
    )
    step_var = float((posterior["walk_sd"] ** 2).mean())
    mean = logit.mean(axis=0)
    sd = logit.std(axis=0)
    ahead = np.arange(len(mean)) - now
    future = ahead > 0
    mean[future] = mean[now]
    sd[future] = np.sqrt(sd[now] ** 2 + ahead[future] * step_var)
    return {"mean": np.round(mean, 3).tolist(), "sd": np.round(sd, 3).tolist()}


def optional(value):
    """None for a missing value, else the value."""
    return None if pd.isna(value) else value


def poll_records(obs: pd.DataFrame) -> list[dict]:
    """One record per poll, as the model saw it (two-party margin in D-side points).

    Args:
        obs: Poll observations from matchups (race or generic ballot).

    Returns:
        One dict per poll, with its pollster, dates, sample, result and link.
    """
    margin = 100 * (obs["dem"] - obs["rep"]) / (obs["dem"] + obs["rep"])
    return [
        {
            "pollster": row.pollster,
            "start": row.start_date,
            "end": row.end_date,
            "date": row.date.date().isoformat(),
            "n": None if pd.isna(row.sample_size) else int(row.sample_size),
            "n_two_party": round(float(row.n_two_party)),
            "population": row.population,
            "partisan": optional(row.partisan),
            "dem": row.dem,
            "rep": row.rep,
            "margin": round(float(m), 1),
            "url": optional(row.url),
        }
        for row, m in zip(obs.itertuples(index=False), margin, strict=True)
    ]


def build_timeline(
    modeled_races: pd.DataFrame,
    posterior: dict[str, np.ndarray],
    race_obs: pd.DataFrame,
    generic_obs: pd.DataFrame,
) -> dict:
    """Week-by-week estimates behind the step charts, on the logit scale.

    Each week's estimate is summarized by its mean and SD (the posterior is close to normal on
    this scale), so the site can draw any band from a few numbers per week. Weeks after this one
    hold the mean at this week's value and widen as the random walk allows.

    Args:
        modeled_races: Races passed to model.build_inputs, in the same order.
        posterior: Output of model.fit.
        race_obs: Race poll observations (polls of unmodeled races are skipped).
        generic_obs: Generic ballot poll observations.

    Returns:
        {"start": first week's date, "national": {mean, sd}, "prior": {race_id: [mean, sd]},
        "races": {race_id: {mean, sd}}, "polls": {"generic": [...], "races": {race_id: [...]}}}.
        "national" is the mood each week; "prior" is each race's lean before its polls,
        independent of the national mood; "races" is the mood plus the race's fitted lean each
        week, for polled races only; "polls" are the polls behind them (see poll_records).
    """
    races = RaceSet.from_races(modeled_races)
    spread = RACE_SPREAD[races.office]
    nat = posterior["nat"]
    race_obs = race_obs[race_obs["race_id"].isin(set(modeled_races["race_id"]))]
    polled = set(race_obs["race_id"])
    return {
        "start": START_DATE.isoformat(),
        "national": weekly(nat, posterior),
        "prior": {
            race_id: [round(float(races.expected[i]), 4), round(float(spread[i]), 4)]
            for i, race_id in enumerate(modeled_races["race_id"])
        },
        "races": {
            race_id: weekly(nat + posterior["lean"][:, i, None], posterior)
            for i, race_id in enumerate(modeled_races["race_id"])
            if race_id in polled
        },
        "polls": {
            "generic": poll_records(generic_obs),
            "races": {
                race_id: poll_records(polls)
                for race_id, polls in race_obs.groupby("race_id", sort=False)
            },
        },
    }
