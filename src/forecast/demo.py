"""A made-up election with a known outcome, polled with known errors, for the About page.

One state with five House districts and a Senate seat. We pick the true national environment,
each race's true result, and each pollster's true bias, draw noisy polls from that truth (all
of them shifted by an industry-wide error the model cannot see), then fit the real model to
those polls and compare its forecast with the truth.
"""

from dataclasses import dataclass
from datetime import timedelta

import numpy as np
import pandas as pd

from forecast import model, simulate
from forecast.config import ELECTION_DATE, PRIORS, START_DATE

# Logit shift that moves a 50-50 race by about one point of margin.
_POINT = 0.02
_POLL_WEEKS = 30
_POLL_DAYS = 7 * _POLL_WEEKS


@dataclass(frozen=True)
class DemoRace:
    """One race and its true candidate-quality effect (logit, + helps the Democrat)."""

    race_id: str
    label: str
    pvi: float
    incumbent_side: str | None
    quality: float
    n_polls: int


RACES = (
    DemoRace("H-AA-01", "District 1", 12, "D", 0.00, 0),
    DemoRace("H-AA-02", "District 2", 4, None, -0.10, 3),
    DemoRace("H-AA-03", "District 3", 0, None, 0.12, 6),
    DemoRace("H-AA-04", "District 4", -5, "R", 0.06, 2),
    DemoRace("H-AA-05", "District 5", -14, None, 0.00, 0),
    DemoRace("S-AA", "Senate", -2, "R", 0.08, 10),
)
# True house effects in margin points (+ overstates Democrats). They average zero, since a
# shared offset is indistinguishable from the industry-wide miss below.
POLLSTERS = {"Pollster A": 2.5, "Pollster B": -2.5, "Pollster C": -0.5, "Pollster D": 0.5}
# Industry-wide polling miss in margin points, shared by every poll (+ overstates Democrats).
SCENARIOS = {"none": 0.0, "dem": 3.0, "rep": -3.0}
NATIONAL_MARGIN = 4.0
N_GENERIC = 50


@dataclass(frozen=True)
class Truth:
    """The made-up world: national environment by week and each race's lean (logit)."""

    nat: np.ndarray
    lean: np.ndarray


def _truth(rng: np.random.Generator) -> Truth:
    walk = np.cumsum(rng.normal(0, PRIORS.walk_weekly, model.election_week() + 1))
    nat = walk - walk[-1] + NATIONAL_MARGIN * _POINT
    share = 0.5 + np.array([r.pvi for r in RACES]) / 100
    incumbency = np.array([{"D": 1, "R": -1}.get(r.incumbent_side, 0) for r in RACES])
    lean = (
        np.log(share / (1 - share))
        + PRIORS.incumbency_mean * incumbency
        + np.array([r.quality for r in RACES])
    )
    return Truth(nat=nat, lean=lean)


def _polls(truth: Truth, bias: float, rng: np.random.Generator) -> pd.DataFrame:
    """Draw generic ballot and race polls from the truth, with every bias added in."""
    last_day = ELECTION_DATE - timedelta(days=7)
    plan = [(None, N_GENERIC, 1000)] + [(i, r.n_polls, 600) for i, r in enumerate(RACES)]
    rows = []
    for race, count, n in plan:
        for _ in range(count):
            date = last_day - timedelta(days=int(rng.integers(0, _POLL_DAYS)))
            pollster = rng.choice(list(POLLSTERS))
            week = (date - START_DATE).days // 7
            logit = (
                truth.nat[week]
                + (0.0 if race is None else truth.lean[race])
                + (POLLSTERS[pollster] + bias) * _POINT
                + rng.normal(0, PRIORS.poll_extra_noise / 2)
            )
            share = rng.binomial(n, 1 / (1 + np.exp(-logit))) / n
            rows.append(
                {
                    "race_id": None if race is None else RACES[race].race_id,
                    "pollster": str(pollster),
                    "date": pd.Timestamp(date),
                    "population": "lv",
                    "partisan_sign": 0,
                    "dem": 100 * share,
                    "rep": 100 * (1 - share),
                    "n_two_party": n,
                }
            )
    return pd.DataFrame(rows).sort_values("date", ignore_index=True)


def _margin(logit: np.ndarray) -> np.ndarray:
    return (2 / (1 + np.exp(-logit)) - 1) * 100


def _interval(values: np.ndarray) -> dict[str, float]:
    lo, hi = np.percentile(values, [10, 90])
    return {
        "mean": round(float(values.mean()), 1),
        "lo": round(float(lo), 1),
        "hi": round(float(hi), 1),
    }


def _races_table() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "race_id": [r.race_id for r in RACES],
            "office": ["senate" if r.race_id.startswith("S") else "house" for r in RACES],
            "state": "AA",
            "pvi": [r.pvi for r in RACES],
            "incumbent_side": [r.incumbent_side for r in RACES],
        }
    )


def run_scenario(bias: float, *, seed: int = 0, **fit_args) -> dict:
    """Simulate polls with an industry-wide bias, fit the model, and compare with the truth.

    Args:
        bias: Polling miss shared by every poll, in margin points (+ overstates Democrats).
        seed: Random seed; the made-up truth is the same for every bias.
        **fit_args: Passed to model.fit (warmup, samples, chains).

    Returns:
        JSON-serializable dict with the national trend, each race, and each pollster, all as
        margins in points, holding both the truth and the model's estimate.
    """
    truth = _truth(np.random.default_rng(seed))
    rng = np.random.default_rng(seed + 1)
    polls = _polls(truth, bias, rng)
    generic = polls[polls["race_id"].isna()]
    race_polls = polls[polls["race_id"].notna()]
    races = _races_table()

    inputs = model.build_inputs(races, race_polls, generic)
    posterior = model.fit(inputs, seed=seed, **fit_args)
    errors = simulate.election_day_errors(races, len(posterior["lean"]), rng)
    shares = simulate.simulate_shares(posterior, errors)

    first_week = model.election_week() - _POLL_WEEKS - 1
    weeks = range(first_week, model.election_week() + 1)
    nat_draws = _margin(posterior["nat"])
    true_margin = _margin(truth.nat[-1] + truth.lean)
    sim_margin = (2 * shares - 1) * 100
    poll_margin = race_polls.assign(margin=race_polls["dem"] - race_polls["rep"])
    pollster_draws = posterior["house_sd"][:, None] * posterior["house_z"]
    return {
        "bias": bias,
        "national": [
            {
                "date": (START_DATE + timedelta(weeks=w)).isoformat(),
                "truth": round(float(_margin(truth.nat[w])), 2),
                **_interval(nat_draws[:, w]),
            }
            for w in weeks
        ],
        "generic_polls": [
            {"date": d.date().isoformat(), "margin": round(m, 1), "pollster": p}
            for d, m, p in zip(
                generic["date"], generic["dem"] - generic["rep"], generic["pollster"], strict=True
            )
        ],
        "races": [
            {
                "id": r.race_id,
                "label": r.label,
                "pvi": r.pvi,
                "incumbent_side": r.incumbent_side,
                "n_polls": r.n_polls,
                "poll_average": (
                    round(
                        float(
                            poll_margin.loc[poll_margin["race_id"] == r.race_id, "margin"].mean()
                        ),
                        1,
                    )
                    if r.n_polls
                    else None
                ),
                "truth": round(float(true_margin[i]), 1),
                "p_d": round(float((sim_margin[:, i] > 0).mean()), 3),
                **_interval(sim_margin[:, i]),
            }
            for i, r in enumerate(RACES)
        ],
        "pollsters": [
            {"name": name, "truth": true_effect, **_interval(_margin(pollster_draws[:, j]))}
            for name, true_effect in POLLSTERS.items()
            for j in [inputs.pollsters.index(name)]
        ],
        "house_seats": {
            "truth": int((true_margin[:5] > 0).sum()),
            **_interval((sim_margin[:, :5] > 0).sum(axis=1).astype(float)),
        },
    }


def build_demo(*, seed: int = 0, **fit_args) -> dict:
    """Run every scenario in SCENARIOS.

    Args:
        seed: Random seed shared by all scenarios, so only the polling miss differs.
        **fit_args: Passed to model.fit.

    Returns:
        {"scenarios": {name: run_scenario output}, "national_margin": true election-day margin}.
    """
    return {
        "national_margin": NATIONAL_MARGIN,
        "scenarios": {
            name: run_scenario(bias, seed=seed, **fit_args) for name, bias in SCENARIOS.items()
        },
    }
