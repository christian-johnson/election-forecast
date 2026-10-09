"""Command line entry point.

`forecast fetch` downloads polls and races, `forecast run` fits the model, and `forecast update`
does both.
"""

import argparse
import logging
import os

# Must be set before JAX initializes so MCMC chains run in parallel on CPU cores.
os.environ.setdefault("XLA_FLAGS", f"--xla_force_host_platform_device_count={os.cpu_count()}")

import numpy as np
import orjson
import pandas as pd

from forecast import (
    breakdown,
    export,
    fetch,
    likelihood,
    matchups,
    model,
    polls,
    races,
    simulate,
)
from forecast.config import DATA_DIR, OFFICES, SITE_DATA_DIR

logger = logging.getLogger("forecast")

RACES_CSV = DATA_DIR / "races.csv"
CANDIDATES_CSV = DATA_DIR / "candidates.csv"
POLLS_CSV = DATA_DIR / "polls.csv"
FORECAST_JSON = SITE_DATA_DIR / "forecast.json"
STEPS_JSON = SITE_DATA_DIR / "steps.json"
TIMELINE_JSON = SITE_DATA_DIR / "timeline.json"
LIKELIHOOD_JSON = SITE_DATA_DIR / "likelihood.json"


def run_fetch() -> None:
    """Download polls and race tables and save them as tidy CSVs under data/."""
    race_tables, candidate_tables = [], []
    for office in OFFICES:
        r, c = races.parse_races(office, fetch.fetch_wikipedia(office))
        race_tables.append(r)
        candidate_tables.append(c)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    pd.concat(race_tables).to_csv(RACES_CSV, index=False)
    pd.concat(candidate_tables).to_csv(CANDIDATES_CSV, index=False)
    poll_rows = polls.flatten_polls(fetch.fetch_polls())
    poll_rows.to_csv(POLLS_CSV, index=False)
    logger.info("Saved %d races and %d poll answers", sum(map(len, race_tables)), len(poll_rows))


def _read_csv(path):
    return pd.read_csv(path, keep_default_na=False, na_values=[""])


def run_model(*, warmup: int, samples: int, chains: int, seed: int) -> dict:
    """Fit the model to the saved CSVs and write every JSON file the site reads.

    Args:
        warmup: NUTS warmup iterations per chain.
        samples: NUTS draws kept per chain.
        chains: Number of chains.
        seed: Random seed for both MCMC and the election simulation.

    Returns:
        The forecast document that was written.
    """
    race_table = _read_csv(RACES_CSV)
    candidates = _read_csv(CANDIDATES_CSV)
    poll_rows = _read_csv(POLLS_CSV)

    matched = matchups.match_answers(poll_rows, race_table, candidates)
    race_table, candidates = matchups.assign_sides(race_table, candidates, matched)
    race_obs = matchups.race_poll_observations(matched, candidates)
    generic_obs = matchups.generic_poll_observations(poll_rows)
    modeled = race_table[race_table["rule"] == "model"]

    inputs = model.build_inputs(modeled, race_obs, generic_obs)
    n_polls = {"race": len(race_obs), "generic": len(generic_obs)}
    logger.info("Fitting %d races with observations %s", len(modeled), n_polls)
    posterior = model.fit(inputs, warmup=warmup, samples=samples, chains=chains, seed=seed)

    rng = np.random.default_rng(seed)
    miss = simulate.national_miss(len(posterior["lean"]), rng)
    shares = simulate.simulate_shares(posterior, miss)
    document = export.build_forecast(race_table, race_obs, posterior, shares, n_polls)
    _write_json(FORECAST_JSON, document)
    _write_json(STEPS_JSON, breakdown.build_steps(modeled, posterior, miss, rng))
    _write_json(
        TIMELINE_JSON,
        breakdown.build_timeline(modeled, posterior, race_obs, generic_obs),
    )
    house = (race_table["office"] == "house").to_numpy()
    house_seats = (simulate.winners(race_table, shares)[:, house] == "D").sum(axis=1)
    _write_json(
        LIKELIHOOD_JSON,
        likelihood.build_likelihood(
            modeled, race_obs, generic_obs, posterior, likelihood.Elections(miss, house_seats)
        ),
    )
    return document


def _write_json(path, document) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(
        orjson.dumps(document, option=orjson.OPT_SERIALIZE_NUMPY | orjson.OPT_APPEND_NEWLINE)
    )
    logger.info("Wrote %s", path)


def main() -> None:
    """Parse arguments and run the requested stage."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["fetch", "run", "update"])
    parser.add_argument("--warmup", type=int, default=1000)
    parser.add_argument("--samples", type=int, default=1000)
    parser.add_argument("--chains", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    fit_args = {
        "warmup": args.warmup,
        "samples": args.samples,
        "chains": args.chains,
        "seed": args.seed,
    }
    if args.stage in {"fetch", "update"}:
        run_fetch()
    if args.stage in {"run", "update"}:
        run_model(**fit_args)
