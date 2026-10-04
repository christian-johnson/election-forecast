"""Command line entry point.

`forecast fetch` downloads polls and races, `forecast run` fits the model, `forecast update` does
both, `forecast demographics` downloads the (rarely changing) Census group composition, and
`forecast demo` refits the made-up election shown on the About page.
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
    crosstabs,
    demo,
    demographics,
    export,
    fetch,
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
CROSSTABS_CSV = DATA_DIR / "crosstabs.csv"
COMPOSITION_CSV = DATA_DIR / "composition.csv"
FORECAST_JSON = SITE_DATA_DIR / "forecast.json"
DEMO_JSON = SITE_DATA_DIR / "demo.json"
STEPS_JSON = SITE_DATA_DIR / "steps.json"


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


def run_demographics() -> None:
    """Download Census ACS tables and save each state's and district's group mix."""
    tables = {table: fetch.fetch_acs(table) for table in demographics.ACS_TABLES}
    shares = demographics.composition(tables)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    shares.to_csv(COMPOSITION_CSV, index=False)
    logger.info("Saved group composition for %d places", shares["geo"].nunique())


def _read_csv(path):
    return pd.read_csv(path, keep_default_na=False, na_values=[""])


def run_model(*, warmup: int, samples: int, chains: int, seed: int) -> dict:
    """Fit the model to the saved CSVs and write the site's forecast and step-by-step JSON.

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
    crosstab_obs = (
        crosstabs.crosstab_observations(_read_csv(CROSSTABS_CSV))
        if CROSSTABS_CSV.exists()
        else None
    )

    matched = matchups.match_answers(poll_rows, race_table, candidates)
    race_table, candidates = matchups.assign_sides(race_table, candidates, matched)
    race_obs = matchups.race_poll_observations(matched, candidates)
    generic_obs = matchups.generic_poll_observations(poll_rows)
    modeled = race_table[race_table["rule"] == "model"]

    composition = None
    if COMPOSITION_CSV.exists():
        composition = demographics.race_composition(modeled, pd.read_csv(COMPOSITION_CSV))
    else:
        logger.warning("No %s; run `forecast demographics` first", COMPOSITION_CSV.name)

    inputs = model.build_inputs(modeled, race_obs, generic_obs, crosstab_obs, composition)
    n_polls = {
        "race": len(race_obs),
        "generic": len(generic_obs),
        "crosstab": 0 if crosstab_obs is None else len(crosstab_obs),
    }
    logger.info("Fitting %d races with observations %s", len(modeled), n_polls)
    posterior = model.fit(inputs, warmup=warmup, samples=samples, chains=chains, seed=seed)

    rng = np.random.default_rng(seed)
    errors = simulate.election_day_errors(
        modeled, len(posterior["lean"]), rng, composition=composition
    )
    shares = simulate.simulate_shares(posterior, errors)
    document = export.build_forecast(race_table, race_obs, posterior, shares, n_polls)
    _write_json(FORECAST_JSON, document)
    _write_json(STEPS_JSON, breakdown.build_steps(modeled, posterior, errors))
    return document


def _write_json(path, document) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(
        orjson.dumps(document, option=orjson.OPT_SERIALIZE_NUMPY | orjson.OPT_APPEND_NEWLINE)
    )
    logger.info("Wrote %s", path)


def run_demo(*, warmup: int, samples: int, chains: int, seed: int) -> dict:
    """Fit the About page's made-up election under each polling-error scenario.

    Args:
        warmup: NUTS warmup iterations per chain.
        samples: NUTS draws kept per chain.
        chains: Number of chains.
        seed: Random seed for the made-up truth, the polls, and the fit.

    Returns:
        The demo document that was written.
    """
    document = demo.build_demo(warmup=warmup, samples=samples, chains=chains, seed=seed)
    _write_json(DEMO_JSON, document)
    return document


def main() -> None:
    """Parse arguments and run the requested stage."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["fetch", "run", "update", "demographics", "demo"])
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
    if args.stage == "demographics":
        run_demographics()
    if args.stage == "demo":
        run_demo(**fit_args)
    if args.stage in {"fetch", "update"}:
        run_fetch()
    if args.stage in {"run", "update"}:
        run_model(**fit_args)
