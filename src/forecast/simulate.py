"""Turn posterior draws into simulated elections, including races decided by rule."""

import numpy as np
import pandas as pd

from forecast.config import ELECTION_DAY_ERROR, ElectionDayError
from forecast.model import election_week

# A side wins with more than half of the two-party vote.
WIN_SHARE = 0.5


def election_day_errors(
    modeled_races: pd.DataFrame,
    n_draws: int,
    rng: np.random.Generator,
    error: ElectionDayError = ELECTION_DAY_ERROR,
    composition: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    """Draw the polling error that the polls cannot reveal, as logit shifts.

    Args:
        modeled_races: The races passed to model.build_inputs, in the same order.
        n_draws: Number of simulated elections.
        rng: Random generator.
        error: Scales of the election-day error terms.
        composition: Group mix of each race (see demographics.race_composition), if known.

    Returns:
        Arrays (draws, n_modeled_races) keyed by source: "national" (one shift shared by every
        race), "state" (shared within each state), "group" (a miss per demographic group,
        applied by each race's group mix) and "race" (independent per race).
    """
    n_races = len(modeled_races)
    states, state_idx = np.unique(modeled_races["state"].to_numpy(), return_inverse=True)
    group = np.zeros((n_draws, n_races))
    if composition is not None:
        group = rng.normal(0, error.group, (n_draws, composition.shape[1])) @ composition.T
    return {
        "national": np.repeat(rng.normal(0, error.national, (n_draws, 1)), n_races, axis=1),
        "state": rng.normal(0, error.state, (n_draws, len(states)))[:, state_idx],
        "group": group,
        "race": rng.normal(0, error.race, (n_draws, n_races)),
    }


def simulate_shares(posterior: dict[str, np.ndarray], errors: dict[str, np.ndarray]) -> np.ndarray:
    """Simulate the D-side two-party share in every modeled race on election day.

    Args:
        posterior: Output of model.fit.
        errors: Output of election_day_errors, with as many draws as the posterior.

    Returns:
        Array (draws, n_modeled_races) of D-side two-party vote shares.
    """
    nat = posterior["nat"][:, election_week()]
    logit = posterior["lean"] + nat[:, None] + sum(errors.values())
    return 1 / (1 + np.exp(-logit))


def winners(races: pd.DataFrame, shares: np.ndarray) -> np.ndarray:
    """Party credited with each race's win in each draw.

    Args:
        races: Full races table (all rules); rows with rule == "model" must appear in the same
            order as the columns of shares.
        shares: Output of simulate_shares.

    Returns:
        Array (draws, n_races) of party codes ("D", "R", or "I").
    """
    n_draws = shares.shape[0]
    out = np.empty((n_draws, len(races)), dtype="<U1")
    modeled = (races["rule"] == "model").to_numpy()
    d_party = races["d_party"].fillna("D").to_numpy()
    r_party = races["r_party"].fillna("R").to_numpy()
    out[:, modeled] = np.where(shares > WIN_SHARE, d_party[modeled], r_party[modeled])
    fixed = ~modeled
    out[:, fixed] = np.where(races["rule"].to_numpy()[fixed] == "D", d_party[fixed], r_party[fixed])
    return out
