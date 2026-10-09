"""Turn posterior draws into simulated elections, including races decided by rule."""

import numpy as np
import pandas as pd

from forecast.config import HISTORY, POINT, History
from forecast.model import election_week

# A side wins with more than half of the two-party vote.
WIN_SHARE = 0.5


def national_miss(n_draws: int, rng: np.random.Generator, history: History = HISTORY) -> np.ndarray:
    """Draw the polling miss shared by every race, which the polls cannot reveal.

    Args:
        n_draws: Number of simulated elections.
        rng: Random generator.
        history: Typical polling misses in past elections.

    Returns:
        Logit shift for each simulated election, shape (draws, 1).
    """
    return rng.normal(0, history.national_miss * POINT, (n_draws, 1))


def simulate_shares(posterior: dict[str, np.ndarray], miss: np.ndarray) -> np.ndarray:
    """Simulate the D-side two-party share in every modeled race on election day.

    Args:
        posterior: Output of model.fit.
        miss: Output of national_miss, with as many draws as the posterior.

    Returns:
        Array (draws, n_modeled_races) of D-side two-party vote shares.
    """
    nat = posterior["nat"][:, election_week()]
    logit = posterior["lean"] + nat[:, None] + miss
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
