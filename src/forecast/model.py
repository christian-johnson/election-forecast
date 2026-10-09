"""Joint Bayesian model of every poll, fit with NUTS.

All quantities are on the logit scale of the Democratic-side two-party share. A shared set of
latent parameters (the national environment over time, each race's lean, pollster house
effects) is defined once in `forecast_model`; each data source is a `Component` that adds its
own likelihood term on top of them. Adding a data source (approval, ...) means writing a new
Component and appending it to `ModelInputs.components` - the latent core and the rest of the
pipeline stay unchanged.
"""

import logging
from dataclasses import dataclass, field
from typing import Protocol

import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist
import pandas as pd
from jax import random
from numpyro.infer import MCMC, NUTS

from forecast.config import ELECTION_DATE, HISTORY, OFFICES, POINT, PRIORS, START_DATE, Priors

logger = logging.getLogger(__name__)

_SHARE_CLIP = 0.02
# Fixed by past elections (config.HISTORY), in logit units.
INCUMBENCY = HISTORY.incumbency * POINT
RACE_SPREAD = np.array([HISTORY.race_spread[office] for office in OFFICES]) * POINT
STATE_MISS = HISTORY.state_miss * POINT
RACE_MISS = HISTORY.race_miss * POINT


@dataclass(frozen=True)
class Latent:
    """Shared latent parameters that every Component can read.

    Attributes:
        nat: National environment by week, i.e. the generic ballot (n_weeks,).
        lean: Each modeled race's lean relative to the national environment (n_races,).
        poll_bias: Miss shared by every poll of each race, beyond the national one (n_races,).
        house: Pollster house effects (n_pollsters,).
        priors: Prior scales, for components that sample their own parameters.
    """

    nat: jnp.ndarray
    lean: jnp.ndarray
    poll_bias: jnp.ndarray
    house: jnp.ndarray
    priors: Priors


class Component(Protocol):
    """A data source that adds a likelihood term on top of the shared latent parameters."""

    def observe(self, latent: Latent) -> None:
        """Register this component's observations with numpyro."""


@dataclass(frozen=True)
class RaceSet:
    """Fixed attributes of the modeled (contested) races, aligned with the races table rows.

    Attributes:
        baseline: logit of the PVI-implied D two-party share in a tied national environment.
        office: Index into config.OFFICES.
        state: Index of each race's state, for misses shared within a state.
        incumbency: +1 if the D-side candidate is the incumbent, -1 for the R side, else 0.
    """

    baseline: np.ndarray
    office: np.ndarray
    state: np.ndarray
    incumbency: np.ndarray

    @property
    def expected(self) -> np.ndarray:
        """Each race's lean before its polls: PVI baseline plus the incumbency edge."""
        return self.baseline + INCUMBENCY * self.incumbency

    @classmethod
    def from_races(cls, races: pd.DataFrame) -> "RaceSet":
        """Build from the modeled rows of the races table (rule == "model")."""
        share = np.clip(0.5 + races["pvi"].to_numpy(float) / 100, _SHARE_CLIP, 1 - _SHARE_CLIP)
        return cls(
            baseline=np.log(share / (1 - share)),
            office=races["office"].map(OFFICES.index).to_numpy(int),
            state=np.unique(races["state"].to_numpy(), return_inverse=True)[1],
            incumbency=races["incumbent_side"].map({"D": 1, "R": -1}).fillna(0).to_numpy(float),
        )


@dataclass(frozen=True)
class TwoPartyPolls:
    """Polls of the D side vs the R side: of one race, or the generic ballot.

    Attributes:
        name: Prefix for this component's numpyro sites.
        y: Observed logit D-side two-party share.
        sampling_var: Binomial sampling variance of y.
        week: Week index of each poll's midpoint.
        pollster: Pollster index.
        race: Index into the modeled races, or None for generic ballot polls.
    """

    name: str
    y: np.ndarray
    sampling_var: np.ndarray
    week: np.ndarray
    pollster: np.ndarray
    race: np.ndarray | None = None

    def observe(self, latent: Latent) -> None:
        """Each poll measures the national environment (plus race lean) and its biases."""
        mu = latent.nat[self.week] + latent.house[self.pollster]
        if self.race is not None:
            mu = mu + latent.lean[self.race] + latent.poll_bias[self.race]
        extra = numpyro.sample(
            f"{self.name}_noise", dist.HalfNormal(latent.priors.poll_extra_noise)
        )
        scale = jnp.sqrt(self.sampling_var + extra**2)
        numpyro.sample(f"{self.name}_obs", dist.Normal(mu, scale), obs=self.y)


@dataclass(frozen=True)
class ModelInputs:
    """Everything the model needs: race structure, index sizes, and data components."""

    races: RaceSet
    n_weeks: int
    pollsters: list[str]
    components: list[Component] = field(default_factory=list)


def week_index(dates: pd.Series) -> np.ndarray:
    """Week number since START_DATE, clipped to the election week."""
    days = (pd.to_datetime(dates) - pd.Timestamp(START_DATE)).dt.days.to_numpy()
    return np.clip(days // 7, 0, election_week())


def election_week() -> int:
    """Index of the week containing election day."""
    return (ELECTION_DATE - START_DATE).days // 7


def _two_party_polls(name, obs, pollster_index, race_index=None) -> TwoPartyPolls:
    share = (obs["dem"] / (obs["dem"] + obs["rep"])).to_numpy(float)
    n = obs["n_two_party"].to_numpy(float)
    return TwoPartyPolls(
        name=name,
        y=np.log(share / (1 - share)),
        sampling_var=1 / (n * share * (1 - share)),
        week=week_index(obs["date"]),
        pollster=obs["pollster"].map(pollster_index).to_numpy(int),
        race=None if race_index is None else obs["race_id"].map(race_index).to_numpy(int),
    )


def pollster_names(race_obs: pd.DataFrame, generic_obs: pd.DataFrame) -> list[str]:
    """Every pollster in the fit, in the order of its house effect."""
    return sorted(set(race_obs["pollster"]) | set(generic_obs["pollster"]))


def build_inputs(
    modeled_races: pd.DataFrame,
    race_obs: pd.DataFrame,
    generic_obs: pd.DataFrame,
) -> ModelInputs:
    """Index the data and assemble the model's components.

    Args:
        modeled_races: Races with rule == "model", in the order the model should index them.
        race_obs: Output of matchups.race_poll_observations.
        generic_obs: Output of matchups.generic_poll_observations.

    Returns:
        ModelInputs ready for `fit`.
    """
    race_index = {r: i for i, r in enumerate(modeled_races["race_id"])}
    race_obs = race_obs[race_obs["race_id"].isin(race_index)]
    pollsters = pollster_names(race_obs, generic_obs)
    pollster_index = {p: i for i, p in enumerate(pollsters)}
    office_of = dict(zip(modeled_races["race_id"], modeled_races["office"], strict=True))
    race_office = race_obs["race_id"].map(office_of)
    # One component per office so each gets its own extra-noise scale.
    components: list[Component] = [
        _two_party_polls("generic", generic_obs, pollster_index),
        *(
            _two_party_polls(office, race_obs[race_office == office], pollster_index, race_index)
            for office in OFFICES
            if (race_office == office).any()
        ),
    ]
    return ModelInputs(
        races=RaceSet.from_races(modeled_races),
        n_weeks=election_week() + 1,
        pollsters=pollsters,
        components=components,
    )


def forecast_model(inputs: ModelInputs, priors: Priors = PRIORS) -> None:
    """Numpyro model: the shared latent core, then every component's likelihood."""
    races = inputs.races

    walk_sd = numpyro.sample("walk_sd", dist.HalfNormal(priors.walk_weekly))
    nat_start = numpyro.sample("nat_start", dist.Normal(0.0, 0.2))
    steps = numpyro.sample("nat_steps", dist.Normal(0.0, 1.0).expand([inputs.n_weeks - 1]))
    nat = numpyro.deterministic(
        "nat", nat_start + jnp.concatenate([jnp.zeros(1), jnp.cumsum(steps * walk_sd)])
    )

    # How far each race runs from its PVI and incumbency, as wide as past races ran from theirs.
    race_z = numpyro.sample("race_z", dist.Normal(0.0, 1.0).expand([len(races.baseline)]))
    lean_race = numpyro.deterministic("lean_race", RACE_SPREAD[races.office] * race_z)
    lean = numpyro.deterministic("lean", races.expected + lean_race)
    # Polls of one state, and of one race, can miss together, as much as they have in the past.
    n_states = len(np.unique(races.state))
    state_z = numpyro.sample("state_bias_z", dist.Normal(0.0, 1.0).expand([n_states]))
    race_bias_z = numpyro.sample("race_bias_z", dist.Normal(0.0, 1.0).expand([len(races.state)]))
    poll_bias = STATE_MISS * state_z[races.state] + RACE_MISS * race_bias_z

    house_sd = numpyro.sample("house_sd", dist.HalfNormal(priors.house_effect_scale))
    house_z = numpyro.sample("house_z", dist.Normal(0.0, 1.0).expand([len(inputs.pollsters)]))
    latent = Latent(
        nat=nat, lean=lean, poll_bias=poll_bias, house=house_sd * house_z, priors=priors
    )
    for component in inputs.components:
        component.observe(latent)


def fit(
    inputs: ModelInputs,
    *,
    warmup: int = 1000,
    samples: int = 1000,
    chains: int = 4,
    seed: int = 0,
) -> dict[str, np.ndarray]:
    """Sample the posterior with NUTS.

    Args:
        inputs: Output of build_inputs.
        warmup: Warmup iterations per chain.
        samples: Kept draws per chain.
        chains: Number of chains (run in parallel when host devices allow).
        seed: Random seed.

    Returns:
        Posterior draws for "nat" (draws, n_weeks) and "lean" (draws, n_races), plus the
        scalar and vector parameters, with chains flattened.
    """
    mcmc = MCMC(
        NUTS(forecast_model, target_accept_prob=0.9),
        num_warmup=warmup,
        num_samples=samples,
        num_chains=chains,
        progress_bar=False,
    )
    mcmc.run(random.PRNGKey(seed), inputs, extra_fields=("diverging",))
    divergences = int(mcmc.get_extra_fields()["diverging"].sum())
    if divergences:
        logger.warning("NUTS reported %d divergent transitions", divergences)
    return {k: np.asarray(v) for k, v in mcmc.get_samples().items()}
