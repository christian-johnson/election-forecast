"""Joint Bayesian model of every poll, fit with NUTS.

All quantities are on the logit scale of the Democratic-side two-party share. A shared set of
latent parameters (the national environment over time, each race's lean, pollster house
effects) is defined once in `forecast_model`; each data source is a `Component` that adds its
own likelihood term on top of them. Adding a data source (crosstabs, approval, ...) means
writing a new Component and appending it to `ModelInputs.components` - the latent core and the
rest of the pipeline stay unchanged.
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

from forecast.config import ELECTION_DATE, OFFICES, PRIORS, START_DATE, Priors
from forecast.demographics import GROUP_KEYS, GROUPS

logger = logging.getLogger(__name__)

POPULATIONS = ("lv", "rv", "v", "a")
_SHARE_CLIP = 0.02


def _centering_matrix() -> np.ndarray:
    """Matrix that subtracts, within each dimension, the 2024-voter-weighted mean swing."""
    dims = np.array([g.dimension for g in GROUPS])
    shares = np.array([g.voter_share for g in GROUPS], float)
    same = dims[:, None] == dims[None, :]
    weights = same * shares[None, :] / (same * shares[None, :]).sum(axis=1, keepdims=True)
    return np.eye(len(GROUPS)) - weights


_CENTER = _centering_matrix()
_GROUP_OFFSET = np.array([g.offset for g in GROUPS])


@dataclass(frozen=True)
class Latent:
    """Shared latent parameters that every Component can read.

    Attributes:
        nat: National environment by week, i.e. the generic ballot (n_weeks,).
        lean: Each modeled race's lean relative to the national environment (n_races,).
        house: Pollster house effects (n_pollsters,).
        population: Offset for each poll population in POPULATIONS; likely voters are 0.
        partisan: Shift toward the sponsor's party in partisan-sponsored polls.
        group_shift: Each demographic group's lean relative to the national environment: its
            2024 lean plus its swing since (len(GROUPS),).
        priors: Prior scales, for components that sample their own parameters.
    """

    nat: jnp.ndarray
    lean: jnp.ndarray
    house: jnp.ndarray
    population: jnp.ndarray
    partisan: jnp.ndarray
    group_shift: jnp.ndarray
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
        incumbency: +1 if the D-side candidate is the incumbent, -1 for the R side, else 0.
        composition: Group mix relative to the nation, from demographics.race_composition
            (n_races, len(GROUPS)); zeros when unknown.
    """

    baseline: np.ndarray
    office: np.ndarray
    incumbency: np.ndarray
    composition: np.ndarray

    @classmethod
    def from_races(cls, races: pd.DataFrame, composition: np.ndarray | None = None) -> "RaceSet":
        """Build from the modeled rows of the races table (rule == "model")."""
        share = np.clip(0.5 + races["pvi"].to_numpy(float) / 100, _SHARE_CLIP, 1 - _SHARE_CLIP)
        return cls(
            baseline=np.log(share / (1 - share)),
            office=races["office"].map(OFFICES.index).to_numpy(int),
            incumbency=races["incumbent_side"].map({"D": 1, "R": -1}).fillna(0).to_numpy(float),
            composition=(
                np.zeros((len(races), len(GROUPS))) if composition is None else composition
            ),
        )


@dataclass(frozen=True)
class TwoPartyPolls:
    """Polls of the D side vs the R side: of one race, the generic ballot, or one group.

    Attributes:
        name: Prefix for this component's numpyro sites.
        y: Observed logit D-side two-party share.
        sampling_var: Binomial sampling variance of y.
        week: Week index of each poll's midpoint.
        pollster: Pollster index.
        population: Index into POPULATIONS.
        partisan: +1 for Democratic sponsors, -1 for Republican, 0 otherwise.
        race: Index into the modeled races, or None for generic ballot polls.
        group: Index into GROUPS for generic ballot crosstabs, else None.
    """

    name: str
    y: np.ndarray
    sampling_var: np.ndarray
    week: np.ndarray
    pollster: np.ndarray
    population: np.ndarray
    partisan: np.ndarray
    race: np.ndarray | None = None
    group: np.ndarray | None = None

    def observe(self, latent: Latent) -> None:
        """Each poll measures the national environment (plus race lean) and pollster biases."""
        mu = (
            latent.nat[self.week]
            + latent.house[self.pollster]
            + latent.population[self.population]
            + latent.partisan * self.partisan
        )
        if self.race is not None:
            mu = mu + latent.lean[self.race]
        if self.group is not None:
            mu = mu + latent.group_shift[self.group]
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


def _two_party_polls(name, obs, pollster_index, race_index=None, *, groups=False) -> TwoPartyPolls:
    share = (obs["dem"] / (obs["dem"] + obs["rep"])).to_numpy(float)
    n = obs["n_two_party"].to_numpy(float)
    return TwoPartyPolls(
        name=name,
        y=np.log(share / (1 - share)),
        sampling_var=1 / (n * share * (1 - share)),
        week=week_index(obs["date"]),
        pollster=obs["pollster"].map(pollster_index).to_numpy(int),
        population=obs["population"].map(POPULATIONS.index).to_numpy(int),
        partisan=obs["partisan_sign"].to_numpy(float),
        race=None if race_index is None else obs["race_id"].map(race_index).to_numpy(int),
        group=obs["group"].map(GROUP_KEYS.index).to_numpy(int) if groups else None,
    )


def build_inputs(
    modeled_races: pd.DataFrame,
    race_obs: pd.DataFrame,
    generic_obs: pd.DataFrame,
    crosstab_obs: pd.DataFrame | None = None,
    composition: np.ndarray | None = None,
) -> ModelInputs:
    """Index the data and assemble the model's components.

    Args:
        modeled_races: Races with rule == "model", in the order the model should index them.
        race_obs: Output of matchups.race_poll_observations.
        generic_obs: Output of matchups.generic_poll_observations.
        crosstab_obs: Output of crosstabs.crosstab_observations, if any.
        composition: Output of demographics.race_composition for modeled_races, if known.

    Returns:
        ModelInputs ready for `fit`.
    """
    race_index = {r: i for i, r in enumerate(modeled_races["race_id"])}
    race_obs = race_obs[race_obs["race_id"].isin(race_index)]
    crosstab_obs = crosstab_obs if crosstab_obs is not None else generic_obs.iloc[:0]
    pollsters = sorted(
        set(race_obs["pollster"]) | set(generic_obs["pollster"]) | set(crosstab_obs["pollster"])
    )
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
    if len(crosstab_obs):
        components.append(_two_party_polls("crosstab", crosstab_obs, pollster_index, groups=True))
    return ModelInputs(
        races=RaceSet.from_races(modeled_races, composition),
        n_weeks=election_week() + 1,
        pollsters=pollsters,
        components=components,
    )


def forecast_model(inputs: ModelInputs, priors: Priors = PRIORS) -> None:
    """Numpyro model: the shared latent core, then every component's likelihood."""
    n_offices = len(OFFICES)
    races = inputs.races

    walk_sd = numpyro.sample("walk_sd", dist.HalfNormal(priors.walk_weekly))
    nat_start = numpyro.sample("nat_start", dist.Normal(0.0, 0.2))
    steps = numpyro.sample("nat_steps", dist.Normal(0.0, 1.0).expand([inputs.n_weeks - 1]))
    nat = numpyro.deterministic(
        "nat", nat_start + jnp.concatenate([jnp.zeros(1), jnp.cumsum(steps * walk_sd)])
    )

    office_shift = numpyro.sample(
        "office_shift", dist.Normal(0.0, priors.office_shift).expand([n_offices])
    )
    incumbency = numpyro.sample(
        "incumbency",
        dist.Normal(priors.incumbency_mean, priors.incumbency_sd).expand([n_offices]),
    )
    race_sd = numpyro.sample("race_sd", dist.HalfNormal(priors.race_scale).expand([n_offices]))
    race_z = numpyro.sample("race_z", dist.Normal(0.0, 1.0).expand([len(races.baseline)]))
    swing_raw = numpyro.sample(
        "group_swing_raw", dist.Normal(0.0, priors.group_swing).expand([len(GROUPS)])
    )
    # Swings are relative to the national shift, so each dimension's weighted mean is zero.
    group_swing = numpyro.deterministic("group_swing", _CENTER @ swing_raw)
    # Each piece of the lean is recorded so the site can show how it builds up the forecast.
    lean_office = numpyro.deterministic(
        "lean_office",
        office_shift[races.office] + incumbency[races.office] * races.incumbency,
    )
    lean_groups = numpyro.deterministic("lean_groups", races.composition @ group_swing)
    lean_race = numpyro.deterministic("lean_race", race_sd[races.office] * race_z)
    lean = numpyro.deterministic("lean", races.baseline + lean_office + lean_groups + lean_race)

    house_sd = numpyro.sample("house_sd", dist.HalfNormal(priors.house_effect_scale))
    house_z = numpyro.sample("house_z", dist.Normal(0.0, 1.0).expand([len(inputs.pollsters)]))
    population_raw = numpyro.sample(
        "population_effect",
        dist.Normal(0.0, priors.population_effect).expand([len(POPULATIONS) - 1]),
    )
    partisan = numpyro.sample(
        "partisan_effect", dist.Normal(priors.partisan_sponsor_mean, priors.partisan_sponsor_sd)
    )
    latent = Latent(
        nat=nat,
        lean=lean,
        house=house_sd * house_z,
        population=jnp.concatenate([jnp.zeros(1), population_raw]),
        partisan=partisan,
        group_shift=_GROUP_OFFSET + group_swing,
        priors=priors,
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
