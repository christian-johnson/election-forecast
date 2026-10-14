"""How the forecast has moved: refit the model on only the polls finished by each past date.

Nothing is stored between runs. Each past forecast is rebuilt from today's data with later polls
hidden, so the race list and candidates are today's.
"""

from datetime import date, timedelta

import numpy as np
from tqdm import tqdm

from forecast import model, simulate
from forecast.config import ELECTION_DATE, OFFICES
from forecast.export import seat_summary
from forecast.matchups import Sources


def past_dates(today: date, days: int, step: int) -> list[date]:
    """Dates to refit: every `step` days from `days` before the election, up to before today.

    Args:
        today: The date of the published forecast, which is not refit.
        days: How many days before the election the trend starts.
        step: Days between refits.

    Returns:
        Dates in increasing order.
    """
    start = ELECTION_DATE - timedelta(days=days)
    end = min(today, ELECTION_DATE)
    return [start + timedelta(days=d) for d in range(0, (end - start).days, step)]


def _point(offices: dict) -> dict:
    """Mean seats and chance of control per party, from build_forecast-style office summaries."""
    point = {}
    for office, summary in offices.items():
        point[office] = {"seats": {p: summary["seats"][p]["mean"] for p in ("D", "R")}}
        if "p_control" in summary:
            point[office]["p_control"] = summary["p_control"]
    return point


def _forecast_as_of(sources: Sources, as_of: date, fit_args: dict) -> dict:
    """Fit the model to the polls finished by `as_of` and summarize each office's seats."""
    obs = sources.observations(as_of)
    inputs = model.build_inputs(obs.modeled, obs.race_obs, obs.generic_obs)
    posterior = model.fit(inputs, **fit_args)
    rng = np.random.default_rng(fit_args["seed"])
    miss = simulate.national_miss(len(posterior["lean"]), rng)
    wins = simulate.winners(obs.races, simulate.simulate_shares(posterior, miss))
    office_of = obs.races["office"].to_numpy()
    return _point({o: seat_summary(o, wins[:, office_of == o]) for o in OFFICES})


def build_trend(sources: Sources, forecast: dict, dates: list[date], fit_args: dict) -> dict:
    """Rebuild the forecast as it stood on each past date, ending at today's published one.

    Args:
        sources: The fetched tables, including polls finished after each date.
        forecast: Today's document from export.build_forecast.
        dates: Past dates to refit, from past_dates.
        fit_args: Keyword arguments for model.fit, including "seed".

    Returns:
        JSON-serializable dict with "dates" (ISO, today last) and, per office, series of mean
        seats ("seats") and chance of control ("p_control", House and Senate) for each party.
    """
    points = [
        _forecast_as_of(sources, as_of, fit_args)
        for as_of in tqdm(dates, desc="Refitting past forecasts")
    ]
    points.append(_point(forecast["offices"]))
    offices = {
        office: {
            measure: {p: [point[office][measure][p] for point in points] for p in ("D", "R")}
            for measure in points[-1][office]
        }
        for office in OFFICES
    }
    today = forecast["updated"][:10]
    return {"dates": [*(d.isoformat() for d in dates), today], "offices": offices}
