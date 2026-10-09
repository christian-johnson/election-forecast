"""End-to-end: mocked downloads -> tidy CSVs -> model fit -> site JSON."""

import orjson
import pandas as pd
import pytest

from forecast import cli, fetch
from forecast.config import HISTORY
from forecast.model import week_index
from forecast.races import name_key, parse_pvi, parse_races

_CSV_NAMES = ("RACES_CSV", "CANDIDATES_CSV", "POLLS_CSV")


@pytest.fixture
def forecast_doc(tmp_path, monkeypatch, raw_polls, wiki_pages):
    """Run fetch + model against synthetic sources in a temp directory."""
    monkeypatch.setattr(fetch, "fetch_polls", lambda: raw_polls)
    monkeypatch.setattr(fetch, "fetch_wikipedia", wiki_pages.__getitem__)
    monkeypatch.setattr(cli, "DATA_DIR", tmp_path / "data")
    for name in _CSV_NAMES:
        monkeypatch.setattr(cli, name, tmp_path / "data" / getattr(cli, name).name)
    monkeypatch.setattr(cli, "FORECAST_JSON", tmp_path / "site" / "forecast.json")
    monkeypatch.setattr(cli, "STEPS_JSON", tmp_path / "site" / "steps.json")
    monkeypatch.setattr(cli, "TIMELINE_JSON", tmp_path / "site" / "timeline.json")
    monkeypatch.setattr(cli, "LIKELIHOOD_JSON", tmp_path / "site" / "likelihood.json")
    cli.run_fetch()
    cli.run_model(warmup=300, samples=300, chains=1, seed=0)
    return orjson.loads(cli.FORECAST_JSON.read_bytes())


@pytest.fixture
def steps_doc(forecast_doc):  # noqa: ARG001 - the pipeline must run first
    return orjson.loads(cli.STEPS_JSON.read_bytes())


@pytest.fixture
def timeline_doc(forecast_doc):  # noqa: ARG001 - the pipeline must run first
    return orjson.loads(cli.TIMELINE_JSON.read_bytes())


@pytest.fixture
def likelihood_doc(forecast_doc):  # noqa: ARG001 - the pipeline must run first
    return orjson.loads(cli.LIKELIHOOD_JSON.read_bytes())


def _race(doc, office, race_id):
    return next(r for r in doc["offices"][office]["races"] if r["id"] == race_id)


def test_pipeline_writes_complete_forecast(forecast_doc):
    # Arrange
    house = forecast_doc["offices"]["house"]
    senate = forecast_doc["offices"]["senate"]

    # Act
    house_seats = sum(s["mean"] for s in house["seats"].values())
    senate_seats = sum(s["mean"] for s in senate["seats"].values())
    senate_dist = senate["dem_seat_dist"]

    # Assert
    assert forecast_doc["n_polls"] == {"race": 34, "generic": 18}
    assert {len(forecast_doc["offices"][o]["races"]) for o in ("senate", "governor")} == {1, 2}
    assert house_seats == pytest.approx(5, abs=0.2)
    # Two synthetic races plus the 65 real seats not up this cycle; means are rounded.
    assert senate_seats == pytest.approx(67, abs=0.2)
    # Democrats hold 34 seats not up, so the histogram starts at 34 or above.
    assert senate_dist["start"] >= 34
    assert sum(senate_dist["counts"]) == pytest.approx(1000, abs=10)
    assert 2 < forecast_doc["generic_ballot"]["today"]["mean"] < 10


def test_uncontested_race_is_decided_by_rule(forecast_doc):
    race = _race(forecast_doc, "house", "H-OH-03")

    assert race["rule"] == "D"
    assert race["p_d"] == 1.0
    assert race["margin"] is None


def test_well_polled_races_follow_the_polls(forecast_doc):
    assert _race(forecast_doc, "house", "H-OH-01")["p_d"] > 0.95
    assert _race(forecast_doc, "governor", "G-IA")["p_d"] > 0.95
    assert 0.2 < _race(forecast_doc, "senate", "S-MI")["p_d"] < 0.9


def test_unpolled_race_follows_partisan_lean(forecast_doc):
    race = _race(forecast_doc, "house", "H-OH-02")

    assert race["n_polls"] == 0
    assert race["p_d"] < 0.1


def test_polled_independent_takes_weaker_partys_side(forecast_doc):
    alaska = _race(forecast_doc, "house", "H-AK-01")
    nebraska = _race(forecast_doc, "senate", "S-NE")

    assert (alaska["d_name"], alaska["d_party"]) == ("Bill Hill", "I")
    assert (nebraska["d_name"], nebraska["d_party"]) == ("Dan Osborn", "I")
    assert "I" in forecast_doc["offices"]["senate"]["seats"]


def test_parse_races_reads_new_map_pvi_and_dfl(wiki_pages):
    races, candidates = parse_races("house", wiki_pages["house"])
    races = races.set_index("race_id")

    assert races.loc["H-OH-01", "pvi"] == 3.0
    assert races.loc["H-OH-01", "redistricted"]
    assert not races.loc["H-MN-02", "redistricted"]
    assert races.loc["H-AK-01", "district"] == 1
    assert races.loc["H-MN-02", "incumbent_party"] == "D"
    assert candidates.loc[candidates["name"] == "Kim Lee", "party"].item() == "D"
    assert candidates.loc[candidates["name"] == "Greg Landsman", "incumbent"].item()


@pytest.mark.parametrize(("text", "expected"), [("D+7", 7.0), ("R+15[2]", -15.0), ("EVEN", 0.0)])
def test_parse_pvi(text, expected):
    assert parse_pvi(text) == expected


def test_name_key_ignores_initials_suffixes_and_accents():
    assert name_key("Nick Begich III") == name_key("Nick Begich")
    assert name_key("María Elvira Salazar") == "maria salazar"


def test_steps_end_at_the_published_forecast(forecast_doc, steps_doc):
    # Arrange
    modeled = [
        r for o in forecast_doc["offices"].values() for r in o["races"] if r["rule"] == "model"
    ]

    # Act
    finals = {race_id: race["steps"][-1] for race_id, race in steps_doc["races"].items()}

    # Assert
    assert set(finals) == {r["id"] for r in modeled}
    for race in modeled:
        assert finals[race["id"]]["p_d"] == race["p_d"]
        assert finals[race["id"]]["mean"] == pytest.approx(race["margin"]["mean"], abs=0.1)
        assert sum(finals[race["id"]]["hist"]) == pytest.approx(1000, abs=25)


def test_steps_build_from_national_polls_to_race(steps_doc):
    # Arrange
    oh01 = steps_doc["races"]["H-OH-01"]
    oh02 = steps_doc["races"]["H-OH-02"]

    # Act
    polls, miss, prior, after = oh01["steps"]
    *_, unpolled_prior, unpolled_after = oh02["steps"]

    # Assert: step 1 is the national mood alone; the national miss widens it without moving it.
    assert [s["key"] for s in steps_doc["steps"]] == ["national", "national_miss", "prior", "polls"]
    assert polls["mean"] == pytest.approx(steps_doc["national"]["mean"], abs=0.1)
    assert miss["hi"] - miss["lo"] > polls["hi"] - polls["lo"]
    assert miss["mean"] == pytest.approx(polls["mean"], abs=1)
    # OH-01 (D+3, Democratic incumbent) polls far ahead of what PVI and incumbency expect.
    assert prior["mean"] == pytest.approx(oh01["expected"], abs=1)
    assert after["mean"] > prior["mean"] > polls["mean"]
    assert after["p_d"] > prior["p_d"]
    # OH-02 has no polls, so its polls step barely moves from its R+20 starting point.
    assert unpolled_after["mean"] == pytest.approx(unpolled_prior["mean"], abs=1.5)
    assert steps_doc["history"]["race_spread"] == HISTORY.race_spread
    assert set(steps_doc["noise"]) == {"house", "senate", "governor"}


def test_timeline_covers_every_week_and_polled_race(forecast_doc, timeline_doc):
    # Arrange
    modeled = {
        r["id"]: r
        for o in forecast_doc["offices"].values()
        for r in o["races"]
        if r["rule"] == "model"
    }
    n_weeks = len(timeline_doc["national"]["mean"])

    polls = timeline_doc["polls"]

    # Act
    polled = {race_id for race_id, race in modeled.items() if race["n_polls"]}
    every_poll = polls["generic"] + [p for race in polls["races"].values() for p in race]

    # Assert
    assert set(timeline_doc["prior"]) == set(modeled)
    assert set(timeline_doc["races"]) == polled
    # Every poll the model saw is listed once, with a link to its source.
    assert len(polls["generic"]) == forecast_doc["n_polls"]["generic"]
    assert {r: len(p) for r, p in polls["races"].items()} == {
        r: modeled[r]["n_polls"] for r in polled
    }
    assert all(p["url"].startswith("https://example.com/polls/") for p in every_poll)
    assert {len(r["sd"]) for r in timeline_doc["races"].values()} == {n_weeks}
    # After this week the mood holds its value and only grows less certain toward election day.
    now = int(week_index(pd.Series([pd.Timestamp.now()]))[0])
    mean, sd = timeline_doc["national"]["mean"], timeline_doc["national"]["sd"]
    assert len(set(mean[now:])) == 1
    assert sd[now:] == sorted(sd[now:])
    assert sd[-1] > min(sd)


def test_walkthrough_shows_polls_and_replays_the_fit(forecast_doc, likelihood_doc):
    # Arrange
    national = likelihood_doc["national"]
    races = likelihood_doc["races"]
    draws = likelihood_doc["draws"]
    modeled = {
        r["id"]: r
        for o in forecast_doc["offices"].values()
        for r in o["races"]
        if r["rule"] == "model"
    }

    # Act
    n_weeks = len(national["weekly"]["mean"])
    house_d = draws["house_d"]
    p_d = sum(seats >= likelihood_doc["house"]["majority"] for seats in house_d) / len(house_d)

    offices = [r["office"] for r in races]

    # Assert: every generic poll is shown, and races alternate offices, the most polled first.
    assert len(national["polls"]) == forecast_doc["n_polls"]["generic"]
    assert [len(r["polls"]) for r in races] == [modeled[r["id"]]["n_polls"] for r in races]
    assert len(races[0]["polls"]) == max(len(r["polls"]) for r in races)
    assert len(set(offices[: len(set(offices))])) == len(set(offices))
    assert all(len(r["weekly"]["sd"]) == n_weeks for r in races)
    # Each draw holds the whole path, one lean per shown race, and the House it implies.
    assert {len(path) for path in draws["nat"]} == {n_weeks}
    assert {len(leans) for leans in draws["lean"]} == {len(races)}
    assert len(draws["miss"]) == len(draws["nat"])
    assert len(house_d) == len(draws["nat"])
    assert all(0 <= seats <= 5 for seats in house_d)
    assert p_d == pytest.approx(likelihood_doc["house"]["p_d"], abs=0.15)
    assert likelihood_doc["history"]["national_miss"] == HISTORY.national_miss
    assert {p["name"] for p in likelihood_doc["pollsters"]} >= {"Acme Polling", "Beta Research"}
