"""End-to-end: mocked downloads -> tidy CSVs -> model fit -> site JSON."""

import orjson
import pandas as pd
import pytest

from forecast import cli, fetch
from forecast.crosstabs import crosstab_observations
from forecast.demo import RACES, run_scenario
from forecast.demographics import composition
from forecast.races import name_key, parse_pvi, parse_races

_CSV_NAMES = ("RACES_CSV", "CANDIDATES_CSV", "POLLS_CSV", "CROSSTABS_CSV", "COMPOSITION_CSV")


@pytest.fixture
def run_pipeline(tmp_path, monkeypatch, raw_polls, wiki_pages, acs_tables):
    """Run fetch + model against synthetic sources in a temp directory, with optional crosstabs."""
    monkeypatch.setattr(fetch, "fetch_polls", lambda: raw_polls)
    monkeypatch.setattr(fetch, "fetch_wikipedia", wiki_pages.__getitem__)
    monkeypatch.setattr(fetch, "fetch_acs", acs_tables.__getitem__)
    monkeypatch.setattr(cli, "DATA_DIR", tmp_path / "data")
    for name in _CSV_NAMES:
        monkeypatch.setattr(cli, name, tmp_path / "data" / getattr(cli, name).name)
    monkeypatch.setattr(cli, "FORECAST_JSON", tmp_path / "site" / "forecast.json")
    monkeypatch.setattr(cli, "STEPS_JSON", tmp_path / "site" / "steps.json")
    cli.run_demographics()
    cli.run_fetch()

    def run(crosstab_rows=None):
        cli.CROSSTABS_CSV.unlink(missing_ok=True)
        if crosstab_rows is not None:
            pd.DataFrame(crosstab_rows).to_csv(cli.CROSSTABS_CSV, index=False)
        cli.run_model(warmup=300, samples=300, chains=1, seed=0)
        return orjson.loads(cli.FORECAST_JSON.read_bytes())

    return run


@pytest.fixture
def forecast_doc(run_pipeline):
    return run_pipeline()


@pytest.fixture
def steps_doc(forecast_doc):  # noqa: ARG001 - the pipeline must run first
    return orjson.loads(cli.STEPS_JSON.read_bytes())


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
    assert forecast_doc["n_polls"] == {"race": 34, "generic": 18, "crosstab": 0}
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


def test_crosstab_swing_moves_races_by_group_mix(run_pipeline):
    # Arrange: crosstabs show Hispanic voters swinging hard to Republicans since 2024 (51-48).
    rows = [
        {
            "pollster": "Acme Polling",
            "start_date": f"2026-08-{d:02d}",
            "end_date": f"2026-08-{d + 2:02d}",
            "population": "lv",
            "partisan": None,
            "dimension": "race",
            "group": "hispanic",
            "dem": 30,
            "rep": 65,
            "n": 300,
        }
        for d in range(1, 21)
    ]

    # Act
    without = _race(run_pipeline(), "house", "H-MN-02")
    with_crosstabs = run_pipeline(rows)

    # Assert: MN-02 is unpolled and, in the synthetic Census data, mostly Hispanic.
    assert with_crosstabs["n_polls"]["crosstab"] == 20
    shift = _race(with_crosstabs, "house", "H-MN-02")["margin"]["mean"] - without["margin"]["mean"]
    assert shift < -1.5


def test_composition_maps_census_geographies(acs_tables):
    shares = composition(acs_tables).pivot_table(index="geo", columns="group", values="share")

    assert {"US", "MN", "MN-02", "AK-01"} <= set(shares.index)
    assert shares.loc["MN-02", "race:hispanic"] == pytest.approx(0.6)
    # Black adults are suppressed in the synthetic data, so they fall into "other".
    assert shares.loc["MN-02", "race:other"] == pytest.approx(0.1)
    assert shares.filter(like="age:").sum(axis=1).to_numpy() == pytest.approx(1.0, abs=1e-3)


def test_crosstabs_reject_unknown_groups():
    row = {"pollster": "A", "start_date": "2026-08-01", "end_date": "2026-08-02",
           "population": "lv", "partisan": None, "dimension": "race", "group": "martian",
           "dem": 50, "rep": 40, "n": 100}  # fmt: skip

    with pytest.raises(ValueError, match="race:martian"):
        crosstab_observations(pd.DataFrame([row]))


def test_demo_recovers_known_truth_without_polling_miss():
    # Act
    result = run_scenario(0.0, warmup=300, samples=300, chains=1)

    # Assert
    covered = [r["lo"] <= r["truth"] <= r["hi"] for r in result["races"]]
    assert len(covered) == len(RACES)
    assert sum(covered) >= 5
    election_day = result["national"][-1]
    assert election_day["lo"] <= election_day["truth"] <= election_day["hi"]
    effects = {p["name"]: p["mean"] for p in result["pollsters"]}
    assert effects["Pollster A"] > effects["Pollster B"]


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
    polls, miss, lean, *_, race_polls, _ = steps_doc["races"]["H-OH-01"]["steps"]
    national = steps_doc["national"]

    # Step 1 is the national mood alone; the shared miss widens it without moving it.
    assert polls["mean"] == pytest.approx(national["mean"], abs=0.1)
    assert miss["hi"] - miss["lo"] > polls["hi"] - polls["lo"]
    assert miss["mean"] == pytest.approx(polls["mean"], abs=1)
    # OH-01 (D+3) polls far ahead of its lean, and its polls step says so.
    assert lean["mean"] > polls["mean"]
    assert race_polls["p_d"] > lean["p_d"]
