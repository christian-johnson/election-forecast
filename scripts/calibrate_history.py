"""Measure, from past elections, how far results stray from PVI and how far polls miss.

Run once: `uv run python scripts/calibrate_history.py`, then copy the printed numbers into
`config.HISTORY`. The misses behind them are saved to site/data/history.json for the About page.
It measures:

- How far each office's results landed from what PVI, the national vote and incumbency predict
  in 2022 and 2024. Both years use Cook's 2022 PVI (built from the 2016 and 2020 presidential
  votes), so neither year's result is part of the PVI it is checked against.
- How far the late polls missed, from FiveThirtyEight's archive of graded polls (1998-2022): the
  miss shared by every race in a year, then the further miss within one state and one race.
"""

import re
from io import BytesIO, StringIO

import httpx
import numpy as np
import orjson
import pandas as pd

from forecast.config import SITE_DATA_DIR
from forecast.fetch import USER_AGENT, WIKIPEDIA_URL
from forecast.races import flatten_columns, locate, name_key, parse_pvi, party_code

YEARS = (2022, 2024)
PAGES = {
    "house": "{year}_United_States_House_of_Representatives_elections",
    "senate": "{year}_United_States_Senate_elections",
    "governor": "{year}_United_States_gubernatorial_elections",
}
HISTORY_JSON = SITE_DATA_DIR / "history.json"
RAW_POLLS_URL = (
    "https://raw.githubusercontent.com/fivethirtyeight/data/master/pollster-ratings/raw_polls.csv"
)
# Only polls from the last three weeks, in races polled at least this often.
LATE_DAYS = 21
MIN_POLLS = 3
# Top-two and ranked-choice elections are not plain D-vs-R contests.
SKIP_STATES = {"AK", "LA"}
# A third candidate this strong means the race was not a D-vs-R contest.
MAX_OTHER_PCT = 15.0
# Logit units per point of margin near a 50-50 race.
POINT = 0.02
# Half-width of a normal 80% interval, in SDs.
Z80 = 1.2816

_FOOTNOTE = re.compile(r"\[[^\]]*\]")
_CANDIDATE = re.compile(r"▌\s*(?P<name>[^▌()]+?)\s*\((?P<party>[^()]*)\)[^▌%]*?(?P<pct>[\d.]+)%")
_RUNOFF = re.compile(r"(?:Instant runoff|Runoff):")
_PLACE = ("Location", "State", "States")
_MEMBER = ("Member", "Senator", "Governor")


def _get(url: str) -> bytes:
    response = httpx.get(url, headers={"User-Agent": USER_AGENT}, timeout=120)
    response.raise_for_status()
    return response.content


def _record(office: str, row: pd.Series, cand_col: str, member_col: str) -> dict | None:
    """Two-party D share and incumbency of one contested race, or None."""
    first_round = _RUNOFF.split(_FOOTNOTE.sub("", str(row[cand_col])))[0]
    votes = {"D": 0.0, "R": 0.0, "O": 0.0}
    names = set()
    for match in _CANDIDATE.finditer(first_round):
        names.add(name_key(match["name"]))
        party = party_code(match["party"].strip())
        pct = float(match["pct"])
        votes[party] = max(votes[party], pct) if party == "O" else votes[party] + pct
    if not (votes["D"] and votes["R"]) or votes["O"] > MAX_OTHER_PCT:
        return None
    member = re.split(r"\s+Redistricted from", _FOOTNOTE.sub("", str(row[member_col])))[0]
    running = name_key(member) in names
    side = {"D": 1, "R": -1}.get(party_code(str(row["Party"])), 0)
    return {
        "office": office,
        "share": votes["D"] / (votes["D"] + votes["R"]),
        "incumbency": side if running else 0,
    }


def parse_results(office: str, html: str) -> pd.DataFrame:
    """Every contested race on one Wikipedia results page, with its PVI and incumbency."""
    pvi, records = {}, {}
    for table in (flatten_columns(t) for t in pd.read_html(StringIO(html))):
        place = next((c for c in table.columns if c in _PLACE), None)
        if place is None:
            continue
        pvi_col = next((c for c in table.columns if "PVI" in c), None)
        cand_col = next((c for c in table.columns if "andidates" in c), None)
        member_col = next((c for c in table.columns if c in _MEMBER), None)
        for _, row in table.iterrows():
            loc = locate(office, row[place])
            if loc is None or loc[0] in SKIP_STATES:
                continue
            if pvi_col and loc not in pvi:
                pvi[loc] = parse_pvi(row[pvi_col])
            if cand_col and member_col and loc not in records:
                records[loc] = _record(office, row, cand_col, member_col)
    rows = [
        {"state": state, "district": district, "pvi": pvi[(state, district)], **record}
        for (state, district), record in records.items()
        if record is not None and (state, district) in pvi
    ]
    return pd.DataFrame(rows)


def pvi_spread(results: pd.DataFrame) -> dict:
    """Measure how far results strayed from PVI + national vote + incumbency, by office.

    Each year's national vote and the incumbency edge are fit on House races. The spread is the
    distance within which 80% of races landed, as the matching normal SD, so that one outlier
    such as Vermont's Republican governor does not set the spread for every race.

    Returns:
        {"incumbency": margin points, "race_spread": {office: margin points}} and, under
        "pvi_misses", every race's miss by office.
    """
    share = np.clip(0.5 + results["pvi"] / 100, 0.02, 0.98)
    y = np.log(results["share"] / (1 - results["share"])) - np.log(share / (1 - share))
    years = pd.get_dummies(results["year"]).to_numpy(float)
    design = np.column_stack([years, results["incumbency"]])
    house = (results["office"] == "house").to_numpy()
    coef, *_ = np.linalg.lstsq(design[house], y[house], rcond=None)
    resid = (y - design @ coef) / POINT
    spread = {}
    for office, r in resid.groupby(results["office"]):
        spread[office] = round(float(r.abs().quantile(0.8)) / Z80, 1)
        print(
            f"{office}: {len(r)} races, median {r.median():+.1f}, "
            f"80% within {Z80 * spread[office]:.1f}"
        )
    return {
        "incumbency": round(float(coef[-1]) / POINT, 1),
        "race_spread": spread,
        "pvi_misses": {
            office: r.round(1).tolist() for office, r in resid.groupby(results["office"])
        },
    }


def poll_misses(polls: pd.DataFrame) -> dict:
    """Measure the miss shared by every race in a year, and the further state and race misses.

    Returns:
        Typical (SD) misses in margin points: "national_miss", "state_miss" and "race_miss",
        plus the largest national miss and its year, and under "national_misses" and
        "local_misses" each year's miss and each race's miss beyond its year's.
    """
    polls = polls[
        polls["type_simple"].isin(["Sen-G", "Gov-G", "House-G"])
        & (polls["cand1_party"] == "DEM")
        & (polls["cand2_party"] == "REP")
        & (polls["time_to_election"] <= LATE_DAYS)
        & (polls["cycle"] % 2 == 0)
    ]
    p1, p2 = polls["cand1_pct"] / 100, polls["cand2_pct"] / 100
    polls = polls.assign(
        miss=polls["margin_poll"] - polls["margin_actual"],
        sampling_var=1e4 * (p1 + p2 - (p1 - p2) ** 2) / polls["samplesize"],
    )
    races = (
        polls.groupby(["cycle", "type_simple", "location"])
        .agg(miss=("miss", "mean"), n=("miss", "size"), var=("sampling_var", "mean"))
        .reset_index()
    )
    races = races[races["n"] >= MIN_POLLS]
    year_miss = races.groupby("cycle")["miss"].mean()
    print("Average late-poll miss by year (+ means polls too D):")
    print(year_miss.round(1).to_string())

    # What is left after the national miss, less the polls' own sampling noise, is split into
    # a part shared by races in the same state and a part for each race.
    resid = races["miss"] - races["cycle"].map(year_miss)
    local_var = resid.var() - (races["var"] / races["n"]).mean()
    groups = resid.groupby([races["cycle"], races["location"].str[:2]])
    cross = ((groups.sum() ** 2 - groups.apply(lambda r: (r**2).sum())) / 2).sum()
    state_var = cross / (groups.size() * (groups.size() - 1) / 2).sum()
    largest = year_miss.abs().idxmax()
    return {
        "national_miss": round(float(np.sqrt((year_miss**2).mean())), 1),
        "largest_national_miss": round(float(year_miss[largest]), 1),
        "largest_national_miss_year": int(largest),
        "state_miss": round(float(np.sqrt(state_var)), 1),
        "race_miss": round(float(np.sqrt(local_var - state_var)), 1),
        "elections": f"{len(year_miss)} elections, {year_miss.index.min()}-{year_miss.index.max()}",
        "race_averages": len(races),
        "national_misses": {int(y): round(float(m), 1) for y, m in year_miss.items()},
        "local_misses": resid.round(1).tolist(),
    }


def main() -> None:
    """Download past results and polls and print the measured spreads."""
    results = pd.concat(
        parse_results(
            office, _get(WIKIPEDIA_URL.format(title=page.format(year=year))).decode()
        ).assign(year=year)
        for year in YEARS
        for office, page in PAGES.items()
    )
    measured = pvi_spread(results) | poll_misses(pd.read_csv(BytesIO(_get(RAW_POLLS_URL))))
    data_keys = ("pvi_misses", "national_misses", "local_misses")
    HISTORY_JSON.write_bytes(
        orjson.dumps({key: measured.pop(key) for key in data_keys}, option=orjson.OPT_NON_STR_KEYS)
    )
    print("\nFor config.HISTORY (margin points):")
    for key, value in measured.items():
        print(f"    {key} = {value!r}")


if __name__ == "__main__":
    main()
