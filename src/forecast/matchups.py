"""Link polls to races and candidates, decide each race's two sides, and build poll observations.

Every race is reduced to a contest between a Democratic side and a Republican side. A side is
the set of that party's candidates, so races with several candidates per party (Louisiana,
Alaska) are modeled on party vote share. When an independent outpolls a party's strongest
candidate (e.g. Nebraska Senate, Alaska at-large), that independent takes the party's side.
"""

import re

import numpy as np
import pandas as pd

from forecast.config import DEFAULT_SAMPLE_SIZE, MAX_SAMPLE_SIZE
from forecast.races import name_key

# Answers that are not people: these never disqualify a poll when left unmatched.
_NON_CANDIDATE = re.compile(
    r"undecided|other|someone|not sure|unsure|don.?t know|refused|would not|wouldn.?t|"
    r"none|neither|third|skip|no answer",
    re.IGNORECASE,
)
# An unmatched named answer this large means the poll asked about a different matchup.
_MAX_UNMATCHED_PCT = 10.0
_POPULATION_RANK = {"lv": 0, "rv": 1, "v": 2, "a": 3}


def _match_candidate(choice: str, cands: pd.DataFrame) -> int | None:
    """Return the candidates-table index matching a poll answer, or None."""
    exact = cands.index[cands["name"].str.lower() == choice.lower()]
    if len(exact):
        return exact[0]
    key = name_key(choice)
    keyed = cands.index[cands["name"].map(name_key) == key]
    if len(keyed):
        return keyed[0]
    last = key.split(" ")[-1]
    by_last = cands.index[cands["name"].map(lambda n: name_key(n).split(" ")[-1]) == last]
    return by_last[0] if len(by_last) == 1 else None


def match_answers(polls: pd.DataFrame, races: pd.DataFrame, candidates: pd.DataFrame):
    """Attach race_id and candidate index to every race-poll answer.

    Args:
        polls: Answer rows from polls.flatten_polls.
        races: Race table from races.parse_races (all offices concatenated).
        candidates: Candidate table from races.parse_races (all offices concatenated).

    Returns:
        Race-poll answer rows (generic ballot rows excluded) with added columns race_id and
        candidate (index into candidates, or NaN when unmatched). Polls for races not in the
        race table are dropped.
    """
    keys = races.set_index(["office", "state", "district"])["race_id"]
    race_polls = polls[polls["office"] != "generic"].copy()
    race_polls["race_id"] = [
        keys.get((o, s, d))
        for o, s, d in race_polls[["office", "state", "district"]].itertuples(index=False)
    ]
    race_polls = race_polls.dropna(subset=["race_id"])
    by_race = dict(tuple(candidates.groupby("race_id")))
    race_polls["candidate"] = [
        _match_candidate(choice, by_race[race]) if race in by_race else None
        for choice, race in race_polls[["choice", "race_id"]].itertuples(index=False)
    ]
    return race_polls


def _valid_polls(matched: pd.DataFrame) -> pd.DataFrame:
    """Drop polls that include a large unmatched named answer (primaries, hypotheticals)."""
    unmatched = matched["candidate"].isna() & ~matched["choice"].str.contains(_NON_CANDIDATE)
    bad = matched.loc[unmatched & (matched["pct"] >= _MAX_UNMATCHED_PCT), "poll_id"]
    return matched[~matched["poll_id"].isin(bad) & matched["candidate"].notna()]


def assign_sides(races: pd.DataFrame, candidates: pd.DataFrame, matched: pd.DataFrame):
    """Decide which candidates make up each race's D and R sides, and how to forecast it.

    Args:
        races: Race table.
        candidates: Candidate table.
        matched: Output of match_answers.

    Returns:
        (races, candidates). candidates gains a "side" column ("D", "R" or None). races gains
        d_name, r_name (lead candidate of each side), d_party, r_party (party credited with a
        win, "D"/"R"/"I"), incumbent_side ("D", "R" or None), and rule: "model" when both
        sides exist, otherwise the party that wins by default ("D" or "R").
    """
    candidates = candidates.copy()
    valid = _valid_polls(matched)
    candidates["mentions"] = (
        valid.groupby("candidate")["poll_id"].nunique().reindex(candidates.index, fill_value=0)
    )
    support = valid.groupby("candidate")["pct"].mean().reindex(candidates.index, fill_value=0.0)
    candidates["side"] = candidates["party"].where(candidates["party"].isin(["D", "R"]))

    records = []
    for race_id, cands in candidates.groupby("race_id"):
        # A polled independent who outpolls a party's strongest candidate takes that side.
        independents = support[cands.index[cands["party"] == "O"]]
        if len(independents) and independents.max() > 0:
            side_support = {
                side: support[cands.index[cands["side"] == side]].to_numpy().max(initial=0.0)
                for side in ("D", "R")
            }
            weaker = min(side_support, key=side_support.get)
            if independents.max() > side_support[weaker]:
                candidates.loc[cands.index[cands["side"] == weaker], "side"] = None
                candidates.loc[independents.idxmax(), "side"] = weaker
        race_cands = candidates.loc[cands.index]
        record = {"race_id": race_id}
        for side in ("D", "R"):
            members = race_cands[race_cands["side"] == side].sort_values(
                ["incumbent", "mentions"], ascending=False, kind="stable"
            )
            lead = members.iloc[0] if len(members) else None
            record[f"{side.lower()}_name"] = None if lead is None else lead["name"]
            record[f"{side.lower()}_party"] = (
                None if lead is None else (side if lead["party"] == side else "I")
            )
        incumbent_sides = race_cands.loc[race_cands["incumbent"], "side"].dropna()
        record["incumbent_side"] = incumbent_sides.iloc[0] if len(incumbent_sides) else None
        has_d, has_r = record["d_name"] is not None, record["r_name"] is not None
        record["rule"] = "model" if has_d and has_r else ("D" if has_d else "R")
        records.append(record)
    races = races.merge(pd.DataFrame(records), on="race_id", how="left")
    return races, candidates.drop(columns="mentions")


def _dedupe(obs: pd.DataFrame) -> pd.DataFrame:
    """Keep one version per poll, preferring likely voters, when a pollster reports several."""
    obs = obs.assign(rank=obs["population"].map(_POPULATION_RANK).fillna(9))
    obs = obs.sort_values(["rank", "sample_size"], ascending=[True, False])
    return obs.drop_duplicates(["race_id", "pollster", "start_date", "end_date"]).drop(
        columns="rank"
    )


def _finish(obs: pd.DataFrame) -> pd.DataFrame:
    start = pd.to_datetime(obs["start_date"])
    end = pd.to_datetime(obs["end_date"])
    obs["date"] = start + (end - start) / 2
    total = obs["dem"] + obs["rep"]
    n = obs["sample_size"].fillna(DEFAULT_SAMPLE_SIZE).clip(upper=MAX_SAMPLE_SIZE)
    obs["n_two_party"] = np.maximum(n * total / 100, 50)
    return _dedupe(obs).reset_index(drop=True)


def race_poll_observations(matched: pd.DataFrame, candidates: pd.DataFrame) -> pd.DataFrame:
    """Collapse race polls to one D-side vs R-side observation per poll.

    Args:
        matched: Output of match_answers.
        candidates: Candidate table with the "side" column from assign_sides.

    Returns:
        One row per usable poll with poll_id, race_id, pollster, dates, population,
        partisan (sponsor), url, dem and rep (side totals in percent), n_two_party and midpoint
        date.
    """
    valid = _valid_polls(matched)
    valid = valid.assign(side=candidates.loc[valid["candidate"].astype(int), "side"].to_numpy())
    shares = valid.pivot_table(
        index="poll_id", columns="side", values="pct", aggfunc="sum", fill_value=0.0
    ).reindex(columns=["D", "R"], fill_value=0.0)
    shares = shares[(shares["D"] > 0) & (shares["R"] > 0)]
    meta_cols = ["race_id", "pollster", "start_date", "end_date", "sample_size", "population"]
    meta = valid.groupby("poll_id")[[*meta_cols, "partisan", "url"]].first()
    obs = meta.join(shares.rename(columns={"D": "dem", "R": "rep"}), how="inner").reset_index()
    return _finish(obs)


def generic_poll_observations(polls: pd.DataFrame) -> pd.DataFrame:
    """Collapse generic ballot polls to one Democratic vs Republican observation per poll.

    Args:
        polls: Answer rows from polls.flatten_polls.

    Returns:
        Same columns as race_poll_observations, with race_id None.
    """
    generic = polls[polls["office"] == "generic"]
    shares = generic.pivot_table(index="poll_id", columns="choice", values="pct", aggfunc="sum")
    shares = shares.rename(columns={"Dem": "dem", "Rep": "rep"})[["dem", "rep"]].dropna()
    meta_cols = [
        "pollster",
        "start_date",
        "end_date",
        "sample_size",
        "population",
        "partisan",
        "url",
    ]
    meta = generic.groupby("poll_id")[meta_cols].first()
    obs = meta.join(shares, how="inner").reset_index().assign(race_id=None)
    return _finish(obs)
