"""Parse the Wikipedia 2026 election pages into tidy race and candidate tables."""

import re
import unicodedata
from io import StringIO

import pandas as pd

STATES = {
    "Alabama": "AL", "Alaska": "AK", "Arizona": "AZ", "Arkansas": "AR", "California": "CA",
    "Colorado": "CO", "Connecticut": "CT", "Delaware": "DE", "Florida": "FL", "Georgia": "GA",
    "Hawaii": "HI", "Idaho": "ID", "Illinois": "IL", "Indiana": "IN", "Iowa": "IA",
    "Kansas": "KS", "Kentucky": "KY", "Louisiana": "LA", "Maine": "ME", "Maryland": "MD",
    "Massachusetts": "MA", "Michigan": "MI", "Minnesota": "MN", "Mississippi": "MS",
    "Missouri": "MO", "Montana": "MT", "Nebraska": "NE", "Nevada": "NV", "New Hampshire": "NH",
    "New Jersey": "NJ", "New Mexico": "NM", "New York": "NY", "North Carolina": "NC",
    "North Dakota": "ND", "Ohio": "OH", "Oklahoma": "OK", "Oregon": "OR", "Pennsylvania": "PA",
    "Rhode Island": "RI", "South Carolina": "SC", "South Dakota": "SD", "Tennessee": "TN",
    "Texas": "TX", "Utah": "UT", "Vermont": "VT", "Virginia": "VA", "Washington": "WA",
    "West Virginia": "WV", "Wisconsin": "WI", "Wyoming": "WY",
}  # fmt: skip

OFFICE_PREFIX = {"house": "H", "senate": "S", "governor": "G"}
_FOOTNOTE = re.compile(r"\[[^\]]*\]")
_CANDIDATE = re.compile(r"^(?P<name>.*)\((?P<party>[^()]*)\)$")
_SUFFIXES = {"jr", "sr", "ii", "iii", "iv"}


def name_key(name: str) -> str:
    """Reduce a person's name to "first last" for fuzzy matching across sources.

    Args:
        name: A full name, possibly with accents, middle initials, or suffixes.

    Returns:
        Lowercase "first last" key.

    Example:
        >>> name_key("Dan S. Sullivan Jr.")
        'dan sullivan'
    """
    ascii_name = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode()
    tokens = [
        t
        for t in re.sub(r"[^a-z\s-]", " ", ascii_name.lower()).split()
        if len(t) > 1 and t not in _SUFFIXES
    ]
    return f"{tokens[0]} {tokens[-1]}" if tokens else ""


def party_code(label: str) -> str:
    """Map a Wikipedia party label to D, R, or O (other), counting fusion and DFL labels."""
    if "Democratic" in label or label == "DFL":
        return "D"
    if "Republican" in label:
        return "R"
    return "O"


def parse_pvi(text: str) -> float:
    """Convert a Cook PVI label such as "D+7", "R+15" or "EVEN" to signed points (D positive)."""
    text = _FOOTNOTE.sub("", str(text)).strip()
    if text.upper() == "EVEN":
        return 0.0
    party, points = text.split("+")
    return float(points) if party == "D" else -float(points)


def parse_candidates(cell: str) -> list[tuple[str, str]]:
    """Split a Wikipedia "Candidates" cell into (name, party label) pairs.

    Args:
        cell: Text such as "▌Jane Doe (Democratic)[12] ▌John Roe (Republican) 51.2%".

    Returns:
        List of (name, party label) tuples, in listed order.
    """
    out = []
    for raw in str(cell).split("▌"):
        piece = re.sub(r"[\d.]+%", "", _FOOTNOTE.sub("", raw)).strip()
        match = _CANDIDATE.match(piece)
        if match:
            out.append((match["name"].strip(), match["party"].strip()))
    return out


def flatten_columns(table: pd.DataFrame) -> pd.DataFrame:
    """Keep the last header row of a Wikipedia table, without footnote marks or repeats."""
    cols = table.columns.get_level_values(-1) if table.columns.nlevels > 1 else table.columns
    table = table.copy()
    table.columns = [_FOOTNOTE.sub("", str(c)).strip() for c in cols]
    return table.loc[:, ~table.columns.duplicated()]


def locate(office: str, place: str) -> tuple[str, int] | None:
    """Return (state abbreviation, district number) for a table's first column, if valid."""
    place = re.sub(r"\s+", " ", _FOOTNOTE.sub("", str(place)))
    place = re.sub(r" ?\(Class \d\)", "", place).strip()
    if office != "house":
        return (STATES[place], 0) if place in STATES else None
    match = re.match(r"^(?P<state>.+?) (?P<district>\d+|at-large)$", place)
    if not match or match["state"] not in STATES:
        return None
    district = 1 if match["district"] == "at-large" else int(match["district"])
    return STATES[match["state"]], district


def _collect_rows(office: str, tables: list[pd.DataFrame]):
    """Find each race's candidate row and PVI across all of a page's tables.

    Returns:
        (rows, pvi, redistricted). rows and pvi are keyed by (state, district). Redistricted
        districts are listed in a table with both old and new PVIs; a 2026 column wins.
    """
    place_col = "Location" if office == "house" else "State"
    pvi: dict[tuple[str, int], tuple[bool, float]] = {}
    rows: dict[tuple[str, int], pd.Series] = {}
    redistricted: set[tuple[str, int]] = set()
    for table in tables:
        if place_col not in table.columns:
            continue
        pvi_cols = sorted((c for c in table.columns if "PVI" in c), key=lambda c: "2026" not in c)
        current = bool(pvi_cols) and "2026" in pvi_cols[0]
        for _, row in table.iterrows():
            loc = locate(office, row[place_col])
            if loc is None:
                continue
            if len(pvi_cols) > 1:
                redistricted.add(loc)
            if pvi_cols and (loc not in pvi or (current and not pvi[loc][0])):
                pvi[loc] = (current, parse_pvi(row[pvi_cols[0]]))
            if "Candidates" in table.columns and loc not in rows:
                rows[loc] = row
    return rows, {loc: value for loc, (_, value) in pvi.items()}, redistricted


def parse_races(office: str, html: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Extract every race of one office from its Wikipedia elections page.

    Args:
        office: One of "house", "senate", "governor".
        html: Page HTML from fetch.fetch_wikipedia.

    Returns:
        (races, candidates). races has one row per race with columns race_id, office, state,
        district, pvi, redistricted (new map for 2026), incumbent, incumbent_party.
        candidates has one row per listed general election candidate with columns race_id,
        name, party, party_label, incumbent.

    Raises:
        ValueError: If no race tables are found on the page.
    """
    rows, pvi, redistricted = _collect_rows(
        office, [flatten_columns(t) for t in pd.read_html(StringIO(html))]
    )
    if not rows:
        msg = f"No {office} race tables found"
        raise ValueError(msg)

    member_col = {"house": "Member", "senate": "Senator", "governor": "Governor"}[office]
    race_records, candidate_records = [], []
    for (state, district), row in sorted(rows.items()):
        race_id = f"{OFFICE_PREFIX[office]}-{state}" + (f"-{district:02d}" if district else "")
        party_label = str(row.get("Party", ""))
        member = _FOOTNOTE.sub("", str(row.get(member_col, "")))
        member = re.split(r"\s+Redistricted from", member)[0].strip()
        if member.startswith(("None", "Vacant")) or member == "nan":
            member, party_label = "", "nan"
        race_records.append(
            {
                "race_id": race_id,
                "office": office,
                "state": state,
                "district": district,
                "pvi": pvi.get((state, district)),
                "redistricted": (state, district) in redistricted,
                "incumbent": member,
                "incumbent_party": party_code(party_label) if party_label != "nan" else None,
            }
        )
        for name, label in parse_candidates(row["Candidates"]):
            is_incumbent = member.startswith(name) or name_key(name) == name_key(member)
            candidate_records.append(
                {
                    "race_id": race_id,
                    "name": name,
                    "party": party_code(label),
                    "party_label": label,
                    "incumbent": is_incumbent,
                }
            )
    return pd.DataFrame(race_records), pd.DataFrame(candidate_records)
