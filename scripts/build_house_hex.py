"""Build the House hex map layout (one hex per district) from Census district centroids.

Each state is a solid clump of hexes, kept apart from other states with room for a label above.

Run once: `uv run python scripts/build_house_hex.py`. District numbering, not district lines,
is what the layout keys on, so it stays valid after mid-decade redistricting as long as each
state's seat count is unchanged (it is fixed until the 2030 apportionment).
"""

import io
import zipfile
from pathlib import Path

import httpx
import numpy as np
import orjson
import pandas as pd
from scipy.optimize import linear_sum_assignment

GAZETTEER_URL = (
    "https://www2.census.gov/geo/docs/maps-data/data/gazetteer/2024_Gazetteer/"
    "2024_Gaz_119CDs_national.zip"
)
OUT = Path(__file__).resolve().parents[1] / "site" / "data" / "house_hex.json"
# Hex radii per km of true distance; small enough that crowded regions spread into empty ones.
GEO_SCALE = 0.013
# Layout units are hex radii: free space between state boxes and room for each state's label.
GAP = 0.8
LABEL_HEIGHT = 1.4
MAX_ASPECT = 2.5
# States on neighboring tiles drift together until their boxes are this close.
NEIGHBOR_GAP = 0.5
PULL_STEPS, MAX_STEPS = 10000, 30000
# Distance from a pointy-top hex's center to its box edge, in hex radii.
HEX_HALF = np.array([np.sqrt(3) / 2, 1.0])
LABEL_SPACE = np.array([0.0, LABEL_HEIGHT])
GROW_ORIGINS = np.array([[0.0, 0.0], [np.sqrt(3) / 2, 0.0], [np.sqrt(3) / 2, -0.5]])
# Alaska and Hawaii are moved to the lower left, as on most US maps (lon, lat).
INSETS = {"AK": (-118.0, 26.0), "HI": (-108.0, 25.0)}
NON_STATES = {"DC", "PR"}
# A standard state tile grid (one square per state). It decides which side of each other two
# crowded states end up on, and which states are pulled together.
TILE_GRID = """
.  .  .  .  .  .  .  .  .  .  ME .
.  .  .  .  .  .  .  .  .  VT NH .
WA ID MT ND MN WI MI .  NY MA .  .
OR NV WY SD IA IL IN OH PA NJ CT RI
CA UT CO NE MO KY WV VA MD DE .  .
.  AZ NM KS AR TN NC SC .  .  .  .
.  .  .  OK LA MS AL GA .  .  .  .
AK HI .  TX .  .  .  .  FL .  .  .
"""
TILES = {
    name: (col, -row)
    for row, line in enumerate(TILE_GRID.strip().splitlines())
    for col, name in enumerate(line.split())
    if name != "."
}


def albers(lon: np.ndarray, lat: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Albers equal-area conic projection for the contiguous US, in km."""
    radius = 6371.0
    phi1, phi2, phi0, lam0 = np.radians([29.5, 45.5, 37.5, -96.0])
    n = (np.sin(phi1) + np.sin(phi2)) / 2
    c = np.cos(phi1) ** 2 + 2 * n * np.sin(phi1)
    rho0 = radius * np.sqrt(c - 2 * n * np.sin(phi0)) / n
    phi, lam = np.radians(lat), np.radians(lon)
    rho = radius * np.sqrt(c - 2 * n * np.sin(phi)) / n
    theta = n * (lam - lam0)
    return rho * np.sin(theta), rho0 - rho * np.cos(theta)


def load_centroids() -> pd.DataFrame:
    """Download district centroids; returns columns id, state, lon, lat, area."""
    response = httpx.get(GAZETTEER_URL, timeout=120)
    response.raise_for_status()
    with zipfile.ZipFile(io.BytesIO(response.content)) as archive:
        text = archive.read(archive.namelist()[0]).decode("latin-1")
    table = pd.read_csv(io.StringIO(text), sep="\t", dtype={"GEOID": str})
    table.columns = table.columns.str.strip()
    # "ZZ" marks water areas not assigned to any district.
    table = table[~table["USPS"].isin(NON_STATES) & table["GEOID"].str[2:].str.isdigit()]
    district = table["GEOID"].str[2:].astype(int).clip(lower=1)
    out = pd.DataFrame(
        {
            "id": table["USPS"] + "-" + district.map("{:02d}".format),
            "state": table["USPS"],
            "lon": table["INTPTLONG"],
            "lat": table["INTPTLAT"],
            "area": table["ALAND_SQMI"],
        }
    )
    for state, (lon, lat) in INSETS.items():
        # Keep each inset state's own districts spread around its new anchor.
        mask = out["state"] == state
        out.loc[mask, "lon"] = lon + (out.loc[mask, "lon"] - out.loc[mask, "lon"].mean()) * 0.3
        out.loc[mask, "lat"] = lat + (out.loc[mask, "lat"] - out.loc[mask, "lat"].mean()) * 0.3
    return out.reset_index(drop=True)


def clump(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Pack one state's districts into a solid hex clump shaped like the state.

    Cells are the n nearest to the origin under the districts' own covariance, so long states
    give long clumps; districts are then matched to cells keeping their relative positions.

    Returns:
        (n, 2) cell centers in hex radii, centered on the origin, y pointing north.
    """
    n = len(x)
    offset = np.stack([x - x.mean(), y - y.mean()], axis=1)
    if n == 1:
        return np.zeros((1, 2))
    evals, evecs = np.linalg.eigh(np.cov(offset.T) + 1e-6 * np.eye(2))
    # Cap elongation, more so for small states, so thin states form a clump rather than a line.
    aspect = min(MAX_ASPECT, 1 + n / 12)
    evals = np.maximum(evals, evals.max() / aspect**2)
    whiten = evecs / np.sqrt(evals)
    span = int(np.ceil(np.sqrt(n))) + 4
    q, r = np.meshgrid(np.arange(-2 * span, 2 * span + 1), np.arange(-span, span + 1))
    q, r = q.ravel(), r.ravel()
    grid = np.stack([np.sqrt(3) * (q + r / 2), -1.5 * r], axis=1)

    def nearest(origin: np.ndarray) -> np.ndarray:
        # Lexsort breaks distance ties by position, so the layout is deterministic.
        dist = (((grid - origin) @ whiten) ** 2).sum(axis=1).round(6)
        return grid[np.lexsort((grid[:, 0], grid[:, 1], dist))[:n]]

    def spread(cells: np.ndarray) -> float:
        return float((((cells - cells.mean(axis=0)) @ whiten) ** 2).sum())

    # Growing from a cell center, an edge, or a vertex gives differently shaped clumps.
    cells = min((nearest(o) for o in GROW_ORIGINS), key=spread)
    cells -= cells.mean(axis=0)
    scale = np.sqrt((cells**2).sum() / max((offset**2).sum(), 1e-9))
    cost = ((offset[:, None, :] * scale - cells[None, :, :]) ** 2).sum(axis=2)
    _, cell = linear_sum_assignment(cost)
    return cells[cell]


def place(home: np.ndarray, half: np.ndarray, tiles: np.ndarray) -> np.ndarray:
    """Push state boxes apart until none overlap, keeping each near its home position.

    Overlapping states separate along the axis on which their tiles differ most, keeping the
    tile grid's left/right and above/below order. States on neighboring tiles are pulled back
    together, and larger states move less than smaller ones.

    Args:
        home: (k, 2) geographic centers in hex radii.
        half: (k, 2) half-width and half-height of each state's box, gap included.
        tiles: (k, 2) tile grid positions, y pointing north.

    Returns:
        (k, 2) box centers.
    """
    span = half[:, None, :] + half[None, :, :]
    tile_delta = tiles[:, None, :] - tiles[None, :, :]
    # Diagonal tiles fall back to the axis on which their homes differ most.
    home_delta = home[:, None, :] - home[None, :, :]
    tie = np.abs(tile_delta[..., 0]) == np.abs(tile_delta[..., 1])
    axis = np.where(
        tie,
        (np.abs(home_delta) / span).argmax(axis=2),
        np.abs(tile_delta).argmax(axis=2),
    )
    pick = np.eye(2)[axis]
    side = np.sign((np.where(tie[..., None], home_delta, tile_delta) * pick).sum(axis=2))
    area = half.prod(axis=1)
    # Share of each pairwise push that falls on the first state of the pair.
    share = area[None, :] / (area[:, None] + area[None, :])
    a, b = np.nonzero(np.triu(np.abs(tile_delta).sum(axis=2) == 1))
    center = home.copy()
    for step in range(MAX_STEPS):
        delta = center[:, None, :] - center[None, :, :]
        hit = (np.abs(delta) < span).all(axis=2)
        np.fill_diagonal(hit, val=False)
        need = (span * pick).sum(axis=2) - side * (delta * pick).sum(axis=2)
        settling = step >= PULL_STEPS
        if not hit.any() and settling:
            break
        # The small overshoot lets halving steps actually finish.
        amount = np.where(hit, np.clip(need, 0, None) + 0.02, 0) * share
        move = (amount * side)[..., None] * pick
        gap = (np.abs(delta[a, b]) - span[a, b]).max(axis=1)
        tug = 0.02 * np.clip(gap - NEIGHBOR_GAP, 0, None)[:, None]
        direction = delta[a, b] / np.linalg.norm(delta[a, b], axis=1, keepdims=True)
        pull = 0.002 * (home - center)
        np.add.at(pull, a, -tug * direction)
        np.add.at(pull, b, tug * direction)
        # The last steps only separate, so the pulls cannot leave states overlapping.
        center += 0.5 * move.sum(axis=1) + (0 if settling else pull)
    else:
        print("Warning: placement did not converge")
    return center


def main() -> None:
    """Build and write the layout."""
    districts = load_centroids()
    x, y = albers(districts["lon"].to_numpy(), districts["lat"].to_numpy())
    states = districts["state"].to_numpy()
    area = districts["area"].to_numpy()
    names = np.unique(states)
    cells = np.zeros((len(districts), 2))
    home, low, high = (np.zeros((len(names), 2)) for _ in range(3))
    for i, name in enumerate(names):
        mask = states == name
        cells[mask] = clump(x[mask], y[mask])
        weight = area[mask] / area[mask].sum()
        home[i] = x[mask] @ weight, y[mask] @ weight
        low[i] = cells[mask].min(axis=0) - HEX_HALF
        high[i] = cells[mask].max(axis=0) + HEX_HALF + LABEL_SPACE
    half = (high - low) / 2 + GAP / 2
    tiles = np.array([TILES[name] for name in names], dtype=float)
    center = place(home * GEO_SCALE, half, tiles)
    shift = center - (low + high) / 2
    index = np.searchsorted(names, states)
    pos = cells + shift[index]
    # Screen coordinates grow downward.
    pos[:, 1] *= -1
    pos -= pos.min(axis=0) - HEX_HALF - LABEL_SPACE
    layout = [
        {"id": d, "state": s, "x": round(float(px), 3), "y": round(float(py), 3)}
        for d, s, (px, py) in zip(districts["id"], states, pos, strict=True)
    ]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_bytes(
        orjson.dumps(sorted(layout, key=lambda h: h["id"]), option=orjson.OPT_APPEND_NEWLINE)
    )
    print(f"Wrote {len(layout)} hexes to {OUT}")


if __name__ == "__main__":
    main()
