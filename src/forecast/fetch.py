"""Download raw inputs: polls (VoteHub), race tables (Wikipedia), demographics (Census ACS)."""

import httpx
import orjson

VOTEHUB_URL = "https://api.votehub.com/polls"
ACS_URL = (
    "https://www2.census.gov/programs-surveys/acs/summary_file/2024/table-based-SF/data/"
    "1YRData/acsdt1y2024-{table}.dat"
)
WIKIPEDIA_URL = "https://en.wikipedia.org/api/rest_v1/page/html/{title}"
# Wikipedia asks API clients to identify themselves with a contact URL.
USER_AGENT = "midterm-forecast/0.1 (https://github.com/christian-johnson/election-forecast)"

WIKIPEDIA_PAGES = {
    "house": "2026_United_States_House_of_Representatives_elections",
    "senate": "2026_United_States_Senate_elections",
    "governor": "2026_United_States_gubernatorial_elections",
}


def _get(url: str) -> bytes:
    response = httpx.get(
        url, headers={"User-Agent": USER_AGENT}, timeout=120, follow_redirects=True
    )
    response.raise_for_status()
    return response.content


def fetch_polls() -> list[dict]:
    """Download every poll VoteHub tracks.

    Returns:
        Raw poll records as returned by the VoteHub API.

    Raises:
        httpx.HTTPError: If the request fails.
    """
    return orjson.loads(_get(VOTEHUB_URL))


def fetch_wikipedia(office: str) -> str:
    """Download the rendered HTML of the Wikipedia elections page for one office.

    Args:
        office: One of "house", "senate", "governor".

    Returns:
        Page HTML.

    Raises:
        httpx.HTTPError: If the request fails.
    """
    return _get(WIKIPEDIA_URL.format(title=WIKIPEDIA_PAGES[office])).decode()


def fetch_acs(table: str) -> str:
    """Download one ACS 2024 1-year detailed table for every geography (no API key needed).

    Args:
        table: Lowercase table id, e.g. "b15003".

    Returns:
        Pipe-delimited table text.

    Raises:
        httpx.HTTPError: If the request fails.
    """
    return _get(ACS_URL.format(table=table)).decode()
