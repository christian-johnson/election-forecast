"""Download raw inputs: polls (VoteHub) and race tables (Wikipedia)."""

import httpx
import orjson

from forecast.polls import POLL_TYPES

# The endpoint behind votehub.com/polls; the public api.votehub.com/polls stopped updating in 2026.
VOTEHUB_URL = "https://polling.votehub.com/site/polls"
PAGE_SIZE = 500
WIKIPEDIA_URL = "https://en.wikipedia.org/api/rest_v1/page/html/{title}"
# Wikipedia asks API clients to identify themselves with a contact URL.
USER_AGENT = "midterm-forecast/0.1 (https://github.com/christian-johnson/election-forecast)"

WIKIPEDIA_PAGES = {
    "house": "2026_United_States_House_of_Representatives_elections",
    "senate": "2026_United_States_Senate_elections",
    "governor": "2026_United_States_gubernatorial_elections",
}


def _get(url: str, params: dict | None = None) -> bytes:
    response = httpx.get(
        url,
        params=params,
        headers={"User-Agent": USER_AGENT},
        timeout=120,
        follow_redirects=True,
    )
    response.raise_for_status()
    return response.content


def fetch_polls() -> list[dict]:
    """Download every VoteHub poll of the types in polls.POLL_TYPES, a page at a time.

    Returns:
        Raw poll records as returned by VoteHub.

    Raises:
        httpx.HTTPError: If a request fails.
    """
    records = []
    for poll_type in POLL_TYPES:
        offset, total = 0, 1
        while offset < total:
            params = {
                "poll_type": poll_type,
                "in_averages_only": "false",
                "limit": PAGE_SIZE,
                "offset": offset,
            }
            page = orjson.loads(_get(VOTEHUB_URL, params))
            if not page["items"]:
                break
            records.extend(page["items"])
            offset, total = offset + len(page["items"]), page["total"]
    return records


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
