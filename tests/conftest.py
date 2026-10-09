"""Synthetic Wikipedia pages and VoteHub polls standing in for the live sources."""

import pytest


def _table(groups: list[tuple[str, int]], columns: list[str], rows: list[list[str]]) -> str:
    """HTML table with a grouped two-row header, like Wikipedia's election tables."""
    top = "".join(f'<th colspan="{span}">{name}</th>' for name, span in groups)
    sub = "".join(f"<th>{c}</th>" for c in columns)
    body = "".join("<tr>" + "".join(f"<td>{v}</td>" for v in row) + "</tr>" for row in rows)
    return f"<table><tr>{top}</tr><tr>{sub}</tr>{body}</table>"


def _flat_table(columns: list[str], rows: list[list[str]]) -> str:
    head = "".join(f"<th>{c}</th>" for c in columns)
    body = "".join("<tr>" + "".join(f"<td>{v}</td>" for v in row) + "</tr>" for row in rows)
    return f"<table><tr>{head}</tr>{body}</table>"


HOUSE_HTML = (
    "<html><body>"
    # Redistricting summary listing the old and new PVI; the 2026 value must win.
    + _table(
        [("District", 1), ("Incumbent", 2)],
        ["Location", "Original 2025 PVI", "New 2026 PVI"],
        [["Ohio 1", "R+2", "D+3"]],
    )
    + _table(
        [("District", 2), ("Incumbent", 4), ("Candidates[9]", 1)],
        ["Location", "2026 PVI[9]", "Member", "Party", "First elected", "Status", "Candidates[9]"],
        [
            [
                "Ohio 1",
                "R+2",
                "Greg Landsman",
                "Democratic",
                "2022",
                "Incumbent running",
                "▌Greg Landsman (Democratic)[3] ▌Pat Roe (Republican)",
            ],
            [
                "Ohio 2",
                "R+20",
                "Dave Taylor",
                "Republican",
                "2024",
                "Incumbent running",
                "▌Ann Smith (Democratic) ▌Dave Taylor (Republican)",
            ],
            [
                "Ohio 3",
                "D+30",
                "Joyce Beatty",
                "Democratic",
                "2012",
                "Incumbent running",
                "▌Joyce Beatty (Democratic)",
            ],
        ],
    )
    + _table(
        [("District", 2), ("Incumbent", 4), ("Candidates", 1)],
        ["Location", "2026 PVI", "Member", "Party", "First elected", "Status", "Candidates"],
        [
            [
                "Alaska\xa0at-large",
                "R+6",
                "Nick Begich III",
                "Republican",
                "2024",
                "Incumbent running",
                "▌Nick Begich III (Republican) ▌Bill Hill (Independent) ▌Eric Hafner (Democratic)",
            ],
            [
                "Minnesota 2",
                "D+1",
                "Angie Craig",
                "DFL",
                "2018",
                "Incumbent retiring",
                "▌Kim Lee (DFL) ▌Tyler Kistner (Republican)",
            ],
        ],
    )
    + "</body></html>"
)

SENATE_HTML = (
    "<html><body>"
    + _table(
        [("Constituency", 2), ("Incumbent", 3), ("Candidates", 1)],
        ["State", "PVI[27]", "Senator", "Party", "Electoral history", "Candidates"],
        [
            [
                "Michigan",
                "EVEN",
                "Gary Peters",
                "Democratic",
                "2014",
                "▌Abdul El-Sayed (Democratic)[1] ▌Mike Rogers (Republican)[2]",
            ],
            [
                "Nebraska",
                "R+10",
                "Pete Ricketts",
                "Republican",
                "2023",
                "▌Dan Osborn (Independent) ▌Pete Ricketts (Republican)",
            ],
        ],
    )
    + "</body></html>"
)

GOVERNOR_HTML = (
    "<html><body>"
    + _table(
        [("Constituency", 2), ("Incumbent", 1)],
        ["State", "PVI[7]", "Governor"],
        [["Iowa", "R+6", "Kim Reynolds"]],
    )
    + _flat_table(
        ["State", "Governor", "Party", "Status", "Candidates"],
        [
            [
                "Iowa",
                "Kim Reynolds",
                "Republican",
                "Term-limited",
                "▌Rob Sand (Democratic) ▌Zach Lahn (Republican)",
            ]
        ],
    )
    + "</body></html>"
)


def _poll(poll_id, poll_type, subject, answers, *, end="2026-09-15", **extra):
    return {
        "id": poll_id,
        "poll_type": poll_type,
        "subject": subject,
        "seat_name": extra.get("seat"),
        "pollster": extra.get("pollster", "Acme Polling"),
        "start_date": "2026-09-10",
        "end_date": end,
        "sample_size": 1000,
        "population": extra.get("population", "lv"),
        "partisan": None,
        "url": f"https://example.com/polls/{poll_id}.pdf",
        "answers": [{"choice": c, "pct": p} for c, p in answers],
    }


def _repeat(n, *args, **kwargs):
    return [
        _poll(f"{args[0]}{i}", *args[1:], end=f"2026-09-{11 + i:02d}", **kwargs) for i in range(n)
    ]


@pytest.fixture
def raw_polls() -> list[dict]:
    """A VoteHub-shaped feed: generic ballot, general election, primary and hypothetical polls."""
    pollsters = ["Acme Polling", "Beta Research", "Gamma Group"]
    generic = [
        _poll(
            f"gen{i}",
            "generic-ballot",
            "2026",
            [("Dem", 51), ("Rep", 45)],
            pollster=p,
            end=f"2026-08-{11 + i:02d}",
        )
        for i, p in enumerate(pollsters * 6)
    ]
    return [
        *generic,
        *_repeat(8, "oh1-", "us-representative", "2026 OH-01", [
            ("Greg Landsman", 60), ("Pat Roe", 38)], seat="OH-01"),
        *_repeat(6, "ak-", "us-representative", "2026 AK-01", [
            ("Nick Begich", 52), ("Bill Hill", 40), ("Eric Hafner", 5)], seat="AK-01"),
        *_repeat(8, "mi-", "us-senator", "2026 Michigan", [
            ("Abdul El-Sayed", 49), ("Mike Rogers", 47)]),
        *_repeat(4, "ne-", "us-senator", "2026 Nebraska", [
            ("Dan Osborn", 46), ("Pete Ricketts", 48)]),
        *_repeat(8, "ia-", "governor", "2026 Iowa", [("Rob Sand", 58), ("Zach Lahn", 38)]),
        # Primary and hypothetical polls must not be used.
        _poll("prim", "us-senator", "2026 Michigan Democratic", [
            ("Abdul El-Sayed", 30), ("Haley Stevens", 40)]),
        _poll("hypo", "us-senator", "2026 Michigan", [
            ("Haley Stevens", 50), ("Mike Rogers", 20)]),
        # Too old to count.
        _poll("old", "generic-ballot", "2024", [("Dem", 40), ("Rep", 60)], end="2024-10-01"),
    ]  # fmt: skip


@pytest.fixture
def wiki_pages() -> dict[str, str]:
    """Synthetic Wikipedia HTML for each office."""
    return {"house": HOUSE_HTML, "senate": SENATE_HTML, "governor": GOVERNOR_HTML}
