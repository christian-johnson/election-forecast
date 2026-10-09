# 2026 midterm forecast

A polls-only Bayesian forecast for the 2026 House, Senate, and governor races. The Python
pipeline in `src/forecast/` writes `site/data/forecast.json`; `site/` is a static page served by
GitHub Pages.

## Running

```bash
uv run forecast fetch    # download polls + race tables into data/
uv run forecast run      # fit the model, write site/data/forecast.json
uv run forecast update   # both
uv run poe test
```

`.github/workflows/forecast.yml` runs `forecast update` daily, commits `data/` and the site
JSON, and deploys `site/` to GitHub Pages. Pushes to `main` redeploy the site without refitting.
To preview locally: `python -m http.server -d site`.

## Data

- **Polls**: [VoteHub](https://votehub.com/polls/) API, no key needed (`fetch.py`, `polls.py`).
- **Races, candidates, incumbents, Cook PVI**: the Wikipedia 2026 House, Senate, and
  gubernatorial election pages (`races.py`). For redistricted states the 2026-map PVI is used.
- **Past elections** (`scripts/calibrate_history.py`, run once; results pasted into
  `config.HISTORY`, and the misses behind them saved to `site/data/history.json`): 2022 and 2024 results and PVI from Wikipedia set the incumbency edge and how
  far races stray from PVI; FiveThirtyEight's
  [poll archive](https://github.com/fivethirtyeight/data/tree/master/pollster-ratings)
  (1998-2022) sets the size of national, state and race polling misses.
- **House hex map**: `scripts/build_house_hex.py`, run once (needs the dev dependencies).

## Model

Everything is on the logit scale of the Democratic-side two-party share (`model.py`):

- **National environment**: a weekly random walk, measured directly by generic ballot polls.
- **Race lean**: PVI baseline + incumbency + the race's own effect (candidate quality). The
  incumbency edge and the spread of race effects are fixed from past elections, not fitted.
- **Poll biases**: pollster house effects shared across all poll types, extra noise per poll type,
  and a miss shared by all polls of one state and of one race (sized from past elections).
- **Election day** (`simulate.py`): a national polling miss the polls cannot reveal, sized from
  past elections.

Clicking a race on the site opens its forecast built up in four steps (`breakdown.py`,
`site/data/steps.json`, with the weekly estimates and polls in `site/data/timeline.json`): the
national mood from polls, plus the national polling miss, the race before its polls, and after.
The last step is the published forecast.

The About page (`site/about.html`, `site/walkthrough.js`) builds the model up as you scroll
(`likelihood.py`, `site/data/likelihood.json`): a pinned one-line equation highlights each term
while a pinned chart shows the data behind it, ending with a replay of the fit's draws across the
most-polled races and the House.

Races are decided by rule when one side has no candidate (`matchups.assign_sides`). A polled
independent who outpolls a party's strongest candidate takes that party's side (e.g. Nebraska
Senate); races with several candidates per party are modeled on party vote share.

## Adding a data source

Each data source is a `Component` with an `observe(latent)` method that adds its likelihood on
top of the shared `Latent` parameters. To add one (approval, ...):

1. Parse the data into a tidy table.
2. Write a component class in `model.py` (it may sample its own parameters inside `observe`).
3. Append it to `components` in `build_inputs`.

The rest of the pipeline (simulation, export, site) needs no changes.
