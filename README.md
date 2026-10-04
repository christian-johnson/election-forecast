# 2026 midterm forecast

A polls-only Bayesian forecast for the 2026 House, Senate, and governor races. The Python
pipeline in `src/forecast/` writes `site/data/forecast.json`; `site/` is a static page served by
GitHub Pages.

## Running

```bash
uv run forecast fetch    # download polls + race tables into data/
uv run forecast run      # fit the model, write site/data/forecast.json
uv run forecast update   # both
uv run forecast demographics  # Census group mix per state/district (rarely needed)
uv run forecast demo     # refit the About page's made-up election (after model changes)
uv run poe test
```

`.github/workflows/forecast.yml` runs `forecast update` daily, commits `data/` and the site
JSON, and deploys `site/` to GitHub Pages. Pushes to `main` redeploy the site without refitting.
To preview locally: `python -m http.server -d site`.

## Data

- **Polls**: [VoteHub](https://votehub.com/polls/) API, no key needed (`fetch.py`, `polls.py`).
- **Races, candidates, incumbents, Cook PVI**: the Wikipedia 2026 House, Senate, and
  gubernatorial election pages (`races.py`). For redistricted states the 2026-map PVI is used.
- **Group composition**: Census ACS 2024 1-year tables, no key needed (`demographics.py`), saved
  to `data/composition.csv`. Race uses adult citizens; districts redrawn for 2026 use their
  state's mix because the ACS still has the old lines.
- **2024 group vote**: Pew Research Center validated voters (`demographics.GROUPS`).
- **Crosstabs**: `data/crosstabs.csv`, kept by hand from pollsters' national generic ballot
  crosstabs. One row per poll and group:

  | column | meaning |
  |---|---|
  | `pollster`, `start_date`, `end_date` | as in the topline poll |
  | `population` | `lv`, `rv`, `v` or `a` |
  | `partisan` | `D`, `R`, or blank for nonpartisan sponsors |
  | `dimension`, `group` | one of `race`: white/black/hispanic/asian/other, `education`: college/noncollege, `age`: 18-29/30-44/45-64/65+ |
  | `dem`, `rep` | percent for each party within the group |
  | `n` | respondents in the group (groups under 30 are skipped) |
- **House hex map**: `scripts/build_house_hex.py`, run once (needs the dev dependencies).

## Model

Everything is on the logit scale of the Democratic-side two-party share (`model.py`):

- **National environment**: a weekly random walk, measured directly by generic ballot polls.
- **Race lean**: PVI baseline + office offset + incumbency + a partially pooled race effect
  (candidate quality). Unpolled races fall back to the pooled prior.
- **Group swing**: each demographic group's swing since 2024, relative to the national shift.
  Crosstabs measure it directly; each race moves by its group mix minus the nation's, averaged
  over the race, education and age dimensions.
- **Poll biases**: pollster house effects shared across all poll types, population (LV/RV/A),
  and partisan sponsorship, plus extra noise per office.
- **Election day** (`simulate.py`): national, state, demographic-group, and race-level polling
  error the polls cannot reveal (scales in `config.py`).

Clicking a race on the site opens its forecast built up step by step (`breakdown.py`,
`site/data/steps.json`). The election-day logit is a sum of terms, added one at a time: the
national mood from polls, a shared polling miss, the PVI baseline, office and incumbency,
demographics, the race's own effect, and local (state, group, race) misses. The last step is the
published forecast.

The About page (`site/about.html`) shows the model fit to a made-up election with a known
answer (`demo.py`), with and without an industry-wide polling miss.

Races are decided by rule when one side has no candidate (`matchups.assign_sides`). A polled
independent who outpolls a party's strongest candidate takes that party's side (e.g. Nebraska
Senate); races with several candidates per party are modeled on party vote share.

## Adding a data source

Each data source is a `Component` with an `observe(latent)` method that adds its likelihood on
top of the shared `Latent` parameters. To add one (crosstabs, approval, ...):

1. Parse the data into a tidy table.
2. Write a component class in `model.py` (it may sample its own parameters inside `observe`).
3. Append it to `components` in `build_inputs`.

The rest of the pipeline (simulation, export, site) needs no changes.
