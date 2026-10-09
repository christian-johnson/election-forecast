"use strict";

// The About page's walk through the model: the one-line posterior above a chart that steps
// through its pieces with the real data, from the national polls to the full fit.

const term = (name, tex) => String.raw`\htmlClass{term term-${name}}{${tex}}`;
const EQUATION = [
  String.raw`p(\theta \mid y) \propto \prod_i \mathcal{N}\big(`,
  term("y", "y_i"),
  String.raw`\,\big|\,`,
  term("eta", String.raw`\eta_{t_i}`),
  "+",
  term("lambda", String.raw`\lambda_{r_i}`),
  "+",
  term("delta", String.raw`\delta_{r_i}`),
  "+",
  term("h", "h_{p_i}"),
  String.raw`,\ `,
  term("v", "v_i"),
  "+",
  term("tau", String.raw`\tau_{o_i}^2`),
  String.raw`\big)\,`,
  term("prior", String.raw`p(\theta)`),
].join(" ");
const ELECTION = [
  term("m", "M_r"),
  "=",
  term("eta", String.raw`\eta_E`),
  "+",
  term("lambda", String.raw`\lambda_r`),
  "+",
  term("eps", String.raw`\varepsilon`),
].join(" ");


// The model's prior on the national mood when polling starts (0.2 logit), in margin points.
const START_SD = 10;
// Pollsters with at least this many polls are named as examples of house effects.
const REGULAR_POLLS = 10;
const OFFICE_LABEL = { house: "House", senate: "Senate", governor: "governor" };
const SHORT_OFFICE = { house: "", senate: "Sen.", governor: "Gov." };
const PARTY_ADJ = { D: "Democratic", R: "Republican" };

// How long each animation plays, in milliseconds; the finale's replay of draws takes longer.
const ANIMATION_MS = 700;
const DURATION = {
  mood: ANIMATION_MS * 1.3,
  house: ANIMATION_MS,
  widen: ANIMATION_MS,
  prior: ANIMATION_MS,
  "race-polls": ANIMATION_MS,
  finale: 10000,
};
const reducedMotion = window.matchMedia("(prefers-reduced-motion: reduce)");

const walk = {
  data: null,
  history: null,
  prep: null,
  steps: [],
  step: 0,
  width: 0,
  narrow: false,
  timer: null,
};
const scenes = {};

const day = (s) => new Date(`${s}T12:00`);
const clamp01 = (v) => Math.min(1, Math.max(0, v));
const tickMargin = (v) => (v === 0 ? "Even" : fmt(v).replace(".0", ""));
const partyColor = (m) => css(m >= 0 ? "--dem" : "--rep");
const pviText = (pvi) => (pvi === 0 ? "EVEN" : pvi > 0 ? `D+${pvi}` : `R+${-pvi}`);
const raceName = (r) => (r.office === "house" ? r.label : `${r.label} ${OFFICE_LABEL[r.office]}`);
const fade = (sel, opacity) =>
  sel.interrupt().transition().duration(400).attr("opacity", +opacity);

/** Week-by-week estimate from only the polls up to each week, as a running (Kalman) filter. */
function runningEstimate(polls, weeks, step, extra) {
  const byWeek = d3.group(polls, (p) => p.week);
  let mean = 0;
  let variance = START_SD ** 2;
  return weeks.map((t, i) => {
    if (i > 0) variance += step ** 2;
    for (const p of byWeek.get(i) ?? []) {
      const gain = variance / (variance + p.sd ** 2 + extra ** 2);
      mean += gain * (p.margin - mean);
      variance *= 1 - gain;
    }
    const half = Z80 * Math.sqrt(variance);
    return { t, mid: mean, lo: mean - half, hi: mean + half };
  });
}

function prepare(data) {
  const start = day(data.start);
  const electionDay = day(data.election_date);
  const nWeeks = data.national.weekly.mean.length;
  // The last week runs up to election day.
  const weeks = d3.range(nWeeks).map((i) =>
    i === nWeeks - 1 ? electionDay : new Date(+start + i * WEEK_MS),
  );
  const weekOf = (t) => Math.min(nWeeks - 1, Math.max(0, Math.floor((t - start) / WEEK_MS)));
  const house = Object.fromEntries(data.pollsters.map((p) => [p.name, p.mean]));
  const polls = (list) =>
    list.map((p) => {
      const t = day(p.date);
      const share = p.dem / (p.dem + p.rep);
      return {
        ...p,
        t,
        week: weekOf(t),
        house: house[p.pollster] ?? 0,
        // Sampling SD of the two-party margin, in points.
        sd: 200 * Math.sqrt((share * (1 - share)) / p.n_two_party),
        size: Math.min(p.n ?? ASSUMED_N, MAX_N),
      };
    });
  const series = ({ mean, sd }) =>
    mean.map((v, i) => ({
      t: weeks[i],
      mid: logitMargin(v),
      lo: logitMargin(v - Z80 * sd[i]),
      hi: logitMargin(v + Z80 * sd[i]),
    }));
  const nat = data.national.weekly;
  const nationalPolls = polls(data.national.polls);
  const races = data.races.map((r) => ({
    ...r,
    polls: polls(r.polls),
    fit: series(r.weekly),
    prior: series({
      mean: nat.mean.map((v) => v + r.prior[0]),
      sd: nat.sd.map((v) => Math.hypot(v, r.prior[1])),
    }),
  }));
  return {
    start,
    electionDay,
    today: day(data.as_of),
    weeks,
    nationalPolls,
    national: series(nat),
    // The national mood with the national polling miss added on top of the fit's uncertainty.
    nationalWide: series({
      mean: nat.mean,
      sd: nat.sd.map((v) => Math.hypot(v, data.history.national_miss * POINT)),
    }),
    running: runningEstimate(nationalPolls, weeks, data.walk, data.noise.generic),
    race: races[0],
    races,
    // For the finale each draw's whole line carries its national polling miss, so the lines show
    // the mood's uncertainty and the miss together and end at the simulated result.
    draws: data.draws.nat.map((path, k) => {
      const shifted = path.map((v) => v + data.draws.miss[k]);
      return {
        nat: shifted.map((v) => logitMargin(v / 1000)),
        races: data.draws.lean[k].map((lean) => shifted.map((v) => logitMargin((v + lean) / 1000))),
        seats: data.draws.house_d[k],
      };
    }),
  };
}

/** Share of polls that miss the fitted mood by more than their own 80% range. */
function shareOutside(prep, shift, extra) {
  const { nationalPolls: polls, national } = prep;
  const out = polls.filter(
    (p) => Math.abs(p.margin - shift(p) - national[p.week].mid) > Z80 * Math.hypot(p.sd, extra),
  );
  return pct(out.length / polls.length);
}

const shareWithin = (values, half) => values.filter((v) => Math.abs(v) <= half).length / values.length;

function fills(data, prep, history) {
  const h = data.history;
  const race = prep.race;
  const regulars = data.pollsters
    .filter((p) => p.n_polls >= REGULAR_POLLS)
    .sort((a, b) => b.mean - a.mean);
  const [mostD, mostR] = [regulars[0], regulars.at(-1)];
  return {
    "n-generic": prep.nationalPolls.length,
    "nat-day-lo": fmt(prep.nationalWide.at(-1).lo),
    "nat-day-hi": fmt(prep.nationalWide.at(-1).hi),
    walk: data.walk.toFixed(2),
    "outside-raw": shareOutside(prep, () => 0, 0),
    "outside-adj": shareOutside(prep, (p) => p.house, data.noise.generic),
    "most-d": mostD.name,
    "most-d-lean": fmt(mostD.mean),
    "most-r": mostR.name,
    "most-r-lean": fmt(mostR.mean),
    "tau-generic": data.noise.generic.toFixed(1),
    "largest-year": h.largest_national_miss_year,
    "largest-side": h.largest_national_miss > 0 ? "Democrats" : "Republicans",
    "largest-miss": Math.abs(h.largest_national_miss).toFixed(1),
    "miss-half": (Z80 * h.national_miss).toFixed(1),
    "miss-inside": (() => {
      const misses = Object.values(history.national_misses);
      return `${misses.filter((m) => Math.abs(m) <= Z80 * h.national_miss).length} of ${misses.length}`;
    })(),
    incumbency: h.incumbency.toFixed(1),
    ...Object.fromEntries(
      Object.entries(h.race_spread).map(([o, s]) => [`spread-${o}`, (Z80 * s).toFixed(1)]),
    ),
    "race-name": raceName(race),
    "race-pvi": pviText(race.pvi),
    "race-incumbent": race.incumbent_side
      ? `, plus the ${PARTY_ADJ[race.incumbent_side]} incumbent's edge,`
      : "",
    "race-n-polls": race.polls.length,
    "local-n": history.local_misses.length,
    "local-half": (Z80 * Math.hypot(h.state_miss, h.race_miss)).toFixed(1),
    "race-direction": race.prior[0].mid < prep.national[0].mid ? "down" : "up",
    "n-polls": (prep.nationalPolls.length + data.n_race_polls).toLocaleString(),
    "shown-draws": prep.draws.length,
    "n-draws": data.n_draws.toLocaleString(),
    "shown-races": prep.races.length,
    "n-races": data.n_races,
    "p-house": pct(data.house.p_d),
  };
}

/** Where a tracing step's cursor is: most of it walks up to today, the rest to election day. */
function cursorAt(p) {
  const { start, today, electionDay } = walk.prep;
  const toToday = clamp01(p / 0.75);
  const toElection = clamp01((p - 0.8) / 0.2);
  return toElection > 0
    ? new Date(+today + toElection * (electionDay - today))
    : new Date(+start + toToday * (today - start));
}

const areaPath = (x, y) =>
  d3
    .area()
    .curve(d3.curveBasis)
    .x((d) => x(d.t))
    .y0((d) => y(d.lo))
    .y1((d) => y(d.hi));
const linePath = (x, y) =>
  d3
    .line()
    .curve(d3.curveBasis)
    .x((d) => x(d.t))
    .y((d) => y(d.mid));

/** A line and its 80% band. Drawn for looking only, so clicks reach the polls underneath. */
function estimate(g, series, x, y, { dashed = false } = {}) {
  g.attr("pointer-events", "none");
  g.append("path").attr("fill", css("--band")).attr("d", areaPath(x, y)(series));
  g.append("path")
    .attr("fill", "none")
    .attr("stroke", css("--ink"))
    .attr("stroke-width", 2)
    .attr("stroke-dasharray", dashed ? "5 4" : null)
    .attr("d", linePath(x, y)(series));
  return g;
}

/** Axes, gridlines and today's line for a chart of margins (D up) over time. */
function timeChart(el, values) {
  el.innerHTML = "";
  const width = el.clientWidth;
  const height = el.clientHeight;
  const m = { top: 24, right: 16, bottom: 26, left: 48 };
  const { start, electionDay, today } = walk.prep;
  const x = d3.scaleTime().domain([start, electionDay]).range([m.left, width - m.right]);
  const [lo, hi] = d3.extent([...values, 0]);
  const y = d3
    .scaleLinear()
    .domain([lo - 1, hi + 1])
    .nice()
    .range([height - m.bottom, m.top]);
  const svg = d3.select(el).append("svg").attr("width", width).attr("height", height);
  svg
    .append("g")
    .attr("transform", `translate(${m.left},0)`)
    .call(
      d3
        .axisLeft(y)
        .ticks(5)
        .tickFormat(tickMargin)
        .tickSize(-(width - m.left - m.right)),
    )
    .call((a) => a.select(".domain").remove())
    .call((a) => a.selectAll(".tick line").attr("stroke", (v) => css(v === 0 ? "--axis" : "--grid")))
    .call((a) => a.selectAll(".tick text").attr("fill", css("--ink-2")));
  svg
    .append("g")
    .attr("transform", `translate(0,${height - m.bottom})`)
    .call(
      d3
        .axisBottom(x)
        .tickValues([
          ...x.ticks(width < 500 ? 4 : 7).filter((t) => x(electionDay) - x(t) > 40),
          electionDay,
        ])
        .tickFormat((t) => (+t === +electionDay ? "Nov 3" : x.tickFormat()(t)))
        .tickSizeOuter(0),
    )
    .call(styleAxis)
    .call((a) => a.select(".tick:last-of-type text").attr("font-weight", 700));
  const clip = `clip-${el.dataset.scene}`;
  svg
    .append("clipPath")
    .attr("id", clip)
    .append("rect")
    .attr("x", m.left)
    .attr("y", m.top)
    .attr("width", width - m.left - m.right)
    .attr("height", height - m.top - m.bottom);
  // Tracing steps reveal an estimate from the left by widening this.
  const trace = svg
    .append("clipPath")
    .attr("id", `${clip}-trace`)
    .append("rect")
    .attr("height", height)
    .attr("width", 0);
  const plot = svg.append("g").attr("clip-path", `url(#${clip})`);
  const overlay = svg.append("g").attr("pointer-events", "none");
  overlay
    .append("line")
    .attr("x1", x(today))
    .attr("x2", x(today))
    .attr("y1", m.top)
    .attr("y2", height - m.bottom)
    .attr("stroke", css("--ink-2"))
    .attr("stroke-dasharray", "3 3");
  overlay
    .append("text")
    .attr("x", x(today))
    .attr("y", m.top - 8)
    .attr("text-anchor", "middle")
    .attr("font-size", 11)
    .attr("fill", css("--ink-2"))
    .text("Today");
  return { svg, plot, overlay, trace, clip: `${clip}-trace`, x, y, m, width, height };
}

/** Bubbles of polls (sized by sample, older to the left), each with a hidden whisker. */
function pollBubbles(g, polls, x, maxR) {
  const r = d3.scaleSqrt().domain([0, MAX_N]).range([0, maxR]).clamp(true);
  const a = g
    .selectAll("a")
    .data(polls.toSorted((p, q) => q.size - p.size))
    .join("a")
    .attr("class", "bubble")
    .attr("href", (p) => safeUrl(p.url))
    .attr("target", "_blank")
    .attr("rel", "noopener")
    .on("mousemove", (e, p) => showTooltip(e, pollTooltip(p)))
    .on("mouseleave", hideTooltip);
  a.append("line")
    .attr("x1", (p) => x(p.t))
    .attr("x2", (p) => x(p.t))
    .attr("stroke", (p) => partyColor(p.margin))
    .attr("stroke-opacity", 0.2)
    .attr("opacity", 0);
  a.append("circle")
    .attr("cx", (p) => x(p.t))
    .attr("r", (p) => Math.max(1.5, r(p.size)))
    .attr("fill", (p) => partyColor(p.margin))
    .attr("fill-opacity", 0.25)
    .attr("stroke", (p) => partyColor(p.margin));
  return a;
}

/** Move polls down by `shift`, with whiskers `half` points either side. */
function placePolls(sel, y, shift, half) {
  sel.select("circle").attr("cy", (p) => y(p.margin - shift(p)));
  sel
    .select("line")
    .attr("y1", (p) => y(p.margin - shift(p) - half(p)))
    .attr("y2", (p) => y(p.margin - shift(p) + half(p)));
}

/** Key for the line and band, in a chart's top left corner. */
function bandLegend(g, x, y, label = "Estimate") {
  const key = g.append("g").attr("transform", `translate(${x},${y})`);
  key
    .append("line")
    .attr("x2", 18)
    .attr("y1", 6)
    .attr("y2", 6)
    .attr("stroke", css("--ink"))
    .attr("stroke-width", 2);
  key.append("rect").attr("x", 0).attr("y", 16).attr("width", 18).attr("height", 10).attr("fill", css("--band"));
  key
    .selectAll("text")
    .data([
      [label, 10],
      ["80% range", 25],
    ])
    .join("text")
    .attr("x", 26)
    .attr("y", (d) => d[1])
    .attr("font-size", 11)
    .attr("fill", css("--ink-2"))
    .text((d) => d[0]);
}

/** A band partway (p from 0 to 1) from one series' 80% range to another's. */
const between = (from, to, p) =>
  from.map((d, i) => ({ t: d.t, lo: d.lo + p * (to[i].lo - d.lo), hi: d.hi + p * (to[i].hi - d.hi) }));

scenes.national = {
  render(el) {
    const { nationalPolls, national, running } = walk.prep;
    const values = nationalPolls.map((p) => p.margin);
    const c = timeChart(el, [d3.quantile(values, 0.01), d3.quantile(values, 0.99)]);
    const polls = pollBubbles(c.plot.append("g"), nationalPolls, c.x, walk.narrow ? 4 : 5);
    const traced = estimate(c.plot.append("g").attr("clip-path", `url(#${c.clip})`), running, c.x, c.y);
    // Behind the fit's own band: the range once the national miss is added.
    const wide = c.plot
      .append("path")
      .attr("pointer-events", "none")
      .attr("fill", css("--band"))
      .attr("opacity", 0);
    const fit = estimate(c.plot.append("g").attr("opacity", 0), national, c.x, c.y);
    bandLegend(c.overlay, c.m.left + 8, c.m.top + 4);
    Object.assign(this, { c, polls, traced, wide, fit });
  },
  update(step, p, entered) {
    const { c, polls, traced, wide, fit } = this;
    const { national, nationalWide } = walk.prep;
    const tracing = step === "mood";
    const widening = step === "widen";
    const whiskers = step === "sampling" || step === "house";
    if (entered) {
      fade(traced, tracing);
      fade(fit, whiskers || widening);
      fade(wide, widening);
      fade(polls.selectAll("line"), whiskers);
      if (!tracing) fade(polls, 1);
    }
    if (tracing) {
      const t = cursorAt(p);
      c.trace.attr("width", c.x(t));
      polls.interrupt().attr("opacity", (d) => (d.t <= t ? 1 : 0.12));
    }
    if (widening) wide.attr("d", areaPath(c.x, c.y)(between(national, nationalWide, p)));
    // Polls move by their house effects in that step, and stay moved afterwards.
    const move = step === "house" ? p : widening ? 1 : 0;
    const extra = move * walk.data.noise.generic;
    placePolls(
      polls,
      c.y,
      (d) => move * d.house,
      (d) => Z80 * Math.hypot(d.sd, extra),
    );
  },
};

/** A label on a pill that dims the lines behind it; place(y) sets it just above a line at y. */
function lineLabel(g, x, text, color) {
  const label = g.append("g").attr("pointer-events", "none");
  const pill = label
    .append("rect")
    .attr("rx", 9)
    .attr("height", 18)
    .attr("fill", css("--page"))
    .attr("fill-opacity", 0.85);
  const content = label
    .append("text")
    .attr("x", 8)
    .attr("y", 13)
    .attr("font-size", 12)
    .attr("fill", color)
    .text(text);
  pill.attr("width", content.node().getComputedTextLength() + 16);
  label.place = (y) => label.attr("transform", `translate(${x},${y - 22})`);
  return label;
}

scenes.race = {
  render(el) {
    const { race, national, nationalPolls } = walk.prep;
    const nationalValues = nationalPolls.map((p) => p.margin);
    const values = [
      d3.quantile(nationalValues, 0.01),
      d3.quantile(nationalValues, 0.99),
      ...race.polls.map((p) => p.margin),
      ...race.prior.flatMap((d) => [d.lo, d.hi]),
      ...race.fit.flatMap((d) => [d.lo, d.hi]),
      ...national.map((d) => d.mid),
    ];
    const c = timeChart(el, d3.extent(values));
    const line = linePath(c.x, c.y);
    // The national polls, house effects removed, as a dim backdrop to the shift.
    const backdrop = pollBubbles(
      c.plot.append("g").attr("pointer-events", "none"),
      nationalPolls,
      c.x,
      walk.narrow ? 4 : 5,
    ).attr("opacity", 0);
    backdrop.selectAll("line").attr("opacity", 1);
    placePolls(
      backdrop,
      c.y,
      (d) => d.house,
      (d) => Z80 * Math.hypot(d.sd, walk.data.noise.generic),
    );
    c.plot
      .append("path")
      .attr("pointer-events", "none")
      .attr("fill", "none")
      .attr("stroke", css("--muted"))
      .attr("stroke-width", 1.5)
      .attr("d", line(national));
    const priorBand = c.plot
      .append("path")
      .attr("pointer-events", "none")
      .attr("fill", css("--band"))
      .attr("opacity", 0)
      .attr("d", areaPath(c.x, c.y)(race.prior));
    // The PVI guess: the national mood's line, shifted by the race's lean.
    const guess = c.plot
      .append("path")
      .attr("pointer-events", "none")
      .attr("fill", "none")
      .attr("stroke", css("--ink"))
      .attr("stroke-width", 2)
      .attr("stroke-dasharray", "5 4");
    const polls = pollBubbles(c.plot.append("g"), race.polls, c.x, walk.narrow ? 6 : 8);
    placePolls(polls, c.y, () => 0, () => 0);
    const fit = estimate(c.plot.append("g").attr("clip-path", `url(#${c.clip})`), race.fit, c.x, c.y);

    const labelX = c.m.left + 4;
    const above = (series) => c.y(series[0].mid) - 2;
    lineLabel(c.overlay, labelX, "National mood", css("--ink-2")).place(above(national));
    const guessLabel = lineLabel(
      c.overlay,
      labelX,
      `${raceName(race)} after national polls + PVI${race.incumbent_side ? " + incumbency" : ""}`,
      css("--ink"),
    );
    const fitLabel = lineLabel(
      c.overlay,
      labelX,
      `${raceName(race)} after all polls + PVI${race.incumbent_side ? " + incumbency" : ""}`,
      css("--ink"),
    )
      .attr("opacity", 0)
      .place(above(race.fit));
    Object.assign(this, { c, line, backdrop, priorBand, guess, guessLabel, polls, fit, fitLabel, above });
  },
  update(step, p, entered) {
    const { c, line, backdrop, priorBand, guess, guessLabel, polls, fit, fitLabel, above } = this;
    const { race, national } = walk.prep;
    const tracing = step === "race-polls";
    if (entered) {
      fade(backdrop, tracing ? 0 : 0.25);
      fade(priorBand, tracing ? 0.5 : 0);
      fade(fit, tracing);
      fade(fitLabel, tracing);
      guessLabel.interrupt().attr("opacity", 1);
    }
    // The guess slides from the national mood to the race; afterwards it stays put.
    const shift = tracing ? 1 : p;
    const moved = national.map((d, i) => ({ t: d.t, mid: d.mid + shift * (race.prior[i].mid - d.mid) }));
    guess.attr("d", line(moved));
    guessLabel.place(above(moved));
    const t = tracing ? cursorAt(p) : walk.prep.start;
    c.trace.attr("width", c.x(t));
    polls.attr("opacity", (d) => (tracing && d.t <= t ? 1 : 0));
  },
};

/** Histogram of past misses between `top` and `bottom`, over the 80% range it implies. */
function missHistogram(svg, values, { x, top, bottom, spread, step, label }) {
  const [lo, hi] = x.domain().toSorted((a, b) => a - b);
  const bins = d3
    .bin()
    .domain([lo, hi])
    .thresholds(d3.range(lo, hi + step, step))(values.map((v) => Math.max(lo, Math.min(hi - 1e-9, v))));
  const y = d3
    .scaleLinear()
    .domain([0, d3.max(bins, (b) => b.length)])
    .range([bottom, top + 16]);
  const half = Z80 * spread;
  svg
    .append("rect")
    .attr("x", x(half))
    .attr("width", x(-half) - x(half))
    .attr("y", top + 16)
    .attr("height", bottom - top - 16)
    .attr("fill", css("--band"));
  const bars = svg
    .append("g")
    .selectAll("rect")
    .data(bins)
    .join("rect")
    .attr("x", (b) => x(b.x1) + 0.5)
    .attr("width", (b) => Math.max(0.5, x(b.x0) - x(b.x1) - 1))
    .attr("y", (b) => y(b.length))
    .attr("height", (b) => bottom - y(b.length))
    .attr("fill", (b) => partyColor((b.x0 + b.x1) / 2));
  svg
    .append("text")
    .attr("x", x.range()[0])
    .attr("y", top + 10)
    .attr("font-size", 12)
    .attr("font-weight", 600)
    .attr("fill", css("--ink"))
    .text(`${label} · ${pct(shareWithin(values, half))} within ±${half.toFixed(1)} (shaded)`);
  return bars;
}

/** Margin axis for misses, with what each direction means at its ends. */
function missAxis(svg, x, y, [left, right]) {
  svg
    .append("g")
    .attr("transform", `translate(0,${y})`)
    .call(
      d3
        .axisBottom(x)
        .ticks(7)
        .tickFormat((v) => (v === 0 ? "0" : Math.abs(v)))
        .tickSizeOuter(0),
    )
    .call(styleAxis);
  svg
    .selectAll("text.end")
    .data([
      [x.range()[0], "start", left],
      [x.range()[1], "end", right],
    ])
    .join("text")
    .attr("x", (d) => d[0])
    .attr("y", y + 36)
    .attr("text-anchor", (d) => d[1])
    .attr("font-size", 11)
    .attr("fill", css("--ink-2"))
    .text((d) => d[2]);
}

/** Grow histogram bars up from their base to the heights kept by keepHeights. */
function growBars(bars) {
  bars
    .interrupt()
    .attr("y", function () {
      return +this.dataset.y + +this.dataset.h;
    })
    .attr("height", 0)
    .transition()
    .duration(ANIMATION_MS / 2)
    .delay((_, i, nodes) => (i / nodes.length) * (ANIMATION_MS / 2))
    .attr("y", function () {
      return this.dataset.y;
    })
    .attr("height", function () {
      return this.dataset.h;
    });
}

/** Remember each bar's drawn height, so it can grow back to it. */
function keepHeights(bars) {
  bars.each(function () {
    this.dataset.y = this.getAttribute("y");
    this.dataset.h = this.getAttribute("height");
  });
  return bars;
}

scenes.years = {
  render(el) {
    el.innerHTML = "";
    const width = el.clientWidth;
    const height = el.clientHeight;
    const m = { top: 28, right: 16, bottom: 28, left: 56 };
    const misses = Object.entries(walk.history.national_misses).map(([year, miss]) => ({
      year,
      miss,
    }));
    const half = Z80 * walk.data.history.national_miss;
    const bound = Math.max(half, d3.max(misses, (d) => Math.abs(d.miss))) * 1.15;
    const x = d3
      .scaleBand()
      .domain(misses.map((d) => d.year))
      .range([m.left, width - m.right])
      .padding(0.25);
    const y = d3
      .scaleLinear()
      .domain([-bound, bound])
      .nice()
      .range([height - m.bottom, m.top]);
    const svg = d3.select(el).append("svg").attr("width", width).attr("height", height);
    svg
      .append("rect")
      .attr("x", m.left)
      .attr("width", width - m.left - m.right)
      .attr("y", y(half))
      .attr("height", y(-half) - y(half))
      .attr("fill", css("--band"));
    svg
      .append("text")
      .attr("x", width - m.right)
      .attr("y", m.top - 12)
      .attr("text-anchor", "end")
      .attr("font-size", 11)
      .attr("fill", css("--ink-2"))
      .text(
        `Shaded: ε's 80% range, ±${half.toFixed(1)} · ` +
          `${misses.filter((d) => Math.abs(d.miss) <= half).length} of ${misses.length} elections inside`,
      );
    svg
      .append("g")
      .attr("transform", `translate(${m.left},0)`)
      .call(
        d3
          .axisLeft(y)
          .ticks(6)
          .tickFormat((v) => (v === 0 ? "0" : Math.abs(v))),
      )
      .call(styleAxis);
    svg
      .selectAll("text.side")
      .data([
        [m.top - 12, "Polls too Democratic"],
        [height - m.bottom + 22, "Polls too Republican"],
      ])
      .join("text")
      .attr("x", m.left)
      .attr("y", (d) => d[0])
      .attr("font-size", 11)
      .attr("fill", css("--ink-2"))
      .text((d) => d[1]);
    svg
      .append("line")
      .attr("x1", m.left)
      .attr("x2", width - m.right)
      .attr("y1", y(0))
      .attr("y2", y(0))
      .attr("stroke", css("--axis"));
    const bars = svg
      .append("g")
      .selectAll("rect")
      .data(misses)
      .join("rect")
      .attr("x", (d) => x(d.year))
      .attr("width", x.bandwidth())
      .attr("y", (d) => Math.min(y(0), y(d.miss)))
      .attr("height", (d) => Math.abs(y(d.miss) - y(0)))
      .attr("fill", (d) => partyColor(d.miss))
      .on("mousemove", (e, d) =>
        showTooltip(
          e,
          `<div class="title">${d.year}</div><div>Late polls overstated ${d.miss > 0 ? "Democrats" : "Republicans"} by ${Math.abs(d.miss).toFixed(1)} pts on average</div>`,
        ),
      )
      .on("mouseleave", hideTooltip);
    svg
      .append("g")
      .selectAll("text")
      .data(misses)
      .join("text")
      .attr("x", (d) => x(d.year) + x.bandwidth() / 2)
      .attr("y", y(0) + 4)
      .attr("dy", (d) => (d.miss >= 0 ? 12 : -6))
      .attr("text-anchor", "middle")
      .attr("font-size", width < 500 ? 9 : 11)
      .attr("fill", css("--ink-2"))
      .text((d) => (width < 500 ? `'${d.year.slice(2)}` : d.year));
    Object.assign(this, { bars: keepHeights(bars), zero: y(0) });
  },
  update(step, p, entered) {
    if (!entered) return;
    this.bars
      .interrupt()
      .attr("y", this.zero)
      .attr("height", 0)
      .transition()
      .duration(ANIMATION_MS / 2)
      .delay((_, i, nodes) => (i / nodes.length) * (ANIMATION_MS / 2))
      .attr("y", function () {
        return this.dataset.y;
      })
      .attr("height", function () {
        return this.dataset.h;
      });
  },
};

scenes.pvi = {
  render(el) {
    el.innerHTML = "";
    const width = el.clientWidth;
    const height = el.clientHeight;
    const m = { top: 8, right: 16, bottom: 46, left: 16 };
    const x = marginScale(-30, 30, m.left, width - m.right);
    const svg = d3.select(el).append("svg").attr("width", width).attr("height", height);
    const rowH = (height - m.top - m.bottom) / 3;
    const { race_spread: spread } = walk.data.history;
    this.bars = Object.entries(OFFICE_LABEL).flatMap(([office, label], i) => {
      const values = walk.history.pvi_misses[office];
      const top = m.top + i * rowH;
      const bars = missHistogram(svg, values, {
        x,
        top,
        bottom: top + rowH - 10,
        spread: spread[office],
        step: 2,
        label: `${label[0].toUpperCase()}${label.slice(1)}: ${values.length} results vs. their PVI guess`,
      });
      return keepHeights(bars).nodes();
    });
    missAxis(svg, x, height - m.bottom, ["Democrats beat the guess", "Republicans beat the guess"]);
  },
  update(step, p, entered) {
    if (entered) growBars(d3.selectAll(this.bars));
  },
};

scenes.local = {
  render(el) {
    el.innerHTML = "";
    const width = el.clientWidth;
    const height = el.clientHeight;
    const m = { top: 8, right: 16, bottom: 46, left: 16 };
    const x = marginScale(-20, 20, m.left, width - m.right);
    const svg = d3.select(el).append("svg").attr("width", width).attr("height", height);
    const { state_miss: stateMiss, race_miss: raceMiss } = walk.data.history;
    const values = walk.history.local_misses;
    const bars = missHistogram(svg, values, {
      x,
      top: m.top,
      bottom: height - m.bottom,
      spread: Math.hypot(stateMiss, raceMiss),
      step: 1,
      label: `Late poll average vs. result, ${values.length} races`,
    });
    this.bars = keepHeights(bars);
    missAxis(svg, x, height - m.bottom, ["Polls too Democratic", "Polls too Republican"]);
  },
  update(step, p, entered) {
    if (entered) growBars(this.bars);
  },
};

/** One small panel of the finale: polls in SVG, with the draws painted on a canvas above. */
function finalePanel(div, { title, polls, paths }) {
  const width = div.clientWidth;
  const height = div.clientHeight;
  const m = { top: 20, right: 6, bottom: 6, left: 34 };
  const { start, electionDay } = walk.prep;
  const x = d3.scaleTime().domain([start, electionDay]).range([m.left, width - m.right]);
  // Most of the draws, not every outlier, set the scale.
  const drawn = paths.flat().sort(d3.ascending);
  const values = [
    ...polls.map((p) => p.margin),
    d3.quantileSorted(drawn, 0.01),
    d3.quantileSorted(drawn, 0.99),
    0,
  ];
  const y = d3
    .scaleLinear()
    .domain(d3.extent(values))
    .nice()
    .range([height - m.bottom, m.top]);
  const svg = d3.select(div).append("svg").attr("width", width).attr("height", height);
  svg
    .append("g")
    .attr("transform", `translate(${m.left},0)`)
    .call(
      d3
        .axisLeft(y)
        .ticks(3)
        .tickFormat(tickMargin)
        .tickSize(-(width - m.left - m.right)),
    )
    .call((a) => a.select(".domain").remove())
    .call((a) => a.selectAll(".tick line").attr("stroke", (v) => css(v === 0 ? "--axis" : "--grid")))
    .call((a) => a.selectAll(".tick text").attr("fill", css("--ink-2")).attr("font-size", 9));
  svg
    .append("line")
    .attr("x1", x(walk.prep.today))
    .attr("x2", x(walk.prep.today))
    .attr("y1", m.top)
    .attr("y2", height - m.bottom)
    .attr("stroke", css("--ink-2"))
    .attr("stroke-dasharray", "2 3");
  svg
    .append("text")
    .attr("x", m.left)
    .attr("y", 12)
    .attr("font-size", 11)
    .attr("font-weight", 600)
    .attr("fill", css("--ink"))
    .text(title);
  svg
    .append("g")
    .selectAll("circle")
    .data(polls)
    .join("circle")
    .attr("cx", (p) => x(p.t))
    .attr("cy", (p) => y(p.margin))
    .attr("r", 1.5)
    .attr("fill", (p) => partyColor(p.margin))
    .attr("fill-opacity", 0.5);
  const dpr = window.devicePixelRatio || 1;
  const canvas = d3
    .select(div)
    .append("canvas")
    .attr("width", width * dpr)
    .attr("height", height * dpr)
    .style("width", `${width}px`)
    .style("height", `${height}px`)
    .node();
  const ctx = canvas.getContext("2d");
  ctx.scale(dpr, dpr);
  return { ctx, x, y, m, width, height, paths };
}

/** Paint the first k draws faintly, and the latest one boldly. */
function paintDraws(panel, k, colors) {
  const { ctx, x, y, m, width, height, paths } = panel;
  const xs = walk.prep.weeks.map((t) => x(t));
  const stroke = (path) => {
    ctx.beginPath();
    path.forEach((v, i) => (i ? ctx.lineTo(xs[i], y(v)) : ctx.moveTo(xs[i], y(v))));
    ctx.stroke();
  };
  ctx.clearRect(0, 0, width, height);
  ctx.save();
  ctx.beginPath();
  ctx.rect(m.left, m.top, width - m.left - m.right, height - m.top - m.bottom);
  ctx.clip();
  ctx.strokeStyle = colors.ink;
  ctx.globalAlpha = 0.07;
  ctx.lineWidth = 1;
  for (let j = 0; j < k - 1; j++) stroke(paths[j]);
  if (k) {
    ctx.globalAlpha = 1;
    ctx.strokeStyle = colors.draw;
    ctx.lineWidth = 1.5;
    stroke(paths[k - 1]);
  }
  ctx.restore();
}

scenes.finale = {
  render(el) {
    const { races, nationalPolls, draws } = walk.prep;
    el.innerHTML = `<div class="finale">
      <div class="finale-top"><div class="panel"></div><div class="panel"></div></div>
      <div class="finale-races"></div>
    </div>`;
    // About three race panels fit across; the rest trail off to the right.
    el.querySelector(".finale-races").style.setProperty("--panel-w", `${el.clientWidth / 3.3}px`);
    const raceDivs = races.map(() => {
      const div = document.createElement("div");
      div.className = "panel";
      el.querySelector(".finale-races").append(div);
      return div;
    });
    const [natDiv, seatsDiv] = el.querySelectorAll(".finale-top .panel");
    this.panels = [
      finalePanel(natDiv, {
        title: "National mood",
        polls: nationalPolls,
        paths: draws.map((d) => d.nat),
      }),
      ...races.map((r, j) =>
        finalePanel(raceDivs[j], {
          title: walk.narrow
            ? `${r.label} ${SHORT_OFFICE[r.office]}`
            : `${raceName(r)} · ${r.polls.length} poll${r.polls.length === 1 ? "" : "s"}`,
          polls: r.polls,
          paths: draws.map((d) => d.races[j]),
        }),
      ),
    ];
    this.colors = { ink: css("--ink"), draw: css("--truth") };
    this.seats = seatsChart(seatsDiv);
    this.k = -1;
  },
  update(step, p) {
    const k = Math.round(p * walk.prep.draws.length);
    if (k === this.k) return;
    this.k = k;
    for (const panel of this.panels) paintDraws(panel, k, this.colors);
    this.seats(k);
  },
};

/**
 * Histogram of House seats over the draws so far, read like the House page's: Democratic seats
 * to the left, the axis in Republican seats. Returns a function that shows the first k draws.
 */
function seatsChart(div) {
  const width = div.clientWidth;
  const height = div.clientHeight;
  const m = { top: 30, right: 8, bottom: 34, left: 8 };
  const { draws } = walk.prep;
  const { majority, total } = walk.data.house;
  const seats = draws.map((d) => d.seats);
  const [lo, hi] = d3.extent([...seats, majority]);
  const domain = d3.range(hi + 1, lo - 2, -1);
  const x = d3.scaleBand().domain(domain).range([m.left, width - m.right]).padding(0.1);
  const most = d3.max(d3.rollup(seats, (v) => v.length, (s) => s).values());
  const y = d3
    .scaleLinear()
    .domain([0, most])
    .range([height - m.bottom, m.top + 16]);
  const svg = d3.select(div).append("svg").attr("width", width).attr("height", height);
  svg
    .append("text")
    .attr("x", m.left)
    .attr("y", 12)
    .attr("font-size", 11)
    .attr("font-weight", 600)
    .attr("fill", css("--ink"))
    .text("House");
  const tally = svg
    .append("text")
    .attr("x", m.left)
    .attr("y", 26)
    .attr("font-size", 10)
    .attr("fill", css("--ink-2"));
  const bars = svg
    .append("g")
    .selectAll("rect")
    .data(domain)
    .join("rect")
    .attr("x", (s) => x(s))
    .attr("width", x.bandwidth())
    .attr("fill", (s) => css(s >= majority ? "--dem" : "--rep"))
    .attr("stroke", css("--truth"))
    .attr("y", y(0))
    .attr("height", 0);
  // Between the smallest Democratic majority and one seat short of it.
  const xm = (x(majority) + x.bandwidth() + x(majority - 1)) / 2;
  svg
    .append("line")
    .attr("x1", xm)
    .attr("x2", xm)
    .attr("y1", m.top + 16)
    .attr("y2", height - m.bottom)
    .attr("stroke", css("--ink-2"))
    .attr("stroke-dasharray", "3 3");
  // Labelled on whichever side of the line has more room.
  const roomRight = xm < (m.left + width - m.right) / 2;
  svg
    .append("text")
    .attr("x", roomRight ? xm + 3 : xm - 3)
    .attr("y", m.top + 26)
    .attr("text-anchor", roomRight ? "start" : "end")
    .attr("font-size", 10)
    .attr("fill", css("--ink-2"))
    .text(`${majority} needed for control`);
  svg
    .append("g")
    .attr("transform", `translate(0,${height - m.bottom})`)
    .call(
      d3
        .axisBottom(x)
        .tickValues(domain.filter((s) => (total - s) % (width < 300 ? 20 : 10) === 0))
        .tickFormat((s) => total - s)
        .tickSizeOuter(0),
    )
    .call(styleAxis);
  svg
    .append("text")
    .attr("x", (m.left + width - m.right) / 2)
    .attr("y", height - 4)
    .attr("text-anchor", "middle")
    .attr("font-size", 10)
    .attr("fill", css("--ink-2"))
    .text("Republican House seats");
  return (k) => {
    const counts = d3.rollup(seats.slice(0, k), (v) => v.length, (s) => s);
    const current = k ? seats[k - 1] : null;
    bars
      .attr("y", (s) => y(counts.get(s) ?? 0))
      .attr("height", (s) => y(0) - y(counts.get(s) ?? 0))
      .attr("stroke-width", (s) => (s === current ? 2 : 0));
    const wins = seats.slice(0, k).filter((s) => s >= majority).length;
    tally.text(k ? `Democrats win ${pct(wins / k)} · ${k} draws` : "No draws yet");
  };
}

function highlight(terms) {
  document.querySelectorAll(".equation-bar .term").forEach((t) => {
    const name = [...t.classList].find((c) => c.startsWith("term-")).slice(5);
    t.classList.toggle("active", terms.includes(name));
  });
}

/** Show step i, playing its animation from the start. */
function goTo(i) {
  walk.timer?.stop();
  walk.step = i;
  const step = walk.steps[i];
  const key = step.dataset.step;
  const scene = scenes[step.dataset.scene];
  walk.steps.forEach((s) => (s.hidden = s !== step));
  highlight(step.dataset.term.split(" "));
  document
    .querySelectorAll("#stage .scene")
    .forEach((s) => s.classList.toggle("active", s.dataset.scene === step.dataset.scene));
  document.getElementById("walk-prev").disabled = i === 0;
  document.getElementById("walk-next").disabled = i === walk.steps.length - 1;
  document.getElementById("walk-count").textContent = `${i + 1} of ${walk.steps.length}`;
  document.getElementById("walk-replay").hidden = !DURATION[key];
  const duration = reducedMotion.matches ? 0 : (DURATION[key] ?? 0);
  scene.update(key, 0, true);
  if (!duration) {
    scene.update(key, 1, false);
    return;
  }
  walk.timer = d3.timer((elapsed) => {
    const p = Math.min(1, elapsed / duration);
    scene.update(key, p, false);
    if (p === 1) walk.timer.stop();
  });
}

function renderWalkthrough() {
  walk.width = window.innerWidth;
  walk.narrow = walk.width < 800;
  document.querySelectorAll("#stage .scene").forEach((el) => scenes[el.dataset.scene].render(el));
  goTo(walk.step);
}

function renderText() {
  const values = fills(walk.data, walk.prep, walk.history);
  document.querySelectorAll("[data-fill]").forEach((el) => {
    el.textContent = values[el.dataset.fill];
  });
  renderMathInElement(document.getElementById("walkthrough"), {
    delimiters: [{ left: "\\(", right: "\\)", display: false }],
    ignoredClasses: ["equation", "equation-note"],
  });
}

const katexOptions = { trust: (context) => context.command === "\\htmlClass", strict: false };
katex.render(EQUATION, document.getElementById("equation"), { ...katexOptions, displayMode: true });
const note = document.getElementById("equation-note");
note.textContent = "Each draw of θ then becomes one simulated election: ";
note.append(document.createElement("span"));
katex.render(ELECTION, note.lastChild, katexOptions);
walk.steps = [...document.querySelectorAll(".walk-step")];

document.getElementById("walk-prev").addEventListener("click", () => goTo(walk.step - 1));
document.getElementById("walk-next").addEventListener("click", () => goTo(walk.step + 1));
document.getElementById("walk-replay").addEventListener("click", () => goTo(walk.step));
// Arrow keys step through the walkthrough while it is on screen.
document.addEventListener("keydown", (e) => {
  const box = document.getElementById("walk").getBoundingClientRect();
  if (!walk.prep || box.bottom < 0 || box.top > window.innerHeight) return;
  if (e.key === "ArrowRight" && walk.step < walk.steps.length - 1) goTo(walk.step + 1);
  if (e.key === "ArrowLeft" && walk.step > 0) goTo(walk.step - 1);
});

let walkResize;
window.addEventListener("resize", () => {
  clearTimeout(walkResize);
  // Phones resize as their address bar shows and hides; only a new width needs a redraw.
  walkResize = setTimeout(() => {
    if (walk.prep && window.innerWidth !== walk.width) renderWalkthrough();
  }, 150);
});
window
  .matchMedia("(prefers-color-scheme: dark)")
  .addEventListener("change", () => walk.prep && renderWalkthrough());

Promise.all([d3.json("data/likelihood.json"), d3.json("data/history.json")])
  .then(([data, history]) => {
    Object.assign(walk, { data, history, prep: prepare(data) });
    renderText();
    renderWalkthrough();
  })
  .catch((err) => {
    document.getElementById("glossary").textContent = "Could not load the model data.";
    console.error(err);
  });
