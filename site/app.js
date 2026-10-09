"use strict";

const US_ATLAS =
  "https://cdn.jsdelivr.net/npm/us-atlas@3/states-albers-10m.json";
const FIPS = {
  "01": "AL",
  "02": "AK",
  "04": "AZ",
  "05": "AR",
  "06": "CA",
  "08": "CO",
  "09": "CT",
  10: "DE",
  11: "DC",
  12: "FL",
  13: "GA",
  15: "HI",
  16: "ID",
  17: "IL",
  18: "IN",
  19: "IA",
  20: "KS",
  21: "KY",
  22: "LA",
  23: "ME",
  24: "MD",
  25: "MA",
  26: "MI",
  27: "MN",
  28: "MS",
  29: "MO",
  30: "MT",
  31: "NE",
  32: "NV",
  33: "NH",
  34: "NJ",
  35: "NM",
  36: "NY",
  37: "NC",
  38: "ND",
  39: "OH",
  40: "OK",
  41: "OR",
  42: "PA",
  44: "RI",
  45: "SC",
  46: "SD",
  47: "TN",
  48: "TX",
  49: "UT",
  50: "VT",
  51: "VA",
  53: "WA",
  54: "WV",
  55: "WI",
  56: "WY",
};
const PARTY_NAME = { D: "Democrats", R: "Republicans", I: "Independents" };
const BINS = [
  { min: 0.95, css: "--safe-d", label: "Safe D" },
  { min: 0.8, css: "--likely-d", label: "Likely D" },
  { min: 0.6, css: "--lean-d", label: "Lean D" },
  { min: 0.4, css: "--tossup", label: "Toss-up" },
  { min: 0.2, css: "--lean-r", label: "Lean R" },
  { min: 0.05, css: "--likely-r", label: "Likely R" },
  { min: -1, css: "--safe-r", label: "Safe R" },
];
/* const SEAT_NOTES = {
  house:
    "All 435 seats are up. Bars show the average across simulations; whiskers show the 80% range.",
  senate:
    "Includes the 65 seats not up this year. King and Sanders are counted as Democrats; " +
    "Democrats need 51 seats for control because the Vice President breaks ties.",
  governor: "Includes the 14 governorships not up this year.",
};
*/

const OFFICES = ["house", "senate", "governor"];
const initialOffice = OFFICES.find((o) => `#${o}` === location.hash) || "house";
const state = {
  data: null,
  hexes: null,
  us: null,
  office: initialOffice,
  sortKey: "closeness",
  sortDir: 1,
  query: "",
};
const bin = (race) => BINS.find((b) => race.p_d >= b.min);
const fill = (race) => css(bin(race).css);
function marginText(m, race) {
  const lead = m >= 0 ? race.d_party : race.r_party;
  return `${lead}+${Math.abs(m).toFixed(1)}`;
}

// Rating labels name the party of each side's candidate, so independents read correctly.
const ratingText = (b, race) =>
  b.label.replace(/ D$/, ` ${race.d_party}`).replace(/ R$/, ` ${race.r_party}`);

function favorite(race) {
  return race.p_d >= 0.5
    ? { party: race.d_party, p: race.p_d }
    : { party: race.r_party, p: 1 - race.p_d };
}

function candidate(name, party, side) {
  if (!name) return "—";
  return party === side ? esc(name) : `${esc(name)} (${party})`;
}

function decidedText(race) {
  return race.rule === "D"
    ? "No Republican on ballot"
    : "No Democrat on ballot";
}

function raceTooltip(race) {
  const fav = favorite(race);
  const info = stepsData && stepsData.races[race.id];
  const body = info
    ? miniHistogram(info, race)
    : race.rule !== "model"
      ? `<div class="muted">${decidedText(race)}</div>`
      : "";
  return `
    <div class="title">${race.label}
      <span class="rating"><span class="swatch" style="background:var(${bin(race).css})"></span>${ratingText(bin(race), race)}</span></div>
    <div>Forecast: ${pct(fav.p)} ${fav.party}</div>
    ${body}
    <div class="muted">Click to see full analysis</div>`;
}

// Small margin histogram as markup; the axis is optional so table rows stay compact.
function miniHistogram(info, race, { width = 220, height = 60, axis = true } = {}) {
  const hist = info.steps.at(-1).hist;
  const m = { top: 4, right: 10, bottom: axis ? 18 : 2, left: 10 };
  const end = info.start + info.width * hist.length;
  // Stretch the axis to reach Even, so lopsided races still show where the race would flip.
  const x = marginScale(Math.min(info.start, 0), Math.max(end, 0), m.left, width - m.right);
  const y = d3
    .scaleLinear()
    .domain([0, d3.max(hist)])
    .range([height - m.bottom, m.top]);
  const svg = d3.create("svg").attr("width", width).attr("height", height);
  const barW = Math.max(0.5, x(info.start) - x(info.start + info.width) - 1);
  // One path per party color instead of a rect per bar, since tables draw hundreds of these.
  const bars = d3.group(marginBins(hist, info.start, info.width), marginColor);
  for (const [color, bins] of bars)
    svg
      .append("path")
      .attr("fill", color)
      .attr(
        "d",
        bins.map((b) => `M${x(b.x1)},${y(0)}V${y(b.v)}h${barW}V${y(0)}Z`).join(""),
      );
  svg
    .append("line")
    .attr("x1", x(0))
    .attr("x2", x(0))
    .attr("y1", m.top)
    .attr("y2", height - m.bottom)
    .attr("stroke", css("--ink"));
  if (axis)
    svg
      .append("g")
      .attr("transform", `translate(0,${height - m.bottom})`)
      .attr("font-size", 10)
      .call(
        d3
          .axisBottom(x)
          .ticks(3)
          .tickSize(3)
          .tickSizeOuter(0)
          .tickFormat((v) => (v === 0 ? "Even" : marginText(v, race).replace(".0", ""))),
      )
      .call(styleAxis);
  return `<div class="mini-hist">${svg.node().outerHTML}</div>`;
}

// An 80% range as offsets stacked after a value, like a LaTeX superscript over a subscript.
const pm = (plus, minus, title) =>
  `<span class="pm" title="${title}"><span>+${plus}</span><span>−${minus}</span></span>`;

// The projected winner's margin, with the 80% range seen from the winner's side.
function leadMargin(race) {
  const { mean, lo, hi } = race.margin;
  const d = mean >= 0;
  return {
    party: d ? race.d_party : race.r_party,
    value: Math.abs(mean),
    plus: d ? hi - mean : mean - lo,
    minus: d ? mean - lo : hi - mean,
  };
}

function renderMeta() {
  const d = state.data;
  const updated = new Date(d.updated).toLocaleString(undefined, {
    dateStyle: "long",
  });
  document.getElementById("updated").textContent = `. Updated ${updated}`;
}

const SEATS_TITLE = {
  house: "Control of the House",
  senate: "Control of the Senate",
  governor: "Governorships",
};
// Line label on the seat histogram, keyed by office.
const CONTROL_LABEL = {
  house: (need) => `${need} needed for control of the House`,
  senate: (need) => `${need} needed for Republican control of the Senate`,
};

function renderSeats() {
  const o = state.data.offices[state.office];
  const container = document.getElementById("seats");
  document.getElementById("seats-title").textContent = SEATS_TITLE[state.office];
  const label = (p, side) => {
    const s = o.seats[p];
    const mean = Math.round(s.mean);
    const control = o.p_control
      ? `<div><strong>${pct(o.p_control[p])}</strong> chance of control</div>`
      : "";
    return `<div class="${side}">
      <div class="party"><span class="swatch" style="background:var(--${p === "D" ? "dem" : "rep"})"></span>${PARTY_NAME[p]}</div>
      ${control}
      <div>Projected ${state.office === "governor" ? "governorships" : "seats"}:
        ${mean}${pm(s.hi - mean, mean - s.lo, `80% range ${s.lo}–${s.hi}`)}</div>
    </div>`;
  };
  const ind =
    o.seats.I && o.seats.I.mean >= 1
      ? `<div>Independents: ${o.seats.I.mean.toFixed(1)} expected</div>`
      : "";
  container.innerHTML = `<div class="seat-labels">${label("D", "left")}${ind}${label("R", "right")}</div>`;

  const dist = o.dem_seat_dist;
  const need = o.majority && o.majority.D;
  // Republican wins are the bars below the Democratic majority (or below half for governors).
  const repWin = (seats) => (need ? seats < need : seats * 2 < o.total);
  const atLeast = d3.cumsum([...dist.counts].reverse()).reverse();
  const atMost = d3.cumsum(dist.counts);
  const bins = dist.counts.map((v, i) => {
    const seats = dist.start + i;
    return repWin(seats)
      ? { seats, v, party: "R", count: o.total - seats, p: atMost[i] / 1000 }
      : { seats, v, party: "D", count: seats, p: atLeast[i] / 1000 };
  });
  const unit = state.office === "governor" ? "governorships" : "seats";
  const color = (b) =>
    css(
      need
        ? b.seats >= need ? "--dem" : "--rep"
        : b.seats * 2 > o.total ? "--dem" : b.seats * 2 < o.total ? "--rep" : "--ind",
    );
  const width = container.clientWidth || 600;
  const height = 180;
  const m = { top: 24, right: 12, bottom: 40, left: 12 };
  // Bins count Democratic seats, but the axis reads in Republican seats so Democrats sit on the
  // left, matching the labels above. Seats Democrats don't win are counted as Republican.
  const x = d3
    .scaleLinear()
    .domain([dist.start - 0.5, dist.start + bins.length - 0.5])
    .range([width - m.right, m.left]);
  const repSeats = d3.scaleLinear().domain(x.domain().map((d) => o.total - d)).range(x.range());
  const y = d3
    .scaleLinear()
    .domain([0, d3.max(dist.counts)])
    .range([height - m.bottom, m.top]);
  const svg = d3
    .select(container)
    .append("svg")
    .attr("width", width)
    .attr("height", height)
    .attr("role", "img")
    .attr(
      "aria-label",
      `Simulated Republican ${unit}: ${Math.round(o.seats.R.mean)} on average, 80% range ${o.seats.R.lo} to ${o.seats.R.hi}`,
    );
  const barW = Math.max(1, x(0) - x(1) - 1);
  // Dim the bars outside the tail that the hovered bar's "at least" figure adds up.
  const inTail = (hovered, b) =>
    hovered.party === "D" ? b.seats >= hovered.seats : b.seats <= hovered.seats;
  const bars = svg
    .selectAll("rect.bar")
    .data(bins)
    .join("rect")
    .attr("class", "bar")
    .attr("x", (b) => x(b.seats + 0.5))
    .attr("width", barW)
    .attr("y", (b) => y(b.v))
    .attr("height", (b) => y(0) - y(b.v))
    .attr("fill", color);
  // Invisible full-width columns take the hover, so the gaps between bars don't flicker.
  svg
    .selectAll("rect.hit")
    .data(bins)
    .join("rect")
    .attr("class", "hit")
    .attr("x", (b) => x(b.seats + 0.5))
    .attr("width", x(0) - x(1))
    .attr("y", m.top)
    .attr("height", height - m.bottom - m.top)
    .attr("fill", "transparent")
    .on("mouseenter", (e, hovered) =>
      bars.attr("opacity", (b) => (inTail(hovered, b) ? 1 : 0.3)),
    )
    .on("mousemove", (e, b) =>
      showTooltip(
        e,
        `<div>${pct(b.p)} chance of at least ${b.count} ${b.party === "D" ? "Democratic" : "Republican"} ${unit}</div>`,
      ),
    )
    .on("mouseleave", () => {
      bars.attr("opacity", 1);
      hideTooltip();
    });

  svg
    .append("g")
    .attr("transform", `translate(0,${height - m.bottom})`)
    .call(d3.axisBottom(repSeats).ticks(width < 500 ? 5 : 10).tickFormat(d3.format("d")).tickSizeOuter(0))
    .call(styleAxis);
  svg
    .append("text")
    .attr("x", (m.left + width - m.right) / 2)
    .attr("y", height - 6)
    .attr("text-anchor", "middle")
    .attr("fill", css("--ink-2"))
    .attr("font-size", 12)
    .text(`Republican ${unit} across simulations`);

  if (need) {
    const lineX = x(need - 0.5);
    const repNeed = o.majority.R;
    svg
      .append("line")
      .attr("x1", lineX)
      .attr("x2", lineX)
      .attr("y1", m.top - 8)
      .attr("y2", height - m.bottom)
      .attr("stroke", css("--ink"))
      .attr("stroke-width", 1.5)
      .attr("stroke-dasharray", "3 3");
    const text = svg
      .append("text")
      .attr("y", m.top - 12)
      .attr("text-anchor", "middle")
      .attr("fill", css("--ink"))
      .attr("font-size", 12)
      .attr("font-weight", 600)
      .text(CONTROL_LABEL[state.office](repNeed));
    // Centered over the line, but nudged inward if that would run off the chart.
    const half = text.node().getComputedTextLength() / 2;
    text.attr("x", Math.max(half, Math.min(lineX, width - half)));
  }
}

function renderLegend() {
  const items = BINS.map(
    (b) =>
      `<span><span class="swatch" style="background:var(${b.css})"></span>${b.label}</span>`,
  );
  // Every House seat is up, so only the state maps have shapes without a race.
  if (state.office !== "house")
    items.push(
      `<span><span class="swatch" style="background:var(--none)"></span>No race</span>`,
    );
  document.getElementById("legend").innerHTML = items.join("");
}

const OFFICE_NAME = { house: "House", senate: "Senate", governor: "Governor" };
const modal = document.getElementById("race-modal");
const modalState = { race: null, step: 0, detail: null };
let stepsData = null;
// Loaded in the background: the map tooltips use it once it arrives.
const stepsLoaded = d3.json("data/steps.json").then(
  (data) => (stepsData = data),
  (err) => {
    console.error(err);
    throw err;
  },
);
// Every poll and the weekly estimates, for the pop-up's timeline; loaded on first open.
let detailLoaded = null;
const loadDetail = () =>
  (detailLoaded ??= d3.json("data/timeline.json").then((timeline) => ({
    timeline,
    polls: timeline.polls,
  })));
// The first steps describe the national vote, before moving to this race.
const NATIONAL_STEPS = 2;

const pviText = (pvi) =>
  pvi === 0 ? "EVEN" : pvi > 0 ? `D+${pvi}` : `R+${-pvi}`;
const natText = (m) => (m >= 0 ? `D+${m.toFixed(1)}` : `R+${(-m).toFixed(1)}`);
const plural = (n, word) => `${n} ${word}${n === 1 ? "" : "s"}`;

function nationalText(nat, n) {
  const spread = (nat.hi - nat.lo) / 2;
  return (
    `Start with the national environment. A national mood that drifts from week to week, ` +
    `fit to ${n.generic} generic ballot polls and ${n.race} state and district polls (recent ` +
    `ones count most), stands at ${natText(nat.mean)} on election day, plus or minus about ` +
    `${spread.toFixed(1)} points (80% range). That range covers sampling noise, disagreement ` +
    `between pollsters, and how far opinion can still move before November.`
  );
}

function missText(h) {
  const side = h.largest_national_miss > 0 ? "Democrats" : "Republicans";
  return (
    `Polls can all miss in the same direction, and no amount of polling reveals that ahead ` +
    `of time. Over the ${h.poll_elections}, the late polls missed by about ` +
    `${h.national_miss} points in a typical year, and by ` +
    `${Math.abs(h.largest_national_miss)} in ${h.largest_national_miss_year} (overstating ` +
    `${side}). So every race gets a shared miss of that size: the center stays put and the ` +
    `range widens.`
  );
}

const placeName = () => (state.office === "house" ? "this district" : "this state");

function priorText(race, info) {
  const h = stepsData.history;
  const lean =
    race.pvi === 0
      ? `Its Cook PVI is EVEN: in the last two presidential elections it voted in line ` +
        `with the nation.`
      : `Its Cook PVI is ${pviText(race.pvi)}: in the last two presidential elections it ` +
        `voted about ${Math.abs(race.pvi)} points more ` +
        `${race.pvi > 0 ? "Democratic" : "Republican"} than the nation, worth about ` +
        `${2 * Math.abs(race.pvi)} points of margin.`;
  const incumbent = race.incumbent_running
    ? ` ${race.incumbent} is running again, worth about ${h.incumbency} points.`
    : " It is an open seat.";
  return (
    `Now move to ${placeName()}, before looking at its own polls. ${lean}${incumbent} Added ` +
    `to the national picture, that puts it at ${marginText(info.expected, race)}. That is ` +
    `only a starting point: in ${h.pvi_elections}, 80% of ${OFFICE_NAME[state.office]} ` +
    `races landed within ${Math.round(Z80 * h.race_spread[state.office])} points of what ` +
    `PVI, incumbency and the national vote predicted, so the range widens to match.`
  );
}

function pollsText(race, step) {
  const h = stepsData.history;
  if (!race.n_polls)
    return (
      `Nobody has polled ${placeName()}, so its forecast stays at the starting point from ` +
      `the last step. This is the published forecast.`
    );
  return (
    `Now add ${placeName()}'s own ${plural(race.n_polls, "poll")}, each adjusted for its ` +
    `pollster's lean. They move it to ${marginText(step.mean, race)}. They count for less ` +
    `than their sample sizes suggest: ${OFFICE_NAME[state.office]} polls scatter by about ` +
    `${stepsData.noise[state.office]} points beyond sampling error, and all of a race's ` +
    `polls can miss together (by ${h.race_miss} points in a typical past race, plus ` +
    `${h.state_miss} shared across a state). This is the published forecast.`
  );
}

function stepText(key, race, info, step) {
  switch (key) {
    case "national":
      return nationalText(stepsData.national, state.data.n_polls);
    case "national_miss":
      return missText(stepsData.history);
    case "prior":
      return priorText(race, info);
    case "polls":
      return pollsText(race, step);
    default:
      return "";
  }
}

function pointChange(delta) {
  const points = Math.round(100 * delta);
  if (points === 0) return " (no change";
  return ` (${points > 0 ? "+" : "−"}${Math.abs(points)} points`;
}

function winLine(race, step) {
  const fav = favorite({ ...race, p_d: step.p_d });
  return `${PARTY_NAME[fav.party]} ${pct(fav.p)} to win`;
}

// Horizontal margins shared by the histogram and the timeline below it, so they share one axis.
const STEP_X = { left: 60, right: 18 };
const stepScale = (info, width) =>
  marginScale(
    info.start,
    info.start + info.width * info.steps[0].hist.length,
    STEP_X.left,
    width - STEP_X.right,
  );

/**
 * From the race's step on, an arrow from the national center to where PVI and incumbency move
 * this race. It grows out of the national center when the race first comes in.
 */
function renderLeanArrow(svg, race, info, k, x, arrowY) {
  const arrow = svg.select(".lean-arrow");
  const shown = arrow.attr("visibility") === "visible";
  if (k < 2) {
    arrow.attr("visibility", "hidden");
    return;
  }
  const from = x(info.steps[1].mean);
  const to = x(info.steps[2].mean);
  const label =
    `PVI ${pviText(race.pvi)}` + (race.incumbent_running ? " + incumbent" : "");
  arrow.attr("visibility", "visible");
  const line = arrow
    .select("line")
    .attr("x1", from)
    .attr("y1", arrowY)
    .attr("y2", arrowY)
    .interrupt();
  const text = arrow
    .select("text")
    .attr("y", arrowY - 6)
    .text(label)
    .interrupt();
  if (shown) {
    line.attr("x2", to);
    text.attr("x", (from + to) / 2);
  } else {
    line.attr("x2", from).transition().duration(600).attr("x2", to);
    text.attr("x", from).transition().duration(600).attr("x", (from + to) / 2);
  }
}

function renderStepChart(race, info, k) {
  const el = document.getElementById("step-chart");
  const step = info.steps[k];
  const width = el.clientWidth || 600;
  const height = 200;
  const m = { top: 24, bottom: 26, ...STEP_X };
  const n = step.hist.length;
  const x = stepScale(info, width);
  // Fit the taller of this step and the previous one's outline, so wide steps stay readable.
  const yMax = 1.2 * d3.max([...step.hist, ...(k > 0 ? info.steps[k - 1].hist : [])]);
  const y = d3
    .scaleLinear()
    .domain([0, yMax])
    .range([height - m.bottom, m.top]);
  let svg = d3.select(el).select("svg");
  if (svg.empty() || +svg.attr("width") !== width || svg.attr("data-race") !== race.id) {
    el.innerHTML = "";
    svg = d3
      .select(el)
      .append("svg")
      .attr("width", width)
      .attr("height", height)
      .attr("data-race", race.id);
    svg.append("g").attr("class", "bars");
    svg.append("path").attr("class", "ghost");
    svg.append("line").attr("class", "zero");
    svg.append("g").attr("class", "x-axis");
    svg.append("g").attr("class", "side-labels");
  }
  svg.attr("aria-label", `Simulated margins: ${winLine(race, step)}`).attr("role", "img");
  const bins = marginBins(step.hist, info.start, info.width);
  const barW = Math.max(1, x(info.start) - x(info.start + info.width) - 1);
  svg
    .select(".bars")
    .selectAll("rect")
    .data(bins, (b) => b.i)
    .join((enter) =>
      enter
        .append("rect")
        .attr("x", (b) => x(b.x1))
        .attr("width", barW)
        .attr("y", y(0))
        .attr("height", 0),
    )
    .attr("fill", marginColor)
    .transition()
    .duration(600)
    .attr("y", (b) => y(b.v))
    .attr("height", (b) => y(0) - y(b.v));

  // Outline of the previous step, so the change is visible. It starts on the old vertical scale
  // (where the bars just were) and rescales along with the bars.
  const prev = k > 0 ? info.steps[k - 1].hist : null;
  const outline = (scale) =>
    d3
      .line()
      .curve(d3.curveStepAfter)
      .x((d) => x(d[0]))
      .y((d) => scale(d[1]))([
      ...prev.map((v, i) => [info.start + i * info.width, v]),
      [info.start + n * info.width, prev.at(-1)],
    ]);
  const oldY = y.copy().domain([0, svg.property("yMax") ?? yMax]);
  svg.property("yMax", yMax);
  const ghost = svg
    .select(".ghost")
    .attr("fill", "none")
    .attr("stroke", css("--ink"))
    .attr("stroke-width", 1.5)
    .attr("stroke-dasharray", "4 3")
    .interrupt();
  if (prev) ghost.attr("d", outline(oldY)).transition().duration(600).attr("d", outline(y));
  else ghost.attr("d", null);

  const zeroX = x(0);
  svg
    .select(".zero")
    .attr("x1", zeroX)
    .attr("x2", zeroX)
    .attr("y1", m.top - 8)
    .attr("y2", height - m.bottom)
    .attr("stroke", css("--ink"))
    .attr("stroke-width", 1)
    .attr("visibility", zeroX > m.left && zeroX < width - m.right ? "visible" : "hidden");
  svg
    .select(".x-axis")
    .attr("transform", `translate(0,${height - m.bottom})`)
    .call(
      d3
        .axisBottom(x)
        .ticks(width < 500 ? 4 : 7)
        .tickFormat((v) => (v === 0 ? "Even" : marginText(v, race).replace(".0", "")))
        .tickSizeOuter(0),
    )
    .call(styleAxis);

  const labels = [
    { text: `${race.d_party} wins ${pct(step.p_d)}`, x: Math.max(m.left, Math.min(zeroX, width - m.right)) - 6, anchor: "end" },
    { text: `${race.r_party} wins ${pct(1 - step.p_d)}`, x: Math.max(m.left, Math.min(zeroX, width - m.right)) + 6, anchor: "start" },
  ];
  svg
    .select(".side-labels")
    .selectAll("text")
    .data(labels)
    .join("text")
    .attr("x", (d) => d.x)
    .attr("y", m.top - 10)
    .attr("text-anchor", (d) => d.anchor)
    .attr("fill", css("--ink"))
    .attr("font-size", 12)
    .attr("font-weight", 600)
    .text((d) => d.text);
}

/** Weekly estimate shown under step k: national mood, or this race before or after its polls. */
function timeSeries(race, k) {
  const tl = modalState.detail.timeline;
  const nat = tl.national;
  let { mean, sd } = nat;
  const own = tl.races[race.id];
  if (k === 3 && own) ({ mean, sd } = own);
  else if (k >= 2) {
    const [lean, spread] = tl.prior[race.id];
    mean = nat.mean.map((v) => v + lean);
    sd = nat.sd.map((v) => Math.hypot(v, spread));
  }
  // From step 2 on, a polling miss shared by every poll would shift every week's reading alike.
  if (k >= 1) {
    const miss = stepsData.history.national_miss * POINT;
    sd = sd.map((v) => Math.hypot(v, miss));
  }
  const start = new Date(`${tl.start}T12:00`).getTime();
  const electionDay = new Date(`${state.data.election_date}T12:00`);
  return mean.map((v, i) => ({
    // The last week runs up to Election Day, at the top of the chart.
    t: i === mean.length - 1 ? electionDay : new Date(start + i * WEEK_MS),
    mid: logitMargin(v),
    lo: logitMargin(v - Z80 * sd[i]),
    hi: logitMargin(v + Z80 * sd[i]),
  }));
}

function stepPolls(race, k) {
  const { polls } = modalState.detail;
  if (k < 2) return polls.generic.map((p, i) => ({ ...p, key: `g${i}` }));
  if (k === 3) return (polls.races[race.id] ?? []).map((p, i) => ({ ...p, key: `r${i}` }));
  return [];
}

function timeNote(race, k) {
  const miss = k >= 1 ? " The band includes the national polling miss from step 2." : "";
  if (k < 2)
    return (
      "Below: each bubble is a generic ballot poll, sized by its sample. The line and band " +
      `are the national mood each week, up to Election Day at the top.${miss}`
    );
  if (k === 2 || !race.n_polls)
    return `Below: where PVI and incumbency put this race each week, given the national mood.${miss}`;
  return (
    "Below: each bubble is a poll of this race, sized by its sample. The line and band are " +
    `the model's estimate each week.${miss}`
  );
}

/** Key for the line and band, in the bottom right corner of the timeline. */
function timeLegend(svg, right, bottom) {
  const w = 132;
  const g = svg.append("g").attr("transform", `translate(${right - w - 4},${bottom - 44})`);
  g.append("rect")
    .attr("width", w)
    .attr("height", 40)
    .attr("rx", 4)
    .attr("fill", css("--surface"))
    .attr("fill-opacity", 0.85);
  g.append("line")
    .attr("x1", 8)
    .attr("x2", 26)
    .attr("y1", 13)
    .attr("y2", 13)
    .attr("stroke", css("--ink"))
    .attr("stroke-width", 2);
  g.append("rect").attr("x", 8).attr("y", 23).attr("width", 18).attr("height", 10).attr("fill", css("--band"));
  g.selectAll("text")
    .data([
      ["Estimate", 17],
      ["80% range", 32],
    ])
    .join("text")
    .attr("x", 34)
    .attr("y", (d) => d[1])
    .attr("font-size", 11)
    .attr("fill", css("--ink-2"))
    .text((d) => d[0]);
}

function renderTimeChart(race, info, k) {
  const el = document.getElementById("step-time");
  const width = el.clientWidth || 600;
  const narrow = width < 500;
  const height = narrow ? 214 : 264;
  // Room at the top for the lean arrow, between the histogram's axis and Election Day.
  const m = { top: 34, bottom: 6, ...STEP_X };
  const x = stepScale(info, width);
  const series = timeSeries(race, k);
  const today = new Date(state.data.updated);
  const y = d3
    .scaleTime()
    .domain([series.at(-1).t, series[0].t])
    .range([m.top, height - m.bottom]);
  const r = d3
    .scaleSqrt()
    .domain([0, MAX_N])
    .range([0, narrow ? 4 : 5])
    .clamp(true);

  let svg = d3.select(el).select("svg");
  if (svg.empty() || +svg.attr("width") !== width || svg.attr("data-race") !== race.id) {
    el.innerHTML = "";
    svg = d3
      .select(el)
      .append("svg")
      .attr("width", width)
      .attr("height", height)
      .attr("data-race", race.id)
      .attr("role", "img");
    svg
      .append("clipPath")
      .attr("id", "step-time-clip")
      .append("rect")
      .attr("x", m.left)
      .attr("width", width - m.left - m.right)
      .attr("height", height);
    const plot = svg.append("g").attr("clip-path", "url(#step-time-clip)");
    // Bubbles go underneath, so the estimate stays visible through crowded national polls.
    plot.append("g").attr("class", "bubbles");
    // Everything drawn after the bubbles is for looking only, so clicks reach the polls below.
    const overlay = svg.append("g").attr("pointer-events", "none");
    plot.append("path").attr("class", "band").attr("fill", css("--band")).attr("pointer-events", "none");
    svg
      .append("marker")
      .attr("id", "lean-arrowhead")
      .attr("viewBox", "0 0 10 10")
      .attr("refX", 9)
      .attr("refY", 5)
      .attr("markerWidth", 4)
      .attr("markerHeight", 4)
      .attr("orient", "auto")
      .append("path")
      .attr("d", "M0,0L10,5L0,10Z")
      .attr("fill", css("--ink"));
    const lean = overlay.append("g").attr("class", "lean-arrow").attr("visibility", "hidden");
    lean
      .append("line")
      .attr("stroke", css("--ink"))
      .attr("stroke-width", 1.5)
      .attr("marker-end", "url(#lean-arrowhead)");
    lean
      .append("text")
      .attr("text-anchor", "middle")
      .attr("font-size", 11)
      .attr("fill", css("--ink"));
    timeLegend(overlay, width - m.right, height - m.bottom);
    plot
      .append("path")
      .attr("class", "line")
      .attr("pointer-events", "none")
      .attr("fill", "none")
      .attr("stroke", css("--ink"))
      .attr("stroke-width", 2);
    const zeroX = x(0);
    if (zeroX > m.left && zeroX < width - m.right)
      overlay
        .append("line")
        .attr("x1", zeroX)
        .attr("x2", zeroX)
        .attr("y1", m.top)
        .attr("y2", height - m.bottom)
        .attr("stroke", css("--axis"));
    overlay
      .append("line")
      .attr("x1", m.left)
      .attr("x2", width - m.right)
      .attr("y1", y(today))
      .attr("y2", y(today))
      .attr("stroke", css("--ink-2"))
      .attr("stroke-dasharray", "3 3");
    overlay
      .append("text")
      .attr("x", width - m.right - 4)
      .attr("y", y(today) - 4)
      .attr("text-anchor", "end")
      .attr("font-size", 11)
      .attr("fill", css("--ink-2"))
      .text("Today");
    svg
      .append("g")
      .attr("transform", `translate(${m.left},0)`)
      .call(
        d3
          .axisLeft(y)
          .tickValues([series.at(-1).t, ...y.ticks(narrow ? 3 : 5).filter((t) => y(t) > y(today) + 14)])
          .tickFormat((t) => d3.timeFormat(+t === +series.at(-1).t ? "%b %-d" : "%b %Y")(t))
          .tickSizeOuter(0),
      )
      .call(styleAxis)
      .call((g) => g.select(".tick text").attr("font-weight", 700));
  }
  svg.attr("aria-label", timeNote(race, k));

  // Smoothed, since each week is estimated on its own and the raw edges are jagged.
  const area = d3
    .area()
    .curve(d3.curveBasis)
    .x0((d) => x(d.lo))
    .x1((d) => x(d.hi))
    .y((d) => y(d.t));
  const line = d3
    .line()
    .curve(d3.curveBasis)
    .x((d) => x(d.mid))
    .y((d) => y(d.t));
  svg.select(".band").transition().duration(600).attr("d", area(series));
  renderLeanArrow(svg, race, info, k, x, 20);
  svg.select(".line").transition().duration(600).attr("d", line(series));

  const rows = stepPolls(race, k).map((p) => ({ ...p, size: Math.min(p.n ?? ASSUMED_N, MAX_N) }));
  svg
    .select(".bubbles")
    .selectAll("a")
    .data(
      rows.toSorted((a, b) => b.size - a.size),
      (p) => p.key,
    )
    .join(
      (enter) => {
        const a = enter
          .append("a")
          .attr("class", "bubble")
          .attr("target", "_blank")
          .attr("rel", "noopener")
          .attr("opacity", 0);
        a.append("circle");
        a.transition().duration(600).attr("opacity", 1);
        return a;
      },
      (update) => update,
      (exit) => exit.transition().duration(400).attr("opacity", 0).remove(),
    )
    .attr("href", (p) => safeUrl(p.url))
    .on("mousemove", (e, p) => showTooltip(e, pollTooltip(p)))
    .on("mouseleave", hideTooltip)
    .select("circle")
    .attr("cx", (p) => x(p.margin))
    .attr("cy", (p) => y(new Date(`${p.date}T12:00`)))
    .attr("r", (p) => Math.max(1.5, r(p.size)))
    .attr("fill", (p) => css(p.margin >= 0 ? "--dem" : "--rep"))
    .attr("fill-opacity", 0.25)
    .attr("stroke", (p) => css(p.margin >= 0 ? "--dem" : "--rep"));
}

function statsHtml(race, info, k) {
  const step = info.steps[k];
  const prevP = k > 0 ? info.steps[k - 1].p_d : null;
  const change =
    prevP === null
      ? ""
      : `<br><span class="muted">${race.d_party} win chance was ${pct(prevP)} before this ` +
        `step${pointChange(step.p_d - prevP)})</span>`;
  return (
    `<strong>${winLine(race, step)}${k < NATIONAL_STEPS ? " the national vote" : ""}</strong> · ` +
    `projected margin ${marginText(step.mean, race)} ` +
    `<span class="muted">(80%: ${marginText(step.lo, race)} to ${marginText(step.hi, race)})</span>` +
    change
  );
}

/** Show versions[k] in el, with el's height fixed to the tallest of all versions. */
function reserveHeight(el, versions, k) {
  el.style.minHeight = "";
  let tallest = 0;
  for (const html of versions) {
    el.innerHTML = html;
    tallest = Math.max(tallest, el.offsetHeight);
  }
  el.style.minHeight = `${tallest}px`;
  el.innerHTML = versions[k];
}

function renderStep() {
  const race = modalState.race;
  const info = stepsData.races[race.id];
  const k = modalState.step;
  const meta = stepsData.steps;
  const body = document.getElementById("modal-body");
  if (!body.querySelector(".stepper")) {
    body.innerHTML = `
      <ol class="stepper">${meta
        .map((s, i) => `<li><button data-step="${i}">${i + 1}. ${s.label}</button></li>`)
        .join("")}</ol>
      <h3 id="step-title"></h3>
      <p id="step-text" class="step-text"></p>
      <p id="step-stats" class="step-stats"></p>
      <div id="step-chart"></div>
      <div id="step-time"></div>
      <p id="time-note" class="modal-note"></p>
      <div class="step-nav">
        <button id="step-prev">Back</button>
        <button id="step-next">Next</button>
      </div>
      <p class="modal-note">Each histogram is 4,000 simulated elections, going from the national
      picture down to this race; the dashed outline is the previous step.</p>`;
    body.querySelectorAll("[data-step]").forEach((b) =>
      b.addEventListener("click", () => goToStep(+b.dataset.step)),
    );
    body.querySelector("#step-prev").addEventListener("click", () => goToStep(modalState.step - 1));
    body.querySelector("#step-next").addEventListener("click", () => goToStep(modalState.step + 1));
  }
  body.querySelectorAll(".stepper button").forEach((b) => {
    const i = +b.dataset.step;
    b.setAttribute("aria-current", i === k ? "step" : "false");
    b.classList.toggle("done", i < k);
  });
  body.querySelector("#step-title").textContent = `Step ${k + 1} of ${meta.length}: ${meta[k].label}`;
  const texts = meta.map((s, i) => esc(stepText(s.key, race, info, info.steps[i])));
  const stats = meta.map((_, i) => statsHtml(race, info, i));
  // Keep the charts still: each block is as tall as its longest version across the steps.
  reserveHeight(body.querySelector("#step-text"), texts, k);
  reserveHeight(body.querySelector("#step-stats"), stats, k);
  body.querySelector("#step-prev").disabled = k === 0;
  body.querySelector("#step-next").disabled = k === meta.length - 1;
  renderStepChart(race, info, k);
  renderTimeChart(race, info, k);
  body.querySelector("#time-note").textContent = timeNote(race, k);
}

function goToStep(k) {
  modalState.step = Math.max(0, Math.min(k, stepsData.steps.length - 1));
  renderStep();
}

async function openRace(race) {
  hideTooltip();
  modalState.race = race;
  modalState.step = 0;
  document.getElementById("modal-title").textContent =
    `${race.label} ${OFFICE_NAME[state.office]}`;
  document.getElementById("modal-sub").innerHTML =
    `${candidate(race.d_name, race.d_party, "D")} vs ${candidate(race.r_name, race.r_party, "R")}`;
  const body = document.getElementById("modal-body");
  body.innerHTML = "";
  // The dialog sits in the browser's top layer, so the tooltip must live inside it to show.
  modal.append(tooltip);
  modal.showModal();
  if (race.rule !== "model") {
    body.innerHTML = `<p>${decidedText(race)}, so this race is called for the
      ${PARTY_NAME[race.rule]} without modeling.</p>`;
    return;
  }
  body.innerHTML = "<p>Loading…</p>";
  try {
    [, modalState.detail] = await Promise.all([stepsLoaded, loadDetail()]);
  } catch (err) {
    console.error(err);
    detailLoaded = null;
    body.innerHTML = "<p>Could not load the step-by-step data.</p>";
    return;
  }
  if (modalState.race !== race) return;
  body.innerHTML = "";
  renderStep();
}

// Hover outlines go on their own top layer, so neighbors never cover them.
function bindHover(selection, svg) {
  const outline = svg.append("path").attr("class", "hover-outline");
  selection
    .filter((d) => d.race)
    .classed("active", true)
    .classed("tossup", (d) => bin(d.race).css === "--tossup")
    .on("mouseenter", function () {
      outline.attr("d", this.getAttribute("d")).attr("visibility", "visible");
    })
    .on("mousemove", (e, d) => showTooltip(e, raceTooltip(d.race)))
    .on("mouseleave", () => {
      outline.attr("visibility", "hidden");
      hideTooltip();
    })
    .on("click", (e, d) => openRace(d.race));
}

function renderStateMap(races) {
  const byState = new Map(races.map((r) => [r.state, r]));
  const features = topojson
    .feature(state.us, state.us.objects.states)
    .features.map((f) => ({ f, race: byState.get(FIPS[f.id]) }));
  const svg = d3.select("#map").append("svg").attr("viewBox", "0 0 975 610");
  svg
    .selectAll("path")
    .data(features)
    .join("path")
    .attr("class", "map-shape")
    .attr("d", (d) => d3.geoPath()(d.f))
    .attr("fill", (d) => (d.race ? fill(d.race) : css("--none")))
    .call(bindHover, svg);
}

function renderHexMap(races) {
  const byId = new Map(races.map((r) => [r.label, r]));
  // Layout coordinates are in hex radii; hexes are drawn slightly small to leave a gap.
  const R = 10,
    drawn = R - 0.8;
  const hexes = state.hexes.map((h) => ({
    ...h,
    c: [h.x * R, h.y * R],
    race: byId.get(h.id),
  }));
  const width = d3.max(hexes, (h) => h.c[0]) + R;
  const height = d3.max(hexes, (h) => h.c[1]) + R;
  const corners = d3.range(6).map((i) => {
    const a = (Math.PI / 180) * (60 * i - 30);
    return [drawn * Math.cos(a), drawn * Math.sin(a)];
  });
  const svg = d3
    .select("#map")
    .append("svg")
    .attr("viewBox", `0 0 ${width} ${height}`);
  svg
    .selectAll("text")
    .data(d3.group(hexes, (h) => h.state))
    .join("text")
    .attr("class", "hex-label")
    .attr("x", ([, hs]) => (d3.min(hs, (h) => h.c[0]) + d3.max(hs, (h) => h.c[0])) / 2)
    .attr("y", ([, hs]) => d3.min(hs, (h) => h.c[1]) - R - 3)
    .text(([s]) => s);
  svg
    .selectAll("path")
    .data(hexes)
    .join("path")
    .attr("class", "map-shape hex")
    .attr(
      "d",
      (h) =>
        "M" +
        corners.map(([dx, dy]) => [h.c[0] + dx, h.c[1] + dy]).join("L") +
        "Z",
    )
    .attr("fill", (h) => (h.race ? fill(h.race) : css("--none")))
    .call(bindHover, svg);
}

function renderMap() {
  document.getElementById("map").innerHTML = "";
  const races = state.data.offices[state.office].races;
  if (state.office === "house") renderHexMap(races);
  else renderStateMap(races);
}

const SORT_VALUE = {
  label: (r) => r.id,
  d_name: (r) => r.d_name || "",
  r_name: (r) => r.r_name || "",
  p_d: (r) => -r.p_d,
  closeness: (r) => (r.margin ? Math.abs(r.margin.mean) : Infinity),
};

function renderTable() {
  const q = state.query.toLowerCase();
  const rows = state.data.offices[state.office].races.filter(
    (r) =>
      !q ||
      [r.label, r.state, r.d_name, r.r_name].some(
        (v) => v && v.toLowerCase().includes(q),
      ),
  );
  const key = SORT_VALUE[state.sortKey];
  rows.sort((a, b) => {
    const va = key(a),
      vb = key(b);
    return (va < vb ? -1 : va > vb ? 1 : 0) * state.sortDir;
  });
  document.querySelectorAll("#races th").forEach((th) => {
    th.setAttribute(
      "aria-sort",
      th.dataset.key === state.sortKey
        ? state.sortDir > 0
          ? "ascending"
          : "descending"
        : "none",
    );
  });
  document.querySelector("#races tbody").innerHTML = rows
    .map((r) => {
      const info = stepsData && stepsData.races[r.id];
      let projection = `<span class="range">${decidedText(r)}</span>`;
      if (r.margin) {
        const lead = leadMargin(r);
        const range = `80% range ${marginText(r.margin.lo, r)} to ${marginText(r.margin.hi, r)}`;
        projection = `<div class="projection">
          <span class="lead">${lead.party} +${lead.value.toFixed(1)}${pm(lead.plus.toFixed(1), lead.minus.toFixed(1), range)}</span>
          ${info ? miniHistogram(info, r, { width: 140, height: 28, axis: false }) : ""}
        </div>`;
      }
      const b = bin(r);
      const fav = favorite(r);
      return `<tr data-id="${esc(r.id)}" tabindex="0">
      <td class="race">${r.label}</td>
      <td>${candidate(r.d_name, r.d_party, "D")}</td>
      <td>${candidate(r.r_name, r.r_party, "R")}</td>
      <td class="forecast"><span class="swatch" style="background:var(${b.css})"></span>${ratingText(b, r)}
        <span class="muted">${pct(fav.p)}${b.css === "--tossup" ? ` ${fav.party}` : ""}</span></td>
      <td>${projection}</td>
    </tr>`;
    })
    .join("");
}

function render() {
  document
    .querySelectorAll(".tabs button")
    .forEach((b) =>
      b.setAttribute("aria-selected", b.dataset.office === state.office),
    );
  renderSeats();
  renderLegend();
  renderMap();
  renderTable();
}

function bindControls() {
  document.querySelectorAll(".tabs button").forEach((b) =>
    b.addEventListener("click", () => {
      state.office = b.dataset.office;
      history.replaceState(null, "", `#${state.office}`);
      render();
    }),
  );
  document.querySelectorAll("#races th").forEach((th) =>
    th.addEventListener("click", () => {
      state.sortDir = state.sortKey === th.dataset.key ? -state.sortDir : 1;
      state.sortKey = th.dataset.key;
      renderTable();
    }),
  );
  const tbody = document.querySelector("#races tbody");
  const openRow = (tr) => {
    const race = state.data.offices[state.office].races.find((r) => r.id === tr.dataset.id);
    if (race) openRace(race);
  };
  tbody.addEventListener("click", (e) => {
    const tr = e.target.closest("tr[data-id]");
    if (tr) openRow(tr);
  });
  tbody.addEventListener("keydown", (e) => {
    const tr = e.target.closest("tr[data-id]");
    if (tr && (e.key === "Enter" || e.key === " ")) {
      e.preventDefault();
      openRow(tr);
    }
  });
  document.getElementById("modal-close").addEventListener("click", () => modal.close());
  modal.addEventListener("close", () => {
    hideTooltip();
    document.body.append(tooltip);
  });
  modal.addEventListener("click", (e) => {
    if (e.target === modal) modal.close();
  });
  modal.addEventListener("keydown", (e) => {
    if (e.key === "ArrowRight") goToStep(modalState.step + 1);
    if (e.key === "ArrowLeft") goToStep(modalState.step - 1);
  });
  document.getElementById("search").addEventListener("input", (e) => {
    state.query = e.target.value.trim();
    renderTable();
  });
  window
    .matchMedia("(prefers-color-scheme: dark)")
    .addEventListener("change", render);
  window.addEventListener("resize", renderSeats);
}

Promise.all([
  d3.json("data/forecast.json"),
  d3.json("data/house_hex.json"),
  d3.json(US_ATLAS),
])
  .then(([data, hexes, us]) => {
    Object.assign(state, { data, hexes, us });
    renderMeta();
    bindControls();
    render();
    // The table's histograms appear once the step data arrives; a failure is already logged.
    stepsLoaded.then(renderTable, () => {});
  })
  .catch((err) => {
    document.getElementById("updated").textContent =
      ". Could not load the forecast data.";
    console.error(err);
  });
