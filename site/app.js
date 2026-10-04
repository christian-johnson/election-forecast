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
const esc = (s) =>
  String(s).replace(/[&<>"']/g, (c) => `&#${c.charCodeAt(0)};`);
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

// Bins of a margin histogram stored as counts per bin, `width` points wide from `start`.
const marginBins = (hist, start, width) =>
  hist.map((v, i) => ({ i, v, x0: start + i * width, x1: start + (i + 1) * width }));
const marginColor = (b) => css((b.x0 + b.x1) / 2 >= 0 ? "--dem" : "--rep");

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
const modalState = { race: null, step: 0 };
let stepsData = null;
// Loaded in the background: the map tooltips use it once it arrives.
const stepsLoaded = d3.json("data/steps.json").then(
  (data) => (stepsData = data),
  (err) => {
    console.error(err);
    throw err;
  },
);
// The first steps describe the national vote, before moving to this race.
const NATIONAL_STEPS = 2;

const pviText = (pvi) =>
  pvi === 0 ? "EVEN" : pvi > 0 ? `D+${pvi}` : `R+${-pvi}`;
const natText = (m) => (m >= 0 ? `D+${m.toFixed(1)}` : `R+${(-m).toFixed(1)}`);
const plural = (n, word) => `${n} ${word}${n === 1 ? "" : "s"}`;

function stepText(key, race) {
  const place = state.office === "house" ? "this district" : "this state";
  const nat = stepsData.national;
  const n = state.data.n_polls;
  switch (key) {
    case "polls":
      return (
        `Start with the polls alone. ${n.generic} generic ballot polls, plus every race poll ` +
        `in the country, put the national mood at ${natText(nat.mean)} on election day ` +
        `(80%: ${natText(nat.lo)} to ${natText(nat.hi)}). This spread is the polls' own ` +
        `statistical uncertainty: sampling noise, disagreement between pollsters, and how much ` +
        `opinion can still drift before November.`
      );
    case "miss":
      return (
        `Polls can all miss in the same direction, as they did in 2016 and 2020. No amount of ` +
        `polling reveals that kind of miss, so the model adds one of typical size, a few points ` +
        `either way. The center stays put; the range widens.`
      );
    case "lean": {
      if (race.pvi === 0)
        return `Now move to ${place}. Its partisan lean is EVEN (Cook PVI): in the last two presidential elections it voted in line with the nation, so the estimate stays where it is.`;
      const dir = race.pvi > 0 ? "Democratic" : "Republican";
      return (
        `Now move to ${place}. Its partisan lean is ${pviText(race.pvi)} (Cook PVI): in the ` +
        `last two presidential elections it voted about ${Math.abs(race.pvi)} points more ` +
        `${dir} than the nation, which shifts the margin about ${2 * Math.abs(race.pvi)} ` +
        `points toward ${race.pvi > 0 ? "Democrats" : "Republicans"}.`
      );
    }
    case "office": {
      const inc = race.incumbent_running
        ? `${race.incumbent} is running again, which is worth a little extra.`
        : "It is an open seat, so neither side has an incumbent's edge.";
      return (
        `${OFFICE_NAME[state.office]} races don't track the generic ballot exactly, and ` +
        `incumbents tend to run a bit ahead of their party. ${inc}`
      );
    }
    case "groups":
      return n.crosstab
        ? `Adjust for who lives here. ${plural(n.crosstab, "crosstab row")} show how groups ` +
            `(by race, education and age) have swung since 2024, and ${place} moves with ` +
            `the groups it has more of than the nation.`
        : `Adjust for who lives here, using Census data on race, education and age. No ` +
            `crosstabs are loaded yet, so group swings can only be inferred from race polls ` +
            `and this step changes little.`;
    case "race":
      return race.n_polls
        ? `Add ${place}'s own ${plural(race.n_polls, "poll")}, each adjusted for its ` +
            `pollster's known lean. This is where the candidates themselves show up: how far ` +
            `this race runs ahead of or behind what the steps above imply.`
        : `Nobody has polled ${place}, so the candidates can't be measured directly. The ` +
            `model adds the typical race-to-race spread it learned from polled races, which ` +
            `widens the range.`;
    case "local":
      return (
        `Finally, polls can miss locally too: all of a state's polls can be off together, a ` +
        `demographic group can be mismeasured, and any single race can surprise. This is ` +
        `the published forecast.`
      );
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

function renderStepChart(race, info, k) {
  const el = document.getElementById("step-chart");
  const step = info.steps[k];
  const width = el.clientWidth || 600;
  const height = 200;
  const m = { top: 24, right: 18, bottom: 26, left: 18 };
  const n = step.hist.length;
  const x = marginScale(info.start, info.start + info.width * n, m.left, width - m.right);
  const yMax = d3.max(info.steps, (s) => d3.max(s.hist));
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

  // Outline of the previous step, so the change is visible.
  const prev = k > 0 ? info.steps[k - 1].hist : null;
  const ghost = prev
    ? d3
        .line()
        .curve(d3.curveStepAfter)
        .x((d) => x(d[0]))
        .y((d) => y(d[1]))([
        ...prev.map((v, i) => [info.start + i * info.width, v]),
        [info.start + n * info.width, prev.at(-1)],
      ])
    : null;
  svg
    .select(".ghost")
    .attr("fill", "none")
    .attr("stroke", css("--ink"))
    .attr("stroke-width", 1.5)
    .attr("stroke-dasharray", "4 3")
    .attr("d", ghost);

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

function renderStep() {
  const race = modalState.race;
  const info = stepsData.races[race.id];
  const k = modalState.step;
  const step = info.steps[k];
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
      <div class="step-nav">
        <button id="step-prev">Back</button>
        <button id="step-next">Next</button>
      </div>
      <table class="step-table">
        <thead><tr><th>Step</th><th class="num">Estimate</th><th class="num">Win probability</th></tr></thead>
        <tbody></tbody>
      </table>
      <p class="modal-note">Each chart is 4,000 simulated elections. Each step adds one piece
      to the one before; the dashed outline is the previous step.</p>`;
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
  body.querySelector("#step-text").textContent = stepText(meta[k].key, race);
  const prevP = k > 0 ? info.steps[k - 1].p_d : null;
  const change =
    prevP === null
      ? ""
      : `<br><span class="muted">${race.d_party} win chance was ${pct(prevP)} before this ` +
        `step${pointChange(step.p_d - prevP)})</span>`;
  body.querySelector("#step-stats").innerHTML =
    `<strong>${winLine(race, step)}${k < NATIONAL_STEPS ? " the national vote" : ""}</strong> · ` +
    `projected margin ${marginText(step.mean, race)} ` +
    `<span class="muted">(80%: ${marginText(step.lo, race)} to ${marginText(step.hi, race)})</span>` +
    change;
  body.querySelector("#step-prev").disabled = k === 0;
  body.querySelector("#step-next").disabled = k === meta.length - 1;
  body.querySelector(".step-table tbody").innerHTML = info.steps
    .map(
      (st, i) => `<tr class="${i === k ? "current" : ""}" data-step="${i}">
        <td>${i + 1}. ${meta[i].label}</td>
        <td class="num">${marginText(st.mean, race)}</td>
        <td class="num">${pct(st.p_d)} ${race.d_party}</td></tr>`,
    )
    .join("");
  body.querySelectorAll(".step-table tbody tr").forEach((tr) =>
    tr.addEventListener("click", () => goToStep(+tr.dataset.step)),
  );
  renderStepChart(race, info, k);
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
  modal.showModal();
  if (race.rule !== "model") {
    body.innerHTML = `<p>${decidedText(race)}, so this race is called for the
      ${PARTY_NAME[race.rule]} without modeling.</p>`;
    return;
  }
  if (!stepsData) {
    body.innerHTML = "<p>Loading…</p>";
    try {
      await stepsLoaded;
    } catch {
      body.innerHTML = "<p>Could not load the step-by-step data.</p>";
      return;
    }
    body.innerHTML = "";
  }
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
