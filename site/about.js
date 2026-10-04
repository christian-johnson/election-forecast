"use strict";

const SCENARIO_TEXT = {
  none:
    "With honest polls, the model lands close to the truth, including in the two districts " +
    "nobody polled, and it sorts out which pollsters lean which way.",
  dem:
    "Every poll overstates Democrats by 3 points. The model cannot see a miss that every " +
    "pollster shares, so its estimates shift toward Democrats with the polls. The ranges are " +
    "wide enough to allow for this kind of miss, which is why the forecast is less certain " +
    "than the polls alone would suggest.",
  rep:
    "Every poll overstates Republicans by 3 points, so the model leans too far toward " +
    "Republicans. Close races are where this matters: an underdog in a race the model calls " +
    "likely for the other side can still win, as the ranges allow.",
};

const state = { data: null, scenario: "none" };

const fmt = (m) =>
  Math.abs(m) < 0.05
    ? "Even"
    : m > 0
      ? `D+${m.toFixed(1)}`
      : `R+${(-m).toFixed(1)}`;

const swatch = (cls) => `<span class="key ${cls}"></span>`;

function legend(id, items) {
  document.getElementById(id).innerHTML = items
    .map(([cls, label]) => `<span>${swatch(cls)}${label}</span>`)
    .join("");
}

function marginAxis(g, x, height) {
  g.call(
    d3
      .axisBottom(x)
      .ticks(6)
      .tickFormat((m) => (m === 0 ? "Even" : fmt(m).replace(".0", "")))
      .tickSize(-height),
  )
    .call((a) => a.select(".domain").remove())
    .call((a) => a.selectAll(".tick line").attr("stroke", css("--grid")))
    .call((a) => a.selectAll(".tick text").attr("fill", css("--ink-2")));
}

function renderNational(s) {
  const el = document.getElementById("national");
  el.innerHTML = "";
  const width = el.clientWidth;
  const height = 260;
  const m = { top: 8, right: 12, bottom: 28, left: 48 };
  const points = s.national.map((d) => ({ ...d, t: new Date(d.date) }));
  const polls = s.generic_polls.map((d) => ({ ...d, t: new Date(d.date) }));
  const x = d3
    .scaleTime()
    .domain(d3.extent(points, (d) => d.t))
    .range([m.left, width - m.right]);
  const values = [
    ...points.flatMap((d) => [d.lo, d.hi, d.truth]),
    ...polls.map((d) => d.margin),
  ];
  const y = d3
    .scaleLinear()
    .domain(d3.extent(values))
    .nice()
    .range([height - m.bottom, m.top]);
  const svg = d3
    .select(el)
    .append("svg")
    .attr("width", width)
    .attr("height", height)
    .attr("role", "img")
    .attr(
      "aria-label",
      `National margin: truth ${fmt(points.at(-1).truth)} on election day, model ${fmt(points.at(-1).mean)}`,
    );

  svg
    .append("g")
    .attr("transform", `translate(${m.left},0)`)
    .call(
      d3
        .axisLeft(y)
        .ticks(5)
        .tickFormat((v) => (v === 0 ? "Even" : fmt(v).replace(".0", "")))
        .tickSize(-(width - m.left - m.right)),
    )
    .call((a) => a.select(".domain").remove())
    .call((a) =>
      a
        .selectAll(".tick line")
        .attr("stroke", (v) => css(v === 0 ? "--axis" : "--grid")),
    )
    .call((a) => a.selectAll(".tick text").attr("fill", css("--ink-2")));
  svg
    .append("g")
    .attr("transform", `translate(0,${height - m.bottom})`)
    .call(d3.axisBottom(x).ticks(width < 500 ? 4 : 7).tickSizeOuter(0))
    .call((a) => a.select(".domain").attr("stroke", css("--axis")))
    .call((a) => a.selectAll(".tick line").attr("stroke", css("--axis")))
    .call((a) => a.selectAll(".tick text").attr("fill", css("--ink-2")));

  svg
    .append("path")
    .datum(points)
    .attr("fill", css("--band"))
    .attr(
      "d",
      d3
        .area()
        .x((d) => x(d.t))
        .y0((d) => y(d.lo))
        .y1((d) => y(d.hi)),
    );
  svg
    .append("g")
    .selectAll("circle")
    .data(polls)
    .join("circle")
    .attr("cx", (d) => x(d.t))
    .attr("cy", (d) => y(d.margin))
    .attr("r", 3)
    .attr("fill", "none")
    .attr("stroke", css("--muted"))
    .attr("stroke-width", 1.5);
  const line = (key) =>
    d3
      .line()
      .x((d) => x(d.t))
      .y((d) => y(d[key]));
  svg
    .append("path")
    .datum(points)
    .attr("fill", "none")
    .attr("stroke", css("--ink"))
    .attr("stroke-width", 2)
    .attr("d", line("mean"));
  svg
    .append("path")
    .datum(points)
    .attr("fill", "none")
    .attr("stroke", css("--truth"))
    .attr("stroke-width", 2)
    .attr("d", line("truth"));

  const cross = svg
    .append("line")
    .attr("y1", m.top)
    .attr("y2", height - m.bottom)
    .attr("stroke", css("--axis"))
    .attr("visibility", "hidden");
  const bisect = d3.bisector((d) => d.t).center;
  svg
    .append("rect")
    .attr("x", m.left)
    .attr("y", m.top)
    .attr("width", width - m.left - m.right)
    .attr("height", height - m.top - m.bottom)
    .attr("fill", "transparent")
    .on("mousemove", (e) => {
      const d = points[bisect(points, x.invert(d3.pointer(e)[0]))];
      cross.attr("x1", x(d.t)).attr("x2", x(d.t)).attr("visibility", "visible");
      showTooltip(
        e,
        `<div class="title">Week of ${d.t.toLocaleDateString(undefined, { month: "short", day: "numeric", year: "numeric" })}</div>
        <div>Truth: ${fmt(d.truth)}</div>
        <div>Model: ${fmt(d.mean)} <span class="muted">(80%: ${fmt(d.lo)} to ${fmt(d.hi)})</span></div>`,
      );
    })
    .on("mouseleave", () => {
      cross.attr("visibility", "hidden");
      hideTooltip();
    });
}

/** One row per item: model 80% range, model estimate, truth, and optional extra markers. */
function dotPlot(id, rows, { tooltipHtml, extra }) {
  const el = document.getElementById(id);
  el.innerHTML = "";
  const width = el.clientWidth;
  const narrow = width < 560;
  const rowH = 40;
  const m = { top: 4, right: 12, bottom: 26, left: narrow ? 128 : 150 };
  const height = m.top + rows.length * rowH + m.bottom;
  const values = rows.flatMap((r) => [r.lo, r.hi, r.truth, r.poll_average ?? 0]);
  const bound = d3.max(values, Math.abs) * 1.05;
  const x = marginScale(-bound, bound, m.left, width - m.right).nice();
  const svg = d3
    .select(el)
    .append("svg")
    .attr("width", width)
    .attr("height", height);
  svg
    .append("g")
    .attr("transform", `translate(0,${height - m.bottom})`)
    .call((g) => marginAxis(g, x, height - m.top - m.bottom));
  svg
    .append("line")
    .attr("x1", x(0))
    .attr("x2", x(0))
    .attr("y1", m.top)
    .attr("y2", height - m.bottom)
    .attr("stroke", css("--axis"));

  const row = svg
    .selectAll("g.row")
    .data(rows)
    .join("g")
    .attr("class", "row")
    .attr("transform", (_, i) => `translate(0,${m.top + i * rowH + rowH / 2})`);
  row
    .append("rect")
    .attr("x", 0)
    .attr("y", -rowH / 2)
    .attr("width", width)
    .attr("height", rowH)
    .attr("fill", "transparent");
  row
    .append("text")
    .attr("x", 0)
    .attr("y", 4)
    .attr("fill", css("--ink"))
    .attr("font-size", 13)
    .text((r) => r.label);
  row
    .append("rect")
    .attr("x", (r) => x(r.hi))
    .attr("width", (r) => Math.max(2, x(r.lo) - x(r.hi)))
    .attr("y", -5)
    .attr("height", 10)
    .attr("rx", 4)
    .attr("fill", css("--band"));
  if (extra) extra(row, x);
  row
    .append("circle")
    .attr("cx", (r) => x(r.mean))
    .attr("r", 5)
    .attr("fill", css("--ink"))
    .attr("stroke", css("--surface"))
    .attr("stroke-width", 2);
  row
    .append("path")
    .attr("d", d3.symbol(d3.symbolDiamond, 90))
    .attr("transform", (r) => `translate(${x(r.truth)},0)`)
    .attr("fill", css("--truth"))
    .attr("stroke", css("--surface"))
    .attr("stroke-width", 2);
  row
    .on("mousemove", (e, r) => showTooltip(e, tooltipHtml(r)))
    .on("mouseleave", hideTooltip);
}

function renderRaces(s) {
  const rows = s.races.map((r) => ({
    ...r,
    label: `${r.label} · ${r.n_polls} poll${r.n_polls === 1 ? "" : "s"}`,
    name: r.label,
  }));
  dotPlot("races", rows, {
    extra: (row, x) =>
      row
        .filter((r) => r.poll_average !== null)
        .append("circle")
        .attr("cx", (r) => x(r.poll_average))
        .attr("r", 5)
        .attr("fill", "none")
        .attr("stroke", css("--muted"))
        .attr("stroke-width", 2),
    tooltipHtml: (r) => {
      const lean = r.pvi === 0 ? "EVEN" : r.pvi > 0 ? `D+${r.pvi}` : `R+${-r.pvi}`;
      const inc = r.incumbent_side ? ` · ${r.incumbent_side} incumbent` : " · open seat";
      const avg =
        r.poll_average === null
          ? "No polls"
          : `Poll average: ${fmt(r.poll_average)}`;
      const winner = r.truth > 0 ? "D" : "R";
      const pWinner = winner === "D" ? r.p_d : 1 - r.p_d;
      return `<div class="title">${r.name}</div>
        <div class="muted">PVI ${lean}${inc}</div>
        <div>Truth: ${fmt(r.truth)}</div>
        <div>Model: ${fmt(r.mean)} <span class="muted">(80%: ${fmt(r.lo)} to ${fmt(r.hi)})</span></div>
        <div>${avg}</div>
        <div>Model gave the winner (${winner}) ${pct(pWinner)}</div>`;
    },
  });
}

function renderPollsters(s) {
  const rows = s.pollsters.map((p) => ({ ...p, label: p.name }));
  dotPlot("pollsters", rows, {
    tooltipHtml: (p) => `<div class="title">${p.name}</div>
      <div>True lean: ${fmt(p.truth)}</div>
      <div>Model: ${fmt(p.mean)} <span class="muted">(80%: ${fmt(p.lo)} to ${fmt(p.hi)})</span></div>`,
  });
}

function renderSummary(s) {
  const inside = s.races.filter((r) => r.lo <= r.truth && r.truth <= r.hi).length;
  const called = s.races.filter((r) => r.p_d >= 0.5 === r.truth > 0).length;
  const seats = s.house_seats;
  document.getElementById("summary").textContent =
    `The true result fell inside the model's 80% range in ${inside} of ${s.races.length} races, ` +
    `and the model's favorite won ${called} of ${s.races.length}. ` +
    `Democrats won ${seats.truth} of 5 House seats; the model expected ${seats.mean} ` +
    `(80% range ${seats.lo}–${seats.hi}).`;
  document.getElementById("races-note").textContent =
    "Margin on election day. The bar is the model's 80% range, including a possible shared " +
    "polling miss.";
  document.getElementById("takeaway").textContent = SCENARIO_TEXT[state.scenario];
}

function render() {
  const s = state.data.scenarios[state.scenario];
  document
    .querySelectorAll("#scenarios button")
    .forEach((b) =>
      b.setAttribute("aria-selected", b.dataset.scenario === state.scenario),
    );
  renderSummary(s);
  renderNational(s);
  renderRaces(s);
  renderPollsters(s);
}

legend("national-legend", [
  ["truth", "Truth"],
  ["model", "Model estimate"],
  ["band", "Model 80% range"],
  ["poll", "Poll"],
]);
legend("races-legend", [
  ["truth", "Truth"],
  ["model", "Model estimate"],
  ["band", "Model 80% range"],
  ["poll", "Poll average"],
]);

document.querySelectorAll("#scenarios button").forEach((b) =>
  b.addEventListener("click", () => {
    state.scenario = b.dataset.scenario;
    render();
  }),
);
let resizeTimer;
window.addEventListener("resize", () => {
  clearTimeout(resizeTimer);
  resizeTimer = setTimeout(() => state.data && render(), 150);
});
window
  .matchMedia("(prefers-color-scheme: dark)")
  .addEventListener("change", () => state.data && render());

d3.json("data/demo.json")
  .then((data) => {
    state.data = data;
    render();
  })
  .catch((err) => {
    document.getElementById("summary").textContent =
      "Could not load the test data.";
    console.error(err);
  });
