"use strict";

// Shared by the forecast and about pages: colors, formatting, tooltips, and chart axes.

const tooltip = document.getElementById("tooltip");

const css = (name) =>
  getComputedStyle(document.documentElement).getPropertyValue(name).trim();
const pct = (p) =>
  p > 0.99 ? ">99%" : p < 0.01 ? "<1%" : `${Math.round(p * 100)}%`;
const fmt = (m) =>
  Math.abs(m) < 0.05
    ? "Even"
    : m > 0
      ? `D+${m.toFixed(1)}`
      : `R+${(-m).toFixed(1)}`;
const esc = (s) =>
  String(s).replace(
    /[&<>"']/g,
    (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[c],
  );
const safeUrl = (u) => (/^https?:\/\//.test(u ?? "") ? u : null);
const shortDate = (s) =>
  new Date(`${s}T12:00`).toLocaleDateString(undefined, {
    month: "short",
    day: "numeric",
    year: "numeric",
  });

// Half-width of a normal 80% interval in SDs, matching the 80% ranges shown everywhere.
const Z80 = 1.2816;
// D-side margin in points from a logit of the D-side two-party share.
const logitMargin = (logit) => (2 / (1 + Math.exp(-logit)) - 1) * 100;
// Logit units per point of margin near a 50-50 race (config.POINT).
const POINT = 0.02;
// Length of one step of the national random walk, as stored in the weekly estimates.
const WEEK_MS = 7 * 864e5;

const POPULATION_NAME = {
  lv: "Likely voters",
  rv: "Registered voters",
  v: "Voters",
  a: "Adults",
};
// The model assumes this sample size when a pollster reports none (config.DEFAULT_SAMPLE_SIZE),
// and caps larger ones (config.MAX_SAMPLE_SIZE).
const ASSUMED_N = 600;
const MAX_N = 3000;

function pollTooltip(p) {
  const n = p.n === null ? `not reported (assumed ${ASSUMED_N})` : p.n.toLocaleString();
  const sponsor = p.partisan
    ? `<div class="muted">Sponsored by ${p.partisan === "D" ? "Democrats" : "Republicans"}</div>`
    : "";
  return `<div class="title">${esc(p.pollster)}</div>
    <div class="muted">${shortDate(p.start)} to ${shortDate(p.end)}</div>
    <div>${POPULATION_NAME[p.population] ?? esc(p.population)}, n = ${n}</div>
    ${sponsor}
    <div>D ${p.dem}% · R ${p.rep}% (two-party ${fmt(p.margin)})</div>
    ${safeUrl(p.url) ? '<div class="muted">Click to open the poll</div>' : ""}`;
}

/**
 * Horizontal scale for margins (D-side points), with Democrats on the left and Republicans on
 * the right, as everywhere on the site.
 */
const marginScale = (lo, hi, left, right) =>
  d3.scaleLinear().domain([hi, lo]).range([left, right]);

// Bins of a margin histogram stored as counts per bin, `width` points wide from `start`.
const marginBins = (hist, start, width) =>
  hist.map((v, i) => ({ i, v, x0: start + i * width, x1: start + (i + 1) * width }));
const marginColor = (b) => css((b.x0 + b.x1) / 2 >= 0 ? "--dem" : "--rep");

function styleAxis(axis) {
  axis.select(".domain").attr("stroke", css("--axis"));
  axis.selectAll(".tick line").attr("stroke", css("--axis"));
  axis.selectAll(".tick text").attr("fill", css("--ink-2"));
}

function showTooltip(event, html) {
  tooltip.innerHTML = html;
  tooltip.hidden = false;
  const pad = 14;
  const { width, height } = tooltip.getBoundingClientRect();
  const x = Math.min(event.clientX + pad, window.innerWidth - width - 4);
  const y =
    event.clientY + pad + height > window.innerHeight
      ? event.clientY - height - pad
      : event.clientY + pad;
  tooltip.style.left = `${Math.max(4, x)}px`;
  tooltip.style.top = `${y}px`;
}

const hideTooltip = () => (tooltip.hidden = true);
