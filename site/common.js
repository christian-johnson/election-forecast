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
    ${safeUrl(p.url) ? `<div class="muted">${tapOrClick()} to open the poll</div>` : ""}`;
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

// Touch screens have no hover: the first tap on an item shows its tooltip, a tap on the tooltip
// (or the item again) acts on it, and a tap elsewhere or a scroll clears it.
let pointerType = "mouse";
let tooltipOwner = null;
let onTooltipHide = null;
let onTooltipTap = null;
const isTouch = () => pointerType === "touch";
const tapOrClick = () => (isTouch() ? "Tap here" : "Click");
document.addEventListener(
  "pointerdown",
  (e) => {
    pointerType = e.pointerType;
    if (
      isTouch() &&
      tooltipOwner &&
      !tooltipOwner.contains(e.target) &&
      !tooltip.contains(e.target)
    )
      hideTooltip();
  },
  true,
);
document.addEventListener("scroll", () => isTouch() && hideTooltip(), {
  capture: true,
  passive: true,
});
tooltip.addEventListener("click", () => {
  onTooltipTap?.();
  hideTooltip();
});

/**
 * Show the tooltip by the pointer; `onHide` undoes any highlight when it goes away, and `onTap`
 * runs when the tooltip is tapped on a touch screen.
 */
function showTooltip(event, html, onHide = null, onTap = null) {
  tooltip.innerHTML = html;
  tooltip.hidden = false;
  tooltipOwner = event.currentTarget;
  onTooltipHide = onHide;
  onTooltipTap = onTap;
  tooltip.classList.toggle("tappable", isTouch() && onTap !== null);
  // Above a finger, so it isn't covered; below a mouse pointer unless that runs off the screen.
  const pad = isTouch() ? 28 : 14;
  const { width, height } = tooltip.getBoundingClientRect();
  const x = Math.min(event.clientX + (isTouch() ? -width / 2 : pad), window.innerWidth - width - 4);
  const above = event.clientY - height - pad;
  const below = event.clientY + pad;
  const y =
    (isTouch() && above >= 4) || below + height > window.innerHeight ? Math.max(4, above) : below;
  tooltip.style.left = `${Math.max(4, x)}px`;
  tooltip.style.top = `${y}px`;
}

function hideTooltip() {
  tooltip.hidden = true;
  const undo = onTooltipHide;
  tooltipOwner = onTooltipHide = onTooltipTap = null;
  undo?.();
}

/**
 * Give each element of a d3 selection a tooltip: on hover with a mouse, or on the first tap with
 * touch. A click, a tap on the tooltip, or a second tap goes through to `onClick` and to any link.
 *
 * Args:
 *   selection: d3 selection of the elements, bound to their data.
 *   html: Function from a datum to the tooltip's markup.
 *   options.onShow: Called with (event, datum) as the tooltip shows, to highlight the element.
 *   options.onHide: Called when the tooltip goes away, to undo the highlight.
 *   options.onClick: Called with (event, datum) on a click or a second tap.
 */
function bindTooltip(selection, html, { onShow, onHide, onClick } = {}) {
  const show = (e, d) => {
    const target = e.currentTarget;
    onShow?.(e, d);
    // Replaying the click on the item reaches `onClick` and follows any link.
    showTooltip(e, html(d), onHide, () =>
      target.dispatchEvent(new MouseEvent("click", { bubbles: true, cancelable: true })),
    );
  };
  selection
    .on("pointermove", (e, d) => e.pointerType === "mouse" && show(e, d))
    .on("pointerleave", (e) => e.pointerType === "mouse" && hideTooltip())
    .on("click", (e, d) => {
      if (isTouch() && (tooltip.hidden || tooltipOwner !== e.currentTarget)) {
        e.preventDefault();
        show(e, d);
      } else onClick?.(e, d);
    });
}

/** Call `step(+1)` on a swipe left across `el` and `step(-1)` on a swipe right. */
function onSwipe(el, step) {
  let start = null;
  el.addEventListener(
    "touchstart",
    (e) => {
      start = e.touches.length === 1 ? e.touches[0] : null;
    },
    { passive: true },
  );
  el.addEventListener("touchend", (e) => {
    if (!start) return;
    const end = e.changedTouches[0];
    const dx = end.clientX - start.clientX;
    const dy = end.clientY - start.clientY;
    start = null;
    // Mostly sideways, so a vertical scroll that drifts doesn't change the step.
    if (Math.abs(dx) > 50 && Math.abs(dx) > 2 * Math.abs(dy)) step(dx < 0 ? 1 : -1);
  });
}
