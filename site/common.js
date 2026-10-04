"use strict";

// Shared by the forecast and about pages: colors, formatting, tooltips, and chart axes.

const tooltip = document.getElementById("tooltip");

const css = (name) =>
  getComputedStyle(document.documentElement).getPropertyValue(name).trim();
const pct = (p) =>
  p > 0.99 ? ">99%" : p < 0.01 ? "<1%" : `${Math.round(p * 100)}%`;

/**
 * Horizontal scale for margins (D-side points), with Democrats on the left and Republicans on
 * the right, as everywhere on the site.
 */
const marginScale = (lo, hi, left, right) =>
  d3.scaleLinear().domain([hi, lo]).range([left, right]);

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
