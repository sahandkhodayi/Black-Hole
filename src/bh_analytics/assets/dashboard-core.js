const $ = (id) => document.getElementById(id);
let experiments = [];
let selected = null;
let trainingSnapshot = null;
let trainingPollTimer = null;
const TRAINING_POLL_MS = 60_000;

const esc = (value) => String(value ?? "—").replace(/[&<>"']/g, (char) => ({
  "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;"
})[char]);

const isNumber = (value) => typeof value === "number" && Number.isFinite(value);
const fmt = (value, digits = 3) => isNumber(value)
  ? value.toLocaleString(undefined, { maximumSignificantDigits: digits })
  : "Not recorded";

async function api(url, options) {
  const response = await fetch(url, options);
  const data = await response.json();
  if (!response.ok) throw new Error(data.error || response.statusText);
  return data;
}

function mean(rows, key) {
  const values = (rows || []).map((row) => row[key]).filter(isNumber);
  return values.length ? values.reduce((sum, value) => sum + value, 0) / values.length : null;
}
