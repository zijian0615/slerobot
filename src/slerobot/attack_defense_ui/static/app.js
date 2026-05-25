const $ = (sel) => document.querySelector(sel);

const btnAttack = $("#btn-attack");
const btnDetect = $("#btn-detect");
const btnMitigation = $("#btn-mitigation");
const btnStop = $("#btn-stop");
const btnHomesetOnly = $("#btn-homeset-only");
const chkHomeset = $("#chk-homeset");
const repoLabel = $("#repo-label");
const statusPill = $("#status-pill");
const modeLabel = $("#mode-label");
const pidLabel = $("#pid-label");
const stepLabel = $("#step-label");
const terminal = $("#terminal");
const logMeta = $("#log-meta");
const feedMeta = $("#feed-meta");
const camFront = $("#cam-front");
const camSide = $("#cam-side");
const sideCaption = $("#side-caption");
const camOverlaySection = $("#cam-overlay-section");
const camFrontCam = $("#cam-front-cam");
const camSideCam = $("#cam-side-cam");
const frontCamCaption = $("#front-cam-caption");
const sideCamCaption = $("#side-cam-caption");
const actionChart = $("#action-chart");
const chartLegend = $("#chart-legend");

const MODE_BUTTONS = [btnAttack, btnDetect, btnMitigation];

let currentMode = null;

const ACTION_KEYS = ["j0", "j1", "j2", "j3", "j4", "j5", "j7"];
const CHART_COLORS = [
  "#00ff88",
  "#2ab0ff",
  "#ff2a4a",
  "#ffd24a",
  "#c86bff",
  "#ff8c42",
  "#7fffd4",
];

let pollTimer = null;
let telemetryTimer = null;
let isRunning = false;
let lastSeries = null;
let chartResizeObserver = null;

function drawMatrix() {
  const canvas = $("#matrix");
  const ctx = canvas.getContext("2d");
  const chars = "01アイウエオATTACKDETECTMITIGATIONCAMROI";
  let cols;
  let drops;

  function resize() {
    canvas.width = window.innerWidth;
    canvas.height = window.innerHeight;
    cols = Math.floor(canvas.width / 14);
    drops = Array(cols).fill(0);
  }

  function tick() {
    ctx.fillStyle = "rgba(3, 8, 6, 0.08)";
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    ctx.fillStyle = "#00ff66";
    ctx.font = "12px monospace";
    for (let i = 0; i < cols; i++) {
      const ch = chars[Math.floor(Math.random() * chars.length)];
      const x = i * 14;
      const y = drops[i] * 14;
      ctx.fillText(ch, x, y);
      if (y > canvas.height && Math.random() > 0.975) drops[i] = 0;
      drops[i]++;
    }
    requestAnimationFrame(tick);
  }

  resize();
  window.addEventListener("resize", resize);
  tick();
}

function resizeActionChart() {
  const wrap = actionChart.parentElement;
  if (!wrap) return;
  const rect = wrap.getBoundingClientRect();
  const legendH = chartLegend.offsetHeight || 24;
  const h = Math.max(100, rect.height - legendH - 8);
  const w = Math.max(120, rect.width);
  const dpr = window.devicePixelRatio || 1;
  actionChart.width = Math.floor(w * dpr);
  actionChart.height = Math.floor(h * dpr);
  actionChart.style.width = `${w}px`;
  actionChart.style.height = `${h}px`;
  const ctx = actionChart.getContext("2d");
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  if (lastSeries) {
    drawCombinedChart(lastSeries);
  }
}

function normalizeSeries(values) {
  const finite = values.filter((v) => Number.isFinite(v));
  if (finite.length === 0) return [];
  let min = Math.min(...finite);
  let max = Math.max(...finite);
  if (Math.abs(max - min) < 1e-9) {
    min -= 1;
    max += 1;
  }
  return finite.map((v) => (v - min) / (max - min));
}

function chartMutedColor() {
  return getComputedStyle(document.documentElement).getPropertyValue("--muted").trim() || "#5f9f78";
}

function drawCombinedChart(series) {
  const ctx = actionChart.getContext("2d");
  const w = actionChart.clientWidth;
  const h = actionChart.clientHeight;
  const padL = 28;
  const padR = 8;
  const padT = 10;
  const padB = 16;

  ctx.clearRect(0, 0, w, h);
  ctx.fillStyle = "rgba(0, 20, 12, 0.55)";
  ctx.fillRect(0, 0, w, h);

  ctx.strokeStyle = "rgba(0, 255, 120, 0.12)";
  ctx.lineWidth = 1;
  for (let i = 0; i <= 4; i++) {
    const y = padT + (i / 4) * (h - padT - padB);
    ctx.beginPath();
    ctx.moveTo(padL, y);
    ctx.lineTo(w - padR, y);
    ctx.stroke();
  }

  const legendItems = [];

  ACTION_KEYS.forEach((key, idx) => {
    const raw = series[key] || [];
    const norm = normalizeSeries(raw);
    if (norm.length < 2) return;

    const color = CHART_COLORS[idx % CHART_COLORS.length];
    ctx.strokeStyle = color;
    ctx.lineWidth = 1.6;
    ctx.beginPath();
    norm.forEach((v, i) => {
      const x = padL + (i / (norm.length - 1)) * (w - padL - padR);
      const y = padT + (1 - v) * (h - padT - padB);
      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    });
    ctx.stroke();

    const lastRaw = raw.filter((v) => Number.isFinite(v)).pop();
    legendItems.push({ key, color, last: lastRaw });
  });

  ctx.fillStyle = chartMutedColor();
  ctx.font = "9px monospace";
  ctx.fillText("0", 4, h - padB);
  ctx.fillText("1", 4, padT + 8);

  chartLegend.innerHTML = legendItems
    .map(
      ({ key, color, last }) =>
        `<span class="legend-item"><span class="legend-swatch" style="background:${color}"></span>${key} ${last != null ? last.toFixed(2) : "—"}</span>`,
    )
    .join("");
}

function initCharts() {
  resizeActionChart();
  if (chartResizeObserver) {
    chartResizeObserver.disconnect();
  }
  chartResizeObserver = new ResizeObserver(() => resizeActionChart());
  chartResizeObserver.observe(actionChart.parentElement);
  window.addEventListener("resize", resizeActionChart);
}

function setImg(el, b64) {
  if (!b64) return;
  el.src = `data:image/jpeg;base64,${b64}`;
}

async function api(path, body) {
  const res = await fetch(path, {
    method: body ? "POST" : "GET",
    headers: body ? { "Content-Type": "application/json" } : {},
    body: body ? JSON.stringify(body) : undefined,
  });
  const data = await res.json();
  if (!res.ok || data.ok === false) {
    throw new Error(data.error || `HTTP ${res.status}`);
  }
  return data;
}

function applyViewLayout(mode) {
  currentMode = mode;
  camOverlaySection.classList.toggle("hidden", mode !== "detect");
  requestAnimationFrame(resizeActionChart);
}

function updateRepoPreview(data) {
  if (!data) return;
  const idx = data.next_repo_index ?? data.repo_index;
  const repos = data.next_repo_ids;
  if (repos && idx != null) {
    repoLabel.textContent = `#${idx} · ${repos.attack} | ${repos.detect} | ${repos.mitigation}`;
  } else if (data.repo_id) {
    repoLabel.textContent = data.repo_id;
  }
}

function setBusyHomeset(busy) {
  const disabled = Boolean(busy);
  btnHomesetOnly.disabled = disabled;
  chkHomeset.disabled = disabled;
  if (!isRunning) {
    MODE_BUTTONS.forEach((btn) => {
      btn.disabled = disabled;
    });
  }
}

function setRunning(mode, pid) {
  isRunning = true;
  applyViewLayout(mode);
  MODE_BUTTONS.forEach((btn) => {
    btn.disabled = true;
  });
  btnStop.disabled = false;
  btnHomesetOnly.disabled = true;
  statusPill.textContent = mode ? mode.toUpperCase() : "RUNNING";
  statusPill.className = `pill running ${mode || ""}`;
  modeLabel.textContent = mode ? `MODE · ${mode.toUpperCase()}` : "";
  pidLabel.textContent = pid ? `PID ${pid}` : "";
  if (!telemetryTimer) {
    telemetryTimer = setInterval(pollTelemetry, 100);
  }
}

function setIdle() {
  isRunning = false;
  currentMode = null;
  camOverlaySection.classList.add("hidden");
  MODE_BUTTONS.forEach((btn) => {
    btn.disabled = false;
  });
  btnStop.disabled = true;
  btnHomesetOnly.disabled = false;
  chkHomeset.disabled = false;
  statusPill.textContent = "STANDBY";
  statusPill.className = "pill idle";
  modeLabel.textContent = "—";
  pidLabel.textContent = "";
  stepLabel.textContent = "step —";
  feedMeta.textContent = "waiting…";
  if (telemetryTimer) {
    clearInterval(telemetryTimer);
    telemetryTimer = null;
  }
}

function renderLogs(lines) {
  terminal.textContent = lines.join("\n");
  terminal.scrollTop = terminal.scrollHeight;
  logMeta.textContent = `${lines.length} lines`;
}

async function pollTelemetry() {
  try {
    const data = await api("/api/telemetry/snapshot");
    if (data.step) {
      stepLabel.textContent = `step ${data.step}`;
      feedMeta.textContent = data.mode ? `${data.mode.toUpperCase()} · live` : "live";
    }

    const mode = data.mode || currentMode;
    if (mode && mode !== currentMode) {
      applyViewLayout(mode);
    }

    const cams = data.cameras || {};
    const attn = data.attention || {};
    const warnings = data.warnings || {};

    setImg(camFront, cams.front);
    setImg(camSide, cams.side);

    if (mode === "mitigation") {
      sideCaption.textContent = warnings.side ? "SIDE · WARNING" : "SIDE";
      sideCaption.classList.toggle("warn", Boolean(warnings.side));
    } else {
      sideCaption.textContent = "SIDE";
      sideCaption.classList.remove("warn");
    }

    if (mode === "detect") {
      setImg(camFrontCam, attn.front);
      setImg(camSideCam, attn.side);
      frontCamCaption.textContent = "FRONT · GRAD-CAM";
      sideCamCaption.textContent = warnings.side ? "SIDE · GRAD-CAM · WARNING" : "SIDE · GRAD-CAM";
      sideCamCaption.classList.toggle("warn", Boolean(warnings.side));
    }

    lastSeries = data.series || {};
    drawCombinedChart(lastSeries);
  } catch (err) {
    console.debug(err);
  }
}

async function pollStatus() {
  try {
    const data = await api("/api/status");
    renderLogs(data.logs || []);
    updateRepoPreview(data);
    if (data.homeset_running) {
      setBusyHomeset(true);
    } else if (data.running) {
      setRunning(data.mode, data.pid);
    } else {
      setIdle();
    }
  } catch (err) {
    console.error(err);
  }
}

async function startMode(mode) {
  const homeset = chkHomeset.checked;
  try {
    setBusyHomeset(true);
    appendTerminal(
      `\n>>> ${mode.toUpperCase()}${homeset ? " (home set first)" : ""} starting…\n`,
    );
    const data = await api("/api/start", { mode, homeset });
    appendTerminal(
      `\n>>> ${mode.toUpperCase()} started · repo ${data.repo_id} · pid ${data.pid}\n`,
    );
    updateRepoPreview(data);
    await pollStatus();
  } catch (err) {
    appendTerminal(`\n[ERROR] ${err.message}\n`);
    alert(err.message);
    setIdle();
    await pollStatus();
  }
}

async function runHomesetOnly() {
  try {
    setBusyHomeset(true);
    appendTerminal("\n>>> Running home set (moveLinear.py)…\n");
    await api("/api/homeset", {});
    appendTerminal("\n>>> Home set done\n");
    await pollStatus();
  } catch (err) {
    appendTerminal(`\n[ERROR] ${err.message}\n`);
    alert(err.message);
    setIdle();
    await pollStatus();
  }
}

async function stopSession() {
  try {
    const data = await api("/api/stop", {});
    appendTerminal(`\n>>> ${data.message || "Stopped"}\n`);
    await pollStatus();
  } catch (err) {
    appendTerminal(`\n[ERROR] ${err.message}\n`);
    alert(err.message);
  }
}

function appendTerminal(text) {
  terminal.textContent += text;
  terminal.scrollTop = terminal.scrollHeight;
}

btnAttack.addEventListener("click", () => startMode("attack"));
btnDetect.addEventListener("click", () => startMode("detect"));
btnMitigation.addEventListener("click", () => startMode("mitigation"));
btnStop.addEventListener("click", stopSession);
btnHomesetOnly.addEventListener("click", runHomesetOnly);

drawMatrix();
initCharts();
pollStatus();
pollTimer = setInterval(pollStatus, 1200);
