const $ = (sel) => document.querySelector(sel);

const btnAttack = $("#btn-attack");
const btnDefense = $("#btn-defense");
const btnStop = $("#btn-stop");
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
const chartsGrid = $("#charts-grid");

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
const chartCanvases = {};

function drawMatrix() {
  const canvas = $("#matrix");
  const ctx = canvas.getContext("2d");
  const chars = "01アイウエオｱｲｳｴｵATTACKDEFENSECAMROI";
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

function initCharts() {
  chartsGrid.innerHTML = "";
  ACTION_KEYS.forEach((key, idx) => {
    const wrap = document.createElement("div");
    wrap.className = "chart-card";
    const label = document.createElement("span");
    label.className = "chart-label";
    label.textContent = key;
    const canvas = document.createElement("canvas");
    canvas.width = 220;
    canvas.height = 72;
    wrap.append(label, canvas);
    chartsGrid.appendChild(wrap);
    chartCanvases[key] = { canvas, color: CHART_COLORS[idx % CHART_COLORS.length] };
  });
}

function drawSeries(canvas, values, color) {
  const ctx = canvas.getContext("2d");
  const w = canvas.width;
  const h = canvas.height;
  const pad = 6;
  ctx.clearRect(0, 0, w, h);
  ctx.fillStyle = "rgba(0, 20, 12, 0.55)";
  ctx.fillRect(0, 0, w, h);

  const finite = values.filter((v) => Number.isFinite(v));
  if (finite.length < 2) return;

  let min = Math.min(...finite);
  let max = Math.max(...finite);
  if (Math.abs(max - min) < 1e-9) {
    min -= 1;
    max += 1;
  }

  ctx.strokeStyle = color;
  ctx.lineWidth = 1.5;
  ctx.beginPath();
  finite.forEach((v, i) => {
    const x = pad + (i / (finite.length - 1)) * (w - pad * 2);
    const y = h - pad - ((v - min) / (max - min)) * (h - pad * 2);
    if (i === 0) ctx.moveTo(x, y);
    else ctx.lineTo(x, y);
  });
  ctx.stroke();

  const last = finite[finite.length - 1];
  ctx.fillStyle = color;
  ctx.font = "10px monospace";
  ctx.fillText(last.toFixed(3), pad, 12);
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

function setRunning(mode, pid) {
  isRunning = true;
  btnAttack.disabled = true;
  btnDefense.disabled = true;
  btnStop.disabled = false;
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
  btnAttack.disabled = false;
  btnDefense.disabled = false;
  btnStop.disabled = true;
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

    const cams = data.cameras || {};
    const attn = data.attention || {};
    const warnings = data.warnings || {};

    setImg(camFront, cams.front);
    if (attn.side) {
      setImg(camSide, attn.side);
      sideCaption.textContent = warnings.side ? "SIDE · WARNING" : "SIDE · CAM";
      sideCaption.classList.toggle("warn", Boolean(warnings.side));
    } else {
      setImg(camSide, cams.side);
      sideCaption.textContent = "SIDE";
      sideCaption.classList.remove("warn");
    }

    const series = data.series || {};
    ACTION_KEYS.forEach((key) => {
      const chart = chartCanvases[key];
      if (!chart) return;
      drawSeries(chart.canvas, series[key] || [], chart.color);
    });
  } catch (err) {
    console.debug(err);
  }
}

async function pollStatus() {
  try {
    const data = await api("/api/status");
    renderLogs(data.logs || []);
    if (data.running) {
      setRunning(data.mode, data.pid);
    } else {
      setIdle();
    }
  } catch (err) {
    console.error(err);
  }
}

async function startMode(mode) {
  try {
    const data = await api("/api/start", { mode });
    appendTerminal(`\n>>> ${mode.toUpperCase()} session started (pid ${data.pid})\n`);
    await pollStatus();
  } catch (err) {
    appendTerminal(`\n[ERROR] ${err.message}\n`);
    alert(err.message);
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
btnDefense.addEventListener("click", () => startMode("defense"));
btnStop.addEventListener("click", stopSession);

drawMatrix();
initCharts();
pollStatus();
pollTimer = setInterval(pollStatus, 1200);
