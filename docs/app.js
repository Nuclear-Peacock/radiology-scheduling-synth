// docs/app.js
// Educational Use Only (Non-Clinical)
// Pure-JS neural network inference (no ONNX/WASM). Uses exported JSON weights.

function $(id){ return document.getElementById(id); }
function clamp(x,a,b){ return Math.max(a, Math.min(b,x)); }

async function fetchJson(path){
  const res = await fetch(path, { cache: "no-store" });
  if (!res.ok) throw new Error(`Failed to fetch ${path} (${res.status})`);
  return await res.json();
}

// ---- MLP inference (layers: [{W: [[...]], b:[...]}], ReLU between layers) ----
function reluVec(v){
  const out = new Array(v.length);
  for (let i=0;i<v.length;i++) out[i] = v[i] > 0 ? v[i] : 0;
  return out;
}
function linearForward(x, layer){
  const W = layer.W; // [out][in]
  const b = layer.b; // [out]
  const outDim = W.length;
  const y = new Array(outDim);
  for (let o=0;o<outDim;o++){
    const row = W[o];
    let s = b[o] || 0;
    for (let i=0;i<row.length;i++){
      s += row[i] * x[i];
    }
    y[o] = s;
  }
  return y;
}
function mlpForward(x, model){
  let h = x;
  for (let li=0; li<model.layers.length; li++){
    h = linearForward(h, model.layers[li]);
    if (li < model.layers.length - 1) h = reluVec(h);
  }
  return h;
}
function sigmoid(z){
  if (z >= 0) {
    const ez = Math.exp(-z);
    return 1 / (1 + ez);
  } else {
    const ez = Math.exp(z);
    return ez / (1 + ez);
  }
}

// ---- UI helpers ----
function setText(id, text){
  const el = $(id);
  if (el) el.textContent = text;
}

function readKnobs(){
  const get = (id, fallback) => ($(id) ? parseFloat($(id).value) : fallback);
  return {
    edFlux: get("edFlux", 1.0),
    inFlux: get("inFlux", 1.0),
    opFlux: get("opFlux", 1.0),

    mixXR: get("mixXR", 30),
    mixUS: get("mixUS", 20),
    mixCT: get("mixCT", 30),
    mixMR: get("mixMR", 20),

    ctDown: get("ctDown", 0.10),
    mrDown: get("mrDown", 0.10),

    transport: get("transport", 18),

    nsRate: get("nsRate", 0.08),
    overT: get("overT", 0.60),
    reserve: get("reserve", 0.10),
  };
}

function updateSliderLabels(){
  const pairs = [
    ["edFlux","v_edFlux"], ["inFlux","v_inFlux"], ["opFlux","v_opFlux"],
    ["mixXR","v_mixXR"], ["mixUS","v_mixUS"], ["mixCT","v_mixCT"], ["mixMR","v_mixMR"],
    ["ctDown","v_ctDown"], ["mrDown","v_mrDown"], ["transport","v_transport"],
    ["nsRate","v_ns"], ["overT","v_overT"], ["reserve","v_reserve"],
  ];
  for (const [sid, vid] of pairs){
    const s = $(sid);
    const v = $(vid);
    if (s && v) v.textContent = s.value;
  }
}

function setSlider(id, value){
  const el = $(id);
  if (!el) return;
  el.value = value;
}

function applyPreset(p){
  setSlider("edFlux", p.edFlux);
  setSlider("inFlux", p.inFlux);
  setSlider("opFlux", p.opFlux);

  setSlider("mixXR", p.mixXR);
  setSlider("mixUS", p.mixUS);
  setSlider("mixCT", p.mixCT);
  setSlider("mixMR", p.mixMR);

  setSlider("ctDown", p.ctDown);
  setSlider("mrDown", p.mrDown);
  setSlider("transport", p.transport);

  setSlider("nsRate", p.nsRate);
  setSlider("overT", p.overT);
  setSlider("reserve", p.reserve);

  updateSliderLabels();
}

const PRESETS = {
  normal: {
    edFlux: 1.00, inFlux: 1.00, opFlux: 1.00,
    mixXR: 30, mixUS: 20, mixCT: 30, mixMR: 20,
    ctDown: 0.10, mrDown: 0.10, transport: 18,
    nsRate: 0.08, overT: 0.60, reserve: 0.10,
  },
  winter: {
    edFlux: 1.35, inFlux: 1.20, opFlux: 0.95,
    mixXR: 28, mixUS: 18, mixCT: 36, mixMR: 18,
    ctDown: 0.12, mrDown: 0.10, transport: 28,
    nsRate: 0.09, overT: 0.55, reserve: 0.12,
  },
  mri_down: {
    edFlux: 1.05, inFlux: 1.05, opFlux: 1.00,
    mixXR: 30, mixUS: 18, mixCT: 33, mixMR: 19,
    ctDown: 0.10, mrDown: 0.25, transport: 18,
    nsRate: 0.08, overT: 0.60, reserve: 0.10,
  },
  ed_ct_surge: {
    edFlux: 1.60, inFlux: 1.05, opFlux: 0.90,
    mixXR: 26, mixUS: 18, mixCT: 40, mixMR: 16,
    ctDown: 0.10, mrDown: 0.10, transport: 20,
    nsRate: 0.08, overT: 0.55, reserve: 0.20,
  }
};

// ---- Charts (simple canvas bar charts) ----
function clearCanvas(c){
  const ctx = c.getContext("2d");
  ctx.clearRect(0, 0, c.width, c.height);
  return ctx;
}

function drawBarChart(canvasId, labels, values, opts = {}){
  const c = $(canvasId);
  if (!c) return;
  const ctx = clearCanvas(c);

  const W = c.width, H = c.height;
  const pad = 28;
  const maxV = Math.max(1e-6, ...(opts.max ? [opts.max] : values));
  const n = values.length;
  const gap = 10;
  const barW = (W - pad*2 - gap*(n-1)) / n;

  // baseline
  ctx.globalAlpha = 0.35;
  ctx.strokeStyle = "#23283a";
  ctx.beginPath();
  ctx.moveTo(pad, H - pad);
  ctx.lineTo(W - pad, H - pad);
  ctx.stroke();
  ctx.globalAlpha = 1;

  ctx.font = "12px system-ui";

  for (let i=0;i<n;i++){
    const v = values[i];
    const h = clamp(v / maxV, 0, 1) * (H - pad*2);
    const x = pad + i*(barW + gap);
    const y = (H - pad) - h;

    // bar
    ctx.globalAlpha = 0.9;
    ctx.fillStyle = "#2b6cff";
    ctx.fillRect(x, y, barW, h);

    // value label
    ctx.globalAlpha = 1;
    ctx.fillStyle = "#e8ecf3";
    const txt = opts.fmt ? opts.fmt(v) : String(v);
    ctx.fillText(txt, x, Math.max(12, y - 6));

    // x label
    ctx.globalAlpha = 0.85;
    ctx.fillStyle = "#b9c2d6";
    ctx.fillText(labels[i], x, H - 10);
    ctx.globalAlpha = 1;
  }
}

// ---- Educational scenario vector for the neural nets ----
function buildScenarioVector(knobs){
  const mixSum = Math.max(1e-6, knobs.mixXR + knobs.mixUS + knobs.mixCT + knobs.mixMR);
  const xr = knobs.mixXR / mixSum;
  const us = knobs.mixUS / mixSum;
  const ct = knobs.mixCT / mixSum;
  const mr = knobs.mixMR / mixSum;

  const v = [
    clamp((knobs.edFlux - 0.5) / 1.5, 0, 1),
    clamp((knobs.inFlux - 0.5) / 1.5, 0, 1),
    clamp((knobs.opFlux - 0.7) / 0.5, 0, 1),

    xr, us, ct, mr,

    clamp(knobs.ctDown / 0.3, 0, 1),
    clamp(knobs.mrDown / 0.3, 0, 1),

    clamp(knobs.transport / 60, 0, 1),

    clamp(knobs.nsRate / 0.25, 0, 1),
    clamp((knobs.overT - 0.2) / 0.75, 0, 1),
    clamp(knobs.reserve / 0.30, 0, 1),

    1.0,                // constant
    ct * knobs.edFlux,   // interactions
    mr * knobs.inFlux,
    knobs.ctDown * ct,
    knobs.mrDown * mr,
    knobs.nsRate * knobs.opFlux
  ];

  return v;
}

function padOrTruncate(vec, targetDim){
  const x = new Array(targetDim).fill(0);
  const n = Math.min(vec.length, targetDim);
  for (let i=0;i<n;i++) x[i] = vec[i];
  return x;
}

// ---- KPI computation (NN outputs used as nonlinear modifiers) ----
function computeKPIs(knobs, durationModel, noshowModel){
  const baseED = 180, baseIP = 140, baseOP = 220;

  const ED = baseED * knobs.edFlux;
  const IP = baseIP * knobs.inFlux;
  const OP = baseOP * knobs.opFlux;

  const mixSum = Math.max(1e-6, knobs.mixXR + knobs.mixUS + knobs.mixCT + knobs.mixMR);
  const mix = {
    XR: knobs.mixXR / mixSum,
    US: knobs.mixUS / mixSum,
    CT: knobs.mixCT / mixSum,
    MR: knobs.mixMR / mixSum
  };

  // inventory
  const scanners = { CT: 5, MR: 3, US: 2, XR: 9 };
  let availCT = scanners.CT * 1440 * (1 - clamp(knobs.ctDown, 0, 0.9));
  let availMR = scanners.MR * 1440 * (1 - clamp(knobs.mrDown, 0, 0.9));

  const scenario = buildScenarioVector(knobs);

  // duration NN -> multiplier
  const durInDim = durationModel.layers[0].W[0].length;
  const xDur = padOrTruncate(scenario, durInDim);
  const durOut = mlpForward(xDur, durationModel)[0];
  const durMultiplier = clamp(1 + durOut / 60, 0.6, 1.8);

  // no-show NN -> prob (blend with slider)
  const nsInDim = noshowModel.layers[0].W[0].length;
  const xNS = padOrTruncate(scenario, nsInDim);
  const nsLogit = mlpForward(xNS, noshowModel)[0];

  // Cap NN component to 25% to keep it believable for learners
  const nsNN = clamp(sigmoid(nsLogit), 0.01, 0.25);
  const nsBase = clamp(knobs.nsRate, 0, 0.40);
  const nsFrac = clamp(0.70 * nsBase + 0.30 * nsNN, 0.0, 0.40);

  const allowOverbook = nsFrac >= knobs.overT;
  const overbookExtra = allowOverbook ? Math.round(0.08 * OP) : 0;

  const baseDur = { XR: 10, US: 40, CT: 22, MR: 55 };
  const effDur = {
    XR: baseDur.XR * durMultiplier,
    US: baseDur.US * durMultiplier,
    CT: baseDur.CT * durMultiplier,
    MR: baseDur.MR * durMultiplier
  };

  const total = ED + IP + (OP + overbookExtra);
  const demandCT = total * mix.CT * effDur.CT;
  const demandMR = total * mix.MR * effDur.MR;

  // reserve affects OP access to CT
  const reserve = clamp(knobs.reserve, 0, 0.8);
  const opCTAvail = availCT * (1 - reserve);

  const ctUtil = clamp(demandCT / Math.max(1, availCT), 0, 1.5);
  const mrUtil = clamp(demandMR / Math.max(1, availMR), 0, 1.5);

  // waits
  const edTTS = 18 + (ctUtil > 0.6 ? (ctUtil - 0.6) * 120 : 0) + (ctUtil > 0.9 ? (ctUtil - 0.9) * 350 : 0);
  const ipTTS = 45 + knobs.transport + (ctUtil > 0.6 ? (ctUtil - 0.6) * 140 : 0) + (ctUtil > 0.9 ? (ctUtil - 0.9) * 350 : 0);

  const opCTPressure = clamp(demandCT / Math.max(1, opCTAvail), 0, 2);
  const opOnTime = clamp(
    0.93 - (ctUtil - 0.6) * 0.35 - (overbookExtra > 0 ? 0.06 : 0) - Math.max(0, opCTPressure - 0.9) * 0.10,
    0.20, 0.98
  );

  const overtime = Math.max(0, demandCT - availCT) / Math.max(1, scanners.CT)
                + Math.max(0, demandMR - availMR) / Math.max(1, scanners.MR);

  const idle = Math.max(0, availCT - demandCT) / Math.max(1, scanners.CT)
            + Math.max(0, availMR - demandMR) / Math.max(1, scanners.MR);

  const bumps = Math.round((overbookExtra / Math.max(1, OP)) * 12 + Math.max(0, ctUtil - 0.85) * 14);

  const opBooked = OP + overbookExtra;
  const throughput = Math.round(ED + IP + opBooked * (1 - nsFrac));

  return {
    edTTS, ipTTS,
    opOnTimePct: opOnTime * 100,
    ctUtilPct: ctUtil * 100,
    mrUtilPct: mrUtil * 100,
    overtimeMin: overtime,
    idleMin: idle,
    nsPct: nsFrac * 100,
    bumpsPerDay: bumps,
    throughput,
    durMultiplier,
    nsNN_pct: nsNN * 100,
    volumes: { ED, IP, OP },
    overbookExtra
  };
}

document.addEventListener("DOMContentLoaded", async () => {
  const statusEl = $("status");
  updateSliderLabels();

  // keep labels updating
  const ids = ["edFlux","inFlux","opFlux","mixXR","mixUS","mixCT","mixMR","ctDown","mrDown","transport","nsRate","overT","reserve"];
  ids.forEach(id => { const el = $(id); if (el) el.addEventListener("input", updateSliderLabels); });

  // wire presets
  $("preset_normal")?.addEventListener("click", () => { applyPreset(PRESETS.normal); $("runBtn").click(); });
  $("preset_winter")?.addEventListener("click", () => { applyPreset(PRESETS.winter); $("runBtn").click(); });
  $("preset_mri_down")?.addEventListener("click", () => { applyPreset(PRESETS.mri_down); $("runBtn").click(); });
  $("preset_ed_surge")?.addEventListener("click", () => { applyPreset(PRESETS.ed_ct_surge); $("runBtn").click(); });

  let durationModel = null;
  let noshowModel = null;

  try {
    statusEl.textContent = "Loading JS neural network weights…";
    durationModel = await fetchJson("./models/duration_mlp.json");
    noshowModel = await fetchJson("./models/noshow_mlp.json");
    statusEl.textContent =
`Loaded JS neural nets ✅
• duration_mlp.json layers: ${durationModel.layers.length}
• noshow_mlp.json layers: ${noshowModel.layers.length}
Educational Use Only (Non-Clinical).`;
  } catch (e) {
    statusEl.textContent = `FAILED to load JS models: ${e.message}\nCheck docs/models/*.json`;
  }

  function run(){
    const knobs = readKnobs();
    const sum = knobs.mixXR + knobs.mixUS + knobs.mixCT + knobs.mixMR;
    if (Math.abs(sum - 100) > 5) {
      alert(`Modality mix sums to ${sum}%. Please adjust closer to 100% for best results.`);
    }
    if (!durationModel || !noshowModel) {
      statusEl.textContent = "Models not loaded. Check docs/models/*.json exist and refresh.";
      return;
    }

    const k = computeKPIs(knobs, durationModel, noshowModel);

    // KPIs
    setText("k_edTTS", `${Math.round(k.edTTS)} min`);
    setText("k_inTTS", `${Math.round(k.ipTTS)} min`);
    setText("k_opOnTime", `${k.opOnTimePct.toFixed(1)}%`);
    setText("k_ctUtil", `${k.ctUtilPct.toFixed(1)}%`);
    setText("k_mrUtil", `${k.mrUtilPct.toFixed(1)}%`);
    setText("k_overtime", `${Math.round(k.overtimeMin)} min`);
    setText("k_idle", `${Math.round(k.idleMin)} min`);
    setText("k_ns", `${k.nsPct.toFixed(1)}%`);
    setText("k_bumps", `${k.bumpsPerDay}`);
    setText("k_throughput", `${k.throughput}`);

    // Charts
    drawBarChart(
      "chartUtil",
      ["CT util", "MR util"],
      [k.ctUtilPct, k.mrUtilPct],
      { max: 120, fmt: v => `${v.toFixed(1)}%` }
    );

    drawBarChart(
      "chartWait",
      ["ED TTS", "IP TTS"],
      [k.edTTS, k.ipTTS],
      { max: 240, fmt: v => `${Math.round(v)}m` }
    );

    drawBarChart(
      "chartOps",
      ["On-time%", "No-show%", "Overtime", "Idle"],
      [k.opOnTimePct, k.nsPct, k.overtimeMin, k.idleMin],
      { max: 240, fmt: v => (v > 100 ? `${Math.round(v)}m` : `${v.toFixed(1)}`) }
    );

    // Status: 3–5 interpretive lines
    statusEl.textContent =
`Models loaded ✅ | Last run: ${new Date().toLocaleTimeString()}
Volumes (ED/IP/OP): ${k.volumes.ED.toFixed(0)} / ${k.volumes.IP.toFixed(0)} / ${k.volumes.OP.toFixed(0)}
Mix (XR/US/CT/MR): ${knobs.mixXR}% / ${knobs.mixUS}% / ${knobs.mixCT}% / ${knobs.mixMR}%
Downtime (CT/MR): ${(knobs.ctDown*100).toFixed(0)}% / ${(knobs.mrDown*100).toFixed(0)}% | ED CT reserve: ${(knobs.reserve*100).toFixed(0)}%
NN effects: duration×${k.durMultiplier.toFixed(2)} | no-show NN component ${k.nsNN_pct.toFixed(1)}% | overbook +${k.overbookExtra}
Educational Use Only (Non-Clinical).`;
  }

  $("runBtn")?.addEventListener("click", run);

  // first run
  if (durationModel && noshowModel) run();
});
