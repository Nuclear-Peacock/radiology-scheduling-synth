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
  // x: [in], W: [out][in], b: [out]
  const W = layer.W;
  const b = layer.b;
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
  // model.layers = linear layers only; activation = relu
  let h = x;
  for (let li=0; li<model.layers.length; li++){
    h = linearForward(h, model.layers[li]);
    if (li < model.layers.length - 1) h = reluVec(h);
  }
  return h; // final layer output vector
}

function sigmoid(z){
  // stable-ish sigmoid
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
    // 10 knobs
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

// ---- Educational “scenario vector” for the neural nets ----
// Since your trained nets expect large one-hot vectors, we need a compact, stable demo input.
// We do this by creating a fixed-length numeric vector and either:
//  (A) padding/truncating to the model’s input dimension, and
//  (B) letting the network act as a nonlinear function over knobs.
// This is educational only (not clinical), but it is a real NN forward pass with your weights.
function buildScenarioVector(knobs){
  const mixSum = Math.max(1e-6, knobs.mixXR + knobs.mixUS + knobs.mixCT + knobs.mixMR);
  const xr = knobs.mixXR / mixSum;
  const us = knobs.mixUS / mixSum;
  const ct = knobs.mixCT / mixSum;
  const mr = knobs.mixMR / mixSum;

  // Normalize ranges to roughly 0..1 so numbers are reasonable
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
  ];

  // add a bias-ish constant + interactions
  v.push(1.0);
  v.push(ct * knobs.edFlux);        // more ED + CT mix
  v.push(mr * knobs.inFlux);        // more IP + MR mix
  v.push(knobs.ctDown * ct);        // CT downtime pressure
  v.push(knobs.mrDown * mr);        // MR downtime pressure
  v.push(knobs.nsRate * knobs.opFlux); // OP/no-show interplay

  return v;
}

function padOrTruncate(vec, targetDim){
  const x = new Array(targetDim).fill(0);
  const n = Math.min(vec.length, targetDim);
  for (let i=0;i<n;i++) x[i] = vec[i];
  return x;
}

// ---- KPI computation using NN outputs as nonlinear modifiers ----
function computeKPIs(knobs, durationModel, noshowModel){
  // Baseline daily volumes (same as generator defaults)
  const baseED = 180, baseIP = 140, baseOP = 220;

  const ED = baseED * knobs.edFlux;
  const IP = baseIP * knobs.inFlux;
  const OP = baseOP * knobs.opFlux;

  // Mix normalized
  const mixSum = Math.max(1e-6, knobs.mixXR + knobs.mixUS + knobs.mixCT + knobs.mixMR);
  const mix = {
    XR: knobs.mixXR / mixSum,
    US: knobs.mixUS / mixSum,
    CT: knobs.mixCT / mixSum,
    MR: knobs.mixMR / mixSum
  };

  // Capacity (your hospital)
  const scanners = { CT: 5, MR: 3, US: 2, XR: 9 };
  let availCT = scanners.CT * 1440 * (1 - clamp(knobs.ctDown, 0, 0.9));
  let availMR = scanners.MR * 1440 * (1 - clamp(knobs.mrDown, 0, 0.9));

  // Scenario vector -> pad to each model’s input dim
  const scenario = buildScenarioVector(knobs);

  // Duration model output (regression)
  // We use NN output as a multiplier around baseline durations (educational).
  const durInDim = durationModel.layers[0].W[0].length;
  const xDur = padOrTruncate(scenario, durInDim);
  const durOut = mlpForward(xDur, durationModel)[0]; // scalar
  const durMultiplier = clamp(1 + durOut / 60, 0.6, 1.8); // convert to sane multiplier

  // No-show model output (logits -> probability)
  const nsInDim = noshowModel.layers[0].W[0].length;
  const xNS = padOrTruncate(scenario, nsInDim);
  const nsLogit = mlpForward(xNS, noshowModel)[0];
  // Blend learner nsRate with NN “shape” (still only OP)
  const nsNN = clamp(sigmoid(nsLogit), 0.01, 0.40);
  const nsBase = clamp(knobs.nsRate, 0, 0.40);
  const nsFrac = clamp(0.65 * nsBase + 0.35 * nsNN, 0.0, 0.40);

  // Overbooking policy
  const allowOverbook = nsFrac >= knobs.overT;
  const overbookExtra = allowOverbook ? Math.round(0.08 * OP) : 0;

  // Baseline per-modality minutes (rough)
  const baseDur = { XR: 10, US: 40, CT: 22, MR: 55 };
  const effDur = {
    XR: baseDur.XR * durMultiplier,
    US: baseDur.US * durMultiplier,
    CT: baseDur.CT * durMultiplier,
    MR: baseDur.MR * durMultiplier
  };

  // Demand minutes
  const total = ED + IP + (OP + overbookExtra);
  const demandCT = total * mix.CT * effDur.CT;
  const demandMR = total * mix.MR * effDur.MR;

  // ED reserve: reduces OP access to CT (pressure effect)
  const reserve = clamp(knobs.reserve, 0, 0.8);
  const opCTAvail = availCT * (1 - reserve);

  const ctUtil = clamp(demandCT / Math.max(1, availCT), 0, 1.5);
  const mrUtil = clamp(demandMR / Math.max(1, availMR), 0, 1.5);

  // Queueing-ish waits
  const edTTS = 18 + (ctUtil > 0.6 ? (ctUtil - 0.6) * 120 : 0) + (ctUtil > 0.9 ? (ctUtil - 0.9) * 350 : 0);
  const ipTTS = 45 + knobs.transport + (ctUtil > 0.6 ? (ctUtil - 0.6) * 140 : 0) + (ctUtil > 0.9 ? (ctUtil - 0.9) * 350 : 0);

  // OP on-time start drops with utilization, overbooking, and OP CT pressure
  const opCTPressure = clamp(demandCT / Math.max(1, opCTAvail), 0, 2);
  const opOnTime = clamp(0.93 - (ctUtil - 0.6) * 0.35 - (overbookExtra > 0 ? 0.06 : 0) - Math.max(0, opCTPressure - 0.9) * 0.10, 0.20, 0.98);

  // Overtime/idle
  const overtime = Math.max(0, demandCT - availCT) / Math.max(1, scanners.CT)
                + Math.max(0, demandMR - availMR) / Math.max(1, scanners.MR);

  const idle = Math.max(0, availCT - demandCT) / Math.max(1, scanners.CT)
            + Math.max(0, availMR - demandMR) / Math.max(1, scanners.MR);

  const bumps = Math.round((overbookExtra / Math.max(1, OP)) * 12 + Math.max(0, ctUtil - 0.85) * 14);

  // Throughput
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
    nsNN: nsNN * 100
  };
}

document.addEventListener("DOMContentLoaded", async () => {
  const statusEl = $("status");
  updateSliderLabels();

  // Wire slider labels
  const ids = ["edFlux","inFlux","opFlux","mixXR","mixUS","mixCT","mixMR","ctDown","mrDown","transport","nsRate","overT","reserve"];
  ids.forEach(id => { const el = $(id); if (el) el.addEventListener("input", updateSliderLabels); });

  let durationModel = null;
  let noshowModel = null;

  // Load the JSON-weight models
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
    statusEl.textContent = `FAILED to load JS models: ${e.message}`;
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

    statusEl.textContent =
`Models loaded ✅ | Last run: ${new Date().toLocaleTimeString()}
Duration multiplier (NN): ${k.durMultiplier.toFixed(2)}
No-show (NN component): ${k.nsNN.toFixed(1)}%
Educational Use Only (Non-Clinical).`;
  }

  const runBtn = $("runBtn");
  if (runBtn) runBtn.addEventListener("click", run);

  // Run once on load so the page is “alive”
  if (durationModel && noshowModel) run();
});
