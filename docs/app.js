// ui/app.js
// Educational Use Only (Non-Clinical)

function $(id){ return document.getElementById(id); }

const statusEl = $("status");

// Show slider values
const sliders = [
  ["edFlux","v_edFlux"], ["inFlux","v_inFlux"], ["opFlux","v_opFlux"],
  ["mixXR","v_mixXR"], ["mixUS","v_mixUS"], ["mixCT","v_mixCT"], ["mixMR","v_mixMR"],
  ["ctDown","v_ctDown"], ["mrDown","v_mrDown"], ["transport","v_transport"],
  ["nsRate","v_ns"], ["overT","v_overT"], ["reserve","v_reserve"],
];

function updateLabels(){
  for (const [sid, vid] of sliders){
    const s = $(sid);
    const v = $(vid);
    if (!s || !v) continue;
    v.textContent = `${s.value}`;
  }
}
updateLabels();
for (const [sid] of sliders){
  const s = $(sid);
  if (s) s.addEventListener("input", updateLabels);
}

// We’ll load ONNX runtime from CDN to keep GitHub Pages simple
async function loadOrt(){
  if (window.ort) return window.ort;
  statusEl.textContent = "Loading ONNX Runtime (web)…";
  await new Promise((resolve, reject) => {
    const script = document.createElement("script");
    script.src = "https://cdn.jsdelivr.net/npm/onnxruntime-web/dist/ort.min.js";
    script.onload = resolve;
    script.onerror = reject;
    document.head.appendChild(script);
  });
  return window.ort;
}

async function fetchJson(path){
  const res = await fetch(path);
  if (!res.ok) throw new Error(`Failed to fetch ${path}: ${res.status}`);
  return await res.json();
}

let ort = null;
let durationSession = null;
let noshowSession = null;
let durationMeta = null;
let noshowMeta = null;

async function loadModels(){
  ort = await loadOrt();

  statusEl.textContent = "Loading model metadata…";
  durationMeta = await fetchJson("./public/models/duration_features.json");
  noshowMeta = await fetchJson("./public/models/noshow_features.json");

  statusEl.textContent = "Loading ONNX models…";
  durationSession = await ort.InferenceSession.create("./public/models/duration.onnx");
  noshowSession = await ort.InferenceSession.create("./public/models/noshow.onnx");

  statusEl.textContent =
`Loaded:
- duration.onnx (features: ${durationMeta.feature_columns.length})
- noshow.onnx (features: ${noshowMeta.feature_columns.length})

Note: This UI uses an educational “expected value” simulation for KPIs.`;
}

function clamp(x, lo, hi){ return Math.max(lo, Math.min(hi, x)); }

// Minimal feature encoder for demo:
// Because one-hot feature space is large, we won’t attempt full per-exam inference here.
// Instead: use the *model metadata* as proof models are loaded,
// and compute durations/no-show risks via lightweight educational proxies.
// (Next step: we can add true ONNX inference over sampled orders.)
function proxyDurationMinutes(modality){
  // Baseline minutes
  if (modality === "CT") return 22;
  if (modality === "MR") return 55;
  if (modality === "US") return 40;
  return 10; // XR
}

function proxyNoShowProb(baseRate, leadDays){
  // Increase risk with lead time (educational)
  return clamp(baseRate * (1 + 0.04 * leadDays), 0.0, 0.95);
}

// Educational KPI simulation (fast expected values)
function runEducationalSim(knobs){
  // knobs:
  // {edFlux, inFlux, opFlux, mixXR, mixUS, mixCT, mixMR, ctDown, mrDown, transport, nsRate, overT, reserve}

  // Synthetic daily demand baseline (matches generator defaults)
  const baseED = 180, baseIN = 140, baseOP = 220;

  // Volumes
  const nED = Math.round(baseED * knobs.edFlux);
  const nIN = Math.round(baseIN * knobs.inFlux);
  const nOP = Math.round(baseOP * knobs.opFlux);

  // Mix
  const mix = {
    XR: knobs.mixXR/100,
    US: knobs.mixUS/100,
    CT: knobs.mixCT/100,
    MR: knobs.mixMR/100
  };

  // Scanner inventory (fixed from your hospital)
  const scanners = {
    CT: 5,
    MR: 3,
    US: 2,
    XR: 6 + 3 // XR rooms + portable XR as throughput capacity proxy
  };

  // Available minutes/day (educational):
  // CT/MR are 24/7; US roughly 16h; XR ~17h but ED rooms 24/7; simplify
  const avail = {
    CT: 24*60*scanners.CT,
    MR: 24*60*scanners.MR,
    US: 16*60*scanners.US,
    XR: 20*60*scanners.XR
  };

  // Apply downtime (fraction of day per scanner group)
  const ctAvail = avail.CT * (1 - clamp(knobs.ctDown, 0, 0.9));
  const mrAvail = avail.MR * (1 - clamp(knobs.mrDown, 0, 0.9));

  // ED CT reserve reduces outpatient-usable CT capacity
  const opCTAvail = ctAvail * (1 - clamp(knobs.reserve, 0, 0.8));
  const edCTAvail = ctAvail - opCTAvail;

  // Demand minutes by setting and modality
  function demandMinutes(total, modality){
    const per = proxyDurationMinutes(modality);
    return total * per;
  }

  // Split counts by modality
  function splitCounts(n){
    return {
      XR: Math.round(n * mix.XR),
      US: Math.round(n * mix.US),
      CT: Math.round(n * mix.CT),
      MR: Math.round(n * mix.MR),
    };
  }

  const ed = splitCounts(nED);
  const ip = splitCounts(nIN);
  const op = splitCounts(nOP);

  // Outpatient no-shows: expected shows
  // Use a simple average leadDays=10 for expected value, plus overbooking threshold effect
  const avgLead = 10;
  const pNS = proxyNoShowProb(knobs.nsRate, avgLead);
  const expectedShows = nOP * (1 - pNS);

  // Overbooking policy: if pNS >= threshold, book extra up to 8% volume (educational)
  const allowOverbook = (pNS >= knobs.overT);
  const overbookExtra = allowOverbook ? Math.round(0.08 * nOP) : 0;

  const opEffectiveBooked = nOP + overbookExtra;
  const opExpectedShowCount = opEffectiveBooked * (1 - pNS);

  // Compute utilization by modality (approx)
  // CT: ED+IP consume ED CT capacity first; outpatient consumes opCT capacity
  const edCTmins = demandMinutes(ed.CT, "CT");
  const ipCTmins = demandMinutes(ip.CT, "CT");
  const opCTmins = demandMinutes(opEffectiveBooked * mix.CT, "CT");

  const ctUsed = Math.min(ctAvail, edCTmins + ipCTmins + opCTmins);
  const ctUtil = ctUsed / Math.max(1, ctAvail);

  // MR used
  const mrMins = demandMinutes(ed.MR + ip.MR + (opEffectiveBooked * mix.MR), "MR");
  const mrUsed = Math.min(mrAvail, mrMins);
  const mrUtil = mrUsed / Math.max(1, mrAvail);

  // Overtime proxy: when used > available → overtime minutes = excess / scanners
  const overtimeMin = Math.max(0, (edCTmins + ipCTmins + opCTmins) - ctAvail) / Math.max(1, scanners.CT)
                    + Math.max(0, mrMins - mrAvail) / Math.max(1, scanners.MR);

  // Idle proxy: available - used (only CT+MR shown, rough)
  const idleMin = Math.max(0, (ctAvail - ctUsed) / Math.max(1, scanners.CT))
                + Math.max(0, (mrAvail - mrUsed) / Math.max(1, scanners.MR));

  // Time-to-scan proxies:
  // As utilization approaches 1, waits increase nonlinearly (queueing intuition)
  function ttsFromUtil(util, base){
    return base + (util > 0.6 ? (util - 0.6) * 120 : 0) + (util > 0.9 ? (util - 0.9) * 400 : 0);
  }
  const edTTS = ttsFromUtil(clamp(ctUtil, 0, 1.2), 18);         // minutes
  const ipTTS = (ttsFromUtil(clamp(ctUtil, 0, 1.2), 45) + knobs.transport); // minutes (transport adds)
  const opOnTime = clamp(0.92 - (ctUtil - 0.6) * 0.35 - (overbookExtra > 0 ? 0.06 : 0), 0.2, 0.98);

  // Bumps proxy: overbooking + high utilization increases bumps
  const bumps = Math.round((overbookExtra / Math.max(1, nOP)) * 10 + Math.max(0, ctUtil - 0.85) * 12);

  // Throughput proxy (completed/day)
  const completed = Math.round(nED + nIN + opExpectedShowCount);

  return {
    edTTS_min: edTTS,
    ipTTS_min: ipTTS,
    opOnTime_pct: opOnTime * 100,
    ctUtil_pct: ctUtil * 100,
    mrUtil_pct: mrUtil * 100,
    overtime_min: overtimeMin,
    idle_min: idleMin,
    ns_pct: pNS * 100,
    bumps_per_day: bumps,
    throughput: completed,
  };
}

function fmtMin(x){
  if (!isFinite(x)) return "—";
  return `${Math.round(x)} min`;
}
function fmtPct(x){
  if (!isFinite(x)) return "—";
  return `${x.toFixed(1)}%`;
}
function fmtNum(x){
  if (!isFinite(x)) return "—";
  return `${Math.round(x)}`;
}

function readKnobs(){
  return {
    edFlux: parseFloat($("edFlux").value),
    inFlux: parseFloat($("inFlux").value),
    opFlux: parseFloat($("opFlux").value),
    mixXR: parseFloat($("mixXR").value),
    mixUS: parseFloat($("mixUS").value),
    mixCT: parseFloat($("mixCT").value),
    mixMR: parseFloat($("mixMR").value),
    ctDown: parseFloat($("ctDown").value),
    mrDown: parseFloat($("mrDown").value),
    transport: parseFloat($("transport").value),
    nsRate: parseFloat($("nsRate").value),
    overT: parseFloat($("overT").value),
    reserve: parseFloat($("reserve").value),
  };
}

function setKpis(k){
  $("k_edTTS").textContent = fmtMin(k.edTTS_min);
  $("k_inTTS").textContent = fmtMin(k.ipTTS_min);
  $("k_opOnTime").textContent = fmtPct(k.opOnTime_pct);
  $("k_ctUtil").textContent = fmtPct(k.ctUtil_pct);
  $("k_mrUtil").textContent = fmtPct(k.mrUtil_pct);
  $("k_overtime").textContent = fmtMin(k.overtime_min);
  $("k_idle").textContent = fmtMin(k.idle_min);
  $("k_ns").textContent = fmtPct(k.ns_pct);
  $("k_bumps").textContent = fmtNum(k.bumps_per_day);
  $("k_throughput").textContent = fmtNum(k.throughput);
}

$("runBtn").addEventListener("click", () => {
  const knobs = readKnobs();

  // warn if modality mix doesn't sum to 100
  const sum = knobs.mixXR + knobs.mixUS + knobs.mixCT + knobs.mixMR;
  if (Math.abs(sum - 100) > 5) {
    alert(`Modality mix sums to ${sum}%. Please adjust closer to 100% for best results.`);
  }

  const kpis = runEducationalSim(knobs);
  setKpis(kpis);
});

// Load models on page load
loadModels().catch(err => {
  statusEl.textContent = `Failed to load models: ${err.message}`;
});
