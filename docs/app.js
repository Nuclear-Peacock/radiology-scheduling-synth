// docs/app.js
// Educational Use Only (Non-Clinical)

function $(id){ return document.getElementById(id); }
function clamp(x,a,b){ return Math.max(a, Math.min(b,x)); }

document.addEventListener("DOMContentLoaded", () => {
  const statusEl = $("status");

  // ---------- Slider labels ----------
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
      if (s && v) v.textContent = s.value;
    }
  }
  updateLabels();
  sliders.forEach(([sid]) => {
    const el = $(sid);
    if (el) el.addEventListener("input", updateLabels);
  });

  // ---------- KPI helpers ----------
  function setText(id, text){
    const el = $(id);
    if (el) el.textContent = text;
  }

  function baseDuration(mod){
    if (mod === "CT") return 22;
    if (mod === "MR") return 55;
    if (mod === "US") return 40;
    return 10; // XR
  }

  // Educational expected-value simulation (fast, deterministic)
  function simulate(knobs){
    // Baseline daily volumes (same ballpark as generator)
    const baseED = 180, baseIP = 140, baseOP = 220;

    const ED = baseED * knobs.edFlux;
    const IP = baseIP * knobs.inFlux;
    const OP = baseOP * knobs.opFlux;

    const mix = {
      XR: knobs.mixXR/100,
      US: knobs.mixUS/100,
      CT: knobs.mixCT/100,
      MR: knobs.mixMR/100
    };

    // Inventory (your hospital)
    const scanners = { CT: 5, MR: 3, US: 2, XR: 9 }; // XR rooms + portable as rough capacity proxy

    // Available minutes/day
    let availCT = scanners.CT * 1440;
    let availMR = scanners.MR * 1440;

    // Downtime reduces availability
    availCT *= (1 - clamp(knobs.ctDown, 0, 0.9));
    availMR *= (1 - clamp(knobs.mrDown, 0, 0.9));

    // ED CT reserve reduces outpatient-usable CT capacity (simplified)
    const opCTAvail = availCT * (1 - clamp(knobs.reserve, 0, 0.8));

    // Demand minutes by modality
    const total = ED + IP + OP;
    const demandCT = total * mix.CT * baseDuration("CT");
    const demandMR = total * mix.MR * baseDuration("MR");

    const ctUtil = clamp(demandCT / Math.max(1, availCT), 0, 1.5);
    const mrUtil = clamp(demandMR / Math.max(1, availMR), 0, 1.5);

    // Wait proxies: increase nonlinearly with utilization
    const edTTS = 18 + (ctUtil > 0.6 ? (ctUtil - 0.6) * 120 : 0) + (ctUtil > 0.9 ? (ctUtil - 0.9) * 350 : 0);
    const ipTTS = 45 + knobs.transport + (ctUtil > 0.6 ? (ctUtil - 0.6) * 140 : 0) + (ctUtil > 0.9 ? (ctUtil - 0.9) * 350 : 0);

    // No-show (OP only): base rate amplified by system pressure (educational)
    const nsFrac = clamp(knobs.nsRate * (1 + 0.20 * Math.max(0, ctUtil - 0.7)), 0, 0.40);

    // Overbooking: if nsFrac >= threshold, add bookings (educational)
    const allowOverbook = nsFrac >= knobs.overT;
    const overbookExtra = allowOverbook ? Math.round(0.08 * OP) : 0;

    // On-time starts drop with utilization and overbooking
    const opOnTime = clamp(0.93 - (ctUtil - 0.6) * 0.35 - (overbookExtra > 0 ? 0.06 : 0), 0.20, 0.98);

    // Overtime & idle proxies
    const overtime = Math.max(0, demandCT - availCT) / Math.max(1, scanners.CT)
                  + Math.max(0, demandMR - availMR) / Math.max(1, scanners.MR);

    const idle = Math.max(0, availCT - demandCT) / Math.max(1, scanners.CT)
              + Math.max(0, availMR - demandMR) / Math.max(1, scanners.MR);

    // Bumps proxy (mostly when overbooking + high util)
    const bumps = Math.round((overbookExtra / Math.max(1, OP)) * 12 + Math.max(0, ctUtil - 0.85) * 14);

    // Throughput: ED+IP all complete; OP completes expected shows among booked
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
      throughput
    };
  }

  function readKnobs(){
    // If any element missing, default safely
    const get = (id, fallback) => {
      const el = $(id);
      return el ? parseFloat(el.value) : fallback;
    };
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

  function renderKPIs(k){
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
  }

  // ---------- Wire up the Run button FIRST (so it always works) ----------
  const runBtn = $("runBtn");
  if (runBtn) {
    runBtn.addEventListener("click", () => {
      const knobs = readKnobs();
      const sum = knobs.mixXR + knobs.mixUS + knobs.mixCT + knobs.mixMR;
      if (Math.abs(sum - 100) > 5) {
        alert(`Modality mix sums to ${sum}%. Please adjust closer to 100% for best results.`);
      }
      const kpis = simulate(knobs);
      renderKPIs(kpis);
    });
  }

  // ---------- Load models (optional; app still runs without them) ----------
  async function fetchJson(path){
    const res = await fetch(path);
    if (!res.ok) throw new Error(`Failed to fetch ${path} (${res.status})`);
    return await res.json();
  }

  async function loadModels(){
    const ort = window.ort;
    if (!ort || !ort.InferenceSession) {
      statusEl.textContent = "ONNX Runtime not available. (UI still works; models not loaded.)";
      return;
    }
    // WASM path for ONNX runtime
    ort.env.wasm.wasmPaths = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.18.0/dist/";

    statusEl.textContent = "Loading model metadata…";
    const durationMeta = await fetchJson("./public/models/duration_features.json");
    const noshowMeta = await fetchJson("./public/models/noshow_features.json");

    statusEl.textContent = "Loading ONNX models…";
    await ort.InferenceSession.create("./public/models/duration.onnx");
    await ort.InferenceSession.create("./public/models/noshow.onnx");

    statusEl.textContent =
`Loaded models successfully:
• duration.onnx (features: ${durationMeta.feature_columns.length})
• noshow.onnx (features: ${noshowMeta.feature_columns.length})

Note: current KPIs use an educational expected-value simulator.
(Next step can run true per-order ONNX inference.)`;
  }

  loadModels().catch(err => {
    statusEl.textContent = `Model load failed: ${err.message}\n(UI still works without models.)`;
  });

  // Run once on load so user sees numbers immediately
  renderKPIs(simulate(readKnobs()));
});
