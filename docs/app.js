// docs/app.js
// Educational Use Only (Non-Clinical)

function $(id){ return document.getElementById(id); }
function clamp(x,a,b){ return Math.max(a, Math.min(b,x)); }

document.addEventListener("DOMContentLoaded", () => {
  const statusEl = $("status");

  // ----- Always set a status immediately so it never stays "Loading…" -----
  statusEl.textContent = "JS loaded ✅. Initializing…";

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

  // ---------- KPI engine (educational expected-value) ----------
  function baseDuration(mod){
    if (mod === "CT") return 22;
    if (mod === "MR") return 55;
    if (mod === "US") return 40;
    return 10; // XR
  }

  function simulate(knobs){
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

    const scanners = { CT: 5, MR: 3, US: 2, XR: 9 };

    let availCT = scanners.CT * 1440 * (1 - clamp(knobs.ctDown, 0, 0.9));
    let availMR = scanners.MR * 1440 * (1 - clamp(knobs.mrDown, 0, 0.9));

    // reserve affects OP access to CT (used as a pressure factor)
    const reserve = clamp(knobs.reserve, 0, 0.8);
    const opCTAvail = availCT * (1 - reserve);

    const total = ED + IP + OP;
    const demandCT = total * mix.CT * baseDuration("CT");
    const demandMR = total * mix.MR * baseDuration("MR");

    const ctUtil = clamp(demandCT / Math.max(1, availCT), 0, 1.5);
    const mrUtil = clamp(demandMR / Math.max(1, availMR), 0, 1.5);

    // wait proxies
    const edTTS = 18 + (ctUtil > 0.6 ? (ctUtil - 0.6) * 120 : 0) + (ctUtil > 0.9 ? (ctUtil - 0.9) * 350 : 0);
    const ipTTS = 45 + knobs.transport + (ctUtil > 0.6 ? (ctUtil - 0.6) * 140 : 0) + (ctUtil > 0.9 ? (ctUtil - 0.9) * 350 : 0);

    // OP no-shows and overbooking
    const nsFrac = clamp(knobs.nsRate * (1 + 0.25 * Math.max(0, (demandCT / Math.max(1, opCTAvail)) - 0.8)), 0, 0.40);
    const allowOverbook = nsFrac >= knobs.overT;
    const overbookExtra = allowOverbook ? Math.round(0.08 * OP) : 0;

    const opOnTime = clamp(0.93 - (ctUtil - 0.6) * 0.35 - (overbookExtra > 0 ? 0.06 : 0), 0.20, 0.98);

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
      throughput
    };
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

  function setText(id, text){ const el = $(id); if (el) el.textContent = text; }

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

  // ---------- Wire button (always) ----------
  const runBtn = $("runBtn");
  if (runBtn) {
    runBtn.addEventListener("click", () => {
      const knobs = readKnobs();
      const sum = knobs.mixXR + knobs.mixUS + knobs.mixCT + knobs.mixMR;
      if (Math.abs(sum - 100) > 5) {
        alert(`Modality mix sums to ${sum}%. Please adjust closer to 100% for best results.`);
      }
      renderKPIs(simulate(knobs));
      statusEl.textContent = statusEl.textContent.replace(/\n\nLast run:.*/s, "") + `\n\nLast run: ${new Date().toLocaleTimeString()}`;
    });
  }

  // Run once on load so the page shows numbers
  renderKPIs(simulate(readKnobs()));
  statusEl.textContent = "UI ready ✅. Loading models (optional)…";

  // ---------- Model load with timeout (so it never hangs) ----------
  function withTimeout(promise, ms, label){
    let t;
    const timeout = new Promise((_, reject) => {
      t = setTimeout(() => reject(new Error(`${label} timed out after ${ms}ms`)), ms);
    });
    return Promise.race([promise, timeout]).finally(() => clearTimeout(t));
  }

  async function fetchJson(path){
    const res = await fetch(path);
    if (!res.ok) throw new Error(`Failed to fetch ${path} (${res.status})`);
    return await res.json();
  }

  async function loadModels(){
    const ort = window.ort;
    if (!ort || !ort.InferenceSession) {
      statusEl.textContent = "UI ready ✅. ONNX Runtime not available (models not loaded).";
      return;
    }

    const durMeta = await withTimeout(fetchJson("./public/models/duration_features.json"), 4000, "duration_features.json");
    const nsMeta  = await withTimeout(fetchJson("./public/models/noshow_features.json"), 4000, "noshow_features.json");

    await withTimeout(ort.InferenceSession.create("./public/models/duration.onnx"), 12000, "duration.onnx");
    await withTimeout(ort.InferenceSession.create("./public/models/noshow.onnx"), 12000, "noshow.onnx");

    statusEl.textContent =
`UI ready ✅. Models loaded ✅
• duration.onnx (features: ${durMeta.feature_columns.length})
• noshow.onnx (features: ${nsMeta.feature_columns.length})

Current KPIs use an educational expected-value simulator.
Next step can apply true per-order inference.`;
  }

  loadModels().catch(err => {
    statusEl.textContent =
`UI ready ✅. Model load failed (UI still works):
${err.message}

Common fixes:
• Ensure docs/public/models/*.onnx and *.json exist
• Hard refresh the page (Ctrl/Cmd+Shift+R)`;
  });
});
