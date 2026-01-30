// docs/app.js
// Educational Use Only (Non-Clinical)

function $(id){ return document.getElementById(id); }

const statusEl = $("status");

// =======================
// Slider value display
// =======================
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
  const s = $(sid);
  if (s) s.addEventListener("input", updateLabels);
});

// =======================
// ONNX Runtime Setup
// =======================
const ort = window.ort;
if (!ort || !ort.InferenceSession) {
  statusEl.textContent = "ONNX Runtime failed to load. Check index.html script tag.";
  throw new Error("ONNX Runtime missing");
}

// Tell ORT where its WASM files live
ort.env.wasm.wasmPaths = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.18.0/dist/";

// =======================
// Load models + metadata
// =======================
let durationSession = null;
let noshowSession = null;
let durationMeta = null;
let noshowMeta = null;

async function fetchJson(path){
  const res = await fetch(path);
  if (!res.ok) throw new Error(`Failed to fetch ${path}`);
  return await res.json();
}

async function loadModels(){
  statusEl.textContent = "Loading model metadata…";

  durationMeta = await fetchJson("./public/models/duration_features.json");
  noshowMeta = await fetchJson("./public/models/noshow_features.json");

  statusEl.textContent = "Loading ONNX models…";

  durationSession = await ort.InferenceSession.create("./public/models/duration.onnx");
  noshowSession = await ort.InferenceSession.create("./public/models/noshow.onnx");

  statusEl.textContent =
`Loaded models successfully:
• duration.onnx (features: ${durationMeta.feature_columns.length})
• noshow.onnx (features: ${noshowMeta.feature_columns.length})

This UI uses an educational expected-value simulation.
Models are loaded and ready.`;
}

// =======================
// Educational KPI engine
// =======================
function clamp(x,a,b){ return Math.max(a, Math.min(b,x)); }

function baseDuration(mod){
  if (mod==="CT") return 22;
  if (mod==="MR") return 55;
  if (mod==="US") return 40;
  return 10;
}

function simulate(knobs){
  const baseED=180, baseIP=140, baseOP=220;

  const ED = baseED * knobs.edFlux;
  const IP = baseIP * knobs.inFlux;
  const OP = baseOP * knobs.opFlux;

  const mix = {
    XR: knobs.mixXR/100,
    US: knobs.mixUS/100,
    CT: knobs.mixCT/100,
    MR: knobs.mixMR/100
  };

  const scanners={CT:5,MR:3,US:2,XR:9};
  const availCT=5*1440*(1-knobs.ctDown);
  const availMR=3*1440*(1-knobs.mrDown);

  const demandCT = (ED+IP+OP)*mix.CT*baseDuration("CT");
  const demandMR = (ED+IP+OP)*mix.MR*baseDuration("MR");

  const ctUtil = clamp(demandCT/availCT,0,1.5);
  const mrUtil = clamp(demandMR/availMR,0,1.5);

  const edTTS = 20 + Math.max(0,ctUtil-0.7)*120;
  const ipTTS = 50 + knobs.transport + Math.max(0,ctUtil-0.7)*140;

  const ns = clamp(knobs.nsRate * (1+0.4*ctUtil),0,0.4);

  const onTime = clamp(0.95 - (ctUtil-0.7)*0.4,0.2,0.95);

  const overtime = Math.max(0,demandCT-availCT)/5 + Math.max(0,demandMR-availMR)/3;
  const idle = Math.max(0,availCT-demandCT)/5 + Math.max(0,availMR-demandMR)/3;

  const bumps = Math.round((knobs.overT<ns?OP*0.05:0)+(ctUtil>0.9?OP*0.04:0));

  return {
    edTTS, ipTTS,
    onTime:onTime*100,
    ctUtil:ctUtil*100,
    mrUtil:mrUtil*100,
    overtime, idle,
    ns:ns*100,
    bumps,
    throughput: Math.round(ED+IP+OP*(1-ns))
  };
}

// =======================
// UI wiring
// =======================
function readKnobs(){
  return {
    edFlux:+$("edFlux").value,
    inFlux:+$("inFlux").value,
    opFlux:+$("opFlux").value,
    mixXR:+$("mixXR").value,
    mixUS:+$("mixUS").value,
    mixCT:+$("mixCT").value,
    mixMR:+$("mixMR").value,
    ctDown:+$("ctDown").value,
    mrDown:+$("mrDown").value,
    transport:+$("transport").value,
    nsRate:+$("nsRate").value,
    overT:+$("overT").value,
    reserve:+$("reserve").value
  };
}

$("runBtn").onclick=()=>{
  const k=simulate(readKnobs());
  $("k_edTTS").textContent=Math.round(k.edTTS)+" min";
  $("k_inTTS").textContent=Math.round(k.ipTTS)+" min";
  $("k_opOnTime").textContent=k.onTime.toFixed(1)+"%";
  $("k_ctUtil").textContent=k.ctUtil.toFixed(1)+"%";
  $("k_mrUtil").textContent=k.mrUtil.toFixed(1)+"%";
  $("k_overtime").textContent=Math.round(k.overtime)+" min";
  $("k_idle").textContent=Math_
