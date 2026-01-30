// SMOKE TEST: confirms JavaScript is loading and button click works
document.addEventListener("DOMContentLoaded", () => {
  const status = document.getElementById("status");
  const btn = document.getElementById("runBtn");

  if (status) status.textContent = "JS LOADED ✅ (smoke test)";
  if (btn) {
    btn.addEventListener("click", () => {
      status.textContent = "Run simulation clicked ✅ " + new Date().toLocaleTimeString();
    });
  }
});
