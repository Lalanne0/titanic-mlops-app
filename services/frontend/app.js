/* ===================================================
   Titanic MLOps -- App Logic (SPA)
   =================================================== */

(() => {
  "use strict";

  // -- State -----------------------------------------------
  let lastPassengerInput = null;
  let lastPrediction = null;
  let modelLoaded = false;

  // -- DOM refs --------------------------------------------
  const $ = (sel) => document.querySelector(sel);
  const $$ = (sel) => document.querySelectorAll(sel);

  // -- View management -------------------------------------
  function showView(name) {
    $$(".view").forEach((v) => v.classList.remove("active"));
    const target = $(`#view-${name}`);
    if (target) target.classList.add("active");

    // sidebar active state
    $$(".sidebar-link[data-view]").forEach((l) => l.classList.remove("active"));
    const navLink = $(`.sidebar-link[data-view="${name}"]`);
    if (navLink) navLink.classList.add("active");

    // re-render icons in the new view
    window.lucide && window.lucide.createIcons();
  }

  // -- Toast notifications ---------------------------------
  function toast(message, type = "info") {
    const iconMap = { success: "check-circle", error: "x-circle", info: "info" };
    const container = $("#toast-container");
    const el = document.createElement("div");
    el.className = `toast ${type}`;
    el.innerHTML = `<i data-lucide="${iconMap[type] || "info"}"></i> ${escapeHtml(message)}`;
    container.appendChild(el);
    window.lucide && window.lucide.createIcons();
    setTimeout(() => el.remove(), 3200);
  }

  // -- Loading overlay -------------------------------------
  function showLoading(text = "Processing...") {
    $("#loading-text").textContent = text;
    $("#loading-overlay").classList.add("active");
  }

  function hideLoading() {
    $("#loading-overlay").classList.remove("active");
  }

  // -- Helpers ---------------------------------------------
  function escapeHtml(str) {
    const d = document.createElement("div");
    d.textContent = str;
    return d.innerHTML;
  }

  async function apiFetch(path, options = {}) {
    const resp = await fetch(`/api${path}`, {
      headers: { "Content-Type": "application/json", ...options.headers },
      ...options,
    });
    if (!resp.ok) {
      const body = await resp.json().catch(() => ({}));
      throw new Error(body.detail || `HTTP ${resp.status}`);
    }
    return resp.json();
  }

  // -- Model status ----------------------------------------
  async function checkModelStatus() {
    try {
      const h = await apiFetch("/health");
      modelLoaded = !!h.model_loaded;
    } catch {
      modelLoaded = false;
    }
    updateModelWarnings();
    await refreshModelBadge();
  }

  function updateModelWarnings() {
    const banner = $("#no-model-banner");
    const inline = $("#no-model-inline");
    const predictBtn = $("#btn-predict");

    if (modelLoaded) {
      if (banner) banner.style.display = "none";
      if (inline) inline.style.display = "none";
      if (predictBtn) predictBtn.disabled = false;
    } else {
      if (banner) banner.style.display = "block";
      if (inline) inline.style.display = "block";
      if (predictBtn) predictBtn.disabled = true;
    }
  }

  // -- Load model badge ------------------------------------
  async function refreshModelBadge() {
    try {
      const info = await apiFetch("/model-info");
      if (info.model_version) {
        $("#badge-version").textContent = `v${info.model_version}`;
        $("#badge-name").textContent = info.model_name || "titanic-survivor";
      } else {
        $("#badge-version").textContent = "No model";
      }
    } catch {
      $("#badge-version").textContent = "Unavailable";
    }
  }

  // -- Render metrics grid (reusable) ----------------------
  function renderMetrics(gridEl, result) {
    const metrics = [
      { label: "Accuracy", value: (result.accuracy * 100).toFixed(1) + "%" },
      { label: "F1 Score", value: (result.f1_score * 100).toFixed(1) + "%" },
      { label: "Precision", value: (result.precision * 100).toFixed(1) + "%" },
      { label: "Recall", value: (result.recall * 100).toFixed(1) + "%" },
    ];

    gridEl.innerHTML = metrics
      .map(
        (m) => `
      <div class="metric-card">
        <div class="metric-value">${m.value}</div>
        <div class="metric-label">${m.label}</div>
      </div>`
      )
      .join("");
  }

  function renderDetails(detailsEl, result) {
    const lines = [];
    lines.push(`Model version: <strong>v${result.model_version}</strong>`);
    lines.push(`Trained on <strong>${result.n_samples}</strong> samples`);
    lines.push(`MLflow run ID: <code>${result.run_id}</code>`);
    if (result.drift_report) {
      lines.push(`Drift report: <code>${result.drift_report}</code>`);
    }
    detailsEl.innerHTML = `<p>${lines.join("<br/>")}</p>`;
  }

  // =========================================================
  //  Predict
  // =========================================================
  async function handlePredict(e) {
    e.preventDefault();

    if (!modelLoaded) {
      toast("No model loaded. Train a model first.", "error");
      return;
    }

    const payload = {
      Pclass: parseInt($("#field-pclass").value, 10),
      Sex: $("#field-sex").value,
      Age: $("#field-age").value ? parseFloat($("#field-age").value) : null,
      SibSp: parseInt($("#field-sibsp").value, 10) || 0,
      Parch: parseInt($("#field-parch").value, 10) || 0,
      Fare: parseFloat($("#field-fare").value),
      Embarked: $("#field-embarked").value,
    };

    if (isNaN(payload.Fare) || payload.Fare < 0) {
      toast("Please enter a valid fare.", "error");
      return;
    }

    lastPassengerInput = payload;
    showLoading("Running prediction...");

    try {
      const result = await apiFetch("/predict", {
        method: "POST",
        body: JSON.stringify(payload),
      });
      lastPrediction = result;
      renderResult(result);
      showView("result");
      toast("Prediction complete", "success");
    } catch (err) {
      toast(err.message, "error");
    } finally {
      hideLoading();
    }
  }

  function renderResult(result) {
    const card = $("#result-card");
    const survived = result.survived;
    const prob = result.probability;

    card.className = `card result-card ${survived ? "survived" : "not-survived"}`;

    const iconName = survived ? "shield-check" : "shield-x";
    $("#result-icon").innerHTML = `<i data-lucide="${iconName}"></i>`;

    $("#result-verdict").textContent = survived ? "Survived" : "Did Not Survive";

    const pct = (prob * 100).toFixed(1);
    $("#result-probability").textContent = survived
      ? `${pct}% chance of survival`
      : `${(100 - prob * 100).toFixed(1)}% chance of not surviving`;

    $("#prob-bar-fill").style.width = `${prob * 100}%`;
    $("#result-meta").textContent = `Model version: ${result.model_version}`;

    window.lucide && window.lucide.createIcons();
  }

  // =========================================================
  //  Feedback ("Prediction was wrong")
  // =========================================================
  async function handleFeedback() {
    if (!lastPassengerInput || lastPrediction === null) return;

    showLoading("Submitting feedback...");

    try {
      await apiFetch("/feedback", {
        method: "POST",
        body: JSON.stringify({
          passenger_data: lastPassengerInput,
          actual_survived: !lastPrediction.survived,
        }),
      });
      showView("feedback");
      toast("Feedback recorded, dataset updated", "success");
    } catch (err) {
      toast(err.message, "error");
    } finally {
      hideLoading();
    }
  }

  // =========================================================
  //  Train / Retrain
  // =========================================================
  async function handleTrain() {
    showLoading("Training model... this may take a moment.");

    try {
      const result = await apiFetch("/retrain", { method: "POST" });

      // Show results in the train view
      const resultsEl = $("#train-results");
      resultsEl.style.display = "block";
      $("#train-results-subtitle").textContent = `Version ${result.model_version} is now live.`;
      renderMetrics($("#train-metrics-grid"), result);
      renderDetails($("#train-details"), result);

      modelLoaded = true;
      updateModelWarnings();
      await refreshModelBadge();

      toast("Model trained successfully", "success");
      window.lucide && window.lucide.createIcons();
    } catch (err) {
      toast(err.message, "error");
    } finally {
      hideLoading();
    }
  }

  async function handleRetrainFromFeedback() {
    showLoading("Retraining model with updated data...");

    try {
      const result = await apiFetch("/retrain", { method: "POST" });

      // Switch to the train view to display results
      $("#train-results").style.display = "block";
      $("#train-results-title").textContent = "Retraining Complete";
      $("#train-results-subtitle").textContent = `Version ${result.model_version} is now live with your corrections.`;
      renderMetrics($("#train-metrics-grid"), result);
      renderDetails($("#train-details"), result);
      showView("train");

      modelLoaded = true;
      updateModelWarnings();
      await refreshModelBadge();

      toast("Model retrained successfully", "success");
      window.lucide && window.lucide.createIcons();
    } catch (err) {
      toast(err.message, "error");
    } finally {
      hideLoading();
    }
  }

  // =========================================================
  //  Drift Simulation
  // =========================================================
  async function handleSimulateDrift() {
    const input = $("#field-drift-n-main");
    const n = parseInt(input.value, 10) || 500;
    showLoading(`Injecting ${n} synthetic samples...`);

    try {
      const result = await apiFetch(`/simulate-drift?n_samples=${n}`, {
        method: "POST",
      });

      const resultsEl = $("#drift-results");
      resultsEl.style.display = "block";

      const injected = result.drift_injection?.n_injected || n;
      const totalRows = result.drift_injection?.total_rows || "unknown";
      $("#drift-results-subtitle").textContent =
        `${injected} synthetic samples injected. The dataset now contains drifted data.`;

      const details = [];
      details.push(`Samples injected: <strong>${injected}</strong>`);
      details.push(`Total dataset size: <strong>${totalRows} rows</strong>`);
      if (result.drift_report && !result.drift_report.startsWith("Could not")) {
        details.push(`Drift report generated: <code>${result.drift_report}</code>`);
      } else if (result.drift_report) {
        details.push(`Note: ${result.drift_report}`);
      }
      $("#drift-details").innerHTML = `<p>${details.join("<br/>")}</p>`;

      // Point the report link directly to the generated report
      const reportLink = $("#btn-drift-report");
      if (reportLink && result.drift_report && !result.drift_report.startsWith("Could not")) {
        reportLink.href = `/monitoring/report/${result.drift_report}`;
      }

      toast("Drift injected. Check the drift report before retraining.", "info");
      window.lucide && window.lucide.createIcons();
    } catch (err) {
      toast(err.message, "error");
    } finally {
      hideLoading();
    }
  }

  async function handleDriftRetrain() {
    showLoading("Retraining model on drifted data...");

    try {
      const result = await apiFetch("/retrain", { method: "POST" });

      // Show results in the train view
      $("#train-results").style.display = "block";
      $("#train-results-title").textContent = "Retraining Complete (Drifted Data)";
      $("#train-results-subtitle").textContent = `Version ${result.model_version} trained on corrupted data.`;
      renderMetrics($("#train-metrics-grid"), result);
      renderDetails($("#train-details"), result);
      showView("train");

      modelLoaded = true;
      updateModelWarnings();
      await refreshModelBadge();

      toast("Model retrained on drifted data", "success");
      window.lucide && window.lucide.createIcons();
    } catch (err) {
      toast(err.message, "error");
    } finally {
      hideLoading();
    }
  }

  // =========================================================
  //  Reset Dataset
  // =========================================================
  async function handleResetDataset() {
    showLoading("Resetting dataset to original...");

    try {
      const result = await apiFetch("/reset-data", { method: "POST" });

      const resultEl = $("#reset-result");
      resultEl.style.display = "block";
      $("#reset-result-title").textContent = "Dataset Reset Successfully";
      $("#reset-result-body").innerHTML =
        `<p>${escapeHtml(result.message)}<br/>Dataset now has <strong>${result.total_rows}</strong> rows.</p>`;

      toast("Dataset restored to original", "success");
      window.lucide && window.lucide.createIcons();
    } catch (err) {
      toast(err.message, "error");
    } finally {
      hideLoading();
    }
  }

  // =========================================================
  //  Health Check
  // =========================================================
  async function loadHealth() {
    const grid = $("#health-grid");
    grid.innerHTML = `<div class="health-item"><span class="hi-label"><i data-lucide="loader"></i> Loading...</span></div>`;
    window.lucide && window.lucide.createIcons();

    const items = [];

    // API health
    try {
      const h = await apiFetch("/health");
      items.push({
        icon: "server",
        label: "API Service",
        value: `<span class="status-dot green"></span> Healthy`,
      });
      items.push({
        icon: "cpu",
        label: "Model Loaded",
        value: h.model_loaded
          ? `<span class="status-dot green"></span> v${h.model_version}`
          : `<span class="status-dot red"></span> Not loaded`,
      });
    } catch {
      items.push({
        icon: "server",
        label: "API Service",
        value: `<span class="status-dot red"></span> Unreachable`,
      });
    }

    // Model info
    try {
      const m = await apiFetch("/model-info");
      if (m.model_run_id) {
        items.push({
          icon: "git-branch",
          label: "MLflow Run",
          value: `<code style="font-size:.75rem;color:var(--text-muted)">${m.model_run_id.substring(0, 12)}...</code>`,
        });
      }
    } catch {
      /* skip */
    }

    // Dataset info
    try {
      const d = await apiFetch("/dataset-info");
      items.push({
        icon: "table-2",
        label: "Dataset Size",
        value: `${d.total_rows} rows`,
      });
      items.push({
        icon: "percent",
        label: "Survival Rate",
        value: `${(d.survival_rate * 100).toFixed(1)}%`,
      });
      items.push({
        icon: "user",
        label: "Avg Age",
        value: `${d.age_stats.mean} years`,
      });
      items.push({
        icon: "users",
        label: "Sex Distribution",
        value: `${d.sex_distribution.male || 0} M / ${d.sex_distribution.female || 0} F`,
      });
    } catch {
      items.push({
        icon: "table-2",
        label: "Dataset",
        value: `<span class="status-dot red"></span> Unavailable`,
      });
    }

    // Monitoring health
    try {
      await fetch("/monitoring/health");
      items.push({
        icon: "activity",
        label: "Monitoring Service",
        value: `<span class="status-dot green"></span> Healthy`,
      });
    } catch {
      items.push({
        icon: "activity",
        label: "Monitoring Service",
        value: `<span class="status-dot yellow"></span> Unknown`,
      });
    }

    grid.innerHTML = items
      .map(
        (item) => `
      <div class="health-item">
        <span class="hi-label"><i data-lucide="${item.icon}"></i> ${item.label}</span>
        <span class="hi-value">${item.value}</span>
      </div>`
      )
      .join("");

    window.lucide && window.lucide.createIcons();
  }

  // =========================================================
  //  Event Binding
  // =========================================================
  function init() {
    // Sidebar navigation
    $$(".sidebar-link[data-view]").forEach((link) => {
      link.addEventListener("click", () => {
        const view = link.dataset.view;
        showView(view);
        if (view === "health") loadHealth();
      });
    });

    // Landing CTAs
    $("#btn-start-predict").addEventListener("click", () => showView("form"));
    $("#btn-start-health").addEventListener("click", () => {
      showView("health");
      loadHealth();
    });

    // No-model banner train buttons
    const bannerTrain = $("#btn-banner-train");
    if (bannerTrain) bannerTrain.addEventListener("click", () => showView("train"));

    const inlineTrain = $("#btn-inline-train");
    if (inlineTrain) inlineTrain.addEventListener("click", () => showView("train"));

    // Back buttons
    $("#btn-back-form").addEventListener("click", () => showView("landing"));
    $("#btn-back-result").addEventListener("click", () => showView("form"));
    $("#btn-back-feedback").addEventListener("click", () => showView("landing"));
    $("#btn-back-train").addEventListener("click", () => showView("landing"));
    $("#btn-back-drift").addEventListener("click", () => showView("landing"));
    $("#btn-back-health").addEventListener("click", () => showView("landing"));

    const backReset = $("#btn-back-reset");
    if (backReset) backReset.addEventListener("click", () => showView("landing"));

    // Prediction form
    $("#predict-form").addEventListener("submit", handlePredict);

    // Result actions
    $("#btn-wrong").addEventListener("click", handleFeedback);
    $("#btn-try-again").addEventListener("click", () => showView("form"));

    // Feedback retrain
    const retrainFeedback = $("#btn-retrain-feedback");
    if (retrainFeedback) retrainFeedback.addEventListener("click", handleRetrainFromFeedback);

    // Train view
    $("#btn-train-run").addEventListener("click", handleTrain);

    const postTrainPredict = $("#btn-post-train-predict");
    if (postTrainPredict) postTrainPredict.addEventListener("click", () => showView("form"));

    // Drift view
    $("#btn-drift-run").addEventListener("click", handleSimulateDrift);

    const driftRetrain = $("#btn-drift-retrain");
    if (driftRetrain) driftRetrain.addEventListener("click", handleDriftRetrain);

    // Reset data
    const resetBtn = $("#btn-reset-data-run");
    if (resetBtn) resetBtn.addEventListener("click", handleResetDataset);

    // Health refresh
    $("#btn-refresh-health").addEventListener("click", loadHealth);

    // Init Lucide icons
    window.lucide && window.lucide.createIcons();

    // Check model status on load
    checkModelStatus();
  }

  // Start
  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init);
  } else {
    init();
  }
})();
