
const LOCAL_DATA_PATH = "data/benchmark_results.csv";

const state = {
  rows: [],
  metricNames: [],
  directions: {},
  backendAvailable: false,
  uploadedYamlText: null,
  options: {
    datasets: [],
    models: [],
    methods: [],
  },
};

const els = {
  datasetSelect: document.getElementById("datasetSelect"),
  modelSelect: document.getElementById("modelSelect"),
  methodSelect: document.getElementById("methodSelect"),
  metricEditor: document.getElementById("metricEditor"),
  runMetricEditor: document.getElementById("runMetricEditor"),
  rankBtn: document.getElementById("rankBtn"),
  resetBtn: document.getElementById("resetBtn"),
  reloadBtn: document.getElementById("reloadBtn"),
  runSelectionBtn: document.getElementById("runSelectionBtn"),
  runConfigBtn: document.getElementById("runConfigBtn"),
  tableHead: document.querySelector("#resultTable thead"),
  tableBody: document.querySelector("#resultTable tbody"),
  summary: document.getElementById("summary"),
  topCards: document.getElementById("topCards"),
  configUpload: document.getElementById("configUpload"),
  configStatus: document.getElementById("configStatus"),
  jobStatus: document.getElementById("jobStatus"),
  modeHint: document.getElementById("modeHint"),
};

const RUN_METRIC_CATALOG = [
  "validity",
  "distance_l0",
  "distance_l1",
  "distance_l2",
  "distance_linf",
  "ynn",
  "runtime_seconds",
];

function parseNumber(value) {
  const n = Number(value);
  return Number.isFinite(n) ? n : NaN;
}

function mean(values) {
  const ok = values.filter((v) => Number.isFinite(v));
  if (!ok.length) return NaN;
  return ok.reduce((a, b) => a + b, 0) / ok.length;
}

function uniqueSorted(values) {
  return [...new Set(values)].sort((a, b) => String(a).localeCompare(String(b)));
}

function inferDirection(metric) {
  if (metric === "validity" || metric === "ynn") return "max";
  if (metric.startsWith("distance_") || metric.includes("runtime") || metric.includes("elapsed")) return "min";
  return "max";
}

function setSelectOptions(selectEl, values, includeAll = true) {
  const oldValue = selectEl.value;
  selectEl.innerHTML = "";
  if (includeAll) {
    const all = document.createElement("option");
    all.value = "";
    all.textContent = "All";
    selectEl.appendChild(all);
  }
  values.forEach((value) => {
    const option = document.createElement("option");
    option.value = String(value);
    option.textContent = String(value);
    selectEl.appendChild(option);
  });
  if ([...selectEl.options].some((o) => o.value === oldValue)) {
    selectEl.value = oldValue;
  }
}

function getSelectedMethods() {
  return [...els.methodSelect.selectedOptions].map((o) => o.value);
}

function buildMetricEditor() {
  els.metricEditor.innerHTML = "";
  state.metricNames.forEach((metric) => {
    const direction = state.directions[metric];
    const row = document.createElement("div");
    row.className = "metricRow";
    row.dataset.metric = metric;

    row.innerHTML = `
      <label class="metricName">
        <input type="checkbox" class="metric-use" ${metric === "validity" ? "checked" : ""} />
        <span>${metric}</span>
        <small>${direction === "max" ? "maximize" : "minimize"}</small>
      </label>
      <label>Weight
        <input class="metric-weight" type="number" step="0.1" min="0" value="1" />
      </label>
      <label>Min
        <input class="metric-min" type="number" step="any" placeholder="none" />
      </label>
      <label>Max
        <input class="metric-max" type="number" step="any" placeholder="none" />
      </label>
      <label>Direction
        <select class="metric-direction">
          <option value="max" ${direction === "max" ? "selected" : ""}>Max</option>
          <option value="min" ${direction === "min" ? "selected" : ""}>Min</option>
        </select>
      </label>
    `;

    els.metricEditor.appendChild(row);
  });
}

function buildRunMetricEditor() {
  els.runMetricEditor.innerHTML = "";
  const runMetrics = RUN_METRIC_CATALOG.filter((m) => state.metricNames.includes(m) || ["validity", "ynn", "runtime_seconds"].includes(m));
  runMetrics.forEach((metric) => {
    const row = document.createElement("div");
    row.className = "metricRow";
    row.dataset.metric = metric;
    const checked = ["validity", "distance_l1", "distance_l2", "ynn", "runtime_seconds"].includes(metric);
    row.innerHTML = `
      <label class="metricName">
        <input type="checkbox" class="run-metric-use" ${checked ? "checked" : ""} />
        <span>${metric}</span>
      </label>
      <div></div><div></div><div></div><div></div>
    `;
    els.runMetricEditor.appendChild(row);
  });
}

function collectMetricConfig() {
  return [...els.metricEditor.querySelectorAll(".metricRow")].map((row) => {
    const metric = row.dataset.metric;
    const use = row.querySelector(".metric-use").checked;
    const weight = parseNumber(row.querySelector(".metric-weight").value);
    const minVal = parseNumber(row.querySelector(".metric-min").value);
    const maxVal = parseNumber(row.querySelector(".metric-max").value);
    const direction = row.querySelector(".metric-direction").value;

    return {
      metric,
      use,
      direction,
      weight: Number.isFinite(weight) && weight > 0 ? weight : 1,
      min: Number.isFinite(minVal) ? minVal : null,
      max: Number.isFinite(maxVal) ? maxVal : null,
    };
  });
}

function collectRunMetrics() {
  return [...els.runMetricEditor.querySelectorAll(".metricRow")]
    .filter((row) => row.querySelector(".run-metric-use").checked)
    .map((row) => row.dataset.metric);
}

function aggregateByMethod(rows, metrics) {
  const groups = new Map();
  rows.forEach((row) => {
    if (!groups.has(row.method)) groups.set(row.method, []);
    groups.get(row.method).push(row);
  });

  const entries = [];
  groups.forEach((methodRows, method) => {
    const entry = { method, count: methodRows.length };
    metrics.forEach((metric) => {
      entry[metric] = mean(methodRows.map((r) => parseNumber(r[metric])));
    });
    entries.push(entry);
  });

  return entries;
}

function applyConstraints(rows, metricConfig) {
  return rows.filter((row) => {
    for (const cfg of metricConfig) {
      if (cfg.min === null && cfg.max === null) continue;
      const val = parseNumber(row[cfg.metric]);
      if (!Number.isFinite(val)) return false;
      if (cfg.min !== null && val < cfg.min) return false;
      if (cfg.max !== null && val > cfg.max) return false;
    }
    return true;
  });
}
function scoreMethods(methodRows, selectedMetrics) {
  const ranges = {};
  selectedMetrics.forEach((cfg) => {
    const values = methodRows.map((r) => parseNumber(r[cfg.metric])).filter((v) => Number.isFinite(v));
    if (!values.length) {
      ranges[cfg.metric] = null;
      return;
    }
    ranges[cfg.metric] = { min: Math.min(...values), max: Math.max(...values) };
  });

  methodRows.forEach((row) => {
    let total = 0;
    let totalWeight = 0;
    selectedMetrics.forEach((cfg) => {
      const range = ranges[cfg.metric];
      const val = parseNumber(row[cfg.metric]);
      if (!range || !Number.isFinite(val)) return;

      let normalized = 1;
      if (range.max > range.min) {
        if (cfg.direction === "max") {
          normalized = (val - range.min) / (range.max - range.min);
        } else {
          normalized = (range.max - val) / (range.max - range.min);
        }
      }

      total += normalized * cfg.weight;
      totalWeight += cfg.weight;
    });

    row.score = totalWeight > 0 ? total / totalWeight : NaN;
  });

  methodRows.sort((a, b) => (b.score ?? -Infinity) - (a.score ?? -Infinity));
  methodRows.forEach((row, idx) => {
    row.rank = idx + 1;
  });

  return methodRows;
}

function renderCards(rows, selectedMetrics) {
  els.topCards.innerHTML = "";
  const top = rows.slice(0, 3);
  if (!top.length) {
    els.topCards.innerHTML = "<p>No matching methods.</p>";
    return;
  }

  top.forEach((row) => {
    const metricLines = selectedMetrics
      .slice(0, 3)
      .map((m) => `${m.metric}: ${Number.isFinite(row[m.metric]) ? row[m.metric].toFixed(4) : "n/a"}`)
      .join("<br>");

    const card = document.createElement("article");
    card.className = "card";
    card.innerHTML = `
      <span class="rankTag">#${row.rank}</span>
      <h3>${row.method}</h3>
      <p><strong>Score:</strong> ${Number.isFinite(row.score) ? row.score.toFixed(4) : "n/a"}</p>
      <p><strong>Rows used:</strong> ${row.count}</p>
      <p>${metricLines}</p>
    `;
    els.topCards.appendChild(card);
  });
}

function formatCell(val) {
  if (typeof val === "number") return Number.isFinite(val) ? val.toFixed(4) : "";
  const parsed = parseNumber(val);
  if (Number.isFinite(parsed)) return parsed.toFixed(4);
  return String(val ?? "");
}

function renderTable(rows, selectedMetrics) {
  const columns = ["rank", "method", "score", "count", ...selectedMetrics.map((m) => m.metric)];

  els.tableHead.innerHTML = `<tr>${columns.map((c) => `<th>${c}</th>`).join("")}</tr>`;

  els.tableBody.innerHTML = rows
    .map((row) => `<tr>${columns.map((col) => `<td>${formatCell(row[col])}</td>`).join("")}</tr>`)
    .join("");
}
function rankMethods() {
  const dataset = els.datasetSelect.value;
  const model = els.modelSelect.value;
  const selectedMethods = getSelectedMethods();
  const metricConfig = collectMetricConfig();
  const selectedMetrics = metricConfig.filter((m) => m.use);

  if (!selectedMetrics.length) {
    els.summary.textContent = "Select at least one ranking metric.";
    els.tableHead.innerHTML = "";
    els.tableBody.innerHTML = "";
    els.topCards.innerHTML = "";
    return;
  }

  let filtered = state.rows.filter((r) => r.status === "success");
  if (dataset) filtered = filtered.filter((r) => r.dataset === dataset);
  if (model) filtered = filtered.filter((r) => r.model === model);
  if (selectedMethods.length) {
    const methodSet = new Set(selectedMethods);
    filtered = filtered.filter((r) => methodSet.has(r.method));
  }

  filtered = applyConstraints(filtered, metricConfig);

  if (!filtered.length) {
    els.summary.textContent = "No rows satisfy current filters/constraints.";
    els.tableHead.innerHTML = "";
    els.tableBody.innerHTML = "";
    els.topCards.innerHTML = "";
    return;
  }

  const methodRows = aggregateByMethod(filtered, state.metricNames);
  const ranked = scoreMethods(methodRows, selectedMetrics);

  els.summary.textContent = `Evaluated ${filtered.length} experiment rows across ${ranked.length} methods.`;
  renderCards(ranked, selectedMetrics);
  renderTable(ranked, selectedMetrics);
}

function applyConfigToUI(config) {
  if (config?.dataset?.name) {
    const name = String(config.dataset.name);
    if ([...els.datasetSelect.options].some((o) => o.value === name)) {
      els.datasetSelect.value = name;
    }
  }

  if (config?.model?.name) {
    const name = String(config.model.name);
    if ([...els.modelSelect.options].some((o) => o.value === name)) {
      els.modelSelect.value = name;
    }
  }

  const evalList = Array.isArray(config?.evaluation) ? config.evaluation : [];
  const metricSet = new Set();

  evalList.forEach((item) => {
    const name = String(item?.name || "").toLowerCase();
    if (name === "validity") metricSet.add("validity");
    if (name === "ynn") metricSet.add("ynn");
    if (name === "runtime") metricSet.add("runtime_seconds");
    if (name === "distance") {
      const metrics = Array.isArray(item.metrics) ? item.metrics : ["l0", "l1", "l2", "linf"];
      metrics.forEach((m) => metricSet.add(`distance_${String(m).toLowerCase()}`));
    }
  });

  [...els.runMetricEditor.querySelectorAll(".metricRow")].forEach((row) => {
    const checkbox = row.querySelector(".run-metric-use");
    if (metricSet.size) {
      checkbox.checked = metricSet.has(row.dataset.metric);
    }
  });
}
function updateJobStatus(text) {
  els.jobStatus.textContent = text;
}

async function postJson(url, body) {
  const response = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!response.ok) {
    const text = await response.text();
    throw new Error(text || `HTTP ${response.status}`);
  }
  return response.json();
}

async function pollJob(jobId) {
  let keepPolling = true;
  while (keepPolling) {
    await new Promise((resolve) => setTimeout(resolve, 1800));
    const response = await fetch(`/api/jobs/${jobId}`);
    if (!response.ok) {
      updateJobStatus(`Job ${jobId}: status check failed (${response.status}).`);
      return;
    }
    const payload = await response.json();
    const progress = payload.progress || {};
    updateJobStatus(
      `Job ${jobId}\nstatus: ${payload.status}\ncompleted: ${progress.completed ?? 0}/${progress.total ?? 0}\nfailed: ${progress.failed ?? 0}${payload.error ? `\nerror: ${payload.error}` : ""}`
    );

    if (["completed", "failed"].includes(payload.status)) {
      keepPolling = false;
      await loadData();
    }
  }
}

async function runSelection() {
  if (!state.backendAvailable) {
    updateJobStatus("Backend unavailable. Running new experiments requires server mode.");
    return;
  }

  const dataset = els.datasetSelect.value;
  const model = els.modelSelect.value;
  const selectedMethods = getSelectedMethods();
  const methods = selectedMethods.length ? selectedMethods : [...els.methodSelect.options].map((o) => o.value);
  const metrics = collectRunMetrics();

  if (!dataset || !model) {
    updateJobStatus("Select dataset and model before running experiments.");
    return;
  }
  if (!methods.length) {
    updateJobStatus("No methods selected or available.");
    return;
  }

  try {
    const payload = await postJson("/api/run-selection", {
      dataset,
      model,
      methods,
      metrics,
    });
    updateJobStatus(`Job ${payload.job_id} submitted.`);
    pollJob(payload.job_id);
  } catch (err) {
    updateJobStatus(`Run failed to submit: ${err.message}`);
  }
}

async function runUploadedConfig() {
  if (!state.backendAvailable) {
    updateJobStatus("Backend unavailable. Running uploaded config requires server mode.");
    return;
  }
  if (!state.uploadedYamlText) {
    updateJobStatus("Upload a YAML config first.");
    return;
  }

  const metrics = collectRunMetrics();
  const methods = getSelectedMethods();

  try {
    const payload = await postJson("/api/run-config", {
      yaml_text: state.uploadedYamlText,
      metrics,
      methods,
    });
    updateJobStatus(`Job ${payload.job_id} submitted from uploaded config.`);
    pollJob(payload.job_id);
  } catch (err) {
    updateJobStatus(`Config run submission failed: ${err.message}`);
  }
}
function onConfigUpload(event) {
  const file = event.target.files?.[0];
  if (!file) return;

  const reader = new FileReader();
  reader.onload = () => {
    try {
      const raw = String(reader.result || "");
      state.uploadedYamlText = raw;
      const parsed = window.jsyaml.load(raw);
      if (!parsed || typeof parsed !== "object") {
        throw new Error("Parsed YAML is empty or invalid.");
      }
      applyConfigToUI(parsed);
      els.configStatus.textContent = `Loaded config: ${file.name}`;
      rankMethods();
    } catch (err) {
      els.configStatus.textContent = `Config parse error: ${err.message}`;
      state.uploadedYamlText = null;
    }
  };
  reader.readAsText(file);
}

function resetFilters() {
  els.datasetSelect.value = "";
  els.modelSelect.value = "";
  [...els.methodSelect.options].forEach((o) => {
    o.selected = false;
  });
  buildMetricEditor();
  buildRunMetricEditor();
  els.configUpload.value = "";
  els.configStatus.textContent = "No config uploaded.";
  state.uploadedYamlText = null;
  rankMethods();
}

function parseLocalCsv() {
  return new Promise((resolve, reject) => {
    Papa.parse(LOCAL_DATA_PATH, {
      download: true,
      header: true,
      skipEmptyLines: true,
      complete: (result) => {
        if (result.errors.length) {
          reject(new Error(result.errors[0].message));
        } else {
          resolve(result.data);
        }
      },
      error: reject,
    });
  });
}

async function fetchBackendData() {
  const response = await fetch("/api/results");
  if (!response.ok) throw new Error(`HTTP ${response.status}`);
  const payload = await response.json();
  const optionsResp = await fetch("/api/options");
  const options = optionsResp.ok ? await optionsResp.json() : null;
  return { rows: payload.rows || [], options };
}

function initFromRows(rows) {
  state.rows = rows;
  const datasets = state.options.datasets.length ? state.options.datasets : uniqueSorted(rows.map((r) => r.dataset));
  const models = state.options.models.length ? state.options.models : uniqueSorted(rows.map((r) => r.model));
  const methods = state.options.methods.length ? state.options.methods : uniqueSorted(rows.map((r) => r.method));

  const excluded = new Set([
    "config_file",
    "run_name",
    "dataset",
    "model",
    "method",
    "status",
    "error",
    "traceback",
  ]);

  const sample = rows[0] || {};
  const metricNames = Object.keys(sample)
    .filter((key) => !excluded.has(key))
    .filter((key) => rows.some((r) => Number.isFinite(parseNumber(r[key]))));

  state.metricNames = uniqueSorted(metricNames);
  state.metricNames.forEach((m) => {
    state.directions[m] = inferDirection(m);
  });

  setSelectOptions(els.datasetSelect, datasets, true);
  setSelectOptions(els.modelSelect, models, true);
  setSelectOptions(els.methodSelect, methods, false);

  buildMetricEditor();
  buildRunMetricEditor();
  rankMethods();
}

async function loadData() {
  try {
    const backend = await fetchBackendData();
    state.backendAvailable = true;
    state.options = {
      datasets: backend.options?.datasets || [],
      models: backend.options?.models || [],
      methods: backend.options?.methods || [],
    };
    els.modeHint.textContent = "Server mode: enabled. You can run new experiments from this page.";
    initFromRows(backend.rows);
    return;
  } catch (_err) {
    state.backendAvailable = false;
    state.options = { datasets: [], models: [], methods: [] };
  }

  try {
    const rows = await parseLocalCsv();
    els.modeHint.textContent = "Static mode: running new experiments is disabled (no backend detected).";
    initFromRows(rows);
  } catch (err) {
    els.summary.textContent = `Failed to load data: ${err.message}`;
  }
}

els.rankBtn.addEventListener("click", rankMethods);
els.resetBtn.addEventListener("click", resetFilters);
els.reloadBtn.addEventListener("click", loadData);
els.runSelectionBtn.addEventListener("click", runSelection);
els.runConfigBtn.addEventListener("click", runUploadedConfig);
els.configUpload.addEventListener("change", onConfigUpload);

loadData();
