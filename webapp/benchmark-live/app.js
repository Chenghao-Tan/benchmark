const DATA_PATH = "data/benchmark_results.csv";

const state = {
  rows: [],
  metricNames: [],
  directions: {},
};

const els = {
  datasetSelect: document.getElementById("datasetSelect"),
  modelSelect: document.getElementById("modelSelect"),
  methodSelect: document.getElementById("methodSelect"),
  metricEditor: document.getElementById("metricEditor"),
  rankBtn: document.getElementById("rankBtn"),
  resetBtn: document.getElementById("resetBtn"),
  tableHead: document.querySelector("#resultTable thead"),
  tableBody: document.querySelector("#resultTable tbody"),
  summary: document.getElementById("summary"),
  topCards: document.getElementById("topCards"),
  configUpload: document.getElementById("configUpload"),
  configStatus: document.getElementById("configStatus"),
};

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

function renderTable(rows, selectedMetrics) {
  const columns = ["rank", "method", "score", "count", ...selectedMetrics.map((m) => m.metric)];

  els.tableHead.innerHTML = `<tr>${columns.map((c) => `<th>${c}</th>`).join("")}</tr>`;

  els.tableBody.innerHTML = rows
    .map((row) => {
      return `<tr>${columns
        .map((col) => {
          const val = row[col];
          const text = Number.isFinite(val) ? Number(val).toFixed(4) : String(val ?? "");
          return `<td>${text}</td>`;
        })
        .join("")}</tr>`;
    })
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

  const rows = [...els.metricEditor.querySelectorAll(".metricRow")];
  rows.forEach((row) => {
    row.querySelector(".metric-use").checked = false;
  });

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

  rows.forEach((row) => {
    const metric = row.dataset.metric;
    if (metricSet.has(metric)) {
      row.querySelector(".metric-use").checked = true;
    }
  });
}

function onConfigUpload(event) {
  const file = event.target.files?.[0];
  if (!file) return;

  const reader = new FileReader();
  reader.onload = () => {
    try {
      const raw = String(reader.result || "");
      const parsed = window.jsyaml.load(raw);
      if (!parsed || typeof parsed !== "object") {
        throw new Error("Parsed YAML is empty or invalid.");
      }
      applyConfigToUI(parsed);
      els.configStatus.textContent = `Loaded config: ${file.name}`;
      rankMethods();
    } catch (err) {
      els.configStatus.textContent = `Config parse error: ${err.message}`;
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
  els.configUpload.value = "";
  els.configStatus.textContent = "No config uploaded.";
  rankMethods();
}

function initFromRows(rows) {
  state.rows = rows;
  const datasets = uniqueSorted(rows.map((r) => r.dataset));
  const models = uniqueSorted(rows.map((r) => r.model));
  const methods = uniqueSorted(rows.map((r) => r.method));

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

  state.metricNames = metricNames;
  metricNames.forEach((m) => {
    state.directions[m] = inferDirection(m);
  });

  setSelectOptions(els.datasetSelect, datasets, true);
  setSelectOptions(els.modelSelect, models, true);
  setSelectOptions(els.methodSelect, methods, false);

  buildMetricEditor();
  rankMethods();
}

function loadData() {
  Papa.parse(DATA_PATH, {
    download: true,
    header: true,
    skipEmptyLines: true,
    complete: (result) => {
      if (result.errors.length) {
        els.summary.textContent = `CSV load error: ${result.errors[0].message}`;
        return;
      }
      initFromRows(result.data);
    },
    error: (err) => {
      els.summary.textContent = `CSV load error: ${err.message}`;
    },
  });
}

els.rankBtn.addEventListener("click", rankMethods);
els.resetBtn.addEventListener("click", resetFilters);
els.configUpload.addEventListener("change", onConfigUpload);

loadData();
