const LOCAL_DATA_PATH = "data/benchmark_results.csv";

const METRIC_DEFS = [
  { id: "validity", label: "Validity", column: "validity", direction: "max" },
  { id: "proximity", label: "Proximity (L2)", column: "distance_l2", direction: "min" },
  { id: "sparsity", label: "Sparsity (L0)", column: "distance_l0", direction: "min" },
  { id: "plausibility", label: "Plausibility (YNN)", column: "ynn", direction: "max" },
  { id: "runtime", label: "Runtime", column: "runtime_seconds", direction: "min" },
];

const METRIC_PRESETS = {
  balanced: METRIC_DEFS.map((m) => m.id),
  quality: ["validity", "plausibility", "sparsity"],
  speed: ["runtime", "proximity", "sparsity"],
};

const state = {
  rows: [],
  options: {
    datasets: [],
    models: [],
    methods: [],
  },
  selectedDataset: "",
  selectedModel: "",
  methodSearch: "",
  useAllMethods: true,
  selectedMethods: new Set(),
  selectedMetrics: new Set(METRIC_DEFS.map((m) => m.id)),
  activePreset: "balanced",
  metricBounds: {},
};

const els = {
  datasetRail: document.getElementById("datasetRail"),
  modelRail: document.getElementById("modelRail"),
  methodSearch: document.getElementById("methodSearch"),
  methodPalette: document.getElementById("methodPalette"),
  metricPalette: document.getElementById("metricPalette"),
  modeAllMethodsBtn: document.getElementById("modeAllMethodsBtn"),
  modeCustomMethodsBtn: document.getElementById("modeCustomMethodsBtn"),
  selectAllMethodsBtn: document.getElementById("selectAllMethodsBtn"),
  clearMethodsBtn: document.getElementById("clearMethodsBtn"),
  selectAllMetricsBtn: document.getElementById("selectAllMetricsBtn"),
  clearMetricsBtn: document.getElementById("clearMetricsBtn"),
  presetBtns: [...document.querySelectorAll(".presetBtn")],
  rankBtn: document.getElementById("rankBtn"),
  resetBtn: document.getElementById("resetBtn"),
  reloadBtn: document.getElementById("reloadBtn"),
  selectionHeadline: document.getElementById("selectionHeadline"),
  methodSelectedCount: document.getElementById("methodSelectedCount"),
  metricSelectedCount: document.getElementById("metricSelectedCount"),
  selectionPills: document.getElementById("selectionPills"),
  statusMessage: document.getElementById("statusMessage"),
  visualSection: document.getElementById("visualSection"),
  tableSection: document.getElementById("tableSection"),
  tradeoffPlot: document.getElementById("tradeoffPlot"),
  radarPlot: document.getElementById("radarPlot"),
  summary: document.getElementById("summary"),
  tableHead: document.querySelector("#resultTable thead"),
  tableBody: document.querySelector("#resultTable tbody"),
};

let radarChart = null;

function parseNumber(value) {
  if (value === null || value === undefined) return NaN;
  if (typeof value === "string" && value.trim() === "") return NaN;
  const n = Number(value);
  return Number.isFinite(n) ? n : NaN;
}

function mean(values) {
  const valid = values.filter((v) => Number.isFinite(v));
  if (!valid.length) return NaN;
  return valid.reduce((acc, v) => acc + v, 0) / valid.length;
}

function uniqueSorted(values) {
  return [...new Set(values)].sort((a, b) => String(a).localeCompare(String(b)));
}

function escapeHtml(value) {
  return String(value ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#39;");
}

function formatCell(value, digits = 4) {
  const n = parseNumber(value);
  return Number.isFinite(n) ? n.toFixed(digits) : "-";
}

function isSuccessful(row) {
  if (row.status === undefined || row.status === null || String(row.status).trim() === "") return true;
  return String(row.status).trim().toLowerCase() === "success";
}

function hideResultPanels() {
  els.visualSection.classList.add("hidden");
  els.tableSection.classList.add("hidden");
}

function showResultPanels() {
  els.visualSection.classList.remove("hidden");
  els.tableSection.classList.remove("hidden");
}

function clearResults(message) {
  els.statusMessage.textContent = message;
  els.summary.textContent = "";
  els.tradeoffPlot.innerHTML = "";
  if (radarChart) { radarChart.destroy(); radarChart = null; }
  if (els.radarPlot) els.radarPlot.innerHTML = "";
  els.tableHead.innerHTML = "";
  els.tableBody.innerHTML = "";
  hideResultPanels();
}

function markResultsStale() {
  const wasVisible = !els.visualSection.classList.contains("hidden") || !els.tableSection.classList.contains("hidden");
  if (wasVisible) {
    clearResults("Configuration changed. Click Show Benchmark to refresh visuals and ranking.");
  }
}

function renderScopeRail(container, options, selectedValue, type) {
  container.innerHTML = "";
  options.forEach((option) => {
    const btn = document.createElement("button");
    btn.type = "button";
    btn.className = `scopeToken${selectedValue === option ? " active" : ""}`;
    btn.dataset.type = type;
    btn.dataset.value = option;
    btn.textContent = option;
    container.appendChild(btn);
  });
}

function renderMethodPalette() {
  const filter = state.methodSearch.trim().toLowerCase();
  const methods = filter
    ? state.options.methods.filter((m) => m.toLowerCase().includes(filter))
    : state.options.methods;

  els.methodPalette.innerHTML = "";

  if (!methods.length) {
    els.methodPalette.innerHTML = '<span class="small">No methods match your search.</span>';
    return;
  }

  methods.forEach((method) => {
    const btn = document.createElement("button");
    btn.type = "button";
    const isActive = !state.useAllMethods && state.selectedMethods.has(method);
    btn.className = `token${isActive ? " active" : ""}${state.useAllMethods ? " subtle" : ""}`;
    btn.dataset.method = method;
    btn.textContent = method;
    els.methodPalette.appendChild(btn);
  });
}

function renderMetricPalette() {
  els.metricPalette.innerHTML = "";

  METRIC_DEFS.forEach((metric) => {
    const btn = document.createElement("button");
    btn.type = "button";
    btn.className = `token${state.selectedMetrics.has(metric.id) ? " active" : ""}`;
    btn.dataset.metricId = metric.id;
    btn.innerHTML = `${escapeHtml(metric.label)} <span class="dir">(${metric.direction === "max" ? "max" : "min"})</span>`;
    els.metricPalette.appendChild(btn);
  });

  els.presetBtns.forEach((button) => {
    button.classList.toggle("active", button.dataset.preset === state.activePreset);
  });
}

function selectedMetricConfig() {
  return METRIC_DEFS
    .filter((metric) => state.selectedMetrics.has(metric.id))
    .map((metric) => ({
      metricId: metric.id,
      label: metric.label,
      column: metric.column,
      direction: metric.direction,
    }));
}

function getMethodSubset() {
  if (state.useAllMethods) return state.options.methods;
  return state.options.methods.filter((method) => state.selectedMethods.has(method));
}

function ensureCustomSelectionSeed() {
  if (state.useAllMethods) return;
  if (state.selectedMethods.size) return;
  if (state.options.methods.length) {
    state.selectedMethods = new Set([state.options.methods[0]]);
  }
}

function updateSelectionUI() {
  const methodCount = state.useAllMethods ? state.options.methods.length : state.selectedMethods.size;

  els.modeAllMethodsBtn.classList.toggle("active", state.useAllMethods);
  els.modeCustomMethodsBtn.classList.toggle("active", !state.useAllMethods);

  els.methodSelectedCount.textContent = state.useAllMethods
    ? `All (${state.options.methods.length})`
    : `${state.selectedMethods.size} selected`;

  els.metricSelectedCount.textContent = `${state.selectedMetrics.size} selected`;

  els.selectionHeadline.textContent = state.selectedDataset && state.selectedModel
    ? `Scope: ${state.selectedDataset} / ${state.selectedModel}`
    : "Choose a dataset and model to begin.";

  const methodLabel = state.useAllMethods ? "Methods: all" : `Methods: ${state.selectedMethods.size}`;
  const metricLabel = state.selectedMetrics.size ? `Metrics: ${state.selectedMetrics.size}` : "Metrics: none";

  els.selectionPills.innerHTML = [
    `<span class="selectionPill">Dataset: ${escapeHtml(state.selectedDataset || "not selected")}</span>`,
    `<span class="selectionPill">Model: ${escapeHtml(state.selectedModel || "not selected")}</span>`,
    `<span class="selectionPill">${escapeHtml(methodLabel)} (visible ${methodCount})</span>`,
    `<span class="selectionPill">${escapeHtml(metricLabel)}</span>`,
  ].join("");

  const canRun = Boolean(state.selectedDataset && state.selectedModel && state.selectedMetrics.size > 0 && (state.useAllMethods || state.selectedMethods.size > 0));
  els.rankBtn.disabled = !canRun;
}

function aggregateByMethod(rows, metricColumns) {
  const groups = new Map();

  rows.forEach((row) => {
    const key = String(row.method);
    if (!groups.has(key)) groups.set(key, []);
    groups.get(key).push(row);
  });

  const aggregated = [];
  groups.forEach((methodRows, method) => {
    const item = { method, count: methodRows.length };
    metricColumns.forEach((column) => {
      item[column] = mean(methodRows.map((r) => parseNumber(r[column])));
    });
    aggregated.push(item);
  });

  return aggregated;
}

function computeMetricBounds(rows) {
  const bounds = {};

  METRIC_DEFS.forEach((metric) => {
    const values = rows.map((r) => parseNumber(r[metric.column])).filter((v) => Number.isFinite(v));
    const observedMax = values.length ? Math.max(...values) : NaN;

    if (metric.column === "validity" || metric.column === "ynn") {
      bounds[metric.column] = { min: 0, max: 1 };
      return;
    }

    const max = Number.isFinite(observedMax) && observedMax > 0 ? observedMax : 1;
    bounds[metric.column] = { min: 0, max };
  });

  return bounds;
}

function computeMetricRanges(rows, metricConfig) {
  const ranges = {};
  metricConfig.forEach((cfg) => {
    const fixed = state.metricBounds[cfg.column];
    if (fixed && Number.isFinite(fixed.min) && Number.isFinite(fixed.max) && fixed.max > fixed.min) {
      ranges[cfg.column] = { min: fixed.min, max: fixed.max };
      return;
    }

    const values = rows.map((r) => parseNumber(r[cfg.column])).filter((v) => Number.isFinite(v));
    if (!values.length) {
      ranges[cfg.column] = null;
      return;
    }
    ranges[cfg.column] = {
      min: Math.min(...values),
      max: Math.max(...values),
    };
  });
  return ranges;
}

function normalizeMetricValue(value, direction, range) {
  if (!range || !Number.isFinite(value)) return NaN;
  if (range.max <= range.min) return 1;

  const normalized = direction === "max"
    ? (value - range.min) / (range.max - range.min)
    : (range.max - value) / (range.max - range.min);

  return Math.max(0, Math.min(1, normalized));
}

function scoreRows(rows, metricConfig, ranges) {
  rows.forEach((row) => {
    let total = 0;
    let used = 0;

    metricConfig.forEach((cfg) => {
      const value = parseNumber(row[cfg.column]);
      const normalized = normalizeMetricValue(value, cfg.direction, ranges[cfg.column]);
      if (!Number.isFinite(normalized)) return;
      total += normalized;
      used += 1;
    });

    row.score = used ? total / used : NaN;
  });

  rows.sort((a, b) => {
    const av = Number.isFinite(a.score) ? a.score : -Infinity;
    const bv = Number.isFinite(b.score) ? b.score : -Infinity;
    return bv - av;
  });

  rows.forEach((row, index) => {
    row.rank = index + 1;
  });

  return rows;
}

function pickTradeoffMetrics() {
  return {
    x: { metricId: "proximity", label: "Proximity (L2)", column: "distance_l2", direction: "min" },
    y: { metricId: "validity", label: "Validity", column: "validity", direction: "max" },
  };
}

function renderTradeoffPlot(rows) {
  const pair = pickTradeoffMetrics();

  const width = 700;
  const height = 340;
  const pad = { left: 64, right: 20, top: 18, bottom: 56 };

  const xValues = rows.map((r) => parseNumber(r[pair.x.column])).filter((v) => Number.isFinite(v));
  const yValues = rows.map((r) => parseNumber(r[pair.y.column])).filter((v) => Number.isFinite(v));

  if (!xValues.length || !yValues.length) {
    els.tradeoffPlot.innerHTML = '<div class="tradeoffEmpty">Validity or Proximity (L2) values are missing for this selection.</div>';
    return;
  }

  const xBound = state.metricBounds?.[pair.x.column];
  const yBound = state.metricBounds?.[pair.y.column];

  const xMin = xBound && Number.isFinite(xBound.min) ? xBound.min : Math.min(...xValues);
  const xMaxRaw = xBound && Number.isFinite(xBound.max) ? xBound.max : Math.max(...xValues);
  const yMin = yBound && Number.isFinite(yBound.min) ? yBound.min : Math.min(...yValues);
  const yMaxRaw = yBound && Number.isFinite(yBound.max) ? yBound.max : Math.max(...yValues);

  const xMax = xMaxRaw > xMin ? xMaxRaw : xMin + 1;
  const yMax = yMaxRaw > yMin ? yMaxRaw : yMin + 1;

  const xSpan = xMax - xMin;
  const ySpan = yMax - yMin;

  const chartW = width - pad.left - pad.right;
  const chartH = height - pad.top - pad.bottom;

  const toX = (xv) => pad.left + ((xv - xMin) / xSpan) * chartW;
  const toY = (yv) => pad.top + chartH - ((yv - yMin) / ySpan) * chartH;

  const points = rows.map((row, index) => {
    const xv = parseNumber(row[pair.x.column]);
    const yv = parseNumber(row[pair.y.column]);
    if (!Number.isFinite(xv) || !Number.isFinite(yv)) return "";

    const cx = toX(xv);
    const cy = toY(yv);
    const score = Number.isFinite(row.score) ? row.score : 0;
    const fill = `hsl(${170 - score * 55} 70% ${44 - score * 8}%)`;

    const label = `${row.method}`;
    const approxLabelWidth = label.length * 7;
    const lx = Math.max(cx - 12, pad.left + approxLabelWidth + 4);
    const ly = Math.min(Math.max(cy + 4, pad.top + 12), pad.top + chartH - 4);

    return `
      <g>
        <circle cx="${cx.toFixed(2)}" cy="${cy.toFixed(2)}" r="7.5" fill="${fill}" stroke="#ffffff" stroke-width="1.4" opacity="0.95">
          <title>${escapeHtml(row.method)} | ${pair.x.label}: ${formatCell(xv, 3)} | ${pair.y.label}: ${formatCell(yv, 3)} | score: ${formatCell(row.score, 3)}</title>
        </circle>
        <text class="pointLabel" text-anchor="end" x="${lx.toFixed(2)}" y="${ly.toFixed(2)}">${escapeHtml(label)}</text>
      </g>
    `;
  }).join("");

  const axisColor = "#9d9788";
  const tickColor = "#8d8779";
  const gridColor = "#e1d8c8";
  const tickCount = 5;

  const fmtTick = (value) => {
    if (!Number.isFinite(value)) return "";
    if (Math.abs(value) >= 10) return value.toFixed(1);
    return value.toFixed(2);
  };

  const xTicks = Array.from({ length: tickCount + 1 }, (_, i) => xMin + (xSpan * i) / tickCount);
  const yTicks = Array.from({ length: tickCount + 1 }, (_, i) => yMin + (ySpan * i) / tickCount);

  const xTickMarkup = xTicks.map((tick) => {
    const x = toX(tick);
    return `
      <g>
        <line x1="${x.toFixed(2)}" y1="${pad.top}" x2="${x.toFixed(2)}" y2="${(pad.top + chartH).toFixed(2)}" stroke="${gridColor}" stroke-width="1" />
        <text x="${x.toFixed(2)}" y="${(pad.top + chartH + 16).toFixed(2)}" text-anchor="middle" font-size="11" fill="${tickColor}">${fmtTick(tick)}</text>
      </g>
    `;
  }).join("");

  const yTickMarkup = yTicks.map((tick) => {
    const y = toY(tick);
    return `
      <g>
        <line x1="${pad.left}" y1="${y.toFixed(2)}" x2="${(pad.left + chartW).toFixed(2)}" y2="${y.toFixed(2)}" stroke="${gridColor}" stroke-width="1" />
        <text x="${(pad.left - 8).toFixed(2)}" y="${(y + 4).toFixed(2)}" text-anchor="end" font-size="11" fill="${tickColor}">${fmtTick(tick)}</text>
      </g>
    `;
  }).join("");

  els.tradeoffPlot.innerHTML = `
    <svg viewBox="0 0 ${width} ${height}" width="100%" height="340" role="img" aria-label="Tradeoff scatter plot with raw metric values">
      ${xTickMarkup}
      ${yTickMarkup}
      <line x1="${pad.left}" y1="${pad.top + chartH}" x2="${pad.left + chartW}" y2="${pad.top + chartH}" stroke="${axisColor}" />
      <line x1="${pad.left}" y1="${pad.top}" x2="${pad.left}" y2="${pad.top + chartH}" stroke="${axisColor}" />
      <text x="${(pad.left + chartW / 2).toFixed(2)}" y="${(height - 16).toFixed(2)}" text-anchor="middle" font-size="12" fill="#595f58">${escapeHtml(pair.x.label)}</text>
      <text x="16" y="${(pad.top + chartH / 2).toFixed(2)}" text-anchor="middle" transform="rotate(-90, 16, ${(pad.top + chartH / 2).toFixed(2)})" font-size="12" fill="#595f58">${escapeHtml(pair.y.label)}</text>
      ${points}
    </svg>
  `;
}
function polygonArea(points) {
  if (!points.length) return 0;
  let sum = 0;
  for (let i = 0; i < points.length; i += 1) {
    const a = points[i];
    const b = points[(i + 1) % points.length];
    sum += a.x * b.y - b.x * a.y;
  }
  return Math.abs(sum) / 2;
}

function renderRadarFallbackSvg(series, metricConfig) {
  const size = 360;
  const center = size / 2;
  const radius = 132;
  const rings = [0.25, 0.5, 0.75, 1];

  const axes = metricConfig.map((cfg, index) => {
    const angle = -Math.PI / 2 + (2 * Math.PI * index) / metricConfig.length;
    const x = center + Math.cos(angle) * radius;
    const y = center + Math.sin(angle) * radius;
    const lx = center + Math.cos(angle) * (radius + 16);
    const ly = center + Math.sin(angle) * (radius + 16);
    return { label: cfg.label, angle, x, y, lx, ly };
  });

  const ringMarkup = rings.map((level) => {
    const points = axes.map((axis) => {
      const x = center + Math.cos(axis.angle) * radius * level;
      const y = center + Math.sin(axis.angle) * radius * level;
      return `${x.toFixed(2)},${y.toFixed(2)}`;
    }).join(" ");
    return `<polygon points="${points}" fill="none" stroke="#d8d0c1" stroke-width="1"></polygon>`;
  }).join("");

  const axisMarkup = axes.map((axis) => `
    <line x1="${center}" y1="${center}" x2="${axis.x.toFixed(2)}" y2="${axis.y.toFixed(2)}" stroke="#b9b09f" stroke-width="1"></line>
  `).join("");

  const labelMarkup = axes.map((axis) => `
    <text x="${axis.lx.toFixed(2)}" y="${axis.ly.toFixed(2)}" text-anchor="middle" dominant-baseline="middle" font-size="11" fill="#4e5a52">${escapeHtml(axis.label)}</text>
  `).join("");

  const polygons = series.map((item) => {
    const points = item.values.map((value, index) => {
      const axis = axes[index];
      const v = Math.max(0, Math.min(1, value));
      const x = center + Math.cos(axis.angle) * radius * v;
      const y = center + Math.sin(axis.angle) * radius * v;
      return `${x.toFixed(2)},${y.toFixed(2)}`;
    }).join(" ");

    return `
      <polygon points="${points}" fill="${item.fill}" stroke="${item.stroke}" stroke-width="2" opacity="0.9">
        <title>${escapeHtml(item.method)} (area ${item.areaPct.toFixed(1)}%)</title>
      </polygon>
    `;
  }).join("");

  const legendItems = series.map((item) => `
    <div class="radarLegendItem">
      <span class="radarSwatch" style="background:${item.stroke}"></span>
      <span class="radarName">${escapeHtml(item.method)}</span>
      <span class="radarArea">area ${item.areaPct.toFixed(1)}%</span>
    </div>
  `).join("");

  els.radarPlot.innerHTML = `
    <div class="radarWrap">
      <div class="radarCanvasWrap">
        <svg viewBox="0 0 ${size} ${size}" width="100%" height="100%" role="img" aria-label="Top method radar profile">
          ${ringMarkup}
          ${axisMarkup}
          ${polygons}
          ${labelMarkup}
        </svg>
      </div>
      <div class="radarLegend">${legendItems}</div>
    </div>
  `;
}

function renderRadarPlot(rows, metricConfig, ranges) {
  if (!els.radarPlot) return;

  if (radarChart) {
    radarChart.destroy();
    radarChart = null;
  }

  if (metricConfig.length < 2) {
    els.radarPlot.innerHTML = '<div class="tradeoffEmpty">Select at least 2 metrics to show a radar profile.</div>';
    return;
  }

  const top = rows.slice(0, Math.min(3, rows.length));
  if (!top.length) {
    els.radarPlot.innerHTML = '<div class="tradeoffEmpty">No methods available for radar plot.</div>';
    return;
  }

  const labels = metricConfig.map((m) => m.label);

  const palette = [
    { stroke: "#0f7a56", fill: "rgba(15,122,86,0.20)" },
    { stroke: "#126b8a", fill: "rgba(18,107,138,0.18)" },
    { stroke: "#da8e2f", fill: "rgba(218,142,47,0.20)" },
  ];

  const maxPolygonPoints = metricConfig.map((_, i) => {
    const angle = -Math.PI / 2 + (2 * Math.PI * i) / metricConfig.length;
    return { x: Math.cos(angle), y: Math.sin(angle) };
  });
  const maxArea = polygonArea(maxPolygonPoints) || 1;

  const datasets = [];
  const series = [];

  top.forEach((row, idx) => {
    const style = palette[idx % palette.length];
    const values = metricConfig.map((cfg) => {
      const raw = parseNumber(row[cfg.column]);
      const norm = normalizeMetricValue(raw, cfg.direction, ranges[cfg.column]);
      return Number.isFinite(norm) ? norm : 0;
    });

    datasets.push({
      label: row.method,
      data: values,
      rawValues: metricConfig.map((cfg) => parseNumber(row[cfg.column])),
      borderColor: style.stroke,
      backgroundColor: style.fill,
      pointBackgroundColor: style.stroke,
      pointBorderColor: "#ffffff",
      pointBorderWidth: 1,
      pointRadius: 3,
      borderWidth: 2,
      fill: true,
      tension: 0,
    });

    const pointsObj = values.map((v, i) => {
      const angle = -Math.PI / 2 + (2 * Math.PI * i) / values.length;
      return { x: Math.cos(angle) * v, y: Math.sin(angle) * v };
    });
    const areaPct = (polygonArea(pointsObj) / maxArea) * 100;

    series.push({
      method: row.method,
      values,
      stroke: style.stroke,
      fill: style.fill,
      areaPct,
    });
  });

  const hasChartLibrary = typeof window !== "undefined" && typeof window.Chart === "function";
  if (!hasChartLibrary) {
    renderRadarFallbackSvg(series, metricConfig);
    return;
  }

  const legendItems = series.map((item) => `
    <div class="radarLegendItem">
      <span class="radarSwatch" style="background:${item.stroke}"></span>
      <span class="radarName">${escapeHtml(item.method)}</span>
      <span class="radarArea">area ${item.areaPct.toFixed(1)}%</span>
    </div>
  `).join("");

  els.radarPlot.innerHTML = `
    <div class="radarWrap">
      <div class="radarCanvasWrap">
        <canvas id="radarCanvas"></canvas>
      </div>
      <div class="radarLegend">${legendItems}</div>
    </div>
  `;

  const canvas = document.getElementById("radarCanvas");
  if (!canvas || !canvas.getContext) {
    renderRadarFallbackSvg(series, metricConfig);
    return;
  }

  try {
    radarChart = new window.Chart(canvas.getContext("2d"), {
      type: "radar",
      data: {
        labels,
        datasets,
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        animation: false,
        plugins: {
          legend: { display: false },
          tooltip: {
            enabled: true,
            callbacks: {
              label: (context) => {
                const metric = metricConfig[context.dataIndex];
                const raw = context.dataset.rawValues?.[context.dataIndex];
                const utility = context.parsed?.r;
                return `${context.dataset.label} | ${metric?.label ?? "metric"}: utility ${formatCell(utility, 3)} | raw ${formatCell(raw, 3)}`;
              },
            },
          },
        },
        scales: {
          r: {
            min: 0,
            max: 1,
            ticks: {
              stepSize: 0.25,
              showLabelBackdrop: false,
              color: "#6a716a",
            },
            grid: { color: "#d8d0c1" },
            angleLines: { color: "#b9b09f" },
            pointLabels: {
              color: "#4e5a52",
              font: { size: 11, weight: "600" },
            },
          },
        },
      },
    });
  } catch (error) {
    renderRadarFallbackSvg(series, metricConfig);
    return;
  }

  if (typeof requestAnimationFrame === "function") {
    requestAnimationFrame(() => {
      if (radarChart) {
        radarChart.resize();
        radarChart.update("none");
      }
    });
  }
}
function renderTable(rows, metricConfig, ranges) {
  const headers = ["rank", "method", "score", ...metricConfig.map((cfg) => cfg.label)];

  els.tableHead.innerHTML = `<tr>${headers.map((h) => `<th>${escapeHtml(h)}</th>`).join("")}</tr>`;
  els.tableBody.innerHTML = rows.map((row) => {
    const scorePct = Number.isFinite(row.score) ? Math.max(0, Math.min(100, row.score * 100)) : 0;

    const metricCells = metricConfig.map((cfg) => {
      const value = parseNumber(row[cfg.column]);
      const normalized = normalizeMetricValue(value, cfg.direction, ranges[cfg.column]);
      const strength = Number.isFinite(normalized) ? 0.12 + normalized * 0.58 : 0.06;
      return `<td><div class="heatmapCell" style="--strength:${strength.toFixed(3)}">${formatCell(value, 3)}</div></td>`;
    }).join("");

    return `
      <tr>
        <td><span class="rankBadge">#${row.rank}</span></td>
        <td>${escapeHtml(row.method)}</td>
        <td>
          <div class="scoreCell">
            <div class="scoreTop"><span>${formatCell(row.score, 3)}</span><span>${scorePct.toFixed(1)}%</span></div>
            <div class="miniTrack"><div class="miniFill" style="width:${scorePct.toFixed(2)}%"></div></div>
          </div>
        </td>
        ${metricCells}
      </tr>
    `;
  }).join("");
}

function rankMethods() {
  try {
    const dataset = state.selectedDataset;
    const model = state.selectedModel;
    const metricConfig = selectedMetricConfig();

    if (!dataset || !model) {
      clearResults("Select both dataset and model first.");
      return;
    }

    if (!metricConfig.length) {
      clearResults("Select at least one metric.");
      return;
    }

    const metricColumns = metricConfig.map((m) => m.column);
    const aggregateColumns = [...new Set([...metricColumns, "validity", "distance_l2"])];

    const contextRows = state.rows
      .filter((r) => isSuccessful(r))
      .filter((r) => String(r.dataset) === dataset)
      .filter((r) => String(r.model) === model);

    if (!contextRows.length) {
      clearResults("No successful rows found for this dataset/model.");
      return;
    }

    const methodSubset = new Set(getMethodSubset());

    const filteredRows = contextRows.filter((r) => methodSubset.has(String(r.method)));
    if (!filteredRows.length) {
      clearResults("No rows remain after method filtering.");
      return;
    }

    const baselineAggregated = aggregateByMethod(contextRows, aggregateColumns);
    const aggregated = aggregateByMethod(filteredRows, aggregateColumns);

    if (!aggregated.length) {
      clearResults("No methods available for ranking.");
      return;
    }

    const ranges = computeMetricRanges(baselineAggregated, metricConfig);
    const ranked = scoreRows(aggregated, metricConfig, ranges);

    els.statusMessage.textContent = "Benchmark updated.";
    els.summary.textContent = `Rows evaluated: ${filteredRows.length}. Methods ranked: ${ranked.length}. Normalization uses fixed global metric bounds from the full benchmark CSV.`;

    showResultPanels();
    renderTradeoffPlot(ranked);
    renderRadarPlot(ranked, metricConfig, ranges);
    renderTable(ranked, metricConfig, ranges);
  } catch (error) {
    clearResults(`Ranking error: ${error.message}`);
  }
}

function setMethodsModeAll() {
  state.useAllMethods = true;
  state.selectedMethods.clear();
  renderMethodPalette();
  updateSelectionUI();
  markResultsStale();
}

function setMethodsModeCustom() {
  state.useAllMethods = false;
  ensureCustomSelectionSeed();
  renderMethodPalette();
  updateSelectionUI();
  markResultsStale();
}

function toggleMethodSelection(method) {
  if (state.useAllMethods) {
    state.useAllMethods = false;
    state.selectedMethods = new Set([method]);
  } else if (state.selectedMethods.has(method)) {
    state.selectedMethods.delete(method);
    if (!state.selectedMethods.size) {
      state.useAllMethods = true;
    }
  } else {
    state.selectedMethods.add(method);
  }

  renderMethodPalette();
  updateSelectionUI();
  markResultsStale();
}

function selectAllMethodsCustom() {
  state.useAllMethods = false;
  state.selectedMethods = new Set(state.options.methods);
  renderMethodPalette();
  updateSelectionUI();
  markResultsStale();
}

function clearMethodsCustom() {
  state.useAllMethods = false;
  state.selectedMethods.clear();
  ensureCustomSelectionSeed();
  renderMethodPalette();
  updateSelectionUI();
  markResultsStale();
}

function selectAllMetrics() {
  state.selectedMetrics = new Set(METRIC_DEFS.map((m) => m.id));
  state.activePreset = "balanced";
  renderMetricPalette();
  updateSelectionUI();
  markResultsStale();
}

function clearMetrics() {
  state.selectedMetrics.clear();
  state.activePreset = "";
  renderMetricPalette();
  updateSelectionUI();
  markResultsStale();
}

function applyMetricPreset(name) {
  const preset = METRIC_PRESETS[name];
  if (!preset) return;
  state.selectedMetrics = new Set(preset);
  state.activePreset = name;
  renderMetricPalette();
  updateSelectionUI();
  markResultsStale();
}

function toggleMetric(metricId) {
  if (state.selectedMetrics.has(metricId)) {
    state.selectedMetrics.delete(metricId);
  } else {
    state.selectedMetrics.add(metricId);
  }
  state.activePreset = "";
  renderMetricPalette();
  updateSelectionUI();
  markResultsStale();
}

function resetFilters() {
  state.selectedDataset = "";
  state.selectedModel = "";
  state.methodSearch = "";
  els.methodSearch.value = "";
  state.useAllMethods = true;
  state.selectedMethods.clear();
  state.selectedMetrics = new Set(METRIC_DEFS.map((m) => m.id));
  state.activePreset = "balanced";

  renderScopeRail(els.datasetRail, state.options.datasets, state.selectedDataset, "dataset");
  renderScopeRail(els.modelRail, state.options.models, state.selectedModel, "model");
  renderMethodPalette();
  renderMetricPalette();
  updateSelectionUI();
  clearResults("Filters reset. Choose scope and click Show Benchmark.");
}

function parseLocalCsv() {
  return new Promise((resolve, reject) => {
    if (!window.Papa) {
      reject(new Error("PapaParse is not loaded."));
      return;
    }

    Papa.parse(LOCAL_DATA_PATH, {
      download: true,
      header: true,
      skipEmptyLines: true,
      complete: (result) => {
        if (result.errors && result.errors.length) {
          reject(new Error(result.errors[0].message));
          return;
        }
        resolve(result.data || []);
      },
      error: reject,
    });
  });
}

function sanitizeRows(rows) {
  return rows
    .filter((row) => row && typeof row === "object")
    .map((row) => {
      const clean = { ...row };
      clean.dataset = String(clean.dataset ?? "").trim();
      clean.model = String(clean.model ?? "").trim();
      clean.method = String(clean.method ?? "").trim();
      clean.status = String(clean.status ?? "").trim();
      return clean;
    })
    .filter((row) => row.dataset && row.model && row.method);
}

function initFromRows(rows) {
  state.rows = sanitizeRows(rows);
  state.options.datasets = uniqueSorted(state.rows.map((r) => r.dataset));
  state.options.models = uniqueSorted(state.rows.map((r) => r.model));
  state.options.methods = uniqueSorted(state.rows.map((r) => r.method));
  state.metricBounds = computeMetricBounds(state.rows.filter((r) => isSuccessful(r)));

  state.selectedDataset = "";
  state.selectedModel = "";
  state.methodSearch = "";
  els.methodSearch.value = "";
  state.useAllMethods = true;
  state.selectedMethods.clear();
  state.selectedMetrics = new Set(METRIC_DEFS.map((m) => m.id));
  state.activePreset = "balanced";

  renderScopeRail(els.datasetRail, state.options.datasets, state.selectedDataset, "dataset");
  renderScopeRail(els.modelRail, state.options.models, state.selectedModel, "model");
  renderMethodPalette();
  renderMetricPalette();
  updateSelectionUI();
  clearResults("Select dataset/model and click Show Benchmark.");
}

async function loadData() {
  try {
    const rows = await parseLocalCsv();
    initFromRows(rows);
  } catch (error) {
    clearResults(`Failed to load CSV: ${error.message}`);
  }
}

els.rankBtn.addEventListener("click", rankMethods);
els.resetBtn.addEventListener("click", resetFilters);
els.reloadBtn.addEventListener("click", loadData);

els.modeAllMethodsBtn.addEventListener("click", setMethodsModeAll);
els.modeCustomMethodsBtn.addEventListener("click", setMethodsModeCustom);
els.selectAllMethodsBtn.addEventListener("click", selectAllMethodsCustom);
els.clearMethodsBtn.addEventListener("click", clearMethodsCustom);

els.selectAllMetricsBtn.addEventListener("click", selectAllMetrics);
els.clearMetricsBtn.addEventListener("click", clearMetrics);

els.datasetRail.addEventListener("click", (event) => {
  const button = event.target.closest("button.scopeToken");
  if (!button || button.dataset.type !== "dataset") return;

  const value = button.dataset.value || "";
  state.selectedDataset = state.selectedDataset === value ? "" : value;
  renderScopeRail(els.datasetRail, state.options.datasets, state.selectedDataset, "dataset");
  updateSelectionUI();
  markResultsStale();
});

els.modelRail.addEventListener("click", (event) => {
  const button = event.target.closest("button.scopeToken");
  if (!button || button.dataset.type !== "model") return;

  const value = button.dataset.value || "";
  state.selectedModel = state.selectedModel === value ? "" : value;
  renderScopeRail(els.modelRail, state.options.models, state.selectedModel, "model");
  updateSelectionUI();
  markResultsStale();
});

els.methodSearch.addEventListener("input", (event) => {
  state.methodSearch = String(event.target.value || "");
  renderMethodPalette();
});

els.methodPalette.addEventListener("click", (event) => {
  const button = event.target.closest("button.token");
  if (!button || !button.dataset.method) return;
  toggleMethodSelection(button.dataset.method);
});

els.metricPalette.addEventListener("click", (event) => {
  const button = event.target.closest("button.token");
  if (!button || !button.dataset.metricId) return;
  toggleMetric(button.dataset.metricId);
});

els.presetBtns.forEach((button) => {
  button.addEventListener("click", () => applyMetricPreset(button.dataset.preset));
});

loadData();
















































