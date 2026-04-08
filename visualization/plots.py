"""Plotly figures for Streamlit dashboard (also exportable to HTML)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go


def _safe_float(x: Any, default: float = 0.0) -> float:
    """JSON null / NaN-safe float for Plotly (instance metrics use null for failed CF rows)."""
    if x is None:
        return default
    try:
        v = float(x)
    except (TypeError, ValueError):
        return default
    if np.isnan(v):
        return default
    return v


def fig_radar(summary_row: pd.Series, title: str) -> go.Figure:
    axes = [
        ("valid_mean", "validity"),
        ("cost_l1_mean", "cost↓"),
        ("sparsity_l0_mean", "sparsity"),
        ("PGI_rec_mean", "PGI_rec"),
        ("RIS_rec_mean", "RIS_rec↓"),
        ("plausibility_density_mean", "plausibility↓"),
        ("fairness_delta", "fair_gap↓"),
        ("hypervolume_mean", "hypervol"),
    ]
    values: list[float] = []
    labels: list[str] = []
    for key, lab in axes:
        v = float(summary_row[key]) if key in summary_row.index else float("nan")
        if np.isnan(v):
            v = 0.0
        if "↓" in lab or lab in ("RIS_rec↓", "plausibility↓", "fair_gap↓", "cost↓"):
            v = 1.0 / (1.0 + abs(v)) if lab != "validity" else min(1.0, max(0.0, v))
        elif lab == "validity":
            v = min(1.0, max(0.0, v))
        elif lab == "PGI_rec":
            v = min(1.0, v / (v + 1.0) if v >= 0 else 0.0)
        else:
            v = min(1.0, v / (v + 1.0))
        values.append(float(v))
        labels.append(lab)

    fig = go.Figure(
        data=go.Scatterpolar(r=values + [values[0]], theta=labels + [labels[0]], fill="toself")
    )
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
        showlegend=False,
        title=title,
        height=480,
    )
    return fig


def fig_pareto_3d(records: list[dict[str, Any]], title: str) -> go.Figure:
    costs = [_safe_float(r.get("cost_l1")) for r in records]
    ris = [_safe_float(r.get("RIS_rec")) for r in records]
    pgi = [
        1.0 - min(1.0, max(0.0, _safe_float(r.get("PGI_rec")))) for r in records
    ]
    hv = [_safe_float(r.get("hypervolume")) for r in records]
    valid = [bool(r.get("valid", False)) for r in records]
    fig = go.Figure(
        data=[
            go.Scatter3d(
                x=costs,
                y=ris,
                z=pgi,
                mode="markers",
                marker=dict(
                    size=[8 + 40 * float(min(h, 1.0)) for h in hv],
                    color=[("green" if v else "gray") for v in valid],
                    opacity=0.85,
                ),
                text=[
                    f"valid={v} cost={c:.3f} RIS={rr:.3f} 1-PGI={p:.3f}"
                    for v, c, rr, p in zip(valid, costs, ris, pgi)
                ],
                hoverinfo="text",
            )
        ]
    )
    fig.update_layout(
        title=title,
        scene=dict(xaxis_title="cost", yaxis_title="RIS_rec", zaxis_title="1−PGI (proxy)"),
        height=640,
    )
    return fig


def fig_violin_metric(records: list[dict[str, Any]], metric: str, title: str) -> go.Figure:
    vals = [r.get(metric) for r in records]
    vals_f = [float(v) for v in vals if v is not None and not (isinstance(v, float) and np.isnan(v))]
    valid = [r.get("valid", False) for r in records][: len(vals_f)]
    fig = go.Figure()
    fig.add_trace(
        go.Violin(
            y=vals_f,
            box_visible=True,
            meanline_visible=True,
            fillcolor="lightseagreen",
            opacity=0.65,
            x0=title,
        )
    )
    fig.update_layout(title=f"{metric} distribution", height=400)
    return fig


def fig_reliability_heatmap(summary: pd.DataFrame) -> go.Figure:
    cols = [c for c in summary.columns if any(x in c for x in ("PGI", "PGU", "RIS", "ROS", "RRS"))]
    cols = [c for c in cols if c.endswith("_mean")][:12]
    if not cols:
        fig = go.Figure()
        fig.update_layout(title="No reliability columns in summary")
        return fig
    row = summary.iloc[0][cols].astype(float)
    fig = go.Figure(
        data=go.Heatmap(
            z=[row.values.tolist()],
            x=cols,
            y=["metrics"],
            colorscale="Viridis",
        )
    )
    fig.update_layout(title="Reliability metrics (first summary row)", height=200)
    return fig


def _umap_factual_vector(f: list[Any]) -> list[float] | None:
    if not f:
        return None
    out: list[float] = []
    for v in f:
        if v is None:
            out.append(0.0)
        else:
            try:
                out.append(float(v))
            except (TypeError, ValueError):
                out.append(0.0)
    return out


def _umap_cf_vector(c: list[Any], f: list[Any]) -> list[float] | None:
    """
    Impute missing CF coordinates from factual so failed/partial recourses still appear.
    (Raw JSON uses null per feature when the method could not change that value.)
    """
    if not f:
        return None
    fv = _umap_factual_vector(f)
    if fv is None:
        return None
    if not c or len(c) != len(f):
        return list(fv)
    out: list[float] = []
    for i in range(len(f)):
        cv = c[i]
        if cv is not None:
            try:
                out.append(float(cv))
            except (TypeError, ValueError):
                out.append(fv[i])
        else:
            out.append(fv[i])
    return out


def fig_umap_points(
    records: list[dict[str, Any]],
    feature_dim: int | None = None,
) -> go.Figure:
    try:
        from umap import UMAP
    except ImportError:
        fig = go.Figure()
        fig.update_layout(title="umap-learn not installed")
        return fig

    X: list[list[float]] = []
    labels: list[str] = []
    hover: list[str] = []
    n_imputed = 0
    for r in records:
        f, c = r.get("factual"), r.get("counterfactual")
        if not f:
            continue
        fvec = _umap_factual_vector(f)
        if fvec is None:
            continue
        cvec = _umap_cf_vector(c or [], f)
        if cvec is None:
            cvec = list(fvec)
        if c:
            n_imputed += sum(
                1
                for i in range(min(len(c), len(f)))
                if c[i] is None
            )
        iid = r.get("instance_id", "?")
        X.append(fvec)
        labels.append("factual")
        hover.append(f"factual id={iid}")
        X.append(cvec)
        labels.append("cf")
        hover.append(f"CF id={iid}")

    if len(X) < 5:
        fig = go.Figure()
        fig.update_layout(
            title="Not enough points for UMAP (need at least 3 instances with factual vectors)"
        )
        return fig
    arr = np.asarray(X, dtype=np.float64)
    d = arr.shape[1]
    if feature_dim is not None and feature_dim > 0 and d != feature_dim:
        fig = go.Figure()
        fig.update_layout(
            title=f"Feature dimension mismatch (data has {d}, expected {feature_dim})"
        )
        return fig
    n = arr.shape[0]
    n_neighbors = max(2, min(30, max(5, n // 25)))
    emb = UMAP(
        n_components=2,
        n_neighbors=n_neighbors,
        min_dist=0.15,
        random_state=42,
    ).fit_transform(arr)
    # When CF equals factual (failed recourse or null-CF imputation), UMAP places both
    # rows at the same 2D location — markers stack and the plot looks "sparse". Nudge
    # only the CF embedding slightly so factual vs CF both remain visible.
    emb = np.asarray(emb, dtype=np.float64).copy()
    rng = np.random.default_rng(42)
    spread = float(max(np.std(emb[:, 0]), np.std(emb[:, 1]), 1e-9))
    jitter_scale = 0.035 * spread
    n_stacked_cf = 0
    for i in range(0, n, 2):
        if i + 1 >= n:
            break
        if np.allclose(arr[i], arr[i + 1], rtol=1e-5, atol=1e-8):
            emb[i + 1, 0] += float(rng.normal(0.0, jitter_scale))
            emb[i + 1, 1] += float(rng.normal(0.0, jitter_scale))
            n_stacked_cf += 1
    msize = 10 if n < 80 else 8
    fig = go.Figure(
        data=go.Scatter(
            x=emb[:, 0],
            y=emb[:, 1],
            mode="markers",
            marker=dict(
                size=msize,
                color=[0 if l == "factual" else 1 for l in labels],
                colorscale="Earth",
                opacity=0.65,
                line=dict(width=0.5, color="rgba(255,255,255,0.7)"),
            ),
            text=hover,
            hovertemplate="%{text}<extra></extra>",
        )
    )
    n_inst = len(X) // 2
    impute_note = f"{n_imputed} CF coordinates were null in JSON" if n_imputed else "no null CF coordinates"
    stack_note = (
        f"; {n_stacked_cf} CF points jittered (were identical to factual in feature space)"
        if n_stacked_cf
        else ""
    )
    title = f"UMAP: {n} points ({n_inst} instances, factual + CF; {impute_note}{stack_note})"
    fig.update_layout(title=title, height=560)
    return fig


def fig_parallel_instance(record: dict[str, Any]) -> go.Figure:
    names = record.get("feature_names") or [f"x{i}" for i in range(len(record.get("factual", [])))]
    f = record.get("factual", [])
    c = record.get("counterfactual", [])
    if len(f) != len(c):
        fig = go.Figure()
        return fig
    def _y(v: Any) -> float:
        if v is None:
            return float("nan")
        try:
            return float(v)
        except (TypeError, ValueError):
            return float("nan")

    fy = [_y(v) for v in f]
    cy = [_y(v) for v in c]
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=names, y=fy, mode="lines+markers", name="factual", line=dict(color="steelblue")
        )
    )
    fig.add_trace(
        go.Scatter(x=names, y=cy, mode="lines+markers", name="CF", line=dict(color="coral"))
    )
    fig.update_layout(
        title="Parallel-style line comparison",
        height=min(900, max(400, len(names) * 14)),
    )
    return fig


def fig_fairness_bars(summary: pd.DataFrame) -> go.Figure:
    keys = ["fairness_gap_validity", "fairness_gap_cost", "fairness_delta"]
    present = [k for k in keys if k in summary.columns]
    if not present:
        return go.Figure(layout=dict(title="No fairness aggregates"))
    v = summary.iloc[0][present].astype(float)
    fig = go.Figure(data=[go.Bar(x=present, y=v.values)])
    fig.update_layout(title="Fairness gaps (summary)", height=360)
    return fig


def save_fig_html(fig: go.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(path), include_plotlyjs="cdn")


def fig_sankey_feature_changes(records: list[dict[str, Any]], top_k: int = 12) -> go.Figure:
    """Aggregate |Δ| per feature across instances → simple Sankey-style flow (approximate)."""
    from collections import defaultdict

    agg: dict[str, float] = defaultdict(float)
    for r in records:
        names = r.get("feature_names")
        f, c = r.get("factual"), r.get("counterfactual")
        if not names or not f or not c:
            continue
        for i, n in enumerate(names):
            fv, cv = f[i], c[i]
            if fv is None or cv is None:
                continue
            try:
                agg[n] += abs(float(cv) - float(fv))
            except (TypeError, ValueError):
                continue
    items = sorted(agg.items(), key=lambda x: -x[1])[:top_k]
    if len(items) < 2:
        return go.Figure(layout=dict(title="Sankey: insufficient feature changes"))

    sources = list(range(len(items) - 1))
    targets = [i + 1 for i in sources]
    values = [items[i][1] for i in sources]
    labels = [x[0] for x in items]
    fig = go.Figure(
        data=[
            go.Sankey(
                node=dict(label=labels, pad=15, thickness=20),
                link=dict(source=sources, target=targets, value=values),
            )
        ]
    )
    fig.update_layout(title="Sankey of top feature-change mass (approx)", height=480)
    return fig
