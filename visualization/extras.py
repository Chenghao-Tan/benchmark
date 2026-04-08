"""
Creative extras: Recourse Arena (Elo), storytelling HTML, difficulty badges.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go


def compute_elo_from_scores(names: list[str], scores: list[float]) -> dict[str, float]:
    """Elo-style scalar from deviation of each score from the batch mean (demo proxy)."""
    if not names:
        return {}
    arr = np.asarray(scores, dtype=np.float64)
    m = float(np.nanmean(arr))
    out: dict[str, float] = {}
    for n, s in zip(names, scores):
        out[n] = 1500.0 + 400.0 * float(np.tanh(float(s) - m))
    return out


def fig_elo_heatmap(names: list[str], scores: list[float]) -> go.Figure:
    elo = compute_elo_from_scores(names, scores)
    n = len(names)
    win = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i == j:
                win[i, j] = 0.5
            else:
                win[i, j] = 1.0 if scores[i] > scores[j] else 0.0 if scores[i] < scores[j] else 0.5
    fig = go.Figure(
        data=go.Heatmap(z=win, x=names, y=names, colorscale="Blues", zmin=0, zmax=1)
    )
    fig.update_layout(
        title="Pairwise win-rate heatmap (from Reliable_Recourse_Score)",
        height=400 + 20 * n,
    )
    fig.add_annotation(
        x=0.5,
        y=-0.15,
        xref="paper",
        yref="paper",
        showarrow=False,
        text="Elo (text): " + ", ".join(f"{k}={v:.0f}" for k, v in elo.items()),
    )
    return fig


def difficulty_badges(summary_row: pd.Series) -> list[str]:
    badges: list[str] = []
    ris = float(summary_row.get("RIS_rec_mean", float("nan")))
    pgi = float(summary_row.get("PGI_rec_mean", float("nan")))
    rob = float(summary_row.get("robustness_score_mean", float("nan")))
    fg = float(summary_row.get("fairness_delta", float("nan")))

    if not math.isnan(ris) and ris < 0.3:
        badges.append("Stable")
    if not math.isnan(pgi) and pgi > 0.2:
        badges.append("Faithful")
    if not math.isnan(rob) and rob > 0.6:
        badges.append("Robust")
    if not math.isnan(fg) and fg < 0.15:
        badges.append("Fair")
    return badges or ["Unclassified"]


def write_story_html(
    out_path: Path,
    method_name: str,
    figs: list[tuple[str, go.Figure]],
) -> None:
    """Minimal auto-report with embedded Plotly HTML fragments."""
    parts = [
        "<html><head><meta charset='utf-8'><title>Recourse story</title></head><body>",
        f"<h1>Recourse path report — {method_name}</h1>",
        "<p>Typical denied-instance counterfactual path (demo).</p>",
    ]
    for title, fig in figs:
        parts.append(f"<h2>{title}</h2>")
        parts.append(fig.to_html(full_html=False, include_plotlyjs="cdn"))
    parts.append("</body></html>")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(parts), encoding="utf-8")
