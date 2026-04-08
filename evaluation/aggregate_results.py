"""
Aggregate per-instance JSON records into summary Parquet + composite leaderboard scores.
"""

from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from evaluation.per_instance import fairness_gaps_from_records


def _slug(s: str) -> str:
    s = re.sub(r"[^\w\-]+", "_", s.strip().lower())
    return s.strip("_") or "run"


def records_to_summary_frame(records: list[dict[str, Any]]) -> pd.DataFrame:
    """One row with mean/std/median/min/max for numeric instance fields."""
    if not records:
        return pd.DataFrame()

    skip_cols = {"factual", "counterfactual", "index", "protected_group"}
    numeric_keys: list[str] = []
    for k, v in records[0].items():
        if k in skip_cols:
            continue
        if k == "instance_id":
            continue
        if isinstance(v, bool):
            numeric_keys.append(k)
        elif isinstance(v, (int, float)) and not isinstance(v, bool):
            numeric_keys.append(k)
        elif isinstance(v, np.floating):
            numeric_keys.append(k)
        elif isinstance(v, np.integer):
            numeric_keys.append(k)

    row: dict[str, Any] = {"n_instances": len(records)}
    fg = fairness_gaps_from_records(records)
    row["fairness_gap_validity"] = fg["fairness_gap_validity"]
    row["fairness_gap_cost"] = fg["fairness_gap_cost"]
    row["fairness_delta"] = (
        float(
            (fg["fairness_gap_cost"] if not math.isnan(fg["fairness_gap_cost"]) else 0.0)
            + (
                fg["fairness_gap_validity"]
                if not math.isnan(fg["fairness_gap_validity"])
                else 0.0
            )
        )
        if fg
        else float("nan")
    )

    for key in numeric_keys:
        vals = []
        for r in records:
            if key not in r:
                continue
            v = r[key]
            if isinstance(v, bool):
                vals.append(1.0 if v else 0.0)
            elif isinstance(v, (int, float)) and not isinstance(v, bool):
                vals.append(float(v))
            elif isinstance(v, np.floating):
                vals.append(float(v))
            elif isinstance(v, np.integer):
                vals.append(float(v))
        arr = np.asarray(vals, dtype=np.float64)
        arr = arr[~np.isnan(arr)]
        if arr.size == 0:
            row[f"{key}_mean"] = float("nan")
            row[f"{key}_std"] = float("nan")
            row[f"{key}_median"] = float("nan")
            row[f"{key}_min"] = float("nan")
            row[f"{key}_max"] = float("nan")
            continue
        row[f"{key}_mean"] = float(np.mean(arr))
        row[f"{key}_std"] = float(np.std(arr, ddof=0))
        row[f"{key}_median"] = float(np.median(arr))
        row[f"{key}_min"] = float(np.min(arr))
        row[f"{key}_max"] = float(np.max(arr))

    return pd.DataFrame([row])


def reliable_recourse_score(summary: pd.Series) -> float:
    """
    Composite leaderboard score (higher = better), approximately in [0, 1].
    validity_mean * norm(PGI) * (1-RIS) * (1-fairness) * inv_hypervolume
    """
    def g(name: str, default: float = float("nan")) -> float:
        if name not in summary.index:
            return default
        v = summary[name]
        try:
            return float(v)
        except Exception:
            return default

    validity = g("valid_mean", 0.0)
    if math.isnan(validity):
        validity = 0.0
    pgi = g("PGI_rec_mean", float("nan"))
    ris = g("RIS_rec_mean", 0.0)
    if math.isnan(ris):
        ris = 0.0
    fg = g("fairness_delta", 0.0)
    if math.isnan(fg):
        fg = 0.0
    hv = g("hypervolume_mean", float("nan"))

    pgi_n = min(1.0, pgi / (pgi + 1.0)) if not math.isnan(pgi) and pgi >= 0 else 0.5
    ris_n = max(0.0, min(1.0, 1.0 - ris / (ris + 1.0)))
    fair_n = max(0.0, 1.0 - min(1.0, fg))
    hv_n = 1.0 / (1.0 + hv) if not math.isnan(hv) and hv >= 0 else 1.0

    score = validity * pgi_n * ris_n * fair_n * hv_n
    return float(max(0.0, min(1.0, score)))


def attach_composite(summary: pd.DataFrame) -> pd.DataFrame:
    if summary.empty:
        return summary
    s = summary.iloc[0]
    out = summary.copy()
    out["Reliable_Recourse_Score"] = reliable_recourse_score(s)
    return out


def save_instances_json(records: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    def convert(o: Any) -> Any:
        if isinstance(o, float) and (math.isnan(o) or math.isinf(o)):
            return None
        if isinstance(o, np.floating):
            x = float(o)
            return None if math.isnan(x) or math.isinf(x) else x
        if isinstance(o, np.integer):
            return int(o)
        if isinstance(o, dict):
            return {k: convert(v) for k, v in o.items()}
        if isinstance(o, list):
            return [convert(v) for v in o]
        return o

    with path.open("w", encoding="utf-8") as f:
        json.dump(convert(records), f, indent=2)


def save_summary_parquet(summary: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    summary.to_parquet(path, index=False)


def result_basename(dataset: str, model: str, method: str) -> str:
    return f"{_slug(dataset)}_{_slug(model)}_{_slug(method)}"


def write_eval_outputs(
    records: list[dict[str, Any]],
    results_dir: Path,
    dataset: str,
    model: str,
    method: str,
) -> tuple[Path, Path]:
    base = result_basename(dataset, model, method)
    raw_path = results_dir / "raw" / f"{base}_instances.json"
    summ_path = results_dir / "summary" / f"{base}.parquet"
    save_instances_json(records, raw_path)
    summary = attach_composite(records_to_summary_frame(records))
    summary["bundle_id"] = base
    save_summary_parquet(summary, summ_path)
    return raw_path, summ_path
