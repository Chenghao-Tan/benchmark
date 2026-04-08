"""
Full per-instance evaluation: classic + extended + OpenXAI-style reliability metrics.
"""

from __future__ import annotations

import time
import tracemalloc
from copy import deepcopy
from typing import Any

import numpy as np
import pandas as pd

from dataset.dataset_object import DatasetObject
from evaluation.instance_metrics import (
    build_pareto_hypervolume,
    compute_classic_cost_sparsity,
    compute_diversity_spread,
    compute_feature_action_cost,
    compute_plausibility_nll,
    compute_robustness_score,
    compute_temporal_decay_rate,
    instance_valid,
    memory_peak_mb_sample,
)
from evaluation.reliability_metrics import compute_all_reliability
from preprocess.preprocess_utils import resolve_feature_metadata
from method.method_object import MethodObject
from model.model_object import ModelObject


def default_evaluation_full_config() -> dict[str, Any]:
    return {
        "robustness": {"n_trials": 20, "noise_scale": 0.01},
        "temporal_decay": {"n_steps": 10, "trend_scale": 0.05},
        "reliability": {
            "pgi_m": 30,
            "pgi_eps": 0.05,
            "ris_p": 20,
            "ris_noise_scale": 0.01,
            "ros_m": 20,
            "logit_noise": 0.01,
            "skip_ris": False,
        },
        "fairness": {"protected_feature": None},
        "action_cost_weights": {},
        "hyperparam_sensitivity": {"enabled": False},
    }


def merge_eval_full_config(base_cfg: dict, yaml_cfg: dict | None) -> dict[str, Any]:
    merged = deepcopy(base_cfg)
    if not yaml_cfg:
        return merged
    for key, value in yaml_cfg.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = {**merged[key], **value}
        else:
            merged[key] = value
    return merged


def _feature_columns(dataset: DatasetObject) -> list[str]:
    cols = list(dataset.get(target=False).columns)
    return cols


def evaluate_instances(
    factuals: DatasetObject,
    counterfactuals: DatasetObject,
    trainset: DatasetObject,
    model: ModelObject,
    method: MethodObject,
    eval_full_cfg: dict[str, Any],
    instance_indices: list[int] | None = None,
    base_seed: int | None = None,
) -> list[dict[str, Any]]:
    """
    Returns list of per-instance metric dicts (JSON-serializable where possible).
    """
    fdf = factuals.get(target=False)
    cdf = counterfactuals.get(target=False)
    train_X = trainset.get(target=False)

    feature_type, _, _ = resolve_feature_metadata(factuals)
    feature_cols = _feature_columns(factuals)
    train_std = train_X[feature_cols].std(ddof=0).replace(0, 1.0)

    desired_class = getattr(method, "_desired_class", None)
    protected = eval_full_cfg.get("fairness", {}).get("protected_feature")
    weights = eval_full_cfg.get("action_cost_weights") or {}

    idx_list = instance_indices if instance_indices is not None else list(range(len(fdf)))

    records: list[dict[str, Any]] = []

    for row_pos, i in enumerate(idx_list):
        if i < 0 or i >= len(fdf):
            continue
        idx = fdf.index[i]
        factual_row = fdf.loc[[idx]]
        cf_row = cdf.loc[[idx]]

        inst_seed = None if base_seed is None else int(base_seed) + row_pos

        rec: dict[str, Any] = {"instance_id": int(i), "index": str(idx)}

        # Timing single-row recourse generation
        tracemalloc.start()
        t0 = time.perf_counter()
        try:
            _ = method.get_counterfactuals(factual_row)
        except Exception:
            pass
        runtime_s = time.perf_counter() - t0
        _, peak_b = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        mem_mb = memory_peak_mb_sample()
        rec["runtime_seconds"] = float(runtime_s)
        rec["memory_peak_mb"] = float(mem_mb)
        rec["memory_traced_peak_mb"] = float(peak_b) / (1024 * 1024)

        valid = instance_valid(
            factual_row,
            cf_row,
            feature_cols,
            model,
            desired_class,
        )
        classic = compute_classic_cost_sparsity(factual_row, cf_row, feature_cols)
        action = compute_feature_action_cost(
            factual_row, cf_row, feature_cols, weights
        )
        plaus = compute_plausibility_nll(cf_row, feature_cols, train_X)
        rec["valid"] = bool(valid)
        rec["cost_l1"] = classic["cost_l1"]
        rec["sparsity_l0"] = classic["sparsity_l0"]
        rec["feature_action_cost_l0"] = action["feature_action_cost_l0"]
        rec["feature_action_cost_l1"] = action["feature_action_cost_l1"]
        rec["plausibility_density"] = plaus

        rb = eval_full_cfg["robustness"]
        rec["robustness_score"] = compute_robustness_score(
            factual_row,
            cf_row,
            feature_cols,
            model,
            train_std,
            n_trials=int(rb.get("n_trials", 20)),
            noise_scale=float(rb.get("noise_scale", 0.01)),
            seed=inst_seed,
        )

        td = eval_full_cfg["temporal_decay"]
        rec["temporal_decay_rate"] = compute_temporal_decay_rate(
            factual_row,
            cf_row,
            feature_cols,
            feature_type,
            model,
            n_steps=int(td.get("n_steps", 10)),
            trend_scale=float(td.get("trend_scale", 0.05)),
        )

        rec["diversity_spread"] = float("nan")
        rec["hypervolume"] = float("nan")  # filled after loop using dataset-wide refs

        rel_cfg = deepcopy(eval_full_cfg)
        rel = compute_all_reliability(
            factual_row,
            cf_row,
            feature_cols,
            model,
            method,
            train_X,
            {"reliability": rel_cfg.get("reliability", {})},
            dataset=factuals,
            seed=inst_seed,
        )
        for k, v in rel.items():
            rec[k] = v

        if protected and protected in factual_row.columns:
            rec["protected_group"] = str(factual_row[protected].iloc[0])
        else:
            rec["protected_group"] = None

        rec["feature_names"] = feature_cols
        rec["factual"] = factual_row[feature_cols].values.flatten().tolist()
        rec["counterfactual"] = cf_row[feature_cols].values.flatten().tolist()

        # hyperparam sensitivity: optional NaN unless multi-run supplies values
        hs = eval_full_cfg.get("hyperparam_sensitivity", {})
        if hs.get("enabled") and isinstance(hs.get("validity_samples"), list):
            vs = [float(x) for x in hs["validity_samples"]]
            rec["hyperparam_sensitivity"] = (
                float(np.std(vs)) if len(vs) > 1 else float("nan")
            )
        else:
            rec["hyperparam_sensitivity"] = float("nan")

        records.append(rec)

    costs = [r["cost_l1"] for r in records if not np.isnan(r["cost_l1"])]
    pls = [r["plausibility_density"] for r in records if not np.isnan(r["plausibility_density"])]
    max_cost = float(np.max(costs)) if costs else 1.0
    max_plaus = float(np.max(pls)) if pls else 50.0
    max_cost = max(max_cost, 1e-6)
    max_plaus = max(max_plaus, 1e-6)

    for r in records:
        r["hypervolume"] = build_pareto_hypervolume(
            bool(r["valid"]),
            float(r["cost_l1"]),
            float(r["plausibility_density"]),
            max_cost=max_cost,
            max_plaus=max_plaus,
        )

    return records


def fairness_gaps_from_records(records: list[dict[str, Any]]) -> dict[str, float]:
    """Aggregate |maj-min| for mean cost and validity rate by protected_group."""
    groups: dict[str, list[dict[str, Any]]] = {}
    for r in records:
        g = r.get("protected_group")
        if g is None:
            continue
        groups.setdefault(str(g), []).append(r)

    if len(groups) < 2:
        return {"fairness_gap_cost": float("nan"), "fairness_gap_validity": float("nan")}

    def mean_valid(sub: list[dict[str, Any]]) -> float:
        if not sub:
            return float("nan")
        return float(np.mean([1.0 if s["valid"] else 0.0 for s in sub]))

    def mean_cost(sub: list[dict[str, Any]]) -> float:
        if not sub:
            return float("nan")
        return float(np.mean([s["cost_l1"] for s in sub]))

    vals_v = [mean_valid(v) for v in groups.values()]
    vals_c = [mean_cost(v) for v in groups.values()]
    return {
        "fairness_gap_validity": float(np.nanmax(vals_v) - np.nanmin(vals_v)),
        "fairness_gap_cost": float(np.nanmax(vals_c) - np.nanmin(vals_c)),
    }
