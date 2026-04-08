"""
OpenXAI-inspired reliability metrics adapted for counterfactual recourse.

References: OpenXAI (arXiv:2206.11104) — prediction-gap and stability notions
mapped from attributions to recourse deltas.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import torch

from model.model_object import ModelObject
from utils.seed import seed_context


def _proba_tensor(model: ModelObject, X: pd.DataFrame) -> torch.Tensor:
    return model.get_prediction(X, proba=True).detach().cpu().float()


def _l1_proba_diff(a: torch.Tensor, b: torch.Tensor) -> float:
    return float(torch.abs(a - b).sum(dim=-1).mean().item())


def compute_pgi_rec(
    factual: pd.DataFrame,
    counterfactual: pd.DataFrame,
    feature_columns: list[str],
    model: ModelObject,
    feature_std: pd.Series,
    M: int = 30,
    eps_scale: float = 0.05,
    seed: int | None = None,
) -> float:
    """
    Prediction Gap on Important (changed) features — higher = changes drive the flip.
    """
    if factual.shape[0] != 1 or counterfactual.shape[0] != 1:
        raise ValueError("PGI_rec expects single-row DataFrames")

    x0 = factual[feature_columns].to_numpy(dtype=np.float64).ravel()
    x1 = counterfactual[feature_columns].to_numpy(dtype=np.float64).ravel()
    delta = np.abs(x1 - x0)
    changed = delta > 1e-5
    if not changed.any():
        return float("nan")

    std = feature_std.reindex(feature_columns).fillna(1.0).to_numpy(dtype=np.float64)
    base = _proba_tensor(model, factual)

    with seed_context(seed):
        gaps: list[float] = []
        for _ in range(M):
            eps = np.random.randn(len(feature_columns)) * (eps_scale * std + 1e-8)
            eps *= changed.astype(np.float64)
            perturbed = x1 + eps
            row = pd.DataFrame([perturbed], columns=feature_columns, index=factual.index)
            Xp = factual.copy()
            Xp[feature_columns] = row.values
            other = _proba_tensor(model, Xp)
            gaps.append(_l1_proba_diff(base, other))
    return float(np.mean(gaps))


def compute_pgu_rec(
    factual: pd.DataFrame,
    counterfactual: pd.DataFrame,
    feature_columns: list[str],
    model: ModelObject,
    feature_std: pd.Series,
    M: int = 30,
    eps_scale: float = 0.05,
    seed: int | None = None,
) -> float:
    """
    Prediction Gap on Unimportant (unchanged) features — lower = better.
    """
    if factual.shape[0] != 1 or counterfactual.shape[0] != 1:
        raise ValueError("PGU_rec expects single-row DataFrames")

    x0 = factual[feature_columns].to_numpy(dtype=np.float64).ravel()
    x1 = counterfactual[feature_columns].to_numpy(dtype=np.float64).ravel()
    delta = np.abs(x1 - x0)
    unchanged = delta <= 1e-5
    if not unchanged.any():
        return float("nan")

    std = feature_std.reindex(feature_columns).fillna(1.0).to_numpy(dtype=np.float64)
    base = _proba_tensor(model, counterfactual)

    with seed_context(seed):
        gaps: list[float] = []
        for _ in range(M):
            eps = np.random.randn(len(feature_columns)) * (eps_scale * std + 1e-8)
            eps *= unchanged.astype(np.float64)
            perturbed = x1 + eps
            row = pd.DataFrame([perturbed], columns=feature_columns, index=factual.index)
            Xp = counterfactual.copy()
            Xp[feature_columns] = row.values
            other = _proba_tensor(model, Xp)
            gaps.append(_l1_proba_diff(base, other))
    return float(np.mean(gaps))


def compute_ros_rec(
    counterfactual: pd.DataFrame,
    feature_columns: list[str],
    model: ModelObject,
    M: int = 20,
    logit_noise: float = 0.01,
    seed: int | None = None,
) -> float:
    """
    Output stability at the counterfactual: small logit perturbations (Torch models).
    Lower mean L1(proba shift) = more stable. Non-torch models return NaN.
    """
    if not hasattr(model, "_model") or not isinstance(
        getattr(model, "_model", None), torch.nn.Module
    ):
        return float("nan")
    if counterfactual.shape[0] != 1:
        raise ValueError("ROS_rec expects a single-row counterfactual")

    X_tensor = torch.tensor(
        counterfactual[feature_columns].to_numpy(dtype=np.float32),
        dtype=torch.float32,
        device=model._device,
    )
    model._model.eval()
    base_logits = model.forward(X_tensor).detach()
    if base_logits.ndim == 1:
        base_logits = base_logits.unsqueeze(0)

    def logits_to_proba(logits: torch.Tensor) -> torch.Tensor:
        if logits.shape[-1] == 1:
            return torch.sigmoid(logits)
        return torch.softmax(logits, dim=-1)

    base_p = logits_to_proba(base_logits).cpu()

    with seed_context(seed):
        shifts: list[float] = []
        for _ in range(M):
            noise = torch.randn_like(base_logits) * logit_noise
            noisy_logits = base_logits + noise
            noisy_p = logits_to_proba(noisy_logits).cpu()
            shifts.append(float(torch.abs(base_p - noisy_p).sum(dim=-1).mean().item()))
    return float(np.mean(shifts))


def compute_rrs_rec(
    model: ModelObject,
    counterfactual: pd.DataFrame,
    feature_columns: list[str],
) -> float:
    """
    Relative representation stability — requires embedding hook; not wired for all models.
    """
    del model, counterfactual, feature_columns
    return float("nan")


def compute_ris_rec(
    method: Any,
    model: ModelObject,
    factual: pd.DataFrame,
    counterfactual: pd.DataFrame,
    feature_columns: list[str],
    feature_std: pd.Series,
    P: int = 20,
    noise_scale: float = 0.01,
    seed: int | None = None,
    max_attempts: int = 200,
) -> float:
    """
    Relative input stability: max relative change in recourse delta under noisy factuals
    that keep the same predicted class. Regenerates recourses (expensive).
    """
    if factual.shape[0] != 1 or counterfactual.shape[0] != 1:
        raise ValueError("RIS_rec expects single-row DataFrames")

    x0 = factual[feature_columns].copy()
    cf0 = counterfactual[feature_columns].copy()
    delta = (cf0.to_numpy(dtype=np.float64) - x0.to_numpy(dtype=np.float64)).ravel()
    norm_delta = float(np.linalg.norm(delta) + 1e-8)

    orig_pred = model.get_prediction(x0, proba=False).argmax(dim=-1).item()

    std = feature_std.reindex(feature_columns).fillna(1.0).to_numpy(dtype=np.float64)
    ratios: list[float] = []

    with seed_context(seed):
        attempts = 0
        accepted = 0
        while accepted < P and attempts < max_attempts:
            attempts += 1
            noise = np.random.randn(len(feature_columns)) * (noise_scale * std + 1e-8)
            x_p_vals = x0.to_numpy(dtype=np.float64).ravel() + noise
            x_p = pd.DataFrame([x_p_vals], columns=feature_columns, index=x0.index)
            x_p_full = factual.copy()
            x_p_full[feature_columns] = x_p.values

            try:
                new_pred = model.get_prediction(x_p_full, proba=False).argmax(
                    dim=-1
                ).item()
            except Exception:
                continue
            if new_pred != orig_pred:
                continue

            cf_p = method.get_counterfactuals(x_p_full)
            if cf_p.shape[0] != 1:
                continue
            if cf_p.isna().any().any():
                continue
            cf_p = cf_p.reindex(columns=feature_columns)
            delta_p = (
                cf_p.to_numpy(dtype=np.float64) - x_p.to_numpy(dtype=np.float64)
            ).ravel()
            ratio = float(np.linalg.norm(delta_p - delta) / norm_delta)
            ratios.append(ratio)
            accepted += 1

    if not ratios:
        return float("nan")
    return float(max(ratios))


def compute_ground_truth_faithfulness(
    factual: pd.DataFrame,
    counterfactual: pd.DataFrame,
    feature_columns: list[str],
    dataset: Any,
) -> dict[str, float]:
    """
    Spearman-style alignment between |Δ| and optional ground-truth feature ranks.
    If dataset.has_ground_truth and ground_truth_importance (list aligned with
    feature_columns), returns correlation metrics; else NaNs.
    """
    out = {
        "FA_rec": float("nan"),
        "RA_rec": float("nan"),
        "SA_rec": float("nan"),
        "SRA_rec": float("nan"),
        "RC_rec": float("nan"),
        "PRA_rec": float("nan"),
    }
    if not getattr(dataset, "has_ground_truth", False):
        return out

    gt = getattr(dataset, "ground_truth_importance", None)
    if gt is None or len(gt) != len(feature_columns):
        return out

    x0 = factual[feature_columns].to_numpy(dtype=np.float64).ravel()
    x1 = counterfactual[feature_columns].to_numpy(dtype=np.float64).ravel()
    mag = np.abs(x1 - x0)
    from scipy.stats import spearmanr

    gt_arr = np.asarray(gt, dtype=np.float64)
    if np.allclose(mag, 0) or np.allclose(gt_arr, 0):
        return out

    rho, _ = spearmanr(mag, gt_arr)
    out["RC_rec"] = float(rho) if not np.isnan(rho) else float("nan")
    # Compact placeholders — full OpenXAI suite can extend rank transforms here
    out["FA_rec"] = out["RC_rec"]
    out["RA_rec"] = out["RC_rec"]
    out["SA_rec"] = out["RC_rec"]
    out["SRA_rec"] = out["RC_rec"]
    out["PRA_rec"] = out["RC_rec"]
    return out


def compute_all_reliability(
    factual: pd.DataFrame,
    counterfactual: pd.DataFrame,
    feature_columns: list[str],
    model: ModelObject,
    method: Any,
    train_features: pd.DataFrame,
    config: dict[str, Any],
    dataset: Any,
    seed: int | None = None,
) -> dict[str, float]:
    """Run PGI/PGU/ROS/RIS (+ optional GT block) with shared hyperparameters."""
    rel_cfg = config.get("reliability", {})
    M = int(rel_cfg.get("pgi_m", 30))
    eps = float(rel_cfg.get("pgi_eps", 0.05))
    P = int(rel_cfg.get("ris_p", 20))
    ris_noise = float(rel_cfg.get("ris_noise_scale", 0.01))
    ros_m = int(rel_cfg.get("ros_m", 20))
    logit_n = float(rel_cfg.get("logit_noise", 0.01))

    std = train_features[feature_columns].std(ddof=0).replace(0, 1.0)

    out: dict[str, float] = {}
    out["PGI_rec"] = compute_pgi_rec(
        factual,
        counterfactual,
        feature_columns,
        model,
        std,
        M=M,
        eps_scale=eps,
        seed=seed,
    )
    out["PGU_rec"] = compute_pgu_rec(
        factual,
        counterfactual,
        feature_columns,
        model,
        std,
        M=M,
        eps_scale=eps,
        seed=seed,
    )
    out["ROS_rec"] = compute_ros_rec(
        counterfactual,
        feature_columns,
        model,
        M=ros_m,
        logit_noise=logit_n,
        seed=seed,
    )
    out["RRS_rec"] = compute_rrs_rec(model, counterfactual, feature_columns)

    if rel_cfg.get("skip_ris", False):
        out["RIS_rec"] = float("nan")
    else:
        out["RIS_rec"] = compute_ris_rec(
            method,
            model,
            factual,
            counterfactual,
            feature_columns,
            std,
            P=P,
            noise_scale=ris_noise,
            seed=seed,
        )

    out.update(
        compute_ground_truth_faithfulness(factual, counterfactual, feature_columns, dataset)
    )
    return out
