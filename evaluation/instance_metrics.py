"""
Per-instance metrics: robustness, temporal decay, plausibility, diversity, hypervolume,
action cost, runtime, and hyperparameter sensitivity helpers.
"""

from __future__ import annotations

import time
import tracemalloc
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde
from sklearn.neighbors import KernelDensity

def compute_robustness_score(
    factual: pd.DataFrame,
    counterfactual: pd.DataFrame,
    feature_columns: list[str],
    model: Any,
    train_std: pd.Series,
    n_trials: int = 20,
    noise_scale: float = 0.01,
    seed: int | None = None,
) -> float:
    """
    Fraction of trials where the CF prediction still differs from noisy-factual prediction
    (recourse still 'flips' relative to perturbed baseline).
    """
    if factual.shape[0] != 1 or counterfactual.shape[0] != 1:
        raise ValueError("robustness expects single-row frames")

    std = train_std.reindex(feature_columns).fillna(1.0).to_numpy(dtype=np.float64)
    x0 = factual[feature_columns].to_numpy(dtype=np.float64).ravel()
    cf = counterfactual.copy()

    rng = np.random.default_rng(seed)
    successes = 0
    for _ in range(n_trials):
        noise = rng.standard_normal(len(feature_columns)) * (noise_scale * std + 1e-8)
        x_noisy = x0 + noise
        noisy_df = factual.copy()
        noisy_df[feature_columns] = x_noisy.reshape(1, -1)

        try:
            p_f = model.get_prediction(noisy_df, proba=False).argmax(dim=-1).item()
            p_c = model.get_prediction(cf, proba=False).argmax(dim=-1).item()
        except Exception:
            continue
        if p_f != p_c:
            successes += 1
    return successes / max(n_trials, 1)


def compute_temporal_decay_rate(
    factual: pd.DataFrame,
    counterfactual: pd.DataFrame,
    feature_columns: list[str],
    feature_type: dict[str, str],
    model: Any,
    n_steps: int = 10,
    trend_scale: float = 0.05,
) -> float:
    """
    After drifting continuous factuals linearly, fraction of steps where recourse invalid.
    """
    if factual.shape[0] != 1 or counterfactual.shape[0] != 1:
        raise ValueError("temporal_decay expects single-row frames")

    cf = counterfactual.copy()
    invalid_steps = 0
    for t in range(1, n_steps + 1):
        drifted = factual.copy()
        for col in feature_columns:
            if str(feature_type.get(col, "")).lower() != "numerical":
                continue
            base = float(factual[col].iloc[0])
            drifted.loc[:, col] = base + trend_scale * t * abs(base) + trend_scale * t
        try:
            p_f = model.get_prediction(drifted, proba=False).argmax(dim=-1).item()
            p_c = model.get_prediction(cf, proba=False).argmax(dim=-1).item()
        except Exception:
            invalid_steps += 1
            continue
        if p_f == p_c:
            invalid_steps += 1
    return invalid_steps / max(n_steps, 1)


def compute_diversity_spread(
    counterfactuals_k: pd.DataFrame,
    feature_columns: list[str],
    train_std: pd.Series,
) -> float:
    """Mean pairwise L2 in z-space (per-feature / std). k>=2 rows required."""
    if counterfactuals_k.shape[0] < 2:
        return float("nan")
    std = train_std.reindex(feature_columns).replace(0, 1.0).fillna(1.0)
    z = counterfactuals_k[feature_columns].to_numpy(dtype=np.float64) / std.to_numpy()
    dists: list[float] = []
    n = z.shape[0]
    for i in range(n):
        for j in range(i + 1, n):
            dists.append(float(np.linalg.norm(z[i] - z[j])))
    return float(np.mean(dists)) if dists else float("nan")


def compute_hypervolume_scalar(
    objectives: np.ndarray,
    ref_point: np.ndarray,
) -> float:
    """
    Minimization hypervolume. Falls back to axis-aligned volume for a single point
    if pymoo is unavailable.
    """
    if objectives.size == 0:
        return float("nan")
    try:
        from pymoo.indicators.hv import Hypervolume

        return float(Hypervolume(ref_point=ref_point).do(objectives))
    except Exception:
        pt = objectives[0]
        vol = 1.0
        for i in range(len(pt)):
            vol *= max(0.0, float(ref_point[i]) - float(pt[i]))
        return float(vol)


def build_pareto_hypervolume(
    valid: bool,
    cost_l1: float,
    plausibility_nll: float,
    max_cost: float,
    max_plaus: float,
) -> float:
    """Single-point degenerate Pareto: hypervolume of one point vs reference."""
    if not valid or np.isnan(cost_l1) or np.isnan(plausibility_nll):
        return float("nan")
    # minimize all three: f1 = 1-validity, f2 = cost, f3 = plausibility
    pt = np.array([[1.0 - float(valid), cost_l1, plausibility_nll]])
    ref = np.array([1.0, max(max_cost, 1e-6), max(max_plaus, 1e-6)])
    return compute_hypervolume_scalar(pt, ref)


def compute_plausibility_nll(
    counterfactual: pd.DataFrame,
    feature_columns: list[str],
    train_features: pd.DataFrame,
) -> float:
    """Negative log density under Gaussian KDE (fallback KernelDensity). Lower = more plausible."""
    if counterfactual.shape[0] != 1:
        raise ValueError("plausibility expects one CF row")

    X = train_features[feature_columns].dropna().to_numpy(dtype=np.float64)
    x = counterfactual[feature_columns].to_numpy(dtype=np.float64).ravel()
    if X.shape[0] < 5 or X.shape[1] == 0:
        return float("nan")

    try:
        kde = gaussian_kde(X.T)
        v = float(kde.logpdf(x.reshape(-1, 1)))
        return float(-v)
    except Exception:
        try:
            kd = KernelDensity(kernel="gaussian", bandwidth=0.5).fit(X)
            v = kd.score_samples(x.reshape(1, -1))[0]
            return float(-v)
        except Exception:
            return float("nan")


def compute_feature_action_cost(
    factual: pd.DataFrame,
    counterfactual: pd.DataFrame,
    feature_columns: list[str],
    weights: dict[str, float] | None,
) -> dict[str, float]:
    w = {c: float(weights.get(c, 1.0)) for c in feature_columns} if weights else {c: 1.0 for c in feature_columns}
    delta = (
        counterfactual[feature_columns].to_numpy(dtype=np.float64)
        - factual[feature_columns].to_numpy(dtype=np.float64)
    ).ravel()
    w_vec = np.array([w[c] for c in feature_columns], dtype=np.float64)
    l1 = float(np.sum(w_vec * np.abs(delta)))
    l0 = float(np.sum((np.abs(delta) > 1e-8).astype(np.float64) * w_vec))
    return {"feature_action_cost_l0": l0, "feature_action_cost_l1": l1}


def compute_classic_cost_sparsity(
    factual: pd.DataFrame,
    counterfactual: pd.DataFrame,
    feature_columns: list[str],
) -> dict[str, float]:
    delta = counterfactual[feature_columns].to_numpy(dtype=np.float64) - factual[
        feature_columns
    ].to_numpy(dtype=np.float64)
    cost_l1 = float(np.abs(delta).sum())
    sparsity = float(np.sum(np.abs(delta) > 1e-8))
    return {"cost_l1": cost_l1, "sparsity_l0": sparsity}


def instance_valid(
    factual: pd.DataFrame,
    counterfactual: pd.DataFrame,
    feature_columns: list[str],
    model: Any,
    desired_class: int | str | None,
) -> bool:
    if counterfactual[feature_columns].isna().any().any():
        return False
    try:
        p0 = model.get_prediction(factual, proba=False).argmax(dim=-1).item()
        p1 = model.get_prediction(counterfactual, proba=False).argmax(dim=-1).item()
    except Exception:
        return False
    if p0 == p1:
        return False
    if desired_class is not None:
        class_map = model.get_class_to_index()
        want = class_map.get(desired_class, None)
        if want is None:
            return p1 != p0
        return p1 == want
    return True


def fairness_submetrics(
    valid: bool,
    cost_l1: float,
    group_key: str | None,
) -> dict[str, Any]:
    return {"group": group_key, "valid": valid, "cost_l1": cost_l1}


def measure_method_call_seconds(method: Any, factuals: pd.DataFrame) -> tuple[float, float]:
    """Returns (elapsed_seconds, peak_traced_memory_bytes) for get_counterfactuals."""
    tracemalloc.start()
    t0 = time.perf_counter()
    _ = method.get_counterfactuals(factuals)
    elapsed = time.perf_counter() - t0
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return elapsed, float(peak)


def memory_peak_mb_sample() -> float:
    try:
        import psutil

        proc = psutil.Process()
        rss = proc.memory_info().rss
        return rss / (1024 * 1024)
    except Exception:
        return float("nan")
