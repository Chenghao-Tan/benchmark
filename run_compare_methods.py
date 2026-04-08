#!/usr/bin/env python3
"""
Run the same experiment YAML against multiple recourse methods; write all results under one folder
for the Streamlit dashboard (one JSON + Parquet per method).

Examples:
  # Preset: toydata + linear model — sensible defaults for each method
  python run_compare_methods.py --config experiment/toy/toydata_linear_compare.yaml --preset toy_linear --output results/compare_toy/

  # Explicit list (YAML method block merged except name; no preset-specific hyperparams)
  python run_compare_methods.py --config experiment/toy/toydata_linear_toy_reproduce.yaml --methods toy,wachter --output results/compare_toy/
"""

from __future__ import annotations

import argparse
import copy
from pathlib import Path
from typing import Any

import yaml

import dataset  # noqa: F401
import evaluation  # noqa: F401
import method  # noqa: F401
import model  # noqa: F401
import preprocess  # noqa: F401

from experiment import Experiment

# Default order for --preset toy_linear (all support LinearModel on tabular toydata).
PRESET_TOY_LINEAR: list[str] = ["toy", "wachter", "dice", "gs", "claproar"]

# Hyperparameters tuned for small toy runs; merged with seed/device/desired_class from YAML.
METHOD_DEFAULTS_TOY_LINEAR: dict[str, dict[str, Any]] = {
    "toy": {
        "max_iterations": 300,
        "step_size": 0.05,
        "lambda_": 0.08,
        "clamp": True,
    },
    "wachter": {
        "lr": 0.05,
        "lambda_": 0.1,
        "n_iter": 400,
        "t_max_min": 0.5,
        "norm": 1,
        "clamp": True,
        "loss_type": "BCE",
    },
    "dice": {
        "num": 1,
        "posthoc_sparsity_param": 0.1,
    },
    "gs": {
        "n_search_samples": 400,
        "p_norm": 2,
        "step": 0.15,
        "max_iter": 800,
    },
    "claproar": {
        "individual_cost_lambda": 0.1,
        "external_cost_lambda": 0.1,
        "learning_rate": 0.02,
        "max_iter": 120,
        "tol": 1e-4,
    },
}


def build_method_cfg(
    base_cfg: dict[str, Any],
    method_name: str,
    *,
    use_preset_defaults: bool,
) -> dict[str, Any]:
    """Keep seed / device / desired_class from YAML; add preset or inherited kwargs."""
    base_m = copy.deepcopy(base_cfg.get("method", {}))
    out: dict[str, Any] = {"name": method_name}
    for k in ("seed", "device", "desired_class"):
        if k in base_m and base_m[k] is not None:
            out[k] = copy.deepcopy(base_m[k])

    if use_preset_defaults and method_name in METHOD_DEFAULTS_TOY_LINEAR:
        out.update(copy.deepcopy(METHOD_DEFAULTS_TOY_LINEAR[method_name]))
    else:
        for k, v in base_m.items():
            if k in ("name", "seed", "device", "desired_class"):
                continue
            out[k] = copy.deepcopy(v)

    out["name"] = method_name
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Multi-method compare -> one results_dir")
    parser.add_argument("--config", required=True, help="Base experiment YAML")
    parser.add_argument(
        "--methods",
        default=None,
        help="Comma-separated registry names (optional if --preset is set)",
    )
    parser.add_argument(
        "--preset",
        choices=["toy_linear"],
        default=None,
        help="Use bundled method list + default hyperparameters for toydata+linear",
    )
    parser.add_argument(
        "--output",
        default="results/compare/",
        help="Shared results directory for dashboard",
    )
    parser.add_argument("--n_instances", type=int, default=10)
    args = parser.parse_args()

    with Path(args.config).open("r", encoding="utf-8") as f:
        base_cfg = yaml.safe_load(f)
    if not isinstance(base_cfg, dict):
        raise SystemExit("Config must be a YAML mapping")

    use_preset = args.preset == "toy_linear"
    if args.methods:
        methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    elif use_preset:
        methods = list(PRESET_TOY_LINEAR)
    else:
        raise SystemExit("Provide --methods and/or --preset toy_linear")

    if not methods:
        raise SystemExit("No methods to run")

    out = Path(args.output)
    base_name = base_cfg.get("name", "compare")

    for m in methods:
        cfg = copy.deepcopy(base_cfg)
        cfg["method"] = build_method_cfg(cfg, m, use_preset_defaults=use_preset)
        cfg["name"] = f"{base_name}__{m}"
        print(f"\n=== Method: {m} (experiment name: {cfg['name']}) ===")
        exp = Experiment(cfg)
        exp.run(
            eval_full=True,
            sample_instances=args.n_instances,
            results_dir=out,
            launch_dashboard=False,
        )

    print(
        f"\nDone. Open dashboard with:\n  streamlit run visualization/dashboard.py -- --results_dir {out.resolve()}"
    )


if __name__ == "__main__":
    main()
