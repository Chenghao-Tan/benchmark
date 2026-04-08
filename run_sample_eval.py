#!/usr/bin/env python3
"""
Fast sample run: first N test instances, extended metrics, JSON + Parquet, optional Streamlit.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

# Trigger registration (same as experiment)
import dataset  # noqa: F401
import evaluation  # noqa: F401
import method  # noqa: F401
import model  # noqa: F401
import preprocess  # noqa: F401

from experiment import Experiment


def main() -> None:
    parser = argparse.ArgumentParser(description="Sample recourse evaluation + dashboard")
    parser.add_argument(
        "--config",
        default="experiment/toy/toydata_linear_toy_reproduce.yaml",
        help="Experiment YAML path",
    )
    parser.add_argument("--n_instances", type=int, default=10)
    parser.add_argument("--output", default="results/sample/")
    parser.add_argument(
        "--no-dashboard",
        action="store_true",
        help="Do not spawn Streamlit (still prints launch command)",
    )
    args = parser.parse_args()

    cfg_path = Path(args.config)
    with cfg_path.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    if not isinstance(config, dict):
        raise SystemExit("Config must be a YAML mapping")

    # Fast defaults for interactive / demo runs (override via evaluation_full in YAML)
    ef = config.setdefault("evaluation_full", {})
    rel = ef.setdefault("reliability", {})
    rel.setdefault("skip_ris", True)
    rel.setdefault("pgi_m", 12)
    rel.setdefault("ros_m", 8)
    rb = ef.setdefault("robustness", {})
    rb.setdefault("n_trials", 8)
    td = ef.setdefault("temporal_decay", {})
    td.setdefault("n_steps", 5)

    exp = Experiment(config)
    out = Path(args.output)
    exp.run(
        eval_full=True,
        sample_instances=args.n_instances,
        results_dir=out,
        launch_dashboard=not args.no_dashboard,
    )

    dash = Path(__file__).resolve().parent / "visualization" / "dashboard.py"
    cmd = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        str(dash),
        "--",
        "--results_dir",
        str(out.resolve()),
    ]
    print("\nDashboard command:\n  " + " ".join(cmd))


if __name__ == "__main__":
    main()
