from __future__ import annotations

import argparse
import sys
from copy import deepcopy
from pathlib import Path

import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiment import Experiment
from utils.reproduce import (
    apply_device,
    evaluate_counterfactuals,
    load_config,
    materialize_datasets,
    select_factuals,
)


def _apply_quick(config: dict) -> dict:
    cfg = deepcopy(config)
    cfg["model"]["epochs"] = 40
    cfg["method"]["max_iter"] = 50
    return cfg


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default=str(Path(__file__).with_name("credit_linear_reproduce.yaml")),
    )
    parser.add_argument("--num-factuals", type=int, default=5)
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)
    if args.quick:
        config = _apply_quick(config)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    config = apply_device(config, device)

    experiment = Experiment(config)
    trainset, testset = materialize_datasets(experiment)
    experiment._target_model.fit(trainset)
    experiment._method.fit(trainset)
    factuals = select_factuals(
        experiment._target_model,
        testset,
        desired_class=1,
        num_factuals=args.num_factuals,
        selection="negative_class",
    )
    counterfactuals = experiment._method.predict(factuals)
    metrics = evaluate_counterfactuals(experiment, factuals, counterfactuals)
    comparison = pd.DataFrame(
        [
            {
                "metric": "feature_change_std_mean",
                "measured": float(metrics["feature_change_std_mean"].iloc[0]),
                "reference": 0.03,
            },
            {
                "metric": "individual_cost",
                "measured": float(metrics["individual_cost"].iloc[0]),
                "reference": 0.5,
            },
        ]
    )
    print(metrics.to_string(index=False))
    print()
    print(comparison.to_string(index=False))


if __name__ == "__main__":
    main()
