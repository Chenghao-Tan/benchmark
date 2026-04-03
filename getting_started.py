"""Simple end-to-end benchmark example for new users.

This script runs a compact benchmark pipeline on the Adult dataset using a
linear target model and the Wachter recourse method. It is intended as a
readable first example, not a reproduction-quality experiment.
"""

from __future__ import annotations

import torch

from experiment import Experiment


def resolve_device() -> str:
    """Return the best available device for the quickstart run."""
    return "cuda" if torch.cuda.is_available() else "cpu"


def build_config(device: str) -> dict:
    """Build the onboarding experiment config used by this script.

    Args:
        device: Device name shared by the target model and recourse method.

    Returns:
        dict: Experiment configuration compatible with ``Experiment``.
    """
    return {
        "name": "adult_wachter_getting_started",
        "logger": {
            "level": "INFO",
            "path": "./logs/adult_wachter_getting_started.log",
        },
        "caching": {
            "path": "./cache/",
        },
        "dataset": {
            "name": "adult",
        },
        "preprocess": [
            {
                "name": "scale",
                "seed": 7,
                "scaling": "normalize",
                "range": True,
            },
            {
                "name": "encode",
                "seed": 7,
                "encoding": "onehot",
            },
            {
                "name": "split",
                "seed": 7,
                "split": 0.2,
                "sample": 8,
            },
        ],
        "model": {
            "name": "linear",
            "seed": 7,
            "device": device,
            "epochs": 20,
            "learning_rate": 0.03,
            "batch_size": 64,
            "optimizer": "adam",
            "criterion": "bce",
            "output_activation": "sigmoid",
            "save_name": None,
        },
        "method": {
            "name": "wachter",
            "seed": 7,
            "device": device,
            "desired_class": 1,
            "lr": 0.01,
            "lambda_": 0.01,
            "n_iter": 200,
            "t_max_min": 0.1,
            "norm": 1,
            "clamp": True,
            "loss_type": "BCE",
        },
        "evaluation": [
            {
                "name": "validity",
            },
            {
                "name": "distance",
            },
        ],
    }


def main() -> None:
    """Run the root quickstart experiment and print the final metrics."""
    device = resolve_device()
    config = build_config(device)

    print("Benchmark getting-started example")
    print("Dataset: adult")
    print("Preprocess: scale -> encode -> split -> finalize")
    print("Model: linear")
    print("Method: wachter")
    print("Evaluation: validity + distance")
    print(f"Device: {device}")
    print("Test sample size: 8")
    print()

    experiment = Experiment(config)
    metrics = experiment.run()

    print("Final metrics")
    print(metrics.to_string(index=False))


if __name__ == "__main__":
    main()
