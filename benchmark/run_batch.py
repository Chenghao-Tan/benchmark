from __future__ import annotations

import argparse
import traceback
from pathlib import Path
from time import perf_counter

import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]

import sys

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiment import Experiment


def run_single_config(config_path: Path) -> dict[str, object]:
    with config_path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)
    if not isinstance(config, dict):
        raise ValueError(f"Config must parse to dict: {config_path}")

    dataset_name = str(config.get("dataset", {}).get("name", "unknown"))
    model_name = str(config.get("model", {}).get("name", "unknown"))
    method_name = str(config.get("method", {}).get("name", "unknown"))
    run_name = str(config.get("name", config_path.stem))

    start = perf_counter()
    base = {
        "config_file": str(config_path.relative_to(PROJECT_ROOT).as_posix()),
        "run_name": run_name,
        "dataset": dataset_name,
        "model": model_name,
        "method": method_name,
    }

    try:
        experiment = Experiment(config)
        metrics = experiment.run()
        elapsed = perf_counter() - start
        row = base | {
            "status": "success",
            "elapsed_seconds": elapsed,
        }
        row.update(metrics.iloc[0].to_dict())
        return row
    except Exception as error:
        elapsed = perf_counter() - start
        return base | {
            "status": "failed",
            "elapsed_seconds": elapsed,
            "error": str(error),
            "traceback": traceback.format_exc(limit=1),
        }


def run_batch(config_dir: Path, output_csv: Path, stop_on_error: bool = False) -> pd.DataFrame:
    config_files = sorted(config_dir.glob("*.yaml"))
    if not config_files:
        raise FileNotFoundError(f"No YAML files found in: {config_dir}")

    rows: list[dict[str, object]] = []
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    total = len(config_files)
    for index, config_path in enumerate(config_files, start=1):
        print(f"[{index}/{total}] Running {config_path.name}")
        row = run_single_config(config_path)
        rows.append(row)
        pd.DataFrame(rows).to_csv(output_csv, index=False)

        if row["status"] == "failed":
            print(f"  FAILED: {row.get('error', 'unknown error')}")
            if stop_on_error:
                break
        else:
            print("  OK")

    results = pd.DataFrame(rows)
    results.to_csv(output_csv, index=False)
    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config-dir",
        default=str(Path(__file__).resolve().parent / "configs"),
        help="Directory containing experiment YAML files",
    )
    parser.add_argument(
        "--output-csv",
        default=str(
            PROJECT_ROOT
            / "results"
            / "benchmark_batch_2x2x5"
            / "benchmark_results.csv"
        ),
        help="Path of consolidated CSV results file",
    )
    parser.add_argument(
        "--stop-on-error",
        action="store_true",
        help="Stop batch run on first failed experiment",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_dir = Path(args.config_dir).resolve()
    output_csv = Path(args.output_csv).resolve()

    results = run_batch(
        config_dir=config_dir,
        output_csv=output_csv,
        stop_on_error=bool(args.stop_on_error),
    )
    succeeded = int((results["status"] == "success").sum())
    failed = int((results["status"] == "failed").sum())
    print(
        f"Completed batch run. Success: {succeeded}, Failed: {failed}, "
        f"CSV: {output_csv.as_posix()}"
    )


if __name__ == "__main__":
    main()
