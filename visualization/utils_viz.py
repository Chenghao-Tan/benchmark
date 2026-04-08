"""Load JSON instance records and Parquet summaries for the dashboard."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd


def parse_args_results_dir() -> Path:
    import argparse

    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--results_dir", type=str, default="./results/")
    args, _ = p.parse_known_args()
    return Path(args.results_dir).resolve()


def list_instance_jsons(results_dir: Path) -> list[Path]:
    raw = results_dir / "raw"
    if not raw.is_dir():
        return []
    return sorted(raw.glob("*_instances.json"))


def list_summary_parquets(results_dir: Path) -> list[Path]:
    summ = results_dir / "summary"
    if not summ.is_dir():
        return []
    return sorted(summ.glob("*.parquet"))


def load_instances(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_all_summaries(paths: list[Path]) -> pd.DataFrame:
    if not paths:
        return pd.DataFrame()
    frames: list[pd.DataFrame] = []
    for p in paths:
        chunk = pd.read_parquet(p)
        chunk = chunk.copy()
        chunk["source_file"] = p.name
        frames.append(chunk)
    return pd.concat(frames, axis=0, ignore_index=True)


def method_label_from_filename(path: Path) -> str:
    stem = path.stem.replace("_instances", "")
    return stem


def build_leaderboard(summary: pd.DataFrame) -> pd.DataFrame:
    if summary.empty or "Reliable_Recourse_Score" not in summary.columns:
        return summary
    return summary.sort_values(
        "Reliable_Recourse_Score", ascending=False, na_position="last"
    ).reset_index(drop=True)
