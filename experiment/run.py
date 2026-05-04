from __future__ import annotations

import argparse
import csv
import os
import platform
import socket
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_ROOT = PROJECT_ROOT / "experiment"
DEFAULT_OUTPUT_ROOT = EXPERIMENT_ROOT / "_runs"
STRUCTURED_SUFFIXES = {".csv", ".json", ".yaml", ".yml"}


@dataclass(frozen=True)
class MethodSpec:
    method_name: str
    script_path: Path
    support_status: str
    command_builder: Callable[[argparse.Namespace, "MethodSpec", Path], list[str]] | None
    unsupported_reason: str | None = None
    expected_artifacts: tuple[str, ...] = ()


@dataclass
class MethodResult:
    method_name: str
    script_path: str
    support_status: str
    run_status: str
    command: list[str] = field(default_factory=list)
    return_code: int | None = None
    start_utc: str | None = None
    end_utc: str | None = None
    duration_seconds: float | None = None
    stdout_path: str | None = None
    stderr_path: str | None = None
    log_dir: str | None = None
    unsupported_reason: str | None = None
    structured_artifacts: list[str] = field(default_factory=list)
    stdout_tail: list[str] = field(default_factory=list)
    error_message: str | None = None

    def to_summary_row(self) -> dict[str, object]:
        return {
            "method_name": self.method_name,
            "support_status": self.support_status,
            "run_status": self.run_status,
            "return_code": self.return_code,
            "duration_seconds": self.duration_seconds,
            "start_utc": self.start_utc,
            "end_utc": self.end_utc,
            "stdout_path": self.stdout_path,
            "stderr_path": self.stderr_path,
            "log_dir": self.log_dir,
            "script_path": self.script_path,
            "command": " ".join(self.command),
            "unsupported_reason": self.unsupported_reason,
            "structured_artifacts": ";".join(self.structured_artifacts),
            "stdout_tail": " | ".join(self.stdout_tail),
            "error_message": self.error_message,
        }


def _parse_name_filter(raw_value: str | None) -> list[str] | None:
    if raw_value is None:
        return None
    resolved: list[str] = []
    seen: set[str] = set()
    for token in raw_value.split(","):
        name = token.strip()
        if not name or name in seen:
            continue
        seen.add(name)
        resolved.append(name)
    return resolved or None


def _timestamp_slug() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _default_output_dir() -> Path:
    return DEFAULT_OUTPUT_ROOT / _timestamp_slug()


def _default_python_executable() -> str:
    candidates: list[Path] = []
    conda_exe = os.environ.get("CONDA_EXE")
    if conda_exe:
        conda_root = Path(conda_exe).resolve().parent.parent
        candidates.append(conda_root / "envs" / "benchmark" / "python.exe")
        candidates.append(conda_root / "envs" / "benchmark" / "python")
    user_profile = os.environ.get("USERPROFILE")
    if user_profile:
        home = Path(user_profile)
        candidates.append(home / "anaconda3" / "envs" / "benchmark" / "python.exe")
        candidates.append(home / "anaconda3" / "envs" / "benchmark" / "python")
    for candidate in candidates:
        if candidate.exists():
            return str(candidate)
    return sys.executable


def _build_noarg_command(
    args: argparse.Namespace,
    spec: MethodSpec,
    method_output_dir: Path,
) -> list[str]:
    del method_output_dir
    return [args.python, str(spec.script_path)]


def _build_config_command(config_relpath: str) -> Callable[[argparse.Namespace, MethodSpec, Path], list[str]]:
    def _builder(
        args: argparse.Namespace,
        spec: MethodSpec,
        method_output_dir: Path,
    ) -> list[str]:
        del method_output_dir
        return [args.python, str(spec.script_path), "--config", config_relpath]

    return _builder


def _build_path_command(config_relpath: str) -> Callable[[argparse.Namespace, MethodSpec, Path], list[str]]:
    def _builder(
        args: argparse.Namespace,
        spec: MethodSpec,
        method_output_dir: Path,
    ) -> list[str]:
        del method_output_dir
        return [args.python, str(spec.script_path), "-p", config_relpath]

    return _builder


def _build_dual_config_command(
    first_flag: str,
    first_value: str,
    second_flag: str,
    second_value: str,
) -> Callable[[argparse.Namespace, MethodSpec, Path], list[str]]:
    def _builder(
        args: argparse.Namespace,
        spec: MethodSpec,
        method_output_dir: Path,
    ) -> list[str]:
        del method_output_dir
        return [
            args.python,
            str(spec.script_path),
            first_flag,
            first_value,
            second_flag,
            second_value,
        ]

    return _builder


def _build_mace_command(
    args: argparse.Namespace,
    spec: MethodSpec,
    method_output_dir: Path,
) -> list[str]:
    output_json = method_output_dir / "mace_summary.json"
    return [
        args.python,
        str(spec.script_path),
        "--output-json",
        str(output_json),
    ]


def _build_method_registry() -> dict[str, MethodSpec]:
    return {
        # "apas": MethodSpec(
        #     method_name="apas",
        #     script_path=EXPERIMENT_ROOT / "apas" / "reproduce.py",
        #     support_status="supported",
        #     command_builder=_build_config_command("./experiment/apas/config.yaml"),
        # ),
        # "arg_ensembling": MethodSpec(
        #     method_name="arg_ensembling",
        #     script_path=EXPERIMENT_ROOT / "arg_ensembling" / "reproduce.py",
        #     support_status="supported",
        #     command_builder=_build_config_command("./experiment/arg_ensembling/config.yaml"),
        # ),
        "cchvae": MethodSpec(
            method_name="cchvae",
            script_path=EXPERIMENT_ROOT / "cchvae" / "reproduce.py",
            support_status="supported",
            command_builder=_build_path_command(
                "./experiment/cchvae/credit_cchvae_sklearn_logistic_regression_cchvae_reproduce.yaml"
            ),
        ),
        # "cemsp": MethodSpec(
        #     method_name="cemsp",
        #     script_path=EXPERIMENT_ROOT / "cemsp" / "reproduce.py",
        #     support_status="supported",
        #     command_builder=_build_config_command("./experiment/cemsp/config.yaml"),
        # ),
        "cfrl": MethodSpec(
            method_name="cfrl",
            script_path=EXPERIMENT_ROOT / "cfrl" / "reproduce.py",
            support_status="supported",
            command_builder=_build_noarg_command,
        ),
        "cfvae": MethodSpec(
            method_name="cfvae",
            script_path=EXPERIMENT_ROOT / "cfvae" / "reproduce.py",
            support_status="unsupported",
            command_builder=None,
            unsupported_reason="Requires external --weights-dir artifact path.",
        ),
        "clue": MethodSpec(
            method_name="clue",
            script_path=EXPERIMENT_ROOT / "clue" / "reproduce.py",
            support_status="unsupported",
            command_builder=None,
            unsupported_reason=(
                "Requires external artifact paths for the BNN/VAE/VAEAC checkpoints."
            ),
        ),
        # "cogs": MethodSpec(
        #     method_name="cogs",
        #     script_path=EXPERIMENT_ROOT / "cogs" / "reproduce.py",
        #     support_status="supported",
        #     command_builder=_build_config_command("./experiment/cogs/config.yaml"),
        # ),
        # "cols": MethodSpec(
        #     method_name="cols",
        #     script_path=EXPERIMENT_ROOT / "cols" / "reproduce.py",
        #     support_status="supported",
        #     command_builder=_build_config_command("./experiment/cols/reproduce_configs.yaml"),
        # ),
        # "cvas_proj": MethodSpec(
        #     method_name="cvas_proj",
        #     script_path=EXPERIMENT_ROOT / "cvas_proj" / "reproduce.py",
        #     support_status="supported",
        #     command_builder=_build_dual_config_command(
        #         "--current-config",
        #         "./experiment/cvas_proj/german_mlp_cvas_proj_current.yaml",
        #         "--future-config",
        #         "./experiment/cvas_proj/german_mlp_cvas_proj_future.yaml",
        #     ),
        # ),
        "dice": MethodSpec(
            method_name="dice",
            script_path=EXPERIMENT_ROOT / "dice" / "reproduce.py",
            support_status="supported",
            command_builder=_build_config_command(
                "./experiment/dice/compas_mlp_dice_reproduce.yaml"
            ),
        ),
        # "diverse_dist": MethodSpec(
        #     method_name="diverse_dist",
        #     script_path=EXPERIMENT_ROOT / "diverse_dist" / "reproduce.py",
        #     support_status="supported",
        #     command_builder=_build_config_command("./experiment/diverse_dist/config.yaml"),
        # ),
        "face": MethodSpec(
            method_name="face",
            script_path=EXPERIMENT_ROOT / "face" / "reproduce.py",
            support_status="supported",
            command_builder=_build_noarg_command,
        ),
        "gs": MethodSpec(
            method_name="gs",
            script_path=EXPERIMENT_ROOT / "gs" / "reproduce.py",
            support_status="supported",
            command_builder=_build_path_command(
                "./experiment/gs/news_popularity_randomforest_gs_reproduce.yaml"
            ),
        ),
        "larr": MethodSpec(
            method_name="larr",
            script_path=EXPERIMENT_ROOT / "larr" / "reproduce.py",
            support_status="supported",
            command_builder=_build_noarg_command,
        ),
        # "mace": MethodSpec(
        #     method_name="mace",
        #     script_path=EXPERIMENT_ROOT / "mace" / "reproduce.py",
        #     support_status="supported",
        #     command_builder=_build_mace_command,
        #     expected_artifacts=("mace_summary.json",),
        # ),
        "probe": MethodSpec(
            method_name="probe",
            script_path=EXPERIMENT_ROOT / "probe" / "reproduce.py",
            support_status="supported",
            command_builder=_build_path_command(
                "./experiment/probe/compas_mlp_probe_reproduce.yaml"
            ),
        ),
        # "proplace": MethodSpec(
        #     method_name="proplace",
        #     script_path=EXPERIMENT_ROOT / "proplace" / "reproduce.py",
        #     support_status="supported",
        #     command_builder=_build_config_command("./experiment/proplace/config.yaml"),
        # ),
        "rbr": MethodSpec(
            method_name="rbr",
            script_path=EXPERIMENT_ROOT / "rbr" / "reproduce.py",
            support_status="supported",
            command_builder=_build_dual_config_command(
                "--current-config",
                "./experiment/rbr/german_mlp_rbr_reproduce_current.yaml",
                "--future-config",
                "./experiment/rbr/german_mlp_rbr_reproduce_future.yaml",
            ),
        ),
        "roar": MethodSpec(
            method_name="roar",
            script_path=EXPERIMENT_ROOT / "roar" / "reproduce.py",
            support_status="supported",
            command_builder=_build_dual_config_command(
                "--current-config",
                "./experiment/roar/german_mlp_roar_reproduce_current.yaml",
                "--future-config",
                "./experiment/roar/german_mlp_roar_reproduce_future.yaml",
            ),
        ),
        # "sns": MethodSpec(
        #     method_name="sns",
        #     script_path=EXPERIMENT_ROOT / "sns" / "reproduce.py",
        #     support_status="supported",
        #     command_builder=_build_config_command("./experiment/sns/config.yaml"),
        # ),
        # "trex": MethodSpec(
        #     method_name="trex",
        #     script_path=EXPERIMENT_ROOT / "trex" / "reproduce.py",
        #     support_status="supported",
        #     command_builder=_build_config_command("./experiment/trex/config.yaml"),
        # ),
    }


def _discover_reproduce_scripts() -> list[Path]:
    return sorted(
        path
        for path in EXPERIMENT_ROOT.iterdir()
        if path.is_dir() and (path / "reproduce.py").exists()
    )


def _resolve_specs() -> list[MethodSpec]:
    registry = _build_method_registry()
    specs: list[MethodSpec] = []
    for method_dir in _discover_reproduce_scripts():
        name = method_dir.name
        if name in registry:
            specs.append(registry[name])
            continue
        specs.append(
            MethodSpec(
                method_name=name,
                script_path=method_dir / "reproduce.py",
                support_status="unsupported",
                command_builder=None,
                unsupported_reason="Method discovered but not registered in experiment/run.py.",
            )
        )
    return specs


def _filter_specs(
    specs: list[MethodSpec],
    include_names: list[str] | None,
    exclude_names: list[str] | None,
) -> list[MethodSpec]:
    available = [spec.method_name for spec in specs]
    if include_names is not None:
        invalid = [name for name in include_names if name not in available]
        if invalid:
            raise ValueError(f"Unknown method names in --methods: {invalid}")
        allowed = set(include_names)
        specs = [spec for spec in specs if spec.method_name in allowed]
    if exclude_names is not None:
        invalid = [name for name in exclude_names if name not in available]
        if invalid:
            raise ValueError(f"Unknown method names in --exclude: {invalid}")
        blocked = set(exclude_names)
        specs = [spec for spec in specs if spec.method_name not in blocked]
    if not specs:
        raise ValueError("No methods remain after applying filters.")
    return specs


def _ensure_output_dir(output_dir: Path, rerun: bool) -> None:
    if output_dir.exists() and any(output_dir.iterdir()) and not rerun:
        raise FileExistsError(
            f"Output directory already exists and is not empty: {output_dir}. "
            "Pass --rerun to reuse it or choose a different --output-dir."
        )
    output_dir.mkdir(parents=True, exist_ok=True)


def _list_structured_files(root: Path) -> set[Path]:
    if not root.exists():
        return set()
    return {
        path.resolve()
        for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in STRUCTURED_SUFFIXES
    }


def _relative_to_project(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return resolved.as_posix()


def _tail_lines(text: str, limit: int = 10) -> list[str]:
    stripped = [line.rstrip() for line in text.splitlines() if line.strip()]
    if not stripped:
        return []
    return stripped[-limit:]


def _collect_structured_artifacts(
    spec: MethodSpec,
    before_files: set[Path],
    after_files: set[Path],
    method_output_dir: Path,
) -> list[str]:
    artifact_paths = set(after_files - before_files)
    for expected_name in spec.expected_artifacts:
        expected_path = method_output_dir / expected_name
        if expected_path.exists():
            artifact_paths.add(expected_path.resolve())
    return sorted(_relative_to_project(path) for path in artifact_paths)


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _run_supported_method(
    args: argparse.Namespace,
    spec: MethodSpec,
    output_dir: Path,
) -> MethodResult:
    method_output_dir = output_dir / spec.method_name
    method_output_dir.mkdir(parents=True, exist_ok=True)
    stdout_path = method_output_dir / "stdout.txt"
    stderr_path = method_output_dir / "stderr.txt"
    before_files = _list_structured_files(spec.script_path.parent) | _list_structured_files(
        method_output_dir
    )

    start_wall = time.perf_counter()
    start_utc = _utc_now()
    command = spec.command_builder(args, spec, method_output_dir)
    child_env = os.environ.copy()
    child_env.setdefault("PYTHONIOENCODING", "utf-8")
    child_env.setdefault("PYTHONUTF8", "1")

    completed = subprocess.run(
        command,
        cwd=str(PROJECT_ROOT),
        text=True,
        capture_output=True,
        encoding="utf-8",
        errors="replace",
        env=child_env,
        check=False,
    )

    end_utc = _utc_now()
    duration = round(time.perf_counter() - start_wall, 3)
    _write_text(stdout_path, completed.stdout)
    _write_text(stderr_path, completed.stderr)
    after_files = _list_structured_files(spec.script_path.parent) | _list_structured_files(
        method_output_dir
    )

    run_status = "passed" if completed.returncode == 0 else "failed"
    return MethodResult(
        method_name=spec.method_name,
        script_path=_relative_to_project(spec.script_path),
        support_status=spec.support_status,
        run_status=run_status,
        command=command,
        return_code=int(completed.returncode),
        start_utc=start_utc,
        end_utc=end_utc,
        duration_seconds=duration,
        stdout_path=_relative_to_project(stdout_path),
        stderr_path=_relative_to_project(stderr_path),
        log_dir=_relative_to_project(method_output_dir),
        structured_artifacts=_collect_structured_artifacts(
            spec=spec,
            before_files=before_files,
            after_files=after_files,
            method_output_dir=method_output_dir,
        ),
        stdout_tail=_tail_lines(completed.stdout),
        error_message=(
            f"Subprocess exited with code {completed.returncode}"
            if completed.returncode != 0
            else None
        ),
    )


def _run_unsupported_method(spec: MethodSpec) -> MethodResult:
    return MethodResult(
        method_name=spec.method_name,
        script_path=_relative_to_project(spec.script_path),
        support_status=spec.support_status,
        run_status="skipped_unsupported",
        unsupported_reason=spec.unsupported_reason,
        error_message=spec.unsupported_reason,
    )


def _write_summary_csv(results: list[MethodResult], path: Path) -> None:
    rows = [result.to_summary_row() for result in results]
    fieldnames = list(rows[0].keys()) if rows else [
        "method_name",
        "support_status",
        "run_status",
    ]
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _build_manifest(
    args: argparse.Namespace,
    selected_specs: list[MethodSpec],
    output_dir: Path,
) -> dict[str, object]:
    return {
        "created_utc": _utc_now(),
        "project_root": PROJECT_ROOT.as_posix(),
        "output_dir": output_dir.as_posix(),
        "python": args.python,
        "strict": bool(args.strict),
        "rerun": bool(args.rerun),
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "selected_methods": [spec.method_name for spec in selected_specs],
    }


def _build_summary_payload(
    manifest: dict[str, object],
    results: list[MethodResult],
) -> dict[str, object]:
    counts = {
        "total": len(results),
        "passed": sum(result.run_status == "passed" for result in results),
        "failed": sum(result.run_status == "failed" for result in results),
        "skipped_unsupported": sum(
            result.run_status == "skipped_unsupported" for result in results
        ),
    }
    return {
        "manifest": manifest,
        "counts": counts,
        "results": [result.to_summary_row() for result in results],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--methods", default=None)
    parser.add_argument("--exclude", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--python", default=_default_python_executable())
    parser.add_argument("--strict", action="store_true")
    parser.add_argument("--rerun", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    include_names = _parse_name_filter(args.methods)
    exclude_names = _parse_name_filter(args.exclude)
    output_dir = (
        Path(args.output_dir).resolve()
        if args.output_dir is not None
        else _default_output_dir().resolve()
    )
    args.python = str(Path(args.python))

    specs = _filter_specs(_resolve_specs(), include_names, exclude_names)
    _ensure_output_dir(output_dir, rerun=bool(args.rerun))

    manifest = _build_manifest(args=args, selected_specs=specs, output_dir=output_dir)
    results: list[MethodResult] = []

    for spec in specs:
        print(f"[run] {spec.method_name} ({spec.support_status})", flush=True)
        if spec.support_status != "supported" or spec.command_builder is None:
            result = _run_unsupported_method(spec)
            results.append(result)
            print(f"[skip] {spec.method_name}: {result.unsupported_reason}", flush=True)
            continue

        result = _run_supported_method(args=args, spec=spec, output_dir=output_dir)
        results.append(result)
        print(
            f"[{result.run_status}] {spec.method_name} rc={result.return_code} "
            f"duration={result.duration_seconds}s",
            flush=True,
        )
        if args.strict and result.run_status == "failed":
            break

    summary_payload = _build_summary_payload(manifest=manifest, results=results)
    summary_yaml_path = output_dir / "summary.yaml"
    summary_csv_path = output_dir / "summary.csv"
    manifest_yaml_path = output_dir / "manifest.yaml"

    _write_text(
        summary_yaml_path,
        yaml.safe_dump(summary_payload, sort_keys=False, allow_unicode=False),
    )
    _write_summary_csv(results, summary_csv_path)
    _write_text(
        manifest_yaml_path,
        yaml.safe_dump(manifest, sort_keys=False, allow_unicode=False),
    )

    counts = summary_payload["counts"]
    print(
        "Suite complete: "
        f"passed={counts['passed']} failed={counts['failed']} "
        f"skipped_unsupported={counts['skipped_unsupported']} "
        f"summary={_relative_to_project(summary_yaml_path)}",
        flush=True,
    )

    if args.strict and counts["failed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
