"""Deterministic seeding helpers for reproducible benchmark runs."""

from __future__ import annotations

import contextlib
import random

import numpy as np
import torch


def _capture_torch_backend_state() -> dict[str, bool | None]:
    state: dict[str, bool | None] = {
        "deterministic_algorithms": None,
        "deterministic_algorithms_warn_only": None,
        "cudnn_deterministic": None,
        "cudnn_benchmark": None,
    }

    if hasattr(torch, "are_deterministic_algorithms_enabled"):
        state["deterministic_algorithms"] = torch.are_deterministic_algorithms_enabled()
    if hasattr(torch, "is_deterministic_algorithms_warn_only_enabled"):
        state["deterministic_algorithms_warn_only"] = (
            torch.is_deterministic_algorithms_warn_only_enabled()
        )

    cudnn_backend = getattr(torch.backends, "cudnn", None)
    if cudnn_backend is not None:
        if hasattr(cudnn_backend, "deterministic"):
            state["cudnn_deterministic"] = bool(cudnn_backend.deterministic)
        if hasattr(cudnn_backend, "benchmark"):
            state["cudnn_benchmark"] = bool(cudnn_backend.benchmark)

    return state


def _enable_deterministic_torch_backends() -> None:
    if hasattr(torch, "use_deterministic_algorithms"):
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except TypeError:
            torch.use_deterministic_algorithms(True)

    cudnn_backend = getattr(torch.backends, "cudnn", None)
    if cudnn_backend is not None:
        if hasattr(cudnn_backend, "deterministic"):
            cudnn_backend.deterministic = True
        if hasattr(cudnn_backend, "benchmark"):
            cudnn_backend.benchmark = False


def _restore_torch_backend_state(state: dict[str, bool | None]) -> None:
    deterministic_algorithms = state.get("deterministic_algorithms")
    warn_only = state.get("deterministic_algorithms_warn_only")
    if deterministic_algorithms is not None and hasattr(
        torch, "use_deterministic_algorithms"
    ):
        try:
            if warn_only is None:
                torch.use_deterministic_algorithms(deterministic_algorithms)
            else:
                torch.use_deterministic_algorithms(
                    deterministic_algorithms, warn_only=warn_only
                )
        except TypeError:
            torch.use_deterministic_algorithms(deterministic_algorithms)

    cudnn_backend = getattr(torch.backends, "cudnn", None)
    if cudnn_backend is not None:
        cudnn_deterministic = state.get("cudnn_deterministic")
        cudnn_benchmark = state.get("cudnn_benchmark")
        if cudnn_deterministic is not None and hasattr(cudnn_backend, "deterministic"):
            cudnn_backend.deterministic = cudnn_deterministic
        if cudnn_benchmark is not None and hasattr(cudnn_backend, "benchmark"):
            cudnn_backend.benchmark = cudnn_benchmark


@contextlib.contextmanager
def seed_context(seed: int | None):
    """Temporarily seed Python, NumPy, and torch for deterministic execution.

    The previous random states are restored when the context exits so callers
    can scope reproducibility to a single training or inference block.

    Args:
        seed: Requested seed. When ``None``, a temporary random seed is drawn.

    Yields:
        int: The active seed used inside the context.
    """
    python_state = random.getstate()
    numpy_state = np.random.get_state()
    torch_state = torch.random.get_rng_state()
    cuda_state = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    torch_backend_state = _capture_torch_backend_state()

    active_seed = (
        seed if seed is not None else random.SystemRandom().randrange(0, 2**32 - 1)
    )
    random.seed(active_seed)
    np.random.seed(active_seed)
    torch.manual_seed(active_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(active_seed)
    _enable_deterministic_torch_backends()

    try:
        yield active_seed
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)
        torch.random.set_rng_state(torch_state)
        if cuda_state is not None:
            torch.cuda.set_rng_state_all(cuda_state)
        _restore_torch_backend_state(torch_backend_state)
