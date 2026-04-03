"""Cache directory helpers used by benchmark components."""

from __future__ import annotations

from pathlib import Path

global_cache_dir = "./cache/"


def set_cache_dir(path: str) -> None:
    """Set the base cache directory and ensure it exists.

    Args:
        path: New base path for benchmark cache artifacts.
    """
    global global_cache_dir
    global_cache_dir = path
    Path(global_cache_dir).mkdir(parents=True, exist_ok=True)


def get_cache_dir(sub: str) -> str:
    """Return a cache subdirectory path and ensure it exists.

    Args:
        sub: Cache subdirectory name.

    Returns:
        str: POSIX-style path to the requested cache directory with a trailing
        slash.
    """
    path = Path(global_cache_dir) / sub
    path.mkdir(parents=True, exist_ok=True)
    return f"{path.as_posix()}/"
