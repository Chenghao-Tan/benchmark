"""Logging helpers for benchmark scripts and experiments."""

from __future__ import annotations

import logging
import sys
from pathlib import Path

global_logger_initialized = False


def setup_logger(
    level: str = "INFO", path: str | None = None, name: str = "benchmark"
) -> logging.Logger:
    """Configure the root logger once and return a named logger.

    Args:
        level: Logging level name.
        path: Optional log file path. When provided, parent directories are
            created automatically.
        name: Name of the logger to return.

    Returns:
        logging.Logger: Configured logger instance.
    """
    global global_logger_initialized

    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, level.upper(), logging.INFO))

    if global_logger_initialized:
        return logging.getLogger(name)

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    root_logger.addHandler(stream_handler)

    if path is not None:
        log_path = Path(path)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_path)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)

    global_logger_initialized = True
    return logging.getLogger(name)
