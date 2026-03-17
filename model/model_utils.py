from __future__ import annotations

import logging
from pathlib import Path

import torch

from utils.caching import get_cache_dir


def resolve_device(device: str) -> str:
    device = device.lower()
    if device not in {"cpu", "cuda"}:
        raise ValueError(f"Unsupported device: {device}")
    if device == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA is not available in the current environment")
    return device


def logits_to_prediction(logits: torch.Tensor, proba: bool = True) -> torch.Tensor:
    probabilities = torch.softmax(logits, dim=1)
    if proba:
        return probabilities
    indices = probabilities.argmax(dim=1)
    return torch.nn.functional.one_hot(indices, num_classes=probabilities.shape[1]).to(
        dtype=torch.float32
    )


def build_optimizer(optimizer_name: str, parameters, learning_rate: float):
    optimizer_name = optimizer_name.lower()
    if optimizer_name == "adam":
        return torch.optim.Adam(parameters, lr=learning_rate)
    if optimizer_name == "sgd":
        return torch.optim.SGD(parameters, lr=learning_rate)
    raise ValueError(f"Unsupported optimizer: {optimizer_name}")


def save_torch_model(model: torch.nn.Module, save_name: str | None) -> None:
    if save_name is None:
        return
    model_dir = Path(get_cache_dir("models"))
    save_path = model_dir / f"{save_name}.pt"
    torch.save(model.state_dict(), save_path)
    logging.getLogger(__name__).info(
        "Saved model checkpoint to %s", save_path.as_posix()
    )
