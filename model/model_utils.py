"""Shared helpers for target model implementations."""

from __future__ import annotations

import logging
from pathlib import Path

import torch

from utils.caching import get_cache_dir


def resolve_device(device: str) -> str:
    """Validate and normalize the requested execution device.

    Args:
        device: Requested device name.

    Returns:
        str: Normalized device string.

    Raises:
        ValueError: If the device is unsupported or CUDA is unavailable.
    """
    device = device.lower()
    if device not in {"cpu", "cuda"}:
        raise ValueError(f"Unsupported device: {device}")
    if device == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA is not available in the current environment")
    return device


def logits_to_prediction(
    logits: torch.Tensor,
    proba: bool = True,
    output_activation: str = "softmax",
) -> torch.Tensor:
    """Convert model logits into probabilities or one-hot predictions.

    Args:
        logits: Raw model outputs.
        proba: When ``True``, return probabilities. Otherwise return one-hot
            hard predictions.
        output_activation: Output convention expected from the model head.

    Returns:
        torch.Tensor: Probability matrix or one-hot predictions.

    Raises:
        ValueError: If ``output_activation`` is unsupported or incompatible with
            the logits shape.
    """
    output_activation = output_activation.lower()
    if output_activation == "softmax":
        probabilities = torch.softmax(logits, dim=1)
        if proba:
            return probabilities
        indices = probabilities.argmax(dim=1)
        return torch.nn.functional.one_hot(
            indices, num_classes=probabilities.shape[1]
        ).to(dtype=torch.float32)

    if output_activation == "sigmoid":
        if logits.ndim == 1:
            logits = logits.unsqueeze(1)
        if logits.shape[1] != 1:
            raise ValueError(
                "sigmoid output activation requires a single-logit output layer"
            )
        positive_probability = torch.sigmoid(logits)
        probabilities = torch.cat(
            [1.0 - positive_probability, positive_probability], dim=1
        )
        if proba:
            return probabilities
        indices = (positive_probability.reshape(-1) >= 0.5).to(dtype=torch.long)
        return torch.nn.functional.one_hot(indices, num_classes=2).to(
            dtype=torch.float32
        )

    raise ValueError(f"Unsupported output activation: {output_activation}")


def build_optimizer(optimizer_name: str, parameters, learning_rate: float):
    """Construct a torch optimizer from a benchmark configuration name.

    Args:
        optimizer_name: Optimizer identifier.
        parameters: Iterable of model parameters.
        learning_rate: Optimizer learning rate.

    Returns:
        torch.optim.Optimizer: Configured optimizer instance.

    Raises:
        ValueError: If the optimizer name is unsupported.
    """
    optimizer_name = optimizer_name.lower()
    if optimizer_name == "adam":
        return torch.optim.Adam(parameters, lr=learning_rate)
    if optimizer_name == "sgd":
        return torch.optim.SGD(parameters, lr=learning_rate)
    if optimizer_name in {"rms", "rmsprop"}:
        return torch.optim.RMSprop(parameters, lr=learning_rate)
    raise ValueError(f"Unsupported optimizer: {optimizer_name}")


def save_torch_model(model: torch.nn.Module, save_name: str | None) -> None:
    """Persist a torch model checkpoint in the benchmark cache directory.

    Args:
        model: Trained torch model to serialize.
        save_name: Checkpoint name without the ``.pt`` suffix. When ``None``,
            no checkpoint is written.
    """
    if save_name is None:
        return
    model_dir = Path(get_cache_dir("models"))
    save_path = model_dir / f"{save_name}.pt"
    torch.save(model.state_dict(), save_path)
    logging.getLogger(__name__).info(
        "Saved model checkpoint to %s", save_path.as_posix()
    )
