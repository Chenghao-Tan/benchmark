from __future__ import annotations

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from method.hare.support import FeatureSchema, ModelAdapter


def _enforce_constraints(
    directions: torch.Tensor,
    categorical_indices: list[int] | None,
    immutable_indices: list[int] | None,
) -> torch.Tensor:
    constrained = directions.clone()
    for indices in (categorical_indices or [], immutable_indices or []):
        constrained[:, indices] = 0.0
    return constrained


class PrototypeDirections(nn.Module):
    def __init__(self, num_candidates: int, num_features: int):
        super().__init__()
        self._directions = nn.Parameter(torch.randn(num_candidates, num_features))
        self._identity = nn.Parameter(
            torch.eye(num_candidates, dtype=torch.float32),
            requires_grad=False,
        )

    def forward(self) -> torch.Tensor:
        return F.normalize(self._directions, p=2, dim=1)

    @property
    def identity(self) -> torch.Tensor:
        return self._identity


class ReferenceSamplingLoss(nn.Module):
    def __init__(
        self,
        identity: torch.Tensor,
        model: ModelAdapter,
        baseline: torch.Tensor,
        target_index: int,
        lambda_: float,
    ) -> None:
        super().__init__()
        self._identity = identity
        self._model = model
        self._baseline = baseline
        self._target_index = int(target_index)
        self._lambda = float(lambda_)
        self.mean_similarity = torch.tensor(0.0)

    def forward(self, directions: torch.Tensor) -> torch.Tensor:
        similarity = directions @ directions.t() - (2.0 * self._identity)
        diversity_loss = similarity.max(dim=1)[0].mean()
        self.mean_similarity = diversity_loss.detach().clone()

        sampled = self._baseline + directions
        probabilities = self._model.predict_proba(sampled)
        if not isinstance(probabilities, torch.Tensor):
            raise TypeError("ReferenceSamplingLoss requires a differentiable model")

        target_probability = probabilities[:, self._target_index]
        validity_loss = F.binary_cross_entropy(
            target_probability,
            torch.ones_like(target_probability),
        )
        return validity_loss + self._lambda * diversity_loss


def _hyperspherical_sampling(
    baseline: pd.DataFrame,
    model: ModelAdapter,
    num_candidates: int,
    lambda_: float,
    categorical_indices: list[int],
    immutable_indices: list[int],
    target_index: int,
    epochs: int,
    lr: float,
) -> pd.DataFrame:
    if num_candidates < 1:
        return baseline.iloc[0:0].copy(deep=True)
    if not model.supports_gradients:
        return baseline.iloc[0:0].copy(deep=True)

    device = model._target_model._device
    baseline_tensor = torch.tensor(
        baseline.to_numpy(dtype="float32"),
        dtype=torch.float32,
        device=device,
    ).view(1, -1)
    prototypes = PrototypeDirections(num_candidates, baseline.shape[1]).to(device)
    optimizer = torch.optim.Adam(prototypes.parameters(), lr=float(lr))
    loss_fn = ReferenceSamplingLoss(
        identity=prototypes.identity,
        model=model,
        baseline=baseline_tensor,
        target_index=target_index,
        lambda_=lambda_,
    )

    for _ in range(max(1, int(epochs))):
        optimizer.zero_grad()
        loss = loss_fn(prototypes())
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        directions = prototypes()
        directions = _enforce_constraints(
            directions,
            categorical_indices=categorical_indices,
            immutable_indices=immutable_indices,
        )
        sampled = baseline_tensor + directions
        probabilities = model.predict_proba(sampled)
        if not isinstance(probabilities, torch.Tensor):
            return baseline.iloc[0:0].copy(deep=True)
        target_probability = probabilities[:, int(target_index)]
        mask = target_probability >= 0.5
        sampled_array = sampled[mask].detach().cpu().numpy()

    return pd.DataFrame(sampled_array, columns=baseline.columns)


def actionable_sampling(
    baseline: pd.DataFrame,
    factual: pd.DataFrame,
    model: ModelAdapter,
    schema: FeatureSchema,
    target_index: int,
    num_candidates: int,
    radius: float = 1.0,
    lambda_: float = 10.0,
    lr: float = 0.1,
    epochs: int = 100,
) -> pd.DataFrame:
    del factual, radius
    baseline = baseline.loc[:, schema.feature_names].copy(deep=True).reset_index(drop=True)
    sampled = _hyperspherical_sampling(
        baseline=baseline,
        model=model,
        num_candidates=int(num_candidates),
        lambda_=float(lambda_),
        categorical_indices=list(schema.binary_indices),
        immutable_indices=list(schema.immutable_indices),
        target_index=int(target_index),
        epochs=int(epochs),
        lr=float(lr),
    )
    return pd.concat([baseline, sampled], axis=0, ignore_index=True)
