from __future__ import annotations

import datetime

import numpy as np
import torch
import torch.optim as optim


def reconstruct_binary_constraints(
    instance: torch.Tensor,
    binary_feature_indices: list[int],
) -> torch.Tensor:
    if not binary_feature_indices:
        return instance

    candidate = instance.clone()
    squeeze_output = False
    if candidate.ndim == 1:
        candidate = candidate.unsqueeze(0)
        squeeze_output = True

    for feature_index in binary_feature_indices:
        candidate[:, feature_index] = torch.clamp(
            torch.round(candidate[:, feature_index]), min=0.0, max=1.0
        )

    if squeeze_output:
        return candidate.squeeze(0)
    return candidate


def wachter_recourse(
    model,
    x: np.ndarray,
    binary_feature_indices: list[int],
    lr: float,
    lambda_param: float,
    y_target: list[float],
    n_iter: int,
    t_max_min: float,
    norm: int,
    clamp: bool,
    loss_type: str,
) -> np.ndarray:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    x = torch.from_numpy(x).float().to(device)
    y_target_tensor = torch.tensor(y_target, dtype=torch.float32, device=device)
    target_class = int(torch.argmax(y_target_tensor).item())
    x_new = x.clone().detach().requires_grad_(True)
    optimizer = optim.Adam([x_new], lr=lr, amsgrad=True)

    if loss_type.upper() == "MSE":
        loss_fn = torch.nn.MSELoss()
    elif loss_type.upper() == "BCE":
        loss_fn = torch.nn.BCELoss()
    else:
        raise ValueError(f"loss_type {loss_type} not supported")

    t0 = datetime.datetime.now()
    t_max = datetime.timedelta(minutes=float(t_max_min))

    while True:
        success = False
        for _ in range(n_iter):
            optimizer.zero_grad()
            x_new_enc = reconstruct_binary_constraints(x_new, binary_feature_indices)
            probabilities = model.predict_proba(x_new_enc.unsqueeze(0))
            target_probability = probabilities[:, target_class]

            if loss_type.upper() == "MSE":
                f_x_loss = torch.log(
                    target_probability.clamp(min=1e-6, max=1 - 1e-6)
                    / (1 - target_probability.clamp(min=1e-6, max=1 - 1e-6))
                )
                loss_target = torch.tensor([1.0], dtype=torch.float32, device=device)
            else:
                f_x_loss = probabilities.squeeze(0)
                loss_target = y_target_tensor

            cost = torch.dist(x_new_enc, x, norm)
            loss = loss_fn(f_x_loss, loss_target) + float(lambda_param) * cost
            loss.backward()
            optimizer.step()

            if clamp:
                with torch.no_grad():
                    x_new.clamp_(0.0, 1.0)

            predicted = int(
                torch.argmax(model.predict_proba(x_new.unsqueeze(0)), dim=1)[0]
            )
            if predicted == target_class:
                success = True
                break

        if success or (datetime.datetime.now() - t0 > t_max):
            break

    output = reconstruct_binary_constraints(x_new, binary_feature_indices)
    output_array = output.detach().cpu().numpy()
    if output_array.ndim > 1:
        output_array = output_array.squeeze(axis=0)
    return output_array
