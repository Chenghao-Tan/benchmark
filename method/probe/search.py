from __future__ import annotations

import datetime

import numpy as np
import torch
import torch.distributions.normal as normal_distribution
import torch.optim as optim
from torch.distributions.multivariate_normal import MultivariateNormal

from method.wachter.search import reconstruct_binary_constraints


def gradient(y, x, grad_outputs=None):
    if grad_outputs is None:
        grad_outputs = torch.tensor(1.0, device=y.device)
    return torch.autograd.grad(y, [x], grad_outputs=grad_outputs, create_graph=True)[0]


def compute_jacobian(inputs, output):
    return gradient(output, inputs)


def compute_invalidation_rate_closed(model, x, sigma2):
    probabilities = model.predict_proba(x.unsqueeze(0)).squeeze(0)
    logit_x = torch.log(probabilities[1] / probabilities[0])
    jacobian_x = compute_jacobian(x, logit_x).reshape(-1)
    denom = torch.sqrt(torch.tensor(sigma2, device=x.device)) * torch.norm(
        jacobian_x, 2
    )
    arg = logit_x / denom.clamp(min=1e-6)
    normal = normal_distribution.Normal(loc=0.0, scale=1.0)
    return 1 - normal.cdf(arg)


def reparametrization_trick(mu, sigma2, device, n_samples):
    std = torch.sqrt(torch.tensor(sigma2, device=device))
    epsilon = MultivariateNormal(
        loc=torch.zeros(mu.shape[0], device=device),
        covariance_matrix=torch.eye(mu.shape[0], device=device),
    ).sample((n_samples,))
    return mu.reshape(-1) + std * epsilon


def compute_invalidation_rate(model, random_samples):
    yhat = model.predict_proba(random_samples)[:, 1]
    hat = (yhat > 0.5).float()
    return 1 - torch.mean(hat, 0)


def probe_recourse(
    model,
    x: np.ndarray,
    binary_feature_indices: list[int],
    lr: float = 0.07,
    lambda_param: float = 5,
    y_target: list[float] | None = None,
    n_iter: int = 500,
    t_max_min: float = 1.0,
    norm: int = 1,
    clamp: bool = False,
    loss_type: str = "MSE",
    invalidation_target: float = 0.45,
    inval_target_eps: float = 0.005,
    noise_variance: float = 0.01,
) -> np.ndarray:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if y_target is None:
        y_target = [0.0, 1.0]

    x = torch.from_numpy(x).float().to(device)
    y_target_tensor = torch.tensor(y_target, dtype=torch.float32, device=device)
    x_new = x.clone().detach().requires_grad_(True)
    optimizer = optim.Adam([x_new], lr, amsgrad=True)

    if loss_type.upper() == "MSE":
        loss_fn = torch.nn.MSELoss()
    else:
        loss_fn = torch.nn.BCELoss()

    t0 = datetime.datetime.now()
    t_max = datetime.timedelta(minutes=float(t_max_min))
    random_samples = reparametrization_trick(
        x_new.detach(), noise_variance, device, n_samples=1000
    )
    invalidation_rate = compute_invalidation_rate(model, random_samples)

    while invalidation_rate > invalidation_target + inval_target_eps:
        for _ in range(n_iter):
            optimizer.zero_grad()
            x_new_enc = reconstruct_binary_constraints(x_new, binary_feature_indices)
            f_x_new_binary = model.predict_proba(x_new_enc.unsqueeze(0)).squeeze(0)

            cost = torch.dist(x_new_enc, x, norm)
            invalidation_rate_closed = compute_invalidation_rate_closed(
                model, x_new_enc, noise_variance
            )
            loss_invalidation = invalidation_rate_closed - invalidation_target
            if loss_invalidation < 0:
                loss_invalidation = torch.zeros_like(loss_invalidation)

            if loss_type.upper() == "MSE":
                f_x_loss = torch.log(
                    f_x_new_binary[1].clamp(min=1e-6, max=1 - 1e-6)
                    / (1 - f_x_new_binary[1].clamp(min=1e-6, max=1 - 1e-6))
                )
                target_loss = torch.tensor([1.0], device=device)
            else:
                f_x_loss = f_x_new_binary
                target_loss = y_target_tensor

            loss = (
                3 * loss_invalidation
                + loss_fn(f_x_loss, target_loss)
                + lambda_param * cost
            )
            loss.backward()
            optimizer.step()

            random_samples = reparametrization_trick(
                x_new.detach(), noise_variance, device, n_samples=1000
            )
            invalidation_rate = compute_invalidation_rate(model, random_samples)

            if clamp:
                with torch.no_grad():
                    x_new.clamp_(0.0, 1.0)

            predicted = int(
                torch.argmax(model.predict_proba(x_new.unsqueeze(0)), dim=1)[0]
            )
            if predicted == int(torch.argmax(y_target_tensor).item()) and (
                invalidation_rate < invalidation_target + inval_target_eps
            ):
                break

        if datetime.datetime.now() - t0 > t_max:
            break
        if invalidation_rate < invalidation_target + inval_target_eps:
            break

    output = reconstruct_binary_constraints(x_new, binary_feature_indices)
    output_array = output.detach().cpu().numpy()
    if output_array.ndim > 1:
        output_array = output_array.squeeze(axis=0)
    return output_array
