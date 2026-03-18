from __future__ import annotations

import math
from typing import Optional

import numpy as np
import torch
from sklearn.utils import check_random_state


@torch.no_grad()
def l2_projection(x: torch.Tensor, radius: float) -> torch.Tensor:
    norm = torch.linalg.norm(x, ord=2, axis=-1)
    denom = torch.max(norm, torch.tensor(radius, device=x.device))
    scale = (radius / denom).unsqueeze(1)
    return scale * x


class OptimisticLikelihood(torch.nn.Module):
    def __init__(
        self,
        x_dim: torch.Tensor,
        epsilon_op: torch.Tensor,
        sigma: torch.Tensor,
        device: torch.device,
    ):
        super().__init__()
        self.device = device
        self.x_dim = x_dim.to(self.device)
        self.epsilon_op = epsilon_op.to(self.device)
        self.sigma = sigma.to(self.device)

    @torch.no_grad()
    def projection(self, v: torch.Tensor) -> torch.Tensor:
        v = v.clone()
        v = torch.max(v, torch.tensor(0, device=self.device))
        return l2_projection(v, float(self.epsilon_op)).to(self.device)

    def forward(self, v: torch.Tensor, x: torch.Tensor, x_feas: torch.Tensor):
        c = torch.linalg.norm(x - x_feas, axis=-1)
        d = v[..., 1] + self.sigma
        p = self.x_dim
        L = (
            torch.log(d)
            + (c - v[..., 0]) ** 2 / (2 * d**2)
            + (p - 1) * torch.log(self.sigma)
        )
        v_grad = torch.zeros_like(v, device=self.device)
        v_grad[..., 0] = -(c - v[..., 0]) / d**2
        v_grad[..., 1] = 1 / d - (c - v[..., 0]) ** 2 / d**3
        return L, v_grad

    def optimize(self, x: torch.Tensor, x_feas: torch.Tensor, max_iter: int = 1000):
        v = torch.zeros([x.shape[0], 2], device=self.device)
        lr = 1 / torch.sqrt(torch.tensor(max_iter, device=self.device).float())
        min_loss = float("inf")
        num_stable_iter = 0
        for _ in range(max_iter):
            F, grad = self.forward(v, x, x_feas)
            v = self.projection(v - lr * grad)
            loss_sum = F.sum().data.item()
            loss_diff = min_loss - loss_sum
            if loss_diff <= 1e-10:
                num_stable_iter += 1
                if num_stable_iter >= 10:
                    break
            else:
                num_stable_iter = 0
            min_loss = min(min_loss, loss_sum)
        return v


class PessimisticLikelihood(torch.nn.Module):
    def __init__(
        self,
        x_dim: torch.Tensor,
        epsilon_pe: torch.Tensor,
        sigma: torch.Tensor,
        device: torch.device,
    ):
        super().__init__()
        self.device = device
        self.epsilon_pe = epsilon_pe.to(self.device)
        self.sigma = sigma.to(self.device)
        self.x_dim = x_dim.to(self.device)

    @torch.no_grad()
    def projection(self, u: torch.Tensor) -> torch.Tensor:
        u = u.clone()
        u = torch.max(u, torch.tensor(0, device=self.device))
        result = l2_projection(u, float(self.epsilon_pe) / math.sqrt(float(self.x_dim)))
        return result.to(self.device)

    def forward(
        self, u: torch.Tensor, x: torch.Tensor, x_feas: torch.Tensor, zeta: float = 1e-6
    ):
        c = torch.linalg.norm(x - x_feas, axis=-1)
        d = u[..., 1] + self.sigma
        p = self.x_dim
        sqrt_p = torch.sqrt(p.float())
        inside = (zeta + self.epsilon_pe**2 - p * u[..., 0] ** 2 - u[..., 1] ** 2) / (
            p - 1
        )
        f = torch.sqrt(inside)
        L = (
            -torch.log(d)
            - (c + sqrt_p * u[..., 0]) ** 2 / (2 * d**2)
            - (p - 1) * torch.log(f + self.sigma)
        )
        u_grad = torch.zeros_like(u, device=self.device)
        u_grad[..., 0] = -sqrt_p * (c + sqrt_p * u[..., 0]) / d**2 - (
            p * u[..., 0]
        ) / (f * (f + self.sigma))
        u_grad[..., 1] = (
            -1 / d
            + (c + sqrt_p * u[..., 0]) ** 2 / d**3
            + u[..., 1] / (f * (f + self.sigma))
        )
        return L, u_grad

    def optimize(self, x: torch.Tensor, x_feas: torch.Tensor, max_iter: int = 1000):
        u = torch.zeros([x.shape[0], 2], device=self.device)
        lr = 1.0 / torch.sqrt(torch.tensor(max_iter, device=self.device).float())
        min_loss = float("inf")
        num_stable_iter = 0
        for _ in range(max_iter):
            F, grad = self.forward(u, x, x_feas)
            u = self.projection(u - lr * grad)
            loss_sum = F.sum().data.item()
            loss_diff = min_loss - loss_sum
            if loss_diff <= 1e-10:
                num_stable_iter += 1
                if num_stable_iter >= 10:
                    break
            else:
                num_stable_iter = 0
            min_loss = min(min_loss, loss_sum)
        return u


class RBRLoss(torch.nn.Module):
    def __init__(
        self,
        X_feas: torch.Tensor,
        X_feas_pos: torch.Tensor,
        X_feas_neg: torch.Tensor,
        epsilon_op: float,
        epsilon_pe: float,
        sigma: float,
        device: torch.device,
    ):
        super().__init__()
        self.device = device
        self.X_feas = X_feas.to(self.device)
        self.X_feas_pos = X_feas_pos.to(self.device)
        self.X_feas_neg = X_feas_neg.to(self.device)
        self.epsilon_op = torch.tensor(epsilon_op, device=self.device)
        self.epsilon_pe = torch.tensor(epsilon_pe, device=self.device)
        self.sigma = torch.tensor(sigma, device=self.device)
        self.x_dim = torch.tensor(self.X_feas.shape[-1], device=self.device)

        self.op = OptimisticLikelihood(
            self.x_dim, self.epsilon_op, self.sigma, self.device
        )
        self.pe = PessimisticLikelihood(
            self.x_dim, self.epsilon_pe, self.sigma, self.device
        )

    def forward(self, x: torch.Tensor):
        if self.X_feas_pos.shape[0] == 0 or self.X_feas_neg.shape[0] == 0:
            F = torch.linalg.norm(self.X_feas - x, ord=2, axis=-1)
            return F, F, F

        x_pos = x.unsqueeze(0).expand(self.X_feas_pos.shape[0], -1)
        x_neg = x.unsqueeze(0).expand(self.X_feas_neg.shape[0], -1)
        v = self.op.optimize(x_pos, self.X_feas_pos)
        u = self.pe.optimize(x_neg, self.X_feas_neg)
        numer, _ = self.op.forward(v, x_pos, self.X_feas_pos)
        denom, _ = self.pe.forward(u, x_neg, self.X_feas_neg)
        F = numer - denom
        return F, denom, numer


def robust_bayesian_recourse(
    raw_model,
    x0: np.ndarray,
    train_data: np.ndarray,
    num_samples: int = 200,
    perturb_radius: float = 0.2,
    delta_plus: float = 1.0,
    sigma: float = 1.0,
    epsilon_op: float = 0.5,
    epsilon_pe: float = 1.0,
    max_iter: int = 1000,
    dev: str = "cpu",
    random_state: Optional[int] = None,
    verbose: bool = False,
) -> np.ndarray:
    def predict_fn_np(x):
        preds_tensor = raw_model.predict(x)
        if preds_tensor.ndim == 1:
            preds_tensor = preds_tensor.unsqueeze(0)
        preds = preds_tensor.detach().cpu().numpy()
        if preds.ndim == 2 and preds.shape[1] > 1:
            preds = preds.argmax(axis=1)
        elif preds.dtype.kind == "f":
            if preds.ndim == 2 and preds.shape[1] == 1:
                preds = preds.squeeze()
            preds = (preds >= 0.5).astype(int)
        elif preds.ndim == 2 and preds.shape[1] == 1:
            probs = 1 / (1 + np.exp(-preds))
            preds = (probs >= 0.5).astype(int)
        else:
            preds = preds.astype(int)
        if np.asarray(preds).size == 1:
            return int(np.asarray(preds).item())
        return preds

    def dist(a: torch.Tensor, b: torch.Tensor):
        return torch.linalg.norm(a - b, ord=1, axis=-1)

    def uniform_ball(x: torch.Tensor, r: float, n: int, rng_state):
        rng_local = check_random_state(rng_state)
        d = x.shape[0]
        V = rng_local.randn(n, d)
        V = V / np.linalg.norm(V, axis=1).reshape(-1, 1)
        V = V * (rng_local.random(n) ** (1.0 / d)).reshape(-1, 1)
        V = V * r + x.cpu().numpy()
        return torch.from_numpy(V).float().to(dev)

    def simplex_projection(x, delta):
        (p,) = x.shape
        if torch.linalg.norm(x, ord=1) == delta and torch.all(x >= 0):
            return x
        u, _ = torch.sort(x, descending=True)
        cssv = torch.cumsum(u, 0)
        rho = torch.nonzero(u * torch.arange(1, p + 1).to(dev) > (cssv - delta))[-1, 0]
        theta = (cssv[rho] - delta) / (rho + 1.0)
        return torch.clip(x - theta, min=0)

    def projection(x, delta):
        x_abs = torch.abs(x)
        if x_abs.sum() <= delta:
            return x
        proj = simplex_projection(x_abs, delta=delta)
        proj *= torch.sign(x)
        return proj

    rng = check_random_state(random_state)
    if train_data is None:
        raise ValueError("train_data must be provided to robust_bayesian_recourse")

    x0_t = torch.from_numpy(x0.copy()).float().to(dev)
    train_t = torch.tensor(train_data).float().to(dev)
    train_label = torch.tensor(predict_fn_np(train_t)).to(dev)
    x_label = torch.tensor(predict_fn_np(x0_t.clone()), device=dev)

    dists = dist(train_t, x0_t)
    order = torch.argsort(dists)
    candidates = train_t[order[train_label[order] == (1 - x_label)]][:1000]
    best_x_b = None
    best_dist = torch.tensor(float("inf"), device=dev)
    for x_c in candidates:
        lambdas = torch.linspace(0, 1, 100, device=dev)
        for lam in lambdas:
            x_b = (1 - lam) * x0_t + lam * x_c
            label = predict_fn_np(x_b)
            if label == 1 - x_label:
                curdist = dist(x0_t, x_b)
                if curdist < best_dist:
                    best_x_b = x_b.detach().clone()
                    best_dist = curdist.detach().clone()
                break
    if best_x_b is None:
        opp_idx = (train_label == (1 - x_label)).nonzero(as_tuple=False)
        if opp_idx.shape[0] == 0:
            return x0.copy()
        first_idx = opp_idx[0, 0].item()
        best_x_b = train_t[first_idx].detach().clone()
        best_dist = dist(x0_t, best_x_b)

    delta = best_dist + delta_plus
    X_feas = uniform_ball(best_x_b, perturb_radius, num_samples, rng).float().to(dev)
    y_feas = predict_fn_np(X_feas)
    if (y_feas == 1).any():
        X_feas_pos = X_feas[y_feas == 1].reshape([int((y_feas == 1).sum().item()), -1])
    else:
        X_feas_pos = torch.empty((0, X_feas.shape[1]), device=dev)
    if (y_feas == 0).any():
        X_feas_neg = X_feas[y_feas == 0].reshape([int((y_feas == 0).sum().item()), -1])
    else:
        X_feas_neg = torch.empty((0, X_feas.shape[1]), device=dev)

    loss_fn = RBRLoss(
        X_feas,
        X_feas_pos,
        X_feas_neg,
        epsilon_op,
        epsilon_pe,
        sigma,
        device=dev,
    )

    x_t = best_x_b.detach().clone()
    x_t.requires_grad_(True)
    min_loss = float("inf")
    num_stable_iter = 0
    step = 1.0 / math.sqrt(1e3)
    for _ in range(max_iter):
        if x_t.grad is not None:
            x_t.grad.data.zero_()
        F, _, _ = loss_fn(x_t)
        F_sum = F.sum()
        F_sum.backward()
        if torch.ge(torch.linalg.norm((x_t.detach() - x0_t), ord=1), float(delta)):
            break
        with torch.no_grad():
            x_new = x_t - step * x_t.grad
            x_new = projection(x_new - x0_t, float(delta)) + x0_t
        for i, e in enumerate(x_new.data):
            x_t.data[i] = e
        loss_sum = F_sum.item()
        loss_diff = min_loss - loss_sum
        if loss_diff <= 1e-10:
            num_stable_iter += 1
            if num_stable_iter >= 10:
                break
        else:
            num_stable_iter = 0
        min_loss = min(min_loss, loss_sum)

    return x_t.detach().cpu().numpy().squeeze()
