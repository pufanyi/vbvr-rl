"""COS (Chain-of-Step) interpolation paths for N-step piecewise flow matching.

Controls how the flow-matching path traverses
    noise → v_0 → v_1 → … → v_{N-1}  (= x_final)
in sigma-space, with piecewise boundaries τ_0 > τ_1 > … > τ_{K-1}
(K = N-1 intermediates).

Terminology
-----------
anchors     [noise, v_0, v_1, …, v_{N-1}]  — len N+1
boundaries  [1.0,  τ_0, τ_1, …, τ_{K-1}, 0.0]  — len N+1
segment i   σ ∈ [boundaries[i+1], boundaries[i]], interpolating anchors[i] ↔ anchors[i+1]

Path families
-------------
**Passthrough** – the path literally passes through every waypoint at its τ.

**Target-blend** – the velocity field smoothly blends among waypoints;
the path does NOT pass exactly through them.

Supported path types (N-step):
  linear           Piecewise linear passthrough (C0 at boundaries).
  target_cosine    Smooth cosine target blend (C1 everywhere).
  target_sigmoid   Smooth sigmoid target blend (C1 everywhere).

Legacy 2-step-only paths (raise for N>2):
  cosine, cubic_hermite, smooth_blend, quadratic_bezier, target_linear
"""

from __future__ import annotations

import math
from typing import Literal

import torch
from torch import Tensor

PathType = Literal[
    "linear",
    "cosine",
    "cubic_hermite",
    "smooth_blend",
    "quadratic_bezier",
    "target_linear",
    "target_cosine",
    "target_sigmoid",
]


def compute_cos_path(
    path_type: PathType,
    sigma: Tensor,
    taus: list[float],
    noise: Tensor,
    waypoints: list[Tensor],
    *,
    boundary_noise_std: float = 0.0,
    smooth_blend_delta: float = 0.05,
    sigmoid_steepness: float = 10.0,
) -> tuple[Tensor, Tensor]:
    """Compute interpolated sample *x_t* and velocity target *dx/dσ*.

    Args:
        path_type: Interpolation strategy.
        sigma: Per-sample sigma, broadcastable ``(B, 1, 1, 1, 1)``.
        taus: Descending boundary list, len K (number of intermediates).
        noise: Pure Gaussian noise, same shape as waypoints[0].
        waypoints: ``[v_0, …, v_{N-1}]`` with ``v_{N-1} = x_final``.
            len = K+1.
        boundary_noise_std: Gaussian std added to intermediate waypoints
            for samples below each waypoint's sigma position.
        smooth_blend_delta: Half-width of the blending window for
            ``smooth_blend`` (2-step only).
        sigmoid_steepness: Logistic steepness for ``target_sigmoid``.

    Returns:
        ``(x_t, target)`` with the same shape as noise.
    """
    _validate_nstep_inputs(path_type, taus, waypoints)
    K = len(taus)

    if path_type == "linear":
        return _linear_n(sigma, taus, noise, waypoints, boundary_noise_std)
    if path_type == "target_cosine":
        return _target_cosine_n(sigma, taus, noise, waypoints, boundary_noise_std)
    if path_type == "target_sigmoid":
        return _target_sigmoid_n(sigma, taus, noise, waypoints, boundary_noise_std, sigmoid_steepness)

    # Legacy 2-step-only paths
    if K != 1:
        raise ValueError(f"Path type {path_type!r} only supports 2-step (1 tau), got {K} taus")

    tau = taus[0]
    x_tau = waypoints[0]
    x_final = waypoints[-1]

    # Apply boundary noise for legacy paths
    if boundary_noise_std > 0:
        high = sigma >= tau
        x_tau = torch.where(high, x_tau, x_tau + torch.randn_like(x_tau) * boundary_noise_std)

    if path_type == "cosine":
        return _cosine(sigma, tau, noise, x_tau, x_final)
    if path_type == "cubic_hermite":
        return _cubic_hermite(sigma, tau, noise, x_tau, x_final)
    if path_type == "smooth_blend":
        return _smooth_blend(sigma, tau, noise, x_tau, x_final, smooth_blend_delta)
    if path_type == "quadratic_bezier":
        return _quadratic_bezier(sigma, tau, noise, x_tau, x_final)
    if path_type == "target_linear":
        return _target_linear(sigma, tau, noise, x_tau, x_final)
    raise ValueError(f"Unknown COS path type: {path_type!r}")


def _validate_nstep_inputs(path_type: PathType, taus: list[float], waypoints: list[Tensor]) -> None:
    """Validate COS chain structure before dispatching to a path implementation."""
    if not waypoints:
        raise ValueError(f"COS path {path_type!r} requires at least one waypoint")

    expected_taus = len(waypoints) - 1
    if len(taus) != expected_taus:
        raise ValueError(
            "COS path expects len(taus) == len(waypoints) - 1, "
            f"got {len(taus)} taus for {len(waypoints)} waypoints "
            f"(path_type={path_type!r}). "
            f"For {len(waypoints)} videos/waypoints, use exactly {expected_taus} tau boundaries."
        )

    if any(not 0.0 < tau < 1.0 for tau in taus):
        raise ValueError(f"COS taus must lie strictly inside (0, 1), got {taus}")

    if any(taus[i] <= taus[i + 1] for i in range(len(taus) - 1)):
        raise ValueError(f"COS taus must be strictly descending, got {taus}")


# ======================================================================
# N-step path implementations
# ======================================================================


def _linear_n(
    sigma: Tensor,
    taus: list[float],
    noise: Tensor,
    waypoints: list[Tensor],
    boundary_noise_std: float = 0.0,
) -> tuple[Tensor, Tensor]:
    """N-step piecewise linear passthrough.  C0 at every boundary."""
    K = len(taus)
    N = K + 1  # number of segments
    boundaries = [1.0] + taus + [0.0]

    # Build anchor list: [noise, v_0, ..., v_{N-1}]
    anchors: list[Tensor] = [noise] + list(waypoints)

    # Boundary noise: perturb intermediate waypoints for samples below them.
    if boundary_noise_std > 0:
        for k in range(K):  # waypoints[k] = anchors[k+1], skip final
            tau_k = boundaries[k + 1]
            below = sigma < tau_k
            noisy = anchors[k + 1] + torch.randn_like(anchors[k + 1]) * boundary_noise_std
            anchors[k + 1] = torch.where(below, noisy, anchors[k + 1])

    x_t = torch.zeros_like(noise)
    target = torch.zeros_like(noise)

    for i in range(N):
        hi = boundaries[i]
        lo = boundaries[i + 1]
        seg_len = hi - lo

        mask = sigma >= lo if i == 0 else (sigma >= lo) & (sigma < hi)

        s = (sigma - lo) / seg_len
        x_t_seg = s * anchors[i] + (1.0 - s) * anchors[i + 1]
        tgt_seg = (anchors[i] - anchors[i + 1]) / seg_len

        x_t = torch.where(mask, x_t_seg, x_t)
        target = torch.where(mask, tgt_seg, target)

    return x_t, target


def _target_cosine_n(
    sigma: Tensor,
    taus: list[float],
    noise: Tensor,
    waypoints: list[Tensor],
    boundary_noise_std: float = 0.0,
) -> tuple[Tensor, Tensor]:
    r"""N-step smooth cosine target blend.  C1 everywhere.

    For each tau_j, a smooth transition function α_j(σ) goes from 0 to 1:

    * Below the transition window (σ ≤ lo_j):  α_j = 0
    * Lower half  [lo_j, τ_j]:   α_j = ¼(1 − cos(π·t))    (0 → ½)
    * Upper half  [τ_j, hi_j]:   α_j = ½ + ¼(1 − cos(π·t)) (½ → 1)
    * Above the transition window (σ ≥ hi_j):  α_j = 1

    where lo_j = τ_{j+1} (or 0) and hi_j = τ_{j-1} (or 1).

    Weights: w_0 = α_0,  w_k = α_k − α_{k-1},  w_K = 1 − α_{K-1}.

    Effective target:  x_eff = Σ w_k · v_k
    Noisy sample:      x_t = σ · noise + (1−σ) · x_eff
    Velocity target:   dx/dσ = noise − x_eff + (1−σ) · dx_eff/dσ
    """
    K = len(taus)
    if K == 0:
        x_eff = waypoints[0]
        x_t = sigma * noise + (1.0 - sigma) * x_eff
        target = noise - x_eff
        return x_t, target

    pi = math.pi
    # Extended boundaries for transition windows
    boundaries = [1.0] + taus + [0.0]

    # Apply boundary noise to intermediates
    if boundary_noise_std > 0:
        waypoints = list(waypoints)  # shallow copy
        for k in range(K):
            tau_k = boundaries[k + 1]
            below = sigma < tau_k
            noisy = waypoints[k] + torch.randn_like(waypoints[k]) * boundary_noise_std
            waypoints[k] = torch.where(below, noisy, waypoints[k])

    # Compute α_j and dα_j/dσ for each transition
    alphas: list[Tensor] = []
    dalphas: list[Tensor] = []

    for j in range(K):
        tau_j = taus[j]
        lo = boundaries[j + 2]  # next lower tau, or 0
        hi = boundaries[j]  # next higher tau, or 1

        lo_len = tau_j - lo
        hi_len = hi - tau_j

        t_lower = (sigma - lo) / lo_len
        a_lo = 0.25 * (1.0 - torch.cos(pi * t_lower))
        da_lo = 0.25 * pi / lo_len * torch.sin(pi * t_lower)

        t_upper = (sigma - tau_j) / hi_len
        a_hi = 0.5 + 0.25 * (1.0 - torch.cos(pi * t_upper))
        da_hi = 0.25 * pi / hi_len * torch.sin(pi * t_upper)

        zero = torch.zeros_like(sigma)
        one = torch.ones_like(sigma)

        in_lower = (sigma > lo) & (sigma <= tau_j)
        in_upper = (sigma > tau_j) & (sigma < hi)
        above = sigma >= hi

        alpha_j = torch.where(in_lower, a_lo, torch.where(in_upper, a_hi, torch.where(above, one, zero)))
        dalpha_j = torch.where(in_lower, da_lo, torch.where(in_upper, da_hi, zero))

        alphas.append(alpha_j)
        dalphas.append(dalpha_j)

    # Compute weights: w_0 = α_0, w_k = α_k − α_{k-1}, w_K = 1 − α_{K-1}
    x_eff = torch.zeros_like(noise)
    dx_eff = torch.zeros_like(noise)

    for k in range(K + 1):
        if k == 0:
            w = alphas[0]
            dw = dalphas[0]
        elif k == K:
            w = 1.0 - alphas[K - 1]
            dw = -dalphas[K - 1]
        else:
            w = alphas[k] - alphas[k - 1]
            dw = dalphas[k] - dalphas[k - 1]
        x_eff = x_eff + w * waypoints[k]
        dx_eff = dx_eff + dw * waypoints[k]

    x_t = sigma * noise + (1.0 - sigma) * x_eff
    target = noise - x_eff + (1.0 - sigma) * dx_eff

    return x_t, target


def _sigmoid_ease01(u: Tensor, steepness: float = 10.0) -> tuple[Tensor, Tensor]:
    """Normalized sigmoid easing on [0, 1] with zero endpoint slope."""
    u = u.clamp(0.0, 1.0)
    s = u * u * (3.0 - 2.0 * u)
    ds_du = 6.0 * u * (1.0 - u)

    lo = torch.sigmoid(torch.as_tensor(-0.5 * steepness, dtype=u.dtype, device=u.device))
    hi = torch.sigmoid(torch.as_tensor(0.5 * steepness, dtype=u.dtype, device=u.device))
    denom = hi - lo

    z = torch.sigmoid(steepness * (s - 0.5))
    y = (z - lo) / denom
    dy_du = (steepness * z * (1.0 - z) * ds_du) / denom
    return y, dy_du


def _target_sigmoid_n(
    sigma: Tensor,
    taus: list[float],
    noise: Tensor,
    waypoints: list[Tensor],
    boundary_noise_std: float = 0.0,
    sigmoid_steepness: float = 10.0,
) -> tuple[Tensor, Tensor]:
    r"""N-step smooth sigmoid target blend.  C1 everywhere.

    This mirrors ``target_cosine`` but replaces each half-cosine transition with
    a normalized sigmoid easing curve.  Each transition keeps exact anchor
    values: α_j(lo_j)=0, α_j(τ_j)=0.5, α_j(hi_j)=1.
    """
    K = len(taus)
    if K == 0:
        x_eff = waypoints[0]
        x_t = sigma * noise + (1.0 - sigma) * x_eff
        target = noise - x_eff
        return x_t, target

    boundaries = [1.0] + taus + [0.0]

    if boundary_noise_std > 0:
        waypoints = list(waypoints)
        for k in range(K):
            tau_k = boundaries[k + 1]
            below = sigma < tau_k
            noisy = waypoints[k] + torch.randn_like(waypoints[k]) * boundary_noise_std
            waypoints[k] = torch.where(below, noisy, waypoints[k])

    alphas: list[Tensor] = []
    dalphas: list[Tensor] = []

    for j in range(K):
        tau_j = taus[j]
        lo = boundaries[j + 2]
        hi = boundaries[j]

        lo_len = tau_j - lo
        hi_len = hi - tau_j

        u_lower = (sigma - lo) / lo_len
        y_lo, dy_lo = _sigmoid_ease01(u_lower, sigmoid_steepness)
        a_lo = 0.5 * y_lo
        da_lo = 0.5 * dy_lo / lo_len

        u_upper = (sigma - tau_j) / hi_len
        y_hi, dy_hi = _sigmoid_ease01(u_upper, sigmoid_steepness)
        a_hi = 0.5 + 0.5 * y_hi
        da_hi = 0.5 * dy_hi / hi_len

        zero = torch.zeros_like(sigma)
        one = torch.ones_like(sigma)

        in_lower = (sigma > lo) & (sigma <= tau_j)
        in_upper = (sigma > tau_j) & (sigma < hi)
        above = sigma >= hi

        alpha_j = torch.where(in_lower, a_lo, torch.where(in_upper, a_hi, torch.where(above, one, zero)))
        dalpha_j = torch.where(in_lower, da_lo, torch.where(in_upper, da_hi, zero))

        alphas.append(alpha_j)
        dalphas.append(dalpha_j)

    x_eff = torch.zeros_like(noise)
    dx_eff = torch.zeros_like(noise)

    for k in range(K + 1):
        if k == 0:
            w = alphas[0]
            dw = dalphas[0]
        elif k == K:
            w = 1.0 - alphas[K - 1]
            dw = -dalphas[K - 1]
        else:
            w = alphas[k] - alphas[k - 1]
            dw = dalphas[k] - dalphas[k - 1]
        x_eff = x_eff + w * waypoints[k]
        dx_eff = dx_eff + dw * waypoints[k]

    x_t = sigma * noise + (1.0 - sigma) * x_eff
    target = noise - x_eff + (1.0 - sigma) * dx_eff

    return x_t, target


# ======================================================================
# Legacy 2-step path implementations (used via dispatch above)
# ======================================================================


def _cosine(
    sigma: Tensor,
    tau: float,
    noise: Tensor,
    x_tau: Tensor,
    x_final: Tensor,
) -> tuple[Tensor, Tensor]:
    """Cosine reparameterisation per segment.  C1 at boundary (velocity -> 0)."""
    high = sigma >= tau
    pi = math.pi

    t_h = (sigma - tau) / (1.0 - tau)
    s_h = 0.5 * (1.0 - torch.cos(pi * t_h))
    ds_dsigma_h = 0.5 * pi * torch.sin(pi * t_h) / (1.0 - tau)
    x_t_h = s_h * noise + (1.0 - s_h) * x_tau
    tgt_h = ds_dsigma_h * (noise - x_tau)

    t_l = sigma / tau
    s_l = 0.5 * (1.0 - torch.cos(pi * t_l))
    ds_dsigma_l = 0.5 * pi * torch.sin(pi * t_l) / tau
    x_t_l = s_l * x_tau + (1.0 - s_l) * x_final
    tgt_l = ds_dsigma_l * (x_tau - x_final)

    return torch.where(high, x_t_h, x_t_l), torch.where(high, tgt_h, tgt_l)


def _cubic_hermite(
    sigma: Tensor,
    tau: float,
    noise: Tensor,
    x_tau: Tensor,
    x_final: Tensor,
) -> tuple[Tensor, Tensor]:
    """Catmull-Rom cubic Hermite spline.  C1 at boundary."""
    high = sigma >= tau
    v_tau = noise - x_final

    t = (sigma - tau) / (1.0 - tau)
    t2 = t * t
    t3 = t2 * t
    h00 = 2.0 * t3 - 3.0 * t2 + 1.0
    h10 = t3 - 2.0 * t2 + t
    h01 = -2.0 * t3 + 3.0 * t2
    h11 = t3 - t2
    m0_h = (1.0 - tau) * v_tau
    m1_h = noise - x_tau
    x_t_h = h00 * x_tau + h10 * m0_h + h01 * noise + h11 * m1_h
    dh00 = 6.0 * t2 - 6.0 * t
    dh10 = 3.0 * t2 - 4.0 * t + 1.0
    dh01 = -6.0 * t2 + 6.0 * t
    dh11 = 3.0 * t2 - 2.0 * t
    dx_dt_h = dh00 * x_tau + dh10 * m0_h + dh01 * noise + dh11 * m1_h
    tgt_h = dx_dt_h / (1.0 - tau)

    t = sigma / tau
    t2 = t * t
    t3 = t2 * t
    h00 = 2.0 * t3 - 3.0 * t2 + 1.0
    h10 = t3 - 2.0 * t2 + t
    h01 = -2.0 * t3 + 3.0 * t2
    h11 = t3 - t2
    m0_l = x_tau - x_final
    m1_l = tau * v_tau
    x_t_l = h00 * x_final + h10 * m0_l + h01 * x_tau + h11 * m1_l
    dh00 = 6.0 * t2 - 6.0 * t
    dh10 = 3.0 * t2 - 4.0 * t + 1.0
    dh01 = -6.0 * t2 + 6.0 * t
    dh11 = 3.0 * t2 - 2.0 * t
    dx_dt_l = dh00 * x_final + dh10 * m0_l + dh01 * x_tau + dh11 * m1_l
    tgt_l = dx_dt_l / tau

    return torch.where(high, x_t_h, x_t_l), torch.where(high, tgt_h, tgt_l)


def _smooth_blend(
    sigma: Tensor,
    tau: float,
    noise: Tensor,
    x_tau: Tensor,
    x_final: Tensor,
    delta: float,
) -> tuple[Tensor, Tensor]:
    """Linear with smoothstep blending in ``[tau - delta, tau + delta]``."""
    s_h = (sigma - tau) / (1.0 - tau)
    x_t_h = s_h * noise + (1.0 - s_h) * x_tau
    tgt_h = (noise - x_tau) / (1.0 - tau)

    s_l = sigma / tau
    x_t_l = s_l * x_tau + (1.0 - s_l) * x_final
    tgt_l = (x_tau - x_final) / tau

    lo = tau - delta
    hi = tau + delta
    u = ((sigma - lo) / (hi - lo)).clamp(0.0, 1.0)
    alpha = 3.0 * u * u - 2.0 * u * u * u
    dalpha_dsigma = 6.0 * u * (1.0 - u) / (hi - lo)

    x_t = (1.0 - alpha) * x_t_l + alpha * x_t_h
    target = (1.0 - alpha) * tgt_l + alpha * tgt_h + dalpha_dsigma * (x_t_h - x_t_l)
    return x_t, target


def _quadratic_bezier(
    sigma: Tensor,
    tau: float,
    noise: Tensor,
    x_tau: Tensor,
    x_final: Tensor,
) -> tuple[Tensor, Tensor]:
    """Quadratic polynomial through three anchor points."""
    diff_tau = x_tau - x_final
    diff_1 = noise - x_final
    denom = tau * (tau - 1.0)
    a = (diff_tau - tau * diff_1) / denom
    b = diff_1 - a

    sigma2 = sigma * sigma
    x_t = a * sigma2 + b * sigma + x_final
    target = 2.0 * a * sigma + b
    return x_t, target


def _target_linear(
    sigma: Tensor,
    tau: float,
    noise: Tensor,
    x_tau: Tensor,
    x_final: Tensor,
) -> tuple[Tensor, Tensor]:
    """No-passthrough, hard target switch at tau.  C0 in velocity."""
    high = sigma >= tau

    x_t_h = sigma * noise + (1.0 - sigma) * x_tau
    tgt_h = noise - x_tau

    x_theta = tau * noise + (1.0 - tau) * x_tau
    s_l = sigma / tau
    x_t_l = s_l * x_theta + (1.0 - s_l) * x_final
    tgt_l = (x_theta - x_final) / tau

    return torch.where(high, x_t_h, x_t_l), torch.where(high, tgt_h, tgt_l)
