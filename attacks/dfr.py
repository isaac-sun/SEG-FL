from __future__ import annotations

import math
from collections import OrderedDict
from typing import List, Mapping, Optional

import torch


def estimate_dfr_sigma_from_updates(
    updates: Mapping[int, Mapping[str, torch.Tensor]],
) -> Optional[float]:
    """Estimate a non-zero DFR scale from available honest updates.

    The paper estimates ``sigma`` from the first global increment.  Before
    that increment exists, the first-round honest updates provide the only
    observable task-scale signal.  We estimate the population standard
    deviation across floating-point coordinates for each update and average
    those variances, avoiding a model-sized concatenation.
    """
    variances: list[float] = []
    for update in updates.values():
        total_count = 0
        total_sum = 0.0
        total_sq_sum = 0.0
        for value in update.values():
            if not torch.is_floating_point(value):
                continue
            flat = value.detach().float()
            count = int(flat.numel())
            if count == 0:
                continue
            total_count += count
            total_sum += float(flat.sum().item())
            total_sq_sum += float(flat.square().sum().item())
        if total_count:
            mean = total_sum / total_count
            variance = max(total_sq_sum / total_count - mean * mean, 0.0)
            variances.append(variance)

    if not variances:
        return None
    sigma = float(sum(variances) / len(variances)) ** 0.5
    if not math.isfinite(sigma):
        raise FloatingPointError("non-finite DFR bootstrap sigma")
    return sigma


def estimate_dfr_sigma(
    global_state_dict: OrderedDict,
    global_history: List[OrderedDict],
) -> Optional[float]:
    """
    Estimate DFR sigma from the first observed global-model increment.

    Fraboni et al. fit a zero-centred univariate Gaussian to
    ``theta(1) - theta(0)`` and use its standard deviation as sigma.
    ``global_history`` contains states before the current global state.
    """
    if len(global_history) < 1:
        return None

    theta_0 = global_history[0]
    theta_1 = global_history[1] if len(global_history) >= 2 else global_state_dict

    delta_flat = torch.cat([
        (theta_1[k].float() - theta_0[k].float()).reshape(-1).cpu()
        for k in theta_0
    ])
    # The fitted Gaussian uses the population standard deviation.
    return delta_flat.std(unbiased=False).item()


def dfr_attack(
    global_state_dict: OrderedDict,
    sigma: float,
    round_num: int = 1,
    gamma: float = 1.0,
    generator: torch.Generator | None = None,
) -> OrderedDict:
    """
    Disguised free-rider update from Fraboni et al.

        delta_f^t = phi(t) * epsilon_t
        phi(t) = sigma * t^(-gamma),  epsilon_t ~ N(0, I)

    The returned object is a model delta, equivalent to uploading
    ``theta^t + delta_f^t``. All tensors are kept on the canonical state
    device (CPU in this project).
    """
    if sigma < 0:
        raise ValueError("sigma must be non-negative")
    if gamma < 0:
        raise ValueError("gamma must be non-negative")

    phi_t = sigma * (max(round_num, 1) ** (-gamma))
    update = OrderedDict()
    for key, value in global_state_dict.items():
        noise = torch.randn(
            value.shape,
            dtype=value.dtype,
            device=value.device,
            generator=generator,
        )
        update[key] = noise * phi_t
    return update


def plain_free_rider_update(global_state_dict: OrderedDict) -> OrderedDict:
    """Return the zero delta used while DFR sigma is not observable yet."""
    return OrderedDict((key, torch.zeros_like(value))
                       for key, value in global_state_dict.items())
