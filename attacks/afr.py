from __future__ import annotations

from collections import OrderedDict
from typing import List, Tuple

import math
import torch

from attacks.sdfr import sdfr_attack


def estimate_decay_rate(
    update_norm_t: float,
    update_norm_1: float,
    round_num: int,
    eps: float = 1e-12,
) -> float:
    """
    Estimate the positive OU decay rate from Zhu et al. Algorithm 1.

    The paper estimates the rate from ``l(t)=|E(theta(t)-theta(t-1))|``
    and ``l(1)``. Its typeset pseudocode omits the minus sign even though
    Lemma 1 and the surrounding derivation require decay ``exp(-lambda*t)``.
    This implementation uses the mathematically consistent form
    ``-log(l(t)/l(1)) / (t-1)``.
    """
    if round_num <= 1 or update_norm_1 <= eps or update_norm_t <= eps:
        return 0.0
    ratio = max(update_norm_t / update_norm_1, eps)
    return max(-math.log(ratio) / (round_num - 1), 0.0)


def estimate_e_cos_beta(c: float, decay_rate: float, round_num: int) -> float:
    """Zhu et al. Lemma 1: E[cos beta] = C^2/(C^2+exp(2*lambda*t))."""
    if c <= 0:
        raise ValueError("AFR constant C must be positive")
    exponent = min(2.0 * max(decay_rate, 0.0) * max(round_num, 0), 700.0)
    c_sq = c * c
    return c_sq / (c_sq + math.exp(exponent))


def global_update_norm(
    global_state_dict: OrderedDict,
    previous_state_dict: OrderedDict,
) -> float:
    """Compute ||theta(t)-theta(t-1)|| without flattening the full model."""
    norm_sq = 0.0
    for key in global_state_dict:
        delta = global_state_dict[key].float() - previous_state_dict[key].float()
        norm_sq += delta.pow(2).sum().item()
    return math.sqrt(norm_sq)


def update_norm(update: OrderedDict) -> float:
    """Compute an update's L2 norm without flattening the model."""
    return math.sqrt(sum(
        value.float().pow(2).sum().item() for value in update.values()
    ))


def estimate_c_from_norms(
    local_update_norm: float,
    mean_update_norm: float,
    n_total: int,
    decay_rate: float,
    round_num: int,
    eps: float = 1e-8,
) -> float:
    """Estimate C by combining Zhu et al. Lemmas 1 and 2.

    Lemma 2 recovers E[cos beta] from ``||U_i|| / ||U_bar||`` using the
    honest-client calibration updates observed during DFR warm-up. Lemma 1
    then yields C.
    """
    if n_total < 2 or local_update_norm <= eps or mean_update_norm <= eps:
        raise ValueError("positive warm-up update norms are required to estimate C")
    norm_ratio_sq = (local_update_norm / mean_update_norm) ** 2
    n = float(n_total)
    cosine = ((n * n) / norm_ratio_sq - n) / (n * n - n)
    cosine = min(max(cosine, eps), 1.0 - eps)
    exponent = min(2.0 * max(decay_rate, 0.0) * max(round_num, 0), 700.0)
    c_sq = cosine * math.exp(exponent) / (1.0 - cosine)
    return math.sqrt(c_sq)


# ── AFR attack ─────────────────────────────────────────────────────────────

def afr_attack(
    global_state_dict: OrderedDict,
    global_history: List[OrderedDict],
    n_total: int = 10,
    e_cos_beta: float = 0.0,
    noisy_frac: float = 0.1,
    seed: int | None = None,
    eps: float = 1e-10,
) -> Tuple[OrderedDict, float]:
    """
    Advanced free-rider attack from Zhu et al. Algorithm 1.

    It augments the scaled-delta update with sparse Gaussian noise:

        |phi(t)| = sqrt(n^2 / (n + (n^2-n)E[cos beta]) - 1) * |U_f|

    Selected coordinates receive ``|phi(t)| * N(0, 1/d)``. Bernoulli
    selection is used tensor-by-tensor to avoid allocating a 66-million
    element permutation solely to select ``d`` coordinates. Its expected
    selected dimension is exactly ``D * noisy_frac``.
    """
    if n_total < 2:
        raise ValueError("AFR requires at least two clients")
    if not 0.0 < noisy_frac <= 1.0:
        raise ValueError("noisy_frac must be in (0, 1]")
    if not 0.0 <= e_cos_beta <= 1.0:
        raise ValueError("e_cos_beta must be in [0, 1]")

    base_update = sdfr_attack(global_state_dict, global_history, eps=eps)
    total_dim = sum(value.numel() for value in base_update.values())
    base_norm_sq = sum(
        value.float().pow(2).sum().item() for value in base_update.values()
    )
    base_norm = math.sqrt(base_norm_sq)

    n = n_total
    denom = n + (n * n - n) * e_cos_beta
    multiplier_sq = max((n * n) / (denom + eps) - 1.0, 0.0)
    phi_t = math.sqrt(multiplier_sq) * base_norm
    expected_noisy = max(int(total_dim * noisy_frac), 1)
    noise_std = phi_t / math.sqrt(expected_noisy)

    gen = torch.Generator()
    if seed is not None:
        gen.manual_seed(seed)
    else:
        gen.manual_seed(torch.randint(0, 2**31, (1,)).item())

    update = OrderedDict()
    for key, value in base_update.items():
        selection = torch.rand(value.shape, generator=gen) < noisy_frac
        noise = torch.randn(value.shape, dtype=value.dtype, generator=gen)
        noise.mul_(noise_std).mul_(selection)
        update[key] = value + noise

    return update, base_norm
