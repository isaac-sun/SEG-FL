from collections import OrderedDict
from typing import List

import math
import torch


def sdfr_attack(
    global_state_dict: OrderedDict,
    global_history: List[OrderedDict],
    eps: float = 1e-10,
) -> OrderedDict:
    """
    Scaled-delta free-rider attack, Zhu et al. Eq. (9).

        U_f(theta) = ||theta(t)-theta(t-1)|| / ||theta(t-1)-theta(t-2)||
                     * (theta(t)-theta(t-1))

    The paper's construction requires three consecutive global states.
    Callers must provide two prior global states before invoking this function;
    the attack scheduler may obtain them from DFR warm-up rounds.
    """
    if len(global_history) < 2:
        raise ValueError("SDFR requires theta(t), theta(t-1), and theta(t-2)")

    w_t = global_state_dict
    w_t1 = global_history[-1]
    w_t2 = global_history[-2]

    delta_t_norm_sq = 0.0
    delta_prev_norm_sq = 0.0
    delta_t = OrderedDict()
    update = OrderedDict()

    for key in w_t:
        dt = w_t[key].float() - w_t1[key].float()
        dp = w_t1[key].float() - w_t2[key].float()
        delta_t[key] = w_t[key] - w_t1[key]
        delta_t_norm_sq += dt.pow(2).sum().item()
        delta_prev_norm_sq += dp.pow(2).sum().item()

    current_norm = math.sqrt(delta_t_norm_sq)
    previous_norm = math.sqrt(delta_prev_norm_sq)
    if previous_norm <= eps:
        raise FloatingPointError(
            "SDFR denominator is zero; two preceding global states are identical"
        )
    scale = current_norm / previous_norm
    if not math.isfinite(scale):
        raise FloatingPointError(f"non-finite SDFR scale: {scale}")

    for key in delta_t:
        update[key] = delta_t[key] * scale

    return update
