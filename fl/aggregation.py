from __future__ import annotations

from collections import OrderedDict


def fedavg_aggregate(
    global_state_dict: OrderedDict,
    updates: dict,
    weights: dict = None,
) -> OrderedDict:
    """
    Standard FedAvg aggregation of client model deltas.

        theta(t+1) = theta(t) + sum_i (N_i / N) * U_i(theta)

    The free-rider constructions evaluated by SVRFL assume this vanilla,
    sample-size-weighted update without server momentum or a server LR.
    """
    client_ids = list(updates.keys())
    n = len(client_ids)

    if n == 0:
        return OrderedDict({k: v.clone() for k, v in global_state_dict.items()})

    if weights is None:
        weights = {cid: 1.0 / n for cid in client_ids}
    missing = set(client_ids) - set(weights)
    if missing:
        raise ValueError(f"missing aggregation weights for clients: {sorted(missing)}")
    weight_sum = sum(float(weights[cid]) for cid in client_ids)
    if weight_sum <= 0:
        raise ValueError("aggregation weights must have a positive sum")
    normalized = {cid: float(weights[cid]) / weight_sum for cid in client_ids}

    new_state_dict = OrderedDict()

    for key in global_state_dict:
        device = global_state_dict[key].device

        # Weighted average of client deltas
        avg_delta = sum(
            normalized[cid] * updates[cid][key].to(device)
            for cid in client_ids
        )
        if not avg_delta.is_floating_point() or not avg_delta.isfinite().all():
            raise FloatingPointError(f"non-finite aggregated delta for {key}")
        new_state_dict[key] = global_state_dict[key] + avg_delta

    return new_state_dict
