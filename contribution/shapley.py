"""Per-group Monte Carlo Shapley values for federated segmentation (CSDD).

Same permutation sampler and coalition construction as ECL-FL / 20NEWS-FL;
the value function is the drop of the pixel-mean cross-entropy of group g on
the server reference set:

    v_g(S) = F_g(w_t) - F_g(w_{t,S}),   F_g(w) = sum_{p in P_g} l_p(w) / |P_g|

Coalitions use equal weights by default, as in the paper (Eq. 14):

    w_{t,S} = w_t + 1/|S| * sum_{j in S} delta_j

One inference of a coalition model gives the cross-entropy sum of every
(reference image, pixel class) cell.  Every grouping is a map from cells to
groups, so the group losses of all groupings come from that one pass; the
Monte Carlo marginals are taken on the concatenated loss vector and split per
grouping afterwards.  All groupings therefore share the same permutations.
"""

from __future__ import annotations

from collections import OrderedDict

import numpy as np
import torch
import torch.nn.functional as F


class ReferenceEvaluator:
    """Reference set + groupings; turns a model state into group losses."""

    def __init__(self, image_set, groupings: dict, num_classes: int, device: str,
                 amp: bool = True):
        self.image_set = image_set
        self.groupings = groupings
        self.names = list(groupings)
        self.num_classes = num_classes
        self.device = device
        self.amp = amp
        self.pixel_counts = image_set.pixel_counts()          # [I, C]
        cells = len(image_set) * num_classes
        flat_counts = torch.from_numpy(self.pixel_counts.reshape(-1)).to(device)
        self.slices = {}
        ids, offset = [], 0
        for name in self.names:
            g = groupings[name]
            cell = torch.from_numpy(g.cell_group.reshape(-1)).to(device)
            if cell.numel() != cells:
                raise ValueError(f"grouping {name} has the wrong number of cells")
            ids.append(torch.where(cell >= 0, cell + offset, torch.full_like(cell, -1)))
            self.slices[name] = slice(offset, offset + g.num_groups)
            offset += g.num_groups
        self.total_groups = offset
        self.cell_ids = torch.stack(ids)                         # [G, cells]
        self.group_pixels = self._group_sum(flat_counts)          # [total_groups]
        if (self.group_pixels <= 0).any():
            empty = [(n, int(i) - self.slices[n].start) for n in self.names
                     for i in range(self.slices[n].start, self.slices[n].stop)
                     if self.group_pixels[i] <= 0]
            raise ValueError(f"empty groups in the reference set: {empty}")

    def _group_sum(self, flat_values: torch.Tensor) -> torch.Tensor:
        out = torch.zeros(self.total_groups, dtype=torch.float64, device=self.device)
        for ids in self.cell_ids:
            keep = ids >= 0
            out.index_add_(0, ids[keep], flat_values[keep].double())
        return out

    def group_weights(self, name: str) -> np.ndarray:
        """Pixel share of each group (weights for the overall SV)."""
        pixels = self.group_pixels[self.slices[name]].cpu().numpy()
        return pixels / pixels.sum()

    @torch.no_grad()
    def evaluate(self, model, state) -> dict:
        """Loss vector over all groupings + confusion + per-class CE."""
        model.load_fl_state(state)
        model.to(self.device)
        model.eval()
        return evaluate_image_set(model, self.image_set, self.num_classes, self.device,
                                  self.amp, evaluator=self)

    def split(self, vector: np.ndarray) -> dict:
        return {name: vector[self.slices[name]] for name in self.names}


@torch.no_grad()
def evaluate_image_set(model, image_set, num_classes: int, device: str, amp: bool,
                       evaluator: ReferenceEvaluator | None = None) -> dict:
    """Per-cell CE sums, confusion matrix, per-class mean CE.

    ``model`` must already hold the state to evaluate and be in eval mode.
    """
    use_amp = amp and device == "cuda"
    num_images = len(image_set)
    ce_cells = torch.zeros(num_images * num_classes, dtype=torch.float64, device=device)
    confusion = torch.zeros(num_classes * num_classes, dtype=torch.float64, device=device)
    for start, x, y in image_set.batches():
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
            logits = model(x)
        logits = logits.float()
        loss = F.cross_entropy(logits, y, ignore_index=255, reduction="none")
        if not torch.isfinite(loss).all():
            raise FloatingPointError("non-finite loss during evaluation")
        valid = y != 255
        idx = torch.arange(start, start + y.shape[0], device=y.device).view(-1, 1, 1)
        cells = (idx * num_classes + y)[valid]
        ce_cells.index_add_(0, cells, loss[valid].double())
        pred = logits.argmax(dim=1)
        confusion += torch.bincount(y[valid] * num_classes + pred[valid],
                                    minlength=num_classes * num_classes).double()
    class_ce_sum = ce_cells.view(num_images, num_classes).sum(0)
    confusion = confusion.view(num_classes, num_classes)
    class_pixels = confusion.sum(1)
    result = {
        "confusion": confusion.cpu().numpy(),
        "class_ce": (class_ce_sum / class_pixels.clamp_min(1)).cpu().numpy(),
    }
    if evaluator is not None:
        result["group_loss"] = (evaluator._group_sum(ce_cells)
                                / evaluator.group_pixels).cpu().numpy()
    return result


def segmentation_metrics(confusion: np.ndarray, class_ce: np.ndarray) -> dict:
    """mIoU (over classes present in the labels), pixel accuracy, mean CE."""
    confusion = np.asarray(confusion, dtype=np.float64)
    tp = np.diag(confusion)
    gt = confusion.sum(1)
    pred = confusion.sum(0)
    union = gt + pred - tp
    present = gt > 0
    iou = np.where(union > 0, tp / np.maximum(union, 1), np.nan)
    acc = np.where(gt > 0, tp / np.maximum(gt, 1), np.nan)
    total = gt.sum()
    return {
        "miou": float(np.nanmean(iou[present])) if present.any() else float("nan"),
        "pixel_acc": float(tp.sum() / total) if total else float("nan"),
        "loss": float(np.sum(class_ce * gt) / total) if total else float("nan"),
        "class_iou": iou,
        "class_acc": acc,
        "class_ce": np.asarray(class_ce, dtype=np.float64),
    }


def _coalition_state(global_state, delta_sum, total_weight):
    state = OrderedDict()
    for key, value in global_state.items():
        state[key] = value + delta_sum[key] / total_weight
        if not torch.isfinite(state[key]).all():
            raise FloatingPointError(f"non-finite coalition parameter: {key}")
    return state


def estimate_round_shapley(
    model,
    updates: dict,
    global_state: OrderedDict,
    evaluator: ReferenceEvaluator,
    *,
    num_mc_samples: int = 20,
    sample_counts: dict | None = None,
    cache_coalitions: bool = True,
    rng: np.random.Generator | None = None,
):
    """Return ``(sv, evaluations)``.

    ``sv``: ``grouping -> {client_id: ndarray[num_groups]}``.
    ``evaluations``: coalition tuple -> evaluation dict (``()`` = global model,
    ``(cid,)`` = the client's submitted model under equal weights).
    ``sample_counts=None`` gives equal-weight coalitions (paper Eq. 14).
    """
    device = evaluator.device
    client_ids = list(updates)
    weights = sample_counts or {cid: 1 for cid in client_ids}
    eval_global = OrderedDict((k, v.to(device)) for k, v in global_state.items())
    eval_updates = {cid: OrderedDict((k, v.to(device)) for k, v in u.items())
                    for cid, u in updates.items()}
    evaluations = {(): evaluator.evaluate(model, eval_global)}
    base = evaluations[()]["group_loss"]
    sums = {cid: np.zeros(evaluator.total_groups, dtype=np.float64) for cid in client_ids}
    for _ in range(num_mc_samples if client_ids else 0):
        order = (rng.permutation(client_ids).tolist() if rng is not None
                 else np.random.permutation(client_ids).tolist())
        coalition = []
        previous = base
        delta_sum = OrderedDict((k, torch.zeros_like(v)) for k, v in eval_global.items())
        total_weight = 0.0
        for cid in order:
            coalition.append(cid)
            weight = float(weights.get(cid, 1))
            total_weight += weight
            for key, acc in delta_sum.items():
                acc.add_(eval_updates[cid][key], alpha=weight)
            key = tuple(sorted(coalition))
            result = evaluations.get(key) if cache_coalitions else None
            if result is None:
                result = evaluator.evaluate(
                    model, _coalition_state(eval_global, delta_sum, total_weight))
                if cache_coalitions:
                    evaluations[key] = result
            current = result["group_loss"]
            sums[cid] += previous - current
            previous = current
    sv = {name: {} for name in evaluator.names}
    for cid in client_ids:
        parts = evaluator.split(sums[cid] / max(num_mc_samples, 1))
        for name in evaluator.names:
            sv[name][cid] = parts[name]
    return sv, evaluations


def per_group_to_overall(per_group_sv: dict, group_weights: np.ndarray) -> dict:
    return {cid: float(np.dot(group_weights, sv)) for cid, sv in per_group_sv.items()}


def compute_class_metrics(per_class_sv: dict, eps: float = 1e-10) -> dict:
    """Unchanged from 20NEWS-FL: A (positive sum), P, V, dominant class."""
    metrics = {}
    for cid, sv_arr in per_class_sv.items():
        positive = np.maximum(sv_arr, 0.0)
        a_i = float(np.sum(positive))
        p_i = float(np.max(positive) / (a_i + eps)) if a_i > 0 else 0.0
        metrics[cid] = {
            "class_sv_variance": float(np.var(sv_arr)),
            "positive_class_sv_sum": a_i,
            "concentration_ratio": p_i,
            "dominant_class": int(np.argmax(positive)),
        }
    return metrics
