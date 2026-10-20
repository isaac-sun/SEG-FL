#!/usr/bin/env python3
"""Federated free-rider detection (CSDD) on VOC 2012 semantic segmentation.

One invocation runs one data partition for a list of seeds and conditions,
mirroring one partition of the ECL-FL / 20NEWS-FL formal matrix:

    baseline (no attack) + {DFR, SDFR, AFR} x {10%, 30%, 50%} malicious

For a fixed (partition, seed), every condition shares the same client
partition, attacker order (10% subset of 30% subset of 50%), participation
schedule and model initialization.  Every round computes the per-class
Shapley values of all pre-registered groupings (``data.voc.GROUPINGS``; called
"Shapley axes" in the outputs, as in ECL-FL) from the same coalition
evaluations.  The 20NEWS-format tables use the main grouping ``class21``.  Each finished condition+seed is written to
``<results-dir>/runs/<run>/`` and skipped on restart; an interrupted run
resumes from its last finished round.
"""

from __future__ import annotations

import argparse
import copy
import json
import logging
import os
import random
import sys
import time
from collections import OrderedDict
from dataclasses import asdict

import numpy as np
import pandas as pd
import torch

from attacks.afr import (
    afr_attack,
    estimate_c_from_norms,
    estimate_decay_rate,
    estimate_e_cos_beta,
    global_update_norm,
    update_norm,
)
from attacks.dfr import dfr_attack, estimate_dfr_sigma, estimate_dfr_sigma_from_updates
from attacks.sdfr import sdfr_attack
from config import Config
from contribution.shapley import (
    ReferenceEvaluator,
    compute_class_metrics,
    estimate_round_shapley,
    per_group_to_overall,
    segmentation_metrics,
)
from data.voc import (VOC_CLASSES, WHOLE_SET_AXES, ImageSet, build_groupings, center_crop,
                      load_voc, whole_set_grouping)
from fl.aggregation import fedavg_aggregate
from fl.client import FLClient
from fl.server import FLServer
from models.deeplab import create_model
from utils.partition import dirichlet_partition, iid_partition, partition_statistics
from utils.seed import set_seed

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger("seg-fl")

SCORE_COLUMNS = {
    # name: (column, lower_is_malicious)
    "V_mean": ("v_mean", True),          # paper Eq. 20, primary score
    "V_sum": ("v_sum", True),
    "P_mean": ("p_mean", True),
    "P_sum": ("p_sum", True),
    "SV_mean": ("sv_mean", True),
    "SV_sum": ("sv_sum", True),
}
RECORD_KEYS = ("round_details", "axis_records", "per_class_records",
               "axis_group_shapley", "per_class_client_metrics", "round_curves")


# ── naming (same conventions as 20NEWS-FL / ECL-FL) ────────────────────────

def alpha_tag(alpha: float) -> str:
    alpha = float(alpha)
    if alpha.is_integer():
        return f"{int(alpha)}p0"
    return f"{alpha:g}".replace(".", "p")


def partition_id(method: str, alpha: float | None) -> str:
    return "iid" if method == "iid" else f"dirichlet_alpha{alpha_tag(alpha)}"


def experiment_name(attack: str, ratio: float, part_id: str) -> str:
    if attack == "none":
        return f"baseline_no_attack_partition_{part_id}"
    return f"attack_{attack}_mal{int(round(ratio * 100))}_partition_{part_id}"


def build_participation_schedule(num_clients, ratio, num_rounds, seed):
    num_selected = max(1, int(round(num_clients * ratio)))
    rng = np.random.default_rng(seed)
    return [sorted(rng.choice(num_clients, num_selected, replace=False).tolist())
            for _ in range(num_rounds)]


# ── attacks (logic copied from ECL-FL / 20NEWS-FL ``_apply_attack``) ────────
# Attacks act on the trainable parameters only.  A free rider has no data,
# so its BatchNorm statistics are the global ones (zero delta).

def params_view(state: OrderedDict, keys) -> OrderedDict:
    return OrderedDict((k, state[k]) for k in keys)


def with_zero_buffers(param_update: OrderedDict, global_state: OrderedDict) -> OrderedDict:
    return OrderedDict((k, param_update[k] if k in param_update else torch.zeros_like(v))
                       for k, v in global_state.items())


def apply_attack(attack_type, global_sd, global_history, config, round_index,
                 dfr_sigma_est, client_id, afr_c_est, param_keys):
    gp = params_view(global_sd, param_keys)
    hp = [params_view(h, param_keys) for h in global_history]
    effective = attack_type
    if attack_type in {"sdfr", "afr"} and round_index < config.attack_warmup_rounds:
        effective = "dfr"
    if effective == "dfr":
        sigma = config.dfr_sigma_override if config.dfr_sigma_override is not None else dfr_sigma_est
        if sigma is None:
            raise RuntimeError("DFR requires a bootstrap sigma")
        update = dfr_attack(gp, sigma=sigma, round_num=round_index + 1, gamma=config.dfr_gamma)
    elif effective == "sdfr":
        update = sdfr_attack(gp, hp)
    elif attack_type == "afr":
        afr_c = config.afr_c_override if config.afr_c_override is not None else afr_c_est
        if afr_c is None:
            raise RuntimeError("AFR C was not estimated during warm-up")
        first_norm = global_update_norm(hp[1], hp[0])
        current_norm = global_update_norm(gp, hp[-1])
        decay_rate = estimate_decay_rate(current_norm, first_norm, round_num=round_index)
        e_cos_beta = estimate_e_cos_beta(afr_c, decay_rate, round_num=round_index)
        update, _ = afr_attack(gp, hp, n_total=config.num_clients, e_cos_beta=e_cos_beta,
                               noisy_frac=config.afr_noisy_frac,
                               seed=config.seed + round_index * config.num_clients + client_id)
    else:
        raise ValueError(f"unknown attack type: {attack_type}")
    return with_zero_buffers(update, global_sd)


def assert_finite(update, label):
    for key, value in update.items():
        if not torch.isfinite(value).all():
            raise FloatingPointError(f"{label} contains non-finite values in {key}")


# ── detection scores ─────────────────────────────────────────────────────

def pairwise_auc(benign, malicious, lower_is_malicious=True) -> float:
    """Pr(benign ranked safer than malicious); ties count one half."""
    b = np.asarray(benign, dtype=float)[:, None]
    m = np.asarray(malicious, dtype=float)[None, :]
    if b.size == 0 or m.size == 0:
        return float("nan")
    wins = (b > m) if lower_is_malicious else (b < m)
    return float((wins.sum() + 0.5 * (b == m).sum()) / (b.size * m.size))


def client_totals(round_details: list[dict], num_clients: int,
                  malicious_ids: set) -> list[dict]:
    frame = pd.DataFrame(round_details)
    rows = []
    for cid in range(num_clients):
        part = frame[frame["client_id"] == cid] if len(frame) else frame
        n = int(len(part))
        v_sum = float(part["class_sv_variance"].sum()) if n else 0.0
        p_sum = float(part["positive_class_sv_sum"].sum()) if n else 0.0
        sv_sum = float(part["round_shapley_value"].sum()) if n else 0.0
        rows.append({
            "client_id": cid,
            "is_malicious": cid in malicious_ids,
            "participations": n,
            "v_sum": v_sum, "v_mean": v_sum / n if n else 0.0,
            "p_sum": p_sum, "p_mean": p_sum / n if n else 0.0,
            "sv_sum": sv_sum, "sv_mean": sv_sum / n if n else 0.0,
        })
    return rows


def auc_record(totals: list[dict], suffix: str = "") -> dict:
    frame = pd.DataFrame(totals)
    malicious = frame["is_malicious"].astype(bool)
    return {f"auc_{name}{suffix}": pairwise_auc(frame.loc[~malicious, column],
                                                frame.loc[malicious, column], lower)
            for name, (column, lower) in SCORE_COLUMNS.items()}


def axis_client_totals(axis_records: list[dict], num_clients: int, malicious_ids: set):
    """``client_totals`` for every grouping ("Shapley axis", as in ECL-FL) with
    an ``axis`` column, plus the AUCs per axis (``auc_<score>__<axis>``)."""
    frame = pd.DataFrame(axis_records)
    rows, aucs = [], {}
    if not len(frame):
        return rows, aucs
    for name, part in frame.groupby("axis", sort=False):
        totals = client_totals(part.to_dict("records"), num_clients, malicious_ids)
        rows.extend(dict(t, axis=name) for t in totals)
        if malicious_ids:
            record = auc_record(totals, f"__{name}")
            if name in WHOLE_SET_AXES:   # one group: V and P are undefined
                record = {k: (float("nan") if k.startswith(("auc_V_", "auc_P_")) else v)
                          for k, v in record.items()}
            aucs.update(record)
    return rows, aucs


# ── round-level checkpoint ──────────────────────────────────────────────────

CHECKPOINT_VERSION = 1


def checkpoint_identity(config) -> dict:
    keys = ("experiment_name", "seed", "attack_type", "malicious_ratio", "num_rounds",
            "num_clients", "participation_ratio", "partition_id", "num_mc_samples",
            "local_epochs", "local_steps", "batch_size", "local_lr", "momentum",
            "crop_size", "reference_per_class", "grouping_seed", "compute_shapley",
            "head_prior_init", "global_axis")
    return {key: getattr(config, key) for key in keys}


def save_checkpoint(path: str, payload: dict) -> None:
    torch.save(payload, path + ".tmp")
    os.replace(path + ".tmp", path)


def load_checkpoint(path: str, config):
    if not os.path.exists(path):
        return None
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except Exception as exc:  # truncated or corrupt file: start the run over
        logger.warning("ignoring unreadable checkpoint %s (%s)", path, exc)
        return None
    if (payload.get("version") != CHECKPOINT_VERSION
            or payload.get("identity") != checkpoint_identity(config)):
        logger.warning("ignoring checkpoint %s: configuration changed", path)
        return None
    return payload


def rng_state() -> dict:
    state = {"python": random.getstate(), "numpy": np.random.get_state(),
             "torch": torch.get_rng_state()}
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(state: dict) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if "cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])


# ── one run ───────────────────────────────────────────────────────────────

def _class_columns(prefix: str, values, suffix: str = "") -> dict:
    return {f"{prefix}{c}{suffix}": float(v) for c, v in enumerate(values)}


def run_experiment(config, data, evaluator, test_set, partition, malicious_ids,
                   schedule, checkpoint_path=None):
    """Run one condition+seed; resumable after every finished round."""
    set_seed(config.seed)
    model = create_model(config, class_share=evaluator.pixel_counts.sum(0))
    param_keys = model.param_keys()
    train_model = copy.deepcopy(model)
    eval_model = copy.deepcopy(model)
    server = FLServer(eval_model, test_set, config)
    clients = {cid: FLClient(cid, indices, data.train_images, data.train_labels, config)
               for cid, indices in partition.items()}
    sample_counts = {cid: len(indices) for cid, indices in partition.items()}
    main = config.main_grouping
    weights_by_grouping = {name: evaluator.group_weights(name) for name in evaluator.names}
    num_mal = len(malicious_ids)
    common = {
        "experiment_name": config.experiment_name,
        "attack_type": config.attack_type,
        "run_seed": config.seed,
        "malicious_ratio": config.malicious_ratio if num_mal else 0.0,
        "partition_id": config.partition_id,
        "dirichlet_alpha": config.dirichlet_alpha,
    }

    state = {
        "dfr_sigma_est": None,
        "afr_c_est": None,
        "last_warmup_mean_local_norm": None,
        "participation": {cid: 0 for cid in range(config.num_clients)},
        "cumulative_sv": {cid: 0.0 for cid in range(config.num_clients)},
        "class_var_history": {cid: [] for cid in range(config.num_clients)},
        "pos_sum_history": {cid: [] for cid in range(config.num_clients)},
        "records": {key: [] for key in RECORD_KEYS},
    }
    # The untrained global model, for the round-0 share of the loss drop.
    initial_ref = evaluator.evaluate(eval_model, server.get_global_state())
    initial_ref = segmentation_metrics(initial_ref["confusion"], initial_ref["class_ce"])
    initial_test = server.evaluate()
    state["initial"] = {"ref_loss": initial_ref["loss"], "ref_miou": initial_ref["miou"],
                        "test_loss": initial_test["loss"], "test_miou": initial_test["miou"]}
    start_round = 0
    resumed = load_checkpoint(checkpoint_path, config) if checkpoint_path else None
    if resumed is not None:
        start_round = resumed["next_round"]
        initial = state["initial"]
        state = resumed["state"]
        state.setdefault("initial", initial)
        server.global_state = resumed["global_state"]
        server.global_history.clear()
        server.global_history.extend(resumed["global_history"])
        for cid, generator_state in resumed["client_generators"].items():
            clients[cid].generator.set_state(generator_state)
        restore_rng_state(resumed["rng"])
        logger.info("resume %s seed=%d from round %d", config.experiment_name,
                    config.seed, start_round)

    records = state["records"]
    participation = state["participation"]
    for round_t in range(start_round, config.num_rounds):
        started = time.perf_counter()
        selected = schedule[round_t]
        global_sd = server.get_global_state()
        global_history = list(server.global_history)
        gp = params_view(global_sd, param_keys)
        hp = [params_view(h, param_keys) for h in global_history]

        if (config.attack_type == "afr" and state["afr_c_est"] is None
                and config.afr_c_override is None
                and round_t >= config.attack_warmup_rounds):
            first_norm = global_update_norm(hp[1], hp[0])
            current_norm = global_update_norm(gp, hp[-1])
            decay_rate = estimate_decay_rate(current_norm, first_norm, round_num=round_t)
            state["afr_c_est"] = estimate_c_from_norms(
                state["last_warmup_mean_local_norm"], current_norm,
                config.num_clients, decay_rate, round_num=round_t - 1)
        if (config.attack_type in {"dfr", "afr"} and state["dfr_sigma_est"] is None
                and config.dfr_sigma_override is None):
            state["dfr_sigma_est"] = estimate_dfr_sigma(gp, hp)

        updates, train_loss, honest_ids = {}, {}, []
        attacking = set(malicious_ids).intersection(selected)
        for cid in selected:
            if cid in malicious_ids:
                continue
            participation[cid] += 1
            updates[cid] = clients[cid].train(train_model, global_sd)
            train_loss[cid] = clients[cid].last_train_loss
            assert_finite(updates[cid], f"client {cid} update")
            honest_ids.append(cid)
        if (config.attack_type in {"dfr", "sdfr", "afr"} and state["dfr_sigma_est"] is None
                and config.dfr_sigma_override is None):
            state["dfr_sigma_est"] = estimate_dfr_sigma_from_updates(
                {cid: params_view(updates[cid], param_keys) for cid in honest_ids})
            if state["dfr_sigma_est"] is None or state["dfr_sigma_est"] <= 0.0:
                raise RuntimeError("could not bootstrap a positive DFR sigma")
        for cid in selected:
            if cid not in malicious_ids:
                continue
            participation[cid] += 1
            updates[cid] = apply_attack(config.attack_type, global_sd, global_history,
                                        config, round_t, state["dfr_sigma_est"], cid,
                                        state["afr_c_est"], param_keys)
            train_loss[cid] = float("nan")   # free riders do not train
            assert_finite(updates[cid], f"client {cid} update")
        if config.attack_type == "afr" and round_t < config.attack_warmup_rounds:
            if not honest_ids:
                raise RuntimeError("AFR warm-up requires an honest selected client")
            state["last_warmup_mean_local_norm"] = float(np.mean(
                [update_norm(params_view(updates[cid], param_keys)) for cid in honest_ids]))
        client_seconds = time.perf_counter() - started

        shapley_started = time.perf_counter()
        sv, evaluations = {}, {}
        if config.compute_shapley:
            sv, evaluations = estimate_round_shapley(
                eval_model, updates, global_sd, evaluator,
                num_mc_samples=config.num_mc_samples,
                sample_counts=None if config.shapley_equal_weights else {
                    cid: sample_counts[cid] for cid in selected},
                cache_coalitions=config.shapley_cache_coalitions,
                rng=np.random.default_rng(
                    (int(config.seed) + 7) * 1_000_003 + (round_t + 1) * 10_007))
        shapley_seconds = time.perf_counter() - shapley_started

        # Each submitted model (global + update) on the reference set: the
        # per-class loss / accuracy / IoU of each client per round.  Under
        # equal weights these are the singleton coalitions already evaluated.
        client_eval_started = time.perf_counter()
        client_eval = {}
        for cid in selected:
            result = evaluations.get((cid,)) if config.shapley_equal_weights else None
            if result is None:
                local_state = OrderedDict((k, v + updates[cid][k]) for k, v in global_sd.items())
                result = evaluator.evaluate(eval_model, local_state)
            client_eval[cid] = segmentation_metrics(result["confusion"], result["class_ce"])
        client_eval_seconds = time.perf_counter() - client_eval_started

        aggregation_started = time.perf_counter()
        new_sd = fedavg_aggregate(global_sd, updates,
                                  weights={cid: sample_counts[cid] for cid in selected})
        assert_finite(new_sd, "aggregated global state")
        server.update_global_model(new_sd)
        test = server.evaluate()
        ref_result = evaluator.evaluate(eval_model, server.get_global_state())
        ref = segmentation_metrics(ref_result["confusion"], ref_result["class_ce"])
        aggregation_eval_seconds = time.perf_counter() - aggregation_started
        round_seconds = time.perf_counter() - started
        logger.info("%s seed=%d round %2d: test_miou=%.4f acc=%.4f loss=%.4f "
                    "clients=%.1fs shapley=%.1fs total=%.1fs",
                    config.experiment_name, config.seed, round_t, test["miou"],
                    test["pixel_acc"], test["loss"], client_seconds, shapley_seconds,
                    round_seconds)

        curve = dict(common, round=round_t, test_miou=test["miou"],
                     test_pixel_acc=test["pixel_acc"], test_loss=test["loss"],
                     ref_miou=ref["miou"], ref_pixel_acc=ref["pixel_acc"],
                     ref_loss=ref["loss"])
        curve.update(_class_columns("test_class_", test["class_iou"], "_iou"))
        curve.update(_class_columns("test_class_", test["class_acc"], "_acc"))
        curve.update(_class_columns("test_class_", test["class_ce"], "_loss"))
        curve.update(_class_columns("ref_class_", ref["class_ce"], "_loss"))
        for name in evaluator.names:
            loss = evaluator.split(ref_result["group_loss"])[name]
            curve.update(_class_columns(f"ref_{name}_group_", loss, "_loss"))
        curve.update({"client_phase_seconds": client_seconds,
                      "shapley_phase_seconds": shapley_seconds,
                      "client_eval_seconds": client_eval_seconds,
                      "aggregation_eval_seconds": aggregation_eval_seconds,
                      "round_seconds": round_seconds})
        records["round_curves"].append(curve)

        metrics_by_grouping, overall_by_grouping = {}, {}
        for name in sv:
            metrics_by_grouping[name] = compute_class_metrics(sv[name])
            overall_by_grouping[name] = per_group_to_overall(sv[name], weights_by_grouping[name])
        for cid in selected:
            m = metrics_by_grouping.get(main, {}).get(cid, {})
            overall = overall_by_grouping.get(main, {}).get(cid, 0.0)
            pc = participation[cid]
            state["cumulative_sv"][cid] += overall
            state["class_var_history"][cid].append(m.get("class_sv_variance", 0.0))
            state["pos_sum_history"][cid].append(m.get("positive_class_sv_sum", 0.0))
            local = client_eval[cid]
            records["round_details"].append(dict(
                common, round=round_t, client_id=cid,
                test_miou=test["miou"], test_pixel_acc=test["pixel_acc"], test_loss=test["loss"],
                is_malicious=cid in malicious_ids, is_attacking=cid in attacking,
                participation_count_so_far=pc, num_samples=sample_counts[cid],
                round_shapley_value=overall,
                cumulative_shapley_value=state["cumulative_sv"][cid],
                mean_shapley_value=state["cumulative_sv"][cid] / pc,
                class_sv_variance=m.get("class_sv_variance", 0.0),
                positive_class_sv_sum=m.get("positive_class_sv_sum", 0.0),
                concentration_ratio=m.get("concentration_ratio", 0.0),
                dominant_class=m.get("dominant_class", -1),
                mean_class_sv_variance=float(np.mean(state["class_var_history"][cid])),
                mean_positive_class_sv_sum=float(np.mean(state["pos_sum_history"][cid])),
                local_train_loss=train_loss[cid],
                local_ref_loss=local["loss"], local_ref_miou=local["miou"],
                local_ref_pixel_acc=local["pixel_acc"],
                update_norm=float(update_norm(params_view(updates[cid], param_keys))),
                client_phase_seconds=client_seconds, shapley_phase_seconds=shapley_seconds,
                aggregation_eval_seconds=aggregation_eval_seconds, round_seconds=round_seconds,
            ))
            for name in sv:
                gm = metrics_by_grouping[name][cid]
                records["axis_records"].append(dict(
                    common, round=round_t, client_id=cid, axis=name,
                    is_malicious=cid in malicious_ids,
                    round_shapley_value=overall_by_grouping[name][cid],
                    class_sv_variance=gm["class_sv_variance"],
                    positive_class_sv_sum=gm["positive_class_sv_sum"],
                    concentration_ratio=gm["concentration_ratio"],
                    dominant_class=gm["dominant_class"]))
                group_names = evaluator.groupings[name].group_names
                for g, value in enumerate(sv[name][cid]):
                    records["axis_group_shapley"].append(dict(
                        round=round_t, client_id=cid, is_malicious=cid in malicious_ids,
                        axis=name, group=g, group_name=group_names[g],
                        shapley=float(value)))
            if main in sv:
                record = dict(common, round=round_t, client_id=cid,
                              is_malicious=cid in malicious_ids)
                record.update(_class_columns("class_", sv[main][cid]))
                records["per_class_records"].append(record)
            record = dict(common, round=round_t, client_id=cid,
                          is_malicious=cid in malicious_ids, is_attacking=cid in attacking,
                          local_train_loss=train_loss[cid])
            record.update(_class_columns("class_", local["class_ce"], "_loss"))
            record.update(_class_columns("class_", local["class_acc"], "_acc"))
            record.update(_class_columns("class_", local["class_iou"], "_iou"))
            records["per_class_client_metrics"].append(record)

        if checkpoint_path:
            save_checkpoint(checkpoint_path, {
                "version": CHECKPOINT_VERSION,
                "identity": checkpoint_identity(config),
                "next_round": round_t + 1,
                "state": state,
                "global_state": server.global_state,
                "global_history": list(server.global_history),
                "client_generators": {cid: c.generator.get_state()
                                      for cid, c in clients.items()},
                "rng": rng_state(),
            })
    return state


# ── per-run outputs (written as soon as a condition+seed finishes) ─────────

def learning_speed(initial: dict, curves: pd.DataFrame) -> dict:
    """How much of the 25-round reference-loss drop happens in round 0.

    V is a variance of per-class marginal loss drops, so it scales roughly
    with the square of a round's loss drop: ``round0_v_scale`` estimates how
    many times larger round-0 V is than a typical later round.
    """
    losses = np.concatenate([[initial["ref_loss"]], curves["ref_loss"].to_numpy(float)])
    drops = -np.diff(losses)
    total = losses[0] - losses[-1]
    later = np.median(np.abs(drops[1:])) if len(drops) > 1 else float("nan")
    return {
        "untrained_test_miou": initial["test_miou"],
        "untrained_ref_loss": initial["ref_loss"],
        "round0_ref_loss_drop": float(drops[0]),
        "round0_share_of_loss_drop": float(drops[0] / total) if total > 0 else float("nan"),
        "round0_v_scale": float((drops[0] / later) ** 2) if later and later > 0 else float("nan"),
        "rounds_to_90pct_of_final_miou": int(np.argmax(
            curves["test_miou"].to_numpy(float) >= 0.9 * curves["test_miou"].iloc[-1])),
    }


def experiment_summary(config, state, malicious_ids, totals, axis_aucs,
                       wall_seconds) -> dict:
    details = pd.DataFrame(state["records"]["round_details"])
    curves = pd.DataFrame(state["records"]["round_curves"])
    frame = pd.DataFrame(totals)
    honest = frame[~frame["is_malicious"]]
    malicious = frame[frame["is_malicious"]]
    seen_h = honest[honest["participations"] > 0]
    seen_m = malicious[malicious["participations"] > 0]

    def mean(part, column):
        return float(part[column].mean()) if len(part) else 0.0

    num_mal = len(malicious_ids)
    summary = {
        "experiment_name": config.experiment_name,
        "attack_type": config.attack_type,
        "run_seed": config.seed,
        "partition_seed": config.partition_seed,
        "partition_id": config.partition_id,
        "malicious_ratio": config.malicious_ratio if num_mal else 0.0,
        "num_malicious_clients": num_mal,
        "partition_method": config.partition_method,
        "dirichlet_alpha": config.dirichlet_alpha,
        "malicious_client_ids": ",".join(map(str, sorted(malicious_ids))),
        "main_grouping": config.main_grouping,
        "num_rounds": config.num_rounds,
        "num_clients": config.num_clients,
        "participation_ratio": config.participation_ratio,
        "initial_test_miou": float(curves["test_miou"].iloc[0]),
        "final_test_miou": float(curves["test_miou"].iloc[-1]),
        "best_test_miou": float(curves["test_miou"].max()),
        "final_test_pixel_acc": float(curves["test_pixel_acc"].iloc[-1]),
        "final_test_loss": float(curves["test_loss"].iloc[-1]),
        **learning_speed(state["initial"], curves),
        "avg_round_shapley_honest": mean(seen_h, "sv_mean"),
        "avg_round_shapley_malicious": mean(seen_m, "sv_mean"),
        "avg_cumulative_shapley_honest": mean(honest, "sv_sum"),
        "avg_cumulative_shapley_malicious": mean(malicious, "sv_sum"),
        "shapley_gap_honest_vs_malicious": mean(seen_h, "sv_mean") - mean(seen_m, "sv_mean"),
        "avg_class_sv_variance_honest": mean(seen_h, "v_mean"),
        "avg_class_sv_variance_malicious": mean(seen_m, "v_mean"),
        "avg_positive_class_sv_sum_honest": mean(seen_h, "p_mean"),
        "avg_positive_class_sv_sum_malicious": mean(seen_m, "p_mean"),
        "avg_local_ref_loss_honest": float(details.loc[~details["is_malicious"], "local_ref_loss"].mean()),
        "avg_local_ref_loss_malicious": (float(details.loc[details["is_malicious"], "local_ref_loss"].mean())
                                         if num_mal else float("nan")),
        "dfr_sigma": state["dfr_sigma_est"],
        "afr_c": state["afr_c_est"],
        "total_experiment_seconds": float(curves["round_seconds"].sum()),
        "wall_seconds_this_process": wall_seconds,
        "mean_round_seconds": float(curves["round_seconds"].mean()),
        "mean_client_phase_seconds": float(curves["client_phase_seconds"].mean()),
        "mean_shapley_phase_seconds": float(curves["shapley_phase_seconds"].mean()),
        "mean_client_eval_seconds": float(curves["client_eval_seconds"].mean()),
        "mean_aggregation_eval_seconds": float(curves["aggregation_eval_seconds"].mean()),
    }
    if malicious_ids and config.compute_shapley:
        summary.update(auc_record(totals))
        summary.update(axis_aucs)
    return summary


def finish_run(config, run_dir, state, malicious_ids, partition_frame, config_record,
               wall_seconds, make_plots=True):
    """Write CSVs, the 20NEWS-style workbook and plots; mark the run done."""
    from utils.export import write_run_workbook

    records = state["records"]
    totals = client_totals(records["round_details"], config.num_clients, malicious_ids)
    by_axis, axis_aucs = axis_client_totals(records["axis_records"],
                                            config.num_clients, malicious_ids)
    common = {"experiment_name": config.experiment_name,
              "partition_id": config.partition_id,
              "attack_type": config.attack_type,
              "malicious_ratio": config.malicious_ratio if malicious_ids else 0.0,
              "run_seed": config.seed}
    tables = {key: pd.DataFrame(records[key]) for key in RECORD_KEYS}
    tables["client_totals"] = pd.DataFrame(totals).assign(**common)
    tables["axis_client_totals"] = pd.DataFrame(by_axis).assign(**common)
    for key, frame in tables.items():
        frame.to_csv(os.path.join(run_dir, f"{key}.csv"), index=False)
    summary = experiment_summary(config, state, malicious_ids, totals, axis_aucs,
                                 wall_seconds)
    run_config = dict(config_record)
    run_config.update(asdict(config))
    write_run_workbook(os.path.join(run_dir, "experiment_results.xlsx"), tables,
                       summary, partition_frame, run_config)
    if make_plots:
        try:
            from visualization.plots import plot_run
            plot_run(run_dir, tables, summary)
        except Exception:  # plots must never cost a finished run
            logger.exception("plotting failed for %s", run_dir)
    write_json(os.path.join(run_dir, "summary.json"), summary)
    checkpoint = os.path.join(run_dir, "checkpoint.pt")
    if os.path.exists(checkpoint):
        os.remove(checkpoint)
    logger.info("done %s seed=%d final_miou=%.4f %s", config.experiment_name,
                config.seed, summary["final_test_miou"],
                {k: round(v, 3) for k, v in summary.items()
                 if k.startswith("auc_V_mean") and v == v})


def refresh_summary(root: str, make_plots: bool) -> None:
    """Rebuild the combined workbook (and plots) after every finished run."""
    try:
        import summarize
        summarize.main(["--results-root", root] + ([] if make_plots else ["--no-plots"]))
    except SystemExit:
        pass
    except Exception:
        logger.exception("could not refresh the combined summary under %s", root)


# ── CLI ───────────────────────────────────────────────────────────────────

def csv_list(cast):
    return lambda text: tuple(cast(item) for item in text.split(",") if item.strip())


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-dir", default=Config.data_dir)
    p.add_argument("--results-dir", default="results/run")
    p.add_argument("--partition-method", choices=("iid", "dirichlet"), default="iid")
    p.add_argument("--dirichlet-alpha", type=float, default=0.5)
    p.add_argument("--attack-types", type=csv_list(str), default=("dfr", "sdfr", "afr"))
    p.add_argument("--malicious-ratios", type=csv_list(float), default=(0.1, 0.3, 0.5))
    p.add_argument("--experiment-seeds", type=csv_list(int), default=(42, 43, 44, 45, 46))
    p.add_argument("--no-baseline", action="store_true", help="skip the no-attack condition")
    p.add_argument("--no-shapley", action="store_true")
    p.add_argument("--no-global-axis", action="store_true",
                   help="skip the classless whole-val-set Shapley axis (faster)")
    p.add_argument("--num-rounds", type=int, default=Config.num_rounds)
    p.add_argument("--num-clients", type=int, default=Config.num_clients)
    p.add_argument("--participation-ratio", type=float, default=Config.participation_ratio)
    p.add_argument("--num-mc-samples", type=int, default=Config.num_mc_samples)
    p.add_argument("--local-epochs", type=int, default=Config.local_epochs)
    p.add_argument("--local-steps", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=Config.batch_size)
    p.add_argument("--eval-batch-size", type=int, default=Config.eval_batch_size)
    p.add_argument("--local-lr", type=float, default=Config.local_lr)
    p.add_argument("--crop-size", type=int, default=Config.crop_size)
    p.add_argument("--reference-per-class", type=int, default=Config.reference_per_class)
    p.add_argument("--min-samples-per-client", type=int, default=Config.min_samples_per_client)
    p.add_argument("--pretrained-backbone", default=Config.pretrained_backbone)
    p.add_argument("--no-head-prior", action="store_true",
                   help="leave the classifier bias at its default initialisation")
    p.add_argument("--no-pretrained", action="store_true",
                   help="random backbone initialisation (tests only)")
    p.add_argument("--device", choices=("auto", "cuda", "cpu", "mps"), default="auto")
    p.add_argument("--require-cuda", action="store_true")
    p.add_argument("--torch-threads", type=int, default=None)
    p.add_argument("--summary-root", default=None,
                   help="directory whose combined workbook is refreshed after every "
                        "finished run (default: --results-dir)")
    p.add_argument("--no-plots", action="store_true")
    return p.parse_args(argv)


def resolve_device(choice: str, require_cuda: bool) -> str:
    if require_cuda and not torch.cuda.is_available():
        raise SystemExit("--require-cuda was given but CUDA is not available")
    if choice != "auto":
        return choice
    return "cuda" if torch.cuda.is_available() else "cpu"


def write_json(path, payload):
    with open(path + ".tmp", "w") as handle:
        json.dump(payload, handle, indent=2, default=float)
    os.replace(path + ".tmp", path)


def main(argv=None):
    args = parse_args(argv)
    if args.torch_threads:
        torch.set_num_threads(args.torch_threads)
    device = resolve_device(args.device, args.require_cuda)
    if device == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
    part_id = partition_id(args.partition_method, args.dirichlet_alpha)

    base = Config(
        data_dir=args.data_dir,
        results_dir=args.results_dir,
        num_clients=args.num_clients,
        partition_method=args.partition_method,
        dirichlet_alpha=args.dirichlet_alpha if args.partition_method == "dirichlet" else None,
        partition_id=part_id,
        min_samples_per_client=args.min_samples_per_client,
        num_rounds=args.num_rounds,
        participation_ratio=args.participation_ratio,
        num_mc_samples=args.num_mc_samples,
        local_epochs=args.local_epochs,
        local_steps=args.local_steps,
        batch_size=args.batch_size,
        eval_batch_size=args.eval_batch_size,
        local_lr=args.local_lr,
        crop_size=args.crop_size,
        reference_per_class=args.reference_per_class,
        pretrained_backbone=args.pretrained_backbone,
        require_pretrained=not args.no_pretrained,
        head_prior_init=not args.no_head_prior,
        compute_shapley=not args.no_shapley,
        global_axis=not args.no_global_axis and not args.no_shapley,
        device=device,
    )
    os.makedirs(os.path.join(base.results_dir, "runs"), exist_ok=True)
    logger.info("device=%s partition=%s results=%s", device, part_id, base.results_dir)

    data = load_voc(base)
    ref_images = center_crop(data.train_images[data.reference_index].numpy(), base.crop_size)
    ref_labels = center_crop(data.train_labels[data.reference_index].numpy(), base.crop_size)
    reference_set = ImageSet(torch.from_numpy(np.ascontiguousarray(ref_images)),
                             torch.from_numpy(np.ascontiguousarray(ref_labels)),
                             base.eval_batch_size, device, base.num_classes)
    test_set = ImageSet(data.test_images, data.test_labels, base.eval_batch_size, device,
                        base.num_classes)
    groupings = build_groupings(len(reference_set), base.num_classes, base.grouping_seed)
    whole_sets = {"global": test_set} if base.global_axis else {}
    for name in whole_sets:
        groupings[name] = whole_set_grouping(
            name, "whole VOC val set (all labelled pixels), no class split; classless control")
    evaluator = ReferenceEvaluator(reference_set, groupings, base.num_classes, device,
                                   base.eval_amp, whole_sets=whole_sets)
    logger.info("train pool %d images, reference %d, test %d; groupings %s",
                len(data.client_pool), len(reference_set), len(test_set),
                {n: g.num_groups for n, g in groupings.items()})
    pd.DataFrame({
        "image_index": data.reference_index,
        "dominant_class": data.dominant[data.reference_index],
        "random_image21_group": groupings["random_image21"].cell_group[:, 0],
    }).to_csv(os.path.join(base.results_dir, "reference_set.csv"), index=False)
    pd.DataFrame([{"grouping": n, "num_groups": g.num_groups, "definition": g.definition,
                   "groups": json.dumps(g.group_names),
                   "group_pixel_share": json.dumps(np.round(evaluator.group_weights(n), 5).tolist())}
                  for n, g in groupings.items()]).to_csv(
        os.path.join(base.results_dir, "groupings.csv"), index=False)

    pool = data.client_pool
    pool_labels = data.dominant[pool]
    attacks = [("none", 0.0)] if not args.no_baseline else []
    attacks += [(a, r) for a in args.attack_types for r in args.malicious_ratios]
    config_record = asdict(base)
    config_record.update({
        "train_pool_images": int(len(pool)),
        "reference_images": int(len(reference_set)),
        "test_images": int(len(test_set)),
        "data_source": data.meta.get("source"),
        "data_train_source": data.meta.get("train_source"),
        "groupings": {n: g.num_groups for n, g in groupings.items()},
        "torch_version": torch.__version__,
    })
    write_json(os.path.join(base.results_dir, "experiment_config.json"), config_record)

    for seed in args.experiment_seeds:
        if base.partition_method == "iid":
            positions = iid_partition(pool, base.num_clients, seed=seed)
        else:
            positions = dirichlet_partition(
                pool_labels, base.num_clients, alpha=base.dirichlet_alpha,
                min_samples_per_client=base.min_samples_per_client, seed=seed)
        partition = {cid: pool[np.asarray(pos, dtype=int)].tolist()
                     for cid, pos in positions.items()}
        stats = partition_statistics(pool_labels, positions, base.num_classes)
        pd.DataFrame(stats).assign(run_seed=seed).to_csv(
            os.path.join(base.results_dir, f"partition_seed{seed}.csv"), index=False)
        attacker_order = np.random.default_rng(seed).permutation(base.num_clients)
        schedule = build_participation_schedule(base.num_clients, base.participation_ratio,
                                                base.num_rounds, seed=seed + 1_000_003)
        logger.info("seed %d client images=%s attacker_order=%s", seed,
                    [s["total_samples"] for s in stats], attacker_order.tolist())

        for attack, ratio in attacks:
            name = experiment_name(attack, ratio, part_id)
            run_dir = os.path.join(base.results_dir, "runs", f"{name}_seed{seed}")
            if os.path.exists(os.path.join(run_dir, "summary.json")):
                logger.info("skip completed %s seed=%d", name, seed)
                continue
            os.makedirs(run_dir, exist_ok=True)
            config = copy.deepcopy(base)
            config.seed = seed
            config.partition_seed = seed
            config.experiment_name = name
            config.attack_type = attack
            config.malicious_ratio = ratio
            num_mal = int(round(base.num_clients * ratio)) if attack != "none" else 0
            malicious_ids = set(attacker_order[:num_mal].tolist())
            started = time.perf_counter()
            state = run_experiment(config, data, evaluator, test_set, partition,
                                   malicious_ids, schedule,
                                   checkpoint_path=os.path.join(run_dir, "checkpoint.pt"))
            finish_run(config, run_dir, state, malicious_ids,
                       partition_frame=pd.DataFrame(stats).assign(run_seed=seed),
                       config_record=config_record,
                       wall_seconds=time.perf_counter() - started,
                       make_plots=not args.no_plots)
            refresh_summary(args.summary_root or base.results_dir, not args.no_plots)

    logger.info("partition %s finished", part_id)


if __name__ == "__main__":
    sys.exit(main())
