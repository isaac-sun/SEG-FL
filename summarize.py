#!/usr/bin/env python3
"""Combine finished SEG-FL runs into one workbook.

Reads every ``runs/*/summary.json`` below ``--results-root`` and writes
``seg_csdd_summary.xlsx`` with these sheets:

* ``AUC by Condition``: mean and sample SD over seeds of each score's AUC for
  the main grouping ``class21`` (V_mean = paper Eq. 20 is the primary score;
  lower score = free rider);
* ``Axis Comparison`` / ``Axis AUC by Condition`` / ``Axis AUC by Run`` /
  ``Axis Definitions``: every score (V_mean, P_mean, SV_mean and the three V
  aggregations) on every grouping ("Shapley axis": class21, class20_nobg,
  voc_super5, random_class5, random_image21) of the same runs; same layout as
  ECL-FL;
* ``Axis Tests``: paired Wilcoxon signed-rank tests over all attack runs
  (pooled and per partition): H2, class21 against every control axis; and
  class21 V_mean against the classless Shapley value (SV_mean) on ``global``
  (whole val set) and on the reference set;
* ``AUC by Run``: one row per condition+seed;
* ``Seed Macro``: per-seed macro AUC over all attack conditions;
* ``Client Totals``: per-client cumulative and per-participation scores;
* ``Test mIoU``: final test mIoU per run, with the matched no-attack run;
* ``Round Curves``: test mIoU / pixel accuracy / loss per round;
* ``Completeness``: finished runs per partition;
* ``Robustness AUC by Condition`` / ``Robustness AUC by Run``: class21 only,
  primary V_mean next to the post-hoc V aggregations (H3), see ``axis_scores``;
* ``round_shapley_details`` / ``experiment_summary`` / ``per_class_records`` /
  ``per_class_client_metrics`` / ``per_class_global_metrics``: all runs
  concatenated, same sheets and columns as each run's
  ``experiment_results.xlsx`` (20NEWS-FL layout); ``axis_shapley_records``
  holds every axis.

Charts are written to ``<results-root>/plots/`` unless ``--no-plots``.
"""

from __future__ import annotations

import argparse
import glob
import json
import os

import numpy as np
import pandas as pd

AUC_COLUMNS = ["auc_V_mean", "auc_V_sum", "auc_P_mean", "auc_P_sum",
               "auc_SV_mean", "auc_SV_sum"]
MAIN_GROUPING = "class21"
RUN_TABLES = ("client_totals", "round_curves", "round_details", "axis_records",
              "per_class_records", "per_class_client_metrics")
EXCEL_MAX_ROWS = 1_048_575
CONDITION_KEYS = ["partition_id", "attack_type", "malicious_ratio"]
RUN_KEYS = ["partition_id", "experiment_name", "run_seed"]


def load(root: str, seeds=None):
    summaries = []
    tables = {name: [] for name in RUN_TABLES}
    for path in sorted(glob.glob(os.path.join(root, "**", "runs", "*", "summary.json"),
                                 recursive=True)):
        run_dir = os.path.dirname(path)
        with open(path) as handle:
            summary = json.load(handle)
        if seeds and int(summary.get("run_seed", -1)) not in seeds:
            continue
        summaries.append(summary)
        for name in RUN_TABLES:
            file = os.path.join(run_dir, f"{name}.csv")
            if os.path.exists(file) and os.path.getsize(file) > 1:
                tables[name].append(pd.read_csv(file))
    if not summaries:
        raise SystemExit(f"no finished runs below {root}")
    tables = {name: pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
              for name, frames in tables.items()}
    return pd.DataFrame(summaries), tables


def condition_table(frame: pd.DataFrame, keys: list, columns: list) -> pd.DataFrame:
    if not len(frame):
        return pd.DataFrame()
    grouped = frame.groupby(keys, sort=False)[columns]
    return grouped.mean().add_suffix("_mean").join(
        grouped.std(ddof=1).add_suffix("_sd")).join(
        grouped.size().rename("num_seeds")).reset_index()


# Post-hoc V aggregations (robustness checks; the primary score stays V_mean).
LATE_FROM_ROUND = 3   # after the 2 DFR warm-up rounds and round 2
ROBUST_SCORES = {
    "V_rank_mean": "mean over participations of the within-round percentile rank of V",
    "log10V_mean": "mean over participations of log10(V)",
    "V_late_mean": f"participation mean of V over rounds >= {LATE_FROM_ROUND}",
}
AXIS_SCORES = ("V_mean", "P_mean", "SV_mean", "V_rank_mean", "log10V_mean", "V_late_mean")
WHOLE_SET_AXES = ("global",)   # one group: only SV is defined
AXIS_DEFINITIONS = {
    "global": "whole VOC val set (1,449 images, all labelled pixels), no class split: "
              "classless Shapley control; V and P undefined",
    "class21": "one group per VOC pixel class, background included; primary arm",
    "class20_nobg": "as class21 without the background pixels",
    "voc_super5": "official VOC categories: background, person, animal, vehicle, indoor",
    "random_class5": "classes merged at random into groups with the voc_super5 sizes "
                     "(control for the semantics of voc_super5)",
    "random_image21": "reference images split at random into 21 groups, all pixels of an "
                      "image together (control: same group count as class21, no class "
                      "semantics)",
}


def axis_scores(axis_records: pd.DataFrame, runs: pd.DataFrame) -> pd.DataFrame:
    """AUC of every score on every Shapley axis; one row per attack run and axis.

    Same as ECL-FL ``summarize.axis_scores``.  V_mean, P_mean and SV_mean equal
    the per-run ``auc_*__<axis>``; the three V aggregations are post-hoc
    robustness checks.  Lower = more suspicious.  A client without any (late)
    participation gets the lowest possible value.
    """
    from main import pairwise_auc

    if not len(axis_records) or "axis" not in axis_records:
        return pd.DataFrame()
    frame = axis_records.copy()
    frame["v_rank"] = frame.groupby(RUN_KEYS + ["axis", "round"])["class_sv_variance"].rank(pct=True)
    frame["log10_v"] = np.log10(frame["class_sv_variance"].clip(lower=1e-30))
    order = list(dict.fromkeys(frame["axis"]))
    groups = dict(iter(frame.groupby(RUN_KEYS + ["axis"], sort=False)))
    rows = []
    for _, run in runs[runs["attack_type"] != "none"].iterrows():
        malicious = {int(c) for c in str(run["malicious_client_ids"]).split(",")
                     if c.strip().isdigit()}
        clients = list(range(int(run.get("num_clients", frame["client_id"].max() + 1))))
        is_mal = np.array([c in malicious for c in clients])
        for axis in order:
            part = groups.get((run["partition_id"], run["experiment_name"], run["run_seed"], axis))
            if part is None:
                continue
            by_client = part.groupby("client_id")
            late = part[part["round"] >= LATE_FROM_ROUND].groupby("client_id")
            scores = {
                "V_mean": by_client["class_sv_variance"].mean(),
                "P_mean": by_client["positive_class_sv_sum"].mean(),
                "SV_mean": by_client["round_shapley_value"].mean(),
                "V_rank_mean": by_client["v_rank"].mean(),
                "log10V_mean": by_client["log10_v"].mean(),
                "V_late_mean": late["class_sv_variance"].mean(),
            }
            fill = dict.fromkeys(scores, 0.0)
            fill["log10V_mean"] = float(part["log10_v"].min())
            record = {k: run[k] for k in CONDITION_KEYS + ["run_seed", "experiment_name"]}
            record["axis"] = axis
            for name, series in scores.items():
                if axis in WHOLE_SET_AXES and name != "SV_mean":
                    record[f"auc_{name}"] = np.nan
                    continue
                values = series.reindex(clients).fillna(fill[name]).to_numpy()
                record[f"auc_{name}"] = pairwise_auc(values[~is_mal], values[is_mal], True)
            rows.append(record)
    return pd.DataFrame(rows)


def axis_tables(axis_runs: pd.DataFrame):
    """Per-condition means and the side-by-side comparison of the axes."""
    if not len(axis_runs):
        return pd.DataFrame(), pd.DataFrame()
    order = list(dict.fromkeys(axis_runs["axis"]))
    cols = [f"auc_{name}" for name in AXIS_SCORES]
    by_condition = condition_table(axis_runs, CONDITION_KEYS + ["axis"], cols)
    long = axis_runs.melt(id_vars=CONDITION_KEYS + ["run_seed", "axis"], value_vars=cols,
                          var_name="score", value_name="auc")
    long["score"] = pd.Categorical(long["score"].str[len("auc_"):], AXIS_SCORES, ordered=True)
    overall = long.pivot_table(index="score", columns="axis", values="auc",
                               aggfunc="mean", observed=True).reset_index()
    overall[CONDITION_KEYS] = ["all", "all", float("nan")]
    per_condition = long.pivot_table(index=CONDITION_KEYS + ["score"], columns="axis",
                                     values="auc", aggfunc="mean", observed=True).reset_index()
    comparison = pd.concat([overall, per_condition], ignore_index=True)
    comparison = comparison[CONDITION_KEYS + ["score"] + [a for a in order if a in comparison]]
    return by_condition, comparison


def axis_tests(axis_runs: pd.DataFrame) -> pd.DataFrame:
    """Paired Wilcoxon signed-rank tests over all attack runs (pooled and per
    partition).  H2: class21 V_mean / P_mean vs the same score on each control
    axis.  Classless control: class21 V_mean vs SV_mean on ``global`` (whole
    val set) and on class21 itself (reference set)."""
    from scipy.stats import wilcoxon

    if not len(axis_runs) or MAIN_GROUPING not in set(axis_runs["axis"]):
        return pd.DataFrame()
    axes = list(dict.fromkeys(axis_runs["axis"]))
    pairs = [(score, other, score) for score in ("V_mean", "P_mean")
             for other in axes if other != MAIN_GROUPING and other not in WHOLE_SET_AXES]
    pairs += [("V_mean", other, "SV_mean") for other in axes if other in WHOLE_SET_AXES]
    pairs += [("V_mean", MAIN_GROUPING, "SV_mean")]
    long = axis_runs.melt(id_vars=CONDITION_KEYS + ["run_seed", "axis"],
                          value_vars=[f"auc_{s}" for s in AXIS_SCORES],
                          var_name="score", value_name="auc")
    long["key"] = long["axis"] + ":" + long["score"].str[len("auc_"):]
    wide = long.pivot_table(index=CONDITION_KEYS + ["run_seed"], columns="key",
                            values="auc").reset_index()
    rows = []
    for scope, part in [("all", wide)] + list(wide.groupby("partition_id")):
        for main_score, other, other_score in pairs:
            a, b = f"{MAIN_GROUPING}:{main_score}", f"{other}:{other_score}"
            if a not in part or b not in part:
                continue
            pair = part[[a, b]].dropna()
            diff = pair[a] - pair[b]
            record = {"scope": scope, "main": a, "control": b, "num_runs": int(len(pair)),
                      "main_mean_auc": float(pair[a].mean()) if len(pair) else np.nan,
                      "control_mean_auc": float(pair[b].mean()) if len(pair) else np.nan,
                      "mean_difference": float(diff.mean()) if len(pair) else np.nan,
                      "main_better_runs": int((diff > 0).sum()),
                      "control_better_runs": int((diff < 0).sum()),
                      "wilcoxon_statistic": np.nan, "p_value": np.nan}
            if len(pair) >= 2 and (diff != 0).any():
                result = wilcoxon(pair[a], pair[b], zero_method="wilcox")
                record["wilcoxon_statistic"] = float(result.statistic)
                record["p_value"] = float(result.pvalue)
            rows.append(record)
    return pd.DataFrame(rows)


def _write(writer, frame: pd.DataFrame, sheet: str, root: str, csv_name: str) -> None:
    if len(frame) > EXCEL_MAX_ROWS:   # keep the full table as CSV
        frame.to_csv(os.path.join(root, f"all_{csv_name}.csv"), index=False)
        frame = frame.head(EXCEL_MAX_ROWS)
    frame.to_excel(writer, sheet_name=sheet, index=False)


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-root", default="results/full_experiment")
    parser.add_argument("--output", default=None)
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument("--seeds", default="",
                        help="comma-separated run seeds to include (default: all)")
    parser.add_argument("--slim", action="store_true",
                        help="leave out the large per-class sheets (per_class_records, "
                             "per_class_client_metrics)")
    args = parser.parse_args(argv)
    seeds = {int(x) for x in args.seeds.split(",") if x.strip()}
    runs, tables = load(args.results_root, seeds)
    totals, curves = tables["client_totals"], tables["round_curves"]
    output = args.output or os.path.join(args.results_root, "seg_csdd_summary.xlsx")

    attacks = runs[runs["attack_type"] != "none"].copy()
    present = [c for c in AUC_COLUMNS if c in attacks]
    by_condition = condition_table(attacks, CONDITION_KEYS, present) if present else pd.DataFrame()
    seed_macro = (attacks.groupby(["partition_id", "run_seed"])[present].mean().reset_index()
                  if len(attacks) and present else pd.DataFrame())
    axis_runs = axis_scores(tables["axis_records"], runs)
    axis_condition, axis_comparison = axis_tables(axis_runs)
    tests = axis_tests(axis_runs)

    base = runs[runs["attack_type"] == "none"][
        ["partition_id", "run_seed", "final_test_miou", "final_test_pixel_acc"]
    ].rename(columns={"final_test_miou": "no_attack_test_miou",
                      "final_test_pixel_acc": "no_attack_test_pixel_acc"})
    test_miou = runs[CONDITION_KEYS + ["run_seed", "final_test_miou", "best_test_miou",
                                       "final_test_pixel_acc"]].merge(
        base, on=["partition_id", "run_seed"], how="left")
    test_miou["miou_drop_vs_no_attack"] = (
        test_miou["no_attack_test_miou"] - test_miou["final_test_miou"])

    robust_runs = pd.DataFrame()
    robust_condition = pd.DataFrame()
    if len(axis_runs):   # main grouping: primary V_mean vs the post-hoc aggregations (H3)
        cols = ["auc_V_mean"] + [f"auc_{n}" for n in ROBUST_SCORES]
        robust_runs = axis_runs[axis_runs["axis"] == MAIN_GROUPING][
            CONDITION_KEYS + ["run_seed", "experiment_name"] + cols]
        robust_condition = condition_table(robust_runs, CONDITION_KEYS, cols)
    completeness = runs.groupby("partition_id").agg(
        finished_runs=("experiment_name", "size"),
        seeds=("run_seed", lambda s: ",".join(map(str, sorted(set(s)))))).reset_index()

    tmp = f"{output}.{os.getpid()}.tmp.xlsx"
    with pd.ExcelWriter(tmp) as writer:
        by_condition.to_excel(writer, sheet_name="AUC by Condition", index=False)
        if len(axis_runs):
            axis_comparison.to_excel(writer, sheet_name="Axis Comparison", index=False)
            axis_condition.to_excel(writer, sheet_name="Axis AUC by Condition", index=False)
            axis_runs.to_excel(writer, sheet_name="Axis AUC by Run", index=False)
            tests.to_excel(writer, sheet_name="Axis Tests", index=False)
            pd.DataFrame([{"axis": a, "definition": AXIS_DEFINITIONS.get(a, "")}
                          for a in dict.fromkeys(axis_runs["axis"])]).to_excel(
                writer, sheet_name="Axis Definitions", index=False)
        runs.sort_values(CONDITION_KEYS + ["run_seed"]).to_excel(
            writer, sheet_name="AUC by Run", index=False)
        seed_macro.to_excel(writer, sheet_name="Seed Macro", index=False)
        _write(writer, totals, "Client Totals", args.results_root, "client_totals")
        test_miou.to_excel(writer, sheet_name="Test mIoU", index=False)
        curves_short = curves[[c for c in curves.columns
                               if not c.startswith(("test_class_", "ref_class_", "ref_"))
                               or c in ("ref_miou", "ref_pixel_acc", "ref_loss")]]
        curves_short.to_excel(writer, sheet_name="Round Curves", index=False)
        completeness.to_excel(writer, sheet_name="Completeness", index=False)
        robust_condition.to_excel(writer, sheet_name="Robustness AUC by Condition", index=False)
        robust_runs.to_excel(writer, sheet_name="Robustness AUC by Run", index=False)
        pd.DataFrame([{"score": k, "definition": v} for k, v in ROBUST_SCORES.items()]).to_excel(
            writer, sheet_name="Robustness Definitions", index=False)
        runs.to_excel(writer, sheet_name="experiment_summary", index=False)
        for sheet, name in (("round_shapley_details", "round_details"),
                            ("per_class_records", "per_class_records"),
                            ("per_class_client_metrics", "per_class_client_metrics"),
                            ("per_class_global_metrics", "round_curves"),
                            ("axis_shapley_records", "axis_records")):
            if args.slim and name.startswith("per_class"):
                continue
            _write(writer, tables[name], sheet, args.results_root, name)
    os.replace(tmp, output)
    if not args.no_plots:
        try:
            from visualization.plots import plot_summary
            plot_summary(args.results_root, runs, axis_runs, robust_runs)
        except Exception as exc:  # charts are optional
            print(f"plotting failed: {exc}")
    print(f"wrote {output} ({len(runs)} runs)")
    if len(axis_comparison):
        print("axis comparison, all attack conditions (mean AUC):")
        print(axis_comparison[axis_comparison["partition_id"] == "all"].drop(
            columns=CONDITION_KEYS).round(3).to_string(index=False))


if __name__ == "__main__":
    main()
