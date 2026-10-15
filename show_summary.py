#!/usr/bin/env python3
"""Print the main tables of a combined workbook (used by run_seg.ipynb).

Usage: python3 show_summary.py results/pilot
"""

import os
import sys

import pandas as pd

root = sys.argv[1] if len(sys.argv) > 1 else "results/pilot"
path = os.path.join(root, "seg_csdd_summary.xlsx")
if not os.path.exists(path):
    sys.exit(f"{path} not found (no finished runs yet?)")
pd.set_option("display.width", 220)
pd.set_option("display.max_columns", 40)
sheets = pd.ExcelFile(path).sheet_names


def show(title, frame):
    print(f"== {title}")
    print(frame.round(3).to_string(index=False) if len(frame) else "(empty)")
    print()


show("Completeness", pd.read_excel(path, "Completeness"))
auc = pd.read_excel(path, "AUC by Condition")
show("AUC by Condition, class21 (mean over seeds)",
     auc[[c for c in auc.columns if c in ("partition_id", "attack_type", "malicious_ratio",
                                          "num_seeds") or c.endswith("_mean_mean")]])
if "Axis Comparison" in sheets:
    show("Axis Comparison (mean AUC; class21 = main, others = controls)",
         pd.read_excel(path, "Axis Comparison"))
    tests = pd.read_excel(path, "Axis Tests")
    show("Axis Tests (class21 vs control, paired Wilcoxon, pooled)",
         tests[tests["scope"] == "all"])
miou = pd.read_excel(path, "Test mIoU")
show("Final test mIoU", miou.groupby(["partition_id", "attack_type", "malicious_ratio"])
     [["final_test_miou", "best_test_miou", "miou_drop_vs_no_attack"]].mean().reset_index())
curves = pd.read_excel(path, "Round Curves")
if len(curves):
    cols = [c for c in ("round_seconds", "client_phase_seconds", "shapley_phase_seconds")
            if c in curves]
    show("Mean seconds per round", curves[cols].mean().to_frame().T)
    first = curves[curves["attack_type"] == "none"].pivot_table(
        index="round", columns="partition_id", values="test_miou")
    show("No-attack test mIoU by round (convergence check)", first.reset_index())
