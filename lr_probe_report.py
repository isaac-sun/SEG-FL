#!/usr/bin/env python3
"""Compare the learning rates of ``run_lr_probe.sh``.

Usage: python3 lr_probe_report.py results/lr_probe

Per learning rate: test mIoU at a few rounds, the share of the 25-round
reference-loss drop that happens in round 0, and ``round0_v_scale`` (round-0
drop / median later drop, squared: roughly how many times larger round-0 V is
than a later round's).  Suggestion: among the rates whose round-0 share is at
most ``--max-round0-share`` (default 0.5), the one with the best final test
mIoU; if none qualifies, the smallest rate.  Writes ``lr_probe.xlsx`` and
``lr_probe.png`` next to the runs.
"""

import argparse
import glob
import json
import os
import warnings

import pandas as pd

warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)

parser = argparse.ArgumentParser()
parser.add_argument("root", nargs="?", default="results/lr_probe")
parser.add_argument("--max-round0-share", type=float, default=0.5)
args = parser.parse_args()

rows, curves = [], []
for path in sorted(glob.glob(os.path.join(args.root, "lr*", "runs", "*", "summary.json"))):
    with open(path) as handle:
        summary = json.load(handle)
    run_dir = os.path.dirname(path)
    config = pd.read_excel(os.path.join(run_dir, "experiment_results.xlsx"),
                           sheet_name="experiment_config").set_index("parameter")["value"]
    lr = float(config["local_lr"])
    curve = pd.read_csv(os.path.join(run_dir, "round_curves.csv"))
    curves.append(curve.assign(local_lr=lr))
    record = {"local_lr": lr}
    for r in (0, 1, 2, 4, 9, 24):
        if r < len(curve):
            record[f"test_miou_r{r}"] = curve["test_miou"].iloc[r]
    for key in ("untrained_test_miou", "round0_share_of_loss_drop", "round0_v_scale",
                "rounds_to_90pct_of_final_miou", "final_test_miou", "best_test_miou",
                "mean_round_seconds"):
        record[key] = summary.get(key)
    rows.append(record)
if not rows:
    raise SystemExit(f"no finished probe runs below {args.root}")
table = pd.DataFrame(rows).sort_values("local_lr", ascending=False)
pd.set_option("display.width", 220)
print(table.round(4).to_string(index=False))
ok = table[table["round0_share_of_loss_drop"] <= args.max_round0_share]
pick = (ok.loc[ok["final_test_miou"].idxmax()] if len(ok)
        else table.loc[table["local_lr"].idxmin()])
reason = (f"round-0 share <= {args.max_round0_share} and best final mIoU" if len(ok)
          else f"no rate has round-0 share <= {args.max_round0_share}; smallest rate")
print(f"\nsuggested LOCAL_LR={pick['local_lr']:g}  ({reason})")
with pd.ExcelWriter(os.path.join(args.root, "lr_probe.xlsx")) as writer:
    table.to_excel(writer, sheet_name="LR Probe", index=False)
    pd.concat(curves).to_excel(writer, sheet_name="Round Curves", index=False)
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for curve in curves:
        lr = curve["local_lr"].iloc[0]
        axes[0].plot(curve["round"], curve["test_miou"], marker="o", ms=3, label=f"lr {lr:g}")
        axes[1].plot(curve["round"], curve["ref_loss"], marker="o", ms=3, label=f"lr {lr:g}")
    axes[0].set_title("no attack: test mIoU by round")
    axes[1].set_title("no attack: reference-set CE by round")
    for ax in axes:
        ax.set_xlabel("round")
        ax.grid(alpha=0.3)
        ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(args.root, "lr_probe.png"), dpi=140)
except Exception as exc:  # the chart is optional
    print(f"plotting failed: {exc}")
