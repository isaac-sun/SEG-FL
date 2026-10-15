"""Per-run Excel workbook with the sheet layout of 20NEWS-FL.

Sheet names and the shared columns follow 20NEWS-FL ``experiment_results.xlsx``
(``round_shapley_details``, ``experiment_summary``, ``per_class_records``,
``client_data_distribution``, ``experiment_config``).  "class" is the VOC
pixel class (main grouping ``class21``).  Accuracy/loss columns become pixel
accuracy, IoU and cross-entropy; ``per_class_client_metrics`` holds each
submitted model's per-class metrics on the reference set.  The other
groupings ("Shapley axes", same names as ECL-FL) are in
``axis_shapley_records`` / ``axis_client_totals`` / ``axis_group_shapley``.
"""

from __future__ import annotations

import json
import os

import pandas as pd

SHEETS = (
    # sheet name, table key
    ("round_shapley_details", "round_details"),
    ("experiment_summary", None),
    ("per_class_records", "per_class_records"),
    ("per_class_client_metrics", "per_class_client_metrics"),
    ("per_class_global_metrics", "round_curves"),
    ("client_totals", "client_totals"),
    # Every grouping ("Shapley axis", as in ECL-FL), main grouping included.
    ("axis_shapley_records", "axis_records"),
    ("axis_client_totals", "axis_client_totals"),
    ("axis_group_shapley", "axis_group_shapley"),
)
EXCEL_MAX_ROWS = 1_048_575


def _config_frame(config: dict) -> pd.DataFrame:
    rows = []
    for key, value in config.items():
        if isinstance(value, (list, tuple, dict)):
            value = json.dumps(value, default=str)
        rows.append({"parameter": key, "value": value})
    return pd.DataFrame(rows)


def write_run_workbook(path: str, tables: dict, summary: dict,
                       partition: pd.DataFrame, config: dict) -> str:
    tmp = path + ".tmp.xlsx"
    with pd.ExcelWriter(tmp) as writer:
        for sheet, key in SHEETS:
            frame = pd.DataFrame([summary]) if key is None else tables[key]
            frame.head(EXCEL_MAX_ROWS).to_excel(writer, sheet_name=sheet, index=False)
        partition.to_excel(writer, sheet_name="client_data_distribution", index=False)
        _config_frame(config).to_excel(writer, sheet_name="experiment_config", index=False)
    os.replace(tmp, path)
    return path
