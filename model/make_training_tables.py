# model/make_training_tables.py
"""
Educational Use Only (Non-Clinical)
Creates training tables from synthetic CSVs.

Outputs:
- data/derived/train_duration.csv
- data/derived/train_noshow.csv
"""

from __future__ import annotations
import os
import pandas as pd


def main() -> None:
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    raw_dir = os.path.join(repo_root, "data", "raw")
    out_dir = os.path.join(repo_root, "data", "derived")
    os.makedirs(out_dir, exist_ok=True)

    orders = pd.read_csv(os.path.join(raw_dir, "orders.csv"))
    sched = pd.read_csv(os.path.join(raw_dir, "schedule_planned.csv"))
    exe = pd.read_csv(os.path.join(raw_dir, "execution_actual.csv"))

    # Parse timestamps
    orders["order_time"] = pd.to_datetime(orders["order_time"])
    sched["scheduled_start"] = pd.to_datetime(sched["scheduled_start"])
    exe["actual_start"] = pd.to_datetime(exe["actual_start"], errors="coerce")
    exe["actual_end"] = pd.to_datetime(exe["actual_end"], errors="coerce")

    # Join: schedule has sps_id/order_id; execution has sps_id/order_id; orders has order_id
    df = sched.merge(exe, on=["sps_id", "order_id", "scanner_id", "patient_type", "resource_modality"], how="left")
    df = df.merge(orders, on=["order_id", "patient_type"], how="left", suffixes=("", "_order"))

    # Features
    df["hour"] = df["scheduled_start"].dt.hour.astype(int)
    df["dow"] = df["scheduled_start"].dt.dayofweek.astype(int)

    # For outpatients, a simple lead time feature (days between order and scheduled)
    df["lead_time_days"] = ((df["scheduled_start"] - df["order_time"]).dt.total_seconds() / 86400.0).fillna(0.0)
    df["lead_time_days"] = df["lead_time_days"].clip(lower=0.0)

    # Duration target (only where completed and actual timestamps exist)
    df["duration_min"] = (df["actual_end"] - df["actual_start"]).dt.total_seconds() / 60.0

    # ---- Duration training table (all settings, completed only) ----
    dur = df[(df["completed"] == 1) & (df["duration_min"].notna())].copy()

    # Keep columns we will use
    dur_cols = [
        "patient_type", "resource_modality", "exam_code", "priority", "scanner_id",
        "requires_contrast", "requires_sedation", "isolation",
        "mobility", "hour", "dow", "lead_time_days",
        "duration_min",
    ]
    # Some columns may not exist if your generator didn’t include them (safe fallback)
    for c in dur_cols:
        if c not in dur.columns:
            dur[c] = 0
    dur = dur[dur_cols]

    dur.to_csv(os.path.join(out_dir, "train_duration.csv"), index=False)

    # ---- No-show training table (outpatient only) ----
    ns = df[df["patient_type"] == "OUTPATIENT"].copy()
    ns_cols = [
        "resource_modality", "exam_code", "priority", "scanner_id",
        "requires_contrast", "requires_sedation", "isolation",
        "hour", "dow", "lead_time_days",
        "no_show",
    ]
    for c in ns_cols:
        if c not in ns.columns:
            ns[c] = 0
    ns = ns[ns_cols]
    ns.to_csv(os.path.join(out_dir, "train_noshow.csv"), index=False)

    print("Wrote:")
    print("- data/derived/train_duration.csv")
    print("- data/derived/train_noshow.csv")


if __name__ == "__main__":
    main()
