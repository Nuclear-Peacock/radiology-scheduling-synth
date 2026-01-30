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


def _pick_col(df: pd.DataFrame, candidates: list[str]) -> str:
    for c in candidates:
        if c in df.columns:
            return c
    raise KeyError(f"None of these columns exist: {candidates}. Existing columns: {list(df.columns)}")


def main() -> None:
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    raw_dir = os.path.join(repo_root, "data", "raw")
    out_dir = os.path.join(repo_root, "data", "derived")
    os.makedirs(out_dir, exist_ok=True)

    orders = pd.read_csv(os.path.join(raw_dir, "orders.csv"))
    sched = pd.read_csv(os.path.join(raw_dir, "schedule_planned.csv"))
    exe = pd.read_csv(os.path.join(raw_dir, "execution_actual.csv"))

    # Parse timestamps we know exist
    orders["order_time"] = pd.to_datetime(orders["order_time"], errors="coerce")
    sched["scheduled_start"] = pd.to_datetime(sched["scheduled_start"], errors="coerce")
    exe["actual_start"] = pd.to_datetime(exe["actual_start"], errors="coerce")
    exe["actual_end"] = pd.to_datetime(exe["actual_end"], errors="coerce")

    # IMPORTANT: execution_actual also has scheduled_start; drop it to avoid merge column collisions
    if "scheduled_start" in exe.columns:
        exe = exe.drop(columns=["scheduled_start"])
    if "scheduled_end" in exe.columns:
        exe = exe.drop(columns=["scheduled_end"])

    # Merge schedule + execution first (keep schedule's scheduled_start)
    df = sched.merge(
        exe,
        on=["sps_id", "order_id", "scanner_id", "patient_type", "resource_modality"],
        how="left",
        suffixes=("_sched", "_exe"),
    )

    # Merge orders (brings order_time and order features)
    df = df.merge(
        orders,
        on=["order_id", "patient_type"],
        how="left",
        suffixes=("", "_order"),
    )

    # Use scheduled_start from schedule table
    sched_start_col = _pick_col(df, ["scheduled_start", "scheduled_start_sched", "scheduled_start_x"])
    df[sched_start_col] = pd.to_datetime(df[sched_start_col], errors="coerce")

    # Features
    df["hour"] = df[sched_start_col].dt.hour.fillna(0).astype(int)
    df["dow"] = df[sched_start_col].dt.dayofweek.fillna(0).astype(int)

    # Lead time (OP only really matters, but safe everywhere)
    df["lead_time_days"] = ((df[sched_start_col] - df["order_time"]).dt.total_seconds() / 86400.0).fillna(0.0)
    df["lead_time_days"] = df["lead_time_days"].clip(lower=0.0)

    # Duration target (only where completed and actual timestamps exist)
    df["duration_min"] = (df["actual_end"] - df["actual_start"]).dt.total_seconds() / 60.0

    # ---- Duration training table (all settings, completed only) ----
    dur = df[(df.get("completed", 0) == 1) & (df["duration_min"].notna())].copy()

    dur_cols = [
        "patient_type", "resource_modality", "exam_code", "priority", "scanner_id",
        "requires_contrast", "requires_sedation", "isolation",
        "mobility", "hour", "dow", "lead_time_days",
        "duration_min",
    ]
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
