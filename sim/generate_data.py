# sim/generate_data.py
"""
Educational Use Only (Non-Clinical)
-----------------------------------
This script generates fully synthetic radiology workflow data for education/research prototyping.
It is NOT validated for clinical or operational use. Do NOT use for real patient care or scheduling.

Outputs:
- data/raw/orders.csv
- data/raw/schedule_planned.csv
- data/raw/execution_actual.csv
"""

from __future__ import annotations

import os
import math
from dataclasses import dataclass
from datetime import datetime, timedelta, time
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import yaml


# -----------------------------
# Defaults (if config lacks keys)
# -----------------------------
DEFAULTS = {
    "start_date": "2025-01-01",
    "num_days": 180,
    "random_seed": 42,
    # Base daily volumes (scaled by flux multipliers)
    "base_daily_orders": {
        "ED": 180,
        "INPATIENT": 140,
        "OUTPATIENT": 220,
    },
    # Basic priority mixes by setting
    "priority_mix": {
        "ED": {"STAT": 0.35, "URGENT": 0.50, "ROUTINE": 0.15},
        "INPATIENT": {"STAT": 0.10, "URGENT": 0.50, "ROUTINE": 0.40},
        "OUTPATIENT": {"STAT": 0.01, "URGENT": 0.09, "ROUTINE": 0.90},
    },
    # Duration templates (minutes) – intentionally coarse/educational
    "duration_templates": {
        "XR": (6, 12),
        "XR_PORTABLE": (10, 22),
        "US": (25, 55),
        "CT": (12, 28),
        "MR": (30, 70),
    },
}

# Synthetic "exam codes" per modality (just for variety)
EXAM_CODES = {
    "XR": ["XR_CHEST_2V", "XR_ABD_2V", "XR_EXTREMITY", "XR_SPINE"],
    "XR_PORTABLE": ["XR_PORT_CHEST", "XR_PORT_ABD"],
    "US": ["US_ABD", "US_PELVIS", "US_DVT", "US_RUQ"],
    "CT": ["CT_HEAD_WO", "CT_CHEST_W", "CT_ABDPEL_W", "CT_TRAUMA"],
    "MR": ["MR_BRAIN_WO", "MR_SPINE_WO", "MR_KNEE_WO", "MR_ABD_W"],
}


@dataclass
class Config:
    start_date: datetime
    num_days: int
    seed: int

    ED_flux: float
    INPATIENT_flux: float
    OUTPATIENT_flux: float

    modality_mix: Dict[str, float]  # XR/US/CT/MR
    CT_downtime_rate: float
    MR_downtime_rate: float
    INPATIENT_transport_delay_min: int

    OUTPATIENT_no_show_rate: float
    overbooking_threshold: float
    ED_CT_reserve_capacity: float


def _load_yaml(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _parse_date(s: str) -> datetime:
    return datetime.strptime(s, "%Y-%m-%d")


def _parse_hhmm(s: str) -> time:
    hh, mm = s.split(":")
    return time(int(hh), int(mm))


def _time_to_dt(day: datetime, t: time) -> datetime:
    return datetime(day.year, day.month, day.day, t.hour, t.minute, 0)


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _choose_from_dist(rng: np.random.Generator, dist: Dict[str, float]) -> str:
    keys = list(dist.keys())
    probs = np.array([dist[k] for k in keys], dtype=float)
    probs = probs / probs.sum()
    return keys[int(rng.choice(len(keys), p=probs))]


def _sample_priority(rng: np.random.Generator, setting: str) -> str:
    mix = DEFAULTS["priority_mix"][setting]
    return _choose_from_dist(rng, mix)


def _sample_duration_minutes(rng: np.random.Generator, modality: str) -> int:
    lo, hi = DEFAULTS["duration_templates"][modality]
    # add mild lognormal-ish tail without getting too wild
    base = rng.integers(lo, hi + 1)
    noise = rng.normal(0, 2.5)
    d = max(3, int(round(base + noise)))
    return d


def _order_time_for_setting(rng: np.random.Generator, day: datetime, setting: str) -> datetime:
    # Simple time-of-day patterns (educational)
    if setting == "OUTPATIENT":
        # Most outpatient orders placed daytime
        hour = int(rng.choice([8, 9, 10, 11, 13, 14, 15, 16], p=[.1, .15, .17, .13, .15, .12, .1, .08]))
        minute = int(rng.integers(0, 60))
        return day + timedelta(hours=hour, minutes=minute)
    if setting == "ED":
        # ED: broader distribution, busier midday/evening
        hour = int(rng.choice(range(24), p=_ed_hour_probs()))
        minute = int(rng.integers(0, 60))
        return day + timedelta(hours=hour, minutes=minute)
    # Inpatient: clustered around rounds and afternoon changes
    hour = int(rng.choice([7, 8, 9, 10, 13, 14, 15, 16, 19, 21], p=[.08, .12, .16, .10, .12, .12, .10, .08, .06, .06]))
    minute = int(rng.integers(0, 60))
    return day + timedelta(hours=hour, minutes=minute)


def _ed_hour_probs() -> np.ndarray:
    # 24 probabilities sum to 1
    # Low overnight, higher 10am–10pm
    probs = np.array([
        0.015, 0.012, 0.010, 0.010, 0.012, 0.015,  # 0-5
        0.020, 0.025, 0.035, 0.045, 0.055, 0.060,  # 6-11
        0.060, 0.055, 0.055, 0.055, 0.055, 0.055,  # 12-17
        0.055, 0.050, 0.045, 0.035, 0.025, 0.020   # 18-23
    ], dtype=float)
    return probs / probs.sum()


def _normalize_mix(m: Dict[str, float]) -> Dict[str, float]:
    total = float(sum(m.values()))
    if total <= 0:
        return {"XR": 0.25, "US": 0.25, "CT": 0.25, "MR": 0.25}
    return {k: float(v) / total for k, v in m.items()}


def _map_modality_for_resources(requested_modality: str, is_portable: bool) -> str:
    if requested_modality == "XR" and is_portable:
        return "XR_PORTABLE"
    return requested_modality


def _is_open(resources_row: pd.Series, dt: datetime) -> bool:
    ot = _parse_hhmm(str(resources_row["open_time"]))
    ct = _parse_hhmm(str(resources_row["close_time"]))
    start = _time_to_dt(dt.replace(hour=0, minute=0, second=0, microsecond=0), ot)
    end = _time_to_dt(dt.replace(hour=0, minute=0, second=0, microsecond=0), ct)
    # Treat 23:59 as end-of-day
    if end.hour == 23 and end.minute == 59:
        end = dt.replace(hour=23, minute=59, second=59)
    return start <= dt <= end


def _scanner_day_window(resources_row: pd.Series, day: datetime) -> Tuple[datetime, datetime]:
    ot = _parse_hhmm(str(resources_row["open_time"]))
    ct = _parse_hhmm(str(resources_row["close_time"]))
    start = _time_to_dt(day, ot)
    end = _time_to_dt(day, ct)
    if ct.hour == 23 and ct.minute == 59:
        end = day.replace(hour=23, minute=59, second=59)
    return start, end


def _generate_downtime_events(
    rng: np.random.Generator,
    resources: pd.DataFrame,
    day: datetime,
    ct_rate: float,
    mr_rate: float
) -> Dict[str, List[Tuple[datetime, datetime]]]:
    """
    Returns {scanner_id: [(down_start, down_end), ...]}
    Simple model: each CT/MR scanner has probability=rate of one downtime event per day.
    """
    events: Dict[str, List[Tuple[datetime, datetime]]] = {}
    for _, r in resources.iterrows():
        sid = str(r["scanner_id"])
        mod = str(r["modality"])
        if mod not in ("CT", "MR"):
            continue
        rate = ct_rate if mod == "CT" else mr_rate
        if rng.random() < rate:
            day_start, day_end = _scanner_day_window(r, day)
            # place downtime in the middle 80% of operating window
            span_min = int((day_end - day_start).total_seconds() / 60)
            if span_min < 120:
                continue
            start_offset = int(rng.integers(int(span_min * 0.1), int(span_min * 0.8)))
            down_start = day_start + timedelta(minutes=start_offset)
            down_len = int(max(10, rng.normal(45 if mod == "CT" else 60, 15)))
            down_end = min(day_end, down_start + timedelta(minutes=down_len))
            events.setdefault(sid, []).append((down_start, down_end))
    return events


def _overlaps(a_start: datetime, a_end: datetime, b_start: datetime, b_end: datetime) -> bool:
    return a_start < b_end and b_start < a_end


def _next_available_time(
    scanner_available: datetime,
    scheduled_start: datetime
) -> datetime:
    return max(scanner_available, scheduled_start)


def main() -> None:
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    resources_path = os.path.join(repo_root, "data", "raw", "resources.csv")
    config_path = os.path.join(repo_root, "sim", "config.yaml")

    if not os.path.exists(resources_path):
        raise FileNotFoundError(f"Missing {resources_path}. Create data/raw/resources.csv first.")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Missing {config_path}. Create sim/config.yaml first.")

    cfg_raw = _load_yaml(config_path)

    # Read config with defaults
    start_date = _parse_date(str(cfg_raw.get("start_date", DEFAULTS["start_date"])))
    num_days = int(cfg_raw.get("num_days", DEFAULTS["num_days"]))
    seed = int(cfg_raw.get("random_seed", DEFAULTS["random_seed"]))

    ED_flux = float(cfg_raw.get("ED_flux", cfg_raw.get("ED_flux_multiplier", 1.0)))
    IN_flux = float(cfg_raw.get("INPATIENT_flux", cfg_raw.get("INPATIENT_flux_multiplier", 1.0)))
    OP_flux = float(cfg_raw.get("OUTPATIENT_flux", cfg_raw.get("OUTPATIENT_density_multiplier", 1.0)))

    mix = cfg_raw.get("modality_mix", {"XR": 0.30, "US": 0.20, "CT": 0.30, "MR": 0.20})
    mix = _normalize_mix({k: float(v) for k, v in mix.items()})

    CT_down = float(cfg_raw.get("CT_downtime_rate", 0.10))
    MR_down = float(cfg_raw.get("MR_downtime_rate", 0.10))
    transport_mean = int(cfg_raw.get("INPATIENT_transport_delay_min", cfg_raw.get("INPATIENT_transport_delay_mean_min", 18)))

    op_no_show = float(cfg_raw.get("OUTPATIENT_no_show_rate", cfg_raw.get("OUTPATIENT_base_no_show_rate", 0.08)))
    overbook_t = float(cfg_raw.get("overbooking_threshold", 0.60))
    ed_ct_reserve = float(cfg_raw.get("ED_CT_reserve_capacity", cfg_raw.get("ED_CT_reserve_capacity_pct", 0.10)))

    config = Config(
        start_date=start_date,
        num_days=num_days,
        seed=seed,
        ED_flux=ED_flux,
        INPATIENT_flux=IN_flux,
        OUTPATIENT_flux=OP_flux,
        modality_mix=mix,
        CT_downtime_rate=CT_down,
        MR_downtime_rate=MR_down,
        INPATIENT_transport_delay_min=transport_mean,
        OUTPATIENT_no_show_rate=op_no_show,
        overbooking_threshold=overbook_t,
        ED_CT_reserve_capacity=ed_ct_reserve,
    )

    rng = np.random.default_rng(config.seed)
    resources = pd.read_csv(resources_path)

    # ---------- Generate Orders ----------
    base_daily = DEFAULTS["base_daily_orders"]
    all_orders: List[dict] = []
    order_id_counter = 1

    for day_i in range(config.num_days):
        day = config.start_date + timedelta(days=day_i)
        day0 = day.replace(hour=0, minute=0, second=0, microsecond=0)

        # Daily order counts
        n_ed = int(rng.poisson(base_daily["ED"] * config.ED_flux))
        n_in = int(rng.poisson(base_daily["INPATIENT"] * config.INPATIENT_flux))
        n_op = int(rng.poisson(base_daily["OUTPATIENT"] * config.OUTPATIENT_flux))

        for setting, n in [("ED", n_ed), ("INPATIENT", n_in), ("OUTPATIENT", n_op)]:
            for _ in range(n):
                requested_mod = _choose_from_dist(rng, config.modality_mix)

                # Portable decision: inpatient has some portable XR; ED/OP generally not portable
                is_portable = False
                if requested_mod == "XR" and setting == "INPATIENT" and rng.random() < 0.25:
                    is_portable = True

                modality_for_resource = _map_modality_for_resources(requested_mod, is_portable)

                exam_code = str(rng.choice(EXAM_CODES[modality_for_resource]))
                priority = _sample_priority(rng, setting)

                # Simple flags (educational)
                requires_contrast = 0
                requires_sedation = 0
                isolation = 1 if (setting != "OUTPATIENT" and rng.random() < 0.06) else 0

                if requested_mod in ("CT", "MR") and rng.random() < (0.35 if setting != "OUTPATIENT" else 0.25):
                    requires_contrast = 1
                if requested_mod == "MR" and setting != "OUTPATIENT" and rng.random() < 0.06:
                    requires_sedation = 1

                mobility = str(rng.choice(["ambulatory", "wheelchair", "stretcher"], p=[0.55, 0.20, 0.25])) if setting != "OUTPATIENT" else "ambulatory"

                order_time = _order_time_for_setting(rng, day0, setting)

                # Outpatient scheduled time is in the future (lead time) – simplified
                lead_days = 0
                if setting == "OUTPATIENT":
                    lead_days = int(max(0, rng.normal(9, 6)))
                order = {
                    "order_id": f"O{order_id_counter:08d}",
                    "patient_type": setting,
                    "order_time": order_time.isoformat(),
                    "requested_modality": requested_mod,
                    "resource_modality": modality_for_resource,  # XR vs XR_PORTABLE
                    "exam_code": exam_code,
                    "priority": priority,
                    "requires_contrast": requires_contrast,
                    "requires_sedation": requires_sedation,
                    "isolation": isolation,
                    "mobility": mobility,
                    "outpatient_lead_days": lead_days,
                }
                all_orders.append(order)
                order_id_counter += 1

    orders_df = pd.DataFrame(all_orders)
    # Sort by order time, then priority rough order
    priority_rank = {"STAT": 0, "URGENT": 1, "ROUTINE": 2}
    orders_df["priority_rank"] = orders_df["priority"].map(priority_rank).fillna(2).astype(int)
    orders_df["order_time_dt"] = pd.to_datetime(orders_df["order_time"])
    orders_df = orders_df.sort_values(["order_time_dt", "priority_rank"]).reset_index(drop=True)

    # ---------- Build Planned Schedule ----------
    # Maintain scanner next-free times
    scanner_next_free: Dict[str, datetime] = {}
    scanner_day_close: Dict[Tuple[str, datetime], datetime] = {}

    # Precompute day windows for each scanner and day
    for _, r in resources.iterrows():
        sid = str(r["scanner_id"])
        for day_i in range(config.num_days):
            day = config.start_date + timedelta(days=day_i)
            open_dt, close_dt = _scanner_day_window(r, day.replace(hour=0, minute=0, second=0, microsecond=0))
            scanner_day_close[(sid, day.date())] = close_dt
            # initialize next_free at open
            # but only for that day; we’ll update as we schedule
            # We'll set next_free lazily when needed.

    # Track planned schedules rows
    planned_rows: List[dict] = []
    sps_counter = 1

    def pick_scanner(modality_needed: str, setting: str) -> List[str]:
        # If XR_PORTABLE needed, use those only
        if modality_needed == "XR_PORTABLE":
            eligible = resources[resources["modality"] == "XR_PORTABLE"]
        else:
            eligible = resources[resources["modality"] == modality_needed]
        # Simple location preference
        if setting == "ED":
            # prefer ED-labeled scanners first
            ed_first = eligible.sort_values(by="location", key=lambda s: (s != "ED").astype(int))
            return [str(x) for x in ed_first["scanner_id"].tolist()]
        if setting == "INPATIENT":
            # portable for inpatient already handled; otherwise prefer MAIN
            main_first = eligible.sort_values(by="location", key=lambda s: (s != "MAIN").astype(int))
            return [str(x) for x in main_first["scanner_id"].tolist()]
        # outpatient
        main_first = eligible.sort_values(by="location", key=lambda s: (s != "MAIN").astype(int))
        return [str(x) for x in main_first["scanner_id"].tolist()]

    def compute_planned_start(order_row: pd.Series) -> datetime:
        ot = pd.to_datetime(order_row["order_time_dt"])
        setting = str(order_row["patient_type"])
        modality_needed = str(order_row["resource_modality"])
        # Outpatient: schedule in future during business hours (use order_time + lead_days)
        if setting == "OUTPATIENT":
            lead = int(order_row["outpatient_lead_days"])
            day = (ot + timedelta(days=lead)).replace(hour=0, minute=0, second=0, microsecond=0)
            # choose a daytime slot 7:00–18:00
            hour = int(rng.integers(7, 18))
            minute = int(rng.choice([0, 10, 20, 30, 40, 50]))
            return day + timedelta(hours=hour, minutes=minute)
        # ED/IP: schedule soon after order time
        # ED STAT sooner:
        pr = str(order_row["priority"])
        if pr == "STAT":
            return ot + timedelta(minutes=int(rng.integers(0, 20)))
        if pr == "URGENT":
            return ot + timedelta(minutes=int(rng.integers(10, 90)))
        return ot + timedelta(minutes=int(rng.integers(30, 240)))

    # Capacity reservation: reserve a % of CT capacity for ED by blocking outpatient CT bookings
    # Implementation: for each CT scanner per day, define "reserved minutes" at the end of day.
    # Outpatient CT cannot schedule into the reserved tail.
    reserved_tail_minutes_per_ct = int(round(24 * 60 * config.ED_CT_reserve_capacity))

    for idx, o in orders_df.iterrows():
        setting = str(o["patient_type"])
        modality_needed = str(o["resource_modality"])

        planned_start = compute_planned_start(o)
        planned_duration = _sample_duration_minutes(rng, modality_needed)

        # add coarse buffers for contrast/sedation/isolation (educational)
        if int(o["requires_contrast"]) == 1 and modality_needed in ("CT", "MR"):
            planned_duration += int(rng.integers(8, 16))
        if int(o["requires_sedation"]) == 1 and modality_needed == "MR":
            planned_duration += int(rng.integers(20, 40))
        if int(o["isolation"]) == 1:
            planned_duration += int(rng.integers(5, 12))

        candidate_scanners = pick_scanner(modality_needed, setting)
        if not candidate_scanners:
            # should not happen if resources.csv is correct
            continue

        # Choose earliest available among eligible scanners
        best_sid = None
        best_start = None

        for sid in candidate_scanners:
            # Lazy init next free based on scanner's open time for that day
            # Ensure planned_start respects scanner opening
            r = resources.loc[resources["scanner_id"] == sid].iloc[0]
            open_dt, close_dt = _scanner_day_window(r, planned_start.replace(hour=0, minute=0, second=0, microsecond=0))

            # Apply outpatient CT reserve tail capacity
            effective_close = close_dt
            if modality_needed == "CT" and setting == "OUTPATIENT" and reserved_tail_minutes_per_ct > 0:
                effective_close = close_dt - timedelta(minutes=reserved_tail_minutes_per_ct)

            key = (sid, planned_start.date())
            if sid not in scanner_next_free or scanner_next_free[sid].date() != planned_start.date():
                scanner_next_free[sid] = open_dt

            start_time = max(planned_start, scanner_next_free[sid], open_dt)
            end_time = start_time + timedelta(minutes=planned_duration)

            # If outpatient and would exceed effective close, push to next day at open
            if setting == "OUTPATIENT" and end_time > effective_close:
                # Move to next day open
                next_day = (planned_start + timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)
                open_dt2, close_dt2 = _scanner_day_window(r, next_day)
                effective_close2 = close_dt2
                if modality_needed == "CT" and reserved_tail_minutes_per_ct > 0:
                    effective_close2 = close_dt2 - timedelta(minutes=reserved_tail_minutes_per_ct)
                start_time = open_dt2
                end_time = start_time + timedelta(minutes=planned_duration)
                if end_time > effective_close2:
                    # if still impossible, skip (extreme edge)
                    continue

            if best_start is None or start_time < best_start:
                best_start = start_time
                best_sid = sid

        if best_sid is None or best_start is None:
            continue

        planned_end = best_start + timedelta(minutes=planned_duration)

        # Simple overbooking (educational):
        # estimate predicted no-show probability (not a model yet): higher with longer lead time
        pred_no_show = 0.0
        if setting == "OUTPATIENT":
            lead = int(o["outpatient_lead_days"])
            pred_no_show = min(0.95, max(0.01, config.OUTPATIENT_no_show_rate * (1.0 + 0.04 * lead)))
            # If high predicted no-show, allow mild overbooking by NOT advancing scanner_next_free fully
            # (creates tighter packing / overlap risk → bump/late starts in simulation)
            if pred_no_show >= config.overbooking_threshold:
                advance_minutes = int(round(planned_duration * 0.70))  # aggressive
            else:
                advance_minutes = planned_duration
        else:
            advance_minutes = planned_duration

        # Update scanner availability
        scanner_next_free[best_sid] = best_start + timedelta(minutes=advance_minutes)

        planned_rows.append({
            "sps_id": f"S{sps_counter:08d}",
            "order_id": str(o["order_id"]),
            "patient_type": setting,
            "resource_modality": modality_needed,
            "scanner_id": best_sid,
            "scheduled_start": best_start.isoformat(),
            "scheduled_end": planned_end.isoformat(),
            "planned_duration_min": planned_duration,
            "pred_no_show_proxy": round(pred_no_show, 4),
            "educational_note": "Synthetic planned schedule (not clinical)."
        })
        sps_counter += 1

    schedule_df = pd.DataFrame(planned_rows)

    # ---------- Simulate Actual Execution ----------
    # Build downtime events per day
    downtime_by_day: Dict[datetime.date, Dict[str, List[Tuple[datetime, datetime]]]] = {}
    for day_i in range(config.num_days):
        day = (config.start_date + timedelta(days=day_i)).replace(hour=0, minute=0, second=0, microsecond=0)
        downtime_by_day[day.date()] = _generate_downtime_events(
            rng=rng,
            resources=resources,
            day=day,
            ct_rate=config.CT_downtime_rate,
            mr_rate=config.MR_downtime_rate
        )

    # Simulate in scanner order
    schedule_df["scheduled_start_dt"] = pd.to_datetime(schedule_df["scheduled_start"])
    schedule_df = schedule_df.sort_values(["scanner_id", "scheduled_start_dt"]).reset_index(drop=True)

    exec_rows: List[dict] = []
    scanner_actual_next_free: Dict[Tuple[str, datetime.date], datetime] = {}

    # Create fast lookup for order fields
    orders_lookup = orders_df.set_index("order_id")

    for _, s in schedule_df.iterrows():
        sps_id = str(s["sps_id"])
        order_id = str(s["order_id"])
        scanner_id = str(s["scanner_id"])
        setting = str(s["patient_type"])
        modality = str(s["resource_modality"])

        scheduled_start = pd.to_datetime(s["scheduled_start_dt"]).to_pydatetime()
        planned_duration = int(s["planned_duration_min"])
        day_key = scheduled_start.date()

        # Initialize scanner day window and actual next free
        r = resources.loc[resources["scanner_id"] == scanner_id].iloc[0]
        open_dt, close_dt = _scanner_day_window(r, scheduled_start.replace(hour=0, minute=0, second=0, microsecond=0))
        key = (scanner_id, day_key)
        if key not in scanner_actual_next_free:
            scanner_actual_next_free[key] = open_dt

        # Determine arrival time
        order_row = orders_lookup.loc[order_id]
        order_time = pd.to_datetime(order_row["order_time_dt"]).to_pydatetime()

        # Outpatient: arrival around scheduled, may be late
        patient_arrival = scheduled_start
        if setting == "OUTPATIENT":
            late = int(max(0, rng.normal(3, 7)))  # average a bit late
            patient_arrival = scheduled_start + timedelta(minutes=late)
        else:
            # ED/IP: arrival after order time + transport/coordination
            base = max(order_time, scheduled_start - timedelta(minutes=15))
            if setting == "INPATIENT":
                transport = int(max(0, rng.normal(config.INPATIENT_transport_delay_min, 8)))
                patient_arrival = base + timedelta(minutes=transport)
            else:
                patient_arrival = base + timedelta(minutes=int(max(0, rng.normal(8, 6))))

        # No-show only for outpatient
        no_show = 0
        canceled = 0
        completed = 0

        actual_start = None
        actual_end = None
        delay_reason = "none"

        if setting == "OUTPATIENT":
            lead_days = int(order_row["outpatient_lead_days"])
            p = min(0.95, max(0.01, config.OUTPATIENT_no_show_rate * (1.0 + 0.04 * lead_days)))
            if rng.random() < p:
                no_show = 1
                delay_reason = "no_show"
                # no actual start/end
            else:
                # show
                completed = 1
        else:
            completed = 1  # ED/IP always show in this synthetic model

        if completed == 1:
            # Start after patient arrival AND after scanner is free
            start_candidate = max(patient_arrival, scanner_actual_next_free[key], open_dt, scheduled_start)

            # Apply downtime delays if overlap with downtime window
            d_events = downtime_by_day.get(day_key, {}).get(scanner_id, [])
            for (ds, de) in d_events:
                # if the scan would start during downtime, push it to downtime end
                if ds <= start_candidate < de:
                    start_candidate = de
                    delay_reason = "downtime"
                # if downtime overlaps the scan window, also push start (simplified)
                scan_end_candidate = start_candidate + timedelta(minutes=planned_duration)
                if _overlaps(start_candidate, scan_end_candidate, ds, de):
                    start_candidate = de
                    delay_reason = "downtime"

            # Add “ED interrupt” effect: ED STAT bumps can cause small delays to non-ED
            if setting == "OUTPATIENT" and modality in ("CT", "MR") and rng.random() < 0.10:
                # mild bump risk
                bump = int(rng.integers(5, 25))
                start_candidate += timedelta(minutes=bump)
                delay_reason = "bumped_by_urgent"

            if setting == "INPATIENT" and modality in ("CT", "MR", "US") and rng.random() < 0.08:
                extra = int(rng.integers(5, 20))
                start_candidate += timedelta(minutes=extra)
                delay_reason = "transport_or_prep"

            # Actual duration = planned +/- noise
            dur_noise = int(round(rng.normal(0, max(2, planned_duration * 0.08))))
            actual_duration = max(3, planned_duration + dur_noise)

            actual_start = start_candidate
            actual_end = actual_start + timedelta(minutes=actual_duration)

            # Clamp within day close? (Allow overtime by not clamping)
            scanner_actual_next_free[key] = actual_end

        exec_rows.append({
            "sps_id": sps_id,
            "order_id": order_id,
            "scanner_id": scanner_id,
            "patient_type": setting,
            "resource_modality": modality,
            "order_time": order_time.isoformat(),
            "scheduled_start": scheduled_start.isoformat(),
            "scheduled_end": pd.to_datetime(s["scheduled_end"]).to_pydatetime().isoformat(),
            "patient_arrival_time": patient_arrival.isoformat(),
            "actual_start": None if actual_start is None else actual_start.isoformat(),
            "actual_end": None if actual_end is None else actual_end.isoformat(),
            "completed": completed,
            "canceled": canceled,
            "no_show": no_show,
            "delay_reason": delay_reason,
            "educational_note": "Synthetic execution (not clinical)."
        })

    execution_df = pd.DataFrame(exec_rows)

    # ---------- Write outputs ----------
    out_dir = os.path.join(repo_root, "data", "raw")
    _ensure_dir(out_dir)

    # clean helper columns
    orders_out = orders_df.drop(columns=["priority_rank", "order_time_dt"])
    schedule_out = schedule_df.drop(columns=["scheduled_start_dt"])

    orders_out.to_csv(os.path.join(out_dir, "orders.csv"), index=False)
    schedule_out.to_csv(os.path.join(out_dir, "schedule_planned.csv"), index=False)
    execution_df.to_csv(os.path.join(out_dir, "execution_actual.csv"), index=False)

    print("Wrote:")
    print("- data/raw/orders.csv")
    print("- data/raw/schedule_planned.csv")
    print("- data/raw/execution_actual.csv")
    print("\nEducational Use Only (Non-Clinical).")


if __name__ == "__main__":
    main()
