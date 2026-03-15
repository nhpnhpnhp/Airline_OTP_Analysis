"""
transformations.py — Data-cleaning rules R1–R8 and the clean_chunk orchestrator.
"""
import numpy as np
import pandas as pd

from .config import (
    CANCEL_NULL_COLS, DIVERT_NULL_COLS, HHMM_COLS, DELAY_CAUSE_COLS,
)
from .utils import hhmm_to_minutes, cyclic_encode


def apply_r1_cancelled(df):
    """R1: Nullify operational fields for cancelled flights."""
    mask = df["CANCELLED"] == 1
    n = mask.sum()
    div_cols = [c for c in df.columns if c.startswith("DIV_")]
    for c in CANCEL_NULL_COLS + div_cols:
        if c in df.columns:
            df.loc[mask, c] = pd.NA if df[c].dtype.name.startswith(("Int", "string")) else np.nan
    return df, n


def apply_r2_diverted(df):
    """R2: Nullify arrival outcomes for diverted flights."""
    mask = df["DIVERTED"] == 1
    n = mask.sum()
    for c in DIVERT_NULL_COLS:
        if c in df.columns:
            df.loc[mask, c] = pd.NA if df[c].dtype.name.startswith(("Int", "string")) else np.nan
    return df, n


def apply_r3_hhmm(df, stats):
    """R3: Convert HHMM columns to minutes + cyclic for scheduled times."""
    for c in HHMM_COLS:
        if c not in df.columns:
            continue
        mc = c + "_MIN"
        df[mc] = hhmm_to_minutes(df[c])
        parsed = df[mc].notna().sum()
        na = df[mc].isna().sum()
        stats.setdefault(c, {"parsed": 0, "na": 0})
        stats[c]["parsed"] += int(parsed)
        stats[c]["na"] += int(na)
        df[mc] = df[mc].astype("Int32")
    # Cyclic for scheduled
    if "CRS_DEP_TIME_MIN" in df.columns:
        s, co = cyclic_encode(df["CRS_DEP_TIME_MIN"].astype("float32"))
        df["CRS_DEP_SIN"] = s; df["CRS_DEP_COS"] = co
    if "CRS_ARR_TIME_MIN" in df.columns:
        s, co = cyclic_encode(df["CRS_ARR_TIME_MIN"].astype("float32"))
        df["CRS_ARR_SIN"] = s; df["CRS_ARR_COS"] = co
    return df


def apply_r4_date(df, dow_mismatches):
    """R4: Create FL_DATE, validate DAY_OF_WEEK."""
    df["FL_DATE"] = pd.to_datetime(
        df["YEAR"].astype(str) + "-01-" + df["DAY_OF_MONTH"].astype(str),
        errors="coerce")
    computed_dow = df["FL_DATE"].dt.dayofweek.astype("Int16")  # Mon=0..Sun=6
    # BTS uses Mon=1..Sun=7, convert: bts_dow - 1 for comparison
    bts_dow = df["DAY_OF_WEEK"] - 1  # transform to 0-based
    mismatch = (bts_dow != computed_dow) & bts_dow.notna() & computed_dow.notna()
    n = int(mismatch.sum())
    dow_mismatches["count"] = dow_mismatches.get("count", 0) + n
    # Overwrite with computed (0-based Mon=0)
    df["DAY_OF_WEEK"] = computed_dow
    return df


def apply_r5_weekend(df):
    """R5: Flag weekend flights."""
    df["IS_WEEKEND"] = (df["DAY_OF_WEEK"].isin({5, 6})).astype("Int16")
    return df


def apply_r6_route(df):
    """R6: Create ORIGIN-DEST route column."""
    df["ROUTE"] = df["ORIGIN"].astype(str) + "-" + df["DEST"].astype(str)
    return df


def apply_r7_delay_cat(df, target_col):
    """R7: Bin arrival delay into categories."""
    src = "ARR_DELAY" if target_col == "ARR_DEL15" else "ARR_DELAY_NEW"
    if src in df.columns:
        bins = [-np.inf, 0, 15, 60, 180, np.inf]
        labels = ["<=0", "1-15", "16-60", "61-180", ">180"]
        df["ARR_DELAY_CAT"] = pd.cut(df[src].astype("float64"), bins=bins, labels=labels)
    return df


def apply_r8_dominant(df):
    """R8: Identify the dominant delay cause."""
    avail = [c for c in DELAY_CAUSE_COLS if c in df.columns]
    if not avail:
        return df
    sub = df[avail].astype("float32")
    has_any = sub.notna().any(axis=1) & (sub.fillna(0).sum(axis=1) > 0)
    dom = sub.idxmax(axis=1).where(has_any, other=pd.NA)
    df["DOMINANT_DELAY_CAUSE"] = dom.astype("string")
    return df


def clean_chunk(df, target_col, hhmm_stats, dow_mismatches, rule_counts):
    """Apply R1–R8 to a chunk."""
    df, n1 = apply_r1_cancelled(df)
    rule_counts["R1"] = rule_counts.get("R1", 0) + n1
    df, n2 = apply_r2_diverted(df)
    rule_counts["R2"] = rule_counts.get("R2", 0) + n2
    apply_r3_hhmm(df, hhmm_stats)
    rule_counts["R3"] = rule_counts.get("R3", 0) + len(df)
    apply_r4_date(df, dow_mismatches)
    rule_counts["R4"] = rule_counts.get("R4", 0) + len(df)
    apply_r5_weekend(df)
    apply_r6_route(df)
    apply_r7_delay_cat(df, target_col)
    apply_r8_dominant(df)
    return df
