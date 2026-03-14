#!/usr/bin/env python3
"""
preprocess_flights.py — Two-pass out-of-core preprocessing pipeline for
US Airline On-Time Performance data (January 2021–2025).

Usage:
  python preprocess_flights.py \
    --inputs data/raw/T_ONTIME_REPORTING_2021.csv ... \
    --chunksize 300000 --target ARR_DEL15 \
    --out_dir data/processed --report reports/quality_report.md
"""
import argparse, hashlib, json, logging, os, sys, time, warnings
from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

warnings.filterwarnings("ignore", category=FutureWarning)
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger("preprocess")

# ── defaults ────────────────────────────────────────────────────────────
FILES = [
    "data/raw/T_ONTIME_REPORTING_2021.csv",
    "data/raw/T_ONTIME_REPORTING_2022.csv",
    "data/raw/T_ONTIME_REPORTING_2023.csv",
    "data/raw/T_ONTIME_REPORTING_2024.csv",
    "data/raw/T_ONTIME_REPORTING_2025.csv",
]
CHUNKSIZE = 300_000
TARGET = "ARR_DEL15"
# TRAIN_YEARS is used by Pass 1 (Data Cleaning) to build freq/OTP mappings.
TRAIN_YEARS = {2021, 2022, 2023, 2024}
# ── [ML DISABLED] TEST_YEARS — only used for ML train/test split ─────
# TEST_YEARS = {2025}

CANCEL_NULL_COLS = [
    "DEP_TIME", "ARR_TIME", "WHEELS_OFF", "WHEELS_ON", "TAXI_OUT", "TAXI_IN",
    "ACTUAL_ELAPSED_TIME", "AIR_TIME",
    "DEP_DELAY", "DEP_DELAY_NEW", "DEP_DEL15", "DEP_DELAY_GROUP",
    "ARR_DELAY", "ARR_DELAY_NEW", "ARR_DEL15", "ARR_DELAY_GROUP",
    "CARRIER_DELAY", "WEATHER_DELAY", "NAS_DELAY", "SECURITY_DELAY", "LATE_AIRCRAFT_DELAY",
    "FIRST_DEP_TIME", "TOTAL_ADD_GTIME", "LONGEST_ADD_GTIME",
]
DIVERT_NULL_COLS = [
    "ARR_TIME", "ARR_DELAY", "ARR_DELAY_NEW", "ARR_DEL15", "ARR_DELAY_GROUP", "ARR_TIME_BLK",
    "WHEELS_ON", "TAXI_IN",
]
HHMM_COLS = ["CRS_DEP_TIME", "DEP_TIME", "WHEELS_OFF", "WHEELS_ON",
             "CRS_ARR_TIME", "ARR_TIME", "FIRST_DEP_TIME"]
DELAY_CAUSE_COLS = ["CARRIER_DELAY", "WEATHER_DELAY", "NAS_DELAY",
                    "SECURITY_DELAY", "LATE_AIRCRAFT_DELAY"]
FREQ_ENCODE_COLS = ["OP_CARRIER", "ORIGIN", "DEST", "ROUTE", "DEP_TIME_BLK"]
OTP_GROUP_COLS = ["ORIGIN", "OP_CARRIER"]

# ── [ML DISABLED] Track A/B feature definitions & forbidden sets ─────
"""
TRACK_A_FEATURES = [
    "YEAR", "DAY_OF_MONTH", "DAY_OF_WEEK", "IS_WEEKEND",
    "CRS_DEP_TIME_MIN", "CRS_ARR_TIME_MIN",
    "CRS_DEP_SIN", "CRS_DEP_COS", "CRS_ARR_SIN", "CRS_ARR_COS",
    "CRS_ELAPSED_TIME", "DISTANCE", "DISTANCE_GROUP",
    "OP_CARRIER_FREQ", "CARRIER_HIST_OTP",
    "ORIGIN_FREQ", "ORIGIN_HIST_OTP", "DEST_FREQ",
    "ROUTE_FREQ", "DEP_TIME_BLK_FREQ",
]
TRACK_B_EXTRA = ["DEP_DELAY", "DEP_DELAY_NEW", "DEP_DEL15", "TAXI_OUT"]

TRACK_A_FORBIDDEN = {
    "DEP_TIME", "DEP_DELAY", "DEP_DELAY_NEW", "DEP_DEL15", "DEP_DELAY_GROUP",
    "TAXI_OUT", "WHEELS_OFF", "WHEELS_ON", "TAXI_IN",
    "ARR_TIME", "ARR_DELAY", "ARR_DELAY_NEW", "ARR_DEL15", "ARR_DELAY_GROUP",
    "ACTUAL_ELAPSED_TIME", "AIR_TIME",
    "CARRIER_DELAY", "WEATHER_DELAY", "NAS_DELAY", "SECURITY_DELAY", "LATE_AIRCRAFT_DELAY",
    "FIRST_DEP_TIME", "TOTAL_ADD_GTIME", "LONGEST_ADD_GTIME",
    "CANCELLED", "DIVERTED",
}
TRACK_B_FORBIDDEN = {
    "ARR_TIME", "ARR_DELAY", "ARR_DELAY_NEW", "ARR_DEL15", "ARR_DELAY_GROUP",
    "ARR_TIME_BLK", "ACTUAL_ELAPSED_TIME", "AIR_TIME",
    "CARRIER_DELAY", "WEATHER_DELAY", "NAS_DELAY", "SECURITY_DELAY", "LATE_AIRCRAFT_DELAY",
    "WHEELS_ON", "TAXI_IN", "FIRST_DEP_TIME", "TOTAL_ADD_GTIME", "LONGEST_ADD_GTIME",
    "CANCELLED", "DIVERTED",
}
"""

# ── helpers ─────────────────────────────────────────────────────────────

def hash_inputs(paths):
    h = hashlib.md5()
    for p in sorted(paths):
        h.update(str(p).encode())
    return h.hexdigest()


def hhmm_to_minutes(s):
    """R3: Robust HHMM→minutes parser. Handles 600, '0600', blanks."""
    v = pd.to_numeric(s, errors="coerce")
    hh = (v // 100).astype("Int32")
    mm = (v % 100).astype("Int32")
    result = hh * 60 + mm
    result = result.where((result >= 0) & (result <= 1439), other=pd.NA)
    return result


def cyclic_encode(minutes_col, period=1440):
    rad = 2 * np.pi * minutes_col / period
    return np.sin(rad).astype("float32"), np.cos(rad).astype("float32")


def standardize_columns(df):
    df.columns = df.columns.str.strip().str.upper()
    # Drop trailing unnamed columns (BTS artefact)
    df = df[[c for c in df.columns if not c.startswith("UNNAMED")]]
    return df


DTYPE_MAP_INT = {
    "YEAR": "Int16", "DAY_OF_MONTH": "Int16", "DAY_OF_WEEK": "Int16",
    "OP_CARRIER_AIRLINE_ID": "Int32", "OP_CARRIER_FL_NUM": "Int32",
    "ORIGIN_AIRPORT_ID": "Int32", "ORIGIN_AIRPORT_SEQ_ID": "Int32",
    "ORIGIN_CITY_MARKET_ID": "Int32", "ORIGIN_STATE_FIPS": "Int16", "ORIGIN_WAC": "Int16",
    "DEST_AIRPORT_ID": "Int32", "DEST_AIRPORT_SEQ_ID": "Int32",
    "DEST_CITY_MARKET_ID": "Int32", "DEST_STATE_FIPS": "Int16", "DEST_WAC": "Int16",
    "DEP_DEL15": "Int16", "DEP_DELAY_GROUP": "Int16",
    "ARR_DEL15": "Int16", "ARR_DELAY_GROUP": "Int16",
    "CANCELLED": "Int16", "DIVERTED": "Int16",
    "DISTANCE_GROUP": "Int16", "FLIGHTS": "Int16",
    "DIV_AIRPORT_LANDINGS": "Int16", "DIV_REACHED_DEST": "Int16",
}
DTYPE_MAP_FLOAT = {
    "DEP_DELAY": "float32", "DEP_DELAY_NEW": "float32",
    "ARR_DELAY": "float32", "ARR_DELAY_NEW": "float32",
    "TAXI_OUT": "float32", "TAXI_IN": "float32",
    "CRS_ELAPSED_TIME": "float32", "ACTUAL_ELAPSED_TIME": "float32",
    "AIR_TIME": "float32", "DISTANCE": "float32",
    "CARRIER_DELAY": "float32", "WEATHER_DELAY": "float32",
    "NAS_DELAY": "float32", "SECURITY_DELAY": "float32", "LATE_AIRCRAFT_DELAY": "float32",
    "TOTAL_ADD_GTIME": "float32", "LONGEST_ADD_GTIME": "float32",
    "DIV_ACTUAL_ELAPSED_TIME": "float32", "DIV_ARR_DELAY": "float32", "DIV_DISTANCE": "float32",
}
STR_COLS = {"OP_UNIQUE_CARRIER", "OP_CARRIER", "TAIL_NUM", "ORIGIN", "ORIGIN_CITY_NAME",
            "ORIGIN_STATE_ABR", "ORIGIN_STATE_NM", "DEST", "DEST_CITY_NAME",
            "DEST_STATE_ABR", "DEST_STATE_NM", "DEP_TIME_BLK", "ARR_TIME_BLK",
            "CANCELLATION_CODE"}


def cast_dtypes(df):
    for c, dt in DTYPE_MAP_INT.items():
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").astype(dt)
    for c, dt in DTYPE_MAP_FLOAT.items():
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").astype(dt)
    for c in STR_COLS:
        if c in df.columns:
            df[c] = df[c].astype("string")
    return df


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
    df["IS_WEEKEND"] = (df["DAY_OF_WEEK"].isin({5, 6})).astype("Int16")
    return df


def apply_r6_route(df):
    df["ROUTE"] = df["ORIGIN"].astype(str) + "-" + df["DEST"].astype(str)
    return df


def apply_r7_delay_cat(df, target_col):
    src = "ARR_DELAY" if target_col == "ARR_DEL15" else "ARR_DELAY_NEW"
    if src in df.columns:
        bins = [-np.inf, 0, 15, 60, 180, np.inf]
        labels = ["<=0", "1-15", "16-60", "61-180", ">180"]
        df["ARR_DELAY_CAT"] = pd.cut(df[src].astype("float64"), bins=bins, labels=labels)
    return df


def apply_r8_dominant(df):
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


# ── PASS 1 ──────────────────────────────────────────────────────────────

def run_pass1(input_files, chunksize, target_col, out_dir):
    """Build frequency maps and OTP stats from train years (2021-2024)."""
    mappings_dir = Path(out_dir) / "mappings"
    mappings_dir.mkdir(parents=True, exist_ok=True)
    meta_path = mappings_dir / "_meta.json"

    inp_hash = hash_inputs(input_files)
    if meta_path.exists():
        meta = json.loads(meta_path.read_text())
        if (meta.get("completed_pass1") and meta.get("inputs_hash") == inp_hash
                and meta.get("chunksize") == chunksize and meta.get("target") == target_col):
            log.info("Pass 1 checkpoint found & valid — skipping.")
            return load_mappings(mappings_dir)
        log.info("Pass 1 checkpoint stale — re-running.")

    log.info("=== PASS 1: Building train-only mappings (years 2021–2024) ===")
    freq_counts = {c: {} for c in FREQ_ENCODE_COLS}
    otp_counts = {c: {} for c in OTP_GROUP_COLS}  # key -> [total, ontime]
    total_train_operated = 0

    train_files = [f for f in input_files if any(str(y) in str(f) for y in TRAIN_YEARS)]
    for fpath in train_files:
        log.info(f"  Pass1 reading {fpath}")
        for ci, chunk in enumerate(pd.read_csv(fpath, chunksize=chunksize, low_memory=False)):
            chunk = standardize_columns(chunk)
            chunk = cast_dtypes(chunk)
            # R1/R2 masking for correct OTP
            chunk, _ = apply_r1_cancelled(chunk)
            chunk, _ = apply_r2_diverted(chunk)
            apply_r3_hhmm(chunk, {})  # need scheduled times parsed
            apply_r4_date(chunk, {})
            apply_r6_route(chunk)
            # Operated only for OTP
            operated = chunk[(chunk["CANCELLED"] == 0) & (chunk["DIVERTED"] == 0)]
            total_train_operated += len(operated)
            # Freq counts (ALL train rows, not just operated)
            for col in FREQ_ENCODE_COLS:
                if col in chunk.columns:
                    vc = chunk[col].value_counts()
                    for k, v in vc.items():
                        freq_counts[col][k] = freq_counts[col].get(k, 0) + int(v)
            # OTP counts (operated only)
            for col in OTP_GROUP_COLS:
                if col in operated.columns and target_col in operated.columns:
                    grp = operated.groupby(col)[target_col].agg(["count", "sum"])
                    for k, row in grp.iterrows():
                        prev = otp_counts[col].get(k, [0, 0])
                        otp_counts[col][k] = [prev[0] + int(row["count"]),
                                              prev[1] + int(row["sum"])]
            if (ci + 1) % 3 == 0:
                log.info(f"    chunk {ci+1} done")

    # Build normalised freq maps
    freq_maps = {}
    for col, counts in freq_counts.items():
        total = sum(counts.values()) or 1
        freq_maps[col] = {k: v / total for k, v in counts.items()}

    # Build OTP maps: mean(1 - ARR_DEL15) = (total - delayed) / total
    otp_maps = {}
    global_total = 0; global_ontime_sum = 0
    for col, mapping in otp_counts.items():
        otp_maps[col] = {}
        for k, (cnt, delayed) in mapping.items():
            otp_maps[col][k] = (cnt - delayed) / cnt if cnt > 0 else 0.5
            global_total += cnt; global_ontime_sum += (cnt - delayed)
    global_otp = global_ontime_sum / global_total if global_total > 0 else 0.5
    # de-duplicate: global counted twice (once per group col)
    global_otp = (total_train_operated - sum(v[1] for v in otp_counts["ORIGIN"].values())) / total_train_operated if total_train_operated > 0 else 0.5

    # Save mappings
    for col, m in freq_maps.items():
        pd.DataFrame(list(m.items()), columns=["key", "freq"]).to_parquet(
            mappings_dir / f"freq_{col}.parquet", index=False)
    for col, m in otp_maps.items():
        pd.DataFrame(list(m.items()), columns=["key", "otp"]).to_parquet(
            mappings_dir / f"otp_{col}.parquet", index=False)
    # Save global OTP
    (mappings_dir / "global_otp.json").write_text(json.dumps({"global_otp": global_otp}))

    meta = {"completed_pass1": True, "inputs_hash": inp_hash,
            "chunksize": chunksize, "target": target_col,
            "total_train_operated": total_train_operated}
    meta_path.write_text(json.dumps(meta, indent=2))
    log.info(f"Pass 1 complete. Train operated rows: {total_train_operated:,}")
    return freq_maps, otp_maps, global_otp


def load_mappings(mappings_dir):
    freq_maps = {}
    for col in FREQ_ENCODE_COLS:
        p = mappings_dir / f"freq_{col}.parquet"
        if p.exists():
            tmp = pd.read_parquet(p)
            freq_maps[col] = dict(zip(tmp["key"], tmp["freq"]))
    otp_maps = {}
    for col in OTP_GROUP_COLS:
        p = mappings_dir / f"otp_{col}.parquet"
        if p.exists():
            tmp = pd.read_parquet(p)
            otp_maps[col] = dict(zip(tmp["key"], tmp["otp"]))
    gp = mappings_dir / "global_otp.json"
    global_otp = json.loads(gp.read_text())["global_otp"] if gp.exists() else 0.5
    return freq_maps, otp_maps, global_otp


# ── PASS 2 ──────────────────────────────────────────────────────────────

# ── [ML DISABLED] apply_freq_otp — frequency/OTP encoding for ML ─────
"""
def apply_freq_otp(df, freq_maps, otp_maps, global_otp, unseen_log):
    \"\"\"R10: Apply frequency encoding and historical OTP from train mappings.\"\"\"
    col_map = {"OP_CARRIER": "OP_CARRIER_FREQ", "ORIGIN": "ORIGIN_FREQ",
               "DEST": "DEST_FREQ", "ROUTE": "ROUTE_FREQ", "DEP_TIME_BLK": "DEP_TIME_BLK_FREQ"}
    for src, dst in col_map.items():
        if src in df.columns and src in freq_maps:
            mapped = df[src].map(freq_maps[src])
            unseen = mapped.isna() & df[src].notna()
            unseen_log.setdefault(src, {"total": 0, "unseen": 0})
            unseen_log[src]["total"] += int(df[src].notna().sum())
            unseen_log[src]["unseen"] += int(unseen.sum())
            df[dst] = mapped.fillna(0).astype("float32")
    otp_col_map = {"ORIGIN": "ORIGIN_HIST_OTP", "OP_CARRIER": "CARRIER_HIST_OTP"}
    for src, dst in otp_col_map.items():
        if src in df.columns and src in otp_maps:
            mapped = df[src].map(otp_maps[src])
            unseen = mapped.isna() & df[src].notna()
            unseen_log.setdefault(f"otp_{src}", {"total": 0, "unseen": 0})
            unseen_log[f"otp_{src}"]["total"] += int(df[src].notna().sum())
            unseen_log[f"otp_{src}"]["unseen"] += int(unseen.sum())
            df[dst] = mapped.fillna(global_otp).astype("float32")
    return df
"""


def write_parquet_partition(df, base_dir, year_val, writers_state):
    """Append rows to a partitioned parquet file for the given year."""
    part_dir = Path(base_dir) / f"YEAR={year_val}"
    part_dir.mkdir(parents=True, exist_ok=True)
    fpath = part_dir / "part-0.parquet"
    # Drop YEAR column from data (it's in the partition path)
    df_write = df.drop(columns=["YEAR"], errors="ignore")
    table = pa.Table.from_pandas(df_write, preserve_index=False)
    key = str(fpath)
    if key not in writers_state:
        writers_state[key] = pq.ParquetWriter(str(fpath), table.schema, compression="snappy")
    try:
        writers_state[key].write_table(table)
    except (pa.ArrowInvalid, pa.ArrowTypeError):
        # Schema evolved — close old writer, rewrite
        writers_state[key].close()
        writers_state[key] = pq.ParquetWriter(str(fpath), table.schema, compression="snappy")
        writers_state[key].write_table(table)


# ── [ML DISABLED] build_ml_rows — build Track A / Track B datasets ───
"""
def build_ml_rows(df, target_col, freq_maps, otp_maps, global_otp, unseen_log):
    \"\"\"From operated chunk, build Track A / Track B rows.\"\"\"
    # Apply freq/otp
    df = apply_freq_otp(df.copy(), freq_maps, otp_maps, global_otp, unseen_log)
    # Drop rows with missing target
    df = df[df[target_col].notna()].copy()
    if df.empty:
        return None, None

    # Track A
    a_cols = [c for c in TRACK_A_FEATURES if c in df.columns] + [target_col]
    track_a = df[a_cols].copy()

    # Track B
    b_cols = [c for c in TRACK_A_FEATURES + TRACK_B_EXTRA if c in df.columns] + [target_col]
    track_b = df[b_cols].copy()

    return track_a, track_b
"""


def run_pass2(input_files, chunksize, target_col, out_dir, overwrite):
    """Full cleaning + write all outputs (Data Cleaning only)."""
    log.info("=== PASS 2: Full cleaning & output generation ===")
    clean_full_dir = Path(out_dir) / "clean_full"
    clean_op_dir = Path(out_dir) / "clean_operated"
    # [ML DISABLED] ML output directories
    # ml_a_dir = Path(out_dir) / "ml_track_a"
    # ml_b_dir = Path(out_dir) / "ml_track_b"
    for d in [clean_full_dir, clean_op_dir]:
        d.mkdir(parents=True, exist_ok=True)

    # If overwrite, remove existing partitions
    if overwrite:
        import shutil
        for d in [clean_full_dir, clean_op_dir]:
            for sub in d.glob("YEAR=*"):
                shutil.rmtree(sub, ignore_errors=True)

    hhmm_stats = {}
    dow_mismatches = {"count": 0}
    rule_counts = {}
    year_stats = {}  # year -> {rows_read, rows_full, rows_operated, cancelled, diverted}
    writers_full = {}
    writers_op = {}

    # [ML DISABLED] ML accumulators: collect train/test
    # unseen_log = {}
    # ml_a_train, ml_a_test = [], []
    # ml_b_train, ml_b_test = [], []
    consistency = {"distance_neg": 0, "time_min_oor": 0}

    for fpath in input_files:
        log.info(f"  Pass2 reading {fpath}")
        for ci, chunk in enumerate(pd.read_csv(fpath, chunksize=chunksize, low_memory=False)):
            chunk = standardize_columns(chunk)
            chunk = cast_dtypes(chunk)

            # Determine year
            if "YEAR" not in chunk.columns or chunk["YEAR"].isna().all():
                log.warning(f"  chunk {ci} has no YEAR column, skipping")
                continue
            years_in_chunk = chunk["YEAR"].dropna().unique()

            chunk = clean_chunk(chunk, target_col, hhmm_stats, dow_mismatches, rule_counts)

            # Consistency: distance < 0
            if "DISTANCE" in chunk.columns:
                consistency["distance_neg"] += int((chunk["DISTANCE"] < 0).sum())

            # Per-year stats
            for yr in years_in_chunk:
                yr = int(yr)
                yr_mask = chunk["YEAR"] == yr
                yr_chunk = chunk[yr_mask]
                st = year_stats.setdefault(yr, {"rows_read": 0, "rows_full": 0,
                    "rows_operated": 0, "cancelled": 0, "diverted": 0})
                st["rows_read"] += len(yr_chunk)
                st["rows_full"] += len(yr_chunk)
                st["cancelled"] += int((yr_chunk["CANCELLED"] == 1).sum()) if "CANCELLED" in yr_chunk.columns else 0
                st["diverted"] += int((yr_chunk["DIVERTED"] == 1).sum()) if "DIVERTED" in yr_chunk.columns else 0

                # Write clean_full
                write_parquet_partition(yr_chunk, clean_full_dir, yr, writers_full)

                # Operated
                op = yr_chunk[(yr_chunk["CANCELLED"] == 0) & (yr_chunk["DIVERTED"] == 0)]
                st["rows_operated"] += len(op)
                if not op.empty:
                    write_parquet_partition(op, clean_op_dir, yr, writers_op)

                    # [ML DISABLED] ML datasets
                    # ta, tb = build_ml_rows(op, target_col, freq_maps, otp_maps, global_otp, unseen_log)
                    # if ta is not None:
                    #     if yr in TRAIN_YEARS:
                    #         ml_a_train.append(ta); ml_b_train.append(tb)
                    #     else:
                    #         ml_a_test.append(ta); ml_b_test.append(tb)

            if (ci + 1) % 3 == 0:
                log.info(f"    chunk {ci+1} done")

    # Close writers
    for w in list(writers_full.values()) + list(writers_op.values()):
        w.close()

    # [ML DISABLED] Write ML datasets
    # ml_counts = {}
    # for name, parts, d in [("ml_track_a_train", ml_a_train, ml_a_dir),
    #                         ("ml_track_a_test", ml_a_test, ml_a_dir),
    #                         ("ml_track_b_train", ml_b_train, ml_b_dir),
    #                         ("ml_track_b_test", ml_b_test, ml_b_dir)]:
    #     if parts:
    #         combined = pd.concat(parts, ignore_index=True)
    #         out_path = d / f"{name}.parquet"
    #         combined.to_parquet(out_path, engine="pyarrow", compression="snappy", index=False)
    #         ml_counts[name] = len(combined)
    #         log.info(f"  Wrote {name}: {len(combined):,} rows, cols={list(combined.columns)}")
    #     else:
    #         ml_counts[name] = 0
    #         log.warning(f"  {name} has 0 rows!")

    log.info("Pass 2 complete.")
    return {
        "year_stats": year_stats, "hhmm_stats": hhmm_stats,
        "dow_mismatches": dow_mismatches, "rule_counts": rule_counts,
        "consistency": consistency,
    }


# ── QUALITY REPORT ──────────────────────────────────────────────────────

def generate_report(report_path, input_files, chunksize, target_col, out_dir,
                    stats, start_time, end_time):
    Path(report_path).parent.mkdir(parents=True, exist_ok=True)
    ys = stats["year_stats"]
    lines = []
    a = lines.append

    a("# Quality Report — Airline OTP Preprocessing Pipeline\n")
    a(f"Generated: {datetime.now().isoformat()}\n")

    # Section 1: run config
    a("## 1. Run Configuration\n")
    a(f"- **Machine**: local (~16 GB RAM)")
    a(f"- **Python**: {sys.version.split()[0]}")
    a(f"- **pandas**: {pd.__version__}")
    a(f"- **pyarrow**: {pa.__version__}")
    a(f"- **Files**: {', '.join(str(f) for f in input_files)}")
    a(f"- **Chunksize**: {chunksize:,}")
    a(f"- **Target**: `{target_col}`")
    a(f"- **Start**: {start_time}")
    a(f"- **End**: {end_time}")
    elapsed = (datetime.fromisoformat(end_time) - datetime.fromisoformat(start_time)).total_seconds()
    a(f"- **Runtime**: {elapsed:.1f}s\n")

    # Section 2: ingestion
    a("## 2. Ingestion Summary (per YEAR)\n")
    a("| YEAR | Rows Read | Rows Full | Rows Operated | % Cancelled | % Diverted |")
    a("|------|-----------|-----------|---------------|-------------|------------|")
    for yr in sorted(ys.keys()):
        s = ys[yr]
        pct_c = s["cancelled"] / s["rows_read"] * 100 if s["rows_read"] else 0
        pct_d = s["diverted"] / s["rows_read"] * 100 if s["rows_read"] else 0
        a(f"| {yr} | {s['rows_read']:,} | {s['rows_full']:,} | {s['rows_operated']:,} | {pct_c:.2f}% | {pct_d:.2f}% |")
    tot_read = sum(s["rows_read"] for s in ys.values())
    tot_full = sum(s["rows_full"] for s in ys.values())
    tot_op = sum(s["rows_operated"] for s in ys.values())
    a(f"| **Total** | **{tot_read:,}** | **{tot_full:,}** | **{tot_op:,}** | | |\n")

    # Section 3: schema — read one output to get dtypes
    a("## 3. Schema & Dtype Table (clean_full)\n")
    try:
        sample_dir = Path(out_dir) / "clean_full"
        parts = list(sample_dir.rglob("*.parquet"))
        if parts:
            sample_df = pd.read_parquet(parts[0], engine="pyarrow")
            a("| Column | Dtype | % Missing (sample partition) |")
            a("|--------|-------|-----------------------------|")
            for c in sample_df.columns:
                miss = sample_df[c].isna().mean() * 100
                a(f"| {c} | {sample_df[c].dtype} | {miss:.2f}% |")
            a("")
            derived = [c for c in sample_df.columns if c in {
                "FL_DATE", "IS_WEEKEND", "ROUTE", "ARR_DELAY_CAT", "DOMINANT_DELAY_CAUSE",
                "CRS_DEP_TIME_MIN", "CRS_ARR_TIME_MIN", "DEP_TIME_MIN", "ARR_TIME_MIN",
                "WHEELS_OFF_MIN", "WHEELS_ON_MIN", "FIRST_DEP_TIME_MIN",
                "CRS_DEP_SIN", "CRS_DEP_COS", "CRS_ARR_SIN", "CRS_ARR_COS"}]
            a(f"**Derived columns**: {', '.join(derived)}\n")
    except Exception as e:
        a(f"*Could not read sample partition: {e}*\n")

    # Section 4: transformation log
    a("## 4. Transformation Log (R1–R10)\n")
    rc = stats["rule_counts"]
    a(f"- **R1** Cancelled nullification: {rc.get('R1', 0):,} rows affected")
    a(f"- **R2** Diverted nullification: {rc.get('R2', 0):,} rows affected")
    a(f"- **R3** HHMM parsing: {rc.get('R3', 0):,} rows processed")
    a(f"- **R4** FL_DATE / DOW validation: {rc.get('R4', 0):,} rows processed")
    a(f"- **R5** IS_WEEKEND: applied to all rows")
    a(f"- **R6** ROUTE: applied to all rows")
    a(f"- **R7** ARR_DELAY_CAT: applied to operated rows")
    a(f"- **R8** DOMINANT_DELAY_CAUSE: applied to delayed rows")
    a(f"- **R9** Operated subset: filtered per year (see Section 2)")
    a(f"- **R10** Freq/OTP encoding: from train mappings\n")
    a("### HHMM Parsing Stats\n")
    a("| Column | Parsed | NA |")
    a("|--------|--------|----|")
    for c, v in stats["hhmm_stats"].items():
        a(f"| {c} | {v['parsed']:,} | {v['na']:,} |")
    a("")

    # Section 5: consistency
    a("## 5. Consistency Checks\n")
    a(f"- **DAY_OF_WEEK mismatches** (BTS vs computed): {stats['dow_mismatches']['count']:,}")
    a(f"  - Decision: overwrite with computed (Mon=0..Sun=6)")
    a(f"- **DISTANCE < 0**: {stats['consistency']['distance_neg']:,}")
    a(f"  - Action: kept as-is (data quality flag)\n")

    # [ML DISABLED] Section 6: ML dataset audit
    # a("## 6. ML Dataset Audit\n")
    # mc = stats["ml_counts"]
    # a(f"### Row Counts")
    # a(f"- Track A train: {mc.get('ml_track_a_train', 0):,}")
    # a(f"- Track A test:  {mc.get('ml_track_a_test', 0):,}")
    # a(f"- Track B train: {mc.get('ml_track_b_train', 0):,}")
    # a(f"- Track B test:  {mc.get('ml_track_b_test', 0):,}\n")
    # a(f"### Track A Features")
    # a(f"```\n{TRACK_A_FEATURES}\n```\n")
    # a(f"### Track B Features")
    # a(f"```\n{TRACK_A_FEATURES + TRACK_B_EXTRA}\n```\n")
    # a(f"### Leakage Columns Removed")
    # a(f"- Track A forbidden: `{sorted(TRACK_A_FORBIDDEN)}`")
    # a(f"- Track B forbidden: `{sorted(TRACK_B_FORBIDDEN)}`\n")
    # a(f"### Unseen Category Rates (2025 test)\n")
    # a("| Source | Total | Unseen | Rate |")
    # a("|--------|-------|--------|------|")
    # for k, v in stats["unseen_log"].items():
    #     rate = v["unseen"] / v["total"] * 100 if v["total"] else 0
    #     a(f"| {k} | {v['total']:,} | {v['unseen']:,} | {rate:.3f}% |")
    # a(f"\nUnseen values mapped to 0 (freq) or global OTP mean (OTP).\n")

    # Section 6: artifacts (renumbered from 7)
    a("## 6. Artifacts Produced\n")
    a("| Directory | Description | Partition |")
    a("|-----------|-------------|-----------|")
    a(f"| `{out_dir}/clean_full/` | Full cleaned data | `YEAR=YYYY/part-0.parquet` |")
    a(f"| `{out_dir}/clean_operated/` | Operated only | `YEAR=YYYY/part-0.parquet` |")
    # [ML DISABLED] ML artifact rows
    # a(f"| `{out_dir}/ml_track_a/` | ML pre-flight | `ml_track_a_train.parquet`, `ml_track_a_test.parquet` |")
    # a(f"| `{out_dir}/ml_track_b/` | ML post-pushback | `ml_track_b_train.parquet`, `ml_track_b_test.parquet` |")
    a(f"| `{out_dir}/mappings/` | Train-only freq/OTP maps | Parquet + JSON |")
    a("")

    report_text = "\n".join(lines)
    Path(report_path).write_text(report_text, encoding="utf-8")
    log.info(f"Report written to {report_path}")
    return report_text


# ── ACCEPTANCE CHECKS ───────────────────────────────────────────────────

def acceptance_checks(out_dir, report_path, target_col):
    log.info("=== Running acceptance checks ===")
    errors = []
    rp = Path(report_path)
    if not rp.exists():
        errors.append(f"Report not found: {report_path}")
    od = Path(out_dir)
    for yr in range(2021, 2026):
        for sub in ["clean_full", "clean_operated"]:
            d = od / sub / f"YEAR={yr}"
            if not d.exists():
                errors.append(f"Missing partition: {d}")
    # [ML DISABLED] ML file existence & column/leakage checks
    # for fname in ["ml_track_a/ml_track_a_train.parquet", "ml_track_a/ml_track_a_test.parquet",
    #                "ml_track_b/ml_track_b_train.parquet", "ml_track_b/ml_track_b_test.parquet"]:
    #     p = od / fname
    #     if not p.exists():
    #         errors.append(f"Missing ML file: {p}")
    #
    # # Column checks
    # for name, expected_features in [("ml_track_a", TRACK_A_FEATURES),
    #                                  ("ml_track_b", TRACK_A_FEATURES + TRACK_B_EXTRA)]:
    #     train_p = od / name / f"{name}_train.parquet"
    #     if train_p.exists():
    #         cols = set(pd.read_parquet(train_p, engine="pyarrow").columns)
    #         expected = set(expected_features) | {target_col}
    #         missing = expected - cols
    #         extra = cols - expected
    #         if missing:
    #             errors.append(f"{name} train missing cols: {missing}")
    #         if extra:
    #             errors.append(f"{name} train extra cols: {extra}")
    #
    # # Track A must not have forbidden (target is allowed)
    # ta_train = od / "ml_track_a" / "ml_track_a_train.parquet"
    # if ta_train.exists():
    #     cols = set(pd.read_parquet(ta_train, engine="pyarrow").columns)
    #     leaked = cols & (TRACK_A_FORBIDDEN - {target_col})
    #     if leaked:
    #         errors.append(f"Track A LEAKAGE: {leaked}")
    # # Track B must not have forbidden (target is allowed)
    # tb_train = od / "ml_track_b" / "ml_track_b_train.parquet"
    # if tb_train.exists():
    #     cols = set(pd.read_parquet(tb_train, engine="pyarrow").columns)
    #     leaked = cols & (TRACK_B_FORBIDDEN - {target_col})
    #     if leaked:
    #         errors.append(f"Track B LEAKAGE: {leaked}")

    if errors:
        msg = "ACCEPTANCE FAILED:\n" + "\n".join(f"  - {e}" for e in errors)
        log.error(msg)
        raise AssertionError(msg)
    log.info("All acceptance checks PASSED ✓")


# ── MAIN ────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Airline OTP Preprocessing Pipeline")
    p.add_argument("--inputs", nargs="+", default=FILES, help="CSV input file paths")
    p.add_argument("--chunksize", type=int, default=CHUNKSIZE)
    p.add_argument("--target", default=TARGET, choices=["ARR_DEL15", "ARR_DELAY_NEW"])
    p.add_argument("--out_dir", default="data/processed")
    p.add_argument("--report", default="reports/quality_report.md")
    p.add_argument("--overwrite", action="store_true", help="Overwrite existing partitions")
    return p.parse_args()


def main():
    args = parse_args()
    start = datetime.now().isoformat()

    # Validate inputs
    for f in args.inputs:
        if not Path(f).exists():
            log.error(f"Input file missing: {f}")
            sys.exit(1)
    # Check required cols in first file
    sample = pd.read_csv(args.inputs[0], nrows=0)
    sample.columns = sample.columns.str.strip().str.upper()
    required = {"YEAR", "DAY_OF_MONTH", "DAY_OF_WEEK", "OP_CARRIER", "ORIGIN", "DEST",
                "CRS_DEP_TIME", "CRS_ARR_TIME", "CANCELLED", "DIVERTED", args.target}
    missing = required - set(sample.columns)
    if missing:
        log.error(f"Required columns missing from {args.inputs[0]}: {missing}")
        sys.exit(1)

    # Pass 1
    freq_maps, otp_maps, global_otp = run_pass1(
        args.inputs, args.chunksize, args.target, args.out_dir)

    # Pass 2
    stats = run_pass2(args.inputs, args.chunksize, args.target, args.out_dir,
                      args.overwrite)

    end = datetime.now().isoformat()

    # Report
    report_text = generate_report(args.report, args.inputs, args.chunksize,
                                  args.target, args.out_dir, stats, start, end)

    # Acceptance
    acceptance_checks(args.out_dir, args.report, args.target)

    # Console summary
    print("\n" + "=" * 60)
    print("PREPROCESSING COMPLETE")
    print("=" * 60)
    total = sum(s["rows_read"] for s in stats["year_stats"].values())
    print(f"Total rows processed: {total:,}")
    print(f"Report: {args.report}")
    print(f"Outputs: {args.out_dir}/")
    print("=" * 60)


if __name__ == "__main__":
    main()
