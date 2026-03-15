"""
pipeline.py — Core two-pass processing logic (Pass 1: mappings, Pass 2: clean + ML).
"""
import json
import shutil
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from .config import (
    log, TRAIN_YEARS, FREQ_ENCODE_COLS, OTP_GROUP_COLS,
)
from .utils import hash_inputs, standardize_columns, cast_dtypes
from .transformations import (
    apply_r1_cancelled, apply_r2_diverted, apply_r3_hhmm,
    apply_r4_date, apply_r6_route, clean_chunk,
)
from .ml_preparation import build_ml_rows


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
    """Reload previously-saved frequency and OTP mappings from disk."""
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


def run_pass2(input_files, chunksize, target_col, out_dir, overwrite,
              freq_maps, otp_maps, global_otp):
    """Full cleaning + write all outputs (clean data + ML datasets)."""
    log.info("=== PASS 2: Full cleaning & output generation ===")
    clean_full_dir = Path(out_dir) / "clean_full"
    clean_op_dir = Path(out_dir) / "clean_operated"
    ml_a_dir = Path(out_dir) / "ml_track_a"
    ml_b_dir = Path(out_dir) / "ml_track_b"
    for d in [clean_full_dir, clean_op_dir, ml_a_dir, ml_b_dir]:
        d.mkdir(parents=True, exist_ok=True)

    # If overwrite, remove existing partitions
    if overwrite:
        for d in [clean_full_dir, clean_op_dir]:
            for sub in d.glob("YEAR=*"):
                shutil.rmtree(sub, ignore_errors=True)

    hhmm_stats = {}
    dow_mismatches = {"count": 0}
    rule_counts = {}
    year_stats = {}  # year -> {rows_read, rows_full, rows_operated, cancelled, diverted}
    writers_full = {}
    writers_op = {}

    # ML accumulators
    unseen_log = {}
    ml_a_train, ml_a_test = [], []
    ml_b_train, ml_b_test = [], []
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

                    # ML datasets
                    ta, tb = build_ml_rows(op, target_col, freq_maps, otp_maps, global_otp, unseen_log)
                    if ta is not None:
                        if yr in TRAIN_YEARS:
                            ml_a_train.append(ta); ml_b_train.append(tb)
                        else:
                            ml_a_test.append(ta); ml_b_test.append(tb)

            if (ci + 1) % 3 == 0:
                log.info(f"    chunk {ci+1} done")

    # Close writers
    for w in list(writers_full.values()) + list(writers_op.values()):
        w.close()

    # Write ML datasets
    ml_counts = {}
    for name, parts, d in [("ml_track_a_train", ml_a_train, ml_a_dir),
                            ("ml_track_a_test", ml_a_test, ml_a_dir),
                            ("ml_track_b_train", ml_b_train, ml_b_dir),
                            ("ml_track_b_test", ml_b_test, ml_b_dir)]:
        if parts:
            combined = pd.concat(parts, ignore_index=True)
            out_path = d / f"{name}.parquet"
            combined.to_parquet(out_path, engine="pyarrow", compression="snappy", index=False)
            ml_counts[name] = len(combined)
            log.info(f"  Wrote {name}: {len(combined):,} rows, cols={list(combined.columns)}")
        else:
            ml_counts[name] = 0
            log.warning(f"  {name} has 0 rows!")

    log.info("Pass 2 complete.")
    return {
        "year_stats": year_stats, "hhmm_stats": hhmm_stats,
        "dow_mismatches": dow_mismatches, "rule_counts": rule_counts,
        "consistency": consistency,
        "ml_counts": ml_counts, "unseen_log": unseen_log,
    }
