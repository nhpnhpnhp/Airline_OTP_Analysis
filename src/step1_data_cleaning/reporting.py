"""
reporting.py — Quality report generation and acceptance checks.

ML Dataset Audit and ML file existence/leakage checks are fully enabled.
"""
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd
import pyarrow as pa

from .config import log
from .ml_config import (
    TRACK_A_FEATURES, TRACK_B_EXTRA, TRACK_A_FORBIDDEN, TRACK_B_FORBIDDEN,
)


def generate_report(report_path, input_files, chunksize, target_col, out_dir,
                    stats, start_time, end_time):
    """Write the full quality report as a Markdown file."""
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

    # Section 3: schema
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

    # Section 6: ML dataset audit
    a("## 6. ML Dataset Audit\n")
    mc = stats["ml_counts"]
    a(f"### Row Counts")
    a(f"- Track A train: {mc.get('ml_track_a_train', 0):,}")
    a(f"- Track A test:  {mc.get('ml_track_a_test', 0):,}")
    a(f"- Track B train: {mc.get('ml_track_b_train', 0):,}")
    a(f"- Track B test:  {mc.get('ml_track_b_test', 0):,}\n")
    a(f"### Track A Features")
    a(f"```\n{TRACK_A_FEATURES}\n```\n")
    a(f"### Track B Features")
    a(f"```\n{TRACK_A_FEATURES + TRACK_B_EXTRA}\n```\n")
    a(f"### Leakage Columns Removed")
    a(f"- Track A forbidden: `{sorted(TRACK_A_FORBIDDEN)}`")
    a(f"- Track B forbidden: `{sorted(TRACK_B_FORBIDDEN)}`\n")
    a(f"### Unseen Category Rates (2025 test)\n")
    a("| Source | Total | Unseen | Rate |")
    a("|--------|-------|--------|------|")
    for k, v in stats["unseen_log"].items():
        rate = v["unseen"] / v["total"] * 100 if v["total"] else 0
        a(f"| {k} | {v['total']:,} | {v['unseen']:,} | {rate:.3f}% |")
    a(f"\nUnseen values mapped to 0 (freq) or global OTP mean (OTP).\n")

    # Section 7: artifacts
    a("## 7. Artifacts Produced\n")
    a("| Directory | Description | Partition |")
    a("|-----------|-------------|-----------|")
    a(f"| `{out_dir}/clean_full/` | Full cleaned data | `YEAR=YYYY/part-0.parquet` |")
    a(f"| `{out_dir}/clean_operated/` | Operated only | `YEAR=YYYY/part-0.parquet` |")
    a(f"| `{out_dir}/ml_track_a/` | ML pre-flight | `ml_track_a_train.parquet`, `ml_track_a_test.parquet` |")
    a(f"| `{out_dir}/ml_track_b/` | ML post-pushback | `ml_track_b_train.parquet`, `ml_track_b_test.parquet` |")
    a(f"| `{out_dir}/mappings/` | Train-only freq/OTP maps | Parquet + JSON |")
    a("")

    report_text = "\n".join(lines)
    Path(report_path).write_text(report_text, encoding="utf-8")
    log.info(f"Report written to {report_path}")
    return report_text


# ── ACCEPTANCE CHECKS ───────────────────────────────────────────────────

def acceptance_checks(out_dir, report_path, target_col):
    """Run post-pipeline acceptance checks including ML file/leakage validation."""
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

    # ML file existence & column/leakage checks
    for fname in ["ml_track_a/ml_track_a_train.parquet", "ml_track_a/ml_track_a_test.parquet",
                   "ml_track_b/ml_track_b_train.parquet", "ml_track_b/ml_track_b_test.parquet"]:
        p = od / fname
        if not p.exists():
            errors.append(f"Missing ML file: {p}")

    # Column checks
    for name, expected_features in [("ml_track_a", TRACK_A_FEATURES),
                                     ("ml_track_b", TRACK_A_FEATURES + TRACK_B_EXTRA)]:
        train_p = od / name / f"{name}_train.parquet"
        if train_p.exists():
            cols = set(pd.read_parquet(train_p, engine="pyarrow").columns)
            expected = set(expected_features) | {target_col}
            missing = expected - cols
            extra = cols - expected
            if missing:
                errors.append(f"{name} train missing cols: {missing}")
            if extra:
                errors.append(f"{name} train extra cols: {extra}")

    # Track A must not have forbidden (target is allowed)
    ta_train = od / "ml_track_a" / "ml_track_a_train.parquet"
    if ta_train.exists():
        cols = set(pd.read_parquet(ta_train, engine="pyarrow").columns)
        leaked = cols & (TRACK_A_FORBIDDEN - {target_col})
        if leaked:
            errors.append(f"Track A LEAKAGE: {leaked}")
    # Track B must not have forbidden (target is allowed)
    tb_train = od / "ml_track_b" / "ml_track_b_train.parquet"
    if tb_train.exists():
        cols = set(pd.read_parquet(tb_train, engine="pyarrow").columns)
        leaked = cols & (TRACK_B_FORBIDDEN - {target_col})
        if leaked:
            errors.append(f"Track B LEAKAGE: {leaked}")

    if errors:
        msg = "ACCEPTANCE FAILED:\n" + "\n".join(f"  - {e}" for e in errors)
        log.error(msg)
        raise AssertionError(msg)
    log.info("All acceptance checks PASSED ✓")
