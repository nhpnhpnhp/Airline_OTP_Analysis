#!/usr/bin/env python3
"""
main.py — Entry point for the Airline OTP preprocessing pipeline.

Usage:
  python -m src.step1_data_cleaning.main \
    --inputs data/raw/T_ONTIME_REPORTING_2021.csv ... \
    --chunksize 300000 --target ARR_DEL15 \
    --out_dir data/processed --report reports/quality_report.md
"""
import argparse
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

from .config import log, FILES, CHUNKSIZE, TARGET
from .pipeline import run_pass1, run_pass2
from .reporting import generate_report, acceptance_checks


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

    # Pass 2 (now receives mapping objects for ML)
    stats = run_pass2(args.inputs, args.chunksize, args.target, args.out_dir,
                      args.overwrite, freq_maps, otp_maps, global_otp)

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
