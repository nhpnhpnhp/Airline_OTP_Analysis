#!/usr/bin/env python3
"""
eda_03_airport_pain.py — Airport Pain Index cho Top 20 sân bay bận rộn nhất.

Pain Index = (Tỷ lệ trễ) × (Trung bình phút trễ ARR_DELAY_NEW)

Output:
  - reports/figures/03_airport_pain_index.png
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# ── Paths ────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "processed" / "clean_operated"
FIG_DIR = PROJECT_ROOT / "reports" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

sns.set_theme(style="whitegrid", font_scale=1.1)


def load_data() -> pd.DataFrame:
    """Đọc toàn bộ clean_operated — chỉ load cột cần thiết."""
    cols = ["ORIGIN", "ARR_DEL15", "ARR_DELAY_NEW"]
    df = pd.read_parquet(DATA_DIR, engine="pyarrow", columns=cols)
    df["ARR_DEL15"] = df["ARR_DEL15"].astype("Int16")
    df["ARR_DELAY_NEW"] = pd.to_numeric(df["ARR_DELAY_NEW"], errors="coerce")
    print(f"✅ Loaded {len(df):,} rows")
    return df


def compute_pain_index(df: pd.DataFrame, top_n: int = 20) -> pd.DataFrame:
    """Tính Airport Pain Index cho top N sân bay bận rộn nhất."""
    # Top N sân bay theo tổng chuyến bay (ORIGIN)
    top_airports = df["ORIGIN"].value_counts().head(top_n).index

    sub = df[df["ORIGIN"].isin(top_airports)].copy()
    agg = (
        sub.groupby("ORIGIN")
        .agg(
            flights=("ARR_DEL15", "count"),
            delay_rate=("ARR_DEL15", "mean"),
            avg_delay_min=("ARR_DELAY_NEW", "mean"),
        )
        .reset_index()
    )
    agg["pain_index"] = agg["delay_rate"] * agg["avg_delay_min"]
    agg["delay_pct"] = agg["delay_rate"] * 100
    agg = agg.sort_values("pain_index", ascending=False)
    return agg


def plot_pain_index(agg: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(12, 7))

    # Color scale
    norm = plt.Normalize(agg["pain_index"].min(), agg["pain_index"].max())
    cmap = plt.cm.YlOrRd
    colors = [cmap(norm(v)) for v in agg["pain_index"]]

    bars = ax.barh(
        agg["ORIGIN"], agg["pain_index"],
        color=colors, edgecolor="white", linewidth=0.8, height=0.7,
    )
    # Annotate
    for bar, row in zip(bars, agg.itertuples()):
        ax.text(
            bar.get_width() + 0.15, bar.get_y() + bar.get_height() / 2,
            f"{row.pain_index:.1f}  (Trễ {row.delay_pct:.1f}% | "
            f"TB {row.avg_delay_min:.0f} min | {row.flights:,} chuyến)",
            va="center", fontsize=8.5,
        )

    ax.invert_yaxis()
    ax.set_xlabel("Pain Index  =  Tỷ lệ trễ × TB phút trễ", fontsize=12)
    ax.set_title(
        "Airport Pain Index — Top 20 sân bay bận rộn nhất\n"
        "(Chỉ số càng cao → hành khách càng \"khốn khổ\")",
        fontsize=14, fontweight="bold",
    )
    ax.set_xlim(0, agg["pain_index"].max() * 1.45)
    sns.despine(left=True)
    fig.tight_layout()
    out = FIG_DIR / "03_airport_pain_index.png"
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"💾 Saved {out}")


# ── Main ─────────────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("EDA 03 — Airport Pain Index")
    print("=" * 60)
    df = load_data()
    agg = compute_pain_index(df, top_n=20)
    print(agg[["ORIGIN", "flights", "delay_pct", "avg_delay_min", "pain_index"]].to_string(index=False))
    plot_pain_index(agg)
    print("✅ EDA 03 hoàn tất!")


if __name__ == "__main__":
    main()
