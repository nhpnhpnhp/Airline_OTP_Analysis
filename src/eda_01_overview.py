#!/usr/bin/env python3
"""
eda_01_overview.py — Tổng quan On-Time Performance / Trễ chuyến.

Output:
  - reports/figures/01_otp_pie_chart.png
  - reports/figures/01_delay_rate_by_year.png
"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# ── Paths ────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "processed" / "clean_operated"
FIG_DIR = PROJECT_ROOT / "reports" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

# ── Style ────────────────────────────────────────────────────────────────
sns.set_theme(style="whitegrid", font_scale=1.15)
PALETTE = ["#2ecc71", "#e74c3c"]  # xanh = đúng giờ, đỏ = trễ


def load_data() -> pd.DataFrame:
    """Đọc toàn bộ clean_operated (partitioned Parquet).
    Chỉ load các cột cần thiết để tiết kiệm RAM (~2.5M+ rows).
    YEAR nằm trong partition path → cần thêm vào danh sách cột.
    """
    df = pd.read_parquet(DATA_DIR, engine="pyarrow", columns=["ARR_DEL15", "YEAR"])
    df["ARR_DEL15"] = df["ARR_DEL15"].astype("Int16")
    print(f"Loaded {len(df):,} rows  |  Columns: {list(df.columns)}")
    return df


# ── Chart 1: Pie chart — Đúng giờ vs Trễ ────────────────────────────────
def plot_otp_pie(df: pd.DataFrame) -> None:
    on_time = (df["ARR_DEL15"] == 0).sum()
    delayed = (df["ARR_DEL15"] == 1).sum()
    total = on_time + delayed
    otp_rate = on_time / total * 100

    fig, ax = plt.subplots(figsize=(7, 7))
    wedges, texts, autotexts = ax.pie(
        [on_time, delayed],
        labels=["Đúng giờ (On-Time)", "Trễ chuyến (Delayed)"],
        autopct="%1.1f%%",
        colors=PALETTE,
        startangle=140,
        explode=(0.03, 0.03),
        textprops={"fontsize": 13},
        wedgeprops={"edgecolor": "white", "linewidth": 2},
    )
    for t in autotexts:
        t.set_fontweight("bold")
        t.set_fontsize(14)
    ax.set_title(
        f"Tỷ lệ On-Time Performance tổng thể\n(OTP = {otp_rate:.1f}%  |  N = {total:,})",
        fontsize=15, fontweight="bold", pad=20,
    )
    fig.tight_layout()
    out = FIG_DIR / "01_otp_pie_chart.png"
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"💾 Saved {out}")


# ── Chart 2: Bar chart — Tỷ lệ trễ qua các năm ─────────────────────────
def plot_delay_by_year(df: pd.DataFrame) -> None:
    yearly = (
        df.groupby("YEAR")["ARR_DEL15"]
        .mean()
        .reset_index()
        .rename(columns={"ARR_DEL15": "delay_rate"})
    )
    yearly["delay_pct"] = yearly["delay_rate"] * 100

    fig, ax = plt.subplots(figsize=(9, 5.5))
    bars = ax.bar(
        yearly["YEAR"].astype(str),
        yearly["delay_pct"],
        color=sns.color_palette("Reds_d", n_colors=len(yearly)),
        edgecolor="white", linewidth=1.2, width=0.6,
    )
    # Annotate values
    for bar, pct in zip(bars, yearly["delay_pct"]):
        ax.text(
            bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.4,
            f"{pct:.1f}%", ha="center", va="bottom", fontweight="bold", fontsize=12,
        )
    ax.set_xlabel("Năm", fontsize=13)
    ax.set_ylabel("Tỷ lệ trễ chuyến (%)", fontsize=13)
    ax.set_title(
        "Xu hướng tỷ lệ trễ chuyến qua các năm (Jan 2021 – 2025)",
        fontsize=14, fontweight="bold",
    )
    ax.set_ylim(0, yearly["delay_pct"].max() * 1.25)
    sns.despine()
    fig.tight_layout()
    out = FIG_DIR / "01_delay_rate_by_year.png"
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"💾 Saved {out}")


# ── Main ─────────────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("EDA 01 — Tổng quan OTP / Trễ chuyến")
    print("=" * 60)
    df = load_data()
    plot_otp_pie(df)
    plot_delay_by_year(df)
    print("✅ EDA 01 hoàn tất!")


if __name__ == "__main__":
    main()
