#!/usr/bin/env python3
"""
eda_02_airline_otp.py — Phân tích OTP theo hãng bay.

Output:
  - reports/figures/02_airline_otp_ranking.png
  - reports/figures/02_airline_otp_stability.png
"""
from pathlib import Path

import matplotlib.pyplot as plt
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
    df = pd.read_parquet(DATA_DIR, engine="pyarrow",
                         columns=["OP_CARRIER", "ARR_DEL15", "YEAR"])
    df["ARR_DEL15"] = df["ARR_DEL15"].astype("Int16")
    print(f"Loaded {len(df):,} rows")
    return df


# ── Chart 1: Xếp hạng OTP theo hãng ────────────────────────────────────
def plot_airline_ranking(df: pd.DataFrame) -> None:
    carrier_otp = (
        df.groupby("OP_CARRIER")["ARR_DEL15"]
        .agg(["mean", "count"])
        .reset_index()
        .rename(columns={"mean": "delay_rate", "count": "flights"})
    )
    carrier_otp["otp_pct"] = (1 - carrier_otp["delay_rate"]) * 100
    carrier_otp = carrier_otp.sort_values("otp_pct", ascending=True)

    n = len(carrier_otp)
    colors = []
    for i, row in enumerate(carrier_otp.itertuples()):
        if i < 5:
            colors.append("#e74c3c")   # bottom 5 — đỏ
        elif i >= n - 5:
            colors.append("#2ecc71")   # top 5 — xanh
        else:
            colors.append("#95a5a6")   # trung bình — xám

    fig, ax = plt.subplots(figsize=(10, max(7, n * 0.4)))
    bars = ax.barh(
        carrier_otp["OP_CARRIER"], carrier_otp["otp_pct"],
        color=colors, edgecolor="white", linewidth=0.8, height=0.7,
    )
    # Annotate
    for bar, pct, flights in zip(
        bars, carrier_otp["otp_pct"], carrier_otp["flights"]
    ):
        ax.text(
            bar.get_width() + 0.3, bar.get_y() + bar.get_height() / 2,
            f"{pct:.1f}%  ({flights:,} chuyến)",
            va="center", fontsize=9,
        )
    ax.set_xlabel("Tỷ lệ đúng giờ — OTP (%)", fontsize=12)
    ax.set_title(
        "Xếp hạng On-Time Performance theo hang bay\n"
        "(Top 5 tot nhat = xanh  |  Top 5 te nhat = do)",
        fontsize=13, fontweight="bold",
    )
    ax.set_xlim(0, 105)
    ax.axvline(x=carrier_otp["otp_pct"].median(), color="#3498db",
               linestyle="--", linewidth=1.2, label="Median")
    ax.legend(fontsize=10)
    sns.despine(left=True)
    fig.tight_layout()
    out = FIG_DIR / "02_airline_otp_ranking.png"
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"💾 Saved {out}")


# ── Chart 2: Stability — OTP theo hãng qua các năm ──────────────────────
def plot_airline_stability(df: pd.DataFrame) -> None:
    yearly = (
        df.groupby(["YEAR", "OP_CARRIER"])["ARR_DEL15"]
        .mean()
        .reset_index()
        .rename(columns={"ARR_DEL15": "delay_rate"})
    )
    yearly["otp_pct"] = (1 - yearly["delay_rate"]) * 100

    # Chọn top 10 hãng có nhiều chuyến nhất để biểu đồ dễ đọc
    top_carriers = (
        df["OP_CARRIER"].value_counts().head(10).index.tolist()
    )
    plot_df = yearly[yearly["OP_CARRIER"].isin(top_carriers)]

    fig, ax = plt.subplots(figsize=(11, 6))
    palette = sns.color_palette("tab10", n_colors=len(top_carriers))
    for i, carrier in enumerate(top_carriers):
        sub = plot_df[plot_df["OP_CARRIER"] == carrier].sort_values("YEAR")
        ax.plot(
            sub["YEAR"].astype(str), sub["otp_pct"],
            marker="o", linewidth=2, markersize=7,
            label=carrier, color=palette[i],
        )
        # Label cuối dòng
        if not sub.empty:
            last = sub.iloc[-1]
            ax.text(
                str(int(last["YEAR"])), last["otp_pct"] + 0.5,
                carrier, fontsize=8, fontweight="bold", color=palette[i],
            )

    ax.set_xlabel("Năm", fontsize=12)
    ax.set_ylabel("OTP (%)", fontsize=12)
    ax.set_title(
        "Độ ổn định OTP theo hãng bay qua các năm (Top 10 hãng lớn nhất)",
        fontsize=13, fontweight="bold",
    )
    ax.legend(
        bbox_to_anchor=(1.02, 1), loc="upper left",
        fontsize=9, title="Hãng bay",
    )
    sns.despine()
    fig.tight_layout()
    out = FIG_DIR / "02_airline_otp_stability.png"
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"💾 Saved {out}")


# ── Main ─────────────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("EDA 02 — Phân tích OTP theo hãng bay")
    print("=" * 60)
    df = load_data()
    plot_airline_ranking(df)
    plot_airline_stability(df)
    print("✅ EDA 02 hoàn tất!")


if __name__ == "__main__":
    main()
