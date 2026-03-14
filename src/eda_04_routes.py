#!/usr/bin/env python3
"""
eda_04_routes.py — Phân tích tuyến bay: Volume & Delay Rate.

Output:
  - reports/figures/04_top_routes_delay.png
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
    cols = ["ROUTE", "ARR_DEL15"]
    df = pd.read_parquet(DATA_DIR, engine="pyarrow", columns=cols)
    df["ARR_DEL15"] = df["ARR_DEL15"].astype("Int16")
    print(f"✅ Loaded {len(df):,} rows")
    return df


def analyse_routes(df: pd.DataFrame, top_volume: int = 50, top_delay: int = 10):
    """Tính volume & delay rate cho từng tuyến, lọc top."""
    route_stats = (
        df.groupby("ROUTE")["ARR_DEL15"]
        .agg(["count", "mean"])
        .reset_index()
        .rename(columns={"count": "volume", "mean": "delay_rate"})
    )
    route_stats["delay_pct"] = route_stats["delay_rate"] * 100

    # Top 50 tuyến có volume lớn nhất
    top_vol = route_stats.nlargest(top_volume, "volume")
    # Trong đó, top 10 có tỷ lệ trễ cao nhất
    worst_routes = top_vol.nlargest(top_delay, "delay_rate")
    return top_vol, worst_routes


def plot_worst_routes(worst: pd.DataFrame) -> None:
    worst = worst.sort_values("delay_pct", ascending=True)

    fig, ax = plt.subplots(figsize=(11, 6))
    colors = sns.color_palette("Reds_r", n_colors=len(worst))

    bars = ax.barh(
        worst["ROUTE"], worst["delay_pct"],
        color=colors, edgecolor="white", linewidth=0.8, height=0.7,
    )
    for bar, row in zip(bars, worst.itertuples()):
        ax.text(
            bar.get_width() + 0.3,
            bar.get_y() + bar.get_height() / 2,
            f"{row.delay_pct:.1f}%  ({row.volume:,} chuyến)",
            va="center", fontsize=10,
        )

    ax.set_xlabel("Tỷ lệ trễ chuyến (%)", fontsize=12)
    ax.set_title(
        "Top 10 tuyến bay có tỷ lệ trễ cao nhất\n"
        "(trong nhóm 50 tuyến có lưu lượng lớn nhất)",
        fontsize=14, fontweight="bold",
    )
    ax.set_xlim(0, worst["delay_pct"].max() * 1.35)
    sns.despine(left=True)
    fig.tight_layout()
    out = FIG_DIR / "04_top_routes_delay.png"
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"💾 Saved {out}")


# ── Main ─────────────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("EDA 04 — Phân tích tuyến bay")
    print("=" * 60)
    df = load_data()
    top_vol, worst = analyse_routes(df)
    print(f"\nTop 50 tuyến bận rộn nhất (volume range: "
          f"{top_vol['volume'].min():,} – {top_vol['volume'].max():,})")
    print(f"\nTop 10 tuyến trễ nhiều nhất (trong nhóm volume lớn):")
    print(worst[["ROUTE", "volume", "delay_pct"]].to_string(index=False))
    plot_worst_routes(worst)
    print("✅ EDA 04 hoàn tất!")


if __name__ == "__main__":
    main()
