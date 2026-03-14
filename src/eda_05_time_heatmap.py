#!/usr/bin/env python3
"""
eda_05_time_heatmap.py — Heatmap: Khung giờ × Ngày trong tuần → Tỷ lệ trễ.

Output:
  - reports/figures/05_delay_heatmap.png
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

sns.set_theme(style="white", font_scale=1.1)

DAY_LABELS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]


def load_data() -> pd.DataFrame:
    """Đọc toàn bộ clean_operated — chỉ load cột cần thiết."""
    cols = ["DEP_TIME_BLK", "DAY_OF_WEEK", "ARR_DEL15"]
    df = pd.read_parquet(DATA_DIR, engine="pyarrow", columns=cols)
    df["ARR_DEL15"] = pd.to_numeric(df["ARR_DEL15"], errors="coerce").astype("float32")
    df["DAY_OF_WEEK"] = df["DAY_OF_WEEK"].astype("Int16")
    print(f"✅ Loaded {len(df):,} rows")
    return df


def build_pivot(df: pd.DataFrame) -> pd.DataFrame:
    """Tạo pivot table: DEP_TIME_BLK × DAY_OF_WEEK → mean(ARR_DEL15)."""
    pivot = df.pivot_table(
        index="DEP_TIME_BLK",
        columns="DAY_OF_WEEK",
        values="ARR_DEL15",
        aggfunc="mean",
    )
    # Sắp xếp khung giờ theo thứ tự thời gian
    pivot = pivot.sort_index()
    # Đổi tên cột thành ngày
    pivot.columns = [DAY_LABELS[int(c)] if int(c) < 7 else str(c)
                     for c in pivot.columns]
    return pivot


def plot_heatmap(pivot: pd.DataFrame) -> None:
    # Convert to percentage
    pivot_pct = pivot * 100

    fig, ax = plt.subplots(figsize=(10, max(8, len(pivot_pct) * 0.45)))
    hm = sns.heatmap(
        pivot_pct, annot=True, fmt=".1f", cmap="YlOrRd",
        linewidths=0.8, linecolor="white",
        cbar_kws={"label": "Tỷ lệ trễ (%)", "shrink": 0.8},
        ax=ax,
    )
    ax.set_xlabel("Ngày trong tuần", fontsize=13)
    ax.set_ylabel("Khung giờ khởi hành (DEP_TIME_BLK)", fontsize=13)
    ax.set_title(
        "Heatmap — Tỷ lệ trễ chuyến theo Khung giờ × Ngày trong tuần\n"
        "(Giá trị = % chuyến trễ  |  Màu đậm = \"Điểm nóng\")",
        fontsize=14, fontweight="bold", pad=15,
    )
    plt.yticks(rotation=0)
    fig.tight_layout()
    out = FIG_DIR / "05_delay_heatmap.png"
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"💾 Saved {out}")


# ── Main ─────────────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("EDA 05 — Heatmap: Khung giờ × Ngày trong tuần")
    print("=" * 60)
    df = load_data()
    pivot = build_pivot(df)
    print("\nPivot table (% trễ):")
    print((pivot * 100).round(1).to_string())
    plot_heatmap(pivot)
    print("✅ EDA 05 hoàn tất!")


if __name__ == "__main__":
    main()
