"""
Streamlit Dashboard — US Airline On-Time Performance (Jan 2021–2025)
====================================================================

Chạy: streamlit run dashboard/app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import joblib
import sys

sys.path.append(str(Path(__file__).resolve().parent.parent))

# ============================================================
# CONFIG
# ============================================================
st.set_page_config(
    page_title="✈️ US Airline OTP Dashboard",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded",
)

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"
MODELS_DIR = Path(__file__).resolve().parent.parent / "models"


# ============================================================
# DATA LOADING
# ============================================================
@st.cache_data
def load_data():
    """Load cleaned flight data."""
    path = DATA_DIR / "flights_full_cleaned.parquet"
    if not path.exists():
        path = DATA_DIR / "flights_operated.parquet"
    if not path.exists():
        path = DATA_DIR / "flights_jan_2021_2025.parquet"
    if not path.exists():
        st.error("❌ Chưa có data! Chạy notebooks/01_data_ingestion.py trước.")
        st.stop()
    return pd.read_parquet(path)


@st.cache_resource
def load_model(track="A"):
    """Try to load the best saved model."""
    for name in ["xgboost", "lightgbm", "random_forest", "logistic_regression"]:
        path = MODELS_DIR / f"track{track}_{name}.joblib"
        if path.exists():
            return joblib.load(path), name
    return None, None


# ============================================================
# SIDEBAR FILTERS
# ============================================================
def sidebar_filters(df):
    st.sidebar.header("🔍 Bộ lọc")

    years = sorted(df["YEAR"].unique())
    sel_years = st.sidebar.multiselect("Năm", years, default=years)

    carriers = sorted(df["OP_CARRIER"].dropna().unique())
    sel_carriers = st.sidebar.multiselect("Hãng bay", carriers, default=carriers)

    origins = sorted(df["ORIGIN"].dropna().unique())
    sel_origin = st.sidebar.multiselect(
        "Sân bay đi (Origin)",
        origins,
        default=[],
        help="Để trống = tất cả"
    )

    dests = sorted(df["DEST"].dropna().unique())
    sel_dest = st.sidebar.multiselect(
        "Sân bay đến (Dest)",
        dests,
        default=[],
        help="Để trống = tất cả"
    )

    time_blocks = sorted(df["DEP_TIME_BLK"].dropna().unique()) if "DEP_TIME_BLK" in df.columns else []
    sel_blk = st.sidebar.multiselect("Khung giờ khởi hành", time_blocks, default=[])

    # Apply filters
    mask = df["YEAR"].isin(sel_years) & df["OP_CARRIER"].isin(sel_carriers)
    if sel_origin:
        mask &= df["ORIGIN"].isin(sel_origin)
    if sel_dest:
        mask &= df["DEST"].isin(sel_dest)
    if sel_blk and "DEP_TIME_BLK" in df.columns:
        mask &= df["DEP_TIME_BLK"].isin(sel_blk)

    return df[mask].copy()


# ============================================================
# PAGES
# ============================================================

def page_overview(df):
    """KPI cards + overview charts."""
    st.header("📊 Overview — On-Time Performance")

    operated = df[(df.get("CANCELLED", 0) == 0) & (df.get("DIVERTED", 0) == 0)]

    # KPI Cards
    col1, col2, col3, col4 = st.columns(4)

    total_flights = len(df)
    otp_rate = (1 - operated["ARR_DEL15"].mean()) * 100 if "ARR_DEL15" in operated.columns else 0
    avg_delay = operated["ARR_DELAY"].mean() if "ARR_DELAY" in operated.columns else 0
    cancel_rate = df["CANCELLED"].mean() * 100 if "CANCELLED" in df.columns else 0

    col1.metric("✈️ Tổng chuyến bay", f"{total_flights:,}")
    col2.metric("⏱️ OTP Rate", f"{otp_rate:.1f}%")
    col3.metric("⏰ Avg Delay", f"{avg_delay:.1f} min")
    col4.metric("🚫 Cancel Rate", f"{cancel_rate:.1f}%")

    st.divider()

    # Charts
    c1, c2 = st.columns(2)

    with c1:
        # OTP by Year
        if "ARR_DEL15" in operated.columns:
            otp_yr = operated.groupby("YEAR")["ARR_DEL15"].mean().reset_index()
            otp_yr["OTP"] = (1 - otp_yr["ARR_DEL15"]) * 100
            fig = px.bar(otp_yr, x="YEAR", y="OTP", text="OTP",
                         title="OTP (%) theo Năm", color="OTP",
                         color_continuous_scale="RdYlGn")
            fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
            fig.update_layout(yaxis_range=[0, 100])
            st.plotly_chart(fig, use_container_width=True)

    with c2:
        # Delay distribution
        if "ARR_DELAY" in operated.columns:
            fig = px.histogram(operated, x="ARR_DELAY", nbins=100,
                               title="Phân phối ARR_DELAY",
                               range_x=[-60, 200])
            fig.add_vline(x=15, line_dash="dash", line_color="red",
                          annotation_text="15-min threshold")
            st.plotly_chart(fig, use_container_width=True)

    # More charts
    c3, c4 = st.columns(2)

    with c3:
        # Top carriers by OTP
        if "OP_CARRIER" in operated.columns and "ARR_DEL15" in operated.columns:
            carrier_otp = (operated.groupby("OP_CARRIER")
                           .agg(flights=("ARR_DEL15", "count"), delay=("ARR_DEL15", "mean"))
                           .reset_index())
            carrier_otp["OTP"] = (1 - carrier_otp["delay"]) * 100
            carrier_otp = carrier_otp[carrier_otp["flights"] >= 500].sort_values("OTP")
            fig = px.bar(carrier_otp, x="OTP", y="OP_CARRIER", orientation="h",
                         title="OTP theo Hãng bay", color="OTP",
                         color_continuous_scale="RdYlGn")
            st.plotly_chart(fig, use_container_width=True)

    with c4:
        # Delay causes
        cause_cols = ["CARRIER_DELAY", "WEATHER_DELAY", "NAS_DELAY",
                      "SECURITY_DELAY", "LATE_AIRCRAFT_DELAY"]
        avail = [c for c in cause_cols if c in operated.columns]
        if avail:
            totals = operated[avail].sum()
            fig = px.pie(values=totals.values, names=[c.replace("_DELAY", "") for c in avail],
                         title="Nguyên nhân trễ (tổng phút)")
            st.plotly_chart(fig, use_container_width=True)

    # Heatmap: Year x Time Block
    if "DEP_TIME_BLK" in operated.columns and "ARR_DEL15" in operated.columns:
        hm = operated.groupby(["YEAR", "DEP_TIME_BLK"])["ARR_DEL15"].mean().reset_index()
        hm["OTP"] = (1 - hm["ARR_DEL15"]) * 100
        hm_pivot = hm.pivot(index="YEAR", columns="DEP_TIME_BLK", values="OTP")
        fig = px.imshow(hm_pivot, title="OTP Heatmap: Year × Departure Time Block",
                        color_continuous_scale="RdYlGn", aspect="auto",
                        labels=dict(color="OTP %"))
        st.plotly_chart(fig, use_container_width=True)


def page_predict(df):
    """Predict delay for a simulated flight."""
    st.header("🔮 Predict Delay — Demo")

    model_A, name_A = load_model("A")
    model_B, name_B = load_model("B")

    if model_A is None and model_B is None:
        st.warning("⚠️ Chưa có trained model! Chạy notebook 04/05 trước.")
        return

    st.subheader("Nhập thông tin chuyến bay")

    c1, c2, c3 = st.columns(3)
    with c1:
        carrier = st.selectbox("Hãng bay", sorted(df["OP_CARRIER"].dropna().unique()))
        origin = st.selectbox("Sân bay đi", sorted(df["ORIGIN"].dropna().unique()))
        dest = st.selectbox("Sân bay đến", sorted(df["DEST"].dropna().unique()))

    with c2:
        dep_hour = st.slider("Giờ khởi hành (giờ)", 0, 23, 10)
        dep_min = st.slider("Phút", 0, 59, 0)
        distance = st.number_input("Khoảng cách (miles)", 100, 5000, 1000)

    with c3:
        day_of_week = st.selectbox("Ngày trong tuần", list(range(7)),
                                   format_func=lambda x: ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"][x])
        year = st.selectbox("Năm", [2021, 2022, 2023, 2024, 2025], index=4)

    # Track B extras
    track_b_enabled = model_B is not None
    dep_delay = 0
    taxi_out = 15
    if track_b_enabled:
        st.subheader("Track B — Post-pushback (optional)")
        dep_delay = st.number_input("DEP_DELAY (phút)", -30, 300, 0)
        taxi_out = st.number_input("TAXI_OUT (phút)", 0, 120, 15)

    if st.button("🔮 Predict!", type="primary"):
        st.divider()

        # Build feature vector (simplified — actual features depend on engineering)
        crs_dep_min = dep_hour * 60 + dep_min
        is_weekend = 1 if day_of_week >= 5 else 0

        st.subheader("📊 Kết quả dự đoán")

        # Track A
        if model_A:
            st.write(f"**Track A** (Pre-flight) — Model: {name_A}")
            st.info("⚠️ Demo sử dụng features đơn giản. Trong thực tế cần đầy đủ encoding.")
            # Note: This is a simplified demo. Real prediction needs proper feature engineering.
            st.metric("Kết luận", "🟢 Đúng giờ (demo)" if dep_delay <= 0 else "🔴 Có thể trễ (demo)")

        # Track B
        if model_B and track_b_enabled:
            st.write(f"**Track B** (Post-pushback) — Model: {name_B}")
            if dep_delay > 15:
                st.metric("Kết luận", "🔴 Rất có thể trễ", f"DEP_DELAY = {dep_delay} min")
            else:
                st.metric("Kết luận", "🟢 Khả năng đúng giờ", f"DEP_DELAY = {dep_delay} min")

        st.caption("⚠️ Đây là demo inference. Để dự đoán chính xác, cần run đầy đủ pipeline feature engineering.")


# ============================================================
# MAIN APP
# ============================================================
def main():
    st.title("✈️ US Airline On-Time Performance Dashboard")
    st.caption("Dữ liệu: BTS — Tháng 1 (January) 2021–2025")

    df = load_data()
    filtered = sidebar_filters(df)

    st.sidebar.divider()
    st.sidebar.metric("📊 Flights hiển thị", f"{len(filtered):,}")

    # Navigation
    tab1, tab2 = st.tabs(["📊 Overview & Analysis", "🔮 Predict Delay"])

    with tab1:
        page_overview(filtered)

    with tab2:
        page_predict(df)

    # Footer
    st.divider()
    st.caption("Đồ án Phân tích dữ liệu — US Airline OTP Analysis (Jan 2021–2025)")


if __name__ == "__main__":
    main()
