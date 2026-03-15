"""
ml_preparation.py — ML data-preparation functions (previously [ML DISABLED]).

Provides frequency/OTP encoding and Track A/B row building.
"""
from .ml_config import TRACK_A_FEATURES, TRACK_B_EXTRA


def apply_freq_otp(df, freq_maps, otp_maps, global_otp, unseen_log):
    """R10: Apply frequency encoding and historical OTP from train mappings."""
    col_map = {
        "OP_CARRIER": "OP_CARRIER_FREQ", "ORIGIN": "ORIGIN_FREQ",
        "DEST": "DEST_FREQ", "ROUTE": "ROUTE_FREQ", "DEP_TIME_BLK": "DEP_TIME_BLK_FREQ",
    }
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


def build_ml_rows(df, target_col, freq_maps, otp_maps, global_otp, unseen_log):
    """From operated chunk, build Track A / Track B rows."""
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
