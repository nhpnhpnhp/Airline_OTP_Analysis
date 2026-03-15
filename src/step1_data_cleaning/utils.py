"""
utils.py — Generic helper functions for the preprocessing pipeline.
"""
import hashlib

import numpy as np
import pandas as pd

from .config import DTYPE_MAP_INT, DTYPE_MAP_FLOAT, STR_COLS


def hash_inputs(paths):
    """Create an MD5 hash from sorted input-file paths (for caching)."""
    h = hashlib.md5()
    for p in sorted(paths):
        h.update(str(p).encode())
    return h.hexdigest()


def hhmm_to_minutes(s):
    """R3: Robust HHMM→minutes parser. Handles 600, '0600', blanks."""
    v = pd.to_numeric(s, errors="coerce")
    hh = (v // 100).astype("Int32")
    mm = (v % 100).astype("Int32")
    result = hh * 60 + mm
    result = result.where((result >= 0) & (result <= 1439), other=pd.NA)
    return result


def cyclic_encode(minutes_col, period=1440):
    """Sine/cosine encoding for a minutes-of-day column."""
    rad = 2 * np.pi * minutes_col / period
    return np.sin(rad).astype("float32"), np.cos(rad).astype("float32")


def standardize_columns(df):
    """Strip whitespace and upper-case column names; drop trailing unnamed cols."""
    df.columns = df.columns.str.strip().str.upper()
    # Drop trailing unnamed columns (BTS artefact)
    df = df[[c for c in df.columns if not c.startswith("UNNAMED")]]
    return df


def cast_dtypes(df):
    """Cast columns to their canonical dtypes (Int, float, string)."""
    for c, dt in DTYPE_MAP_INT.items():
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").astype(dt)
    for c, dt in DTYPE_MAP_FLOAT.items():
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").astype(dt)
    for c in STR_COLS:
        if c in df.columns:
            df[c] = df[c].astype("string")
    return df
