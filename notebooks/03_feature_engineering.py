# %% [markdown]
# # 🔧 03 — Feature Engineering
# **Mục tiêu**: Tạo features cho Track A (pre-flight) và Track B (post-pushback).

# %%
import pandas as pd
import numpy as np
import sys
sys.path.append('..')
from src.features import (
    engineer_features, get_feature_target_split, temporal_train_test_split,
    TRACK_A_FEATURES, TRACK_B_EXTRA_FEATURES, LEAKAGE_COLUMNS, TARGET,
    frequency_encode, label_encode_column, compute_historical_otp
)

# %%
df = pd.read_parquet('../data/processed/flights_operated.parquet')
print(f"Shape: {df.shape}")
print(f"Years: {sorted(df['YEAR'].unique())}")
print(f"Target '{TARGET}' distribution:\n{df[TARGET].value_counts()}")
print(f"Delay rate: {df[TARGET].mean():.4f}")

# %% [markdown]
# ## 3.1 Leakage Prevention
# **Rule nghiêm ngặt**: KHÔNG dùng bất kỳ thông tin nào xảy ra SAU giờ bay cho Track A.
#
# | ✅ Track A (Pre-flight) | ❌ KHÔNG dùng cho Track A |
# |---|---|
# | YEAR, DAY_OF_WEEK, CRS_DEP_TIME | DEP_TIME, DEP_DELAY |
# | OP_CARRIER, ORIGIN, DEST | ARR_TIME, ARR_DELAY |
# | CRS_ELAPSED_TIME, DISTANCE | TAXI_OUT, AIR_TIME |
# | DEP_TIME_BLK | CARRIER_DELAY, NAS_DELAY... |
#
# Track B cho phép thêm: DEP_DELAY, DEP_DEL15, TAXI_OUT

# %%
# Track A features
df_A = engineer_features(df.copy(), track="A")
X_A, y_A, feats_A = get_feature_target_split(df_A, track="A")
print(f"\nTrack A features: {feats_A}")

# %%
# Track B features
df_B = engineer_features(df.copy(), track="B")
X_B, y_B, feats_B = get_feature_target_split(df_B, track="B")
print(f"\nTrack B features: {feats_B}")

# %% [markdown]
# ## 3.2 Temporal Train/Test Split

# %%
# Temporal split: Train 2021-2024, Test 2025
train_A, test_A = temporal_train_test_split(df_A)
train_B, test_B = temporal_train_test_split(df_B)

X_train_A, y_train_A, _ = get_feature_target_split(train_A, "A")
X_test_A, y_test_A, _ = get_feature_target_split(test_A, "A")

X_train_B, y_train_B, _ = get_feature_target_split(train_B, "B")
X_test_B, y_test_B, _ = get_feature_target_split(test_B, "B")

print(f"\n--- Track A ---")
print(f"Train: {X_train_A.shape}, Test: {X_test_A.shape}")
print(f"Train delay rate: {y_train_A.mean():.4f}")
print(f"Test delay rate:  {y_test_A.mean():.4f}")

print(f"\n--- Track B ---")
print(f"Train: {X_train_B.shape}, Test: {X_test_B.shape}")

# %% [markdown]
# ## 3.3 Save Engineered Data

# %%
# Save cho notebook 04/05
train_A.to_parquet('../data/processed/train_A.parquet', index=False)
test_A.to_parquet('../data/processed/test_A.parquet', index=False)
train_B.to_parquet('../data/processed/train_B.parquet', index=False)
test_B.to_parquet('../data/processed/test_B.parquet', index=False)
print("✅ Saved train/test splits for Track A & B")

# %% [markdown]
# ## 3.4 Feature Checks

# %%
# Class imbalance
print(f"Class balance (overall):")
print(f"  On-time (0): {(y_A==0).sum():,} ({(y_A==0).mean()*100:.1f}%)")
print(f"  Delayed (1): {(y_A==1).sum():,} ({(y_A==1).mean()*100:.1f}%)")
print(f"  Ratio: {(y_A==0).sum()/(y_A==1).sum():.1f}:1")

# %%
# Feature correlations with target
if len(feats_A) > 0:
    corrs = X_A.corrwith(y_A).abs().sort_values(ascending=False)
    print("Feature correlations with ARR_DEL15:")
    print(corrs.to_string())

# %% [markdown]
# ---
# **Next**: `04_model_trackA.ipynb` → Track A Modeling
