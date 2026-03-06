# %% [markdown]
# # 🤖 04 — Track A: Pre-flight Delay Prediction
# **Mục tiêu**: Dự đoán ARR_DEL15 chỉ dùng features biết trước giờ bay.
# 
# Models: Logistic Regression → Random Forest → XGBoost/LightGBM.

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sys, warnings
warnings.filterwarnings('ignore')
sys.path.append('..')

from src.features import get_feature_target_split
from src.train import train_all_models, save_model, get_model_configs
from src.evaluate import (
    full_evaluation, plot_permutation_importance, shap_analysis,
    drift_analysis, compare_models
)

# %%
train = pd.read_parquet('../data/processed/train_A.parquet')
test = pd.read_parquet('../data/processed/test_A.parquet')

X_train, y_train, feats = get_feature_target_split(train, track="A")
X_test, y_test, _ = get_feature_target_split(test, track="A")

print(f"Features ({len(feats)}): {feats}")
print(f"Train: {X_train.shape}, positive rate: {y_train.mean():.4f}")
print(f"Test:  {X_test.shape}, positive rate: {y_test.mean():.4f}")

# %% [markdown]
# ## 4.1 Train Models

# %%
# Train tất cả models
trained_models = train_all_models(X_train, y_train)

# %% [markdown]
# ## 4.2 Evaluate

# %%
# Full evaluation pipeline
results, comparison = full_evaluation(trained_models, X_test, y_test, track="A")
comparison

# %% [markdown]
# ## 4.3 Feature Importance

# %%
# Permutation importance cho best model
best_name = comparison['roc_auc'].idxmax()
best_model = trained_models[best_name]
print(f"Best model: {best_name} (ROC-AUC = {comparison.loc[best_name,'roc_auc']:.4f})")

plot_permutation_importance(best_model, X_test, y_test, model_name=f"TrackA_{best_name}")

# %%
# SHAP (cho tree-based models)
for name in ['xgboost', 'lightgbm', 'random_forest']:
    if name in trained_models:
        shap_analysis(trained_models[name], X_test, model_name=f"TrackA_{name}")
        break  # Chỉ cần 1

# %% [markdown]
# ## 4.4 Drift Analysis

# %%
# Kiểm tra performance theo năm
full_data = pd.concat([train, test])
drift_df = drift_analysis(best_model, full_data, feats, model_name=f"TrackA_{best_name}")

# %% [markdown]
# ## 4.5 Save Models

# %%
for name, model in trained_models.items():
    save_model(model, name, track="A")
print("✅ All Track A models saved!")

# %% [markdown]
# ## 4.6 Key Findings — Track A
# 
# **Ghi chú kết quả vào đây sau khi chạy:**
# 
# | Metric | Logistic Regression | Random Forest | XGBoost/LightGBM |
# |--------|--------------------|--------------|--------------------|
# | ROC-AUC | ___ | ___ | ___ |
# | F1 | ___ | ___ | ___ |
# | PR-AUC | ___ | ___ | ___ |
# | Precision | ___ | ___ | ___ |
# | Recall | ___ | ___ | ___ |
# 
# **Top features**: ___
#
# **Nhận xét**: 
# - Track A (pre-flight) có giới hạn vì không có DEP_DELAY
# - Features quan trọng nhất thường là: ___

# %% [markdown]
# ---
# **Next**: `05_model_trackB.ipynb` → Track B (Post-pushback)
