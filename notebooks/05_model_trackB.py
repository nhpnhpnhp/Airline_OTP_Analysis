# %% [markdown]
# # 🤖 05 — Track B: Post-pushback Delay Prediction
# **Mục tiêu**: Dự đoán ARR_DEL15 với features sau pushback (DEP_DELAY, TAXI_OUT...).
# So sánh với Track A để chứng minh improvement.

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sys, warnings
warnings.filterwarnings('ignore')
sys.path.append('..')

from src.features import get_feature_target_split
from src.train import train_all_models, save_model
from src.evaluate import (
    full_evaluation, plot_permutation_importance, shap_analysis,
    drift_analysis, compare_models
)

# %%
train = pd.read_parquet('../data/processed/train_B.parquet')
test = pd.read_parquet('../data/processed/test_B.parquet')

X_train, y_train, feats = get_feature_target_split(train, track="B")
X_test, y_test, _ = get_feature_target_split(test, track="B")

print(f"Features ({len(feats)}): {feats}")
print(f"Train: {X_train.shape}, positive rate: {y_train.mean():.4f}")
print(f"Test:  {X_test.shape}, positive rate: {y_test.mean():.4f}")

# %% [markdown]
# ## 5.1 Train Models

# %%
trained_models = train_all_models(X_train, y_train)

# %% [markdown]
# ## 5.2 Evaluate

# %%
results, comparison = full_evaluation(trained_models, X_test, y_test, track="B")
comparison

# %% [markdown]
# ## 5.3 Feature Importance & SHAP

# %%
best_name = comparison['roc_auc'].idxmax()
best_model = trained_models[best_name]
print(f"Best model: {best_name} (ROC-AUC = {comparison.loc[best_name,'roc_auc']:.4f})")

plot_permutation_importance(best_model, X_test, y_test, model_name=f"TrackB_{best_name}")

# %%
for name in ['xgboost', 'lightgbm', 'random_forest']:
    if name in trained_models:
        shap_analysis(trained_models[name], X_test, model_name=f"TrackB_{name}")
        break

# %% [markdown]
# ## 5.4 Drift Analysis

# %%
full_data = pd.concat([train, test])
drift_df = drift_analysis(best_model, full_data, feats, model_name=f"TrackB_{best_name}")

# %% [markdown]
# ## 5.5 Track A vs Track B Comparison

# %%
# Load Track A results for comparison
print("\n" + "=" * 60)
print("📊 TRACK A vs TRACK B COMPARISON")
print("=" * 60)
print("\nKỳ vọng: Track B >> Track A vì có DEP_DELAY (strong signal)")
print("Nếu Track A ≈ Track B → DEP_DELAY không phải driver chính")
print("Nếu Track B >> Track A → DEP_DELAY/TAXI_OUT rất quan trọng")
print("\n⚠️ LƯU Ý: Track A mới là track thực tế hữu ích (dự đoán trước bay)")
print("   Track B chỉ để chứng minh thêm info giúp cải thiện bao nhiêu")

# %% [markdown]
# ## 5.6 Save Models

# %%
for name, model in trained_models.items():
    save_model(model, name, track="B")
print("✅ All Track B models saved!")

# %% [markdown]
# ## 5.7 Key Findings — Track B
# 
# | Metric | Logistic Regression | Random Forest | XGBoost/LightGBM |
# |--------|--------------------|--------------|--------------------|
# | ROC-AUC | ___ | ___ | ___ |
# | F1 | ___ | ___ | ___ |
# | PR-AUC | ___ | ___ | ___ |
#
# **Improvement over Track A**: ROC-AUC ↑ ___
# **Top new features**: DEP_DELAY >> tất cả features khác (expected)
#
# **Kết luận**: Departure delay là predictor mạnh nhất cho arrival delay,
# nhưng chỉ biết sau khi máy bay đã rời cổng → không dùng được cho dự đoán sớm.
