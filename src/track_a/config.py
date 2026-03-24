"""Configuration for the Track A workflow."""

from pathlib import Path

RANDOM_SEED = 42
TARGET_COL = "ARR_DEL15"
TRACK_A_DIR = Path("data/processed/ml_track_a")
TRAIN_PATH = TRACK_A_DIR / "ml_track_a_train.parquet"
TEST_PATH = TRACK_A_DIR / "ml_track_a_test.parquet"
CLEAN_OPERATED_DIR = Path("data/processed/clean_operated")

REPORT_DIR = Path("reports/track_a")
FIG_DIR = REPORT_DIR / "figures"
MODEL_DIR = REPORT_DIR / "models"

FEATURES = [
    "YEAR", "DAY_OF_MONTH", "DAY_OF_WEEK", "IS_WEEKEND",
    "CRS_DEP_TIME_MIN", "CRS_ARR_TIME_MIN",
    "CRS_DEP_SIN", "CRS_DEP_COS", "CRS_ARR_SIN", "CRS_ARR_COS",
    "CRS_ELAPSED_TIME", "DISTANCE", "DISTANCE_GROUP",
    "OP_CARRIER_FREQ", "CARRIER_HIST_OTP",
    "ORIGIN_FREQ", "ORIGIN_HIST_OTP", "DEST_FREQ",
    "ROUTE_FREQ", "DEP_TIME_BLK_FREQ",
]

LOGREG_CONFIG = {
    "epochs": 10,
    "batch_size": 262_144,
    "learning_rate": 0.08,
    "l2": 1e-4,
}

TREE_CONFIG = {
    "max_depth": 4,
    "max_thresholds": 16,
    "min_samples_leaf": 1_500,
    "sample_size": 120_000,
    "min_gain": 1e-4,
}

VALIDATION_FRACTION = 0.10
PERMUTATION_SAMPLE = 50_000
