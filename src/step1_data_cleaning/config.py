"""
config.py — General constants for the Airline OTP preprocessing pipeline.
"""
import logging
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
log = logging.getLogger("preprocess")

# ── default input files ─────────────────────────────────────────────────
FILES = [
    "data/raw/T_ONTIME_REPORTING_2021.csv",
    "data/raw/T_ONTIME_REPORTING_2022.csv",
    "data/raw/T_ONTIME_REPORTING_2023.csv",
    "data/raw/T_ONTIME_REPORTING_2024.csv",
    "data/raw/T_ONTIME_REPORTING_2025.csv",
]

CHUNKSIZE = 300_000
TARGET = "ARR_DEL15"

# TRAIN_YEARS is used by Pass 1 (Data Cleaning) to build freq/OTP mappings.
TRAIN_YEARS = {2021, 2022, 2023, 2024}

# ── column groups ───────────────────────────────────────────────────────
CANCEL_NULL_COLS = [
    "DEP_TIME", "ARR_TIME", "WHEELS_OFF", "WHEELS_ON", "TAXI_OUT", "TAXI_IN",
    "ACTUAL_ELAPSED_TIME", "AIR_TIME",
    "DEP_DELAY", "DEP_DELAY_NEW", "DEP_DEL15", "DEP_DELAY_GROUP",
    "ARR_DELAY", "ARR_DELAY_NEW", "ARR_DEL15", "ARR_DELAY_GROUP",
    "CARRIER_DELAY", "WEATHER_DELAY", "NAS_DELAY", "SECURITY_DELAY", "LATE_AIRCRAFT_DELAY",
    "FIRST_DEP_TIME", "TOTAL_ADD_GTIME", "LONGEST_ADD_GTIME",
]
DIVERT_NULL_COLS = [
    "ARR_TIME", "ARR_DELAY", "ARR_DELAY_NEW", "ARR_DEL15", "ARR_DELAY_GROUP", "ARR_TIME_BLK",
    "WHEELS_ON", "TAXI_IN",
]
HHMM_COLS = [
    "CRS_DEP_TIME", "DEP_TIME", "WHEELS_OFF", "WHEELS_ON",
    "CRS_ARR_TIME", "ARR_TIME", "FIRST_DEP_TIME",
]
DELAY_CAUSE_COLS = [
    "CARRIER_DELAY", "WEATHER_DELAY", "NAS_DELAY",
    "SECURITY_DELAY", "LATE_AIRCRAFT_DELAY",
]
FREQ_ENCODE_COLS = ["OP_CARRIER", "ORIGIN", "DEST", "ROUTE", "DEP_TIME_BLK"]
OTP_GROUP_COLS = ["ORIGIN", "OP_CARRIER"]

# ── dtype maps ──────────────────────────────────────────────────────────
DTYPE_MAP_INT = {
    "YEAR": "Int16", "DAY_OF_MONTH": "Int16", "DAY_OF_WEEK": "Int16",
    "OP_CARRIER_AIRLINE_ID": "Int32", "OP_CARRIER_FL_NUM": "Int32",
    "ORIGIN_AIRPORT_ID": "Int32", "ORIGIN_AIRPORT_SEQ_ID": "Int32",
    "ORIGIN_CITY_MARKET_ID": "Int32", "ORIGIN_STATE_FIPS": "Int16", "ORIGIN_WAC": "Int16",
    "DEST_AIRPORT_ID": "Int32", "DEST_AIRPORT_SEQ_ID": "Int32",
    "DEST_CITY_MARKET_ID": "Int32", "DEST_STATE_FIPS": "Int16", "DEST_WAC": "Int16",
    "DEP_DEL15": "Int16", "DEP_DELAY_GROUP": "Int16",
    "ARR_DEL15": "Int16", "ARR_DELAY_GROUP": "Int16",
    "CANCELLED": "Int16", "DIVERTED": "Int16",
    "DISTANCE_GROUP": "Int16", "FLIGHTS": "Int16",
    "DIV_AIRPORT_LANDINGS": "Int16", "DIV_REACHED_DEST": "Int16",
}
DTYPE_MAP_FLOAT = {
    "DEP_DELAY": "float32", "DEP_DELAY_NEW": "float32",
    "ARR_DELAY": "float32", "ARR_DELAY_NEW": "float32",
    "TAXI_OUT": "float32", "TAXI_IN": "float32",
    "CRS_ELAPSED_TIME": "float32", "ACTUAL_ELAPSED_TIME": "float32",
    "AIR_TIME": "float32", "DISTANCE": "float32",
    "CARRIER_DELAY": "float32", "WEATHER_DELAY": "float32",
    "NAS_DELAY": "float32", "SECURITY_DELAY": "float32", "LATE_AIRCRAFT_DELAY": "float32",
    "TOTAL_ADD_GTIME": "float32", "LONGEST_ADD_GTIME": "float32",
    "DIV_ACTUAL_ELAPSED_TIME": "float32", "DIV_ARR_DELAY": "float32", "DIV_DISTANCE": "float32",
}
STR_COLS = {
    "OP_UNIQUE_CARRIER", "OP_CARRIER", "TAIL_NUM", "ORIGIN", "ORIGIN_CITY_NAME",
    "ORIGIN_STATE_ABR", "ORIGIN_STATE_NM", "DEST", "DEST_CITY_NAME",
    "DEST_STATE_ABR", "DEST_STATE_NM", "DEP_TIME_BLK", "ARR_TIME_BLK",
    "CANCELLATION_CODE",
}
