# Quality Report — Airline OTP Preprocessing Pipeline

Generated: 2026-03-12T10:45:08.688787

## 1. Run Configuration

- **Machine**: local (~16 GB RAM)
- **Python**: 3.13.5
- **pandas**: 2.3.3
- **pyarrow**: 23.0.1
- **Files**: data/raw/T_ONTIME_REPORTING_2021.csv, data/raw/T_ONTIME_REPORTING_2022.csv, data/raw/T_ONTIME_REPORTING_2023.csv, data/raw/T_ONTIME_REPORTING_2024.csv, data/raw/T_ONTIME_REPORTING_2025.csv
- **Chunksize**: 300,000
- **Target**: `ARR_DEL15`
- **Start**: 2026-03-12T10:43:59.146788
- **End**: 2026-03-12T10:45:08.688448
- **Runtime**: 69.5s

## 2. Ingestion Summary (per YEAR)

| YEAR | Rows Read | Rows Full | Rows Operated | % Cancelled | % Diverted |
|------|-----------|-----------|---------------|-------------|------------|
| 2021 | 361,428 | 361,428 | 357,204 | 1.01% | 0.16% |
| 2022 | 537,902 | 537,902 | 503,529 | 6.18% | 0.21% |
| 2023 | 538,837 | 538,837 | 527,197 | 1.91% | 0.25% |
| 2024 | 547,271 | 547,271 | 525,370 | 3.73% | 0.28% |
| 2025 | 539,747 | 539,747 | 522,269 | 3.02% | 0.22% |
| **Total** | **2,525,185** | **2,525,185** | **2,435,569** | | |

## 3. Schema & Dtype Table (clean_full)

| Column | Dtype | % Missing (sample partition) |
|--------|-------|-----------------------------|
| DAY_OF_MONTH | Int16 | 0.00% |
| DAY_OF_WEEK | Int16 | 0.00% |
| OP_UNIQUE_CARRIER | string | 0.00% |
| OP_CARRIER_AIRLINE_ID | Int32 | 0.00% |
| OP_CARRIER | string | 0.00% |
| TAIL_NUM | string | 0.14% |
| OP_CARRIER_FL_NUM | Int32 | 0.00% |
| ORIGIN_AIRPORT_ID | Int32 | 0.00% |
| ORIGIN_AIRPORT_SEQ_ID | Int32 | 0.00% |
| ORIGIN_CITY_MARKET_ID | Int32 | 0.00% |
| ORIGIN | string | 0.00% |
| ORIGIN_CITY_NAME | string | 0.00% |
| ORIGIN_STATE_ABR | string | 0.00% |
| ORIGIN_STATE_FIPS | Int16 | 0.00% |
| ORIGIN_STATE_NM | string | 0.00% |
| ORIGIN_WAC | Int16 | 0.00% |
| DEST_AIRPORT_ID | Int32 | 0.00% |
| DEST_AIRPORT_SEQ_ID | Int32 | 0.00% |
| DEST_CITY_MARKET_ID | Int32 | 0.00% |
| DEST | string | 0.00% |
| DEST_CITY_NAME | string | 0.00% |
| DEST_STATE_ABR | string | 0.00% |
| DEST_STATE_FIPS | Int16 | 0.00% |
| DEST_STATE_NM | string | 0.00% |
| DEST_WAC | Int16 | 0.00% |
| CRS_DEP_TIME | int64 | 0.00% |
| DEP_TIME | float64 | 1.01% |
| DEP_DELAY | float32 | 1.01% |
| DEP_DELAY_NEW | float32 | 1.01% |
| DEP_DEL15 | Int16 | 1.01% |
| DEP_DELAY_GROUP | Int16 | 1.01% |
| DEP_TIME_BLK | string | 0.00% |
| TAXI_OUT | float32 | 1.01% |
| WHEELS_OFF | float64 | 1.01% |
| WHEELS_ON | float64 | 1.17% |
| TAXI_IN | float32 | 1.17% |
| CRS_ARR_TIME | int64 | 0.00% |
| ARR_TIME | float64 | 1.17% |
| ARR_DELAY | float32 | 1.17% |
| ARR_DELAY_NEW | float32 | 1.17% |
| ARR_DEL15 | Int16 | 1.17% |
| ARR_DELAY_GROUP | Int16 | 1.17% |
| ARR_TIME_BLK | string | 0.16% |
| CANCELLED | Int16 | 0.00% |
| CANCELLATION_CODE | string | 98.99% |
| DIVERTED | Int16 | 0.00% |
| CRS_ELAPSED_TIME | float32 | 0.00% |
| ACTUAL_ELAPSED_TIME | float32 | 1.17% |
| AIR_TIME | float32 | 1.17% |
| FLIGHTS | Int16 | 0.00% |
| DISTANCE | float32 | 0.00% |
| DISTANCE_GROUP | Int16 | 0.00% |
| CARRIER_DELAY | float32 | 90.33% |
| WEATHER_DELAY | float32 | 90.33% |
| NAS_DELAY | float32 | 90.33% |
| SECURITY_DELAY | float32 | 90.33% |
| LATE_AIRCRAFT_DELAY | float32 | 90.33% |
| FIRST_DEP_TIME | float64 | 99.45% |
| TOTAL_ADD_GTIME | float32 | 99.45% |
| LONGEST_ADD_GTIME | float32 | 99.45% |
| DIV_AIRPORT_LANDINGS | Int16 | 1.01% |
| DIV_REACHED_DEST | Int16 | 99.84% |
| DIV_ACTUAL_ELAPSED_TIME | float32 | 99.87% |
| DIV_ARR_DELAY | float32 | 99.87% |
| DIV_DISTANCE | float32 | 99.84% |
| CRS_DEP_TIME_MIN | Int32 | 0.00% |
| DEP_TIME_MIN | Int32 | 1.01% |
| WHEELS_OFF_MIN | Int32 | 1.01% |
| WHEELS_ON_MIN | Int32 | 1.19% |
| CRS_ARR_TIME_MIN | Int32 | 0.00% |
| ARR_TIME_MIN | Int32 | 1.19% |
| FIRST_DEP_TIME_MIN | Int32 | 99.45% |
| CRS_DEP_SIN | float32 | 0.00% |
| CRS_DEP_COS | float32 | 0.00% |
| CRS_ARR_SIN | float32 | 0.00% |
| CRS_ARR_COS | float32 | 0.00% |
| FL_DATE | datetime64[ns] | 0.00% |
| IS_WEEKEND | Int16 | 0.00% |
| ROUTE | object | 0.00% |
| ARR_DELAY_CAT | category | 1.17% |
| DOMINANT_DELAY_CAUSE | string | 90.33% |

**Derived columns**: CRS_DEP_TIME_MIN, DEP_TIME_MIN, WHEELS_OFF_MIN, WHEELS_ON_MIN, CRS_ARR_TIME_MIN, ARR_TIME_MIN, FIRST_DEP_TIME_MIN, CRS_DEP_SIN, CRS_DEP_COS, CRS_ARR_SIN, CRS_ARR_COS, FL_DATE, IS_WEEKEND, ROUTE, ARR_DELAY_CAT, DOMINANT_DELAY_CAUSE

## 4. Transformation Log (R1–R10)

- **R1** Cancelled nullification: 83,899 rows affected
- **R2** Diverted nullification: 5,717 rows affected
- **R3** HHMM parsing: 2,525,185 rows processed
- **R4** FL_DATE / DOW validation: 2,525,185 rows processed
- **R5** IS_WEEKEND: applied to all rows
- **R6** ROUTE: applied to all rows
- **R7** ARR_DELAY_CAT: applied to operated rows
- **R8** DOMINANT_DELAY_CAUSE: applied to delayed rows
- **R9** Operated subset: filtered per year (see Section 2)
- **R10** Freq/OTP encoding: from train mappings

### HHMM Parsing Stats

| Column | Parsed | NA |
|--------|--------|----|
| CRS_DEP_TIME | 2,525,185 | 0 |
| DEP_TIME | 2,441,123 | 84,062 |
| WHEELS_OFF | 2,441,031 | 84,154 |
| WHEELS_ON | 2,434,629 | 90,556 |
| CRS_ARR_TIME | 2,525,185 | 0 |
| ARR_TIME | 2,434,461 | 90,724 |
| FIRST_DEP_TIME | 16,023 | 2,509,162 |

## 5. Consistency Checks

- **DAY_OF_WEEK mismatches** (BTS vs computed): 0
  - Decision: overwrite with computed (Mon=0..Sun=6)
- **DISTANCE < 0**: 0
  - Action: kept as-is (data quality flag)

## 6. ML Dataset Audit

### Row Counts
- Track A train: 1,913,300
- Track A test:  522,269
- Track B train: 1,913,300
- Track B test:  522,269

### Track A Features
```
['YEAR', 'DAY_OF_MONTH', 'DAY_OF_WEEK', 'IS_WEEKEND', 'CRS_DEP_TIME_MIN', 'CRS_ARR_TIME_MIN', 'CRS_DEP_SIN', 'CRS_DEP_COS', 'CRS_ARR_SIN', 'CRS_ARR_COS', 'CRS_ELAPSED_TIME', 'DISTANCE', 'DISTANCE_GROUP', 'OP_CARRIER_FREQ', 'CARRIER_HIST_OTP', 'ORIGIN_FREQ', 'ORIGIN_HIST_OTP', 'DEST_FREQ', 'ROUTE_FREQ', 'DEP_TIME_BLK_FREQ']
```

### Track B Features
```
['YEAR', 'DAY_OF_MONTH', 'DAY_OF_WEEK', 'IS_WEEKEND', 'CRS_DEP_TIME_MIN', 'CRS_ARR_TIME_MIN', 'CRS_DEP_SIN', 'CRS_DEP_COS', 'CRS_ARR_SIN', 'CRS_ARR_COS', 'CRS_ELAPSED_TIME', 'DISTANCE', 'DISTANCE_GROUP', 'OP_CARRIER_FREQ', 'CARRIER_HIST_OTP', 'ORIGIN_FREQ', 'ORIGIN_HIST_OTP', 'DEST_FREQ', 'ROUTE_FREQ', 'DEP_TIME_BLK_FREQ', 'DEP_DELAY', 'DEP_DELAY_NEW', 'DEP_DEL15', 'TAXI_OUT']
```

### Leakage Columns Removed
- Track A forbidden: `['ACTUAL_ELAPSED_TIME', 'AIR_TIME', 'ARR_DEL15', 'ARR_DELAY', 'ARR_DELAY_GROUP', 'ARR_DELAY_NEW', 'ARR_TIME', 'CANCELLED', 'CARRIER_DELAY', 'DEP_DEL15', 'DEP_DELAY', 'DEP_DELAY_GROUP', 'DEP_DELAY_NEW', 'DEP_TIME', 'DIVERTED', 'FIRST_DEP_TIME', 'LATE_AIRCRAFT_DELAY', 'LONGEST_ADD_GTIME', 'NAS_DELAY', 'SECURITY_DELAY', 'TAXI_IN', 'TAXI_OUT', 'TOTAL_ADD_GTIME', 'WEATHER_DELAY', 'WHEELS_OFF', 'WHEELS_ON']`
- Track B forbidden: `['ACTUAL_ELAPSED_TIME', 'AIR_TIME', 'ARR_DEL15', 'ARR_DELAY', 'ARR_DELAY_GROUP', 'ARR_DELAY_NEW', 'ARR_TIME', 'ARR_TIME_BLK', 'CANCELLED', 'CARRIER_DELAY', 'DIVERTED', 'FIRST_DEP_TIME', 'LATE_AIRCRAFT_DELAY', 'LONGEST_ADD_GTIME', 'NAS_DELAY', 'SECURITY_DELAY', 'TAXI_IN', 'TOTAL_ADD_GTIME', 'WEATHER_DELAY', 'WHEELS_ON']`

### Unseen Category Rates (2025 test)

| Source | Total | Unseen | Rate |
|--------|-------|--------|------|
| OP_CARRIER | 2,435,569 | 0 | 0.000% |
| ORIGIN | 2,435,569 | 77 | 0.003% |
| DEST | 2,435,569 | 78 | 0.003% |
| ROUTE | 2,435,569 | 3,541 | 0.145% |
| DEP_TIME_BLK | 2,435,569 | 0 | 0.000% |
| otp_ORIGIN | 2,435,569 | 77 | 0.003% |
| otp_OP_CARRIER | 2,435,569 | 0 | 0.000% |

Unseen values mapped to 0 (freq) or global OTP mean (OTP).

## 7. Artifacts Produced

| Directory | Description | Partition |
|-----------|-------------|-----------|
| `data/processed/clean_full/` | Full cleaned data | `YEAR=YYYY/part-0.parquet` |
| `data/processed/clean_operated/` | Operated only | `YEAR=YYYY/part-0.parquet` |
| `data/processed/ml_track_a/` | ML pre-flight | `ml_track_a_train.parquet`, `ml_track_a_test.parquet` |
| `data/processed/ml_track_b/` | ML post-pushback | `ml_track_b_train.parquet`, `ml_track_b_test.parquet` |
| `data/processed/mappings/` | Train-only freq/OTP maps | Parquet + JSON |
