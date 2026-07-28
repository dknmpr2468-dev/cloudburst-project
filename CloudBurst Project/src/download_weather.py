"""
download_weather.py  (rewritten)

Builds the training dataset so that every feature is derived EXACTLY the way
nc_feature_extract.py derives it at inference time:

  * Same 17 hourly variables (+ "rain", kept as a legacy extra)
  * Same derived wind_shear_10_100m = wind_speed_100m - wind_speed_10m
  * "Current" features = DAILY AGGREGATES over the event's calendar day
    (NOT the instantaneous event-hour value anymore)
  * Cumulative windows anchored at 23:00 of the event day, looking back
    1 / 3 / 7 days with the same >= / <= inclusive filters as nc
  * Timestamps: UTC from Open-Meteo, tz stripped — identical to nc

NOTE: the "Time" column in the Excel is no longer used for feature
extraction (only "Date"), because nc has no event hour on a grid and the
two pipelines must match.
"""

import numpy as np
import pandas as pd

import openmeteo_requests
import requests_cache
from retry_requests import retry

# ==========================================================
# OPEN-METEO SETUP (identical to nc_feature_extract.py)
# ==========================================================

cache_session = requests_cache.CachedSession(".cache", expire_after=-1)

retry_session = retry(cache_session, retries=5, backoff_factor=0.2)

openmeteo = openmeteo_requests.Client(session=retry_session)

ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"

# ==========================================================
# INPUT
# ==========================================================

EXCEL_FILE = r"D:\clou feature\cloudburst_full_dataset.xlsx"

CHECKPOINT_EVERY = 20

# ==========================================================
# HOURLY VARIABLES
# 17 vars matching nc_feature_extract.HOURLY_VARS, plus "rain"
# (kept only for the legacy rain_*_sum columns).
# wind_speed_100m is fetched ONLY to derive wind_shear_10_100m.
# ==========================================================

HOURLY_VARS = [
    "temperature_2m",
    "relative_humidity_2m",
    "dew_point_2m",
    "precipitation",
    "rain",                     # legacy extra (not a model feature)
    "pressure_msl",
    "surface_pressure",
    "cloud_cover",
    "cloud_cover_low",
    "cloud_cover_mid",
    "cloud_cover_high",
    "wind_speed_10m",
    "wind_speed_100m",          # for wind shear only
    "wind_direction_10m",
    "wind_gusts_10m",
    "soil_moisture_0_to_7cm",
    "vapour_pressure_deficit",
    "weather_code",
]

# ==========================================================
# "CURRENT DAY" FEATURES — aggregated over the event's calendar
# day, using the same rules as nc_feature_extract.DAILY_AGG.
# ==========================================================

DAILY_AGG = {
    "temperature_2m": "mean",
    "relative_humidity_2m": "mean",
    "dew_point_2m": "mean",
    "precipitation": "sum",
    "rain": "sum",              # legacy extra
    "pressure_msl": "mean",
    "surface_pressure": "mean",
    "cloud_cover": "mean",
    "cloud_cover_low": "mean",
    "cloud_cover_mid": "mean",
    "cloud_cover_high": "mean",
    "wind_speed_10m": "mean",
    "wind_direction_10m": "mean",
    "wind_gusts_10m": "max",
    "soil_moisture_0_to_7cm": "mean",
    "vapour_pressure_deficit": "mean",
    "weather_code": "max",
    "wind_shear_10_100m": "mean",
}

# ==========================================================
# CUMULATIVE WINDOW SPEC
# (output_column, source_column, lookback_days, aggregation)
#
# First block = the windows nc_feature_extract produces (model features).
# Second block = legacy extras kept for backward compatibility.
# ==========================================================

CUMULATIVE_SPECS = [

    # ----- MODEL FEATURES (must match nc_feature_extract exactly) -----

    ("relative_humidity_1d_mean",       "relative_humidity_2m",     1, "mean"),
    ("dew_point_7d_mean",               "dew_point_2m",             7, "mean"),

    ("precipitation_1d_sum",            "precipitation",            1, "sum"),
    ("precipitation_3d_sum",            "precipitation",            3, "sum"),
    ("precipitation_7d_sum",            "precipitation",            7, "sum"),

    ("pressure_msl_7d_mean",            "pressure_msl",             7, "mean"),
    ("surface_pressure_7d_mean",        "surface_pressure",         7, "mean"),

    ("cloud_cover_1d_mean",             "cloud_cover",              1, "mean"),
    ("cloud_cover_3d_mean",             "cloud_cover",              3, "mean"),
    ("cloud_cover_7d_mean",             "cloud_cover",              7, "mean"),

    ("cloud_cover_low_1d_mean",         "cloud_cover_low",          1, "mean"),
    ("cloud_cover_low_3d_mean",         "cloud_cover_low",          3, "mean"),
    ("cloud_cover_low_7d_mean",         "cloud_cover_low",          7, "mean"),

    ("cloud_cover_mid_1d_mean",         "cloud_cover_mid",          1, "mean"),
    ("cloud_cover_mid_3d_mean",         "cloud_cover_mid",          3, "mean"),
    ("cloud_cover_mid_7d_mean",         "cloud_cover_mid",          7, "mean"),

    ("cloud_cover_high_1d_mean",        "cloud_cover_high",         1, "mean"),
    ("cloud_cover_high_3d_mean",        "cloud_cover_high",         3, "mean"),
    ("cloud_cover_high_7d_mean",        "cloud_cover_high",         7, "mean"),

    ("wind_speed_1d_mean",              "wind_speed_10m",           1, "mean"),
    ("wind_speed_7d_mean",              "wind_speed_10m",           7, "mean"),

    ("wind_direction_1d_mean",          "wind_direction_10m",       1, "mean"),
    ("wind_direction_3d_mean",          "wind_direction_10m",       3, "mean"),
    ("wind_direction_7d_mean",          "wind_direction_10m",       7, "mean"),

    ("wind_gusts_1d_max",               "wind_gusts_10m",           1, "max"),
    ("wind_gusts_3d_max",               "wind_gusts_10m",           3, "max"),
    ("wind_gusts_7d_max",               "wind_gusts_10m",           7, "max"),

    ("vapour_pressure_deficit_1d_mean", "vapour_pressure_deficit",  1, "mean"),

    ("weather_code_1d_max",             "weather_code",             1, "max"),
    ("weather_code_3d_max",             "weather_code",             3, "max"),
    ("weather_code_7d_max",             "weather_code",             7, "max"),

    ("wind_shear_10_100m_1d",           "wind_shear_10_100m",       1, "mean"),
    ("wind_shear_10_100m_3d",           "wind_shear_10_100m",       3, "mean"),
    ("wind_shear_10_100m_7d",           "wind_shear_10_100m",       7, "mean"),

    ("precipitation_1d_max_hourly",     "precipitation",            1, "max"),
    ("precipitation_3d_max_hourly",     "precipitation",            3, "max"),
    ("precipitation_7d_max_hourly",     "precipitation",            7, "max"),

    # ----- LEGACY EXTRAS (kept, not model features) -----

    ("temperature_1d_mean",             "temperature_2m",           1, "mean"),
    ("temperature_3d_mean",             "temperature_2m",           3, "mean"),
    ("temperature_7d_mean",             "temperature_2m",           7, "mean"),

    ("relative_humidity_3d_mean",       "relative_humidity_2m",     3, "mean"),
    ("relative_humidity_7d_mean",       "relative_humidity_2m",     7, "mean"),

    ("dew_point_1d_mean",               "dew_point_2m",             1, "mean"),
    ("dew_point_3d_mean",               "dew_point_2m",             3, "mean"),

    ("rain_1d_sum",                     "rain",                     1, "sum"),
    ("rain_3d_sum",                     "rain",                     3, "sum"),
    ("rain_7d_sum",                     "rain",                     7, "sum"),

    ("pressure_msl_1d_mean",            "pressure_msl",             1, "mean"),
    ("pressure_msl_3d_mean",            "pressure_msl",             3, "mean"),

    ("surface_pressure_1d_mean",        "surface_pressure",         1, "mean"),
    ("surface_pressure_3d_mean",        "surface_pressure",         3, "mean"),

    ("wind_speed_3d_mean",              "wind_speed_10m",           3, "mean"),

    ("soil_moisture_1d_mean",           "soil_moisture_0_to_7cm",   1, "mean"),
    ("soil_moisture_3d_mean",           "soil_moisture_0_to_7cm",   3, "mean"),
    ("soil_moisture_7d_mean",           "soil_moisture_0_to_7cm",   7, "mean"),
]

# "Current day" output columns share the source variable's name,
# same as before (temperature_2m, precipitation, ...), plus the
# derived wind_shear_10_100m.
CURRENT_COLUMNS = [c for c in HOURLY_VARS if c != "wind_speed_100m"]
CURRENT_COLUMNS.append("wind_shear_10_100m")

CUMULATIVE_COLUMNS = [spec[0] for spec in CUMULATIVE_SPECS]

# Representative columns used to decide whether a row is already done.
# Includes new-spec features so previously processed rows get re-filled.
RESUME_CHECK_COLUMNS = [
    "temperature_7d_mean",
    "soil_moisture_7d_mean",
    "wind_direction_7d_mean",
    "weather_code_7d_max",
    "wind_shear_10_100m_7d",
    "precipitation_7d_max_hourly",
]

# ==========================================================
# SAFE AGGREGATIONS (identical to nc_feature_extract.py)
# ==========================================================

def safe_mean(data, column):
    if len(data) == 0:
        return np.nan
    return float(data[column].mean())


def safe_sum(data, column):
    if len(data) == 0:
        return np.nan
    return float(data[column].sum())


def safe_max(data, column):
    if len(data) == 0:
        return np.nan
    return float(data[column].max())


def aggregate(data, column, how):
    if how == "sum":
        return safe_sum(data, column)
    if how == "max":
        return safe_max(data, column)
    return safe_mean(data, column)


# ==========================================================
# WEATHER FETCH (mirrors nc_feature_extract.fetch_weather_df)
# ==========================================================

def fetch_weather_df(latitude, longitude, start_date, end_date):
    """
    Hourly weather for one point -> tidy DataFrame with naive-UTC "time",
    one column per HOURLY_VARS entry, plus derived "wind_shear_10_100m".
    """

    params = {
        "latitude": latitude,
        "longitude": longitude,
        "start_date": start_date,
        "end_date": end_date,
        "hourly": HOURLY_VARS,
    }

    responses = openmeteo.weather_api(ARCHIVE_URL, params=params)

    if len(responses) == 0:
        raise RuntimeError("No response from Open-Meteo")

    hourly = responses[0].Hourly()

    data = {
        "time": pd.date_range(
            start=pd.to_datetime(hourly.Time(), unit="s", utc=True),
            end=pd.to_datetime(hourly.TimeEnd(), unit="s", utc=True),
            freq=pd.Timedelta(seconds=hourly.Interval()),
            inclusive="left",
        ).tz_localize(None)
    }

    # Variables come back in the same order as HOURLY_VARS.
    for i, feature in enumerate(HOURLY_VARS):
        data[feature] = hourly.Variables(i).ValuesAsNumpy()

    wdf = pd.DataFrame(data)

    # Derived: low-level wind shear between 10 m and 100 m.
    wdf["wind_shear_10_100m"] = wdf["wind_speed_100m"] - wdf["wind_speed_10m"]

    return wdf


# ==========================================================
# LOAD DATASET
# ==========================================================

df = pd.read_excel(EXCEL_FILE)

df["Date"] = pd.to_datetime(
    df["Date"],
    format="mixed",
    dayfirst=True,
    errors="coerce",
)

for col in CURRENT_COLUMNS + CUMULATIVE_COLUMNS:
    if col not in df.columns:
        df[col] = np.nan

print("=" * 60)
print("Dataset Loaded Successfully")
print("Rows :", len(df))
print("Feature columns :", len(CURRENT_COLUMNS) + len(CUMULATIVE_COLUMNS))
print("=" * 60)

# ==========================================================
# MAIN LOOP
# ==========================================================

for index, row in df.iterrows():

    # ---- Skip rows already fully processed under the NEW spec ----
    if all(pd.notna(row[c]) for c in RESUME_CHECK_COLUMNS):
        continue

    print(f"\nProcessing Row {index + 1}/{len(df)}")

    if pd.isna(row["Date"]):
        print("Invalid Date")
        continue

    if pd.isna(row["Latitude"]) or pd.isna(row["Longitude"]):
        print("Missing Latitude/Longitude")
        continue

    lat = float(row["Latitude"])
    lon = float(row["Longitude"])

    # ------------------------------------------------------
    # Anchor exactly like nc_feature_extract:
    #   target day  = the event's calendar day
    #   anchor time = 23:00 of that day (last hourly slot)
    # The Excel "Time" column is intentionally NOT used.
    # ------------------------------------------------------

    target_date = row["Date"].normalize()
    event_datetime = target_date + pd.Timedelta(hours=23)

    start_date = (target_date - pd.Timedelta(days=7)).strftime("%Y-%m-%d")
    end_date = target_date.strftime("%Y-%m-%d")

    print("Latitude :", lat)
    print("Longitude:", lon)
    print("Start :", start_date)
    print("End   :", end_date)

    try:

        weather_df = fetch_weather_df(lat, lon, start_date, end_date)

        # ---- Lookback windows (same inclusive filters as nc) ----

        windows = {
            days: weather_df[
                (weather_df["time"] >= event_datetime - pd.Timedelta(days=days))
                & (weather_df["time"] <= event_datetime)
            ]
            for days in (1, 3, 7)
        }

        # ---- Target calendar day (for "current" aggregates) ----

        target_day = weather_df[
            weather_df["time"].dt.normalize() == target_date
        ]

        # ---- Current-day aggregates ----

        for col in CURRENT_COLUMNS:
            df.loc[index, col] = aggregate(
                target_day, col, DAILY_AGG.get(col, "mean")
            )

        print("Current-Day Aggregates Added")

        # ---- Cumulative window features ----

        for out_col, src_col, days, how in CUMULATIVE_SPECS:
            df.loc[index, out_col] = aggregate(windows[days], src_col, how)

        print("Cumulative Features Added")
        print(f"Row {index + 1} Completed")

    except Exception as e:

        print("=" * 60)
        print(f"Error in Row : {index + 1}")
        print(f"Latitude     : {lat}")
        print(f"Longitude    : {lon}")
        print(f"Date         : {target_date.date()}")
        print(f"Reason       : {e}")
        print("=" * 60)
        continue

    # ---- Checkpoint save ----

    if (index + 1) % CHECKPOINT_EVERY == 0:
        df.to_excel(EXCEL_FILE, index=False)
        print(f"\nCheckpoint Saved : {index + 1} Rows")

# ==========================================================
# FINAL SAVE
# ==========================================================

print("\n" + "=" * 70)
print("Saving Final Excel File...")
print("=" * 70)

df.to_excel(EXCEL_FILE, index=False)

print("\n" + "=" * 70)
print("FEATURE EXTRACTION COMPLETED")
print("=" * 70)
print(f"Total Rows : {len(df)}")
print(EXCEL_FILE)
