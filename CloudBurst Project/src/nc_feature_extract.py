import os
import json
import logging

import numpy as np
import pandas as pd
import rasterio

from functools import lru_cache

import openmeteo_requests
import requests_cache
from retry_requests import retry

# ==========================================================
# LOGGING
# ==========================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger(__name__)

# ==========================================================
# TERRAIN DATA (elevation + slope rasters live in this folder)
# ==========================================================

SLOPE_FOLDER = r"D:\elevation slope\terrain_outputs"

ELEVATION_FOLDER = r"D:\elevation slope\terrain_outputs"

# ==========================================================
# HOURLY WEATHER VARIABLES TO DOWNLOAD FROM OPEN-METEO
# (everything the 58-feature model needs is derived from these)
#
# NOTE: wind_speed_100m is fetched ONLY to derive wind_shear_10_100m
#       (= wind_speed_100m - wind_speed_10m); it is not a model feature itself.
# ==========================================================

HOURLY_VARS = [

    "temperature_2m",
    "relative_humidity_2m",
    "dew_point_2m",
    "precipitation",
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

# How each "current day" feature is aggregated over the target day.
# (In training these were instantaneous values at the event hour; on a grid
#  there is no event hour, so we use a per-day aggregate.)

DAILY_AGG = {

    "temperature_2m": "mean",
    "relative_humidity_2m": "mean",
    "dew_point_2m": "mean",
    "precipitation": "sum",
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

# The exact 58 features the model expects (used only as a completeness check).

MODEL_FEATURES = [

    "Latitude", "Longitude",
    "temperature_2m", "relative_humidity_2m", "dew_point_2m", "precipitation",
    "pressure_msl", "surface_pressure",
    "cloud_cover", "cloud_cover_low", "cloud_cover_mid", "cloud_cover_high",
    "wind_speed_10m", "wind_direction_10m", "wind_gusts_10m",
    "soil_moisture_0_to_7cm",
    "relative_humidity_1d_mean", "dew_point_7d_mean",
    "precipitation_1d_sum", "precipitation_3d_sum", "precipitation_7d_sum",
    "pressure_msl_7d_mean", "surface_pressure_7d_mean",
    "cloud_cover_1d_mean", "cloud_cover_3d_mean", "cloud_cover_7d_mean",
    "cloud_cover_low_1d_mean", "cloud_cover_low_3d_mean", "cloud_cover_low_7d_mean",
    "cloud_cover_mid_1d_mean", "cloud_cover_mid_3d_mean", "cloud_cover_mid_7d_mean",
    "cloud_cover_high_1d_mean", "cloud_cover_high_3d_mean", "cloud_cover_high_7d_mean",
    "wind_speed_1d_mean", "wind_speed_7d_mean",
    "wind_direction_1d_mean", "wind_direction_3d_mean", "wind_direction_7d_mean",
    "wind_gusts_1d_max", "wind_gusts_3d_max", "wind_gusts_7d_max",
    "vapour_pressure_deficit", "weather_code", "wind_shear_10_100m",
    "vapour_pressure_deficit_1d_mean",
    "weather_code_1d_max", "weather_code_3d_max", "weather_code_7d_max",
    "wind_shear_10_100m_1d", "wind_shear_10_100m_3d", "wind_shear_10_100m_7d",
    "precipitation_1d_max_hourly", "precipitation_3d_max_hourly", "precipitation_7d_max_hourly",
    "Elevation", "Slope",

]

# ==========================================================
# OPEN-METEO SETUP (identical config to download_weather.py)
# ==========================================================

_cache_session = requests_cache.CachedSession(
    ".cache",
    expire_after=-1
)

_retry_session = retry(
    _cache_session,
    retries=5,
    backoff_factor=0.2
)

openmeteo = openmeteo_requests.Client(
    session=_retry_session
)

ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"

print("Configuration Loaded Successfully")

# ==========================================================
# RASTER CACHE
# ==========================================================

@lru_cache(maxsize=8)
def open_raster(path):
    return rasterio.open(path)


# ==========================================================
# RASTER VALUE EXTRACTION
# ==========================================================

def extract_raster_value(raster_path, latitude, longitude):
    """
    Extract a single raster value at latitude/longitude.
    """

    try:

        src = open_raster(raster_path)

        row, col = src.index(longitude, latitude)

        # Guard against points outside the raster grid.
        if row < 0 or col < 0 or row >= src.height or col >= src.width:
            return np.nan

        from rasterio.windows import Window

        value = src.read(
            1,
            window=Window(col, row, 1, 1)
        )[0, 0]

        if src.nodata is not None and value == src.nodata:
            return np.nan

        return float(value)

    except Exception as e:

        logger.error(f"Error reading {raster_path}")

        logger.error(e)

        return np.nan


# ==========================================================
# LOCATE THE CORRECT TIFF BY KEYWORD
# ==========================================================

@lru_cache(maxsize=8)
def find_tif_by_keyword(folder, keywords):
    """
    Return the first .tif in `folder` whose file name contains any of
    `keywords`. `keywords` is a tuple so this stays cacheable.
    Falls back to the first .tif if nothing matches.
    """

    if not os.path.exists(folder):
        return None

    tif_files = [
        f for f in os.listdir(folder)
        if f.lower().endswith(".tif")
    ]

    if len(tif_files) == 0:
        return None

    for f in tif_files:
        name = f.lower()
        if any(k in name for k in keywords):
            return os.path.join(folder, f)

    # No keyword match -> fall back to the first tif.
    return os.path.join(folder, tif_files[0])


# ==========================================================
# ELEVATION
# ==========================================================

def extract_elevation(latitude, longitude):

    tif = find_tif_by_keyword(
        ELEVATION_FOLDER,
        ("elev", "dem", "altitude")
    )

    if tif is None:
        logger.warning("Elevation raster not found.")
        return np.nan

    return extract_raster_value(tif, latitude, longitude)


# ==========================================================
# SLOPE
# ==========================================================

def extract_slope(latitude, longitude):

    tif = find_tif_by_keyword(
        SLOPE_FOLDER,
        ("slope",)
    )

    if tif is None:
        logger.warning("Slope raster not found.")
        return np.nan

    return extract_raster_value(tif, latitude, longitude)


print("Terrain Functions Loaded Successfully")

# ==========================================================
# WEATHER FETCH (mirrors download_weather.py fetching logic)
# ==========================================================

def fetch_weather_df(latitude, longitude, start_date, end_date):
    """
    Download hourly weather for a single point and return a tidy DataFrame with
    a naive-UTC "time" column, one column per HOURLY_VARS entry, plus a derived
    "wind_shear_10_100m" column (wind_speed_100m - wind_speed_10m).
    """

    params = {

        "latitude": latitude,
        "longitude": longitude,

        "start_date": start_date,
        "end_date": end_date,

        "hourly": HOURLY_VARS

    }

    responses = openmeteo.weather_api(ARCHIVE_URL, params=params)

    if len(responses) == 0:
        raise RuntimeError("No response from Open-Meteo")

    response = responses[0]

    hourly = response.Hourly()

    data = {

        "time": pd.date_range(

            start=pd.to_datetime(hourly.Time(), unit="s", utc=True),

            end=pd.to_datetime(hourly.TimeEnd(), unit="s", utc=True),

            freq=pd.Timedelta(seconds=hourly.Interval()),

            inclusive="left"

        ).tz_localize(None)

    }

    # Variables come back in the same order as the HOURLY_VARS list.
    for i, feature in enumerate(HOURLY_VARS):
        data[feature] = hourly.Variables(i).ValuesAsNumpy()

    df = pd.DataFrame(data)

    # Derived: low-level wind shear between 10 m and 100 m.
    df["wind_shear_10_100m"] = df["wind_speed_100m"] - df["wind_speed_10m"]

    return df


# ==========================================================
# SAFE AGGREGATIONS
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


def daily_aggregate(day_data, column):
    """Aggregate one feature over the target day using DAILY_AGG."""

    how = DAILY_AGG.get(column, "mean")

    if how == "sum":
        return safe_sum(day_data, column)
    if how == "max":
        return safe_max(day_data, column)
    return safe_mean(day_data, column)


# ==========================================================
# MAIN EXTRACTION FUNCTION (called by the batch script)
# ==========================================================

def extract_daily_features_from_72h_nc(latitude, longitude, date_str):
    """
    Build the FULL 58-feature training set for ONE grid point on ONE date.

    Returns a list containing a single feature dict so the batch script can
    `.extend()` the results.

    Features produced (exactly the 58 the model was trained on, + a "Date"
    metadata field the model ignores):
      * Metadata / terrain : Latitude, Longitude, Elevation, Slope
      * Current day        : 17 aggregates over the target day
      * Cumulative windows : 1d / 3d / 7d lookbacks ending on the date
    """

    target_date = pd.to_datetime(date_str).normalize()

    # Anchor at the last hourly slot of the target day, then look back.
    event_datetime = target_date + pd.Timedelta(hours=23)

    start_date = (target_date - pd.Timedelta(days=7)).strftime("%Y-%m-%d")
    end_date = target_date.strftime("%Y-%m-%d")

    weather_df = fetch_weather_df(
        latitude,
        longitude,
        start_date,
        end_date
    )

    # ------------------------------------------------------
    # Lookback windows (same semantics as training)
    # ------------------------------------------------------

    last1 = weather_df[
        (weather_df["time"] >= event_datetime - pd.Timedelta(days=1)) &
        (weather_df["time"] <= event_datetime)
    ]

    last3 = weather_df[
        (weather_df["time"] >= event_datetime - pd.Timedelta(days=3)) &
        (weather_df["time"] <= event_datetime)
    ]

    last7 = weather_df[
        (weather_df["time"] >= event_datetime - pd.Timedelta(days=7)) &
        (weather_df["time"] <= event_datetime)
    ]

    # The target calendar day (used for the "current weather" aggregate).
    target_day = weather_df[
        weather_df["time"].dt.normalize() == target_date
    ]

    # ------------------------------------------------------
    # Assemble the feature record
    # ------------------------------------------------------

    features = {

        "Date": end_date,
        "Latitude": float(latitude),
        "Longitude": float(longitude),

        "Elevation": extract_elevation(latitude, longitude),
        "Slope": extract_slope(latitude, longitude),

    }

    # ----- Current-day aggregates (17 features) -----

    features["temperature_2m"] = daily_aggregate(target_day, "temperature_2m")
    features["relative_humidity_2m"] = daily_aggregate(target_day, "relative_humidity_2m")
    features["dew_point_2m"] = daily_aggregate(target_day, "dew_point_2m")
    features["precipitation"] = daily_aggregate(target_day, "precipitation")
    features["pressure_msl"] = daily_aggregate(target_day, "pressure_msl")
    features["surface_pressure"] = daily_aggregate(target_day, "surface_pressure")
    features["cloud_cover"] = daily_aggregate(target_day, "cloud_cover")
    features["cloud_cover_low"] = daily_aggregate(target_day, "cloud_cover_low")
    features["cloud_cover_mid"] = daily_aggregate(target_day, "cloud_cover_mid")
    features["cloud_cover_high"] = daily_aggregate(target_day, "cloud_cover_high")
    features["wind_speed_10m"] = daily_aggregate(target_day, "wind_speed_10m")
    features["wind_direction_10m"] = daily_aggregate(target_day, "wind_direction_10m")
    features["wind_gusts_10m"] = daily_aggregate(target_day, "wind_gusts_10m")
    features["soil_moisture_0_to_7cm"] = daily_aggregate(target_day, "soil_moisture_0_to_7cm")
    features["vapour_pressure_deficit"] = daily_aggregate(target_day, "vapour_pressure_deficit")
    features["weather_code"] = daily_aggregate(target_day, "weather_code")
    features["wind_shear_10_100m"] = daily_aggregate(target_day, "wind_shear_10_100m")

    # ----- Cumulative windows -----

    # relative humidity / dew point
    features["relative_humidity_1d_mean"] = safe_mean(last1, "relative_humidity_2m")
    features["dew_point_7d_mean"] = safe_mean(last7, "dew_point_2m")

    # precipitation sums
    features["precipitation_1d_sum"] = safe_sum(last1, "precipitation")
    features["precipitation_3d_sum"] = safe_sum(last3, "precipitation")
    features["precipitation_7d_sum"] = safe_sum(last7, "precipitation")

    # pressure means
    features["pressure_msl_7d_mean"] = safe_mean(last7, "pressure_msl")
    features["surface_pressure_7d_mean"] = safe_mean(last7, "surface_pressure")

    # cloud cover (total / low / mid / high)
    features["cloud_cover_1d_mean"] = safe_mean(last1, "cloud_cover")
    features["cloud_cover_3d_mean"] = safe_mean(last3, "cloud_cover")
    features["cloud_cover_7d_mean"] = safe_mean(last7, "cloud_cover")

    features["cloud_cover_low_1d_mean"] = safe_mean(last1, "cloud_cover_low")
    features["cloud_cover_low_3d_mean"] = safe_mean(last3, "cloud_cover_low")
    features["cloud_cover_low_7d_mean"] = safe_mean(last7, "cloud_cover_low")

    features["cloud_cover_mid_1d_mean"] = safe_mean(last1, "cloud_cover_mid")
    features["cloud_cover_mid_3d_mean"] = safe_mean(last3, "cloud_cover_mid")
    features["cloud_cover_mid_7d_mean"] = safe_mean(last7, "cloud_cover_mid")

    features["cloud_cover_high_1d_mean"] = safe_mean(last1, "cloud_cover_high")
    features["cloud_cover_high_3d_mean"] = safe_mean(last3, "cloud_cover_high")
    features["cloud_cover_high_7d_mean"] = safe_mean(last7, "cloud_cover_high")

    # wind speed means
    features["wind_speed_1d_mean"] = safe_mean(last1, "wind_speed_10m")
    features["wind_speed_7d_mean"] = safe_mean(last7, "wind_speed_10m")

    # wind direction means
    features["wind_direction_1d_mean"] = safe_mean(last1, "wind_direction_10m")
    features["wind_direction_3d_mean"] = safe_mean(last3, "wind_direction_10m")
    features["wind_direction_7d_mean"] = safe_mean(last7, "wind_direction_10m")

    # wind gust maxima
    features["wind_gusts_1d_max"] = safe_max(last1, "wind_gusts_10m")
    features["wind_gusts_3d_max"] = safe_max(last3, "wind_gusts_10m")
    features["wind_gusts_7d_max"] = safe_max(last7, "wind_gusts_10m")

    # vapour pressure deficit (window mean)
    features["vapour_pressure_deficit_1d_mean"] = safe_mean(last1, "vapour_pressure_deficit")

    # weather code maxima (worst weather in each window)
    features["weather_code_1d_max"] = safe_max(last1, "weather_code")
    features["weather_code_3d_max"] = safe_max(last3, "weather_code")
    features["weather_code_7d_max"] = safe_max(last7, "weather_code")

    # wind shear (window means)
    features["wind_shear_10_100m_1d"] = safe_mean(last1, "wind_shear_10_100m")
    features["wind_shear_10_100m_3d"] = safe_mean(last3, "wind_shear_10_100m")
    features["wind_shear_10_100m_7d"] = safe_mean(last7, "wind_shear_10_100m")

    # peak hourly rainfall in each window
    features["precipitation_1d_max_hourly"] = safe_max(last1, "precipitation")
    features["precipitation_3d_max_hourly"] = safe_max(last3, "precipitation")
    features["precipitation_7d_max_hourly"] = safe_max(last7, "precipitation")

    return [features]


# ==========================================================
# COMPLETENESS CHECK
# ==========================================================

def missing_model_features(record):
    """Return any of the 58 model features absent from a produced record."""
    return [f for f in MODEL_FEATURES if f not in record]


# ==========================================================
# SINGLE-POINT DEMO
# ==========================================================

if __name__ == "__main__":

    LATITUDE = 31.6167
    LONGITUDE = 77.6167
    DATE_STR = "2025-05-24"

    demo = extract_daily_features_from_72h_nc(
        LATITUDE,
        LONGITUDE,
        DATE_STR
    )

    print("\n========== FEATURE VALUES ==========\n")

    for key, value in demo[0].items():
        print(f"{key:32} : {value}")

    print("\n====================================\n")

    # Confirm every model feature is present.
    missing = missing_model_features(demo[0])
    if missing:
        print(f"WARNING: {len(missing)} model feature(s) missing:")
        for m in missing:
            print("  -", m)
    else:
        print(f"OK: all {len(MODEL_FEATURES)} model features produced.")

    # Write a single-point JSON (same shape as the batch output).
    output_file = f"point_{LATITUDE}_{LONGITUDE}_{DATE_STR}.json"

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(demo, f, indent=2, allow_nan=True)

    logger.info(f"Saved : {output_file}")
