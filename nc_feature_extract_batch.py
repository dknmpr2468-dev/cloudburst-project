import logging
import json
import os
import geopandas as gpd
from shapely.geometry import Point
import numpy as np
from tqdm import tqdm

# Import the feature extraction function
from nc_feature_extract import extract_daily_features_from_72h_nc

# ==========================================================
# CONFIGURATION
# ==========================================================

STATE_SHAPEFILE = r"D:\generative point\state\STATE_BOUNDARY.shp"

# Change only this
STATE_NAME = "Uttarakhand"

# Grid spacing (km)
GRID_SPACING_KM = 4.0

# Prediction Date
DATE_STR = "2025-08-06"

# Output JSON
OUTPUT_JSON = (
    f"{STATE_NAME.lower().replace(' ', '_')}_{DATE_STR}_features.json"
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger(__name__)

# ==========================================================
# GENERATE GRID POINTS
# ==========================================================

def generate_grid_points(shapefile_path, spacing_km, state_name):

    gdf = gpd.read_file(shapefile_path)

    if gdf.crs != "EPSG:4326":
        gdf = gdf.to_crs("EPSG:4326")

    state_col = next(
        (c for c in ("STATE", "STATE_NAME", "ST_NM") if c in gdf.columns),
        None
    )

    if state_col is None:
        raise ValueError(
            f"No state column found. Columns: {list(gdf.columns)}"
        )

    values = sorted(
        gdf[state_col].dropna().str.strip().unique()
    )

    gdf = gdf[
        gdf[state_col].str.strip().str.lower()
        == state_name.strip().lower()
    ]

    if gdf.empty:
        raise ValueError(
            f"State '{state_name}' not found.\nAvailable: {values}"
        )

    minx, miny, maxx, maxy = gdf.total_bounds

    avg_lat = (miny + maxy) / 2

    lat_spacing = spacing_km / 111.0

    lon_spacing = spacing_km / (
        111.0 * np.cos(np.radians(avg_lat))
    )

    radius_km = spacing_km / 2

    radius_lat_deg = radius_km / 111.0

    radius_lon_deg = radius_km / (
        111.0 * np.cos(np.radians(avg_lat))
    )

    from shapely.prepared import prep

    hp_geom = prep(gdf.geometry.union_all())

    points = []

    for lat in np.arange(
        miny - radius_lat_deg,
        maxy + radius_lat_deg,
        lat_spacing
    ):

        for lon in np.arange(
            minx - radius_lon_deg,
            maxx + radius_lon_deg,
            lon_spacing
        ):

            point = Point(lon, lat)

            if hp_geom.intersects(point):
                points.append((lat, lon))

    logger.info(f"Total Grid Points : {len(points)}")

    return points
# ==========================================================
# FEATURE EXTRACTION (WITH AUTO RESUME)
# ==========================================================

def batch_extract_features(points_list, date_str, output_file):

    # ------------------------------------------
    # Resume from existing JSON if present
    # ------------------------------------------

    all_features = []

    if os.path.exists(output_file):

        try:

            with open(output_file, "r", encoding="utf-8") as f:
                all_features = json.load(f)

            start_index = len(all_features)

            logger.info("=" * 60)
            logger.info("RESUME MODE ENABLED")
            logger.info(f"Already Completed : {start_index}")
            logger.info(f"Remaining Points  : {len(points_list) - start_index}")
            logger.info("=" * 60)

        except Exception:

            logger.warning("Existing JSON is corrupted.")
            logger.warning("Starting from beginning...")

            all_features = []
            start_index = 0

    else:

        start_index = 0

        logger.info("No previous output found.")
        logger.info("Starting from point 1...")

    # ------------------------------------------
    # Continue Remaining Points
    # ------------------------------------------

    for i in tqdm(
        range(start_index, len(points_list)),
        desc="Extracting features"
    ):

        lat, lon = points_list[i]

        try:

            results = extract_daily_features_from_72h_nc(
                lat,
                lon,
                date_str
            )

            all_features.extend(results)

            # Save progress after every successful point
            with open(
                output_file,
                "w",
                encoding="utf-8"
            ) as f:

                json.dump(
                    all_features,
                    f,
                    indent=2,
                    allow_nan=True
                )

        except Exception as e:

            logger.error(
                f"Failed ({lat}, {lon}) : {e}"
            )

            # Stop only if API limit reached
            if (
                "limit exceeded" in str(e).lower()
                or "429" in str(e)
            ):

                logger.warning("=" * 60)
                logger.warning("OPEN-METEO API LIMIT REACHED")
                logger.warning(f"Progress Saved : {len(all_features)} points")
                logger.warning("Run this script again after one hour.")
                logger.warning("=" * 60)

                return

            # Skip only failed point for other errors
            continue

    logger.info("=" * 60)
    logger.info("ALL GRID POINTS COMPLETED SUCCESSFULLY")
    logger.info(f"Total Points Saved : {len(all_features)}")
    logger.info("=" * 60)
    # ==========================================================
# MAIN
# ==========================================================

if __name__ == "__main__":

    logger.info("=" * 60)
    logger.info(f"State          : {STATE_NAME}")
    logger.info(f"Date           : {DATE_STR}")
    logger.info(f"Grid Spacing   : {GRID_SPACING_KM} km")
    logger.info(f"Output File    : {OUTPUT_JSON}")
    logger.info("=" * 60)

    # Generate Grid Points
    points = generate_grid_points(
        STATE_SHAPEFILE,
        GRID_SPACING_KM,
        STATE_NAME
    )

    logger.info(
        f"Generated {len(points)} grid points for {STATE_NAME}"
    )

    # Start / Resume Feature Extraction
    batch_extract_features(
        points,
        DATE_STR,
        OUTPUT_JSON
    )

    logger.info("=" * 60)
    logger.info("PROCESS FINISHED")
    logger.info("=" * 60)