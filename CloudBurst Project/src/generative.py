# ==========================================
# IMPORTS
# ==========================================

import os
import json
import logging
import numpy as np
import pandas as pd

from tqdm import tqdm

from nc_feature_extract import extract_daily_features_from_72h_nc

# ==========================================
# LOGGING
# ==========================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger(__name__)

# ==========================================
# OUTPUT PATH
# ==========================================

OUTPUT_FOLDER = r"D:\Generative_Point\Output_JSON"

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# ==========================================
# DATE
# ==========================================

DATE_STR = "2026-07-17"

# ==========================================
# STATE-WISE GRID SPACING (KM)
# ==========================================

STATE_GRID_SPACING = {

    "Jammu and Kashmir":4,

    "Ladakh":5,

    "Himachal Pradesh":3,

    "Uttarakhand":3,

    "Punjab":5,

    "Haryana":5,

    "Delhi":2,

    "Uttar Pradesh":8,

    "Bihar":6,

    "Sikkim":2,

    "Arunachal Pradesh":3,

    "Assam":4,

    "Meghalaya":3,

    "Nagaland":3,

    "Manipur":3,

    "Mizoram":3,

    "Tripura":3

}
# ==========================================
# STATE BOUNDING BOXES
# ==========================================

STATE_BOUNDARIES = {

    "Jammu and Kashmir": {
        "north": 37.10,
        "south": 32.30,
        "east": 80.40,
        "west": 73.40
    },

    "Ladakh": {
        "north": 36.90,
        "south": 32.50,
        "east": 80.90,
        "west": 76.80
    },

    "Himachal Pradesh": {
        "north": 33.30,
        "south": 30.30,
        "east": 79.10,
        "west": 75.50
    },

    "Uttarakhand": {
        "north": 31.50,
        "south": 28.40,
        "east": 81.10,
        "west": 77.50
    },

    "Punjab": {
        "north": 32.60,
        "south": 29.50,
        "east": 76.90,
        "west": 73.90
    },

    "Haryana": {
        "north": 30.90,
        "south": 27.60,
        "east": 77.60,
        "west": 74.50
    },

    "Delhi": {
        "north": 28.88,
        "south": 28.40,
        "east": 77.35,
        "west": 76.84
    },

    "Uttar Pradesh": {
        "north": 30.40,
        "south": 23.90,
        "east": 84.60,
        "west": 77.10
    },

    "Bihar": {
        "north": 27.50,
        "south": 24.30,
        "east": 88.30,
        "west": 83.20
    },

    "Sikkim": {
        "north": 28.20,
        "south": 27.00,
        "east": 88.90,
        "west": 88.00
    },

    "Arunachal Pradesh": {
        "north": 29.50,
        "south": 26.60,
        "east": 97.50,
        "west": 91.30
    },

    "Assam": {
        "north": 28.20,
        "south": 24.00,
        "east": 96.10,
        "west": 89.70
    },

    "Meghalaya": {
        "north": 26.10,
        "south": 25.00,
        "east": 92.90,
        "west": 89.80
    },

    "Nagaland": {
        "north": 27.10,
        "south": 25.10,
        "east": 95.30,
        "west": 93.20
    },

    "Manipur": {
        "north": 25.70,
        "south": 23.80,
        "east": 94.80,
        "west": 93.00
    },

    "Mizoram": {
        "north": 24.50,
        "south": 21.90,
        "east": 93.30,
        "west": 92.20
    },

    "Tripura": {
        "north": 24.50,
        "south": 22.90,
        "east": 92.40,
        "west": 91.10
    }

}

# ==========================================
# GRID GENERATION FUNCTION
# ==========================================

def generate_grid_points(state_name):

    boundary = STATE_BOUNDARIES[state_name]

    spacing_km = STATE_GRID_SPACING[state_name]

    radius_km = spacing_km / 2

    avg_lat = (boundary["north"] + boundary["south"]) / 2

    lat_spacing = spacing_km / 111.0

    lon_spacing = spacing_km / (
        111.0 * np.cos(np.radians(avg_lat))
    )

    points = []

    latitudes = np.arange(
        boundary["south"],
        boundary["north"],
        lat_spacing
    )

    longitudes = np.arange(
        boundary["west"],
        boundary["east"],
        lon_spacing
    )

    for lat in latitudes:

        for lon in longitudes:

            points.append((lat, lon))

    logger.info("=" * 50)
    logger.info(f"State               : {state_name}")
    logger.info(f"Grid Spacing (km)   : {spacing_km}")
    logger.info(f"Radius (km)         : {radius_km}")
    logger.info(f"Generated Points    : {len(points)}")
    logger.info("=" * 50)

    return points
# ==========================================
# FEATURE EXTRACTION
# ==========================================

def batch_extract_features(state_name, points):

    logger.info(f"\nStarting Feature Extraction : {state_name}")

    all_features = []

    total_points = len(points)

    for idx, (lat, lon) in enumerate(tqdm(points, desc=state_name)):

        try:

            daily_features = extract_daily_features_from_72h_nc(

                lat=lat,
                lon=lon,
                base_date_str=DATE_STR

            )

            all_features.extend(daily_features)

        except Exception as e:

            logger.error(

                f"{state_name} -> Point ({lat:.4f}, {lon:.4f}) Failed"

            )

            logger.error(str(e))

            continue

    output_file = os.path.join(

        OUTPUT_FOLDER,

        state_name.replace(" ", "_")

        + "_"

        + DATE_STR

        + ".json"

    )

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

    logger.info("=" * 60)

    logger.info(f"{state_name} Completed")

    logger.info(f"Points Processed : {total_points}")

    logger.info(f"Records Saved    : {len(all_features)}")

    logger.info(f"Output File      : {output_file}")

    logger.info("=" * 60)


# ==========================================
# PROCESS ALL STATES
# ==========================================

logger.info("\nStarting Batch Processing...\n")

for state_name in STATE_BOUNDARIES.keys():

    logger.info(f"\nGenerating Grid : {state_name}")

    points = generate_grid_points(state_name)

    batch_extract_features(

        state_name,

        points

    )

logger.info("\nAll States Completed Successfully.")
# ==========================================
# MERGE ALL JSON FILES
# ==========================================

logger.info("\nMerging All State JSON Files...\n")

all_records = []

json_files = [

    f for f in os.listdir(OUTPUT_FOLDER)

    if f.endswith(".json")

]

for file in json_files:

    file_path = os.path.join(OUTPUT_FOLDER, file)

    try:

        with open(file_path, "r", encoding="utf-8") as f:

            data = json.load(f)

            if isinstance(data, list):

                all_records.extend(data)

            else:

                logger.warning(f"{file} is not a list. Skipped.")

    except Exception as e:

        logger.error(f"Failed to read {file}")

        logger.error(str(e))

# ==========================================
# CREATE DATAFRAME
# ==========================================

df = pd.DataFrame(all_records)

# Remove duplicates if any
df.drop_duplicates(inplace=True)

# Sort by Latitude & Longitude if available
sort_columns = []

if "latitude" in df.columns:
    sort_columns.append("latitude")

if "longitude" in df.columns:
    sort_columns.append("longitude")

if sort_columns:
    df.sort_values(sort_columns, inplace=True)

df.reset_index(drop=True, inplace=True)

# ==========================================
# SAVE FINAL FILES
# ==========================================

final_json = os.path.join(
    OUTPUT_FOLDER,
    "Pan_North_NE_Cloudburst_Dataset.json"
)

final_csv = os.path.join(
    OUTPUT_FOLDER,
    "Pan_North_NE_Cloudburst_Dataset.csv"
)

df.to_json(
    final_json,
    orient="records",
    indent=2
)

df.to_csv(
    final_csv,
    index=False
)

logger.info("=" * 60)
logger.info("FINAL DATASET CREATED")
logger.info(f"Total Records : {len(df)}")
logger.info(f"JSON : {final_json}")
logger.info(f"CSV  : {final_csv}")
logger.info("=" * 60)
# ==========================================
# CHECKPOINT SYSTEM
# ==========================================

CHECKPOINT_FILE = os.path.join(
    OUTPUT_FOLDER,
    "completed_states.json"
)


def load_checkpoint():

    if os.path.exists(CHECKPOINT_FILE):

        with open(CHECKPOINT_FILE, "r", encoding="utf-8") as f:

            return json.load(f)

    return []


def save_checkpoint(completed_states):

    with open(CHECKPOINT_FILE, "w", encoding="utf-8") as f:

        json.dump(
            completed_states,
            f,
            indent=2
        )


# ==========================================
# RESUME PROCESSING
# ==========================================

completed_states = load_checkpoint()

logger.info("=" * 60)
logger.info("RESUME MODE ENABLED")
logger.info(f"Completed States : {completed_states}")
logger.info("=" * 60)


for state_name in STATE_BOUNDARIES.keys():

    if state_name in completed_states:

        logger.info(f"Skipping {state_name} (Already Completed)")
        continue

    logger.info(f"\nProcessing {state_name}")

    try:

        points = generate_grid_points(state_name)

        batch_extract_features(
            state_name,
            points
        )

        completed_states.append(state_name)

        save_checkpoint(completed_states)

        logger.info(f"{state_name} Saved to Checkpoint")

    except Exception as e:

        logger.error(f"{state_name} Failed")

        logger.error(str(e))

        logger.info("Run the script again to resume from this state.")

        break


logger.info("=" * 60)
logger.info("ALL POSSIBLE STATES PROCESSED")
logger.info("=" * 60)

