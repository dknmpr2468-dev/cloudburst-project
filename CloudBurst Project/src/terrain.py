import pandas as pd
import rasterio

# =====================================
# FILE PATHS
# =====================================

input_file = r"D:\elevation slope\cloudburst_dataset_with_features_20260715_195016.xlsx"

elevation_file = r"D:\elevation slope\terrain_outputs\elevation.tif"

slope_file = r"D:\elevation slope\terrain_outputs\slope.tif"

# Same Excel file will be updated
output_file = r"D:\elevation slope\cloudburst_dataset_with_features_20260715_195016.xlsx"

# =====================================
# READ EXCEL
# =====================================

print("Reading Excel File...")

df = pd.read_excel(input_file)

print(f"Total Rows : {len(df)}")

# =====================================
# OPEN RASTERS
# =====================================

elev_src = rasterio.open(elevation_file)
slope_src = rasterio.open(slope_file)

print("Elevation CRS :", elev_src.crs)
print("Slope CRS :", slope_src.crs)

# =====================================
# CREATE COLUMNS IF NOT EXIST
# =====================================

if "Elevation" not in df.columns:
    df["Elevation"] = None

if "Slope" not in df.columns:
    df["Slope"] = None

# =====================================
# FETCH ELEVATION & SLOPE
# =====================================

for i, row in df.iterrows():

    # Skip rows already completed
    if pd.notna(row["Elevation"]) and pd.notna(row["Slope"]):
        continue

    lat = row["Latitude"]
    lon = row["Longitude"]

    try:

        # Elevation
        elev = list(elev_src.sample([(lon, lat)]))[0][0]

        # Slope
        slope = list(slope_src.sample([(lon, lat)]))[0][0]

        df.loc[i, "Elevation"] = float(elev)
        df.loc[i, "Slope"] = float(slope)

    except Exception as e:

        print(f"Row {i+1} Error : {e}")

        df.loc[i, "Elevation"] = None
        df.loc[i, "Slope"] = None

    if (i + 1) % 100 == 0:
        print(f"{i+1}/{len(df)} rows completed")

# =====================================
# SAVE UPDATED EXCEL
# =====================================

df.to_excel(output_file, index=False)

print("\n====================================")
print("Terrain Feature Extraction Completed")
print("====================================")
print(f"Rows Processed : {len(df)}")
print("Elevation Added")
print("Slope Added")
print(f"Saved Successfully : {output_file}")

# Close raster files
elev_src.close()
slope_src.close()