import pandas as pd
import rasterio

# =====================================
# FILE PATHS
# =====================================

input_excel = r"D:\elevation slope\cloudburst_dataset_with_features_20260715_195016.xlsx"

elevation_raster = r"D:\elevation slope\terrain_outputs\elevation.tif"

slope_raster = r"D:\elevation slope\terrain_outputs\slope.tif"

output_excel = r"D:\elevation slope\terrain_features.xlsx"

# =====================================
# READ INPUT EXCEL
# =====================================

print("Reading Input Excel...")

df = pd.read_excel(input_excel)

# Keep only Latitude & Longitude
terrain_df = df[["Latitude", "Longitude"]].copy()

# =====================================
# OPEN RASTER FILES
# =====================================

elev_src = rasterio.open(elevation_raster)
slope_src = rasterio.open(slope_raster)

print("Elevation CRS :", elev_src.crs)
print("Slope CRS :", slope_src.crs)

# =====================================
# CREATE NEW COLUMNS
# =====================================

terrain_df["Elevation"] = None
terrain_df["Slope"] = None

# =====================================
# FETCH VALUES
# =====================================

print("\nFetching Terrain Features...\n")

for i, row in terrain_df.iterrows():

    lat = row["Latitude"]
    lon = row["Longitude"]

    try:

        elevation = list(elev_src.sample([(lon, lat)]))[0][0]
        slope = list(slope_src.sample([(lon, lat)]))[0][0]

        terrain_df.loc[i, "Elevation"] = float(elevation)
        terrain_df.loc[i, "Slope"] = float(slope)

    except Exception as e:

        print(f"Row {i+1} Error : {e}")

        terrain_df.loc[i, "Elevation"] = None
        terrain_df.loc[i, "Slope"] = None

    if (i + 1) % 100 == 0:
        print(f"{i+1}/{len(terrain_df)} rows completed")

# =====================================
# SAVE NEW EXCEL
# =====================================

terrain_df.to_excel(output_excel, index=False)

# Close raster files
elev_src.close()
slope_src.close()

print("\n===================================")
print("Terrain Feature Extraction Completed")
print("===================================")
print(f"Total Rows : {len(terrain_df)}")
print(f"Output File : {output_excel}")