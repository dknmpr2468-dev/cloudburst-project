import ee
import geemap

# ==================== AUTH & INITIALIZE ====================
ee.Authenticate()  # safe — skips if already done

# ←←← CHANGE THIS TO YOUR PROJECT ID ↓↓↓
ee.Initialize(project='cloud-burst-502513')   # REQUIRED in 2026

# ==================== NASA SRTM DEM (year 2000) ====================
dem = ee.Image('USGS/SRTMGL1_003').select('elevation')

region = ee.Geometry.Rectangle([65.6, 3.87, 97.42, 35.51])
filename = 'INDIA2000_dem.tif'

# ==================== DOWNLOAD WITH PROGRESS BAR ====================
print("🚀 Starting SRTM download (this may take a few minutes)...")
geemap.download_ee_image(
    image=dem,
    filename=filename,      # different name so you don't overwrite Copernicus
    region=region,
    scale=30,                     # native 30m resolution
    crs='EPSG:4326',
)

print(f"✅ SRTM download finished! File saved as {filename}")