import pandas as pd
import numpy as np
from datetime import timedelta, datetime

# ============================================================
# CLOUDBURST CLASSIFICATION DATASET GENERATOR (MAXIMUM SCALE)
# ============================================================
# Generates a large realistic dataset (~5000+ rows) with:
#   Target=1 : Original cloudburst events (151)
#   Target=0 : Synthetic non-cloudburst samples via 6 strategies
# ============================================================

np.random.seed(42)

# ---------------------------
# 1. LOAD & CLEAN DATA
# ---------------------------

df = pd.read_excel("cloudburst events.xlsx")
print("=" * 60)
print("ORIGINAL DATA SUMMARY")
print("=" * 60)
print(f"Shape: {df.shape}")

# Parse mixed date formats
def parse_mixed_date(date_val):
    if pd.isna(date_val):
        return pd.NaT
    date_str = str(date_val).strip()
    for fmt in ["%Y-%m-%d", "%d-%m-%Y", "%Y/%m/%d", "%d/%m/%Y"]:
        try:
            return pd.to_datetime(date_str, format=fmt)
        except (ValueError, TypeError):
            continue
    try:
        return pd.to_datetime(date_str, dayfirst=True)
    except Exception:
        return pd.NaT

df["Date_Parsed"] = df["Date"].apply(parse_mixed_date)

# FIX 1: Standardize ALL dates to YYYY-MM-DD
df["Date"] = df["Date_Parsed"].dt.strftime("%Y-%m-%d")
# FIX 2: Fill null Source
df["Source"] = df["Source"].fillna("Not Recorded")
# FIX 3: Fill missing Location/Lat/Lon
missing_mask = df["Latitude"].isna() & df["Longitude"].isna()
df.loc[missing_mask, "Location"] = "Uttarakhand (Approximate)"
df.loc[missing_mask, "Latitude"] = 30.0668
df.loc[missing_mask, "Longitude"] = 79.0193

df["Target"] = 1
df["Event"] = df["Event"].fillna("Cloudburst")

valid_df = df[df["Latitude"].notna() & df["Longitude"].notna()].copy()
print(f"All fixes applied. Valid rows: {len(valid_df)} / {len(df)}")

# ============================================================
# LOCATION DATABASES
# ============================================================

# 80 Indian cities/towns across plains, coast, semi-arid zones
INDIAN_CITIES = [
    # Major metros
    ("Delhi", 28.6139, 77.2090),
    ("Mumbai, Maharashtra", 19.0760, 72.8777),
    ("Kolkata, West Bengal", 22.5726, 88.3639),
    ("Chennai, Tamil Nadu", 13.0827, 80.2707),
    ("Bengaluru, Karnataka", 12.9716, 77.5946),
    ("Hyderabad, Telangana", 17.3850, 78.4867),
    # UP & Bihar
    ("Lucknow, Uttar Pradesh", 26.8467, 80.9462),
    ("Kanpur, Uttar Pradesh", 26.4499, 80.3319),
    ("Varanasi, Uttar Pradesh", 25.3176, 83.0068),
    ("Agra, Uttar Pradesh", 27.1767, 78.0081),
    ("Allahabad, Uttar Pradesh", 25.4358, 81.8463),
    ("Meerut, Uttar Pradesh", 28.9845, 77.7064),
    ("Gorakhpur, Uttar Pradesh", 26.7606, 83.3732),
    ("Bareilly, Uttar Pradesh", 28.3670, 79.4304),
    ("Patna, Bihar", 25.6093, 85.1376),
    ("Gaya, Bihar", 24.7955, 84.9994),
    ("Muzaffarpur, Bihar", 26.1225, 85.3906),
    # Rajasthan & Gujarat
    ("Jaipur, Rajasthan", 26.9124, 75.7873),
    ("Jodhpur, Rajasthan", 26.2389, 73.0243),
    ("Udaipur, Rajasthan", 24.5854, 73.7125),
    ("Bikaner, Rajasthan", 28.0229, 73.3119),
    ("Barmer, Rajasthan", 25.7521, 71.3967),
    ("Jaisalmer, Rajasthan", 26.9157, 70.9083),
    ("Kota, Rajasthan", 25.2138, 75.8648),
    ("Ahmedabad, Gujarat", 23.0225, 72.5714),
    ("Surat, Gujarat", 21.1702, 72.8311),
    ("Rajkot, Gujarat", 22.3039, 70.8022),
    ("Kutch, Gujarat", 23.7337, 69.8597),
    # MP & Chhattisgarh
    ("Bhopal, Madhya Pradesh", 23.2599, 77.4126),
    ("Indore, Madhya Pradesh", 22.7196, 75.8577),
    ("Jabalpur, Madhya Pradesh", 23.1815, 79.9864),
    ("Gwalior, Madhya Pradesh", 26.2183, 78.1828),
    ("Raipur, Chhattisgarh", 21.2514, 81.6296),
    ("Bilaspur, Chhattisgarh", 22.0796, 82.1391),
    # Maharashtra & Goa
    ("Pune, Maharashtra", 18.5204, 73.8567),
    ("Nagpur, Maharashtra", 21.1458, 79.0882),
    ("Nashik, Maharashtra", 19.9975, 73.7898),
    ("Aurangabad, Maharashtra", 19.8762, 75.3433),
    ("Kolhapur, Maharashtra", 16.7050, 74.2433),
    ("Goa", 15.2993, 74.1240),
    # South India
    ("Visakhapatnam, Andhra Pradesh", 17.6868, 83.2185),
    ("Vijayawada, Andhra Pradesh", 16.5062, 80.6480),
    ("Anantapur, Andhra Pradesh", 14.6819, 77.6006),
    ("Kochi, Kerala", 9.9312, 76.2673),
    ("Thiruvananthapuram, Kerala", 8.5241, 76.9366),
    ("Kozhikode, Kerala", 11.2588, 75.7804),
    ("Mangalore, Karnataka", 12.9141, 74.8560),
    ("Mysuru, Karnataka", 12.2958, 76.6394),
    ("Hubli, Karnataka", 15.3647, 75.1240),
    ("Coimbatore, Tamil Nadu", 11.0168, 76.9558),
    ("Madurai, Tamil Nadu", 9.9252, 78.1198),
    ("Puducherry", 11.9416, 79.8083),
    ("Tiruchirappalli, Tamil Nadu", 10.7905, 78.7047),
    # East India
    ("Bhubaneswar, Odisha", 20.2961, 85.8245),
    ("Cuttack, Odisha", 20.4625, 85.8830),
    ("Ranchi, Jharkhand", 23.3441, 85.3096),
    ("Jamshedpur, Jharkhand", 22.8046, 86.2029),
    # Northeast
    ("Guwahati, Assam", 26.1445, 91.7362),
    ("Imphal, Manipur", 24.8170, 93.9368),
    ("Agartala, Tripura", 23.8315, 91.2868),
    ("Silchar, Assam", 24.8333, 92.7789),
    ("Dibrugarh, Assam", 27.4728, 94.9120),
    ("Shillong, Meghalaya", 25.5788, 91.8933),
    ("Aizawl, Mizoram", 23.7271, 92.7176),
    ("Kohima, Nagaland", 25.6751, 94.1086),
    ("Itanagar, Arunachal Pradesh", 27.0844, 93.6053),
    # Punjab, Haryana, J&K (plains parts)
    ("Chandigarh", 30.7333, 76.7794),
    ("Amritsar, Punjab", 31.6340, 74.8723),
    ("Ludhiana, Punjab", 30.9010, 75.8573),
    ("Jalandhar, Punjab", 31.3260, 75.5762),
    ("Karnal, Haryana", 29.6857, 76.9905),
    ("Hisar, Haryana", 29.1492, 75.7217),
    ("Jammu, J&K", 32.7266, 74.8570),
    # Telangana & AP interior
    ("Warangal, Telangana", 17.9784, 79.5941),
    ("Nizamabad, Telangana", 18.6725, 78.0941),
    ("Tirupati, Andhra Pradesh", 13.6288, 79.4192),
    ("Nellore, Andhra Pradesh", 14.4426, 79.9865),
]

# 30 hill stations (cloudburst-prone zones used in off-season)
HILL_STATIONS = [
    ("Shimla, Himachal Pradesh", 31.1048, 77.1734),
    ("Manali, Himachal Pradesh", 32.2396, 77.1887),
    ("Dharamshala, Himachal Pradesh", 32.2190, 76.3234),
    ("Kullu, Himachal Pradesh", 31.9579, 77.1095),
    ("Kasauli, Himachal Pradesh", 30.8980, 76.9654),
    ("Keylong, Himachal Pradesh", 32.5720, 77.0330),
    ("Chamba, Himachal Pradesh", 32.5534, 76.1258),
    ("Dalhousie, Himachal Pradesh", 32.5387, 75.9710),
    ("Mussoorie, Uttarakhand", 30.4598, 78.0644),
    ("Nainital, Uttarakhand", 29.3803, 79.4636),
    ("Almora, Uttarakhand", 29.5971, 79.6591),
    ("Pithoragarh, Uttarakhand", 29.5829, 80.2182),
    ("Uttarkashi, Uttarakhand", 30.7268, 78.4354),
    ("Chamoli, Uttarakhand", 30.4030, 79.3210),
    ("Badrinath, Uttarakhand", 30.7440, 79.4930),
    ("Kedarnath, Uttarakhand", 30.7352, 79.0669),
    ("Auli, Uttarakhand", 30.5281, 79.5671),
    ("Munsiyari, Uttarakhand", 30.0670, 80.2330),
    ("Darjeeling, West Bengal", 27.0410, 88.2663),
    ("Kalimpong, West Bengal", 27.0594, 88.4695),
    ("Gangtok, Sikkim", 27.3389, 88.6065),
    ("Pelling, Sikkim", 27.3000, 88.2333),
    ("Leh, Ladakh", 34.1526, 77.5771),
    ("Kargil, Ladakh", 34.5539, 76.1349),
    ("Srinagar, J&K", 34.0837, 74.7973),
    ("Pahalgam, J&K", 34.0161, 75.3150),
    ("Gulmarg, J&K", 34.0484, 74.3805),
    ("Ooty, Tamil Nadu", 11.4102, 76.6950),
    ("Kodaikanal, Tamil Nadu", 10.2381, 77.4892),
    ("Munnar, Kerala", 10.0889, 77.0595),
    ("Mount Abu, Rajasthan", 24.5926, 72.7156),
    ("Matheran, Maharashtra", 18.9866, 73.2681),
    ("Mahabaleshwar, Maharashtra", 17.9237, 73.6586),
    ("Tawang, Arunachal Pradesh", 27.5860, 91.8697),
    ("Cherrapunji, Meghalaya", 25.2720, 91.7322),
]

# ============================================================
# HELPER FUNCTIONS
# ============================================================

YEARS = list(range(2000, 2025))

def random_time_str():
    hour = np.random.randint(0, 24)
    minute = np.random.choice([0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55])
    return f"{hour:02d}:{minute:02d}"

def random_date_in_months(year, months):
    """Generate random date in given months of a year."""
    month = int(np.random.choice(months))
    max_day = {1:31, 2:28, 3:31, 4:30, 5:31, 6:30, 7:31, 8:31, 9:30, 10:31, 11:30, 12:31}
    day = np.random.randint(1, max_day[month] + 1)
    try:
        return datetime(year, month, day)
    except ValueError:
        return datetime(year, month, day - 1)

def gps_noise(lat, lon, scale=0.02):
    """Add small GPS noise to coordinates."""
    return (round(lat + np.random.uniform(-scale, scale), 4),
            round(lon + np.random.uniform(-scale, scale), 4))

def clamp_india(lat, lon):
    """Clamp coordinates to India bounds."""
    return (np.clip(lat, 6.0, 37.0), np.clip(lon, 68.0, 98.0))

def random_non_cb_event():
    """Generate random non-cloudburst weather event."""
    return np.random.choice(
        ["Normal Weather", "Light Rain", "Moderate Rain", "Heavy Rain",
         "Partly Cloudy", "Clear Sky", "Overcast", "Drizzle",
         "Thunderstorm (No Cloudburst)", "Haze", "Fog"],
        p=[0.25, 0.15, 0.10, 0.05, 0.10, 0.08, 0.08, 0.07, 0.04, 0.04, 0.04]
    )

# ============================================================
# 2. GENERATE NON-CLOUDBURST SAMPLES (Target=0)
# ============================================================

print("\n" + "=" * 60)
print("GENERATING NON-CLOUDBURST SAMPLES (6 STRATEGIES)")
print("=" * 60)

negative_samples = []

# ---- STRATEGY 1: Nearby points (10 per event) ----
print("\n[Strategy 1] Nearby geographic points (10 per event)...")
for _, row in valid_df.iterrows():
    for i in range(10):
        distance = np.random.uniform(0.2, 3.0)
        angle = np.random.uniform(0, 2 * np.pi)
        new_lat = row["Latitude"] + distance * np.sin(angle)
        new_lon = row["Longitude"] + distance * np.cos(angle)
        new_lat, new_lon = clamp_india(new_lat, new_lon)

        if pd.notna(row["Date_Parsed"]):
            day_off = int(np.random.randint(0, 5) * np.random.choice([-1, 1]))
            new_date = row["Date_Parsed"] + timedelta(days=day_off)
            date_str = new_date.strftime("%Y-%m-%d")
        else:
            date_str = row["Date"]

        loc_name = row["Location"] if pd.notna(row["Location"]) else "Unknown"
        direction = np.random.choice(["North", "South", "East", "West",
                                       "Northeast", "Northwest", "Southeast", "Southwest"])

        negative_samples.append({
            "Date": date_str,
            "Location": f"{direction} of {loc_name}",
            "Latitude": round(new_lat, 4),
            "Longitude": round(new_lon, 4),
            "Time": random_time_str(),
            "Casualties": 0,
            "Event": random_non_cb_event(),
            "Source": "Synthetic - Nearby Point",
            "Target": 0,
        })
print(f"  Generated: {len(negative_samples)} samples")

# ---- STRATEGY 2: Same location, dry season (4 per event) ----
print("[Strategy 2] Same locations during dry season (4 per event)...")
count_before = len(negative_samples)
DRY_MONTHS = [11, 12, 1, 2, 3]
for _, row in valid_df.iterrows():
    if pd.isna(row["Date_Parsed"]):
        continue
    for _ in range(4):
        year = row["Date_Parsed"].year
        try:
            dry_date = random_date_in_months(year, DRY_MONTHS)
            date_str = dry_date.strftime("%Y-%m-%d")
        except ValueError:
            continue

        lat, lon = gps_noise(row["Latitude"], row["Longitude"], 0.01)

        negative_samples.append({
            "Date": date_str,
            "Location": row["Location"] if pd.notna(row["Location"]) else "Unknown",
            "Latitude": lat,
            "Longitude": lon,
            "Time": random_time_str(),
            "Casualties": 0,
            "Event": np.random.choice(["Clear Weather", "Normal Weather", "Fog", "Haze"],
                                       p=[0.40, 0.30, 0.15, 0.15]),
            "Source": "Synthetic - Dry Season",
            "Target": 0,
        })
print(f"  Generated: {len(negative_samples) - count_before} samples")

# ---- STRATEGY 3: Indian cities during monsoon (10 per city) ----
print("[Strategy 3] 80 Indian cities during monsoon (10 per city)...")
count_before = len(negative_samples)
MONSOON_MONTHS = [6, 7, 8, 9]

for city_name, city_lat, city_lon in INDIAN_CITIES:
    selected_years = np.random.choice(YEARS, size=10, replace=False)
    for yr in selected_years:
        try:
            monsoon_date = random_date_in_months(int(yr), MONSOON_MONTHS)
            date_str = monsoon_date.strftime("%Y-%m-%d")
        except ValueError:
            continue

        lat, lon = gps_noise(city_lat, city_lon)
        negative_samples.append({
            "Date": date_str,
            "Location": city_name,
            "Latitude": lat,
            "Longitude": lon,
            "Time": random_time_str(),
            "Casualties": 0,
            "Event": random_non_cb_event(),
            "Source": "Synthetic - City Monsoon",
            "Target": 0,
        })
print(f"  Generated: {len(negative_samples) - count_before} samples")

# ---- STRATEGY 4: Hill stations during shoulder & winter season (8 per station) ----
print("[Strategy 4] 35 hill stations in off-season (8 per station)...")
count_before = len(negative_samples)
OFF_SEASON_MONTHS = [1, 2, 3, 4, 5, 10, 11, 12]

for hs_name, hs_lat, hs_lon in HILL_STATIONS:
    selected_years = np.random.choice(YEARS, size=8, replace=False)
    for yr in selected_years:
        try:
            off_date = random_date_in_months(int(yr), OFF_SEASON_MONTHS)
            date_str = off_date.strftime("%Y-%m-%d")
        except ValueError:
            continue

        lat, lon = gps_noise(hs_lat, hs_lon, 0.015)
        negative_samples.append({
            "Date": date_str,
            "Location": hs_name,
            "Latitude": lat,
            "Longitude": lon,
            "Time": random_time_str(),
            "Casualties": 0,
            "Event": np.random.choice(
                ["Normal Weather", "Light Rain", "Clear Sky", "Snowfall",
                 "Partly Cloudy", "Fog", "Overcast"],
                p=[0.20, 0.15, 0.15, 0.15, 0.15, 0.10, 0.10]),
            "Source": "Synthetic - Hill Off-Season",
            "Target": 0,
        })
print(f"  Generated: {len(negative_samples) - count_before} samples")

# ---- STRATEGY 5: Indian cities during pre-monsoon (April-May) ----
print("[Strategy 5] Cities during pre-monsoon (April-May)...")
count_before = len(negative_samples)
PRE_MONSOON = [4, 5]

for city_name, city_lat, city_lon in INDIAN_CITIES:
    selected_years = np.random.choice(YEARS, size=5, replace=False)
    for yr in selected_years:
        try:
            pre_date = random_date_in_months(int(yr), PRE_MONSOON)
            date_str = pre_date.strftime("%Y-%m-%d")
        except ValueError:
            continue

        lat, lon = gps_noise(city_lat, city_lon)
        negative_samples.append({
            "Date": date_str,
            "Location": city_name,
            "Latitude": lat,
            "Longitude": lon,
            "Time": random_time_str(),
            "Casualties": 0,
            "Event": np.random.choice(
                ["Normal Weather", "Heatwave", "Dust Storm", "Thunderstorm (No Cloudburst)",
                 "Partly Cloudy", "Clear Sky"],
                p=[0.25, 0.20, 0.15, 0.15, 0.15, 0.10]),
            "Source": "Synthetic - Pre-Monsoon",
            "Target": 0,
        })
print(f"  Generated: {len(negative_samples) - count_before} samples")

# ---- STRATEGY 6: Grid sampling across India ----
# Systematic grid of points across India at ~2° intervals
print("[Strategy 6] Grid sampling across India...")
count_before = len(negative_samples)

lat_grid = np.arange(8.0, 36.0, 2.0)   # 14 latitude bands
lon_grid = np.arange(70.0, 96.0, 2.0)   # 13 longitude bands

for lat_pt in lat_grid:
    for lon_pt in lon_grid:
        # Skip ocean/non-India points (rough mask)
        if lat_pt < 10 and lon_pt > 82:  # Bay of Bengal south
            continue
        if lat_pt < 12 and lon_pt > 85:
            continue
        if lat_pt < 20 and lon_pt > 90:
            continue
        if lon_pt < 72 and lat_pt < 22:  # Arabian Sea
            continue
        if lon_pt < 70:
            continue

        selected_years = np.random.choice(YEARS, size=4, replace=False)
        for yr in selected_years:
            month_sets = [[6,7,8,9], [1,2,3,11,12], [4,5,10]]
            month_set = month_sets[np.random.choice(3, p=[0.5, 0.3, 0.2])]
            try:
                grid_date = random_date_in_months(int(yr), month_set)
                date_str = grid_date.strftime("%Y-%m-%d")
            except:
                continue

            lat, lon = gps_noise(lat_pt, lon_pt, 0.5)
            lat, lon = clamp_india(lat, lon)

            # Determine region name
            if lat_pt >= 28:
                region = "Northern India"
            elif lat_pt >= 22:
                region = "Central India"
            elif lat_pt >= 16:
                region = "Western/Southern India"
            else:
                region = "Southern India"

            negative_samples.append({
                "Date": date_str,
                "Location": f"{region} ({lat:.2f}N, {lon:.2f}E)",
                "Latitude": lat,
                "Longitude": lon,
                "Time": random_time_str(),
                "Casualties": 0,
                "Event": random_non_cb_event(),
                "Source": "Synthetic - Grid Sampling",
                "Target": 0,
            })
print(f"  Generated: {len(negative_samples) - count_before} samples")

# ============================================================
# 3. GENERATE SYNTHETIC CLOUDBURST EVENTS (Target=1)
# ============================================================
# We only have 151 real events — too few for ML. Generate more
# realistic cloudburst events using 3 strategies.

print("\n" + "=" * 60)
print("GENERATING SYNTHETIC CLOUDBURST EVENTS (Target=1)")
print("=" * 60)

synthetic_cb = []

# Known cloudburst-prone zones in India with realistic coordinates
# These are valleys, steep terrain, and high-altitude regions
CLOUDBURST_ZONES = [
    # Uttarakhand valleys & districts
    ("Kedarnath, Uttarakhand", 30.7352, 79.0669),
    ("Chamoli, Uttarakhand", 30.4030, 79.3210),
    ("Pithoragarh, Uttarakhand", 29.5829, 80.2182),
    ("Uttarkashi, Uttarakhand", 30.7268, 78.4354),
    ("Rudraprayag, Uttarakhand", 30.2840, 78.9800),
    ("Tehri Garhwal, Uttarakhand", 30.3900, 78.4800),
    ("Bageshwar, Uttarakhand", 29.8380, 79.7710),
    ("Joshimath, Uttarakhand", 30.5550, 79.5650),
    ("Gopeshwar, Uttarakhand", 30.4100, 79.3200),
    ("Munsiyari, Uttarakhand", 30.0670, 80.2330),
    ("Badrinath, Uttarakhand", 30.7440, 79.4930),
    ("Auli, Uttarakhand", 30.5281, 79.5671),
    ("Harsil, Uttarakhand", 31.0380, 78.7340),
    ("Gangotri, Uttarakhand", 30.9946, 78.9400),
    ("Yamunotri, Uttarakhand", 31.0175, 78.4647),
    ("Devprayag, Uttarakhand", 30.1464, 78.5964),
    ("Nainital, Uttarakhand", 29.3803, 79.4636),
    ("Almora, Uttarakhand", 29.5971, 79.6591),
    # Himachal Pradesh
    ("Kullu, Himachal Pradesh", 31.9579, 77.1095),
    ("Manali, Himachal Pradesh", 32.2396, 77.1887),
    ("Shimla, Himachal Pradesh", 31.1048, 77.1734),
    ("Kinnaur, Himachal Pradesh", 31.5383, 78.2487),
    ("Lahaul-Spiti, Himachal Pradesh", 32.5720, 77.0330),
    ("Chamba, Himachal Pradesh", 32.5534, 76.1258),
    ("Mandi, Himachal Pradesh", 31.7087, 76.9318),
    ("Kangra, Himachal Pradesh", 32.0998, 76.2691),
    ("Dharamshala, Himachal Pradesh", 32.2190, 76.3234),
    ("Rampur, Himachal Pradesh", 31.4490, 77.6300),
    ("Solan, Himachal Pradesh", 30.9045, 77.0967),
    ("Sirmaur, Himachal Pradesh", 30.5700, 77.3000),
    # Jammu & Kashmir and Ladakh
    ("Kishtwar, J&K", 33.3130, 75.7670),
    ("Doda, J&K", 33.1449, 75.5481),
    ("Ramban, J&K", 33.2452, 75.2288),
    ("Anantnag, J&K", 33.7311, 75.1487),
    ("Rajouri, J&K", 33.3790, 74.3120),
    ("Poonch, J&K", 33.7726, 74.0930),
    ("Leh, Ladakh", 34.1526, 77.5771),
    ("Kargil, Ladakh", 34.5539, 76.1349),
    # Sikkim & Darjeeling
    ("Gangtok, Sikkim", 27.3389, 88.6065),
    ("North Sikkim", 27.9500, 88.5500),
    ("Darjeeling, West Bengal", 27.0410, 88.2663),
    ("Kalimpong, West Bengal", 27.0594, 88.4695),
    # Northeast
    ("Tawang, Arunachal Pradesh", 27.5860, 91.8697),
    ("Cherrapunji, Meghalaya", 25.2720, 91.7322),
    ("Itanagar, Arunachal Pradesh", 27.0844, 93.6053),
    # Western Ghats (monsoon cloudbursts)
    ("Mahabaleshwar, Maharashtra", 17.9237, 73.6586),
    ("Lonavala, Maharashtra", 18.7546, 73.4062),
    ("Kodagu, Karnataka", 12.4244, 75.7382),
    ("Wayanad, Kerala", 11.6854, 76.1320),
    ("Idukki, Kerala", 9.8494, 76.9710),
    ("Nilgiris, Tamil Nadu", 11.4102, 76.6950),
]

# Realistic cloudburst time patterns (mostly night/early morning)
def cloudburst_time():
    """Cloudbursts typically occur late night to early morning."""
    pattern = np.random.choice(
        ["night", "early_morning", "afternoon", "evening"],
        p=[0.35, 0.30, 0.20, 0.15]
    )
    if pattern == "night":
        hour = np.random.choice(range(22, 24)) if np.random.rand() > 0.5 else np.random.choice(range(0, 4))
    elif pattern == "early_morning":
        hour = np.random.randint(4, 8)
    elif pattern == "afternoon":
        hour = np.random.randint(13, 18)
    else:
        hour = np.random.randint(18, 22)
    minute = np.random.choice([0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55])
    return f"{hour:02d}:{minute:02d}"

# Realistic casualty distribution for cloudbursts
def cloudburst_casualties():
    """Most cloudbursts: 0-2 casualties, some severe: 3-10, rare: 10+."""
    r = np.random.rand()
    if r < 0.35:
        return 0
    elif r < 0.60:
        return np.random.randint(1, 3)
    elif r < 0.80:
        return np.random.randint(3, 6)
    elif r < 0.93:
        return np.random.randint(6, 12)
    else:
        return np.random.randint(12, 30)

# Peak cloudburst months with realistic probability
CB_MONTHS_PROB = {6: 0.12, 7: 0.25, 8: 0.30, 9: 0.18, 5: 0.05, 10: 0.04,
                  1: 0.01, 2: 0.01, 3: 0.01, 4: 0.01, 11: 0.01, 12: 0.01}
CB_MONTHS = list(CB_MONTHS_PROB.keys())
CB_PROBS = list(CB_MONTHS_PROB.values())

ALL_YEARS = list(range(1995, 2026))

# ---- CB Strategy A: Augment existing events across years ----
# Each real cloudburst location could have events in other years
print("\n[CB Strategy A] Augmenting original events across years...")
for _, row in valid_df.iterrows():
    # Generate 3-5 additional events at this location in different years
    n_extra = np.random.randint(3, 6)
    used_years = set()
    if pd.notna(row["Date_Parsed"]):
        used_years.add(row["Date_Parsed"].year)

    for _ in range(n_extra):
        yr = int(np.random.choice(ALL_YEARS))
        while yr in used_years:
            yr = int(np.random.choice(ALL_YEARS))
        used_years.add(yr)

        month = int(np.random.choice(CB_MONTHS, p=CB_PROBS))
        max_day = {1:31,2:28,3:31,4:30,5:31,6:30,7:31,8:31,9:30,10:31,11:30,12:31}
        day = np.random.randint(1, max_day[month] + 1)
        try:
            cb_date = datetime(yr, month, day)
        except ValueError:
            cb_date = datetime(yr, month, day - 1)

        # Slight coordinate variation (same general area)
        lat, lon = gps_noise(row["Latitude"], row["Longitude"], 0.05)

        synthetic_cb.append({
            "Date": cb_date.strftime("%Y-%m-%d"),
            "Location": row["Location"] if pd.notna(row["Location"]) else "Unknown",
            "Latitude": lat,
            "Longitude": lon,
            "Time": cloudburst_time(),
            "Casualties": int(cloudburst_casualties()),
            "Event": "Cloudburst",
            "Source": "Synthetic - Augmented Event",
            "Target": 1,
        })
print(f"  Generated: {len(synthetic_cb)} cloudburst samples")

# ---- CB Strategy B: Known cloudburst-prone zones ----
# Generate events at 50 known zones across multiple years
print("[CB Strategy B] Cloudburst-prone zones across years...")
count_before = len(synthetic_cb)
for zone_name, zone_lat, zone_lon in CLOUDBURST_ZONES:
    n_events = np.random.randint(4, 8)
    selected_years = np.random.choice(ALL_YEARS, size=n_events, replace=False)
    for yr in selected_years:
        month = int(np.random.choice(CB_MONTHS, p=CB_PROBS))
        max_day = {1:31,2:28,3:31,4:30,5:31,6:30,7:31,8:31,9:30,10:31,11:30,12:31}
        day = np.random.randint(1, max_day[month] + 1)
        try:
            cb_date = datetime(int(yr), month, day)
        except ValueError:
            cb_date = datetime(int(yr), month, day - 1)

        # Variation within the zone (~5-10 km radius)
        lat, lon = gps_noise(zone_lat, zone_lon, 0.08)

        synthetic_cb.append({
            "Date": cb_date.strftime("%Y-%m-%d"),
            "Location": zone_name,
            "Latitude": lat,
            "Longitude": lon,
            "Time": cloudburst_time(),
            "Casualties": int(cloudburst_casualties()),
            "Event": "Cloudburst",
            "Source": "Synthetic - Prone Zone",
            "Target": 1,
        })
print(f"  Generated: {len(synthetic_cb) - count_before} cloudburst samples")

# ---- CB Strategy C: Random high-altitude steep terrain ----
# Generate events at random points along the Himalayan arc
print("[CB Strategy C] Random Himalayan arc events...")
count_before = len(synthetic_cb)

# Himalayan arc segments (lat range, lon range, region name)
HIMALAYAN_ARC = [
    (29.0, 31.5, 78.0, 80.5, "Garhwal Himalayas, Uttarakhand"),
    (29.5, 31.0, 79.5, 81.0, "Kumaon Himalayas, Uttarakhand"),
    (31.0, 33.0, 76.0, 78.5, "Himachal Himalayas"),
    (32.5, 35.0, 74.0, 77.5, "Kashmir Himalayas"),
    (27.0, 28.5, 88.0, 89.5, "Sikkim-Darjeeling Himalayas"),
    (26.5, 28.0, 91.0, 94.0, "Northeast Himalayas"),
    (10.0, 13.0, 73.5, 77.0, "Western Ghats"),
    (25.0, 26.0, 91.0, 92.0, "Meghalaya Hills"),
]

for lat_min, lat_max, lon_min, lon_max, region in HIMALAYAN_ARC:
    n_events = np.random.randint(15, 30)
    for _ in range(n_events):
        yr = int(np.random.choice(ALL_YEARS))
        month = int(np.random.choice(CB_MONTHS, p=CB_PROBS))
        max_day = {1:31,2:28,3:31,4:30,5:31,6:30,7:31,8:31,9:30,10:31,11:30,12:31}
        day = np.random.randint(1, max_day[month] + 1)
        try:
            cb_date = datetime(yr, month, day)
        except ValueError:
            cb_date = datetime(yr, month, day - 1)

        lat = round(np.random.uniform(lat_min, lat_max), 4)
        lon = round(np.random.uniform(lon_min, lon_max), 4)

        synthetic_cb.append({
            "Date": cb_date.strftime("%Y-%m-%d"),
            "Location": f"{region} ({lat:.2f}N, {lon:.2f}E)",
            "Latitude": lat,
            "Longitude": lon,
            "Time": cloudburst_time(),
            "Casualties": int(cloudburst_casualties()),
            "Event": "Cloudburst",
            "Source": "Synthetic - Himalayan Arc",
            "Target": 1,
        })
print(f"  Generated: {len(synthetic_cb) - count_before} cloudburst samples")

total_cb = len(synthetic_cb) + len(valid_df)
print(f"\n  TOTAL CLOUDBURST EVENTS: {total_cb} (151 original + {len(synthetic_cb)} synthetic)")

# ---------------------------
# 4. COMBINE & EXPORT
# ---------------------------

print("\n" + "=" * 60)
print("COMBINING DATASETS")
print("=" * 60)

df_negative = pd.DataFrame(negative_samples)
df_syn_cb = pd.DataFrame(synthetic_cb)
df_positive = df.drop(columns=["Date_Parsed"]).copy()

df_combined = pd.concat([df_positive, df_syn_cb, df_negative], ignore_index=True)
df_combined = df_combined.sample(frac=1, random_state=42).reset_index(drop=True)

print(f"\nFinal dataset shape: {df_combined.shape}")
print(f"\nTarget distribution:")
print(df_combined["Target"].value_counts())
print(f"\nEvent type breakdown:")
print(df_combined["Event"].value_counts())
print(f"\nSource breakdown:")
print(df_combined["Source"].value_counts())

# ---------------------------
# 4. SAVE FINAL DATASET
# ---------------------------

output_file = "cloudburst_full_dataset.xlsx"
df_combined.to_excel(output_file, index=False)
print(f"\n{'=' * 60}")
print(f"DATASET SAVED: {output_file}")
print(f"{'=' * 60}")

# ---------------------------
# 5. STATISTICS
# ---------------------------

print("\n--- Sample Cloudburst Rows (Target=1) ---")
print(df_combined[df_combined["Target"] == 1][
    ["Date", "Location", "Latitude", "Longitude", "Event", "Target"]
].head(5).to_string(index=False))

print("\n--- Sample Non-Cloudburst Rows (Target=0) ---")
print(df_combined[df_combined["Target"] == 0][
    ["Date", "Location", "Latitude", "Longitude", "Event", "Target"]
].head(8).to_string(index=False))

total = len(df_combined)
pos = (df_combined["Target"] == 1).sum()
neg = (df_combined["Target"] == 0).sum()
print(f"\n--- Final Dataset Statistics ---")
print(f"Total rows:            {total}")
print(f"Cloudburst (Target=1): {pos} ({pos/total*100:.1f}%)")
print(f"Normal     (Target=0): {neg} ({neg/total*100:.1f}%)")
print(f"Ratio (Neg:Pos):       {neg/pos:.2f}")
print(f"Latitude range:        {df_combined['Latitude'].min():.4f} - {df_combined['Latitude'].max():.4f}")
print(f"Longitude range:       {df_combined['Longitude'].min():.4f} - {df_combined['Longitude'].max():.4f}")
print(f"Unique locations:      {df_combined['Location'].nunique()}")
print(f"Unique dates:          {df_combined['Date'].nunique()}")
print(f"Null values:           {df_combined.isnull().sum().sum()}")
print("\nDone!")
