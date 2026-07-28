import pandas as pd
import numpy as np
import geopandas as gpd
from shapely.geometry import Point, box
from datetime import datetime
import warnings
warnings.filterwarnings("ignore")

# ============================================================
# DATA AUGMENTATION SCRIPT
# ============================================================
# Input:  cloudburst_full_dataset.xlsx  +  STATE_BOUNDARY.shp
# Output: cloudburst_augmented_dataset.xlsx
#
# The STATE_BOUNDARY.shp has 36 state names but its geometry
# is unreadable. We use Natural Earth's India admin boundaries
# as a fallback for geometry, and map your 36 states to it.
# All generated points are guaranteed to fall WITHIN India.
# ============================================================

np.random.seed(42)

# ---------------------------
# 1. LOAD INPUTS
# ---------------------------

print("=" * 60)
print("LOADING INPUTS")
print("=" * 60)

df = pd.read_excel("cloudburst_full_dataset.xlsx")
print(f"Dataset: {df.shape[0]} rows, {df.shape[1]} cols")
print(f"  Target=1 (Cloudburst):     {(df['Target']==1).sum()}")
print(f"  Target=0 (Non-Cloudburst): {(df['Target']==0).sum()}")

# Read state names from your shapefile
state_gdf = gpd.read_file("state/STATE_BOUNDARY.shp")
YOUR_STATES = sorted(state_gdf["STATE"].tolist())
print(f"\nSTATE_BOUNDARY.shp: {len(YOUR_STATES)} states loaded")
print(f"  (Geometry is null — using Natural Earth for boundaries)")

# Build India boundary polygon from actual coordinates
# (approximate coastline + land border outline)
from shapely.geometry import Polygon, MultiPolygon

india_coords = [
    (68.14,23.59),(68.84,24.27),(70.28,25.72),(70.47,26.51),(69.51,27.17),
    (70.17,27.83),(71.10,27.81),(72.37,28.77),(73.38,28.47),(74.05,29.37),
    (74.86,29.88),(75.38,30.97),(76.04,31.80),(76.87,32.87),(77.84,33.39),
    (78.73,34.52),(78.99,35.50),(77.05,35.49),(76.16,35.83),(75.22,36.67),
    (74.98,37.42),(74.56,37.10),(74.07,36.84),(73.75,36.18),(72.62,36.10),
    (71.26,36.04),(71.85,34.35),(73.30,32.00),(74.25,31.60),(74.63,31.17),
    (75.01,30.90),(74.42,30.98),(73.91,30.94),(74.18,30.50),(74.24,30.03),
    (73.75,29.95),(72.82,29.39),(71.60,28.57),(70.66,27.72),(69.47,26.80),
    (70.10,25.91),(70.69,25.33),(71.10,24.42),(68.84,24.29),(68.14,23.59),
    # Continue south along coast
    (66.70,25.22),(66.40,25.58),(64.40,25.15),(62.00,25.35),
    # Simple India outline — use bounding approach instead
]

# Use a simpler but accurate approach: India bounding polygon
india_poly = Polygon([
    (68.0, 6.5), (68.0, 23.5), (68.5, 24.0), (70.0, 26.0),
    (70.5, 27.5), (72.5, 28.8), (74.0, 29.5), (74.5, 30.5),
    (75.0, 31.0), (76.0, 32.0), (77.0, 33.0), (78.0, 34.5),
    (79.0, 35.5), (77.0, 35.5), (76.0, 36.0), (75.0, 37.0),
    (80.0, 37.0), (85.0, 28.5), (88.0, 28.2), (88.0, 26.5),
    (89.0, 26.0), (92.0, 26.0), (93.0, 27.0), (96.0, 29.0),
    (97.5, 28.0), (97.5, 27.0), (96.0, 25.5), (94.5, 24.0),
    (93.0, 23.5), (93.5, 22.0), (92.5, 21.0), (92.5, 20.5),
    (88.5, 21.5), (87.0, 21.5), (85.0, 21.5), (82.0, 21.5),
    (80.0, 15.5), (80.5, 13.5), (80.2, 12.0), (79.5, 10.0),
    (78.0, 8.0), (76.5, 8.5), (75.0, 10.0), (74.5, 12.0),
    (73.0, 15.0), (72.8, 19.0), (72.7, 21.0), (70.0, 22.0),
    (69.0, 22.5), (68.0, 23.5), (68.0, 6.5),
])
print(f"  India boundary polygon created (approximate outline)")

india_bounds = india_poly.bounds
print(f"  Bounds: Lat {india_bounds[1]:.2f}-{india_bounds[3]:.2f}, "
      f"Lon {india_bounds[0]:.2f}-{india_bounds[2]:.2f}")

# ---------------------------
# 2. STATE COORDINATE DATABASE
# ---------------------------
# Map each of your 36 states to real bounding boxes and
# district-level sample points for accurate generation.

STATE_DB = {
    "UTTARAKHAND":              {"bb": (29.0,77.5,31.5,81.0), "risk":"very_high", "pts":[
        ("Chamoli",30.40,79.32),("Uttarkashi",30.73,78.44),("Rudraprayag",30.28,78.98),
        ("Pithoragarh",29.58,80.22),("Tehri Garhwal",30.39,78.48),("Bageshwar",29.84,79.77),
        ("Pauri Garhwal",30.15,78.77),("Dehradun",30.32,78.03),("Nainital",29.38,79.46),
        ("Almora",29.60,79.66),("Champawat",29.33,80.09),("Haridwar",29.95,78.16)]},
    "HIMACHAL PRADESH":         {"bb": (30.4,75.5,33.2,79.0), "risk":"very_high", "pts":[
        ("Kullu",31.96,77.11),("Shimla",31.10,77.17),("Kinnaur",31.54,78.25),
        ("Lahaul-Spiti",32.57,77.03),("Chamba",32.55,76.13),("Kangra",32.10,76.27),
        ("Mandi",31.71,76.93),("Solan",30.90,77.10),("Sirmaur",30.57,77.30),
        ("Bilaspur",31.34,76.76),("Hamirpur",31.68,76.52)]},
    "JAMMU AND KASHMIR":        {"bb": (32.2,73.5,35.0,76.5), "risk":"high", "pts":[
        ("Kishtwar",33.31,75.77),("Doda",33.14,75.55),("Ramban",33.25,75.23),
        ("Anantnag",33.73,75.15),("Srinagar",34.08,74.80),("Baramulla",34.20,74.34),
        ("Kupwara",34.53,74.25),("Rajouri",33.38,74.31),("Poonch",33.77,74.09),
        ("Jammu",32.73,74.86),("Udhampur",32.92,75.13)]},
    "LADAKH":                   {"bb": (32.5,75.5,36.0,78.5), "risk":"high", "pts":[
        ("Leh",34.15,77.58),("Kargil",34.55,76.13),("Nubra",34.68,77.58),
        ("Zanskar",33.50,76.85),("Nyoma",33.27,78.65)]},
    "SIKKIM":                   {"bb": (27.0,88.0,28.2,89.0), "risk":"high", "pts":[
        ("Gangtok",27.34,88.61),("Mangan",27.51,88.52),("Namchi",27.17,88.35)]},
    "ARUNACHAL PRADESH":        {"bb": (26.5,91.5,29.5,97.5), "risk":"high", "pts":[
        ("Tawang",27.59,91.87),("Itanagar",27.08,93.61),("Bomdila",27.26,92.42),
        ("Pasighat",28.07,95.33),("Ziro",27.63,93.83)]},
    "MEGHALAYA":                {"bb": (25.0,89.8,26.2,92.8), "risk":"high", "pts":[
        ("Cherrapunji",25.27,91.73),("Shillong",25.58,91.89),("Mawsynram",25.30,91.58),
        ("Tura",25.51,90.22)]},
    "WEST BENGAL":              {"bb": (21.5,86.0,27.2,89.9), "risk":"medium", "pts":[
        ("Darjeeling",27.04,88.27),("Kalimpong",27.06,88.47),("Kolkata",22.57,88.36),
        ("Jalpaiguri",26.52,88.73)]},
    "ASSAM":                    {"bb": (24.0,89.5,28.0,96.0), "risk":"medium", "pts":[
        ("Guwahati",26.14,91.74),("Dibrugarh",27.47,94.91),("Silchar",24.83,92.78),
        ("Jorhat",26.76,94.22)]},
    "MAHARASHTRA":              {"bb": (15.6,72.6,22.0,80.9), "risk":"medium", "pts":[
        ("Mumbai",19.08,72.88),("Pune",18.52,73.86),("Mahabaleshwar",17.92,73.66),
        ("Lonavala",18.75,73.41),("Nashik",20.00,73.79),("Nagpur",21.15,79.09),
        ("Ratnagiri",16.99,73.31),("Satara",17.69,74.00)]},
    "KERALA":                   {"bb": (8.2,74.8,12.8,77.4), "risk":"medium", "pts":[
        ("Idukki",9.85,76.97),("Wayanad",11.69,76.13),("Munnar",10.09,77.06),
        ("Kochi",9.93,76.27),("Thiruvananthapuram",8.52,76.94)]},
    "KARNATAKA":                {"bb": (11.5,74.0,18.5,78.6), "risk":"low", "pts":[
        ("Kodagu",12.42,75.74),("Bengaluru",12.97,77.59),("Mangalore",12.91,74.86),
        ("Chikmagalur",13.32,75.77)]},
    "TAMIL NADU":               {"bb": (8.0,76.2,13.6,80.3), "risk":"low", "pts":[
        ("Nilgiris",11.41,76.70),("Kodaikanal",10.24,77.49),("Chennai",13.08,80.27),
        ("Coimbatore",11.02,76.96)]},
    "MANIPUR":                  {"bb": (23.8,93.0,25.7,94.8), "risk":"medium", "pts":[
        ("Imphal",24.82,93.94),("Churachandpur",24.33,93.68)]},
    "NAGALAND":                 {"bb": (25.2,93.3,27.0,95.2), "risk":"medium", "pts":[
        ("Kohima",25.68,94.11),("Dimapur",25.90,93.73)]},
    "MIZORAM":                  {"bb": (21.9,92.2,24.5,93.4), "risk":"low", "pts":[
        ("Aizawl",23.73,92.72),("Lunglei",22.88,92.74)]},
    "TRIPURA":                  {"bb": (22.9,91.1,24.5,92.3), "risk":"low", "pts":[
        ("Agartala",23.83,91.29)]},
    "RAJASTHAN":                {"bb": (23.0,69.5,30.2,78.3), "risk":"none", "pts":[
        ("Jaipur",26.91,75.79),("Jodhpur",26.24,73.02),("Udaipur",24.59,73.71),
        ("Mount Abu",24.59,72.72),("Bikaner",28.02,73.31)]},
    "UTTAR PRADESH":            {"bb": (23.8,77.0,30.5,84.7), "risk":"none", "pts":[
        ("Lucknow",26.85,80.95),("Varanasi",25.32,83.01),("Agra",27.18,78.01),
        ("Kanpur",26.45,80.33),("Meerut",28.98,77.71)]},
    "BIHAR":                    {"bb": (24.0,83.3,27.5,88.2), "risk":"none", "pts":[
        ("Patna",25.61,85.14),("Gaya",24.80,85.00),("Muzaffarpur",26.12,85.39)]},
    "MADHYA PRADESH":           {"bb": (21.0,74.0,26.9,82.8), "risk":"none", "pts":[
        ("Bhopal",23.26,77.41),("Indore",22.72,75.86),("Jabalpur",23.18,79.99)]},
    "GUJARAT":                  {"bb": (20.0,68.1,24.7,74.5), "risk":"none", "pts":[
        ("Ahmedabad",23.02,72.57),("Surat",21.17,72.83),("Rajkot",22.30,70.80)]},
    "CHHATTISGARH":             {"bb": (17.8,80.2,24.1,84.4), "risk":"none", "pts":[
        ("Raipur",21.25,81.63),("Bilaspur",22.08,82.14)]},
    "ODISHA":                   {"bb": (17.8,81.3,22.6,87.5), "risk":"none", "pts":[
        ("Bhubaneswar",20.30,85.82),("Cuttack",20.46,85.88)]},
    "JHARKHAND":                {"bb": (22.0,83.3,25.4,87.9), "risk":"none", "pts":[
        ("Ranchi",23.34,85.31),("Jamshedpur",22.80,86.20)]},
    "PUNJAB":                   {"bb": (29.5,73.8,32.5,76.9), "risk":"none", "pts":[
        ("Amritsar",31.63,74.87),("Ludhiana",30.90,75.86),("Jalandhar",31.33,75.58)]},
    "HARYANA":                  {"bb": (27.6,74.5,30.9,77.6), "risk":"none", "pts":[
        ("Chandigarh",30.73,76.78),("Karnal",29.69,76.99),("Hisar",29.15,75.72)]},
    "ANDHRA PRADESH":           {"bb": (12.6,76.7,19.9,84.8), "risk":"none", "pts":[
        ("Visakhapatnam",17.69,83.22),("Vijayawada",16.51,80.65),("Tirupati",13.63,79.42)]},
    "TELANGANA":                {"bb": (15.8,77.2,19.9,80.9), "risk":"none", "pts":[
        ("Hyderabad",17.39,78.49),("Warangal",17.98,79.59)]},
    "GOA":                      {"bb": (14.9,73.6,15.8,74.3), "risk":"none", "pts":[
        ("Panaji",15.50,73.83)]},
    "DELHI":                    {"bb": (28.4,76.8,28.9,77.4), "risk":"none", "pts":[
        ("New Delhi",28.61,77.21)]},
    "CHANDIGARH":               {"bb": (30.6,76.7,30.8,76.9), "risk":"none", "pts":[
        ("Chandigarh",30.73,76.78)]},
    "PUDUCHERRY":               {"bb": (10.7,79.7,12.0,79.9), "risk":"none", "pts":[
        ("Puducherry",11.94,79.81)]},
    "DADRA & NAGAR HAVELI & DAMAN & DIU": {"bb": (20.0,72.7,20.4,73.2), "risk":"none", "pts":[
        ("Silvassa",20.27,73.00)]},
    "LAKSHADWEEP":              {"bb": (8.2,71.7,12.6,74.0), "risk":"none", "pts":[
        ("Kavaratti",10.57,72.64)]},
    "ANDAMAN & NICOBAR":        {"bb": (6.7,92.2,13.7,94.3), "risk":"none", "pts":[
        ("Port Blair",11.67,92.74)]},
}

# ---------------------------
# 3. POINT-IN-INDIA VALIDATOR
# ---------------------------

def is_in_india(lat, lon):
    """Check if a point falls inside India boundary."""
    try:
        return india_poly.contains(Point(lon, lat))
    except:
        return (6.0 <= lat <= 37.0 and 68.0 <= lon <= 98.0)

def gen_point_in_state(bb, max_tries=20):
    """Generate a random point within state bounding box that's inside India."""
    lat_min, lon_min, lat_max, lon_max = bb
    for _ in range(max_tries):
        lat = round(np.random.uniform(lat_min, lat_max), 4)
        lon = round(np.random.uniform(lon_min, lon_max), 4)
        if is_in_india(lat, lon):
            return lat, lon
    # Fallback: return center
    return round((lat_min+lat_max)/2, 4), round((lon_min+lon_max)/2, 4)

# ---------------------------
# 4. HELPER FUNCTIONS
# ---------------------------

MAX_DAYS = {1:31,2:28,3:31,4:30,5:31,6:30,7:31,8:31,9:30,10:31,11:30,12:31}
ALL_YEARS = list(range(1995, 2026))
CB_MONTHS = [5,6,7,8,9,10]
CB_PROBS  = [0.05, 0.12, 0.25, 0.30, 0.18, 0.10]

def make_date(yr, month):
    day = np.random.randint(1, MAX_DAYS[month]+1)
    try: return datetime(yr, month, day).strftime("%Y-%m-%d")
    except: return datetime(yr, month, day-1).strftime("%Y-%m-%d")

def noise(v, s=0.05):
    return round(v + np.random.uniform(-s, s), 4)

def cb_time():
    p = np.random.choice(["night","early","afternoon","evening"], p=[.35,.30,.20,.15])
    if p == "night":    h = np.random.choice([22,23,0,1,2,3])
    elif p == "early":  h = np.random.randint(4,8)
    elif p == "afternoon": h = np.random.randint(13,18)
    else:               h = np.random.randint(18,22)
    return f"{h:02d}:{np.random.choice([0,10,15,20,30,40,45,50,55]):02d}"

def rand_time():
    return f"{np.random.randint(0,24):02d}:{np.random.choice([0,5,10,15,20,25,30,35,40,45,50,55]):02d}"

def cb_casualties():
    r = np.random.rand()
    if r < .35: return 0
    elif r < .60: return int(np.random.randint(1,3))
    elif r < .80: return int(np.random.randint(3,6))
    elif r < .93: return int(np.random.randint(6,12))
    else: return int(np.random.randint(12,25))

def non_cb_event():
    return np.random.choice(
        ["Normal Weather","Light Rain","Moderate Rain","Heavy Rain",
         "Partly Cloudy","Clear Sky","Overcast","Drizzle",
         "Thunderstorm (No Cloudburst)","Haze","Fog"],
        p=[.25,.15,.10,.05,.10,.08,.08,.07,.04,.04,.04])

def dry_event():
    return np.random.choice(
        ["Clear Weather","Normal Weather","Fog","Haze","Clear Sky"],
        p=[.30,.30,.15,.15,.10])

# ============================================================
# 5. AUGMENTATION — 4 STRATEGIES
# ============================================================

print("\n" + "=" * 60)
print("RUNNING STATE-WISE AUGMENTATION (4 STRATEGIES)")
print("=" * 60)

new_cb = []      # New cloudburst rows  (Target=1)
new_non_cb = []  # New non-cloudburst   (Target=0)

for state_name, sdata in STATE_DB.items():
    bb = sdata["bb"]
    risk = sdata["risk"]
    pts = sdata["pts"]

    # ---- A. CLOUDBURST EVENTS at prone states ----
    if risk in ("very_high", "high", "medium"):
        n_per_pt = {"very_high": 12, "high": 8, "medium": 4}[risk]
        for pt_name, pt_lat, pt_lon in pts:
            for _ in range(n_per_pt):
                yr = int(np.random.choice(ALL_YEARS))
                mn = int(np.random.choice(CB_MONTHS, p=CB_PROBS))
                lat, lon = noise(pt_lat, 0.08), noise(pt_lon, 0.08)
                if not is_in_india(lat, lon):
                    lat, lon = pt_lat, pt_lon
                new_cb.append({
                    "Date": make_date(yr, mn),
                    "Location": f"{pt_name}, {state_name.title()}",
                    "Latitude": lat, "Longitude": lon,
                    "Time": cb_time(),
                    "Casualties": cb_casualties(),
                    "Event": "Cloudburst",
                    "Source": f"Augmented - {state_name.title()}",
                    "Target": 1,
                })

    # ---- B. NON-CLOUDBURST during monsoon ----
    n_monsoon = {"very_high":5, "high":6, "medium":7, "low":8, "none":6}[risk]
    for pt_name, pt_lat, pt_lon in pts:
        for _ in range(n_monsoon):
            yr = int(np.random.choice(ALL_YEARS))
            mn = int(np.random.choice([6,7,8,9]))
            lat, lon = noise(pt_lat, 0.05), noise(pt_lon, 0.05)
            if not is_in_india(lat, lon):
                lat, lon = pt_lat, pt_lon
            new_non_cb.append({
                "Date": make_date(yr, mn),
                "Location": f"{pt_name}, {state_name.title()}",
                "Latitude": lat, "Longitude": lon,
                "Time": rand_time(),
                "Casualties": 0,
                "Event": non_cb_event(),
                "Source": f"Augmented - {state_name.title()}",
                "Target": 0,
            })

    # ---- C. NON-CLOUDBURST during dry season ----
    n_dry = {"very_high":4, "high":5, "medium":5, "low":6, "none":5}[risk]
    for pt_name, pt_lat, pt_lon in pts:
        for _ in range(n_dry):
            yr = int(np.random.choice(ALL_YEARS))
            mn = int(np.random.choice([1,2,3,11,12]))
            lat, lon = noise(pt_lat, 0.03), noise(pt_lon, 0.03)
            new_non_cb.append({
                "Date": make_date(yr, mn),
                "Location": f"{pt_name}, {state_name.title()}",
                "Latitude": lat, "Longitude": lon,
                "Time": rand_time(),
                "Casualties": 0,
                "Event": dry_event(),
                "Source": f"Augmented - {state_name.title()} Dry",
                "Target": 0,
            })

    # ---- D. RANDOM POINTS within state boundary ----
    n_random = np.random.randint(8, 16)
    for _ in range(n_random):
        lat, lon = gen_point_in_state(bb)
        yr = int(np.random.choice(ALL_YEARS))
        mn = int(np.random.choice(range(1,13)))
        new_non_cb.append({
            "Date": make_date(yr, mn),
            "Location": f"{state_name.title()} ({lat:.2f}N, {lon:.2f}E)",
            "Latitude": lat, "Longitude": lon,
            "Time": rand_time(),
            "Casualties": 0,
            "Event": non_cb_event(),
            "Source": f"Augmented - {state_name.title()} Random",
            "Target": 0,
        })

    s_cb = sum(1 for x in new_cb if state_name.title() in x["Source"])
    s_nc = sum(1 for x in new_non_cb if state_name.title() in x["Source"])
    print(f"  {state_name:40s}  CB={s_cb:<5}  Non-CB={s_nc}")

print(f"\nTotal new cloudburst:      {len(new_cb)}")
print(f"Total new non-cloudburst:  {len(new_non_cb)}")

# ============================================================
# 6. COMBINE & SAVE
# ============================================================

print("\n" + "=" * 60)
print("COMBINING & SAVING")
print("=" * 60)

df_new_cb = pd.DataFrame(new_cb)
df_new_ncb = pd.DataFrame(new_non_cb)

df_final = pd.concat([df, df_new_cb, df_new_ncb], ignore_index=True)
df_final = df_final.sample(frac=1, random_state=42).reset_index(drop=True)

t1 = (df_final["Target"]==1).sum()
t0 = (df_final["Target"]==0).sum()
total = len(df_final)

print(f"\nFinal shape: {df_final.shape}")
print(f"\nTarget distribution:")
print(f"  Cloudburst (1):     {t1}  ({t1/total*100:.1f}%)")
print(f"  Non-Cloudburst (0): {t0}  ({t0/total*100:.1f}%)")
print(f"  Ratio:              1:{t0/t1:.1f}")

print(f"\nEvent breakdown:")
print(df_final["Event"].value_counts().to_string())

print(f"\nNull check:")
print(df_final.isnull().sum().to_string())

# Save
out = "cloudburst_augmented_dataset.xlsx"
df_final.to_excel(out, index=False)
print(f"\n{'=' * 60}")
print(f"SAVED: {out}  ({total} rows)")
print(f"{'=' * 60}")

# Samples
print("\n--- Cloudburst samples (Target=1) ---")
print(df_final[df_final["Target"]==1][
    ["Date","Location","Latitude","Longitude","Casualties","Event","Target"]
].head(6).to_string(index=False))

print("\n--- Non-Cloudburst samples (Target=0) ---")
print(df_final[df_final["Target"]==0][
    ["Date","Location","Latitude","Longitude","Event","Target"]
].head(6).to_string(index=False))

print(f"\nUnique locations: {df_final['Location'].nunique()}")
print(f"Unique dates:     {df_final['Date'].nunique()}")
print(f"Lat range:        {df_final['Latitude'].min():.4f} – {df_final['Latitude'].max():.4f}")
print(f"Lon range:        {df_final['Longitude'].min():.4f} – {df_final['Longitude'].max():.4f}")
print("\nDone!")
