import os
import requests
import pandas as pd
from time import sleep

# -----------------------------
# Configuration
# -----------------------------
OUTPUT_DIR = "../data/raw"
os.makedirs(OUTPUT_DIR, exist_ok=True)

START_DATE = "2019-01-01"
END_DATE = "2024-04-30"

URL = "https://archive-api.open-meteo.com/v1/archive"

# -----------------------------
# Regions and Cities
# -----------------------------
regions = {
    "northern": {
        "Delhi": (28.6139, 77.2090),
        "Chandigarh": (30.7333, 76.7794),
        "Lucknow": (26.8467, 80.9462),
        "Jaipur": (26.9124, 75.7873)
    },

    "western": {
        "Mumbai": (19.0760, 72.8777),
        "Ahmedabad": (23.0225, 72.5714),
        "Pune": (18.5204, 73.8567),
        "Surat": (21.1702, 72.8311)
    },

    "eastern": {
        "Kolkata": (22.5726, 88.3639),
        "Bhubaneswar": (20.2961, 85.8245),
        "Patna": (25.5941, 85.1376),
        "Ranchi": (23.3441, 85.3096)
    },

    "southern": {
        "Chennai": (13.0827, 80.2707),
        "Bengaluru": (12.9716, 77.5946),
        "Hyderabad": (17.3850, 78.4867),
        "Kochi": (9.9312, 76.2673)
    },

    "northeastern": {
        "Guwahati": (26.1445, 91.7362),
        "Shillong": (25.5788, 91.8933),
        "Agartala": (23.8315, 91.2868),
        "Imphal": (24.8170, 93.9368)
    }
}

# -----------------------------
# Download Function
# -----------------------------
def download_city(city_name, latitude, longitude):

    params = {
        "latitude": latitude,
        "longitude": longitude,
        "start_date": START_DATE,
        "end_date": END_DATE,
        "hourly": "temperature_2m,relative_humidity_2m,wind_speed_10m,cloud_cover",
        "timezone": "Asia/Kolkata"
    }

    try:

        response = requests.get(URL, params=params, timeout=60)
        response.raise_for_status()

        data = response.json()

        if "hourly" not in data:
            print(f"❌ API Error for {city_name}")
            print(data)
            return None

        df = pd.DataFrame(data["hourly"])
        df["city"] = city_name

        print(f"✅ {city_name} downloaded ({len(df)} rows)")

        return df

    except Exception as e:
        print(f"❌ Failed: {city_name}")
        print(e)
        return None


# -----------------------------
# Main Loop
# -----------------------------
for region_name, cities in regions.items():

    print("\n" + "=" * 60)
    print(f"Downloading {region_name.upper()} Region")
    print("=" * 60)

    regional_data = []

    for city, (lat, lon) in cities.items():

        df = download_city(city, lat, lon)

        if df is not None:
            regional_data.append(df)

        sleep(1)

    if len(regional_data) == 0:
        print(f"No data downloaded for {region_name}")
        continue

    final_df = pd.concat(regional_data, ignore_index=True)

    filename = f"weather_{region_name}_india_multi.csv"

    filepath = os.path.join(OUTPUT_DIR, filename)

    final_df.to_csv(filepath, index=False)

    print(f"\n💾 Saved: {filename}")
    print(f"Rows: {len(final_df):,}")

print("\n")
print("=" * 60)
print("🎉 ALL WEATHER DATA DOWNLOADED SUCCESSFULLY")
print("=" * 60)