import os
import pandas as pd

# ==========================================
# Paths
# ==========================================

RAW_DIR = "data/raw"
PROCESSED_DIR = "data/processed"

os.makedirs(PROCESSED_DIR, exist_ok=True)

# ==========================================
# Region Mapping
# ==========================================

REGIONS = {
    "northern": "Northern Region Hourly Demand",
    "western": "Western Region Hourly Demand",
    "eastern": "Eastern Region Hourly Demand",
    "southern": "Southern Region Hourly Demand",
    "northeastern": "North-Eastern Region Hourly Demand"
}

# ==========================================
# Load Hourly Dataset
# ==========================================

print("Loading hourly electricity dataset...")

load = pd.read_csv(f"{RAW_DIR}/study1_hourly.csv")

load["datetime"] = pd.to_datetime(load["datetime"])

# ==========================================
# Merge Function
# ==========================================

def merge_region(region_name, demand_column):

    print("\n" + "="*60)
    print(f"Merging {region_name.upper()} Region")
    print("="*60)

    weather_file = f"{RAW_DIR}/weather_{region_name}_india_multi.csv"

    if not os.path.exists(weather_file):
        print(f"Weather file missing: {weather_file}")
        return

    weather = pd.read_csv(weather_file)

    weather["time"] = pd.to_datetime(weather["time"])

    # -----------------------------
    # Average Weather
    # -----------------------------

    weather_avg = weather.groupby("time").agg(
        temperature_avg=("temperature_2m", "mean"),
        humidity_avg=("relative_humidity_2m", "mean"),
        wind_speed_avg=("wind_speed_10m", "mean"),
        cloud_cover_avg=("cloud_cover", "mean")
    ).reset_index()

    weather_avg = weather_avg.rename(columns={"time": "datetime"})

    # -----------------------------
    # City-wise Temperatures
    # -----------------------------

    city_temp = weather.pivot(
        index="time",
        columns="city",
        values="temperature_2m"
    )

    city_temp.columns = [
        f"temp_{c.lower().replace(' ', '_')}"
        for c in city_temp.columns
    ]

    city_temp = city_temp.reset_index()

    city_temp = city_temp.rename(columns={"time": "datetime"})

    # -----------------------------
    # Combine Weather
    # -----------------------------

    weather_full = pd.merge(
        weather_avg,
        city_temp,
        on="datetime"
    )

    # -----------------------------
    # Select Demand Columns
    # -----------------------------

    keep_columns = [

        "datetime",

        demand_column,

        "share_res_pct",
        "share_nonfossil_pct",

        "gen_coal_mu",
        "gen_hydro_mu",
        "gen_res_mu",
        "gen_total_mu"

    ]

    load_region = load[keep_columns].copy()

    load_region = load_region.rename(
        columns={demand_column: "load"}
    )

    # -----------------------------
    # Merge
    # -----------------------------

    merged = pd.merge(

        load_region,

        weather_full,

        on="datetime",

        how="inner"

    )

    merged = merged.sort_values("datetime")

    merged = merged.drop_duplicates(subset="datetime")

    merged = merged.set_index("datetime")

    # -----------------------------
    # Save
    # -----------------------------

    output_file = f"{PROCESSED_DIR}/{region_name}_merged.csv"

    merged.to_csv(output_file)

    print(f"Saved: {output_file}")

    print(f"Rows : {len(merged):,}")

    print(f"Columns : {merged.shape[1]}")

    print("Missing Values")

    print(merged.isna().sum().sum())


# ==========================================
# Run All Regions
# ==========================================

for region, demand in REGIONS.items():

    merge_region(region, demand)

print("\n")
print("="*60)
print("ALL REGIONS MERGED SUCCESSFULLY")
print("="*60)