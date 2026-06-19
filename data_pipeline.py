"""
data_pipeline.py
-----------------
Starter pipeline for the "Net-Zero Smart Energy Zones" project (Sambalpur test bed).

Builds the consolidated hourly ledger described in Section 4 of the Master Blueprint:
  - REAL:               Ambient_Temp_C, Solar_Rad_kWh_m2          (NASA POWER hourly API)
  - REAL (manual file):  Grid_Load_MW                              (Kaggle India hourly load CSV)
  - REAL (manual file):  Emission_Factor_kgCO2_per_MWh             (CEA CO2 Baseline Database)
  - POLICY-CALIBRATED:   Carbon_Credit_Price_INR                   (bounded around the CCTS Rs 250-1500 band)
  - GROUNDED-SIMULATED:  Zone_Soil_Moisture_Pct, Sequestration_Rate_kgCO2  (driven by real weather + allometric model)
  - DERIVED:             Net_Carbon_Balance

Two manual downloads are required before running this end to end (see TODO markers below):
  1. Kaggle "Hourly Load India" dataset  -> data/raw/india_hourly_load.csv
  2. CEA CO2 Baseline Database (V21.0)   -> note the all-India / regional emission factor value(s)

Usage:
    python data_pipeline.py --year 2024 --lat 21.4669 --lon 83.9756 --out data/processed/sambalpur_grid_telemetry.csv
"""

import argparse
import datetime as dt
import os
import time
import numpy as np
import pandas as pd
import requests

# --------------------------------------------------------------------------------------
# 1. REAL DATA: NASA POWER hourly weather
# --------------------------------------------------------------------------------------

NASA_POWER_HOURLY_URL = "https://power.larc.nasa.gov/api/temporal/hourly/point"


def fetch_nasa_power_weather(lat: float, lon: float, start: str, end: str) -> pd.DataFrame:
    """
    Pull hourly temperature + solar radiation from the free NASA POWER API.
    No API key required. CC BY 4.0 licensed, hourly data available from 2001 to near-real-time.

    start/end format: 'YYYYMMDD'
    """
    params = {
        "parameters": "T2M,ALLSKY_SFC_SW_DWN,PRECTOTCORR",
        "community": "RE",
        "longitude": lon,
        "latitude": lat,
        "start": start,
        "end": end,
        "format": "JSON",
        "time-standard": "UTC",
    }
    resp = requests.get(NASA_POWER_HOURLY_URL, params=params, timeout=120)
    resp.raise_for_status()
    payload = resp.json()

    params_data = payload["properties"]["parameter"]
    temp = params_data["T2M"]
    solar = params_data["ALLSKY_SFC_SW_DWN"]
    precip = params_data["PRECTOTCORR"]

    idx = pd.to_datetime(list(temp.keys()), format="%Y%m%d%H")
    df = pd.DataFrame({
        "Datetime": idx,
        "Ambient_Temp_C": list(temp.values()),
        "Solar_Rad": list(solar.values()),
        "Precip_mm": list(precip.values()),
    }).sort_values("Datetime").reset_index(drop=True)

    # NASA POWER uses -999 as a missing-value sentinel; convert to NaN then interpolate
    df = df.replace(-999.0, np.nan)
    df[["Ambient_Temp_C", "Solar_Rad", "Precip_mm"]] = df[
        ["Ambient_Temp_C", "Solar_Rad", "Precip_mm"]
    ].interpolate(limit_direction="both")
    return df


# Approximate monthly climate normals for Sambalpur, Odisha (tropical wet-and-dry climate:
# very hot pre-monsoon summer, monsoon Jun-Sep, mild winter). These are general climatological
# patterns, NOT a substitute for the real NASA POWER series -- only used if the live API call
# fails (e.g. no network access to power.larc.nasa.gov from this environment).
SAMBALPUR_MONTHLY_TEMP_C = {
    1: 19, 2: 23, 3: 29, 4: 34, 5: 36, 6: 33,
    7: 29, 8: 28, 9: 28, 10: 27, 11: 23, 12: 19,
}
SAMBALPUR_MONTHLY_RAIN_MM = {  # rough wet-season weighting, not station-verified
    1: 1, 2: 1, 3: 1, 4: 2, 5: 4, 6: 12,
    7: 16, 8: 15, 9: 10, 10: 4, 11: 1, 12: 1,
}


def climatology_fallback_weather(start: str, end: str) -> pd.DataFrame:
    """
    FALLBACK (used only when the NASA POWER API cannot be reached from this environment).
    Builds an hourly weather series from monthly climate normals + a diurnal sinusoidal
    swing + noise. This is clearly a stand-in, not measured data -- replace by re-running
    this script with full internet access so it can pull the real NASA POWER series.
    """
    idx = pd.date_range(
        start=dt.datetime.strptime(start, "%Y%m%d"),
        end=dt.datetime.strptime(end, "%Y%m%d") + dt.timedelta(hours=23),
        freq="h",
    )
    rng = np.random.default_rng(0)
    month_mean = idx.month.map(SAMBALPUR_MONTHLY_TEMP_C).to_numpy(dtype=float)
    month_rain = idx.month.map(SAMBALPUR_MONTHLY_RAIN_MM).to_numpy(dtype=float)
    hour = idx.hour.to_numpy()

    diurnal_swing = 7.0 * np.sin((hour - 9) / 24 * 2 * np.pi)  # warmest mid-afternoon
    noise = rng.normal(0, 1.3, size=len(idx))
    temp = month_mean + diurnal_swing + noise

    daylight = np.clip(np.sin((hour - 6) / 12 * np.pi), 0, 1)
    solar = daylight * (4.5 + 1.5 * np.sin(2 * np.pi * (idx.dayofyear.to_numpy() - 80) / 365)) \
        + rng.normal(0, 0.2, size=len(idx))
    solar = np.clip(solar, 0, None)

    rain_chance = np.clip(month_rain / 20.0, 0, 1)
    precip = (rng.random(len(idx)) < rain_chance / 24) * rng.exponential(4.0, size=len(idx))

    return pd.DataFrame({
        "Datetime": idx, "Ambient_Temp_C": temp, "Solar_Rad": solar, "Precip_mm": precip,
    })


def get_weather(lat: float, lon: float, start: str, end: str) -> tuple[pd.DataFrame, str]:
    """Try the real NASA POWER API first; fall back to climatology if unreachable."""
    try:
        df = fetch_nasa_power_weather(lat, lon, start, end)
        return df, "REAL (NASA POWER API)"
    except Exception as e:
        print(f"      [FALLBACK] NASA POWER unreachable ({e.__class__.__name__}: {e}).")
        print("      [FALLBACK] Using calibrated climatological demo weather instead.")
        print("      [FALLBACK] Re-run this script from an environment with internet access")
        print("      [FALLBACK] to power.larc.nasa.gov to pull the real hourly series.")
        df = climatology_fallback_weather(start, end)
        return df, "DEMO (climatology fallback, NOT live NASA data)"


# --------------------------------------------------------------------------------------
# 2. REAL DATA: Grid load (manual download) + CEA emission factor
# --------------------------------------------------------------------------------------

# --------------------------------------------------------------------------------------
# 2. REAL DATA: Grid load (manual download) + CEA emission factor
# --------------------------------------------------------------------------------------

ANNUAL_CEA_EF = {
    2019: 0.81,
    2020: 0.79,
    2021: 0.77,
    2022: 0.74,
    2023: 0.7117
}

REGIONS_CONFIG = {
    "East": {
        "lat": 21.4669,
        "lon": 83.9756,
        "load_col": "Eastern Region Hourly Demand",
        "biomass_low": 60.0,
        "biomass_high": 140.0,
        "carbon_fraction": 0.47
    },
    "West": {
        "lat": 26.2389,
        "lon": 73.0243,
        "load_col": "Western Region Hourly Demand",
        "biomass_low": 10.0,
        "biomass_high": 30.0,
        "carbon_fraction": 0.45
    },
    "South": {
        "lat": 11.6854,
        "lon": 76.1320,
        "load_col": "Southern Region Hourly Demand",
        "biomass_low": 150.0,
        "biomass_high": 300.0,
        "carbon_fraction": 0.49
    }
}

def load_entire_grid_load(path: str) -> pd.DataFrame:
    """Load the entire raw excel file to avoid redundant reads."""
    if not os.path.exists(path):
        alternative_paths = [
            "hourlyLoadDataIndia.xlsx",
            "data/raw/hourlyLoadDataIndia.xlsx",
            "india_hourly_load.csv"
        ]
        for alt in alternative_paths:
            if os.path.exists(alt):
                path = alt
                break
                
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Grid load file not found at {path}. Place the Excel file at workspace root "
            "or data/raw/ directory."
        )
        
    print(f"      Loading entire grid load file from: {path} ...")
    if path.endswith('.xlsx'):
        raw = pd.read_excel(path, sheet_name='Sheet1')
    else:
        raw = pd.read_csv(path)
        
    # Standardize datetime column
    dt_col = None
    for c in ["datetime", "Datetime", "DATETIME"]:
        if c in raw.columns:
            dt_col = c
            break
    if dt_col is None:
        raise KeyError(f"No datetime column found. Columns: {list(raw.columns)}")
        
    raw["Datetime"] = pd.to_datetime(raw[dt_col])
    return raw


def get_grid_load_slice(df_all: pd.DataFrame, load_col: str, start_year: int, end_year: int) -> pd.DataFrame:
    """Slice the grid load data by region column and year range."""
    if load_col not in df_all.columns:
        raise KeyError(f"Column '{load_col}' not found. Available columns: {list(df_all.columns)}")
        
    df_slice = df_all[["Datetime", load_col]].copy()
    df_slice = df_slice.rename(columns={load_col: "Grid_Load_MW"})
    
    # Filter years
    df_slice = df_slice[
        (df_slice["Datetime"].dt.year >= start_year) & 
        (df_slice["Datetime"].dt.year <= end_year)
    ].copy()
    
    return df_slice.sort_values("Datetime").reset_index(drop=True)


def get_grid_load(df_all: pd.DataFrame, load_col: str, start_year: int, end_year: int) -> tuple[pd.DataFrame, str]:
    try:
        df = get_grid_load_slice(df_all, load_col, start_year, end_year)
        if len(df) == 0:
            raise ValueError("No rows matched the requested year range.")
        return df, "REAL (Kaggle hourly-load-India)"
    except Exception as e:
        print(f"      [FALLBACK] Real grid-load slice failed ({e}). Generating synthetic data.")
        # Fallback generator for multi-year range
        idx = pd.date_range(
            start=f"{start_year}-01-01 00:00:00",
            end=f"{end_year}-12-31 23:00:00",
            freq="h"
        )
        rng = np.random.default_rng(11)
        hour = idx.hour.to_numpy()
        morning_peak = 0.35 * np.exp(-((hour - 9) ** 2) / (2 * 2.5 ** 2))
        evening_peak = 0.45 * np.exp(-((hour - 19) ** 2) / (2 * 2.5 ** 2))
        weekend_factor = np.where(idx.dayofweek.isin([5, 6]), 0.92, 1.0)
        seasonal = 1 + 0.12 * np.sin(2 * np.pi * (idx.dayofyear.to_numpy() - 172) / 365)
        noise = rng.normal(0, 0.03, size=len(idx))
        load = 1000.0 * (0.55 + morning_peak + evening_peak) * weekend_factor * seasonal * (1 + noise)
        df_demo = pd.DataFrame({"Datetime": idx, "Grid_Load_MW": load})
        return df_demo, "DEMO (synthetic fallback)"


def cea_emission_factor_series(df: pd.DataFrame) -> pd.Series:
    """Apply annual CEA emission factors and apply diurnal solar shapes."""
    hour = df["Datetime"].dt.hour
    year = df["Datetime"].dt.year
    base_factors = year.map(ANNUAL_CEA_EF).fillna(0.7117).to_numpy()
    
    diurnal_swing = -0.18 * np.cos((hour - 13) / 24 * 2 * np.pi)
    factor = base_factors * (1.0 + diurnal_swing)
    return pd.Series(factor * 1000.0, index=df.index, name="Emission_Factor_kgCO2_per_MWh")


# --------------------------------------------------------------------------------------
# 3. POLICY-CALIBRATED: Carbon Credit Price (CCTS-anchored)
# --------------------------------------------------------------------------------------

def simulate_carbon_credit_price(n_hours: int, low: float = 250, high: float = 1500, seed: int = 42) -> np.ndarray:
    rng = np.random.default_rng(seed)
    mid = (low + high) / 2
    theta = 0.01
    sigma = 18.0
    price = np.empty(n_hours)
    price[0] = mid
    for t in range(1, n_hours):
        shock = rng.normal(0, sigma)
        price[t] = price[t - 1] + theta * (mid - price[t - 1]) + shock
        price[t] = np.clip(price[t], low, high)
    return price


# --------------------------------------------------------------------------------------
# 4. GROUNDED-SIMULATED: Green Zone soil moisture + sequestration
# --------------------------------------------------------------------------------------

def simulate_green_zone(weather: pd.DataFrame, hectares: float = 500.0,
                         biomass_density_low: float = 60.0, biomass_density_high: float = 140.0,
                         carbon_fraction: float = 0.47, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    n = len(weather)
    temp = weather["Ambient_Temp_C"].values
    precip = weather["Precip_mm"].values

    # Soil moisture calculation
    base_moisture = 45 - 0.6 * (temp - 28)
    rain_events = rng.poisson(lam=0.03, size=n)
    moisture = base_moisture + 4.0 * precip + rain_events * rng.uniform(8, 20, size=n)
    moisture = np.clip(moisture, 5, 95)

    # Sequestration calculations
    annual_biomass_increment_tC_ha = rng.uniform(biomass_density_low, biomass_density_high) * 0.01
    kgco2_per_ha_per_year = annual_biomass_increment_tC_ha * 1000 * (44 / 12)
    base_hourly_rate = (kgco2_per_ha_per_year * hectares) / 8760.0

    moisture_stress = np.clip(moisture / 60.0, 0.3, 1.2)
    temp_stress = np.clip(1.0 - np.abs(temp - 27) / 40.0, 0.4, 1.1)
    sequestration = base_hourly_rate * moisture_stress * temp_stress * carbon_fraction / 0.47

    out = weather.copy()
    out["Zone_Soil_Moisture_Pct"] = moisture
    out["Sequestration_Rate_kgCO2"] = sequestration
    return out


# --------------------------------------------------------------------------------------
# 5. MERGE + DERIVE TARGET
# --------------------------------------------------------------------------------------

def build_consolidated_ledger(weather: pd.DataFrame, grid_load: pd.DataFrame,
                               region_name: str, config: dict, load_scale_factor: float = 0.005) -> pd.DataFrame:
    df = weather.merge(grid_load, on="Datetime", how="inner")
    
    # Scale grid load
    df["Grid_Load_MW"] = df["Grid_Load_MW"] * load_scale_factor
    
    df["Emission_Factor_kgCO2_per_MWh"] = cea_emission_factor_series(df).values
    df["Carbon_Credit_Price_INR"] = simulate_carbon_credit_price(len(df))
    df = simulate_green_zone(
        df,
        hectares=500.0,
        biomass_density_low=config["biomass_low"],
        biomass_density_high=config["biomass_high"],
        carbon_fraction=config["carbon_fraction"]
    )

    df["Net_Carbon_Balance"] = (
        df["Grid_Load_MW"] * df["Emission_Factor_kgCO2_per_MWh"] - df["Sequestration_Rate_kgCO2"]
    )

    # Cyclical time features
    df["Hour_Sin"] = np.sin(2 * np.pi * df["Datetime"].dt.hour / 24)
    df["Hour_Cos"] = np.cos(2 * np.pi * df["Datetime"].dt.hour / 24)
    df["Month_Sin"] = np.sin(2 * np.pi * df["Datetime"].dt.month / 12)
    df["Month_Cos"] = np.cos(2 * np.pi * df["Datetime"].dt.month / 12)
    df["Is_Weekend"] = df["Datetime"].dt.dayofweek.isin([5, 6]).astype(int)
    
    # Add COVID lockdown anomaly feature
    # Lockdown 1: March-May 2020. Lockdown 2: April-June 2021
    year = df["Datetime"].dt.year
    month = df["Datetime"].dt.month
    is_lockdown_1 = (year == 2020) & (month.isin([3, 4, 5]))
    is_lockdown_2 = (year == 2021) & (month.isin([4, 5, 6]))
    df["Is_COVID_Lockdown"] = (is_lockdown_1 | is_lockdown_2).astype(int)
    
    # Add metadata
    df["Region"] = region_name

    return df


# --------------------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Build the multi-year, multi-region carbon ledger.")
    parser.add_argument("--start-year", type=int, default=2019)
    parser.add_argument("--end-year", type=int, default=2023)
    parser.add_argument("--grid-load-xlsx", type=str, default="data/raw/hourlyLoadDataIndia.xlsx")
    parser.add_argument("--load-scale-factor", type=float, default=0.005)
    parser.add_argument("--out", type=str, default="data/processed/multi_region_telemetry.csv")
    args = parser.parse_args()

    start_str = f"{args.start_year}0101"
    end_str = f"{args.end_year}1231"

    print(f"Loading raw demand file: {args.grid_load_xlsx} ...")
    try:
        df_all_loads = load_entire_grid_load(args.grid_load_xlsx)
        has_real_grid = True
    except Exception as e:
        print(f"      Could not load raw grid demand file ({e}). Will generate DEMO grid patterns.")
        df_all_loads = None
        has_real_grid = False

    ledgers = []
    weather_prov_dict = {}
    load_prov_dict = {}

    for region, config in REGIONS_CONFIG.items():
        print(f"\nProcessing region: {region} ...")
        
        # 1. Fetch weather (5-year range in one API call)
        print(f"   [1/3] Fetching NASA POWER weather data for {region} ({config['lat']}, {config['lon']}) {start_str}-{end_str} ...")
        weather, weather_provenance = get_weather(config["lat"], config["lon"], start_str, end_str)
        print(f"         -> {len(weather)} hourly weather rows [{weather_provenance}]")
        weather_prov_dict[region] = weather_provenance
        
        # 2. Get grid load
        print(f"   [2/3] Extracting grid load for region {region} ({config['load_col']}) ...")
        grid_load, load_provenance = get_grid_load(df_all_loads, config["load_col"], args.start_year, args.end_year)
        print(f"         -> {len(grid_load)} hourly load rows [{load_provenance}]")
        load_prov_dict[region] = load_provenance
        
        # 3. Merge and compute sequestration/pricing
        print("   [3/3] Merging, computing sequestration and carbon pricing...")
        ledger = build_consolidated_ledger(weather, grid_load, region, config, args.load_scale_factor)
        print(f"         -> Ledger shape: {ledger.shape[0]} rows x {ledger.shape[1]} columns")
        
        ledgers.append(ledger)
        
        # Sleep to prevent API rate-limiting
        time.sleep(2)

    # Concatenate all regional ledgers
    print("\n[4/4] Concatenating regional ledgers...")
    master_ledger = pd.concat(ledgers, ignore_index=True)
    print(f"      Total Master Ledger: {master_ledger.shape[0]} rows x {master_ledger.shape[1]} columns")
    
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    master_ledger.to_csv(args.out, index=False)
    print(f"Saved master telemetry to: {args.out}")

    # Write provenance
    manifest_path = os.path.join(os.path.dirname(args.out) or ".", "PROVENANCE.txt")
    with open(manifest_path, "w") as f:
        f.write("Multi-Region, Multi-Year Grid Provenance Log\n")
        f.write("============================================\n")
        f.write(f"Generated: {dt.datetime.now().isoformat(timespec='seconds')}\n")
        f.write(f"Year Range: {args.start_year} - {args.end_year}\n\n")
        
        f.write("Grid load columns mapping:\n")
        for region, config in REGIONS_CONFIG.items():
            f.write(f"   {region}: {config['load_col']} [{load_prov_dict[region]}]\n")
            
        f.write("\nWeather Coordinates & Provenance:\n")
        for region, config in REGIONS_CONFIG.items():
            f.write(f"   {region}: ({config['lat']}, {config['lon']}) [{weather_prov_dict[region]}]\n")
            
        f.write("\nCEA Grid Emission Factors: Annual Variable (0.81 tCO2/MWh in 2019 down to 0.7117 in 2023), diurnally adjusted.\n")
        f.write("Carbon prices: Ornstein-Uhlenbeck CCTS model (Rs 250 - 1500).\n")
        f.write("Forestry Sequestration: Region-specific biomass densities calibrated using Forest Survey of India constants.\n")
        
    print(f"Provenance log written -> {manifest_path}")


if __name__ == "__main__":
    main()

