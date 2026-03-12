# Assignment 7: Saudi Arabia Water Balance
# import necessary libraries
import numpy as np
import xarray as xr
import pandas as pd
import matplotlib.pyplot as plt
import geopandas as gpd
import rioxarray
from shapely.geometry import mapping
import os

# 1. Set file paths and read shapefile
base_dir = "/Users/victor/Documents/Geo/Data2/ERA5_data"
precip_dir = os.path.join(base_dir, "Precipitation")
runoff_dir = os.path.join(base_dir, "Runoff")
evap_dir = os.path.join(base_dir, "Total_Evaporation")

shp_fp = "/Users/victor/Documents/Geo/Data2/Saudi_Shape_File/Saudi_Shape.shp"
shapefile = gpd.read_file(shp_fp)

years = range(2000, 2021)

all_years_precip_monthly = []
all_years_runoff_monthly = []
all_years_evap_monthly = []

# 2. Loop through each year and calculate monthly regional means
for year in years:
    print("Processing year:", year)

    precip_fp = os.path.join(precip_dir, f"era5_OLR_{year}_total_precipitation.nc")
    runoff_fp = os.path.join(runoff_dir, f"ambientera5_OLR_{year}_total_runoff.nc")
    evap_fp = os.path.join(evap_dir, f"era5_OLR_{year}_total_evaporation.nc")

    ds_p = xr.open_dataset(precip_fp)
    ds_r = xr.open_dataset(runoff_fp)
    ds_e = xr.open_dataset(evap_fp)

    precip = ds_p["tp"]
    runoff = ds_r["ro"]
    evap = ds_e["e"]

    precip = precip.rio.set_spatial_dims(x_dim="longitude", y_dim="latitude")
    precip = precip.rio.write_crs("EPSG:4326")

    runoff = runoff.rio.set_spatial_dims(x_dim="longitude", y_dim="latitude")
    runoff = runoff.rio.write_crs("EPSG:4326")

    evap = evap.rio.set_spatial_dims(x_dim="longitude", y_dim="latitude")
    evap = evap.rio.write_crs("EPSG:4326")

    precip_clipped = precip.rio.clip(shapefile.geometry.apply(mapping), shapefile.crs, drop=True)
    runoff_clipped = runoff.rio.clip(shapefile.geometry.apply(mapping), shapefile.crs, drop=True)
    evap_clipped = evap.rio.clip(shapefile.geometry.apply(mapping), shapefile.crs, drop=True)

    precip_monthly = precip_clipped.groupby("valid_time.month").sum("valid_time")
    runoff_monthly = runoff_clipped.groupby("valid_time.month").sum("valid_time")
    evap_monthly = evap_clipped.groupby("valid_time.month").sum("valid_time")

    precip_monthly_mean = precip_monthly.mean(dim=("latitude", "longitude")) * 1000
    runoff_monthly_mean = runoff_monthly.mean(dim=("latitude", "longitude")) * 1000
    evap_monthly_mean = -evap_monthly.mean(dim=("latitude", "longitude")) * 1000

    all_years_precip_monthly.append(precip_monthly_mean.values)
    all_years_runoff_monthly.append(runoff_monthly_mean.values)
    all_years_evap_monthly.append(evap_monthly_mean.values)

# 3. Build monthly dataframe and calculate water balance
all_years_precip_monthly = np.array(all_years_precip_monthly)
all_years_runoff_monthly = np.array(all_years_runoff_monthly)
all_years_evap_monthly = np.array(all_years_evap_monthly)

precip_series = all_years_precip_monthly.flatten()
runoff_series = all_years_runoff_monthly.flatten()
evap_series = all_years_evap_monthly.flatten()

time_index = pd.date_range(start="2000-01-31", periods=len(precip_series), freq="ME")

df_monthly = pd.DataFrame({
    "P": precip_series,
    "R": runoff_series,
    "E": evap_series
}, index=time_index)

df_monthly["Balance"] = df_monthly["P"] - (df_monthly["E"] + df_monthly["R"])

# yearly sums
df_yearly = df_monthly.resample("YE").sum()

print("Mean annual precipitation (mm/yr):", df_yearly["P"].mean())
print("Mean annual runoff (mm/yr):", df_yearly["R"].mean())
print("Mean annual evaporation (mm/yr):", df_yearly["E"].mean())
print("Mean annual water balance (mm/yr):", df_yearly["Balance"].mean())


# 4. Plot monthly and yearly precipitation
plt.figure(figsize=(10, 4))
plt.plot(df_monthly.index, df_monthly["P"], label="Monthly Precipitation", linewidth=1, color="#1f77b4")
plt.plot(df_yearly.index, df_yearly["P"], label="Yearly Precipitation", linewidth=2, linestyle="--", marker="o", color="#0b3c5d")
plt.title("Precipitation in Saudi Arabia (2000-2020)")
plt.ylabel("Precipitation (mm)")
plt.xlabel("Time")
plt.legend()
plt.tight_layout()
plt.savefig("precipitation_timeseries.png", dpi=300)
plt.show()

# 5. Plot monthly and yearly evaporation
plt.figure(figsize=(10, 4))
plt.plot(df_monthly.index, df_monthly["E"], label="Monthly Evaporation", linewidth=1, color="#ff7f0e")
plt.plot(df_yearly.index, df_yearly["E"], label="Yearly Evaporation", linewidth=2, linestyle="--", marker="o",color="#c44e00")
plt.title("Evaporation in Saudi Arabia (2000-2020)")
plt.ylabel("Evaporation (mm)")
plt.xlabel("Time")
plt.legend()
plt.tight_layout()
plt.savefig("evaporation_timeseries.png", dpi=300)
plt.show()

# 6. Plot monthly and yearly runoff
plt.figure(figsize=(10, 4))
plt.plot(df_monthly.index, df_monthly["R"], label="Monthly Runoff", linewidth=1, color="#2ca02c")
plt.plot(df_yearly.index, df_yearly["R"], label="Yearly Runoff", linewidth=2, linestyle="--", marker="o",color="#1b7837")
plt.title("Runoff in Saudi Arabia (2000-2020)")
plt.ylabel("Runoff (mm)")
plt.xlabel("Time")
plt.legend()
plt.tight_layout()
plt.savefig("runoff_timeseries.png", dpi=300)
plt.show()

# 7. Plot monthly and yearly water balance
plt.figure(figsize=(10, 4))
plt.plot(df_monthly.index, df_monthly["Balance"], label="Monthly ΔS", linewidth=1, color="turquoise")
plt.plot(df_yearly.index, df_yearly["Balance"], label="Yearly ΔS", linewidth=2, linestyle="--", marker="o",color="dodgerblue")
plt.axhline(0, color="gray", linestyle="--")
plt.title("Water Balance P - (E + R) in Saudi Arabia (2000-2020)")
plt.ylabel("ΔS (mm)")
plt.xlabel("Time")
plt.legend()
plt.tight_layout()
plt.savefig("water_balance_timeseries.png", dpi=300)
plt.show()

# 8. Plot monthly components together
plt.figure(figsize=(10, 5))
plt.plot(df_monthly.index, df_monthly["P"], label="Precipitation", linewidth=1, color="#1f77b4")
plt.plot(df_monthly.index, df_monthly["E"], label="Evaporation", linewidth=1, color="#ff7f0e")
plt.plot(df_monthly.index, df_monthly["R"], label="Runoff", linewidth=1, color="#2ca02c")
plt.title("Monthly Water Balance Components in Saudi Arabia (2000-2020)")
plt.ylabel("mm")
plt.xlabel("Time")
plt.legend()
plt.tight_layout()
plt.savefig("monthly_components.png", dpi=300)
plt.show()

# 9. Plot yearly components together
plt.figure(figsize=(10, 5))
plt.plot(df_yearly.index, df_yearly["P"], label="Precipitation", linewidth=2, linestyle="--", marker="o", color="#0b3c5d")
plt.plot(df_yearly.index, df_yearly["E"], label="Evaporation", linewidth=2, linestyle="--", marker="o", color="#c44e00")
plt.plot(df_yearly.index, df_yearly["R"], label="Runoff", linewidth=2, linestyle="--", marker="o", color="#1b7837")
plt.title("Yearly Water Balance Components in Saudi Arabia (2000-2020)")
plt.ylabel("mm")
plt.xlabel("Time")
plt.legend()
plt.tight_layout()
plt.savefig("yearly_components.png", dpi=300)
plt.show()