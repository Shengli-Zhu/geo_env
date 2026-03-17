# Assignment 8: Calibrating and validating a linear reservoir rainfall-runoff model for a watershed in South-West KSA
# import necessary libraries
# Assignment 8

import os
import numpy as np
import xarray as xr
import geopandas as gpd
import scipy.optimize as opt
import matplotlib.pyplot as plt
from shapely.geometry import Point
from shapely.prepared import prep

# 1. Set paths
base_dir = "/Users/victor/Documents/Geo/Data2/ERA5_data"

precip_dir = os.path.join(base_dir, "Precipitation")
runoff_dir = os.path.join(base_dir, "Runoff")
evap_dir = os.path.join(base_dir, "Total_Evaporation")

# Modify this path if your watershed shapefile is stored somewhere else
shapefile_path = "/Users/victor/Documents/Geo/Data2/WS_3/WS_3.shp"

# 2. Read watershed shapefile
gdf = gpd.read_file(shapefile_path)
watershed_geom = prep(gdf.union_all())

# 3. Define a function to create a watershed mask
def make_mask(data_array, watershed_geom):
    lons = data_array["longitude"].values
    lats = data_array["latitude"].values

    mask = np.zeros((len(lats), len(lons)), dtype=bool)

    for i, lat in enumerate(lats):
        for j, lon in enumerate(lons):
            mask[i, j] = watershed_geom.contains(Point(float(lon), float(lat))) or watershed_geom.covers(Point(float(lon), float(lat)))

    return xr.DataArray(
        mask,
        coords={"latitude": data_array["latitude"], "longitude": data_array["longitude"]},
        dims=("latitude", "longitude")
    )

# 4. Define a function to load watershed-averaged ERA5 data
def load_basin_average(nc_file, var_name, sign_flip=False):
    ds = xr.open_dataset(nc_file)
    da = ds[var_name]

    mask = make_mask(da.isel(valid_time=0), watershed_geom)
    da_masked = da.where(mask)

    basin_mean = da_masked.mean(dim=["latitude", "longitude"], skipna=True).values

    if var_name in ["tp", "ro", "e"]:
        basin_mean = basin_mean * 1000

    if sign_flip:
        basin_mean = -basin_mean

    return basin_mean, da["valid_time"].values

# 5. Load 2001 and 2002 data for Part 1
precip_file_2001 = os.path.join(precip_dir, "era5_OLR_2001_total_precipitation.nc")
runoff_file_2001 = os.path.join(runoff_dir, "ambientera5_OLR_2001_total_runoff.nc")
evap_file_2001 = os.path.join(evap_dir, "era5_OLR_2001_total_evaporation.nc")

precip_file_2002 = os.path.join(precip_dir, "era5_OLR_2002_total_precipitation.nc")
runoff_file_2002 = os.path.join(runoff_dir, "ambientera5_OLR_2002_total_runoff.nc")
evap_file_2002 = os.path.join(evap_dir, "era5_OLR_2002_total_evaporation.nc")

P_2001, time_2001 = load_basin_average(precip_file_2001, "tp", sign_flip=False)
Q_obs_2001, _ = load_basin_average(runoff_file_2001, "ro", sign_flip=False)
ET_2001, _ = load_basin_average(evap_file_2001, "e", sign_flip=True)

P_2002, time_2002 = load_basin_average(precip_file_2002, "tp", sign_flip=False)
Q_obs_2002, _ = load_basin_average(runoff_file_2002, "ro", sign_flip=False)
ET_2002, _ = load_basin_average(evap_file_2002, "e", sign_flip=True)

print("2001 preprocessing completed")
print("Length of 2001 series:", len(P_2001))
print("P_2001 min/max:", np.min(P_2001), np.max(P_2001))
print("ET_2001 min/max:", np.min(ET_2001), np.max(ET_2001))
print("Q_obs_2001 min/max:", np.min(Q_obs_2001), np.max(Q_obs_2001))

print("\n2002 preprocessing completed")
print("Length of 2002 series:", len(P_2002))
print("P_2002 min/max:", np.min(P_2002), np.max(P_2002))
print("ET_2002 min/max:", np.min(ET_2002), np.max(ET_2002))
print("Q_obs_2002 min/max:", np.min(Q_obs_2002), np.max(Q_obs_2002))

# 6. Plot hydrological variables for 2001–2002 in one figure
P_all = np.concatenate([P_2001, P_2002])
ET_all = np.concatenate([ET_2001, ET_2002])
Q_obs_all = np.concatenate([Q_obs_2001, Q_obs_2002])
time_all = np.concatenate([time_2001, time_2002])

plt.figure(figsize=(14, 4))
plt.plot(time_all, P_all, label="Precipitation")
plt.plot(time_all, ET_all, label="Evaporation")
plt.plot(time_all, Q_obs_all, label="Runoff")
plt.legend()
plt.xlabel("Time")
plt.ylabel("mm")
plt.title("Hydrological Variables in 2001–2002")
plt.tight_layout()
plt.savefig("hydrological_variables_2001_2002.png", dpi=300)
plt.show()

# 7. Define the linear reservoir model
def simulate_runoff(k, P, ET, Q0, dt=1):
    n = len(P)
    Q_sim = np.zeros(n)
    Q_sim[0] = Q0

    for t in range(1, n):
        Q_t = (Q_sim[t-1] + (P[t] - ET[t]) * dt) / (1 + dt / k)
        Q_sim[t] = max(0, Q_t)

    return Q_sim

# 8. Define the Kling-Gupta Efficiency
def kge(Q_obs, Q_sim):
    r = np.corrcoef(Q_obs, Q_sim)[0, 1]
    alpha = np.std(Q_sim) / np.std(Q_obs)
    beta = np.mean(Q_sim) / np.mean(Q_obs)
    KGE = 1 - np.sqrt((r - 1)**2 + (alpha - 1)**2 + (beta - 1)**2)
    return KGE, r, alpha, beta

# 9. Part 2: validation with given k = 0.15
k_test = 0.15
Q_sim_2001_test = simulate_runoff(k_test, P_2001, ET_2001, Q_obs_2001[0])

KGE_2001_test, r_2001_test, alpha_2001_test, beta_2001_test = kge(Q_obs_2001, Q_sim_2001_test)

print("\nValidation results with given k for 2001")
print("k =", k_test)
print("KGE =", KGE_2001_test)
print("Correlation =", r_2001_test)
print("Alpha =", alpha_2001_test)
print("Beta =", beta_2001_test)

# 10. Plot Part 2 validation results
plt.figure(figsize=(12, 4))
plt.plot(time_2001, Q_obs_2001, label="Observed Runoff")
plt.plot(time_2001, Q_sim_2001_test, label="Simulated Runoff")
plt.legend()
plt.xlabel("Time")
plt.ylabel("Runoff (mm)")
plt.title("Observed vs Simulated Runoff in 2001 (k = 0.15)")
plt.tight_layout()
plt.savefig("validation_timeseries_2001_given_k.png", dpi=300)
plt.show()

plt.figure(figsize=(5, 5))
plt.scatter(Q_obs_2001, Q_sim_2001_test, s=8)
max_val = max(np.max(Q_obs_2001), np.max(Q_sim_2001_test))
plt.plot([0, max_val], [0, max_val], "r--")
plt.xlabel("Observed Runoff")
plt.ylabel("Simulated Runoff")
plt.title("Scatter Plot in 2001 (k = 0.15)")
plt.tight_layout()
plt.savefig("validation_scatter_2001_given_k.png", dpi=300)
plt.show()

# 11. Define objective function for calibration
def objective(k, P, ET, Q_obs):
    Q_sim = simulate_runoff(k, P, ET, Q_obs[0])
    KGE_val = kge(Q_obs, Q_sim)[0]
    return 1 - KGE_val

# 12. Calibrate k using 2001
res = opt.minimize_scalar(
    objective,
    bounds=(0.1, 2.0),
    args=(P_2001, ET_2001, Q_obs_2001),
    method="bounded"
)

best_k = res.x
Q_sim_2001_cal = simulate_runoff(best_k, P_2001, ET_2001, Q_obs_2001[0])
KGE_2001_cal, r_2001_cal, alpha_2001_cal, beta_2001_cal = kge(Q_obs_2001, Q_sim_2001_cal)

print("\nCalibration results for 2001")
print("Optimized k =", best_k)
print("KGE =", KGE_2001_cal)
print("Correlation =", r_2001_cal)
print("Alpha =", alpha_2001_cal)
print("Beta =", beta_2001_cal)

# 13. Plot 2001 calibration results
plt.figure(figsize=(12, 4))
plt.plot(time_2001, Q_obs_2001, label="Observed Runoff")
plt.plot(time_2001, Q_sim_2001_cal, label="Simulated Runoff")
plt.legend()
plt.xlabel("Time")
plt.ylabel("Runoff (mm)")
plt.title("Observed vs Simulated Runoff in 2001 (Calibration)")
plt.tight_layout()
plt.savefig("calibration_timeseries_2001.png", dpi=300)
plt.show()

plt.figure(figsize=(5, 5))
plt.scatter(Q_obs_2001, Q_sim_2001_cal, s=8)
max_val = max(np.max(Q_obs_2001), np.max(Q_sim_2001_cal))
plt.plot([0, max_val], [0, max_val], "r--")
plt.xlabel("Observed Runoff")
plt.ylabel("Simulated Runoff")
plt.title("Scatter Plot in 2001 (Calibration)")
plt.tight_layout()
plt.savefig("calibration_scatter_2001.png", dpi=300)
plt.show()

# 14. Load 2002 data
precip_file_2002 = os.path.join(precip_dir, "era5_OLR_2002_total_precipitation.nc")
runoff_file_2002 = os.path.join(runoff_dir, "ambientera5_OLR_2002_total_runoff.nc")
evap_file_2002 = os.path.join(evap_dir, "era5_OLR_2002_total_evaporation.nc")

P_2002, time_2002 = load_basin_average(precip_file_2002, "tp", sign_flip=False)
Q_obs_2002, _ = load_basin_average(runoff_file_2002, "ro", sign_flip=False)
ET_2002, _ = load_basin_average(evap_file_2002, "e", sign_flip=True)

print("\n2002 preprocessing completed")
print("Length of 2002 series:", len(P_2002))
print("P_2002 min/max:", np.min(P_2002), np.max(P_2002))
print("ET_2002 min/max:", np.min(ET_2002), np.max(ET_2002))
print("Q_obs_2002 min/max:", np.min(Q_obs_2002), np.max(Q_obs_2002))

# 15. Validate calibrated k on 2002
Q_sim_2002_val = simulate_runoff(best_k, P_2002, ET_2002, Q_obs_2002[0])
KGE_2002_val, r_2002_val, alpha_2002_val, beta_2002_val = kge(Q_obs_2002, Q_sim_2002_val)

print("\nValidation results for 2002")
print("k used =", best_k)
print("KGE =", KGE_2002_val)
print("Correlation =", r_2002_val)
print("Alpha =", alpha_2002_val)
print("Beta =", beta_2002_val)

# 16. Plot 2002 validation results
plt.figure(figsize=(12, 4))
plt.plot(time_2002, Q_obs_2002, label="Observed Runoff")
plt.plot(time_2002, Q_sim_2002_val, label="Simulated Runoff")
plt.legend()
plt.xlabel("Time")
plt.ylabel("Runoff (mm)")
plt.title("Observed vs Simulated Runoff in 2002 (Validation)")
plt.tight_layout()
plt.savefig("validation_timeseries_2002.png", dpi=300)
plt.show()

plt.figure(figsize=(5, 5))
plt.scatter(Q_obs_2002, Q_sim_2002_val, s=8)
max_val = max(np.max(Q_obs_2002), np.max(Q_sim_2002_val))
plt.plot([0, max_val], [0, max_val], "r--")
plt.xlabel("Observed Runoff")
plt.ylabel("Simulated Runoff")
plt.title("Scatter Plot in 2002 (Validation)")
plt.tight_layout()
plt.savefig("validation_scatter_2002.png", dpi=300)
plt.show()