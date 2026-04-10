# =============================================================================
# Assignment 9: Climate Change Effects
# ErSE316 — Geo-Environmental Modeling & Analysis
# =============================================================================

# ── Section 0: Imports ────────────────────────────────────────────────────────
import os
import numpy as np
import xarray as xr
import geopandas as gpd
import rioxarray
import matplotlib.pyplot as plt
from shapely.geometry import mapping
from collections import namedtuple
from scipy.stats import norm

# ── Section 1: Constants & File Paths ────────────────────────────────────────
DATA_ROOT = "/Users/victor/Documents/Geo/Data/ISIMIP_Data"
SHAPEFILE = "/Users/victor/Documents/Geo/Data/Country_Borders/world-administrative-boundaries.shp"
OUT_DIR   = os.path.dirname(os.path.abspath(__file__))

PATHS = {
    "tas_370":  os.path.join(DATA_ROOT, "SSP370", "Temp_370", "Temp_370.nc"),
    "hurs_370": os.path.join(DATA_ROOT, "SSP370", "Humudity_370", "RH370.nc"),
    "pr_370":   os.path.join(DATA_ROOT, "SSP370", "Precipitation_370", "pr370.nc"),
    "tas_126":  os.path.join(DATA_ROOT, "SSP126", "Temp_126", "Temp126.nc"),
    "hurs_126": os.path.join(DATA_ROOT, "SSP126", "Humidity_126", "RH126.nc"),
    "pr_126":   os.path.join(DATA_ROOT, "SSP126", "Precipitation_126", "PR_126.nc"),
}

YEARS = np.arange(2015, 2101)  # 86 annual values


# ── Section 2: Helper Functions ───────────────────────────────────────────────

# ----------------------------------------------------------
# TREND ANALYSIS FUNCTIONS (Hamed & Rao 1998 + Sen's Slope)
# (From Appendix B of Assignment)
# ----------------------------------------------------------
def hamed_rao_mk_test(x, alpha=0.05):
    """Modified MK test with autocorrelation correction (Hamed & Rao 1998)"""
    n = len(x)
    s = 0
    for k in range(n - 1):
        for j in range(k + 1, n):
            s += np.sign(x[j] - x[k])

    # Calculate variance with autocorrelation correction
    var_s = n * (n - 1) * (2 * n + 5) / 18
    ties = np.unique(x, return_counts=True)[1]
    for t in ties:
        var_s -= t * (t - 1) * (2 * t + 5) / 18

    # Correct for autocorrelation
    n_eff = n
    if n > 10:
        acf = [1] + [np.corrcoef(x[:-i], x[i:])[0, 1] for i in range(1, n // 4)]
        n_eff = n / (1 + 2 * sum((n - i) / n * acf[i] for i in range(1, len(acf))))
        var_s *= n_eff / n

    if s > 0:
        z = (s - 1) / np.sqrt(var_s)
    elif s < 0:
        z = (s + 1) / np.sqrt(var_s)
    else:
        z = 0

    p = 2 * (1 - norm.cdf(abs(z)))
    h = abs(z) > norm.ppf(1 - alpha / 2)

    Trend = namedtuple('Trend', ['trend', 'h', 'p', 'z', 's'])
    trend = 'increasing' if s > 0 else 'decreasing' if s < 0 else 'no trend'
    return Trend(trend=trend, h=h, p=p, z=z, s=s)


def sens_slope(x, y):
    """Sen's slope estimator"""
    slopes = []
    for i in range(len(x)):
        for j in range(i + 1, len(x)):
            slopes.append((y[j] - y[i]) / (x[j] - x[i]))
    return np.median(slopes)


# ----------------------------------------------------------
# Wet Bulb Temperature (from Appendix C of Assignment)
# ----------------------------------------------------------
def calculate_wet_bulb_temperature(temp_k, rh_percent):
    """
    Calculate wet bulb temperature from air temperature and relative humidity.
    (Stull's method, 2011 - accurate to within 0.3°C)

    Args:
        temp_k: Temperature in Kelvin
        rh_percent: Relative humidity in percent
    Returns:
        Wet bulb temperature in Kelvin
    """
    # Convert temperature from Kelvin to Celsius for calculations
    temp_c = temp_k - 273.15

    # Calculation using Stull's method (2011)
    wbt_c = (temp_c * np.arctan(0.151977 * (rh_percent + 8.313659) ** 0.5)
             + np.arctan(temp_c + rh_percent)
             - np.arctan(rh_percent - 1.676331)
             + 0.00391838 * (rh_percent) ** (3 / 2) * np.arctan(0.023101 * rh_percent)
             - 4.686035)

    # Convert back to Kelvin
    wbt_k = wbt_c + 273.15
    return wbt_k


# ----------------------------------------------------------
# Spatial clipping helper
# ----------------------------------------------------------
def clip_to_saudi(da, sa_shape):
    """Clip a DataArray (with lat/lon dims) to Saudi Arabia polygon."""
    return (da
            .rio.set_spatial_dims(x_dim="lon", y_dim="lat")
            .rio.write_crs("EPSG:4326")
            .rio.clip(sa_shape.geometry.apply(mapping), sa_shape.crs, drop=True))


# ----------------------------------------------------------
# Plotting helper
# ----------------------------------------------------------
def plot_two_scenarios(years, y126, y370, ylabel, title, fname,
                       mk126=None, mk370=None, s126=None, s370=None):
    """Plot two-scenario time series with optional MK + Sen's slope annotation."""
    fig, ax = plt.subplots(figsize=(12, 5))

    ax.plot(years, y126, color="#1f77b4", linewidth=1.2, label="SSP1-RCP2.6")
    ax.plot(years, y370, color="#d62728", linewidth=1.2, label="SSP3-RCP7.0")

    # Add MK + Sen's slope annotation and trend lines if provided
    if mk126 is not None and s126 is not None:
        unit = ylabel.split("(")[-1].replace(")", "").strip() if "(" in ylabel else ""
        sig126 = "significant" if mk126.h else "not significant"
        sig370 = "significant" if mk370.h else "not significant"
        txt126 = (f"SSP1-RCP2.6: {mk126.trend} (p={mk126.p:.4f}, {sig126})\n"
                  f"  Sen's slope = {s126:.5f} {unit}/yr")
        txt370 = (f"SSP3-RCP7.0: {mk370.trend} (p={mk370.p:.4f}, {sig370})\n"
                  f"  Sen's slope = {s370:.5f} {unit}/yr")
        ax.text(0.02, 0.97, txt126, transform=ax.transAxes, fontsize=8,
                verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='#1f77b4', alpha=0.15))
        ax.text(0.02, 0.78, txt370, transform=ax.transAxes, fontsize=8,
                verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='#d62728', alpha=0.15))

        # Draw Sen's slope trend lines
        intercept126 = np.median(y126) - s126 * np.median(years)
        intercept370 = np.median(y370) - s370 * np.median(years)
        ax.plot(years, s126 * years + intercept126,
                color="#1f77b4", linestyle="--", linewidth=2, alpha=0.7,
                label=f"SSP1-RCP2.6 trend ({s126:.4f}/yr)")
        ax.plot(years, s370 * years + intercept370,
                color="#d62728", linestyle="--", linewidth=2, alpha=0.7,
                label=f"SSP3-RCP7.0 trend ({s370:.4f}/yr)")

    ax.set_xlabel("Year")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(loc="upper right")
    plt.tight_layout()
    out_path = os.path.join(OUT_DIR, fname)
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"  [Saved] {fname}")


# =============================================================================
# Part 1: Data Loading and Preparation
# =============================================================================
print("=" * 60)
print("Part 1: Data Loading and Preparation")
print("=" * 60)

# Load Saudi Arabia boundary
world = gpd.read_file(SHAPEFILE)
sa = world[world['name'] == 'Saudi Arabia'].copy()
print(f"  Saudi Arabia polygon loaded, CRS: {sa.crs}")

# Open all 6 combined NetCDF datasets
print("  Opening datasets...")
ds_tas_370  = xr.open_dataset(PATHS["tas_370"])
ds_hurs_370 = xr.open_dataset(PATHS["hurs_370"])
ds_pr_370   = xr.open_dataset(PATHS["pr_370"])
ds_tas_126  = xr.open_dataset(PATHS["tas_126"])
ds_hurs_126 = xr.open_dataset(PATHS["hurs_126"])
ds_pr_126   = xr.open_dataset(PATHS["pr_126"])

print(f"  tas_370 : dims={ds_tas_370['tas'].dims}, "
      f"time: {str(ds_tas_370.time.values[0])[:10]} to {str(ds_tas_370.time.values[-1])[:10]}")
print(f"  Sample tas_370[0,0,0] = {ds_tas_370['tas'].values[0,0,0]:.2f} K")
print(f"  Sample hurs_370[0,0,0] = {ds_hurs_370['hurs'].values[0,0,0]:.2f} %")
print(f"  Sample pr_370[0,0,0] = {ds_pr_370['pr'].values[0,0,0]:.6f} kg/m2/s")

# Clip all variables to Saudi Arabia
print("\n  Clipping all variables to Saudi Arabia boundary...")
tas_370  = clip_to_saudi(ds_tas_370['tas'],   sa)
hurs_370 = clip_to_saudi(ds_hurs_370['hurs'], sa)
pr_370   = clip_to_saudi(ds_pr_370['pr'],     sa)
tas_126  = clip_to_saudi(ds_tas_126['tas'],   sa)
hurs_126 = clip_to_saudi(ds_hurs_126['hurs'], sa)
pr_126   = clip_to_saudi(ds_pr_126['pr'],     sa)

print(f"  Clipped tas_370 shape: {tas_370.shape}")
print("  Part 1 complete.\n")


# =============================================================================
# Part 2: Climate Change Trend Analysis
# =============================================================================
print("=" * 60)
print("Part 2: Climate Change Trend Analysis (2015-2100)")
print("=" * 60)

# -- Average Annual Temperature (K -> °C) --
print("  Computing average annual temperature for Saudi Arabia...")
ann_temp_370 = ((tas_370 - 273.15)
                .groupby('time.year').mean('time')
                .mean(dim=['lat', 'lon']).values)
ann_temp_126 = ((tas_126 - 273.15)
                .groupby('time.year').mean('time')
                .mean(dim=['lat', 'lon']).values)

# -- Average Annual Precipitation (kg/m2/s -> mm/yr) --
print("  Computing average annual precipitation for Saudi Arabia...")
ann_pr_370 = ((pr_370 * 86400)                        # kg/m2/s -> mm/day
              .groupby('time.year').sum('time')        # sum daily -> mm/yr
              .mean(dim=['lat', 'lon']).values)
ann_pr_126 = ((pr_126 * 86400)
              .groupby('time.year').sum('time')
              .mean(dim=['lat', 'lon']).values)

# -- Mann-Kendall tests & Sen's Slope --
print("  Running adjusted Mann-Kendall test and Sen's Slope...")
mk_temp_370 = hamed_rao_mk_test(ann_temp_370)
mk_temp_126 = hamed_rao_mk_test(ann_temp_126)
mk_pr_370   = hamed_rao_mk_test(ann_pr_370)
mk_pr_126   = hamed_rao_mk_test(ann_pr_126)

ss_temp_370 = sens_slope(YEARS, ann_temp_370)
ss_temp_126 = sens_slope(YEARS, ann_temp_126)
ss_pr_370   = sens_slope(YEARS, ann_pr_370)
ss_pr_126   = sens_slope(YEARS, ann_pr_126)

print("\n  === Temperature Trend Results ===")
print(f"  SSP3-RCP7.0: {mk_temp_370.trend}, p={mk_temp_370.p:.4f}, "
      f"significant={mk_temp_370.h}, Sen's slope={ss_temp_370:.5f} °C/yr")
print(f"  SSP1-RCP2.6: {mk_temp_126.trend}, p={mk_temp_126.p:.4f}, "
      f"significant={mk_temp_126.h}, Sen's slope={ss_temp_126:.5f} °C/yr")

print("\n  === Precipitation Trend Results ===")
print(f"  SSP3-RCP7.0: {mk_pr_370.trend}, p={mk_pr_370.p:.4f}, "
      f"significant={mk_pr_370.h}, Sen's slope={ss_pr_370:.5f} mm/yr per yr")
print(f"  SSP1-RCP2.6: {mk_pr_126.trend}, p={mk_pr_126.p:.4f}, "
      f"significant={mk_pr_126.h}, Sen's slope={ss_pr_126:.5f} mm/yr per yr")

# -- Plots --
plot_two_scenarios(
    YEARS, ann_temp_126, ann_temp_370,
    ylabel="Temperature (°C)",
    title="Average Annual Temperature over Saudi Arabia (2015-2100)",
    fname="annual_temperature.png",
    mk126=mk_temp_126, mk370=mk_temp_370,
    s126=ss_temp_126, s370=ss_temp_370,
)

plot_two_scenarios(
    YEARS, ann_pr_126, ann_pr_370,
    ylabel="Precipitation (mm/yr)",
    title="Average Annual Precipitation over Saudi Arabia (2015-2100)",
    fname="annual_precipitation.png",
    mk126=mk_pr_126, mk370=mk_pr_370,
    s126=ss_pr_126, s370=ss_pr_370,
)
print("  Part 2 complete.\n")


# =============================================================================
# Part 3: Analysis of Climate Extremes
# =============================================================================
print("=" * 60)
print("Part 3: Analysis of Climate Extremes")
print("=" * 60)

# Spatial mean at daily steps → max per year
print("  Computing annual max of daily spatial-mean temperature...")
max_temp_370 = ((tas_370 - 273.15)
                .mean(dim=['lat', 'lon'])
                .groupby('time.year').max().values)
max_temp_126 = ((tas_126 - 273.15)
                .mean(dim=['lat', 'lon'])
                .groupby('time.year').max().values)

print("  Computing annual max of daily spatial-mean precipitation...")
max_pr_370 = ((pr_370 * 86400)
              .mean(dim=['lat', 'lon'])
              .groupby('time.year').max().values)
max_pr_126 = ((pr_126 * 86400)
              .mean(dim=['lat', 'lon'])
              .groupby('time.year').max().values)

plot_two_scenarios(
    YEARS, max_temp_126, max_temp_370,
    ylabel="Max Daily Temperature (°C)",
    title="Annual Maximum Daily Temperature (spatial mean) - Saudi Arabia (2015-2100)",
    fname="extreme_temperature.png",
)

plot_two_scenarios(
    YEARS, max_pr_126, max_pr_370,
    ylabel="Max Daily Precipitation (mm/day)",
    title="Annual Maximum Daily Precipitation (spatial mean) - Saudi Arabia (2015-2100)",
    fname="extreme_precipitation.png",
)
print("  Part 3 complete.\n")


# =============================================================================
# Part 4: Wet Bulb Temperature Calculation
# =============================================================================
print("=" * 60)
print("Part 4: Wet Bulb Temperature Calculation")
print("=" * 60)

# --- SSP370 ---
print("  Calculating Wet Bulb Temperature for SSP3-RCP7.0...")
wbt_370_k = calculate_wet_bulb_temperature(tas_370, hurs_370)  # returns Kelvin

# Save to NetCDF (following Appendix C pattern)
ds_wbt_370 = xr.Dataset(
    {'wet_bulb_temp': (['time', 'lat', 'lon'], wbt_370_k.values)},
    coords={
        'time': tas_370.time,
        'lat': tas_370.lat,
        'lon': tas_370.lon,
    },
    attrs={
        'description': 'Wet bulb temperature calculated from temperature and relative humidity',
        'units': 'K',
        'calculation_method': "Stull's method (2011)",
    }
)
wbt_370_path = os.path.join(OUT_DIR, "wb_370.nc")
ds_wbt_370.to_netcdf(wbt_370_path)
print(f"  [Saved] wb_370.nc")

# --- SSP126 ---
print("  Calculating Wet Bulb Temperature for SSP1-RCP2.6...")
wbt_126_k = calculate_wet_bulb_temperature(tas_126, hurs_126)

ds_wbt_126 = xr.Dataset(
    {'wet_bulb_temp': (['time', 'lat', 'lon'], wbt_126_k.values)},
    coords={
        'time': tas_126.time,
        'lat': tas_126.lat,
        'lon': tas_126.lon,
    },
    attrs={
        'description': 'Wet bulb temperature calculated from temperature and relative humidity',
        'units': 'K',
        'calculation_method': "Stull's method (2011)",
    }
)
wbt_126_path = os.path.join(OUT_DIR, "wb_126.nc")
ds_wbt_126.to_netcdf(wbt_126_path)
print(f"  [Saved] wb_126.nc")

# Convert WBT to Celsius for analysis
wbt_370_c = wbt_370_k - 273.15
wbt_126_c = wbt_126_k - 273.15

# Average annual WBT (°C) across Saudi Arabia
print("  Computing average annual Wet Bulb Temperature...")
ann_wbt_370 = (wbt_370_c
               .groupby('time.year').mean('time')
               .mean(dim=['lat', 'lon']).values)
ann_wbt_126 = (wbt_126_c
               .groupby('time.year').mean('time')
               .mean(dim=['lat', 'lon']).values)

plot_two_scenarios(
    YEARS, ann_wbt_126, ann_wbt_370,
    ylabel="Wet Bulb Temperature (°C)",
    title="Average Annual Wet Bulb Temperature over Saudi Arabia (2015-2100)",
    fname="annual_wbt.png",
)
print("  Part 4 complete.\n")


# =============================================================================
# Part 5: Wet Bulb Temperature Trend Analysis & Extremes
# =============================================================================
print("=" * 60)
print("Part 5: WBT Trend Analysis & Extremes")
print("=" * 60)

# -- MK test + Sen's Slope on annual mean WBT --
print("  Running adjusted Mann-Kendall test on annual WBT...")
mk_wbt_370 = hamed_rao_mk_test(ann_wbt_370)
mk_wbt_126 = hamed_rao_mk_test(ann_wbt_126)
ss_wbt_370 = sens_slope(YEARS, ann_wbt_370)
ss_wbt_126 = sens_slope(YEARS, ann_wbt_126)

print("\n  === Wet Bulb Temperature Trend Results ===")
print(f"  SSP3-RCP7.0: {mk_wbt_370.trend}, p={mk_wbt_370.p:.4f}, "
      f"significant={mk_wbt_370.h}, Sen's slope={ss_wbt_370:.5f} °C/yr")
print(f"  SSP1-RCP2.6: {mk_wbt_126.trend}, p={mk_wbt_126.p:.4f}, "
      f"significant={mk_wbt_126.h}, Sen's slope={ss_wbt_126:.5f} °C/yr")

plot_two_scenarios(
    YEARS, ann_wbt_126, ann_wbt_370,
    ylabel="Wet Bulb Temperature (°C)",
    title="Annual Mean Wet Bulb Temperature Trend - Saudi Arabia (2015-2100)",
    fname="annual_wbt_trend.png",
    mk126=mk_wbt_126, mk370=mk_wbt_370,
    s126=ss_wbt_126, s370=ss_wbt_370,
)

# -- Annual max WBT (spatial mean first, then annual max) --
print("  Computing annual maximum Wet Bulb Temperature...")
max_wbt_370 = (wbt_370_c
               .mean(dim=['lat', 'lon'])
               .groupby('time.year').max().values)
max_wbt_126 = (wbt_126_c
               .mean(dim=['lat', 'lon'])
               .groupby('time.year').max().values)

plot_two_scenarios(
    YEARS, max_wbt_126, max_wbt_370,
    ylabel="Max Wet Bulb Temperature (°C)",
    title="Annual Maximum Wet Bulb Temperature (spatial mean) - Saudi Arabia (2015-2100)",
    fname="annual_wbt_max.png",
)

print("  Part 5 complete.\n")
print("=" * 60)
print("All parts complete!")
print(f"Output files saved to: {OUT_DIR}")
print("=" * 60)

# Close datasets
ds_tas_370.close()
ds_hurs_370.close()
ds_pr_370.close()
ds_tas_126.close()
ds_hurs_126.close()
ds_pr_126.close()
