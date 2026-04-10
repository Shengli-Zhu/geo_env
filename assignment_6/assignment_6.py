# Assignment 6: Estimation of Open Water Evaporation in Wadi Murwani Reservoir Using ECMWF ERA5
# import necessary libraries
import xarray as xr
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tools

# Part 2: Data Pre-Processing
# 1. load
dset = xr.open_dataset('/Users/victor/Documents/Geo/Data/download.nc')

print("DATA VARS:", list(dset.data_vars))
print("COORDS:", list(dset.coords))

# 2. extract variables
t2m = np.array(dset.variables['t2m'])
tp  = np.array(dset.variables['tp'])
latitude = np.array(dset.variables['latitude'])
longitude = np.array(dset.variables['longitude'])
time_dt = np.array(dset.variables['valid_time'])

# 3. unit conversion
t2m = t2m - 273.15      # K -> °C
tp  = tp * 1000         # m -> mm (ERA5 tp is accumulated depth)

# 4. handle 4D case
if t2m.ndim == 4:
    t2m = np.nanmean(t2m, axis=1)
    tp  = np.nanmean(tp, axis=1)

# 5. squeeze to 1D (single point)
t2m = np.squeeze(t2m)
tp  = np.squeeze(tp)

# 6. build dataframe
df_era5 = pd.DataFrame(index=pd.to_datetime(time_dt))
df_era5['t2m'] = t2m
df_era5['tp']  = tp

# 7. plot time series
df_era5[['t2m','tp']].plot(subplots=True, figsize=(10,6), title='ERA5 at Wadi Murwani (hourly)')
plt.tight_layout()
plt.savefig('era5_time_series.png', dpi=300)
plt.show()

# 8. calculate mean annual precipitation
annual_precip = df_era5['tp'].resample('YE').mean() * 24 * 365.25
mean_annual_precip = np.nanmean(annual_precip)
print(mean_annual_precip)

# Part 3: Calculation of Potential Evaporation (PE)
# 1. prepare inputs for PE calculation
tmin = df_era5['t2m'].resample('D').min().values
tmax = df_era5['t2m'].resample('D').max().values
tmean = df_era5['t2m'].resample('D').mean().values
lat = 22.25
doy = df_era5['t2m'].resample('D').mean().index.dayofyear

# 2. calculate PE using Hargreaves–Samani method
pe = tools.hargreaves_samani_1982(tmin, tmax, tmean, lat, doy)

# 3. plot PE time series
ts_index = df_era5['t2m'].resample('D').mean().index
plt.figure(figsize=(10,4))
plt.plot(ts_index, pe, label='Potential Evaporation')
plt.ylabel('Potential evaporation (mm d$^{-1}$)')
plt.xlabel('Time')
plt.title('Daily Potential Evaporation (Hargreaves–Samani)')
plt.tight_layout()
plt.savefig('potential_evaporation.png', dpi=300)
plt.show()

# 4. calculate mean annual PE
pe_daily = pd.Series(pe, index=ts_index)
annual_pe = pe_daily.resample('YE').sum()   # mm/day summed to mm/yr
mean_annual_pe = np.nanmean(annual_pe)
print(mean_annual_pe)

# 5. estimate annual evaporation volume from the reservoir
area_m2 = 1.6e6
volume_m3_per_year = mean_annual_pe/1000 * area_m2
print(volume_m3_per_year)