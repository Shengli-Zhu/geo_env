# import necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pdb
import xarray as xr

# load the netCDF file for historical climate data
file_path_historical = '/Users/victor/Documents/Geo/Data/Climate_Model_Data/tas_Amon_GFDL-ESM4_historical_r1i1p1f1_gr1_195001-201412.nc'
dset = xr.open_dataset(file_path_historical)

# Part 2 Exploring the Data
print(dset.keys()) 

# Explore the 'tas' variable
tas = dset['tas']

# Print information about the 'tas' variable
print(tas.dims)

# Print the shape of the 'tas' variable
print(tas.dtype)

# Print the time coordinate values
time = dset['time']
print(time)

# Print the units of the 'tas' variable
print(tas.units)


# Part 3: Creation of Climate Change Maps
#1. Calculate the mean air temperature map for 1850–1900
file_path_1850_1949 = '/Users/victor/Documents/Geo/Data/Climate_Model_Data/tas_Amon_GFDL-ESM4_historical_r1i1p1f1_gr1_185001-194912.nc'
dset = xr.open_dataset(file_path_1850_1949)
mean_1850_1900 = dset['tas'].sel(time=slice('18500101', '19001231')).mean(dim='time')
print(mean_1850_1900.dtype)
print(mean_1850_1900.shape)

# 2. & 3. Calculate future means and Plot differences

# Define file paths for future scenarios
base_path = '/Users/victor/Documents/Geo/Data/Climate_Model_Data/'
scenarios = {
    'SSP119': base_path + 'tas_Amon_GFDL-ESM4_ssp119_r1i1p1f1_gr1_201501-210012.nc',
    'SSP245': base_path + 'tas_Amon_GFDL-ESM4_ssp245_r1i1p1f1_gr1_201501-210012.nc',
    'SSP585': base_path + 'tas_Amon_GFDL-ESM4_ssp585_r1i1p1f1_gr1_201501-210012.nc'
}

# Loop through each scenario, calculate the mean for 2071-2100, and plot the differences
for name, file_path in scenarios.items():
    print(f"Processing {name}...")
    
    # Open the future scenario dataset
    dset_future = xr.open_dataset(file_path)
    
    # Calculate the mean air temperature for 2071-2100
    mean_future = dset_future['tas'].sel(time=slice('20710101', '21001231')).mean(dim='time')
    
    # Calculate the difference between future mean and historical mean
    diff = mean_future - mean_1850_1900
    
    # Plot the difference
    plt.figure(figsize=(10, 6))
    
    # Use pcolormesh to plot the difference
    diff.plot(cmap='RdBu_r', cbar_kwargs={'label': 'Temperature Difference (K)'})
    plt.xlabel('Lon')
    plt.ylabel('Lat')
    
    plt.title(f'Temperature Change: {name} (2071-2100) vs Historical')
    
    # Save the plot with a filename that includes the scenario name
    safe_name = name.replace('.', '').replace('-', '_')
    plt.savefig(f'temp_diff_{safe_name}.png', dpi=300, bbox_inches='tight')
    
    plt.show()
    plt.close()
