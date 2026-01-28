# import necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pdb
import xarray as xr

# load the netCDF file
file_path = '/Users/victor/Documents/Geo/Data/N21E039.SRTMGL1_NC.nc'

# open the dataset
dset = xr.open_dataset(file_path)
DEM = np.array(dset.variables['SRTMGL1_DEM'])

# close the dataset
dset.close()

# plot the DEM
plt.imshow(DEM)
cbar = plt.colorbar()
cbar.set_label('Elevation (m asl)')

# set plot title and labels
plt.savefig('assignment_1.png', dpi=300)
plt.show()