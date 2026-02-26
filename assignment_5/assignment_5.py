# Assignment 5: Analysis of the 2009 Jeddah Rainfall Event Using Geostationary Satellite Data
# import necessary libraries
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import glob

# Part 2: Data Visualization and Inspection
# 1. Open the dataset
dset = xr.open_dataset('/Users/victor/Documents/Geo/Data/GridSat_Data/GRIDSAT-B1.2009.11.25.06.v02r01.nc')

# 2. Extract the IR variable and squeeze to 2D
IR = np.array(dset.variables['irwin_cdr']).squeeze()

# 3. Flip the IR data vertically
IR = np.flipud(IR)

# 4. Scale IR values to Kelvin
IR = IR * 0.01 + 200

# 5. Convert IR from Kelvin to Celsius
IR_C = IR - 273.15

# 6. Plot the IR data
plt.figure(figsize=(10,6))
plt.imshow(IR_C, extent=[-180.035,180.035,-70.035,70.035], aspect='auto')
plt.colorbar(label="Brightness temperature (°C)")

# 7. Mark Jeddah on the plot
Jeddah_lon = 39.2
Jeddah_lat = 21.5
plt.scatter(Jeddah_lon, Jeddah_lat, color='red')
plt.title("Brightness Temperature 06 UTC")
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.savefig('brightness_temperature.png', dpi=300)
plt.show()

# 8. Find the lowest brightness temperature at Jeddah across all hours
files = sorted(glob.glob('/Users/victor/Documents/Geo/Data/GridSat_Data/GRIDSAT-B1.2009.11.25.*.v02r01.nc'))

temps = []
times = []

for f in files:
    d = xr.open_dataset(f)
    IR = np.array(d.variables['irwin_cdr']).squeeze()
    IR = np.flipud(IR)
    IR = IR * 0.01 + 200

    lat_index = int((70.035 - Jeddah_lat) / 0.07)
    lon_index = int((Jeddah_lon + 180.035) / 0.07)
    
    Tb = IR[lat_index, lon_index]
    temps.append(Tb)
    times.append(f.split('.')[-3])

temps = np.array(temps)

min_index = np.argmin(temps)

print("Lowest brightness temperature at Jeddah:")
print("Hour:", times[min_index], "UTC")
print("Temperature:", temps[min_index], "K")

# Part 3: Rainfall Estimation
# 1. Define the coefficients for the rainfall estimation formula
A = 1.1183e11
b = 3.6382e-2
c = 1.2

# 2. Calculate the rainfall estimation using the given formula
rain_total = 0

rain_jeddah = []
rain_time = []

for f in files:
    d = xr.open_dataset(f)
    IR = np.array(d.variables['irwin_cdr']).squeeze()
    IR = np.flipud(IR)
    IR = IR * 0.01 + 200   # Kelvin
    
    R = A * np.exp(-b * (IR ** c))  # rainfall rate
    
    hour = f.split('.')[-3]

    # umulative rainfall between 00:00 UTC and 12:00 UTC
    if hour in ["00", "03", "06", "09"]:
        rain_total += R * 3.0   # mm/h * 3h = mm

    # peak rainfall time at Jeddah
    rj = R[lat_index, lon_index]
    rain_jeddah.append(rj)
    rain_time.append(hour)

# Jeddah peak rainfall
rain_jeddah = np.array(rain_jeddah)
peak_i = np.argmax(rain_jeddah)

peak_hour_utc = int(rain_time[peak_i])
peak_hour_local = (peak_hour_utc + 3) % 24  # Jeddah is UTC+3

print("Peak rainfall at Jeddah:")
print("Hour:", f"{peak_hour_utc:02d}", "UTC")
print("Local time (UTC+3):", f"{peak_hour_local:02d}:00")
print("Rain rate (mm/h):", rain_jeddah[peak_i])

# 3. Plot the cumulative rainfall estimation
plt.figure(figsize=(10,6))
plt.imshow(rain_total, extent=[-180.035,180.035,-70.035,70.035], aspect='auto')
plt.colorbar(label='Cumulative rainfall (mm)')
plt.scatter(39.2,21.5,color='red')
plt.title('Cumulative rainfall 25 Nov 2009')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.savefig('cumulative_rainfall.png', dpi=300)
plt.show()

# Plot rainfall rate at Jeddah (00–12 UTC)
t_plot = [int(h) for h in rain_time]          # [0,3,6,9,12]
plt.figure(figsize=(6,4))
plt.plot(t_plot, rain_jeddah, marker='o')
plt.title("Rainfall rate at Jeddah between 00:00 and 12:00 UTC")
plt.xlabel("Time (UTC)")
plt.ylabel("Rainfall rate (mm/h)")
plt.grid(True)
plt.tight_layout()
plt.savefig("rainrate_Jeddah.png", dpi=300)
plt.show()