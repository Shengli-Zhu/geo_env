# import necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
import tools

# c to f conversion function
def c_to_f(c):
    return c * 9/5 + 32

def fmt_c_f(x):
    return f"{x:.2f} °C ({c_to_f(x):.1f} °F)"

# Part 1: Downloading and Importing Jeddah Weather Data
df_isd = tools.read_isd_csv('/Users/victor/Documents/Geo/Data/41024099999.csv')

plot = df_isd.plot(title="ISD data for Jeddah")
plt.show()

# Part 2: Heat Index (HI) Calculation
# 1. Relative Humidity
df_isd['RH'] = tools.dewpoint_to_rh(
    df_isd['DEW'].values,
    df_isd['TMP'].values
)

# 2. Heat Index (must use tools.gen_heat_index as required)
df_isd['HI'] = tools.gen_heat_index(
    df_isd['TMP'].values,
    df_isd['RH'].values
)

# 3. Highest HI
print(df_isd[['TMP','DEW','WND','RH','HI']].max())

# 4. Day/time of highest HI
print(df_isd[['TMP','DEW','WND','RH','HI']].idxmax())

# 5 & 6. Details at maximum HI
tmax = df_isd['HI'].idxmax()
row = df_isd.loc[tmax]

print("\nMaximum Heat Index information:")
print("UTC time:", tmax)
print("Local time (UTC+3):", tmax + pd.Timedelta(hours=3))

tmp_c = float(row['TMP'])
rh = float(row['RH'])
hi_c = float(row['HI'])

print(f"Temperature: {fmt_c_f(tmp_c)}")
print(f"Relative Humidity: {rh:.1f} %")
print(f"Heat Index: {fmt_c_f(hi_c)}")

# 9. Daily data (for discussion)
daily = df_isd.resample('D').mean(numeric_only=True)

# 10. Plot HI time series
plt.figure(figsize=(12,5))
plt.plot(df_isd.index, df_isd['HI'])
plt.title("Heat Index in Jeddah (2024)")
plt.ylabel("HI (°C)")
plt.xlabel("Date")
plt.grid(True)
plt.savefig("HI_2024.png", dpi=300, bbox_inches='tight')
plt.show()

# Part 3: Potential Impact of Climate Change
warming = 2.5238   # from Assignment 2

df_isd['TMP_future'] = df_isd['TMP'] + warming

df_isd['HI_future'] = tools.gen_heat_index(
    df_isd['TMP_future'].values,
    df_isd['RH'].values
)

hi_now_max = df_isd['HI'].max()
hi_future_max = df_isd['HI_future'].max()
increase = hi_future_max - hi_now_max

print("\nClimate change impact:")
print("Current max HI:", fmt_c_f(hi_now_max))
print("Future max HI:", fmt_c_f(hi_future_max))
print(f"Increase: {increase:.2f} °C ({increase*9/5:.1f} °F)")
