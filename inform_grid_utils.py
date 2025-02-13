import pathlib as path
import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta
import xarray as xr
import matplotlib.pyplot as plt
from scipy.stats import norm
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation  


def grid_flight_dat(cesm: xr.open_dataset, cesm_dat: xr.open_dataset, df: pd.DataFrame, air: xr.open_dataset) -> dict:    
    """
    Grids aircraft data and scales it onto a 3D grid.

    This function maps aircraft data onto a predefined grid based on CESM data, 
    using the region defined by the bounds of the flight data.

    NOTE: A low-rate, 1 Hz, flight data file is assumed.

    Args:
        cesm (Any): Free-run CESM dataset.
        cesm_dat (Any): Nudged CESM dataset.
        df (pd.DataFrame): Aircraft dataset (includes LATC, LONC, U, V, T, etc.).
        air (Any): Aircraft dataset in xarray format.

    Returns:
        tuple[dict, np.ndarray, dict]: 
            - A dictionary containing time, latitude, longitude, and pressure altitude for grid cells with data.
            - A 3D numpy array representing the grid.
            - A dictionary containing latitude, longitude, and altitude bounds.
    """
    lat_var = next((var for var in df.columns if 'LAT' in var), None)
    lon_var = next((var for var in df.columns if 'LON' in var), None)
    alt_var = next((var for var in df.columns if 'ALT' in var or 'PSXC' in var), None)
    df_vars = [col for col in df.columns if col.lower() != 'time']
    if not lat_var or not lon_var or not alt_var:
        raise ValueError("Missing essential latitude, longitude, or altitude variables.")

    # Step 2: Create a 3D grid based on CESM & flight data
    def make_grid(cesm, air):
        lat_bounds = [int(np.abs(cesm.lat - np.min(air[lat_var])).argmin())-1, 
                      int(np.abs(cesm.lat - np.max(air[lat_var])).argmin())+1]
        lon_bounds = [int(np.abs(cesm.lon - np.min(air[lon_var])).argmin())-1, 
                      int(np.abs(cesm.lon - np.max(air[lon_var])).argmin())+1]
        alt_bounds = [int(np.abs(cesm.lev - np.min(air[alt_var])).argmin())-1, 
                      int(np.abs(cesm.lev - np.max(air[alt_var])).argmin())+1]

        bounds = {'lat': lat_bounds, 'lon': lon_bounds, 'palt': alt_bounds}
        grid_shape = (alt_bounds[1] - alt_bounds[0], lat_bounds[1] - lat_bounds[0], lon_bounds[1] - lon_bounds[0])
        return np.zeros(grid_shape), bounds

    grid, bounds = make_grid(cesm, air)

    # Step 3: Match aircraft times with CESM times
    da = xr.DataArray(cesm_dat.time, dims="time")
    cesm_times = pd.to_datetime([pd.Timestamp(dt.year, dt.month, dt.day, dt.hour, dt.minute, dt.second)
                        for dt in da.values])    
    aircraft_times = pd.to_datetime(df['Time'])
    # Select model output times that correspond to the flight times. 
    times = np.array(cesm_times[np.isin(cesm_times, aircraft_times)])
    
    # Step 4: Initialize grid arrays
    mean_lat, mean_lon, mean_alt = np.zeros_like(grid, dtype=float), np.zeros_like(grid, dtype=float), np.zeros_like(grid, dtype=float)
    mean_values = {var: np.zeros_like(grid, dtype=float) for var in df_vars}

    # Generate latitude, longitude, and altitude grid values
    lats = np.array(cesm.lat[bounds['lat'][0]:bounds['lat'][1]+1])
    lons = np.array(cesm.lon[bounds['lon'][0]:bounds['lon'][1]+1])
    alts = np.array(cesm.lev[bounds['palt'][0]:bounds['palt'][1]+1])
    alts = alts[::-1]
    # Add 30 minutes
    new_time = times[-1] + np.timedelta64(30, 'm')
    minus_time = times[0] - np.timedelta64(30,'m')
    # Append new value
    times = np.append(times, new_time)
    times = np.append(minus_time, times)
    mid_time = []
    for t in range(0,len(times)-1):
    
        time_start, time_end = times[t], times[t+1]
        # Select aircraft data between time intervals
        air_time_indices = (df['Time'] > times[t]) & (df['Time'] <= times[t+1]) 
        sliced_df_time = df[air_time_indices]
        if not sliced_df_time.empty:
                # Digitize lat, lon, and alt into grid bins
                lat_bins = np.digitize(sliced_df_time['GGLAT'], lats) - 1
                lon_bins = np.digitize(sliced_df_time['GGLON'], lons) - 1
                alt_bins = np.digitize(sliced_df_time['PSXC'], alts) - 1  # Reverse alt indexing
                # Ensure valid indices
                valid_mask = (lat_bins >= 0) & (lat_bins < len(lats) - 1) & \
                             (lon_bins >= 0) & (lon_bins < len(lons) - 1) & \
                             (alt_bins >= 0) & (alt_bins < len(alts) - 1)

                if valid_mask.any():
                    # Filter valid rows
                    sliced_df = sliced_df_time.loc[valid_mask].copy()
                    sliced_df['lat_bin'] = lat_bins[valid_mask]
                    sliced_df['lon_bin'] = lon_bins[valid_mask]
                    sliced_df['alt_bin'] = alt_bins[valid_mask]
    
                    # Group by grid cells and compute means                
                    grouped = sliced_df.groupby(['alt_bin', 'lat_bin', 'lon_bin'])
                    
                    for var in df_vars:
                        mean_values[var][tuple(zip(*grouped[var].mean().index.to_numpy()))] = grouped[var].mean().values
    
                    mean_lat[tuple(zip(*grouped[lat_var].mean().index.to_numpy()))] = grouped[lat_var].mean().values
                    mean_lon[tuple(zip(*grouped[lon_var].mean().index.to_numpy()))] = grouped[lon_var].mean().values
                    mean_alt[tuple(zip(*grouped[alt_var].mean().index.to_numpy()))] = grouped[alt_var].mean().values

                    # Compute mid-time for each grid cell
                    grouped_times = grouped['Time'].agg(lambda x: x.min() + (x.max() - x.min()) / 2)
                    mid_time.extend(grouped_times.values)
                    # Convert mid_time to a NumPy array
    
    # Conver time to numpy array
    mid_time = np.array(mid_time, dtype='datetime64[ns]')
    # There should always be temperature data, so use it to remove grids that do not have data
    mean_t = mean_values['ATX']
    valid_indices = np.argwhere(mean_t != 0)  # Gives (N, 3) array of (alt, lat, lon)
    selected_time = mid_time[:len(valid_indices)]
    # Identify grid cells with non-zero data
    alt_indices, lat_indices, lon_indices = valid_indices[:, 0], valid_indices[:, 1], valid_indices[:, 2]
    selected_time = mid_time[:len(valid_indices)]

    grid_dict = {
        'Time': selected_time,
        'Latitude': mean_lat[mean_t != 0],
        'Longitude': mean_lon[mean_t != 0],
        'Altitude': mean_alt[mean_t != 0],
    }
    grid_dict.update({var: mean_values[var][mean_t != 0] for var in df_vars})
 
        # Confirm the dictionary has data
    if all(len(v) > 0 for v in grid_dict.values()):
        print("✅ Grid dictionary successfully populated with data!")
    else:
        print("⚠️ Warning: Some entries in grid_dict are empty!")
           
    def plot_grid(free,bounds,air_nc):

        # Define the lat/lon boundaries of the square
        lat_min, lat_max = float(free.lat[bounds['lat'][0]-1]), float(free.lat[bounds['lat'][1]+2])  # Latitude range
        lon_min, lon_max = float(free.lon[bounds['lon'][0]-1]), float(free.lon[bounds['lon'][1]+2])  # Longitude range

        # Create a plot with Cartopy's PlateCarree projection
        fig, ax = plt.subplots(subplot_kw={'projection': ccrs.PlateCarree()}, figsize=(10, 6))

        # Add global map features
        ax.add_feature(cfeature.LAND, edgecolor='black', color='lightgray')
        ax.add_feature(cfeature.BORDERS, linestyle=':')
        ax.set_global()
        ax.coastlines()

        # Plot the flight track with proper transformation
        ax.plot(air_nc.LONC, air_nc.LATC, label='flight track', color='black', transform=ccrs.PlateCarree())

        # Plot the square of coordinates
        square_lats = [lat_min, lat_min, lat_max, lat_max, lat_min]
        square_lons = [lon_min, lon_max, lon_max, lon_min, lon_min]
        ax.plot(square_lons, square_lats, color='red', linewidth=2,label='model domain', transform=ccrs.PlateCarree())

        # Set the extent to zoom in on the region around the square
        buffer = 45  # Degrees to add around the square for some padding
        # Ensure that the longitude wrapping is handled correctly
        lon_min_zoom = max(lon_min - buffer, -180)
        lon_max_zoom = min(lon_max + buffer, 180)

        # Set the extent for zooming in on the region around the square
        ax.set_extent([lon_min_zoom, lon_max_zoom, lat_min - buffer, lat_max + buffer], crs=ccrs.PlateCarree())

        # Set title
        ax.set_title('Map with aircraft grid location')
        ax.legend()
        # Show plot
        plt.show()

    print("Here's the flight track and 2-D grid")
    plot_grid(cesm,bounds,air)

    return grid_dict, grid, bounds

def plot_3d_track(grid_data,air_nc):
    # grid_data = grid_flight_dat(cesm_fr, cesm_ndg, df, nc)
    # grid_data = grid_dict
    # Create a figure
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Scatter plot
    sc = ax.scatter(grid_data['GGLON'], grid_data['GGLAT'], grid_data['PSXC'], c=grid_data['ATX'], cmap='viridis', marker='^',label='grid-mean values',s=100)
    # Invert the Z-axis
    ax.scatter(air_nc.LONC, air_nc.LATC, air_nc.PSXC, c=air_nc.ATX, label='3D Flight track',s=12)
    ax.invert_zaxis()

    ax.set_xlabel('latitude (deg)')
    ax.set_ylabel('longitude (deg)')
    ax.set_zlabel('pressure alt (hPa)') 

    ax.legend()
    # # Color bar to show the mapping of color to the fourth dimension
    plt.colorbar(sc, label='Mean Temperature (°C)')

    # Animation function to rotate the view
    def rotate(angle):
        ax.view_init(elev=30, azim=angle)

    # Create animation
    ani = FuncAnimation(fig, rotate, frames=np.arange(-180, 360, 20), interval=100)

    # Show the animation in Jupyter Notebook
    from IPython.display import HTML
    HTML(ani.to_jshtml())

    plt.show()

def write_nc(grid_data, filename="test_grid_data.nc"):
    """
    Automatically creates and saves a NetCDF file from the given grid data dictionary.
    
    :param grid_data: Dictionary containing time series data with "Time" and corresponding variables.
    :param filename: Name of the NetCDF file to be saved.
    """

    # Extract headers dynamically (excluding "Time")
    headers = [key for key in grid_data.keys() if key.lower() != "time"]

    # Create the xarray dataset dynamically
    ds = xr.Dataset(
        {var: (["time"], grid_data[var]) for var in headers},  # Assign all variables dynamically
        coords={"time": grid_data["Time"]},  # Set "Time" as the coordinate
    )
    # # Save to a NetCDF file
    ds.to_netcdf("test_grid_data.nc")
    print("NetCDF file 'grid_data.nc' saved successfully!")