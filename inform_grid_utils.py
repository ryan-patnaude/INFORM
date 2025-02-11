import netCDF4
import pathlib as path
import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta
from fnmatch import fnmatch
from typing import Iterable
import xarray as xr
import matplotlib.pyplot as plt
from scipy.stats import norm

def grid_flight_dat(cesm: xr.open_dataset, cesm_dat: xr.open_dataset, df: pd.DataFrame, air: xr.open_dataset) -> dict:    
    """
    Grids aircraft data and scales it onto a 3D grid.
    
    This function maps aircraft data onto a predefined grid based on CESM data,
    using the region defined by the bounds of the flight data.
    
    NOTE: A low-rate, 1 Hz, flight data file is assumed.
    
    :param gv_dat: A dataset object containing aircraft data (e.g., LATC, LONC, U, V, T).
    :param cesm_dat: A dataset object containing CESM data (e.g., lat, lon, lev).
    :param grid: A 3D numpy array representing the grid, typically created based on flight region.
    :param bounds: A dictionary containing the latitude, longitude, and altitude bounds for the grid.
    
    :return: A dictionary containing latitude (lats), longitude (lons), and pressure altitude (palts)
             for grid cells where data is present.
    """

    def make_grid(cesm: xr.open_dataset, air: xr.open_dataset) -> tuple[np.ndarray, dict[str, list[int]]]:
        """
        Create a grid restricted to the aircraft flight region based on CESM and flight data.
    
        This function reads latitude, longitude, and optionally altitude bounds from aircraft flight data,
        and creates a 3D grid over the specified region. The function assumes low-rate (1 Hz) flight 
        data and standard CESM grid data.
    
        :param cesm_dat: A dataset object containing CESM data with attributes like lat, lon, and lev.
        :param gv_dat: A dataset object containing aircraft data with attributes like LATC and LONC.
    
        :return: 
            - grid: A numpy array representing the grid restricted to the aircraft flight region.
            - bounds: A dictionary containing the calculated latitude and longitude bounds.
        """
        required_cesm_attributes = ['lat', 'lon', 'lev']
        required_gv_attributes = ['LATC', 'LONC', 'PSXC']
        
        for attr in required_cesm_attributes:
            if not hasattr(cesm, attr):
                raise ValueError(f"CESM data is missing required attribute: {attr}")

        for attr in required_gv_attributes:
            if not hasattr(air, attr):
                raise ValueError(f"Flight data is missing required attribute: {attr}")

        try:
            lat_upr_bnd = int(np.abs(cesm.lat - np.max(air.LATC)).argmin())
            lat_lwr_bnd = int(np.abs(cesm.lat - np.min(air.LATC)).argmin())
            lon_upr_bnd = int(np.abs(cesm.lon - np.max(air.LONC)).argmin())
            lon_lwr_bnd = int(np.abs(cesm.lon - np.min(air.LONC)).argmin())
            alt_upr_bnd = int(np.abs(cesm.lev - np.min(air.PSXC)).argmin())
            alt_lwr_bnd = int(np.abs(cesm.lev - np.max(air.PSXC)).argmin())
        except Exception as e:
            raise ValueError(f"Error calculating bounds: {e}")

        bounds = {
            'lat': [lat_lwr_bnd-1, lat_upr_bnd+1],
            'lon': [lon_lwr_bnd-1, lon_upr_bnd+1],
            'palt': [alt_upr_bnd-1, alt_lwr_bnd+1]
        }

        grid_shape = (
            (alt_lwr_bnd+1 - alt_upr_bnd+1), 
            (lat_upr_bnd+1 - lat_lwr_bnd+1), 
            (lon_upr_bnd+1 - lon_lwr_bnd+1)
        )
        grid = np.zeros(grid_shape, dtype=int)
        return grid, bounds
    
    grid, bounds = make_grid(cesm,nc)

    def filt_model_dims(cesm_dat: xr.open_dataset,df: pd.DataFrame):
        """
        Returns the lat, lon, and time from the nudged file that occur during specific research flight
        
        This function maps aircraft data onto a predefined grid based on CESM data
        
        NOTE: A low-rate, 1 Hz, flight data file is assumed.
        
        :param gv_dat: A dataset object containing aircraft data (e.g., LATC, LONC, U, V, T).
        :param cesm_dat: A dataset object containing CESM data (e.g., lat, lon, lev).
        
        :return: A dictionary containing latitude (lats), longitude (lons), and time 
                 for grid cells where data is present.
        """
        da = xr.DataArray(cesm_dat.time, dims="time")
        cesm_times = pd.to_datetime([pd.Timestamp(dt.year, dt.month, dt.day, dt.hour, dt.minute, dt.second)
                            for dt in da.values])
        
        aircraft_times = df.Time
        
        matching_indices = cesm_times.isin(aircraft_times)
        # Assuming cesm_dat.lat and matching_indices are already defined
        # Create a boolean mask for lat values you want to exclude
        lat = np.array(cesm_dat.lat[matching_indices])  # This gives the full lat array
        lon = np.array(cesm_dat.lon[matching_indices])  # This gives the full lat array
        time = np.array(cesm_times[matching_indices])  # This gives the full lat array
        
        # Define a threshold for what is considered a "high" difference
        lat_threshold = 1  # You can adjust this value
        lon_threshold = 1
        
        # Find the indices where the difference exceeds the threshold
        outlier_indices_lat = np.where(np.abs(np.diff(lat)) > lat_threshold)[0]
        outlier_indices_lon = np.where(np.abs(np.diff(lon)) > lon_threshold)[0]
        
        outliers = np.array([])
        
        for i in range(0,len(outlier_indices_lat)-1):
            if lat[outlier_indices_lat[i+1]] < lat[outlier_indices_lat[i]]:
                outliers = np.append(outliers, int(outlier_indices_lat[i+1]))
            elif lat[outlier_indices_lat[i]] < lat[outlier_indices_lat[i+1]]:
                outliers = np.append(outliers, int(outlier_indices_lat[i]))
                
        for i in range(0,len(outlier_indices_lon)-1):
            if lon[outlier_indices_lon[i+1]] < lon[outlier_indices_lon[i]]:
                outliers = np.append(outliers, int(outlier_indices_lon[i+1]))
            elif lon[outlier_indices_lon[i+1]] > lon[outlier_indices_lon[i]]:
                outliers = np.append(outliers, int(outlier_indices_lon[i]))
        # print(outliers)      
        # Create the exclude_mask, same length as lat, initially filled with False
        exclude_mask = np.zeros_like(lat, dtype=bool)  # Shape matches lat
        
        # # Set the outlier indices to True in the exclude_mask
        exclude_mask[outliers.astype(int)] = True  # Mark outliers as True, which will exclude them
        
        final_mask = ~exclude_mask
        
        # check if there are any remaining outliers, usually at the beginning of the array where there's two consecutive outliers
        lat_check = lat[final_mask]
        outlier_indices_lat = np.where(np.abs(np.diff(lat_check)) > lat_threshold)[0]
        if len(outlier_indices_lat) > 0:
            outliers = np.concatenate((outlier_indices_lat, outliers))
            # # Set the outlier indices to True in the exclude_mask
            exclude_mask[outliers.astype(int)] = True  # Mark outliers as True, which will exclude them
        
            final_mask = ~exclude_mask
        
        filter_cesm_dims = {'lat': lat[final_mask],
                            'lon': lon[final_mask],
                            'time': time[final_mask]
                           }
        
        return filter_cesm_dims

    filter_cesm_dims = filt_model_dims(cesm_dat, df)

    mean_u = np.zeros_like(grid, dtype=float)
    mean_v = np.zeros_like(grid, dtype=float)
    mean_t = np.zeros_like(grid, dtype=float)
    mean_lat = np.zeros_like(grid, dtype=float)
    mean_lon = np.zeros_like(grid, dtype=float)
    mean_alt = np.zeros_like(grid, dtype=float)
        
    # Generate latitude, longitude, and altitude grid values
    lats = np.array(cesm.lat[bounds['lat'][0]:bounds['lat'][1]+1])
    lons = np.array(cesm.lon[bounds['lon'][0]:bounds['lon'][1]+1])
    alts = np.array(cesm.lev[bounds['palt'][0]:bounds['palt'][1]+1])
    alts = alts[::-1]
    times = filter_cesm_dims['time']
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
                lat_bins = np.digitize(sliced_df_time['LATC'], lats) - 1
                lon_bins = np.digitize(sliced_df_time['LONC'], lons) - 1
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

                    # print(grouped)
                    # print(grouped)
                    mean_u_vals = grouped['UIC'].mean()
                    mean_v_vals = grouped['VIC'].mean()
                    mean_t_vals = grouped['ATX'].mean()
                    mean_lat_vals = grouped['LATC'].mean()
                    mean_lon_vals = grouped['LONC'].mean()
                    mean_alt_vals = grouped['PSXC'].mean()
    
                    # Assign values to the grid
                    indices = tuple(zip(*mean_u_vals.index.to_numpy()))  # Unpack MultiIndex correctly
                    mean_u[indices] = mean_u_vals.values
                    mean_v[indices] = mean_v_vals.values
                    mean_t[indices] = mean_t_vals.values
                    mean_lat[indices] = mean_lat_vals.values
                    mean_lon[indices] = mean_lon_vals.values
                    mean_alt[indices] = mean_alt_vals.values
    
                    # Compute mid-time for each grid cell
                    grouped_times = grouped['Time'].agg(lambda x: x.min() + (x.max() - x.min()) / 2)
                    mid_time.extend(grouped_times.values)
    
        indices = np.argwhere(mean_t != 0)
        
        # Identify grid cells with non-zero data
        valid_indices = np.argwhere(mean_t != 0)
        alt_indices, lat_indices, lon_indices = valid_indices[:, 0], valid_indices[:, 1], valid_indices[:, 2]
    
        # # Calculate the center values of latitude, longitude, and altitude for each grid cell
        # selected_lats = (lats[lat_indices] + lats[lat_indices + 1]) / 2
        # selected_lons = (lons[lon_indices] + lons[lon_indices + 1]) / 2
        # selected_alts = (alts[alt_indices] + alts[alt_indices + 1]) / 2
        
        grid_dict = {'Time': mid_time,
                     'lats': mean_lat[mean_t !=0],
                     'lons': mean_lon[mean_t !=0],
                     'palts': mean_alt[mean_t !=0],
                     'mean_u': mean_u[mean_t !=0],
                     'mean_v': mean_v[mean_t !=0],
                     'mean_t': mean_t[mean_t !=0],
                    }
        
    return grid_dict, grid, bounds