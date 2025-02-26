import netCDF4
import pathlib as path
import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta
from fnmatch import fnmatch
from typing import Iterable
import xarray as xr

def find_flight_fnames(dir_path: str) -> list[str]:
    """
    find_flight_fnames just searches a directory for all *.nc files and returns a list of them.

    :param dir_path: a path to the directory containing flight netcdf files

    :return: Returns a list of flight netcdf files.
    """
    flight_paths=[]
    flight_fnames = sorted([fname for fname in os.listdir(dir_path) if fnmatch(fname, "*.nc")])
    for i in range(len(flight_fnames)):
        flight_paths.append(dir_path + '/' + flight_fnames[i])
    
    return flight_paths

def find_nc_fnames(dir_path: str) -> list[str]:
    """
    find_flight_fnames just searches a directory for all *.nc files and returns a list of them.
    
    :param dir_path: a path to the directory containing flight netcdf files
    
    :return: Returns a list of flight netcdf files.
    """
    nc_paths=[]
    nc_fnames = sorted([fname for fname in os.listdir(dir_path) if fnmatch(fname, "*.nc")])
    for i in range(len(nc_fnames)):
        nc_paths.append(dir_path + '/' + nc_fnames[i])
        
        nudg_path = [file for file in nc_paths if ".hs." in file]
        free_path = [file for file in nc_paths if ".h0." in file]
        # save dictionary with the paths for 
        paths = {'Free': free_path,'Nudg': nudg_path}
        
    return paths

def open_nc(flight_paths: str) -> netCDF4._netCDF4.Dataset:
    """
    open_flight_nc simply checks to see if the file at the provided path string exists and opens it.

    :param file_path: A path string to a flight data file, e.g. "./test/test_flight.nc"

    :return: Returns xr.open_dataset object.
    """
    fp_path = path.Path(flight_paths)
    if not fp_path.is_file():
        raise FileNotFoundError('testing excptions')

    return xr.open_dataset(flight_paths)

def read_flight_nc_1hz(nc: xr.open_dataset, read_vars) -> pd.DataFrame:
    """
    read_flight_nc reads a set of variables into memory.

    NOTE: a low-rate, 1 Hz, flight data file is assumed

    :param nc: netCDF4._netCDF4.Dataset object opened by open_flight_nc.
    :param read_vars: An list of strings of variable names to be read into memory.

    :return: Returns a pandas data frame.
    """
    long_names = [nc[var].long_name if 'long_name' in nc[var].attrs else None for var in read_vars]
    data = [] # an empty list to accumulate Dataframes of each variable to be read in
    for var in read_vars:
        try:
            if var == "Time":
                # time is provided every second, so need to calculate 25 Hz times efficiently
                # tunits = getattr(nc[var],'units')
                # df = xr.open_dataset(nc)
                time = np.array(nc.Time)
                data.append(pd.DataFrame({var: time}))
                # dt_list = sfm_to_datetime(time, tunits)
                # data.append(pd.DataFrame({'datetime': time}))
            else:
                output = nc[var][:]
                data.append(pd.DataFrame({var: output}))
        except Exception as e:
            print(f"Issue reading {var}: {e}")
            pass
    
    dataframe = pd.concat(data, axis=1, ignore_index=False)
    dataframe.attrs['long_names'] = long_names
    # concatenate the list of dataframes into a single dataframe and return it
    return dataframe

def read_flight_nc_25hz(nc: xr.open_dataset, read_vars) -> pd.DataFrame:
    """
    read_flight_nc reads a set of variables into memory.

    NOTE: a high-rate, usually 25 Hz, flight data file is assumed.

    :param nc: netCDF4._netCDF4.Dataset object opened by open_flight_nc.
    :param read_vars: An optional list of strings of variable names to be read into memory. A default
                      list, vars_to_read, is specified above. Passing in a similar list will read in those variables
                      instead.

    :return: Returns a pandas data frame.
    """

    data = [] # an empty list to accumulate Dataframes of each variable to be read in

    hz = 25
    sub_seconds = np.arange(0,25,1)/25.

# NEED TO EDIT THIS FOR XARRAY

    # for var in read_vars:
    #     try:
    #         if var == "Time":
    #             # time is provided every second, so need to calculate 25 Hz times efficiently
    #             tunits = getattr(nc[var],'units')
    #             time = nc[var][:]

    #             time_25hz = np.zeros((len(time),hz)) # 2-D
    #             for i,inc in enumerate(sub_seconds):
    #                 time_25hz[:,i] = time + inc
    #             output = np.ravel(time_25hz) # ravel to 1-D
    #             data.append(pd.DataFrame({var: output}))
    #             dt_list = sfm_to_datetime(output, tunits)
    #             data.append(pd.DataFrame({'datetime': dt_list}))
    #         else:
    #             ndims = len(np.shape(nc[var][:]))
    #             if ndims == 2:
    #                 # 2-D, 25 Hz variables can just be raveled into 1-D time series
    #                 output = np.ravel(nc[var][:])
    #                 data.append(pd.DataFrame({var: output}))
    #             elif ndims == 1:
    #                 # 1-D variables in 25 Hz data files exist (e.g. GGALT is sampled at 20 Hz, but by default,
    #                 # this is filtered to 1Hz instead of fudged to 25 Hz). Do interpolation to 25 Hz so all time series
    #                 # have same length.
    #                 output_1d = nc[var][:]
    #                 output_2d = np.zeros((len(output_1d),hz))*float("NaN")
    #                 for i in range(len(output_1d)-1):
    #                     output_2d[i,:] = output_1d[i] + sub_seconds*(output_1d[i+1]-output_1d[i]) # divide by 1s omitted
    #                 output = np.ravel(output_2d)
    #                 data.append(pd.DataFrame({var: output}))
    #             else:
    #                 raise RuntimeError(f"Variable {var} is {ndims}-dimensional. Only 1-D or 2-D variables are handled.")
    #     except Exception as e:
    #         #print(f"Issue reading {var}: {e}")
    #         pass
              

    # concatenate the list of dataframes into a single dataframe and return it
    return pd.concat(data, axis=1, ignore_index=False)

# class flight_obj:
#     """
#     flight_obj's are classes that hold flight data (i.e. variables indicated by read_vars) from a provided file path string.
#     The __init__ takes a file path string and a list of vars to read (vars_to_read by default).
#     The __init__ assigns:
#     self.file_path: str; the file path passed in
#     self.read_vars_attempted: list[str]; the originally passed in list of vars to read
#     self.nc: netCDF4._netCDF4.Dataset; the opened netcdf object
#     self.df: pd.DataFrame; a dataframe holding the read in data
#     self.rate: str; a string indicating the rate of the data read in
#     self.read_vars: list[str]; list of the vars that were successfully read in
#     """
#     def __init__(self, file_path: str, read_vars):
#         # assign input vars
#         self.file_path = path.Path(file_path)
#         self.read_vars_attempted = read_vars

#         # open netcdf file if the file exists, assign to self.nc
#         if self.file_path.is_file():
#             self.nc = netCDF4.Dataset(self.file_path)
#         else:
#             raise FileNotFoundError(f"File {self.file_path} did not exist!")

#         # read in the variables, assign DataFrame to self.df,
#         #                               rate to self.rate,
#         #                               vars read in to self.read_vars
#         dim_names = list(self.nc.dimensions.keys())
#         if 'sps25' in dim_names:
#             self.df = read_flight_nc_25hz(self.nc, self.read_vars_attempted)
#             self.rate = "25Hz"
#         else:
#             self.df = read_flight_nc_1hz(self.nc, self.read_vars_attempted)
#             self.rate = "1Hz"
#         self.read_vars = list(self.df.keys())


# Function to read in all the relevant variables from the NSF aircraft datasets
def read_vars(nc):

    var_list = nc.data_vars
    time = 'Time'
    # Spatial variables
    lat, lon, alt = 'GGLAT', 'GGLON', 'GGALT'
    
    # state variables
    temp = 'ATX'
    dwpt = 'DPXC'
    u = 'UIC' if 'UIC' in var_list else 'UIX'
    v = 'VIC' if 'VIC' in var_list else 'VIX'
    w = 'WIC' if 'WIC' in var_list else 'WIX'
    p = 'PSXC'
    ew = 'EWX'
    rh = 'RHUM'
    
    vars_to_read = [time, lat, lon, alt, temp, dwpt, u,  w, p, ew, rh]
    # Thermodynamic data
    if any('THETA' in var for var in var_list): 
        theta_vars = [var for var in var_list if 'THETA' in var]
        vars_to_read.extend(theta_vars)
    # Cloud microphysical
    if any('CONC' in var for var in var_list): # cloud concentrations
        conc_vars = [var for var in var_list if 'CONC' in var and 'D' in var and('_2' not in var and 'R_' not in var and 'CN' not in var)]
        print(conc_vars)
        vars_to_read.extend(conc_vars)
    if any('PLW' in var for var in var_list): # Liquid/Ice water contents
        # v = [var for var in var_list if '2' not in var]
        wc_vars = [var for var in var_list if 'PLW' in var and '_2' not in var]
        vars_to_read.extend(wc_vars)
    # Aerosol data
    if any('UHSAS' in var for var in var_list) or any('CONCN' in var for var in var_list):
        aer_var = [var for var in var_list if ('UHSAS' in var or 'CONCU' in var or 'CONCN' in var) and ('AU' not in var and 'UD' not in var and 'CUH' not in var)]
        uhsas_cells = var_list['CUHSAS_LWII'].CellSizes
        vars_to_read.extend(aer_var)
    
    print('Variables included:')
    print(list(vars_to_read))
    return vars_to_read

def find_sondes(dir_path: str) -> list[str]:
    """
    find_flight_fnames just searches a directory for all *.nc files and returns a list of them.

    :param dir_path: a path to the directory containing flight netcdf files

    :return: Returns a list of flight netcdf files.
    """
    flight_paths=[]
    flight_fnames = sorted([fname for fname in os.listdir(dir_path) if fnmatch(fname, "*.cls")])
    for i in range(len(flight_fnames)):
        flight_paths.append(dir_path + '/' + flight_fnames[i])
    
    return flight_paths

def read_sonde2df(file_path):
    """
    Reads a `.cls` radiosonde file and extracts multiple datasets with their nominal release times.

    :param file_path: Path to the `.cls` file containing radiosonde data.

    :return: A tuple containing:
        - A list of Pandas DataFrames, each representing an individual radiosonde dataset.
        - A list of corresponding nominal release times as Pandas Timestamps.
    
    :raises FileNotFoundError: If the file does not exist.
    :raises ValueError: If the file is incorrectly formatted or missing essential data.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File '{file_path}' not found.")

    # Read the entire file into memory
    with open(file_path, "r", encoding="utf-8") as file:
        lines = file.readlines()

    # Identify all "Nominal Release Time" occurrences
    start_indices = [i for i, line in enumerate(lines) if "Nominal Release Time" in line]
    
    if not start_indices:
        raise ValueError(f"No 'Nominal Release Time' entries found in file: {file_path}")

    datasets = []
    drop_times = []
    # Process each radiosonde dataset
    for idx, start in enumerate(start_indices):
        # Find the start of the tabular data
        data_start = None
        for i in range(start, len(lines) - 2):
            if lines[i].strip().startswith("Time") and "Press" in lines[i]:  # Detect header row
                data_start = i + 3  # Data starts 2 lines after the column names
                break

        if data_start is None:
            print(f"Warning: No data start found for entry at line {start}")
            continue  # Skip this dataset

        # Extract and convert nominal release time
        try:
            date_time_str = lines[start].split("):")[1].strip()
            drop_time = pd.to_datetime(date_time_str, format='%Y, %m, %d, %H:%M:%S')
        except (IndexError, ValueError) as e:
            print(f"Warning: Failed to parse drop time at line {start}: {e}")
            continue  # Skip this dataset

        drop_times.append(drop_time)

        # Extract column names
        columns = lines[data_start - 3].strip().split()

        # Determine dataset end (next "Nominal Release Time" or end of file)
        end = start_indices[idx + 1] if idx + 1 < len(start_indices) else len(lines)

        # Extract and clean data
        data_lines = [line.strip().split() for line in lines[data_start:end]]
        data = [row for row in data_lines if len(row) == len(columns)]

        # Convert to DataFrame
        df = pd.DataFrame(data, columns=columns)

        if df.empty:
            print(f"Warning: Empty dataset at line {start}")
            continue  # Skip empty datasets

        # Remove rows containing 9999.0 in any column
        df = df[(df != "9999.0").all(axis=1)]
        # Convert numeric columns where possible
        df = df.apply(pd.to_numeric, errors='coerce')
        # Store drop time in DataFrame metadata
        df.attrs["drop_time"] = drop_time
        # Append dataset
        datasets.append(df)

    return datasets

# if __name__ == "__main__":
#     inform_utils.()