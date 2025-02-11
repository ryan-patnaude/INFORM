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
    :param read_vars: An optional list of strings of variable names to be read into memory. A default
                      list, vars_to_read, is specified above. Passing in a similar list will read in those variables
                      instead.

    :return: Returns a pandas data frame.
    """

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
    

    # concatenate the list of dataframes into a single dataframe and return it
    return pd.concat(data, axis=1, ignore_index=False)

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

class flight_obj:
    """
    flight_obj's are classes that hold flight data (i.e. variables indicated by read_vars) from a provided file path string.
    The __init__ takes a file path string and a list of vars to read (vars_to_read by default).
    The __init__ assigns:
    self.file_path: str; the file path passed in
    self.read_vars_attempted: list[str]; the originally passed in list of vars to read
    self.nc: netCDF4._netCDF4.Dataset; the opened netcdf object
    self.df: pd.DataFrame; a dataframe holding the read in data
    self.rate: str; a string indicating the rate of the data read in
    self.read_vars: list[str]; list of the vars that were successfully read in
    """
    def __init__(self, file_path: str, read_vars):
        # assign input vars
        self.file_path = path.Path(file_path)
        self.read_vars_attempted = read_vars

        # open netcdf file if the file exists, assign to self.nc
        if self.file_path.is_file():
            self.nc = netCDF4.Dataset(self.file_path)
        else:
            raise FileNotFoundError(f"File {self.file_path} did not exist!")

        # read in the variables, assign DataFrame to self.df,
        #                               rate to self.rate,
        #                               vars read in to self.read_vars
        dim_names = list(self.nc.dimensions.keys())
        if 'sps25' in dim_names:
            self.df = read_flight_nc_25hz(self.nc, self.read_vars_attempted)
            self.rate = "25Hz"
        else:
            self.df = read_flight_nc_1hz(self.nc, self.read_vars_attempted)
            self.rate = "1Hz"
        self.read_vars = list(self.df.keys())

# if __name__ == "__main__":
#     inform_utils.()