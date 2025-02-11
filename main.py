import netCDF4
import pathlib as path
import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta
from fnmatch import fnmatch
from typing import Iterable
import xarray as xr
from itertools import product
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import inform_utils

dir = '/Users/patnaude/Documents/Data/SOCRATES'

vars_to_read = ['Time','GGALT','LATC','LONC', # 4-D Position
                'UIC','VIC','WIC',            # winds
                'ATX','PSXC','EWX','RHUM',           # other state params
                ] 

flight_paths = inform_utils.find_flight_fnames(dir)

nc = inform_utils.open_nc(flight_paths[2])
df = inform_utils.read_flight_nc_1hz(nc,vars_to_read)

cesm_dir = '/Users/patnaude/Documents/Data/cesmdata'

cesm = inform_utils.find_flight_fnames(cesm_dir)
cesm = inform_utils.open_nc(cesm[0])

