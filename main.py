
import inform_utils
import inform_grid_utils

vars_to_read = ['Time','GGALT','LATC','LONC', # 4-D Position
                'UIC','VIC','WIC',            # winds
                'ATX','PSXC','EWX','RHUM',           # other state params
                ] 

dir = '/Users/patnaude/Documents/Data/SOCRATES'
flight_paths = inform_utils.find_flight_fnames(dir)

nc = inform_utils.open_nc(flight_paths[10])
df = inform_utils.read_flight_nc_1hz(nc,vars_to_read)

cesm_dir = '/Users/patnaude/Documents/Data/cesmdata'

# Open model files both free and nudg
cesm_files = inform_utils.find_nc_fnames(cesm_dir)
cesm_ndg = inform_utils.open_nc(cesm_files['Nudg'][0])
cesm_fr = inform_utils.open_nc(cesm_files['Free'][0])

grid_dat, grid, bounds = inform_grid_utils.grid_flight_dat(cesm_fr,cesm_ndg,df,nc)

# Plot 2-D of flight location 
# inform_grid_utils.plot_grid(cesm_fr, bounds, nc)

# Visualize the 3-D flight track and gridded data
# inform_grid_utils.plot_3d_track(grid_dat,nc)


