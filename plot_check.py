import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import inform_grid_utils
import inform_utils

# Assuming grid and bounds are already defined

# grid, bounds = inform_grid_util.grid_flight_dat(cesm_fr, nc)
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