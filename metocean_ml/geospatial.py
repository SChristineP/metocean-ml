import numpy as np
import pandas as pd
import geopandas as gpd
import contextily as ctx
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import pyproj
import shapely
from tqdm import tqdm
from . import spectra_tools

from roaring_landmask import RoaringLandmask
from roaring_landmask import LandmaskProvider
provider = LandmaskProvider.Osm
landmask = RoaringLandmask.new_with_provider(provider)

def is_land(lat,lon):
    """
    Check if the given points (latitude, longitude) are on land or sea.

    Uses the landmask library (https://github.com/gauteh/roaring-landmask) to perform the check.

    Parameters
    ----------
    lat : float or array-like
        The latitudes of the points to check.
    lon : float or array-like
        The longitudes of the points to check.

    Returns
    -------
    numpy.ndarray
        A boolean array where `True` indicates the points are on land, and `False` indicates they are in the sea.
    """
    # Landmask can only handle floats.
    lat = np.array(lat,dtype=float)
    lon = np.array(lon,dtype=float)
    
    return landmask.contains_many(lon,lat).reshape(lat.shape)

def haversine_distances_pairwise(lat0,lon0,lat1,lon1):
    """
    Calculate the great-circle distance (haversine distance) between latitude and longitude pairs.
    This implementation computes pairwise/elementwise distances, meaning all input arrays must have the same shape.

    Parameters
    ----------
    lat0 : float or array-like
        Latitudes of the first set of points.
    lon0 : float or array-like
        Longitudes of the first set of points.
    lat1 : float or array-like
        Latitudes of the second set of points.
    lon1 : float or array-like
        Longitudes of the second set of points.

    Returns
    -------
    numpy.ndarray
        Array of distances in radians. Multiply by Earth's radius (6371 km) to get distance in km.
    """
    
    lat0 = np.radians(lat0)
    lon0 = np.radians(lon0)
    lat1 = np.radians(lat1)
    lon1 = np.radians(lon1)
    
    dLat = lat1-lat0
    dLon = lon1-lon0
    
    d = np.sin(dLat/2)**2 + np.cos(lat0)*np.cos(lat1)*np.sin(dLon/2)**2
    d = 2*np.arcsin(np.sqrt(d))

    return d

def haversine_distances(lat0,lon0,lat1,lon1):
    """
    Calculate haversine distances between all combinations of lat/lon points.

    Parameters
    ----------
    lat0, lon0 : float or array-like
        The first set of latitude and longitude points. These must have equal shapes.
    lat1, lon1 : float or array-like
        The second set of latitude and longitude points. These must have equal shapes.

    Returns
    -------
    numpy.ndarray
        Array with shape (N_0, N_1) where N_0 is the size of the first set of points (lat0, lon0)
        and N_1 is the size of the second set (lat1, lon1). The values represent the arc distance 
        from each point in the first set to each point in the second set. Multiply by Earth's radius 
        (6371 km) to get distance in km.
    """
    lat0 = np.array(lat0)
    lon0 = np.array(lon0)
    lat1 = np.array(lat1)
    lon1 = np.array(lon1)
    
    lat0,lat1 = np.meshgrid(lat0,lat1,indexing='ij')
    lon0,lon1 = np.meshgrid(lon0,lon1,indexing='ij')
    
    return haversine_distances_pairwise(lat0,lon0,lat1,lon1)


def merge_directions_points(points,directions,drop_duplicates="points",on_index=False,merge_tolerance=1):
    '''
    Add information from dataframe of directions with column dir_from,
    to a dataframe of points with column dir_from.
    
    Parameters
    ----------
    points : pandas DataFrame
        DataFrame containing the points with "r" and "dir_from" columns.
    directions : pandas DataFrame
        DataFrame containing directional information.
    drop_duplicates : str, optional, default="points"
        Specifies which columns to drop from duplicate entries. Can be "points" or "directions".
    on_index : bool, optional, default=False
        If True, merge on "dir_idx", otherwise on "dir_from".
    merge_tolerance : int, optional, default=1
        Tolerance for merging points when merging on "dir_from". 
        Not being able to match rows within tolerance will raise an error.

    Parameters
    ----------
    pandas DataFrame
        Merged DataFrame with additional directional information.
    '''

    for param in ["r","dir_from"]:
        assert param in points.columns

    keep = "dir_idx" if on_index else "dir_from"
    if drop_duplicates == "points": 
        drop = [c for c in points.columns if c in directions.columns and c!=keep]
        points = points.drop(drop,axis=1)
    if drop_duplicates == "directions": 
        drop = [c for c in directions.columns if c in points.columns and c!=keep]
        directions = directions.drop(drop,axis=1)

    if on_index:
        if directions["dir_idx"].nunique() != points["dir_idx"].nunique():
            raise IndexError("Non-matching indices. Verify that tables are generated with the same set of directions.")
        points = pd.merge(points, directions, on="dir_idx", how='left')
    else:
        points = pd.merge_asof(
            points.sort_values('dir_from'), 
            directions.sort_values('dir_from'),
            on = 'dir_from',
            direction = 'nearest',
            tolerance = merge_tolerance)

    if 'r' in points.columns and 'group_vel' in points.columns:
        points["delay"] = (1000/3600)*points["r"]/points["group_vel"]
        points["delay"] = points["delay"].fillna(0)

    return points

def wave_properties_from_source_points(origin, spec, lat, lon):
    '''
    Add info from spec to a list of points given by lat and lon.
    Works by interpolating the spectra to the list of directions from origin to each point.
    
    Parameters
    ----------
    origin : tuple
        Latitude and longitude of the origin point.
    spec : object
        Spectral information object.
    lat : array-like
        Latitude values of the points.
    lon : array-like
        Longitude values of the points.

    Returns
    -------
    pandas DataFrame
        DataFrame with added wave properties for each point.
    '''
    origin = np.array(origin)
    points = []
    for i,(x,y) in enumerate(zip(lon,lat)):
        distance = calculate_distance(origin[0],origin[1],y,x)
        dir_from = calculate_bearing(origin[0],origin[1],y,x)
        dir_to = (dir_from + 180) % 360
        points.append({"node":i,"dir_from":dir_from,"dir_to":dir_to,"r":distance,"lat":y,"lon":x})
    points = pd.DataFrame(points).sort_values("dir_to").reset_index(drop=True)
    points["dir_idx"] = np.arange(len(points))

    directions = spectra_tools.directional_spec_info(spec)

    points = merge_directions_points(points,directions,on_index=False)

    return points

def _create_wedge(center_lat, center_lon, dir_from, dir_to, radius_km, edge_resolution=100, arc_resolution=10):
    """
    Create a wedge polygon centered at (center_lon, center_lat) with given direction, radius, and span.
    
    Parameters
    ----------
    center_lat : float
        Latitude of the center of the wedge.
    center_lon : float
        Longitude of the center of the wedge.
    dir_from : float
        Starting direction (in degrees).
    dir_to : float
        Ending direction (in degrees).
    radius_km : float
        Radius of the wedge in kilometers.
    edge_resolution : int, optional, default=100
        Number of points along the edge of the wedge.
    arc_resolution : int, optional, default=10
        Number of points along the arc.

    Returns
    -------
    shapely.geometry.Polygon
        A shapely Polygon object representing the wedge.
    """
    geod = pyproj.Geod(ellps="WGS84")
    transformer = pyproj.Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)

    radius_m = radius_km * 1000  # Convert kilometers to meters
    angles = np.linspace(dir_from, dir_to, arc_resolution)
    long_side = np.linspace(radius_m,0,edge_resolution)

    l_side = [geod.fwd(center_lon,center_lat,angles[0],r)[:2] for r in long_side][::-1]
    r_side = [geod.fwd(center_lon,center_lat,angles[-1],r)[:2] for r in long_side]
    boundary_points = [geod.fwd(center_lon, center_lat, angle, radius_m)[:2] for angle in angles]

    # Close the polygon by adding the center point
    #boundary_points = [(center_lon, center_lat)] + boundary_points + [(center_lon, center_lat)]
    boundary_points = l_side+boundary_points+r_side
    wedge_polygon = shapely.geometry.Polygon(boundary_points)

    # Transform to Web Mercator
    wedge_polygon_mercator = shapely.ops.transform(lambda x, y: transformer.transform(x, y), wedge_polygon)
    return wedge_polygon_mercator


def line_of_sight_plot(center_lat,
                       center_lon,
                       points:pd.DataFrame,
                       as_sectors=True,
                       map_detail=3,
                       color_var="",
                       cmap=plt.get_cmap("Spectral_r"),
                       alpha=0.75,
                       ):
    
    '''
    For a given location, with 'points' from e.g. wave_properties_from_source_points, 
    visualize the potential wave generating areas on a map.
    
    Parameters
    ----------
    center_lat : float
        Latitude of the center point.
    center_lon : float
        Longitude of the center point.
    points : pandas DataFrame
        DataFrame containing latitude, longitude, and other relevant data.
    as_sectors : bool, optional, default=True
        If True, visualize the points as sectors.
    map_detail : int, optional, default=3
        Level of zoom for the map.
    color_var : str, optional, default=""
        Column name to color the sectors based on.
    cmap : colormap, optional, default=plt.get_cmap("Spectral_r")
        Colormap to use for coloring the sectors.
    alpha : float, optional, default=0.75
        Transparency level for the colored sectors.

    Returns
    -------
    fig, ax : matplotlib.figure.Figure, matplotlib.axes.Axes
        The figure and axis for the plot.
    '''
    
    fig,ax = plt.subplots(figsize=[10,10])
    if as_sectors:
        wedges = points.loc[points.groupby(["dir_from"])["r"].idxmax()].reset_index()
        wedges["dir-"] = wedges["dir_from"] - 0.5*wedges["dir_from"].diff(1)
        wedges.loc[0,"dir-"] = wedges.loc[0,"dir_from"] - 0.5*(wedges.loc[0,"dir_from"] - wedges["dir_from"].iloc[-1] + 360)
        wedges["dir+"] = np.roll(wedges["dir-"],-1)
        wedges.loc[len(wedges)-1,"dir+"] += 360
        wedges['wedge_mercator'] = wedges.apply(lambda row: _create_wedge(center_lat, center_lon, row['dir-'], row["dir+"], row['r']),axis=1)
        
        data = {'latitude': points["lat"], 'longitude': points["lon"]}
        gdf = gpd.GeoDataFrame(data, geometry=gpd.points_from_xy(points["lon"], points["lat"]), crs="EPSG:4326")
        gdf = gdf.to_crs(epsg=3857)
        gdf.plot(ax = ax,color="tab:green", alpha=0, markersize=3)

        if color_var in points.columns:
            norm = mcolors.Normalize(vmin=0,vmax=wedges[color_var].max())
            wedges["sector_color"] = norm(wedges[color_var])
        else:
            wedges["sector_color"] = 0.75
            if color_var:
                print(f"Warning: color_var not found in columns of points: {points.columns}")

        for i,row in tqdm(wedges.iterrows(),total=len(wedges)):
            x,y = row["wedge_mercator"].exterior.xy
            ax.fill(x,y,alpha=alpha,color=cmap(row["sector_color"]),edgecolor="none")

        if color_var in points.columns:
            sm = plt.cm.ScalarMappable(norm,cmap)
            fig.colorbar(sm,ax=ax,label=color_var)

    else:
        data = {'latitude': points["lat"], 'longitude': points["lon"]}
        gdf = gpd.GeoDataFrame(data, geometry=gpd.points_from_xy(points["lon"], points["lat"]), crs="EPSG:4326")
        gdf = gdf.to_crs(epsg=3857)
        gdf.plot(ax = ax,color="tab:green", alpha=alpha, markersize=3)

    ctx.add_basemap(ax, source=ctx.providers.OpenStreetMap.Mapnik, zoom=map_detail)
    plt.xticks([],[])
    plt.yticks([],[])
    return fig,ax

def calculate_bearing(lat0, lon0, lat1, lon1):
    '''
    Calculate the bearing between points (elementwise if multiple), in degrees clockwise from north.
    
    Parameters
    ----------
    lat0 : float or np.ndarray
        Latitude(s) of the starting point(s) in degrees.
    lon0 : float or np.ndarray
        Longitude(s) of the starting point(s) in degrees.
    lat1 : float or np.ndarray
        Latitude(s) of the destination point(s) in degrees.
    lon1 : float or np.ndarray
        Longitude(s) of the destination point(s) in degrees.

    Returns
    -------
    np.ndarray
        Bearing(s) from point 0 towards point 1 in degrees, measured clockwise from north.

    References
    ----------
    https://www.igismap.com/formula-to-find-bearing-or-heading-angle-between-two-points-latitude-longitude/
    '''
    
    # Convert input to radians
    lat0 = np.radians(lat0)
    lon0 = np.radians(lon0)
    lat1 = np.radians(lat1)
    lon1 = np.radians(lon1)

    X = np.cos(lat1) * np.sin(lon1-lon0)
    Y = np.cos(lat0) * np.sin(lat1) - np.sin(lat0) * np.cos(lat1) * np.cos(lon1-lon0)
    
    return np.atan2(X,Y)

def calculate_distance(lat0,lon0,lat1,lon1):
    '''
    Calculate the distance between two geographical points in kilometers.
    Returns a distance matrix with all combinations of points in lat0/lon0 and lat1/lon1.
    
    Parameters
    ----------
    lat0 : float or np.ndarray
        Latitude(s) of the starting point(s) in degrees.
    lon0 : float or np.ndarray
        Longitude(s) of the starting point(s) in degrees.
    lat1 : float or np.ndarray
        Latitude(s) of the destination point(s) in degrees.
    lon1 : float or np.ndarray
        Longitude(s) of the destination point(s) in degrees.

    Returns
    -------
    np.ndarray
        Distance(s) between the points in kilometers.
    '''
    return np.squeeze(haversine_distances(lat0,lon0,lat1,lon1))*6371

def distribute_points(lat0:float,
                      lon0:float,
                      bearing:float|np.ndarray,
                      distance:float|np.ndarray,
                      )->tuple[np.ndarray,np.ndarray]:
    '''
    Calculate new latitude and longitude points based on an origin point, direction, and distance.
    
    Parameters
    ----------
    lat0 : float
        Latitude of the origin point in degrees.
    lon0 : float
        Longitude of the origin point in degrees.
    bearing : float or np.ndarray
        Direction(s) in degrees to distribute points. Values are between 0 and 360 degrees.
    distance : float or np.ndarray
        Distance(s) in kilometers to distribute points.

    Returns
    -------
    tuple of np.ndarray
        Tuple containing two arrays: the latitudes and longitudes of the distributed points.
        The shape of each array is (n_bearings, n_distances).
    
    Raises
    ------
    TypeError
        If the input lat0 or lon0 is not a float.
    ValueError
        If the input latitude or longitude is out of valid bounds.
    '''
    
    lat0 = np.radians(lat0)
    lon0 = np.radians(lon0)
    bearing = np.radians(np.array(bearing)%360)

    if (lat0.size != 1) or (lon0.size != 1):
        raise TypeError('Input lat and lon should be floats, not arrays.')
    if lat0>90 or lat0<-90:
        raise ValueError(f'Expected -90<lat<90, got {lat0}.')
    if lon0>180 or lon0<-180:
        raise ValueError(f'Expected -180<lon<180, got {lat0}.')

    distance = np.array(distance)/6371000 # earth average radius [m]
    bearing,distance = np.meshgrid(bearing,distance,indexing='ij')
    
    lat1 = np.arcsin( np.sin(lat0)*np.cos(distance) + np.cos(lat0)*np.sin(distance)*np.cos(bearing))
    lon1 = lon0 + np.arctan2(np.sin(bearing)*np.sin(distance)*np.cos(lat0),np.cos(distance)-np.sin(lat0)*np.sin(lat1))

    return np.degrees(lat1), np.degrees(lon1)

def get_fetch(
    lat,
    lon,
    directions=360,
    step_size_minor=10,
    step_size_major=1e4,
    verbose = False
    ):
    '''
    Calculate the fetch distance for all directions from a given latitude and longitude point.
    
    Parameters
    ----------
    lat : float
        Latitude of the point of interest in degrees.
    lon : float
        Longitude of the point of interest in degrees.
    directions : int, optional
        Number of directions to check, uniformly distributed starting from 0 (default is 360).
    step_size_minor : float, optional
        Step size in meters for smaller increments (default is 10 meters).
    step_size_major : float, optional
        Step size in meters for major increments (default is 10,000 meters).
    verbose : bool, optional
        If True, prints feedback for each iteration (default is False).
    
    Returns
    -------
    pandas.Series
        A series containing fetch distances for each direction, indexed by direction in degrees.
    
    Raises
    ------
    ValueError
        If the point is located on land.
    '''

    if is_land(lat,lon): raise ValueError(f'Point {lat},{lon} is on land.')

    # List of directions to search and respective fetch distance (to be filled)
    directions = np.linspace(0,360,directions,endpoint=False)
    fetch = np.zeros_like(directions)

    search_offset = 0
    while np.any(fetch==0):
        if verbose: print('Iteration {}: Searching {} directions from {} to {} m.'.format(
                search_offset//step_size_major, np.count_nonzero(fetch==0),
                search_offset,search_offset+step_size_major))

        # Distances from center point, we search in blocks e.g. [0,1e4), then [1e4, 2*1e4) etc.
        distances = np.arange(search_offset,search_offset+step_size_major,step_size_minor)

        # Distribute points only in directions where we haven't found fetch yet
        lats,lons = distribute_points(lat,lon,directions[fetch==0],distances)
        land = is_land(lats,lons)

        # Fetch is calculated relative to the block we are searching
        offset_fetch = np.argmax(land,axis=1)*step_size_minor

        # Then nonzero entries (fetch) are increased with the block offset
        offset_fetch[offset_fetch!=0] = offset_fetch[offset_fetch!=0] + search_offset

        # Offset fetch (zero and nonzero entries) are added to the overall fetch vector
        fetch[fetch==0] = offset_fetch
        search_offset += step_size_major

    return pd.Series(data=fetch,index=directions,name='Fetch').rename_axis('Direction')