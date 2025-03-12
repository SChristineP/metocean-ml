import numpy as np
import xarray as xr
import pandas as pd
from scipy.interpolate import CubicSpline, RegularGridInterpolator
from scipy.integrate import simpson
    
def _interpolate_cubic(fp,n):
    '''
    Cubic spline interpolation within the range of fp, for resampling.

    Parameters
    ----------
    fp : np.ndarray
        Values of the function
    n : int
        New sampling resolution.
    '''
    l = len(fp)
    x = np.linspace(0,l-1,n)
    xp = np.arange(l)
    spl = CubicSpline(xp,fp)
    return spl(x)

def interpolate_2D_spec(spec  : np.ndarray,
                        freq0 : np.ndarray,
                        dir0  : np.ndarray,
                        freq1 : np.ndarray,
                        dir1  : np.ndarray,
                        method: str="linear"
                        ) -> np.ndarray:
    '''
    Interpolate 2D wave spectra from fre0 and dir0 to freq1 and dir1.
    
    Parameters
    ---------
    spec : np.ndarray
        N-D array of spectra, must have dimensions [..., frequencies, directions].
    freq0 : np.ndarray
        Array of frequencies.
    dir0 : np.ndarray
        Array of directions.
    freq1 : np.ndarray
        Array of new frequencies.
    dir1 : np.ndarray
        Array of new directions.
    method : str
        The interpolation method used by scipy.interpolate.RegularGridInterpolator(),
        e.g. "nearest", "linear", "cubic", "quintic".
        
    Returns
    -------
    spec : np.ndarray
        The interpolated spectra.
    '''
    # Sort on directions, required for interpolation.
    sorted_indices = np.argsort(dir0)
    dir0 = dir0[sorted_indices]
    spec = spec[...,sorted_indices]

    # Create current and new interpolation points.
    points = tuple(np.arange(s) for s in spec.shape[:-2]) + (freq0,dir0)
    coords = tuple(np.arange(s) for s in spec.shape[:-2]) + (freq1,dir1)
    reorder = tuple(np.arange(1,len(coords)+1))+(0,)
    coords = np.transpose(np.meshgrid(*coords,indexing="ij"),reorder)

    # Define interpolator and interpolate.
    grid_interp = RegularGridInterpolator(points=points,values=spec,fill_value=None,bounds_error=False)
    return grid_interp(coords,method=method)

def scale_2D_spec(  spec:np.ndarray,
                    frequencies : np.ndarray,
                    directions  : np.ndarray,
                    new_frequencies: np.ndarray | int = 20,
                    new_directions: np.ndarray | int = 20,
                    method="linear"
                    ) -> tuple[np.ndarray,np.ndarray,np.ndarray]:
    '''
    Interpolate wave spectra to a new set of specific frequencies and directions.
    
    Parameters
    ----------
    spec : np.ndarray
        Array of spectra, with last two dimensions corresponding to frequencies and directions in order.
    frequences : np.ndarray
        Array of frequency values, corresponding to the second-last dimension of spec.
    directions : np.ndarray
        Array of direction values (degrees), corresponding to the last dimension of spec.
    new_frequencies : int or np.ndarray
        New frequency values, or an integer for number of (interpolated) frequency values.
    new_directions : int or np.ndarray
        New directions, or an integer for number of (uniformly distributed) new directions.
    method : str
        Any interpolation method allowed by scipy's RegularGridInterpolator, such as 
        "nearest", "linear", "cubic", "quintic".

    Returns
    --------
    new_spec : np.ndarray 
        Interpolated spectra, array with same shape as input spectra except the last two dimensions.
    new_frequencies : np.ndarray
        The new set of frequencies
    new_directions : np.ndarray
        The new set of directions.
    '''

    new_frequencies = np.array(new_frequencies)
    new_directions = np.array(new_directions)
    
    # Create freqs and dirs through interpolation if not supplied directly
    if new_frequencies.size == 1:
        new_frequencies = _interpolate_cubic(frequencies,new_frequencies)
    if new_directions.size == 1:
        new_directions = np.linspace(0,360,new_directions,endpoint=False)

    spec = interpolate_2D_spec(spec,frequencies,directions,new_frequencies,new_directions,method=method)
    return spec, new_frequencies, new_directions

def interpolate_dataarray_spec( spec: xr.DataArray,
                                new_frequencies: np.ndarray | int = 20,
                                new_directions: np.ndarray | int = 20,
                                method="linear"
                                ):
    '''
    Interpolate 2D wave spectra to a new shape.
    The last two dimensions of spec must represent frequencies and directions.
    This is just a wrapper for scale_2D_spec, to keep track of the dataarray metadata.
    
    Parameters
    ---------
    spec : xr.DataArray
        Array of spectra. Must have dimensions [..., frequencies, directions].
    new_frequencies : xr.DataArray or np.ndarray or int
        Either an array of new frequences, or an integer for the number of new frequencies.
        If integer, new frequencies will be created with cubic interpolation.
    new_directions : xr.DataArray or np.ndarray or int
        Either an array of new directions, or an integer for the number of new directions.
        If integer, new directions will be created with linear interpolation.
    method : str
        The interpolation method used by scipy.interpolate.RegularGridInterpolator(),
        e.g. "nearest", "linear", "cubic", "quintic".
    
    Returns
    -------
    spec : xr.DataArray
        The 2D-interpolated spectra.
    '''

    # Extract dimension labels and coordinate arrays from spec.
    spec_coords = spec.coords
    spec_dims = list(spec.dims)
    freq_var = spec_dims[-2]
    dir_var = spec_dims[-1]
    free_dims = spec_dims[:-2]

    frequencies = spec_coords[freq_var]
    directions = spec_coords[dir_var]

    new_spec, new_frequencies, new_directions = scale_2D_spec(
        spec.data,frequencies,directions,new_frequencies,new_directions,method=method)
    
    new_coordinates = {k:spec_coords[k] for k in free_dims}
    new_coordinates[freq_var] = new_frequencies
    new_coordinates[dir_var] = new_directions
    return xr.DataArray(new_spec,new_coordinates)

def _reshape_spectra(spec, frequencies, directions):
    '''
    Standardize format of spectra, frequencies and directions for further processing.

    Parameters
    ----------
    spec : np.ndarray or pd.DataFrame or xr.DataArray
        Array of spectra.
    frequencies : np.ndarray
        List of freqency values corresponding to the second-last dimension of the spectra.
    directions : np.ndarray
        List of directions corresponding to the last dimension of the spectra.

    Returns
    -------
    np.ndarray
        Spectra shape [.., frequencies, dimensions]
    np.ndarray
        Frequencies
    np.ndarray
        Directions
    '''
    # Make sure all arrays are numpy.
    spec = np.array(spec)
    frequencies = np.array(frequencies)
    directions = np.array(directions)

    # Check if spec values and shape are OK
    if np.any(spec < 0):
        print("Warning: negative spectra values set to 0")
        spec = np.clip(spec, a_min=0, a_max=None)

    flat_check = (len(spec.shape)<2)
    if not flat_check:
        freq_check = (len(frequencies) != spec.shape[-2])
        dir_check = (len(directions) != spec.shape[-1])
    if flat_check or freq_check or dir_check:
        try:
            spec = spec.reshape(spec.shape[:-1]+(len(frequencies),len(directions)))
        except:
            raise IndexError("Spec shape does not match frequencies and directions.")

    return spec, frequencies, directions
    
def integrated_parameters(
    spec:       np.ndarray|xr.DataArray, 
    frequencies:np.ndarray|xr.DataArray, 
    directions: np.ndarray|xr.DataArray,
    upsample: int = 1000) -> dict:
    """
    Calculate the integrated parameters of a 2D wave spectrum, 
    or some array/list of spectra. Uses simpsons integration rule.

    Implemented: Hs, peak dir, peak freq.
    
    Parameters
    ---------
    spec : np.ndarray or xr.DataArray
        An array of spectra. The shape must be either 
        [..., frequencies, directions] or [..., frequencies*directions].
    frequencies : np.ndarray or xr.DataArray
        Array of spectra frequencies.
    directions: np.ndarray or xr.DataArray
        Array of spectra directions.
    upsample: int (optional, recommended)
        Upsample frequencies and directions by cubic spline interpolation
        to increase the resolution of peak_dir and peak_freq.
        Does not affect other integrated parameters.
        
    Returns
    -------
    spec_parameters : dict[str, np.ndarray]
        A dict with keys Hs, peak_freq, peak_dir, and values are arrays
        of the integrated parameter.
    """

    spec, frequencies, directions = _reshape_spectra(spec, frequencies, directions)
    
    # # Use argmax to find indices of largest value of each spectrum.
    # peak_dir_freq = np.array([np.unravel_index(s.argmax(),s.shape) 
    #     for s in spec.reshape(-1,len(frequencies),len(directions))])
    # peak_dir_freq = peak_dir_freq.reshape(spec.shape[:-2]+(2,))
    # peak_freq = frequencies[peak_dir_freq[...,0]]
    # peak_dir = directions[peak_dir_freq[...,1]]
    peak_freq, peak_dir = peak_freq_dir(spec,frequencies,directions,upsample=upsample)
    
    # Integration requires radians
    if np.max(directions) > 2*np.pi: 
        directions = np.deg2rad(directions)
    
    # Sort on direction before integration
    sorted_indices = np.argsort(directions)
    directions = directions[sorted_indices]
    spec = spec[...,sorted_indices]
    
    # Wraparound correction
    directions = np.concatenate([directions-directions[0],[2*np.pi]])
    spec = np.concatenate([spec,spec[...,:1]],axis=-1)

    # Integration with simpson's rule
    S_f = simpson(spec, x=directions)
    m0 = simpson(S_f, x=frequencies)
    Hs = 4 * np.sqrt(m0)

    spec_parameters = {
        "Hs":       Hs,
        "peak_freq":peak_freq,
        "peak_dir": peak_dir,
        "peak_period":1/peak_freq
    }

    return spec_parameters

def peak_freq_dir(spec:np.ndarray,
                  frequencies:np.ndarray,
                  directions:np.ndarray,
                  upsample:int=1000):
    '''
    Calculate peak frequency and direction of spectra.

    Parameters
    ----------
    spec : np.ndarray
        Some array of spectra.
    frequencies : np.ndarray
        Frequencies corresponding to the second-last dimension of the spectra.
    directions : np.ndarray
        Directions corresponding to the last dimension of the spectra.
    upsample : int
        Upsample the frequency spectra and direction spectra via 
        cubic splines, for higher resolution of the peak values.

    Returns
    -------
    peak_freq : np.ndarray
        Peak frequency, shape corresponding to the number of input spectra.
    peak_dir : np.ndarray
        Peak directions, shape corresponding to the number of input spectra.

    Notes
    ------
    This function first integrates directions to get frequency peak,
    then integrates over frequencies to get direction peak. In some rare
    cases will this be different to the overall 2D spectra peaks,
    and it is much more computationally efficient due to 1D interpolation.
    '''
    
    freq_spec, frequencies = frequency_spectra(spec,frequencies,directions)
    dir_spec, directions = direction_spectra(spec,frequencies,directions)
    
    if upsample:
        xd = np.linspace(0,360,upsample,endpoint=False)
        dir_spec = np.concat([dir_spec,dir_spec[...,:1]],axis=-1)
        directions = np.concat([directions-directions[0],[360]])

        xf = _interpolate_cubic(frequencies,upsample)

        freq_spec = CubicSpline(frequencies,freq_spec,axis=-1)(xf) 
        dir_spec =  CubicSpline(directions, dir_spec, axis=-1,bc_type='periodic')(xd) 

        peak_freq = xf[freq_spec.argmax(axis=-1)]
        peak_dir = xd[dir_spec.argmax(axis=-1)]

    else:
        peak_freq = frequencies[freq_spec.argmax(axis=-1)]
        peak_dir = directions[dir_spec.argmax(axis=-1)]

    return peak_freq, peak_dir

def frequency_spectra(
    spec:       np.ndarray|xr.DataArray, 
    frequencies:np.ndarray|xr.DataArray, 
    directions: np.ndarray|xr.DataArray
    ) -> np.ndarray|xr.DataArray:
    '''
    Get frequency spectra by integrating over directions.
    
    Parameters
    ---------
    spec : np.ndarray or xr.DataArray
        An array of spectra. The shape must be either 
        [..., frequencies, directions] or [..., frequencies*directions].
    frequencies : np.ndarray or xr.DataArray
        Array of spectra frequencies.
    directions: np.ndarray or xr.DataArray
        Array of spectra directions.
        
    Returns
    -------
    1D_spec : np.ndarray or xr.DataArray
        Array of 1D spectra.
    '''

    spec, frequencies, directions = _reshape_spectra(spec,frequencies,directions)
    
    # Integration requires radians
    if np.max(directions) > 2*np.pi: 
        directions = np.deg2rad(directions)
    
    # Sort on direction before integration
    sorted_indices = np.argsort(directions)
    directions = directions[sorted_indices]
    spec = spec[...,sorted_indices]
    
    # Wraparound correction
    directions = np.concatenate([directions-directions[0],[2*np.pi]])
    spec = np.concatenate([spec,spec[...,:1]],axis=-1)

    # Integration with simpson's rule
    S_f = simpson(spec, x=directions)
    
    return S_f, frequencies

def direction_spectra(
    spec:       np.ndarray|xr.DataArray, 
    frequencies:np.ndarray|xr.DataArray, 
    directions: np.ndarray|xr.DataArray
    ) -> np.ndarray|xr.DataArray:
    '''
    Get direction spectra by integrating frequencies.
    
    Parameters
    ---------
    spec : np.ndarray or xr.DataArray
        An array of spectra. The shape must be either 
        [..., frequencies, directions] or [..., frequencies*directions].
    frequencies : np.ndarray or xr.DataArray
        Array of spectra frequencies.
    directions: np.ndarray or xr.DataArray
        Array of spectra directions.
        
    Returns
    -------
    1D_spec : np.ndarray or xr.DataArray
        Array of 1D spectra.
    '''

    spec, frequencies, directions = _reshape_spectra(spec,frequencies,directions)
    
    # Sort on direction before integration
    sorted_indices = np.argsort(directions)
    directions = directions[sorted_indices]
    spec = spec[...,sorted_indices]
    
    # Integration with simpson's rule
    S_d = simpson(spec, x=frequencies, axis=-2)
    
    return S_d, directions


def directional_spec_info(spec:xr.DataArray,
                          directions=360,
                          directions_from_energy = True,
                          energy_smoothing=0.1):
    '''
    Calculate parameters for each direction, such as Tp, freq, group velocity, energy.
    Works by interpolating spectra to the chosen number of directions,
    then calculating the integrated parameters for each direction seperately.

    Parameters
    ----------
    spec : xr.DataArray 
        Spectra array where the two last dimensions are frequency and direction.
    directions : int or np.ndarray
        Number of directions, or a list of specific directions.
    directions_from_energy : bool, default True
        This option allows directions to be distributed with density according
        to the spectral energy density. Used if directions is an integer.
    energy_smoothing : float
        Adjustment of the directions-from-energy algorithm, if used. 
        Higher value will distribute more uniformly while still according to density.

    Returns
    -------
    pd.DataFrame
        Dataframe with one row per direction, and associated metadata extracted from the spectrum.
    '''

    if not isinstance(spec,xr.DataArray): raise TypeError("Spec must be dataarray.")
    if "time" in spec.dims: spec = spec.mean("time")
    if len(spec.dims)!=2: raise TypeError("Unknown spectra shape. Expected [time, freq, dir] or [freq, dir].")
    dim_dir = spec.dims[1]
    
    # Integrate energy over frequency to get energy per direction, and create a cublic spline fit.
    spec = spec.sortby(dim_dir)
    #dir_spec = spec.data.mean(axis=0)
    dir_spec,spec_dirs = direction_spectra(spec,spec[spec.dims[0]].data,spec[spec.dims[1]].data)
    #spec_dirs = np.concatenate([[0],spec[dim_dir].data,[360]])
    if spec_dirs[0] > 1 and spec_dirs[-1] < 359:
        spec_dirs = np.concatenate([[0],spec_dirs,[360]])
        boundary_value = [(dir_spec[-1]+dir_spec[0])/2]
        dir_spec = np.concatenate([boundary_value,dir_spec,boundary_value])
    energy_interpolator = CubicSpline(spec_dirs,dir_spec,bc_type="natural")

    
    if isinstance(directions,int):
        if directions_from_energy:
            # Upscale directional energy distribution, then distribute directions based on energy density.
            xd = np.linspace(start=0,stop=360,num=10000,endpoint=False)
            dirspec = energy_interpolator(xd)
            density = dirspec + energy_smoothing
            dir_dist = density.cumsum()/density.sum()
            dir_ind = dir_dist.searchsorted(np.linspace(0,1,directions,endpoint=False))
            dirs = xd[dir_ind]
        else:
            dirs = np.linspace(0,360,directions,endpoint=False)
        dir_idx = np.arange(len(dirs))

    elif hasattr(directions,"__len__"):
        dirs = directions
        dir_idx = np.arange(len(dirs))
        sort = np.argsort(dirs)
        dirs = dirs[sort]
        dir_idx = dir_idx[sort]
    else:
        raise TypeError("directions must be either an integer number of directions, or a list/array of directions.")
    
    # Upscale the full spectra to find peak freq per direction
    mean_spec = spec
    if spec_dirs[0] > 1 and spec_dirs[-1] < 359:
        spec_boundary = (mean_spec.isel(direction=0).data + mean_spec.isel(direction=-1).data)/2
        dim_freq = mean_spec.dims[-2]
        freqcoord = mean_spec[dim_freq].data
        dir0 = xr.DataArray(np.expand_dims(spec_boundary,1),coords={dim_freq:freqcoord,dim_dir:[0]})
        dir360 = xr.DataArray(np.expand_dims(spec_boundary,1),coords={dim_freq:freqcoord,dim_dir:[360]})
        mean_spec = xr.concat([dir0,mean_spec,dir360],dim="direction")
    mean_spec = interpolate_dataarray_spec(mean_spec,len(dirs),dirs,method="cubic")
    
    # Structure parameters for each direction in a dataframe
    peak_freq = mean_spec.idxmax(mean_spec.dims[0])
    peak_Tp = 1/peak_freq
    dir_energy = energy_interpolator(dirs)
    velocities = 9.80665 * peak_Tp / (4*np.pi)
    dirs_comingfrom = (dirs+180)%360
    directions = pd.DataFrame({
        "dir_to":dirs,
        "dir_from":dirs_comingfrom,
        "dir_idx":dir_idx,
        "Tp":peak_Tp,
        "freq":peak_freq,
        "energy":dir_energy,
        "group_vel":velocities,
    })
    return directions
