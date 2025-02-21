import numpy as np
import pandas as pd

def dirmag_to_uv(wind_direction, wind_speed, going_to=True):
    '''
    Get wind x (east) and y (north) component 
    from speed and direction (degrees, default: going to).

    Parameters
    ---------
    wind_direction : np.ndarray
        Wind direction (degrees)
    wind_speed : np.ndarray
        Wind Speed (degrees)
    going_to : bool
        Controls direction convention, False gives "from" direction.
    '''
    
    wind_direction = np.radians(wind_direction)
    
    if not going_to: 
        wind_direction = (wind_direction+np.pi)%(2*np.pi)
    
    u = wind_speed*np.sin(wind_direction)
    v = wind_speed*np.cos(wind_direction)
    return u,v

def uv_to_dirmag(u, v, going_to=True):
    '''
    Get direction (degrees, default: going to) and magnitude 
    from u (east) and v (north) components.

    Parameters
    ---------
    u : np.ndarray
        Eastwards velocity
    v : np.ndarray
        Northwards velocity
    going_to : bool, default True
        Controls direction convention, False gives "from" direction.
    '''
    direction = np.degrees(np.arctan2(v,u))
    direction = (90-direction)%360
    
    if not going_to:
        direction = (direction+180)%360
    
    magnitude = np.sqrt(u**2+v**2)
    return direction, magnitude

def direct_fetch(fetch:pd.Series,
                 wind_direction:np.ndarray):
    '''
    Calculate the direct fetch, i.e. in the wind direction.

    Parameters
    ----------
    fetch : pd.Series
        Series containing fetch, with direction as index labels.
    wind_direction : np.ndarray | pd.Series
        Direction of the wind, convention "coming from".
    '''

    fetch_dir = fetch.index.values
    if type(wind_direction)==pd.Series:
        wind_direction = wind_direction.values
    else:
        wind_direction = np.array(wind_direction)

    # Normalized absolute difference between fetch and wind direction
    norm_dir = (fetch_dir[np.newaxis,:] - wind_direction[:,np.newaxis])%360
    abs_diff = np.minimum(norm_dir, 360 - norm_dir)

    fetch_idx = abs_diff.argmin(axis=1)
    
    return fetch.values[fetch_idx]


def effective_fetch(fetch:pd.Series,
                    wind_direction:np.ndarray,
                    sector:float = 30):
    '''
    Calculate the effective fetch by a cos^2-weight.
    Reference: SPM (1966).

    Parameters
    ----------
    fetch : pd.Series
        Series containing fetch, with direction as index labels.
    wind_direction : np.ndarray | pd.Series
        Direction of the wind, convention "coming from".
    sector : float
        Size of the sector over which to calculate effective fetch (degrees).
    '''

    fetch_dir = fetch.index.values
    if type(wind_direction)==pd.Series:
        wind_direction = wind_direction.values

    T = len(wind_direction)
    N = int(sector/np.mean(np.diff(fetch.index))) # number of indices within sector

    if N < 5: raise ValueError(f'Number of fetch within sector is too small ({N}).')

    # Normalized absolute difference between fetch and wind direction (T x 360)
    norm_dir = (fetch_dir[np.newaxis,:] - wind_direction[:,np.newaxis])%360
    abs_diff = np.minimum(norm_dir, 360 - norm_dir)

    # Indices of N closest directions (T x N)
    mask = np.argsort(abs_diff,axis=1)[:,:N]

    # Fetch (T x N) and associated direction deviation (T x N)
    sector_fetch = fetch.values[mask]
    row_indices = np.arange(T)[:,np.newaxis]
    sector_dir_diff = np.radians(abs_diff[row_indices,mask])

    # Cosine squared rule
    numerator = np.sum(sector_fetch*np.cos(sector_dir_diff)**2, axis=1)
    effective_fetch = numerator/np.sum(np.cos(sector_dir_diff), axis=1)
    return effective_fetch


def fetch_laws(
        wind,
        fetch,
        depth = 1e6,
        laws = 'holthuijsen'):
    """
    Calculate Hs, Tp and other parameters based on wind speed and fetch according to traditional fetch laws.

    References: Holthuijsen (2006), Kahma & Calkoen (1992), JONSWAP (1973).

    Parameters
    ---------
    wind : float or np.ndarray or pd.Series
        Wind speed [m/s]. If series, the return is a dataframe of wave parameters. 
        Otherwise, a dict of arrays.
    fetch : float or np.ndarray
        Fetch distance [m].
    depth : flaot or np.ndarray 
        Only used by Holthuijsen. Using a large value will negate the depth term of the equations.
    laws : str
        Which set of laws to use: ['Holthuijsen', 'KahmaCalkoen', 'JONSWAP'].
    
    Returns
    --------
    dict or pd.DataFrame
        The wave parameters. DataFrame will be used if the input wind speed is a pandas Series.
        Which parameters depends on the fetch laws used.

    """
    
    if type(wind)==pd.Series:
        index = wind.index
        wind = wind.values
        return_df = True
    else:
        return_df = False

    if laws.lower() == 'holthuijsen':
        wave_parameters = fetch_law_Holthuijsen(wind=wind,fetch=fetch,depth=depth)
    elif laws.lower() == 'kahmacalkoen':
        wave_parameters = fetch_law_KahmaCalkoen(wind_speed=wind,fetch=fetch)
    elif laws.lower() == 'jonswap':
        wave_parameters = fetch_law_JONSWAP(wind_speed=wind,fetch=fetch)
    else:
        raise ValueError(f'Unknown law {laws}.')

    if return_df:
        return pd.DataFrame(data=wave_parameters,index=index)
    else:
        return wave_parameters


def fetch_law_Holthuijsen(
        wind,
        fetch=1e6,
        depth=1e6):
    '''
    Calculate wind from wind speed [m/s], water depth [m], and fetch distance [m],
    using the equations of Breugem & Holthuijsen (2006).
    
    Parameters
    ---------
    wind : float or array.
        The wind speed, in m/s. 
    fetch : float or array.
        The fetch, in meters. 
    depth : float or array.
        The water depth, in meters. 
        
    Returns
    --------
    H_s : float or array.
        Significant wave height.
    T_s : float or array.
        Significant wave period.
    '''
    
    wind = np.array(wind)
    fetch = np.array(fetch)
    depth = np.array(depth)
    
    g = 9.81

    # Dimensionless depth and fetch
    d_hat = (g*depth)/np.power(wind,2)
    F_hat = (g*fetch)/np.power(wind,2)
    
    # Empirical constants, from the reference.
    H_inf = 0.24;   T_inf = 7.69
    k_1 = 4.41e-4;  k_2 = 2.77e-7
    k_3 = 0.343;    k_4 = 0.10
    m_1 = 0.79;     m_2 = 1.45
    m_3 = 1.14;     m_4 = 2.01
    p = 0.572;      q = 0.187

    H_hat = H_inf*np.power(np.tanh(k_3*np.power(d_hat,m_3)) * np.tanh((k_1*np.power(F_hat,m_1))/(np.tanh(k_3*np.power(d_hat,m_3)))),p)
    T_hat = T_inf*np.power(np.tanh(k_4*np.power(d_hat,m_4)) * np.tanh((k_2*np.power(F_hat,m_2))/(np.tanh(k_4*np.power(d_hat,m_4)))),q)

    H_s = H_hat*np.power(wind,2)/g
    T_s = T_hat*wind/g
    return {
        'hs': H_s, 
        'tp':T_s
        }

def fetch_law_KahmaCalkoen(
        wind_speed:np.ndarray,
        fetch:np.ndarray, 
        ):
    """
    Calculates wave parameters from wind speed (m/s) and fetch (m) given Kahma & Calkoen (1992) composite curves.

    Parameters
    ----------
    fetch : np.ndarray
        Fetch distance [m]
    wind_speed : np.ndarray
        Wind speed [m/s]

    Notes
    ---------
        Kahma, K. K., and C. J. Calkoen, 1992: Reconciling Discrepancies in the Observed Growth of
        Wind-generated Waves. J. Phys. Oceanogr., 22, 1389-1405,
        https://doi.org/10.1175/1520-0485(1992)022<1389:RDITOG>2.0.CO;2.

    """

    G = 9.81

    u = np.array(wind_speed)
    x = np.array(fetch)

    xhat = G * x / u / u  # Dimensionless fetch

    ehat = (5.2 * 10**-7) * xhat**0.9  # page 1404 - dimensionless energy

    variance = ehat / G / G * u**4 # variance (m**2)

    hs = 4 * variance ** 0.5 # significant wave height (m)

    xhat = G * x / u / u  # Dimensionless fetch

    omegahat = 13.7 * xhat ** (-0.27)  # page 1404 - dimensionless peak frequency

    wp = omegahat * G / u # angular peak frequency

    fp = wp / 2 / np.pi # peak frequency (Hz)

    tp = 1 / fp # peak period (s)

    lp = G / np.pi / 2 * tp ** 2 # peak wavelenght (m)

    kp = 2 * np.pi / lp # peak wavenumber (rad/m)

    return {
        'hs':hs,
        'tp':tp,
        'peak_wavelength': lp,
        'peak_wavenumber': kp
    }

def fetch_law_JONSWAP(
        wind_speed, 
        fetch):
    """
    Calculate wind-generated waves according to
    JONSWAP (Hasselmann et. al. 1973).

    Parameters
    ----------
    wind_speed : np.ndarray
        Wind speed [m/s]
    fetch : np.ndarray
        Fetch [m]

    
    """
    g = 9.81
    Hs = 0.0016*np.power(g,-0.5)*np.power(fetch,0.5)*wind_speed
    Fp = 3.5*np.power(g,0.67)*np.power(fetch,-0.33)*np.power(wind_speed,-0.33)
    Tp = 1/Fp
    return {
        'hs':Hs,
        'tp':Tp
    }
