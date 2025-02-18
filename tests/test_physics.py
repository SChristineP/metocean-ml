import pytest
import numpy as np
import pandas as pd
from metocean_ml.physics import dirmag_to_uv, uv_to_dirmag, direct_fetch, effective_fetch, fetch_laws, fetch_law_Holthuijsen, fetch_law_KahmaCalkoen, fetch_law_JONSWAP

# Test for dirmag_to_uv
def test_dirmag_to_uv():
    wind_direction = np.array([0, 90, 180, 270])
    wind_speed = np.array([10, 10, 10, 10])
    u, v = dirmag_to_uv(wind_direction, wind_speed)
    
    assert np.allclose(u, [0, 10, 0, -10])
    assert np.allclose(v, [10, 0, -10, 0])
    
    # Test for "from" wind direction
    u, v = dirmag_to_uv(wind_direction, wind_speed, going_to=False)
    assert np.allclose(u, [0, -10, 0, 10])
    assert np.allclose(v, [-10, 0, 10, 0])

# Test for uv_to_dirmag
def test_uv_to_dirmag():
    u = np.array([0, 10, 0, -10])
    v = np.array([10, 0, -10, 0])
    direction, magnitude = uv_to_dirmag(u, v)
    
    assert np.allclose(direction, [0, 90, 180, 270])
    assert np.allclose(magnitude, [10, 10, 10, 10])
    
    # Test for "from" wind direction
    direction, magnitude = uv_to_dirmag(u, v, going_to=False)
    assert np.allclose(direction, [180, 270, 0, 90])

# Test for direct_fetch
def test_direct_fetch():
    fetch = pd.Series([100, 200, 300], index=[0, 90, 180])
    wind_direction = np.array([0, 90, 180])
    
    result = direct_fetch(fetch, wind_direction)
    
    assert np.allclose(result, [100, 200, 300])

# Test for effective_fetch
def test_effective_fetch():
    fetch = pd.Series(np.ones(shape=120), index=np.linspace(0,360,120,endpoint=False))
    wind_direction = np.array(fetch.index)
    
    result = effective_fetch(fetch, wind_direction,sector=20)

    assert np.allclose(result, fetch.values, atol=1e-2)

# Test for fetch_laws
def test_fetch_laws():
    wind_speed = np.array([10, 15, 20])
    fetch = np.array([1000, 2000, 3000])
    
    # Test Holthuijsen
    result = fetch_laws(wind_speed, fetch, laws='holthuijsen')
    assert 'hs' in result
    assert 'tp' in result
    
    # Test KahmaCalkoen
    result = fetch_laws(wind_speed, fetch, laws='kahmacalkoen')
    assert 'hs' in result
    assert 'tp' in result
    
    # Test JONSWAP
    result = fetch_laws(wind_speed, fetch, laws='jonswap')
    assert 'hs' in result
    assert 'tp' in result
    
    # Test invalid law
    with pytest.raises(ValueError):
        fetch_laws(wind_speed, fetch, laws='invalid_law')

# Test for fetch_law_Holthuijsen
def test_fetch_law_Holthuijsen():
    wind = np.array([10, 15, 20])
    fetch = np.array([1000, 2000, 3000])
    depth = 50
    
    result = fetch_law_Holthuijsen(wind, fetch, depth)
    
    assert 'hs' in result
    assert 'tp' in result
    assert np.all(result['hs'] > 0)  # Significant wave height should be positive
    assert np.all(result['tp'] > 0)  # Wave period should be positive

# Test for fetch_law_KahmaCalkoen
def test_fetch_law_KahmaCalkoen():
    wind_speed = np.array([10, 15, 20])
    fetch = np.array([1000, 2000, 3000])
    
    result = fetch_law_KahmaCalkoen(wind_speed, fetch)
    
    assert 'hs' in result
    assert 'tp' in result
    assert np.all(result['hs'] > 0)  # Significant wave height should be positive
    assert np.all(result['tp'] > 0)  # Wave period should be positive

# Test for fetch_law_JONSWAP
def test_fetch_law_JONSWAP():
    wind_speed = np.array([10, 15, 20])
    fetch = np.array([1000, 2000, 3000])
    
    result = fetch_law_JONSWAP(wind_speed, fetch)
    
    assert 'hs' in result
    assert 'tp' in result
    assert np.all(result['hs'] > 0)  # Significant wave height should be positive
    assert np.all(result['tp'] > 0)  # Wave period should be positive
