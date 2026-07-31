import numpy as np
import pytest
from astropy import units as u
from astropy.coordinates import EarthLocation
from astropy.time import Time

import aep8

# Geostationary orbit, where the electron spectrum falls below the floor of the
# AE8 table well within the model's energy range of 0.05-7 MeV.
location = EarthLocation.from_geodetic(
    lon=15 * u.deg, lat=0 * u.deg, height=35786 * u.km
)
time = Time("2025-05-18T02:48:00Z")


@pytest.mark.parametrize("solar", ["min", "max"])
def test_flux_below_table_floor_is_zero(solar):
    """Below the floor of the table, the flux is zero, not 1.

    TRARA1 clamps the log flux at zero, so a naive exp10 bottoms out at 1
    particle / (s cm2) rather than at zero.
    """
    energy = np.linspace(0.05, 7, 1000) * u.MeV
    flux = aep8.model(particle="e", solar=solar).integral_flux(location, time, energy)
    assert not np.any(flux.value == 1)
    assert np.any(flux.value == 0)


@pytest.mark.parametrize("solar", ["min", "max"])
def test_flux_decreases_to_zero(solar):
    """Integral flux is non-increasing with energy and reaches zero."""
    energy = np.linspace(0.05, 7, 1000) * u.MeV
    flux = aep8.model(particle="e", solar=solar).integral_flux(location, time, energy)
    assert np.all(np.diff(flux.value) <= 0)
    assert flux.value[0] > 0
    assert flux.value[-1] == 0
