import numpy as np
import pytest
from astropy import units as u
from astropy.coordinates import EarthLocation
from astropy.time import Time

import aep8


@pytest.fixture
def sample_location():
    lon = np.linspace(-180, 180, 5) * u.deg
    lat = np.linspace(-60, 60, 5) * u.deg
    height = 500 * u.km
    return EarthLocation.from_geodetic(*np.meshgrid(lon, lat), height)


@pytest.fixture
def sample_time():
    return Time("2025-01-01")


def test_vectorized_energy_roundtrip(sample_location, sample_time):
    """Passing an array of energies matches individual scalar calls."""
    energies_MeV = [0.5, 1.0, 2.0, 5.0]
    energy_array = energies_MeV * u.MeV

    result = aep8.flux(
        sample_location,
        sample_time,
        energy_array,
        kind="integral",
        solar="min",
        particle="e",
    )

    assert result.shape == (*sample_location.shape, len(energies_MeV))

    for i, e in enumerate(energies_MeV):
        scalar_result = aep8.flux(
            sample_location,
            sample_time,
            e * u.MeV,
            kind="integral",
            solar="min",
            particle="e",
        )
        np.testing.assert_allclose(
            result[..., i].value,
            scalar_result.value,
            rtol=1e-10,
            err_msg=f"Mismatch at energy={e} MeV",
        )


def test_scalar_energy_backward_compat(sample_location, sample_time):
    """Scalar energy still returns shape without trailing energy dimension."""
    energy = 1 * u.MeV

    result = aep8.flux(
        sample_location,
        sample_time,
        energy,
        kind="integral",
        solar="min",
        particle="e",
    )
    assert result.shape == sample_location.shape


def test_energy_chunking(sample_location, sample_time):
    """More than 25 energies exercises the chunking path."""
    energies = np.linspace(0.1, 5.0, 30) * u.MeV

    result = aep8.flux(
        sample_location,
        sample_time,
        energies,
        kind="integral",
        solar="min",
        particle="e",
    )

    assert result.shape == (*sample_location.shape, 30)

    # Verify a few spot checks against scalar calls
    for idx in [0, 14, 29]:
        scalar_result = aep8.flux(
            sample_location,
            sample_time,
            energies[idx],
            kind="integral",
            solar="min",
            particle="e",
        )
        np.testing.assert_allclose(
            result[..., idx].value,
            scalar_result.value,
            rtol=1e-10,
            err_msg=f"Chunking mismatch at index {idx}",
        )
