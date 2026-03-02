import numpy as np
import pytest
from astropy import units as u
from astropy.coordinates import EarthLocation
from astropy.time import Time

import aep8

# All 4 model combinations and their required dipole model
MODEL_PARAMS = [
    ("e", "min", "JensenANDCain1960"),
    ("e", "max", "JensenANDCain1960"),
    ("p", "min", "JensenANDCain1960"),
    ("p", "max", "GSFC1266"),
]


@pytest.fixture
def sample_location():
    lon = np.linspace(-180, 180, 10) * u.deg
    lat = np.linspace(-60, 60, 10) * u.deg
    height = 500 * u.km
    return EarthLocation.from_geodetic(*np.meshgrid(lon, lat), height)


@pytest.fixture
def sample_time():
    return Time("2025-01-01")


@pytest.mark.parametrize("particle,solar,dipole", MODEL_PARAMS)
@pytest.mark.parametrize("kind", ["differential", "integral"])
def test_roundtrip(sample_location, sample_time, particle, solar, dipole, kind):
    """Verify magnetic_coordinates + flux_from_magnetic_coordinates matches flux()."""
    energy = 1 * u.MeV

    # Reference: single-call flux
    expected = aep8.flux(
        sample_location,
        sample_time,
        energy,
        kind=kind,
        solar=solar,
        particle=particle,
    )

    # Split: compute L/BBo, then look up flux
    L, BBo = aep8.magnetic_coordinates(
        sample_location,
        sample_time,
        dipole=dipole,
    )
    actual = aep8.flux_from_magnetic_coordinates(
        L,
        BBo,
        energy,
        kind=kind,
        solar=solar,
        particle=particle,
    )

    np.testing.assert_allclose(
        actual.value,
        expected.value,
        rtol=1e-10,
        err_msg=f"Roundtrip mismatch for {particle}/{solar}/{kind}",
    )
    assert actual.unit == expected.unit


def test_broadcasting(sample_location, sample_time):
    """Verify magnetic_coordinates handles broadcast shapes correctly."""
    L, BBo = aep8.magnetic_coordinates(
        sample_location,
        sample_time,
        dipole="JensenANDCain1960",
    )
    assert L.shape == sample_location.shape
    assert BBo.shape == sample_location.shape


def test_reuse_across_energies(sample_location, sample_time):
    """Demonstrate computing L/BBo once and reusing for multiple energies."""
    L, BBo = aep8.magnetic_coordinates(
        sample_location,
        sample_time,
        dipole="JensenANDCain1960",
    )

    for energy_val in [0.1, 1.0, 10.0]:
        energy = energy_val * u.MeV
        result = aep8.flux_from_magnetic_coordinates(
            L,
            BBo,
            energy,
            kind="integral",
            solar="min",
            particle="e",
        )
        assert result.shape == L.shape
        assert result.unit == u.cm**-2 * u.s**-1


def test_reuse_across_models(sample_location, sample_time):
    """Demonstrate computing L/BBo once and reusing for AE8min/AE8max/AP8min."""
    # JensenANDCain1960 is shared by whichm 1, 2, 3
    L, BBo = aep8.magnetic_coordinates(
        sample_location,
        sample_time,
        dipole="JensenANDCain1960",
    )
    energy = 1 * u.MeV

    for particle, solar in [("e", "min"), ("e", "max"), ("p", "min")]:
        expected = aep8.flux(
            sample_location,
            sample_time,
            energy,
            kind="integral",
            solar=solar,
            particle=particle,
        )
        actual = aep8.flux_from_magnetic_coordinates(
            L,
            BBo,
            energy,
            kind="integral",
            solar=solar,
            particle=particle,
        )
        np.testing.assert_allclose(
            actual.value,
            expected.value,
            rtol=1e-10,
            err_msg=f"Reuse mismatch for {particle}/{solar}",
        )


def test_scalar_location():
    """Verify that scalar (single-point) inputs work correctly."""
    loc = EarthLocation.from_geodetic(15 * u.deg, -45 * u.deg, 500 * u.km)
    time = Time("2025-01-01")
    energy = 1 * u.MeV

    # AP8max (particle="p", solar="max") requires GSFC1266 dipole
    L, BBo = aep8.magnetic_coordinates(loc, time, dipole="GSFC1266")
    result = aep8.flux_from_magnetic_coordinates(
        L,
        BBo,
        energy,
        kind="integral",
        solar="max",
        particle="p",
    )
    expected = aep8.flux(
        loc,
        time,
        energy,
        kind="integral",
        solar="max",
        particle="p",
    )

    np.testing.assert_allclose(result.value, expected.value, rtol=1e-10)


# --- Vectorized energy tests ---


@pytest.fixture
def simple_location():
    lon = np.linspace(-180, 180, 5) * u.deg
    lat = np.linspace(-60, 60, 5) * u.deg
    height = 500 * u.km
    return EarthLocation.from_geodetic(*np.meshgrid(lon, lat), height)


@pytest.fixture
def simple_time():
    return Time("2025-01-01")


def test_vectorized_energy_roundtrip(simple_location, simple_time):
    """Passing an array of energies matches individual scalar calls."""
    energies_MeV = [0.5, 1.0, 2.0, 5.0]
    energy_array = energies_MeV * u.MeV

    result = aep8.flux(
        simple_location,
        simple_time,
        energy_array,
        kind="integral",
        solar="min",
        particle="e",
    )

    assert result.shape == (*simple_location.shape, len(energies_MeV))

    for i, e in enumerate(energies_MeV):
        scalar_result = aep8.flux(
            simple_location,
            simple_time,
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


def test_vectorized_energy_flux_from_mag_coords(simple_location, simple_time):
    """Vectorized energy in flux_from_magnetic_coordinates matches scalar calls."""
    energies_MeV = [0.5, 1.0, 2.0, 5.0]
    energy_array = energies_MeV * u.MeV

    L, BBo = aep8.magnetic_coordinates(
        simple_location,
        simple_time,
        dipole="JensenANDCain1960",
    )

    result = aep8.flux_from_magnetic_coordinates(
        L,
        BBo,
        energy_array,
        kind="integral",
        solar="min",
        particle="e",
    )

    assert result.shape == (*L.shape, len(energies_MeV))

    for i, e in enumerate(energies_MeV):
        scalar_result = aep8.flux_from_magnetic_coordinates(
            L,
            BBo,
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


def test_scalar_energy_backward_compat(simple_location, simple_time):
    """Scalar energy still returns shape without trailing energy dimension."""
    energy = 1 * u.MeV

    result = aep8.flux(
        simple_location,
        simple_time,
        energy,
        kind="integral",
        solar="min",
        particle="e",
    )
    assert result.shape == simple_location.shape

    L, BBo = aep8.magnetic_coordinates(
        simple_location,
        simple_time,
        dipole="JensenANDCain1960",
    )
    result2 = aep8.flux_from_magnetic_coordinates(
        L,
        BBo,
        energy,
        kind="integral",
        solar="min",
        particle="e",
    )
    assert result2.shape == L.shape


def test_energy_chunking(simple_location, simple_time):
    """More than 25 energies exercises the chunking path."""
    energies = np.linspace(0.1, 5.0, 30) * u.MeV

    result = aep8.flux(
        simple_location,
        simple_time,
        energies,
        kind="integral",
        solar="min",
        particle="e",
    )

    assert result.shape == (*simple_location.shape, 30)

    # Verify a few spot checks against scalar calls
    for idx in [0, 14, 29]:
        scalar_result = aep8.flux(
            simple_location,
            simple_time,
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
