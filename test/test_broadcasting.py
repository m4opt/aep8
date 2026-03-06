import numpy as np
import pytest
from astropy import units as u
from astropy.coordinates import EarthLocation
from astropy.time import Time
from hypothesis import given, settings
from hypothesis.extra.numpy import (
    BroadcastableShapes,
    mutually_broadcastable_shapes,
)

import aep8


@pytest.mark.parametrize("particle", ["e", "p"])
@pytest.mark.parametrize("solar", ["min", "max"])
@pytest.mark.parametrize("kind", ["differential", "integral"])
@settings(deadline=None)
@given(
    shapes=mutually_broadcastable_shapes(num_shapes=3, max_side=50).filter(
        lambda shapes: np.prod(shapes.result_shape) < 2000
    )
)
def test_broadcasting(shapes: BroadcastableShapes, particle, solar, kind):
    location = EarthLocation.from_geocentric(
        *np.random.uniform(-10000, 10000, (3, *shapes.input_shapes[0])),
        unit=u.km,
    )
    time = Time("2020-01-01") + np.random.uniform(0, 1, shapes.input_shapes[1]) * u.year
    energy = np.random.uniform(0, 1, shapes.input_shapes[2]) * u.MeV
    model = aep8.model(solar=solar, particle=particle)
    method = getattr(model, f"{kind}_flux")
    geomag_method = getattr(model, f"{kind}_flux_for_geomagnetic_coordinates")
    with np.errstate(invalid="ignore"):
        result = method(location, time, energy)
        L, B = model.geomagnetic_coordinates(location, time)
        geomag_result = geomag_method(L, B, energy)
        with pytest.warns(DeprecationWarning, match="aep8.flux is deprecated"):
            legacy_result = aep8.flux(
                location, time, energy, kind=kind, solar=solar, particle=particle
            )
    assert result.shape == shapes.result_shape
    assert geomag_result.shape == shapes.result_shape
    assert legacy_result.shape == shapes.result_shape
    assert geomag_result.unit == result.unit
    assert legacy_result.unit == result.unit
    np.testing.assert_array_equal(geomag_result.value, result.value)
    np.testing.assert_array_equal(legacy_result.value, result.value)
