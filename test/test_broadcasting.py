import numpy as np
from astropy import units as u
from astropy.coordinates import EarthLocation
from astropy.time import Time
from hypothesis import given, settings
from hypothesis.extra.numpy import (
    BroadcastableShapes,
    mutually_broadcastable_shapes,
)

from aep8 import flux


@settings(deadline=None)
@given(
    mutually_broadcastable_shapes(num_shapes=2, max_side=50).filter(
        lambda shapes: np.prod(shapes.result_shape) < 2000
    )
)
def test_broadcasting(shapes: BroadcastableShapes):
    location = EarthLocation.from_geocentric(
        *np.random.uniform(-10000, 10000, (3, *shapes.input_shapes[0])),
        unit=u.km,
    )
    time = Time("2020-01-01") + np.random.uniform(0, 1, shapes.input_shapes[1]) * u.year
    result = flux(
        location, time, 10 * u.MeV, kind="integral", solar="max", particle="p"
    )
    assert result.shape == shapes.result_shape
