from typing import Literal

import numpy as np
from astropy import units as u
from astropy.coordinates import EarthLocation
from astropy.time import Time

from ._irbem import fly_in_nasa_aeap1

ntime_max = 100000
nene_max = 25
whichm_dict = {("e", "min"): 1, ("e", "max"): 2, ("p", "min"): 3, ("p", "max"): 4}
whatf_dict = {
    "differential": 1,
    "integral": 3,
}


def flux(
    location: EarthLocation,
    time: Time,
    energy: u.Quantity[u.physical.energy],
    *,
    kind: Literal["integral", "differential"],
    solar: Literal["min", "max"],
    particle: Literal["e", "p"],
) -> u.Quantity:
    """Calculate the flux in the radiation belt using the NASA AE8/AP8 model.

    Parameters
    ----------
    location
        Location at which to calculate the flux.
    time
        Time at which to calculate the flux.
    energy
        Energy at which to calculate the flux.
    kind
        Kind of flux: ``"integral"`` or ``"differential"``.
    solar
        Phase in the solar cycle: ``"min"`` for solar minimum or ``"max"`` for
        solar maximum.
    particle
        Particle species: ``"e"`` for electrons or ``"p"`` for protons.

    Returns
    -------
    :
        Estimated particle flux. If `location` or `time` are arrays, then
        this will also be an array with the same shape. The units are
        1 / (s cm2) for integral flux, or 1 / (MeV s cm2) for differential
        flux.
    """
    arg_arrays: list[np.ndarray] = [
        np.empty(ntime_max, dtype=np.int32),
        np.empty(ntime_max, dtype=np.int32),
        np.empty(ntime_max),
        np.empty(ntime_max),
        np.empty(ntime_max),
        np.empty(ntime_max),
    ]

    ene = np.empty((2, nene_max))
    ene[0, 0] = energy.to_value(u.MeV)
    x, y, z = u.Quantity(location.geocentric).to_value(u.earthRad)
    year, yday, seconds = (
        np.reshape(list(array), time.shape)
        for array in zip(
            *(
                (
                    datetime.year,
                    datetime.timetuple().tm_yday,
                    (
                        datetime
                        - datetime.replace(hour=0, minute=0, second=0, microsecond=0)
                    ).total_seconds(),
                )
                for datetime in np.atleast_1d(time.utc.datetime).ravel()
            )
        )
    )

    whichm = whichm_dict[(particle, solar)]
    whatf = whatf_dict[kind]

    with np.nditer(
        [year, yday, seconds, x, y, z, None],
        ["buffered", "external_loop"],
        [
            ["readonly"],
            ["readonly"],
            ["readonly"],
            ["readonly"],
            ["readonly"],
            ["readonly"],
            ["writeonly", "allocate"],
        ],
        buffersize=ntime_max,
    ) as it:
        for *args, out in it:
            ntime = len(out)
            for arg_array, arg in zip(arg_arrays, args):
                arg_array[:ntime] = arg
            out[:] = fly_in_nasa_aeap1(ntime, 1, whichm, whatf, 1, ene, *arg_arrays)[
                :ntime, 0
            ]

        out = it.operands[-1]

    out[out == -1e31] = np.nan
    out *= u.cm**-2 * u.s**-1
    if kind == "differential":
        out *= u.MeV**-1
    return out
