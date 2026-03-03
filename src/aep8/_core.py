from typing import Literal

import numpy as np
from astropy import units as u
from astropy.coordinates import EarthLocation
from astropy.time import Time

from ._irbem import flux1, flux2, flux3, flux4, geomag3, geomag4

whichm_dict = {
    ("e", "min"): (geomag3, flux1),
    ("e", "max"): (geomag3, flux2),
    ("p", "min"): (geomag3, flux3),
    ("p", "max"): (geomag4, flux4),
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
    geomag_func, flux_func = whichm_dict[(particle, solar)]
    x, y, z = u.Quantity(location.geocentric).to_value(u.earthRad)
    t = np.rint(time.utc.unix).astype(int)
    E = energy.to_value(u.MeV)
    B, L = geomag_func(t, x, y, z)

    if kind == "integral":
        F = flux_func(E, L, B) * u.cm**-2 * u.s**-1
    elif kind == "differential":
        dE = 0.001
        F = (
            (flux_func(E - dE, L, B) - flux_func(E + dE, L, B))
            / (2 * dE)
            * u.cm**-2
            * u.s**-1
            * u.MeV**-1
        )
    else:
        raise NotImplementedError("This line must not be reached")

    return F
