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


def _prepare_energy(energy):
    """Convert energy Quantity to (values_array, scalar_flag) tuple.

    Returns
    -------
    ene_values : numpy.ndarray
        1D array of energy values in MeV.
    scalar : bool
        True if the input was a scalar Quantity (ndim == 0).
    """
    scalar = energy.ndim == 0
    ene_values = np.atleast_1d(energy.to_value(u.MeV)).ravel()
    return ene_values, scalar


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
        Energy at which to calculate the flux. Can be a scalar or an array
        of energies; arrays are processed in batches of up to 25.
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
        this will also be an array with the same shape. If `energy` is an
        array of length *N*, the output gains a trailing dimension of size
        *N*: ``(*broadcast_shape, N)``. The units are 1 / (s cm2) for
        integral flux, or 1 / (MeV s cm2) for differential flux.
    """
    ene_values, scalar_energy = _prepare_energy(energy)
    n_energies = len(ene_values)

    arg_arrays: list[np.ndarray] = [
        np.empty(ntime_max, dtype=np.int32),
        np.empty(ntime_max, dtype=np.int32),
        np.empty(ntime_max),
        np.empty(ntime_max),
        np.empty(ntime_max),
        np.empty(ntime_max),
    ]

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

    all_results = []

    for ene_start in range(0, n_energies, nene_max):
        ene_batch = ene_values[ene_start : ene_start + nene_max]
        nene = len(ene_batch)

        ene = np.empty((2, nene_max))
        ene[0, :nene] = ene_batch

        with np.nditer(
            [year, yday, seconds, x, y, z] + [None] * nene,
            ["buffered", "external_loop"],
            [["readonly"]] * 6 + [["writeonly", "allocate"]] * nene,  # type: ignore[arg-type]
            buffersize=ntime_max,
        ) as it:
            for items in it:
                args = items[:6]
                outs = items[6:]
                ntime = len(outs[0])
                for arg_array, arg in zip(arg_arrays, args):
                    arg_array[:ntime] = arg
                result = fly_in_nasa_aeap1(
                    ntime, 1, whichm, whatf, nene, ene, *arg_arrays
                )
                for i, out_col in enumerate(outs):
                    out_col[:] = result[:ntime, i]

            batch_results = [it.operands[6 + i] for i in range(nene)]

        all_results.extend(batch_results)

    if scalar_energy:
        out = all_results[0]
    else:
        out = np.stack(all_results, axis=-1)

    out = np.maximum(0, out)
    out *= u.cm**-2 * u.s**-1
    if kind == "differential":
        out *= u.MeV**-1
    return out
