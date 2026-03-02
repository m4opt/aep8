from typing import Literal

import numpy as np
from astropy import units as u
from astropy.coordinates import EarthLocation
from astropy.time import Time

from ._irbem import compute_lbbo, fly_in_nasa_aeap1, get_ae8_ap8_flux

ntime_max = 100000
nene_max = 25
whichm_dict = {("e", "min"): 1, ("e", "max"): 2, ("p", "min"): 3, ("p", "max"): 4}
whatf_dict = {
    "differential": 1,
    "integral": 3,
}
kint_dict = {
    "JensenANDCain1960": 2,
    "GSFC1266": 3,
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


def _extract_geocentric(location: EarthLocation):
    """Extract geocentric (x, y, z) coordinates in Earth radii."""
    return u.Quantity(location.geocentric).to_value(u.earthRad)


def _extract_time_components(time: Time):
    """Extract (year, yday, seconds) arrays from an astropy Time."""
    return (
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

    x, y, z = _extract_geocentric(location)
    year, yday, seconds = _extract_time_components(time)

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


def magnetic_coordinates(
    location: EarthLocation,
    time: Time,
    *,
    dipole: Literal["JensenANDCain1960", "GSFC1266"] = "JensenANDCain1960",
) -> tuple[np.ndarray, np.ndarray]:
    """Compute McIlwain L and B/B0 for given locations and times.

    This performs the expensive field-line tracing step of the AE8/AP8
    model.  The results can be reused across multiple calls to
    :func:`flux_from_magnetic_coordinates` for different energies,
    particles, or flux kinds, avoiding redundant computation.

    Parameters
    ----------
    location
        Location at which to compute magnetic coordinates.
    time
        Time at which to compute magnetic coordinates.
    dipole
        Dipole magnetic field model.  Use ``"JensenANDCain1960"``
        (the default) for AE8min, AE8max, and AP8min (whichm 1, 2, 3).
        Use ``"GSFC1266"`` for AP8max (whichm 4).

    Returns
    -------
    L : numpy.ndarray
        McIlwain L parameter.
    BBo : numpy.ndarray
        B/B0 ratio (local field strength over equatorial field strength,
        using the McIlwain dipole approximation).
    """
    kint = kint_dict[dipole]

    arg_arrays: list[np.ndarray] = [
        np.empty(ntime_max, dtype=np.int32),
        np.empty(ntime_max, dtype=np.int32),
        np.empty(ntime_max),
        np.empty(ntime_max),
        np.empty(ntime_max),
        np.empty(ntime_max),
    ]

    x, y, z = _extract_geocentric(location)
    year, yday, seconds = _extract_time_components(time)

    lm_buf = np.empty(ntime_max)
    bbo_buf = np.empty(ntime_max)

    with np.nditer(
        [year, yday, seconds, x, y, z, None, None],
        ["buffered", "external_loop"],
        [
            ["readonly"],
            ["readonly"],
            ["readonly"],
            ["readonly"],
            ["readonly"],
            ["readonly"],
            ["writeonly", "allocate"],
            ["writeonly", "allocate"],
        ],
        buffersize=ntime_max,
    ) as it:
        for *args, out_lm, out_bbo in it:
            ntime = len(out_lm)
            for arg_array, arg in zip(arg_arrays, args):
                arg_array[:ntime] = arg
            lm_out, bbo_out = compute_lbbo(ntime, 1, kint, *arg_arrays)
            out_lm[:] = lm_out[:ntime]
            out_bbo[:] = bbo_out[:ntime]

        L = it.operands[-2]
        BBo = it.operands[-1]

    return L, BBo


def flux_from_magnetic_coordinates(
    L: np.ndarray,
    BBo: np.ndarray,
    energy: u.Quantity[u.physical.energy],
    *,
    kind: Literal["integral", "differential"],
    solar: Literal["min", "max"],
    particle: Literal["e", "p"],
) -> u.Quantity:
    """Look up AE8/AP8 flux from precomputed magnetic coordinates.

    This performs the cheap table-interpolation step of the AE8/AP8
    model.  Use :func:`magnetic_coordinates` to compute the ``L`` and
    ``BBo`` inputs.

    Parameters
    ----------
    L
        McIlwain L parameter (from :func:`magnetic_coordinates`).
    BBo
        B/B0 ratio (from :func:`magnetic_coordinates`).
    energy
        Energy at which to calculate the flux. Can be a scalar or an
        array of energies; arrays are processed in batches of up to 25.
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
        Estimated particle flux, same shape as ``L`` and ``BBo``.
        If `energy` is an array of length *N*, the output gains a
        trailing dimension of size *N*: ``(*shape, N)``.
        Units are 1 / (s cm2) for integral flux, or 1 / (MeV s cm2)
        for differential flux.
    """
    ene_values, scalar_energy = _prepare_energy(energy)
    n_energies = len(ene_values)

    whichm = whichm_dict[(particle, solar)]
    whatf = whatf_dict[kind]

    L = np.asarray(L)
    BBo = np.asarray(BBo)

    l_buf = np.empty(ntime_max)
    bbo_buf = np.empty(ntime_max)

    all_results = []

    for ene_start in range(0, n_energies, nene_max):
        ene_batch = ene_values[ene_start : ene_start + nene_max]
        nene = len(ene_batch)

        ene = np.empty((2, nene_max))
        ene[0, :nene] = ene_batch

        with np.nditer(
            [L, BBo] + [None] * nene,
            ["buffered", "external_loop"],
            [["readonly"], ["readonly"]] + [["writeonly", "allocate"]] * nene,  # type: ignore[arg-type]
            buffersize=ntime_max,
        ) as it:
            for items in it:
                l_chunk, bbo_chunk = items[0], items[1]
                outs = items[2:]
                n = len(outs[0])
                l_buf[:n] = l_chunk
                bbo_buf[:n] = bbo_chunk
                result = get_ae8_ap8_flux(n, whichm, whatf, nene, ene, bbo_buf, l_buf)
                for i, out_col in enumerate(outs):
                    out_col[:] = result[:n, i]

            batch_results = [it.operands[2 + i] for i in range(nene)]

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
