from dataclasses import dataclass
from typing import Literal, TypeAlias
from warnings import warn

import numpy as np
from astropy import units as u
from astropy.coordinates import EarthLocation
from astropy.time import Time
from numpy import typing as npt

from ._irbem import flux1, flux2, flux3, flux4, geomag3, geomag4

Particle: TypeAlias = Literal["e", "p"]
Solar: TypeAlias = Literal["min", "max"]


@dataclass
class Model:
    """Trapped particle flux model for a given particle species and solar cycle phase.

    Notes
    -----
    Do not instantiate this class directly. Call the function
    :meth:`aep8.model` to return an existing model.
    """

    particle: Particle
    """Particle species, ``"e"`` for electrons or ``"p"`` for protons."""

    solar: Solar
    """Phase in the solar cycle, ``"min"`` for solar minimum or ``"max"`` for
    solar maximum."""

    _geomag: np.ufunc
    _flux: np.ufunc

    def geomagnetic_coordinates(
        self,
        location: EarthLocation,
        time: Time,
    ) -> tuple[npt.ArrayLike, npt.ArrayLike]:
        """Calculate the geomagnetic coordinates for a given time and location.

        Parameters
        ----------
        location
            Location at which to calculate the geomagnetic field.
        time
            Time at which to calculate the geomagnetic field.

        Returns
        -------
        L
            McIlwain L.
        BBo
            Local magnetic field divided by magnetic field at the equator.
        """
        x, y, z = u.Quantity(location.geocentric).to_value(u.earthRad)
        t = np.rint(time.utc.unix).astype(int)
        B, L = self._geomag(t, x, y, z)
        return L, B

    def integral_flux(
        self,
        location: EarthLocation,
        time: Time,
        energy: u.Quantity[u.physical.energy],
    ) -> u.Quantity[u.cm**-2 * u.s**-1]:
        """Calculate the integrated flux for a given time and location.

        Parameters
        ----------
        location
            Location at which to calculate the geomagnetic field.
        time
            Time at which to calculate the geomagnetic field.
        energy
            Energy per particle.

        Returns
        -------
        :
            Integrated particle flux up to the given energy.
        """
        L, B = self.geomagnetic_coordinates(location, time)
        return self.integral_flux_for_geomagnetic_coordinates(L, B, energy)

    def integral_flux_for_geomagnetic_coordinates(
        self,
        L: npt.ArrayLike,
        B: npt.ArrayLike,
        energy: u.Quantity[u.physical.energy],
    ) -> u.Quantity[u.cm**-2 * u.s**-1]:
        """Calculate the integrated flux for given geomagnetic coordinates.

        Parameters
        ----------
        L
            See :meth:`Model.geomagnetic_coordinates`.
        B
            See :meth:`Model.geomagnetic_coordinates`.
        energy
            Energy per particle.

        Returns
        -------
        :
            Integrated particle flux up to the given energy.
        """
        return self._flux(energy.to_value(u.MeV), L, B) * u.cm**-2 * u.s**-1

    def differential_flux(
        self,
        location: EarthLocation,
        time: Time,
        energy: u.Quantity[u.physical.energy],
    ) -> u.Quantity[u.cm**-2 * u.s**-1 * u.MeV**-1]:
        """Calculate the differential flux for a given time and location.

        Parameters
        ----------
        location
            Location at which to calculate the geomagnetic field.
        time
            Time at which to calculate the geomagnetic field.
        energy
            Energy per particle.

        Returns
        -------
        :
            Differential particle flux.

        Notes
        -----
        This is a lightweight wrapper around
        :meth:`Model.integral_flux` that performs
        first-order finite differences.
        """
        L, B = self.geomagnetic_coordinates(location, time)
        return self.differential_flux_for_geomagnetic_coordinates(L, B, energy)

    def differential_flux_for_geomagnetic_coordinates(
        self,
        L: npt.ArrayLike,
        B: npt.ArrayLike,
        energy: u.Quantity[u.physical.energy],
    ) -> u.Quantity[u.cm**-2 * u.s**-1 * u.MeV**-1]:
        """Calculate the differential flux for given geomagnetic coordinates.

        Parameters
        ----------
        L
            See :meth:`Model.geomagnetic_coordinates`.
        B
            See :meth:`Model.geomagnetic_coordinates`.
        energy
            Energy per particle.

        Returns
        -------
        :
            Differential particle flux.

        Notes
        -----
        This is a lightweight wrapper around
        :meth:`Model.integral_flux_for_geomagnetic_coordinates` that performs
        first-order finite differences.
        """
        dE_MeV = 0.001
        dE = [-dE_MeV, dE_MeV] * u.MeV
        flux = self.integral_flux_for_geomagnetic_coordinates(
            np.expand_dims(L, -1),
            np.expand_dims(B, -1),
            np.expand_dims(energy, -1) + dE,
        )
        return (flux[..., 0] - flux[..., 1]) / (2 * dE_MeV * u.MeV)


models = {
    (particle, solar): Model(particle, solar, geomag, flux)
    for particle, solar, geomag, flux in [
        ["e", "min", geomag3, flux1],
        ["e", "max", geomag3, flux2],
        ["p", "min", geomag3, flux3],
        ["p", "max", geomag4, flux4],
    ]
}


def model(particle: Particle, solar: Solar) -> Model:
    """Get a trapped particle flux model.

    Parameters
    ----------
    particle
        Particle species: ``"e"`` for electrons or ``"p"`` for protons.
    solar
        Phase in the solar cycle: ``"min"`` for solar minimum or ``"max"`` for
        solar maximum.

    Returns
    -------
    :
        Trapped particle flux model that can be used to evaluate the integral
        or differential flux.
    """
    return models[(particle, solar)]


def flux(
    location: EarthLocation,
    time: Time,
    energy: u.Quantity[u.physical.energy],
    *,
    kind: Literal["integral", "differential"],
    solar: Solar,
    particle: Particle,
) -> u.Quantity:
    """Calculate the flux in the radiation belt using the NASA AE8/AP8 model.

    .. deprecated:: 1.0.0
        This method will be deprecated in aep8 v2.0.0.
        Please use :meth:`aep8.model` instead.

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
        Estimated particle flux. If ``location`` or ``time`` are arrays, then
        this will also be an array with the same shape. The units are
        1 / (s cm2) for integral flux, or 1 / (MeV s cm2) for differential
        flux.
    """
    warn("aep8.flux is deprecated. Please use aep8.model instead.", DeprecationWarning)
    m = model(particle, solar)
    if kind == "integral":
        return m.integral_flux(location, time, energy)
    else:
        return m.differential_flux(location, time, energy)
