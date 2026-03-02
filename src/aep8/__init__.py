from importlib.metadata import version as _version

from ._core import flux, flux_from_magnetic_coordinates, magnetic_coordinates

__all__ = ("flux", "flux_from_magnetic_coordinates", "magnetic_coordinates")
__version__ = _version(__package__)
