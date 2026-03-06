from importlib.metadata import version as _version

from ._core import Model, flux, model

__all__ = ("flux", "model", "Model")
__version__ = _version(__package__)
