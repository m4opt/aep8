from importlib.metadata import version as _version

from ._core import flux

__all__ = ("flux",)
__version__ = _version(__package__)
