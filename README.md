# AEP8

![Map of integral min electron flux at 500 km and 1 MeV](https://github.com/m4opt/aep8/raw/main/test/baseline/test_plot_flux_integral-min-e.png)

This Python package calculates the estimated flux of electrons or protons trapped in the Earth's radiation belt. It is a Python wrapper for the [NASA AE8/AP8 model](https://prbem.github.io/IRBEM/api/radiation_models.html#ae8-and-ap8-models) in the [IRBEM](https://prbem.github.io/IRBEM/) package. It provides an [Astropy](https://www.astropy.org)-friendly interface, allowing you to specify location using [Astropy coordinates](https://docs.astropy.org/en/stable/coordinates/index.html), the time in [Astropy time](https://docs.astropy.org/en/stable/time/index.html), and the energy using [Astropy units](https://docs.astropy.org/en/stable/units/index.html). You can pass it a single time and location, or arrays of times and locations.

## Example

```pycon
>>> from astropy.coordinates import EarthLocation
>>> from astropy.time import Time
>>> from astropy import units as u
>>> from aep8 import flux
>>> loc = EarthLocation.from_geodetic(15 * u.deg, -45 * u.deg, 300 * u.km)
>>> time = Time('2025-01-01 18:37:22')
>>> energy = 10 * u.MeV
>>> flux(loc, time, energy, kind='integral', solar='max', particle='p')
<Quantity 3.04495297 1 / (s cm2)>
```
