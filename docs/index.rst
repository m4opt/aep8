aep8 documentation
==================

.. figure:: _static/test_plot_flux_integral-min-e.png
    :alt: Map of integral min electron flux at 500 km and 1 MeV

This Python package calculates the estimated flux of electrons or protons trapped in the Earth's radiation belt. It is a Python wrapper for the :doc:`NASA AE8/AP8 model <irbem:api/radiation_models>` in the :doc:`IRBEM <irbem:index>` package. It provides an `Astropy <https://www.astropy.org>`_-friendly interface, allowing you to specify the location using :ref:`Astropy coordinates <astropy-coordinates>`, the time in :ref:`Astropy time <astropy-time>`, and the energy using :ref:`Astropy units <astropy-units>`. You can pass it a single time and location, or arrays of times and locations.

*********
Reference
*********

.. automodapi:: aep8
    :no-inheritance-diagram:
