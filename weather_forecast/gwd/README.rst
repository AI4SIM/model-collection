Gravity Wave Drag UC
===============

Description
-----------------
Climate models are complex objects developed over decades and have many components. While the main model renders dynamics (Temperature, Wind speeds, Pressure), models of lesser magnitude substantially refine these predictions by emulating physical processes: radiation, precipitation, evaporation, clouds, as well as Gravity Wave Drag. It caracterizes a phenomenon of vertical propagation of the relief which affects the wind speed over all layers of the atmosphere.

The objective is to substitute the parametrization scheme used in production by a DL model much more efficient at a close-enough accuracy. This experiment is largely inspired from the `work of Chantry et al (2021) <https://agupubs.onlinelibrary.wiley.com/doi/pdfdirect/10.1029/2021MS002477>`_.

Dataset
-----------------
Download the dataset using the Climetlab library developed by ECMWF. An example on how to retrieve the data is largely explained `here <https://git.ecmwf.int/projects/MLFET/repos/maelstrom-nogwd/browse>`_.

Models
-----------------
X:

* ``u``, ``v`` the horizontal wind velocities
* ``T`` the temperature
* ``p`` the pressure
* ``g`` the geopotential
* 4 surface params that describe the subscale orography

Y:

* ``u_t``, ``v_t`` as the gravity-wave drag tendencies
* ``b_f_t`` as the blocking drag tendencies

Two approaches were considered:

1. An MLP based on the work from Chantry et al., with the DoF
2. A 1D-CNN that swipes from lower to upper layer of the atmosphere
