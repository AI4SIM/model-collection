# 3D Correction

## Description

Weather and climate models operate at limited resolution that cannot resolve physical processes like clouds, radiation or turbulence for instance. Theses processes are important and are represented by parameterisation schemes that formulate the physical process by resolved atmospheric variables.

In the atmosphere, absorption, emission or scattering of short-wave and long-wave radiation are important processes. Short-wave radiative fluxes are the major source of energy of the Earth system. Long-wave radiative fluxes are emitted by the Earth surface and the atmosphere and is responsible for the greenhouse effect.

The solution of the radiative transfer equations to obtain the fluxes are computationnaly expensive and, in practice, calculations are performed on a coarser resolution and/or an increased timestep (lower time frequency). An interpolation is then performed to go back to the original grid. The Integrated Forecasting System (IFS), the operational numerical weather prediction model developed by ECMWF, comes with the ecRad scheme (https://github.com/ecmwf-ifs/ecrad) that implements several solvers. Two of them represent the horizontal cloud inhomogeneity: Tripleclouds (Shonk and Hogan, 2008) and SPARTACUS ( Speedy Algorithm for Radiative Transfer through Cloud Sides -- Hogan et al., 2016). SPARTACUS is an extension of Tripleclouds by treating the 3D radiative effects associated with clouds but is currently too expensive for operational weather predictions.

The task of the present use-case is to learn the 3D cloud radiative effects as a corrective term to the Tripleclouds formulation. 

## Dataset

Download the dataset using the Climetlab library developed by ECMWF. An example on how to retrieve the data is largely explained [here](https://git.ecmwf.int/projects/MLFET/repos/maelstrom-radiation/browse).

## Models

X:

- ??


Y:

- ??


One approaches is considered:

1. An 1D Unet ...
