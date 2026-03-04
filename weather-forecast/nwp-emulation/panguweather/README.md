# PanguWeather

## Description

This repository contains an implementation of the **PanguWeather** model. PanguWeather is a deep learning-based weather forecasting model that leverages high-resolution, gridded datasets to produce fast, accurate global forecasts.

This work builds on the research of Bi et al. (2023) and others who have pioneered the use of deep learning architectures for global weather forecasting. We have taken their approach and adapted it for our specific research goals.

## Dataset

We rely on ERA5, the state-of-the-art global reanalysis dataset produced by the European Centre for Medium-Range Weather Forecasts (ECMWF). ERA5 provides:

- **Temporal coverage:** Hourly data from 1979 to present.
- **Spatial resolution:** Approximately 31 km on a reduced Gaussian grid.
- **Variables:** A wide range of atmospheric, land, and oceanic variables, including temperature, pressure, wind, precipitation, and radiation fluxes.


## AI4Sim vs other implementations
### Attention mask
Other implementations implement a "limited area" version of the attention mask, considering borders along all the directions (pressure levels, latitudes and longitudes), following the Swin Transformers architecture.
AI4Sim implements a more versatile version of the mask, including the limited area one and the vanilla PanguWeather one considering the continuity along the longitudes.

### Surface constant masks
AI4Sim uses more recent surface masks (from the ERA5 files). The main differences are:
* Land-sea: continuous values between 0 and 1 for our implementation vs binary for older versions.
* Soil type: different representation of soil types 4, 5 and 6 (total of 7 types).

## MLFlow logging
In the `configs` folder, the `config_pangu_lite_mlflow.yaml` shows how to add MLFlowLogger to the lightning config file. 
If the mlflow experience is logged on a server is strongly recommended to define the following shell variable before 
running the training:
```
export MLFLOW_TRACKING_URI=<server url address>
export MLFLOW_TRACKING_USERNAME=<username>
export MLFLOW_TRACKING_PASSWORD=<password>`
```
## References

- Bi, K., et al. (2023). *Pangu-Weather: A 3D High-Resolution Model for Fast and Accurate Global Weather Forecast.* arXiv:2211.02556.
  
- Hersbach, H., et al. (2020). *The ERA5 global reanalysis.* Quarterly Journal of the Royal Meteorological Society, 146(730), 1999-2049. [doi:10.1002/qj.3803](https://doi.org/10.1002/qj.3803)

- Dueben, P. D. and Bauer, P. (2018). *Challenges and design choices for global weather and climate models based on machine learning.* Geosci. Model Dev., 11, 3999–4009.
