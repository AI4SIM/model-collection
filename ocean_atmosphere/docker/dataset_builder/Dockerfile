FROM continuumio/miniconda3

LABEL maintainer="Atos AI4Sim"

ENV http_proxy=$HTTP_PROXY
ENV https_proxy=$HTTPS_PROXY

RUN conda install -c conda-forge  \
        xesmf \
        xarray \
        dask \
        netCDF4 \
        matplotlib \
        cartopy \
        jupyterlab

CMD ["/bin/bash"]