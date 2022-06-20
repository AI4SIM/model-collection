FROM ubuntu:22.04 as base

LABEL maintainer="Atos AI4Sim"
SHELL ["/bin/bash", "-c"]

# ------------------------------------------------------------
# Required packages
# ------------------------------------------------------------
ARG DEBIAN_FRONTEND=noninteractive
ENV http_proxy=$HTTP_PROXY
ENV https_proxy=$HTTPS_PROXY
RUN echo "Acquire::http::Proxy \"${http_proxy}\";" >> /etc/apt/apt.conf

RUN apt-get --yes -qq update \
    &&  apt-get --yes -qq upgrade \
    &&  apt-get --yes -qq install \
            curl \
            xz-utils \
            bzip2 \
            make \
            cmake \
            wget \
            git \
            gcc \
            g++ \
            gfortran \
            zlib1g-dev \
            m4 \
            htop \
            vim \
            python3 \
            python3-pip \
    &&  apt-get --yes -qq clean\
    &&  rm -rf /var/lib/apt/lists/*

# Download Python packages
ARG PIP_INDEX_URL=https://pypi.python.org/simple/
ARG PIP_TRUSTED_HOST=pypi.python.org
ENV PIP_INDEX_URL $PIP_INDEX_URL
ENV PIP_TRUSTED_HOST $PIP_TRUSTED_HOST

RUN pip3 install \
    --no-cache-dir \
    --trusted-host ${PIP_TRUSTED_HOST} \
    --index-url ${PIP_INDEX_URL} \
    netCDF4 numpy scipy

# ------------------------------------------------------------
# Download sources
# ------------------------------------------------------------

ARG OMPI_VERSION=4.0.2
ARG OMPI_MAJOR_VERSION=4.0
ARG OMPI_URL="https://download.open-mpi.org/release/open-mpi/v${OMPI_MAJOR_VERSION}/openmpi-${OMPI_VERSION}.tar.bz2"

ARG HDF5_MAJOR_VERSION=1.12
ARG HDF5_VERSION=1.12.0
ARG HDF5_URL="https://support.hdfgroup.org/ftp/HDF5/releases/hdf5-${HDF5_MAJOR_VERSION}/hdf5-${HDF5_VERSION}/src/hdf5-${HDF5_VERSION}.tar.gz"

ARG CCDF_VERSION=4.8.1
ARG F90CDF_VERSION=4.5.4
ARG CCDF_URL="https://downloads.unidata.ucar.edu/netcdf-c/${CCDF_VERSION}/netcdf-c-${CCDF_VERSION}.tar.gz"
ARG F90CDF_URL="https://downloads.unidata.ucar.edu/netcdf-fortran/${F90CDF_VERSION}/netcdf-fortran-${F90CDF_VERSION}.tar.gz"

ARG OASIS_VERSION=3.1
ARG OASIS_URL="https://gitlab.com/cerfacs/oasis3-mct/-/archive/OASIS3-MCT_${OASIS_VERSION}/oasis3-mct-OASIS3-MCT_${OASIS_VERSION}.tar.gz"

RUN mkdir -p /tmp/openmpi/ \
    # Download OpenMPI
    &&  cd /tmp/openmpi \
    &&  curl -L ${OMPI_URL} | tar xj \
    &&  mkdir -p /tmp/hdf5/ \
    &&  cd /tmp/hdf5 \
    # Download HDF5
    &&  curl -L ${HDF5_URL} | tar xz \
    &&  mkdir -p /tmp/netcdf/ \
    # Download netCDF-c/fortran/cxx
    &&  cd /tmp/netcdf \
    &&  curl -L ${CCDF_URL} | tar xz \
    &&  curl -L ${F90CDF_URL} | tar xz \
    # Download OASIS
    &&  mkdir -p /tmp/oasis/ \
    &&  cd /tmp/oasis \
    &&  curl -L ${OASIS_URL} | tar xz 
 
# ------------------------------------------------------------
# Environment variables
# ------------------------------------------------------------

ENV OMPI_DIR=/usr/local/ompi
ENV HDF5_DIR=/usr/local/hdf5
ENV NETCDF_DIR=/usr/local/netcdf
ENV OASIS_DIR=/usr/local/oasis
ENV LIB_DIR=/usr/local/lib

ENV PATH=${OMPI_DIR}/bin:${PATH}

ENV LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:${OASIS_DIR}/lib:${NETCDF_DIR}/lib:${HDF5_DIR}/lib:${OMPI_DIR}/lib:${LIB_DIR}

ENV MANPATH=${OMPI_DIR}/share/man:${MANPATH}

# ------------------------------------------------------------
# Compilation
# ------------------------------------------------------------
# OpenMPI
RUN cd /tmp/openmpi/openmpi-${OMPI_VERSION} \
    &&  ./configure \
        --prefix=${OMPI_DIR} \
    &&  make -j 4 install \
    &&  rm -Rf /tmp/openmpi 

# HDF5
RUN cd /tmp/hdf5/hdf5-${HDF5_VERSION} \
    &&  F90=mpif90 \
        CC=mpicc \
        ./configure \
            --with-zlib=/usr/local \
            --prefix=${HDF5_DIR} \
            --enable-fortran \
            --enable-parallel \
            --enable-shared \
    &&  make -j 20 \
    &&  make install \
    &&  rm -Rf /tmp/hdf5

# netCDF-c
RUN cd /tmp/netcdf/netcdf-c-${CCDF_VERSION} \ 
    &&  F90=mpif90 \
        CC=mpicc \
        CPPFLAGS=-I${HDF5_DIR}/include \
        LDFLAGS=-L${HDF5_DIR}/lib \
        ./configure \
            --with-zlib=/usr/local \
            --with-hdf5=${HDF5_DIR} \
            --prefix=${NETCDF_DIR} \
            --enable-parallel \
    &&  make -j 20 \
    &&  make install \
    &&  make clean

# netCDF-fortran
RUN cd /tmp/netcdf/netcdf-fortran-${F90CDF_VERSION} \ 
    &&  F90=mpif90 \
        CC=mpicc \
        CPPFLAGS="-I${HDF5_DIR}/include -I${NETCDF_DIR}/include" \
        LDFLAGS="-L${HDF5_DIR}/lib -L${NETCDF_DIR}/lib" \
        ./configure \
            --with-zlib=/usr/local \
            --with-hdf5=${HDF5_DIR} \
            --prefix=${NETCDF_DIR} \
            --enable-parallel \
            --enable-shared \
    &&  make -j 20 \
    &&  make install \
    &&  make clean \
    &&  rm -Rf /tmp/netcdf

# OASIS
ADD utils_oasis/* /tmp/oasis/

RUN cd /tmp/oasis/oasis3-mct-OASIS3-MCT_${OASIS_VERSION}/util/make_dir \
    &&  mv ../../../make.* . \
    &&  make realclean -f TopMakefileOasis3 \
    &&  make -f TopMakefileOasis3 \
    &&  rm -Rf /tmp/oasis

CMD ["bash"]