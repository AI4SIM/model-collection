FROM oa-mpi-oasis:latest as base

LABEL maintainer="Atos AI4Sim"

##############################################################
# Download MesoNH
##############################################################

ARG MNH_VERSION=5-5-0
ARG MNH_URL="http://mesonh.aero.obs-mip.fr/mesonh/dir_open/dir_MESONH/MNH-V${MNH_VERSION}.tar.gz"

ENV MY_HOME=/home/atmosphere
ARG MNH_VERSION=5-5-0

# Compilation options for MESONH
ARG ARCH=LXgfortran
ARG VER_MPI=MPIAUTO
ARG VER_OASIS=OASISDOCKER
ARG VER_CDF=CDFDOCKER

ARG OPTLEVEL=O3

ARG MNH_BUILD_NAME_MASTER=${ARCH}-R8I4-MNH-V${MNH_VERSION}-${VER_OASIS}-${VER_MPI}-${OPTLEVEL}
ARG MNH_BUILD_NAME_USER=${ARCH}-R8I4-MNH-V${MNH_VERSION}-${VER_OASIS}-MY_SRC-${VER_MPI}-${OPTLEVEL}

RUN mkdir -p ${MY_HOME} \
    &&  cd ${MY_HOME} \
    # Downlaod MesoNH
    &&  curl -L ${MNH_URL} | tar xz \
    && mv MNH-V${MNH_VERSION} mesonh


ENV MNH_DIR=${MY_HOME}/mesonh
ENV MNH_PROFILE_MASTER=${MNH_DIR}/conf/profile_mesonh-${MNH_BUILD_NAME_MASTER}
ENV MNH_PROFILE_USER=${MNH_DIR}/conf/profile_mesonh-${MNH_BUILD_NAME_USER}

ADD utils_mesonh/Makefile.MESONH.mk ${MNH_DIR}/src/

RUN cd ${MNH_DIR}/src \
    # Generate the profile with the specified compilation options
    &&  ./configure \
    # The MNH_PROFILE returns a dummy "error" for ARCH=LXgfortran (Linux +
    # gfortran) because there is no default config file conf/conf_LXgfortran
    # (see l. 163 in ${MNH_DIR}/conf/profile_mesonh). To still build the image
    # after this "error", let's ignore it with: <cmd> || true. A workaround
    # would be to provide for conf/conf_LXgfortran along with the Dockerfile.
    &&  . ../conf/profile_mesonh || true\
    # There can be some problems with the first pass in parallel compilation.
    # Let's first redirect errors so that the RUN instruction does not stop the
    # image build and then compile a second time without error redirect (this
    # hack comes from the jobscript ${MNH_DIR}/src/job_make_mesonh_CRAY_cca)
    #&&  make -j 4 |& tee error${XYZ} \
    &&  make -j 20 \
    # Install the new compiled program in the ${MNH_DIR}/exe folder
    &&  make installmaster

ADD utils_mesonh/ /tmp/utils_mesonh/

ARG VER_USER=MY_SRC

RUN cd ${MNH_DIR}/src \
    && mkdir -p MY_SRC \
    && cp -f /tmp/utils_mesonh/*.F90 MY_SRC/ \
    && rm -rf /tmp/utils_mesonh \
    && unset XYZ \
    && ./configure \
    && . ${MNH_PROFILE_USER} || true \
    && gmake -j 20 user |& tee error${XYZ} \
    && gmake -j 20 user |& tee error${XYZ} \
    && gmake -j 20 user |& tee error${XYZ} \
    && gmake -j 20 user |& tee error${XYZ} \
    && gmake -j 20 user |& tee error${XYZ} \
    && gmake -j 20 user |& tee error${XYZ} \
    && gmake -j 20 user |& tee error${XYZ} \
    && gmake -j 20 user |& tee error${XYZ} \
    && gmake -j 20 user |& tee error${XYZ} \
    && gmake -j 20 user \
    && gmake installuser


CMD ["bash"]