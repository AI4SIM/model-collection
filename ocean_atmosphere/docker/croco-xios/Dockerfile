FROM mpi-oasis:latest as base

LABEL maintainer="Atos AI4Sim"

ENV MY_HOME=/home/ocean
ENV XIOS_DIR=${MY_HOME}/xios
ENV CROCO_DIR=${MY_HOME}/croco

ARG XIOS_URL="https://forge.ipsl.jussieu.fr/ioserver/svn/XIOS/trunk/"
ARG CROCO_VERSION=1.2
ARG CROCO_URL="https://data-croco.ifremer.fr/CODE_ARCHIVE/croco-v${CROCO_VERSION}.tar.gz"

RUN mkdir -p ${MY_HOME} \
    && cd ${MY_HOME} \
    && curl -L ${CROCO_URL} | tar xz \
    && wget -r -np https://forge.ipsl.jussieu.fr/ioserver/svn/XIOS/trunk/ \
    && mv forge.ipsl.jussieu.fr/ioserver/svn/XIOS/trunk xios \
    && rm -rf forge.ipsl.jussieu.fr

ADD utils_xios/* ${XIOS_DIR}/arch/

RUN cd ${XIOS_DIR} \
    && rm -rf /tmp/utils_xios \
    && chmod +x make_xios \
    && ./make_xios \
        --prod \
        --fcm new \
        --arch docker \
        --use_oasis oasis3_mct \
        --netcdf_lib netcdf4_par \
        --job 4

COPY utils_croco /tmp/utils_croco

RUN for CONFDIR in /tmp/utils_croco/config_*/; \
    do \
        cd $CONFDIR; \
        ./jobcomp; \
    done \
    && rm -rf /tmp/utils_croco

CMD ["bash"]