#!/bin/bash

if [ ! -z $1 ] 
then 
    SAVE_SIF_DIR=$1
    echo "$0: Using SAVE_SIF_DIR=${SAVE_SIF_DIR}"
else
    SAVE_SIF_DIR=/net/172.16.118.188/data/ocean_atmosphere
    echo "$0: Using default SAVE_SIF_DIR=${SAVE_SIF_DIR}"
fi

SCRIPT=`realpath $0`
SCRIPTPATH=`dirname $SCRIPT`


mkdir -p ${SAVE_SIF_DIR}

for IMG in mpi-oasis mesonh croco-xios
do
    echo "Building ${IMG} with Docker"
    cd ${SCRIPTPATH}/docker/${IMG}
    docker build --progress plain\
        --build-arg HTTP_PROXY=${http_proxy}  \
        --build-arg HTTPS_PROXY=${http_proxy} \
        -t ${IMG}:latest \
        .

    cd ${SAVE_SIF_DIR}
    echo "Exporting ${IMG}:latest to singularity"
    docker save ${IMG}:latest -o ${IMG}.tar
    singularity build --force ${IMG}.sif docker-archive://${IMG}.tar
    rm ${IMG}.tar
    ln -sf ${SCRIPTPATH}/dataset_builder/${IMG}.sif ${IMG}.sif
done
