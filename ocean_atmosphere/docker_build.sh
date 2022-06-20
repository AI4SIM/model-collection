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

for IMG in mpi-oasis mesonh croco-xios dataset_builder 
do
    echo "Building ${IMG} with Docker"
    cd ${SCRIPTPATH}/docker/${IMG}
    docker build \
        --build-arg HTTP_PROXY=${http_proxy}  \
        --build-arg HTTPS_PROXY=${http_proxy} \
        -t oa-${IMG}:latest \
        .
done

for IMG in mesonh croco-xios dataset_builder 
do
    cd ${SAVE_SIF_DIR}
    echo "Exporting oa-${IMG}:latest to singularity"
    docker save oa-${IMG}:latest -o ${IMG}.tar
    singularity build --force ${IMG}.sif docker-archive://${IMG}.tar
    rm ${IMG}.tar
    ln -sf ${SAVE_SIF_DIR}/${IMG}.sif ${SCRIPTPATH}/dataset_builder/${IMG}.sif 
done