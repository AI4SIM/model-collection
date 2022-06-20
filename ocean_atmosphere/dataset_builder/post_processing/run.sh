#!/bin/bash

if [ ! -z $1 ] 
then 
    DATA_DIR=$1
    echo "$0: Using DATA_DIR=${DATA_DIR}"
else
    DATA_DIR=/net/172.16.118.188/data/ocean_atmosphere/simulations/wmed/
    read -rep "Path to the base directory of the simulation to post-process`echo $'\n '`(DEFAULT: ${DATA_DIR}): " -r
    echo 
    if [[ ! -z "$REPLY" ]]
    then
        DATA_DIR=$REPLY
    fi
    echo "$0: Using DATA_DIR=${DATA_DIR}"
fi

if [ ! -d "${DATA_DIR}" ] 
then
    echo "Directory ${DATA_DIR} DOES NOT exists." 
    exit 9999 # die with error code 9999
fi

SCRIPT=`realpath $0`
SCRIPTPATH=`dirname $SCRIPT`


OCI_DATABUILDER=${SCRIPTPATH}/../dataset_builder.sif

H_DIR=${SCRIPTPATH}

time singularity exec \
    -H ${H_DIR} \
    --bind $DATA_DIR:$DATA_DIR \
    ${OCI_DATABUILDER} \
    python main.py $DATA_DIR