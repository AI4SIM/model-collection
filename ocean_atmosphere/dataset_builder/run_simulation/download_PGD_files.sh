#!/bin/bash

if [ ! -z $1 ] 
then 
    DATA_DIR=$1
    echo "$0: Using DATA_DIR=${DATA_DIR}"
else
    DATA_DIR=/net/172.16.118.188/data/ocean_atmosphere/simulations/
    read -rep "Path to the ocean-atmosphere dataset directory`echo $'\n '`(DEFAULT: ${DATA_DIR}): " -r
    echo 
    if [[ ! -z "$REPLY" ]]
    then
        DATA_DIR=$REPLY
    fi
    echo "$0: Using DATA_DIR=${DATA_DIR}"
fi

BASE_URL="http://mesonh.aero.obs-mip.fr/mesonh/dir_open/dir_PGDFILES/"

mkdir -p ${DATA_DIR}/PGD
cd ${DATA_DIR}/PGD
for DATA in 'gtopo30' 'CLAY_HWSD_MOY_v2' 'SAND_HWSD_MOY_v2' 'ECOCLIMAP_v2.0'
do
    for EXT in '.hdr.gz' '.dir.gz'
    do
        wget ${BASE_URL}${DATA}${EXT} 
    done
done

gunzip *.gz

