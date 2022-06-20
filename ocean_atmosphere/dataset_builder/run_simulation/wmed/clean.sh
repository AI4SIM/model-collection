#!/bin/bash
if [ ! -z $1 ] 
then 
    DATA_DIR=$1
    echo "$0: Using DATA_DIR=${DATA_DIR}"
else
    DATA_DIR=/net/172.16.118.188/data/ocean_atmosphere/simulations
    echo "$0: Using default DATA_DIR=${DATA_DIR}"
fi

WORK_DIR=${DATA_DIR}/wmed

read -p "All simulation data in $WORK_DIR will be deleted. Are you sure? [y/n]" -n 1 -r
echo    # (optional) move to a new line
if [[ ! $REPLY =~ ^[Yy]$ ]]
then
    [[ "$0" = "$BASH_SOURCE" ]] && exit 1 || return 1 # handle exits from shell or function but don't exit interactive shell
fi

cd ${WORK_DIR}/input_oce 
rm *.xml
rm croco.in
rm create_restart_file_from_ini_CROCO_file.py
rm rstrt_SAVE.nc
rm rst_O.nc

rm -rf ${WORK_DIR}/input_atm
rm -rf ${WORK_DIR}/input_oasis
rm -rf ${WORK_DIR}/run
