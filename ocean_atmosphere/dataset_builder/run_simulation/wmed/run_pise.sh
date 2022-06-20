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

SCRIPT=`realpath $0`
SCRIPTPATH=`dirname $SCRIPT`

cat <<EOF > wmed.sbatch
#!/bin/bash
#SBATCH --output=wmed_job.out 
#SBATCH --error=wmed_job.err
#SBATCH --job-name=wmed
#SBATCH --partition=rome7402
#SBATCH --job-name=wmed --nodes 4 --ntasks 343 --partition=rome7402

# global MPI informations on PISE cluster 
export OMPI_MCA_btl_tcp_if_include=10.1.0.1/16 # infiniBand IP
export OMPI_MCA_pml=ob1

export WORK_DIR=${DATA_DIR}/wmed
export OCI_CROCO_XIOS=${SCRIPTPATH}/../../croco-xios.sif
export OCI_MESONH=${SCRIPTPATH}/../../mesonh.sif
export CROCO_EXE=/home/ocean/croco/build/wmed/exe/croco
export XIOS_EXE=/home/ocean/croco/build/wmed/exe/xios_server.exe
export MESONH_EXE=/home/atmosphere/mesonh/exe/MESONH-LXgfortran-R8I4-MNH-V5-5-0-OASISDOCKER-MY_SRC-MPIAUTO-O3

srun --mpi=pmix_v3 -v --multi-prog ./config_pise/pise.conf

EOF

sbatch wmed.sbatch