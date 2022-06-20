#!/bin/bash
#SBATCH --job-name=prepare_wmed --ntasks 9 --partition=rome7402
#SBATCH --output=prepare_wmed_job.out 
#SBATCH --error=prepare_wmed_job.err

DATA_DIR=/net/172.16.118.188/data/ocean_atmosphere/simulations
WORK_DIR=/net/172.16.118.188/data/ocean_atmosphere/simulations/wmed
SCRIPTPATH=/home_nfs/delarouh/model-collection/ocean_atmosphere/dataset_builder/run_simulation/wmed
OCI_CROCO_XIOS=${SCRIPTPATH}/../../croco-xios.sif
OCI_MESONH=${SCRIPTPATH}/../../mesonh.sif

# executables within container
MNH_EXE=/home/atmosphere/mesonh/exe
PREP_PGD=${MNH_EXE}/PREP_PGD-LXgfortran-R8I4-MNH-V5-5-0-OASISDOCKER-MPIAUTO-O3
PREP_REAL=${MNH_EXE}/PREP_REAL_CASE-LXgfortran-R8I4-MNH-V5-5-0-OASISDOCKER-MPIAUTO-O3
# path to croco exe dir
DIR_EXE_CROCO=/home/ocean/croco/build/wmed/exe

# --------------------------------------------------
# -- Prepare run directory
# --------------------------------------------------
mkdir -p ${WORK_DIR}

#TODO: run script to download era5 data (<WD>/wmed/era5)
ERA5_DATA_DIR=${WORK_DIR}/era5

#TODO: run script to download croco inputs (<WD>/input_oce)
cp ${SCRIPTPATH}/input_oce/* ${WORK_DIR}/input_oce/

rm -rf ${WORK_DIR}/input_atm
cp -rf ${SCRIPTPATH}/input_atm ${WORK_DIR}/input_atm

rm -rf ${WORK_DIR}/input_oasis
cp -rf ${SCRIPTPATH}/input_oasis ${WORK_DIR}/input_oasis

# --------------------------------------------------
# -- Prepare MesoNH input files
# --------------------------------------------------
cd ${WORK_DIR}/input_atm
PGD=${DATA_DIR}/PGD

if [ ! -d ${PGD} ]
then
    echo "PGD DOES NOT EXIST"
fi

ln -sf ${PGD}/CLAY_HWSD_MOY_v2.??? .
ln -sf ${PGD}/SAND_HWSD_MOY_v2.??? .
ln -sf ${PGD}/gtopo30.??? .
ln -sf ${PGD}/ECOCLIMAP_v2.0.??? .

# prep_pgd
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
H_DIR=${WORK_DIR}/input_atm

echo "Running PREP_PGD: ${PREP_PGD}"

time srun --mpi=pmix_v3 -v     -n 8 singularity exec         -H $H_DIR         --bind $WORK_DIR:$WORK_DIR         --bind $PGD:$PGD         $OCI_MESONH         ${PREP_PGD}
# prep_real_case
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

YEAR='2006'
MONTH='01'


echo "Running PREP_REAL_CASE: ${PREP_REAL}"
for DAY in '01' '02'
do
  for HOUR in '00' '06' '12' '18'
  do

    echo '======================================='
    echo ' Treatment of the date' $YEAR$MONTH$DAY 'at' $HOUR'h'
    echo '======================================='

    ln -sf ${ERA5_DATA_DIR}/era5.${YEAR}${MONTH}${DAY}.${HOUR} .

    cp PRE_REAL1.nam_SAVE PRE_REAL1.nam

    sed -i  "s/YEAR/$YEAR/g"  PRE_REAL1.nam
    sed -i "s/MONTH/$MONTH/g" PRE_REAL1.nam
    sed -i   "s/DAY/$DAY/g"   PRE_REAL1.nam
    sed -i  "s/HOUR/$HOUR/g"  PRE_REAL1.nam

    time srun --mpi=pmix_v3 -v     -n 8 singularity exec         -H ${H_DIR}         --bind $WORK_DIR:$WORK_DIR         --bind $PGD:$PGD         $OCI_MESONH         $PREP_REAL

    mv OUTPUT_LISTING0 OUTPUT_LISTING0_${YEAR}${MONTH}${DAY}${HOUR}
    mv PRE_REAL1.nam PRE_REAL1.nam_${YEAR}${MONTH}${DAY}${HOUR}

  done
done

# restart file
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

echo "Creating restart file with: create_restart_file_from_PRC_MNH_file.py"

time singularity exec    -H ${H_DIR}     --bind $WORK_DIR:$WORK_DIR     $OCI_MESONH     python3 create_restart_file_from_PRC_MNH_file.py

# --------------------------------------------------
# -- Prepare CROCO input file
# --------------------------------------------------
H_DIR=${WORK_DIR}/input_oce
cd ${H_DIR}
echo "Creating restart file with: create_restart_file_from_ini_CROCO_file.py"
time singularity exec    -H ${H_DIR}     --bind $WORK_DIR:$WORK_DIR     $OCI_MESONH     python3 create_restart_file_from_ini_CROCO_file.py


# --------------------------------------------------
# -- Prepare OASIS input files
# --------------------------------------------------

H_DIR=${WORK_DIR}/input_oasis/create_rmp_files
cd ${H_DIR}
echo "Building test model for interpolation"
time singularity exec    -H ${H_DIR}     --bind $WORK_DIR:$WORK_DIR     $OCI_MESONH     make

./prep_testinterp_atmt_ocnt.sh
./prep_testinterp_ocnt_atmt.sh

H_DIR=${WORK_DIR}/input_oasis/create_rmp_files/data_oasis3
echo "Preparing atm and oce grids for interpolation"
time singularity exec    -H ${H_DIR}     --bind $WORK_DIR:$WORK_DIR     $OCI_MESONH     python3 create_grids_areas_and_masks_files_for_rmp_from_mnh_and_croco.py

H_DIR=${WORK_DIR}/input_oasis/create_rmp_files/create_rmp_files_atmt_ocnt_bilinear/rundir
echo "Running test model for interpolation: atmt_ocnt ${H_DIR}"
cd $H_DIR

time singularity exec     -H $H_DIR     --bind $WORK_DIR:$WORK_DIR     $OCI_MESONH     mpirun -np 1 ./model1 : -np 8 ./model2

H_DIR=${WORK_DIR}/input_oasis/create_rmp_files/create_rmp_files_ocnt_atmt_bilinear/rundir
echo "Running test model for interpolation: ocnt_atmt ${H_DIR}"
cd $H_DIR

time singularity exec     -H $H_DIR     --bind $WORK_DIR:$WORK_DIR     $OCI_MESONH     mpirun -np 1 ./model1 : -np 8 ./model2

cd ${WORK_DIR}/input_oasis

ln -sf ../input_atm/rstrt_SAVE.nc rst_A.nc
ln -sf ../input_oce/rstrt_SAVE.nc rst_O.nc

ln -sf create_rmp_files/create_rmp_files_atmt_ocnt_bilinear/rundir/rmp_atmt_to_ocnt_DISTWGT.nc .
ln -sf create_rmp_files/create_rmp_files_ocnt_atmt_bilinear/rundir/rmp_ocnt_to_atmt_DISTWGT.nc .

# --------------------------------------------------
# -- Prepare RUN
# --------------------------------------------------
H_DIR=${WORK_DIR}/run
echo "Preparing run dir: ${H_DIR}"
mkdir -p ${H_DIR}
cd ${H_DIR}

#~~~~~~ MESONH
ln -sf ../input_atm/PGD_WMED_10km.* .
ln -sf ../input_atm/ECMWF_200601* .
ln -sf ../input_atm/EXSEG1.nam_B_cpl_mnh_croco EXSEG1.nam

#~~~~~~ CROCO
ln -sf ../input_oce/croco_grd.nc .
ln -sf ../input_oce/croco_iniVmodif.nc croco_ini.nc
ln -sf ../input_oce/croco_bry.nc .
ln -sf ../input_oce/croco_runoff.nc .
ln -sf ../input_oce/croco.in croco.in
ln -sf ../input_oce/wmed_bulk_wrf_Y2006M01.nc .

#~~~~~~ OASIS
ln -sf ../input_oasis/namcouple_B_cpl_mnh_croco namcouple
ln -sf ../input_oasis/rmp_*.nc .
ln -sf ../input_oasis/rst_*.nc .

#~~~~~~ XIOS
ln -sf ../input_oce/context_croco.xml .
ln -sf ../input_oce/domain_def_croco.xml .
ln -sf ../input_oce/field_def_croco.xml .
ln -sf ../input_oce/file_def_croco.xml .
ln -sf ../input_oce/iodef.xml .

