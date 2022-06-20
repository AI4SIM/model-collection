#/bin/bash

pwd=`pwd`

ln -sf ${pwd}/../input_atm/rstrt_SAVE.nc rst_A.nc
ln -sf ${pwd}/../input_oce/rstrt_SAVE.nc rst_O.nc

ln -sf ${pwd}/create_rmp_files/create_rmp_files_atmt_ocnt_bilinear/rundir/rmp_atmt_to_ocnt_DISTWGT.nc .
ln -sf ${pwd}/create_rmp_files/create_rmp_files_ocnt_atmt_bilinear/rundir/rmp_ocnt_to_atmt_DISTWGT.nc .

