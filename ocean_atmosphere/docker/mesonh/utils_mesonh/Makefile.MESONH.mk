#MNH_LIC Copyright 1994-2021 CNRS, Meteo-France and Universite Paul Sabatier
#MNH_LIC This is part of the Meso-NH software governed by the CeCILL-C licence
#MNH_LIC version 1. See LICENSE, CeCILL-C_V1-en.txt and CeCILL-C_V1-fr.txt
#MNH_LIC for details. version 1.
##########################################################
#                                                        #
#           Initialisation of some variables             #
#                                                        #
##########################################################
ifdef OBJDIR_PATH
OBJDIR_ROOT=${OBJDIR_PATH}/dir_obj
else
OBJDIR_ROOT=${PWD}/dir_obj
endif
LIB_OBJS_ROOT=lib
#
ARCH_XYZ=${ARCH}-R${MNH_REAL}I${MNH_INT}-${VERSION_XYZ}
##########################################################
#                                                        #
#            Source DIRECTORY                            #
#                                                        #
##########################################################

##########################################################
#           Source MYSRC                                 #
##########################################################
ifdef VER_USER
DIR_USER += ${VER_USER}
endif
##########################################################
#           Source MNH                                   #
##########################################################
ifdef VER_OASIS
CPPFLAGS       += -DCPLOASIS
endif
# PRE_BUG TEST !!!
#DIR_MNH += ARCH_SRC/bug_mnh
# PRE_BUG TEST !!!
#
DIR_MNH += MNH
INC_MNH += -I$(B)include
#CPPFLAGS_MNH =
#
ifdef DIR_MNH
DIR_MASTER += $(DIR_MNH)
CPPFLAGS   += $(CPPFLAGS_MNH)
INC        += $(INC_MNH)

#
# MNH integer 4/8
#
CPPFLAGS   += -DMNH_INT=$(MNH_INT)
#
# MNH real 4/8
#
CPPFLAGS   += -DMNH_REAL=$(MNH_REAL)
#
#  Len of HREC characters 
#
CPPFLAGS   += -DLEN_HREC=$(LEN_HREC)
#

OBJS_NOCB +=  spll_dxf.o spll_dxm.o spll_dyf.o spll_dym.o \
        spll_dzf.o spll_dzm.o spll_mxf.o \
        spll_mxm.o spll_myf.o spll_mym.o spll_mzf.o \
        spll_mzm.o spll_mzf4.o spll_mzm4.o  \
        spll_gx_m_m.o spll_gx_m_u.o spll_gy_m_m.o \
        spll_gy_m_v.o spll_gz_m_m.o spll_gz_m_w.o \
        spll_dzf_mf.o spll_dzm_mf.o spll_mzf_mf.o spll_mzm_mf.o \
        spll_modi_gradient_m_d.o

$(OBJS_NOCB) : OPT = $(OPT_NOCB)

OBJS0 += spll_switch_sbg_lesn.o spll_mode_mppdb.o

$(OBJS0)     : OPT = $(OPT0) 

endif
##########################################################
#           Source SURFEX                                #
##########################################################
# PRE_BUG TEST !!!
#DIR_SURFEX += ARCH_SRC/bug_surfex
# PRE_BUG TEST !!!
#
DIR_SURFEX += ARCH_SRC/surfex
DIR_SURFEX += SURFEX
CPPFLAGS_SURFEX += -DMNH_PARALLEL -Din_surfex 
#
ifdef DIR_SURFEX
DIR_MASTER += $(DIR_SURFEX)
CPPFLAGS   += $(CPPFLAGS_SURFEX)
VER_SURFEX=SURFEX-4-8-0
#ARCH_XYZ    := $(ARCH_XYZ)-$(VER_MYSRC)

OBJS_NOCB +=  spll_mode_cover_301_573.o 

$(OBJS0): OPT = $(OPT0) 

endif
##########################################################
#           Source SURCOUCHE                             #
##########################################################
# PRE_BUG TEST !!!
#DIR_SURCOUCHE += ARCH_SRC/bug_surcouche
# PRE_BUG TEST !!!
#
DIR_SURCOUCHE += LIB/SURCOUCHE/src
#CPPFLAGS_SURCOUCHE = -DMNH_MPI_BSEND
#
ifdef DIR_SURCOUCHE
DIR_MASTER   += $(DIR_SURCOUCHE)
CPPFLAGS     += $(CPPFLAGS_SURCOUCHE)
#VER_SURCOUCHE=
#ARCH_XYZ    := $(ARCH_XYZ)-$(VER_SURCOUCHE)
endif
##########################################################
#           Source MINPACK                             #
##########################################################
DIR_MINPACK += LIB/minpack
#
ifdef DIR_MINPACK
DIR_MASTER   += $(DIR_MINPACK)
endif
##########################################################
#           Source RAD                                   #
##########################################################
# PRE_BUG TEST !!!
#DIR_RAD      += ARCH_SRC/bug_rad
# PRE_BUG TEST !!!
#
DIR_RAD      +=  LIB/RAD/ECMWF_RAD
#CPPFLAGS_RAD =
INC_RAD      = -I$(B)LIB/RAD/ECMWF_RAD
#
ifdef MNH_ECRAD
DIR_RAD      +=  LIB/RAD/ecrad-$(VERSION_ECRAD)_mnh
DIR_RAD      +=  LIB/RAD/ecrad-$(VERSION_ECRAD)
CPPFLAGS_RAD = -DMNH_ECRAD -DVER_ECRAD=$(VER_ECRAD)
INC_RAD      += -I$(B)LIB/RAD/ecrad-$(VERSION_ECRAD)/include
ifeq "$(VER_ECRAD)" "140"
INC_RAD      += -I$(B)LIB/RAD/ecrad-$(VERSION_ECRAD)/drhook/include
IGNORE_DEP_MASTER   += yomhook.D
endif
ifneq "$(VER_ECRAD)" "140"
IGNORE_DEP_MASTER   += read_albedo_data.D read_emiss_data.D
endif

ARCH_XYZ    := $(ARCH_XYZ)-ECRAD$(VER_ECRAD)
endif
#
#
ifdef DIR_RAD
DIR_MASTER  += $(DIR_RAD)
CPPFLAGS    += $(CPPFLAGS_RAD)
INC         += $(INC_RAD)

IGNORE_DEP_MASTER   += olwu.D olwv.D rad1Driv_MACLATMOSPH_60LEVELS_ICRCCM3.D tstrad.D tstrad_chansubset.D tstrad_rttov7.D \
                       tstrad_sx6.D 

ifndef MNH_ECRAD
IGNORE_DEP_MASTER   += read_albedo_data.D read_emiss_data.D
endif


OBJS0 += spll_orrtm_kgb1.o spll_orrtm_kgb14.o spll_orrtm_kgb3_a.o spll_orrtm_kgb4_b.o \
        spll_orrtm_kgb5_c.o spll_orrtm_kgb10.o spll_orrtm_kgb15.o spll_orrtm_kgb3_b.o \
        spll_orrtm_kgb4_c.o spll_orrtm_kgb6.o spll_orrtm_kgb11.o spll_orrtm_kgb16.o \
        spll_orrtm_kgb3_c.o spll_orrtm_kgb5.o spll_orrtm_kgb7.o spll_orrtm_kgb12.o \
        spll_orrtm_kgb2.o spll_orrtm_kgb4.o spll_orrtm_kgb5_a.o spll_orrtm_kgb8.o \
        spll_orrtm_kgb13.o spll_orrtm_kgb3.o spll_orrtm_kgb4_a.o spll_orrtm_kgb5_b.o \
        spll_orrtm_kgb9.o spll_read_xker_raccs.o spll_read_xker_rdryg.o spll_read_xker_sdryg.o \
        spll_suecaebc.o  spll_suecaec.o  spll_suecaeor.o  spll_suecaesd.o \
        spll_suecaess.o  spll_suecaesu.o spll_suecozc.o  spll_suecozo.o
ifdef MNH_ECRAD
OBJS0 += spll_rrtm_kgb1.o spll_rrtm_kgb14.o spll_rrtm_kgb3_a.o spll_rrtm_kgb4_b.o \
        spll_rrtm_kgb5_c.o spll_rrtm_kgb10.o spll_rrtm_kgb15.o spll_rrtm_kgb3_b.o \
        spll_rrtm_kgb4_c.o spll_rrtm_kgb6.o spll_rrtm_kgb11.o spll_rrtm_kgb16.o \
        spll_rrtm_kgb3_c.o spll_rrtm_kgb5.o spll_rrtm_kgb7.o spll_rrtm_kgb12.o \
        spll_rrtm_kgb2.o spll_rrtm_kgb4.o spll_rrtm_kgb5_a.o spll_rrtm_kgb8.o \
        spll_rrtm_kgb13.o spll_rrtm_kgb3.o spll_rrtm_kgb4_a.o spll_rrtm_kgb5_b.o \
        spll_rrtm_kgb9.o spll_read_xker_raccs.o spll_read_xker_rdryg.o spll_read_xker_sdryg.o \
        spll_suecaebc.o  spll_suecaec.o  spll_suecaeor.o  spll_suecaesd.o \
        spll_suecaess.o  spll_suecaesu.o spll_suecozc.o  spll_suecozo.o
IGNORE_DEP_MASTER   += rrtm_rrtm_140gp_mcica.D srtm_spcvrt_mcica.D srtm_srtm_224gp_mcica.D radiation_psrad.D \
                       radiation_psrad_rrtm.D test_spartacus_math.D radiation_adding_ica_sw_test.D \
                       radiation_adding_ica_sw_test2.D srtm_gas_optical_depth_test.D
endif

$(OBJS0): OPT = $(OPT0) 

endif
##########################################################
#           Source RTTOV                                 #
##########################################################
ifdef MNH_RTTOV
ifndef VER_RTTOV
VER_RTTOV      = 8.7
endif
ifeq "$(VER_RTTOV)" "8.7"
DIR_RTTOV      +=  LIB/RTTOV/src
CPPFLAGS_RTTOV = -DMNH_RTTOV
INC_RTTOV      = -I$(B)LIB/RTTOV/src
#
DIR_MASTER  += $(DIR_RTTOV)
CPPFLAGS    += $(CPPFLAGS_RTTOV)
INC         += $(INC_RTTOV)
CPPFLAGS    += $(CPPFLAGS_RTTOV)
CPPFLAGS_MNH += -DMNH_RTTOV_8=MNH_RTTOV_8
endif
ifeq "$(VER_RTTOV)" "11.3"
DIR_RTTOV=${SRC_MESONH}/src/LIB/RTTOV-${VER_RTTOV}
RTTOV_PATH=${DIR_RTTOV}
#
INC_RTTOV     ?= -I${RTTOV_PATH}/include -I${RTTOV_PATH}/mod
LIB_RTTOV     ?= -L${RTTOV_PATH}/lib -lrttov11_coef_io -lrttov11_mw_scatt -lrttov11_main
INC            += $(INC_RTTOV)
LIBS           += $(LIB_RTTOV)
VPATH         += $(RTTOV_PATH)/mod
CPPFLAGS    += $(CPPFLAGS_RTTOV)
CPPFLAGS_MNH += -DMNH_RTTOV_11=MNH_RTTOV_11
endif
ifeq "$(VER_RTTOV)" "13.0"
DIR_RTTOV=${SRC_MESONH}/src/LIB/RTTOV-${VER_RTTOV}
RTTOV_PATH=${DIR_RTTOV}
#
INC_RTTOV     ?= -I${RTTOV_PATH}/include -I${RTTOV_PATH}/mod
LIB_RTTOV     ?= -L${RTTOV_PATH}/lib -lrttov13_coef_io -lrttov13_hdf -lrttov13_mw_scatt -lrttov13_brdf_atlas -lrttov13_main
INC            += $(INC_RTTOV)
LIBS           += $(LIB_RTTOV)
VPATH         += $(RTTOV_PATH)/mod
CPPFLAGS    += $(CPPFLAGS_RTTOV)
CPPFLAGS_MNH += -DMNH_RTTOV_13=MNH_RTTOV_13
endif
endif
##########################################################
#           Source MEGAN                                 #
##########################################################
ifdef MNH_MEGAN
DIR_MEGAN      +=  LIB/MEGAN 
CPPFLAGS_MEGAN = -DMNH_MEGAN
#
DIR_MASTER  += $(DIR_MEGAN)
CPPFLAGS    += $(CPPFLAGS_MEGAN)
INC         += $(INC_MEGAN)
CPPFLAGS_MNH += -DMNH_MEGAN=${MNH_MEGAN}
endif
##########################################################
#           Source NEWLFI                                #
##########################################################
ifdef MNH_IOLFI
CPPFLAGS_MNH += -DMNH_IOLFI
DIR_NEWLFI      += LIB/NEWLFI/src
#CPPFLAGS_NEWLFI = -DSWAPIO -DLINUX
INC_NEWLFI      = -I$(B)LIB/NEWLFI/src
endif
#
ifdef DIR_NEWLFI
#
# Management/parametrisation of size of INTEGER for files > 16 GiB & RECL for LFI
#
LFI_INT?=4
ifneq "$(findstring 8,$(LFI_INT))" ""
OBJS_I8=spll_NEWLFI_ALL.o
$(OBJS_I8) : OPT = $(OPT_BASE) $(OPT_PERF2) $(OPT_INT8)
endif
#
DIR_MASTER          += $(DIR_NEWLFI)
CPPFLAGS            += $(CPPFLAGS_NEWLFI)
OBJS_LISTE_MASTER   += fswap8buff.o
INC                 += $(INC_NEWLFI)
VPATH               += $(DIR_NEWLFI)
#VER_NEWLFI=
#ARCH_XYZ    := $(ARCH_XYZ)-$(VER_NEWLFI)
endif
##########################################################
#           Source COMPRESS                              #
##########################################################
ifdef MNH_COMPRESS
DIR_COMPRESS           = ../LIBTOOLS/lib/COMPRESS/src
INC_COMPRESS           = -I$(B)$(DIR_COMPRESS)
DIR_MASTER            += $(DIR_COMPRESS)
OBJS_LISTE_MASTER     += bitbuff.o nearestpow2.o
INC                   += $(INC_COMPRESS)
VPATH                 += $(DIR_COMPRESS)
CPPFLAGS_COMPRESS     ?= -DLITTLE_endian
CPPFLAGS              += $(CPPFLAGS_COMPRESS)
endif
##########################################################
#           Source S4PY                                  #
##########################################################
ifdef MNH_S4PY
DIR_S4PY               = LIB/s4py
INC_S4PY               = -I$(B)$(DIR_S4PY)
DIR_MASTER            += $(DIR_S4PY)
OBJS_LISTE_MASTER     += init_gfortran.o
INC                   += $(INC_S4PY)
VPATH                 += $(DIR_S4PY)
endif
##########################################################
#           Source FOREFIRE                              #
##########################################################
ifdef MNH_FOREFIRE
DIR_FOREFIRE          += LIB/FOREFIRE
INC_FOREFIRE           = -I$(B)$(DIR_FOREFIRE)
DIR_MASTER            += $(DIR_FOREFIRE)
OBJS_LISTE_MASTER     += C_ForeFire_Interface.o
INC                   += $(INC_FOREFIRE)
VPATH                 += $(DIR_FOREFIRE)
CPPFLAGS              += -DMNH_FOREFIRE
ARCH_XYZ    := $(ARCH_XYZ)-FF
endif
##########################################################
#           Source TOOLS                                 #
##########################################################
ifdef MNH_TOOLS
DIR_TOOLS  += ../LIBTOOLS/tools/lfi2cdf/src
INC_TOOLS  += -I$(B)$(DIR_TOOLS)
DIR_MASTER += $(DIR_TOOLS)
INC        += $(INC_TOOLS)
VPATH      += $(DIR_TOOLS)
endif
##########################################################
#           Source MPIVIDE                               #
##########################################################
#
ifndef VER_MPI
VER_MPI=MPIVIDE
endif
#VER_MPI=MPIVIDE,LAMMPI,LAMMPI-IB,MPICH-IB
#
#   MPIVIDE
#
ifeq "$(VER_MPI)" "MPIVIDE"
DIR_MPI               += LIB/MPIvide
INC_MPI                = -I$(B)$(DIR_MPI)
DIR_MASTER            += $(DIR_MPI)
OBJS_LISTE_MASTER     += mpivide.o
INC                   += $(INC_MPI)
mpivide.o  : CPPFLAGS += -DFUJI -DMNH_INT=$(MNH_INT) -DMNH_REAL=$(MNH_REAL) \
                        -I$(DIR_MPI)/include
VPATH                 += $(DIR_MPI)
endif
#
#   LAMMPI
#
ifeq "$(VER_MPI)" "LAMMPI"
# Standard Lam mpi
#INC_MPI     = -I$(B)/opt/lam/include
#LIB_MPI     = -L/opt/lam/lib   -lmpi -llammpi++ -llammpio -llamf77mpi -lmpi -llam -lpthread -ldl
# default 64 bits SUSE 9 version
INC_MPI     = -I$(B)/usr/include
LIB_MPI     = -lmpi -llammpi++ -llammpio -llamf77mpi -lmpi -llam -lpthread -ldl -lutil 
INC            += $(INC_MPI)
LIBS           += $(LIB_MPI)
endif
#
#   LAMMPI-IB
#
ifeq "$(VER_MPI)" "LAMMPI-IB"
INC_MPI     = -I/home/sila/LAM-7.1.1/include
LIB_MPI     = -L/usr/local/ibgd/driver/infinihost/lib64 -L/home/sila/LAM-7.1.1/lib \
-llammpio -llamf77mpi -lmpi -llam -lutil -lmosal -lmpga -lmtl_common -lvapi -ldl  -lpthread
INC            += $(INC_MPI)
LIBS           += $(LIB_MPI)
endif
#
#   MPICH-IB
#
ifeq "$(VER_MPI)" "MPICH-IB"
INC_MPI     = -I/usr/local/ibgd/mpi/osu/f95/mvapich-0.9.5/include
LIB_MPI     = -L/usr/local/ibgd/driver/infinihost/lib64 \
                 -L/usr/local/ibgd/mpi/osu/f95/mvapich-0.9.5/lib \
                 -lmpich -lmtl_common -lvapi -lmosal -lmpga -lpthread
INC            += $(INC_MPI)
LIBS           += $(LIB_MPI)
endif
#
#   MPICH-2 CNRM
#
ifeq "$(VER_MPI)" "MPICH2"
INC_MPI     = -I/usr/include
LIB_MPI     = -lmpichf90 -lmpich 
INC            += $(INC_MPI)
LIBS           += $(LIB_MPI)
endif

#
#   OPENMPI 1.1 CNRM
#
ifeq "$(VER_MPI)" "OMPICNRM"
MPI_ROOT=/opt/openmpi
INC_MPI = -I${MPI_ROOT}/include  -I${MPI_ROOT}/include/openmpi/ompi -I${MPI_ROOT}/lib64
LIB_MPI     = -L${MPI_ROOT}/lib64 -lmpi -lopen-rte -lopen-pal -lutil -lnsl -ldl -Wl,--export-dynamic -lm -lutil -lnsl -ldl
INC            += $(INC_MPI)
LIBS           += $(LIB_MPI)
endif

#
#   OPENMPI 1.1 BPROC + OPENIB + IFORT
#
ifeq "$(VER_MPI)" "OMPIIFORT"
MPI_ROOT=/home/sila/DEV/OPEN-MPI-11-IFORT-BPROC-OPENIB
INC_MPI     = -I${MPI_ROOT}/include -I${MPI_ROOT}/include/openmpi/ompi -I${MPI_ROOT}/lib
LIB_MPI     = -L${MPI_ROOT}/lib -lmpi -lorte -lopal -lutil -lnsl -ldl -Wl,--export-dynamic -lm -lutil -lnsl -ldl
INC            += $(INC_MPI)
LIBS           += $(LIB_MPI)
endif

#
#   OPENMPI 1.1.4 IFORT BPROC
#
ifeq "$(VER_MPI)" "OMPI114IFORT"
MPI_ROOT=/home/sila/DEV/OPEN-MPI-114-IFORT-BPROC-OPENIB
INC_MPI     = -I${MPI_ROOT}/include -I${MPI_ROOT}/include/openmpi/ompi -I${MPI_ROOT}/lib
LIB_MPI     = -L${MPI_ROOT}/lib -lmpi -lorte -lopal -lutil -lnsl -ldl -Wl,--export-dynamic -lm -lutil -lnsl -ldl
INC            += $(INC_MPI)
LIBS           += $(LIB_MPI)
endif

#
#   OPENMPI 1.2.2 G95 BPROC
#
ifeq "$(VER_MPI)" "OMPI122G95"
MPI_ROOT=/home/sila/DEV/OPEN-MPI-122-G95-BPROC-OPENIB
INC_MPI     = -I${MPI_ROOT}/include -I${MPI_ROOT}/include/openmpi/ompi -I${MPI_ROOT}/lib
LIB_MPI     = -L${MPI_ROOT}/lib -lmpi_f90 -lmpi_f77 -lmpi -lopen-rte -lopen-pal -Wl,--export-dynamic -lm -lutil -lnsl -ldl
INC            += $(INC_MPI)
LIBS           += $(LIB_MPI)
endif
#
#   OPENMPI12X
#
ifeq "$(VER_MPI)" "OMPI12X"
INC_MPI     = -I${MPI_ROOT}/include -I${MPI_ROOT}/include/openmpi/ompi -I${MPI_ROOT}/lib
LIB_MPI     = -L${MPI_ROOT}/lib -lmpi_f90 -lmpi_f77 -lmpi -lopen-rte -lopen-pal -Wl,--export-dynamic -lm -lutil -lnsl -ldl
INC            += $(INC_MPI)
LIBS           += $(LIB_MPI)
endif
#
#   MPI for SGI-ICE 
#
ifeq "$(VER_MPI)" "MPIICE"
INC_MPI     = 
LIB_MPI     = -lmpi
INC            += $(INC_MPI)
LIBS           += $(LIB_MPI)
endif

ARCH_XYZ    := $(ARCH_XYZ)-$(VER_MPI)

##########################################################
#           Librairie GRIBAPI                            #
##########################################################
ifeq "$(MNH_GRIBAPI)" "yes"
DIR_GRIBAPI?=${SRC_MESONH}/src/LIB/grib_api-${VERSION_GRIBAPI}
GRIBAPI_PATH?=${OBJDIR_MASTER}/GRIBAPI-${VERSION_GRIBAPI}
GRIBAPI_INC?=${GRIBAPI_PATH}/include/grib_api.mod
#
ifdef DIR_GRIBAPI
INC_GRIBAPI   ?= -I${GRIBAPI_PATH}/include
LIB_GRIBAPI   ?= -L${GRIBAPI_PATH}/lib -L${GRIBAPI_PATH}/lib64 -lgrib_api_f90 -lgrib_api
INC           += $(INC_GRIBAPI)
LIBS          += $(LIB_GRIBAPI)
VPATH         += $(GRIBAPI_PATH)/include
R64_GRIBAPI=R64
endif
endif

##########################################################
#           ecCodes library                              #
##########################################################
ifneq "$(MNH_GRIBAPI)" "yes"
DIR_ECCODES_SRC?=${SRC_MESONH}/src/LIB/eccodes-${VERSION_ECCODES}-Source
DIR_ECCODES_BUILD?=${OBJDIR_MASTER}/build_eccodes-${VERSION_ECCODES}
DIR_ECCODES_INSTALL?=${OBJDIR_MASTER}/ECCODES-${VERSION_ECCODES}
ECCODES_MOD?=${DIR_ECCODES_INSTALL}/include/grib_api.mod
#
ifdef DIR_ECCODES_SRC
INC_ECCODES   ?= -I${DIR_ECCODES_INSTALL}/include
LIB_ECCODES   ?= -L${DIR_ECCODES_INSTALL}/lib -L${DIR_ECCODES_INSTALL}/lib64 -leccodes_f90 -leccodes
INC           += $(INC_ECCODES)
LIBS          += $(LIB_ECCODES)
VPATH         += $(DIR_ECCODES_INSTALL)/include
endif
endif

##########################################################
#           Librairie OASIS                              #
##########################################################
#
ifeq "$(VER_OASIS)" "OASISAUTO"
OASIS_PATH ?= ${SRC_MESONH}/src/LIB/work_oasis3-mct
OASIS_KEY ?= ${OASIS_PATH}/build/lib/psmile.MPI1/mod_oasis.mod
# INC_OASIS     : includes all *o and *mod for each library
INC_OASIS      ?= -I${OASIS_PATH}/build/lib/psmile.MPI1 -I$(OASIS_PATH)/build/lib/mct -I$(OASIS_PATH)/build/lib/scrip
LIB_OASIS      ?= -L${OASIS_PATH}/lib -lpsmile.MPI1 -lmct -lmpeu -lscrip
INC            += $(INC_OASIS)
LIBS           += $(LIB_OASIS)
VPATH          += ${OASIS_PATH}/build/lib/psmile.MPI1
CPPFLAGS       += -DCPLOASIS

endif

ifeq "$(VER_OASIS)" "OASISBASHRC"
OASIS_PATH ?= ${OASISDIR}
OASIS_KEY ?= ${OASIS_PATH}/build/lib/psmile.MPI1/mod_oasis.mod
# INC_OASIS     : includes all *o and *mod for each library
INC_OASIS      ?= -I${OASIS_PATH}/build/lib/psmile.MPI1 -I$(OASIS_PATH)/build/lib/mct -I$(OASIS_PATH)/build/lib/scrip
LIB_OASIS      ?= -L${OASIS_PATH}/lib -lpsmile.MPI1 -lmct -lmpeu -lscrip
INC            += $(INC_OASIS)
LIBS           += $(LIB_OASIS)
VPATH          += ${OASIS_PATH}/build/lib/psmile.MPI1
CPPFLAGS       += -DCPLOASIS
endif

ifeq "$(VER_OASIS)" "OASISDOCKER"
OASIS_PATH ?= ${OASIS_DIR}
OASIS_KEY ?= ${OASIS_PATH}/build/lib/psmile.MPI1/mod_oasis.mod
# INC_OASIS     : includes all *o and *mod for each library
INC_OASIS      ?= -I${OASIS_PATH}/build/lib/psmile.MPI1 -I$(OASIS_PATH)/build/lib/mct -I$(OASIS_PATH)/build/lib/scrip
LIB_OASIS      ?= -L${OASIS_PATH}/lib -lpsmile.MPI1 -lmct -lmpeu -lscrip
INC            += $(INC_OASIS)
LIBS           += $(LIB_OASIS)
VPATH          += ${OASIS_PATH}/build/lib/psmile.MPI1
CPPFLAGS       += -DCPLOASIS
endif

##########################################################
#           Librairie NETCDF4                            #
##########################################################
# NETCDF4 INPUT/OUTPUT in MesoNH 
ifdef MNH_IOCDF4
CPPFLAGS_MNH += -DMNH_IOCDF4
else
VER_CDF="NONE"
endif
#
# NetCDF  : AUTO install of netcdf-4.X.X on PC linux to avoid problem with compiler
#  
#
ifeq "$(VER_CDF)" "CDFAUTO"
DIR_CDFC?=${SRC_MESONH}/src/LIB/netcdf-c-${VERSION_CDFC}
DIR_CDFCXX?=${SRC_MESONH}/src/LIB/netcdf-cxx4-${VERSION_CDFCXX}
DIR_CDFF?=${SRC_MESONH}/src/LIB/netcdf-fortran-${VERSION_CDFF}
CDF_PATH?=${OBJDIR_MASTER}/NETCDF-${VERSION_CDFF}
CDF_MOD?=${CDF_PATH}/include/netcdf.mod
#
INC_NETCDF     ?= -I${CDF_PATH}/include
LIB_NETCDF     ?= -L${CDF_PATH}/lib -L${CDF_PATH}/lib64 -lnetcdff -lnetcdf -lhdf5hl_fortran -lhdf5_hl -lhdf5_fortran -lhdf5 -lsz -laec -lz -ldl
#
INC            += $(INC_NETCDF)
LIBS           += $(LIB_NETCDF)
#
DIR_HDF?=${SRC_MESONH}/src/LIB/hdf5-${VERSION_HDF}
#
DIR_LIBAEC?=${SRC_MESONH}/src/LIB/libaec-${VERSION_LIBAEC}

endif
#
# NetCDF : CDF this docker image
#
ifeq "$(VER_CDF)" "CDFDOCKER"
#
INC_NETCDF     ?= $(shell $(NETCDF_DIR)/bin/nf-config --fflags)
LIB_NETCDF     ?= $(shell $(NETCDF_DIR)/bin/nf-config --flibs)
#
INC            += $(INC_NETCDF)
LIBS           += $(LIB_NETCDF)
#
endif

#
# NetCDF : CDF LaReunion Local
#
ifeq "$(VER_CDF)" "CDFBASHRC"
#
INC_NETCDF     ?= $(shell $(NETCDF_CONFIG) --fflags)
LIB_NETCDF     ?= $(shell $(NETCDF_CONFIG) --flibs)
#
INC            += $(INC_NETCDF)
LIBS           += $(LIB_NETCDF)
#
endif
#
# NetCDF in beaufix (bull meteo-france)
ifeq "$(VER_CDF)" "CDFBFIX"
CDF_PATH?=/opt/softs/libraries/ICC16.1.150/netcdf-4.4.0
INC_NETCDF     ?= -I${CDF_PATH}/include
LIB_NETCDF     ?= -L${CDF_PATH}/lib -lnetcdf -lnetcdff
INC            += $(INC_NETCDF)
LIBS           += $(LIB_NETCDF)
endif
#
# NetCDF in BGQ
#
ifeq "$(VER_CDF)" "CDFBGQ"
CDF_PATH?=/bglocal/cn/pub/NetCDF/4.3.3.1/seq
INC_NETCDF     ?= -I${CDF_PATH}/include
LIB_NETCDF     ?= -L${CDF_PATH}/lib -lnetcdff -lnetcdf_c++ -lnetcdf
INC            += $(INC_NETCDF)
LIBS           += $(LIB_NETCDF)
HDF5_PATH?=/bglocal/cn/pub/HDF5/1.8.14/seq/
LIB_HDF5       ?= -L${HDF5_PATH}/lib -lhdf5hl_fortran -lhdf5_hl -lhdf5_fortran -lhdf5 -lm
LIBS           += $(LIB_HDF5)
LIBZ_PATH?=/bglocal/cn/pub/zlib/1.2.5
LIB_LIBZ       ?= -L${LIBZ_PATH}/lib -lz
LIBS           += $(LIB_LIBZ)

endif
#
# NetCDF in SGI ICE
#
ifeq "$(VER_CDF)" "CDFICE"
CDF_PATH?=/opt/software/SGI/netcdf/4.0
INC_NETCDF     ?= -I${CDF_PATH}/include
LIB_NETCDF     ?= -L${CDF_PATH}/lib -lnetcdff  -lnetcdf -i_dynamic 
INC            += $(INC_NETCDF)
LIBS           += $(LIB_NETCDF)
endif
#
# NetCDF in NEC SX
#
ifeq "$(VER_CDF)" "CDFSX"
CDF_PATH?=/SXlocal/pub/netcdf/3.6.1
INC_NETCDF     ?= -I${CDF_PATH}/include
LIB_NETCDF     ?= -L${CDF_PATH}/lib -lnetcdf_c++ -lnetcdf
INC            += $(INC_NETCDF)
LIBS           += $(LIB_NETCDF)
endif
#
ifeq "$(VER_CDF)" "CDFMFSX"
CDF_PATH?=/usr/local/SX/lib/NETCDF_size_t32
INC_NETCDF     ?= -I${CDF_PATH}/include
LIB_NETCDF     ?= -L${CDF_PATH}/lib -lnetcdf
INC            += $(INC_NETCDF)
LIBS           += $(LIB_NETCDF)
endif
#
# NetCDF in prefix (bull meteo-france)
ifeq "$(VER_CDF)" "CDFBULL"
CDF_PATH?=/home_nfs/local/Icc13.0.1/netcdf-4.2.1.1
INC_NETCDF     ?= -I${CDF_PATH}/include
LIB_NETCDF     ?= -L${CDF_PATH}/lib -lnetcdf -lnetcdff
INC            += $(INC_NETCDF)
LIBS           += $(LIB_NETCDF)
endif
#

# NetCDF in AIX S
#
ifeq "$(VER_CDF)" "CDFAIX"
CDF_PATH?=/usr/local/pub/NetCDF/3.6.2
INC_NETCDF     ?= -I${CDF_PATH}/include
LIB_NETCDF     ?= -L${CDF_PATH}/lib -lnetcdf_c++ -lnetcdf
INC            += $(INC_NETCDF)
LIBS           += $(LIB_NETCDF)
endif

#
# Linux with gfortran SUSE10.3
#
ifeq "$(VER_CDF)" "CDFGFOR"
INC_NETCDF     ?=  -I/usr/include
LIB_NETCDF     ?=  -lnetcdf -lnetcdff /usr/lib64/libgfortran.so.2
#LIB_NETCDF     ?=  -lnetcdf -lnetcdff 
INC            += $(INC_NETCDF)
LIBS           += $(LIB_NETCDF)
endif

#
# Linux with netcdf CTI 3.6.3
#
ifeq "$(VER_CDF)" "CDFCTI"
CDF_PATH?=/usr
INC_NETCDF     = -I${CDF_PATH}/include
LIB_NETCDF     = -L${CDF_PATH}/lib64 -lnetcdff -lnetcdf -lhdf5_hl -lhdf5 -lsz -lz
INC            += $(INC_NETCDF)
LIBS           += $(LIB_NETCDF)
endif

#
# Linux with gfortran SUSE11.1
#
ifeq "$(VER_CDF)" "CDF3GFOR"
CDF_PATH       ?=/opt/netcdf3
INC_NETCDF     ?=  -I${CDF_PATH}/include
LIB_NETCDF     ?=  -L${CDF_PATH}/lib64  -lnetcdf_c++ -lnetcdf
INC            +=  $(INC_NETCDF)
LIBS           +=  $(LIB_NETCDF)
endif

# for oasis compilation <=> to find correctly netcdf
NETCDF_INCLUDE ?= ${CDF_PATH}/include
NETCDF_LIBRARY ?= $(LIB_NETCDF)
export NETCDF_INCLUDE NETCDF_LIBRARY F90 CC

##########################################################
#           Number of NESTED MODEL                       #
##########################################################
NSOURCE=8
##########################################################
#                                                        #
# PROG_LIST : Main program liste to compile              #
#                                                        #
##########################################################
#
#ifeq "$(ARCH)" "BGQ"
#PROG_LIST += MESONH PREP_IDEAL_CASE PREP_PGD
#else
PROG_LIST += MESONH  LATLON_TO_XY PREP_IDEAL_CASE PREP_REAL_CASE PREP_PGD \
            PREP_NEST_PGD SPAWNING DIAG PREP_SURFEX ZOOM_PGD SPECTRE \
	    MNH2LPDM
ifdef MNH_TOOLS
PROG_LIST += LFI2CDF
endif
#endif
##########################################################
#                                                        #
# LIB_OBJS : Librarie of all *.o                         #
#                                                        #
##########################################################
#
ARCH_XYZ        := $(ARCH_XYZ)-$(OPTLEVEL)
OBJDIR_ROOT     := $(OBJDIR_ROOT)-$(ARCH_XYZ)
LIB_OBJS_ROOT   := $(LIB_OBJS_ROOT)-$(ARCH_XYZ)
#
##########################################################
#                                                        #
# IGNORE_OBJS : some *.o to ignore                       #
#       ---> unused unsupported old routines             #
#                                                        #
##########################################################
#
IGNORE_OBJS += spll_olwu.o spll_olwv.o spll_rad1driv.o spll_radlsw.o spll_suovlp.o \
            spll_ch_init_model0d.o spll_ch_model0d.o spll_ch_svode_fcn.o spll_ch_svode_jac.o
IGNORE_DEP_MASTER += modules_diachro.D
IGNORE_DEP_MASTER += ch_svode.D ch_model0d.D  \
          create_file.D def_var_netcdf.D get_dimlen_netcdf.D \
          handle_err.D init_outfn_isban.D init_outfn_sean.D \
          init_outfn_surf_atmn.D init_outfn_tebn.D init_outfn_watern.D \
          ol_find_file.D ol_read_atm.D ol_time_interp_atm.D \
          read_surf_ol.D write_surf_ol.D \
close_file_ol.D close_namelist_ol.D end_io_surf_oln.D \
init_io_surf_oln.D modd_io_surf_ol.D modd_ol_fileid.D \
open_file_ol.D open_namelist_ol.D read_surf_ol.D write_surf_ol.D offline.D

#
#
##########################################################
#                                                        #
#  VPATH_EXCLUDE : Some sources directory to exclude     #
#                                                        #
##########################################################
#
VPATH_EXCLUDE= %/CVS
#



