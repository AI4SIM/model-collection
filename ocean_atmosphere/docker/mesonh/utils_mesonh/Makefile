#MNH_LIC Copyright 1994-2021 CNRS, Meteo-France and Universite Paul Sabatier
#MNH_LIC This is part of the Meso-NH software governed by the CeCILL-C licence
#MNH_LIC version 1. See LICENSE, CeCILL-C_V1-en.txt and CeCILL-C_V1-fr.txt
#MNH_LIC for details. version 1.
##########################################################
#                                                        #
#  MACRO : dependance_flist                              #
#          generate the dependance list for ALL fortran  #
#          "*.f*" files                                  #
#          searching in PATH=$(1)                        #
##########################################################
#
dependance_flist = $(notdir $(shell find $(1) -follow -type f \
               -name '*.f' -o -name '*.f90' -o -name '*.fx90' -o -name '*.F90'\
               | grep -v ',v' | sed -e 's/\.f.*/.D/g' -e 's/\.F.*/.D/g' ))
obj_flist        = $(notdir $(shell find $(1) -follow -type f \
               -name 'spll_*.f*' | sed -e 's/\(.*\)\(\.\).*/\1.o/g' )) 
#
##########################################################
#                                                        #
# COMPILER & ARCHITECTURE CONFIGURATION                  #
#                                                        #
##########################################################
#
include Rules.$(ARCH)$(F).mk
#
##########################################################
##########################################################
##########################################################
#                                                        #
#    DONT MODIFY LINES AFTER THIS  !!!!!!!!!!!!!         #         
#                                                        #
##########################################################
##########################################################
##########################################################

##########################################################
#                                                        #
#    AUTOMATIQUE  DEPENDANCE GENERATION !!!!!!!!!!!!!    #
#                                                        #
#   "*.D" files contain dependance deduceed              #
#   form all "*.f*" FORTRAN sources files                #
#                                                        #   
# "use" key word                                         #
#     --> dependance on "*.mod" module files             #
# "module" key word key word                             #
#     --> creation of"*.mod"  module files               #
#                                                        #
#   The "*.D" list of files is contructed from every     #
#   directories of fortran sources :                     #
#       ---> user, master , bugfix , etc...              #
#   with or without RCS version control                  #
#                                                        #
#   The "*.D" list is reconstructed automaticaly         #
#   by the commande "spll" , each time a FORTRAN source  #
#   are modified or added .                              #
#                                                        #
#   All modification are allowed !!!!!                   #
#   adding new subroutines                               #
#       or                                               #
#   adding new modules                                   #
#                                                        #
#   REM : if during modification, you deleting some      #
#         FORTRAN subroutines you must also deleted the  #
#         "spll_*.o" files created by this one by hand   #
#         to avoid potential link problems               #
#                                                        #
##########################################################

##########################################################
#                                                        #
# DEP_MASTER : Automatique -->                           #
#    Dependance file of MASTER source                    #
#                                                        #
##########################################################
#
OBJDIR_MASTER=$(OBJDIR_ROOT)/MASTER
DEP_MASTER= $(call dependance_flist,$(DIR_MASTER))
#
#  init VPATH depending on compilation phase
#
ifdef DO_DEP_MASTER
VPATH += $(shell find $(DIR_MASTER) -follow -type d )
VPATH += $(OBJDIR_MASTER)
endif
#
ifdef DO_COMP_MASTER
VPATH += $(OBJDIR_MASTER) $(OBJDIR_MASTER)/MOD
INC += -I$(B)$(OBJDIR_MASTER)/MOD
endif
#
##########################################################
#                                                        #
# DEP_USER : Automatique -->                             #
#    Dependance file of MY source                        #
#                                                        #
##########################################################
#
ifdef VER_USER
OBJDIR_USER=$(OBJDIR_ROOT)/$(VER_USER)
#
#  init VPATH depending on compilation phase
#
ifdef DO_DEP_USER
DEP_USER= $(call dependance_flist,$(DIR_USER))
VPATH += $(shell find $(DIR_USER) -follow -type d )
VPATH += $(OBJDIR_USER)
endif
#
ifdef DO_COMP_USER
VPATH += $(OBJDIR_USER) $(OBJDIR_USER)/MOD $(OBJDIR_MASTER) $(OBJDIR_MASTER)/MOD
INC += -I$(B)$(OBJDIR_USER)/MOD -I$(B)$(OBJDIR_MASTER)/MOD
endif
#
endif
##########################################################
#                                                        #
# VPATH ADJUSTEMENT :                                    #
#                                                        #
##########################################################
#
#
#VPATH := $(filter-out $(VPATH_EXCLUDE),$(VPATH))
#
##########################################################
#                                                        #
# OBJS : all *.o to compile                              #
#                                                        #
##########################################################
#
# MASTER 
#
OBJS_LISTE_MASTER :=  $(OBJS_LISTE_MASTER) $(notdir $(shell find $(OBJDIR_MASTER) -follow -type f -name "spll_*.f*" | sed -e 's/\(.*\)\(\.\).*/\1.o/g' )) 
#OBJS_LISTE_MASTER :=  $(OBJS_LISTE_MASTER) $(obj_flist $(OBJDIR_MASTER)) 
#
OBJS_LISTE_MASTER := $(sort $(filter-out $(IGNORE_OBJS),$(OBJS_LISTE_MASTER))) 
#
LIB_MASTER=$(OBJDIR_MASTER)/$(LIB_OBJS_ROOT)-MASTER.a
#
# USER
#
ifdef OBJDIR_USER
OBJS_LISTE_USER :=  $(OBJS_LISTE_USER) $(notdir $(shell find $(OBJDIR_USER) -follow -type f -name "spll_*.f*" | sed -e 's/\(.*\)\(\.\).*/\1.o/g' )) 
#OBJS_LISTE_USER :=  $(OBJS_LISTE_USER) $(obj_flist $(OBJDIR_USER)) 
#
OBJS_LISTE_USER := $(sort $(filter-out $(IGNORE_OBJS), $(OBJS_LISTE_USER) $(OBJS_LISTE_MASTER) )) 
#
LIB_USER=$(OBJDIR_USER)/$(LIB_OBJS_ROOT)-$(VER_USER).a
endif
##########################################################
#                                                        #
# DEP_ALL : All dependance *.D to generate               #
#                                                        #
##########################################################
#
DEP_ALL_MASTER = $(sort $(filter-out $(IGNORE_DEP_MASTER),$(DEP_MASTER)) )
#
DEP_ALL_USER   = $(sort $(filter-out $(IGNORE_DEP_USER)  ,$(DEP_USER)) )
#

.PHONY : clean objdir depall objs lib prog print

.DELETE_ON_ERROR:  

.SUFFIXES:
.SUFFIXES:  .o .f90 .f .fx90 .mod .D .a

.INTERMEDIATE:   $(LIB_MASTER)

.SECONDARY: iso_fortran_env.mod mpi.mod netcdf.mod openacc.mod

##########################################################
#                                                        #
# MASTER RULES                                           #
#                                                        #
##########################################################
.PHONY : all objdirmaster libmaster bibmaster objmaster progmaster
.PHONY : installmaster cleanlibmaster cleanmaster cleanobjmaster cleanprogmaster

ifeq "$(MNH_GRIBAPI)" "yes"
all : gribapi progmaster
else
all : eccodes_lib progmaster
endif

objdirmaster :  $(OBJDIR_MASTER)/.dummy

$(OBJDIR_MASTER)/.dummy :
	-mkdir -p $(OBJDIR_MASTER)/.dummy
	-mkdir -p $(OBJDIR_MASTER)/MOD

depmaster : objdirmaster 
	@$(MAKE) DO_DEP_MASTER=YES OBJDIR=$(OBJDIR_MASTER) filedepallmaster

ifeq "$(VER_CDF)" "CDFAUTO"
depmaster : cdf
endif

ifeq "$(VER_OASIS)" "OASISAUTO"
depmaster : oasis
endif

ifeq "$(MNH_GRIBAPI)" "yes"
depmaster : gribapi
else
depmaster : eccodes_lib
endif

filedepallmaster :  $(DEP_ALL_MASTER)
	find  $(OBJDIR_MASTER) -follow -name "*.D" >  $(OBJDIR_MASTER)/filemaster
	rm -f  $(OBJDIR_MASTER)/filedepallmaster
	cat $(OBJDIR_MASTER)/filemaster  | xargs cat  >>  $(OBJDIR_MASTER)/filedepallmaster

libmaster :  depmaster
	@$(MAKE)  -I$(B)$(OBJDIR_MASTER)  DO_COMP_MASTER=YES OBJDIR=$(OBJDIR_MASTER) objmaster

bibmaster :  libmaster
	@$(MAKE)  -I$(B)$(OBJDIR_MASTER)  DO_COMP_MASTER=YES OBJDIR=$(OBJDIR_MASTER) $(LIB_MASTER)
	mv $(LIB_MASTER)  $(subst lib-,bib-,$(LIB_MASTER))

objmaster :  $(OBJS_LISTE_MASTER)

$(LIB_MASTER) :  $(OBJS_LISTE_MASTER)
	cd $(OBJDIR_MASTER) ; find . -follow -name '*.o'| xargs $(AR) r $(LIB_MASTER) 

progmaster : libmaster
	@$(MAKE) -I$(B)$(OBJDIR_MASTER)  DO_COMP_MASTER=YES  \
	OBJDIR=$(OBJDIR_MASTER) \
	LIB_MNH=$(LIB_MASTER) \
	PROG_DIR=$(OBJDIR_MASTER) \
	$(PROG_LIST) cleanlibmaster

installmaster : libmaster
	@$(MAKE) -I$(B)$(OBJDIR_MASTER)  DO_COMP_MASTER=YES  OBJDIR=$(OBJDIR_MASTER) install

cleanlibmaster :
	rm -f  $(LIB_MASTER) 

cleanmaster : 
	test -d $(OBJDIR_MASTER) && rm -fr  $(OBJDIR_MASTER)

cleanobjmaster :  
	test -d $(OBJDIR_MASTER) && find $(OBJDIR_MASTER) -follow -name "*.o" -o -name "*.mod" | xargs rm 

cleanprogmaster :
	test -d $(OBJDIR_MASTER) && cd $(OBJDIR_MASTER) && rm -f $(PROG_LIST)

##########################################################
#                                                        #
#  USER RULES                                            #
#                                                        #
##########################################################

user : proguser

objdiruser :  $(OBJDIR_USER)/.dummy

$(OBJDIR_USER)/.dummy :
	-mkdir -p $(OBJDIR_USER)/.dummy
	-mkdir -p $(OBJDIR_USER)/MOD

depuser : objdiruser
	@$(MAKE) DO_DEP_USER=YES OBJDIR=$(OBJDIR_USER) filedepalluser

filedepalluser : $(DEP_ALL_USER)
	find  $(OBJDIR_USER) -follow -name "*.D" >  $(OBJDIR_USER)/fileuser 
	cat $(OBJDIR_USER)/fileuser | sed -e "s!$(OBJDIR_USER)!$(OBJDIR_MASTER)!g"  >  $(OBJDIR_USER)/filenotmaster
	grep -F -v -f   $(OBJDIR_USER)/filenotmaster  $(OBJDIR_MASTER)/filemaster >  $(OBJDIR_USER)/filemaster
	rm -f  $(OBJDIR_USER)/filedepalluser
	cat $(OBJDIR_USER)/fileuser $(OBJDIR_USER)/filemaster | xargs cat >>  $(OBJDIR_USER)/filedepalluser


libuser :   depuser 
	@$(MAKE) -I$(B)$(OBJDIR_USER) DO_COMP_USER=YES OBJDIR=$(OBJDIR_USER) objuser

objuser :  $(OBJS_LISTE_USER)

$(LIB_USER) :  $(OBJS_LISTE_USER)
	rm -f $(LIB_USER)
	cd $(OBJDIR_MASTER) ; find . -name "*.o" | xargs $(AR) r $(LIB_USER) 
	cd $(OBJDIR_USER)   ; find . -name "*.o" | xargs $(AR) r $(LIB_USER) 

proguser : libuser
	@$(MAKE) -I$(B)$(OBJDIR_USER)  DO_COMP_USER=YES \
	OBJDIR=$(OBJDIR_USER) \
	LIB_MNH=$(LIB_USER) \
	PROG_DIR="$(OBJDIR_USER) $(OBJDIR_MASTER)" \
	$(PROG_LIST) # cleanlibuser

cleanlibuser :
	rm -f  $(LIB_USER) 

installuser : 
	@$(MAKE)  -I$(B)$(OBJDIR_USER) DO_COMP_USER=YES  OBJDIR=$(OBJDIR_USER) install	

cleanproguser :
	test -d $(OBJDIR_USER) && cd $(OBJDIR_USER) && rm -f $(PROG_LIST)

##########################################################
#                                                        #
# EXTRA LIB : GRIBAPI                                    #
#                                                        #
##########################################################
ifneq "$(findstring 64,$(shell uname -m))" ""
A64=A64
endif
gribapi : $(GRIBAPI_INC)
$(GRIBAPI_INC) :
	cd ${DIR_GRIBAPI} && ./configure --disable-shared --disable-jpeg --prefix=${GRIBAPI_PATH} CC="$(CC)" \
	FC="$(FC)" FCFLAGS="$(GRIB_FLAGS)" ${GRIBAPI_CONF} && $(MAKE) -j 1 clean && \
	$(MAKE) -j 1 && $(MAKE) -j 1 install && $(MAKE) -j 1 clean

gribapi_clean :
	- [  -d ${GRIBAPI_PATH} ] && rm -fr ${GRIBAPI_PATH}
##########################################################
#                                                        #
# EXTRA LIB : ecCodes                                    #
#                                                        #
##########################################################
eccodes_lib : $(ECCODES_MOD)
$(ECCODES_MOD) :
	- [ ! -d $(DIR_ECCODES_BUILD) ] && mkdir -p $(DIR_ECCODES_BUILD)
	cd ${DIR_ECCODES_BUILD} && \
	cmake ${DIR_ECCODES_SRC} -DCMAKE_INSTALL_PREFIX=${DIR_ECCODES_INSTALL} -DBUILD_SHARED_LIBS=OFF \
	-DENABLE_NETCDF=OFF -DENABLE_JPG=OFF -DENABLE_PYTHON=OFF -DENABLE_EXAMPLES=OFF \
	-DCMAKE_Fortran_COMPILER=$(FC) -DCMAKE_C_COMPILER=$(CC) \
	-DCMAKE_Fortran_FLAGS=$(ECCODES_FFLAGS) -DCMAKE_C_FLAGS=$(ECCODES_CFLAGS) && \
	$(MAKE) && $(MAKE) install && $(MAKE) clean

eccodes_lib_clean :
	- [  -d ${DIR_ECCODES_BUILD} ] && rm -fr ${DIR_ECCODES_BUILD}
	- [  -d ${DIR_ECCODES_INSTALL} ] && rm -fr ${DIR_ECCODES_INSTALL}
##########################################################
#                                                        #
# EXTRA LIB : NETCDF                                     #
#                                                        #
##########################################################
ifeq "$(VER_CDF)" "CDFAUTO"
#
HDF_OPT    ?= ${OPT_BASE_I4:-$OPT_BASE}
NETCDF_OPT ?= ${OPT_BASE_I4:-$OPT_BASE}
#
cdf : $(CDF_MOD)
$(CDF_MOD) :
	cd ${DIR_LIBAEC} && ./configure --disable-shared --prefix=${CDF_PATH} --libdir=${CDF_PATH}/lib64 CC="$(CC)" CFLAGS="$(HDF_OPT)" && \
	$(MAKE) && $(MAKE) install && $(MAKE) clean
	cd ${DIR_HDF} && ./configure --enable-parallel --enable-fortran --disable-shared --prefix=${CDF_PATH} --libdir=${CDF_PATH}/lib64 --with-szlib=${CDF_PATH}/include,${CDF_PATH}/lib64 \
	CC=mpicc CFLAGS="$(HDF_OPT)" FC=mpif90 FCFLAGS="$(NETCDF_OPT)" LDFLAGS="-L${CDF_PATH}/lib64" LIBS="-lsz -laec -lz" && \
	$(MAKE) && $(MAKE) install && $(MAKE) clean
	cd ${DIR_CDFC} && ./configure --enable-parallel --disable-shared --prefix=${CDF_PATH} --libdir=${CDF_PATH}/lib64 --disable-dap \
	CC=mpicc CFLAGS="$(NETCDF_OPT)" CPPFLAGS="${INC_NETCDF}" ${CDF_CONF} LDFLAGS="-L${CDF_PATH}/lib64" LIBS="-lhdf5_hl -lhdf5 -lsz -laec -lz" && \
	$(MAKE) && $(MAKE) install && $(MAKE) clean
ifdef MNH_FOREFIRE
	cd ${DIR_CDFCXX} && ./configure --enable-parallel --disable-shared --prefix=${CDF_PATH} --libdir=${CDF_PATH}/lib64 \
	CXX=mpic++ CXXFLAGS="$(NETCDF_OPT)" FC=mpif90 FCFLAGS="$(NETCDF_OPT) $(NETCDF_SUPPFLAGS)" FFLAGS="$(NETCDF_OPT)"  \
	CPPFLAGS="${INC_NETCDF}" ${CDF_CONF} LDFLAGS="-L${CDF_PATH}/lib64" LIBS="-lnetcdf -lhdf5_hl -lhdf5 -lsz -laec -lz" && \
	$(MAKE) && $(MAKE) install && $(MAKE) clean
endif
	cd ${DIR_CDFF} && ./configure --enable-parallel --disable-shared --prefix=${CDF_PATH} --libdir=${CDF_PATH}/lib64 \
	CC=mpicc CFLAGS="$(NETCDF_OPT)" FC=mpif90 FCFLAGS="$(NETCDF_OPT) $(NETCDF_SUPPFLAGS)" FFLAGS="$(NETCDF_OPT)"  \
	CPPFLAGS="${INC_NETCDF}" ${CDF_CONF} LDFLAGS="-L${CDF_PATH}/lib64" LIBS="-lnetcdf -lhdf5_hl -lhdf5 -lsz -laec -lz" && \
	$(MAKE) && $(MAKE) install && $(MAKE) clean

cleanmaster : cleancdf
cleancdf :
	- [ -d ${CDF_PATH} ] && rm -fr  ${CDF_PATH}
	cd ${DIR_LIBAEC} && $(MAKE) clean
	cd ${DIR_HDF} && $(MAKE) clean 
	cd ${DIR_CDFC} && $(MAKE) clean 
ifdef MNH_FOREFIRE
	cd ${DIR_CDFCXX} && $(MAKE) clean 
endif
	cd ${DIR_CDFF} && $(MAKE) clean
endif
##########################################################
#                                                        #
# EXTRA LIB : OASIS                                      #
#                                                        #
##########################################################
ifeq "$(VER_OASIS)" "OASISAUTO"
#
oasis : $(OASIS_KEY)
$(OASIS_KEY): 
	cd ${SRC_MESONH}/src/LIB/oasis3-mct_v3/util/make_dir && make realclean -f TopMakefileOasis3 && make -f TopMakefileOasis3 && cd ${SRC_MESONH}/src/LIB/toy_v1-0 && make clean && make

cleanmaster : cleanoasis
cleanoasis :
	- [ -d ${OASIS_PATH} ] && rm -fr  ${OASIS_PATH}
endif
##########################################################
#                                                        #
# EXTRA LIB : S4PY                                       #
#                                                        #
##########################################################
ifdef MNH_S4PY
all : libs4py.so

OBJ_S4PY=$(OBJDIR_MASTER)/spll_wcompress_field.o $(OBJDIR_MASTER)/spll_wdecompress_field.o \
$(OBJDIR_MASTER)/spll_wget_compheader.o $(OBJDIR_MASTER)/spll_wlficas.o \
$(OBJDIR_MASTER)/spll_wlfiecr.o $(OBJDIR_MASTER)/spll_wlfifer.o \
$(OBJDIR_MASTER)/spll_wlfilec.o $(OBJDIR_MASTER)/spll_wlfinaf.o \
$(OBJDIR_MASTER)/spll_wlfinfo.o $(OBJDIR_MASTER)/spll_wlfiouv.o $(OBJDIR_MASTER)/spll_wlfipos.o \
$(OBJDIR_MASTER)/spll_NEWLFI_ALL.o $(OBJDIR_MASTER)/spll_lockasgn.o \
$(OBJDIR_MASTER)/spll_lockoff.o $(OBJDIR_MASTER)/spll_lockon.o $(OBJDIR_MASTER)/spll_lockrel.o \
$(OBJDIR_MASTER)/fswap8buff.o $(OBJDIR_MASTER)/spll_remark2.o

libs4py.so : progmaster
	$(F90) -shared $(LDFLAGS) -o $(OBJDIR_MASTER)/$@ $(OBJ_S4PY) $(LIBS)
endif
##########################################################
#                                                        #
# PROGRAM RULES                                          #
#                                                        #
##########################################################
ifdef PROG_LIST

prog :   lib
	@$(MAKE) -I$(B)$(OBJDIR) DEP=YES  $(PROG_LIST)  

install : $(patsubst %,%-${ARCH_XYZ},$(PROG_LIST))


$(PROG_LIST) : OBJ_PROG=$(shell find $(PROG_DIR) -follow -type f -name "spll_*.f*" \
              | xargs grep -l -E -i "^[[:space:]]*program *$@" |  sed -e 's/\.f.*/.o/g' | head -1 \
	      | xargs basename | xargs -I{} find $(PROG_DIR) -follow -name {} -print | head -1 )  

$(PROG_LIST) :  $(LIB_MNH)
#	echo OBJ_PROG=$(OBJ_PROG)
	$(F90) $(LDFLAGS) -o $(OBJDIR)/$@ $(OBJ_PROG) $(LIB_MNH) $(LIBS)

%-${ARCH_XYZ} :%
ifndef DO_COMP_USER
	ln -sf $(OBJDIR)/$(*F) $(SRC_MESONH)/exe/$(*F)-${ARCH_XYZ}
else
#	ln -sf $(OBJDIR)/$(*F) $(SRC_MESONH)/exe/$(*F)-${ARCH_XYZ}-${VER_USER}
	ln -sf $(OBJDIR)/$(*F) $(SRC_MESONH)/exe/$(*F)${XYZ}
endif
	rm -f $(SRC_MESONH)/exe/$(*F)_${ARCH_XYZ}~

##########################################################
#                                                        #
# EXAMPLES RULES                                         #
#                                                        #
##########################################################

PWD_KTEST=$(SRC_MESONH)/MY_RUN/KTEST

EXAMPLES=001_2Drelief 002_3Drelief 003_KW78 004_Reunion 005_ARM 007_16janvier 014_LIMA
examples : $(EXAMPLES)

EXAMPLES_CHIMIE=009_ICARTT 011_KW78CHEM
chimie : $(EXAMPLES_CHIMIE)

$(EXAMPLES) $(EXAMPLES_CHIMIE) 012_dust :
	cd $(PWD_KTEST)/$@ ; run_$@

EXAMPLES_NCL=004_Reunion 007_16janvier

ncl1_examples :  $(patsubst %,%_ncl1,$(EXAMPLES_NCL))

ncl2_examples :  $(patsubst %,%_ncl2,$(EXAMPLES_NCL))

examplesclean :  $(patsubst %,%_clean,$(EXAMPLES))


%_ncl1 : 
	cd $(PWD_KTEST)/$(*F) ; run_$(*F) all_ncl_extractdia

%_ncl2 : 
	cd $(PWD_KTEST)/$(*F) ; run_$(*F) all_ncl_lfi2cdf

%_clean : 
	cd $(PWD_KTEST)/$(*F) ; $(MAKE) clean

#
# rules to remove spll* source on MASTER directory
#
purgelist : $(patsubst %,purge_%,$(notdir $(PL)))
purge_%.f90 :
	cd $(OBJDIR_MASTER) ; grep -l "depfile:$(*F).D" *.f* | xargs rm -f ; rm -f $(*F).D ; rm -f filedepallmaster filemaster 
purge_%.fx90 :
	cd $(OBJDIR_MASTER) ; grep -l "depfile:$(*F).D" *.f* | xargs rm -f ; rm -f $(*F).D ; rm -f filedepallmaster filemaster 
purge_%.f :
	cd $(OBJDIR_MASTER) ; grep -l "depfile:$(*F).D" *.f* | xargs rm -f ; rm -f $(*F).D ; rm -f filedepallmaster filemaster 

#
# rules to remove spll* source on USER directory
#
purgelistu : $(patsubst %,purgeu_%,$(notdir $(PL)))
purgeu_%.f90 :
	cd $(OBJDIR_USER) ; grep -l "depfile:$(*F).D" *.f* | xargs rm -f ; rm -f $(*F).D ; rm -f filedepalluse fileuser
purgeu_%.fx90 :
	cd $(OBJDIR_USER) ; grep -l "depfile:$(*F).D" *.f* | xargs rm -f ; rm -f $(*F).D ; rm -f filedepalluse fileuser
purgeu_%.f :
	cd $(OBJDIR_USER) ; grep -l "depfile:$(*F).D" *.f* | xargs rm -f ; rm -f $(*F).D ; rm -f filedepalluse fileuser

stat_f90 :
	cd $(OBJDIR_MASTER) ; find . -name "*.f90" | wc -l
stat_o :
	cd $(OBJDIR_MASTER) ; find . -name "*.o" | wc -l

endif





cleanuser :  
	test -d $(OBJDIR_USER) && rm -fr  $(OBJDIR_USER)

clean :  examplesclean eccodes_lib_clean
	rm -fr  $(OBJDIR_ROOT)
cleanprog :
	cd $(OBJDIR_ROOT) ; rm -f $(PROG_LIST)

print :
	@echo $(VAR)=$($(VAR))

ARFLAGS=r
#(%.o): %.o
#	$(AR) $(ARFLAGS)   $(OBJDIR)/$(@F) $(*F).o
#	$(RM) $(*F).o

#
#  GENERATION OF DEPENDANCE FILES : *.D
# 

%.D:%.F90
	$(CPP) $(INC) $(CPPFLAGS) $< >  $(OBJDIR)/$(*F).f90 
	spll $(NSOURCE) $(OBJDIR)/$(*F).f90 
	rm -f $(OBJDIR)/$(*F).f90 

%.D:%.f90
	$(CPP) $(INC) $(CPPFLAGS) $< >  $(OBJDIR)/$(*F).f90 
	spll $(NSOURCE) $(OBJDIR)/$(*F).f90 
	rm -f $(OBJDIR)/$(*F).f90 

%.D:%.fx90
	$(CPP)  $(INC) $(CPPFLAGS) $< > $(OBJDIR)/$(*F).fx90
	spll $(NSOURCE) $(OBJDIR)/$(*F).fx90 
	rm -f $(OBJDIR)/$(*F).fx90
 
%.D:%.f
	$(CPP)  $(INC) $(CPPFLAGS) $< > $(OBJDIR)/$(*F).f
	spll $(NSOURCE) $(OBJDIR)/$(*F).f 
	rm -f $(OBJDIR)/$(*F).f
 
#
# GENERATION OF OBJET FILES : *.o
#

%.o:%.f90
	echo "inc=$(INC)"
	$(F90) -I$(OBJDIR) $(INC) -c $(F90FLAGS) $< 
	-mv  $(*F).o $(OBJDIR)/. || echo OK $(*F).o

%.o:%.f
	echo "inc=$(INC)"
	$(F77) -I$(OBJDIR) $(INC) -c $(F77FLAGS)  $<  
	-mv  $(*F).o $(OBJDIR)/. || echo OK $(*F).o

%.o:%.fx90
	cp $< $(OBJDIR)/$(*F).f 
	$(FX90) -I$(OBJDIR) $(INC) -c $(FX90FLAGS) $(OBJDIR)/$(*F).f   
	-mv  $(*F).o $(OBJDIR)/. || echo OK $(*F).o

%.o : %.c
	$(CC) $(INC) $(CFLAGS) $(CPPFLAGS) -c $< -o $(OBJDIR)/$(*F).o
	-mv  $(*F).o $(OBJDIR)/. || echo OK $(*F).o

ifeq "$(DO_COMP_MASTER)" "YES"
include  filedepallmaster
endif 

ifeq "$(DO_COMP_USER)" "YES"
include  filedepalluser
endif 

iso_fortran_env.mod:

mpi.mod:

netcdf.mod:

openacc.mod:

