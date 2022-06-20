! $Id: def_his.F 1571 2014-07-01 12:38:05Z gcambon $
!
!======================================================================
! CROCO is a branch of ROMS developped at IRD and INRIA, in France
! The two other branches from UCLA (Shchepetkin et al)
! and Rutgers University (Arango et al) are under MIT/X style license.
! CROCO specific routines (nesting) are under CeCILL-C license.
!
! CROCO website : http://www.croco-ocean.org
!======================================================================
!
#ifndef AVRH
# include "cppdefs.h"
!
      subroutine def_his (ncid, total_rec, ierr)
#else
      subroutine def_avg (ncid, total_rec, ierr)
#endif
!
! Create/open averages/history netCDF file. In the case when a new
! netCDF file is created, define all variables, their dimensions and
! attributes. In the case when a previously existing netCDF file is
! to be opened for addition of new data, verify that all dimensions
! of the file are consistent with the present model configuration
! and all necessary variables exist. Save netCDF IDs for all needed
! variables. Also determine size of the unlimited dimension.
!
! The difference between def_his and def_avg is as follows: they
! have different netCDF file name (hisname/avgname); netCDF file ID
! (passed as argument); time record index (hisindx/avgindx); array
! of switches which variables to write (wrthis/wrtavg); and different
! sets of netCDF variable IDs (hisTime...hisHbl/avgTime...avgHbl);
! and the first attribute of each variable, long_name, has prefix
! 'averaged'. Because most of the code is identical for both
! routines, the second one is generated from the first entirely
! by CPP.
!
#ifdef MUSTANG
      USE comMUSTANG, ONLY :nk_nivsed_out, nv_out3Dnv_specif,
     &                      nv_out2D_specif, nv_out3Dk_specif
#endif
      implicit none
      logical create_new_file
      integer ncid, total_rec, ierr, rec, lstr,lvar,lenstr, timedim
     &      , r2dgrd(3),  u2dgrd(3), v2dgrd(3), auxil(2), checkdims
     &      , b3dgrd(4)
#ifdef NC4PAR
     &      , csize,cmode
#endif
#if defined SEDIMENT || defined BBL
     &      , indxWrk
#endif
#ifdef SOLVE3D
     &      , r3dgrd(4),  u3dgrd(4), v3dgrd(4), w3dgrd(4), itrc,itrv
#endif
#ifdef MRL_WCI
     &      , iwkb
#endif
#ifdef MUSTANG
     &      , indx,ik
#endif

#include "param.h"
#include "scalars.h"
#include "ncscrum.h"
#include "netcdf.inc"
#ifdef NC4PAR
# include "mpi_cpl.h"
      include 'mpif.h'
#endif
      character*70 text

#ifndef AVRH
# define ncname hisname
# define rec_per_file nrpfhis
# define wrt wrthis
# define vidTime hisTime
# define vidTime2 hisTime2
# define vidTstep hisTstep
# define vidZ hisZ
# define vidUb hisUb
# define vidVb hisVb
# define vidU hisU
# define vidV hisV
# define vidT hisT
# define vidR hisR
# define vidbvf hisbvf
# define vidO hisO
# define vidW hisW
# define vidVisc hisVisc
# define vidDiff hisDiff
# define vidAkv hisAkv
# define vidAkt hisAkt
# define vidAks hisAks
# define vidHbl hisHbl
# define vidHbbl hisHbbl
# define vidTke hisTke
# define vidGls hisGls
# define vidLsc hisLsc
# define vidHm hisHm
# define vidWAVE hisWAVE
# define vidSUP hisSUP
# define vidUST2D hisUST2D
# define vidVST2D hisVST2D
# define vidUST hisUST
# define vidVST hisVST
# define vidWST hisWST
# define vidAkb hisAkb
# define vidAkw hisAkw
# define vidKVF hisKVF
# define vidCALP hisCALP
# define vidKAPS hisKAPS
# define vidHel hisHel
# define vidChC hisChC
# define vidU10 hisU10
# define vidKvO2 hisKvO2
# define vidO2sat hisO2sat
# define vidAOU hisAOU
# define vidwind10 hiswind10
# define vidSed hisSed
# define vidBBL hisBBL
# define vidBostr hisBostr
# define vidBustr hisBustr
# define vidBvstr hisBvstr
# define vidWstr hisWstr
# define vidUWstr hisUWstr
# define vidVWstr hisVWstr
# define vidShflx hisShflx
#ifdef SALINITY
#  define vidSwflx hisSwflx
#endif
#define vidShflx_rsw hisShflx_rsw
#if defined BULK_FLUX || defined OA_COUPLING
#  define vidShflx_rlw hisShflx_rlw
#  define vidShflx_lat hisShflx_lat
#  define vidShflx_sen hisShflx_sen
#endif
#if defined TEMPERATURE && defined BHFLUX
# define vidBhflx hisBhflx
#endif
# if defined SALINITY && defined BWFLUX
#  define vidBwflx hisBwflx
# endif
# ifdef SST_SKIN
#  define vidSST_skin hisSST_skin
# endif
# define vidM hisMust

#else

# define ncname avgname
# define rec_per_file nrpfavg
# define wrt wrtavg
# define vidTime avgTime
# define vidTime2 avgTime2
# define vidTstep avgTstep
# define vidZ avgZ
# define vidUb avgUb
# define vidVb avgVb
# define vidU avgU
# define vidV avgV
# define vidT avgT
# define vidR avgR
# define vidbvf avgbvf
# define vidO avgO
# define vidW avgW
# define vidVisc avgVisc
# define vidDiff avgDiff
# define vidAkv avgAkv
# define vidAkt avgAkt
# define vidAks avgAks
# define vidHbl avgHbl
# define vidHbbl avgHbbl
# define vidTke avgTke
# define vidGls avgGls
# define vidLsc avgLsc
# define vidHel avgHel
# define vidChC avgChC
# define vidU10 avgU10
# define vidKvO2 avgKvO2
# define vidO2sat avgO2sat
# define vidAOU avgAOU
# define vidwind10 avgwind10
# define vidSed avgSed
# define vidBBL avgBBL
# define vidBostr avgBostr
# define vidBustr avgBustr
# define vidBvstr avgBvstr
# define vidWstr avgWstr
# define vidUWstr avgUWstr
# define vidVWstr avgVWstr
# define vidHm avgHm
# define vidWAVE avgWAVE
# define vidSUP avgSUP
# define vidUST2D avgUST2D
# define vidVST2D avgVST2D
# define vidUST avgUST
# define vidWST avgWST
# define vidVST avgVST
# define vidAkb avgAkb
# define vidAkw avgAkw
# define vidKVF avgKVF
# define vidCALP avgCALP
# define vidKAPS avgKAPS
# define vidShflx avgShflx
# define vidSwflx avgSwflx
# define vidShflx_rsw avgShflx_rsw
#if defined BULK_FLUX || defined OA_COUPLING
#  define vidShflx_rlw avgShflx_rlw
#  define vidShflx_lat avgShflx_lat
#  define vidShflx_sen avgShflx_sen
#endif
#ifdef BHFLUX
# define vidBhflx avgBhflx
#endif
#if defined BWFLUX && defined SALINITY
# define vidBwflx avgBwflx
#endif
# ifdef SST_SKIN
#  define vidSST_skin avgSST_skin
# endif

# define vidM avgMust
#endif
!
! Put time record index into file name. In  the case when model
! output is to be arranged into sequence of named files, the naming
! convention is as follows: 'his_root.INDEX.[MPI_node.]nc', where
! INDEX is an integer number such that (i) it is divisible by the
! specified number of records per file; and (ii)
!
!      INDEX + record_within_the_file = total_record
!
! where, 1 =< record_within_the_file =< records_per_file, so that
! total_record changes continuously throughout the sequence of files.
!
      ierr=0
      lstr=lenstr(ncname)
      if (rec_per_file.gt.0) then
        lvar=total_rec-(1+mod(total_rec-1, rec_per_file))
        call insert_time_index (ncname, lstr, lvar, ierr)
        if (ierr .ne. 0) goto 99
      endif
!
! Decide whether to create a new file, or open existing one.
! Overall the whole code below is organized into 3-way switch,
!
! 10  if (create_new_file) then
!        .... create new file, save netCDF ids for all variables;
!     elseif (ncid.eq.-1) then
!        .... try to open existing file and check its dimensions
!       if (cannot be opened or rejected) then
!         create_new_file=.true.
!         goto 10
!       endif   and prepare
!        .... prepare the file for adding new data,
!        .... find and save netCDF ids for all variables
!     else
!        .... just open, no checking, all ids are assumed to be
!        .... already known (MPI single file output only).
!     endif
!
! which is designed to implement flexible opening policy:
! if ldefhis=.true., it forces creation of a new file [if the
! file already exists, it will be overwritten]; on the other hand,
! ldefhis=.false., it is assumed that the file already exists and
! an attempt to open it is made; if the attempt is successful, the
! file is prepared for appending hew data; if it fails, a new file
! is created.
!

#ifdef DO_NOT_OVERWRITE
      if (lvar.eq.total_rec-1) then
        create_new_file=ldefhis
      else
        create_new_file=.false.
      endif
#else
      create_new_file=ldefhis
#endif

      if (ncid.ne.-1) create_new_file=.false.
#if defined MPI & !defined PARALLEL_FILES & !defined NC4PAR
      if (mynode.gt.0) create_new_file=.false.
#endif
!
!=================================================================
!
! Create new history/averages file:    Put global attributes
! ====== === ======= ======== =====    and define all variables.
!
!=================================================================
!
  10  if (create_new_file) then
#ifndef NC4PAR
        ierr=nf_create(ncname(1:lstr),NF_CLOBBER, ncid)
#else
        cmode = ior(nf_netcdf4,nf_classic_model)
        cmode = ior(cmode, nf_mpiio)
        csize=xi_rho*eta_rho/NNODES
        MPI_master_only write(stdout,*)
     &       'CREATE HIS/AVG NC4 PARALLEL FILE'
        ierr=nf_create_par(ncname(1:lstr),cmode,
     &        MPI_COMM_WORLD,MPI_INFO_NULL,ncid)
#endif
        if (ierr .ne. nf_noerr) then
          MPI_master_only write(stdout,'(/3(1x,A)/)')
     &           'ERROR in def_his/avg:',
     &           'Cannot create netCDF file:', ncname(1:lstr)
          goto 99                                         !--> ERROR
        endif
        if (rec_per_file.eq.0) total_rec=0
!
! Put global attributes.
! --- ------ -----------
!
        call put_global_atts (ncid, ierr)
!
! Define dimensions of staggered fields.
! ------ ---------- -- --------- -------
!
        ierr=nf_def_dim (ncid, 'xi_rho',   xi_rho,   r2dgrd(1))
        ierr=nf_def_dim (ncid, 'xi_u',     xi_u,     u2dgrd(1))
        ierr=nf_def_dim (ncid, 'eta_rho',  eta_rho,  r2dgrd(2))
        ierr=nf_def_dim (ncid, 'eta_v',    eta_v,    v2dgrd(2))
#ifdef SEDIMENT
        ierr=nf_def_dim (ncid, 's_b',      NLAY,     b3dgrd(3))
#endif
#ifdef  MUSTANG
        ierr=nf_def_dim (ncid, 's_b',      nk_nivsed_out,  b3dgrd(3))
#endif
#ifdef SOLVE3D
        ierr=nf_def_dim (ncid, 's_rho',    N,        r3dgrd(3))
        ierr=nf_def_dim (ncid, 's_w',      N+1,      w3dgrd(3))
#endif
        ierr=nf_def_dim (ncid, 'time', nf_unlimited, timedim)
# if defined OUTPUTS_SURFACE && ! defined XIOS
        ierr=nf_def_dim (ncid, 'auxil',    6,        auxil(1))
#else
        ierr=nf_def_dim (ncid, 'auxil',    4,        auxil(1))
#endif
        auxil(2)=timedim

        r2dgrd(3)=timedim           ! Free surface

        u2dgrd(2)=r2dgrd(2)         ! 2D UBAR-type
        u2dgrd(3)=timedim

        v2dgrd(1)=r2dgrd(1)         ! 2D VBAR-type
        v2dgrd(3)=timedim

        b3dgrd(1)=r2dgrd(1)         !
        b3dgrd(2)=r2dgrd(2)         ! 3D BED-type
        b3dgrd(4)=timedim           !

#ifdef SOLVE3D
        r3dgrd(1)=r2dgrd(1)         !
        r3dgrd(2)=r2dgrd(2)         ! 3D RHO-type
        r3dgrd(4)=timedim           !

        u3dgrd(1)=u2dgrd(1)         !
        u3dgrd(2)=r2dgrd(2)         ! 3D U-type
        u3dgrd(3)=r3dgrd(3)         !
        u3dgrd(4)=timedim

        v3dgrd(1)=r2dgrd(1)         !
        v3dgrd(2)=v2dgrd(2)         ! 3D V-type
        v3dgrd(3)=r3dgrd(3)         !
        v3dgrd(4)=timedim

        w3dgrd(1)=r2dgrd(1)         !
        w3dgrd(2)=r2dgrd(2)         ! 3D W-type
        w3dgrd(4)=timedim           !
#endif
#if (defined PUT_GRID_INTO_HISTORY && !defined AVRH)\
 || (defined PUT_GRID_INTO_AVERAGES && defined AVRH)
!
! Define grid variables.
! ------ ---- ----------
!
        if (total_rec.le.1) then
# ifdef SOLVE3D
          call def_grid_3d(ncid, r2dgrd, u2dgrd, v2dgrd
     &                                  ,r3dgrd, w3dgrd)
# else
          call def_grid_2d(ncid, r2dgrd, u2dgrd, v2dgrd)
# endif
        endif
#endif

!
! Define evolving model variables.
! ------ -------- ----- ----------
!
! Time step number and time record indices:
!
        ierr=nf_def_var (ncid, 'time_step', nf_int, 2, auxil,
     &                                                 vidTstep)
#ifdef NC4PAR
        ierr=nf_var_par_access(ncid,vidTstep,nf_collective)
#endif
        ierr=nf_put_att_text (ncid, vidTstep, 'long_name', 48,
     &       'time step and record numbers from initialization')
!
! Time.
!
        lvar=lenstr(vname(1,indxTime))
        ierr=nf_def_var (ncid, vname(1,indxTime)(1:lvar),
     &                            NF_DOUBLE, 1, timedim, vidTime)

#ifdef NC4PAR
        ierr=nf_var_par_access(ncid,vidTime,nf_collective)
#endif
#ifndef AVRH
        lvar=lenstr(vname(2,indxTime))
        ierr=nf_put_att_text (ncid, vidTime, 'long_name', lvar,
     &                                vname(2,indxTime)(1:lvar))
#else
        text='averaged '/ /vname(2,indxTime)
        lvar=lenstr(text)
        ierr=nf_put_att_text (ncid, vidTime, 'long_name', lvar,
     &                                             text(1:lvar))
#endif
        lvar=lenstr(vname(3,indxTime))
        ierr=nf_put_att_text (ncid, vidTime, 'units',  lvar,
     &                                vname(3,indxTime)(1:lvar))
        lvar=lenstr(vname(4,indxTime))
        ierr=nf_put_att_text (ncid, vidTime, 'field',  lvar,
     &                                vname(4,indxTime)(1:lvar))

        call nf_add_attribute(ncid, vidTime, indxTime, 5,
     &       NF_FOUT, ierr)

!
! Time2.
!
        lvar=lenstr(vname(1,indxTime2))
        ierr=nf_def_var (ncid, vname(1,indxTime2)(1:lvar),
     &                            NF_DOUBLE, 1, timedim, vidTime2)
#ifdef NC4PAR
        ierr=nf_var_par_access(ncid,vidTime2,nf_collective)
#endif
#ifndef AVRH
        lvar=lenstr(vname(2,indxTime2))
        ierr=nf_put_att_text (ncid, vidTime2, 'long_name', lvar,
     &                                vname(2,indxTime2)(1:lvar))
#else
        text='averaged '/ /vname(2,indxTime2)
        lvar=lenstr(text)
        ierr=nf_put_att_text (ncid, vidTime2, 'long_name', lvar,
     &                                             text(1:lvar))
#endif
        lvar=lenstr(vname(3,indxTime2))
        ierr=nf_put_att_text (ncid, vidTime2, 'units',  lvar,
     &                                vname(3,indxTime2)(1:lvar))
        lvar=lenstr(vname(4,indxTime2))
        ierr=nf_put_att_text (ncid, vidTime2, 'field',  lvar,
     &                                vname(4,indxTime2)(1:lvar))

        call nf_add_attribute(ncid, vidTime2, indxTime2, 5,
     &       NF_FOUT, ierr)

!=======================================================================
!
! Define primary variables
!
!=======================================================================

!
!-----------------------------------------------------------------------
! Free-surface.
!-----------------------------------------------------------------------
!
        if (wrt(indxZ)) then
          lvar=lenstr(vname(1,indxZ))
          ierr=nf_def_var (ncid, vname(1,indxZ)(1:lvar),
     &                              NF_FOUT, 3, r2dgrd, vidZ)
#ifdef NC4PAR
          ierr=nf_var_par_access(ncid,vidZ,nf_collective)
#endif
#ifndef AVRH
          lvar=lenstr(vname(2,indxZ))
          ierr=nf_put_att_text (ncid, vidZ, 'long_name', lvar,
     &                                  vname(2,indxZ)(1:lvar))
#else
          text='averaged '/ /vname(2,indxZ)
          lvar=lenstr(text)
          ierr=nf_put_att_text (ncid, vidZ, 'long_name', lvar,
     &                                            text(1:lvar))
#endif
          lvar=lenstr(vname(3,indxZ))
          ierr=nf_put_att_text (ncid, vidZ, 'units',     lvar,
     &                                  vname(3,indxZ)(1:lvar))
          lvar=lenstr(vname(4,indxZ))
          ierr=nf_put_att_text (ncid, vidZ, 'field',     lvar,
     &                                  vname(4,indxZ)(1:lvar))

          call nf_add_attribute(ncid, vidZ, indxZ, 5, NF_FOUT,
     &         ierr)

        endif
!
!-----------------------------------------------------------------------
! 2D momenta in XI- and ETA-directions.
!-----------------------------------------------------------------------
!
        if (wrt(indxUb)) then
          lvar=lenstr(vname(1,indxUb))
          ierr=nf_def_var (ncid, vname(1,indxUb)(1:lvar),
     &                             NF_FOUT, 3, u2dgrd, vidUb)
#ifdef NC4PAR
          ierr=nf_var_par_access(ncid,vidUb,nf_collective)
#endif
#ifndef AVRH
          lvar=lenstr(vname(2,indxUb))
          ierr=nf_put_att_text (ncid, vidUb, 'long_name', lvar,
     &                                  vname(2,indxUb)(1:lvar))
#else
          text='averaged '/ /vname(2,indxUb)
          lvar=lenstr(text)
          ierr=nf_put_att_text (ncid, vidUb, 'long_name', lvar,
     &                                             text(1:lvar))
#endif
          lvar=lenstr(vname(3,indxUb))
          ierr=nf_put_att_text (ncid, vidUb, 'units',     lvar,
     &                                  vname(3,indxUb)(1:lvar))
          lvar=lenstr(vname(4,indxUb))
          ierr=nf_put_att_text (ncid, vidUb, 'field',    lvar,
     &                                  vname(4,indxUb)(1:lvar))

          call nf_add_attribute(ncid, vidUb, indxUb, 5, NF_FOUT, ierr)
        endif

        if (wrt(indxVb)) then
          lvar=lenstr(vname(1,indxVb))
          ierr=nf_def_var (ncid, vname(1,indxVb)(1:lvar),
     &                              NF_FOUT, 3, v2dgrd, vidVb)
#ifdef NC4PAR
          ierr=nf_var_par_access(ncid,vidVb,nf_collective)
#endif
#ifndef AVRH
          lvar=lenstr(vname(2,indxVb))
          ierr=nf_put_att_text (ncid, vidVb, 'long_name', lvar,
     &                                  vname(2,indxVb)(1:lvar))
#else
          text='averaged '/ /vname(2,indxVb)
          lvar=lenstr(text)
          ierr=nf_put_att_text (ncid, vidVb, 'long_name', lvar,
     &                                             text(1:lvar))
#endif
          lvar=lenstr(vname(3,indxVb))
          ierr=nf_put_att_text (ncid, vidVb, 'units',     lvar,
     &                                  vname(3,indxVb)(1:lvar))
          lvar=lenstr(vname(4,indxVb))
          ierr=nf_put_att_text (ncid, vidVb, 'field',     lvar,
     &                                  vname(4,indxVb)(1:lvar))

          call nf_add_attribute(ncid, vidVb, indxVb, 5, NF_FOUT,
     &         ierr)
        endif

#ifdef MORPHODYN
!
!-----------------------------------------------------------------------
!  Time-dependent bathymetry
!-----------------------------------------------------------------------
!
        if (wrt(indxHm)) then
          lvar=lenstr(vname(1,indxHm))
          ierr=nf_def_var (ncid, vname(1,indxHm)(1:lvar),
     &                              NF_FOUT, 3, r2dgrd, vidHm)
# ifdef NC4PAR
          ierr=nf_var_par_access(ncid,vidHm,nf_collective)
# endif
#ifndef AVRH
          lvar=lenstr(vname(2,indxHm))
          ierr=nf_put_att_text (ncid, vidHm, 'long_name', lvar,
     &                                  vname(2,indxHm)(1:lvar))
#else
          text='averaged '/ /vname(2,indxHm)
          lvar=lenstr(text)
          ierr=nf_put_att_text (ncid, vidHm, 'long_name', lvar,
     &                                             text(1:lvar))
#endif
          lvar=lenstr(vname(3,indxHm))
          ierr=nf_put_att_text (ncid, vidHm, 'units',     lvar,
     &                                  vname(3,indxHm)(1:lvar))
          lvar=lenstr(vname(4,indxHm))
          ierr=nf_put_att_text (ncid, vidHm, 'field',     lvar,
     &                                  vname(4,indxHm)(1:lvar))

          call nf_add_attribute(ncid, vidHm, indxHm, 5, NF_FOUT,
     &         ierr)
        endif
#endif /* MORPHODYN */

#ifdef SOLVE3D
!
!-----------------------------------------------------------------------
! 3D momenta in XI- and ETA-directions.
!-----------------------------------------------------------------------
!
        if (wrt(indxU)) then
          lvar=lenstr(vname(1,indxU))
          ierr=nf_def_var (ncid, vname(1,indxU)(1:lvar),
     &                             NF_FOUT, 4, u3dgrd, vidU)
#ifdef NC4PAR
          ierr=nf_var_par_access(ncid,vidU,nf_collective)
#endif
# ifndef AVRH
          lvar=lenstr(vname(2,indxU))
          ierr=nf_put_att_text (ncid, vidU, 'long_name', lvar,
     &                                  vname(2,indxU)(1:lvar))
# else
          text='averaged '/ /vname(2,indxU)
          lvar=lenstr(text)
          ierr=nf_put_att_text (ncid, vidU, 'long_name', lvar,
     &                                            text(1:lvar))
# endif
          lvar=lenstr(vname(3,indxU))
          ierr=nf_put_att_text (ncid, vidU, 'units',     lvar,
     &                                  vname(3,indxU)(1:lvar))
          lvar=lenstr(vname(4,indxU))
          ierr=nf_put_att_text (ncid, vidU, 'field',     lvar,
     &                                  vname(4,indxU)(1:lvar))

          call nf_add_attribute(ncid, vidU, indxU, 5, NF_FOUT, ierr)
        endif

        if (wrt(indxV)) then
          lvar=lenstr(vname(1,indxV))
          ierr=nf_def_var (ncid, vname(1,indxV)(1:lvar),
     &                             NF_FOUT, 4, v3dgrd, vidV)
#ifdef NC4PAR
          ierr=nf_var_par_access(ncid,vidV,nf_collective)
#endif
# ifndef AVRH
          lvar=lenstr(vname(2,indxV))
          ierr=nf_put_att_text (ncid, vidV, 'long_name', lvar,
     &                                  vname(2,indxV)(1:lvar))
# else
          text='averaged '/ /vname(2,indxV)
          lvar=lenstr(text)
          ierr=nf_put_att_text (ncid, vidV, 'long_name', lvar,
     &                                            text(1:lvar))
# endif
          lvar=lenstr(vname(3,indxV))
          ierr=nf_put_att_text (ncid, vidV, 'units',     lvar,
     &                                  vname(3,indxV)(1:lvar))
          lvar=lenstr(vname(4,indxV))
          ierr=nf_put_att_text (ncid, vidV, 'field',     lvar,
     &                                  vname(4,indxV)(1:lvar))

          call nf_add_attribute(ncid, vidV, indxV, 5, NF_FOUT, ierr)
        endif
!
!-----------------------------------------------------------------------
! Tracer variables.
!-----------------------------------------------------------------------
!
# ifdef TRACERS
        do itrc=1,NT
          if (wrt(indxV+itrc)) then
            lvar=lenstr(vname(1,indxV+itrc))
            ierr=nf_def_var (ncid, vname(1,indxV+itrc)(1:lvar),
     &                             NF_FOUT, 4, r3dgrd, vidT(itrc))
#ifdef NC4PAR
          ierr=nf_var_par_access(ncid,vidT(itrc),nf_collective)
#endif
# ifndef AVRH
            lvar=lenstr(vname(2,indxV+itrc))
            ierr=nf_put_att_text (ncid, vidT(itrc), 'long_name',
     &                         lvar, vname(2,indxV+itrc)(1:lvar))
# else
            text='averaged '/ /vname(2,indxV+itrc)
            lvar=lenstr(text)
            ierr=nf_put_att_text (ncid, vidT(itrc), 'long_name',
     &                                          lvar, text(1:lvar))
# endif
            lvar=lenstr(vname(3,indxV+itrc))
            ierr=nf_put_att_text (ncid, vidT(itrc), 'units', lvar,
     &                               vname(3,indxV+itrc)(1:lvar))
            lvar=lenstr(vname(4,indxV+itrc))
            ierr=nf_put_att_text (ncid, vidT(itrc), 'field', lvar,
     &                               vname(4,indxV+itrc)(1:lvar))

            call nf_add_attribute(ncid,vidT(itrc),indxV+itrc,5,
     &           NF_FOUT, ierr)
          endif
        enddo
# endif
!
!-----------------------------------------------------------------------
! Density anomaly.
!-----------------------------------------------------------------------
!
        if (wrt(indxR)) then
          lvar=lenstr(vname(1,indxR))
          ierr=nf_def_var (ncid, vname(1,indxR)(1:lvar),
     &                             NF_FOUT, 4, r3dgrd, vidR)
#ifdef NC4PAR
          ierr=nf_var_par_access(ncid,vidR,nf_collective)
#endif
# ifndef AVRH
          lvar=lenstr(vname(2,indxR))
          ierr=nf_put_att_text (ncid, vidR, 'long_name', lvar,
     &                                  vname(2,indxR)(1:lvar))
# else
          text='averaged '/ /vname(2,indxR)
          lvar=lenstr(text)
          ierr=nf_put_att_text (ncid, vidR, 'long_name', lvar,
     &                                            text(1:lvar))
# endif
          lvar=lenstr(vname(3,indxR))
          ierr=nf_put_att_text (ncid, vidR, 'units',     lvar,
     &                                  vname(3,indxR)(1:lvar))
          lvar=lenstr(vname(4,indxR))
          ierr=nf_put_att_text (ncid, vidR, 'field',     lvar,
     &                                  vname(4,indxR)(1:lvar))

          call nf_add_attribute(ncid, vidR, indxR, 5, NF_FOUT, ierr)
        endif
!
!-----------------------------------------------------------------------
! Brunt Vaisala Frequency.
!-----------------------------------------------------------------------
!
# if defined ANA_VMIX || defined BVF_MIXING \
  || defined LMD_MIXING || defined LMD_SKPP || defined LMD_BKPP \
  || defined GLS_MIXING
        if (wrt(indxbvf)) then
          lvar=lenstr(vname(1,indxbvf))
          ierr=nf_def_var (ncid, vname(1,indxbvf)(1:lvar),
     &                             NF_FOUT, 4, w3dgrd, vidbvf)
#ifdef NC4PAR
          ierr=nf_var_par_access(ncid,vidbvf,nf_collective)
#endif
# ifndef AVRH
          lvar=lenstr(vname(2,indxbvf))
          ierr=nf_put_att_text (ncid, vidbvf, 'long_name', lvar,
     &                                  vname(2,indxbvf)(1:lvar))
# else
          text='averaged '/ /vname(2,indxbvf)
          lvar=lenstr(text)
          ierr=nf_put_att_text (ncid, vidbvf, 'long_name', lvar,
     &                                            text(1:lvar))
# endif
          lvar=lenstr(vname(3,indxbvf))
          ierr=nf_put_att_text (ncid, vidbvf, 'units',     lvar,
     &                                  vname(3,indxbvf)(1:lvar))
          lvar=lenstr(vname(4,indxbvf))
          ierr=nf_put_att_text (ncid, vidR, 'field',     lvar,
     &                                  vname(4,indxbvf)(1:lvar))

          call nf_add_attribute(ncid, vidbvf, indxbvf, 5, NF_FOUT, ierr)
        endif
# endif
!
!-----------------------------------------------------------------------
! BIO_BioEBUS : AOU and pCO2
!-----------------------------------------------------------------------
!
# if defined BIOLOGY && defined BIO_BioEBUS
        if (wrt(indxAOU)) then
          lvar=lenstr(vname(1,indxAOU))
          ierr=nf_def_var (ncid, vname(1,indxAOU)(1:lvar),
     &                           NF_FOUT, 4, r3dgrd, vidAOU)
#ifdef NC4PAR
          ierr=nf_var_par_access(ncid,vidAOU,nf_collective)
#endif
#   ifndef AVRH
          lvar=lenstr(vname(2,indxAOU))
          ierr=nf_put_att_text (ncid, vidAOU, 'long_name',lvar,
     &                                vname(2,indxAOU)(1:lvar))
#   else
          text='averaged '/ /vname(2,indxAOU)
          lvar=lenstr(text)
          ierr=nf_put_att_text (ncid, vidAOU, 'long_name',lvar,
     &                                          text(1:lvar))
#   endif
          lvar=lenstr(vname(3,indxAOU))
          ierr=nf_put_att_text (ncid, vidAOU, 'units',lvar,
     &                                vname(3,indxAOU)(1:lvar))
          lvar=lenstr(vname(4,indxAOU))
          ierr=nf_put_att_text (ncid, vidAOU, 'field',lvar,
     &                                vname(4,indxAOU)(1:lvar))

          call nf_add_attribute(ncid, vidAOU, indxAOU, 5,
     &    NF_FOUT, ierr)
        endif

        if (wrt(indxWIND10)) then
          lvar=lenstr(vname(1,indxWIND10))
          ierr=nf_def_var (ncid, vname(1,indxWIND10)(1:lvar),
     &                         NF_FOUT, 3, r2dgrd, vidwind10)
#ifdef NC4PAR
          ierr=nf_var_par_access(ncid,vidwind10,nf_collective)
#endif
#  ifndef AVRH
          lvar=lenstr(vname(2,indxWIND10))
          ierr=nf_put_att_text (ncid,vidwind10,'long_name',lvar,
     &                              vname(2,indxWIND10)(1:lvar))
#  else
          text='averaged '/ /vname(2,indxWIND10)
          lvar=lenstr(text)
          ierr=nf_put_att_text (ncid,vidwind10,'long_name',lvar,
     &                                        text(1:lvar))
#  endif
          lvar=lenstr(vname(3,indxWIND10))
          ierr=nf_put_att_text (ncid,vidwind10,'units',lvar,
     &                              vname(3,indxWIND10)(1:lvar))
          lvar=lenstr(vname(4,indxWIND10))
          ierr=nf_put_att_text (ncid,vidwind10,'field',lvar,
     &                              vname(4,indxWIND10)(1:lvar))

          call nf_add_attribute(ncid, vidwind10, indxWIND10, 5,
     &    NF_FOUT, ierr)
        endif
# endif /* BIO_BioEBUS */
!
!-----------------------------------------------------------------------
! S-coordinate "omega" vertical velocity.
!-----------------------------------------------------------------------
!
        if (wrt(indxO)) then
          lvar=lenstr(vname(1,indxO))
          ierr=nf_def_var (ncid, vname(1,indxO)(1:lvar),
     &                             NF_FOUT, 4, w3dgrd, vidO)
#ifdef NC4PAR
          ierr=nf_var_par_access(ncid,vidO,nf_collective)
#endif
# ifndef AVRH
          lvar=lenstr(vname(2,indxO))
          ierr=nf_put_att_text (ncid, vidO, 'long_name', lvar,
     &                                  vname(2,indxO)(1:lvar))
# else
          text='averaged '/ /vname(2,indxO)
          lvar=lenstr(text)
          ierr=nf_put_att_text (ncid, vidO, 'long_name', lvar,
     &                                            text(1:lvar))
# endif
          lvar=lenstr(vname(3,indxO))
          ierr=nf_put_att_text (ncid, vidO, 'units',     lvar,
     &                                  vname(3,indxO)(1:lvar))
          lvar=lenstr(vname(4,indxO))
          ierr=nf_put_att_text (ncid, vidO, 'field',     lvar,
     &                                  vname(4,indxO)(1:lvar))

          call nf_add_attribute(ncid, vidO, indxO, 5, NF_FOUT, ierr)
        endif
!
!-----------------------------------------------------------------------
! True vertical velocity.
!-----------------------------------------------------------------------
!
        if (wrt(indxW)) then
          lvar=lenstr(vname(1,indxW))
# ifdef NBQ
          ierr=nf_def_var (ncid, vname(1,indxW)(1:lvar),
     &                               NF_FOUT, 4, w3dgrd, vidW)
# else
          ierr=nf_def_var (ncid, vname(1,indxW)(1:lvar),
     &                               NF_FOUT, 4, r3dgrd, vidW)
# endif
# ifdef NC4PAR
          ierr=nf_var_par_access(ncid,vidW,nf_collective)
# endif
# ifndef AVRH
          lvar=lenstr(vname(2,indxW))
          ierr=nf_put_att_text (ncid, vidW, 'long_name', lvar,
     &                                  vname(2,indxW)(1:lvar))
# else
          text='averaged '/ /vname(2,indxW)
          lvar=lenstr(text)
          ierr=nf_put_att_text (ncid, vidW, 'long_name', lvar,
     &                                            text(1:lvar))
# endif
          lvar=lenstr(vname(3,indxW))
          ierr=nf_put_att_text (ncid, vidW, 'units',     lvar,
     &                                  vname(3,indxW)(1:lvar))
          lvar=lenstr(vname(4,indxW))
          ierr=nf_put_att_text (ncid, vidW, 'field',     lvar,
     &                                  vname(4,indxW)(1:lvar))

          call nf_add_attribute(ncid, vidW, indxW, 5, NF_FOUT, ierr)
        endif
#endif /* SOLVE3D */

!================================================================
!
! Define secondary variables
!
!================================================================
        if (wrt(indxBostr)) then
          lvar=lenstr(vname(1,indxBostr))
          ierr=nf_def_var (ncid, vname(1,indxBostr)(1:lvar),
     &                             NF_FOUT, 3, r2dgrd, vidBostr)
#ifdef NC4PAR
          ierr=nf_var_par_access(ncid,vidBostr,nf_collective)
#endif
#  ifndef AVRH
          lvar=lenstr(vname(2,indxBostr))
          ierr=nf_put_att_text (ncid, vidBostr, 'long_name', lvar,
     &                                 vname(2,indxBostr)(1:lvar))
#  else
          text='averaged '/ /vname(2,indxBostr)
          lvar=lenstr(text)
          ierr=nf_put_att_text (ncid, vidBostr, 'long_name', lvar,
     &                                               text(1:lvar))
#  endif
          lvar=lenstr(vname(3,indxBostr))
          ierr=nf_put_att_text (ncid, vidBostr, 'units',     lvar,
     &                                 vname(3,indxBostr)(1:lvar))

          call nf_add_attribute(ncid, vidBostr,indxBostr,5,
     &                          NF_FOUT, ierr)
        endif



        if (wrt(indxBustr)) then
          lvar=lenstr(vname(1,indxBustr))
          ierr=nf_def_var (ncid, vname(1,indxBustr)(1:lvar),
     &                             NF_FOUT, 3, u2dgrd, vidBustr)
#ifdef NC4PAR
          ierr=nf_var_par_access(ncid,vidBustr,nf_collective)
#endif
#  ifndef AVRH
          lvar=lenstr(vname(2,indxBustr))
          ierr=nf_put_att_text (ncid, vidBustr, 'long_name', lvar,
     &                                 vname(2,indxBustr)(1:lvar))
#  else
          text='averaged '/ /vname(2,indxBustr)
          lvar=lenstr(text)
          ierr=nf_put_att_text (ncid, vidBustr, 'long_name', lvar,
     &                                               text(1:lvar))
#  endif
          lvar=lenstr(vname(3,indxBustr))
          ierr=nf_put_att_text (ncid, vidBustr, 'units',     lvar,
     &                                 vname(3,indxBustr)(1:lvar))

          call nf_add_attribute(ncid, vidBustr,indxBustr,5,
     &                          NF_FOUT, ierr)

        endif


        if (wrt(indxBvstr)) then
          lvar=lenstr(vname(1,indxBvstr))
          ierr=nf_def_var (ncid, vname(1,indxBvstr)(1:lvar),
     &                             NF_FOUT, 3, v2dgrd, vidBvstr)
#ifdef NC4PAR
          ierr=nf_var_par_access(ncid,vidBvstr,nf_collective)
#endif
#  ifndef AVRH
          lvar=lenstr(vname(2,indxBvstr))
          ierr=nf_put_att_text (ncid, vidBvstr, 'long_name', lvar,
     &                                 vname(2,indxBvstr)(1:lvar))
#  else
          text='averaged '/ /vname(2,indxBvstr)
          lvar=lenstr(text)
          ierr=nf_put_att_text (ncid, vidBvstr, 'long_name', lvar,
     &                                               text(1:lvar))
#  endif
          lvar=lenstr(vname(3,indxBvstr))
          ierr=nf_put_att_text (ncid, vidBvstr, 'units',     lvar,
     &                                 vname(3,indxBvstr)(1:lvar))

          call nf_add_attribute(ncid, vidBvstr,indxBvstr,5,
     &                          NF_FOUT, ierr)

        endif
!
        if (wrt(indxWstr)) then
          lvar=lenstr(vname(1,indxWstr))
          ierr=nf_def_var (ncid, vname(1,indxWstr)(1:lvar),
     &                             NF_FOUT, 3, r2dgrd, vidWstr)
#ifdef NC4PAR
          ierr=nf_var_par_access(ncid,vidWstr,nf_collective)
#endif
#  ifndef AVRH
          lvar=lenstr(vname(2,indxWstr))
          ierr=nf_put_att_text (ncid, vidWstr, 'long_name', lvar,
     &                                 vname(2,indxWstr)(1:lvar))
#  else
          text='averaged '/ /vname(2,indxWstr)
          lvar=lenstr(text)
          ierr=nf_put_att_text (ncid, vidWstr, 'long_name', lvar,
     &                                               text(1:lvar))
#  endif
          lvar=lenstr(vname(3,indxWstr))
          ierr=nf_put_att_text (ncid, vidWstr, 'units',     lvar,
     &                                 vname(3,indxWstr)(1:lvar))

          call nf_add_attribute(ncid, vidWstr, indxWstr, 5,
     &                          NF_FOUT, ierr)
        endif
!
        if (wrt(indxUWstr)) then
          lvar=lenstr(vname(1,indxUWstr))
          ierr=nf_def_var (ncid, vname(1,indxUWstr)(1:lvar),
     &                             NF_FOUT, 3, u2dgrd, vidUWstr)
#ifdef NC4PAR
          ierr=nf_var_par_access(ncid,vidUWstr,nf_collective)
#endif
#  ifndef AVRH
          lvar=lenstr(vname(2,indxUWstr))
          ierr=nf_put_att_text (ncid, vidUWstr, 'long_name', lvar,
     &                                 vname(2,indxUWstr)(1:lvar))
#  else
          text='averaged '/ /vname(2,indxUWstr)
          lvar=lenstr(text)
          ierr=nf_put_att_text (ncid, vidUWstr, 'long_name', lvar,
     &                                               text(1:lvar))
#  endif
          lvar=lenstr(vname(3,indxUWstr))
          ierr=nf_put_att_text (ncid, vidUWstr, 'units',     lvar,
     &                                 vname(3,indxUWstr)(1:lvar))

          call nf_add_attribute(ncid, vidUWstr, indxUWstr, 5,
     &                          NF_FOUT, ierr)
        endif
!
        if (wrt(indxVWstr)) then
          lvar=lenstr(vname(1,indxVWstr))
          ierr=nf_def_var (ncid, vname(1,indxVWstr)(1:lvar),
     &                             NF_FOUT, 3, v2dgrd, vidVWstr)
#ifdef NC4PAR
          ierr=nf_var_par_access(ncid,vidVWstr,nf_collective)
#endif
#  ifndef AVRH
          lvar=lenstr(vname(2,indxVWstr))
          ierr=nf_put_att_text (ncid, vidVWstr, 'long_name', lvar,
     &                                 vname(2,indxVWstr)(1:lvar))
#  else
          text='averaged '/ /vname(2,indxVWstr)
          lvar=lenstr(text)
          ierr=nf_put_att_text (ncid, vidVWstr, 'long_name', lvar,
     &                                               text(1:lvar))
#  endif
          lvar=lenstr(vname(3,indxVWstr))
          ierr=nf_put_att_text (ncid, vidVWstr, 'units',     lvar,
     &                                 vname(3,indxVWstr)(1:lvar))

          call nf_add_attribute(ncid, vidVWstr, indxVWstr, 5,
     &                          NF_FOUT, ierr)
        endif

#ifdef SOLVE3D
# ifdef VIS_COEF_3D
!
! Horizontal viscosity coefficient.
!
        if (wrt(indxVisc)) then
          lvar=lenstr(vname(1,indxVisc))
          ierr=nf_def_var (ncid, vname(1,indxVisc)(1:lvar),
     &                             NF_FOUT, 4, r3dgrd, vidVisc)
#ifdef NC4PAR
          ierr=nf_var_par_access(ncid,vidVisc,nf_collective)
#endif
#  ifndef AVRH
          lvar=lenstr(vname(2,indxVisc))
          ierr=nf_put_att_text (ncid, vidVisc, 'long_name', lvar,
     &                                  vname(2,indxVisc)(1:lvar))
#  else
          text='averaged '/ /vname(2,indxVisc)
          lvar=lenstr(text)
          ierr=nf_put_att_text (ncid, vidVisc, 'long_name', lvar,
     &                                              text(1:lvar))
#  endif
          lvar=lenstr(vname(3,indxVisc))
          ierr=nf_put_att_text (ncid, vidVisc, 'units',     lvar,
     &                                  vname(3,indxVisc)(1:lvar))
          lvar=lenstr(vname(4,indxVisc))
          ierr=nf_put_att_text (ncid, vidVisc, 'field',     lvar,
     &                                  vname(4,indxVisc)(1:lvar))

          call nf_add_attribute(ncid, vidVisc, indxVisc, 5,
     &                          NF_FOUT, ierr)
        endif
# endif
# ifdef DIF_COEF_3D
!
! Horizontal diffusivity coefficient.
!
        if (wrt(indxDiff)) then
          lvar=lenstr(vname(1,indxDiff))
          ierr=nf_def_var (ncid, vname(1,indxDiff)(1:lvar),
     &                             NF_FOUT, 4, r3dgrd, vidDiff)
#ifdef NC4PAR
          ierr=nf_var_par_access(ncid,vidDiff,nf_collective)
#endif
#  ifndef AVRH
          lvar=lenstr(vname(2,indxDiff))
          ierr=nf_put_att_text (ncid, vidDiff, 'long_name', lvar,
     &                                  vname(2,indxDiff)(1:lvar))
#  else
          text='averaged '/ /vname(2,indxDiff)
          lvar=lenstr(text)
          ierr=nf_put_att_text (ncid, vidDiff, 'long_name', lvar,
     &                                              text(1:lvar))
#  endif
          lvar=lenstr(vname(4,indxDiff))
          ierr=nf_put_att_text (ncid, vidDiff, 'field',     lvar,
     &                                  vname(4,indxDiff)(1:lvar))

          call nf_add_attribute(ncid, vidDiff, indxDiff, 5,
     &                          NF_FOUT, ierr)
        endif
# endif
!
!     Vertical viscosity coefficient.
!
        if (wrt(indxAkv)) then
          lvar=lenstr(vname(1,indxAkv))
          ierr=nf_def_var (ncid, vname(1,indxAkv)(1:lvar),
     &                             NF_FOUT, 4, w3dgrd, vidAkv)
#ifdef NC4PAR
          ierr=nf_var_par_access(ncid,vidAkv,nf_collective)
#endif
# ifndef AVRH
          lvar=lenstr(vname(2,indxAkv))
          ierr=nf_put_att_text (ncid, vidAkv, 'long_name', lvar,
     &                                  vname(2,indxAkv)(1:lvar))
# else
          text='averaged '/ /vname(2,indxAkv)
          lvar=lenstr(text)
          ierr=nf_put_att_text (ncid, vidAkv, 'long_name', lvar,
     &                                              text(1:lvar))
# endif
          lvar=lenstr(vname(3,indxAkv))
          ierr=nf_put_att_text (ncid, vidAkv, 'units',     lvar,
     &                                  vname(3,indxAkv)(1:lvar))
          lvar=lenstr(vname(4,indxAkv))
          ierr=nf_put_att_text (ncid, vidAkv, 'field',     lvar,
     &                                  vname(4,indxAkv)(1:lvar))

          call nf_add_attribute(ncid, vidAkv, indxAkv, 5, NF_FOUT,
     &                                                          ierr)
        endif
!
! Vertical diffusion coefficient for potential temperature.
!
# ifdef TEMPERATURE
        if (wrt(indxAkt)) then
          lvar=lenstr(vname(1,indxAkt))
          ierr=nf_def_var (ncid, vname(1,indxAkt)(1:lvar),
     &                              NF_FOUT, 4, w3dgrd, vidAkt)
#ifdef NC4PAR
          ierr=nf_var_par_access(ncid,vidAkt,nf_collective)
#endif
# ifndef AVRH
          lvar=lenstr(vname(2,indxAkt))
          ierr=nf_put_att_text (ncid, vidAkt, 'long_name', lvar,
     &                                  vname(2,indxAkt)(1:lvar))
# else
          text='averaged '/ /vname(2,indxAkt)
          lvar=lenstr(text)
          ierr=nf_put_att_text (ncid, vidAkt, 'long_name', lvar,
     &                                              text(1:lvar))
# endif
          lvar=lenstr(vname(3,indxAkt))
          ierr=nf_put_att_text (ncid, vidAkt, 'units',     lvar,
     &                                  vname(3,indxAkt)(1:lvar))
          lvar=lenstr(vname(4,indxAkt))
          ierr=nf_put_att_text (ncid, vidAkt, 'field',     lvar,
     &                                  vname(4,indxAkt)(1:lvar))

          call nf_add_attribute(ncid, vidAkt, indxAkt, 5, NF_FOUT,
     &                                                           ierr)
        endif
# endif
# ifdef SALINITY
!
! Vertical diffusion coefficient for salinity.
!
        if (wrt(indxAks)) then
          lvar=lenstr(vname(1,indxAks))
          ierr=nf_def_var (ncid, vname(1,indxAks)(1:lvar),
     &                              NF_FOUT, 4, w3dgrd, vidAks)
#ifdef NC4PAR
          ierr=nf_var_par_access(ncid,vidAks,nf_collective)
#endif
#  ifndef AVRH
          lvar=lenstr(vname(2,indxAks))
          ierr=nf_put_att_text (ncid, vidAks, 'long_name', lvar,
     &                                  vname(2,indxAks)(1:lvar))
#  else
          text='averaged '/ /vname(2,indxAks)
          lvar=lenstr(text)
          ierr=nf_put_att_text (ncid, vidAks, 'long_name', lvar,
     &                                              text(1:lvar))
#  endif
          lvar=lenstr(vname(3,indxAks))
          ierr=nf_put_att_text (ncid, vidAks, 'units',     lvar,
     &                                  vname(3,indxAks)(1:lvar))
          lvar=lenstr(vname(4,indxAks))
          ierr=nf_put_att_text (ncid, vidAks, 'field',     lvar,
     &                                  vname(4,indxAks)(1:lvar))

          call nf_add_attribute(ncid, vidAks, indxAks, 5, NF_FOUT,
     &                                                           ierr)
        endif
# endif


# if defined LMD_SKPP || defined GLS_MIXING
!
! Depth of planetary boundary layer.
!
        if (wrt(indxHbl)) then
          lvar=lenstr(vname(1,indxHbl))
          ierr=nf_def_var (ncid, vname(1,indxHbl)(1:lvar),
     &                             NF_FOUT, 3, r2dgrd, vidHbl)
#ifdef NC4PAR
          ierr=nf_var_par_access(ncid,vidHbl,nf_collective)
#endif
#  ifndef AVRH
          lvar=lenstr(vname(2,indxHbl))
          ierr=nf_put_att_text (ncid, vidHbl, 'long_name', lvar,
     &                                  vname(2,indxHbl)(1:lvar))
#  else
          text='averaged '/ /vname(2,indxHbl)
          lvar=lenstr(text)
          ierr=nf_put_att_text (ncid, vidHbl, 'long_name', lvar,
     &                                              text(1:lvar))
#  endif
          lvar=lenstr(vname(3,indxHbl))
          ierr=nf_put_att_text (ncid, vidHbl, 'units',     lvar,
     &                                  vname(3,indxHbl)(1:lvar))
          lvar=lenstr(vname(4,indxHbl))
          ierr=nf_put_att_text (ncid, vidHbl, 'field',     lvar,
     &                                  vname(4,indxHbl)(1:lvar))

          call nf_add_attribute(ncid, vidHbl, indxHbl, 5, NF_FOUT,
     &                                                            ierr)
        endif
# endif
# ifdef LMD_BKPP
!
! Depth of bottom planetary boundary layer.
!
        if (wrt(indxHbbl)) then
          lvar=lenstr(vname(1,indxHbbl))
          ierr=nf_def_var (ncid, vname(1,indxHbbl)(1:lvar),
     &                             NF_FOUT, 3, r2dgrd, vidHbbl)
#ifdef NC4PAR
          ierr=nf_var_par_access(ncid,vidHbbl,nf_collective)
#endif
#  ifndef AVRH
          lvar=lenstr(vname(2,indxHbbl))
          ierr=nf_put_att_text (ncid, vidHbbl, 'long_name', lvar,
     &                                  vname(2,indxHbbl)(1:lvar))
#  else
          text='averaged '/ /vname(2,indxHbbl)
          lvar=lenstr(text)
          ierr=nf_put_att_text (ncid, vidHbbl, 'long_name', lvar,
     &                                              text(1:lvar))
#  endif
          lvar=lenstr(vname(3,indxHbbl))
          ierr=nf_put_att_text (ncid, vidHbbl, 'units',     lvar,
     &                                  vname(3,indxHbbl)(1:lvar))
          lvar=lenstr(vname(4,indxHbbl))
          ierr=nf_put_att_text (ncid, vidHbbl, 'field',     lvar,
     &                                  vname(4,indxHbbl)(1:lvar))

          call nf_add_attribute(ncid, vidHbbl, indxHbbl, 5,
     &                          NF_FOUT, ierr)
        endif
# endif

# ifdef GLS_MIXING
!
! Turbulent Kinetic Energy.
!
        if (wrt(indxTke)) then
          lvar=lenstr(vname(1,indxTke))
          ierr=nf_def_var (ncid, vname(1,indxTke)(1:lvar),
     &                              NF_FOUT, 4, w3dgrd, vidTke)
#ifdef NC4PAR
          ierr=nf_var_par_access(ncid,vidTke,nf_collective)
#endif
#  ifndef AVRH
          lvar=lenstr(vname(2,indxTke))
          ierr=nf_put_att_text (ncid, vidTke, 'long_name', lvar,
     &                                  vname(2,indxTke)(1:lvar))
#  else
          text='averaged '/ /vname(2,indxTke)
          lvar=lenstr(text)
          ierr=nf_put_att_text (ncid, vidTke, 'long_name', lvar,
     &                                              text(1:lvar))
#  endif
          lvar=lenstr(vname(3,indxTke))
          ierr=nf_put_att_text (ncid, vidTke, 'units',     lvar,
     &                                  vname(3,indxTke)(1:lvar))
          lvar=lenstr(vname(4,indxTke))
          ierr=nf_put_att_text (ncid, vidTke, 'field',     lvar,
     &                                  vname(4,indxTke)(1:lvar))

          call nf_add_attribute(ncid, vidTke, indxTke, 5, NF_FOUT,
     &                                                            ierr)
        endif
!
! Generic Length Scale.
!
        if (wrt(indxGls)) then
          lvar=lenstr(vname(1,indxGls))
          ierr=nf_def_var (ncid, vname(1,indxGls)(1:lvar),
     &                              NF_FOUT, 4, w3dgrd, vidGls)
#ifdef NC4PAR
          ierr=nf_var_par_access(ncid,vidGls,nf_collective)
#endif
#  ifndef AVRH
          lvar=lenstr(vname(2,indxGls))
          ierr=nf_put_att_text (ncid, vidGls, 'long_name', lvar,
     &                                  vname(2,indxGls)(1:lvar))
#  else
          text='averaged '/ /vname(2,indxGls)
          lvar=lenstr(text)
          ierr=nf_put_att_text (ncid, vidGls, 'long_name', lvar,
     &                                              text(1:lvar))
#  endif
          lvar=lenstr(vname(3,indxGls))
          ierr=nf_put_att_text (ncid, vidGls, 'units',     lvar,
     &                                  vname(3,indxGls)(1:lvar))
          lvar=lenstr(vname(4,indxGls))
          ierr=nf_put_att_text (ncid, vidGls, 'field',     lvar,
     &                                  vname(4,indxGls)(1:lvar))

          call nf_add_attribute(ncid, vidGls, indxGls, 5, NF_FOUT,
     &                                                            ierr)
        endif
!
! Vertical mixing turbulent length scale.
!
        if (wrt(indxLsc)) then
          lvar=lenstr(vname(1,indxLsc))
          ierr=nf_def_var (ncid, vname(1,indxLsc)(1:lvar),
     &                              NF_FOUT, 4, w3dgrd, vidLsc)
#ifdef NC4PAR
          ierr=nf_var_par_access(ncid,vidLsc,nf_collective)
#endif
#  ifndef AVRH
          lvar=lenstr(vname(2,indxLsc))
          ierr=nf_put_att_text (ncid, vidLsc, 'long_name', lvar,
     &                                  vname(2,indxLsc)(1:lvar))
#  else
          text='averaged '/ /vname(2,indxLsc)
          lvar=lenstr(text)
          ierr=nf_put_att_text (ncid, vidLsc, 'long_name', lvar,
     &                                              text(1:lvar))
#  endif
          lvar=lenstr(vname(3,indxLsc))
          ierr=nf_put_att_text (ncid, vidLsc, 'units',     lvar,
     &                                  vname(3,indxLsc)(1:lvar))
          lvar=lenstr(vname(4,indxLsc))
          ierr=nf_put_att_text (ncid, vidLsc, 'field',     lvar,
     &                                  vname(4,indxLsc)(1:lvar))

          call nf_add_attribute(ncid, vidLsc, indxLsc, 5, NF_FOUT,
     &                                                             ierr)
        endif
# endif /* GLS_MIXING */

!
! Shflx [W/m2]
!
# ifdef TEMPERATURE
        if (wrt(indxShflx)) then
          lvar=lenstr(vname(1,indxShflx))
          ierr=nf_def_var (ncid, vname(1,indxShflx)(1:lvar),
     &                             NF_FOUT, 3, r2dgrd, vidShflx)
#ifdef NC4PAR
          ierr=nf_var_par_access(ncid,vidShflx,nf_collective)
#endif
#  ifndef AVRH
          lvar=lenstr(vname(2,indxShflx))
          ierr=nf_put_att_text (ncid, vidShflx, 'long_name', lvar,
     &                                 vname(2,indxShflx)(1:lvar))
#  else
          text='averaged '/ /vname(2,indxShflx)
          lvar=lenstr(text)
          ierr=nf_put_att_text (ncid, vidShflx, 'long_name', lvar,
     &                                               text(1:lvar))
#  endif
          lvar=lenstr(vname(3,indxShflx))
          ierr=nf_put_att_text (ncid, vidShflx, 'units',     lvar,
     &                                 vname(3,indxShflx)(1:lvar))

          call nf_add_attribute(ncid, vidShflx, indxShflx, 5,
     &                          NF_FOUT, ierr)
        endif
# endif
!
! Swflx [cm/day]
!
# ifdef SALINITY
        if (wrt(indxSwflx)) then
          lvar=lenstr(vname(1,indxSwflx))
          ierr=nf_def_var (ncid, vname(1,indxSwflx)(1:lvar),
     &                             NF_FOUT, 3, r2dgrd, vidSwflx)
#ifdef NC4PAR
          ierr=nf_var_par_access(ncid,vidSwflx,nf_collective)
#endif
#  ifndef AVRH
          lvar=lenstr(vname(2,indxSwflx))
          ierr=nf_put_att_text (ncid, vidSwflx, 'long_name', lvar,
     &                                 vname(2,indxSwflx)(1:lvar))
#  else
          text='averaged '/ /vname(2,indxSwflx)
          lvar=lenstr(text)
          ierr=nf_put_att_text (ncid, vidSwflx, 'long_name', lvar,
     &                                               text(1:lvar))
#  endif
          lvar=lenstr(vname(3,indxSwflx))
          ierr=nf_put_att_text (ncid, vidSwflx, 'units',     lvar,
     &                                 vname(3,indxSwflx)(1:lvar))

          call nf_add_attribute(ncid, vidSwflx, indxSwflx, 5,
     &                          NF_FOUT, ierr)
        endif
# endif

# if defined BHFLUX && defined TEMPERATURE
!
! Bhflx [W/m2]
!
        if (wrt(indxBhflx)) then
          lvar=lenstr(vname(1,indxBhflx))
          ierr=nf_def_var (ncid, vname(1,indxBhflx)(1:lvar),
     &                             NF_FOUT, 3, r2dgrd, vidBhflx)
#ifdef NC4PAR
          ierr=nf_var_par_access(ncid,vidBhflx,nf_collective)
#endif
#  ifndef AVRH
          lvar=lenstr(vname(2,indxBhflx))
          ierr=nf_put_att_text (ncid, vidBhflx, 'long_name', lvar,
     &                                 vname(2,indxBhflx)(1:lvar))
#  else
          text='averaged '/ /vname(2,indxBhflx)
          lvar=lenstr(text)
          ierr=nf_put_att_text (ncid, vidBhflx, 'long_name', lvar,
     &                                               text(1:lvar))
#  endif
          lvar=lenstr(vname(3,indxBhflx))
          ierr=nf_put_att_text (ncid, vidBhflx, 'units',     lvar,
     &                                 vname(3,indxBhflx)(1:lvar))

          call nf_add_attribute(ncid, vidBhflx, indxBhflx, 5,
     &                          NF_FOUT, ierr)

        endif
# endif

# if defined BWFLUX && defined SALINITY
!
! Bwflx [cm/day]
!
        if (wrt(indxBwflx)) then
          lvar=lenstr(vname(1,indxBwflx))
          ierr=nf_def_var (ncid, vname(1,indxBwflx)(1:lvar),
     &                             NF_FOUT, 3, r2dgrd, vidBwflx)
#ifdef NC4PAR
          ierr=nf_var_par_access(ncid,vidBwflx,nf_collective)
#endif
#  ifndef AVRH
          lvar=lenstr(vname(2,indxBwflx))
          ierr=nf_put_att_text (ncid, vidBwflx, 'long_name', lvar,
     &                                 vname(2,indxBwflx)(1:lvar))
#  else
          text='averaged '/ /vname(2,indxBwflx)
          lvar=lenstr(text)
          ierr=nf_put_att_text (ncid, vidBwflx, 'long_name', lvar,
     &                                               text(1:lvar))
#  endif
          lvar=lenstr(vname(3,indxBwflx))
          ierr=nf_put_att_text (ncid, vidBwflx, 'units',     lvar,
     &                                 vname(3,indxBwflx)(1:lvar))

          call nf_add_attribute(ncid, vidBwflx, indxBwflx, 5,
     &                          NF_FOUT, ierr)

        endif
# endif
!
! Short wave Shflx_rsw [W/m2]
!
# ifdef TEMPERATURE
      if (wrt(indxShflx_rsw)) then
        lvar=lenstr(vname(1,indxShflx_rsw))
        ierr=nf_def_var (ncid, vname(1,indxShflx_rsw)(1:lvar),
     &                           NF_FOUT, 3, r2dgrd, vidShflx_rsw)
#ifdef NC4PAR
          ierr=nf_var_par_access(ncid,vidShflx_rsw,nf_collective)
#endif
#  ifndef AVRH
          lvar=lenstr(vname(2,indxShflx_rsw))
          ierr=nf_put_att_text (ncid, vidShflx_rsw, 'long_name', lvar,
     &                                 vname(2,indxShflx_rsw)(1:lvar))
#  else
          text='averaged '/ /vname(2,indxShflx_rsw)
          lvar=lenstr(text)
          ierr=nf_put_att_text (ncid, vidShflx_rsw, 'long_name', lvar,
     &                                               text(1:lvar))
#  endif
          lvar=lenstr(vname(3,indxShflx_rsw))
          ierr=nf_put_att_text (ncid, vidShflx_rsw, 'units',     lvar,
     &                                 vname(3,indxShflx_rsw)(1:lvar))

        call nf_add_attribute(ncid, vidShflx_rsw, indxShflx_rsw,5,
     &                                                   NF_FOUT, ierr)

      endif
# endif


# if (defined BULK_FLUX || defined OA_COUPLING) && defined TEMPERATURE
!
! Long wave Shflx [W/m2]
!
        if (wrt(indxShflx_rlw)) then
          lvar=lenstr(vname(1,indxShflx_rlw))
          ierr=nf_def_var (ncid, vname(1,indxShflx_rlw)(1:lvar),
     &                           NF_FOUT, 3, r2dgrd, vidShflx_rlw)
#ifdef NC4PAR
          ierr=nf_var_par_access(ncid,vidShflx_rlw,nf_collective)
#endif
#  ifndef AVRH
          lvar=lenstr(vname(2,indxShflx_rlw))
          ierr=nf_put_att_text (ncid, vidShflx_rlw, 'long_name', lvar,
     &                               vname(2,indxShflx_rlw)(1:lvar))
#  else
          text='averaged '/ /vname(2,indxShflx_rlw)
          lvar=lenstr(text)
          ierr=nf_put_att_text (ncid, vidShflx_rlw, 'long_name', lvar,
     &                                             text(1:lvar))
#  endif
          lvar=lenstr(vname(3,indxShflx_rlw))
          ierr=nf_put_att_text (ncid, vidShflx_rlw, 'units',     lvar,
     &                               vname(3,indxShflx_rlw)(1:lvar))

          call nf_add_attribute(ncid, vidShflx_rlw, indxShflx_rlw,5,
     &         NF_FOUT, ierr)
        endif
!
! Latent Shflx [W/m2]
!
      if (wrt(indxShflx_lat)) then
        lvar=lenstr(vname(1,indxShflx_lat))
        ierr=nf_def_var (ncid, vname(1,indxShflx_lat)(1:lvar),
     &                           NF_FOUT, 3, r2dgrd, vidShflx_lat)
#ifdef NC4PAR
          ierr=nf_var_par_access(ncid,vidShflx_lat,nf_collective)
#endif
#  ifndef AVRH
          lvar=lenstr(vname(2,indxShflx_lat))
          ierr=nf_put_att_text (ncid, vidShflx_lat, 'long_name', lvar,
     &                               vname(2,indxShflx_lat)(1:lvar))
#  else
          text='averaged '/ /vname(2,indxShflx_lat)
          lvar=lenstr(text)
          ierr=nf_put_att_text (ncid, vidShflx_lat, 'long_name', lvar,
     &                                               text(1:lvar))
#  endif
          lvar=lenstr(vname(3,indxShflx_lat))
          ierr=nf_put_att_text (ncid, vidShflx_lat, 'units',     lvar,
     &                                 vname(3,indxShflx_lat)(1:lvar))

          call nf_add_attribute(ncid, vidShflx_lat, indxShflx_lat, 5,
     &         NF_FOUT, ierr)
        endif
!
! Sensible Shflx [W/m2]
!
      if (wrt(indxShflx_sen)) then
        lvar=lenstr(vname(1,indxShflx_sen))
        ierr=nf_def_var (ncid, vname(1,indxShflx_sen)(1:lvar),
     &                           NF_FOUT, 3, r2dgrd, vidShflx_sen)
#ifdef NC4PAR
          ierr=nf_var_par_access(ncid,vidShflx_sen,nf_collective)
#endif
#  ifndef AVRH
          lvar=lenstr(vname(2,indxShflx_sen))
          ierr=nf_put_att_text (ncid, vidShflx_sen, 'long_name', lvar,
     &                                 vname(2,indxShflx_sen)(1:lvar))
#  else
          text='averaged '/ /vname(2,indxShflx_sen)
          lvar=lenstr(text)
          ierr=nf_put_att_text (ncid, vidShflx_sen, 'long_name', lvar,
     &                                               text(1:lvar))
#  endif
          lvar=lenstr(vname(3,indxShflx_sen))
          ierr=nf_put_att_text (ncid, vidShflx_sen, 'units',     lvar,
     &                                 vname(3,indxShflx_sen)(1:lvar))

        call nf_add_attribute(ncid, vidShflx_sen, indxShflx_sen, 5,
     &       NF_FOUT, ierr)

      endif
# endif  /* BULK_FLUX && TEMPERATURE  */

# if defined BIOLOGY & !defined PISCES
!
! Depth of the euphotic layer.
!
        if (wrt(indxHel)) then
          lvar=lenstr(vname(1,indxHel))
          ierr=nf_def_var (ncid, vname(1,indxHel)(1:lvar),
     &                             NF_FOUT, 3, r2dgrd, vidHel)
#ifdef NC4PAR
          ierr=nf_var_par_access(ncid,vidHel,nf_collective)
#endif
#  ifndef AVRH
          lvar=lenstr(vname(2,indxHel))
          ierr=nf_put_att_text (ncid, vidHel, 'long_name', lvar,
     &                                vname(2,indxHel)(1:lvar))
#  else
          text='averaged '/ /vname(2,indxHel)
          lvar=lenstr(text)
          ierr=nf_put_att_text (ncid, vidHel, 'long_name', lvar,
     &                                              text(1:lvar))
#  endif
          lvar=lenstr(vname(3,indxHel))
          ierr=nf_put_att_text (ncid, vidHel, 'units',     lvar,
     &                                  vname(3,indxHel)(1:lvar))
          lvar=lenstr(vname(4,indxHel))
          ierr=nf_put_att_text (ncid, vidHel, 'field',     lvar,
     &                                  vname(4,indxHel)(1:lvar))

          call nf_add_attribute(ncid, vidHel, indxHel, 5, NF_FOUT,
     &                                                           ierr)
        endif
!
! Chlorophyll/Carbon ratio
!
#  ifdef BIO_NChlPZD
        if (wrt(indxChC)) then
          lvar=lenstr(vname(1,indxChC))
          ierr=nf_def_var (ncid, vname(1,indxChC)(1:lvar),
     &                             NF_FOUT, 4, r3dgrd, vidChC)
#ifdef NC4PAR
          ierr=nf_var_par_access(ncid,vidChC,nf_collective)
#endif
#   ifndef AVRH
          lvar=lenstr(vname(2,indxChC))
          ierr=nf_put_att_text (ncid, vidChC, 'long_name', lvar,
     &                                  vname(2,indxChC)(1:lvar))
#   else
          text='averaged '/ /vname(2,indxChC)
          lvar=lenstr(text)
          ierr=nf_put_att_text (ncid, vidChC, 'long_name', lvar,
     &                                              text(1:lvar))
#   endif
          lvar=lenstr(vname(3,indxChC))
          ierr=nf_put_att_text (ncid, vidChC, 'units',     lvar,
     &                                  vname(3,indxChC)(1:lvar))
          lvar=lenstr(vname(4,indxChC))
          ierr=nf_put_att_text (ncid, vidChC, 'field',     lvar,
     &                                  vname(4,indxChC)(1:lvar))

          call nf_add_attribute(ncid, vidChC, indxChC, 5, NF_FOUT,
     &                                                            ierr)
        endif
#   ifdef OXYGEN
!
! Wind speed at 10 m
!
        if (wrt(indxU10)) then
          lvar=lenstr(vname(1,indxU10))
          ierr=nf_def_var (ncid, vname(1,indxU10)(1:lvar),
     &                             NF_FOUT, 3, r2dgrd, vidU10)
#ifdef NC4PAR
          ierr=nf_var_par_access(ncid,vidU10,nf_collective)
#endif
#  ifndef AVRH
          lvar=lenstr(vname(2,indxU10))
          ierr=nf_put_att_text (ncid, vidU10, 'long_name', lvar,
     &                                  vname(2,indxU10)(1:lvar))
#  else
          text='averaged '/ /vname(2,indxU10)
          lvar=lenstr(text)
          ierr=nf_put_att_text (ncid, vidU10, 'long_name', lvar,
     &                                              text(1:lvar))
#  endif
          lvar=lenstr(vname(3,indxU10))
          ierr=nf_put_att_text (ncid, vidU10, 'units',     lvar,
     &                                  vname(3,indxU10)(1:lvar))
          lvar=lenstr(vname(4,indxU10))
          ierr=nf_put_att_text (ncid, vidU10, 'field',     lvar,
     &                                  vname(4,indxU10)(1:lvar))

          call nf_add_attribute(ncid, vidU10, indxU10, 5, NF_FOUT,
     &                                                            ierr)
        endif
!
! Gas exchange coefficient of O2
!
        if (wrt(indxKvO2)) then
          lvar=lenstr(vname(1,indxKvO2))
          ierr=nf_def_var (ncid, vname(1,indxKvO2)(1:lvar),
     &                             NF_FOUT, 3, r2dgrd, vidKvO2)
#ifdef NC4PAR
          ierr=nf_var_par_access(ncid,vidKvO2,nf_collective)
#endif
#  ifndef AVRH
          lvar=lenstr(vname(2,indxKvO2))
          ierr=nf_put_att_text (ncid, vidKvO2, 'long_name', lvar,
     &                                  vname(2,indxKvO2)(1:lvar))
#  else
          text='averaged '/ /vname(2,indxKvO2)
          lvar=lenstr(text)
          ierr=nf_put_att_text (ncid, vidKvO2, 'long_name', lvar,
     &                                              text(1:lvar))
#  endif
          lvar=lenstr(vname(3,indxKvO2))
          ierr=nf_put_att_text (ncid, vidKvO2, 'units',     lvar,
     &                                  vname(3,indxKvO2)(1:lvar))
          lvar=lenstr(vname(4,indxKvO2))
          ierr=nf_put_att_text (ncid, vidKvO2, 'field',     lvar,
     &                                  vname(4,indxKvO2)(1:lvar))

          call nf_add_attribute(ncid, vidKvO2, indxKvO2, 5,
     &         NF_FOUT, ierr)
        endif
!
! Saturation concentration of O2
!
        if (wrt(indxO2sat)) then
          lvar=lenstr(vname(1,indxO2sat))
          ierr=nf_def_var (ncid, vname(1,indxO2sat)(1:lvar),
     &                             NF_FOUT, 3, r2dgrd, vidO2sat)
#ifdef NC4PAR
          ierr=nf_var_par_access(ncid,vidO2sat,nf_collective)
#endif
#  ifndef AVRH
          lvar=lenstr(vname(2,indxO2sat))
          ierr=nf_put_att_text (ncid, vidO2sat, 'long_name', lvar,
     &                                  vname(2,indxO2sat)(1:lvar))
#  else
          text='averaged '/ /vname(2,indxO2sat)
          lvar=lenstr(text)
          ierr=nf_put_att_text (ncid, vidO2sat, 'long_name', lvar,
     &                                              text(1:lvar))
#  endif
          lvar=lenstr(vname(3,indxO2sat))
          ierr=nf_put_att_text (ncid, vidO2sat, 'units',     lvar,
     &                                  vname(3,indxO2sat)(1:lvar))
          lvar=lenstr(vname(4,indxO2sat))
          ierr=nf_put_att_text (ncid, vidO2sat, 'field',     lvar,
     &                                  vname(4,indxO2sat)(1:lvar))

          call nf_add_attribute(ncid, vidO2sat, indxO2sat, 5,
     &         NF_FOUT, ierr)
        endif
#   endif /* OXYGEN */
#  endif
# endif
!
# ifdef SEDIMENT
!
! bed_thick  Sediment bed layer thickness (m)
! bed_poros  Bed porosity (-)
! bed_frac_sand/silt
!           volume fraction of size class ised in each bed layer (-)
! dflx_ ; e_flx_
!            Deposition/Erosion fluxes
! bdlu_ ; bdlv_
!
        indxWrk=indxSed-1
        if (wrt(indxWrk)) then
          lvar=lenstr(vname(1,indxWrk))
          ierr=nf_def_var (ncid, vname(1,indxWrk)(1:lvar),
     &                                NF_FOUT, 3, r2dgrd, vidSed(1))
#ifdef NC4PAR
          ierr=nf_var_par_access(ncid,vidSed(1),nf_collective)
#endif
#ifndef AVRH
          lvar=lenstr(vname(2,indxWrk))
          ierr=nf_put_att_text (ncid, vidSed(1), 'long_name', lvar,
     &                                  vname(2,indxWrk)(1:lvar))
#else
          text='averaged '/ /vname(2,indxWrk)
          lvar=lenstr(text)
          ierr=nf_put_att_text (ncid, vidSed(1), 'long_name', lvar,
     &                                            text(1:lvar))
#endif
          lvar=lenstr(vname(3,indxWrk))
          ierr=nf_put_att_text (ncid, vidSed(1), 'units',     lvar,
     &                                  vname(3,indxWrk)(1:lvar))
          lvar=lenstr(vname(4,indxWrk))
          ierr=nf_put_att_text (ncid, vidSed(1), 'field',     lvar,
     &                                  vname(4,indxWrk)(1:lvar))

          call nf_add_attribute(ncid, vidSed(1), indxWrk, 5, NF_FOUT,
     &         ierr)
        endif
!
        do itrc=1,2+NST
          indxWrk=indxSed+itrc-1
          if (wrt(indxWrk)) then
            lvar=lenstr(vname(1,indxWrk))
            ierr=nf_def_var (ncid, vname(1,indxWrk)(1:lvar),
     &                             NF_FOUT, 4, b3dgrd, vidSed(itrc+1))
#  ifdef NC4PAR
            ierr=nf_var_par_access(ncid,vidSed(itrc+1),nf_collective)
#  endif
#  ifndef AVRH
            lvar=lenstr(vname(2,indxWrk))
            ierr=nf_put_att_text (ncid,vidSed(itrc+1),'long_name', lvar,
     &                                    vname(2,indxWrk)(1:lvar))
#  else
            text='averaged '/ /vname(2,indxWrk)
            lvar=lenstr(text)
            ierr=nf_put_att_text (ncid,vidSed(itrc+1),'long_name', lvar,
     &                                                text(1:lvar))
#  endif
            lvar=lenstr(vname(3,indxWrk))
            ierr=nf_put_att_text (ncid, vidSed(itrc+1), 'units',   lvar,
     &                                    vname(3,indxWrk)(1:lvar))

            call nf_add_attribute(ncid, vidSed(itrc+1), indxWrk, 5,
     &                                                   NF_FOUT, ierr)
          endif
        enddo

        do itrc=2+NST+1,2+NST
#  ifdef SUSPLOAD
     &                 +2*NST
#  endif
#  ifdef BEDLOAD
     &                 +2*NST
#  endif
          indxWrk=indxSed+itrc-1
          if (wrt(indxWrk)) then
            lvar=lenstr(vname(1,indxWrk))
            ierr=nf_def_var (ncid, vname(1,indxWrk)(1:lvar),
     &                             NF_FOUT, 3, r2dgrd, vidSed(itrc+1))
#  ifdef NC4PAR
            ierr=nf_var_par_access(ncid,vidSed(itrc+1),nf_collective)
#  endif
#  ifndef AVRH
            lvar=lenstr(vname(2,indxWrk))
            ierr=nf_put_att_text (ncid,vidSed(itrc+1),'long_name', lvar,
     &                                    vname(2,indxWrk)(1:lvar))
#  else
            text='averaged '/ /vname(2,indxWrk)
            lvar=lenstr(text)
            ierr=nf_put_att_text (ncid,vidSed(itrc+1),'long_name', lvar,
     &                                                text(1:lvar))
#  endif
            lvar=lenstr(vname(3,indxWrk))
            ierr=nf_put_att_text (ncid,vidSed(itrc+1),'units',     lvar,
     &                                    vname(3,indxWrk)(1:lvar))

            call nf_add_attribute(ncid, vidSed(itrc+1), indxWrk, 5,
     &                                                 NF_FOUT, ierr)
          endif
        enddo
!
#  if defined MIXED_BED || defined COHESIVE_BED
        indxWrk=indxBTCR
        if (wrt(indxWrk)) then
          lvar=lenstr(vname(1,indxWrk))
          ierr=nf_def_var (ncid, vname(1,indxWrk)(1:lvar),
     &                             NF_FOUT, 4, b3dgrd, vidSed(itrc+1))
#   ifdef NC4PAR
          ierr=nf_var_par_access(ncid,vidSed(itrc+1),nf_collective)
#   endif
#   ifndef AVRH
          lvar=lenstr(vname(2,indxWrk))
          ierr=nf_put_att_text (ncid, vidSed(itrc+1), 'long_name', lvar,
     &                                  vname(2,indxWrk)(1:lvar))
#   else
          text='averaged '/ /vname(2,indxWrk)
          lvar=lenstr(text)
          ierr=nf_put_att_text (ncid, vidSed(itrc+1), 'long_name', lvar,
     &                                              text(1:lvar))
#   endif
          lvar=lenstr(vname(3,indxWrk))
          ierr=nf_put_att_text (ncid, vidSed(itrc+1), 'units',     lvar,
     &                                  vname(3,indxWrk)(1:lvar))

          call nf_add_attribute(ncid, vidSed(itrc+1), indxWrk, 5,
     &                                                 NF_FOUT, ierr)
        endif
#  endif
!
# endif /* SEDIMENT */

# if defined BBL && !defined AVRH
!
! Abed       Wind-induced, bed wave excursion amplitude (m)
! Hripple    Bed ripple height (m)
! Lripple    Bed ripple length (m)
! Zbnot      Physical hydraulic bottom roughness (grain roughness) (m)
! Zbapp      Effective apparent hydraulic bottom roughness
! bostrw     Wave-induced kinematic bottom stress (N/m2)
!
        do itrc=1,6
          indxWrk=indxBBL+itrc-1
          if (wrt(indxWrk)) then
            lvar=lenstr(vname(1,indxWrk))
            ierr=nf_def_var (ncid, vname(1,indxWrk)(1:lvar),
     &                               NF_FOUT, 3, r2dgrd, vidBBL(itrc))
#  ifdef NC4PAR
            ierr=nf_var_par_access(ncid,vidBBL(itrc),nf_collective)
#  endif
#  ifndef AVRH
            lvar=lenstr(vname(2,indxWrk))
            ierr=nf_put_att_text (ncid, vidBBL(itrc), 'long_name', lvar,
     &                                    vname(2,indxWrk)(1:lvar))
#  else
            text='averaged '/ /vname(2,indxWrk)
            lvar=lenstr(text)
            ierr=nf_put_att_text (ncid, vidBBL(itrc), 'long_name', lvar,
     &                                                text(1:lvar))
#  endif
            lvar=lenstr(vname(3,indxWrk))
            ierr=nf_put_att_text (ncid, vidBBL(itrc), 'units',     lvar,
     &                                    vname(3,indxWrk)(1:lvar))

            call nf_add_attribute(ncid, vidBBL(itrc), indxWrk, 5,
     &           NF_FOUT, ierr)
          endif
        enddo
# endif /* BBL */
# if defined SST_SKIN && defined TEMPERATURE
! Skin temperature
        if (wrt(indxT)) then
          lvar=lenstr(vname(1,indxSST_skin))
          ierr=nf_def_var (ncid, vname(1,indxSST_skin)(1:lvar),
     &                           NF_FOUT, 3, r2dgrd, vidSST_skin)
#ifdef NC4PAR
          ierr=nf_var_par_access(ncid,vidSST_skin,nf_collective)
#endif
#  ifndef AVRH
          lvar=lenstr(vname(2,indxSST_skin))
          ierr=nf_put_att_text (ncid, vidSST_skin, 'long_name', lvar,
     &                                 vname(2,indxSST_skin)(1:lvar))
#  else
          text='averaged '/ /vname(2,indxSST_skin)
          lvar=lenstr(text)
          ierr=nf_put_att_text (ncid, vidSST_skin, 'long_name', lvar,
     &                                                 text(1:lvar))
#  endif
          lvar=lenstr(vname(3,indxSST_skin))
          ierr=nf_put_att_text (ncid, vidSST_skin, 'units',     lvar,
     &                                 vname(3,indxSST_skin)(1:lvar))
          lvar=lenstr(vname(5,indxSST_skin))

          call nf_add_attribute(ncid, vidSST_skin, indxSST_skin,5,
     &         NF_FOUT , ierr)
        endif
# endif /* SST_SKIN */
!
!-----------------------------------------------------------------------
!  MUSTANG sediment model output
!-----------------------------------------------------------------------
!
# ifdef MUSTANG
        do itrc=1,3
          indx=indxT+ntrc_salt+ntrc_substot+itrc
          lvar=lenstr(vname(1,indx))
          ierr=nf_def_var (ncid, vname(1,indx)(1:lvar),
     &             NF_FOUT, 3, r2dgrd, vidM(itrc))  ! 2D: ksmax,Eptot,Tenfon
#  ifdef NC4PAR
          ierr=nf_var_par_access(ncid,vidM(itrc),nf_collective)
#  endif
          lvar=lenstr(vname(2,indx))
          ierr=nf_put_att_text (ncid, vidM(itrc), 'long_name', lvar,
     &                                  vname(2,indx)(1:lvar))
          lvar=lenstr(vname(3,indx))
          ierr=nf_put_att_text (ncid, vidM(itrc), 'units',     lvar,
     &                                  vname(3,indx)(1:lvar))
          lvar=lenstr(vname(4,indx))
          ierr=nf_put_att_text (ncid, vidM(itrc), 'field',     lvar,
     &                                  vname(4,indx)(1:lvar))
          call nf_add_attribute(ncid,vidM(itrc),indx,5, NF_FOUT, ierr)
        enddo
!
        do itrc=1,3
          indx=indxT+ntrc_salt+ntrc_substot+itrc+3
          ierr=nf_def_var (ncid, vname(1,indx),
     &               NF_FOUT, 4, b3dgrd, vidM(itrc+3)) ! 3D: dzs,tempsed,salsed
#  ifdef NC4PAR
          ierr=nf_var_par_access(ncid,vidM(itrc+3),nf_collective)
#  endif
          lvar=lenstr(vname(2,indx))
          ierr=nf_put_att_text (ncid, vidM(itrc+3), 'long_name', lvar,
     &                                  vname(2,indx)(1:lvar))
          lvar=lenstr(vname(3,indx))
          ierr=nf_put_att_text (ncid, vidM(itrc+3), 'units',     lvar,
     &                                  vname(3,indx)(1:lvar))
          lvar=lenstr(vname(4,indx))
          ierr=nf_put_att_text (ncid, vidM(itrc+3), 'field',     lvar,
     &                                  vname(4,indx)(1:lvar))
        enddo
        do itrc=1,ntrc_subs
          indx=indxT+ntrc_salt+ntrc_substot+itrc+6
          ierr=nf_def_var (ncid, vname(1,indx),
     &               NF_FOUT, 4, b3dgrd, vidM(itrc+6)) ! 3D: traceurSed
#  ifdef NC4PAR
          ierr=nf_var_par_access(ncid,vidM(itrc+6),nf_collective)
#  endif
          lvar=lenstr(vname(2,indx))
          ierr=nf_put_att_text (ncid, vidM(itrc+6), 'long_name', lvar,
     &                                  vname(2,indx)(1:lvar))
          lvar=lenstr(vname(3,indx))
          ierr=nf_put_att_text (ncid, vidM(itrc+6), 'units',     lvar,
     &                                  vname(3,indx)(1:lvar))
          lvar=lenstr(vname(4,indx))
          ierr=nf_put_att_text (ncid, vidM(itrc+6), 'field',     lvar,
     &                                  vname(4,indx)(1:lvar))
          call nf_add_attribute(ncid,vidM(itrc+6),indx, 5, NF_FOUT,ierr)
        enddo
#  ifdef key_MUSTANG_specif_outputs
        ik=0
        indx=indx+2 !ksmi,ksma
        do itrc=1,ntrc_subs
          do itrv=1,nv_out3Dnv_specif
            ik=ik+1
            indx=indx+1
            ierr=nf_def_var (ncid, vname(1,indx),
     &                 NF_FOUT, 3, r2dgrd, vidM(ik+6+ntrc_subs))  ! 3D: traceurSe
#   ifdef NC4PAR
            ierr=nf_var_par_access(ncid,vidM(ik+6+ntrc_subs),
     &                             nf_collective)
#   endif
          lvar=lenstr(vname(2,indx))
          ierr=nf_put_att_text (ncid, vidM(ik+6+ntrc_subs), 'long_name',
     &                                 lvar,vname(2,indx)(1:lvar))
          lvar=lenstr(vname(3,indx))
          ierr=nf_put_att_text (ncid, vidM(ik+6+ntrc_subs), 'units',
     &                                 lvar,vname(3,indx)(1:lvar))
          lvar=lenstr(vname(4,indx))
          ierr=nf_put_att_text (ncid, vidM(ik+6+ntrc_subs), 'field',
     &                                 lvar,vname(4,indx)(1:lvar))
            call nf_add_attribute(ncid,vidM(ik+6+ntrc_subs),indx,5,
     &                 NF_FOUT, ierr)
          enddo
        enddo
        do itrv=1,nv_out2D_specif
          ik=ik+1
          indx=indx+1
          ierr=nf_def_var (ncid, vname(1,indx),
     &              NF_FOUT, 3, r2dgrd, vidM(ik+6+ntrc_subs))  ! 3D: traceurSe
#   ifdef NC4PAR
          ierr=nf_var_par_access(ncid,vidM(ik+6+ntrc_subs),
     &                           nf_collective)
#   endif
          lvar=lenstr(vname(2,indx))
          ierr=nf_put_att_text (ncid, vidM(ik+6+ntrc_subs), 'long_name',
     &                                  lvar,vname(2,indx)(1:lvar))
          lvar=lenstr(vname(3,indx))
          ierr=nf_put_att_text (ncid, vidM(ik+6+ntrc_subs), 'units',
     &                                  lvar,vname(3,indx)(1:lvar))
          lvar=lenstr(vname(4,indx))
          ierr=nf_put_att_text (ncid, vidM(ik+6+ntrc_subs), 'field',
     &                                  lvar,vname(4,indx)(1:lvar))
          call nf_add_attribute(ncid,vidM(ik+6+ntrc_subs),indx, 5,
     &              NF_FOUT, ierr)
        enddo
#  endif
# endif /* MUSTANG */
!
#endif /* SOLVE3D */

#ifdef WAVE_IO
!
! Wave prognostic/diagnostic variables in XI/ETA direction.
! =========================================================
!
# ifndef WAVE_ROLLER
        do iwkb=1,7
# else
        do iwkb=1,9
# endif
          if (wrt(indxHRM+iwkb-1)) then
            lvar=lenstr(vname(1,indxHRM+iwkb-1))
            ierr=nf_def_var (ncid, vname(1,indxHRM+iwkb-1)(1:lvar),
     &                            NF_FOUT, 3, r2dgrd, vidWAVE(iwkb))
#ifdef NC4PAR
            ierr=nf_var_par_access(ncid,vidWAVE(iwkb),nf_collective)
#endif
# ifdef AVRH
            text='averaged '/ /vname(2,indxHRM+iwkb-1)
# else
            text=vname(2,indxHRM+iwkb-1)
# endif
            lvar=lenstr(text)
            ierr=nf_put_att_text (ncid, vidWAVE(iwkb), 'long_name',
     &                                         lvar, text(1:lvar))
            lvar=lenstr(vname(3,indxHRM+iwkb-1))
            ierr=nf_put_att_text (ncid, vidWAVE(iwkb), 'units', lvar,
     &                               vname(3,indxHRM+iwkb-1)(1:lvar))
          endif
        enddo
#endif /* WAVE_IO */
!
#ifdef MRL_WCI
!
! Wave-current interaction diagnostic variables.
! =============================================
!
! sup: quasi-static sea-level response (wave set-up/down)
!
        if (wrt(indxSUP)) then
          lvar=lenstr(vname(1,indxSUP))
          ierr=nf_def_var (ncid, vname(1,indxSUP)(1:lvar), NF_FOUT,
     &                                           3, r2dgrd, vidSUP)
#ifdef NC4PAR
          ierr=nf_var_par_access(ncid,vidSUP,nf_collective)
#endif

#  ifndef AVRH
          lvar=lenstr(vname(2,indxSUP))
          ierr=nf_put_att_text (ncid, vidSUP, 'long_name', lvar,
     &                                  vname(2,indxSUP)(1:lvar))
#  else
          text='averaged '/ /vname(2,indxSUP)
          lvar=lenstr(text)
          ierr=nf_put_att_text (ncid, vidSUP, 'long_name', lvar,
     &                                              text(1:lvar))
#  endif
          lvar=lenstr(vname(3,indxSUP))
          ierr=nf_put_att_text (ncid, vidSUP, 'units',     lvar,
     &                                 vname(3,indxSUP)(1:lvar))
        endif
!
! ust2d & vst2d: 2D, depth-averaged Stokes drift velocities.
!
        if (wrt(indxUST2D)) then
          lvar=lenstr(vname(1,indxUST2D))
          ierr=nf_def_var (ncid, vname(1,indxUST2D)(1:lvar), NF_FOUT,
     &                                           3, u2dgrd, vidUST2D)
#ifdef NC4PAR
          ierr=nf_var_par_access(ncid,vidUST2D,nf_collective)
#endif
#  ifndef AVRH
          lvar=lenstr(vname(2,indxUST2D))
          ierr=nf_put_att_text (ncid, vidUST2D, 'long_name', lvar,
     &                                  vname(2,indxUST2D)(1:lvar))
#  else
          text='averaged '/ /vname(2,indxUST2D)
          lvar=lenstr(text)
          ierr=nf_put_att_text (ncid, vidUST2D, 'long_name', lvar,
     &                                              text(1:lvar))
#  endif
          lvar=lenstr(vname(3,indxUST2D))
          ierr=nf_put_att_text (ncid, vidUST2D, 'units',     lvar,
     &                                 vname(3,indxUST2D)(1:lvar))
        endif
!
        if (wrt(indxVST2D)) then
          lvar=lenstr(vname(1,indxVST2D))
          ierr=nf_def_var (ncid, vname(1,indxVST2D)(1:lvar), NF_FOUT,
     &                                           3, v2dgrd, vidVST2D)
#ifdef NC4PAR
          ierr=nf_var_par_access(ncid,vidVST2D,nf_collective)
#endif
#  ifndef AVRH
          lvar=lenstr(vname(2,indxVST2D))
          ierr=nf_put_att_text (ncid, vidVST2D, 'long_name', lvar,
     &                                  vname(2,indxVST2D)(1:lvar))
#  else
          text='averaged '/ /vname(2,indxVST2D)
          lvar=lenstr(text)
          ierr=nf_put_att_text (ncid, vidVST2D, 'long_name', lvar,
     &                                              text(1:lvar))
#  endif
          lvar=lenstr(vname(3,indxVST2D))
          ierr=nf_put_att_text (ncid, vidVST2D, 'units',     lvar,
     &                                  vname(3,indxVST2D)(1:lvar))
        endif
# ifdef SOLVE3D
!
! ust3d & vst3d: 3D Stokes drift velocities.
!
        if (wrt(indxUST)) then
          lvar=lenstr(vname(1,indxUST))
          ierr=nf_def_var (ncid, vname(1,indxUST)(1:lvar), NF_FOUT,
     &                                           4, u3dgrd, vidUST)
#ifdef NC4PAR
          ierr=nf_var_par_access(ncid,vidUST,nf_collective)
#endif
#  ifndef AVRH
          lvar=lenstr(vname(2,indxUST))
          ierr=nf_put_att_text (ncid, vidUST, 'long_name', lvar,
     &                                  vname(2,indxUST)(1:lvar))
#  else
          text='averaged '/ /vname(2,indxUST)
          lvar=lenstr(text)
          ierr=nf_put_att_text (ncid, vidUST, 'long_name', lvar,
     &                                              text(1:lvar))
#  endif
          lvar=lenstr(vname(3,indxUST))
          ierr=nf_put_att_text (ncid, vidUST, 'units',     lvar,
     &                                 vname(3,indxUST)(1:lvar))
        endif
!
        if (wrt(indxVST)) then
          lvar=lenstr(vname(1,indxVST))
          ierr=nf_def_var (ncid, vname(1,indxVST)(1:lvar), NF_FOUT,
     &                                           4, v3dgrd, vidVST)
#ifdef NC4PAR
          ierr=nf_var_par_access(ncid,vidVST,nf_collective)
#endif
#  ifndef AVRH
          lvar=lenstr(vname(2,indxVST))
          ierr=nf_put_att_text (ncid, vidVST, 'long_name', lvar,
     &                                  vname(2,indxVST)(1:lvar))
#  else
          text='averaged '/ /vname(2,indxUST)
          lvar=lenstr(text)
          ierr=nf_put_att_text (ncid, vidVST, 'long_name', lvar,
     &                                              text(1:lvar))
#  endif
          lvar=lenstr(vname(3,indxVST))
          ierr=nf_put_att_text (ncid, vidVST, 'units',     lvar,
     &                                 vname(3,indxVST)(1:lvar))
        endif
!
! wst: vertical Stokes drift velocity (m/s) at rho-point
!
        if (wrt(indxWST)) then
          lvar=lenstr(vname(1,indxWST))
          ierr=nf_def_var (ncid, vname(1,indxWST)(1:lvar), NF_FOUT,
     &                                           4, r3dgrd, vidWST)
#ifdef NC4PAR
          ierr=nf_var_par_access(ncid,vidWST,nf_collective)
#endif
#  ifndef AVRH
          lvar=lenstr(vname(2,indxWST))
          ierr=nf_put_att_text (ncid, vidWST, 'long_name', lvar,
     &                                  vname(2,indxWST)(1:lvar))
#  else
          text='averaged '/ /vname(2,indxWST)
          lvar=lenstr(text)
          ierr=nf_put_att_text (ncid, vidWST, 'long_name', lvar,
     &                                              text(1:lvar))
#  endif
          lvar=lenstr(vname(3,indxWST))
          ierr=nf_put_att_text (ncid, vidWST, 'units',     lvar,
     &                                 vname(3,indxWST)(1:lvar))
        endif
!
!
! Akb: Vertical eddy viscosity coefficient due to depth-induced wave breaking.
!
        if (wrt(indxAkb)) then
          lvar=lenstr(vname(1,indxAkb))
          ierr=nf_def_var (ncid, vname(1,indxAkb)(1:lvar), NF_FOUT,
     &                                           4, w3dgrd, vidAkb)
#ifdef NC4PAR
          ierr=nf_var_par_access(ncid,vidAkb,nf_collective)
#endif
#  ifndef AVRH
          lvar=lenstr(vname(2,indxAkb))
          ierr=nf_put_att_text (ncid, vidAkb, 'long_name', lvar,
     &                                  vname(2,indxAkb)(1:lvar))
#  else
          text='averaged '/ /vname(2,indxAkb)
          lvar=lenstr(text)
          ierr=nf_put_att_text (ncid, vidAkb, 'long_name', lvar,
     &                                              text(1:lvar))
#  endif
          lvar=lenstr(vname(3,indxAkb))
          ierr=nf_put_att_text (ncid, vidAkb, 'units',     lvar,
     &                                  vname(3,indxAkb)(1:lvar))
        endif
!
! Akw: Vertical eddy diffusivity coefficient due to primary waves.
!
        if (wrt(indxAkw)) then
          lvar=lenstr(vname(1,indxAkw))
          ierr=nf_def_var (ncid, vname(1,indxAkw)(1:lvar), NF_FOUT,
     &                                           4, w3dgrd, vidAkw)
#ifdef NC4PAR
          ierr=nf_var_par_access(ncid,vidAkw,nf_collective)
#endif
#  ifndef AVRH
          lvar=lenstr(vname(2,indxAkw))
          ierr=nf_put_att_text (ncid, vidAkw, 'long_name', lvar,
     &                                  vname(2,indxAkw)(1:lvar))
#  else
          text='averaged '/ /vname(2,indxAkw)
          lvar=lenstr(text)
          ierr=nf_put_att_text (ncid, vidAkw, 'long_name', lvar,
     &                                              text(1:lvar))
#  endif
          lvar=lenstr(vname(3,indxAkw))
          ierr=nf_put_att_text (ncid, vidAkw, 'units',     lvar,
     &                                  vname(3,indxAkw)(1:lvar))
        endif
!
! kvf: vertical vortex force term (u^St du/dz)
!
        if (wrt(indxKVF)) then
          lvar=lenstr(vname(1,indxKVF))
          ierr=nf_def_var (ncid, vname(1,indxKVF)(1:lvar), NF_FOUT,
     &                                           4, r3dgrd, vidKVF)
#ifdef NC4PAR
          ierr=nf_var_par_access(ncid,vidKVF,nf_collective)
#endif
#  ifndef AVRH
          lvar=lenstr(vname(2,indxKVF))
          ierr=nf_put_att_text (ncid, vidKVF, 'long_name', lvar,
     &                                  vname(2,indxKVF)(1:lvar))
#  else
          text='averaged '/ /vname(2,indxKVF)
          lvar=lenstr(text)
          ierr=nf_put_att_text (ncid, vidKVF, 'long_name', lvar,
     &                                              text(1:lvar))
#  endif
          lvar=lenstr(vname(3,indxKVF))
          ierr=nf_put_att_text (ncid, vidKVF, 'units',     lvar,
     &                                 vname(3,indxKVF)(1:lvar))
        endif
!
! calP: surface pressure correction term appeared in prsgrd term
!
        if (wrt(indxCALP)) then
          lvar=lenstr(vname(1,indxCALP))
          ierr=nf_def_var (ncid, vname(1,indxCALP)(1:lvar), NF_FOUT,
     &                                           3, r2dgrd, vidCALP)
#ifdef NC4PAR
          ierr=nf_var_par_access(ncid,vidCALP,nf_collective)
#endif
#  ifndef AVRH
          lvar=lenstr(vname(2,indxCALP))
          ierr=nf_put_att_text (ncid, vidCALP, 'long_name', lvar,
     &                                  vname(2,indxCALP)(1:lvar))
#  else
          text='averaged '/ /vname(2,indxCALP)
          lvar=lenstr(text)
          ierr=nf_put_att_text (ncid, vidCALP, 'long_name', lvar,
     &                                              text(1:lvar))
#  endif
          lvar=lenstr(vname(3,indxCALP))
          ierr=nf_put_att_text (ncid, vidCALP, 'units',     lvar,
     &                                 vname(3,indxCALP)(1:lvar))
        endif
!
! Kapsrf: surface Bernoulli head term appeared in prsgrd term
!
        if (wrt(indxKAPS)) then
          lvar=lenstr(vname(1,indxKAPS))
          ierr=nf_def_var (ncid, vname(1,indxKAPS)(1:lvar), NF_FOUT,
     &                                           3, r2dgrd, vidKAPS)
#ifdef NC4PAR
          ierr=nf_var_par_access(ncid,vidKAPS,nf_collective)
#endif
#  ifndef AVRH
          lvar=lenstr(vname(2,indxKAPS))
          ierr=nf_put_att_text (ncid, vidKAPS, 'long_name', lvar,
     &                                  vname(2,indxKAPS)(1:lvar))
#  else
          text='averaged '/ /vname(2,indxKAPS)
          lvar=lenstr(text)
          ierr=nf_put_att_text (ncid, vidKAPS, 'long_name', lvar,
     &                                              text(1:lvar))
#  endif
          lvar=lenstr(vname(3,indxKAPS))
          ierr=nf_put_att_text (ncid, vidKAPS, 'units',     lvar,
     &                                 vname(3,indxKAPS)(1:lvar))
        endif
# endif  /* SOLVE3D */
#endif   /* MRL_WCI */

!
! Leave definition mode.
!
!
        ierr=nf_enddef(ncid)
        MPI_master_only write(stdout,'(6x,4A,1x,A,i4)')
     &                'DEF_HIS/AVG - Created ',
     &                'new netCDF file ''', ncname(1:lstr), '''.'
     &                 MYID
!
!===================================================================
! Open an existing file and prepare for appending data.
! ==== == ======== ==== === ======= === ========= =====
! Inquire about the dimensions and variables. Check for
! consistency with model dimensions. In the case when file
! is rejected (whether it cannot be opened, or something
! is wrong with its dimensions) create a new file.
!
! After that verify that all necessary variables are already
! defined there and find their netCDF IDs.
!===================================================================
!
      elseif (ncid.eq.-1) then
#ifndef NC4PAR
        ierr=nf_open (ncname(1:lstr), nf_write, ncid)
#else
        ierr=nf_open_par (ncname(1:lstr), IOR(nf_write, nf_mpiio),
     &     MPI_COMM_WORLD, MPI_INFO_NULL, ncid)
#endif
        if (ierr. eq. nf_noerr) then
          ierr=checkdims (ncid, ncname, lstr, rec)
          if (ierr .eq. nf_noerr) then
            if (rec_per_file.eq.0) then
              ierr=rec+1 - total_rec
            else
              ierr=rec+1 - (1+mod(total_rec-1, rec_per_file))
            endif
            if (ierr.gt.0) then
              MPI_master_only write( stdout,
     &                 '(/1x,A,I5,1x,A/8x,3A,I5,/8x,A,I5,1x,A/)'
     &            ) 'DEF_HIS/AVG WARNING: Actual number of records',
     &               rec,  'in netCDF file',  '''',  ncname(1:lstr),
     &             ''' exceeds the record number from restart data',
     &             rec+1-ierr,'/', total_rec,', restart is assumed.'
              rec=rec-ierr
            elseif (rec_per_file.eq.0) then
              total_rec=rec+1           ! <-- set to the next record
#if defined MPI & !defined PARALLEL_FILES
              if (mynode.gt.0) total_rec=total_rec-1
#endif
            endif
            ierr=nf_noerr
          endif
        endif
!
        if (ierr. ne. nf_noerr) then
#if defined MPI & !defined PARALLEL_FILES & !defined NC4PAR
          if (mynode.eq.0) then
            create_new_file=.true.
            goto 10
          else
            MPI_master_only write(stdout,'(/1x,4A,2x,A,I4/)')
     &                  'DEF_HIS/AVG ERROR: ',
     &                  'Cannot open file ''', ncname(1:lstr), '''.'
     &                   MYID
            goto 99                                       !--> ERROR
          endif
#else
          create_new_file=.true.
          goto 10
#endif
        endif
!
! Find netCDF IDs of evolving model variables:
! ---- ------ --- -- -------- ----- ----------
!
! Time step indices:
!
        ierr=nf_inq_varid (ncid, 'time_step', vidTstep)
        if (ierr .ne. nf_noerr) then
          write(stdout,1) 'time_step', ncname(1:lstr)
          goto 99                                         !--> ERROR
        endif
!
! Time.
!
        lvar=lenstr(vname(1,indxTime))
        ierr=nf_inq_varid (ncid,vname(1,indxTime)(1:lvar),vidTime)
        if (ierr .ne. nf_noerr) then
          write(stdout,1) vname(1,indxTime)(1:lvar), ncname(1:lstr)
          goto 99                                         !--> ERROR
        endif
!
! Time2.
!
        lvar=lenstr(vname(1,indxTime2))
        ierr=nf_inq_varid (ncid,vname(1,indxTime2)(1:lvar),vidTime2)
        if (ierr .ne. nf_noerr) then
          write(stdout,1) vname(1,indxTime2)(1:lvar), ncname(1:lstr)
          goto 99                                         !--> ERROR
        endif
!
! Free-surface.
!
        if (wrt(indxZ)) then
          lvar=lenstr(vname(1,indxZ))
          ierr=nf_inq_varid (ncid, vname(1,indxZ)(1:lvar), vidZ)
          if (ierr .ne. nf_noerr) then
            write(stdout,1) vname(1,indxZ)(1:lvar), ncname(1:lstr)
            goto 99                                       !--> ERROR
          endif
        endif
!
! 2D momenta in XI- and ETA-directions.
!
        if (wrt(indxUb)) then
          lvar=lenstr(vname(1,indxUb))
          ierr=nf_inq_varid (ncid, vname(1,indxUb)(1:lvar), vidUb)
          if (ierr .ne. nf_noerr) then
            write(stdout,1) vname(1,indxUb)(1:lvar), ncname(1:lstr)
            goto 99                                       !--> ERROR
          endif
        endif

        if (wrt(indxVb)) then
          lvar=lenstr(vname(1,indxVb))
          ierr=nf_inq_varid (ncid, vname(1,indxVb)(1:lvar), vidVb)
          if (ierr .ne. nf_noerr) then
            write(stdout,1) vname(1,indxVb)(1:lvar), ncname(1:lstr)
            goto 99                                       !--> ERROR
          endif
        endif
!
! time evolving bathymetry
!
# ifdef MORPHODYN
        if (wrt(indxHm)) then
          lvar=lenstr(vname(1,indxHm))
          ierr=nf_inq_varid (ncid, vname(1,indxHm)(1:lvar), vidHm)
          if (ierr .ne. nf_noerr) then
            write(stdout,1) vname(1,indxHm)(1:lvar), ncname(1:lstr)
            goto 99                                       !--> ERROR
          endif
        endif
# endif
!
! bostr     Kinematic bottom stress (N/m2)
!
        if (wrt(indxBostr)) then
          lvar=lenstr(vname(1,indxBostr))
          ierr=nf_inq_varid (ncid,vname(1,indxBostr)(1:lvar),
     &                                                   vidBostr)
          if (ierr .ne. nf_noerr) then
            write(stdout,1) vname(1,indxBostr)(1:lvar), ncname(1:lstr)
            goto 99                                       !--> ERROR
          endif
        endif


!
! bustr     Kinematic U bottom stress (N/m2)
!
        if (wrt(indxBustr)) then
          lvar=lenstr(vname(1,indxBustr))
          ierr=nf_inq_varid (ncid,vname(1,indxBustr)(1:lvar),
     &                                                   vidBustr)
          if (ierr .ne. nf_noerr) then
          write(stdout,1) vname(1,indxBustr)(1:lvar), ncname(1:lstr)
            goto 99                                       !--> ERROR
          endif
        endif


!
! bvstr     Kinematic V bottom stress (N/m2)
!
        if (wrt(indxBvstr)) then
          lvar=lenstr(vname(1,indxBvstr))
          ierr=nf_inq_varid (ncid,vname(1,indxBvstr)(1:lvar),
     &                                                   vidBvstr)
          if (ierr .ne. nf_noerr) then
          write(stdout,1) vname(1,indxBvstr)(1:lvar), ncname(1:lstr)
            goto 99                                       !--> ERROR
          endif
        endif
!
! wstr     Kinematic surface wind stress (N/m2)
!
        if (wrt(indxWstr)) then
          lvar=lenstr(vname(1,indxWstr))
          ierr=nf_inq_varid (ncid,vname(1,indxWstr)(1:lvar),
     &                                                   vidWstr)
          if (ierr .ne. nf_noerr) then
            write(stdout,1) vname(1,indxWstr)(1:lvar), ncname(1:lstr)
            goto 99                                       !--> ERROR
          endif
        endif
!
! uwstr     Kinematic surface u wind stress component (N/m2)
!
        if (wrt(indxUWstr)) then
          lvar=lenstr(vname(1,indxUWstr))
          ierr=nf_inq_varid (ncid,vname(1,indxUWstr)(1:lvar),
     &                                                   vidUWstr)
          if (ierr .ne. nf_noerr) then
            write(stdout,1) vname(1,indxUWstr)(1:lvar), ncname(1:lstr)
            goto 99                                       !--> ERROR
          endif
        endif
!
! vwstr     Kinematic surface v wind stress component (N/m2)
!
        if (wrt(indxVWstr)) then
          lvar=lenstr(vname(1,indxVWstr))
          ierr=nf_inq_varid (ncid,vname(1,indxVWstr)(1:lvar),
     &                                                   vidVWstr)
          if (ierr .ne. nf_noerr) then
            write(stdout,1) vname(1,indxVWstr)(1:lvar), ncname(1:lstr)
            goto 99                                       !--> ERROR
          endif
        endif
#ifdef SOLVE3D
!
! 3D momenta in XI- and ETA-directions.
!
        if (wrt(indxU)) then
          lvar=lenstr(vname(1,indxU))
          ierr=nf_inq_varid (ncid, vname(1,indxU)(1:lvar), vidU)
          if (ierr .ne. nf_noerr) then
            write(stdout,1) vname(1,indxU)(1:lvar), ncname(1:lstr)
            goto 99                                       !--> ERROR
          endif
        endif
!
        if (wrt(indxV)) then
          lvar=lenstr(vname(1,indxV))
          ierr=nf_inq_varid (ncid, vname(1,indxV)(1:lvar), vidV)
          if (ierr .ne. nf_noerr) then
            write(stdout,1) vname(1,indxV)(1:lvar), ncname(1:lstr)
            goto 99                                       !--> ERROR
          endif
        endif
!
! Tracer variables.
!
# ifdef TRACERS
        do itrc=1,NT
          if (wrt(indxV+itrc)) then
            lvar=lenstr(vname(1,indxV+itrc))
            ierr=nf_inq_varid (ncid, vname(1,indxV+itrc)(1:lvar),
     &                                                 vidT(itrc))
            if (ierr .ne. nf_noerr) then
              write(stdout,1) vname(1,indxV+itrc)(1:lvar),
     &                                       ncname(1:lstr)
              goto 99                                     !--> ERROR
            endif
          endif
        enddo
# endif
!
! Density anomaly.
!
        if (wrt(indxR)) then
          lvar=lenstr(vname(1,indxR))
          ierr=nf_inq_varid (ncid, vname(1,indxR)(1:lvar), vidR)
          if (ierr .ne. nf_noerr) then
            write(stdout,1) vname(1,indxR)(1:lvar), ncname(1:lstr)
            goto 99                                       !--> ERROR
          endif
        endif
# if defined ANA_VMIX || defined BVF_MIXING \
  || defined LMD_MIXING || defined LMD_SKPP || defined LMD_BKPP \
  || defined GLS_MIXING
!-----------------------------------------------------------------------------
!
! Brunt Vaisala Frequency.
!
        if (wrt(indxbvf)) then
          lvar=lenstr(vname(1,indxbvf))
          ierr=nf_inq_varid (ncid, vname(1,indxbvf)(1:lvar), vidbvf)
          if (ierr .ne. nf_noerr) then
            write(stdout,1) vname(1,indxbvf)(1:lvar), ncname(1:lstr)
            goto 99                                       !--> ERROR
          endif
        endif
# endif
!-----------------------------------------------------------------------------
! AOU and pCO2
!
#  if defined BIOLOGY && defined BIO_BioEBUS
        if (wrt(indxAOU)) then
          lvar=lenstr(vname(1,indxAOU))
          ierr=nf_inq_varid (ncid, vname(1,indxAOU)(1:lvar),
     &                                               vidAOU)
          if (ierr .ne. nf_noerr) then
            write(stdout,1) vname(1,indxAOU)(1:lvar), ncname(1:lstr)
            goto 99                                       !--> ERROR
          endif
        endif
        if (wrt(indxWIND10)) then
          lvar=lenstr(vname(1,indxWIND10))
          ierr=nf_inq_varid (ncid, vname(1,indxWIND10)(1:lvar),
     &                                             vidwind10)
          if (ierr .ne. nf_noerr) then
            write(stdout,1) vname(1,indxWIND10)(1:lvar),ncname(1:lstr)
            goto 99                                       !--> ERROR
          endif
        endif
#  endif
!
! S-coordinate "omega" vertical velocity.
!
        if (wrt(indxO)) then
          lvar=lenstr(vname(1,indxO))
          ierr=nf_inq_varid (ncid, vname(1,indxO)(1:lvar), vidO)
          if (ierr .ne. nf_noerr) then
            write(stdout,1) vname(1,indxO)(1:lvar), ncname(1:lstr)
            goto 99                                       !--> ERROR
          endif
        endif
!
! True vertical velocity.
!
        if (wrt(indxW)) then
          lvar=lenstr(vname(1,indxW))
          ierr=nf_inq_varid (ncid, vname(1,indxW)(1:lvar), vidW)
          if (ierr .ne. nf_noerr) then
            write(stdout,1) vname(1,indxW)(1:lvar), ncname(1:lstr)
            goto 99                                       !--> ERROR
          endif
        endif
# ifdef VIS_COEF_3D
!
! Horizontal viscosity coefficient.
!
        if (wrt(indxVisc)) then
          lvar=lenstr(vname(1,indxVisc))
          ierr=nf_inq_varid (ncid, vname(1,indxVisc)(1:lvar), vidVisc)
          if (ierr .ne. nf_noerr) then
            write(stdout,1) vname(1,indxVisc)(1:lvar), ncname(1:lstr)
            goto 99                                       !--> ERROR
          endif
        endif
# endif
# ifdef DIF_COEF_3D
!
! Horizontal diffusivity coefficient.
!
        if (wrt(indxDiff)) then
          lvar=lenstr(vname(1,indxDiff))
          ierr=nf_inq_varid (ncid, vname(1,indxDiff)(1:lvar), vidDiff)
          if (ierr .ne. nf_noerr) then
            write(stdout,1) vname(1,indxDiff)(1:lvar), ncname(1:lstr)
            goto 99                                       !--> ERROR
          endif
        endif
# endif
!
! Vertical viscosity coefficient.
!
        if (wrt(indxAkv)) then
          lvar=lenstr(vname(1,indxAkv))
          ierr=nf_inq_varid (ncid, vname(1,indxAkv)(1:lvar), vidAkv)
          if (ierr .ne. nf_noerr) then
            write(stdout,1) vname(1,indxAkv)(1:lvar), ncname(1:lstr)
            goto 99                                       !--> ERROR
          endif
        endif
!
! Vertical diffusion coefficient for potential temperature.
!
# ifdef TEMPERATURE
        if (wrt(indxAkt)) then
          lvar=lenstr(vname(1,indxAkt))
          ierr=nf_inq_varid (ncid,vname(1,indxAkt)(1:lvar), vidAkt)
          if (ierr .ne. nf_noerr) then
            write(stdout,1) vname(1,indxAkt)(1:lvar), ncname(1:lstr)
            goto 99                                       !--> ERROR
          endif
        endif
# endif

# ifdef SALINITY
!
! Vertical diffusion coefficient for salinity.
!
        if (wrt(indxAks)) then
          lvar=lenstr(vname(1,indxAks))
          ierr=nf_inq_varid (ncid,vname(1,indxAks)(1:lvar), vidAks)
          if (ierr .ne. nf_noerr) then
            write(stdout,1) vname(1,indxAks)(1:lvar), ncname(1:lstr)
            goto 99                                       !--> ERROR
          endif
        endif
# endif


# if defined LMD_SKPP || defined GLS_MIXING
!
! Depth of surface planetary boundary layer.
!
        if (wrt(indxHbl)) then
          lvar=lenstr(vname(1,indxHbl))
          ierr=nf_inq_varid (ncid,vname(1,indxHbl)(1:lvar), vidHbl)
          if (ierr .ne. nf_noerr) then
            write(stdout,1) vname(1,indxHbl)(1:lvar), ncname(1:lstr)
            goto 99                                       !--> ERROR
          endif
        endif
# endif
# ifdef LMD_BKPP
!
! Depth of bottom planetary boundary layer.
!
        if (wrt(indxHbbl)) then
          lvar=lenstr(vname(1,indxHbbl))
          ierr=nf_inq_varid (ncid,vname(1,indxHbbl)(1:lvar), vidHbbl)
          if (ierr .ne. nf_noerr) then
            write(stdout,1) vname(1,indxHbbl)(1:lvar), ncname(1:lstr)
            goto 99                                       !--> ERROR
          endif
        endif
# endif

# ifdef GLS_MIXING
!
! Turbulent Kinetic Energy.
!
        if (wrt(indxTke)) then
          lvar=lenstr(vname(1,indxTke))
          ierr=nf_inq_varid (ncid,vname(1,indxTke)(1:lvar), vidTke)
          if (ierr .ne. nf_noerr) then
            write(stdout,1) vname(1,indxTke)(1:lvar), ncname(1:lstr)
            goto 99                                       !--> ERROR
          endif
        endif
!
! Generic Length Scale.
!
        if (wrt(indxGls)) then
          lvar=lenstr(vname(1,indxGls))
          ierr=nf_inq_varid (ncid,vname(1,indxGls)(1:lvar), vidGls)
          if (ierr .ne. nf_noerr) then
            write(stdout,1) vname(1,indxGls)(1:lvar), ncname(1:lstr)
            goto 99                                       !--> ERROR
          endif
        endif
!
! Turbulent Length Scale.
!
        if (wrt(indxLsc)) then
          lvar=lenstr(vname(1,indxLsc))
          ierr=nf_inq_varid (ncid,vname(1,indxLsc)(1:lvar), vidLsc)
          if (ierr .ne. nf_noerr) then
            write(stdout,1) vname(1,indxLsc)(1:lvar), ncname(1:lstr)
            goto 99                                       !--> ERROR
          endif
        endif
# endif

!
! Shflx     Surface net heat flux [W/m2]
!
# ifdef TEMPERATURE
        if (wrt(indxShflx)) then
          lvar=lenstr(vname(1,indxShflx))
          ierr=nf_inq_varid (ncid,vname(1,indxShflx)(1:lvar),
     &                                                   vidShflx)
          if (ierr .ne. nf_noerr) then
            write(stdout,1) vname(1,indxShflx)(1:lvar), ncname(1:lstr)
            goto 99                                       !--> ERROR
          endif
        endif
#  ifdef BHFLUX
!
! Bhflx     Bottom net heat flux [W/m2]
!
        if (wrt(indxBhflx)) then
          lvar=lenstr(vname(1,indxBhflx))
          ierr=nf_inq_varid (ncid,vname(1,indxBhflx)(1:lvar),
     &                                                   vidBhflx)
          if (ierr .ne. nf_noerr) then
          write(stdout,1) vname(1,indxBhflx)(1:lvar), ncname(1:lstr)
            goto 99                                       !--> ERROR
          endif
        endif
#  endif
# endif

#ifdef SALINITY
!
! Swflx    E-P flux [cm/day]
!
        if (wrt(indxSwflx)) then
          lvar=lenstr(vname(1,indxSwflx))
          ierr=nf_inq_varid (ncid,vname(1,indxSwflx)(1:lvar),
     &                                                   vidSwflx)
          if (ierr .ne. nf_noerr) then
            write(stdout,1) vname(1,indxSwflx)(1:lvar), ncname(1:lstr)
            goto 99                                       !--> ERROR
          endif
        endif
# ifdef BWFLUX
!
! Bwflx    Freshwater at bottom flux [cm/day]
!
        if (wrt(indxBwflx)) then
          lvar=lenstr(vname(1,indxBwflx))
          ierr=nf_inq_varid (ncid,vname(1,indxBwflx)(1:lvar),
     &                                                   vidBwflx)
          if (ierr .ne. nf_noerr) then
          write(stdout,1) vname(1,indxBwflx)(1:lvar), ncname(1:lstr)
            goto 99                                       !--> ERROR
          endif
        endif
# endif
#endif
!
! Shflx     Surface shortwave radiation [W/m2]
!
# ifdef TEMPERATURE
        if (wrt(indxShflx_rsw)) then
          lvar=lenstr(vname(1,indxShflx_rsw))
          ierr=nf_inq_varid (ncid,vname(1,indxShflx_rsw)(1:lvar),
     &                                                vidShflx_rsw)
          if (ierr .ne. nf_noerr) then
            write(stdout,1) vname(1,indxShflx_rsw)(1:lvar), ncname(1:lstr)
            goto 99                                       !--> ERROR
          endif
        endif
#endif
# if (defined BULK_FLUX || defined OA_COUPLING) && defined TEMPERATURE
!
        if (wrt(indxShflx_rlw)) then
          lvar=lenstr(vname(1,indxShflx_rlw))
          ierr=nf_inq_varid (ncid,vname(1,indxShflx_rlw)(1:lvar),
     &                                               vidShflx_rlw)
          if (ierr .ne. nf_noerr) then
            write(stdout,1) vname(1,indxShflx_rlw)(1:lvar), ncname(1:lstr)
            goto 99                                       !--> ERROR
          endif
        endif
!
        if (wrt(indxShflx_lat)) then
          lvar=lenstr(vname(1,indxShflx_lat))
          ierr=nf_inq_varid (ncid,vname(1,indxShflx_lat)(1:lvar),
     &                                               vidShflx_lat)
          if (ierr .ne. nf_noerr) then
            write(stdout,1) vname(1,indxShflx_lat)(1:lvar), ncname(1:lstr)
            goto 99                                       !--> ERROR
          endif
        endif
!
        if (wrt(indxShflx_sen)) then
          lvar=lenstr(vname(1,indxShflx_sen))
          ierr=nf_inq_varid (ncid,vname(1,indxShflx_sen)(1:lvar),
     &                                               vidShflx_sen)
          if (ierr .ne. nf_noerr) then
            write(stdout,1) vname(1,indxShflx_sen)(1:lvar), ncname(1:lstr)
            goto 99                                       !--> ERROR
          endif
        endif
!
# endif
!
# if defined BIOLOGY & !defined PISCES
!
! Depth of the euphotic layer.
!
        if (wrt(indxHel)) then
          lvar=lenstr(vname(1,indxHel))
          ierr=nf_inq_varid (ncid,vname(1,indxHel)(1:lvar), vidHel)
          if (ierr .ne. nf_noerr) then
            write(stdout,1) vname(1,indxHel)(1:lvar), ncname(1:lstr)
            goto 99                                       !--> ERROR
          endif
        endif
!
! Chlorophyll/Carbon ratio.
!
#  ifdef BIO_NChlPZD
        if (wrt(indxChC)) then
          lvar=lenstr(vname(1,indxChC))
          ierr=nf_inq_varid (ncid,vname(1,indxChC)(1:lvar), vidChC)
          if (ierr .ne. nf_noerr) then
            write(stdout,1) vname(1,indxChC)(1:lvar), ncname(1:lstr)
            goto 99                                       !--> ERROR
          endif
        endif
#   ifdef OXYGEN
!
! Wind speed at 10 m
        if (wrt(indxU10)) then
          lvar=lenstr(vname(1,indxU10))
          ierr=nf_inq_varid (ncid,vname(1,indxU10)(1:lvar), vidU10)
          if (ierr .ne. nf_noerr) then
            write(stdout,1) vname(1,indxU10)(1:lvar), ncname(1:lstr)
            goto 99                                       !--> ERROR
          endif
        endif
!
! Gas exchange coefficient of O2
        if (wrt(indxKvO2)) then
          lvar=lenstr(vname(1,indxKvO2))
          ierr=nf_inq_varid (ncid,vname(1,indxKvO2)(1:lvar), vidKvO2)
          if (ierr .ne. nf_noerr) then
            write(stdout,1) vname(1,indxKvO2)(1:lvar), ncname(1:lstr)
            goto 99                                       !--> ERROR
          endif
        endif
!
! Saturation concentration of O2
        if (wrt(indxO2sat)) then
          lvar=lenstr(vname(1,indxO2sat))
          ierr=nf_inq_varid (ncid,vname(1,indxO2sat)(1:lvar), vidO2sat)
          if (ierr .ne. nf_noerr) then
            write(stdout,1) vname(1,indxO2sat)(1:lvar), ncname(1:lstr)
            goto 99                                       !--> ERROR
          endif
        endif
#   endif /* OXYGEN */
#  endif
# endif
!
!-----------------------------------------------------------------------
!  USGS sediment model output
!-----------------------------------------------------------------------
!
# ifdef SEDIMENT
!
! Sediment bed layer thickness, porosity, volume fraction of
! size class ised in sediment layer, bed fluxes (2+5*NST r2dgrd variables)
!
        do itrc=1,3+  NST
#  ifdef SUSPLOAD
     &             +2*NST
#  endif
#  ifdef BEDLOAD
     &             +2*NST
#  endif
#  if defined MIXED_BED || defined COHESIVE_BED
     &             +1
#  endif
          indxWrk=indxSed+itrc-2
          if (wrt(indxWrk)) then
            lvar=lenstr(vname(1,indxWrk))
            ierr=nf_inq_varid (ncid,vname(1,indxWrk)(1:lvar),
     &                                                vidSed(itrc))
            if (ierr .ne. nf_noerr) then
              write(stdout,1) vname(1,indxWrk)(1:lvar), ncname(1:lstr)
              goto 99                                       !--> ERROR
            endif
          endif
        enddo
# endif /* SEDIMENT */
# ifdef BBL
!
! Abed       Wind-induced, bed wave excursion amplitude (m)
! Hripple    Bed ripple height (m)
! Lripple    Bed ripple length (m)
! Zbnot      Physical hydraulic bottom roughness (grain roughness) (m)
! Zbapp      Effective apparent hydraulic bottom roughness
! bostrw     Wave-induced kinematic bottom stress (N/m2)
!
        do itrc=1,6
          indxWrk=indxBBL+itrc-1
          if (wrt(indxWrk)) then
            lvar=lenstr(vname(1,indxWrk))
            ierr=nf_inq_varid (ncid,vname(1,indxWrk)(1:lvar),
     &                                                vidBBL(itrc))
            if (ierr .ne. nf_noerr) then
              write(stdout,1) vname(1,indxWrk)(1:lvar), ncname(1:lstr)
              goto 99                                       !--> ERROR
            endif
          endif
        enddo
# endif /* BBL */
# if defined SST_SKIN && defined TEMPERATURE
!
! sst_skin     surface skin temperature (degC)
!
       if (wrt(indxT)) then
         lvar=lenstr(vname(1,indxSST_skin))
         ierr=nf_inq_varid (ncid,vname(1,indxSST_skin)(1:lvar),
     &                                               vidSST_skin)
         if (ierr .ne. nf_noerr) then
           write(stdout,1) vname(1,indxSST_skin)(1:lvar), ncname(1:lstr)
           goto 99                                       !--> ERROR
         endif
       endif
# endif /* SST_SKIN */
!
# ifdef MUSTANG
      do itrc=1,3
        indx=indxT+ntrc_salt+ntrc_substot+itrc
        lvar=lenstr(vname(1,indx))
        ierr=nf_inq_varid (ncid, vname(1,indx)(1:lvar),vidM(itrc)) ! 2D: ksmax,Eptot,Tenfon
      enddo
      do itrc=1,3
        indx=indxT+ntrc_salt+ntrc_substot+itrc+3                 ! 3D: dzs,tempsed,salsed
        lvar=lenstr(vname(1,indx))
        ierr=nf_inq_varid (ncid, vname(1,indx)(1:lvar),vidM(itrc+3))
      enddo
      do itrc=1,ntrc_subs
        indx=indxT+ntrc_salt+ntrc_substot+itrc+6           ! 3D: traceurSed
        lvar=lenstr(vname(1,indx))
        ierr=nf_inq_varid (ncid, vname(1,indx)(1:lvar),vidM(itrc+6))
      enddo
#  ifdef key_MUSTANG_specif_outputs
      ik=0
      do itrc=1,ntrc_subs
        do itrv=1,nv_out3Dnv_specif
          ik=ik+1
          indx=indx+1
          lvar=lenstr(vname(1,indx))
          ierr=nf_inq_varid (ncid, vname(1,indx)(1:lvar),
     &                             vidM(ik+6+ntrc_subs))   ! 3D: traceurSe
        enddo
      enddo
      do itrv=1,nv_out2D_specif
        ik=ik+1
        indx=indx+1
        lvar=lenstr(vname(1,indx))
        ierr=nf_inq_varid (ncid, vname(1,indx)(1:lvar),
     &                           vidM(ik+6+ntrc_subs))   ! 3D: traceurSe
      enddo
#  endif
# endif /* MUSTANG */
!
#endif /* SOLVE3D */
!
#ifdef WAVE_IO
!
! Wave action density, wavenumber in XI/ETA directions
!
# ifndef WAVE_ROLLER
        do iwkb=1,7
# else
        do iwkb=1,9
# endif
          if (wrt(indxHRM+iwkb-1)) then
            lvar=lenstr(vname(1,indxHRM+iwkb-1))
            ierr=nf_inq_varid (ncid, vname(1,indxHRM+iwkb-1)(1:lvar),
     &                                                  vidWAVE(iwkb))
            if (ierr .ne. nf_noerr) then
              write(stdout,1) vname(1,indxHRM+iwkb-1)(1:lvar),
     &                                                 ncname(1:lstr)
              goto 99                                     !--> ERROR
            endif
          endif
        enddo
#endif /* WAVE_IO */

#ifdef MRL_WCI
!
! sup: quasi-static sea-level response (wave set-up/down)
!
        if (wrt(indxSUP)) then
          lvar=lenstr(vname(1,indxSUP))
          ierr=nf_inq_varid (ncid, vname(1,indxSUP)(1:lvar), vidSUP)
          if (ierr .ne. nf_noerr) then
            write(stdout,1) vname(1,indxSUP)(1:lvar), ncname(1:lstr)
            goto 99                                       !--> ERROR
          endif
        endif
!
! ust2d & vst2d: 2D, depth-averaged Stokes drift velocities.
!
        if (wrt(indxUST2D)) then
          lvar=lenstr(vname(1,indxUST2D))
          ierr=nf_inq_varid (ncid, vname(1,indxUST2D)(1:lvar), vidUST2D)
          if (ierr .ne. nf_noerr) then
            write(stdout,1) vname(1,indxUST2D)(1:lvar), ncname(1:lstr)
            goto 99                                       !--> ERROR
          endif
        endif
        if (wrt(indxVST2D)) then
          lvar=lenstr(vname(1,indxVST2D))
          ierr=nf_inq_varid (ncid, vname(1,indxVST2D)(1:lvar), vidVST2D)
          if (ierr .ne. nf_noerr) then
            write(stdout,1) vname(1,indxVST2D)(1:lvar), ncname(1:lstr)
            goto 99                                       !--> ERROR
          endif
        endif
# ifdef SOLVE3D
!
! ust3d & vst3d: 3D Stokes drift velocities.
!
        if (wrt(indxUST)) then
          lvar=lenstr(vname(1,indxUST))
          ierr=nf_inq_varid (ncid, vname(1,indxUST)(1:lvar), vidUST)
          if (ierr .ne. nf_noerr) then
            write(stdout,1) vname(1,indxUST)(1:lvar), ncname(1:lstr)
            goto 99                                       !--> ERROR
          endif
        endif

        if (wrt(indxVST)) then
          lvar=lenstr(vname(1,indxVST))
          ierr=nf_inq_varid (ncid, vname(1,indxVST)(1:lvar), vidVST)
          if (ierr .ne. nf_noerr) then
            write(stdout,1) vname(1,indxVST)(1:lvar), ncname(1:lstr)
            goto 99                                       !--> ERROR
          endif
        endif
!
! wst: vertical Stokes drift velocity (m/s) at rho-point
!
        if (wrt(indxWST)) then
          lvar=lenstr(vname(1,indxWST))
          ierr=nf_inq_varid (ncid, vname(1,indxWST)(1:lvar), vidWST)
          if (ierr .ne. nf_noerr) then
            write(stdout,1) vname(1,indxWST)(1:lvar), ncname(1:lstr)
            goto 99                                       !--> ERROR
          endif
        endif
!
! Akb: Vertical eddy viscosity coefficient due to depth-induced wave breaking.
!
        if (wrt(indxAkb)) then
          lvar=lenstr(vname(1,indxAkb))
          ierr=nf_inq_varid (ncid, vname(1,indxAkb)(1:lvar), vidAkb)
          if (ierr .ne. nf_noerr) then
            write(stdout,1) vname(1,indxAkb)(1:lvar), ncname(1:lstr)
            goto 99                                       !--> ERROR
          endif
        endif
!
! Akw: Vertical eddy diffusivity coefficient due to primary waves.
!
        if (wrt(indxAkw)) then
          lvar=lenstr(vname(1,indxAkw))
          ierr=nf_inq_varid (ncid, vname(1,indxAkw)(1:lvar), vidAkw)
          if (ierr .ne. nf_noerr) then
            write(stdout,1) vname(1,indxAkw)(1:lvar), ncname(1:lstr)
            goto 99                                       !--> ERROR
          endif
        endif
!
! kvf: vertical vortex force term (u^St du/dz)
!
        if (wrt(indxKVF)) then
          lvar=lenstr(vname(1,indxKVF))
          ierr=nf_inq_varid (ncid, vname(1,indxKVF)(1:lvar), vidKVF)
          if (ierr .ne. nf_noerr) then
            write(stdout,1) vname(1,indxKVF)(1:lvar), ncname(1:lstr)
            goto 99                                       !--> ERROR
          endif
        endif
!
! calP: surface pressure correction term appeared in prsgrd term
!
        if (wrt(indxCALP)) then
          lvar=lenstr(vname(1,indxCALP))
          ierr=nf_inq_varid (ncid, vname(1,indxCALP)(1:lvar), vidCALP)
          if (ierr .ne. nf_noerr) then
            write(stdout,1) vname(1,indxCALP)(1:lvar), ncname(1:lstr)
            goto 99                                       !--> ERROR
          endif
        endif
!
! Kapsrf: surface Bernoulli head term appeared in prsgrd term
!
        if (wrt(indxKAPS)) then
          lvar=lenstr(vname(1,indxKAPS))
          ierr=nf_inq_varid (ncid, vname(1,indxKAPS)(1:lvar), vidKAPS)
          if (ierr .ne. nf_noerr) then
            write(stdout,1) vname(1,indxKAPS)(1:lvar), ncname(1:lstr)
            goto 99                                       !--> ERROR
          endif
        endif
# endif  /* SOLVE3D */
#endif   /* MRL_WCI */

      MPI_master_only write(*,'(6x,2A,i4,1x,A,i4)')
     &                     'DEF_HIS/AVG -- Opened ',
     &                     'existing file  from record =', rec
     &                      MYID

#if defined MPI & !defined PARALLEL_FILES & !defined NC4PAR
      else
        ierr=nf_open (ncname(1:lstr), nf_write, ncid)
        if (ierr .ne. nf_noerr) then
          MPI_master_only write(stdout,'(/1x,4A,2x,A,I4/)')
     &                'DEF_HIS/AVG ERROR: ',
     &                'Cannot open file ''', ncname(1:lstr), '''.'
     &                 MYID
          goto 99                                         !--> ERROR
        endif
#endif
      endif             !<--  create_new_file

      ierr=nf_set_fill (ncid, nf_nofill, lvar)
      if (ierr .ne. nf_noerr) then
        write(*,'(6x,2A,i4,1x,A,i4)') 'DEF_HIS/AVG ERROR: Cannot ',
     &    'switch to ''nf_nofill'' more; netCDF error code =', ierr
      endif
   1  format(/1x,'DEF_HIS/AVG ERROR: Cannot find variable ''',
     &                   A, ''' in netCDF file ''', A, '''.'/)
#if (defined PUT_GRID_INTO_HISTORY && !defined AVRH)\
 || (defined PUT_GRID_INTO_AVERAGES && defined AVRH)
!
! Write grid variables.
! ----- ---- ----------
!
        if (total_rec.le.1) call wrt_grid (ncid, ncname, lstr)
#endif

  99  return
      end

#undef ncname
#undef rec_per_file
#undef wrt
#undef vidTime
#undef vidTime2
#undef vidTstep
#undef vidZ
#undef vidUb
#undef vidVb
#undef vidU
#undef vidV
#undef vidT
#undef vidR
#undef vidbvf
#undef vidO
#undef vidW
#undef vidW
#undef vidVisc
#undef vidDiff
#undef vidAkv
#undef vidAkt
#undef vidAks
#undef vidHbl
#undef vidHbbl
#undef vidTke
#undef vidGls
#undef vidLsc
#undef vidHel
#undef vidChC
#undef vidU10
#undef vidKvO2
#undef vidO2sat
#undef vidAOU
#undef vidwind10
#undef vidSed
#undef vidBBL
#undef vidBostr
#undef vidBustr
#undef vidBvstr
#undef vidWstr
#undef vidUWstr
#undef vidVWstr
#undef vidShflx
#undef vidSwflx
#undef vidShflx_rsw
#if defined BULK_FLUX || defined OA_COUPLING
# undef vidShflx_rlw
# undef vidShflx_lat
# undef vidShflx_sen
#endif
#if defined BHFLUX && defined TEMPERATURE
# undef vidBhflx
#endif
#if defined BWFLUX & defined SALINITY
#  undef vidBwflx
#endif
#undef vidSST_skin
#undef vidWAVE
#undef vidSUP
#undef vidUST2D
#undef vidVST2D
#undef vidUST
#undef vidVST
#undef vidWST
#undef vidAkb
#undef vidAkw
#undef vidKVF
#undef vidCALP
#undef vidKAPS
#undef vidHm
#undef vidM

#ifdef AVERAGES
# ifndef AVRH
#  define AVRH
#  include "def_his.F"
# endif
#endif
