! read_inp.F 1615 2014-12-17 13:27:07Z rblod $
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
#include "cppdefs.h"
                                        ! Read/report model input
      subroutine read_inp (ierr)        ! parameters  from  startup 
                                        ! script file using keywords
                                        ! to recognize variables.
!     implicit none

#include "param.h"
#include "scalars.h"
#include "ncscrum.h"
#ifdef SEDIMENT                                        
# include "sediment.h"
#endif
#ifdef FLOATS
# include "ncscrum_floats.h"
#endif
#ifdef STATIONS
# include "nc_sta.h"
#endif
#if defined PSOURCE || defined PSOURCE_NCFILE
# include "sources.h"
#endif
#ifdef WKB_WWAVE
# include "wkb_wwave.h"
#endif
#ifdef ONLINE
# include "online.h"
#endif
#ifdef MPI
      include 'mpif.h'
#endif
#ifdef NBQ
# include "nbq.h"
#endif
#include "mpi_cpl.h"
      integer kwsize, testunit, input
      parameter (kwsize=32, testunit=40, input=15)
      character end_signal*3, keyword*32, fname*180
      parameter (end_signal='end')
      integer ierr, iargc, is,ie, kwlen, lstr, lenstr
#ifdef SOLVE3D
     &                                       , itrc
#endif
      logical dumboolean
#ifdef USE_CALENDAR
      real  dt_his,dt_avg,dt_rst
#endif
!
! Use pre-set default startup filename for known applications,
! or get it as an argument from command line via iargc-getarg

! (override default). NOTE: The usage of the executable should
! be either 
!           croco
!        or
!           croco startup_file_name
!
! WITHOUT the UNIX redirection (<): croco<startup_file like it
! used to be.
!
#if defined BASIN
      fname='TEST_CASES/croco.in.Basin'
#elif defined CANYON
      fname='TEST_CASES/croco.in.Canyon'
#elif defined EQUATOR
      fname='TEST_CASES/croco.in.Equator'
#elif defined GRAV_ADJ
# ifdef NBQ
      fname='TEST_CASES/croco.in.Grav_adj_nbq'
# else
      fname='TEST_CASES/croco.in.Grav_adj'
# endif
#elif defined ISOLITON
      fname='TEST_CASES/croco.in.Isoliton'
#elif defined INNERSHELF
      fname='TEST_CASES/croco.in.Innershelf'
#elif defined INTERNAL
      fname='TEST_CASES/croco.in.Internal'
#elif defined OVERFLOW
      fname='TEST_CASES/croco.in.Overflow'
#elif defined RIVER
      fname='TEST_CASES/croco.in.River'
#elif defined SEAMOUNT
      fname='TEST_CASES/croco.in.Seamount'
#elif defined SHELFRONT
      fname='TEST_CASES/croco.in.Shelfront'
#elif defined SOLITON
      fname='TEST_CASES/croco.in.Soliton'
#elif defined UPWELLING
      fname='TEST_CASES/croco.in.Upwelling'
#elif defined VORTEX
      fname='TEST_CASES/croco.in.Vortex'
#elif defined JET
      fname='TEST_CASES/croco.in.Jet'
#elif defined SHOREFACE
      fname='TEST_CASES/croco.in.ShoreFace'
#elif defined RIP
      fname='TEST_CASES/croco.in.Rip'
#elif defined SANDBAR
      fname='TEST_CASES/croco.in.Sandbar'
#elif defined THACKER
      fname='TEST_CASES/croco.in.Thacker'
#elif defined SINGLE_COLUMN && defined KATO_PHILIPS
      fname='TEST_CASES/croco.in.KATO_PHILLIPS'
#elif defined SINGLE_COLUMN && defined WILLIS_DEARDORFF
      fname='TEST_CASES/croco.in.WILLIS_DEARDORFF'
#elif defined SINGLE_COLUMN && defined DIURNAL_CYCLE
      fname='TEST_CASES/croco.in.DIURNALCYCLE'
#elif defined SINGLE_COLUMN && defined FORCED_EKBBL
      fname='TEST_CASES/croco.in.FORCED_EKBBL'
#elif defined SINGLE_COLUMN && defined FORCED_DBLEEK 
      fname='TEST_CASES/croco.in.FORCED_DBLEEK'
#elif defined SINGLE_COLUMN && defined FORCED_NONROTBBL
      fname='TEST_CASES/croco.in.FORCED_NONROTBBL'
#elif defined SINGLE_COLUMN && defined FORCED_OSCNONROTBBL
      fname='TEST_CASES/croco.in.FORCED_OSCNONROTBBL'
#elif defined TANK
      fname='TEST_CASES/croco.in.Tank'
#elif defined MOVING_BATHY
      fname='TEST_CASES/croco.in.MovBat'
#elif defined ACOUSTIC
      fname='TEST_CASES/croco.in.Acoustic' 
#elif defined SWASH
      fname='TEST_CASES/croco.in.Swash'
#elif defined KH_INST      
      fname='TEST_CASES/croco.in.KH_INST'
#elif defined IGW
# ifdef NBQ
      fname='TEST_CASES/croco.in.IGW_nbq'
# else
      fname='TEST_CASES/croco.in.IGW'
# endif
#elif defined TS_HADV_TEST
# ifdef SOLID_BODY_ROT 
      fname='TEST_CASES/croco.in.TS_adv_body'
# elif defined DIAGONAL_ADV
      fname='TEST_CASES/croco.in.TS_adv_diag'
# elif defined SOLID_BODY_PER
      fname='TEST_CASES/croco.in.TS_adv_rep'
# endif
#elif defined DUNE
# ifdef DUNE3D
      fname='TEST_CASES/croco.in.Dune3d'
# elif defined ANA_DUNE
      fname='TEST_CASES/croco.in.Ana_dune'
# else
      fname='TEST_CASES/croco.in.Dune'
# endif
#elif defined SED_TOY
# ifdef SED_TOY_RESUSP
      fname='TEST_CASES/croco.in.Sed_toy_resusp'
# elif defined SED_TOY_CONSOLID
      fname='TEST_CASES/croco.in.Sed_toy_consolid'
# elif defined SED_TOY_FLOC
      fname='TEST_CASES/croco.in.Sed_toy_floc'
# elif defined SED_TOY_ROUSE
      fname='TEST_CASES/croco.in.Sed_toy_rouse'
# endif
#elif defined TIDAL_FLAT
      fname='TEST_CASES/croco.in.Tidal_flat'
#else
      fname='croco.in'
#endif
#ifdef MPI
      if (mynode.eq.0 .and. iargc().GT.0) call getarg(1,fname) 
      call MPI_Bcast(fname,64,MPI_BYTE, 0, MPI_COMM_WORLD,ierr)
#else
      if (iargc().eq.1) call getarg(1,fname)
#endif
!
! if child grid, use an input name: fname.1, .2, .3, ...
!
#ifdef AGRIF
      if (.Not.Agrif_Root()) then
        lstr=lenstr(fname)
#ifdef AGRIF_ADAPTIVE
        fname=fname(1:lstr) / / '.1'  
#else        
        fname=fname(1:lstr) / / '.' / / Agrif_Cfixed()
#endif 
        lstr=lenstr(fname)
      endif
#endif
#ifdef LOGFILE
# ifndef AGRIF
      MPI_master_only stdout=80
# else 
      stdout=Agrif_Get_Unit()     
      if (.Not.Agrif_Root()) then
        if(mynode.eq.0) then
         open (stdout,file='croco.log.' / / Agrif_Cfixed(),
     &   status='REPLACE',form='formatted',err=98)
        else
         stdout=6
        endif
      else          
# endif 
      if(mynode.eq.0) then      
      open (stdout,file='croco.log',
     & status='REPLACE',form='formatted',err=98)   
      else  
      stdout=6
      endif
# ifdef AGRIF
      endif
# endif      
#endif  

!
      wrthis(indxTime)=.false.
#ifdef AVERAGES
      wrtavg(indxTime)=.false.
#endif

#ifdef TRACERS
# ifdef DIAGNOSTICS_TS
      wrtdia3D(NT+1)=.false.
#  ifdef AVERAGES
      wrtdia3D_avg(NT+1)=.false.
#  endif
#  ifdef DIAGNOSTICS_TS_MLD
      wrtdia2D(NT+1)=.false.
#   ifdef AVERAGES
      wrtdia2D_avg(NT+1)=.false.
#   endif
#  endif
# endif
#endif
#ifdef DIAGNOSTICS_UV
      wrtdiaM(3)=.false.
#ifdef AVERAGES
      wrtdiaM_avg(3)=.false.
#endif
#endif
#ifdef DIAGNOSTICS_VRT
      wrtdiags_vrt(3)=.false.
#ifdef AVERAGES
      wrtdiags_vrt_avg(3)=.false.
#endif
#endif
#ifdef DIAGNOSTICS_EK
      wrtdiags_ek(3)=.false.
#ifdef AVERAGES
      wrtdiags_ek_avg(3)=.false.
#endif
#endif
#if defined DIAGNOSTICS_PV && defined TRACERS
      wrtdiags_pv(NT+1)=.false.
#ifdef AVERAGES
      wrtdiags_pv_avg(NT+1)=.false.
#endif
#endif
#if defined DIAGNOSTICS_EDDY && ! defined XIOS
      wrtdiags_eddy(3)=.false.
# ifdef AVERAGES
      wrtdiags_eddy_avg(3)=.false.
# endif
#endif
#if defined OUTPUTS_SURFACE && ! defined XIOS
      wrtsurf(3)=.false.
# ifdef AVERAGES
      wrtsurf_avg(3)=.false.
# endif
#endif
#ifdef DIAGNOSTICS_BIO
      wrtdiabioFlux(NumFluxTerms+1)=.false.
      wrtdiabioVSink(NumVSinkTerms+1)=.false.
      wrtdiabioGasExc(NumGasExcTerms+1)=.false.
#ifdef AVERAGES
      wrtdiabioFlux_avg(NumFluxTerms+1)=.false.
      wrtdiabioVSink_avg(NumVSinkTerms+1)=.false.
      wrtdiabioGasExc_avg(NumGasExcTerms+1)=.false.   
#endif
#endif
!
! Read in keyword: keep trying, until keyword is found.
! ==== == ======== ==== ======= ===== ======= == ======
!
      ierr=0    ! <-- reset error counter
      call setup_kwds (ierr)
      open (input,file=fname,status='old',form='formatted',err=97)
   1  keyword='                                '
      read(input,'(A)',err=1,end=99) keyword
      if (ichar(keyword(1:1)).eq.33) goto 1
      is=1
   2  if (is.eq.kwsize) then
        goto 1
      elseif (keyword(is:is).eq.' ') then
        is=is+1
        goto 2
      endif
      ie=is
   3  if (keyword(ie:ie).eq.':') then
        keyword(ie:ie)=' '
        goto 4           !--> recognized keyword.               
      elseif (keyword(ie:ie).ne.' ' .and. ie.lt.kwsize) then 
        ie=ie+1
        goto 3
      endif
      goto 1
   4  kwlen=ie-is
      if (is.gt.1) keyword(1:kwlen)=keyword(is:is+kwlen-1)
!
! Read input parameters according to the keyword:
! ==== ===== ========== ========= == === ========
!
! Title
!
      if (keyword(1:kwlen).eq.'title') then
        call cancel_kwd (keyword(1:kwlen), ierr)
        read(input,'(A)',err=95) title
        lstr=lenstr(title)
        MPI_master_only write(stdout,'(/1x,A)') title(1:lstr)
!
#ifdef XIOS
!
! origin_date is the origin date of time axis.
! It must the same used in croco_tool for input files preprocessing
! 
      elseif (keyword(1:kwlen).eq.'origin_date') then
          call cancel_kwd (keyword(1:kwlen), ierr)
          read(input,'(A)',err=95) origin_date
          lstr=lenstr(origin_date)
          MPI_master_only write(stdout,'(/2x,A,1x,A/)')
     &    'Origin date of time axis (dd/mm/yy hh:mm:ss) :',
     &    origin_date(1:lstr)
          origin_date_in_sec=tool_datosec(origin_date(1:lstr))
      READ(origin_date(1:2),fmt='(i2)') origin_day
      READ(origin_date(4:5),fmt='(i2)') origin_month
      READ(origin_date(7:10),fmt='(i4)') origin_year
      READ(origin_date(12:13),fmt='(i2)') origin_hour
      READ(origin_date(15:16),fmt='(i2)') origin_minute
      READ(origin_date(18:19),fmt='(i2)') origin_second
#endif
#ifdef START_DATE
!
! Start date
!
      elseif (keyword(1:kwlen).eq.'start_date') then
        call cancel_kwd (keyword(1:kwlen), ierr)
        read(input,'(A)',err=95) start_date
        lstr=lenstr(start_date)
        MPI_master_only write(stdout,'(/2x,A,1x,A/)') 
     &   'Start date:', start_date(1:lstr)
#endif
#ifdef USE_CALENDAR 
!
! Run start date
!
      elseif (keyword(1:kwlen).eq.'run_start_date') then
        call cancel_kwd (keyword(1:kwlen), ierr)
        read(input,'(A)',err=95) run_start_date
        lstr=lenstr(run_end_date)
        MPI_master_only write(stdout,'(/2x,A,1x,A/)')
     &   'Run start date:', run_start_date(1:lstr)
! Run END date
!
      elseif (keyword(1:kwlen).eq.'run_end_date') then
        call cancel_kwd (keyword(1:kwlen), ierr)
        read(input,'(A)',err=95) run_end_date
        lstr=lenstr(run_end_date)
        MPI_master_only write(stdout,'(/2x,A,1x,A/)')
     &   'Run end date:', run_end_date(1:lstr)
! output time steps
      elseif (keyword(1:kwlen).eq.'output_time_steps') then
        call cancel_kwd (keyword(1:kwlen), ierr)
        read(input,*,err=95) dt_his,dt_avg,dt_rst
        MPI_master_only  write(stdout,
     &  '(2x,A,f6.2,A,1x,A,f6.2,A,1x,A,f6.2,A)')
     &  'DT HIS =',dt_his,'H','DT AVG =',dt_avg,'H',
     &  'DT RST=',dt_rst,'H'
        MPI_master_only write(stdout,*) 
#endif
!     
! Time-stepping parameters
!
      elseif (keyword(1:kwlen).eq.'time_stepping') then
        call cancel_kwd (keyword(1:kwlen), ierr)
        read(input,*,err=95) ntimes,dt,ndtfast, ninfo

        MPI_master_only write(stdout,
     & '(I10,2x,A,1x,A /F10.2,2x,A,2(/I10,2x,A,1x,A)/F10.4,2x,A)')
     &    ntimes,  'ntimes   Total number of timesteps for',
     &                                            '3D equations.',
     &    dt,      'dt       Timestep [sec] for 3D equations',
     &    ndtfast, 'ndtfast  Number of 2D timesteps within each',
     &                                                 '3D step.',
     &    ninfo,   'ninfo    Number of timesteps between',
     &                                     'runtime diagnostics.'
        dtfast=dt/float(ndtfast)     ! set barotropic time step.

        if (NWEIGHT.lt.(2*ndtfast-1)) then
          write(stdout,'(a,i3)')
     &    ' Error - Number of 2D timesteps (2*ndtfast-1): ',
     &    2*ndtfast-1
          write(stdout,'(a,i3)')
     &    '           exceeds barotopic weight dimension: ',NWEIGHT
          goto 95
        endif

#ifdef NBQ
!     
! NBQ Time-stepping parameters
!
      elseif (keyword(1:kwlen).eq.'time_stepping_nbq') then
        call cancel_kwd (keyword(1:kwlen), ierr)
        read(input,*,err=95) ndtnbq, csound_nbq, visc2_nbq
        MPI_master_only write(stdout,
     &    '(I10,2x,A,1x,A,/F10.2,2x,A,/1pe10.3,2x,A,1x,A/)')
     &   ndtnbq,    'ndtnbq      Number of NBQ timesteps within each',
     &                                                 '2D step.',
     &   csound_nbq,'csound_nbq  Sound wave celerity.',
     &   visc2_nbq, 'visc2_nbq   Second viscosity coefficient for',
     &                                      'compressible fluids.'
        dtnbq=dtfast
        ndtnbq=1
#endif

#ifdef SOLVE3D
!
! Vertical S-coordinates parameters.
!
      elseif (keyword(1:kwlen).eq.'S-coord') then
        call cancel_kwd (keyword(1:kwlen), ierr)
        read(input,*,err=95) theta_s, theta_b, Tcline
        MPI_master_only write(stdout,
     &                        '(3(1pe10.3,2x,A,1x,A/),32x,A)')
     &    theta_s, 'theta_s  S-coordinate surface control',
     &                                               'parameter.',
     &    theta_b, 'theta_b  S-coordinate bottom control',
     &                                               'parameter.',
     &    Tcline,  'Tcline   S-coordinate surface/bottom layer',
     &  'width used in', 'vertical coordinate stretching, meters.'
#endif
!
! Initial conditions file name. Check its availability (in the case
! of analytical initial conditions and nrrec=0 initial conditions are
! created internally and no file is needed).
!
      elseif (keyword(1:kwlen).eq.'initial') then
        call cancel_kwd (keyword(1:kwlen), ierr)
        read(input,*,err=95) nrrec
#ifdef ANA_INITIAL
        if (nrrec.gt.0) then 
#endif
          read(input,'(A)',err=95) fname
          lstr=lenstr(fname)
#if defined MPI && defined PARALLEL_FILES
          call insert_node (fname, lstr, mynode2, NNODES2, ierr)
#endif
          open (testunit, file=fname(1:lstr), status='old', err=97)
          close(testunit)
          ininame=fname(1:lstr)
          MPI_master_only write(stdout,'(1x,A,2x,A,4x,A,I3)')
     &     'Initial State File:', ininame(1:lstr), 'Record:',nrrec
#ifdef ANA_INITIAL
        endif
#endif
#ifndef ANA_GRID
!
! Grid file name. Check its availability.
!
      elseif (keyword(1:kwlen).eq.'grid') then
        call cancel_kwd (keyword(1:kwlen), ierr)
        read(input,'(A)',err=95) fname 
        lstr=lenstr(fname)
# if defined MPI && defined PARALLEL_FILES
        call insert_node (fname, lstr, mynode2, NNODES2, ierr)
# endif
        open(testunit,file=fname(1:lstr), status='old', err=97)
        close(testunit)
        grdname=fname(1:lstr)
        MPI_master_only write(stdout,'(10x,A,2x,A)')
     &                   'Grid File:', grdname(1:lstr)
#endif


! forcing file for tides
#if !defined NO_FRCFILE  && \
        ( defined TIDES \
          || (defined MRL_WCI     && !defined ANA_WWAVE && \
                                     !defined WKB_WWAVE && \
                                     !defined OW_COUPLING) \
          || (defined SFLX_CORR   && !defined ANA_SSS   && \
                                   !defined BULK_FLUX   && \
                                   !defined OA_COUPLING) \
          || (defined QCORRECTION && !defined ANA_SST   && \
                                   !defined OA_COUPLING) \
          || (defined BBL         && !defined ANA_BSEDIM  && \
                                   !defined SEDIMENT) \
          || (!defined ANA_STFLUX && !defined BULK_FLUX && \
                                   !defined OA_COUPLING && \
                                   defined SOLVE3D) \
          || (defined SALINITY    && !defined ANA_SSFLUX && \
                                   !defined BULK_FLUX && \
                                   !defined OA_COUPLING && \
                                    defined SOLVE3D) \
          || (!defined ANA_SRFLUX && !defined BULK_FLUX && \
                                   !defined OA_COUPLING && \
                                   defined SOLVE3D) \
          || (!defined ANA_SMFLUX && !defined BULK_FLUX && \
                                   !defined OA_COUPLING) \
          || ( defined BULK_FLUX    && !defined SALINITY ) \
          || ( defined WAVE_OFFLINE && !defined MUSTANG ) )
!
!
! Forcing file name. Check its availability.
!
      elseif (keyword(1:kwlen).eq.'forcing') then
        call cancel_kwd (keyword(1:kwlen), ierr)
        read(input,'(A)',err=95) fname
        lstr=lenstr(fname)
# if defined MPI && defined PARALLEL_FILES
        call insert_node (fname, lstr, mynode2, NNODES2, ierr)
# endif
        open (testunit, file=fname(1:lstr), status='old', err=97)
        close(testunit)
        frcname=fname(1:lstr)
        MPI_master_only write(stdout,'(2x,A,2x,A)')
     &             'Forcing Data File:', frcname(1:lstr)
#endif  /* !defined NO_FRCFILE */

!

# if defined WAVE_OFFLINE && defined MUSTANG
! Forcing file name. Check its availability.
!
      elseif (keyword(1:kwlen).eq.'wave_offline') then
        call cancel_kwd (keyword(1:kwlen), ierr)
        read(input,'(A)',err=95) fname
        lstr=lenstr(fname)
# if defined MPI && defined PARALLEL_FILES
        call insert_node (fname, lstr, mynode, NNODES, ierr)
# endif
        open (testunit, file=fname(1:lstr), status='old', err=97)
        close(testunit)
        wave_file=fname(1:lstr)
        MPI_master_only write(stdout,'(2x,A,2x,A)')
     &             'Wave Data File:', wave_file(1:lstr)
#endif  /* !defined WAVE_OFFLINE */

!
! Biology forcing: iron dust deposition. 
!
#if defined BIOLOGY && defined PISCES
      elseif (keyword(1:kwlen).eq.'biology') then
        call cancel_kwd (keyword(1:kwlen), ierr)
        read(input,'(A)',err=95) fname
        lstr=lenstr(fname)
# if defined MPI && defined PARALLEL_FILES
        call insert_node (fname, lstr, mynode2, NNODES2, ierr)
# endif
c        open (testunit, file=fname(1:lstr), status='old', err=97)
c        close(testunit)
        bioname=fname(1:lstr)
        MPI_master_only write(stdout,'(2x,A,2x,A)')
     &             'Biology Forcing Data File:', bioname(1:lstr)
#endif
!
! Bulk file name. Check its availability.
!
#if defined BULK_FLUX
# ifndef ONLINE /* ONLINE FORCING */
        elseif (keyword(1:kwlen).eq.'bulk_forcing') then
          call cancel_kwd (keyword(1:kwlen), ierr)
          read(input,'(A)',err=95) fname
          lstr=lenstr(fname)
#  if defined MPI && defined PARALLEL_FILES
          call insert_node (fname, lstr, mynode2, NNODES2, ierr)
#  endif
          open (testunit, file=fname(1:lstr), status='old', err=97)
          close(testunit)
          bulkname=fname(1:lstr)
          MPI_master_only write(stdout,'(2x,A,2x,A)')
     &               '   Bulk Data File:', bulkname(1:lstr)
# endif  /* ONLINE FORCING */
#endif
#if (defined TCLIMATOLOGY  && !defined ANA_TCLIMA) || \
      (defined ZCLIMATOLOGY  && !defined ANA_SSH) || \
      (defined M2CLIMATOLOGY && !defined ANA_M2CLIMA) || \
      (defined M3CLIMATOLOGY && !defined ANA_M3CLIMA)
!
! Climatology file name. Check availability.
!
      elseif (keyword(1:kwlen).eq.'climatology') then
        call cancel_kwd (keyword(1:kwlen), ierr)
# ifdef AGRIF
       if (Agrif_Root()) then
# endif
        read(input,'(A)',err=95) fname 
        lstr=lenstr(fname)
# if defined MPI && defined PARALLEL_FILES
        call insert_node (fname, lstr, mynode2, NNODES2, ierr)
# endif
        open (testunit, file=fname(1:lstr), status='old', err=97)
        close(testunit)
        clmname=fname(1:lstr)
        MPI_master_only write(stdout,'(3x,A,2x,A)')
     &            'Climatology File:', clmname(1:lstr)
# ifdef AGRIF
       endif
# endif
#endif
#if !defined ANA_BRY && defined FRC_BRY
!
! Boundary file name. Check availability.
!
        elseif (keyword(1:kwlen).eq.'boundary') then
          call cancel_kwd (keyword(1:kwlen), ierr)
#  ifdef AGRIF
       if (Agrif_Root()) then
#  endif
          read(input,'(A)',err=95) fname
          lstr=lenstr(fname)
#  if defined MPI && defined PARALLEL_FILES
          call insert_node (fname, lstr, mynode2, NNODES2, ierr)
#  endif
          open (testunit, file=fname(1:lstr), status='old', err=97)
          close(testunit)
          bry_file=fname(1:lstr)
          MPI_master_only write(stdout,'(6x,A,2x,A)')
     &          'Boundary File:', bry_file(1:lstr)
#  ifdef AGRIF
       endif
#  endif
#endif
#if defined WKB_WWAVE && !defined ANA_BRY_WKB 
!
! WKB boundary file name. Check availability.
!
        elseif (keyword(1:kwlen).eq.'wkb_boundary') then
          call cancel_kwd (keyword(1:kwlen), ierr)
#  ifdef AGRIF
       if (Agrif_Root()) then
#  endif
          read(input,'(A)',err=95) fname
          lstr=lenstr(fname)
#  if defined MPI && defined PARALLEL_FILES
          call insert_node (fname, lstr, mynode2, NNODES2, ierr)
#  endif
          open (testunit, file=fname(1:lstr), status='old', err=97)
          close(testunit)
          brywkb_file=fname(1:lstr)
          MPI_master_only write(stdout,'(6x,A,2x,A)')
     &          'WKB Boundary File:', brywkb_file(1:lstr)
#  ifdef AGRIF
       endif
#  endif
#endif
!
! Restart file name.
!
      elseif (keyword(1:kwlen).eq.'restart') then
        call cancel_kwd (keyword(1:kwlen), ierr)
        read(input,*,err=95) nrst, nrpfrst
        read(input,'(A)',err=95)  fname 
        lstr=lenstr(fname)
# if defined MPI && defined PARALLEL_FILES
        call insert_node (fname, lstr, mynode2, NNODES2, ierr)
# endif
#ifdef USE_CALENDAR
        nrst=ceiling((dt_rst*3600.)/dt)
#endif
        rstname=fname(1:lstr)
        MPI_master_only write(stdout,
     &             '(7x,A,2x,A,4x,A,I6,4x,A,I4)')
     &             'Restart File:', rstname(1:lstr),
     &             'nrst =', nrst, 'rec/file: ', nrpfrst 
!
! History file name.
!
      elseif (keyword(1:kwlen).eq.'history') then
        call cancel_kwd (keyword(1:kwlen), ierr)
        read(input,*,err=95) ldefhis, nwrt, nrpfhis
        read(input,'(A)',err=95) fname
        lstr=lenstr(fname)
# if defined MPI && defined PARALLEL_FILES
        call insert_node (fname, lstr, mynode2, NNODES2, ierr)
# endif
#ifdef USE_CALENDAR
        nwrt=ceiling((dt_his*3600.)/dt)
#endif
        hisname=fname(1:lstr)
        MPI_master_only write(stdout,
     &             '(7x,A,2x,A,2x,A,1x,L1,2x,A,I4,2x,A,I3)')
     &       'History File:', hisname(1:lstr),  'Create new:',
     &       ldefhis, 'nwrt =', nwrt, 'rec/file =', nrpfhis
#ifdef AVERAGES
!
! Averages file name.
!
      elseif (keyword(1:kwlen).eq.'averages') then
        call cancel_kwd (keyword(1:kwlen), ierr)
        read(input,*,err=95) ntsavg, navg, nrpfavg
        read(input,'(A)',err=95) fname
        lstr=lenstr(fname)
# if defined MPI && defined PARALLEL_FILES
        call insert_node (fname, lstr, mynode2, NNODES2, ierr)
# endif
#ifdef USE_CALENDAR
        navg=ceiling((dt_avg*3600.)/dt)
#endif
        avgname=fname(1:lstr)
        MPI_master_only write(stdout,
     &         '(2(I10,2x,A,1x,A/32x,A/),6x,A,2x,A,1x,A,I3)')
     &      ntsavg, 'ntsavg      Starting timestep for the',
     &         'accumulation of output', 'time-averaged data.',
     &      navg,   'navg        Number of timesteps between',
     &     'writing of time-averaged','data into averages file.',
     &     'Averages File:', avgname(1:lstr),
     &     'rec/file =', nrpfavg
#endif

#if defined DIAGNOSTICS_TS
!
! Diagnostics file name.
!
      elseif (keyword(1:kwlen).eq.'diagnostics') then
        call cancel_kwd (keyword(1:kwlen), ierr)
        read(input,*,err=95) ldefdia, nwrtdia, nrpfdia
        if (nwrtdia.eq.0) nwrtdia = nwrt
        read(input,'(A)',err=95) fname
        lstr=lenstr(fname)
# if defined MPI && defined PARALLEL_FILES
        call insert_node (fname, lstr, mynode2, NNODES2, ierr)
# endif
        dianame=fname(1:lstr)
        MPI_master_only write(stdout,
     &    '(9x,A,2x,A/,32x,A,1x,L1,2x,A,I4,2x,A,I3)')
     &    'Tracer Diag File:',dianame(1:lstr),'Create new:',
     &    ldefdia,'nwrt =',nwrtdia,'rec/file =',nrpfdia
!
# ifdef AVERAGES
      elseif (keyword(1:kwlen).eq.'diag_avg') then
        call cancel_kwd (keyword(1:kwlen), ierr)
        read(input,*,err=95) ldefdia_avg, ntsdia_avg, nwrtdia_avg,
     &                                                nrpfdia_avg
        if (nwrtdia_avg.eq.0) nwrtdia_avg = navg
        read(input,'(A)',err=95) fname
        lstr=lenstr(fname)
# if defined MPI && defined PARALLEL_FILES
        call insert_node (fname, lstr, mynode2, NNODES2, ierr)
# endif
        dianame_avg=fname(1:lstr)
        MPI_master_only write(stdout,
     &    '(5x,A,2x,A/,32x,A,1x,L1,2x,A,I4,2x,A,I3,/32x,A,I10)')
     &    'Tracer AVG Diag File:',dianame_avg(1:lstr),'Create new:',
     &    ldefdia_avg,'nwrt =',nwrtdia_avg,'rec/file =',nrpfdia_avg,
     &    'Starting timestep = ',ntsdia_avg
# endif
#endif /* DIAGOSTICS_TS */
#if defined DIAGNOSTICS_UV
!
! Diagnostics Momentum file name.
!
      elseif (keyword(1:kwlen).eq.'diagnosticsM') then
        call cancel_kwd (keyword(1:kwlen), ierr)
        read(input,*,err=95) ldefdiaM, nwrtdiaM, nrpfdiaM
        if (nwrtdiaM.eq.0) nwrtdiaM = nwrt
        read(input,'(A)',err=95) fname
        lstr=lenstr(fname)
# if defined MPI && defined PARALLEL_FILES
        call insert_node (fname, lstr, mynode2, NNODES2, ierr)
# endif
        dianameM=fname(1:lstr)
        MPI_master_only write(stdout,
     &    '(5x,A,2x,A/,32x,A,1x,L1,2x,A,I4,2x,A,I3)')
     &    'Momentum Diag File:', dianameM(1:lstr),  'Create new:',
     &    ldefdiaM, 'nwrt =', nwrtdiaM, 'rec/file =', nrpfdiaM
!
# ifdef AVERAGES
      elseif (keyword(1:kwlen).eq.'diagM_avg') then
        call cancel_kwd (keyword(1:kwlen), ierr)
        read(input,*,err=95) ldefdiaM_avg, ntsdiaM_avg, nwrtdiaM_avg,
     &                                                  nrpfdiaM_avg
        if (nwrtdiaM_avg.eq.0) nwrtdiaM_avg = navg
        read(input,'(A)',err=95) fname
        lstr=lenstr(fname)
# if defined MPI && defined PARALLEL_FILES
        call insert_node (fname, lstr, mynode2, NNODES2, ierr)
# endif
        dianameM_avg=fname(1:lstr)
        MPI_master_only write(stdout,
     &    '(5x,A,2x,A/,32x,A,1x,L1,2x,A,I4,2x,A,I3,/32x,A,I10)')
     &    'Momentum AVG Diag File:',dianameM_avg(1:lstr),'Create new:',
     &    ldefdiaM_avg,'nwrt =',nwrtdiaM_avg,'rec/file =',nrpfdiaM_avg,
     &    'Starting timestep = ',ntsdiaM_avg

# endif
#endif /*DIAGNOSTICS_UV */


#if defined DIAGNOSTICS_VRT
!
! Diagnostics vrt file name.
!
      elseif (keyword(1:kwlen).eq.'diags_vrt') then
        call cancel_kwd (keyword(1:kwlen), ierr)
        read(input,*,err=95) ldefdiags_vrt, nwrtdiags_vrt, nrpfdiags_vrt
        if (nwrtdiags_vrt.eq.0) nwrtdiags_vrt = nwrt
        read(input,'(A)',err=95) fname
        lstr=lenstr(fname)
# if defined MPI && defined PARALLEL_FILES
        call insert_node (fname, lstr, mynode2, NNODES2, ierr)
# endif
        diags_vrtname=fname(1:lstr)
        MPI_master_only write(stdout,
     &    '(5x,A,2x,A/,32x,A,1x,L1,2x,A,I4,2x,A,I3)')
     &    'Vorticity Diag File:', diags_vrtname(1:lstr),  'Create new:',
     &    ldefdiags_vrt, 'nwrt =', nwrtdiags_vrt,
     &     'rec/file =', nrpfdiags_vrt
!
# ifdef AVERAGES
      elseif (keyword(1:kwlen).eq.'diags_vrt_avg') then
        call cancel_kwd (keyword(1:kwlen), ierr)
        read(input,*,err=95) ldefdiags_vrt_avg, ntsdiags_vrt_avg, 
     &                               nwrtdiags_vrt_avg, nrpfdiags_vrt_avg
        if (nwrtdiags_vrt_avg.eq.0) nwrtdiags_vrt_avg = navg
        read(input,'(A)',err=95) fname
        lstr=lenstr(fname)
# if defined MPI && defined PARALLEL_FILES
        call insert_node (fname, lstr, mynode2, NNODES2, ierr)
# endif
        diags_vrtname_avg=fname(1:lstr)
        MPI_master_only write(stdout,
     &    '(5x,A,2x,A/,32x,A,1x,L1,2x,A,I4,2x,A,I3,/32x,A,I10)')
     &    'Vorticity AVG Diag File:',diags_vrtname_avg(1:lstr),
     &    'Create new:', ldefdiags_vrt_avg,'nwrt =',nwrtdiags_vrt_avg,
     &    'rec/file =',nrpfdiags_vrt_avg,
     &    'Starting timestep = ',ntsdiags_vrt_avg

# endif
#endif /*DIAGNOSTICS_VRT */


#if defined DIAGNOSTICS_EK
!
! Diagnostics ek file name.
!
      elseif (keyword(1:kwlen).eq.'diags_ek') then
        call cancel_kwd (keyword(1:kwlen), ierr)
        read(input,*,err=95) ldefdiags_ek, nwrtdiags_ek, nrpfdiags_ek
        if (nwrtdiags_ek.eq.0) nwrtdiags_ek = nwrt
        read(input,'(A)',err=95) fname
        lstr=lenstr(fname)
# if defined MPI && defined PARALLEL_FILES
        call insert_node (fname, lstr, mynode2, NNODES2, ierr)
# endif
        diags_ekname=fname(1:lstr)
        MPI_master_only write(stdout,
     &    '(5x,A,2x,A/,32x,A,1x,L1,2x,A,I4,2x,A,I3)')
     &    'Kinetic Energy Diag File:', diags_ekname(1:lstr),
     &    'Create new:', ldefdiags_ek, 'nwrt =', nwrtdiags_ek,
     &    'rec/file =', nrpfdiags_ek
!
# ifdef AVERAGES
      elseif (keyword(1:kwlen).eq.'diags_ek_avg') then
        call cancel_kwd (keyword(1:kwlen), ierr)
        read(input,*,err=95) ldefdiags_ek_avg, ntsdiags_ek_avg,
     &                      nwrtdiags_ek_avg,  nrpfdiags_ek_avg
        if (nwrtdiags_ek_avg.eq.0) nwrtdiags_ek_avg = navg
        read(input,'(A)',err=95) fname
        lstr=lenstr(fname)
# if defined MPI && defined PARALLEL_FILES
        call insert_node (fname, lstr, mynode2, NNODES2, ierr)
# endif
        diags_ekname_avg=fname(1:lstr)
        MPI_master_only write(stdout,
     &    '(5x,A,2x,A/,32x,A,1x,L1,2x,A,I4,2x,A,I3,/32x,A,I10)')
     &    'Kinetic Energy AVG Diag File:',diags_ekname_avg(1:lstr),
     &    'Create new:', ldefdiags_ek_avg,'nwrt =',nwrtdiags_ek_avg,
     &    'rec/file =',nrpfdiags_ek_avg,
     &    'Starting timestep = ',ntsdiags_ek_avg

# endif
#endif /*DIAGNOSTICS_EK */

#if defined DIAGNOSTICS_PV
!
! Diagnostics pv file name.
!
      elseif (keyword(1:kwlen).eq.'diags_pv') then
        call cancel_kwd (keyword(1:kwlen), ierr)
        read(input,*,err=95) ldefdiags_pv, nwrtdiags_pv, nrpfdiags_pv
        if (nwrtdiags_pv.eq.0) nwrtdiags_pv = nwrt
        read(input,'(A)',err=95) fname
        lstr=lenstr(fname)
# if defined MPI && defined PARALLEL_FILES
        call insert_node (fname, lstr, mynode2, NNODES2, ierr)
# endif
        diags_pvname=fname(1:lstr)
        MPI_master_only write(stdout,
     &    '(5x,A,2x,A/,32x,A,1x,L1,2x,A,I4,2x,A,I3)')
     &    'PV Diag File:', diags_pvname(1:lstr),
     &    'Create new:', ldefdiags_pv, 'nwrt =', nwrtdiags_pv,
     &    'rec/file =', nrpfdiags_pv
!
# ifdef AVERAGES
      elseif (keyword(1:kwlen).eq.'diags_pv_avg') then
        call cancel_kwd (keyword(1:kwlen), ierr)
        read(input,*,err=95) ldefdiags_pv_avg, ntsdiags_pv_avg,
     &                      nwrtdiags_pv_avg,  nrpfdiags_pv_avg
        if (nwrtdiags_pv_avg.eq.0) nwrtdiags_pv_avg = navg
        read(input,'(A)',err=95) fname
        lstr=lenstr(fname)
# if defined MPI && defined PARALLEL_FILES
        call insert_node (fname, lstr, mynode2, NNODES2, ierr)
# endif
        diags_pvname_avg=fname(1:lstr)
        MPI_master_only write(stdout,
     &    '(5x,A,2x,A/,32x,A,1x,L1,2x,A,I4,2x,A,I3,/32x,A,I10)')
     &    'PV AVG Diag File:',diags_pvname_avg(1:lstr),
     &    'Create new:', ldefdiags_pv_avg,'nwrt =',nwrtdiags_pv_avg,
     &    'rec/file =',nrpfdiags_pv_avg,
     &    'Starting timestep = ',ntsdiags_pv_avg

# endif
#endif /*DIAGNOSTICS_PV */

# if defined DIAGNOSTICS_EDDY && ! defined XIOS
!
! Diagnostics eddy file name.
!
      elseif (keyword(1:kwlen).eq.'diags_eddy') then
        call cancel_kwd (keyword(1:kwlen), ierr)
        read(input,*,err=95) ldefdiags_eddy, nwrtdiags_eddy, nrpfdiags_eddy
        if (nwrtdiags_eddy.eq.0) nwrtdiags_eddy = nwrt
        read(input,'(A)',err=95) fname
        lstr=lenstr(fname)
# if defined MPI && defined PARALLEL_FILES
        call insert_node (fname, lstr, mynode2, NNODES2, ierr)
# endif
        diags_eddyname=fname(1:lstr)
        MPI_master_only write(stdout,
     &    '(5x,A,2x,A/,32x,A,1x,L1,2x,A,I4,2x,A,I3)')
     &    'eddy Diag File:', diags_eddyname(1:lstr),
     &    'Create new:', ldefdiags_eddy, 'nwrt =', nwrtdiags_eddy,
     &    'rec/file =', nrpfdiags_eddy
!
# ifdef AVERAGES
      elseif (keyword(1:kwlen).eq.'diags_eddy_avg') then
        call cancel_kwd (keyword(1:kwlen), ierr)
        read(input,*,err=95) ldefdiags_eddy_avg, ntsdiags_eddy_avg,
     &                      nwrtdiags_eddy_avg,  nrpfdiags_eddy_avg
        if (nwrtdiags_eddy_avg.eq.0) nwrtdiags_eddy_avg = navg
        read(input,'(A)',err=95) fname
        lstr=lenstr(fname)
# if defined MPI && defined PARALLEL_FILES
        call insert_node (fname, lstr, mynode2, NNODES2, ierr)
# endif
        diags_eddyname_avg=fname(1:lstr)
        MPI_master_only write(stdout,
     &    '(5x,A,2x,A/,32x,A,1x,L1,2x,A,I4,2x,A,I3,/32x,A,I10)')
     &    'eddy AVG Diag File:',diags_eddyname_avg(1:lstr),
     &    'Create new:', ldefdiags_eddy_avg,'nwrt =',nwrtdiags_eddy_avg,
     &    'rec/file =',nrpfdiags_eddy_avg,
     &    'Starting timestep = ',ntsdiags_eddy_avg

# endif
#endif /*DIAGNOSTICS_EDDY */

#if defined OUTPUTS_SURFACE && ! defined XIOS
!
! surface outputs file name.
!
      elseif (keyword(1:kwlen).eq.'surf') then
        call cancel_kwd (keyword(1:kwlen), ierr)
        read(input,*,err=95) ldefsurf, nwrtsurf, nrpfsurf
        if (nwrtsurf.eq.0) nwrtsurf = nwrt
        read(input,'(A)',err=95) fname
        lstr=lenstr(fname)
# if defined MPI && defined PARALLEL_FILES
        call insert_node (fname, lstr, mynode2, NNODES2, ierr)
# endif
        surfname=fname(1:lstr)
        MPI_master_only write(stdout,
     &    '(5x,A,2x,A/,32x,A,1x,L1,2x,A,I4,2x,A,I3)')
     &    'Surface outputs File:', surfname(1:lstr),
     &    'Create new:', ldefsurf, 'nwrt =', nwrtsurf,
     &    'rec/file =', nrpfsurf
!
# ifdef AVERAGES
      elseif (keyword(1:kwlen).eq.'surf_avg') then
        call cancel_kwd (keyword(1:kwlen), ierr)
        read(input,*,err=95) ldefsurf_avg, ntssurf_avg,
     &                      nwrtsurf_avg,  nrpfsurf_avg
        if (nwrtsurf_avg.eq.0) nwrtsurf_avg = navg
        read(input,'(A)',err=95) fname
        lstr=lenstr(fname)
# if defined MPI && defined PARALLEL_FILES
        call insert_node (fname, lstr, mynode2, NNODES2, ierr)
# endif
        surfname_avg=fname(1:lstr)
        MPI_master_only write(stdout,
     &    '(5x,A,2x,A/,32x,A,1x,L1,2x,A,I4,2x,A,I3,/32x,A,I10)')
     &    'Surface outputs AVG File:',surfname_avg(1:lstr),
     &    'Create new:', ldefsurf_avg,'nwrt =',nwrtsurf_avg,
     &    'rec/file =',nrpfsurf_avg,
     &    'Starting timestep = ',ntssurf_avg

# endif
#endif /*OUTPUTS_SURFACE */


#ifdef DIAGNOSTICS_BIO
!
! Diagnostics Biology file name.
!
      elseif (keyword(1:kwlen).eq.'diagnostics_bio') then
        call cancel_kwd (keyword(1:kwlen), ierr)
        read(input,*,err=95) ldefdiabio, nwrtdiabio, nrpfdiabio
        read(input,'(A)',err=95) fname
        lstr=lenstr(fname)
# if defined MPI && defined PARALLEL_FILES
        call insert_node (fname, lstr, mynode2, NNODES2, ierr)
# endif
        dianamebio=fname(1:lstr)
        MPI_master_only write(stdout,
     &    '(8x,A,2x,A/,32x,A,1x,L1,2x,A,I4,2x,A,I3)') 
     &    'Biology Diag File:', dianamebio(1:lstr),  'Create new:',
     &    ldefdiabio, 'nwrt =', nwrtdiabio, 'rec/file =', nrpfdiabio
!
# ifdef AVERAGES
      elseif (keyword(1:kwlen).eq.'diagbio_avg') then
        call cancel_kwd (keyword(1:kwlen), ierr)
        read(input,*,err=95) ldefdiabio_avg, ntsdiabio_avg,
     &                       nwrtdiabio_avg, nrpfdiabio_avg
        read(input,'(A)',err=95) fname
        lstr=lenstr(fname)
# if defined MPI && defined PARALLEL_FILES
        call insert_node (fname, lstr, mynode2, NNODES2, ierr)
# endif
        dianamebio_avg=fname(1:lstr)
        MPI_master_only write(stdout,
     &    '(4x,A,2x,A/,32x,A,1x,L1,2x,A,I4,2x,A,I3,/32x,A,I10)')
     &    'Biology AVG Diag File:',dianamebio_avg(1:lstr),'Create new:',
     &    ldefdiabio_avg,'nwrt =',nwrtdiabio_avg,'rec/file =',
     &    nrpfdiabio_avg,'Starting timestep = ',ntsdiabio_avg
# endif
#endif /* DIAGNOSTICS_BIO */

#ifdef FLOATS
!
! Floats file name.
!
      elseif (keyword(1:kwlen).eq.'floats') then
        call cancel_kwd (keyword(1:kwlen), ierr)
#ifdef AGRIF
        if (Agrif_Root()) then
#endif
          read(input,*,err=95) ldefflt, nflt, nrpfflt
          read(input,'(A)',err=95) fposnam
          read(input,'(A)',err=95) fname
          lstr=lenstr(fname)
# if defined MPI && defined PARALLEL_FILES
          call insert_node (fposnam, lstr, mynode2, NNODES2, ierr)
# endif
          fltname=fname(1:lstr)
          MPI_master_only write(stdout,
     &              '(9x,A,2x,A,2x,A,1x,L1,2x,A,I4,2x,A,I3)')
     &        'Float File:',fltname(1:lstr),  'Create new:',
     &        ldefflt, 'nflt =', nflt, 'rec/file =', nrpfflt
#ifdef AGRIF
        else
          ldefflt=Agrif_Parent(ldefflt)
          nflt=Agrif_Parent(nflt)
          nrpfflt=Agrif_Parent(nrpfflt)
        endif
#endif
#endif /* FLOATS */

#ifdef STATIONS
!
! Stations file name.
!
      elseif (keyword(1:kwlen).eq.'stations') then
        call cancel_kwd (keyword(1:kwlen), ierr)
# ifdef AGRIF
        if (Agrif_Root()) then
# endif
          read(input,*,err=95) ldefsta, nsta, nrpfsta
          read(input,'(A)',err=95) staposname
          read(input,'(A)',err=95) fname
          lstr=lenstr(fname)
# if defined MPI && defined PARALLEL_FILES
          call insert_node (staposname, lstr, mynode2, NNODES2, ierr)
# endif
          staname=fname(1:lstr)
          MPI_master_only write(stdout,
     &              '(9x,A,2x,A,2x,A,1x,L1,2x,A,I4,2x,A,I3)')
     &        'Station File:',staname(1:lstr),  'Create new:',
     &        ldefsta, 'nsta =', nsta, 'rec/file =', nrpfsta
# ifdef AGRIF
        else
          ldefsta=Agrif_Parent(ldefsta)
          nsta=Agrif_Parent(nsta)
          nrpfsta=Agrif_Parent(nrpfsta)
        endif
# endif
#endif /* STATIONS */

#ifdef ASSIMILATION
!
! Assimilation input/output file names.
!
      elseif (keyword(1:kwlen).eq.'assimilation') then
        call cancel_kwd (keyword(1:kwlen), ierr)
        read(input,'(A)',err=95) aparnam
        read(input,'(A)',err=95) assname
        fname=aparnam
        lstr=lenstr(aparnam)
        open (testunit,file=aparnam(1:lstr),status='old',err=97)
        close(testunit)
        MPI_master_only write(stdout,'(1x,A,2x,A)')
     &         'Assimilation Parameters File:', aparnam(1:lstr)
        fname=assname
        lstr=lenstr(assname)
        open (testunit,file=assname(1:lstr),status='old',err=97)
        close(testunit)
        MPI_master_only write(stdout,'(12x,A,2x,A)')
     &                    'Assimilation File:', assname(1:lstr)
#endif
!
! Switches for fields to be saved into history file.
!
      elseif (keyword(1:kwlen).eq.'primary_history_fields') then
        call cancel_kwd (keyword(1:kwlen), ierr)
        read(input,*,err=95) wrthis(indxZ),  wrthis(indxUb)
     &                                       ,  wrthis(indxVb)
#ifdef SOLVE3D
     &                    ,  wrthis(indxU),  wrthis(indxV)
# ifdef TRACERS
     &                    , (wrthis(itrc), itrc=indxV+1,indxV+NT) 
# endif
#endif
        if ( wrthis(indxZ) .or. wrthis(indxUb) .or. wrthis(indxVb)
#ifdef SOLVE3D
     &                        .or. wrthis(indxU) .or. wrthis(indxV)
#endif
     &     ) wrthis(indxTime)=.true.

        MPI_master_only write(stdout,'(/1x,A,5(/6x,l1,2x,A,1x,A))')
     &    'Fields to be saved in history file: (T/F)'
     &    , wrthis(indxZ),  'write zeta ', 'free-surface.'
     &    , wrthis(indxUb), 'write UBAR ', '2D U-momentum component.'
     &    , wrthis(indxVb), 'write VBAR ', '2D V-momentum component.'
#ifdef SOLVE3D
     &    , wrthis(indxU),  'write U    ', '3D U-momentum component.'
     &    , wrthis(indxV),  'write V    ', '3D V-momentum component.'
# ifdef TRACERS
        do itrc=1,NT
          if (wrthis(indxV+itrc)) wrthis(indxTime)=.true.
          MPI_master_only write(stdout, '(6x,L1,2x,A,I2,A,I2,A)')
     &                     wrthis(indxV+itrc), 'write T(', itrc,
     &                              ')  Tracer of index', itrc,'.'
        enddo
# endif
#endif

#if !defined SOLVE3D && defined RIP
      elseif (keyword(1:kwlen).eq.'auxiliary_history_fields') then
        call cancel_kwd (keyword(1:kwlen), ierr)
        read(input,*,err=95)
     &                                             wrthis(indxBostr)
     &                                          ,  wrthis(indxBustr)
     &                                          ,  wrthis(indxBvstr)
     &                                          ,  wrthis(indxWstr)
     &                                          ,  wrthis(indxUWstr)
     &                                          ,  wrthis(indxVWstr)
#elif defined SOLVE3D
      elseif (keyword(1:kwlen).eq.'auxiliary_history_fields') then
        call cancel_kwd (keyword(1:kwlen), ierr)
        read(input,*,err=95)
     &                                             wrthis(indxR)
     &                                          ,  wrthis(indxO)
     &                                          ,  wrthis(indxW)
     &                                          ,  wrthis(indxAkv)
# ifdef TEMPERATURE
     &                                          ,  wrthis(indxAkt)
# else 
     &                                          ,  dumboolean
# endif
# ifdef SALINITY
     &                                          ,  wrthis(indxAks)
# else 
     &                                          ,  dumboolean
# endif
# if defined ANA_VMIX || defined BVF_MIXING \
  || defined LMD_MIXING || defined LMD_SKPP || defined LMD_BKPP \
  || defined GLS_MIXING
     &                                          ,  wrthis(indxbvf)
# else
     &                                          ,  dumboolean
# endif
# ifdef VIS_COEF_3D
     &                                          ,  wrthis(indxVisc)
# else 
     &                                          ,  dumboolean
# endif
# ifdef DIF_COEF_3D
     &                                          ,  wrthis(indxDiff)
# else 
     &                                          ,  dumboolean
# endif
# if defined LMD_SKPP || defined GLS_MIXING
     &                                          ,  wrthis(indxHbl)
# else 
     &                                          ,  dumboolean
# endif
# ifdef LMD_BKPP
     &                                          ,  wrthis(indxHbbl)
# else 
     &                                          ,  dumboolean
# endif
     &                                          ,  wrthis(indxBostr)
     &                                          ,  wrthis(indxBustr)
     &                                          ,  wrthis(indxBvstr)
     &                                          ,  wrthis(indxWstr)
     &                                          ,  wrthis(indxUWstr)
     &                                          ,  wrthis(indxVWstr)
# ifdef TEMPERATURE
     &                                          ,  wrthis(indxShflx)
# else 
     &                                          ,  dumboolean
# endif
# ifdef SALINITY
     &                                          ,  wrthis(indxSwflx)
# else 
     &                                          ,  dumboolean
# endif
# ifdef TEMPERATURE
     &                                          ,  wrthis(indxShflx_rsw)
# else 
     &                                          ,  dumboolean
# endif
# if defined BULK_FLUX || defined OA_COUPLING
     &                                          ,  wrthis(indxShflx_rlw)
     &                                          ,  wrthis(indxShflx_lat)
     &                                          ,  wrthis(indxShflx_sen)
# else 
     &                                          ,  dumboolean
     &                                          ,  dumboolean
     &                                          ,  dumboolean
# endif
#ifdef BHFLUX
     &                                          ,  wrthis(indxBhflx)
#else
     &                                          ,  dumboolean
#endif
#if defined BWFLUX && defined SALINITY 
     &                                          ,  wrthis(indxBwflx)
#else
     &                                          ,  dumboolean
#endif       
# ifdef MORPHODYN
     &                                          ,  wrthis(indxHm)
# else 
     &                                          ,  dumboolean
# endif
# if defined BIOLOGY && !defined PISCES
     &                                          ,  wrthis(indxHel)
#  ifdef BIO_NChlPZD
     &                                          ,  wrthis(indxChC)
#   ifdef OXYGEN
     &                                          ,  wrthis(indxU10)
     &                                          ,  wrthis(indxKvO2)
     &                                          ,  wrthis(indxO2sat)
#   endif
#  elif defined BIO_BioEBUS
     &                                          ,  wrthis(indxAOU)
     &                                          ,  wrthis(indxWIND10)
#  endif
# endif

        if ( wrthis(indxR) 
     &                                        .or. wrthis(indxO) 
     &                                        .or. wrthis(indxW)
     &                                        .or. wrthis(indxAkv)
# ifdef TEMPERATURE
     &                                        .or. wrthis(indxAkt)
# endif
# ifdef SALINITY
     &                                        .or. wrthis(indxAks)
# endif
# if defined ANA_VMIX || defined BVF_MIXING \
  || defined LMD_MIXING || defined LMD_SKPP || defined LMD_BKPP \
  || defined GLS_MIXING
     &                                        .or. wrthis(indxbvf)
# endif
# ifdef VIS_COEF_3D
     &                                        .or. wrthis(indxVisc)
# endif
# ifdef DIF_COEF_3D
     &                                        .or. wrthis(indxDiff)
# endif
# if defined LMD_SKPP || defined GLS_MIXING
     &                                        .or. wrthis(indxHbl)
# endif
# ifdef LMD_BKPP
     &                                        .or. wrthis(indxHbbl)
# endif
     &                                        .or. wrthis(indxBostr)
     &                                        .or. wrthis(indxBustr)
     &                                        .or. wrthis(indxBvstr)
     &                                        .or. wrthis(indxWstr)
     &                                        .or. wrthis(indxUWstr)
     &                                        .or. wrthis(indxVWstr)
# ifdef TEMPERATURE
     &                                        .or. wrthis(indxShflx)
# endif
# ifdef SALINITY
     &                                        .or. wrthis(indxSwflx)
# endif
# ifdef TEMPERATURE
     &                                        .or. wrthis(indxShflx_rsw)
# endif
# if defined BULK_FLUX || defined OA_COUPLING
     &                                        .or. wrthis(indxShflx_rlw)
     &                                        .or. wrthis(indxShflx_lat)
     &                                        .or. wrthis(indxShflx_sen)
# endif
# if defined BHFLUX
     &                                        .or. wrthis(indxBhflx)
# endif
# if defined BWFLUX && defined SALINITY
     &                                        .or. wrthis(indxBwflx)
# endif 
# if defined BIOLOGY && !defined PISCES
     &                                        .or. wrthis(indxHel)
#  ifdef BIO_NChlPZD
     &                                        .or. wrthis(indxChC)
#   ifdef OXYGEN
     &                                        .or. wrthis(indxU10)
     &                                        .or. wrthis(indxKvO2)
     &                                        .or. wrthis(indxO2sat)
#   endif
#  elif defined BIO_BioEBUS
     &                                        .or. wrthis(indxAOU)
     &                                        .or. wrthis(indxWIND10)
#  endif
# endif
     &     ) wrthis(indxTime)=.true.


        MPI_master_only write(stdout,'(8(/6x,l1,2x,A,1x,A))')
     &    wrthis(indxR),    'write RHO  ', 'Density anomaly.'
     &  , wrthis(indxO),    'write Omega', 'Omega vertical velocity.'
     &  , wrthis(indxW),    'write W    ', 'True vertical velocity.'
     &  , wrthis(indxAkv),  'write Akv  ', 'Vertical viscosity.'
# ifdef TEMPERATURE
     &  , wrthis(indxAkt),  'write Akt  ',
     &                      'Vertical diffusivity for temperature.'
# endif
# ifdef SALINITY
     &  , wrthis(indxAks),  'write Aks  ',
     &                      'Vertical diffusivity for salinity.'
# endif
# if defined ANA_VMIX || defined BVF_MIXING \
  || defined LMD_MIXING || defined LMD_SKPP || defined LMD_BKPP \
  || defined GLS_MIXING
     &  , wrthis(indxbvf),  'write bvf  ',
     &                         'Brunt Vaisala Frequency.'
#  endif
# ifdef VIS_COEF_3D
     &  , wrthis(indxVisc),  'write Visc3d', 'Horizontal viscosity.'
# endif
# ifdef DIF_COEF_3D
     &  , wrthis(indxDiff),  'write Visc3d', 'Horizontal diffusivity.'
# endif
# if defined LMD_SKPP || defined GLS_MIXING
     &  , wrthis(indxHbl),  'write Hbl  ',
     &                      'Depth of model boundary layer.'
# endif
# ifdef LMD_BKPP
     &  , wrthis(indxHbbl), 'write Hbbl  ',
     &                      'Depth of bottom planetary boundary layer.'
# endif
# if defined BULK_FLUX || defined OA_COUPLING
     &  , wrthis(indxShflx_rlw), 'write shflx_rlw [W/m2]', 
     &                                 'Long Wave heat flux.'
     &  , wrthis(indxShflx_lat), 'write shflx_lat [W/m2]',
     &                                 'Latent heat flux.'
     &  , wrthis(indxShflx_sen), 'write shflx_sen [W/m2]',
     &                                 'Sensible heat flux'
# endif
# ifdef MORPHODYN
     &  , wrthis(indxHm),   'write Hm [m]',
     &                                 'Time evolving bathymetry'
# endif 
# if defined BIOLOGY && !defined PISCES
     &  , wrthis(indxHel),   'write Hel  ',
     &                       'Depth of the euphotic layer'
#  ifdef BIO_NChlPZD
     &  , wrthis(indxChC),   'write ChC  ',
     &                       'Chlorophyll to Carbon ratio'
#   ifdef OXYGEN
     &  , wrthis(indxU10),   'write u10 ',
     &                       'Wind speed at 10 m height'
     &  , wrthis(indxKvO2),  'write Kv_O2 ',
     &                       'Gas transfer coefficient for O2'
     &  , wrthis(indxO2sat), 'write O2sat ',
     &                       'Saturation concentration of O2'
#   endif
#  elif defined BIO_BioEBUS 
     &  , wrthis(indxAOU), 'write AOU ' 
     &  ,                  'Apparent Oxygen Utilization.'
     &  , wrthis(indxWIND10), 'write wind10 ', 'wind speed at 10 m.'  
#  endif
# endif /* BIOLOGY */
     &  , wrthis(indxBostr), 'write Bostr', 'Bottom Stress.'
     &  , wrthis(indxBustr), 'write Bustr', 'U-Bottom Stress.'
     &  , wrthis(indxBvstr), 'write Bvstr', 'V-Bottom Stress.'
     &  , wrthis(indxWstr),  'write Wstress', 'Wind Stress.'
     &  , wrthis(indxUWstr), 'write U-Wstress comp.', 'U-Wind Stress.'
     &  , wrthis(indxVWstr), 'write V-Wstress comp.', 'V-Wind Stress.'
# ifdef TEMPERATURE
     &  , wrthis(indxShflx), 'write Shflx [W/m2]', 
     &                       'Surface net heat flux'
# endif
# ifdef SALINITY
     &  , wrthis(indxSwflx), 'write Swflx [cm/day]', 
     &                       'Surface freshwater flux (E-P)'
# endif
# ifdef TEMPERATURE
     &  , wrthis(indxShflx_rsw),'write Shflx_rsw [W/m2]', 
     &                          'Short-wave surface radiation'
# endif
# ifdef BHFLUX
     &  , wrthis(indxBhflx), 'write Bhflx [W/m2]', 
     &                       'Bottom net heat flux'        
# endif
# if defined BWFLUX && defined SALINITY
     &  , wrthis(indxBwflx), 'write Bwflx [cm/day]', 
     &                       'Bottom freshwater flux (E-P)'
# endif        

!
! Switches for GLS fields to be saved into history file.
!
# ifdef GLS_MIXING
      elseif (keyword(1:kwlen).eq.'gls_history_fields') then
        call cancel_kwd (keyword(1:kwlen), ierr)
        read(input,*,err=95) wrthis(indxTke),  wrthis(indxGls)
     &                    ,  wrthis(indxLsc)
        if (wrthis(indxTke) .or. wrthis(indxGls) .or. wrthis(indxLsc)
     &     ) wrthis(indxTime)=.true.

        MPI_master_only write(stdout,'(/1x,A,5(/6x,l1,2x,A,1x,A))')
     &    'Fields to be saved in history file: (T/F)'
     &   , wrthis(indxTke), 'write TKE ', 'turbulent kinetic energy.  '
     &   , wrthis(indxGls), 'write GLS ', 'generic length scale.'
     &   , wrthis(indxLsc), 'write Lscale ', 
     &                                  'vertical mixing length scale.'
# endif
#endif /* SOLVE3D */

#ifdef AVERAGES
!
! Switches for fields to be saved into averages file.
!
      elseif (keyword(1:kwlen).eq.'primary_averages') then
        call cancel_kwd (keyword(1:kwlen), ierr)
        read(input,*,err=95) wrtavg(indxZ),  wrtavg(indxUb)
     &                                    ,  wrtavg(indxVb)
# ifdef SOLVE3D
     &                    ,  wrtavg(indxU),  wrtavg(indxV)
#  ifdef TRACERS
     &                    , (wrtavg(itrc), itrc=indxV+1,indxV+NT)
#  endif
# endif
        if ( wrtavg(indxZ) .or. wrtavg(indxUb) .or. wrtavg(indxVb)
# ifdef SOLVE3D
     &                     .or. wrtavg(indxU)  .or. wrtavg(indxV)
# endif
     &     ) wrtavg(indxTime)=.true.

        MPI_master_only write(stdout,'(/1x,A,5(/6x,l1,2x,A,1x,A))')
     &  'Fields to be saved in averages file: (T/F)'
     &  , wrtavg(indxZ),  'write zeta ', 'free-surface.'
     &  , wrtavg(indxUb), 'write UBAR ', '2D U-momentum component.'
     &  , wrtavg(indxVb), 'write VBAR ', '2D V-momentum component.'
# ifdef SOLVE3D
     &  , wrtavg(indxU),  'write U    ', '3D U-momentum component.'
     &  , wrtavg(indxV),  'write V    ', '3D V-momentum component.'
#  ifdef TRACERS
         do itrc=1,NT
          if (wrtavg(indxV+itrc)) wrtavg(indxTime)=.true.
          MPI_master_only write(stdout,
     &                     '(6x,L1,2x,A,I2,A,2x,A,I2,A)')
     &                      wrtavg(indxV+itrc), 'write T(',
     &                      itrc,')', 'Tracer of index', itrc,'.'
        enddo
#  endif

      elseif (keyword(1:kwlen).eq.'auxiliary_averages') then
        call cancel_kwd (keyword(1:kwlen), ierr)
        read(input,*,err=95) wrtavg(indxR), wrtavg(indxO)
     &        ,  wrtavg(indxW),  wrtavg(indxAkv)
# ifdef TEMPERATURE
     &                                          ,  wrtavg(indxAkt)
# else 
     &                                          ,  dumboolean
# endif
# ifdef SALINITY
     &                                          ,  wrtavg(indxAks)
# else 
     &                                          ,  dumboolean
# endif
# if defined ANA_VMIX || defined BVF_MIXING \
  || defined LMD_MIXING || defined LMD_SKPP || defined LMD_BKPP \
  || defined GLS_MIXING
     &                                          ,  wrtavg(indxbvf)
# else
     &                                          ,  dumboolean
# endif
# ifdef VIS_COEF_3D
     &                                          ,  wrtavg(indxVisc)
# else 
     &                                          ,  dumboolean
# endif
# ifdef DIF_COEF_3D
     &                                          ,  wrtavg(indxDiff)
# else 
     &                                          ,  dumboolean
# endif
# if defined LMD_SKPP || defined GLS_MIXING
     &                                          ,  wrtavg(indxHbl)
# else 
     &                                          ,  dumboolean
# endif
# ifdef LMD_BKPP
     &                                          ,  wrtavg(indxHbbl)
# else 
     &                                          ,  dumboolean
# endif
     &                                          ,  wrtavg(indxBostr)
     &                                          ,  wrtavg(indxBustr)
     &                                          ,  wrtavg(indxBvstr)
     &                                          ,  wrtavg(indxWstr)
     &                                          ,  wrtavg(indxUWstr)
     &                                          ,  wrtavg(indxVWstr)
# ifdef TEMPERATURE
     &                                          ,  wrtavg(indxShflx)
# else 
     &                                          ,  dumboolean
# endif
# ifdef SALINITY
     &                                          ,  wrtavg(indxSwflx)
# else 
     &                                          ,  dumboolean
# endif
# ifdef TEMPERATURE
     &                                          ,  wrtavg(indxShflx_rsw)
# else 
     &                                          ,  dumboolean
# endif
# if defined BULK_FLUX || defined OA_COUPLING
     &                                          ,  wrtavg(indxShflx_rlw)
     &                                          ,  wrtavg(indxShflx_lat)
     &                                          ,  wrtavg(indxShflx_sen)
# else 
     &                                          ,  dumboolean
     &                                          ,  dumboolean
     &                                          ,  dumboolean
# endif
#ifdef BHFLUX
     &                                          , wrtavg(indxBhflx)
#else
     &                                          , dumboolean
#endif
#if defined BWFLUX && defined SALINITY 
     &                                          , wrtavg(indxBwflx)
#else
     &                                          , dumboolean
#endif      
# ifdef MORPHODYN
     &                                          ,  wrtavg(indxHm)
# else 
     &                                          ,  dumboolean
# endif
# if defined BIOLOGY && !defined PISCES
     &                                          ,  wrtavg(indxHel)
#  ifdef BIO_NChlPZD
     &                                          ,  wrtavg(indxChC)
#   ifdef OXYGEN
     &                                          ,  wrtavg(indxU10)
     &                                          ,  wrtavg(indxKvO2)
     &                                          ,  wrtavg(indxO2sat)
#   endif
#  elif defined BIO_BioEBUS
     &                                          ,  wrtavg(indxAOU)
     &                                          ,  wrtavg(indxWIND10)
#  endif
# endif

        if ( wrtavg(indxR) .or. wrtavg(indxO) .or. wrtavg(indxW)
     &                   .or. wrtavg(indxAkv) 
#  ifdef TEMPERATURE
     &                                        .or. wrtavg(indxAkt)
#  endif
#  ifdef SALINITY
     &                                        .or. wrtavg(indxAks)
#  endif
# if defined ANA_VMIX || defined BVF_MIXING \
  || defined LMD_MIXING || defined LMD_SKPP || defined LMD_BKPP \
  || defined GLS_MIXING
     &                                        .or. wrtavg(indxbvf)
# endif
# ifdef VIS_COEF_3D
     &                                        .or. wrtavg(indxVisc)
# endif
# ifdef DIF_COEF_3D
     &                                        .or. wrtavg(indxDiff)
# endif
#  if defined LMD_SKPP || defined GLS_MIXING
     &                                        .or. wrtavg(indxHbl)
#  endif
#  ifdef LMD_BKPP
     &                                        .or. wrtavg(indxHbbl)
#  endif
     &                                        .or. wrtavg(indxBostr)
     &                                        .or. wrtavg(indxBustr)
     &                                        .or. wrtavg(indxBvstr)
     &                                        .or. wrtavg(indxWstr)
     &                                        .or. wrtavg(indxUWstr)
     &                                        .or. wrtavg(indxVWstr)
#  ifdef TEMPERATURE
     &                                        .or. wrtavg(indxShflx)
#  endif
#  ifdef SALINITY
     &                                        .or. wrtavg(indxSwflx)
#  endif
#  ifdef TEMPERATURE
     &                                        .or. wrtavg(indxShflx_rsw)
#  endif
     &     
#  if defined BULK_FLUX || defined OA_COUPLING
     &                                        .or. wrtavg(indxShflx_rlw)
     &                                        .or. wrtavg(indxShflx_lat)
     &                                        .or. wrtavg(indxShflx_sen)
#  endif
# if defined BHFLUX
     &                                        .or. wrtavg(indxBhflx)
#endif
# if defined BWFLUX && defined SALINITY
     &                                        .or. wrtavg(indxBwflx)
#endif 
#  if defined BIOLOGY && !defined PISCES
     &                                        .or. wrtavg(indxHel)
#   ifdef BIO_NChlPZD
     &                                        .or. wrtavg(indxChC)
#    ifdef OXYGEN
     &                                        .or. wrtavg(indxU10)
     &                                        .or. wrtavg(indxKvO2)
     &                                        .or. wrtavg(indxO2sat)
#    endif 
#   elif defined BIO_BioEBUS
     &                                     .or. wrtavg(indxAOU)
     &                                     .or. wrtavg(indxWIND10)
#   endif
#  endif  /* BIOLOGY */
     &     ) wrtavg(indxTime)=.true.



        MPI_master_only write(stdout,'(8(/6x,l1,2x,A,1x,A))')
     &    wrtavg(indxR),    'write RHO  ', 'Density anomaly'
     &  , wrtavg(indxO),    'write Omega', 'Omega vertical velocity.'
     &  , wrtavg(indxW),    'write W    ', 'True vertical velocity.'
     &  , wrtavg(indxAkv),  'write Akv  ', 'Vertical viscosity'
#  ifdef TEMPERATURE
     &  , wrtavg(indxAkt),  'write Akt  ',
     &                      'Vertical diffusivity for temperature.'
#  endif
#  ifdef SALINITY
     &  , wrtavg(indxAks),  'write Aks  ',
     &                         'Vertical diffusivity for salinity.'
#  endif
# if defined ANA_VMIX || defined BVF_MIXING \
  || defined LMD_MIXING || defined LMD_SKPP || defined LMD_BKPP \
  || defined GLS_MIXING
     &  , wrtavg(indxbvf),  'write bvf  ',
     &                         'Brunt Vaisala Frequency.'
#  endif
#  ifdef VIS_COEF_3D
     &  , wrtavg(indxVisc),'write visc3d', 'Horizontal viscosity'
#  endif
#  ifdef DIF_COEF_3D
     &  , wrtavg(indxDiff),'write diff3d', 'Horizontal diffusivity'
#  endif
#  if defined LMD_SKPP || defined GLS_MIXING
     &  , wrtavg(indxHbl),  'write Hbl  ',
     &                          'Depth of model boundary layer'
#  endif
#  ifdef LMD_BKPP
     &  , wrtavg(indxHbbl),  'write Hbbl  ',
     &                    'Depth of the bottom planetary boundary layer'
#  endif
#  ifdef BULK_FLUX
     &  , wrtavg(indxShflx_rlw), 'write shflx_rlw [W/m2]', 
     &                                 'Long Wave heat flux.'
     &  , wrtavg(indxShflx_lat), 'write shflx_lat[W/m2] ',
     &                                 'Latente heat flux.'
     &  , wrtavg(indxShflx_sen), 'write shflx_sen [W/m2]',
     &                                 'Sensible heat flux.'
#  endif
# ifdef MORPHODYN
     &  , wrtavg(indxHm), 'write Hm [m]',
     &                                 'Time evolving bathymetry.'
# endif
#  if defined BIOLOGY && !defined PISCES
     &  , wrtavg(indxHel),'write Hel  ',
     &                          'Depth of the euphotic layer'
#   ifdef BIO_NChlPZD
     &  , wrtavg(indxChC),'write ChC  ',
     &                          'Chlorophyll to Carbon ratio'
#    ifdef OXYGEN
     &  , wrtavg(indxU10),'write u10 ',
     &         'Wind speed at 10 m height'
     &  , wrtavg(indxKvO2),'write Kv_O2 ',
     &         'Gas transfer coefficient for O2'
     &  , wrtavg(indxO2sat),'write O2sat ',
     &         'Saturation concentration of O2'
#    endif

#   elif defined BIO_BioEBUS
     &  , wrtavg(indxAOU), 'write AOU '
     &  ,                      'Apparent Oxygen Utilization.'
     &  , wrtavg(indxWIND10), 'write wind10 ', 'wind speed at 10 m.'
#   endif
#  endif   /* BIOLOGY */
     &  , wrtavg(indxBostr),'write Bostr', 'Bottom Stress.'
     &  , wrtavg(indxBustr),'write Bustr', 'U-Bottom Stress.'
     &  , wrtavg(indxBvstr),'write Bvstr', 'V-Bottom Stress.'
     &  , wrtavg(indxWstr), 'write Wstr', 'Wind Stress.'
     &  , wrtavg(indxUWstr),'write U-Wstress comp.', 'U-Wind Stress.'
     &  , wrtavg(indxVWstr),'write V-Wstress comp.', 'V-Wind Stress.'
#  ifdef TEMPERATURE
     &  , wrtavg(indxShflx),'write Shflx [W/m2]', 
     &                      'Surface net heat flux.'
#  endif
#  ifdef SALINITY
     &  , wrtavg(indxSwflx),'write Swflx [cm/day]', 
     &                      'Surface freshwater flux (E-P)'
#  endif
#  ifdef TEMPERATURE
     &  , wrtavg(indxShflx_rsw),'write Shflx_rsw [W/m2]', 
     &                      'Short-wave surface radiation.'
# endif     
# ifdef BHFLUX
     &  , wrtavg(indxBhflx), 'write Bhflx [W/m2]', 
     &       'Bottom net heat flux'        
# endif
# if defined BWFLUX && defined SALINITY
     &  , wrtavg(indxBwflx), 'write Bwflx [cm/day]', 
     &                       'Surface freshwater flux (E-P)'
# endif     
!
! Switches for GLS fields to be saved into history file.
!
#  ifdef GLS_MIXING
      elseif (keyword(1:kwlen).eq.'gls_averages') then
        call cancel_kwd (keyword(1:kwlen), ierr)
        read(input,*,err=95) wrtavg(indxTke),  wrtavg(indxGls)
     &                    ,  wrtavg(indxLsc)       
        if ( wrtavg(indxAkk) .or. wrtavg(indxAkp) .or. wrtavg(indxTke)
     &                       .or. wrtavg(indxGls) .or. wrtavg(indxLsc)
     &     ) wrtavg(indxTime)=.true.

        MPI_master_only write(stdout,'(/1x,A,5(/6x,l1,2x,A,1x,A))')
     &    'Fields to be saved in average file: (T/F)'
     &   , wrtavg(indxTke), 'write TKE ', 'turbulent kinetic energy.  '
     &   , wrtavg(indxGls), 'write GLS ', 'generic length scale.'
     &   , wrtavg(indxLsc), 'write Lscale ', 
     &                                  'vertical mixing length scale.'
#  endif
# endif /* SOLVE3D */
#endif /* AVERAGES */


#ifdef FLOATS
!
! Switches for fields to be saved into floats output file.
!
      elseif (keyword(1:kwlen).eq.'float_fields') then
        call cancel_kwd (keyword(1:kwlen), ierr)
#ifdef AGRIF
        if (Agrif_Root()) then
#endif
          read(input,*,err=95) wrtflt(indxfltGrd), 
     &       wrtflt(indxfltTemp), wrtflt(indxfltSalt), 
     &       wrtflt(indxfltRho), wrtflt(indxfltVel)
           MPI_master_only write(stdout,'(/1x,A,5(/6x,l1,2x,A))')
     &      'Fields to be saved in floats output file: (T/F)'
     &     , wrtflt(indxfltGrd),   'write Grid location variables'
     &     , wrtflt(indxfltTemp),  'write temperature.'
     &     , wrtflt(indxfltSalt),  'write salinity.'
     &     , wrtflt(indxfltRho),   'write density.'
     &     , wrtflt(indxfltVel),   'write mean float velocity' 
#ifdef AGRIF
        endif
#endif
#endif /* FLOATS */

#if defined DIAGNOSTICS_TS
!
! Switches for fields to be saved into tracer diagnostics file.
!
      elseif (keyword(1:kwlen).eq.'diag3D_history_fields') then
        call cancel_kwd (keyword(1:kwlen), ierr)
# ifdef TRACERS
        read(input,*,err=95)  (wrtdia3D(itrc), itrc=1, NT)
        do itrc=1,NT
          if (wrtdia3D(itrc)) wrtdia3D(NT+1)=.true.
          MPI_master_only write(stdout, '(6x,L1,2x,A,2x,A,I2,A,I4)')
     &        wrtdia3D(itrc), 
     &        'write Tracer equation terms ',
     &        ' Tracer of index', itrc,'/',NT
        enddo
# endif  /* TRACERS */
# ifdef DIAGNOSTICS_TS_MLD
      elseif (keyword(1:kwlen).eq.'diag2D_history_fields') then
        call cancel_kwd (keyword(1:kwlen), ierr)
#  ifdef TRACERS
        read(input,*,err=95)  (wrtdia2D(itrc), itrc=1, NT)
        do itrc=1,NT
          if (wrtdia2D(itrc)) wrtdia2D(NT+1)=.true.
          MPI_master_only write(stdout, '(6x,L1,2x,A,2x,A,I2,A,I4)')
     &        wrtdia2D(itrc), 
     &        'write Hbl Integrated Tracer equation terms ',
     &        ' Tracer of index', itrc,'/',NT
        enddo
#  endif  /* TRACERS */
# endif  /* DIAGNOSTICS_TS_MLD */

# ifdef AVERAGES
!
! Switches for fields to be saved into tracer diagnostics average file.
!
      elseif (keyword(1:kwlen).eq.'diag3D_average_fields') then
        call cancel_kwd (keyword(1:kwlen), ierr)
#  ifdef TRACERS
        read(input,*,err=95) (wrtdia3D_avg(itrc),itrc=1,NT)
        do itrc=1,NT
          if (wrtdia3D_avg(itrc)) wrtdia3D_avg(NT+1)=.true.
          MPI_master_only write(stdout, '(6x,L1,2x,A,2x,A,I2,A,I4)')
     &        wrtdia3D_avg(itrc), 
     &        'write Avg Tracer equation terms ',
     &        ' Tracer of index', itrc,'/',NT
        enddo
#  endif  /* TRACERS */
!
#  ifdef DIAGNOSTICS_TS_MLD
      elseif (keyword(1:kwlen).eq.'diag2D_average_fields') then
        call cancel_kwd (keyword(1:kwlen), ierr)
#   ifdef TRACERS
        read(input,*,err=95) (wrtdia2D_avg(itrc),itrc=1,NT)
        do itrc=1,NT
          if (wrtdia2D_avg(itrc)) wrtdia2D_avg(NT+1)=.true.
          MPI_master_only write(stdout, '(6x,L1,2x,A,2x,A,I2,A,I4)')
     &        wrtdia2D_avg(itrc), 
     &        'write Avg Hbl Integrated  Tracer equation terms ',
     &        ' Tracer of index', itrc,'/',NT
        enddo
#   endif  /* TRACERS */
#  endif  /* DIAGNOSTICS_TS_MLD */
# endif  /* AVERAGES */

#endif  /* DIAGNOSTICS_TS */


#if defined DIAGNOSTICS_UV
!
! Switches for fields to be saved into momentum diagnostics file.
!
!!          wrtdiaM=.true.
      elseif (keyword(1:kwlen).eq.'diagM_history_fields') then
        call cancel_kwd (keyword(1:kwlen), ierr)
        read(input,*,err=95)  (wrtdiaM(itrc), itrc=1,2)
         do itrc=1,2
           if (wrtdiaM(itrc)) wrtdiaM(3)=.true.
           MPI_master_only write(stdout, '(6x,L1,2x,A,2x,A,I2)')
     &         wrtdiaM(itrc), 
     &         'write momentum equation terms ',
     &         ' Momentum of index', itrc
        enddo

# ifdef AVERAGES
!!         wrtdiaM_avg=.true.
      elseif (keyword(1:kwlen).eq.'diagM_average_fields') then
        call cancel_kwd (keyword(1:kwlen), ierr)
        read(input,*,err=95) (wrtdiaM_avg(itrc),itrc=1,2)  
        do itrc=1,2
          if (wrtdiaM_avg(itrc)) wrtdiaM_avg(3)=.true.
          MPI_master_only write(stdout, '(6x,L1,2x,A,2x,A,I2)')
     &        wrtdiaM_avg(itrc), 
     &        'write averaged momentum  equation terms ',
     &        ' Momentum of index', itrc
       enddo
# endif /* AVERAGES  */

#endif /*DIAGNOSTICS_UV */

#if defined DIAGNOSTICS_VRT
!
! Switches for fields to be saved into vorticity diagnostics file.
!
!!          wrtdiags_vrt=.true.
      elseif (keyword(1:kwlen).eq.'diags_vrt_history_fields') then
        call cancel_kwd (keyword(1:kwlen), ierr)
        read(input,*,err=95)  (wrtdiags_vrt(itrc), itrc=1,1)
         do itrc=1,1
           if (wrtdiags_vrt(itrc)) wrtdiags_vrt(3)=.true.
           MPI_master_only write(stdout, '(6x,L1,2x,A,2x,A,I2)')
     &         wrtdiags_vrt(itrc),
     &         'write vorticity equation terms ',
     &         ' Momentum of index', itrc
        enddo

# ifdef AVERAGES
!!         wrtdiags_vrt_avg=.true.
      elseif (keyword(1:kwlen).eq.'diags_vrt_average_fields') then
        call cancel_kwd (keyword(1:kwlen), ierr)
        read(input,*,err=95) (wrtdiags_vrt_avg(itrc),itrc=1,1)
        do itrc=1,1
          if (wrtdiags_vrt_avg(itrc)) wrtdiags_vrt_avg(3)=.true.
          MPI_master_only write(stdout, '(6x,L1,2x,A,2x,A,I2)')
     &        wrtdiags_vrt_avg(itrc),
     &        'write averaged vorticity  equation terms ',
     &        ' Momentum of index', itrc
       enddo
# endif /* AVERAGES  */

#endif /*DIAGNOSTICS_VRT */

#if defined DIAGNOSTICS_EK
!
! Switches for fields to be saved into kinetic energy diagnostics file.
!
!!          wrtdiags_ek=.true.
      elseif (keyword(1:kwlen).eq.'diags_ek_history_fields') then
        call cancel_kwd (keyword(1:kwlen), ierr)
        read(input,*,err=95)  (wrtdiags_ek(itrc), itrc=1,1)
         do itrc=1,1
           if (wrtdiags_ek(itrc)) wrtdiags_ek(3)=.true.
           MPI_master_only write(stdout, '(6x,L1,2x,A,2x,A,I2)')
     &         wrtdiags_ek(itrc),
     &         'write energy equation terms ',
     &         ' Momentum of index', itrc
        enddo

# ifdef AVERAGES
!!         wrtdiags_ek_avg=.true.
      elseif (keyword(1:kwlen).eq.'diags_ek_average_fields') then
        call cancel_kwd (keyword(1:kwlen), ierr)
        read(input,*,err=95) (wrtdiags_ek_avg(itrc),itrc=1,1)
        do itrc=1,1
          if (wrtdiags_ek_avg(itrc)) wrtdiags_ek_avg(3)=.true.
          MPI_master_only write(stdout, '(6x,L1,2x,A,2x,A,I2)')
     &        wrtdiags_ek_avg(itrc),
     &        'write averaged energy equation terms ',
     &        ' Momentum of index', itrc
       enddo
# endif /* AVERAGES  */
#endif /*DIAGNOSTICS_EK */


#if defined DIAGNOSTICS_PV
!
! Switches for fields to be saved into potential vorticity diagnostics file.
!
!!          wrtdiags_pv=.true.
      elseif (keyword(1:kwlen).eq.'diags_pv_history_fields') then
        call cancel_kwd (keyword(1:kwlen), ierr)
# ifdef TRACERS
        read(input,*,err=95)  (wrtdiags_pv(itrc), itrc=1,NT)
         do itrc=1,NT
           if (wrtdiags_pv(itrc)) wrtdiags_pv(3)=.true.
           MPI_master_only write(stdout, '(6x,L1,2x,A,2x,A,I2)')
     &         wrtdiags_pv(itrc),
     &         'write potential vorticity terms '
        enddo
# endif /* TRACERS  */

# ifdef AVERAGES
!!         wrtdiags_pv_avg=.true.
      elseif (keyword(1:kwlen).eq.'diags_pv_average_fields') then
        call cancel_kwd (keyword(1:kwlen), ierr)
#  ifdef TRACERS
        read(input,*,err=95) (wrtdiags_pv_avg(itrc),itrc=1,NT)
        do itrc=1,NT
          if (wrtdiags_pv_avg(itrc)) wrtdiags_pv_avg(3)=.true.
          MPI_master_only write(stdout, '(6x,L1,2x,A,2x,A,I2)')
     &        wrtdiags_pv_avg(itrc),
     &         'write potential vorticity terms '
       enddo
#  endif /* TRACERS  */
# endif /* AVERAGES  */
#endif /*DIAGNOSTICS_PV */


#if defined DIAGNOSTICS_EDDY && ! defined XIOS
!
! Switches for fields to be saved into eddy diagnostics file.
!
!!          wrtdiags_eddy=.true.
      elseif (keyword(1:kwlen).eq.'diags_eddy_history_fields') then
        call cancel_kwd (keyword(1:kwlen), ierr)
        read(input,*,err=95)  wrtdiags_eddy(1)
         do itrc=1,1
           if (wrtdiags_eddy(itrc)) wrtdiags_eddy(3)=.true.
           MPI_master_only write(stdout, '(6x,L1,2x,A,2x,A,I2)')
     &         wrtdiags_eddy(itrc),
     &         'write Reynolds stress terms '
        enddo

# ifdef AVERAGES
!!         wrtdiags_eddy_avg=.true.
      elseif (keyword(1:kwlen).eq.'diags_eddy_average_fields') then
        call cancel_kwd (keyword(1:kwlen), ierr)
        read(input,*,err=95) wrtdiags_eddy_avg(1)
        do itrc=1,1
          if (wrtdiags_eddy_avg(itrc)) wrtdiags_eddy_avg(3)=.true.
          MPI_master_only write(stdout, '(6x,L1,2x,A,2x,A,I2)')
     &        wrtdiags_eddy_avg(itrc),
     &         'write Reynolds stress terms '
       enddo
# endif /* AVERAGES  */
#endif /*DIAGNOSTICS_EDDY */


#if defined OUTPUTS_SURFACE && ! defined XIOS
!
! Switches for fields to be saved into surface outputs file.
!
!!          wrtsurf=.true.
      elseif (keyword(1:kwlen).eq.'surf_history_fields') then
        call cancel_kwd (keyword(1:kwlen), ierr)
        read(input,*,err=95)  (wrtsurf(itrc), itrc=1,1)
         do itrc=1,1
           if (wrtsurf(itrc)) wrtsurf(3)=.true.
           MPI_master_only write(stdout, '(6x,L1,2x,A,2x,A,I2)')
     &         wrtsurf(itrc),
     &         'write surface outputs ',
     &         ' Momentum of index', itrc
        enddo

# ifdef AVERAGES
!!         wrtsurf_avg=.true.
      elseif (keyword(1:kwlen).eq.'surf_average_fields') then
        call cancel_kwd (keyword(1:kwlen), ierr)
        read(input,*,err=95) (wrtsurf_avg(itrc),itrc=1,1)
        do itrc=1,1
          if (wrtsurf_avg(itrc)) wrtsurf_avg(3)=.true.
          MPI_master_only write(stdout, '(6x,L1,2x,A,2x,A,I2)')
     &        wrtsurf_avg(itrc),
     &        'write averaged surface outputs ',
     &        ' Momentum of index', itrc
       enddo
# endif /* AVERAGES  */

#endif /*OUTPUTS_SURFACE*/

#ifdef DIAGNOSTICS_BIO
!
! Switches for fields to be saved into biology diagnostics file.
!
!          wrtdiabio=.true.
      elseif (keyword(1:kwlen).eq.'diagbioFlux_history_fields') then
!
CR       write(*,*)'NumFluxTerms=',NumFluxTerms
CR        write(*,*)'NumVSinkTerms=',NumVSinkTerms
CR        write(*,*)'NumGasExcTerms=',NumGasExcTerms
!
        call cancel_kwd (keyword(1:kwlen), ierr)
!     ==
        read(input,*,err=95) (wrtdiabioFlux(iflux),iflux=1,NumFluxTerms)
        do iflux=1,NumFluxTerms
          if (wrtdiabioFlux(iflux)) 
     &         wrtdiabioFlux(NumFluxTerms+1)=.true.
          MPI_master_only write(stdout, '(6x,L1,2x,A,2x,A,I2)')
     &         wrtdiabioFlux(iflux), 
     &         'write equation terms ',
     &         'Flux term of index (his)', iflux
        enddo
!     ==
      elseif (keyword(1:kwlen).eq.'diagbioVSink_history_fields') then
        call cancel_kwd (keyword(1:kwlen), ierr)
        read(input,*,err=95) (wrtdiabioVSink(iflux),
     &       iflux=1,NumVSinkTerms)
        do iflux=1,NumVSinkTerms
          if (wrtdiabioVSink(iflux)) 
     &         wrtdiabioVSink(NumVSinkTerms+1)=.true.
          MPI_master_only write(stdout, '(6x,L1,2x,A,2x,A,I2)')
     &         wrtdiabioVSink(iflux), 
     &         'write equation terms ',
     &         'VSink term of index (his)', iflux
        enddo
!     == 
#  if (defined BIO_NChlPZD && defined OXYGEN) || defined BIO_BioEBUS
      elseif (keyword(1:kwlen).eq.'diagbioGasExc_history_fields') then
        call cancel_kwd (keyword(1:kwlen), ierr)
        read(input,*,err=95) (wrtdiabioGasExc(iflux),
     &                                       iflux=1,NumGasExcTerms)
        do iflux=1,NumGasExcTerms
          if (wrtdiabioGasExc(iflux)) 
     &         wrtdiabioGasExc(NumGasExcTerms+1)=.true.
          MPI_master_only write(stdout, '(6x,L1,2x,A,2x,A,I2)')
     &         wrtdiabioGasExc(iflux), 
     &         'write equation terms ',
     &         'GasExc term of index (his)', iflux
        enddo      
#  endif
! === === ===
# ifdef AVERAGES
!         wrtdiabio_avg=.true.
!        write(*,*)'=============='
      elseif (keyword(1:kwlen).eq.'diagbioFlux_average_fields') then
        call cancel_kwd (keyword(1:kwlen), ierr)
!     ==
        read(input,*,err=95) (wrtdiabioFlux_avg(iflux),
     &       iflux=1,NumFluxTerms)
        do iflux=1,NumFluxTerms
          if (wrtdiabioFlux_avg(iflux)) 
     &         wrtdiabioFlux_avg(NumFluxTerms+1)=.true.
          MPI_master_only write(stdout, '(6x,L1,2x,A,2x,A,I2)')
     &         wrtdiabioFlux_avg(iflux), 
     &         'write equation terms ',
     &         'Flux term of index (avg)', iflux
        enddo
!     ==
      elseif (keyword(1:kwlen).eq.'diagbioVSink_average_fields') then
        call cancel_kwd (keyword(1:kwlen), ierr)
        read(input,*,err=95) (wrtdiabioVSink_avg(iflux),
     &                              iflux=1,NumVSinkTerms)
        do iflux=1,NumVSinkTerms
          if (wrtdiabioVSink_avg(iflux)) 
     &         wrtdiabioVSink_avg(NumVSinkTerms+1)=.true.
          MPI_master_only write(stdout, '(6x,L1,2x,A,2x,A,I2)')
     &         wrtdiabioVSink_avg(iflux), 
     &         'write equation terms ',
     &         'VSink term of index (avg)', iflux
        enddo
!     == 
#  if (defined BIO_NChlPZD && defined OXYGEN) || defined BIO_BioEBUS
      elseif (keyword(1:kwlen).eq.'diagbioGasExc_average_fields') then
        call cancel_kwd (keyword(1:kwlen), ierr)   
        read(input,*,err=95) (wrtdiabioGasExc_avg(iflux),
     &                                  iflux=1,NumGasExcTerms)
        do iflux=1,NumGasExcTerms
          if (wrtdiabioGasExc_avg(iflux)) 
     &         wrtdiabioGasExc_avg(NumGasExcTerms+1)=.true.
          MPI_master_only write(stdout, '(6x,L1,2x,A,2x,A,I2)')
     &         wrtdiabioGasExc_avg(iflux), 
     &         'write equation terms ',
     &         'GasExc term of index (avg)', iflux
        enddo      
#  endif
# endif  /* AVERAGES */
#endif /* DIAGNOSTICS_BIO */

#ifdef STATIONS
!
! Switches for fields to be saved into stations output file.
!
      elseif (keyword(1:kwlen).eq.'station_fields') then
        call cancel_kwd (keyword(1:kwlen), ierr)
# ifdef AGRIF
        if (Agrif_Root()) then
# endif
          read(input,*,err=95) wrtsta(indxstaGrd),
     &       wrtsta(indxstaTemp), wrtsta(indxstaSalt),
     &       wrtsta(indxstaRho), wrtsta(indxstaVel)
           MPI_master_only write(stdout,'(/1x,A,5(/6x,l1,2x,A))')
     &      'Fields to be saved in stations output  (T/F)'
     &     , wrtsta(indxstaGrd),   'write Grid location variables'
     &     , wrtsta(indxstaTemp),  'write temperature.'
     &     , wrtsta(indxstaSalt),  'write salinity.'
     &     , wrtsta(indxstaRho),   'write density.'
     &     , wrtsta(indxstaVel),   'write mean station velocity'
# ifdef AGRIF
        endif
# endif
#endif /* STATIONS */

!
! Boussinesq Approximation mean density.
!
      elseif (keyword(1:kwlen).eq.'rho0') then
        call cancel_kwd (keyword(1:kwlen), ierr)
        read(input,*,err=95) rho0
        MPI_master_only write(stdout,'(F10.4,2x,A,1x,A)')
     &        rho0, 'rho0     Boussinesq approximation',
     &                           'mean density, kg/m3.'
#if defined UV_VIS2 || defined UV_VIS4
!
! Horizontal viscosity coefficients.
!
      elseif (keyword(1:kwlen).eq.'lateral_visc') then
        call cancel_kwd (keyword(1:kwlen), ierr)
        read(input,*,err=95) visc2, visc4
#endif
#if defined UV_VIS2 || defined SPONGE_VIS2
        MPI_master_only write(stdout,9) visc2
   9    format(1pe10.3,2x,'visc2    Horizontal Laplacian ',
     &       'mixing coefficient [m2/s]',/,32x,'for momentum.')
#endif
#ifdef UV_VIS4
        MPI_master_only write(stdout,10) visc4
  10    format(1pe10.3,2x,'visc4    Horizontal biharmonic ',
     &       'mixing coefficient [m4/s]',/,32x,'for momentum.')
#endif
!
! Bottom drag coefficients.
!
      elseif (keyword(1:kwlen).eq.'bottom_drag') then
        call cancel_kwd (keyword(1:kwlen), ierr)
        read(input,*,err=95) rdrg, rdrg2, Zobt, Cdb_min, Cdb_max
        MPI_master_only write(stdout,'(5(1pe10.3,2x,A/))')
     &     rdrg, 'rdrg     Linear bottom drag coefficient (m/si).',
     &    rdrg2, 'rdrg2    Quadratic bottom drag coefficient.',
     &     Zobt, 'Zobt     Bottom roughness for logarithmic law (m).',
     &  Cdb_min, 'Cdb_min  Minimum bottom drag coefficient.',
     &  Cdb_max, 'Cdb_max  Maximum bottom drag coefficient.'
!
! Lateral boundary slipperness.
!
      elseif (keyword(1:kwlen).eq.'gamma2') then
        call cancel_kwd (keyword(1:kwlen), ierr)
        read(input,*,err=95) gamma2
        MPI_master_only write(stdout,'(f10.2,2x,A,1x,A)')
     &     gamma2, 'gamma2   Slipperiness parameter:',
     &                     'free-slip +1, or no-slip -1.'
#ifdef SOLVE3D
# if defined TS_DIF2 || defined SPONGE_DIF2
!
! Horizontal Laplacian mixing coefficients for tracers.
!
      elseif (keyword(1:kwlen).eq.'tracer_diff2') then
        call cancel_kwd (keyword(1:kwlen), ierr)
#  ifdef TRACERS
        read(input,*,err=95) (tnu2(itrc),itrc=1,NT)
        do itrc=1,NT
          MPI_master_only write(stdout,7) tnu2(itrc), itrc, itrc
   7      format(1pe10.3,'  tnu2(',i2,')  Horizontal Laplacian '
     &     ,'mixing coefficient (m2/s)',/,32x,'for tracer ',i2,'.')
        enddo
#  endif /* TRACERS */
# endif
# ifdef TS_DIF4
!
! Horizontal biharmonic mixing coefficients for tracer.
!
      elseif (keyword(1:kwlen).eq.'tracer_diff4') then
        call cancel_kwd (keyword(1:kwlen), ierr)
#  ifdef TRACERS
        read(input,*,err=95) (tnu4(itrc),itrc=1,NT)
        do itrc=1,NT
          MPI_master_only write(stdout,8) tnu4(itrc), itrc, itrc
   8      format(1pe10.3,'  tnu4(',i2,')  Horizontal biharmonic'
     &    ,' mixing coefficient [m4/s]',/,32x,'for tracer ',i2,'.')
        enddo
#  endif /* TRACERS */

# endif
# if !defined LMD_MIXING && !defined BVF_MIXING && \
     !defined GLS_MIXING
!
! Background vertical viscosity and mixing coefficients for tracers.
!
      elseif (keyword(1:kwlen).eq.'vertical_mixing') then
        call cancel_kwd (keyword(1:kwlen), ierr)
#  ifdef TRACERS
         read(input,*,err=95) Akv_bak,(Akt_bak(itrc),itrc=1,NT)
#  endif /* TRACERS */
        MPI_master_only write(stdout,'(1pe10.3,2x,A,1x,A)')
     &      Akv_bak, 'Akv_bak    Background vertical viscosity',
     &                                     'coefficient, m2/s.'
#  ifdef TRACERS
        do itrc=1,NT
          MPI_master_only write(stdout,
     &           '(1pe10.3,2x,A,I2,A,1x,A/32x,A,I2,A)')
     &            Akt_bak(itrc), 'Akt_bak(', itrc, ')',
     &           'Background vertical mixing coefficient, m2/s,',
     &                                  'for tracer ', itrc, '.' 
        enddo
#  endif /* TRACERS */
# endif
# ifdef BODYFORCE
      elseif (keyword(1:kwlen).eq.'bodyforce') then
        call cancel_kwd (keyword(1:kwlen), ierr)
        read(input,*,err=95) levsfrc,levbfrc
        if (levsfrc.lt.1 .or. levsfrc.gt.N) then
          MPI_master_only write(stdout,19) 'LEVSFRC = ',levsfrc
  19      format(' READ_INP - Illegal bodyforce level, ',A,i4)
          ierr=ierr+1
        endif
        if (levbfrc.lt.1 .or. levbfrc.gt.N) then
          MPI_master_only write(stdout,19) 'LEVBFRC = ',levbfrc
          ierr=ierr+1
        endif
        MPI_master_only write(stdout,20) levsfrc, levbfrc
  20    format(4x,i6,2x,'levsfrc     ',
     &           'Deepest level to apply surface stress as a ',
     &           'bodyforce.',/,
     &         4x,i6,2x,'levbfrc     ',
     &           'Shallowest level to apply bottom stress as a ',
     &           'bodyforce.')
# endif
#endif
#if  (defined SPONGE && !defined SPONGE_GRID) 
!
! Parameters for sponge layers
!
!  if SPONGE_GRID is defined, they are set generically
!  in set_nudgcof routine
!
      elseif (keyword(1:kwlen).eq.'sponge') then
        call cancel_kwd (keyword(1:kwlen), ierr)
        read(input,*,err=95) x_sponge
        MPI_master_only write(stdout,'(1pe10.2,2x,A,1x,A)')
     &     x_sponge,'x_sponge Thickness of sponge',
     &     'and/or nudging layer (m)'
!
#elif  (defined SPONGE && defined SPONGE_GRID) 
!
!  if SPONGE_GRID is defined, they are set generically
!  in set_nudgcof routine
!
      elseif (keyword(1:kwlen).eq.'sponge') then
         call cancel_kwd (keyword(1:kwlen), ierr)	
	     MPI_master_only write(stdout,'(/,1x,A,/,25x,A/)')
     &   'SPONGE_GRID is defined: x_sponge parameter in sponge/nudging',
     &   'layer is set generically in set_nudgcof.F routine'
#endif
!
#if  defined T_FRC_BRY     || defined M2_FRC_BRY    || \
     defined M3_FRC_BRY    || defined Z_FRC_BRY     || \
     defined TCLIMATOLOGY  || defined M2CLIMATOLOGY || \
     defined M3CLIMATOLOGY || defined ZCLIMATOLOGY
!
! Nudging parameters for OBC and nudging layers 
! (converted from [days] to [sec^-1]
!
      elseif (keyword(1:kwlen).eq.'nudg_cof') then
        call cancel_kwd (keyword(1:kwlen), ierr)
# if  defined AGRIF && !defined AGRIF_OBC_M2ORLANSKI && \
     !defined AGRIF_OBC_M3ORLANSKI && !defined AGRIF_OBC_TORLANSKI
        if (Agrif_Root()) then
# endif
          read(input,*,err=95) tauT_in,tauT_out,tauM_in,tauM_out
          tauT_in =1./(tauT_in *86400.)
          tauT_out=1./(tauT_out*86400.)
          tauM_in =1./(tauM_in *86400.)
          tauM_out=1./(tauM_out*86400.)
          MPI_master_only write(stdout,'(1pe10.3,2x,A)') 
     &        tauT_in,'tauT_in  Nudging coefficients [sec^-1]'
          MPI_master_only write(stdout,'(1pe10.3,2x,A)') 
     &       tauT_out,'tauT_out Nudging coefficients [sec^-1]'
          MPI_master_only write(stdout,'(1pe10.3,2x,A)') 
     &        tauM_in,'tauM_in  Nudging coefficients [sec^-1]'
          MPI_master_only write(stdout,'(1pe10.3,2x,A/)') 
     &       tauM_out,'tauM_out Nudging coefficients [sec^-1]'
# if defined AGRIF && !defined AGRIF_OBC_M2ORLANSKI && \
     !defined AGRIF_OBC_M3ORLANSKI && !defined AGRIF_OBC_TORLANSKI
        endif
# endif
#endif
#ifdef SOLVE3D
# if !defined NONLIN_EOS
!
! Parameters for linear equations of state.
!
      elseif (keyword(1:kwlen).eq.'lin_EOS_cff') then
        call cancel_kwd (keyword(1:kwlen), ierr)
        read(input,*,err=95) R0, T0, S0, Tcoef, Scoef
        MPI_master_only write(stdout,'(5(f10.4,2x,A,1x,A/))')
     &       T0, 'T0       Background value for potential',
     &                                     'temperature (Celsius).',
     &       S0, 'S0       Background salinity (PSU),', 'constant.',
     &       R0, 'R0       Background density (kg/m3) used in',
     &                                                'linear EOS.',
     &    Tcoef, 'Tcoef    Thermal expansion coefficient',
     &                                           '(kg/m3/Celsius).',
     &    Scoef, 'Scoef    Saline contraction coefficient',
     &                                                '(kg/m3/PSU).'
# endif
# ifdef SEDIMENT
!
! Sediments input file name for USGS sediment model.
!
      elseif (keyword(1:kwlen).eq.'sediments') then
        call cancel_kwd (keyword(1:kwlen), ierr)
        read(input,'(A)',err=95) sedname
        
        MPI_master_only write(stdout,
     &              '(/9x,A,2x,A)')
     &        'Sediment input file:',sedname
      elseif (keyword(1:kwlen).eq.'sediment_history_fields') then
        call cancel_kwd (keyword(1:kwlen), ierr)
        read(input,*,err=95) 
     &                   (wrthis(itrc), itrc=indxSed-1,indxSed+NST+1
#   ifdef SUSPLOAD
     &                                       +2*NST
#   endif
#   ifdef BEDLOAD
     &                                       +2*NST 
#   endif     
#   if defined MIXED_BED || defined COHESIVE_BED
     &                                       + 1
#   endif     
     &  ) 
#   ifdef AVERAGES
        do itrc=indxSed,indxSed+NST+1
#   ifdef SUSPLOAD
     &                                       +2*NST
#   endif
#   ifdef BEDLOAD
     &                                       +2*NST 
#   endif
         wrtavg(itrc)=wrthis(itrc)
        enddo
#   endif
        MPI_master_only write(stdout,'(3(/6x,L1,2x,A,1x,A))')
     &    wrthis(indxATHK), 'write act_thick ', 
     &                            'thickness of active bed layer.'
     &  , wrthis(indxBTHK), 'write bed_thick ', 
     &                            'thickness of sediment bed layer.'
     &  , wrthis(indxBPOR), 'write bed_poros ', 
     &                             'porosity of sediment bed layer.'
        do itrc=1,NST
            MPI_master_only write(stdout, '(6x,L1,2x,A,I2,A,I2,A)')
     &          wrthis(indxBFRA(itrc)), 'write bed_frac(',itrc,
     &                   ') Sediment fraction of index ',itrc,'.'
        enddo
#   ifdef SUSPLOAD
        do itrc=1,NST
            MPI_master_only write(stdout, '(6x,L1,2x,A,I2,A,I2,A)')
     &          wrthis(indxDFLX(itrc)), 'write deposition flux(',itrc,
     &                   ') for sediment of index ',itrc,'.'
        enddo
        do itrc=1,NST
            MPI_master_only write(stdout, '(6x,L1,2x,A,I2,A,I2,A)')
     &          wrthis(indxEFLX(itrc)), 'write erosion flux(',itrc,
     &                   ') for sediment of index ',itrc,'.'
        enddo
#   endif
#   ifdef BEDLOAD
        do itrc=1,NST
            MPI_master_only write(stdout, '(6x,L1,2x,A,I2,A,I2,A)')
     &          wrthis(indxBDLU(itrc)), 'write xi-bedload flux(',itrc,
     &                   ') for sediment of index ',itrc,'.'
        enddo
        do itrc=1,NST
            MPI_master_only write(stdout, '(6x,L1,2x,A,I2,A,I2,A)')
     &          wrthis(indxBDLU(itrc)), 'write eta-bedload flux(',itrc,
     &                   ') for sediment of index ',itrc,'.'
        enddo
#   endif
#   if defined MIXED_BED || defined COHESIVE_BED
        MPI_master_only write(stdout,'(6x,L1,2x,A,1x,A)')
     &      wrthis(indxBTCR), 'write bed_tau_crit ',
     &                            'critical stress of bed layer.'
#   endif
# elif defined MUSTANG
!
! Sediments input file name for Mustang.
!
      elseif (keyword(1:kwlen).eq.'sediments_mustang') then
        call cancel_kwd (keyword(1:kwlen), ierr)
        read(input,'(A)',err=95) sedname_subst
        read(input,'(A)',err=95) sedname_must
        lstr=lenstr(sedname_subst)
        MPI_master_only write(stdout,'(7x,A,10x,A)')
     &         'Sediment input file for Substance:' ,
     &          sedname_subst(1:lstr)
        lstr=lenstr(sedname_must)
        MPI_master_only write(stdout,'(7x,A,2x,A/)')
     &         'Sediment input file for Mustang (version):' ,
     &          sedname_must(1:lstr)

# endif /* SEDIMENT */
#endif
#if defined SUBSTANCE && !defined MUSTANG
      elseif (keyword(1:kwlen).eq.'substance') then
        call cancel_kwd (keyword(1:kwlen), ierr)
        read(input,'(A)',err=95) subsname
        lstr=lenstr(subsname)
        MPI_master_only write(stdout,'(7x,A,12x,A)')
     &         'Input file for Substance:' ,
     &          subsname(1:lstr)
#endif
#ifdef BBL
      elseif (keyword(1:kwlen).eq.'bbl_history_fields') then
        call cancel_kwd (keyword(1:kwlen), ierr)
        read(input,*,err=95) 
     &                    (wrthis(itrc), itrc=indxBBL,indxBBL+5) 

        MPI_master_only write(stdout,'(6(/6x,L1,2x,A,1x,A))')
     &    wrthis(indxAbed), 'write Abed ', 
     &                            'bed wave excursion amplitude.'
     &  , wrthis(indxHrip), 'write Hripple ', 
     &                                       'Bed ripple height.'
     &  , wrthis(indxLrip), 'write Lripple ', 
     &                                       'Bed ripple length.'
     &  , wrthis(indxZbnot), 'write Zbnot ', 
     &                              'Physical bottom roughness.'
     &  , wrthis(indxZbapp), 'write Zbapp ', 
     &                              'Apparent bottom roughness.'
     &  , wrthis(indxBostrw), 'write Bostrw ', 
     &                              'Wave-induced bottom stress.'

#endif /* BBL */

#ifdef MRL_WCI
!
! MRL wave-current interaction variables history fields
!
        elseif (keyword(1:kwlen).eq.'wci_history_fields') then
          call cancel_kwd (keyword(1:kwlen), ierr)
          read(input,*,err=95) wrthis(indxSUP),  wrthis(indxUST2D)
     &                                        ,  wrthis(indxVST2D)
# ifdef SOLVE3D
     &         , wrthis(indxUST), wrthis(indxVST), wrthis(indxwST)
     &         , wrthis(indxAKB), wrthis(indxAKW), wrthis(indxKVF)
     &                        , wrthis(indxCALP), wrthis(indxKAPS)
# endif
        MPI_master_only write(stdout,'(/1x,A,3(/6x,l1,2x,A))')
     &    'Fields to be saved in MRL-WCI 2D history  (T/F)',
     &    wrthis(indxSUP),
     &    'write SUP    quasi-static sea-level [m]',
     &    wrthis(indxUST2D),
     &    'write UST2D  depth-averaged  XI-Stokes velocity [m/s]',
     &    wrthis(indxVST2D),
     &    'write VST2D  depth-averaged ETA-Stokes velocity [m/s]'
# ifdef SOLVE3D
        MPI_master_only write(stdout,'(/1x,A,8(/6x,l1,2x,A))')
     &    'Fields to be saved in MRL-WCI 3D history  (T/F)',
     &    wrthis(indxUST),
     &    'write UST  3D  XI-Stokes velocity [m/s]',
     &    wrthis(indxVST),
     &    'write VST  3D ETA-Stokes velocity [m/s]',
     &    wrthis(indxWST),
     &    'write WST  3D vertical Stokes velocity [m/s]',
     &    wrthis(indxAkb),
     &    'write Akb  eddy viscosity due to wave breaking [m2/s]',
     &    wrthis(indxAkw),
     &    'write Akw  eddy diffusivity due to primary waves [m2/s]',
     &    wrthis(indxKVF),
     &    'write KVF  vertical vortex force [m/s2]',
     &    wrthis(indxCALP),
     &    'write CALP surface pressure correction [m]',
     &    wrthis(indxKAPS),
     &    'write KAPS surface Bernoulli head [m]'
# endif
# ifdef AVERAGES
!
! MRL wave-current interaction variables average fields
!
        elseif (keyword(1:kwlen).eq.'wci_average_fields') then
          call cancel_kwd (keyword(1:kwlen), ierr)
          read(input,*,err=95) wrtavg(indxSUP),  wrtavg(indxUST2D)
     &                                        ,  wrtavg(indxVST2D)
#  ifdef SOLVE3D
     &         , wrtavg(indxUST), wrtavg(indxVST), wrtavg(indxwST)
     &         , wrtavg(indxAKB), wrtavg(indxAKW), wrtavg(indxKVF)
     &                        , wrtavg(indxCALP), wrtavg(indxKAPS)
#  endif
        MPI_master_only write(stdout,'(/1x,A,3(/6x,l1,2x,A))')
     &    'Fields to be saved in MRL-WCI 2D average  (T/F)',
     &    wrtavg(indxSUP),
     &    'write SUP    quasi-static sea-level [m]',
     &    wrtavg(indxUST2D),
     &    'write UST2D  depth-averaged  XI-Stokes velocity [m/s]',
     &    wrtavg(indxVST2D),
     &    'write VST2D  depth-averaged ETA-Stokes velocity [m/s]'
#  ifdef SOLVE3D
        MPI_master_only write(stdout,'(/1x,A,8(/6x,l1,2x,A))')
     &    'Fields to be saved in MRL-WCI 3D average  (T/F)',
     &    wrtavg(indxUST),
     &    'write UST  3D  XI-Stokes velocity [m/s]',
     &    wrtavg(indxVST),
     &    'write VST  3D ETA-Stokes velocity [m/s]',
     &    wrtavg(indxWST),
     &    'write WST  3D vertical Stokes velocity [m/s]',
     &    wrtavg(indxAkb),
     &    'write Akb  eddy viscosity due to wave breaking [m2/s]',
     &    wrtavg(indxAkw),
     &    'write Akw  eddy diffusivity due to primary waves [m2/s]',
     &    wrtavg(indxKVF),
     &    'write KVF  vertical vortex force [m/s2]',
     &    wrtavg(indxCALP),
     &    'write CALP surface pressure correction [m]',
     &    wrtavg(indxKAPS),
     &    'write KAPS surface Bernoulli head [m]'
#  endif
# endif /* AVERAGES */
#endif /* MRL_WCI */

#if defined MRL_WCI || defined OW_COUPLING
!
! Wave history fields
!
      elseif (keyword(1:kwlen).eq.'wave_history_fields') then
        call cancel_kwd (keyword(1:kwlen), ierr)
        read(input,*,err=95) (wrthis(i), i=indxHRM,indxHRM+8)
        MPI_master_only write(stdout,'(/1x,A,9(/6x,l1,2x,A))')
     &    'Fields to be saved in wave history  (T/F)',
     &    wrthis(indxHRM), 'write HRM  wave height (m)',
     &    wrthis(indxFRQ), 'write FRQ  intrinsic frequency (rad/s)',
     &    wrthis(indxWAC), 'write WAC  primary wave action (m3/s)',
     &    wrthis(indxWKX), 'write WKX  wavenumber  xi-dir component',
     &    wrthis(indxWKE), 'write WKE  wavenumber eta-dir component',
     &    wrthis(indxEPB), 'write EPB  breaking dissipation (m3/s3)',
     &    wrthis(indxEPD), 'write EPD  frictional dissipation (m3/s3)',
     &    wrthis(indxWAR), 'write WAR  roller wave action (m3/s)',
     &    wrthis(indxEPR), 
     &                   'write EPR  roller dissipation (m3/s3): [0,1]'
# ifdef AVERAGES
!
! Wave average fields
! 
      elseif (keyword(1:kwlen).eq.'wave_average_fields') then
        call cancel_kwd (keyword(1:kwlen), ierr)
        read(input,*,err=95) (wrtavg(i), i=indxHRM,indxHRM+8)
        MPI_master_only write(stdout,'(/1x,A,9(/6x,l1,2x,A))')
     &    'Fields to be saved in wave average  (T/F)',
     &    wrtavg(indxHRM), 'write HRM  wave height (m)',
     &    wrtavg(indxFRQ), 'write FRQ  intrinsic frequency (rad/s)',
     &    wrtavg(indxWAC), 'write WAC  primary wave action (m3/s)',
     &    wrtavg(indxWKX), 'write WKX  wavenumber  xi-dir component',
     &    wrtavg(indxWKE), 'write WKE  wavenumber eta-dir component',
     &    wrtavg(indxEPB), 'write EPB  breaking dissipation (m3/s3)',
     &    wrtavg(indxEPD), 'write EPD  frictional dissipation (m3/s3)',
     &    wrtavg(indxWAR), 'write WAR  roller wave action (m3/s)',
     &    wrtavg(indxEPR),
     &                   'write EPR  roller dissipation (m3/s3): [0,1]'
# endif
#endif /* WKB_WWAVE || OW_COUPLING || WAVE_OFFLINE */

#if defined WKB_WWAVE
!
! WKB primary waves and empirical breaking model parameters
!
      elseif (keyword(1:kwlen).eq.'wkb_wwave') then
        call cancel_kwd (keyword(1:kwlen), ierr)
        read(input,*,err=95) wkb_amp,wkb_ang,wkb_prd,wkb_tide,wkb_btg,
     &                       wkb_gam
        MPI_master_only write(stdout,
     &             '(/1x,A,4(/3x,f10.4,2x,A),/3x,A,2(/3x,f10.4,2x,A))')
     &    'Primary waves and breaking parameters for WKB wave model',
     &   wkb_amp,   'wkb_amp    offshore wave amplitude [m]',
     &   wkb_ang,   'wkb_ang    offshore wave angle [deg]',
     &   wkb_prd,   'wkb_prd    offshore wave period [s]',
     &   wkb_tide,  'wkb_tide   constant offshore water level [m]',
# ifdef WAVE_BREAK_TG86
     &  'breaking model (Thornton & Guza, 1983a) is specifield.',
# elif defined WAVE_BREAK_TG86A
     &  'breaking model (Thornton & Guza, 1983b) is specifield.',
# elif defined WAVE_BREAK_CT93
     &  'breaking model (Church & Thornton, 1993) is specifield.',
# else
     &  'breaking model (Church & Thornton, 1993) is default.',
# endif
     &   wkb_btg,   'wkb_btg    B parameter',
     &   wkb_gam,   'wkb_gam    gamma paramemer (Hrms/h ratio)'

# ifdef WAVE_ROLLER
!
! Svendsen's surface roller model parameters
!
      elseif (keyword(1:kwlen).eq.'wkb_roller') then
        call cancel_kwd (keyword(1:kwlen), ierr)
        read(input,*,err=95) wkb_rsb, wkb_roller
        MPI_master_only write(stdout,'(/1x,A,2(/3x,f10.4,2x,A))')
     &  'Svenden (1984) surface roller model parameters.',
     &   wkb_rsb,   'wkb_rsb     sin(beta) roller dissipation',
     &   wkb_roller,'wkb_roller  breaking contrib to roller: [0,1]'
# endif
#endif /* WKB_WWAVE */

#ifdef ANA_PSOURCE
!
! Set-up point Sources/Sink number (Nsrc), direction (Dsrc), I- and
! J-grid locations (Isrc,Jsrc), and logical switch for type of tracer
! to apply (Lsrc). Currently, the direction can be along XI-direction
! (Dsrc = 0) or along ETA-direction (Dsrc > 0).  The mass sources are
! located at U- or V-points so the grid locations should range from
! 1 =< Isrc =< L  and  1 =< Jsrc =< M.
!
# ifdef PSOURCE_NCFILE
      elseif (keyword(1:kwlen).eq.'psource_ncfile') then
        call cancel_kwd (keyword(1:kwlen), ierr)
! river runoff file name. Check its availability.
        read(input,'(A)',err=95) fname
        lstr=lenstr(fname)
        open (testunit, file=fname(1:lstr), status='old', err=97)
        close(testunit)
        qbarname=fname(1:lstr)
        MPI_master_only write(stdout,'(2x,A,2x,A)')
     &             'Runoff Data File:', qbarname(1:lstr)
        read(input,*,err=95) Nsrc
        MPI_master_only write(stdout,'(/6x,i6,2x,A,1x,A)')
     &                               Nsrc, 'Number of point sources'
        do is=1,Nsrc
#  ifdef PSOURCE_NCFILE_TS
          read(input,*,err=95) Isrc(is),Jsrc(is),Dsrc(is),qbardir(is),
     &                                    (Lsrc(is,itrc), itrc=1,NT)
#  else
          read(input,*,err=95) Isrc(is),Jsrc(is),Dsrc(is),qbardir(is),
     &                                     (Lsrc(is,itrc), itrc=1,NT),
#   ifdef TRACERS
     &                                     (Tsrc0(is,itrc), itrc=1,NT)
#   endif /* TRACERS */
#  endif /* PSOURCE_NCFILE_TS  */ 

        MPI_master_only write(stdout,'(3(/6x,i6,2x,A),(/6x,F6.0,2x,A))')
     &    Isrc(is), 'I point source indice'
     &  , Jsrc(is), 'J point source indice'
     &  , Dsrc(is), 'Orientation of point source flow'
!         MPI_master_only write(stdout,'(/6x,F6.0,2x,A)')
     &  , qbardir(is), 'Direction of point source flow'
#  ifdef MPI
        if (iminmpi.LE.Isrc(is) .AND. Isrc(is).LE.imaxmpi .AND.
     &       jminmpi.LE.Jsrc(is) .AND. Jsrc(is).LE.jmaxmpi) then
          Isrc_mpi(is,mynode)=Isrc(is)-iminmpi+1
          Jsrc_mpi(is,mynode)=Jsrc(is)-jminmpi+1
        else
          Isrc_mpi(is,mynode)=-1
          Jsrc_mpi(is,mynode)=-1
        endif
#  endif
#  ifdef TRACERS
        do itrc=1,NT
          MPI_master_only write(stdout,
     &         '(6x,L1,2x,A,I2,A,2x,A,I2,A)')
     &         Lsrc(is,itrc), 'write Lsrc(',
     &         itrc,')', 'Tracer of index', itrc,'.'
        enddo
#  endif /* TRACERS */
#  ifndef PSOURCE_NCFILE_TS
#   ifdef TRACERS
        do itrc=1,NT
          MPI_master_only write(stdout,
     &         '(6x,1pe10.3,2x,A,I2,A,2x,A,I2,A)')
     &         Tsrc0(is,itrc), 'write Tsrc(',
     &         itrc,')', 'Tracer of index', itrc,'.'
        enddo
#   endif /* TRACERS */
#  endif
      enddo
# else   
!<- PSOURCE_NCFILE 
      elseif (keyword(1:kwlen).eq.'psource') then
        call cancel_kwd (keyword(1:kwlen), ierr)
        
        read(input,*,err=95) Nsrc
        MPI_master_only write(stdout,'(/6x,i6,2x,A,1x,A)')
     &                               Nsrc, 'Number of point sources'
        do is=1,Nsrc
          read(input,*,err=95) Isrc(is), Jsrc(is), Dsrc(is), Qbar(is)
#  if defined SOLVE3D && defined TRACERS
     &                                    ,(Lsrc(is,itrc), itrc=1,NT)
     &                                    ,(Tsrc0(is,itrc), itrc=1,NT)
#  endif
          MPI_master_only write(stdout,'(3(/6x,i6,2x,A))')
     &    Isrc(is), 'I point source indice'
     &  , Jsrc(is), 'J point source indice'
     &  , Dsrc(is), 'Direction of point source flow'
          MPI_master_only write(stdout,'(1pe10.3,2x,A)')
     &    Qbar(is), 'Total transport at point source'
#  ifdef MPI
          MPI_master_only write(*,*)'Isrc(is)=',Isrc(is)
          MPI_master_only write(*,*)'Jsrc(is)=',Jsrc(is)
          if (iminmpi.LE.Isrc(is) .AND. Isrc(is).LE.imaxmpi .AND.
     &        jminmpi.LE.Jsrc(is) .AND. Jsrc(is).LE.jmaxmpi) then
           Isrc_mpi(is,mynode)=Isrc(is)-iminmpi+1
           Jsrc_mpi(is,mynode)=Jsrc(is)-jminmpi+1
          else
           Isrc_mpi(is,mynode)=-1
           Jsrc_mpi(is,mynode)=-1
          endif
#  endif
#  if defined SOLVE3D && defined TRACERS
          do itrc=1,NT
            MPI_master_only write(stdout,
     &                     '(6x,L1,2x,A,I2,A,2x,A,I2,A)')
     &                      Lsrc(is,itrc), 'write Lsrc(',
     &                      itrc,')', 'Tracer of index', itrc,'.'
          enddo
          do itrc=1,NT
            MPI_master_only write(stdout,
     &                     '(6x,1pe10.3,2x,A,I2,A,2x,A,I2,A)')
     &                      Tsrc0(is,itrc), 'write Tsrc(',
     &                      itrc,')', 'Tracer of index', itrc,'.'
          enddo
#  endif
        enddo
# endif   /*  define PSOURCE_NCFILE  */
#endif /* ANA_PSOURCE */
!
!
! Hydrothermal forcing file name. Check its availability.
!
#if defined BHFLUX || ( defined BWFLUX && defined SALINITY)
      elseif (keyword(1:kwlen).eq.'bottom_forcing') then
        call cancel_kwd (keyword(1:kwlen), ierr)
        read(input,'(A)',err=95) fname
        lstr=lenstr(fname)
# if defined MPI && defined PARALLEL_FILES && defined MPI_NOLAND
        call insert_node (fname, lstr, mynode2, NNODES2, ierr)
# endif
# if defined MPI && defined PARALLEL_FILES && !defined MPI_NOLAND
        call insert_node (fname, lstr, mynode, NNODES, ierr)
# endif
        open (testunit, file=fname(1:lstr), status='old', err=97)
        close(testunit)
        btfname=fname(1:lstr)
        MPI_master_only write(stdout,'(2x,A,2x,A)')
     &       'Bottom forcing file:', fname(1:lstr)
#endif  /* BHFLUX || (SALINITY && BWFLUX )  */
!
! Online forcing with CFSR
!
#ifdef ONLINE   /* JG Online */
      elseif (keyword(1:kwlen).eq.'online') then
        call cancel_kwd (keyword(1:kwlen), ierr)
        read(input,*,err=95) yearnum, monthnum, recordsperday, yearend, 
     &                       monthend
        read(input,'(A)',err=95)  fname 
        lstr=lenstr(fname)
        pathbulk=fname(1:lstr)
        MPI_master_only write(stdout,
     &         '(7x,A,1x,I4,A,1x,I2,A)')
     &         'Online forcing: first forcing, year',yearnum,
     &         ', month', monthnum,'.'
        MPI_master_only write(stdout,
     &         '(7x,A,1x,I4,A,1x,I2,A)')
     &         'Online forcing: last forcing year',yearend,
     &         ', month', monthend,'.'
        MPI_master_only write(stdout,
     &         '(7x,A,A,1x,A4,1x,I2,A)')
     &         'Online forcing: datasets in ',fname(1:lstr),
     &         'with',recordsperday, ' records per day.'
#endif /* ONLINE */

      else
        MPI_master_only write(stdout,'(/3(1x,A)/)')
     &                  'WARNING: Unrecognized keyword:',
     &                   keyword(1:kwlen),' --> DISREGARDED.'
      endif
      if (keyword(1:kwlen) .eq. end_signal) goto 99
      goto 1
!
! Error while reading input parameters.
!
  95  write(stdout,'(/1x,4A/)') 'READ_INP ERROR while reading block',
     &                    ' with keyword ''', keyword(1:kwlen), '''.' 
      ierr=ierr+1
      goto 99 
  97  write(stdout,'(/1x,4A/)') 'READ_INP ERROR: Cannot find input ',
     &                                'file ''', fname(1:lstr), '''.'
#ifdef LOGFILE
  98  write(6,'(/1x,4A/)') 'READ_INP ERROR: Cannot open log file ',
     &                                'file   , croco.log', '''.'
#endif
      ierr=ierr+1
  99  close (input) 
!
! Check that all keywords were canceled
! Complain if some of them are left
!
      if (ierr.eq.0) then
        call check_kwds (ierr)
!
! Check CPP-switches for consistency. This operation is split into
! two stages because the first subroutine, "check_switches1", is
! generated by special program cppcheck (file cppcheck.F) by
! examining and documentation all available switches in cppdefs.h.
! This subroutine creates log of all switches defined in "cppdefs.h",
! as well as traps multiply defined global configurations (project
! switches, such as REGIONAL, etc).
! The second routine, "check_switches2" is hand written and it
! contains traps for mutually exclussive definition of all other
! CPP-switches (i.e. those which are NOT project selection switches,
! for example, it traps multiply defined vertical mixing schemes or
! lateral boundary conditions).
!
! Both codes are written in transparent mode: they assumed that error
! variable (ierr) is initialized at entry and they add 1 for each
! error discovered. 
!
        call check_srcs
        call check_switches1 (ierr)
        call check_switches2 (ierr)
      endif
      if (ierr.ne.0) then
        write(stdout,'(/1x,2A,I3,1x,A/)') 'READ_INP ERROR: ',
     & 'A total of', ierr, 'configuration errors discovered.'
        return
      endif
#ifdef MPI
!      call MPI_Barrier (MPI_COMM_WORLD, ierr)
#endif
      return
      end

c
