! $Id: init_scalars.F 1564 2014-06-24 17:39:21Z gcambon $
!
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

      subroutine init_scalars (ierr)
!
! Set initial values for  globally accessable (stored in common
! blocks) scalar variables. Typically this is associated with
! model utility switches, i/o contlol, time indices, global
! summation counters, etc; NONE of the physical parameters of
! the model is set here.
!
      implicit none
      integer ierr, i,j,itrc, lvar, lenstr
#if defined ONLINE
      integer blkvar_id
#endif /* ONLINE */
      character*20 nametrc, unitt
      character*60 vname1, vname3
      integer omp_get_num_threads
#include "param.h"
#include "scalars.h"
#include "mixing.h"
#include "ncscrum.h"
#include "diagnostics.h"
#ifdef DIAGNOSTICS_VRT
# include "diags_vrt.h"
#endif
#ifdef DIAGNOSTICS_EK
# include "diags_ek.h"
#endif
#ifdef DIAGNOSTICS_PV
# include "diags_pv.h"
#endif
#ifdef DIAGNOSTICS_EDDY
# include "diags_eddy.h"
#endif
#include "boundary.h"
#if defined ONLINE
# include "online.h"
#endif /* ONLINE */
#if defined TCLIMATOLOGY
# include "climat.h"
#endif /* TCLIMATOLOGY */
#if defined WKB_WWAVE
# include "wkb_wwave.h"
#endif

!      call system('uname -prs')

!
! Find out how many threads are created and check whether the number
! of threads exceeds maximum allowed, as well as check if the number
! of tiles is divisible by the number of threads, so the job can be
! evenly partitioned, complain and signal to terminate, if something
! is wrong.
!
C$OMP PARALLEL
C$OMP CRITICAL (isca_cr_rgn)
      numthreads=omp_get_num_threads()
C$OMP END CRITICAL (isca_cr_rgn)
C$OMP END PARALLEL
      MPI_master_only write(stdout,'(1x,A,3(1x,A,I3),A)') 'NUMBER',
     &    'OF THREADS:',numthreads,'BLOCKING:',NSUB_X,'x',NSUB_E,'.'

      if (numthreads.gt.NPP) then
        MPI_master_only write(stdout,'(/1x,A,I3/)')
     &    'ERROR: Requested number of threads exceeds NPP =', NPP
        ierr=ierr+1                                      !--> ERROR
      elseif (mod(NSUB_X*NSUB_E,numthreads).ne.0) then
        MPI_master_only write(stdout,
     &                '(/1x,A,1x,A,I3,4x,A,I3,4x,A,I4,A)') 'ERROR:',
     &                'wrong choice of numthreads =', numthreads,
     &                'NSUB_X =', NSUB_X, 'NSUB_E =', NSUB_E, '.'
        ierr=ierr+1                                      !--> ERROR
      endif
!
! Set evolving time indices to their default initial values.
!
      time=0.
      tdays=0.
      PREDICTOR_2D_STEP=.FALSE.
      iic=0
      kstp=1
      krhs=1
      knew=1

      ntstart=1
#ifdef SOLVE3D
      nstp=1
      nrhs=1
      nnew=1
#endif
      nfast=1

#ifdef FLOATS
      nfp1=0     ! set time step indices for
      nf=3       ! 5th order AB4AM4 corrected
      nfm1=2     ! scheme
      nfm2=1
      nfm3=0
#endif
#ifdef WKB_WWAVE
      wstp=1
      wnew=1
#endif
!
! Reset intra-thread communication flags and counters.
!
      synchro_flag=.true.
      first_time=0
      may_day_flag=0
!
! Reset process ID and CPU_time clocks (needed by timers).
!
      do j=0,NPP
        do i=0,31
          proc(i,j)=0
          CPU_time(i,j)=0.E0
        enddo
      enddo
      trd_count=0
!
! Reset record counters for averages/history output.
!
      nrecrst=0
      nrechis=0
#ifdef AVERAGES
      nrecavg=0
#endif
#ifdef DIAGNOSTICS_TS
      nrecdia=0
# ifdef AVERAGES
      nrecdia_avg=0
# endif
#endif
#ifdef DIAGNOSTICS_UV
      nrecdiaM=0
# ifdef AVERAGES
      nrecdiaM_avg=0
# endif
#endif
#ifdef DIAGNOSTICS_BIO
      nrecdiabio=0
# ifdef AVERAGES
      nrecdiabio_avg=0
# endif
#endif

!
! Reset global sums and counters needed by volume/energy
! diagnostics and volume conservation constraint.
!
      tile_count=0
      bc_count=0
      avgke=0.
      avgpe=0.
      avgkp=0.
      volume=0.

      hmin=+1.E+20      ! Set these
      hmax=-1.E+20      ! to large,
      grdmin=+1.E+20    ! positive or
      grdmax=-1.E+20    ! negative
      Cu_min=+1.E+20    ! unrealistic
      Cu_max=-1.E+20    ! values
#ifdef SPHERICAL
      lonmin=+1.E+20
      lonmax=-1.E+20
      latmin=+1.E+20
      latmax=-1.E+20
#endif

      bc_crss=QuadZero
#ifdef OBC_VOLCONS
      bc_flux=QuadZero
      ubar_xs=QuadZero
#endif
#ifdef SOLVE3D
      rx0=-1.E+20
      rx1=-1.E+20
#endif
#ifdef BIOLOGY
      bio_count=0
      do i=0,2*NT+1
        global_sum(i)=QuadZero
      enddo
#endif
!
! Initialize netCDF files IDs to closed status.
!
      ncidrst=-1
      ncidhis=-1
#ifdef AVERAGES
      ncidavg=-1
#endif
      ncidfrc=-1
      ncidbulk=-1
      ncidclm=-1
      ncidqbar=-1
#if (defined WAVE_OFFLINE && defined MUSTANG)
      ncidwave=-1
#endif
      ncidbtf=-1
#ifdef DIAGNOSTICS_TS
      nciddia=-1
# ifdef AVERAGES
      nciddia_avg=-1
# endif
#endif
#ifdef DIAGNOSTICS_UV
      nciddiaM=-1
# ifdef AVERAGES
      nciddiaM_avg=-1
# endif
#endif
#ifdef DIAGNOSTICS_VRT
      nciddiags_vrt=-1
      nrecdiags_vrt=0
# ifdef AVERAGES
      nciddiags_vrt_avg=-1
      nrecdiags_vrt_avg=0
# endif
#endif
#ifdef DIAGNOSTICS_EK
      nciddiags_ek=-1
      nrecdiags_ek=0
# ifdef AVERAGES
      nciddiags_ek_avg=-1
      nrecdiags_ek_avg=0
# endif
#endif
#ifdef DIAGNOSTICS_PV
      nciddiags_pv=-1
      nrecdiags_pv=0
# ifdef AVERAGES
      nciddiags_pv_avg=-1
      nrecdiags_pv_avg=0
# endif
#endif
#if defined DIAGNOSTICS_EDDY && ! defined XIOS
      nciddiags_eddy=-1
      nrecdiags_eddy=0
# ifdef AVERAGES
      nciddiags_eddy_avg=-1
      nrecdiags_eddy_avg=0
# endif
#endif
#if defined OUTPUTS_SURFACE && ! defined XIOS
      ncidsurf=-1
      nrecsurf=0
#ifdef AVERAGES
      ncidsurf_avg=-1
      nrecsurf_avg=0
# endif
#endif
#ifdef DIAGNOSTICS_BIO
      nciddiabio=-1
# ifdef AVERAGES
      nciddiabio_avg=-1
# endif
#endif
#ifndef ANA_BRY
       bry_id=-1
#endif
#if defined WKB_WWAVE && !defined ANA_BRY_WKB
       brywkb_id=-1
#endif
      call get_date (date_str)

#ifdef SOLVE3D
# ifdef TS_MIX_ISO_FILT
      csmooth=dt/(86400.)
      ismooth=int(1./csmooth)
# endif
#endif
!
!---------------------------------------------------------------
! Define names of variables in NetCDF output files.
! Inner dimension is for variable type and outer is for variable
! attributes (name, long-name, units, field).
!---------------------------------------------------------------
!
!vname(5,indx?)  => standard name (for all)
!vname(6,indx?)  => coordinate (for 2D grid fields)
!vname(7,indx?)  => axis (for variable dimension, Z-axis and time)
!vname(8,indx?)  => _FillValue
!vname(9,indx?)  => positive for vertical axis
!vname(10,indx?) => time origin
!vname(11,indx?) => calendar for time
!vname(12,indx?) => ?
!vname(13,indx?) => ?
!vname(14,indx?) => ?

! Initialisation at one blanck space
      vname(:,:)='  '


!#define CR
CR      write(stdout,'(1x,A,I3)') 'indxTime =',indxTime
#ifdef MILES
      vname(1,indxTime)='ocean_time'
#else
      vname(1,indxTime)='scrum_time'
#endif
      vname(2,indxTime)='time since initialization'
#ifdef USE_CALENDAR
      vname(3,indxTime)='seconds since 1900-01-01'
#else
      vname(3,indxTime)='second'
#endif
      vname(4,indxTime)='time, scalar, series'
      vname(5,indxTime)='time'
      vname(6,indxTime)='    '
      vname(7,indxTime)='T'
      vname(8,indxTime)='.NO.'
      vname(9,indxTime)=' '
#ifdef START_DATE
      vname(10,indxTime)=start_date
!     vname(11,indxTime)='Gregorian'
#endif

CR      write(stdout,'(1x,A,I3)') 'indxTime2 =',indxTime2
      vname(1,indxTime2)='time'
      vname(2,indxTime2)='time since initialization'
#ifdef USE_CALENDAR
      vname(3,indxTime2)='seconds since 1900-01-01'
#else
      vname(3,indxTime2)='second'
#endif
      vname(4,indxTime2)='time, scalar, series'
      vname(5,indxTime2)='time'
      vname(6,indxTime2)='  '
      vname(7,indxTime2)='T '
      vname(8,indxTime2)='.NO.'
      vname(9,indxTime2)=' '
#ifdef START_DATE
      vname(10,indxTime2)=start_date
!     vname(11,indxTime2)='Gregorian'
#endif

CR      write(stdout,'(1x,A,I3)') 'indxZ =',indxZ
      vname(1,indxZ)='zeta                                        '
      vname(2,indxZ)='free-surface                                '
      vname(3,indxZ)='meter                                       '
      vname(4,indxZ)='free-surface, scalar, series                '
      vname(5,indxZ)='sea_surface_height                          '
#ifdef SPHERICAL
      vname(6,indxZ)='lat_rho lon_rho                             '
#else
      vname(6,indxZ)='x_rho y_rho                                 '
#endif
      vname(7,indxZ)='                                            '

CR      write(stdout,'(1x,A,I3)') 'indxUb =',indxUb
      vname(1,indxUb)='ubar                                       '
      vname(2,indxUb)='vertically integrated u-momentum component '
      vname(3,indxUb)='meter second-1                             '
      vname(4,indxUb)='ubar-velocity, scalar, series              '
      vname(5,indxUb)='barotropic_sea_water_x_'
     &                                  // 'velocity_at_u_location'
#ifdef SPHERICAL
      vname(6,indxUb)='lat_u lon_u                                '
!#else
!      vname(6,indxUb)='x_u y_u                                    '
#endif
      vname(7,indxUb)='                                           '

CR      write(stdout,'(1x,A,I3)') 'indxVb = ',indxVb
      vname(1,indxVb)='vbar                                       '
      vname(2,indxVb)='vertically integrated v-momentum component '
      vname(3,indxVb)='meter second-1                             '
      vname(4,indxVb)='vbar-velocity, scalar, series              '
      vname(5,indxVb)='barotropic_sea_water_y_'
     &                                  // 'velocity_at_v_location'
#ifdef SPHERICAL
      vname(6,indxVb)='lat_v lon_v                                '
!#else
!      vname(6,indxVb)='x_v y_v                                    '
#endif
      vname(7,indxVb)='                                           '
      vname(11,indxVb)=' '

#ifdef MORPHODYN
CR      write(stdout,'(1x,A,I3)') 'indxHm =',indxHm
      vname(1,indxHm)='hmorph                                     '
      vname(2,indxHm)='moving bathymetry                          '
      vname(3,indxHm)='meter                                      '
      vname(4,indxHm)='moving_bathymetry , scalar, series         '
      vname(5,indxHm)='moving_bathymetry                          '
# ifdef SPHERICAL
      vname(6,indxHm)='lat_rho lon_rho                          '
# else
      vname(6,indxHm)='x_rho y_rho                                '
# endif
      vname(7,indxHm)='                                           '
#endif

CR      write(stdout,'(1x,A,I3)') 'indxBostr =',indxBostr
      vname(1,indxBostr)='bostr                                   '
      vname(2,indxBostr)='Kinematic bottom stress                 '
      vname(3,indxBostr)='N/m2                                    '
      vname(4,indxBostr)='                                        '
      vname(5,indxBostr)='                                        '
#ifdef SPHERICAL
      vname(6,indxBostr)='lat_rho lon_rho                         '
#else
      vname(6,indxBostr)='x_rho y_rho                             '
#endif
      vname(7,indxBostr)='                                        '


CR      write(stdout,'(1x,A,I3)') 'indxBustr =',indxBustr
      vname(1,indxBustr)='bustr                                   '
      vname(2,indxBustr)='Kinematic u bottom stress component     '
      vname(3,indxBustr)='N/m2                                    '
      vname(4,indxBustr)='                                        '
      vname(5,indxBustr)='                                        '
#  ifdef SPHERICAL
      vname(6,indxBustr)='lat_u lon_u                             '
!#  else
!      vname(6,indxBustr)='x_rho y_rho                            '
#  endif
      vname(7,indxBustr)='                                        '

CR      write(stdout,'(1x,A,I3)') 'indxBvstr =',indxBvstr
      vname(1,indxBvstr)='bvstr                                   '
      vname(2,indxBvstr)='Kinematic v bottom stress component     '
      vname(3,indxBvstr)='N/m2                                    '
      vname(4,indxBvstr)='                                        '
      vname(5,indxBvstr)='                                        '
#  ifdef SPHERICAL
      vname(6,indxBvstr)='lat_v lon_v                             '
!#  else
!      vname(6,indxBvstr)='x_rho y_rho                            '
#  endif
      vname(7,indxBvstr)='                                        '


CR      write(stdout,'(1x,A,I3)') 'indxWstr=',indxWstr
      vname(1,indxWstr)='wstr                                     '
      vname(2,indxWstr)='Kinematic wind stress                    '
      vname(3,indxWstr)='N/m2                                     '
      vname(4,indxWstr)='                                         '
      vname(5,indxWstr)='magnitude_of_surface_downward_stress     '
#ifdef SPHERICAL
      vname(6,indxWstr)='lat_rho lon_rho                          '
#else
      vname(6,indxWstr)='x_rho y_rho                              '
#endif
      vname(7,indxWstr)='                                         '

CR      write(stdout,'(1x,A,I3)') 'indxUWstr =',indxUWstr
      vname(1,indxUWstr)='sustr                                   '
      vname(2,indxUWstr)='Kinematic u wind stress component       '
      vname(3,indxUWstr)='N/m2                                    '
      vname(4,indxUWstr)='                                        '
      vname(5,indxUWstr)='surface_downward_eastward_stress        '
#ifdef SPHERICAL
      vname(6,indxUWstr)='lat_u lon_u                             '
!#else
!      vname(6,indxUWstr)='x_u y_u                                 '
#endif
      vname(7,indxUWstr)='                                        '
      vname(8,indxUWstr)='                                        '

CR      write(stdout,'(1x,A,I3)') 'indxVWstr =',indxVWstr
      vname(1,indxVWstr)='svstr                                   '
      vname(2,indxVWstr)='Kinematic v wind stress component       '
      vname(3,indxVWstr)='N/m2                                    '
      vname(4,indxVWstr)='                                        '
      vname(5,indxVWstr)='surface_downward_northward_stress       '
#ifdef SPHERICAL
      vname(6,indxVWstr)='lat_v lon_v                             '
!#else
!      vname(6,indxVWstr)='x_v y_v                                 '
#endif
      vname(7,indxVWstr)='                                        '

#ifdef SOLVE3D
CR      write(stdout,'(1x,A,I3)') 'indxU = ',indxU
      vname(1,indxU)='u                                           '
      vname(2,indxU)='u-momentum component                        '
      vname(3,indxU)='meter second-1                              '
      vname(4,indxU)='u-velocity, scalar, series                  '
      vname(5,indxU)='sea_water_x_velocity_at_u_location          '
#ifdef SPHERICAL
      vname(6,indxU)='lat_u lon_u                                 '
!#else
!      vname(6,indxU)='x_u y_u                                     '
#endif
      vname(7,indxU)='                                            '

CR      write(stdout,'(1x,A,I3)') 'indxV = ',indxV
      vname(1,indxV)='v                                           '
      vname(2,indxV)='v-momentum component                        '
      vname(3,indxV)='meter second-1                              '
      vname(4,indxV)='v-velocity, scalar, series                  '
      vname(5,indxV)='sea_water_y_velocity_at_v_location          '
#ifdef SPHERICAL
      vname(6,indxV)='lat_v lon_v                                 '
!#else
!      vname(6,indxV)='x_v y_v                                     '
#endif
      vname(7,indxV)='                                            '
#ifdef TEMPERATURE
CR      write(stdout,'(1x,A,I3)') 'indxT = ',indxT
      vname(1,indxT)='temp                                        '
      vname(2,indxT)='potential temperature                       '
      vname(3,indxT)='Celsius                                     '
      vname(4,indxT)='temperature, scalar, series                 '
      vname(5,indxT)='sea_water_potential_temperature             '
#ifdef SPHERICAL
      vname(6,indxT)='lat_rho lon_rho                             '
#else
      vname(6,indxT)='x_rho y_rho                                 '
#endif
      vname(7,indxT)='                                            '
#endif /* TEMPERATURE */

# ifdef SALINITY
CR      write(stdout,'(1x,A,I3)') 'indxS = ',indxS
      vname(1,indxS)='salt                                        '
      vname(2,indxS)='salinity                                    '
      vname(3,indxS)='PSU                                         '
      vname(4,indxS)='salinity, scalar, series                    '
      vname(5,indxS)='sea_water_salinity                          '
#  ifdef SPHERICAL
      vname(6,indxS)='lat_rho lon_rho                             '
#  else
      vname(6,indxS)='x_rho y_rho                                 '
#  endif
      vname(7,indxS)='                                            '
# endif /* SALINITY */

# ifdef PASSIVE_TRACER
      do i=1,ntrc_pas
        if( i < 10) then
           write(nametrc,'(A,I1)') 'tpas0',i
        else
           write(nametrc,'(A,I2)') 'tpas',i
        endif
CR      write(stdout,'(1x,A,I3,1x,I3)') 'indTPAS Nb',i, indxTPAS(i)
        vname(1,indxTPAS(i))=TRIM(nametrc)
        vname(2,indxTPAS(i))=TRIM(nametrc)//" passive tracer"
        vname(3,indxTPAS(i))='no unit                             '
        vname(4,indxTPAS(i))='passive tracer, scalar, series      '
        vname(5,indxTPAS(i))='                                   '
#  ifdef SPHERICAL
        vname(6,indxTPAS(i))='lat_rho lon_rho                     '
#  else
        vname(6,indxTPAS(i))='x_rho y_rho                         '
#  endif
        vname(7,indxTPAS(i))='                                    '
      enddo
# endif /* PASSIVE_TRACER */

# ifdef BIOLOGY
#  ifdef PISCES
      vname(1,indxDIC)='DIC                                       '
      vname(2,indxDIC)='Dissolved inorganic carbon                '
      vname(3,indxDIC)='umol C L-1                                '
      vname(4,indxDIC)='DIC, scalar, series                       '
      vname(5,indxDIC)='mole_concentration_of_dissolved_'
     &                           // 'inorganic_carbon_in_sea_water'
      vname(6,indxDIC)='lat_rho lon_rho                           '
      vname(7,indxDIC)='                                          '

      vname(1,indxTAL)='TALK                                      '
      vname(2,indxTAL)='Total Alkalinity                          '
      vname(3,indxTAL)='umol C L-1                                '
      vname(4,indxTAL)='Alkalinity, scalar, series                '
      vname(5,indxTAL)='                                          '
      vname(6,indxTAL)='lat_rho lon_rho                           '
      vname(7,indxTAL)='                                          '

      vname(1,indxOXY)='O2                                        '
      vname(2,indxOXY)='Oxygen                                    '
      vname(3,indxOXY)='umol L-1                                  '
      vname(4,indxOXY)='Oxygen, scalar, series                    '
      vname(5,indxOXY)='                                          '
      vname(6,indxOXY)='lat_rho lon_rho                           '
      vname(7,indxOXY)='                                          '

      vname(1,indxCAL)='CACO3                                     '
      vname(2,indxCAL)='Calcite                                   '
      vname(3,indxCAL)='umol C L-1                                '
      vname(4,indxCAL)='Calcite, scalar, series                   '
      vname(5,indxCAL)='                                          '
      vname(6,indxCAL)='                                          '
      vname(7,indxCAL)='                                          '

      vname(1,indxPO4)='PO4                                       '
      vname(2,indxPO4)='Phosphate                                 '
      vname(3,indxPO4)='umol P L-1                                '
      vname(4,indxPO4)='Phosphate, scalar, series                 '
      vname(5,indxPO4)='                                          '
      vname(6,indxPO4)='lat_rho lon_rho                           '
      vname(7,indxPO4)='                                          '

      vname(1,indxPOC)='POC                                       '
      vname(2,indxPOC)='Small particulate organic carbon          '
      vname(3,indxPOC)='umol C L-1                                '
      vname(4,indxPOC)='Small particles, scalar, series           '
      vname(5,indxPOC)='                                          '
      vname(6,indxPOC)='lat_rho lon_rho                           '
      vname(7,indxPOC)='                                          '

      vname(1,indxSIL)='Si                                        '
      vname(2,indxSIL)='Silicate                                  '
      vname(3,indxSIL)='umol Si L-1                               '
      vname(4,indxSIL)='Silicate, scalar, series                  '
      vname(5,indxSIL)='                                          '
      vname(6,indxSIL)='lat_rho lon_rho                           '
      vname(7,indxSIL)='                                          '

      vname(1,indxPHY)='NANO                                      '
      vname(2,indxPHY)='Nanophytoplankton                         '
      vname(3,indxPHY)='umol C L-1                                '
      vname(4,indxPHY)='Nanophytoplankton, scalar, series         '
      vname(5,indxPHY)='                                          '
      vname(6,indxPHY)='lat_rho lon_rho                           '
      vname(7,indxPHY)='                                          '

      vname(1,indxZOO)='ZOO                                       '
      vname(2,indxZOO)='Microzooplankton                          '
      vname(3,indxZOO)='umol C L-1                                '
      vname(4,indxZOO)='Microzooplankton, scalar, series          '
      vname(5,indxZOO)='                                          '
      vname(6,indxZOO)='lat_rho lon_rho                           '
      vname(7,indxZOO)='                                          '

      vname(1,indxDOC)='DOC                                       '
      vname(2,indxDOC)='Dissolved organic carbon                  '
      vname(3,indxDOC)='umol C L-1                                '
      vname(4,indxDOC)='DOC, scalar, series                       '
      vname(5,indxDOC)='                                          '
      vname(6,indxDOC)='lat_rho lon_rho                           '
      vname(7,indxDOC)='                                          '

      vname(1,indxDIA)='DIA                                       '
      vname(2,indxDIA)='Diatoms                                   '
      vname(3,indxDIA)='umol C L-1                                '
      vname(4,indxDIA)='Diatoms, scalar, series                   '
      vname(5,indxDIA)='                                          '
      vname(6,indxDIA)='lat_rho lon_rho                           '
      vname(7,indxDIA)='                                          '

      vname(1,indxMES)='MESO                                      '
      vname(2,indxMES)='Mesozooplankton                           '
      vname(3,indxMES)='umol C L-1                                '
      vname(4,indxMES)='Mesozooplankton, scalar, series           '
      vname(5,indxMES)='                                          '
      vname(6,indxMES)='lat_rho lon_rho                           '
      vname(7,indxMES)='                                          '

      vname(1,indxBSI)='BSI                                       '
      vname(2,indxBSI)='Diatoms silicon                           '
      vname(3,indxBSI)='umol Si L-1                               '
      vname(4,indxBSI)='BSi, scalar, series                       '
      vname(5,indxBSI)='                                          '
      vname(6,indxBSI)='lat_rho lon_rho                           '
      vname(7,indxBSI)='                                          '

      vname(1,indxFER)='FER                                       '
      vname(2,indxFER)='Dissolved iron                            '
      vname(3,indxFER)='umol Fe L-1                               '
      vname(4,indxFER)='Iron, scalar, series                      '
      vname(5,indxFER)='                                          '
      vname(6,indxFER)='lat_rho lon_rho                           '
      vname(7,indxFER)='                                          '

      vname(1,indxBFE)='BFE                                       '
      vname(2,indxBFE)='Iron in big particles                     '
      vname(3,indxBFE)='umol Fe L-1                               '
      vname(4,indxBFE)='BFE, scalar, series                       '
      vname(5,indxBFE)='                                          '
      vname(6,indxBFE)='lat_rho lon_rho                           '
      vname(7,indxBFE)='                                          '

      vname(1,indxGOC)='GOC                                       '
      vname(2,indxGOC)='Big particulate organic carbon            '
      vname(3,indxGOC)='umol C L-1                                '
      vname(4,indxGOC)='GOC, scalar, series                       '
      vname(5,indxGOC)='                                          '
      vname(6,indxGOC)='lat_rho lon_rho                           '
      vname(7,indxGOC)='                                          '

      vname(1,indxSFE)='SFE                                       '
      vname(2,indxSFE)='Iron in nanophytoplankton                 '
      vname(3,indxSFE)='umol Fe L-1                               '
      vname(4,indxSFE)='SFE, scalar, series                       '
      vname(5,indxSFE)='                                          '
      vname(6,indxSFE)='lat_rho lon_rho                           '
      vname(7,indxSFE)='                                          '

      vname(1,indxDFE)='DFE                                       '
      vname(2,indxDFE)='Iron in diatoms                           '
      vname(3,indxDFE)='umol Fe L-1                               '
      vname(4,indxDFE)='DFE, scalar, series                       '
      vname(5,indxDFE)='                                          '
      vname(6,indxDFE)='lat_rho lon_rho                           '
      vname(7,indxDFE)='                                          '

      vname(1,indxDSI)='DSI                                       '
      vname(2,indxDSI)='Biogenic silica                           '
      vname(3,indxDSI)='umol Si L-1                               '
      vname(4,indxDSI)='DSI, scalar, series                       '
      vname(5,indxDSI)='                                          '
      vname(6,indxDSI)='lat_rho lon_rho                           '
      vname(7,indxDSI)='                                          '

      vname(1,indxNFE)='NFE                                       '
      vname(2,indxNFE)='Iron in small particles                   '
      vname(3,indxNFE)='umol Fe L-1                               '
      vname(4,indxNFE)='NFE, scalar, series                       '
      vname(5,indxNFE)='                                          '
      vname(6,indxNFE)='lat_rho lon_rho                           '
      vname(7,indxNFE)='                                          '

      vname(1,indxNCH)='NCHL                                      '
      vname(2,indxNCH)='Chlorophyll in nano                       '
      vname(3,indxNCH)='mg Chl m-3                                '
      vname(4,indxNCH)='NCHL, scalar, series                      '
      vname(5,indxNCH)='                                          '
      vname(6,indxNCH)='lat_rho lon_rho                           '
      vname(7,indxNCH)='                                          '

      vname(1,indxDCH)='DCHL                                      '
      vname(2,indxDCH)='Chlorophyll in diatoms                    '
      vname(3,indxDCH)='mg Chl m-3                                '
      vname(4,indxDCH)='DCHL, scalar, series                      '
      vname(5,indxDCH)='                                          '
      vname(6,indxDCH)='lat_rho lon_rho                           '
      vname(7,indxDCH)='                                          '

      vname(1,indxNO3)='NO3                                       '
      vname(2,indxNO3)='Nitrate                                   '
      vname(3,indxNO3)='umol N L-1                                '
      vname(4,indxNO3)='Nitrate, scalar, series                   '
      vname(5,indxNO3)='                                          '
      vname(6,indxNO3)='lat_rho lon_rho                           '
      vname(7,indxNO3)='                                          '

      vname(1,indxNH4)='NH4                                       '
      vname(2,indxNH4)='Ammonium                                  '
      vname(3,indxNH4)='umol N L-1                                '
      vname(4,indxNH4)='Ammonium, scalar, series                  '
      vname(5,indxNH4)='                                          '
      vname(6,indxNH4)='lat_rho lon_rho                           '
      vname(7,indxNH4)='                                          '

#   if defined key_ligand
      vname(1,indxLGW)='LGW                                       '
      vname(2,indxLGW)='Ligands                                   '
      vname(3,indxLGW)='umol L-1                                  '
      vname(4,indxLGW)='Ligands, scalar, series                   '
      vname(5,indxLGW)='                                          '
      vname(6,indxLGW)='lat_rho lon_rho                           '
      vname(7,indxLGW)='                                          '
#   endif

#   if defined key_pisces_quota
      vname(1,indxDON)='DON                                       '
      vname(2,indxDON)='DON                                       '
      vname(3,indxDON)='umol L-1                                  '
      vname(4,indxDON)='DON, scalar, series                       '
      vname(5,indxDON)='                                          '
      vname(6,indxDON)='lat_rho lon_rho                           '
      vname(7,indxDON)='                                          '

      vname(1,indxDOP)='DOP                                       '
      vname(2,indxDOP)='DOP                                       '
      vname(3,indxDOP)='umol L-1                                  '
      vname(4,indxDOP)='DOP, scalar, series                       '
      vname(5,indxDOP)='                                          '
      vname(6,indxDOP)='lat_rho lon_rho                           '
      vname(7,indxDOP)='                                          '

      vname(1,indxPON)='PON                                       '
      vname(2,indxPON)='PON                                       '
      vname(3,indxPON)='umol L-1                                  '
      vname(4,indxPON)='PON, scalar, series                       '
      vname(5,indxPON)='                                          '
      vname(6,indxPON)='lat_rho lon_rho                           '
      vname(7,indxPON)='                                          '

      vname(1,indxPOP)='POP                                       '
      vname(2,indxPOP)='POP                                       '
      vname(3,indxPOP)='umol L-1                                  '
      vname(4,indxPOP)='POP, scalar, series                       '
      vname(5,indxPOP)='                                          '
      vname(6,indxPOP)='lat_rho lon_rho                           '
      vname(7,indxPOP)='                                          '

      vname(1,indxNPH)='NPH                                       '
      vname(2,indxNPH)='N quota of nanophyto                      '
      vname(3,indxNPH)='umol L-1                                  '
      vname(4,indxNPH)='NPH, scalar, series                       '
      vname(5,indxNPH)='                                          '
      vname(6,indxNPH)='lat_rho lon_rho                           '
      vname(7,indxNPH)='                                          '

      vname(1,indxPPH)='PPH                                       '
      vname(2,indxPPH)='P quota of nanophyto                      '
      vname(3,indxPPH)='umol L-1                                  '
      vname(4,indxPPH)='PPH, scalar, series                       '
      vname(5,indxPPH)='                                          '
      vname(6,indxPPH)='lat_rho lon_rho                           '
      vname(7,indxPPH)='                                          '

      vname(1,indxNDI)='NDI                                       '
      vname(2,indxNDI)='N quota of diatoms                        '
      vname(3,indxNDI)='umol L-1                                  '
      vname(4,indxNDI)='NDI, scalar, series                       '
      vname(5,indxNDI)='                                          '
      vname(6,indxNDI)='lat_rho lon_rho                           '
      vname(7,indxNDI)='                                          '

      vname(1,indxPDI)='PDI                                       '
      vname(2,indxPDI)='P quota of diatoms                        '
      vname(3,indxPDI)='umol L-1                                  '
      vname(4,indxPDI)='PDI, scalar, series                       '
      vname(5,indxPDI)='                                          '
      vname(6,indxPDI)='lat_rho lon_rho                           '
      vname(7,indxPDI)='                                          '

      vname(1,indxPIC)='PIC                                       '
      vname(2,indxPIC)='Picophytoplankton                         '
      vname(3,indxPIC)='umolC L-1                                 '
      vname(4,indxPIC)='Picophytoplankton, scalar, series         '
      vname(5,indxPIC)='                                          '
      vname(6,indxPIC)='lat_rho lon_rho                           '
      vname(7,indxPIC)='                                          '

      vname(1,indxNPI)='NPI                                       '
      vname(2,indxNPI)='N quota of picophyto                      '
      vname(3,indxNPI)='umol L-1                                  '
      vname(4,indxNPI)='NPI, scalar, series                       '
      vname(5,indxNPI)='                                          '
      vname(6,indxNPI)='lat_rho lon_rho                           '
      vname(7,indxNPI)='                                          '

      vname(1,indxPPI)='PPI                                       '
      vname(2,indxPPI)='P quota of picophyto                      '
      vname(3,indxPPI)='umol L-1                                  '
      vname(4,indxPPI)='PPI, scalar, series                       '
      vname(5,indxPPI)='                                          '
      vname(6,indxPPI)='lat_rho lon_rho                           '
      vname(7,indxPPI)='                                          '

      vname(1,indxPFE)='PFE                                       '
      vname(2,indxPFE)='Iron quota of picophyto                   '
      vname(3,indxPFE)='umol L-1                                  '
      vname(4,indxPFE)='PFE, scalar, series                       '
      vname(5,indxPFE)='                                          '
      vname(6,indxPFE)='lat_rho lon_rho                           '
      vname(7,indxPFE)='                                          '

      vname(1,indxPCH)='PCH                                       '
      vname(2,indxPCH)='Chlorophyll of picophyto                  '
      vname(3,indxPCH)='umol L-1                                  '
      vname(4,indxPCH)='PCH, scalar, series                       '
      vname(5,indxPCH)='                                          '
      vname(6,indxPCH)='lat_rho lon_rho                           '
      vname(7,indxPCH)='                                          '

      vname(1,indxGON)='GON                                       '
      vname(2,indxGON)='GON                                       '
      vname(3,indxGON)='umol L-1                                  '
      vname(4,indxGON)='GON, scalar, series                       '
      vname(5,indxGON)='                                          '
      vname(6,indxGON)='lat_rho lon_rho                           '
      vname(7,indxGON)='                                          '

      vname(1,indxGOP)='GOP                                       '
      vname(2,indxGOP)='GOP                                       '
      vname(3,indxGOP)='umol L-1                                  '
      vname(4,indxGOP)='GOP, scalar, series                       '
      vname(5,indxGOP)='                                          '
      vname(6,indxGOP)='lat_rho lon_rho                           '
      vname(7,indxGOP)='                                          '
#   endif

#  elif defined BIO_NChlPZD

CR      write(stdout,'(1x,A,I3)') 'indxNO3 =',indxNO3
      vname(1,indxNO3)='NO3                                       '
      vname(2,indxNO3)='NO3 Nutrient                              '
      vname(3,indxNO3)='mMol N m-3                                '
      vname(4,indxNO3)='NO3, scalar, series                       '
      vname(5,indxNO3)='                                          '
      vname(6,indxNO3)='lat_rho lon_rho                           '
      vname(7,indxNO3)='                                          '

CR      write(stdout,'(1x,A,I3)') 'indxChla =',indxChla
      vname(1,indxChla)='CHLA                                     '
      vname(2,indxChla)='Chlorophyll A                            '
      vname(3,indxChla)='mg Chla m-3                              '
      vname(4,indxChla)='Chlorophyl, scalar, series               '
      vname(5,indxChla)='                                         '
      vname(6,indxChla)='lat_rho lon_rho                          '
      vname(7,indxChla)='                                         '

CR      write(stdout,'(1x,A,I3)') 'indxPhy1 =',indxPhy1
      vname(1,indxPhy1)='PHYTO                                    '
      vname(2,indxPhy1)='Phytoplankton                            '
      vname(3,indxPhy1)='mMol N m-3                               '
      vname(4,indxPhy1)='Phytoplankton, scalar, series            '
      vname(5,indxPhy1)='                                         '
      vname(6,indxPhy1)='lat_rho lon_rho                          '
      vname(7,indxPhy1)='                                         '

CR      write(stdout,'(1x,A,I3)') 'indxZoo1 =',indxZoo1
      vname(1,indxZoo1)='ZOO                                      '
      vname(2,indxZoo1)='Zooplankton                              '
      vname(3,indxZoo1)='mMol N m-3                               '
      vname(4,indxZoo1)='Zooplankton, scalar, series              '
      vname(5,indxZoo1)='                                         '
      vname(6,indxZoo1)='lat_rho lon_rho                          '
      vname(7,indxZoo1)='                                         '

CR      write(stdout,'(1x,A,I3)') 'indxDet1 =',indxDet1
      vname(1,indxDet1)='DET                                      '
      vname(2,indxDet1)='Detritus Nutrient                        '
      vname(3,indxDet1)='mMol N m-3                               '
      vname(4,indxDet1)='Detritus, scalar, series                 '
      vname(5,indxDet1)='                                         '
      vname(6,indxDet1)='lat_rho lon_rho                          '
      vname(7,indxDet1)='                                         '
#   ifdef OXYGEN
CR      write(stdout,'(1x,A,I3)') 'indxO2 =',indxO2
      vname(1,indxO2)='O2                                         '
      vname(2,indxO2)='Dissolved Oxygen	                          '
      vname(3,indxO2)='mMol O2 m-3                                '
      vname(4,indxO2)='Dissolved Oxygen	, scalar, series          '
      vname(5,indxO2)='                                           '
      vname(6,indxO2)='lat_rho lon_rho                            '
      vname(7,indxO2)='                                           '
#   endif

CR      write(stdout,'(1x,A,I3)') 'ntrc_salt=',ntrc_salt
CR      write(stdout,'(1x,A,I3)') 'ntrc_pas =',ntrc_pas
CR      write(stdout,'(1x,A,I3)') 'ntrc_bio =',ntrc_bio
CR      write(stdout,'(1x,A,I3)') 'ntrc_sed =',ntrc_sed
CR      write(stdout,'(1x,A,I3)') 'ntrc_diabio=',ntrc_diabio

#  elif defined BIO_N2ChlPZD2

CR      write(stdout,'(1x,A,I3)') 'indxNO3 =',indxNO3
      vname(1,indxNO3)='NO3                                       '
      vname(2,indxNO3)='NO3 Nutrient                              '
      vname(3,indxNO3)='mMol N m-3                                '
      vname(4,indxNO3)='NO3, scalar, series                       '
      vname(5,indxNO3)='                                          '
      vname(6,indxNO3)='                                          '
      vname(7,indxNO3)='                                          '

CR      write(stdout,'(1x,A,I3)') 'indxNH4 =',indxNH4
      vname(1,indxNH4)='NH4                                       '
      vname(2,indxNH4)='NH4 Nutrient                              '
      vname(3,indxNH4)='mMol N m-3                                '
      vname(4,indxNH4)='NH4, scalar, series                       '
      vname(5,indxNH4)='                                          '
      vname(6,indxNH4)='lat_rho lon_rho                           '
      vname(7,indxNH4)='  '

CR      write(stdout,'(1x,A,I3)') 'indxChla =',indxChla
      vname(1,indxChla)='CHLA                                     '
      vname(2,indxChla)='Chlorophyll A                            '
      vname(3,indxChla)='mg Chla m-3                              '
      vname(4,indxChla)='Chlorophyl, scalar, series               '
      vname(5,indxChla)='                                         '
      vname(6,indxChla)='lat_rho lon_rho                          '
      vname(7,indxChla)='                                         '

CR      write(stdout,'(1x,A,I3)') 'indxPhy1 =',indxPhy1
      vname(1,indxPhy1)='PHYTO                                    '
      vname(2,indxPhy1)='Phytoplankton                            '
      vname(3,indxPhy1)='mMol N m-3                               '
      vname(4,indxPhy1)='Phytoplankton, scalar, series            '
      vname(5,indxPhy1)='                                         '
      vname(6,indxPhy1)='lat_rho lon_rho                          '
      vname(7,indxPhy1)='                                         '

CR      write(stdout,'(1x,A,I3)') 'indxZoo1 =',indxZoo1
      vname(1,indxZoo1)='ZOO                                      '
      vname(2,indxZoo1)='Zooplankton                              '
      vname(3,indxZoo1)='mMol N m-3                               '
      vname(4,indxZoo1)='Zooplankton, scalar, series              '
      vname(5,indxZoo1)='                                         '
      vname(6,indxZoo1)='lat_rho lon_rho                          '
      vname(7,indxZoo1)='                                         '

CR      write(stdout,'(1x,A,I3)') 'indxDet1 =',indxDet1
      vname(1,indxDet1)='SDET                                     '
      vname(2,indxDet1)='Small Detritus Nutrient                  '
      vname(3,indxDet1)='mMol N m-3                               '
      vname(4,indxDet1)='Small Detritus, scalar, series           '
      vname(5,indxDet1)='                                         '
      vname(6,indxDet1)='lat_rho lon_rh                           '
      vname(7,indxDet1)='                                         '

CR      write(stdout,'(1x,A,I3)') 'indxDet2 =',indxDet2
      vname(1,indxDet2)='LDET                                     '
      vname(2,indxDet2)='Large Detritus Nutrient                  '
      vname(3,indxDet2)='mMol N m-3                               '
      vname(4,indxDet2)='Large Detritus, scalar, series           '
      vname(5,indxDet2)='                                         '
      vname(6,indxDet2)='lat_rho lon_rho                          '
      vname(7,indxDet2)='                                         '

#  elif defined BIO_BioEBUS

CR      write(stdout,'(1x,A,I3)') 'indxNO3 =',indxNO3
      vname(1,indxNO3)='NO3                                       '
      vname(2,indxNO3)='Nitrate                                   '
      vname(3,indxNO3)='mmol N m-3                                '
      vname(4,indxNO3)='NO3, scalar, series                       '
      vname(5,indxNO3)='                                          '
      vname(6,indxNO3)='                                          '
      vname(7,indxNO3)='                                          '

CR      write(stdout,'(1x,A,I3)') 'indxNO2 =',indxNO2
      vname(1,indxNO2)='NO2                                       '
      vname(2,indxNO2)='Nitrite                                   '
      vname(3,indxNO2)='mmol N m-3                                '
      vname(4,indxNO2)='NO2, scalar, series                       '
      vname(5,indxNO2)='                                          '
      vname(6,indxNO2)='                                          '
      vname(7,indxNO2)='                                          '

CR      write(stdout,'(1x,A,I3)') 'indxNH4 =',indxNH4
      vname(1,indxNH4)='NH4                                       '
      vname(2,indxNH4)='Ammonium                                  '
      vname(3,indxNH4)='mmol N m-3                                '
      vname(4,indxNH4)='NH4, scalar, series                       '
      vname(5,indxNH4)='                                          '
      vname(6,indxNH4)='                                          '
      vname(7,indxNH4)='                                          '

CR      write(stdout,'(1x,A,I3)') 'indxPhy1 =',indxPhy1
      vname(1,indxPhy1)='SPHYTO                                   '
      vname(2,indxPhy1)='Small Phytoplankton                      '
      vname(3,indxPhy1)='mmol N m-3                               '
      vname(4,indxPhy1)='Small Phytoplankton, scalar, series      '
      vname(5,indxPhy1)='                                         '
      vname(6,indxPhy1)='                                         '
      vname(7,indxPhy1)='                                         '

CR      write(stdout,'(1x,A,I3)') 'indxPhy2 =',indxPhy2
      vname(1,indxPhy2)='LPHYTO                                   '
      vname(2,indxPhy2)='Large Phytoplankton                      '
      vname(3,indxPhy2)='mmol N m-3                               '
      vname(4,indxPhy2)='Large Phytoplankton, scalar, series      '
      vname(5,indxPhy2)='                                         '
      vname(6,indxPhy2)='                                         '
      vname(7,indxPhy2)='                                         '

CR      write(stdout,'(1x,A,I3)') 'indxZoo1 =',indxZoo1
      vname(1,indxZoo1)='SZOO                                     '
      vname(2,indxZoo1)='Small Zooplankton                        '
      vname(3,indxZoo1)='mmol N m-3                               '
      vname(4,indxZoo1)='Small Zooplankton, scalar, series        '
      vname(5,indxZoo1)='                                         '
      vname(6,indxZoo1)='                                         '
      vname(7,indxZoo1)='                                         '

CR      write(stdout,'(1x,A,I3)') 'indxZoo2 =',indxZoo2
      vname(1,indxZoo2)='LZOO                                     '
      vname(2,indxZoo2)='Large Zooplankton                        '
      vname(3,indxZoo2)='mmol N m-3                               '
      vname(4,indxZoo2)='Large Zooplankton, scalar, series        '
      vname(5,indxZoo2)='                                         '
      vname(6,indxZoo2)='                                         '
      vname(7,indxZoo2)='                                         '

CR      write(stdout,'(1x,A,I3)') 'indxDet1 =',indxDet1
      vname(1,indxDet1)='SDET                                     '
      vname(2,indxDet1)='Small Detritus                           '
      vname(3,indxDet1)='mmol N m-3                               '
      vname(4,indxDet1)='Small Detritus, scalar, series           '
      vname(5,indxDet1)='                                         '
      vname(6,indxDet1)='                                         '
      vname(7,indxDet1)='                                         '

CR      write(stdout,'(1x,A,I3)') 'indxDet2 =',indxDet2
      vname(1,indxDet2)='LDET                                     '
      vname(2,indxDet2)='Large Detritus                           '
      vname(3,indxDet2)='mmol N m-3                               '
      vname(4,indxDet2)='Large Detritus, scalar, series           '
      vname(5,indxDet2)='                                         '
      vname(6,indxDet2)='                                         '
      vname(7,indxDet2)='                                         '

CR      write(stdout,'(1x,A,I3)') 'indxDON =',indxDON
      vname(1,indxDON)='DON                                       '
      vname(2,indxDON)='Dissolved Organic Nitrogen                '
      vname(3,indxDON)='mmol N m-3                                '
      vname(4,indxDON)='Dissolved Organic Nitrogen, scalar, series'
      vname(5,indxDON)='                                          '
      vname(6,indxDON)='                                          '
      vname(7,indxDON)='                                          '

CR      write(stdout,'(1x,A,I3)') 'indxO2 =',indxO2
      vname(1,indxO2)='O2                                         '
      vname(2,indxO2)='Dissolved Oxygen                           '
      vname(3,indxO2)='mmol O  m-3                                '
      vname(4,indxO2)='Dissolved Oxygen, scalar, series           '
      vname(5,indxO2)='                                           '
      vname(6,indxO2)='                                           '
      vname(7,indxO2)='                                           '

CR      write(stdout,'(1x,A,I3)') 'indxAOU =',indxAOU
      vname(1,indxAOU)='AOU                                        '
      vname(2,indxAOU)='Apparent Oxygen Utilization                '
      vname(3,indxAOU)='mmol O2 m-3                                '
      vname(4,indxAOU)='Apparent Oxygen Utilization, scalar, series'
      vname(5,indxAOU)='                                           '
      vname(6,indxAOU)='                                           '
      vname(7,indxAOU)='                                           '

CR      write(stdout,'(1x,A,I3)') 'indxWIND10 =',indxWIND10
      vname(1,indxWIND10)='wind10                                 '
      vname(2,indxWIND10)='surface wind speed (10 m)              '
      vname(3,indxWIND10)='meter second-1                         '
      vname(4,indxWIND10)='surface wind speed, scalar, series     '
      vname(5,indxWIND10)='                                       '
      vname(6,indxWIND10)='                                       '
      vname(7,indxWIND10)='                                       '

#    ifdef NITROUS_OXIDE

CR      write(stdout,'(1x,A,I3)') 'indxN2O =',indxN2O
      vname(1,indxN2O)='N2O                                       '
      vname(2,indxN2O)='Nitrous Oxide                             '
      vname(3,indxN2O)='mmol N m-3                                '
      vname(4,indxN2O)='N2O, scalar, series                       '
      vname(5,indxN2O)='                                          '
      vname(6,indxN2O)='                                          '
      vname(7,indxN2O)='                                          '

#    endif

#  endif /* PISCES || BIO_NChlPZD || BIO_N2ChlPZD2 || BIO_BioEBUS */
!----------------------------------------------------------------------------

# endif /* BIOLOGY */

# ifdef SEDIMENT
      do i=1,NSAND
       if( i < 10) then
          write(nametrc,'(A,I1)') 'sand_0',i
       else
          write(nametrc,'(A,I2)') 'sand_',i
       endif
CR        write(stdout,'(1x,A,I3,1x,I3)') 'indxsand Nb',i, indxSAND(i)
        vname(1,indxsand(i))=TRIM(nametrc)
        vname(2,indxsand(i))=TRIM(nametrc)//" sediment"
        vname(3,indxsand(i))='kg/m3                               '
        vname(4,indxsand(i))='sand, scalar, series                '
        vname(5,indxsand(i))='                                    '
        vname(6,indxsand(i))='lat_rho lon_rho                     '
        vname(7,indxsand(i))='                                    '
      enddo

      do i=1,NMUD
        if( i < 10) then
          write(nametrc,'(A,I1)') 'mud_0',i
        else
          write(nametrc,'(A,I2)') 'mud_',i
        endif
CR        write(stdout,'(1x,A,I3,1x,I3)') 'indxmud Nb',i, indxMUD(i)
        vname(1,indxmud(i))=TRIM(nametrc)
        vname(2,indxmud(i))=TRIM(nametrc)//" sediment"
        vname(3,indxmud(i))='kg/m3                                '
        vname(4,indxmud(i))='mud, scalar, series                  '
        vname(5,indxmud(i))='                                     '
        vname(6,indxmud(i))='lat_rho lon_rho                      '
        vname(7,indxmud(i))='                                     '
      enddo

      do i=1,NGRAV
        if( i < 10) then
          write(nametrc,'(A,I1)') 'gravel_0',i
        else
          write(nametrc,'(A,I2)') 'gravel_',i
        endif
CR        write(stdout,'(1x,A,I3,1x,I3)') 'indgrav Nb',i, indxGRAV(i)
        vname(1,indxgrav(i))=TRIM(nametrc)
        vname(2,indxgrav(i))=TRIM(nametrc)//" sediment"
        vname(3,indxgrav(i))='kg/m3                               '
        vname(4,indxgrav(i))='gravel, scalar, series              '
        vname(5,indxgrav(i))='                                    '
        vname(6,indxgrav(i))='lat_rho lon_rho                     '
        vname(7,indxgrav(i))='                                    '
      enddo

CR      write(stdout,'(1x,A,I3)') 'indxATHK =',indxATHK
      vname(1,indxATHK)='act_thick                                '
      vname(2,indxATHK)='Thickness of sediment active layer       '
      vname(3,indxATHK)='m                                        '
      vname(5,indxATHK)='                                         '
      vname(6,indxATHK)='lat_rho lon_rho                          '
      vname(7,indxATHK)='                                         '

CR      write(stdout,'(1x,A,I3)') 'indxBTHK =',indxBTHK
      vname(1,indxBTHK)='bed_thick                                '
      vname(2,indxBTHK)='Thickness of sediment bed layer          '
      vname(3,indxBTHK)='m                                        '
      vname(5,indxBTHK)='                                         '
      vname(6,indxBTHK)='lat_rho lon_rho                          '
      vname(7,indxBTHK)='                                         '

CR      write(stdout,'(1x,A,I3)') 'indxBPOR =',indxBPOR
      vname(1,indxBPOR)='bed_poros                                '
      vname(2,indxBPOR)='Porosity of sediment bed layer           '
      vname(3,indxBPOR)='no units                                 '
      vname(5,indxBPOR)='                                         '
      vname(6,indxBPOR)='lat_rho lon_rho                          '
      vname(7,indxBPOR)='                                         '


# if defined MIXED_BED || defined COHESIVE_BED
CR      write(stdout,'(1x,A,I3)') 'indxBTCR =',indxBTCR
      vname(1,indxBTCR)='bed_tau_crit                             '
      vname(2,indxBTCR)='Bed tau critical in each layer           '
      vname(3,indxBTCR)='N/m2                                     '
      vname(5,indxBTCR)='                                         '
      vname(6,indxBTCR)='lat_rho lon_rho                          '
      vname(7,indxBTCR)='                                         '
# endif

      do i=isand1,isand2
        if( i < 10) then
          write(nametrc,'(A,I1)') 'bed_frac_sand_0',i
        else
          write(nametrc,'(A,I2)') 'bed_frac_sand_',i
        endif
CR      write(stdout,'(1x,A,I3,1x,I3)') 'indxBFRA_sand  Nb',i, indxBFRA(i)
        vname(1,indxBFRA(i))=TRIM(nametrc)
        vname(2,indxBFRA(i))='vol. fraction of sand in bed layer  '
        vname(3,indxBFRA(i))='no units                            '
        vname(5,indxBFRA(i))='                                    '
        vname(6,indxBFRA(i))='lat_rho lon_rho                     '
        vname(7,indxBFRA(i))='                                    '
      enddo
      itrc=0
      do i=imud1,imud2
        itrc=itrc+1
        if( itrc < 10) then
          write(nametrc,'(A,I1)') 'bed_frac_mud_0',itrc
        else
          write(nametrc,'(A,I2)') 'bed_frac_mud_',itrc
        endif
CR      write(stdout,'(1x,A,I3,1x,I3)') 'indxBFRA_mud  Nb',i, indxBFRA(i)
        vname(1,indxBFRA(i))  =TRIM(nametrc)
        vname(2,indxBFRA(i))  ='vol. fraction of mud in bed layer '
        vname(3,indxBFRA(i))  ='no units                          '
        vname(5,indxBFRA(i))='                                    '
        vname(6,indxBFRA(i))='lat_rho lon_rho                     '
        vname(7,indxBFRA(i))='                                    '
      enddo
      itrc=0
      do i=igrav1,igrav2
        itrc=itrc+1
        if( itrc < 10) then
          write(nametrc,'(A,I1)') 'bed_frac_grav_0',itrc
        else
          write(nametrc,'(A,I2)') 'bed_frac_grav_',itrc
        endif
CR      write(stdout,'(1x,A,I3,1x,I3)') 'indxBFRA_grav  Nb',i, indxBFRA(i)
        vname(1,indxBFRA(i))=TRIM(nametrc)
        vname(2,indxBFRA(i))='vol. fraction of gravel in bed layer'
        vname(3,indxBFRA(i))='no units                            '
        vname(5,indxBFRA(i))='                                    '
        vname(6,indxBFRA(i))='lat_rho lon_rho                     '
        vname(7,indxBFRA(i))='                                    '
      enddo

#  ifdef SUSPLOAD
      do i=isand1,isand2
        if( i < 10) then
          write(nametrc,'(A,I1)') 'dflx_sand_0',i
        else
          write(nametrc,'(A,I2)') 'dflx_sand_',i
        endif
CR        write(stdout,'(1x,A,I3,1x,I3)') 'sand_dflx Nb',i, indxDFLX(i)
        vname(1,indxDFLX(i))=TRIM(nametrc)
        vname(2,indxDFLX(i))=TRIM(nametrc)//" sediment"
        vname(3,indxDFLX(i))='kg/m2/s                             '
        vname(4,indxDFLX(i))='sand/s, scalar, series              '
        vname(5,indxDFLX(i))='                                    '
        vname(6,indxDFLX(i))='lat_rho lon_rho                     '
        vname(7,indxDFLX(i))='                                    '
      enddo

      itrc=0
      do i=imud1,imud2
        itrc=itrc+1
        if( itrc < 10) then
          write(nametrc,'(A,I1)') 'dflx_mud_0',itrc
        else
          write(nametrc,'(A,I2)') 'dflx_mud_',itrc
        endif
CR        write(stdout,'(1x,A,I3,1x,I3)') 'mud_dflx Nb',i, indxDFLX(i)
        vname(1,indxDFLX(i))=TRIM(nametrc)
        vname(2,indxDFLX(i))=TRIM(nametrc)//" sediment"
        vname(3,indxDFLX(i))='kg/m2/s                             '
        vname(4,indxDFLX(i))='mud, scalar, series                 '
        vname(5,indxDFLX(i))='                                    '
        vname(6,indxDFLX(i))='lat_rho lon_rho                     '
        vname(7,indxDFLX(i))='                                    '
      enddo

      itrc=0
      do i=igrav1,igrav2
       itrc=itrc+1
        if( itrc < 10) then
          write(nametrc,'(A,I1)') 'dflx_grav_0',itrc
        else
          write(nametrc,'(A,I2)') 'dflx_grav_',itrc
        endif
CR        write(stdout,'(1x,A,I3,1x,I3)') 'gravel_dflx Nb',i, indxDFLX(i)
        vname(1,indxDFLX(i))=TRIM(nametrc)
        vname(2,indxDFLX(i))=TRIM(nametrc)
        vname(3,indxDFLX(i))='kg/m2/s                             '
        vname(4,indxDFLX(i))='gravel, scalar, series              '
        vname(5,indxDFLX(i))='                                    '
        vname(6,indxDFLX(i))='lat_rho lon_rho                     '
        vname(7,indxDFLX(i))='                                    '
      enddo

      do i=isand1,isand2
        if( i < 10) then
          write(nametrc,'(A,I1)') 'eflx_sand_0',i
        else
          write(nametrc,'(A,I2)') 'eflx_sand_',i
        endif
CR        write(stdout,'(1x,A,I3,1x,I3)') 'sand_eflx Nb',i, indxEFLX(i)
        vname(1,indxEFLX(i))=TRIM(nametrc)
        vname(2,indxEFLX(i))=TRIM(nametrc)//" sediment"
        vname(3,indxEFLX(i))='kg/m2/s                             '
        vname(4,indxEFLX(i))='sand, scalar, series                '
        vname(5,indxEFLX(i))='                                    '
        vname(6,indxEFLX(i))='lat_rho lon_rho                     '
        vname(7,indxEFLX(i))='                                    '
      enddo

      itrc=0
      do i=imud1,imud2
        itrc=itrc+1
        if( itrc < 10) then
          write(nametrc,'(A,I1)') 'eflx_mud_0',itrc
        else
          write(nametrc,'(A,I2)') 'eflx_mud_',itrc
        endif
CR        write(stdout,'(1x,A,I3,1x,I3)') 'mud_eflx Nb',i, indxEFLX(i)
        vname(1,indxEFLX(i))=TRIM(nametrc)
        vname(2,indxEFLX(i))=TRIM(nametrc)//" sediment"
        vname(3,indxEFLX(i))='kg/m2/s                             '
        vname(4,indxEFLX(i))='mud, scalar, series                 '
        vname(5,indxEFLX(i))='                                    '
        vname(6,indxEFLX(i))='lat_rho lon_rho                     '
        vname(7,indxEFLX(i))='                                    '
      enddo

      itrc=0
      do i=igrav1,igrav2
       itrc=itrc+1
        if( itrc < 10) then
          write(nametrc,'(A,I1)') 'dflx_grav_0',itrc
        else
          write(nametrc,'(A,I2)') 'dflx_grav_',itrc
        endif
CR        write(stdout,'(1x,A,I3,1x,I3)') 'gravel_eflx Nb',i, indxEFLX(i)
        vname(1,indxEFLX(i))=TRIM(nametrc)
        vname(2,indxEFLX(i))=TRIM(nametrc)//" sediment"
        vname(3,indxEFLX(i))='kg/m2/s                             '
        vname(4,indxEFLX(i))='gravel, scalar, series              '
        vname(5,indxEFLX(i))='                                    '
        vname(6,indxEFLX(i))='lat_rho lon_rho                     '
        vname(7,indxEFLX(i))='                                    '
      enddo
#  endif
#  ifdef BEDLOAD
      do i=isand1,isand2
        if( i < 10) then
          write(nametrc,'(A,I1)') 'bdlu_sand_0',i
        else
          write(nametrc,'(A,I2)') 'bdlu_sand_',i
        endif
CR        write(stdout,'(1x,A,I3,1x,I3)') 'sand_bdlu Nb',i, indxBDLU(i)
        vname(1,indxBDLU(i))=TRIM(nametrc)
        vname(2,indxBDLU(i))=TRIM(nametrc)
        vname(3,indxBDLU(i))='kg/m2/s                             '
        vname(4,indxBDLU(i))='sand, scalar, series                '
        vname(5,indxBDLU(i))='                                    '
        vname(6,indxBDLU(i))='lat_rho lon_rho                     '
        vname(7,indxBDLU(i))='                                    '
      enddo

      itrc=0
      do i=imud1,imud2
       itrc=itrc+1
        if( itrc < 10) then
          write(nametrc,'(A,I1)') 'bdlu_mud_0',itrc
        else
          write(nametrc,'(A,I2)') 'bdlu_mud_',itrc
        endif
CR        write(stdout,'(1x,A,I3,1x,I3)') 'mud_bdlu Nb',i, indxBDLU(i)
        vname(1,indxBDLU(i))=TRIM(nametrc)
        vname(2,indxBDLU(i))=TRIM(nametrc)
        vname(3,indxBDLU(i))='kg/m2/s                             '
        vname(4,indxBDLU(i))='mud, scalar, series                 '
        vname(5,indxBDLU(i))='                                    '
        vname(6,indxBDLU(i))='lat_rho lon_rho                     '
        vname(7,indxBDLU(i))='                                    '
      enddo

      itrc=0
      do i=igrav1,igrav2
        itrc=itrc+1
        if( itrc < 10) then
          write(nametrc,'(A,I1)') 'bdlu_grav_0',itrc
        else
          write(nametrc,'(A,I2)') 'bdlu_grav_',itrc
        endif
CR        write(stdout,'(1x,A,I3,1x,I3)') 'gravel_bdlu Nb',i, indxBDLU(i)
        vname(1,indxBDLU(i))=TRIM(nametrc)
        vname(2,indxBDLU(i))=TRIM(nametrc)
        vname(3,indxBDLU(i))='kg/m2/s                             '
        vname(4,indxBDLU(i))='gravel, scalar, series              '
        vname(5,indxBDLU(i))='                                    '
        vname(6,indxBDLU(i))='lat_rho lon_rho                     '
        vname(7,indxBDLU(i))='                                    '
      enddo

      do i=isand1,isand2
        if( i < 10) then
          write(nametrc,'(A,I1)') 'bdlv_sand_0',i
        else
          write(nametrc,'(A,I2)') 'bdlv_sand_',i
        endif
CR        write(stdout,'(1x,A,I3,1x,I3)') 'sand_bdlv_ Nb',i, indxBDLV(i)
        vname(1,indxBDLV(i))=TRIM(nametrc)
        vname(2,indxBDLV(i))=TRIM(nametrc)
        vname(3,indxBDLV(i))='kg/m2/s                             '
        vname(4,indxBDLV(i))='sand bdlv, scalar, series           '
        vname(5,indxBDLV(i))='                                    '
        vname(6,indxBDLV(i))='lat_rho lon_rho                     '
        vname(7,indxBDLV(i))='                                    '
      enddo

      itrc=0
      do i=imud1,imud2
        itrc=itrc+1
        if( itrc < 10) then
          write(nametrc,'(A,I1)') 'bdlv_mud_0',itrc
        else
          write(nametrc,'(A,I2)') 'bdlv_mud_',itrc
        endif
CR        write(stdout,'(1x,A,I3,1x,I3)') 'mud_bdlv Nb',i, indxBDLV(i)
        vname(1,indxBDLV(i))=TRIM(nametrc)
        vname(2,indxBDLV(i))=TRIM(nametrc)
        vname(3,indxBDLV(i))='kg/m2/s                             '
        vname(4,indxBDLV(i))='mud bdlv, scalar, series            '
        vname(5,indxBDLV(i))='                                    '
        vname(6,indxBDLV(i))='lat_rho lon_rho                     '
        vname(7,indxBDLV(i))='                                    '
      enddo

      itrc=0
      do i=igrav1,igrav2
        itrc=itrc+1
        if( itrc < 10) then
          write(nametrc,'(A,I1)') 'bdlv_grav_0',itrc
        else
          write(nametrc,'(A,I2)') 'bdlv_grav_',itrc
        endif
CR        write(stdout,'(1x,A,I3,1x,I3)') 'gravel_bdlv_ Nb',i, indxBDLV(i)
        vname(1,indxBDLV(i))=TRIM(nametrc)
        vname(2,indxBDLV(i))=TRIM(nametrc)
        vname(3,indxBDLV(i))='kg/m2/s                             '
        vname(4,indxBDLV(i))='gravel bdlv, scalar, series         '
        vname(5,indxBDLV(i))='                                    '
        vname(6,indxBDLV(i))='lat_rho lon_rho                     '
        vname(7,indxBDLV(i))='                                    '
      enddo
#  endif
# endif /* SEDIMENT */

# ifdef BBL
CR      write(stdout,'(1x,A,I3)') 'indxAbed=',indxBBL
      vname(1,indxBBL)  ='Abed                                    '
      vname(2,indxBBL)  ='Bed wave excursion amplitude            '
      vname(3,indxBBL)  ='m                                       '
      vname(5,indxBBL)='                                          '
      vname(6,indxBBL)='lat_rho lon_rho                           '
      vname(7,indxBBL)='                                          '

CR      write(stdout,'(1x,A,I3)') 'indxHrip =',indxBBL+1
      vname(1,indxHrip)='Hripple                                  '
      vname(2,indxHrip)='Bed ripple height                        '
      vname(3,indxHrip)='m                                        '
      vname(5,indxHrip)='                                         '
      vname(6,indxHrip)='lat_rho lon_rho                          '
      vname(7,indxHrip)='                                         '

CR      write(stdout,'(1x,A,I3)') 'indxLrip =',indxBBL+2
      vname(1,indxLrip)='Lripple                                  '
      vname(2,indxLrip)='Bed ripple length                        '
      vname(3,indxLrip)='m                                        '
      vname(5,indxLrip)='                                         '
      vname(6,indxLrip)='lat_rho lon_rho                          '
      vname(7,indxLrip)='                                         '

CR      write(stdout,'(1x,A,I3)') 'indxZbnot=',indxBBL+3
      vname(1,indxZbnot)='Zbnot                                   '
      vname(2,indxZbnot)='Physical hydraulic bottom roughness     '
      vname(3,indxZbnot)='m                                       '
      vname(5,indxZbnot)='                                        '
      vname(6,indxZbnot)='lat_rho lon_rho                         '
      vname(7,indxZbnot)='                                        '

CR      write(stdout,'(1x,A,I3)') 'indxZbapp=',indxBBL+4
      vname(1,indxZbapp)='Zbapp                                   '
      vname(2,indxZbapp)='Apparent hydraulic bottom roughness     '
      vname(3,indxZbapp)='m                                       '
      vname(5,indxZbapp)='                                        '
      vname(6,indxZbapp)='lat_rho lon_rho                         '
      vname(7,indxZbapp)='                                        '

CR      write(stdout,'(1x,A,I3)') 'indxBostrw=',indxBBL+5
      vname(1,indxBostrw)='bostrw                                 '
      vname(2,indxBostrw)='Wave-induced kinematic bottom stress   '
      vname(3,indxBostrw)='N/m2                                   '
      vname(5,indxBostrw)='                                       '
      vname(6,indxBostrw)='lat_rho lon_rho                        '
      vname(7,indxBostrw)='                                       '
# endif /* BBL */

CR      write(stdout,'(1x,A,I3)') 'indxShflx =',indxShflx
      vname(1,indxShflx)='shflux                                  '
      vname(2,indxShflx)='surface net heat flux                   '
      vname(3,indxShflx)='Watts meter-2                           '
      vname(4,indxShflx)='surface heat flux, scalar, series       '
#ifdef SPHERICAL
      vname(6,indxShflx)='lat_rho lon_rho                         '
#else
      vname(6,indxShflx)='x_rho y_rho                             '
#endif
      vname(7,indxShflx)='                                        '

#ifdef SALINITY
CR      write(stdout,'(1x,A,I3)') 'indxSwflx =',indxSwflx
      vname(1,indxSwflx)='swflux                                  '
      vname(2,indxSwflx)='surface freshwater flux (E-P)           '
      vname(3,indxSwflx)='centimeter day-1                        '
      vname(4,indxSwflx)='surface freshwater flux, scalar, series '
      vname(5,indxSwflx)='                                        '
#ifdef SPHERICAL
      vname(6,indxSwflx)='lat_rho lon_rho                         '
#else
      vname(6,indxSwflx)='x_rho y_rho                             '
#endif
      vname(7,indxSwflx)='                                        '
      vname(8,indxSwflx)='                                        '
#endif

CR      write(stdout,'(1x,A,I3)') 'indxShflx_rsw =',indxShflx_rsw
# ifdef BULK_FLUX
      vname(1,indxShflx_rsw)='radsw                               '
# else
      vname(1,indxShflx_rsw)='swrad                               '
# endif
      vname(2,indxShflx_rsw)='Short-wave surface radiation        '
      vname(3,indxShflx_rsw)='Watts meter-2                       '
      vname(4,indxShflx_rsw)='                                    '
      vname(5,indxShflx_rsw)='                                    '
      vname(6,indxShflx_rsw)='lat_rho lon_rho                     '
      vname(7,indxShflx_rsw)='                                    '
# ifdef DIURNAL_INPUT_SRFLX
#  ifdef BULK_FLUX
      vname(1,indxShflx_rswbio)='radswbio                         '
#  else
      vname(1,indxShflx_rswbio)='swradbio                         '
#  endif
      vname(2,indxShflx_rswbio)='Short-wave surface radiation     '
      vname(3,indxShflx_rswbio)='Watts meter-2                    '
      vname(4,indxShflx_rswbio)='                                 '
      vname(5,indxShflx_rswbio)='                                 '
      vname(6,indxShflx_rswbio)='lat_rho lon_rho                  '
      vname(7,indxShflx_rswbio)='                                 '
# endif /* DIURNAL_INPUT_SRFLX  */
# if defined BULK_FLUX || defined OA_COUPLING
CR      write(stdout,'(1x,A,I3)') 'indxShflx_rlw =',indxShflx_rlw
      vname(1,indxShflx_rlw)='shflx_rlw                           '
      vname(2,indxShflx_rlw)='Long-wave surface radiation         '
      vname(3,indxShflx_rlw)='Watts meter-2                       '
      vname(5,indxShflx_rlw)='                                    '
      vname(6,indxShflx_rlw)='lat_rho lon_rho                     '
      vname(7,indxShflx_rlw)='                                    '

CR      write(stdout,'(1x,A,I3)') 'indxShflx_lat =',indxShflx_lat
      vname(1,indxShflx_lat)='shflx_lat                           '
      vname(2,indxShflx_lat)='Latent surface heat flux            '
      vname(3,indxShflx_lat)='Watts meter-2                       '
      vname(4,indxShflx_sen)='                                    '
      vname(5,indxShflx_lat)='                                    '
      vname(6,indxShflx_lat)='lat_rho lon_rho                     '
      vname(7,indxShflx_lat)='                                    '

CR      write(stdout,'(1x,A,I3)') 'indxShflx_sen =',indxShflx_sen
      vname(1,indxShflx_sen)='shflx_sen                           '
      vname(2,indxShflx_sen)='Sensible surface heat flux          '
      vname(3,indxShflx_sen)='Watts meter-2                       '
      vname(4,indxShflx_sen)='                                    '
      vname(4,indxShflx_sen)='                                    '
      vname(6,indxShflx_sen)='lat_rho lon_rho                     '
      vname(7,indxShflx_sen)='                                    '
      vname(7,indxShflx_sen)='  '
#  ifdef READ_PATM
CR      write(stdout,'(1x,A,I3)') 'inxPatm =',inxPatm
      vname(1,indxPatm)='patm                                     '
      vname(2,indxPatm)='Atmospheric Pressure                     '
      vname(3,indxPatm)='millibar                                 '
      vname(4,indxPatm)='                                         '
      vname(4,indxPatm)='                                         '
      vname(6,indxPatm)='lat_rho lon_rho                          '
      vname(7,indxPatm)='                                         '
      vname(7,indxPatm)='  '
#  endif

# endif /* BULK_FLUX */

# ifdef SST_SKIN
CR      write(stdout,'(1x,A,I3)') 'indxSST_skin =',indxSST_skin
      vname(1,indxSST_skin)='sst_skin                             '
      vname(2,indxSST_skin)='Skin temperature                     '
      vname(3,indxSST_skin)='Celsius                              '
      vname(4,indxSST_skin)='sst_skin, scalars, series            '
      vname(5,indxSST_skin)='skin_temperature                     '
      vname(6,indxSST_skin)='lat_rho lon_rho                      '
      vname(7,indxSST_skin)='                                     '
# endif

CR      write(stdout,'(1x,A,I3)') 'indxO =',indxO
      vname(1,indxO)='omega                                       '
      vname(2,indxO)='S-coordinate vertical momentum component    '
      vname(3,indxO)='meter second-1                              '
      vname(4,indxO)='omega, scalar, series                       '
      vname(5,indxO)='                                            '
      vname(6,indxO)='lat_rho lon_rho                             '
      vname(7,indxO)='                                            '

CR      write(stdout,'(1x,A,I3)') 'indxW =',indxW
      vname(1,indxW)='w                                           '
      vname(2,indxW)='vertical momentum component                 '
      vname(3,indxW)='meter second-1                              '
      vname(4,indxW)='w-velocity, scalar, series                  '
      vname(5,indxW)='upward_sea_water_velocity                   '
      vname(6,indxW)='lat_rho lon_rho                             '

CR      write(stdout,'(1x,A,I3)') 'indxR =',indxR
      vname(1,indxR)='rho                                         '
      vname(2,indxR)='density anomaly                             '
      vname(3,indxR)='kilogram meter-3                            '
      vname(4,indxR)='density, scalar, series                     '
      vname(5,indxR)='sea_water_sigma_t                           '
      vname(7,indxR)='                                            '

# if defined ANA_VMIX || defined BVF_MIXING \
      || defined LMD_MIXING || defined LMD_SKPP || defined LMD_BKPP \
      || defined GLS_MIXING
CR      write(stdout,'(1x,A,I3)') 'indxbvf =',indxbvf
      vname(1,indxbvf)='bvf                                       '
      vname(2,indxbvf)='Brunt Vaisala Frequency                   '
      vname(3,indxbvf)='second-1                                  '
      vname(4,indxbvf)='bvf, scalar, series                       '
      vname(5,indxbvf)='brunt_vaisala_frequency                   '
      vname(7,indxbvf)='                                          '
# endif

# ifdef VIS_COEF_3D
CR      write(stdout,'(1x,A,I3)') 'indxVisc =',indxVisc
      vname(1,indxVisc)='visc3d                                   '
      vname(2,indxVisc)='horizontal viscosity coefficient         '
#  ifdef UV_VIS2
      vname(3,indxVisc)='meter2 second-1                          '
      vname(5,indxVisc)='ocean_momentum_xy_laplacian              '
#  else
      vname(3,indxVisc)='meter4 second-1                          '
      vname(5,indxVisc)='ocean_momentum_xy_biharmonic_diffusivity '
#  endif
      vname(4,indxVisc)='visc3d, scalar, series                   '
      vname(6,indxVisc)='lat_rho lon_rho                          '
      vname(7,indxVisc)='                                         '
# endif
# ifdef DIF_COEF_3D
CR      write(stdout,'(1x,A,I3)') 'indxDiff =',indxDiff
      vname(1,indxDiff)='diff3d                                   '
      vname(2,indxDiff)='horizontal diffusivity coefficient       '
#  ifdef TS_DIF2
      vname(3,indxDiff)='meter2 second-1                          '
      vname(5,indxDiff)='ocean_tracer_xy_laplacian_diffusivity    '
#  else
      vname(3,indxDiff)='meter4 second-1                          '
      vname(5,indxDiff)='ocean_tracer_xy_biharmonic_diffusivity   '
#  endif
      vname(4,indxDiff)='diff3d, scalar, series                   '
      vname(6,indxDiff)='lat_rho lon_rho                          '
      vname(7,indxDiff)='                                         '
# endif

CR      write(stdout,'(1x,A,I3)') 'indxAkv =',indxAkv
      vname(1,indxAkv)='AKv                                       '
      vname(2,indxAkv)='vertical viscosity coefficient            '
      vname(3,indxAkv)='meter2 second-1                           '
      vname(4,indxAkv)='AKv, scalar, series                       '
      vname(5,indxAkv)='ocean_vertical_momentum_diffusivity_'
     &                                          // 'at_w_location '
      vname(6,indxAkv)='lat_rho lon_rho                           '
      vname(7,indxAkv)='                                          '

# ifdef TEMPERATURE
CR      write(stdout,'(1x,A,I3)') 'indxAkt =',indxAkt
      vname(1,indxAkt)='AKt                                       '
      vname(2,indxAkt)='temperature vertical diffusion coefficient'
      vname(3,indxAkt)='meter2 second-1                           '
      vname(4,indxAkt)='AKt, scalar, series                       '
      vname(5,indxAkt)='ocean_vertical_heat_diffusivity_'
     &                                         //  'at_w_location '
      vname(6,indxAkt)='lat_rho lon_rho                           '
      vname(7,indxAkt)='                                          '
# endif


# ifdef SALINITY
CR      write(stdout,'(1x,A,I3)') 'indxAks =',indxAks
      vname(1,indxAks)='AKs                                       '
      vname(2,indxAks)='salinity vertical diffusion coefficient   '
      vname(3,indxAks)='meter2 second-1                           '
      vname(4,indxAks)='AKs, scalar, series                       '
      vname(5,indxAks)='ocean_vertical_salt_diffusivity_'
     &               / /  'at_w_location                          '
      vname(6,indxAks)='lat_rho lon_rho                           '
      vname(7,indxAks)='                                          '
# endif
# if defined LMD_SKPP || defined GLS_MIXING
CR      write(stdout,'(1x,A,I3)') 'indxHbl =',indxHbl
      vname(1,indxHbl)='hbl                                       '
      vname(2,indxHbl)='depth of planetary boundary layer         '
      vname(3,indxHbl)='meter                                     '
      vname(4,indxHbl)='hbl, scalar, series                       '
      vname(5,indxHbl)='ocean_mixed_layer_thickness_defined_'
     &                                       // 'by_mixing_scheme '
      vname(6,indxHbl)='lat_rho lon_rho                           '
      vname(7,indxHbl)='                                          '
# endif
# ifdef LMD_BKPP
CR      write(stdout,'(1x,A,I3)') 'indxHbbl =',indxHbbl
      vname(1,indxHbbl)='hbbl                                     '
      vname(2,indxHbbl)='depth of bottom boundary layer           '
      vname(3,indxHbbl)='meter                                    '
      vname(4,indxHbbl)='hbbl, scalar, series                     '
      vname(5,indxHbbl)='                                         '
      vname(6,indxHbbl)='lat_rho lon_rho                          '
      vname(7,indxHbbl)='                                         '
# endif
# ifdef GLS_MIXING
CR      write(stdout,'(1x,A,I3)') 'indxTke =',indxTke
      vname(1,indxTke)='tke                                       '
      vname(2,indxTke)='turbulent kinetic energy                  '
      vname(3,indxTke)='meter2 second-2                           '
      vname(4,indxTke)='tke, scalar, series                       '
      vname(5,indxTke)='ocean_turbulent_kinetic_energy'
     &                                         // '_at_w_location '
      vname(6,indxTke)='lat_rho lon_rho                           '
      vname(7,indxTke)='                                          '
CR      write(stdout,'(1x,A,I3)') 'indxGls =',indxGls
      vname(1,indxGls)='gls                                       '
      vname(2,indxGls)='turbulent generic length scale            '
      vname(3,indxGls)='meter3 second-2                           '
      vname(4,indxGls)='gls, scalar, series                       '
      vname(5,indxGls)='ocean_generic_length_scale'
     &                                         // '_at_w_location '
      vname(6,indxGls)='lat_rho lon_rho                           '
      vname(7,indxGls)='                                          '
CR      write(stdout,'(1x,A,I3)') 'indxLsc =',indxLsc   
      vname(1,indxLsc)='Lscale                                    '
      vname(2,indxLsc)='vertical mixing turbulent length scale    '
      vname(3,indxLsc)='meter                                     '
      vname(4,indxLsc)='Lscale, scalar, series                    '
      vname(5,indxLsc)='ocean_turbulent_length_scale'
     &                                         // '_at_w_location '
      vname(6,indxLsc)='lat_rho lon_rho                           '
      vname(7,indxLsc)='                                          '
# endif /* GLS_MIXING */

# if defined BIOLOGY && !defined PISCES
CR      write(stdout,'(1x,A,I3)') 'indxHel =',indxHel
      vname(1,indxHel)='hel                                       '
      vname(2,indxHel)='depth of euphotic layer                   '
      vname(3,indxHel)='meter                                     '
      vname(4,indxHel)='hel, scalar, series                       '
      vname(5,indxHel)='                                          '
      vname(6,indxHel)='lat_rho lon_rho                           '
      vname(7,indxHel)='                                          '
#  if (defined BIO_NChlPZD || defined BIO_N2ChlPZD2)
CR      write(stdout,'(1x,A,I3)') 'indxChC =',indxChC
      vname(1,indxChC)='theta                                     '
      vname(2,indxChC)='Chlorophyll to Carbon ratio               '
      vname(3,indxChC)='mgChl/mgC                                 '
      vname(4,indxChC)='theta, scalar, series                     '
      vname(5,indxChC)='                                          '
      vname(6,indxChC)='lat_rho lon_rho                           '
      vname(7,indxChC)='                                          '
#   ifdef OXYGEN
CR      write(stdout,'(1x,A,I3)') 'indxU10 =',indxU10
      vname(1,indxU10)='U10                                       '
      vname(2,indxU10)='surface wind speed 10 m                   '
      vname(3,indxU10)='meter second-1                            '
      vname(4,indxU10)='surface wind speed, scalar, series        '
      vname(5,indxU10)='                                          '
      vname(6,indxU10)='lat_rho lon_rho                           '
      vname(7,indxU10)='                                          '

CR      write(stdout,'(1x,A,I3)') 'indxKvO2 =',indxKvO2
      vname(1,indxKvO2)='KvO2                                     '
      vname(2,indxKvO2)='gas transfer coef for O2                 '
      vname(3,indxKvO2)='meter second-1                           '
      vname(4,indxKvO2)='gas transfer coef for O2, scalar, series '
      vname(5,indxKvO2)='                                         '
      vname(6,indxKvO2)='lat_rho lon_rho                          '
      vname(7,indxKvO2)='                                         '

CR      write(stdout,'(1x,A,I3)') 'indxO2sat =',indxO2sat
      vname(1,indxO2sat)='O2sat                                   '
      vname(2,indxO2sat)='saturation oxygen                       '
      vname(3,indxO2sat)='mMol O2 m-3                             '
      vname(4,indxO2sat)='saturation oxygen, scalar, series       '
      vname(5,indxO2sat)='                                        '
      vname(6,indxO2sat)='lat_rho lon_rho                         '
      vname(7,indxO2sat)='                                        '
#   endif
#  endif
# endif /* BIOLOGY */
#endif /* SOLVE3D */


# ifdef EXACT_RESTART

CR      write(stdout,'(1x,A,I3)') 'indxrufrc =',indxrufrc
      vname(1,indxrufrc)='rufrc                                   '
      vname(2,indxrufrc)='barotropic forcing terms                '
      vname(3,indxrufrc)='meter second-2                          '
      vname(4,indxrufrc)='rufrc, scalar, series                   '
      vname(5,indxrufrc)='barotropic_forcing_terms                '
      vname(6,indxrufrc)='lat_u lon_u                             '
      vname(7,indxrufrc)='                                        '

CR      write(stdout,'(1x,A,I3)') 'indxrvfrc =',indxrvfrc
      vname(1,indxrvfrc)='rvfrc                                   '
      vname(2,indxrvfrc)='barotropic forcing terms                '
      vname(3,indxrvfrc)='meter second-2                          '
      vname(4,indxrvfrc)='rvfrc, scalar, series                   '
      vname(5,indxrvfrc)='barotropic_forcing_terms                '
      vname(6,indxrvfrc)='lat_v lon_v                             '
      vname(7,indxrvfrc)='                                        '

# ifdef M3FAST

CR      write(stdout,'(1x,A,I3)') 'indxru_nbq =',indxru_nbq
      vname(1,indxru_nbq)='ru_nbq                                 '
      vname(2,indxru_nbq)='Fast momentum forcing terms            '
      vname(3,indxru_nbq)='meter second-2                         '
      vname(4,indxru_nbq)='ru_nbq, scalar, series                 '
      vname(5,indxru_nbq)='Fast_momentum_forcing_terms            '
      vname(6,indxru_nbq)='lat_u lon_u                            '
      vname(7,indxru_nbq)='                                       '

CR      write(stdout,'(1x,A,I3)') 'indxrv_nbq =',indxrv_nbq
      vname(1,indxrv_nbq)='rv_nbq                                 '
      vname(2,indxrv_nbq)='Fast momentum forcing terms            '
      vname(3,indxrv_nbq)='meter second-2                         '
      vname(4,indxrv_nbq)='rv_nbq, scalar, series                 '
      vname(5,indxrv_nbq)='Fast_momentum_forcing_terms            '
      vname(6,indxrv_nbq)='lat_v lon_v                            '
      vname(7,indxrv_nbq)='                                       '

CR      write(stdout,'(1x,A,I3)') 'indxru_nbq_avg2 =',indxru_nbq_avg2
      vname(1,indxru_nbq_avg2)='ru_nbq_avg2                       '
      vname(2,indxru_nbq_avg2)='Fast momentum forcing terms       '
      vname(3,indxru_nbq_avg2)='meter second-2                    '
      vname(4,indxru_nbq_avg2)='ru_nbq_avg2, scalar, series       '
      vname(5,indxru_nbq_avg2)='Fast_momentum_forcing_terms       '
      vname(6,indxru_nbq_avg2)='lat_u lon_u                       '
      vname(7,indxru_nbq_avg2)='                                  '

CR      write(stdout,'(1x,A,I3)') 'indxrv_nbq_avg2 =',indxrv_nbq_avg2
      vname(1,indxrv_nbq_avg2)='rv_nbq_avg2                       '
      vname(2,indxrv_nbq_avg2)='Fast momentum forcing terms       '
      vname(3,indxrv_nbq_avg2)='meter second-2                    '
      vname(4,indxrv_nbq_avg2)='rv_nbq_avg2, scalar, series       '
      vname(5,indxrv_nbq_avg2)='Fast_momentum_forcing_terms       '
      vname(6,indxrv_nbq_avg2)='lat_v lon_v                       '
      vname(7,indxrv_nbq_avg2)='                                  '

CR      write(stdout,'(1x,A,I3)') 'indxqdmu_nbq =',indxqdmu_nbq
      vname(1,indxqdmu_nbq)='qdmu_nbq                             '
      vname(2,indxqdmu_nbq)='Fast momentum forcing terms          '
      vname(3,indxqdmu_nbq)='meter second-2                       '
      vname(4,indxqdmu_nbq)='qdmu_nbq, scalar, series             '
      vname(5,indxqdmu_nbq)='Fast_momentum_forcing_terms          '
      vname(6,indxqdmu_nbq)='lat_u lon_u                          '
      vname(7,indxqdmu_nbq)='                                     '

CR      write(stdout,'(1x,A,I3)') 'indxqdmv_nbq =',indxqdmv_nbq
      vname(1,indxqdmv_nbq)='qdmv_nbq                             '
      vname(2,indxqdmv_nbq)='Fast momentum forcing terms          '
      vname(3,indxqdmv_nbq)='meter second-2                       '
      vname(4,indxqdmv_nbq)='qdmv_nbq, scalar, series             '
      vname(5,indxqdmv_nbq)='Fast_momentum_forcing_terms          '
      vname(6,indxqdmv_nbq)='lat_v lon_v                          '
      vname(7,indxqdmv_nbq)='                                     '

# endif  /* M3FAST */

# ifdef TS_MIX_ISO_FILT

!
! density gradients for use in t3dmix
!
CR      write(stdout,'(1x,A,I3)') 'indxdRdx =',indxdRdx
      vname(1,indxdRdx)='dRdx                                 '
      vname(2,indxdRdx)='Density gradient                     '
      vname(3,indxdRdx)='                                     '
      vname(4,indxdRdx)='dRdx, scalar, series                 '
      vname(5,indxdRdx)='Density_gradient                     '
      vname(6,indxdRdx)='lat_u lon_u                          '
      vname(7,indxdRdx)='                                     '

CR      write(stdout,'(1x,A,I3)') 'indxdRde =',indxdRde
      vname(1,indxdRde)='dRde                                 '
      vname(2,indxdRde)='Density_gradient                     '
      vname(3,indxdRde)='meter second-2                       '
      vname(4,indxdRde)='dRde, scalar, series                 '
      vname(5,indxdRde)='Density_gradient                     '
      vname(6,indxdRde)='lat_v lon_v                          '
      vname(7,indxdRde)='                                     '


# endif

# endif /* EXACT_RESTART */

CR      write(stdout,'(1x,A,I3)') 'indxSSH =',indxSSH
      vname(1,indxSSH)='SSH                                       '
      vname(2,indxSSH)='sea surface height                        '
      vname(3,indxSSH)='meter                                     '
      vname(4,indxSSH)='SSH, scalar, series                       '
      vname(5,indxSSH)='sea_surface_height_above_sea_level        '
      vname(6,indxSSH)='lat_rho lon_rho                           '
      vname(7,indxSSH)='                                          '

CR      write(stdout,'(1x,A,I3)') 'indxSUSTR =',indxSUSTR
      vname(1,indxSUSTR)='sustr                                   '
      vname(2,indxSUSTR)='surface u-momentum stress               '
      vname(3,indxSUSTR)='Newton meter-2                          '
      vname(4,indxSUSTR)='surface u-mom. stress, scalar, series   '
      vname(5,indxSUSTR)='surface_downward_x_stress               '
      vname(6,indxSUSTR)='lat_u lon_u                             '
      vname(7,indxSUSTR)='                                        '

CR      write(stdout,'(1x,A,I3)') 'indxSVSTR =',indxSVSTR
      vname(1,indxSVSTR)='svstr                                   '
      vname(2,indxSVSTR)='surface v-momentum stress               '
      vname(3,indxSVSTR)='Newton meter-2                          '
      vname(4,indxSVSTR)='surface v-mom. stress, scalar, series   '
      vname(5,indxSVSTR)='surface_downward_y_stress               '     
      vname(6,indxSVSTR)='lat_v lon_v                             '
      vname(7,indxSVSTR)='                                        '

#ifdef SOLVE3D
!# if !defined BULK_FLUX
# ifdef QCORRECTION
CR      write(stdout,'(1x,A,I3)') 'indxSST =',indxSST
      vname(1,indxSST)='SST                                       '
      vname(2,indxSST)='sea surface temperature                   '
      vname(3,indxSST)='Celsius                                   '
      vname(4,indxSST)='SST, scalar, series                       '
      vname(5,indxSST)='sea_surface_temperature                   '
      vname(6,indxSST)='lat_rho lon_rho                           '
      vname(7,indxSST)='                                          '
      vname(8,indxSST)=' '

CR      write(stdout,'(1x,A,I3)') 'indxdQdSST =',indxdQdSST
      vname(1,indxdQdSST)='dQdSST                                 '
      vname(2,indxdQdSST)='surface heat flux sensitivity to SST   '
      vname(3,indxdQdSST)='Watts meter-2 Celsius-1                '
      vname(4,indxdQdSST)='dQdSST, scalar, series                 '
      vname(5,indxdQdSST)='heat_flux_correction                   '
      vname(6,indxdQdSST)='lat_rho lon_rho                        '
      vname(7,indxdQdSST)='                                       '
# endif
# ifdef PSOURCE_NCFILE
CR      write(stdout,'(1x,A,I3)') 'indxQBAR =',indxQBAR
      vname(1,indxQBAR)='Qbar                                     '
      vname(2,indxQBAR)='Runoff Discharge                         '
      vname(3,indxQBAR)='m3.s-1                                   '
      vname(4,indxQBAR)='Qbar, scalar, series                     '
#  ifdef PSOURCE_NCFILE_TS
CR      write(stdout,'(1x,A,I3)') 'indxTsrc(1) =',indxTsrc
        vname(1,indxTsrc)='temp_src                               '
        vname(2,indxTsrc)='Temp. source runoff                    '
        vname(3,indxTsrc)='deg. Celsius                           '
        vname(4,indxTsrc)='Tsrc(1), scalar, series                '

CR      write(stdout,'(1x,A,I3)') 'indxTsrc(2) =',indxTsrc+1
        vname(1,indxTsrc+1)='salt_src                             '
        vname(2,indxTsrc+1)='Salt. runoff                         '
        vname(3,indxTsrc+1)='psu                                  '
        vname(4,indxTsrc+1)='Tsrc(2), scalar, series              '
#   ifdef BIOLOGY
        do itrc=3,NT
          nametrc=vname(1,indxT+itrc-1)
          unitt=vname(3,indxT+itrc-1)
          write(vname1,*) trim(nametrc),'_src                     '
          vname(1,indxTsrc+itrc-1)=vname1
CR       write(stdout,'(20x,A)') vname(1,indxTVmix+itrc-1)
          vname(2,indxTsrc+itrc-1)='Tracer source concentration   '
          write(vname3,*) trim(unitt),'                           '
          vname(3,indxTsrc+itrc-1)=vname3
          vname(4,indxTsrc+itrc-1)='                              '
          vname(5,indxTsrc+itrc-1)='                              '
          vname(6,indxTsrc+itrc-1)='                              '
          vname(7,indxTsrc+itrc-1)='                              '
        enddo
#   endif
#  endif
# endif

#if defined BHFLUX
CR      write(stdout,'(1x,A,I3)') 'indxBhflx =',indxBhflx
      vname(1,indxBhflx)='bhflux                                 '
      vname(2,indxBhflx)='bottom net heat flux                   '
      vname(3,indxBhflx)='Watts meter-2                          '
      vname(4,indxBhflx)='bottom heat flux, scalar, series       '
      vname(5,indxBhflx)='                                       '
      vname(6,indxBhflx)='lat_rho lon_rho                        '
#endif
#if defined BWFLUX && defined SALINITY
CR      write(stdout,'(1x,A,I3)') 'indxBwflx =',indxBwflx
      vname(1,indxBwflx)='bwflux                                 '
      vname(2,indxBwflx)='bottom freshwater flux (E-P)          '
      vname(3,indxBwflx)='centimeter day-1                       '
      vname(4,indxBwflx)='bottom freshwater flux, scalar, series'
      vname(5,indxBwflx)='                                       '
      vname(6,indxBwflx)='lat_rho lon_rho                        '
#endif
# if defined SALINITY && defined SFLX_CORR
CR      write(stdout,'(1x,A,I3)') 'indxSSS =',indxSSS
      vname(1,indxSSS)='SSS                                       '
      vname(2,indxSSS)='sea surface salinity                      '
      vname(3,indxSSS)='PSU                                       '
      vname(4,indxSSS)='SSS, scalar, series                       '
      vname(5,indxSSS)='sea_surface_salinity                      '
      vname(6,indxSSS)='lat_rho lon_rho                           '
      vname(7,indxSSS)='                                          '
# endif

# if defined BBL
#  if (!defined ANA_BSEDIM && !defined SEDIMENT)
CR      write(stdout,'(1x,A,I3)') 'indxBSS =',indxBSS
      vname(1,indxBSS)='Ssize                                     '
      vname(2,indxBSS)='bottom sediment grain diameter size       '
      vname(3,indxBSS)='meter                                     '
      vname(4,indxBSS)='Ssize, scalar, series                     '
      vname(5,indxBSS)='                                          '
      vname(6,indxBSS)='lat_rho lon_rho                           '
      vname(7,indxBSS)='                                          '

CR      write(stdout,'(1x,A,I3)') 'indxBSD =',indxBSD
      vname(1,indxBSD)='Sdens                                     '
      vname(2,indxBSD)='bottom sediment grain density             '
      vname(3,indxBSD)='kilogram meter-3                          '
      vname(4,indxBSD)='Sdens, scalar, series                     '
      vname(5,indxBSD)='                                          '
      vname(6,indxBSD)='lat_rho lon_rho                           '
      vname(7,indxBSD)='                                          '
#  endif /* !SEDIMENT */
# endif /* BBL */
#endif /* SOLVE3D */

#if (defined MUSTANG || defined MRL_WCI || defined BBL) \
     && !defined  ANA_WWAVE

CR      write(stdout,'(1x,A,I3)') 'indxWWA',indxWWA

# ifdef MUSTANG
      vname(1,indxWWA)='hs                                        '
      vname(2,indxWWA)='significant wave height                   '
      vname(3,indxWWA)='meter                                     '
      vname(4,indxWWA)='Awave,scalar, series                      '
      vname(5,indxWWA)='                                          '
      vname(6,indxWWA)='lat_rho lon_rho                           '

CR      write(stdout,'(1x,A,I3)') 'indxWWD =',indxWWD
      vname(1,indxWWD)='dir                                       '
      vname(2,indxWWD)='wave direction                            '
      vname(3,indxWWD)='degrees                                   '
      vname(4,indxWWD)='Dwave, scalar, series                     '
      vname(5,indxWWD)='                                          '
      vname(6,indxWWD)='lat_rho lon_rho                           '
      vname(7,indxWWD)='                                          '

CR      write(stdout,'(1x,A,I3)') 'indxWWP =',indxWWP
      vname(1,indxWWP)='t01                                       '
      vname(2,indxWWP)='wave Period                               '
      vname(3,indxWWP)='second                                    '
      vname(4,indxWWP)='Pwave, scalar, series                     '
      vname(5,indxWWP)='                                          '
      vname(6,indxWWP)='lat_rho lon_rho                           '
      vname(7,indxWWP)='                                          '

      vname(1,indxWWU)='ubr                                       '
      vname(2,indxWWU)='wave orbital velocity                     '
      vname(3,indxWWU)='meter per second                          '
      vname(4,indxWWU)='Uwave,scalar, series                      '
      vname(5,indxWWU)='                                          '
      vname(6,indxWWU)='lat_rho lon_rho                           '
      vname(7,indxWWU)='                                          '
# else
CR      write(stdout,'(1x,A,I3)') 'indxWWA',indxWWA
      vname(1,indxWWA)='Awave                                     '
      vname(2,indxWWA)='wind induced wave amplitude               '
      vname(3,indxWWA)='meter                                     '
      vname(4,indxWWA)='Awave,scalar, series                      '
      vname(5,indxWWA)='                                          '
      vname(6,indxWWA)='lat_rho lon_rho                           '

CR      write(stdout,'(1x,A,I3)') 'indxWWD =',indxWWD
      vname(1,indxWWD)='Dwave                                     '
      vname(2,indxWWD)='wind induced wave direction               '
      vname(3,indxWWD)='degrees                                   '
      vname(4,indxWWD)='Dwave, scalar, series                     '
      vname(5,indxWWD)='                                          '
      vname(6,indxWWD)='lat_rho lon_rho                           '
      vname(7,indxWWD)='                                          '

CR      write(stdout,'(1x,A,I3)') 'indxWWP =',indxWWP
      vname(1,indxWWP)='Pwave                                     '
      vname(2,indxWWP)='wind induced wave Period                  '
      vname(3,indxWWP)='second                                    '
      vname(4,indxWWP)='Pwave, scalar, series                     '
      vname(5,indxWWP)='                                          '
      vname(6,indxWWP)='lat_rho lon_rho                           '
      vname(7,indxWWP)='                                          '
# endif  /* MUSTANG */

# ifdef WAVE_OFFLINE_BREAKING
CR      write(stdout,'(1x,A,I3)') 'indxWEB =',indxWEB
      vname(1,indxWEB)='Eb                                        '
      vname(2,indxWEB)='breaking dissipation                      '
      vname(3,indxWEB)='m3s-3                                     '
      vname(4,indxWEB)='Eb, scalar, series                        '
      vname(5,indxWEB)='                                          '
      vname(6,indxWEB)='lat_rho lon_rho                           '
      vname(7,indxWEB)='                                          '
# endif /* WAVE_OFFLINE_BREAKING */
# ifdef WAVE_OFFLINE_FRICTION
CR      write(stdout,'(1x,A,I3)') 'indxWED =',indxWED
      vname(1,indxWED)='Ed                                        '
      vname(2,indxWED)='frictional dissipation                    '
      vname(3,indxWED)='m3s-3                                     '
      vname(4,indxWED)='Ed, scalar, series                        '
      vname(5,indxWED)='                                          '
      vname(6,indxWED)='lat_rho lon_rho                           '
      vname(7,indxWED)='                                          '
# endif /* WAVE_OFFLINE_BREAKING */
# ifdef WAVE_OFFLINE_ROLLER
CR      write(stdout,'(1x,A,I3)') 'indxWER =',indxWER
      vname(1,indxWER)='Er                                        '
      vname(2,indxWER)='roller dissipation                        '
      vname(3,indxWER)='m3s-3                                     '
      vname(4,indxWER)='Er, scalar, series                        '
      vname(5,indxWER)='                                          '
      vname(6,indxWER)='lat_rho lon_rho                           '
      vname(7,indxWER)='                                          '
# endif /* WAVE_OFFLINE_BREAKING */
#endif /* !ANA_WWAVE */

#if defined WKB_WWAVE || defined OW_COUPLING \
	              || (defined WAVE_OFFLINE && defined MRL_WCI)
CR      write(stdout,'(1x,A,I3)') 'indxHRM =',indxHRM
      vname(1,indxHRM)='hrm                                       '
      vname(2,indxHRM)='RMS wave height                           '
      vname(3,indxHRM)='meter                                     '

CR      write(stdout,'(1x,A,I3)') 'indxFRQ =',indxFRQ
      vname(1,indxFRQ)='frq                                       '
      vname(2,indxFRQ)='wave frequency                            '
      vname(3,indxFRQ)='rad/second                                '

CR      write(stdout,'(1x,A,I3)') 'indxWAC =',indxWAC
      vname(1,indxWAC)='wac                                       '
      vname(2,indxWAC)='wave action                               '
      vname(3,indxWAC)='meter^3 second^-1                         '

CR      write(stdout,'(1x,A,I3)') 'indxWKX =',indxWKX
      vname(1,indxWKX)='wkx                                       '
      vname(2,indxWKX)='xi-dir wavenumber vector                  '
      vname(3,indxWKX)='radian/meter                              '

CR      write(stdout,'(1x,A,I3)') 'indxWKE =',indxWKE
      vname(1,indxWKE)='wke                                       '
      vname(2,indxWKE)='eta-dir wavenumber vector                 '
      vname(3,indxWKE)='radian/meter                              '

CR      write(stdout,'(1x,A,I3)') 'indxEPB =',indxEPB
      vname(1,indxEPB)='epb                                       '
      vname(2,indxEPB)='breaking dissipation, epsilon_b           '
      vname(3,indxEPB)='meter^3/second^3                          '

CR      write(stdout,'(1x,A,I3)') 'indxEPD =',indxEPD
      vname(1,indxEPD)='epd                                       '
      vname(2,indxEPD)='frictional dissipation, epsilon_d         '
      vname(3,indxEPD)='meter^3/second^3                          '

CR      write(stdout,'(1x,A,I3)') 'indxWAR =',indxWAR
      vname(1,indxWAR)='war                                       '
      vname(2,indxWAR)='wave action of roller component           '
      vname(3,indxWAR)='meter^3 second^-1                         '

CR      write(stdout,'(1x,A,I3)') 'indxEPR =',indxEPR
      vname(1,indxEPR)='epr                                       '
      vname(2,indxEPR)='roller dissipation, epsilon_r             '
      vname(3,indxEPR)='meter^3/second^3                          '
#endif /* WKB_WWAVE || OW_COUPLING || WAVE_OFFLINE */

#ifdef MRL_WCI
CR      write(stdout,'(1x,A,I3)') 'indxSUP =',indxSUP
      vname(1,indxSUP)  ='sup                                     '
      vname(2,indxSUP)  ='quasi-static sea-level response (setup) '
      vname(3,indxSUP)  ='meter                                   '

CR      write(stdout,'(1x,A,I3)') 'indxUST2D =',indxUST2D
      vname(1,indxUST2D)='ust2d                                   '
      vname(2,indxUST2D)='u depth-averaged Stokes drift velocity  '
      vname(3,indxUST2D)='meter second-1                          '

CR      write(stdout,'(1x,A,I3)') 'indxVST2D =',indxVST2D
      vname(1,indxVST2D)='vst2d                                   '
      vname(2,indxVST2D)='v depth-averaged Stokes drift velocity  '
      vname(3,indxVST2D)='meter second-1                          '
# ifdef SOLVE3D
CR      write(stdout,'(1x,A,I3)') 'indxUST =',indxUST
      vname(1,indxUST)  ='ust                                     '
      vname(2,indxUST)  ='u 3D Stokes drift velocity              '
      vname(3,indxUST)  ='meter second-1                          '

CR      write(stdout,'(1x,A,I3)') 'indxVST =',indxVST
      vname(1,indxVST)  ='vst                                     '
      vname(2,indxVST)  ='v 3D Stokes drift velocity              '
      vname(3,indxVST)  ='meter second-1                          '

CR      write(stdout,'(1x,A,I3)') 'indxWST =',indxWST
      vname(1,indxWST)  ='wst                                     '
      vname(2,indxWST)  ='w vertical 3D Stokes drift velocity     '
      vname(3,indxWST)  ='meter second-1                          '

CR      write(stdout,'(1x,A,I3)') 'indxAKB =',indxAKB
      vname(1,indxAKB)  ='Akb                                     '
      vname(2,indxAKB)  ='eddy viscosity due to wave breaking     '
      vname(3,indxAKB)  ='meter2 second-1                         '

CR      write(stdout,'(1x,A,I3)') 'indxAKW =',indxAKW
      vname(1,indxAKW)  ='Akw                                     '
      vname(2,indxAKW)  ='eddy diffusivity due to primary waves   '
      vname(3,indxAKW)  ='meter2 second-1                         '

CR      write(stdout,'(1x,A,I3)') 'indxKVF =',indxKVF
      vname(1,indxKVF)  ='kvf                                     '
      vname(2,indxKVF)  ='vertical vortex force (u^st du/dz)      '
      vname(3,indxKVF)  ='meter second-2                          '

CR      write(stdout,'(1x,A,I3)') 'indxCALP =',indxCALP
      vname(1,indxCALP) ='calP                                    '
      vname(2,indxCALP) ='surface pressure correction in prsgrd   '
      vname(3,indxCALP) ='meter2 second-2                         '

CR      write(stdout,'(1x,A,I3)') 'indxKAPS =',indxKAPS
      vname(1,indxKAPS) ='Kapsrf                                  '
      vname(2,indxKAPS) ='surface Bernoulli head in prsgrd term   '
      vname(3,indxKAPS) ='meter2 second-2                         '
# endif
#endif /* MRL_WCI */

#ifdef ICE
      vname(1,indxAi)='aice                                       '
      vname(2,indxAi)='fraction of cell covered by ice            '
      vname(3,indxAi)='nondimensional                             '
      vname(4,indxAi)='aice, scalar, series                       '
      vname(5,indxAi)='                                           '
      vname(6,indxAi)='lat_rho lon_rho                            '
      vname(7,indxAi)='                                           '
      vname(1,indxUi)='uice                                       '
      vname(2,indxUi)='u-component of ice velocity                '
      vname(3,indxUi)='meter sec-1                                '
      vname(4,indxUi)='uice, scalar, series                       '
      vname(5,indxUi)='                                           '
      vname(6,indxUi)='lat_u lon_u                                '
      vname(7,indxUi)='                                           '

      vname(1,indxVi)='vice                                       '
      vname(2,indxVi)='v-component of ice velocity                '
      vname(3,indxVi)='meter sec-1                                '
      vname(4,indxVi)='vice, scalar, series                       '
      vname(5,indxVi)='                                           '
      vname(6,indxVi)='lat_v lon_v                                '
      vname(7,indxVi)='                                           '

      vname(1,indxHi)='hice                                       '
      vname(2,indxHi)='depth of ice cover                         '
      vname(3,indxHi)='meter                                      '
      vname(4,indxHi)='hice, scalar, series                       '
      vname(5,indxHi)='                                           '
      vname(6,indxHi)='lat_rho lon_rho                            '
      vname(7,indxHi)='                                           '

      vname(1,indxHS)='snow_thick                                 '
      vname(2,indxHS)='depth of snow cover                        '
      vname(3,indxHS)='meter                                      '
      vname(4,indxHS)='snow_thick, scalar, series                 '
      vname(5,indxHS)='                                           '
      vname(6,indxHS)='lat_rho lon_rho                            '
      vname(7,indxHS)='                                           '

      vname(1,indxTIsrf)='tisrf                                   '
      vname(2,indxTIsrf)='temperature of ice surface              '
      vname(3,indxTIsrf)='Kelvin                                  '
      vname(4,indxTIsrf)='tsrf, scalar, series                    '
      vname(5,indxTIsrf)='                                        '
      vname(6,indxTIsrf)='lat_rho lon_rho                         '
      vname(7,indxTIsrf)='                                        '
#endif

#ifdef BULK_FLUX
# ifdef ONLINE /* JG ONLINE */
      do blkvar_id=1,nblkvrs
        itbulkO(blkvar_id)=0
        newbulk(blkvar_id)=.true.
        bulk_timeO(1:2,blkvar_id)=-1.E+20
        bulk_recO(blkvar_id)=0
      enddo
      call get_bulkvar_names
# else
CR      write(stdout,'(1x,A,I3)') 'indxWSPD =',indxWSPD
      vname(1,indxWSPD)='wspd                                     '
      vname(2,indxWSPD)='surface wind speed 10 m                  '
      vname(3,indxWSPD)='meter second-1                           '
      vname(4,indxWSPD)='surface wind speed, scalar, series       '
      vname(5,indxWSPD)='wind_speed                               '
      vname(6,indxWSPD)='lat_rho lon_rho                          '
      vname(7,indxWSPD)='                                         '

CR      write(stdout,'(1x,A,I3)') 'indxTAIR =',indxTAIR
      vname(1,indxTAIR)='tair                                     '
      vname(2,indxTAIR)='surface air temperature 2m               '
      vname(3,indxTAIR)='Celsius                                  '
      vname(4,indxTAIR)='surface air temperature, scalar, series  '
      vname(5,indxTAIR)='air_temperature_at_2m                    '
      vname(6,indxTAIR)='lat_rho lon_rho                          '
      vname(7,indxTAIR)='                                         '

CR      write(stdout,'(1x,A,I3)') 'indxRHUM =',indxRHUM
      vname(1,indxRHUM)='rhum                                     '
      vname(2,indxRHUM)='surface air relative humidity 2m         '
      vname(3,indxRHUM)='fraction                                 '
      vname(4,indxRHUM)='surface relative humidity, scalar, series'
      vname(5,indxRHUM)='relative_humidity_at_2m                  '
      vname(6,indxRHUM)='lat_rho lon_rho                          '
      vname(7,indxRHUM)='                                         '

# ifndef BULK_LW
CR      write(stdout,'(1x,A,I3)') 'indxRADLW =',indxRADLW
      vname(1,indxRADLW)='radlw                                   '
      vname(2,indxRADLW)='net terrestrial longwave radiation      '
      vname(3,indxRADLW)='Watts meter-2                           '
      vname(4,indxRADLW)='terrestrial longwave, scalar, series    '
      vname(5,indxRADLW)='surface_net_downward_longwave_flux      '
      vname(6,indxRADLW)='                                        '
      vname(7,indxRADLW)='                                        '
# else
CR      write(stdout,'(1x,A,I3)') 'indxRADLW =',indxRADLW
      vname(1,indxRADLW)='radlw_in                                '
      vname(2,indxRADLW)='downward longwave radiation             '
      vname(3,indxRADLW)='Watts meter-2                           '
      vname(4,indxRADLW)='downward longwave, scalar, series       '
      vname(5,indxRADLW)='surface_net_downward_longwave_flux      '
      vname(6,indxRADLW)='                                        '
      vname(7,indxRADLW)='                                        '
# endif  /* BULK_LW */


CR      write(stdout,'(1x,A,I3)') 'indxPRATE =',indxPRATE
      vname(1,indxPRATE)='prate                                   '
      vname(2,indxPRATE)='surface precipitation rate              '
      vname(3,indxPRATE)='Kg meter-2 second-1                     '
      vname(4,indxPRATE)='precipitation rate, scalar, series      '
      vname(5,indxPRATE)='                                        '
      vname(6,indxPRATE)='lat_rho lon_rho                         '
      vname(7,indxPRATE)='                                        '

CR      write(stdout,'(1x,A,I3)') 'indxUWND =',indxUWND
      vname(1,indxUWND)='uwnd                                     '
      vname(2,indxUWND)='surface u-wind speed 10 m                '
      vname(3,indxUWND)='meter second-1                           '
      vname(4,indxUWND)='surface wind speed, scalar, series       '
      vname(5,indxUWND)='x_wind                                   '
      vname(6,indxUWND)='lat_u lon_u                              '
      vname(7,indxUWND)='                                         '

CR      write(stdout,'(1x,A,I3)') 'indxVWND =',indxVWND
      vname(1,indxVWND)='vwnd                                     '
      vname(2,indxVWND)='surface v-wind speed 10 m                '
      vname(3,indxVWND)='meter second-1                           '
      vname(4,indxVWND)='surface wind speed, scalar, series       '
      vname(5,indxVWND)='y_wind                                   '
      vname(6,indxVWND)='lat_v lon_v                              '
      vname(7,indxVWND)='                                         '



# endif /* ONLINE */
#endif /* BULK_FLUX */

#ifdef DIAGNOSTICS_TS
      do itrc=1,NT

CR       write(stdout,'(1x,A,I3)') 'indxTXadv = ',indxTXadv+itrc-1
       lvar=lenstr(vname(1,indxT+itrc-1))
       nametrc=vname(1,indxT+itrc-1)(1:lvar)
       lvar=lenstr(nametrc)
       unitt=vname(3,indxT+itrc-1)
       write(vname1,*) trim(nametrc),'_xadv                       '
       vname(1,indxTXadv+itrc-1)=vname1
CR       write(stdout,'(20x,A)') vname(1,indxTXadv+itrc-1)
       vname(2,indxTXadv+itrc-1)='Horizontal (xi) advection term  '
       write(vname3,*) trim(unitt),' second-1                     '
       vname(3,indxTXadv+itrc-1)=vname3
       vname(4,indxTXadv+itrc-1)='                                '
       vname(5,indxTXadv+itrc-1)='                                '
       vname(7,indxTXadv+itrc-1)='                                '

CR       write(stdout,'(1x,A,I3)') 'indxTYadv = ',indxTYadv+itrc-1
       nametrc=vname(1,indxT+itrc-1)
       unitt=vname(3,indxT+itrc-1)
       write(vname1,*) trim(nametrc),'_yadv                       '
       vname(1,indxTYadv+itrc-1)=vname1
CR       write(stdout,'(20x,A)') vname(1,indxTYadv+itrc-1)
       vname(2,indxTYadv+itrc-1)='Horizontal (eta) advection term '
       write(vname3,*) trim(unitt),' second-1                     '
       vname(3,indxTYadv+itrc-1)=vname3
       vname(4,indxTYadv+itrc-1)='                                '
       vname(5,indxTYadv+itrc-1)='                                '
       vname(6,indxTYadv+itrc-1)='                                '
       vname(7,indxTYadv+itrc-1)='                                '

CR       write(stdout,'(1x,A,I3)') 'indxTVadv = ',indxTVadv+itrc-1
       nametrc=vname(1,indxT+itrc-1)
       unitt=vname(3,indxT+itrc-1)
       write(vname1,*) trim(nametrc),'_vadv                       '
       vname(1,indxTVadv+itrc-1)=vname1
CR       write(stdout,'(20x,A)') vname(1,indxTVadv+itrc-1)
       vname(2,indxTVadv+itrc-1)='Vertical advection term         '
       write(vname3,*) trim(unitt),' second-1                     '
       vname(3,indxTVadv+itrc-1)=vname3
       vname(4,indxTVadv+itrc-1)='                                '
       vname(5,indxTVadv+itrc-1)='                                '
       vname(6,indxTVadv+itrc-1)='                                '
       vname(7,indxTVadv+itrc-1)='                                '

CR       write(stdout,'(1x,A,I3)') 'indxTHmix = ',indxTHmix+itrc-1
       nametrc=vname(1,indxT+itrc-1)
       unitt=vname(3,indxT+itrc-1)
       write(vname1,*) trim(nametrc),'_hmix                       '
       vname(1,indxTHmix+itrc-1)=vname1
CR       write(stdout,'(20x,A)') vname(1,indxTHmix+itrc-1)
       vname(2,indxTHmix+itrc-1)='Horizontal mixing term          '
       write(vname3,*) trim(unitt),' second-1                     '
       vname(3,indxTHmix+itrc-1)=vname3
       vname(4,indxTHmix+itrc-1)='                                '
       vname(5,indxTHmix+itrc-1)='                                '
       vname(6,indxTHmix+itrc-1)='                                '
       vname(7,indxTHmix+itrc-1)='                                '

CR       write(stdout,'(1x,A,I3)') 'indxTVmix = ',indxTVmix+itrc-1
       nametrc=vname(1,indxT+itrc-1)
       unitt=vname(3,indxT+itrc-1)
       write(vname1,*) trim(nametrc),'_vmix                       '
       vname(1,indxTVmix+itrc-1)=vname1
CR       write(stdout,'(20x,A)') vname(1,indxTVmix+itrc-1)
       vname(2,indxTVmix+itrc-1)='Vertical mixing term            '
       write(vname3,*) trim(unitt),' second-1                     '
       vname(3,indxTVmix+itrc-1)=vname3
       vname(4,indxTVmix+itrc-1)='                                '
       vname(5,indxTVmix+itrc-1)='                                '
       vname(6,indxTVmix+itrc-1)='                                '
       vname(7,indxTVmix+itrc-1)='                                '

# ifdef DIAGNOSTICS_TSVAR

CR       write(stdout,'(1x,A,I3)') 'indxTVmixt = ',indxTVmixt+itrc-1
       nametrc=vname(1,indxT+itrc-1)
       unitt=vname(3,indxT+itrc-1)
       write(vname1,*) trim(nametrc),'_vmix_trans                 '
       vname(1,indxTVmixt+itrc-1)=vname1
CR       write(stdout,'(20x,A)') vname(1,indxTVmixt+itrc-1)
       vname(2,indxTVmixt+itrc-1)='Vertical mixing term            '
       write(vname3,*) trim(unitt),' second-1                     '
       vname(3,indxTVmixt+itrc-1)=vname3
       vname(4,indxTVmixt+itrc-1)='                                '
       vname(5,indxTVmixt+itrc-1)='                                '
       vname(6,indxTVmixt+itrc-1)='                                '
       vname(7,indxTVmixt+itrc-1)='                                '

# endif

CR       write(stdout,'(1x,A,I3)') 'indxTForc = ',indxTForc+itrc-1
       nametrc=vname(1,indxT+itrc-1)
       unitt=vname(3,indxT+itrc-1)
       write(vname1,*) trim(nametrc),'_forc                       '
       vname(1,indxTForc+itrc-1)=vname1
CR       write(stdout,'(20x,A)') vname(1,indxTForc+itrc-1)
       vname(2,indxTForc+itrc-1)='Forcing term (Q & Nudging)      '
       write(vname3,*) trim(unitt),' second-1                     '
       vname(3,indxTForc+itrc-1)=vname3
       vname(4,indxTForc+itrc-1)='                                '
       vname(5,indxTForc+itrc-1)='                                '
       vname(6,indxTForc+itrc-1)='                                '
       vname(7,indxTForc+itrc-1)='                                '

CR       write(stdout,'(1x,A,I3)') 'indxTrate = ',indxTrate+itrc-1
       nametrc=vname(1,indxT+itrc-1)
       unitt=vname(3,indxT+itrc-1)
       write(vname1,*) trim(nametrc),'_rate                       '
       vname(1,indxTrate+itrc-1)=vname1
CR       write(stdout,'(20x,A)') vname(1,indxTrate+itrc-1)
       vname(2,indxTrate+itrc-1)='Time rate of change             '
       write(vname3,*) trim(unitt),' second-1                     '
       vname(3,indxTrate+itrc-1)=vname3
       vname(4,indxTrate+itrc-1)='                                '
       vname(5,indxTrate+itrc-1)='                                '
       vname(6,indxTrate+itrc-1)='                                '
       vname(7,indxTrate+itrc-1)='                                '
!
! Tracer diagnostics variables averaged over the MLD
!
# ifdef DIAGNOSTICS_TS_MLD
CR       write(stdout,'(1x,A,I3)') 'indxTXadv_mld = ',indxTXadv_mld+itrc-1
       lvar=lenstr(vname(1,indxT+itrc-1))
       nametrc=vname(1,indxT+itrc-1)(1:lvar)
       lvar=lenstr(nametrc)
       unitt=vname(3,indxT+itrc-1)
       write(vname1,*) trim(nametrc),'_xadv_mld                   '
       vname(1,indxTXadv_mld+itrc-1)=vname1
CR       write(stdout,'(20x,A)') vname(1,indxTXadv_mld+itrc-1)
       vname(2,indxTXadv_mld+itrc-1)='MLD xi advection term       '
       write(vname3,*) trim(unitt),' second-1                     '
       vname(3,indxTXadv_mld+itrc-1)=vname3
       vname(4,indxTXadv_mld+itrc-1)='                            '
       vname(5,indxTXadv_mld+itrc-1)='                            '
       vname(6,indxTXadv_mld+itrc-1)='                            '
       vname(7,indxTXadv_mld+itrc-1)='                            '

CR       write(stdout,'(1x,A,I3)') 'indxTYadv_mld = ',indxTYadv_mld+itrc-1
       nametrc=vname(1,indxT+itrc-1)
       unitt=vname(3,indxT+itrc-1)
       write(vname1,*) trim(nametrc),'_yadv_mld                   '
       vname(1,indxTYadv_mld+itrc-1)=vname1
CR       write(stdout,'(20x,A)') vname(1,indxTYadv_mld+itrc-1)
       vname(2,indxTYadv_mld+itrc-1)='MLD eta advection term      '
       write(vname3,*) trim(unitt),' second-1                     '
       vname(3,indxTYadv_mld+itrc-1)=vname3
       vname(4,indxTYadv_mld+itrc-1)='                            '
       vname(5,indxTYadv_mld+itrc-1)='                            '
       vname(6,indxTYadv_mld+itrc-1)='                            '
       vname(7,indxTYadv_mld+itrc-1)='                            '

CR       write(stdout,'(1x,A,I3)') 'indxTVadv_mld = ',indxTVadv_mld+itrc-1
       nametrc=vname(1,indxT+itrc-1)
       unitt=vname(3,indxT+itrc-1)
       write(vname1,*) trim(nametrc),'_vadv_mld                   '
       vname(1,indxTVadv_mld+itrc-1)=vname1
CR       write(stdout,'(20x,A)') vname(1,indxTVadv+itrc-1)
       vname(2,indxTVadv_mld+itrc-1)='MLD vertical advection term '
       write(vname3,*) trim(unitt),' second-1                     '
       vname(3,indxTVadv_mld+itrc-1)=vname3
       vname(4,indxTVadv_mld+itrc-1)='                            '
       vname(5,indxTVadv_mld+itrc-1)='                            '
       vname(6,indxTVadv_mld+itrc-1)='                            '
       vname(7,indxTVadv_mld+itrc-1)='                            '

CR       write(stdout,'(1x,A,I3)') 'indxTHmix_mld = ',indxTHmix_mld+itrc-1
       nametrc=vname(1,indxT+itrc-1)
       unitt=vname(3,indxT+itrc-1)
       write(vname1,*) trim(nametrc),'_hmix_mld                   '
       vname(1,indxTHmix_mld+itrc-1)=vname1
CR       write(stdout,'(20x,A)') vname(1,indxTHmix_mld+itrc-1)
       vname(2,indxTHmix_mld+itrc-1)='MLD xi mixing term          '
       write(vname3,*) trim(unitt),' second-1                     '
       vname(3,indxTHmix_mld+itrc-1)=vname3
       vname(4,indxTHmix_mld+itrc-1)='                            '
       vname(5,indxTHmix_mld+itrc-1)='                            '
       vname(6,indxTHmix_mld+itrc-1)='                            '
       vname(7,indxTHmix_mld+itrc-1)='                            '

CR       write(stdout,'(1x,A,I3)') 'indxTVmix_mld = ',indxTVmix_mld+itrc-1
       nametrc=vname(1,indxT+itrc-1)
       unitt=vname(3,indxT+itrc-1)
       write(vname1,*) trim(nametrc),'_vmix_mld                   '
       vname(1,indxTVmix_mld+itrc-1)=vname1
CR       write(stdout,'(20x,A)') vname(1,indxTVmix_mld+itrc-1)
       vname(2,indxTVmix_mld+itrc-1)='MLD vertical mixing term    '
       write(vname3,*) trim(unitt),' second-1                     '
       vname(3,indxTVmix_mld+itrc-1)=vname3
       vname(4,indxTVmix_mld+itrc-1)='                            '
       vname(5,indxTVmix_mld+itrc-1)='                            '
       vname(6,indxTVmix_mld+itrc-1)='                            '
       vname(7,indxTVmix_mld+itrc-1)='                            '

CR       write(stdout,'(1x,A,I3)') 'indxTForc_mld = ',indxTForc_mld+itrc-1
       nametrc=vname(1,indxT+itrc-1)
       unitt=vname(3,indxT+itrc-1)
       write(vname1,*) trim(nametrc),'_forc_mld                   '
       vname(1,indxTForc_mld+itrc-1)=vname1
CR       write(stdout,'(20x,A)') vname(1,indxTForc_mld+itrc-1)
       vname(2,indxTForc_mld+itrc-1)='MLD Forcing term (Q & Nudg) '
       write(vname3,*) trim(unitt),' second-1                     '
       vname(3,indxTForc_mld+itrc-1)=vname3
       vname(4,indxTForc_mld+itrc-1)='                            '
       vname(5,indxTForc_mld+itrc-1)='                            '
       vname(6,indxTForc_mld+itrc-1)='                            '
       vname(7,indxTForc_mld+itrc-1)='                            '

CR       write(stdout,'(1x,A,I3)') 'indxTrate_mld = ',indxTrate_mld+itrc-1
       nametrc=vname(1,indxT+itrc-1)
       unitt=vname(3,indxT+itrc-1)
       write(vname1,*) trim(nametrc),'_rate_mld                   '
       vname(1,indxTrate_mld+itrc-1)=vname1
CR       write(stdout,'(20x,A)') vname(1,indxTrate_mld+itrc-1)
       vname(2,indxTrate_mld+itrc-1)='MLD time rate of change     '
       write(vname3,*) trim(unitt),' second-1                     '
       vname(3,indxTrate_mld+itrc-1)=vname3
       vname(4,indxTrate_mld+itrc-1)='                            '
       vname(5,indxTrate_mld+itrc-1)='                            '
       vname(6,indxTrate_mld+itrc-1)='                            '
       vname(7,indxTrate_mld+itrc-1)='                            '

CR       write(stdout,'(1x,A,I3)') 'indxTentr_mld = ',indxTentr_mld+itrc-1
       nametrc=vname(1,indxT+itrc-1)
       unitt=vname(3,indxT+itrc-1)
       write(vname1,*) trim(nametrc),'_entr_mld                   '
       vname(1,indxTentr_mld+itrc-1)=vname1
CR       write(stdout,'(20x,A)') vname(1,indxTentr_mld+itrc-1)
       vname(2,indxTentr_mld+itrc-1)='MLD entrainment rate        '
       write(vname3,*) trim(unitt),' second-1                     '
       vname(3,indxTentr_mld+itrc-1)=vname3
       vname(4,indxTentr_mld+itrc-1)='                            '
       vname(5,indxTentr_mld+itrc-1)='                            '
       vname(6,indxTentr_mld+itrc-1)='                            '
       vname(7,indxTentr_mld+itrc-1)='                            '
# endif /* DIAGNOSTICS_TS_MLD */

      enddo  ! itrc
#endif  /* DIAGNOSTICS_TS */

#ifdef DIAGNOSTICS_UV
      do itrc=1,2
CR       write(stdout,'(1x,A,I3)') 'indxMXadv = ',indxMXadv+itrc-1
       lvar=lenstr(vname(1,indxU+itrc-1))
       nametrc=vname(1,indxU+itrc-1)(1:lvar)
       lvar=lenstr(nametrc)
       unitt=vname(3,indxU+itrc-1)
       write(vname1,*) trim(nametrc),'_xadv                       '
       vname(1,indxMXadv+itrc-1)=vname1
CR       write(stdout,'(20x,A)') vname(1,indxMxadv+itrc-1)
       vname(2,indxMXadv+itrc-1)='Horizontal (xi) advection term  '
       write(vname3,*) trim(unitt),' second-1                     '
       vname(3,indxMXadv+itrc-1)=vname3
       vname(4,indxMXadv+itrc-1)='                                '
       write(vname3,*) 'horizontal_Xi_advection_term_',trim(nametrc)
     &               //'_component'
       vname(5,indxMXadv+itrc-1)=vname3
       vname(6,indxMXadv+itrc-1)='                                '
       vname(7,indxMXadv+itrc-1)='                                '

CR       write(stdout,'(1x,A,I3)') 'indxMYadv = ',indxMYadv+itrc-1
       nametrc=vname(1,indxU+itrc-1)
       unitt=vname(3,indxU+itrc-1)
       write(vname1,*) trim(nametrc),'_yadv                       '
       vname(1,indxMYadv+itrc-1)=vname1
CR       write(stdout,'(20x,A)') vname(1,indxMYadv+itrc-1)
       vname(2,indxMYadv+itrc-1)='Horizontal (eta) advection term '
       write(vname3,*) trim(unitt),' second-1                     '
       vname(3,indxMYadv+itrc-1)=vname3
       vname(4,indxMYadv+itrc-1)='                                '
       write(vname3,*) 'horizontal_eta_advection_term_',trim(nametrc)
     &               //'_component'
       vname(5,indxMYadv+itrc-1)=vname3
       vname(6,indxMYadv+itrc-1)='                                '
       vname(7,indxMYadv+itrc-1)='                                '

CR       write(stdout,'(1x,A,I3)') 'indxMVadv = ',indxMVadv+itrc-1
                    nametrc=vname(1,indxU+itrc-1)
       unitt=vname(3,indxU+itrc-1)
       write(vname1,*) trim(nametrc),'_vadv                       '
       vname(1,indxMVadv+itrc-1)=vname1
CR       write(stdout,'(20x,A)') vname(1,indxMVadv+itrc-1)
       vname(2,indxMVadv+itrc-1)='Vertical advection term         '
       write(vname3,*) trim(unitt),' second-1                     '
       vname(3,indxMVadv+itrc-1)=vname3
       vname(4,indxMVadv+itrc-1)='                                '
       write(vname3,*) 'vertical_advection_term_',trim(nametrc)
     &               //'_component'
       vname(5,indxMVadv+itrc-1)=vname3
       vname(6,indxMVadv+itrc-1)='                                '
       vname(7,indxMVadv+itrc-1)='                                '

CR       write(stdout,'(1x,A,I3)') 'indxMCor = ',indxMCor+itrc-1
       nametrc=vname(1,indxU+itrc-1)
       unitt=vname(3,indxU+itrc-1)
       write(vname1,*) trim(nametrc),'_cor                        '
       vname(1,indxMCor+itrc-1)=vname1
CR       write(stdout,'(20x,A)') vname(1,indxMcor+itrc-1)
       vname(2,indxMCor+itrc-1)='Coriolis term                    '
       write(vname3,*) trim(unitt),' second-1                     '
       vname(3,indxMCor+itrc-1)=vname3
       vname(4,indxMCor+itrc-1)='                                 '
       write(vname3,*) 'coriolis_term_in_momentum_',trim(nametrc)
     &               //'_component'
       vname(5,indxMCor+itrc-1)=vname3
       vname(6,indxMCor+itrc-1)='                                 '
       vname(7,indxMCor+itrc-1)='                                 '

CR      write(stdout,'(1x,A,I3)') 'indxMPrsgrd = ',indxMPrsgrd+itrc-1
       nametrc=vname(1,indxU+itrc-1)
       unitt=vname(3,indxU+itrc-1)
       write(vname1,*) trim(nametrc),'_prsgrd                     '
       vname(1,indxMPrsgrd+itrc-1)=vname1
CR       write(stdout,'(20x,A)') vname(1,indxMPrsgrd+itrc-1)
       vname(2,indxMPrsgrd+itrc-1)='Pressure gradient term        '
       write(vname3,*) trim(unitt),' second-1                     '
       vname(3,indxMPrsgrd+itrc-1)=vname3
       vname(4,indxMPrsgrd+itrc-1)='                              '
       write(vname3,*) 'pressure_gradient_term_',trim(nametrc)
     &               //'_component'
       vname(5,indxMPrsgrd+itrc-1)=vname3
       vname(6,indxMPrsgrd+itrc-1)='                              '
       vname(7,indxMPrsgrd+itrc-1)='                              '

CR       write(stdout,'(1x,A,I3)') 'indxMHmix = ',indxMHmix+itrc-1
       nametrc=vname(1,indxU+itrc-1)
       unitt=vname(3,indxU+itrc-1)
       write(vname1,*) trim(nametrc),'_hmix                       '
       vname(1,indxMHmix+itrc-1)=vname1
C     R       write(stdout,'(20x,A)') vname(1,indxMHmix+itrc-1)
       vname(2,indxMHmix+itrc-1)='Horizontal mixing term          '
       write(vname3,*) trim(unitt),' second-1                     '
       vname(3,indxMHmix+itrc-1)=vname3
       vname(4,indxMHmix+itrc-1)='                                '
       write(vname3,*) 'horizontal_mixing_term_',trim(nametrc)
     &               //'_component'
       vname(5,indxMHmix+itrc-1)=vname3
       vname(6,indxMHmix+itrc-1)='                                '
       vname(7,indxMHmix+itrc-1)='                                '

CR       write(stdout,'(1x,A,I3)') 'indxMHdiff = ',indxMHdiff+itrc-1
       nametrc=vname(1,indxU+itrc-1)
       unitt=vname(3,indxU+itrc-1)
       write(vname1,*) trim(nametrc),'_hdiff                      '
       vname(1,indxMHdiff+itrc-1)=vname1
C     R       write(stdout,'(20x,A)') vname(1,indxMHdiff+itrc-1)
       vname(2,indxMHdiff+itrc-1)='Horizontal diffusion term      '
       write(vname3,*) trim(unitt),' second-1                     '
       vname(3,indxMHdiff+itrc-1)=vname3
       vname(4,indxMHdiff+itrc-1)='                               '
       write(vname3,*) 'horizontal_diffusion_term_',trim(nametrc)
     &               //'_component'
       vname(5,indxMHdiff+itrc-1)=vname3
       vname(6,indxMHdiff+itrc-1)='                               '
       vname(7,indxMHdiff+itrc-1)='                               '

CR       write(stdout,'(1x,A,I3)') 'indxMVmix = ',indxMVmix+itrc-1
       nametrc=vname(1,indxU+itrc-1)
       unitt=vname(3,indxU+itrc-1)
       write(vname1,*) trim(nametrc),'_vmix                       '
       vname(1,indxMVmix+itrc-1)=vname1
CR       write(stdout,'(20x,A)') vname(1,indxMVmix+itrc-1)
       vname(2,indxMVmix+itrc-1)='Vertical mixing term            '
       write(vname3,*) trim(unitt),' second-1                     '
       vname(3,indxMVmix+itrc-1)=vname3
       vname(4,indxMVmix+itrc-1)='                                '
       write(vname3,*) 'vertical_mixing_term_',trim(nametrc)
     &               //'_component'
       vname(5,indxMVmix+itrc-1)=vname3
       vname(6,indxMVmix+itrc-1)='                                '
       vname(7,indxMVmix+itrc-1)='                                '

CR       write(stdout,'(1x,A,I3)') 'indxMVmix2 = ',indxMVmix2+itrc-1
       nametrc=vname(1,indxU+itrc-1)
       unitt=vname(3,indxU+itrc-1)
       write(vname1,*) trim(nametrc),'_nudg                       '
       vname(1,indxMVmix2+itrc-1)=vname1
CR       write(stdout,'(20x,A)') vname(1,indxMVmix2+itrc-1)
       vname(2,indxMVmix2+itrc-1)='Momentum nudging term          '
       write(vname3,*) trim(unitt),' second-1                     '
       vname(3,indxMVmix2+itrc-1)=vname3
       vname(4,indxMVmix2+itrc-1)='                               '
       write(vname3,*) 'momentum_nudging_term_',trim(nametrc)
     &               //'_component'
       vname(5,indxMVmix2+itrc-1)=vname3
       vname(6,indxMVmix2+itrc-1)='                               '
       vname(7,indxMVmix2+itrc-1)='                               '

# if defined DIAGNOSTICS_BARO
CR      write(stdout,'(1x,A,I3)') 'indxMBaro = ',indxMBaro+itrc-1
       nametrc=vname(1,indxU+itrc-1)
       unitt=vname(3,indxU+itrc-1)
       write(vname1,*) trim(nametrc),'_baro                       '
       vname(1,indxMBaro+itrc-1)=vname1
CR       write(stdout,'(20x,A)') vname(1,indxMBaro+itrc-1)
       vname(2,indxMBaro+itrc-1)='Barotropic coupling term        '
       write(vname3,*) trim(unitt),' second-1                     '
       vname(3,indxMBaro+itrc-1)=vname3
       vname(4,indxMBaro+itrc-1)='                                '
       write(vname3,*) 'barotropic_coupling_term_',trim(nametrc)
     &               //'_component'
       vname(5,indxMBaro+itrc-1)=vname3
       vname(6,indxMBaro+itrc-1)='                                '
       vname(7,indxMBaro+itrc-1)='                                '
# endif

# if defined M3FAST
CR      write(stdout,'(1x,A,I3)') 'indxMfast = ',indxMfast+itrc-1
       nametrc=vname(1,indxU+itrc-1)
       unitt=vname(3,indxU+itrc-1)
       write(vname1,*) trim(nametrc),'_fast                       '
       vname(1,indxMfast+itrc-1)=vname1
CR       write(stdout,'(20x,A)') vname(1,indxMfast+itrc-1)
       vname(2,indxMfast+itrc-1)='fast term                       '
       write(vname3,*) trim(unitt),' second-1                     '
       vname(3,indxMfast+itrc-1)=vname3
       vname(4,indxMfast+itrc-1)='                                '
       write(vname3,*) 'fast_term_',trim(nametrc)
     &               //'_component'
       vname(5,indxMfast+itrc-1)=vname3
       vname(6,indxMfast+itrc-1)='                                '
       vname(7,indxMfast+itrc-1)='                                '
# endif

# ifdef MRL_WCI
CR       write(stdout,'(1x,A,I3)') 'indxMvf = ',indxMvf+itrc-1
       nametrc=vname(1,indxU+itrc-1)
       unitt=vname(3,indxU+itrc-1)
       write(vname1,*) trim(nametrc),'_vf                         '
       vname(1,indxMvf+itrc-1)=vname1
CR       write(stdout,'(20x,A)') vname(1,indxMvf+itrc-1)
       vname(2,indxMvf+itrc-1)='Vortex body force term            '
       write(vname3,*) trim(unitt),' second-1                     '
       vname(3,indxMvf+itrc-1)=vname3
       vname(4,indxMvf+itrc-1)='                                  '
       vname(5,indxMvf+itrc-1)=vname1
       vname(6,indxMvf+itrc-1)='                                  '
       vname(7,indxMvf+itrc-1)='                                  '

CR       write(stdout,'(1x,A,I3)') 'indxMbrk = ',indxMbrk+itrc-1
       nametrc=vname(1,indxU+itrc-1)
       unitt=vname(3,indxU+itrc-1)
       write(vname1,*) trim(nametrc),'_brk                        '
       vname(1,indxMbrk+itrc-1)=vname1
CR       write(stdout,'(20x,A)') vname(1,indxMbrk+itrc-1)
       vname(2,indxMbrk+itrc-1)='Breaking body force term         '
       write(vname3,*) trim(unitt),' second-1                     '
       vname(3,indxMbrk+itrc-1)=vname3
       vname(4,indxMbrk+itrc-1)='                                 '
       vname(5,indxMbrk+itrc-1)=vname1
       vname(6,indxMbrk+itrc-1)='                                 '
       vname(7,indxMbrk+itrc-1)='                                 '

CR       write(stdout,'(1x,A,I3)') 'indxMStCo = ',indxMStCo+itrc-1
       nametrc=vname(1,indxU+itrc-1)
       unitt=vname(3,indxU+itrc-1)
       write(vname1,*) trim(nametrc),'_StCo                       '
       vname(1,indxMStCo+itrc-1)=vname1
CR       write(stdout,'(20x,A)') vname(1,indxMStCo+itrc-1)
       vname(2,indxMStCo+itrc-1)='Stokes-Coriolis term            '
       write(vname3,*) trim(unitt),' second-1                     '
       vname(3,indxMStCo+itrc-1)=vname3
       vname(4,indxMStCo+itrc-1)='                                '
       vname(5,indxMStCo+itrc-1)=vname1
       vname(6,indxMStCo+itrc-1)='                                '
       vname(7,indxMStCo+itrc-1)='                                '

CR       write(stdout,'(1x,A,I3)') 'indxMVvf = ',indxMVvf+itrc-1
       nametrc=vname(1,indxU+itrc-1)
       unitt=vname(3,indxU+itrc-1)
       write(vname1,*) trim(nametrc),'_Vvf                        '
       vname(1,indxMVvf+itrc-1)=vname1
CR       write(stdout,'(20x,A)') vname(1,indxMVvf+itrc-1)
       vname(2,indxMVvf+itrc-1)='Vertical vortex force term       '
       write(vname3,*) trim(unitt),' second-1                     '
       vname(3,indxMVvf+itrc-1)=vname3
       vname(4,indxMVvf+itrc-1)='                                 '
       vname(5,indxMVvf+itrc-1)=vname1
       vname(6,indxMVvf+itrc-1)='                                 '
       vname(7,indxMVvf+itrc-1)='                                 '

CR       write(stdout,'(1x,A,I3)') 'indxMPrscrt = ',indxMPrscrt+itrc-1
       nametrc=vname(1,indxU+itrc-1)
       unitt=vname(3,indxU+itrc-1)
       write(vname1,*) trim(nametrc),'_Prscrt                     '
       vname(1,indxMPrscrt+itrc-1)=vname1
CR       write(stdout,'(20x,A)') vname(1,indxMPrscrt+itrc-1)
       vname(2,indxMPrscrt+itrc-1)='Pressure correction term      '
       write(vname3,*) trim(unitt),' second-1                     '
       vname(3,indxMPrscrt+itrc-1)=vname3
       vname(4,indxMPrscrt+itrc-1)='                              '
       vname(5,indxMPrscrt+itrc-1)=vname1
       vname(6,indxMPrscrt+itrc-1)='                              '
       vname(7,indxMPrscrt+itrc-1)='                              '

CR       write(stdout,'(1x,A,I3)') 'indxMsbk = ',indxMsbk+itrc-1
       nametrc=vname(1,indxU+itrc-1)
       unitt=vname(3,indxU+itrc-1)
       write(vname1,*) trim(nametrc),'_sbk                        '
       vname(1,indxMsbk+itrc-1)=vname1
CR       write(stdout,'(20x,A)') vname(1,indxMsbk+itrc-1)
       vname(2,indxMsbk+itrc-1)='Surface breaking term            '
       write(vname3,*) trim(unitt),' second-1                     '
       vname(3,indxMsbk+itrc-1)=vname3
       vname(4,indxMsbk+itrc-1)='                                 '
       vname(5,indxMsbk+itrc-1)=vname1
       vname(6,indxMsbk+itrc-1)='                                 '
       vname(7,indxMsbk+itrc-1)='                                 '

CR       write(stdout,'(1x,A,I3)') 'indxMbwf = ',indxMbwf+itrc-1
       nametrc=vname(1,indxU+itrc-1)
       unitt=vname(3,indxU+itrc-1)
       write(vname1,*) trim(nametrc),'_bwf                        '
       vname(1,indxMbwf+itrc-1)=vname1
CR       write(stdout,'(20x,A)') vname(1,indxMbwf+itrc-1)
       vname(2,indxMbwf+itrc-1)='Bed wave friction term           '
       write(vname3,*) trim(unitt),' second-1                     '
       vname(3,indxMbwf+itrc-1)=vname3
       vname(4,indxMbwf+itrc-1)='                                 '
       vname(5,indxMbwf+itrc-1)=vname1
       vname(6,indxMbwf+itrc-1)='                                 '
       vname(7,indxMbwf+itrc-1)='                                 '

CR       write(stdout,'(1x,A,I3)') 'indxMfrc = ',indxMfrc+itrc-1
       nametrc=vname(1,indxU+itrc-1)
       unitt=vname(3,indxU+itrc-1)
       write(vname1,*) trim(nametrc),'_frc                        '
       vname(1,indxMfrc+itrc-1)=vname1
CR       write(stdout,'(20x,A)') vname(1,indxMfrc+itrc-1)
       vname(2,indxMfrc+itrc-1)='Wave streaming                   '
       write(vname3,*) trim(unitt),' second-1                     '
       vname(3,indxMfrc+itrc-1)=vname3
       vname(4,indxMfrc+itrc-1)='                                 '
       vname(5,indxMfrc+itrc-1)=vname1
       vname(6,indxMfrc+itrc-1)='                                 '
       vname(7,indxMfrc+itrc-1)='                                 '
# endif  /* MRL_WCI */

CR       write(stdout,'(1x,A,I3)') 'indxMrate = ',indxMrate+itrc-1
       nametrc=vname(1,indxU+itrc-1)
       unitt=vname(3,indxU+itrc-1)
       write(vname1,*) trim(nametrc),'_rate                       '
       vname(1,indxMrate+itrc-1)=vname1
CR       write(stdout,'(20x,A)') vname(1,indxMrate+itrc-1)
       vname(2,indxMrate+itrc-1)='Time rate of change             '
       write(vname3,*) trim(unitt),' second-1                     '
       vname(3,indxMrate+itrc-1)=vname3
       vname(4,indxMrate+itrc-1)='                                '
       write(vname3,*) 'time_rate_of_change_term_',trim(nametrc)
     &               //'_component'
       vname(5,indxMrate+itrc-1)=vname3
       vname(6,indxMrate+itrc-1)='                                '
       vname(7,indxMrate+itrc-1)='                                '
      enddo
#endif /* DIAGNOSTICS_UV */

#ifdef DIAGNOSTICS_VRT
CR       write(stdout,'(1x,A,I3)') 'indxvrtXadv = ',indxvrtXadv+itrc-1
       nametrc='vrt'
       lvar=lenstr(nametrc)
       unitt=vname(3,indxU)
       write(vname1,*) trim(nametrc),'_xadv                       '
       vname(1,indxvrtXadv)=vname1
CR       write(stdout,'(20x,A)') vname(1,indxvrtxadv)
       vname(2,indxvrtXadv)='Horizontal (xi) advection term       '
       write(vname3,*) trim(unitt),' second-1                     '
       vname(3,indxvrtXadv)=vname3

CR       write(stdout,'(1x,A,I3)') 'indxvrtYadv = ',indxvrtYadv+itrc-1
       nametrc='vrt'
       unitt=vname(3,indxU)
       write(vname1,*) trim(nametrc),'_yadv                       '
       vname(1,indxvrtYadv)=vname1
CR       write(stdout,'(20x,A)') vname(1,indxvrtYadv)
       vname(2,indxvrtYadv)='Horizontal (eta) advection term      '
       write(vname3,*) trim(unitt),' second-1                     '
       vname(3,indxvrtYadv)=vname3

CR       write(stdout,'(1x,A,I3)') 'indxvrtHdiff = ',indxvrtHdiff+itrc-1
       nametrc='vrt'
       unitt=vname(3,indxU)
       write(vname1,*) trim(nametrc),'_hdiff                      '
       vname(1,indxvrtHdiff)=vname1
CR       write(stdout,'(20x,A)') vname(1,indxvrtHdiff)
       vname(2,indxvrtHdiff)='Horizontal diffusion term (implicit)'
       write(vname3,*) trim(unitt),' second-1                     '
       vname(3,indxvrtHdiff)=vname3

CR       write(stdout,'(1x,A,I3)') 'indxvrtCor = ',indxvrtCor+itrc-1
       nametrc='vrt'
       unitt=vname(3,indxU)
       write(vname1,*) trim(nametrc),'_cor                        '
       vname(1,indxvrtCor)=vname1
CR       write(stdout,'(20x,A)') vname(1,indxvrtCor)
       vname(2,indxvrtCor)='Coriolis term                         '
       write(vname3,*) trim(unitt),' second-1                     '
       vname(3,indxvrtCor)=vname3

CR       write(stdout,'(1x,A,I3)') 'indxvrtPrsgrd = ',indxvrtPrsgrd+itrc-1
       nametrc='vrt'
       unitt=vname(3,indxU)
       write(vname1,*) trim(nametrc),'_prsgrd                     '
       vname(1,indxvrtPrsgrd)=vname1
CR       write(stdout,'(20x,A)') vname(1,indxvrtPrsgrd)
       vname(2,indxvrtPrsgrd)='Pressure gradient term             '
       write(vname3,*) trim(unitt),' second-1                     '
       vname(3,indxvrtPrsgrd)=vname3

CR       write(stdout,'(1x,A,I3)') 'indxvrtHmix = ',indxvrtHmix+itrc-1
       nametrc='vrt'
       unitt=vname(3,indxU)
       write(vname1,*) trim(nametrc),'_hmix                       '
       vname(1,indxvrtHmix)=vname1
CR       write(stdout,'(20x,A)') vname(1,indxvrtHmix)
       vname(2,indxvrtHmix)='Horizontal mixing term               '
       write(vname3,*) trim(unitt),' second-1                     '
       vname(3,indxvrtHmix)=vname3

CR       write(stdout,'(1x,A,I3)') 'indxvrtVmix = ',indxvrtVmix+itrc-1
       nametrc='vrt'
       unitt=vname(3,indxU)
       write(vname1,*) trim(nametrc),'_vmix                       '
       vname(1,indxvrtVmix)=vname1
CR       write(stdout,'(20x,A)') vname(1,indxvrtVmix)
       vname(2,indxvrtVmix)='Vertical mixing term                 '
       write(vname3,*) trim(unitt),' second-1                     '
       vname(3,indxvrtVmix)=vname3

CR       write(stdout,'(1x,A,I3)') 'indxvrtrate = ',indxvrtrate+itrc-1
       nametrc='vrt'
       unitt=vname(3,indxU)
       write(vname1,*) trim(nametrc),'_rate                       '
       vname(1,indxvrtrate)=vname1
CR       write(stdout,'(20x,A)') vname(1,indxvrtrate)
       vname(2,indxvrtrate)='Time rate of change                  '
       write(vname3,*) trim(unitt),' second-1                     '
       vname(3,indxvrtrate)=vname3

CR       write(stdout,'(1x,A,I3)') 'indxvrtVmix2 = ',indxvrtVmix2+itrc-1
       nametrc='vrt'
       unitt=vname(3,indxU)
       write(vname1,*) trim(nametrc),'_nudg                       '
       vname(1,indxvrtVmix2)=vname1
CR       write(stdout,'(20x,A)') vname(1,indxvrtVmix2)
       vname(2,indxvrtVmix2)='Momentum nudging term               '
       write(vname3,*) trim(unitt),' second-1                     '
       vname(3,indxvrtVmix2)=vname3

CR       write(stdout,'(1x,A,I3)') 'indxvrtWind = ',indxvrtWind+itrc-1
       nametrc='vrt'
       unitt=vname(3,indxU)
       write(vname1,*) trim(nametrc),'_wind                       '
       vname(1,indxvrtWind)=vname1
CR       write(stdout,'(20x,A)') vname(1,indxvrtWind)
       vname(2,indxvrtWind)='Wind stress curl                     '
       vname(3,indxvrtWind)=' meter second-2                      '

CR       write(stdout,'(1x,A,I3)') 'indxvrtDrag = ',indxvrtDrag+itrc-1
       nametrc='vrt'
       unitt=vname(3,indxU)
       write(vname1,*) trim(nametrc),'_drag                       '
       vname(1,indxvrtDrag)=vname1
CR       write(stdout,'(20x,A)') vname(1,indxvrtDrag)
       vname(2,indxvrtDrag)='Bottom drag curl                     '
       vname(3,indxvrtDrag)=' meter second-2                      '

# if defined DIAGNOSTICS_BARO
CR       write(stdout,'(1x,A,I3)') 'indxvrtBaro = ',indxvrtBaro+itrc-1
       nametrc='vrt'
       unitt=vname(3,indxU)
       write(vname1,*) trim(nametrc),'_baro                       '
       vname(1,indxvrtBaro)=vname1
CR       write(stdout,'(20x,A)') vname(1,indxvrtBaro)
       vname(2,indxvrtBaro)='Barotropic coupling term             '
       write(vname3,*) trim(unitt),' second-1                     '
       vname(3,indxvrtBaro)=vname3
# endif

# if defined M3FAST
CR       write(stdout,'(1x,A,I3)') 'indxvrtfast = ',indxvrtfast+itrc-1
       nametrc='vrt'
       unitt=vname(3,indxU)
       write(vname1,*) trim(nametrc),'_fast                       '
       vname(1,indxvrtfast)=vname1
CR       write(stdout,'(20x,A)') vname(1,indxvrtfast)
       vname(2,indxvrtfast)='Fast term                            '
       write(vname3,*) trim(unitt),' second-1                     '
       vname(3,indxvrtfast)=vname3
# endif

#endif /* DIAGNOSTICS_VRT */


#ifdef DIAGNOSTICS_EK

       nametrc='ke'

CR       write(stdout,'(1x,A,I3)') 'indxekHadv = ',indxekHadv+itrc-1
       unitt=vname(3,indxU)
       write(vname1,*) trim(nametrc),'_hadv                       '
       vname(1,indxekHadv)=vname1
CR       write(stdout,'(20x,A)') vname(1,indxekHadv)
       vname(2,indxekHadv)='Horizontal  advection term            '
       write(vname3,*) trim(unitt),' meter second-2               '
       vname(3,indxekHadv)=vname3

CR       write(stdout,'(1x,A,I3)') 'indxekHdiff = ',indxekHdiff+itrc-1
       unitt=vname(3,indxU)
       write(vname1,*) trim(nametrc),'_hdiff                      '
       vname(1,indxekHdiff)=vname1
CR       write(stdout,'(20x,A)') vname(1,indxekHdiff)
       vname(2,indxekHdiff)='Implicit diffusion term              '
       write(vname3,*) trim(unitt),' meter second-2               '
       vname(3,indxekHdiff)=vname3

CR       write(stdout,'(1x,A,I3)') 'indxekVadv = ',indxekVadv+itrc-1
       unitt=vname(3,indxU)
       write(vname1,*) trim(nametrc),'_vadv                       '
       vname(1,indxekVadv)=vname1
CR       write(stdout,'(20x,A)') vname(1,indxekVadv)
       vname(2,indxekVadv)='Vertical advection term               '
       write(vname3,*) trim(unitt),' meter second-2               '
       vname(3,indxekVadv)=vname3

CR       write(stdout,'(1x,A,I3)') 'indxekCor = ',indxekCor+itrc-1
       unitt=vname(3,indxU)
       write(vname1,*) trim(nametrc),'_cor                        '
       vname(1,indxekCor)=vname1
CR       write(stdout,'(20x,A)') vname(1,indxekCor)
       vname(2,indxekCor)='Coriolis term                          '
       write(vname3,*) trim(unitt),' meter second-2               '
       vname(3,indxekCor)=vname3

CR       write(stdout,'(1x,A,I3)') 'indxekPrsgrd = ',indxekPrsgrd+itrc-1
       unitt=vname(3,indxU)
       write(vname1,*) trim(nametrc),'_prsgrd                     '
       vname(1,indxekPrsgrd)=vname1
CR       write(stdout,'(20x,A)') vname(1,indxekPrsgrd)
       vname(2,indxekPrsgrd)='Pressure gradient term              '
       write(vname3,*) trim(unitt),' meter second-2               '
       vname(3,indxekPrsgrd)=vname3

CR       write(stdout,'(1x,A,I3)') 'indxekHmix = ',indxekHmix+itrc-1
       unitt=vname(3,indxU)
       write(vname1,*) trim(nametrc),'_hmix                       '
       vname(1,indxekHmix)=vname1
CR       write(stdout,'(20x,A)') vname(1,indxekHmix)
       vname(2,indxekHmix)='Horizontal mixing term                '
       write(vname3,*) trim(unitt),' meter second-2               '
       vname(3,indxekHmix)=vname3

CR       write(stdout,'(1x,A,I3)') 'indxekVmix = ',indxekVmix+itrc-1
       unitt=vname(3,indxU)
       write(vname1,*) trim(nametrc),'_vmix                       '
       vname(1,indxekVmix)=vname1
CR       write(stdout,'(20x,A)') vname(1,indxekVmix)
       vname(2,indxekVmix)='Vertical mixing term                  '
       write(vname3,*) trim(unitt),' meter second-2               '
       vname(3,indxekVmix)=vname3

CR       write(stdout,'(1x,A,I3)') 'indxekrate = ',indxekrate+itrc-1
       unitt=vname(3,indxU)
       write(vname1,*) trim(nametrc),'_rate                       '
       vname(1,indxekrate)=vname1
CR       write(stdout,'(20x,A)') vname(1,indxekrate)
       vname(2,indxekrate)='Time rate of change                   '
       write(vname3,*) trim(unitt),' meter second-2               '
       vname(3,indxekrate)=vname3

CR       write(stdout,'(1x,A,I3)') 'indxekvol = ',indxekvol+itrc-1
       unitt=vname(3,indxU)
       write(vname1,*) trim(nametrc),'_vol                        '
       vname(1,indxekvol)=vname1
CR       write(stdout,'(20x,A)') vname(1,indxekvol)
       vname(2,indxekvol)='Volume change                          '
       write(vname3,*) trim(unitt),' meter second-2               '
       vname(3,indxekvol)=vname3

CR       write(stdout,'(1x,A,I3)') 'indxekVmix2 = ',indxekVmix2+itrc-1
       unitt=vname(3,indxU)
       write(vname1,*) trim(nametrc),'_nudg                       '
       vname(1,indxekVmix2)=vname1
CR       write(stdout,'(20x,A)') vname(1,indxekVmix2)
       vname(2,indxekVmix2)='Momentum nudging term                '
       write(vname3,*) trim(unitt),' meter second-2               '
       vname(3,indxekVmix2)=vname3

CR       write(stdout,'(1x,A,I3)') 'indxekWind = ',indxekWind+itrc-1
       unitt=vname(3,indxU)
       write(vname1,*) trim(nametrc),'_wind                       '
       vname(1,indxekWind)=vname1
CR       write(stdout,'(20x,A)') vname(1,indxekWind)
       vname(2,indxekWind)='Wind stress                           '
       write(vname3,*) trim(unitt),' meter second-2               '
       vname(3,indxekWind)=vname3

CR       write(stdout,'(1x,A,I3)') 'indxekDrag = ',indxekDrag+itrc-1
       unitt=vname(3,indxU)
       write(vname1,*) trim(nametrc),'_drag                       '
       vname(1,indxekDrag)=vname1
CR       write(stdout,'(20x,A)') vname(1,indxekDrag)
       vname(2,indxekDrag)='Bottom drag                           '
       write(vname3,*) trim(unitt),' meter second-2               '
       vname(3,indxekDrag)=vname3

# if defined DIAGNOSTICS_BARO
CR       write(stdout,'(1x,A,I3)') 'indxekBaro = ',indxekBaro+itrc-1
       unitt=vname(3,indxU)
       write(vname1,*) trim(nametrc),'_baro                       '
       vname(1,indxekBaro)=vname1
CR       write(stdout,'(20x,A)') vname(1,indxekBaro)
       vname(2,indxekBaro)='Barotropic coupling term              '
       write(vname3,*) trim(unitt),' meter second-2               '
       vname(3,indxekBaro)=vname3
# endif

# if defined M3FAST
CR       write(stdout,'(1x,A,I3)') 'indxekfast = ',indxekfast+itrc-1
       unitt=vname(3,indxU)
       write(vname1,*) trim(nametrc),'_fast                       '
       vname(1,indxekfast)=vname1
CR       write(stdout,'(20x,A)') vname(1,indxekfast)
       vname(2,indxekfast)='Fast term                             '
       write(vname3,*) trim(unitt),' meter second-2               '
       vname(3,indxekfast)=vname3
# endif

# ifdef DIAGNOSTICS_EK_MLD
CR       write(stdout,'(1x,A,I3)') 'indxekHadv = ',indxekHadv+itrc-1
       unitt=vname(3,indxU)
       write(vname1,*) trim(nametrc),'_hadv_mld                   '
       vname(1,indxekHadv_mld)=vname1
CR       write(stdout,'(20x,A)') vname(1,indxekHadv)
       vname(2,indxekHadv_mld)='Horizontal  advection term        '
       write(vname3,*) trim(unitt),' meter second-2               '
       vname(3,indxekHadv_mld)=vname3

CR       write(stdout,'(1x,A,I3)') 'indxekHdiff = ',indxekHdiff+itrc-1
       unitt=vname(3,indxU)
       write(vname1,*) trim(nametrc),'_hdiff_mld                  '
       vname(1,indxekHdiff_mld)=vname1
CR       write(stdout,'(20x,A)') vname(1,indxekHdiff)
       vname(2,indxekHdiff_mld)='Implicit diffusion term          '
       write(vname3,*) trim(unitt),' meter second-2               '
       vname(3,indxekHdiff_mld)=vname3

CR       write(stdout,'(1x,A,I3)') 'indxekVadv = ',indxekVadv+itrc-1
       unitt=vname(3,indxU)
       write(vname1,*) trim(nametrc),'_vadv_mld                   '
       vname(1,indxekVadv_mld)=vname1
CR       write(stdout,'(20x,A)') vname(1,indxekVadv)
       vname(2,indxekVadv_mld)='Vertical advection term           '
       write(vname3,*) trim(unitt),' meter second-2               '
       vname(3,indxekVadv_mld)=vname3

CR       write(stdout,'(1x,A,I3)') 'indxekCor = ',indxekCor+itrc-1
       unitt=vname(3,indxU)
       write(vname1,*) trim(nametrc),'_cor_mld                    '
       vname(1,indxekCor_mld)=vname1
CR       write(stdout,'(20x,A)') vname(1,indxekCor)
       vname(2,indxekCor_mld)='Coriolis term                      '
       write(vname3,*) trim(unitt),' meter second-2               '
       vname(3,indxekCor_mld)=vname3

CR       write(stdout,'(1x,A,I3)') 'indxekPrsgrd = ',indxekPrsgrd+itrc-1
       unitt=vname(3,indxU)
       write(vname1,*) trim(nametrc),'_prsgrd_mld                 '
       vname(1,indxekPrsgrd_mld)=vname1
CR       write(stdout,'(20x,A)') vname(1,indxekPrsgrd)
       vname(2,indxekPrsgrd_mld)='Pressure gradient term          '
       write(vname3,*) trim(unitt),' meter second-2               '
       vname(3,indxekPrsgrd_mld)=vname3

CR       write(stdout,'(1x,A,I3)') 'indxekHmix = ',indxekHmix+itrc-1
       unitt=vname(3,indxU)
       write(vname1,*) trim(nametrc),'_hmix_mld                   '
       vname(1,indxekHmix_mld)=vname1
CR       write(stdout,'(20x,A)') vname(1,indxekHmix)
       vname(2,indxekHmix_mld)='Horizontal mixing term            '
       write(vname3,*) trim(unitt),' meter second-2               '
       vname(3,indxekHmix_mld)=vname3

CR       write(stdout,'(1x,A,I3)') 'indxekVmix = ',indxekVmix+itrc-1
       unitt=vname(3,indxU)
       write(vname1,*) trim(nametrc),'_vmix_mld                   '
       vname(1,indxekVmix_mld)=vname1
CR       write(stdout,'(20x,A)') vname(1,indxekVmix)
       vname(2,indxekVmix_mld)='Vertical mixing term              '
       write(vname3,*) trim(unitt),' meter second-2               '
       vname(3,indxekVmix_mld)=vname3

CR       write(stdout,'(1x,A,I3)') 'indxekrate = ',indxekrate+itrc-1
       unitt=vname(3,indxU)
       write(vname1,*) trim(nametrc),'_rate_mld                   '
       vname(1,indxekrate_mld)=vname1
CR       write(stdout,'(20x,A)') vname(1,indxekrate)
       vname(2,indxekrate_mld)='Time rate of change               '
       write(vname3,*) trim(unitt),' meter second-2               '
       vname(3,indxekrate_mld)=vname3

CR       write(stdout,'(1x,A,I3)') 'indxekvol = ',indxekvol+itrc-1
       unitt=vname(3,indxU)
       write(vname1,*) trim(nametrc),'_vol_mld                    '
       vname(1,indxekvol_mld)=vname1
CR       write(stdout,'(20x,A)') vname(1,indxekvol)
       vname(2,indxekvol_mld)='Volume change                      '
       write(vname3,*) trim(unitt),' meter second-2               '
       vname(3,indxekvol_mld)=vname3

CR       write(stdout,'(1x,A,I3)') 'indxekVmix2 = ',indxekVmix2+itrc-1
       unitt=vname(3,indxU)
       write(vname1,*) trim(nametrc),'_nudg_mld                   '
       vname(1,indxekVmix2_mld)=vname1
CR       write(stdout,'(20x,A)') vname(1,indxekVmix2)
       vname(2,indxekVmix2_mld)='Momentum nudging term            '
       write(vname3,*) trim(unitt),' meter second-2               '
       vname(3,indxekVmix2_mld)=vname3

#  if defined DIAGNOSTICS_BARO
CR       write(stdout,'(1x,A,I3)') 'indxekBaro = ',indxekBaro+itrc-1
       unitt=vname(3,indxU)
       write(vname1,*) trim(nametrc),'_baro_mld                   '
       vname(1,indxekBaro_mld)=vname1
CR       write(stdout,'(20x,A)') vname(1,indxekBaro)
       vname(2,indxekBaro_mld)='Pressure gradient term            '
       write(vname3,*) trim(unitt),' meter second-2               '
       vname(3,indxekBaro_mld)=vname3
#  endif
# endif /* DIAGNOSTICS_EK_MLD */
#endif /* DIAGNOSTICS_EK */

#ifdef DIAGNOSTICS_PV
# ifdef DIAGNOSTICS_PV_FULL
CR       write(stdout,'(1x,A,I3)') 'indxpvpv = ',indxpvpv
       nametrc='u_vmix_trans'
       unitt='meter second-1 second-1'
       write(vname1,*) trim(nametrc)
       vname(1,indxpvpv)=vname1
CR       write(stdout,'(20x,A)') vname(1,indxpvpv)
       vname(2,indxpvpv)='                                '
       write(vname3,*) trim(unitt)
       vname(3,indxpvpv)=vname3
       vname(4,indxpvpv)='                                        '
       vname(5,indxpvpv)='                                        '
       vname(7,indxpvpv)='                                        '

CR       write(stdout,'(1x,A,I3)') 'indxpvpvd = ',indxpvpvd
       nametrc='v_vmix_trans'
       unitt='meter second-1 second-1'
       write(vname1,*) trim(nametrc)
       vname(1,indxpvpvd)=vname1
CR       write(stdout,'(20x,A)') vname(1,indxpvpvd)
       vname(2,indxpvpvd)='                                '
       write(vname3,*) trim(unitt)
       vname(3,indxpvpvd)=vname3
       vname(4,indxpvpvd)='                                       '
       vname(5,indxpvpvd)='                                       '
       vname(7,indxpvpvd)='                                       '
# endif

      do itrc=1,NTA

CR       write(stdout,'(1x,A,I3)') 'indxpvTrhs = ',indxpvTrhs+itrc-1
       lvar=lenstr(vname(1,indxT+itrc-1))
       nametrc=vname(1,indxT+itrc-1)(1:lvar)
       lvar=lenstr(nametrc)
       unitt=vname(3,indxT+itrc-1)
       write(vname1,*) trim(nametrc),'_rhs                        '
       vname(1,indxpvTrhs+itrc-1)=vname1
CR       write(stdout,'(20x,A)') vname(1,indxpvTrhs+itrc-1)
       vname(2,indxpvTrhs+itrc-1)='right hand side                '
       write(vname3,*) trim(unitt),' second-1                     '
       vname(3,indxpvTrhs+itrc-1)=vname3
       vname(4,indxpvTrhs+itrc-1)='                               '
       vname(5,indxpvTrhs+itrc-1)='                               '
       vname(7,indxpvTrhs+itrc-1)='                               '

      enddo  ! itrc

      do itrc=1,2
CR       write(stdout,'(1x,A,I3)') 'indxpvMrhs = ',indxpvMrhs+itrc-1
       lvar=lenstr(vname(1,indxU+itrc-1))
       nametrc=vname(1,indxU+itrc-1)(1:lvar)
       lvar=lenstr(nametrc)
       unitt=vname(3,indxU+itrc-1)
       write(vname1,*) trim(nametrc),'_rhs                        '
       vname(1,indxpvMrhs+itrc-1)=vname1
CR       write(stdout,'(20x,A)') vname(1,indxpvMrhs+itrc-1)
       vname(2,indxpvMrhs+itrc-1)='right hand side                '
       write(vname3,*) trim(unitt),' second-1                     '
       vname(3,indxpvMrhs+itrc-1)=vname3
       vname(4,indxpvMrhs+itrc-1)='                               '
       write(vname3,*) 'right_hand_side_',trim(nametrc)
     &               //'_component'
       vname(5,indxpvMrhs+itrc-1)=vname3
       vname(6,indxpvMrhs+itrc-1)='                               '
       vname(7,indxpvMrhs+itrc-1)='                               '

      enddo  ! 1,2
#endif  /* DIAGNOSTICS_PV */

# if defined DIAGNOSTICS_EDDY && ! defined XIOS
CR       write(stdout,'(1x,A,I3)') 'indxeddyzz = ',indxeddyzz
       write(vname1,*) 'zz                                  '
       vname(1,indxeddyzz)=vname1
CR       write(stdout,'(20x,A)') vname(1,indxeddyzz)
       vname(2,indxeddyzz)='zz                              '
       write(vname3,*) 'meter2                              '
       vname(3,indxeddyzz)=vname3
       vname(4,indxeddyzz)='                                '
       vname(5,indxeddyzz)='                                '
       vname(7,indxeddyzz)='                                '

CR       write(stdout,'(1x,A,I3)') 'indxeddyuu = ',indxeddyuu
       write(vname1,*) 'uu                                        '
       vname(1,indxeddyuu)=vname1
CR       write(stdout,'(20x,A)') vname(1,indxeddyuu)
       vname(2,indxeddyuu)='uu                                    '
       write(vname3,*) 'meter2 second-2                           '
       vname(3,indxeddyuu)=vname3
       vname(4,indxeddyuu)='                                      '
       vname(5,indxeddyuu)='                                      '
       vname(7,indxeddyuu)='                                      '

CR       write(stdout,'(1x,A,I3)') 'indxeddyvv = ',indxeddyvv
       write(vname1,*) 'vv                                        '
       vname(1,indxeddyvv)=vname1
CR       write(stdout,'(20x,A)') vname(1,indxeddyvv)
       vname(2,indxeddyvv)='vv                                    '
       write(vname3,*) 'meter2 second-2                           '
       vname(3,indxeddyvv)=vname3
       vname(4,indxeddyvv)='                                      '
       vname(5,indxeddyvv)='                                      '
       vname(7,indxeddyvv)='                                      '

CR       write(stdout,'(1x,A,I3)') 'indxeddyuv = ',indxeddyuv
       write(vname1,*) 'uv                                        '
       vname(1,indxeddyuv)=vname1
CR       write(stdout,'(20x,A)') vname(1,indxeddyuv)
       vname(2,indxeddyuv)='uv                                    '
       write(vname3,*) 'meter2 second-2                           '
       vname(3,indxeddyuv)=vname3
       vname(4,indxeddyuv)='                                      '
       vname(5,indxeddyuv)='                                      '
       vname(7,indxeddyuv)='                                      '

CR       write(stdout,'(1x,A,I3)') 'indxeddyuw = ',indxeddyuw
       write(vname1,*) 'uw                                        '
       vname(1,indxeddyuw)=vname1
CR       write(stdout,'(20x,A)') vname(1,indxeddyuw)
       vname(2,indxeddyuw)='uw                                    '
       write(vname3,*) 'meter2 second-2                           '
       vname(3,indxeddyuw)=vname3
       vname(4,indxeddyuw)='                                      '
       vname(5,indxeddyuw)='                                      '
       vname(7,indxeddyuw)='                                      '

CR       write(stdout,'(1x,A,I3)') 'indxeddyvw = ',indxeddyvw
       write(vname1,*) 'vw                                        '
       vname(1,indxeddyvw)=vname1
CR       write(stdout,'(20x,A)') vname(1,indxeddyvw)
       vname(2,indxeddyvw)='vw                                    '
       write(vname3,*) 'meter2 second-2                           '
       vname(3,indxeddyvw)=vname3
       vname(4,indxeddyvw)='                                      '
       vname(5,indxeddyvw)='                                      '
       vname(7,indxeddyvw)='                                      '

CR       write(stdout,'(1x,A,I3)') 'indxeddywb = ',indxeddywb
       write(vname1,*) 'wb                                        '
       vname(1,indxeddywb)=vname1
CR       write(stdout,'(20x,A)') vname(1,indxeddywb)
       vname(2,indxeddywb)='wb                                    '
       write(vname3,*) 'meter2 second-2                           '
       vname(3,indxeddywb)=vname3
       vname(4,indxeddywb)='                                      '
       vname(5,indxeddywb)='                                      '
       vname(7,indxeddywb)='                                      '

CR       write(stdout,'(1x,A,I3)') 'indxeddyub = ',indxeddyub
       write(vname1,*) 'ub                                        '
       vname(1,indxeddyub)=vname1
CR       write(stdout,'(20x,A)') vname(1,indxeddyub)
       vname(2,indxeddyub)='ub                                    '
       write(vname3,*) 'meter2 second-2                           '
       vname(3,indxeddyub)=vname3
       vname(4,indxeddyub)='                                      '
       vname(5,indxeddyub)='                                      '
       vname(7,indxeddyub)='                                      '

CR       write(stdout,'(1x,A,I3)') 'indxeddyvb = ',indxeddyvb
       write(vname1,*) 'vb                                        '
       vname(1,indxeddyvb)=vname1
CR       write(stdout,'(20x,A)') vname(1,indxeddyvb)
       vname(2,indxeddyvb)='vb                                    '
       write(vname3,*) 'meter2 second-2                           '
       vname(3,indxeddyvb)=vname3
       vname(4,indxeddyvb)='                                '
       vname(5,indxeddyvb)='                                '
       vname(7,indxeddyvb)='                                '

CR       write(stdout,'(1x,A,I3)') 'indxeddyubu = ',indxeddyubu
       write(vname1,*) 'ubustr                              '
       vname(1,indxeddyubu)=vname1
CR       write(stdout,'(20x,A)') vname(1,indxeddyubu)
       vname(2,indxeddyubu)='ubustr                         '
       write(vname3,*) 'meter3 second-3                     '
       vname(3,indxeddyubu)=vname3
       vname(4,indxeddyubu)='                               '
       vname(5,indxeddyubu)='                               '
       vname(7,indxeddyubu)='                               '

CR       write(stdout,'(1x,A,I3)') 'indxeddyvbv = ',indxeddyvbv
       write(vname1,*) 'vbvstr                              '
       vname(1,indxeddyvbv)=vname1
CR       write(stdout,'(20x,A)') vname(1,indxeddyvbv)
       vname(2,indxeddyvbv)='vbvstr                         '
       write(vname3,*) 'meter3 second-3                     '
       vname(3,indxeddyvbv)=vname3
       vname(4,indxeddyvbv)='                               '
       vname(5,indxeddyvbv)='                               '
       vname(7,indxeddyvbv)='                               '


CR       write(stdout,'(1x,A,I3)') 'indxeddyusu = ',indxeddyusu
       write(vname1,*) 'usustr                              '
       vname(1,indxeddyusu)=vname1
CR       write(stdout,'(20x,A)') vname(1,indxeddyusu)
       vname(2,indxeddyusu)='usustr                         '
       write(vname3,*) 'meter3 second-3                     '
       vname(3,indxeddyusu)=vname3
       vname(4,indxeddyusu)='                               '
       vname(5,indxeddyusu)='                               '
       vname(7,indxeddyusu)='                               '

CR       write(stdout,'(1x,A,I3)') 'indxeddyvsv = ',indxeddyvsv
       write(vname1,*) 'vsvstr                              '
       vname(1,indxeddyvsv)=vname1
CR       write(stdout,'(20x,A)') vname(1,indxeddyvsv)
       vname(2,indxeddyvsv)='vsvstr                         '
       write(vname3,*) 'meter3 second-3                     '
       vname(3,indxeddyvsv)=vname3
       vname(4,indxeddyvsv)='                               '
       vname(5,indxeddyvsv)='                               '
       vname(7,indxeddyvsv)='                               '
#endif  /* DIAGNOSTICS_EDDY */

# if defined OUTPUTS_SURFACE && ! defined XIOS

CR       write(stdout,'(1x,A,I3)') 'indxsurft = ',indxsurft
       write(vname1,*) 'temp'
       vname(1,indxsurft)=vname1
CR       write(stdout,'(20x,A)') vname(1,indxsurft)
       vname(2,indxsurft)='SST                                 '
       write(vname3,*) 'Celsius'
       vname(3,indxsurft)=vname3

CR       write(stdout,'(1x,A,I3)') 'indxsurfs = ',indxsurfs
       write(vname1,*) 'salt'
       vname(1,indxsurfs)=vname1
CR       write(stdout,'(20x,A)') vname(1,indxsurfs)
       vname(2,indxsurfs)='SSS                                 '
       write(vname3,*) 'PSU'
       vname(3,indxsurfs)=vname3

CR       write(stdout,'(1x,A,I3)') 'indxsurfz = ',indxsurfz
       write(vname1,*) 'zeta'
       vname(1,indxsurfz)=vname1
CR       write(stdout,'(20x,A)') vname(1,indxsurfz)
       vname(2,indxsurfz)='SSH                                 '
       write(vname3,*) 'm'
       vname(3,indxsurfz)=vname3

CR       write(stdout,'(1x,A,I3)') 'indxsurfu = ',indxsurfu
       write(vname1,*) 'u'
       vname(1,indxsurfu)=vname1
CR       write(stdout,'(20x,A)') vname(1,indxsurfu)
       vname(2,indxsurfu)='surface u                           '
       write(vname3,*) 'meter second-1                         '
       vname(3,indxsurfu)=vname3

CR       write(stdout,'(1x,A,I3)') 'indxsurfv = ',indxsurfv
       write(vname1,*) 'v'
       vname(1,indxsurfv)=vname1
CR       write(stdout,'(20x,A)') vname(1,indxsurfv)
       vname(2,indxsurfv)='surface v                           '
       write(vname3,*) trim(unitt),'meter second-1             '
       vname(3,indxsurfv)=vname3
# endif /* OUTPUTS_SURFACE */

#ifdef DIAGNOSTICS_BIO
!====================================================================
# ifdef BIO_NChlPZD
CR      write(stdout,'(1x,A,I3)') 'indxbioFlux(NFlux_NewProd)='
CR     &                          ,indxbioFlux+NFlux_NewProd-1
      vname(1,indxbioFlux+NFlux_NewProd-1)='NFlux_NewProd'
      vname(2,indxbioFlux+NFlux_NewProd-1)=
     &  'Flux of N due to new production   '
      vname(3,indxbioFlux+NFlux_NewProd-1)=
     &  'mmol N m-3 s-1                    '
      vname(4,indxbioFlux+NFlux_NewProd-1)=
     &  'NFlux_NewProd, scalar, series     '

      vname(1,indxbioFlux+NFlux_Grazing-1)='NFlux_Grazing'
      vname(2,indxbioFlux+NFlux_Grazing-1)=
     &  'Flux of N due to grazing          '
      vname(3,indxbioFlux+NFlux_Grazing-1)=
     &  'mmol N m-3 s-1                    '
      vname(4,indxbioFlux+NFlux_Grazing-1)=
     &  'NFlux_Grazing, scalar, series     '

      vname(1,indxbioFlux+NFlux_SlopFeed-1)=
     &  'NFlux_SlopFeed                    '
      vname(2,indxbioFlux+NFlux_SlopFeed-1)=
     &  'Flux of N due to sloppy feeding   '
      vname(3,indxbioFlux+NFlux_SlopFeed-1)=
     &  'mmol N m-3 s-1                    '
      vname(4,indxbioFlux+NFlux_SlopFeed-1)=
     &  'NFlux_SlopFeed, scalar, series    '

      vname(1,indxbioFlux+NFlux_Pmort-1)='NFlux_Pmort'
      vname(2,indxbioFlux+NFlux_Pmort-1)=
     &  'Flux of N due to phytoplankton mortality'
      vname(3,indxbioFlux+NFlux_Pmort-1)=
     &  'mmol N m-3 s-1                  '
      vname(4,indxbioFlux+NFlux_Pmort-1)=
     &  'NFlux_Pmort, scalar, series     '

      vname(1,indxbioFlux+NFlux_Zmetab-1)=
     &  'NFlux_Zmetab                  '
      vname(2,indxbioFlux+NFlux_Zmetab-1)=
     &  'Flux of N due to zooplankton metabolism'
      vname(3,indxbioFlux+NFlux_Zmetab-1)=
     &  'mmol N m-3 s-1                 '
      vname(4,indxbioFlux+NFlux_Zmetab-1)=
     &  'NFlux_Zmetab, scalar, series   '

      vname(1,indxbioFlux+NFlux_Zmort-1)=
     &  'NFlux_Zmort                    '
      vname(2,indxbioFlux+NFlux_Zmort-1)=
     &  'Flux of N due to zooplankton mortality'
      vname(3,indxbioFlux+NFlux_Zmort-1)=
     &  'mmol N m-3 s-1                       '
      vname(4,indxbioFlux+NFlux_Zmort-1)=
     &  'NFlux_Zmort, scalar, series '

CR      write(stdout,'(1x,A,I3)') 'indxbioFlux(NFlux_ReminD)='
CR     &                          ,indxbioFlux+NFlux_ReminD-1
      vname(1,indxbioFlux+NFlux_ReminD-1)=
     &  'NFlux_ReminD                         '
      vname(2,indxbioFlux+NFlux_ReminD-1)=
     &  'Flux of N due to remin. of detritus  '
      vname(3,indxbioFlux+NFlux_ReminD-1)=
     &  'mmol N m-3 s-1                       '
      vname(4,indxbioFlux+NFlux_ReminD-1)=
     &  'NFlux_ReminD, scalar, series         '

#  ifdef OXYGEN
CR      write(stdout,'(1x,A,I3)') 'indxbioFlux(OGain_NewProd)='
CR     &                          ,indxbioFlux+OGain_NewProd-1
      vname(1,indxbioFlux+OGain_NewProd-1)=
     &  'OGain_NewProd                       '
      vname(2,indxbioFlux+OGain_NewProd-1)=
     &  'Flux of O2 due to new production    '
      vname(3,indxbioFlux+OGain_NewProd-1)=
     &  'mmol N m-3 s-1                      '
      vname(4,indxbioFlux+OGain_NewProd-1)=
     &  'OGain_NewProd, scalar, series       '

CR      write(stdout,'(1x,A,I3)') 'indxbioFlux(OLoss_Zmetab)='
CR     &                          ,indxbioFlux+OLoss_Zmetab-1
      vname(1,indxbioFlux+OLoss_Zmetab-1)=
     &  'OLoss_Zmetab                        '
      vname(2,indxbioFlux+OLoss_Zmetab-1)=
     &  'Flux of O2 due to zooplankton metabolism'
      vname(3,indxbioFlux+OLoss_Zmetab-1)=
     &  'mmol O2 m-3 s-1                     '
      vname(4,indxbioFlux+OLoss_Zmetab-1)=
     &  'OLoss_Zmetab, scalar, series        '

CR      write(stdout,'(1x,A,I3)') 'indxbioFlux(OLoss_ReminD)='
CR     &                          ,indxbioFlux+OLoss_ReminD-1
      vname(1,indxbioFlux+OLoss_ReminD-1)=
     &     'OLoss_ReminD                     '
      vname(2,indxbioFlux+OLoss_ReminD-1)=
     &     'Flux of O2 due to remin. of small detritus '
      vname(3,indxbioFlux+OLoss_ReminD-1)=
     &     'mmol O2 m-3 s-1                  '
      vname(4,indxbioFlux+OLoss_ReminD-1)=
     &  'OLoss_ReminD, scalar, series        '
#  endif /* OXYGEN */

! vertical sinking terms
CR      write(stdout,'(1x,A,I3)') 'indxbioVSink(NFlux_VSinkP1)='
CR     &                          ,indxbioVSink+NFlux_VSinkP1-1
      vname(1,indxbioVSink+NFlux_VSinkP1-1)=
     &     'NFlux_VSinkP1                    '
      vname(2,indxbioVSink+NFlux_VSinkP1-1)=
     &     'Vertical sinking flux of phytoplankton'
      vname(3,indxbioVSink+NFlux_VSinkP1-1)=
     &     'mmol N m-3 s-1                   '
      vname(4,indxbioVSink+NFlux_VSinkP1-1)=
     &     'NFlux_VSinkP1, scalar, series    '

CR      write(stdout,'(1x,A,I3)') 'indxbioVSink(NFlux_VSinkD1)='
CR     &                          ,indxbioVSink+NFlux_VSinkD1-1
      vname(1,indxbioVSink+NFlux_VSinkD1-1)=
     &     'NFlux_VSinkD1                   '
      vname(2,indxbioVSink+NFlux_VSinkD1-1)=
     &     'Vertical sinking flux of detritus N'
      vname(3,indxbioVSink+NFlux_VSinkD1-1)=
     &     'mmol N m-3 s-1                  '
      vname(4,indxbioVSink+NFlux_VSinkD1-1)=
     &     'NFlux_VSinkD1, scalar, series   '

#  ifdef OXYGEN
!     gas exchange terms
CR      write(stdout,'(1x,A,I3)') 'indxGasExcFlux='
CR     &                          ,indxGasExcFlux
      vname(1,indxGasExcFlux)=
     &     'GasExchangeO2                   '
      vname(2,indxGasExcFlux)=
     &     'Gas exchange of oxygen at the surface'
      vname(3,indxGasExcFlux)=
     &     'mmol O2 m-2 s-1                 '
      vname(4,indxGasExcFlux)=
     &     'GasExchangeO2, scalar, series   '
#  endif /* OXYGEN */

!====================================================================
# elif defined BIO_N2ChlPZD2
!--
!     BioFlux terms
!--
      vname(1,indxbioFlux+NFlux_NewProd-1)=
     &     'NFlux_NewProd                 '
      vname(2,indxbioFlux+NFlux_NewProd-1)=
     &     'Flux of N due to new production'
      vname(3,indxbioFlux+NFlux_NewProd-1)=
     &     'mmol N m-3 s-1                 '
      vname(4,indxbioFlux+NFlux_NewProd-1)=
     &     'NFlux_NewProd, scalar, series '
!--
      vname(1,indxbioFlux+NFlux_RegProd-1)=
     &     'NFlux_RegProd                 '
      vname(2,indxbioFlux+NFlux_RegProd-1)=
     &     'Flux of N due to reg production'
      vname(3,indxbioFlux+NFlux_RegProd-1)=
     &     'mmol N m-3 s-1                 '
      vname(4,indxbioFlux+NFlux_RegProd-1)=
     &     'NFlux_RegProd, scalar, series '
!--
      vname(1,indxbioFlux+NFlux_Nitrific-1)=
     &     'NFlux_Nitrific                 '
      vname(2,indxbioFlux+NFlux_Nitrific-1)=
     &     'Flux of N due to Nitrification '
      vname(3,indxbioFlux+NFlux_Nitrific-1)=
     &     'mmol N m-3 s-1                 '
      vname(4,indxbioFlux+NFlux_Nitrific-1)=
     &     'NFlux_Nitrific, scalar, series '
!--
      vname(1,indxbioFlux+NFlux_Grazing-1)=
     &     'NFlux_Grazing                 '
      vname(2,indxbioFlux+NFlux_Grazing-1)=
     &     'Flux of N due to grazing      '
      vname(3,indxbioFlux+NFlux_Grazing-1)=
     &     'mmol N m-3 s-1                '
      vname(4,indxbioFlux+NFlux_Grazing-1)=
     &     'NFlux_Grazing, scalar, series '
!--
      vname(1,indxbioFlux+NFlux_SlopFeed-1)=
     &     'NFlux_SlopFeed                '
      vname(2,indxbioFlux+NFlux_SlopFeed-1)=
     &     'Flux of N due to sloppy feeding for Z'
      vname(3,indxbioFlux+NFlux_SlopFeed-1)=
     &     'mmol N m-3 s-1                '
      vname(4,indxbioFlux+NFlux_SlopFeed-1)=
     &     'NFlux_SlopFeed, scalar, series'
!--
      vname(1,indxbioFlux+NFlux_Pmort-1)=
     &     'NFlux_Pmort                  '
      vname(2,indxbioFlux+NFlux_Pmort-1)=
     &     'Flux of N due to P mortality '
      vname(3,indxbioFlux+NFlux_Pmort-1)=
     &     'mmol N m-3 s-1               '
      vname(4,indxbioFlux+NFlux_Pmort-1)=
     &     'NFlux_Pmort, scalar, series  '
!--
      vname(1,indxbioFlux+NFlux_Zmetab-1)=
     &     'NFlux_Zmetab                 '
      vname(2,indxbioFlux+NFlux_Zmetab-1)=
     &     'Flux of N due to Z metabolism'
      vname(3,indxbioFlux+NFlux_Zmetab-1)=
     &     'mmol N m-3 s-1               '
      vname(4,indxbioFlux+NFlux_Zmetab-1)=
     &     'NFlux_Zmetab, scalar, series '
!--
      vname(1,indxbioFlux+NFlux_Zmort-1)=
     &     'NFlux_Zmort                  '
      vname(2,indxbioFlux+NFlux_Zmort-1)=
     &     'Flux of N due to Z mortality '
      vname(3,indxbioFlux+NFlux_Zmort-1)=
     &     'mmol N m-3 s-1               '
      vname(4,indxbioFlux+NFlux_Zmort-1)=
     &     'NFlux_Zmort, scalar, series  '
!--
      vname(1,indxbioFlux+NFlux_ReminD1-1)=
     &     'NFlux_ReminD1                 '
      vname(2,indxbioFlux+NFlux_ReminD1-1)=
     &     'Flux of N due to remin. of D1 '
      vname(3,indxbioFlux+NFlux_ReminD1-1)=
     &     'mmol N m-3 s-1                '
      vname(4,indxbioFlux+NFlux_ReminD1-1)=
     &     'NFlux_ReminD1, scalar, series '
!--
      vname(1,indxbioFlux+NFlux_ReminD2-1)=
     &     'NFlux_ReminD2                 '
      vname(2,indxbioFlux+NFlux_ReminD2-1)=
     &     'Flux of N due to remin. of D2 '
      vname(3,indxbioFlux+NFlux_ReminD2-1)=
     &     'mmol N m-3 s-1                '
      vname(4,indxbioFlux+NFlux_ReminD2-1)=
     &     'NFlux_ReminD2, scalar, series '
!--
      vname(1,indxbioFlux+NFlux_ReminD1-1)=
     &     'NFlux_ReminD1                 '
      vname(2,indxbioFlux+NFlux_ReminD1-1)=
     &     'Flux of N due to remin. of D1 '
      vname(3,indxbioFlux+NFlux_ReminD1-1)=
     &     'mmol N m-3 s-1                '
      vname(4,indxbioFlux+NFlux_ReminD1-1)=
     &     'NFlux_ReminD1, scalar, series '
!--
      vname(1,indxbioFlux+NFlux_CoagPhy-1)=
     &     'NFlux_CoagPhy                 '
      vname(2,indxbioFlux+NFlux_CoagPhy-1)=
     &     'Flux of N due to coag. of P   '
      vname(3,indxbioFlux+NFlux_CoagPhy-1)=
     &     'mmol N m-3 s-1                '
      vname(4,indxbioFlux+NFlux_CoagPhy-1)=
     &     'NFlux_CoagPhy, scalar, series '
!
      vname(1,indxbioFlux+NFlux_CoagSDet-1)=
     &     'NFlux_CoagSDet                '
      vname(2,indxbioFlux+NFlux_ReminD2-1)=
     &     'Flux of N due to coag. of SDet'
      vname(3,indxbioFlux+NFlux_ReminD2-1)=
     &     'mmol N m-3 s-1                '
      vname(4,indxbioFlux+NFlux_ReminD2-1)=
     &     'NFlux_CoagSDet, scalar, series'
!--
!     Sinking terms
!--
      vname(1,indxbioVSink+NFlux_VSinkP1-1)=
     &     'NFlux_VSinkP                  '
      vname(2,indxbioVSink+NFlux_VSinkP1-1)=
     &     'Flux of N due to Vert. Sink. of P'
      vname(3,indxbioVSink+NFlux_VSinkP1-1)=
     &     'mmol N m-3 s-1                 '
      vname(4,indxbioVSink+NFlux_VSinkP1-1)=
     &     'NFlux_VSinkP, scalar, series  '
!--
      vname(1,indxbioVSink+NFlux_VSinkD1-1)=
     &     'NFlux_VSinkD1                  '
      vname(2,indxbioVSink+NFlux_VSinkD1-1)=
     &     'Flux of N due to Vert. Sink. of D1'
      vname(3,indxbioVSink+NFlux_VSinkD1-1)=
     &     'mmol N m-3 s-1                 '
      vname(4,indxbioVSink+NFlux_VSinkD1-1)=
     &     'NFlux_VSinkD1, scalar, series  '
!--
      vname(1,indxbioVSink+NFlux_VSinkD2-1)=
     &     'NFlux_VSinkD2                   '
      vname(2,indxbioVSink+NFlux_VSinkD2-1)=
     &     'Flux of N due to Vert. Sink. of D2'
      vname(3,indxbioVSink+NFlux_VSinkD2-1)=
     &     'mmol N m-3 s-1                  '
      vname(4,indxbioVSink+NFlux_VSinkD2-1)=
     &     'NFlux_VSinkD2, scalar, series   '

!====================================================================
#   elif defined BIO_BioEBUS

CR      write(stdout,'(1x,A,I3)') 'indxbioFlux(NFlux_lightlimitP1)='
CR     &                          ,indxbioFlux+NFlux_lightlimitP1-1
      vname(1,indxbioFlux+NFlux_lightlimitP1-1)=
     &     'NFlux_lightlimitP1                            '
      vname(2,indxbioFlux+NFlux_lightlimitP1-1)=
     &     'Limitation of SPhy assimilation by light      '
      vname(3,indxbioFlux+NFlux_lightlimitP1-1)=
     &     'without unit                                  '
      vname(4,indxbioFlux+NFlux_lightlimitP1-1)=
     &     'NFlux_lightlimitP1, scalar, series            '

      vname(1,indxbioFlux+NFlux_lightlimitP2-1)=
     &     'NFlux_lightlimitP2                            '
      vname(2,indxbioFlux+NFlux_lightlimitP2-1)=
     &     'Limitation of LPhy assimilation by light      '
      vname(3,indxbioFlux+NFlux_lightlimitP2-1)=
     &     'without unit                                  '
      vname(4,indxbioFlux+NFlux_lightlimitP2-1)=
     &     'NFlux_lightlimitP2, scalar, series            '

      vname(1,indxbioFlux+NFlux_templimitP1-1)=
     &     'NFlux_templimitP1                             '
      vname(2,indxbioFlux+NFlux_templimitP1-1)=
     &     'Limitation of SPhy assimilation by temperature'
      vname(3,indxbioFlux+NFlux_templimitP1-1)=
     &     'd-1                                           '
      vname(4,indxbioFlux+NFlux_templimitP1-1)=
     &     'NFlux_templimitP1, scalar, series             '

      vname(1,indxbioFlux+NFlux_templimitP2-1)=
     &     'NFlux_templimitP2                             '
      vname(2,indxbioFlux+NFlux_templimitP2-1)=
     &     'Limitation of LPhy assimilation by temperature'
      vname(3,indxbioFlux+NFlux_templimitP2-1)=
     &     'd-1                                           '
      vname(4,indxbioFlux+NFlux_templimitP2-1)=
     &     'NFlux_templimitP2, scalar, series             '

      vname(1,indxbioFlux+NFlux_NO3limitP1-1)=
     &     'NFlux_NO3limitP1                              '
      vname(2,indxbioFlux+NFlux_NO3limitP1-1)=
     &     'Limitation of SPhy assimilation by NO3        '
      vname(3,indxbioFlux+NFlux_NO3limitP1-1)=
     &     'without unit                                  '
      vname(4,indxbioFlux+NFlux_NO3limitP1-1)=
     &     'NFlux_NO3limitP1, scalar, series              '

      vname(1,indxbioFlux+NFlux_NO2limitP1-1)=
     &     'NFlux_NO2limitP1                              '
      vname(2,indxbioFlux+NFlux_NO2limitP1-1)=
     &     'Limitation of SPhy assimilation by NO2        '
      vname(3,indxbioFlux+NFlux_NO2limitP1-1)=
     &     'without unit                                  '
      vname(4,indxbioFlux+NFlux_NO2limitP1-1)=
     &     'NFlux_NO2limitP1, scalar, series              '

      vname(1,indxbioFlux+NFlux_NH4limitP1-1)=
     &     'NFlux_NH4limitP1                              '
      vname(2,indxbioFlux+NFlux_NH4limitP1-1)=
     &     'Limitation of SPhy assimilation by NH4        '
      vname(3,indxbioFlux+NFlux_NH4limitP1-1)=
     &     'without unit                                  '
      vname(4,indxbioFlux+NFlux_NH4limitP1-1)=
     &     'NFlux_NH4limitP1, scalar, series              '

      vname(1,indxbioFlux+NFlux_NO3limitP2-1)=
     &     'NFlux_NO3limitP2                              '
      vname(2,indxbioFlux+NFlux_NO3limitP2-1)=
     &     'Limitation of LPhy assimilation by NO3        '
      vname(3,indxbioFlux+NFlux_NO3limitP2-1)=
     &     'without unit                                  '
      vname(4,indxbioFlux+NFlux_NO3limitP2-1)=
     &     'NFlux_NO3limitP2, scalar, series              '

      vname(1,indxbioFlux+NFlux_NO2limitP2-1)=
     &     'NFlux_NO2limitP2                              '
      vname(2,indxbioFlux+NFlux_NO2limitP2-1)=
     &     'Limitation of LPhy assimilation by NO2        '
      vname(3,indxbioFlux+NFlux_NO2limitP2-1)=
     &     'without unit                                  '
      vname(4,indxbioFlux+NFlux_NO2limitP2-1)=
     &     'NFlux_NO2limitP2, scalar, series              '

      vname(1,indxbioFlux+NFlux_NH4limitP2-1)=
     &     'NFlux_NH4limitP2                              '
      vname(2,indxbioFlux+NFlux_NH4limitP2-1)=
     &     'Limitation of LPhy assimilation by NH4        '
      vname(3,indxbioFlux+NFlux_NH4limitP2-1)=
     &     'without unit                                  '
      vname(4,indxbioFlux+NFlux_NH4limitP2-1)=
     &     'NFlux_NH4limitP2, scalar, series              '

      vname(1,indxbioFlux+NFlux_ProdNO3P1-1)=
     &     'NFlux_ProdNO3P1                               '
      vname(2,indxbioFlux+NFlux_ProdNO3P1-1)=
     &     'Flux of N due to NO3 assimilation by SPhy     '
      vname(3,indxbioFlux+NFlux_ProdNO3P1-1)=
     &     'mmol N m-3 s-1                                '
      vname(4,indxbioFlux+NFlux_ProdNO3P1-1)=
     &     'NFlux_ProdNO3P1, scalar, series               '

      vname(1,indxbioFlux+NFlux_ProdNO3P2-1)=
     &     'NFlux_ProdNO3P2                               '
      vname(2,indxbioFlux+NFlux_ProdNO3P2-1)=
     &     'Flux of N due to NO3 assimilation by LPhy     '
      vname(3,indxbioFlux+NFlux_ProdNO3P2-1)=
     &     'mmol N m-3 s-1                                '
      vname(4,indxbioFlux+NFlux_ProdNO3P2-1)=
     &     'NFlux_ProdNO3P2, scalar, series               '

      vname(1,indxbioFlux+NFlux_ProdNO2P1-1)=
     &     'NFlux_ProdNO2P1                               '
      vname(2,indxbioFlux+NFlux_ProdNO2P1-1)=
     &     'Flux of N due to NO2 assimilation by SPhy     '
      vname(3,indxbioFlux+NFlux_ProdNO2P1-1)=
     &     'mmol N m-3 s-1                                '
      vname(4,indxbioFlux+NFlux_ProdNO2P1-1)=
     &     'NFlux_ProdNO2P1, scalar, series               '

      vname(1,indxbioFlux+NFlux_ProdNO2P2-1)=
     &     'NFlux_ProdNO2P2                               '
      vname(2,indxbioFlux+NFlux_ProdNO2P2-1)=
     &     'Flux of N due to NO2 assimilation by LPhy     '
      vname(3,indxbioFlux+NFlux_ProdNO2P2-1)=
     &     'mmol N m-3 s-1                                '
      vname(4,indxbioFlux+NFlux_ProdNO2P2-1)=
     &     'NFlux_ProdNO2P2, scalar, series               '

      vname(1,indxbioFlux+NFlux_Nitrif1-1)=
     &     'NFlux_Nitrif1                                 '
      vname(2,indxbioFlux+NFlux_Nitrif1-1)=
     &     'Flux of N due to Nitrification (1st stage)    '
      vname(3,indxbioFlux+NFlux_Nitrif1-1)=
     &     'mmol N m-3 s-1                                '
      vname(4,indxbioFlux+NFlux_Nitrif1-1)=
     &     'NFlux_Nitrif1, scalar, series                 '

      vname(1,indxbioFlux+NFlux_Nitrif2-1)=
     &     'NFlux_Nitrif2                                 '
      vname(2,indxbioFlux+NFlux_Nitrif2-1)=
     &     'Flux of N due to Nitrification (2nd stage)    '
      vname(3,indxbioFlux+NFlux_Nitrif2-1)=
     &     'mmol N m-3 s-1                                '
      vname(4,indxbioFlux+NFlux_Nitrif2-1)=
     &     'NFlux_Nitrif2, scalar, series                 '

      vname(1,indxbioFlux+NFlux_ProdNH4P1-1)=
     &     'NFlux_ProdNH4P1                               '
      vname(2,indxbioFlux+NFlux_ProdNH4P1-1)=
     &     'Flux of N due to NH4 assimilation by SPhy     '
      vname(3,indxbioFlux+NFlux_ProdNH4P1-1)=
     &     'mmol N m-3 s-1                                '
      vname(4,indxbioFlux+NFlux_ProdNH4P1-1)=
     &     'NFlux_ProdNH4P1, scalar, series               '

      vname(1,indxbioFlux+NFlux_ProdNH4P2-1)=
     &     'NFlux_ProdNH4P2                               '
      vname(2,indxbioFlux+NFlux_ProdNH4P2-1)=
     &     'Flux of N due to NH4 assimilation by LPhy     '
      vname(3,indxbioFlux+NFlux_ProdNH4P2-1)=
     &     'mmol N m-3 s-1                                '
      vname(4,indxbioFlux+NFlux_ProdNH4P2-1)=
     &     'NFlux_ProdNH4P2, scalar, series               '

      vname(1,indxbioFlux+NFlux_P1Z1Grazing-1)=
     &     'NFlux_P1Z1Grazing                         '
      vname(2,indxbioFlux+NFlux_P1Z1Grazing-1)=
     &     'Flux of N due to grazing of Z1 on P1      '
      vname(3,indxbioFlux+NFlux_P1Z1Grazing-1)=
     &     'mmol N m-3 s-1                            '
      vname(4,indxbioFlux+NFlux_P1Z1Grazing-1)=
     &     'NFlux_P1Z1Grazing, scalar, series         '

      vname(1,indxbioFlux+NFlux_P2Z1Grazing-1)=
     &     'NFlux_P2Z1Grazing                         '
      vname(2,indxbioFlux+NFlux_P2Z1Grazing-1)=
     &     'Flux of N due to grazing of Z1 on P2      '
      vname(3,indxbioFlux+NFlux_P2Z1Grazing-1)=
     &     'mmol N m-3 s-1                            '
      vname(4,indxbioFlux+NFlux_P2Z1Grazing-1)=
     &     'NFlux_P2Z1Grazing, scalar, series         '

      vname(1,indxbioFlux+NFlux_P1mort-1)=
     &     'NFlux_P1mort                              '
      vname(2,indxbioFlux+NFlux_P1mort-1)=
     &     'Flux of N due to P1 mortality             '
      vname(3,indxbioFlux+NFlux_P1mort-1)=
     &     'mmol N m-3 s-1                            '
      vname(4,indxbioFlux+NFlux_P1mort-1)=
     &     'NFlux_P1mort, scalar, series              '

      vname(1,indxbioFlux+NFlux_P2mort-1)=
     &     'NFlux_P2mort                              '
      vname(2,indxbioFlux+NFlux_P2mort-1)=
     &     'Flux of N due to P2 mortality             '
      vname(3,indxbioFlux+NFlux_P2mort-1)=
     &     'mmol N m-3 s-1                            '
      vname(4,indxbioFlux+NFlux_P2mort-1)=
     &     'NFlux_P2mort, scalar, series              '

      vname(1,indxbioFlux+NFlux_P1Z2Grazing-1)=
     &     'NFlux_P1Z2Grazing                         '
      vname(2,indxbioFlux+NFlux_P1Z2Grazing-1)=
     &     'Flux of N due to grazing of Z2 on P1      '
      vname(3,indxbioFlux+NFlux_P1Z2Grazing-1)=
     &     'mmol N m-3 s-1                            '
      vname(4,indxbioFlux+NFlux_P1Z2Grazing-1)=
     &     'NFlux_P1Z2Grazing, scalar, series         '

      vname(1,indxbioFlux+NFlux_P2Z2Grazing-1)=
     &     'NFlux_P2Z2Grazing                         '
      vname(2,indxbioFlux+NFlux_P2Z2Grazing-1)=
     &     'Flux of N due to grazing of Z2 on P2      '
      vname(3,indxbioFlux+NFlux_P2Z2Grazing-1)=
     &     'mmol N m-3 s-1                            '
      vname(4,indxbioFlux+NFlux_P2Z2Grazing-1)=
     &     'NFlux_P2Z2Grazing, scalar, series         '

      vname(1,indxbioFlux+NFlux_Z1Z2Grazing-1)=
     &     'NFlux_Z1Z2Grazing                         '
      vname(2,indxbioFlux+NFlux_Z1Z2Grazing-1)=
     &     'Flux of N due to grazing of Z2 on Z1      '
      vname(3,indxbioFlux+NFlux_Z1Z2Grazing-1)=
     &     'mmol N m-3 s-1                            '
      vname(4,indxbioFlux+NFlux_Z1Z2Grazing-1)=
     &     'NFlux_Z1Z2Grazing, scalar, series         '

      vname(1,indxbioFlux+NFlux_Z1metab-1)=
     &     'NFlux_Z1metab                             '
      vname(2,indxbioFlux+NFlux_Z1metab-1)=
     &     'Flux of N due to Z1 metabolism            '
      vname(3,indxbioFlux+NFlux_Z1metab-1)=
     &     'mmol N m-3 s-1                            '
      vname(4,indxbioFlux+NFlux_Z1metab-1)=
     &     'NFlux_Z1metab, scalar, series             '

      vname(1,indxbioFlux+NFlux_Z1mort-1)=
     &     'NFlux_Z1mort                              '
      vname(2,indxbioFlux+NFlux_Z1mort-1)=
     &     'Flux of N due to Z1 mortality             '
      vname(3,indxbioFlux+NFlux_Z1mort-1)=
     &     'mmol N m-3 s-1                            '
      vname(4,indxbioFlux+NFlux_Z1mort-1)=
     &     'NFlux_Z1mort, scalar, series              '

      vname(1,indxbioFlux+NFlux_Z2metab-1)=
     &     'NFlux_Z2metab                             '
      vname(2,indxbioFlux+NFlux_Z2metab-1)=
     &     'Flux of N due to Z2 metabolism            '
      vname(3,indxbioFlux+NFlux_Z2metab-1)=
     &     'mmol N m-3 s-1                            '
      vname(4,indxbioFlux+NFlux_Z2metab-1)=
     &     'NFlux_Z2metab, scalar, series             '

      vname(1,indxbioFlux+NFlux_Z2mort-1)=
     &     'NFlux_Z2mort                              '
      vname(2,indxbioFlux+NFlux_Z2mort-1)=
     &     'Flux of N due to Z2 mortality             '
      vname(3,indxbioFlux+NFlux_Z2mort-1)=
     &     'mmol N m-3 s-1                            '
      vname(4,indxbioFlux+NFlux_Z2mort-1)=
     &     'NFlux_Z2mort, scalar, series              '

      vname(1,indxbioFlux+NFlux_HydrolD1-1)=
     &     'NFlux_HydrolD1                            '
      vname(2,indxbioFlux+NFlux_HydrolD1-1)=
     &     'Flux of N due to hydrolysis of D1         '
      vname(3,indxbioFlux+NFlux_HydrolD1-1)=
     &     'mmol N m-3 s-1                            '
      vname(4,indxbioFlux+NFlux_HydrolD1-1)=
     &     'NFlux_HydrolD1, scalar, series            '

      vname(1,indxbioFlux+NFlux_ReminOxyD1-1)=
     &     'NFlux_ReminOxyD1                          '
      vname(2,indxbioFlux+NFlux_ReminOxyD1-1)=
     &     'Flux of N due to oxic remin. of D1        '
      vname(3,indxbioFlux+NFlux_ReminOxyD1-1)=
     &     'mmol N m-3 s-1                            '
      vname(4,indxbioFlux+NFlux_ReminOxyD1-1)=
     &     'NFlux_ReminOxyD1, scalar, series          '

      vname(1,indxbioFlux+NFlux_Denitr1D1-1)=
     &     'NFlux_Denitr1D1                           '
      vname(2,indxbioFlux+NFlux_Denitr1D1-1)=
     &     'Flux of N due to Denitrification 1 of D1  '
      vname(3,indxbioFlux+NFlux_Denitr1D1-1)=
     &     'mmol N m-3 s-1                            '
      vname(4,indxbioFlux+NFlux_Denitr1D1-1)=
     &     'NFlux_Denitr1D1, scalar, series           '

      vname(1,indxbioFlux+NFlux_Denitr2D1-1)=
     &     'NFlux_Denitr2D1                           '
      vname(2,indxbioFlux+NFlux_Denitr2D1-1)=
     &     'Flux of N due to Denitrification 2 of D1  '
      vname(3,indxbioFlux+NFlux_Denitr2D1-1)=
     &     'mmol N m-3 s-1                            '
      vname(4,indxbioFlux+NFlux_Denitr2D1-1)=
     &     'NFlux_Denitr2D1, scalar, series           '

      vname(1,indxbioFlux+NFlux_HydrolD2-1)=
     &     'NFlux_HydrolD2                            '
      vname(2,indxbioFlux+NFlux_HydrolD2-1)=
     &     'Flux of N due to hydrolysis of D2         '
      vname(3,indxbioFlux+NFlux_HydrolD2-1)=
     &     'mmol N m-3 s-1                            '
      vname(4,indxbioFlux+NFlux_HydrolD2-1)=
     &     'NFlux_HydrolD2, scalar, series            '

      vname(1,indxbioFlux+NFlux_ReminOxyD2-1)=
     &     'NFlux_ReminOxyD2                          '
      vname(2,indxbioFlux+NFlux_ReminOxyD2-1)=
     &     'Flux of N due to oxic remin. of D2        '
      vname(3,indxbioFlux+NFlux_ReminOxyD2-1)=
     &     'mmol N m-3 s-1                            '
      vname(4,indxbioFlux+NFlux_ReminOxyD2-1)=
     &     'NFlux_ReminOxyD2, scalar, series          '

      vname(1,indxbioFlux+NFlux_Denitr1D2-1)=
     &     'NFlux_Denitr1D2                           '
      vname(2,indxbioFlux+NFlux_Denitr1D2-1)=
     &     'Flux of N due to Denitrification 1 of D2  '
      vname(3,indxbioFlux+NFlux_Denitr1D2-1)=
     &     'mmol N m-3 s-1                            '
      vname(4,indxbioFlux+NFlux_Denitr1D2-1)=
     &     'NFlux_Denitr1D2, scalar, series           '

      vname(1,indxbioFlux+NFlux_Denitr2D2-1)=
     &     'NFlux_Denitr2D2                           '
      vname(2,indxbioFlux+NFlux_Denitr2D2-1)=
     &     'Flux of N due to Denitrification 2 of D2  '
      vname(3,indxbioFlux+NFlux_Denitr2D2-1)=
     &     'mmol N m-3 s-1                            '
      vname(4,indxbioFlux+NFlux_Denitr2D2-1)=
     &     'NFlux_Denitr2D2, scalar, series           '

      vname(1,indxbioFlux+NFlux_ReminOxyDON-1)=
     &     'NFlux_ReminOxyDON                         '
      vname(2,indxbioFlux+NFlux_ReminOxyDON-1)=
     &     'Flux of N due to oxic remin. of DON       '
      vname(3,indxbioFlux+NFlux_ReminOxyDON-1)=
     &     'mmol N m-3 s-1                            '
      vname(4,indxbioFlux+NFlux_ReminOxyDON-1)=
     &     'NFlux_ReminOxyDON, scalar, series         '

      vname(1,indxbioFlux+NFlux_Denitr1DON-1)=
     &     'NFlux_Denitr1DON                          '
      vname(2,indxbioFlux+NFlux_Denitr1DON-1)=
     &     'Flux of N due to Denitrification 1 of DON '
      vname(3,indxbioFlux+NFlux_Denitr1DON-1)=
     &     'mmol N m-3 s-1                            '
      vname(4,indxbioFlux+NFlux_Denitr1DON-1)=
     &     'NFlux_Denitr1DON, scalar, series          '

      vname(1,indxbioFlux+NFlux_Denitr2DON-1)=
     &     'NFlux_Denitr2DON                          '
      vname(2,indxbioFlux+NFlux_Denitr2DON-1)=
     &     'Flux of N due to Denitrification 2 of DON '
      vname(3,indxbioFlux+NFlux_Denitr2DON-1)=
     &     'mmol N m-3 s-1                            '
      vname(4,indxbioFlux+NFlux_Denitr2DON-1)=
     &     'NFlux_Denitr2DON, scalar, series          '

      vname(1,indxbioFlux+NFlux_NO2anammox-1)=
     &     'NFlux_NO2anammox                          '
      vname(2,indxbioFlux+NFlux_NO2anammox-1)=
     &     'Flux of N due to anammox of NO2           '
      vname(3,indxbioFlux+NFlux_NO2anammox-1)=
     &     'mmol N m-3 s-1                            '
      vname(4,indxbioFlux+NFlux_NO2anammox-1)=
     &     'NFlux_NO2anammox, scalar, series          '

      vname(1,indxbioFlux+NFlux_NH4anammox-1)=
     &     'NFlux_NH4anammox                          '
      vname(2,indxbioFlux+NFlux_NH4anammox-1)=
     &     'Flux of N due to anammox of NH4           '
      vname(3,indxbioFlux+NFlux_NH4anammox-1)=
     &     'mmol N m-3 s-1                            '
      vname(4,indxbioFlux+NFlux_NH4anammox-1)=
     &     'NFlux_NH4anammox, scalar, series          '

#    ifdef NITROUS_OXIDE
      vname(1,indxbioFlux+NFlux_paramN2O-1)=
     &     'NFlux_paramN2O                            '
      vname(2,indxbioFlux+NFlux_paramN2O-1)=
     &     'Production of N2O                         '
      vname(3,indxbioFlux+NFlux_paramN2O-1)=
     &     'mmol N2O m-3 s-1                          '
      vname(4,indxbioFlux+NFlux_paramN2O-1)=
     &     'NFlux_paramN2O, scalar, series            '
#    endif

! vertical sinking terms
      vname(1,indxbioVSink+NFlux_VSinkP2-1)=
     &     'NFlux_VSinkP2                             '
      vname(2,indxbioVSink+NFlux_VSinkP2-1)=
     &     'Flux of N due to Vert. Sink. of P2        '
      vname(3,indxbioVSink+NFlux_VSinkP2-1)=
     &     'mmol N m-3 s-1                            '
      vname(4,indxbioVSink+NFlux_VSinkP2-1)=
     &     'NFlux_VSinkP2, scalar, series             '

      vname(1,indxbioVSink+NFlux_VSinkD1-1)=
     &     'NFlux_VSinkD1                             '
      vname(2,indxbioVSink+NFlux_VSinkD1-1)=
     &     'Flux of N due to Vert. Sink. of D1        '
      vname(3,indxbioVSink+NFlux_VSinkD1-1)=
     &     'mmol N m-3 s-1                            '
      vname(4,indxbioVSink+NFlux_VSinkD1-1)=
     &     'NFlux_VSinkD1, scalar, series             '

      vname(1,indxbioVSink+NFlux_VSinkD2-1)=
     &     'NFlux_VSinkD2                             '
      vname(2,indxbioVSink+NFlux_VSinkD2-1)=
     &     'Flux of N due to Vert. Sink. of D2        '
      vname(3,indxbioVSink+NFlux_VSinkD2-1)=
     &     'mmol N m-3 s-1                            '
      vname(4,indxbioVSink+NFlux_VSinkD2-1)=
     &     'NFlux_VSinkD2, scalar, series             '

! gas exchange terms
       vname(1,indxGasExcFlux+O2Flux_GasExc-1)=
     &     'O2Flux_GasExc                             '
      vname(2,indxGasExcFlux+O2Flux_GasExc-1)=
     &     'Flux of O2 at ocean-atmos interface       '
      vname(3,indxGasExcFlux+O2Flux_GasExc-1)=
     &     ' mmol O2 m-3 s-1                          '
      vname(4,indxGasExcFlux+O2Flux_GasExc-1)=
     &     'O2Flux_GasExc, scalar, series             '

#     ifdef NITROUS_OXIDE
      vname(1,indxGasExcFlux+N2OFlux_GasExc-1)=
     &     'N2OFlux_GasExc                            '
      vname(2,indxGasExcFlux+N2OFlux_GasExc-1)=
     &     'Flux of N2O at ocean-atmos interface      '
      vname(3,indxGasExcFlux+N2OFlux_GasExc-1)=
     &     ' mmol N2O m-3 s-1                         '
      vname(4,indxGasExcFlux+N2OFlux_GasExc-1)=
     &     'N2OFlux_GasExc, scalar, series            '
#     endif

!====================================================================
# elif defined PISCES

#  ifdef key_trc_diaadd
      vname(1,indxbioFlux+Nhi-1)=
     &     'PH                              '
      vname(2,indxbioFlux+Nhi-1)=
     &     'PH                              '
      vname(3,indxbioFlux+Nhi-1)=
     &     '-                               '
      vname(4,indxbioFlux+Nhi-1)=
     &     'PH            , scalar, series  '

      vname(1,indxbioFlux+Nco3-1)=
     &     'CO3                             '
      vname(2,indxbioFlux+Nco3-1)=
     &     'Bicarbonates                    '
      vname(3,indxbioFlux+Nco3-1)=
     &     'mol/L                           '
      vname(4,indxbioFlux+Nco3-1)=
     &     'CO3           , scalar, series  '

      vname(1,indxbioFlux+Naksp-1)=
     &     'CO3sat                          '
      vname(2,indxbioFlux+Naksp-1)=
     &     'Saturation CO3                  '
      vname(3,indxbioFlux+Naksp-1)=
     &     'mol/L                           '
      vname(4,indxbioFlux+Naksp-1)=
     &     'CO3sat        , scalar, series  '

      vname(1,indxbioFlux+Netot-1)=
     &     'PAR                             '
      vname(2,indxbioFlux+Netot-1)=
     &     'Photosynthetic Available radiation'
      vname(3,indxbioFlux+Netot-1)=
     &     'W/m2                            '
      vname(4,indxbioFlux+Netot-1)=
     &     'PAR           , scalar, series  '

      vname(1,indxbioFlux+Nprorca-1)=
     &     'PPPHY                           '
      vname(2,indxbioFlux+Nprorca-1)=
     &     'Primary production Nanoplankton '
      vname(3,indxbioFlux+Nprorca-1)=
     &     'molC/m3/s                       '
      vname(4,indxbioFlux+Nprorca-1)=
     &     'PPPHY         , scalar, series  '


      vname(1,indxbioFlux+Nprorcad-1)=
     &     'PPPHY2                          '
      vname(2,indxbioFlux+Nprorcad-1)=
     &     'Primary production Diatoms      '
      vname(3,indxbioFlux+Nprorcad-1)=
     &     'molC/m3/s                       '
      vname(4,indxbioFlux+Nprorcad-1)=
     &     'PPPHY2        , scalar, series  '

      vname(1,indxbioFlux+Npronew-1)=
     &     'PPNEWN                          '
      vname(2,indxbioFlux+Npronew-1)=
     &     'New Production Nanoplankton     '
      vname(3,indxbioFlux+Npronew-1)=
     &     'molC/m3/s                       '
      vname(4,indxbioFlux+Npronew-1)=
     &     'PPNEWN        , scalar, series  '


      vname(1,indxbioFlux+Npronewd-1)=
     &     'PPNEWD                          '
      vname(2,indxbioFlux+Npronewd-1)=
     &     'New Production Diatoms          '
      vname(3,indxbioFlux+Npronewd-1)=
     &     'molC/m3/s                       '
      vname(4,indxbioFlux+Npronewd-1)=
     &     'PPNEWD        , scalar, series  '

      vname(1,indxbioFlux+Nprobsi-1)=
     &     'PBSi                            '
      vname(2,indxbioFlux+Nprobsi-1)=
     &     'Silicate Production Diatoms     '
      vname(3,indxbioFlux+Nprobsi-1)=
     &     'molSi/m3/s                      '
      vname(4,indxbioFlux+Nprobsi-1)=
     &     'PBSi          , scalar, series  '

      vname(1,indxbioFlux+Nprofen-1)=
     &     'PFeN                            '
      vname(2,indxbioFlux+Nprofen-1)=
     &     'Iron Production Nano            '
      vname(3,indxbioFlux+Nprofen-1)=
     &     'molFe/m3/s                      '
      vname(4,indxbioFlux+Nprofen-1)=
     &     'PFeN          , scalar, series  '

      vname(1,indxbioFlux+Nprofed-1)=
     &     'PFeD                            '
      vname(2,indxbioFlux+Nprofed-1)=
     &     'Iron Production Diatoms         '
      vname(3,indxbioFlux+Nprofed-1)=
     &     'molFe/m3/s                      '
      vname(4,indxbioFlux+Nprofed-1)=
     &     'PFeD          , scalar, series  '

        vname(1,indxbioFlux+Npronewo2-1)=
     &     'PPNEWo2                         '
        vname(2,indxbioFlux+Npronewo2-1)=
     &     'o2 production by the NP         '
        vname(3,indxbioFlux+Npronewo2-1)=
     &     'molC/m3/s                       '
        vname(4,indxbioFlux+Npronewo2-1)=
     &     'PPNEWo2         , scalar, series'

        vname(1,indxbioFlux+Nprorego2-1)=
     &     'PPRego2                                    '
        vname(2,indxbioFlux+Nprorego2-1)=
     &     'o2 production by the RP                    '
        vname(3,indxbioFlux+Nprorego2-1)=
     &     'molC/m3/s                                  '
        vname(4,indxbioFlux+Nprorego2-1)=
     &     'PPRego2         , scalar, series           '

        vname(1,indxbioFlux+Ngrapoc-1)=
     &     'grapoc                                     '
        vname(2,indxbioFlux+Ngrapoc-1)=
     &     'Grazing by microzoo                        '
        vname(3,indxbioFlux+Ngrapoc-1)=
     &     'molC/m3/s                                  '
        vname(4,indxbioFlux+Ngrapoc-1)=
     &     'grapoc          , scalar, series           '

        vname(1,indxbioFlux+Ngrapoc2-1)=
     &     'grapoc2                                    '
        vname(2,indxbioFlux+Ngrapoc2-1)=
     &     'Grazing by mesozoo                         '
        vname(3,indxbioFlux+Ngrapoc2-1)=
     &     'molC/m3/s                                  '
        vname(4,indxbioFlux+Ngrapoc2-1)=
     &     'grapoc2          , scalar, series          '

        vname(1,indxbioFlux+Nmico2-1)=
     &     'mico2                                      '
        vname(2,indxbioFlux+Nmico2-1)=
     &     'O2 comsumption by microzoo                 '
        vname(3,indxbioFlux+Nmico2-1)=
     &     'molC/m3/s                                  '
        vname(4,indxbioFlux+Nmico2-1)=
     &     'mico2          , scalar, series            '

        vname(1,indxbioFlux+Nmeso2-1)=
     &     'meso2                                      '
        vname(2,indxbioFlux+Nmeso2-1)
     &     'O2 comsumption by mesozoo                  '
        vname(3,indxbioFlux+Nmeso2-1)=
     &     'molC/m3/s                                  '
        vname(4,indxbioFlux+Nmeso2-1)=
     &     'meso2          , scalar, series            '

        vname(1,indxbioFlux+Nnitrifo2-1)=
     &     'NitO2                                      '
        vname(2,indxbioFlux+Nnitrifo2-1)=
     &     'o2 consumption by Nitrification            '
        vname(3,indxbioFlux+Nnitrifo2-1)=
     &     'molC/m3/s                                  '
        vname(4,indxbioFlux+Nnitrifo2-1)=
     &     'NitO2         , scalar, series             '

        vname(1,indxbioFlux+Nremino2-1)=
     &     'ReminO2                                    '
        vname(2,indxbioFlux+Nremino2-1)=
     &     'o2 consumption by remineralisation         '
        vname(3,indxbioFlux+Nremino2-1)=
     &     'molC/m3/s                                  '
        vname(4,indxbioFlux+Nremino2-1)=
     &     'ReminO2         , scalar, series           '

        vname(1,indxbioFlux+Nfixo2-1)=
     &     'NFixo2                                     '
        vname(2,indxbioFlux+Nfixo2-1)=
     &     'o2 production by Nfix                      '
        vname(3,indxbioFlux+Nfixo2-1)=
     &     'molC/m3/s                                  '
        vname(4,indxbioFlux+Nfixo2-1)=
     &     'NFixo2         , scalar, series            '

      vname(1,indxbioFlux+Nirondep-1)=
     &     'Irondep                                    '
      vname(2,indxbioFlux+Nirondep-1)=
     &     'Iron flux from dust                        '
      vname(3,indxbioFlux+Nirondep-1)=
     &     'molFe/m3/s                                 '
      vname(4,indxbioFlux+Nirondep-1)=
     &     'Irondep           , scalar, series         '

        vname(1,indxbioFlux+Nironsed-1)=
     &     'Nironsed                                   '
        vname(2,indxbioFlux+Nironsed-1)=
     &     'Iron from sediment                         '
        vname(3,indxbioFlux+Nironsed-1)=
     &     'molC/m3/s                                  '
        vname(4,indxbioFlux+Nironsed-1)=
     &     'Nironsed         , scalar, series          '
      vname(1,indxbioVSink+Nfld-1)=
     &     'Cflx                            '
      vname(2,indxbioVSink+Nfld-1)=
     &     'DIC Flux                        '
      vname(3,indxbioVSink+Nfld-1)=
     &     'molC/m2/s                       '
      vname(4,indxbioVSink+Nfld-1)=
     &     'Cflx            , scalar, series'

      vname(1,indxbioVSink+Nflu16-1)=
     &     'Oflx                            '
      vname(2,indxbioVSink+Nflu16-1)=
     &     'O2 Flux                         '
      vname(3,indxbioVSink+Nflu16-1)=
     &     'molO/m2/s                       '
      vname(4,indxbioVSink+Nflu16-1)=
     &     'Oflx            , scalar, series'

      vname(1,indxbioVSink+Nkgco2-1)=
     &     'Kg                              '
      vname(2,indxbioVSink+Nkgco2-1)=
     &     'Gaz Transfer                    '
      vname(3,indxbioVSink+Nkgco2-1)=
     &     'mol/m2/s/uatm                   '
      vname(4,indxbioVSink+Nkgco2-1)=
     &     'Kg              , scalar, series'

      vname(1,indxbioVSink+Natcco2-1)=
     &     'Dpco2                            '
      vname(2,indxbioVSink+Natcco2-1)=
     &     'Delta Co2                        '
      vname(3,indxbioVSink+Natcco2-1)=
     &     'uatm                             '
      vname(4,indxbioVSink+Natcco2-1)=
     &     'Dpco2            , scalar, series'

      vname(1,indxbioVSink+Nsinking-1)=
     &     'EPC100                           '
      vname(2,indxbioVSink+Nsinking-1)=
     &     'Carbon export                    '
      vname(3,indxbioVSink+Nsinking-1)=
     &     'molC/m2/s                        '
      vname(4,indxbioVSink+Nsinking-1)=
     &     'EPC100           , scalar, series'


      vname(1,indxbioVSink+Nsinkfer-1)=
     &     'EPFE100                          '
      vname(2,indxbioVSink+Nsinkfer-1)=
     &     'Biogenic Iron export             '
      vname(3,indxbioVSink+Nsinkfer-1)=
     &     'molFe/m2/s                       '
      vname(4,indxbioVSink+Nsinkfer-1)=
     &     'EPFE100         , scalar, series '


      vname(1,indxbioVSink+Nsinksil-1)=
     &     'EPSI100                          '
      vname(2,indxbioVSink+Nsinksil-1)=
     &     'BSi export                       '
      vname(3,indxbioVSink+Nsinksil-1)=
     &     'molSi/m2/s                       '
      vname(4,indxbioVSink+Nsinksil-1)=
     &     'EPSI100         , scalar, series '

      vname(1,indxbioVSink+Nsinkcal-1)=
     &     'EPCAL100                         '
      vname(2,indxbioVSink+Nsinkcal-1)=
     &     'Calcite export                   '
      vname(3,indxbioVSink+Nsinkcal-1)=
     &     'molC/m2/s                        '
      vname(4,indxbioVSink+Nsinkcal-1)=
     &     'EPCAL100        , scalar, series '

      vname(1,indxbioVSink+Nheup-1)=
     &     'Heup                             '
      vname(2,indxbioVSink+Nheup-1)=
     &     'Euphotic Depth (1%)              '
      vname(3,indxbioVSink+Nheup-1)=
     &     'm                                '
      vname(4,indxbioVSink+Nheup-1)=
     &     'Heup            , scalar, series '

      vname(1,indxbioVSink+Nsildep-1)=
     &     'Sildep                           '
      vname(2,indxbioVSink+Nsildep-1)
     &     'Silicate surface deposition flux '
      vname(3,indxbioVSink+Nsildep-1)=
     &     'molSi/m2/s                       '
      vname(4,indxbioVSink+Nsildep-1)=
     &     'Sildep           , scalar, series'

      vname(1,indxbioVSink+Npo4dep-1)=
     &     'Po4dep                           '
      vname(2,indxbioVSink+Npo4dep-1)=
     &     'Phosphorus surface deposition flux'
      vname(3,indxbioVSink+Npo4dep-1)=
     &     'molP/m2/s                         '
      vname(4,indxbioVSink+Npo4dep-1)=
     &     'Po4dep           , scalar, series '

      vname(1,indxbioVSink+Nno3dep-1)=
     &     'No3dep                            '
      vname(2,indxbioVSink+Nno3dep-1)=
     &     'Nitrate surface deposition flux   '
      vname(3,indxbioVSink+Nno3dep-1)=
     &     'molN/m2/s                         '
      vname(4,indxbioVSink+Nno3dep-1)=
     &     'No3dep           , scalar, series '

      vname(1,indxbioVSink+Nnh4dep-1)=
     &     'Nh4dep                            '
      vname(2,indxbioVSink+Nnh4dep-1)=
     &     'Nitrate surface deposition flux   '
      vname(3,indxbioVSink+Nnh4dep-1)=
     &     'molN/m2/s                         '
      vname(4,indxbioVSink+Nnh4dep-1)=
     &     'Nh4dep           , scalar, series '
      vname(1,indxbioVSink+Nnitrpot-1)=
     &     'Nfix                              '
      vname(2,indxbioVSink+Nnitrpot-1)=
     &     'Nitrogen Fixation                 '
      vname(3,indxbioVSink+Nnitrpot-1)=
     &     'molN/m2/s                         '
      vname(4,indxbioVSink+Nnitrpot-1)=
     &     'Nfix            , scalar, series  '
#  endif
# endif /* PISCES */
#endif /* DIAGNOSTICS_BIO */

#if defined TCLIMATOLOGY && !defined ANA_TCLIMA && defined TRACERS
!
!     flag to read in climatology for tracer type variables.
!
      do itrc=1,NT
         got_tclm(itrc)=.false.
      enddo
#endif /* TCLIMATOLOGY */

      return
      end
