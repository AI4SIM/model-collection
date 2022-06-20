!SFX_LIC Copyright 1994-2014 CNRS, Meteo-France and Universite Paul Sabatier
!SFX_LIC This is part of the SURFEX software governed by the CeCILL-C licence
!SFX_LIC version 1. See LICENSE, CeCILL-C_V1-en.txt and CeCILL-C_V1-fr.txt  
!SFX_LIC for details. version 1.
!     #################
      MODULE MODD_SEAFLUX_n
!     #################
!
!!****  *MODD_SEAFLUX_n - declaration of surface parameters for an inland water surface
!!
!!    PURPOSE
!!    -------
!     Declaration of surface parameters
!
!!
!!**  IMPLICIT ARGUMENTS
!!    ------------------
!!      None 
!!
!!    REFERENCE
!!    ---------
!!
!!    AUTHOR
!!    ------
!!      V. Masson   *Meteo France*
!!
!!    MODIFICATIONS
!!    -------------
!!      Original      01/2004
!!      S. Senesi     01/2014  adapt to fractional seaice, and to seaice scheme
!!      S. Belamari   03/2014  Include NZ0
!!      Modified      03/2014 : M.N. Bouin  ! possibility of wave parameters
!!                                        ! from external source
!!      Modified      11/2014 : J. Pianezze ! add surface pressure, evap-rain and charnock coefficient
!
!*       0.   DECLARATIONS
!             ------------
!
USE MODD_SURF_PAR,   ONLY : XUNDEF
USE MODD_TYPE_DATE_SURF
!
USE MODD_TYPES_GLT,   ONLY : T_GLT
!
!
USE YOMHOOK   ,ONLY : LHOOK,   DR_HOOK
USE PARKIND1  ,ONLY : JPRB
!
IMPLICIT NONE

TYPE SEAFLUX_t
!
! General surface: 
!
  REAL, POINTER, DIMENSION(:)   :: XZS     ! orography
  REAL, POINTER, DIMENSION(:,:) :: XCOVER  ! fraction of each ecosystem       (-)
  LOGICAL, POINTER, DIMENSION(:):: LCOVER  ! GCOVER(i)=T --> ith cover field is not 0.
  LOGICAL                       :: LSBL    ! T: SBL scheme between sea and atm. forcing level
!                                          ! F: no atmospheric layers below forcing level      
  LOGICAL                       :: LHANDLE_SIC ! T: we do weight seaice and open sea fluxes
  CHARACTER(LEN=6)              :: CSEAICE_SCHEME! Name of the seaice scheme 
  REAL, POINTER, DIMENSION(:)   :: XSEABATHY   ! bathymetry
!
  LOGICAL                       :: LINTERPOL_SST ! Interpolation of monthly SST
  CHARACTER(LEN=6)              :: CINTERPOL_SST ! Interpolation method of monthly SST
  LOGICAL                       :: LINTERPOL_SSS ! Interpolation of monthly SSS
  CHARACTER(LEN=6)              :: CINTERPOL_SSS ! Interpolation method of monthly SSS
  LOGICAL                       :: LINTERPOL_SIC ! Interpolation of monthly SIC
  CHARACTER(LEN=6)              :: CINTERPOL_SIC ! Interpolation method of monthly SIC
  LOGICAL                       :: LINTERPOL_SIT ! Interpolation of monthly SIT
  CHARACTER(LEN=6)              :: CINTERPOL_SIT ! Interpolation method of monthly SIT
  REAL                          :: XFREEZING_SST ! Value marking frozen sea in SST data
  REAL                          :: XSIC_EFOLDING_TIME ! For damping of SIC (days)
  REAL                          :: XSIT_EFOLDING_TIME ! For damping of SIT (days)
  REAL                          :: XSEAICE_TSTEP ! Sea ice model time step
  REAL                          :: XCD_ICE_CST   ! Turbulent exchange coefficient for seaice
  REAL                          :: XSI_FLX_DRV   ! Derivative of fluxes on seaice w.r.t to the temperature (W m-2 K-1)
  
!
! Type of formulation for the fluxes
!
  CHARACTER(LEN=6)                  :: CSEA_FLUX   ! type of flux computation
  CHARACTER(LEN=4)                  :: CSEA_ALB    ! type of albedo
  LOGICAL                           :: LPWG        ! flag for gust
  LOGICAL                           :: LPRECIP     ! flag for precip correction
  LOGICAL                           :: LPWEBB      ! flag for Webb correction
  INTEGER                           :: NZ0         ! set to 0,1 or 2 according to Z0 formulation
                                                   ! 0= ARPEGE / 1= Smith (1988) / 2= Direct
  INTEGER                           :: NGRVWAVES   ! set to 0,1 or 2 according to the 
                                                   ! gravity waves model used in coare30_flux
  LOGICAL                           :: LWAVEWIND    ! wave parameters computed from wind only
  REAL                              :: XICHCE      ! CE coef calculation for ECUME
  LOGICAL                           :: LPERTFLUX   ! flag for stochastic flux perturbation
!
! Sea/Ocean:
!
  REAL, POINTER, DIMENSION(:) :: XSST    ! sea surface temperature
  REAL, POINTER, DIMENSION(:) :: XSSS    ! sea surface salinity
  REAL, POINTER, DIMENSION(:) :: XHS     ! significant wave height
  REAL, POINTER, DIMENSION(:) :: XTP     ! wave peak period
  REAL, POINTER, DIMENSION(:) :: XTICE   ! sea ice temperature
  REAL, POINTER, DIMENSION(:) :: XSIC    ! sea ice concentration ( constraint for seaice scheme )
  REAL, POINTER, DIMENSION(:) :: XSST_INI! initial sea surface temperature
  REAL, POINTER, DIMENSION(:) :: XZ0     ! roughness length
  REAL, POINTER, DIMENSION(:) :: XZ0H    ! roughness length for heat
  REAL, POINTER, DIMENSION(:) :: XEMIS   ! emissivity
  REAL, POINTER, DIMENSION(:) :: XDIR_ALB! direct albedo
  REAL, POINTER, DIMENSION(:) :: XSCA_ALB! diffuse albedo
  REAL, POINTER, DIMENSION(:) :: XICE_ALB! sea-ice albedo from seaice model (ESM or embedded)
  REAL, POINTER, DIMENSION(:) :: XUMER   ! U component of sea current (for ESM coupling)
  REAL, POINTER, DIMENSION(:) :: XVMER   ! V component of sea current (for ESM coupling)
!
  REAL, POINTER, DIMENSION(:,:) :: XSST_MTH! monthly sea surface temperature (precedent, current and next)
  REAL, POINTER, DIMENSION(:,:) :: XSSS_MTH! monthly sea surface salinity    (precedent, current and next)
  REAL, POINTER, DIMENSION(:,:) :: XSIC_MTH! monthly sea ice cover           (precedent, current and next)
  REAL, POINTER, DIMENSION(:,:) :: XSIT_MTH! monthly sea ice thickness       (precedent, current and next)
  REAL, POINTER, DIMENSION(:)   :: XFSIC   ! nudging (or forcing) sea ice cover
  REAL, POINTER, DIMENSION(:)   :: XFSIT   ! nudging sea ice thickness
!
  REAL, POINTER, DIMENSION(:) :: XCHARN  ! Charnock coefficient (for ESM coupling)
!
  REAL, POINTER, DIMENSION(:) :: XCPL_SEA_WIND ! 10m wind speed for ESM coupling
  REAL, POINTER, DIMENSION(:) :: XCPL_SEA_FWSU ! zonal wind stress for ESM coupling
  REAL, POINTER, DIMENSION(:) :: XCPL_SEA_FWSV ! meridian wind stress for ESM coupling
  REAL, POINTER, DIMENSION(:) :: XCPL_SEA_SNET ! Solar net heat flux
  REAL, POINTER, DIMENSION(:) :: XCPL_SEA_HEAT ! Non solar net heat flux
  REAL, POINTER, DIMENSION(:) :: XCPL_SEA_EVAP ! Evaporation for ESM coupling
  REAL, POINTER, DIMENSION(:) :: XCPL_SEA_RAIN ! Rainfall for ESM coupling
  REAL, POINTER, DIMENSION(:) :: XCPL_SEA_EVPR ! Evaporatrion - Rainfall for ESM coupling
  REAL, POINTER, DIMENSION(:) :: XCPL_SEA_SNOW ! Snowfall for ESM coupling
  REAL, POINTER, DIMENSION(:) :: XCPL_SEA_FWSM ! wind stress for ESM coupling
  REAL, POINTER, DIMENSION(:) :: XCPL_SEA_PRES ! Surface pressure for ESM coupling
  REAL, POINTER, DIMENSION(:) :: XCPL_SEA_LWFL ! LW flux for ESM coupling
  REAL, POINTER, DIMENSION(:) :: XCPL_SEA_LHFL ! Latent heat flux for ESM coupling
  REAL, POINTER, DIMENSION(:) :: XCPL_SEA_SHFL ! Sensible heat flux for ESM coupling
!  
  REAL, POINTER, DIMENSION(:) :: XCPL_SEAICE_SNET ! Solar net heat flux for ESM coupling
  REAL, POINTER, DIMENSION(:) :: XCPL_SEAICE_HEAT ! Non solar net heat flux
  REAL, POINTER, DIMENSION(:) :: XCPL_SEAICE_EVAP ! Ice sublimation for ESM coupling
!
  REAL, POINTER, DIMENSION(:) :: XPERTFLUX     ! Stochastic flux perturbation pattern
!
! Sea-ice :
!
  TYPE(T_GLT)                           :: TGLT ! Sea-ice state , diagnostics and auxilliaries
                                             ! for the case of embedded Gelato Seaice model
!
! Date:
!
  TYPE (DATE_TIME)                      :: TTIME   ! current date and time
  TYPE (DATE_TIME)                      :: TZTIME  
  LOGICAL                               :: LTZTIME_DONE
  INTEGER                               :: JSX  
!
! Time-step:
!
  REAL                                  :: XTSTEP  ! time step
!
  REAL                                  :: XOUT_TSTEP  ! output writing time step
!
!
!
END TYPE SEAFLUX_t



CONTAINS

!




SUBROUTINE SEAFLUX_INIT(YSEAFLUX)
TYPE(SEAFLUX_t), INTENT(INOUT) :: YSEAFLUX
REAL(KIND=JPRB) :: ZHOOK_HANDLE
IF (LHOOK) CALL DR_HOOK("MODD_SEAFLUX_N:SEAFLUX_INIT",0,ZHOOK_HANDLE)
  NULLIFY(YSEAFLUX%XZS)
  NULLIFY(YSEAFLUX%XCOVER)
  NULLIFY(YSEAFLUX%LCOVER)
  NULLIFY(YSEAFLUX%XSEABATHY)
  NULLIFY(YSEAFLUX%XSST)
  NULLIFY(YSEAFLUX%XSSS)
  NULLIFY(YSEAFLUX%XSIC)
  NULLIFY(YSEAFLUX%XHS)
  NULLIFY(YSEAFLUX%XTP)
  NULLIFY(YSEAFLUX%XTICE)
  NULLIFY(YSEAFLUX%XSST_INI)
  NULLIFY(YSEAFLUX%XZ0)
  NULLIFY(YSEAFLUX%XZ0H)
  NULLIFY(YSEAFLUX%XEMIS)
  NULLIFY(YSEAFLUX%XDIR_ALB)
  NULLIFY(YSEAFLUX%XSCA_ALB)
  NULLIFY(YSEAFLUX%XICE_ALB)
  NULLIFY(YSEAFLUX%XUMER)
  NULLIFY(YSEAFLUX%XVMER)
  NULLIFY(YSEAFLUX%XSST_MTH)
  NULLIFY(YSEAFLUX%XSSS_MTH)
  NULLIFY(YSEAFLUX%XSIC_MTH)
  NULLIFY(YSEAFLUX%XSIT_MTH)
  NULLIFY(YSEAFLUX%XFSIC)
  NULLIFY(YSEAFLUX%XFSIT)
  NULLIFY(YSEAFLUX%XCPL_SEA_WIND)
  NULLIFY(YSEAFLUX%XCPL_SEA_FWSU)
  NULLIFY(YSEAFLUX%XCPL_SEA_FWSV)
  NULLIFY(YSEAFLUX%XCPL_SEA_SNET)
  NULLIFY(YSEAFLUX%XCPL_SEA_HEAT)
  NULLIFY(YSEAFLUX%XCPL_SEA_EVAP)
  NULLIFY(YSEAFLUX%XCPL_SEA_RAIN)
  NULLIFY(YSEAFLUX%XCPL_SEA_EVPR)
  NULLIFY(YSEAFLUX%XCPL_SEA_SNOW)
  NULLIFY(YSEAFLUX%XCPL_SEA_FWSM)
  NULLIFY(YSEAFLUX%XCPL_SEA_PRES)
  NULLIFY(YSEAFLUX%XCPL_SEA_LWFL)
  NULLIFY(YSEAFLUX%XCPL_SEA_LHFL)
  NULLIFY(YSEAFLUX%XCPL_SEA_SHFL)
  NULLIFY(YSEAFLUX%XCPL_SEAICE_SNET)
  NULLIFY(YSEAFLUX%XCPL_SEAICE_HEAT)
  NULLIFY(YSEAFLUX%XCPL_SEAICE_EVAP)
  NULLIFY(YSEAFLUX%XPERTFLUX)
YSEAFLUX%LSBL=.FALSE.
YSEAFLUX%LHANDLE_SIC=.FALSE.
YSEAFLUX%CSEAICE_SCHEME='NONE  '
YSEAFLUX%LINTERPOL_SST=.FALSE.
YSEAFLUX%CINTERPOL_SST=' '
YSEAFLUX%LINTERPOL_SSS=.FALSE.
YSEAFLUX%CINTERPOL_SSS=' '
YSEAFLUX%LINTERPOL_SIC=.FALSE.
YSEAFLUX%CINTERPOL_SIC=' '
YSEAFLUX%LINTERPOL_SIT=.FALSE.
YSEAFLUX%CINTERPOL_SIT=' '
YSEAFLUX%XFREEZING_SST=-1.8
YSEAFLUX%XSIC_EFOLDING_TIME=0. ! means : no damping
YSEAFLUX%XSIT_EFOLDING_TIME=0. ! means : no damping
YSEAFLUX%XSEAICE_TSTEP=XUNDEF 
YSEAFLUX%XCD_ICE_CST=0.
YSEAFLUX%XSI_FLX_DRV=-20. 
YSEAFLUX%CSEA_FLUX=' '
YSEAFLUX%CSEA_ALB=' '
YSEAFLUX%LPWG=.FALSE.
YSEAFLUX%LPRECIP=.FALSE.
YSEAFLUX%LPWEBB=.FALSE.
YSEAFLUX%NZ0=0
YSEAFLUX%NGRVWAVES=0
YSEAFLUX%LWAVEWIND=.TRUE.
YSEAFLUX%XICHCE=0.
YSEAFLUX%LPERTFLUX=.FALSE.
YSEAFLUX%JSX=0
YSEAFLUX%LTZTIME_DONE = .FALSE.
YSEAFLUX%XTSTEP=0.
YSEAFLUX%XOUT_TSTEP=0.
IF (LHOOK) CALL DR_HOOK("MODD_SEAFLUX_N:SEAFLUX_INIT",1,ZHOOK_HANDLE)
END SUBROUTINE SEAFLUX_INIT


END MODULE MODD_SEAFLUX_n
