!SFX_LIC Copyright 1994-2014 CNRS, Meteo-France and Universite Paul Sabatier
!SFX_LIC This is part of the SURFEX software governed by the CeCILL-C licence
!SFX_LIC version 1. See LICENSE, CeCILL-C_V1-en.txt and CeCILL-C_V1-fr.txt  
!SFX_LIC for details. version 1.
!###############
MODULE MODN_SFX_OASIS
!###############
!
!!****  *MODN_SFX_OASIS - declaration of namelist for SFX-OASIS coupling
!!
!!    PURPOSE
!!    -------
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
!!      B. Decharme   *Meteo France*
!!
!!    MODIFICATIONS
!!    -------------
!!      Original       10/13
!!    10/2016 B. Decharme : bug surface/groundwater coupling
!!      Modified       11/2014 : J. Pianezze - add wave coupling parameters
!!                                             and surface pressure parameter for ocean coupling
!
!*       0.   DECLARATIONS
!             ------------
!
IMPLICIT NONE
!
!
REAL             :: XTSTEP_CPL_LAND = -1.0  ! Coupling time step for land
REAL             :: XTSTEP_CPL_SEA  = -1.0  ! Coupling time step for sea
REAL             :: XTSTEP_CPL_LAKE = -1.0  ! Coupling time step for lake
REAL             :: XTSTEP_CPL_WAVE = -1.0  ! Coupling time step for wave
!
!-------------------------------------------------------------------------------
!
! * Land surface variables for Surfex - Oasis coupling
!
!-------------------------------------------------------------------------------
!
! Output variables
!
CHARACTER(LEN=8) :: CRUNOFF     = '        '   ! Surface runoff 
CHARACTER(LEN=8) :: CDRAIN      = '        '   ! Deep drainage 
CHARACTER(LEN=8) :: CCALVING    = '        '   ! Calving flux 
CHARACTER(LEN=8) :: CSRCFLOOD   = '        '   ! Floodplains freshwater flux
!
! Input variables
!
CHARACTER(LEN=8) :: CWTD        = '        '   ! water table depth
CHARACTER(LEN=8) :: CFWTD       = '        '   ! grid-cell fraction of water table rise
CHARACTER(LEN=8) :: CFFLOOD     = '        '   ! Floodplains fraction
CHARACTER(LEN=8) :: CPIFLOOD    = '        '   ! Flood potential infiltartion
!
REAL             :: XFLOOD_LIM = 0.01
!
!-------------------------------------------------------------------------------
!
! * Lake variables for Surfex - Oasis coupling
!
!-------------------------------------------------------------------------------
!
! Input variables
!
CHARACTER(LEN=8) :: CLAKE_EVAP  = '        '   ! Evaporation over lake area
CHARACTER(LEN=8) :: CLAKE_RAIN  = '        '   ! Rainfall over lake area
CHARACTER(LEN=8) :: CLAKE_SNOW  = '        '   ! Snowfall over lake area
CHARACTER(LEN=8) :: CLAKE_WATF  = '        '   ! Net freshwater flux
!
!-------------------------------------------------------------------------------
!
! * Sea variables for Surfex - Oasis coupling 
!
!-------------------------------------------------------------------------------
!
! Sea Output variables
!
CHARACTER(LEN=8) :: CSEA_FWSU = '        '   ! zonal wind stress 
CHARACTER(LEN=8) :: CSEA_FWSV = '        '   ! meridian wind stress 
CHARACTER(LEN=8) :: CSEA_HEAT = '        '   ! Non solar net heat flux
CHARACTER(LEN=8) :: CSEA_SNET = '        '   ! Solar net heat flux
CHARACTER(LEN=8) :: CSEA_WIND = '        '   ! module of 10m wind speed 
CHARACTER(LEN=8) :: CSEA_FWSM = '        '   ! module of wind stress 
CHARACTER(LEN=8) :: CSEA_EVAP = '        '   ! Evaporation 
CHARACTER(LEN=8) :: CSEA_RAIN = '        '   ! Rainfall 
CHARACTER(LEN=8) :: CSEA_SNOW = '        '   ! Snowfall 
CHARACTER(LEN=8) :: CSEA_EVPR = '        '   ! Evaporation - Preci.
CHARACTER(LEN=8) :: CSEA_WATF = '        '   ! Net freshwater flux
CHARACTER(LEN=8) :: CSEA_PRES = '        '   ! Surface pressure 
CHARACTER(LEN=8) :: CSEA_LWFL = '        '   ! Long wave heat flux 
CHARACTER(LEN=8) :: CSEA_LHFL = '        '   ! Latent heat flux 
CHARACTER(LEN=8) :: CSEA_SHFL = '        '   ! Sensible heat flux 
!
! Sea-ice Output variables
!  
CHARACTER(LEN=8) :: CSEAICE_HEAT = '        '   ! Sea-ice non solar net heat flux
CHARACTER(LEN=8) :: CSEAICE_SNET = '        '   ! Sea-ice solar net heat flux 
CHARACTER(LEN=8) :: CSEAICE_EVAP = '        '   ! Sea-ice sublimation 
!
! Sea Input variables
!
CHARACTER(LEN=8) :: CSEA_SST    = '        ' ! Sea surface temperature
CHARACTER(LEN=8) :: CSEA_UCU    = '        ' ! Sea u-current stress
CHARACTER(LEN=8) :: CSEA_VCU    = '        ' ! Sea v-current stress
!
! Sea-ice Input variables
!
CHARACTER(LEN=8) :: CSEAICE_SIT = '        ' ! Sea-ice temperature
CHARACTER(LEN=8) :: CSEAICE_CVR = '        ' ! Sea-ice cover
CHARACTER(LEN=8) :: CSEAICE_ALB = '        ' ! Sea-ice albedo
!
!-------------------------------------------------------------------------------
!
! * Wave variables for Surfex - Oasis coupling 
!
!-------------------------------------------------------------------------------
!
! Wave Output variables
!
CHARACTER(LEN=8) :: CWAVE_U10  = '        '   ! 10m u-wind speed 
CHARACTER(LEN=8) :: CWAVE_V10  = '        '   ! 10m u-wind speed 
!
! Wave Input variables
!
CHARACTER(LEN=8) :: CWAVE_CHA    = '        ' ! Charnock coefficient
CHARACTER(LEN=8) :: CWAVE_UCU    = '        ' ! Wave u-current velocity
CHARACTER(LEN=8) :: CWAVE_VCU    = '        ' ! Wave v-current velocity
CHARACTER(LEN=8) :: CWAVE_HS     = '        ' ! Significant wave height
CHARACTER(LEN=8) :: CWAVE_TP     = '        ' ! Peak period
!
! Switch to add water into sea oasis mask
!
LOGICAL          :: LWATER = .FALSE.
!-------------------------------------------------------------------------------
!
!*       1.    NAMELISTS FOR LAND SURFACE FIELD
!              ------------------------------------------------
!
NAMELIST/NAM_SFX_LAND_CPL/XTSTEP_CPL_LAND, XFLOOD_LIM,          &
                         CRUNOFF,CDRAIN,CCALVING,CWTD,CFWTD,    &
                         CFFLOOD,CPIFLOOD,CSRCFLOOD
!
!
!*       2.    NAMELISTS FOR LAKE FIELD
!              ---------------------------------------------------------------
!
NAMELIST/NAM_SFX_LAKE_CPL/XTSTEP_CPL_LAKE,                              &
                          CLAKE_EVAP,CLAKE_RAIN,CLAKE_SNOW,CLAKE_WATF
!
!
!*       3.    NAMELISTS FOR OCEANIC FIELD
!              ---------------------------------------------------------------
!
NAMELIST/NAM_SFX_SEA_CPL/XTSTEP_CPL_SEA, LWATER,                               &
                          CSEA_FWSU,CSEA_FWSV,CSEA_HEAT,CSEA_SNET,CSEA_WIND,   &
                          CSEA_FWSM,CSEA_EVAP,CSEA_RAIN,CSEA_SNOW,CSEA_EVPR,   &
                          CSEA_WATF,CSEA_PRES,CSEA_LWFL,CSEA_LHFL,CSEA_SHFL,   &
                          CSEAICE_HEAT,CSEAICE_SNET,                           &
                          CSEAICE_EVAP,CSEA_SST,CSEA_UCU,CSEA_VCU,             &
                          CSEAICE_SIT,CSEAICE_CVR,CSEAICE_ALB
!
!*       4.    NAMELISTS FOR WAVE FIELD
!              ---------------------------------------------------------------
!
NAMELIST/NAM_SFX_WAVE_CPL/XTSTEP_CPL_WAVE,                                     &
                          CWAVE_U10, CWAVE_V10,                                &
                          CWAVE_CHA, CWAVE_UCU, CWAVE_VCU, CWAVE_HS, CWAVE_TP
!
!-------------------------------------------------------------------------------
!
END MODULE MODN_SFX_OASIS
