!SFX_LIC Copyright 1994-2014 CNRS, Meteo-France and Universite Paul Sabatier
!SFX_LIC This is part of the SURFEX software governed by the CeCILL-C licence
!SFX_LIC version 1. See LICENSE, CeCILL-C_V1-en.txt and CeCILL-C_V1-fr.txt  
!SFX_LIC for details. version 1.
!#########
SUBROUTINE SFX_OASIS_READ_NAM(HPROGRAM,PTSTEP_SURF,HINIT)
!##################################################################
!
!!****  *SFX_OASIS_READ_NAM* - routine to read the configuration for SFX-OASIS coupling
!!
!!    PURPOSE
!!    -------
!!
!!**  METHOD
!!    ------
!!
!!    EXTERNAL
!!    --------
!!
!!
!!    IMPLICIT ARGUMENTS
!!    ------------------
!!
!!    REFERENCE
!!    ---------
!!
!!
!!    AUTHOR
!!    ------
!!      B. Decharme   *Meteo France*
!!
!!    MODIFICATIONS
!!    -------------
!!      Original    05/2008 
!!    10/2016 B. Decharme : bug surface/groundwater coupling 
!!      Modified    11/2014 : J. Pianezze - add wave coupling parameters
!!                                          and surface pressure for ocean coupling
!-------------------------------------------------------------------------------
!
!*       0.    DECLARATIONS
!              ------------
!
USE MODN_SFX_OASIS
!
USE MODD_SFX_OASIS, ONLY : LOASIS, XRUNTIME,               &
                           LCPL_LAND, LCPL_GW, LCPL_FLOOD, &
                           LCPL_CALVING, LCPL_LAKE,        &
                           LCPL_SEA, LCPL_SEAICE, LCPL_WAVE
!
USE MODE_POS_SURF
!
USE MODI_GET_LUOUT
USE MODI_OPEN_NAMELIST
USE MODI_CLOSE_NAMELIST
!
USE MODI_ABOR1_SFX
!
USE YOMHOOK   ,ONLY : LHOOK,   DR_HOOK
USE PARKIND1  ,ONLY : JPRB
!
IMPLICIT NONE
!
!*       0.1   Declarations of arguments
!              -------------------------
!
CHARACTER(LEN=6), INTENT(IN)           :: HPROGRAM    ! program calling surf. schemes
REAL,             INTENT(IN)           :: PTSTEP_SURF ! Surfex time step
CHARACTER(LEN=*), INTENT(IN), OPTIONAL :: HINIT       ! choice of fields to initialize
!
!*       0.2   Declarations of local parameter
!              -------------------------------
!
INTEGER,          PARAMETER :: KIN   = 1
INTEGER,          PARAMETER :: KOUT  = 0
CHARACTER(LEN=5), PARAMETER :: YLAND = 'land'
CHARACTER(LEN=5), PARAMETER :: YLAKE = 'lake'
CHARACTER(LEN=5), PARAMETER :: YSEA  = 'ocean'
CHARACTER(LEN=5), PARAMETER :: YWAVE = 'wave'
!
!*       0.3   Declarations of local variables
!              -------------------------------
!
LOGICAL            :: GFOUND         ! Return code when searching namelist
INTEGER            :: ILUOUT         ! Listing id
INTEGER            :: ILUNAM         ! logical unit of namelist file
CHARACTER(LEN=20)  :: YKEY
CHARACTER(LEN=50)  :: YCOMMENT
CHARACTER(LEN=3)   :: YINIT
!
REAL(KIND=JPRB) :: ZHOOK_HANDLE
!
!-------------------------------------------------------------------------------
!
IF (LHOOK) CALL DR_HOOK('SFX_OASIS_READ_NAM',0,ZHOOK_HANDLE)
!
!
!*       0.     Initialize :
!               ------------
!
LCPL_LAND    = .FALSE.
LCPL_GW      = .FALSE.
LCPL_FLOOD   = .FALSE.
LCPL_CALVING = .FALSE.
LCPL_LAKE    = .FALSE.
LCPL_SEA     = .FALSE.
LCPL_SEAICE  = .FALSE.
LCPL_WAVE    = .FALSE.
!
IF(.NOT.LOASIS)THEN
  IF (LHOOK) CALL DR_HOOK('SFX_OASIS_READ_NAM',1,ZHOOK_HANDLE)
  RETURN
ENDIF
!
YINIT = 'ALL'
IF(PRESENT(HINIT))YINIT=HINIT
!
CALL GET_LUOUT(HPROGRAM,ILUOUT)
!
!*       1.     Read namelists and check status :
!               --------------------------------
!
CALL OPEN_NAMELIST(HPROGRAM,ILUNAM)
!
CALL POSNAM(ILUNAM,'NAM_SFX_LAND_CPL',GFOUND,ILUOUT)
!
IF (GFOUND) THEN
   READ(UNIT=ILUNAM,NML=NAM_SFX_LAND_CPL)
ELSE
   WRITE(ILUOUT,*)'!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!'
   WRITE(ILUOUT,*)'!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!'
   WRITE(ILUOUT,*)'NAM_SFX_LAND_CPL not found : Surfex land not coupled with river routing'
   WRITE(ILUOUT,*)'!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!'
   WRITE(ILUOUT,*)'!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!'
ENDIF
!
CALL POSNAM(ILUNAM,'NAM_SFX_SEA_CPL',GFOUND,ILUOUT)
!
IF (GFOUND) THEN
   READ(UNIT=ILUNAM,NML=NAM_SFX_SEA_CPL)
ELSE
   WRITE(ILUOUT,*)'!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!'
   WRITE(ILUOUT,*)'!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!'
   WRITE(ILUOUT,*)'NAM_SFX_SEA_CPL not found : Surfex sea not coupled with ocean model'
   WRITE(ILUOUT,*)'!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!'
   WRITE(ILUOUT,*)'!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!'
ENDIF
!
CALL POSNAM(ILUNAM,'NAM_SFX_LAKE_CPL',GFOUND,ILUOUT)
!
IF (GFOUND) THEN
   READ(UNIT=ILUNAM,NML=NAM_SFX_LAKE_CPL)
ELSE
   WRITE(ILUOUT,*)'!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!'
   WRITE(ILUOUT,*)'!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!'
   WRITE(ILUOUT,*)'NAM_SFX_LAKE_CPL not found : Surfex lake not coupled with ocean model'
   WRITE(ILUOUT,*)'!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!'
   WRITE(ILUOUT,*)'!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!'
ENDIF
!
CALL POSNAM(ILUNAM,'NAM_SFX_WAVE_CPL',GFOUND,ILUOUT)
!
IF (GFOUND) THEN
   READ(UNIT=ILUNAM,NML=NAM_SFX_WAVE_CPL)
ELSE
   WRITE(ILUOUT,*)'!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!'
   WRITE(ILUOUT,*)'!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!'
   WRITE(ILUOUT,*)'NAM_SFX_WAVE_CPL not found : Surfex not coupled with wave model'
   WRITE(ILUOUT,*)'!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!'
   WRITE(ILUOUT,*)'!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!'
ENDIF
!
CALL CLOSE_NAMELIST(HPROGRAM,ILUNAM)
!
IF(XTSTEP_CPL_LAND>0.0)LCPL_LAND=.TRUE.
IF(XTSTEP_CPL_LAKE>0.0)LCPL_LAKE=.TRUE.
IF(XTSTEP_CPL_SEA >0.0)LCPL_SEA =.TRUE.
IF(XTSTEP_CPL_WAVE>0.0)LCPL_WAVE=.TRUE.
!
IF(.NOT.LCPL_LAND.AND..NOT.LCPL_SEA.AND..NOT.LCPL_WAVE)THEN
  CALL ABOR1_SFX('SFX_OASIS_READ_NAM: OASIS USED BUT NAMELIST NOT FOUND')
ENDIF
!
!-------------------------------------------------------------------------------
!
!*       2.     Check time step consistency
!               ---------------------------
!
IF(YINIT/='PRE')THEN
  IF(MOD(XRUNTIME,PTSTEP_SURF)/=0.)THEN
    WRITE(ILUOUT,*)'! MOD(XRUNTIME,XTSTEP_SURF)/=0 !!!'     
    WRITE(ILUOUT,*)'! XTSTEP_SURF (model timestep) must be a multiple of $RUNTIME in oasis namcouple !!!'     
    CALL ABOR1_SFX('SFX_OASIS_READ_NAM: XTSTEP_SURF must be a multiple of $RUNTIME in oasis namcouple !!!')
  ENDIF
ENDIF
!
!-------------------------------------------------------------------------------
!
!*       3.     Check status for Land surface fields 
!               ------------------------------------
!
IF(LCPL_LAND)THEN
!
  IF(YINIT/='PRE')THEN
    IF(MOD(XTSTEP_CPL_LAND,PTSTEP_SURF)/=0.)THEN
      WRITE(ILUOUT,*)'! MOD(XTSTEP_SURF,XTSTEP_CPL_LAND) /= 0     !'
      WRITE(ILUOUT,*)'XTSTEP_SURF =',PTSTEP_SURF,'XTSTEP_CPL_LAND = ',XTSTEP_CPL_LAND
      IF(PTSTEP_SURF>XTSTEP_CPL_LAND) &
      WRITE(ILUOUT,*)'! XTSTEP_SURF (model timestep) is superiror to  XTSTEP_CPL_LAND !'         
      CALL ABOR1_SFX('SFX_OASIS_READ_NAM: XTSTEP_SURF and XTSTEP_CPL_LAND not consistent !!!')
    ENDIF
  ENDIF
!
! Land Output variable
!
  YKEY  ='CRUNOFF'
  YCOMMENT='Surface runoff'
  CALL CHECK_FIELD(CRUNOFF,YKEY,YCOMMENT,YLAND,KOUT)
!
  YKEY  ='CDRAIN'
  YCOMMENT='Deep drainage'
  CALL CHECK_FIELD(CDRAIN,YKEY,YCOMMENT,YLAND,KOUT)
!
! Particular case due to calving case
!
  IF(LEN_TRIM(CCALVING)>0)THEN
    LCPL_CALVING = .TRUE.
  ENDIF
!
  IF(LCPL_CALVING)THEN
    YKEY  ='CCALVING'
    YCOMMENT='Calving flux'
    CALL CHECK_FIELD(CCALVING,YKEY,YCOMMENT,YLAND,KOUT)
  ENDIF
!
! Particular case due to water table depth / surface coupling
!    
  IF(LEN_TRIM(CWTD)>0.OR.LEN_TRIM(CFWTD)>0)THEN
    LCPL_GW = .TRUE.
  ENDIF
!
  IF(LCPL_GW)THEN
!
!   Input variable
!
    YKEY  ='CWTD'
    YCOMMENT='Water table depth'
    CALL CHECK_FIELD(CWTD,YKEY,YCOMMENT,YLAND,KIN)
!
    YKEY  ='CFWTD'
    YCOMMENT='Fraction of WTD to rise'
    CALL CHECK_FIELD(CFWTD,YKEY,YCOMMENT,YLAND,KIN)
!
  ENDIF
!
! Particular case due to floodplains coupling
!    
  IF(LEN_TRIM(CSRCFLOOD)>0.OR.LEN_TRIM(CFFLOOD)>0.OR.LEN_TRIM(CPIFLOOD)>0)THEN
    LCPL_FLOOD = .TRUE.
  ENDIF
!
  IF(LCPL_FLOOD)THEN
!
!   Output variable
!
    YKEY  ='CSRCFLOOD'
    YCOMMENT='flood freshwater flux'
    CALL CHECK_FIELD(CSRCFLOOD,YKEY,YCOMMENT,YLAND,KOUT)
!
!   Input variable
!
    YKEY  ='CFFLOOD'
    YCOMMENT='Flood fraction'
    CALL CHECK_FIELD(CFFLOOD,YKEY,YCOMMENT,YLAND,KIN)
!
    YKEY  ='CPIFLOOD'
    YCOMMENT='Flood potential infiltration'
    CALL CHECK_FIELD(CPIFLOOD,YKEY,YCOMMENT,YLAND,KIN)
!
  ENDIF
!
ENDIF
!
!-------------------------------------------------------------------------------
!
!*       4.     Check status for Land surface fields 
!               ------------------------------------
!
IF(LCPL_LAKE)THEN
!
  IF(YINIT/='PRE')THEN
    IF(MOD(XTSTEP_CPL_LAKE,PTSTEP_SURF)/=0.)THEN
      WRITE(ILUOUT,*)'! MOD(XTSTEP_SURF,XTSTEP_CPL_LAKE) /= 0     !'
      WRITE(ILUOUT,*)'XTSTEP_SURF =',PTSTEP_SURF,'XTSTEP_CPL_LAKE = ',XTSTEP_CPL_LAKE
      IF(PTSTEP_SURF>XTSTEP_CPL_LAKE) &
      WRITE(ILUOUT,*)'! XTSTEP_SURF (model timestep) is superiror to  XTSTEP_CPL_LAKE !'     
      CALL ABOR1_SFX('SFX_OASIS_READ_NAM: XTSTEP_SURF and XTSTEP_CPL_LAKE not consistent !!!')          
    ENDIF
  ENDIF
!
! Output variables
!
  YKEY  ='CLAKE_EVAP'
  YCOMMENT='Evaporation rate'
  CALL CHECK_FIELD(CLAKE_EVAP,YKEY,YCOMMENT,YLAKE,KOUT)
!
  YKEY  ='CLAKE_RAIN'
  YCOMMENT='Rainfall rate'
  CALL CHECK_FIELD(CLAKE_RAIN,YKEY,YCOMMENT,YLAKE,KOUT)
!
  YKEY  ='CLAKE_SNOW'
  YCOMMENT='Snowfall rate'
  CALL CHECK_FIELD(CLAKE_SNOW,YKEY,YCOMMENT,YLAKE,KOUT)
!
  YKEY  ='CLAKE_WATF'
  YCOMMENT='Freshwater flux'
  CALL CHECK_FIELD(CLAKE_WATF,YKEY,YCOMMENT,YLAKE,KOUT)
!
ENDIF
!
!-------------------------------------------------------------------------------
!
!*       5.     Check status for Sea fields 
!               ---------------------------
!
IF(LCPL_SEA)THEN
!
  IF(YINIT/='PRE')THEN
    IF(MOD(XTSTEP_CPL_SEA,PTSTEP_SURF)/=0.)THEN
      WRITE(ILUOUT,*)'! MOD(XTSTEP_SURF,XTSTEP_CPL_SEA) /= 0     !'
      WRITE(ILUOUT,*)'XTSTEP_SURF =',PTSTEP_SURF,'XTSTEP_CPL_SEA = ',XTSTEP_CPL_SEA
      IF(PTSTEP_SURF>XTSTEP_CPL_SEA) &
      WRITE(ILUOUT,*)'! XTSTEP_SURF (model timestep) is superiror to  XTSTEP_CPL_SEA !'     
      CALL ABOR1_SFX('SFX_OASIS_READ_NAM: XTSTEP_SURF and XTSTEP_CPL_SEA not consistent !!!')          
    ENDIF
  ENDIF
!
! Sea Output variables
!
  YKEY  ='CSEA_FWSU'
  YCOMMENT='zonal wind stress'
  CALL CHECK_FIELD(CSEA_FWSU,YKEY,YCOMMENT,YSEA,KOUT)
!
  YKEY  ='CSEA_FWSV'
  YCOMMENT='meridian wind stress'
  CALL CHECK_FIELD(CSEA_FWSV,YKEY,YCOMMENT,YSEA,KOUT)
!
  YKEY  ='CSEA_HEAT'
  YCOMMENT='Non solar net heat flux'
  CALL CHECK_FIELD(CSEA_HEAT,YKEY,YCOMMENT,YSEA,KOUT)
!
  YKEY  ='CSEA_SNET'
  YCOMMENT='Solar net heat flux'
  CALL CHECK_FIELD(CSEA_SNET,YKEY,YCOMMENT,YSEA,KOUT)
!
  YKEY  ='CSEA_WIND'
  YCOMMENT='module of 10m wind speed'
  CALL CHECK_FIELD(CSEA_WIND,YKEY,YCOMMENT,YSEA,KOUT)
!
  YKEY  ='CSEA_FWSM'
  YCOMMENT='module of wind stress'
  CALL CHECK_FIELD(CSEA_FWSM,YKEY,YCOMMENT,YSEA,KOUT)
!
  YKEY  ='CSEA_EVAP'
  YCOMMENT='Evaporation rate'
  CALL CHECK_FIELD(CSEA_EVAP,YKEY,YCOMMENT,YSEA,KOUT)
!
  YKEY  ='CSEA_RAIN'
  YCOMMENT='Rainfall rate'
  CALL CHECK_FIELD(CSEA_RAIN,YKEY,YCOMMENT,YSEA,KOUT)
!
  YKEY  ='CSEA_SNOW'
  YCOMMENT='Snowfall rate'
  CALL CHECK_FIELD(CSEA_SNOW,YKEY,YCOMMENT,YSEA,KOUT)
!
  YKEY  ='CSEA_EVPR'
  YCOMMENT='Evap. - Precip. rate'
  CALL CHECK_FIELD(CSEA_EVPR,YKEY,YCOMMENT,YSEA,KOUT)
!
  YKEY  ='CSEA_WATF'
  YCOMMENT='Freshwater flux'
  CALL CHECK_FIELD(CSEA_WATF,YKEY,YCOMMENT,YSEA,KOUT)
!
  YKEY  ='CSEA_PRES'
  YCOMMENT='Surface pressure'
  CALL CHECK_FIELD(CSEA_PRES,YKEY,YCOMMENT,YSEA,KOUT)
!
  YKEY  ='CSEA_LWFL'
  YCOMMENT='Long wave heat flux'
  CALL CHECK_FIELD(CSEA_LWFL,YKEY,YCOMMENT,YSEA,KOUT)
!
  YKEY  ='CSEA_LHFL'
  YCOMMENT='Latent heat flux'
  CALL CHECK_FIELD(CSEA_LHFL,YKEY,YCOMMENT,YSEA,KOUT)
!
  YKEY  ='CSEA_SHFL'
  YCOMMENT='Sensible heat flux'
  CALL CHECK_FIELD(CSEA_SHFL,YKEY,YCOMMENT,YSEA,KOUT)
!
! Sea Input variables
!
  YKEY  ='CSEA_SST'
  YCOMMENT='Sea surface temperature'
  CALL CHECK_FIELD(CSEA_SST,YKEY,YCOMMENT,YSEA,KIN)
!
  YKEY  ='CSEA_UCU'
  YCOMMENT='Sea u-current stress'
  CALL CHECK_FIELD(CSEA_UCU,YKEY,YCOMMENT,YSEA,KIN)
!
  YKEY  ='CSEA_VCU'
  YCOMMENT='Sea v-current stress'
  CALL CHECK_FIELD(CSEA_VCU,YKEY,YCOMMENT,YSEA,KIN)
!
! Sea-ice fluxes
!
  IF(LEN_TRIM(CSEAICE_HEAT)>0.OR.LEN_TRIM(CSEAICE_SNET)>0.OR. &
     LEN_TRIM(CSEAICE_EVAP)>0.OR.LEN_TRIM(CSEAICE_SIT )>0.OR. &
     LEN_TRIM(CSEAICE_CVR )>0.OR.LEN_TRIM(CSEAICE_ALB )>0     )THEN
     LCPL_SEAICE=.TRUE.
  ENDIF
!
  IF(LCPL_SEAICE)THEN
!
!   Sea-ice Output variables
!
    YKEY  ='CSEAICE_HEAT'
    YCOMMENT='Sea-ice non solar net heat flux'
    CALL CHECK_FIELD(CSEAICE_HEAT,YKEY,YCOMMENT,YSEA,KOUT)
!
    YKEY  ='CSEAICE_SNET'
    YCOMMENT='Sea-ice solar net heat flux'
    CALL CHECK_FIELD(CSEAICE_SNET,YKEY,YCOMMENT,YSEA,KOUT)
!
    YKEY  ='CSEAICE_EVAP'
    YCOMMENT='Sea-ice sublimation'
    CALL CHECK_FIELD(CSEAICE_EVAP,YKEY,YCOMMENT,YSEA,KOUT)
!
!   Sea-ice Input variables
!
    YKEY  ='CSEAICE_SIT'
    YCOMMENT='Sea-ice temperature'
    CALL CHECK_FIELD(CSEAICE_SIT,YKEY,YCOMMENT,YSEA,KIN)
!
    YKEY  ='CSEAICE_CVR'
    YCOMMENT='Sea-ice cover'
    CALL CHECK_FIELD(CSEAICE_CVR,YKEY,YCOMMENT,YSEA,KIN)
!
    YKEY  ='CSEAICE_ALB'
    YCOMMENT='Sea-ice albedo'
    CALL CHECK_FIELD(CSEAICE_ALB,YKEY,YCOMMENT,YSEA,KIN)
!
  ENDIF
!  
ENDIF
!
!-------------------------------------------------------------------------------
!
!*       6.     Check status for Wave fields 
!               ---------------------------
!
IF(LCPL_WAVE)THEN
!
  IF(YINIT/='PRE')THEN
    IF(MOD(XTSTEP_CPL_WAVE,PTSTEP_SURF)/=0.)THEN
      WRITE(ILUOUT,*)'! MOD(XTSTEP_SURF,XTSTEP_CPL_WAVE) /= 0     !'
      WRITE(ILUOUT,*)'XTSTEP_SURF =',PTSTEP_SURF,'XTSTEP_CPL_WAVE = ',XTSTEP_CPL_WAVE
      IF(PTSTEP_SURF>XTSTEP_CPL_WAVE) &
      WRITE(ILUOUT,*)'! XTSTEP_SURF (model timestep) is superiror to  XTSTEP_CPL_WAVE !'
      CALL ABOR1_SFX('SFX_OASIS_READ_NAM: XTSTEP_SURF and XTSTEP_CPL_WAVE not consistent !!!')
    ENDIF
  ENDIF
!
! Wave Output variables
!
  YKEY  ='CWAVE_U10'
  YCOMMENT='10m u-wind speed'
  CALL CHECK_FIELD(CWAVE_U10,YKEY,YCOMMENT,YWAVE,KOUT)
!
  YKEY  ='CWAVE_V10'
  YCOMMENT='10m v-wind speed'
  CALL CHECK_FIELD(CWAVE_V10,YKEY,YCOMMENT,YWAVE,KOUT)
!
! Wave Input variables
!
  YKEY  ='CWAVE_CHA'
  YCOMMENT='Charnock Coefficient'
  CALL CHECK_FIELD(CWAVE_CHA,YKEY,YCOMMENT,YWAVE,KIN)
!
  YKEY  ='CWAVE_UCU'
  YCOMMENT='u-current velocity'
  CALL CHECK_FIELD(CWAVE_UCU,YKEY,YCOMMENT,YWAVE,KIN)
!
  YKEY  ='CWAVE_VCU'
  YCOMMENT='v-current velocity'
  CALL CHECK_FIELD(CWAVE_VCU,YKEY,YCOMMENT,YWAVE,KIN)
!
  YKEY  ='CWAVE_HS'
  YCOMMENT='Significant wave height'
  CALL CHECK_FIELD(CWAVE_HS,YKEY,YCOMMENT,YWAVE,KIN)
!
  YKEY  ='CWAVE_TP'
  YCOMMENT='Peak period'
  CALL CHECK_FIELD(CWAVE_TP,YKEY,YCOMMENT,YWAVE,KIN)
!  
ENDIF
!
IF (LHOOK) CALL DR_HOOK('SFX_OASIS_READ_NAM',1,ZHOOK_HANDLE)
!
!-------------------------------------------------------------------------------
CONTAINS
!-------------------------------------------------------------------------------
!
SUBROUTINE CHECK_FIELD(HFIELD,HKEY,HCOMMENT,HTYP,KID)
!
IMPLICIT NONE
!
CHARACTER(LEN=*), INTENT(IN) :: HFIELD
CHARACTER(LEN=*), INTENT(IN) :: HKEY
CHARACTER(LEN=*), INTENT(IN) :: HCOMMENT
CHARACTER(LEN=*), INTENT(IN) :: HTYP
INTEGER,          INTENT(IN) :: KID
!
CHARACTER(LEN=20)  :: YWORK
CHARACTER(LEN=20)  :: YNAMELIST
CHARACTER(LEN=128) :: YCOMMENT1
CHARACTER(LEN=128) :: YCOMMENT2
LOGICAL            :: LSTOP
!
REAL(KIND=JPRB) :: ZHOOK_HANDLE
!
IF (LHOOK) CALL DR_HOOK('SFX_OASIS_READ_NAM:CHECK_FIELD',0,ZHOOK_HANDLE)
!
IF(LEN_TRIM(HFIELD)==0)THEN
!
  IF(KID==0)THEN
    YWORK=TRIM(HTYP)//' - SFX'
  ELSE
    YWORK='SFX - '//TRIM(HTYP)
  ENDIF
!
  SELECT CASE (HTYP)
     CASE(YLAND)
          YNAMELIST='NAM_SFX_LAND_CPL'
     CASE(YSEA)
          YNAMELIST='NAM_SFX_SEA_CPL'
     CASE(YLAKE)
          YNAMELIST='NAM_SFX_LAKE_CPL' 
     CASE(YWAVE)
          YNAMELIST='NAM_SFX_WAVE_CPL'
     CASE DEFAULT
          CALL ABOR1_SFX('SFX_OASIS_READ_NAM: TYPE NOT SUPPORTED OR IMPLEMENTD : '//TRIM(HTYP))               
  END SELECT
!
  YCOMMENT1= 'SFX_OASIS_READ_NAM: '//TRIM(HCOMMENT)//' is not done for '//TRIM(YWORK)//' coupling'
  YCOMMENT2= 'SFX_OASIS_READ_NAM: Namelist key '//TRIM(HKEY)//' is not in '//TRIM(YNAMELIST)
!
  WRITE(ILUOUT,*)TRIM(YCOMMENT1)
  WRITE(ILUOUT,*)TRIM(YCOMMENT2)
!
! For oceanic coupling do not stop the model if a field from surfex to ocean is
! not  done because many particular case can be used
!
  IF((KID==0.OR.KID==1).AND.HTYP/=YLAND)THEN
    LSTOP=.FALSE.
  ELSE
    LSTOP=.TRUE.
  ENDIF
!
  IF(LSTOP)THEN
    CALL ABOR1_SFX(YCOMMENT1)
  ENDIF
!  
ENDIF
!
IF (LHOOK) CALL DR_HOOK('SFX_OASIS_READ_NAM:CHECK_FIELD',1,ZHOOK_HANDLE)
!
END SUBROUTINE CHECK_FIELD
!
!-------------------------------------------------------------------------------
!
END SUBROUTINE SFX_OASIS_READ_NAM
