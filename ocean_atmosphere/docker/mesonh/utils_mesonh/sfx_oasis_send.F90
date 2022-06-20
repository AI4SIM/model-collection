!SFX_LIC Copyright 1994-2014 CNRS, Meteo-France and Universite Paul Sabatier
!SFX_LIC This is part of the SURFEX software governed by the CeCILL-C licence
!SFX_LIC version 1. See LICENSE, CeCILL-C_V1-en.txt and CeCILL-C_V1-fr.txt  
!SFX_LIC for details. version 1.
!#########
SUBROUTINE SFX_OASIS_SEND(KLUOUT,KI,KDATE,OSEND_LAND,OSEND_LAKE,OSEND_SEA,OSEND_WAVE,  &
                          PLAND_RUNOFF,PLAND_DRAIN,PLAND_CALVING,               &
                          PLAND_SRCFLOOD,                                       &
                          PLAKE_EVAP,PLAKE_RAIN,PLAKE_SNOW,PLAKE_WATF,          &
                          PSEA_FWSU,PSEA_FWSV,PSEA_HEAT,PSEA_SNET,PSEA_WIND,    &
                          PSEA_FWSM,PSEA_EVAP,PSEA_RAIN,PSEA_SNOW,PSEA_EVPR,    &
                          PSEA_WATF,PSEA_PRES,PSEA_LWFL,PSEA_LHFL,PSEA_SHFL,    &
                          PSEAICE_HEAT,PSEAICE_SNET,        &
                          PSEAICE_EVAP,PWAVE_U10,PWAVE_V10            )
!###########################################
!
!!****  *SFX_OASIS_SEND* - Send coupling fields
!!
!!    PURPOSE
!!    -------
!!
!!    Attention : all fields are sent in Pa, m/s, W/m2 or kg/m2/s
!!   
!!
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
!!	B. Decharme   *Meteo France*	
!!
!!    MODIFICATIONS
!!    -------------
!!      Original    10/2013
!!      Modified    11/2014 : J. Pianezze - add wave coupling parameters
!!                                          and surface pressure for ocean coupling
!!    10/2016 B. Decharme : bug surface/groundwater coupling
!-------------------------------------------------------------------------------
!
!*       0.    DECLARATIONS
!              ------------
!
USE MODN_SFX_OASIS,  ONLY : XTSTEP_CPL_SEA, XTSTEP_CPL_WAVE, XTSTEP_CPL_LAKE, &
                            XTSTEP_CPL_LAND
!                    
USE MODD_SURF_PAR,   ONLY : XUNDEF, NUNDEF
!
USE MODD_SFX_OASIS
!
USE MODI_GET_LUOUT
!
USE YOMHOOK   ,ONLY : LHOOK,   DR_HOOK
USE PARKIND1  ,ONLY : JPRB
!
#ifdef CPLOASIS
USE MOD_OASIS
#endif
!
IMPLICIT NONE
!
!*       0.1   Declarations of arguments
!              -------------------------
!
INTEGER,             INTENT(IN) :: KLUOUT
INTEGER,             INTENT(IN) :: KI            ! number of points
INTEGER,             INTENT(IN) :: KDATE  ! current coupling time step (s)
LOGICAL,             INTENT(IN) :: OSEND_LAND
LOGICAL,             INTENT(IN) :: OSEND_LAKE
LOGICAL,             INTENT(IN) :: OSEND_SEA
LOGICAL,             INTENT(IN) :: OSEND_WAVE
!
REAL, DIMENSION(KI), INTENT(IN) :: PLAND_RUNOFF    ! Cumulated Surface runoff             (kg/m2)
REAL, DIMENSION(KI), INTENT(IN) :: PLAND_DRAIN     ! Cumulated Deep drainage              (kg/m2)
REAL, DIMENSION(KI), INTENT(IN) :: PLAND_CALVING   ! Cumulated Calving flux               (kg/m2)
REAL, DIMENSION(KI), INTENT(IN) :: PLAND_SRCFLOOD  ! Cumulated flood freshwater flux      (kg/m2)
!
REAL, DIMENSION(KI), INTENT(IN) :: PLAKE_EVAP  ! Cumulated Evaporation              (kg/m2)
REAL, DIMENSION(KI), INTENT(IN) :: PLAKE_RAIN  ! Cumulated Rainfall rate            (kg/m2)
REAL, DIMENSION(KI), INTENT(IN) :: PLAKE_SNOW  ! Cumulated Snowfall rate            (kg/m2)
REAL, DIMENSION(KI), INTENT(IN) :: PLAKE_WATF  ! Cumulated freshwater flux          (kg/m2)
!
REAL, DIMENSION(KI), INTENT(IN) :: PSEA_FWSU  ! Cumulated zonal wind stress       (Pa.s)
REAL, DIMENSION(KI), INTENT(IN) :: PSEA_FWSV  ! Cumulated meridian wind stress    (Pa.s)
REAL, DIMENSION(KI), INTENT(IN) :: PSEA_HEAT  ! Cumulated Non solar net heat flux (J/m2)
REAL, DIMENSION(KI), INTENT(IN) :: PSEA_SNET  ! Cumulated Solar net heat flux     (J/m2)
REAL, DIMENSION(KI), INTENT(IN) :: PSEA_WIND  ! Cumulated 10m wind speed          (m)
REAL, DIMENSION(KI), INTENT(IN) :: PSEA_FWSM  ! Cumulated wind stress             (Pa.s)
REAL, DIMENSION(KI), INTENT(IN) :: PSEA_EVAP  ! Cumulated Evaporation             (kg/m2)
REAL, DIMENSION(KI), INTENT(IN) :: PSEA_RAIN  ! Cumulated Rainfall rate           (kg/m2)
REAL, DIMENSION(KI), INTENT(IN) :: PSEA_SNOW  ! Cumulated Snowfall rate           (kg/m2)
REAL, DIMENSION(KI), INTENT(IN) :: PSEA_EVPR  ! Evap. - Precip. rate              (kg/m2)
REAL, DIMENSION(KI), INTENT(IN) :: PSEA_WATF  ! Cumulated freshwater flux         (kg/m2)
REAL, DIMENSION(KI), INTENT(IN) :: PSEA_PRES  ! Cumulated Surface pressure        (Pa.s)
REAL, DIMENSION(KI), INTENT(IN) :: PSEA_LWFL  ! Cumulated long-wave heat flux     (J/m2)
REAL, DIMENSION(KI), INTENT(IN) :: PSEA_LHFL  ! Cumulated latent heat flux        (J/m2)
REAL, DIMENSION(KI), INTENT(IN) :: PSEA_SHFL  ! Cumulated sensible heat flux      (J/m2)
!
REAL, DIMENSION(KI), INTENT(IN) :: PSEAICE_HEAT ! Cumulated Sea-ice non solar net heat flux (J/m2)
REAL, DIMENSION(KI), INTENT(IN) :: PSEAICE_SNET ! Cumulated Sea-ice solar net heat flux     (J/m2)
REAL, DIMENSION(KI), INTENT(IN) :: PSEAICE_EVAP ! Cumulated Sea-ice sublimation             (kg/m2)
!
REAL, DIMENSION(KI), INTENT(IN) :: PWAVE_U10  ! 
REAL, DIMENSION(KI), INTENT(IN) :: PWAVE_V10  !
!
!*       0.2   Declarations of local variables
!              -------------------------------
!
REAL, DIMENSION(KI,1) :: ZWRITE ! Mean flux send to OASIS (Pa, m/s, W/m2 or kg/m2/s)
!
CHARACTER(LEN=50)     :: YCOMMENT
INTEGER               :: IERR   ! Error info
!
REAL(KIND=JPRB) :: ZHOOK_HANDLE
!
!-------------------------------------------------------------------------------
#ifdef CPLOASIS
!-------------------------------------------------------------------------------
!
IF (LHOOK) CALL DR_HOOK('SFX_OASIS_SEND',0,ZHOOK_HANDLE)
!
!*       1.     Initialize :
!               ------------
!
ZWRITE(:,:) = XUNDEF
!
!-------------------------------------------------------------------------------
!
!*       2.     Send land fields to OASIS:
!               --------------------------
!
IF(OSEND_LAND)THEN
!
! * Send river output fields
!
  YCOMMENT='Surface runoff over land'
  CALL OUTVAR(PLAND_RUNOFF,XTSTEP_CPL_LAND,ZWRITE(:,1))
  CALL OASIS_PUT(NRUNOFF_ID,KDATE,ZWRITE(:,:),IERR)
  CALL CHECK_SFX_SEND(KLUOUT,IERR,YCOMMENT,ZWRITE(:,1))
!
  YCOMMENT='Deep drainage over land'
  CALL OUTVAR(PLAND_DRAIN,XTSTEP_CPL_LAND,ZWRITE(:,1))
  CALL OASIS_PUT(NDRAIN_ID,KDATE,ZWRITE(:,:),IERR)
  CALL CHECK_SFX_SEND(KLUOUT,IERR,YCOMMENT,ZWRITE(:,1))
!
  IF(LCPL_CALVING)THEN
    YCOMMENT='calving flux over land'
    CALL OUTVAR(PLAND_CALVING,XTSTEP_CPL_LAND,ZWRITE(:,1))
    CALL OASIS_PUT(NCALVING_ID,KDATE,ZWRITE(:,:),IERR)
    CALL CHECK_SFX_SEND(KLUOUT,IERR,YCOMMENT,ZWRITE(:,1))
  ENDIF
!
  IF(LCPL_FLOOD)THEN      
    YCOMMENT='flood freshwater flux over land (P-E-I)'
    CALL OUTVAR(PLAND_SRCFLOOD,XTSTEP_CPL_LAND,ZWRITE(:,1))
    CALL OASIS_PUT(NSRCFLOOD_ID,KDATE,ZWRITE(:,:),IERR)
    CALL CHECK_SFX_SEND(KLUOUT,IERR,YCOMMENT,ZWRITE(:,1)) 
  ENDIF
!
ENDIF
!
!-------------------------------------------------------------------------------
!
!*       3.     Send lake fields to OASIS :
!               --------------------------
IF(OSEND_LAKE)THEN
!
! * Send output fields (in kg/m2/s)
!
  IF(NLAKE_EVAP_ID/=NUNDEF)THEN
    YCOMMENT='Evaporation over lake'
    CALL OUTVAR(PLAKE_EVAP,XTSTEP_CPL_LAKE,ZWRITE(:,1))
    CALL OASIS_PUT(NLAKE_EVAP_ID,KDATE,ZWRITE(:,:),IERR)
    CALL CHECK_SFX_SEND(KLUOUT,IERR,YCOMMENT,ZWRITE(:,1))
  ENDIF
!
  IF(NLAKE_RAIN_ID/=NUNDEF)THEN
    YCOMMENT='Rainfall rate over lake'
    CALL OUTVAR(PLAKE_RAIN,XTSTEP_CPL_LAKE,ZWRITE(:,1))
    CALL OASIS_PUT(NLAKE_RAIN_ID,KDATE,ZWRITE(:,:),IERR)
    CALL CHECK_SFX_SEND(KLUOUT,IERR,YCOMMENT,ZWRITE(:,1))
  ENDIF
!
  IF(NLAKE_SNOW_ID/=NUNDEF)THEN
    YCOMMENT='Snowfall rate over lake'
    CALL OUTVAR(PLAKE_SNOW,XTSTEP_CPL_LAKE,ZWRITE(:,1))
    CALL OASIS_PUT(NLAKE_SNOW_ID,KDATE,ZWRITE(:,:),IERR)
    CALL CHECK_SFX_SEND(KLUOUT,IERR,YCOMMENT,ZWRITE(:,1))
  ENDIF
!
  IF(NLAKE_WATF_ID/=NUNDEF)THEN
    YCOMMENT='Freshwater flux over lake (P-E)'
    CALL OUTVAR(PLAKE_WATF,XTSTEP_CPL_LAKE,ZWRITE(:,1))
    CALL OASIS_PUT(NLAKE_WATF_ID,KDATE,ZWRITE(:,:),IERR)
    CALL CHECK_SFX_SEND(KLUOUT,IERR,YCOMMENT,ZWRITE(:,1))
  ENDIF


ENDIF
!
!-------------------------------------------------------------------------------
!
!*       4.     Send sea fields to OASIS :
!               --------------------------
!
IF(OSEND_SEA)THEN
!
! * Send sea output fields (in Pa, m/s, W/m2 or kg/m2/s)
!
  IF(NSEA_FWSU_ID/=NUNDEF)THEN
    YCOMMENT='zonal wind stress over sea'
    CALL OUTVAR(PSEA_FWSU,XTSTEP_CPL_SEA,ZWRITE(:,1))
    CALL OASIS_PUT(NSEA_FWSU_ID,KDATE,ZWRITE(:,:),IERR)
    CALL CHECK_SFX_SEND(KLUOUT,IERR,YCOMMENT,ZWRITE(:,1))
  ENDIF
!
  IF(NSEA_FWSV_ID/=NUNDEF)THEN
    YCOMMENT='meridian wind stress over sea'
    CALL OUTVAR(PSEA_FWSV,XTSTEP_CPL_SEA,ZWRITE(:,1))
    CALL OASIS_PUT(NSEA_FWSV_ID,KDATE,ZWRITE(:,:),IERR)
    CALL CHECK_SFX_SEND(KLUOUT,IERR,YCOMMENT,ZWRITE(:,1))
  ENDIF
!
  IF(NSEA_HEAT_ID/=NUNDEF)THEN
    YCOMMENT='Non solar net heat flux over sea'
    CALL OUTVAR(PSEA_HEAT,XTSTEP_CPL_SEA,ZWRITE(:,1))
    CALL OASIS_PUT(NSEA_HEAT_ID,KDATE,ZWRITE(:,:),IERR)
    CALL CHECK_SFX_SEND(KLUOUT,IERR,YCOMMENT,ZWRITE(:,1))
  ENDIF
!
  IF(NSEA_SNET_ID/=NUNDEF)THEN
    YCOMMENT='Solar net heat flux over sea'
    CALL OUTVAR(PSEA_SNET,XTSTEP_CPL_SEA,ZWRITE(:,1))
    CALL OASIS_PUT(NSEA_SNET_ID,KDATE,ZWRITE(:,:),IERR)
    CALL CHECK_SFX_SEND(KLUOUT,IERR,YCOMMENT,ZWRITE(:,1))
  ENDIF
!
  IF(NSEA_WIND_ID/=NUNDEF)THEN
    YCOMMENT='10m wind speed over sea'
    CALL OUTVAR(PSEA_WIND,XTSTEP_CPL_SEA,ZWRITE(:,1))
    CALL OASIS_PUT(NSEA_WIND_ID,KDATE,ZWRITE(:,:),IERR)
    CALL CHECK_SFX_SEND(KLUOUT,IERR,YCOMMENT,ZWRITE(:,1))
  ENDIF
!
  IF(NSEA_FWSM_ID/=NUNDEF)THEN
    YCOMMENT='wind stress over sea'
    CALL OUTVAR(PSEA_FWSM,XTSTEP_CPL_SEA,ZWRITE(:,1))
    CALL OASIS_PUT(NSEA_FWSM_ID,KDATE,ZWRITE(:,:),IERR)
    CALL CHECK_SFX_SEND(KLUOUT,IERR,YCOMMENT,ZWRITE(:,1))
  ENDIF
!
  IF(NSEA_EVAP_ID/=NUNDEF)THEN
    YCOMMENT='Evaporation over sea'
    CALL OUTVAR(PSEA_EVAP,XTSTEP_CPL_SEA,ZWRITE(:,1))
    CALL OASIS_PUT(NSEA_EVAP_ID,KDATE,ZWRITE(:,:),IERR)
    CALL CHECK_SFX_SEND(KLUOUT,IERR,YCOMMENT,ZWRITE(:,1))
  ENDIF
!
  IF(NSEA_RAIN_ID/=NUNDEF)THEN
    YCOMMENT='Rainfall rate over sea'
    CALL OUTVAR(PSEA_RAIN,XTSTEP_CPL_SEA,ZWRITE(:,1))
    CALL OASIS_PUT(NSEA_RAIN_ID,KDATE,ZWRITE(:,:),IERR)
    CALL CHECK_SFX_SEND(KLUOUT,IERR,YCOMMENT,ZWRITE(:,1))
  ENDIF
!
  IF(NSEA_SNOW_ID/=NUNDEF)THEN
    YCOMMENT='Snowfall rate over sea'
    CALL OUTVAR(PSEA_SNOW,XTSTEP_CPL_SEA,ZWRITE(:,1))
    CALL OASIS_PUT(NSEA_SNOW_ID,KDATE,ZWRITE(:,:),IERR)
    CALL CHECK_SFX_SEND(KLUOUT,IERR,YCOMMENT,ZWRITE(:,1))
  ENDIF
!
  IF(NSEA_EVPR_ID/=NUNDEF)THEN
    YCOMMENT='Evap. - Precip. rate over sea'
    CALL OUTVAR(PSEA_EVPR,XTSTEP_CPL_SEA,ZWRITE(:,1))
    CALL OASIS_PUT(NSEA_EVPR_ID,KDATE,ZWRITE(:,:),IERR)
    CALL CHECK_SFX_SEND(KLUOUT,IERR,YCOMMENT,ZWRITE(:,1))
  ENDIF
!
  IF(NSEA_WATF_ID/=NUNDEF)THEN
    YCOMMENT='Freshwater flux over sea (P-E)'
    CALL OUTVAR(PSEA_WATF,XTSTEP_CPL_SEA,ZWRITE(:,1))
    CALL OASIS_PUT(NSEA_WATF_ID,KDATE,ZWRITE(:,:),IERR)
    CALL CHECK_SFX_SEND(KLUOUT,IERR,YCOMMENT,ZWRITE(:,1))
  ENDIF
!
  IF(NSEA_PRES_ID/=NUNDEF)THEN
    YCOMMENT='Surface pressure'
    CALL OUTVAR(PSEA_PRES,XTSTEP_CPL_SEA,ZWRITE(:,1))
    CALL OASIS_PUT(NSEA_PRES_ID,KDATE,ZWRITE(:,:),IERR)
    CALL CHECK_SFX_SEND(KLUOUT,IERR,YCOMMENT,ZWRITE(:,1))
  ENDIF
!
  IF(NSEA_LWFL_ID/=NUNDEF)THEN
    YCOMMENT='Long-wave heat flux'
    CALL OUTVAR(PSEA_LWFL,XTSTEP_CPL_SEA,ZWRITE(:,1))
    CALL OASIS_PUT(NSEA_LWFL_ID,KDATE,ZWRITE(:,:),IERR)
    CALL CHECK_SFX_SEND(KLUOUT,IERR,YCOMMENT,ZWRITE(:,1))
  ENDIF
!
  IF(NSEA_LHFL_ID/=NUNDEF)THEN
    YCOMMENT='Latent heat flux'
    CALL OUTVAR(PSEA_LHFL,XTSTEP_CPL_SEA,ZWRITE(:,1))
    CALL OASIS_PUT(NSEA_LHFL_ID,KDATE,ZWRITE(:,:),IERR)
    CALL CHECK_SFX_SEND(KLUOUT,IERR,YCOMMENT,ZWRITE(:,1))
  ENDIF
!
  IF(NSEA_SHFL_ID/=NUNDEF)THEN
    YCOMMENT='Sensible heat flux'
    CALL OUTVAR(PSEA_SHFL,XTSTEP_CPL_SEA,ZWRITE(:,1))
    CALL OASIS_PUT(NSEA_SHFL_ID,KDATE,ZWRITE(:,:),IERR)
    CALL CHECK_SFX_SEND(KLUOUT,IERR,YCOMMENT,ZWRITE(:,1))
  ENDIF
!
! * Sea-ice output fields (in W/m2 or kg/m2/s)
!
  IF(LCPL_SEAICE)THEN
!
    IF(NSEAICE_HEAT_ID/=NUNDEF)THEN
      YCOMMENT='Sea-ice non solar net heat flux over sea-ice'
      CALL OUTVAR(PSEAICE_HEAT,XTSTEP_CPL_SEA,ZWRITE(:,1))
      CALL OASIS_PUT(NSEAICE_HEAT_ID,KDATE,ZWRITE(:,:),IERR)
      CALL CHECK_SFX_SEND(KLUOUT,IERR,YCOMMENT,ZWRITE(:,1))
    ENDIF
!
    IF(NSEAICE_SNET_ID/=NUNDEF)THEN
      YCOMMENT='Sea-ice solar net heat flux over sea-ice'
      CALL OUTVAR(PSEAICE_SNET,XTSTEP_CPL_SEA,ZWRITE(:,1))
      CALL OASIS_PUT(NSEAICE_SNET_ID,KDATE,ZWRITE(:,:),IERR)
      CALL CHECK_SFX_SEND(KLUOUT,IERR,YCOMMENT,ZWRITE(:,1))
    ENDIF
!
    IF(NSEAICE_EVAP_ID/=NUNDEF)THEN
      YCOMMENT='Sea-ice sublimation over sea-ice'
      CALL OUTVAR(PSEAICE_EVAP,XTSTEP_CPL_SEA,ZWRITE(:,1))
      CALL OASIS_PUT(NSEAICE_EVAP_ID,KDATE,ZWRITE(:,:),IERR)
      CALL CHECK_SFX_SEND(KLUOUT,IERR,YCOMMENT,ZWRITE(:,1))
    ENDIF
!
  ENDIF
!
ENDIF
!
!-------------------------------------------------------------------------------
!
!*       5.     Send wave fields to OASIS :
!               --------------------------
IF(OSEND_WAVE)THEN
!
! * Send output fields
!
  IF(NWAVE_U10_ID/=NUNDEF)THEN
    YCOMMENT='10m u-wind speed'
    ZWRITE(:,1) = PWAVE_U10(:)
    CALL OASIS_PUT(NWAVE_U10_ID,KDATE,ZWRITE(:,:),IERR)
    CALL CHECK_SFX_SEND(KLUOUT,IERR,YCOMMENT,ZWRITE(:,1))
  ENDIF
!
  IF(NWAVE_V10_ID/=NUNDEF)THEN
    YCOMMENT='10m v-wind speed'
    ZWRITE(:,1) = PWAVE_V10(:)
    CALL OASIS_PUT(NWAVE_V10_ID,KDATE,ZWRITE(:,:),IERR)
    CALL CHECK_SFX_SEND(KLUOUT,IERR,YCOMMENT,ZWRITE(:,1))
  ENDIF
!
ENDIF
!
IF (LHOOK) CALL DR_HOOK('SFX_OASIS_SEND',1,ZHOOK_HANDLE)
!
!-------------------------------------------------------------------------------
CONTAINS
!-------------------------------------------------------------------------------
!
SUBROUTINE CHECK_SFX_SEND(KLUOUT,KERR,HCOMMENT,PWRITE)
!
USE MODI_ABOR1_SFX
!
IMPLICIT NONE
!
INTEGER,          INTENT(IN) :: KLUOUT
INTEGER,          INTENT(IN) :: KERR
CHARACTER(LEN=*), INTENT(IN) :: HCOMMENT
!
REAL, DIMENSION(:), INTENT(OUT):: PWRITE
!
REAL(KIND=JPRB) :: ZHOOK_HANDLE
!
IF (LHOOK) CALL DR_HOOK('SFX_OASIS_SEND:CHECK_SFX_SEND',0,ZHOOK_HANDLE)
!
PWRITE(:) = XUNDEF
!
IF (KERR/=OASIS_OK.AND.KERR<OASIS_SENT) THEN
   WRITE(KLUOUT,'(A,I4)')'Return OASIS code from sending '//TRIM(HCOMMENT)//' : ',KERR
   CALL ABOR1_SFX('SFX_OASIS_SEND: problem sending '//TRIM(HCOMMENT))
ENDIF 
!
IF (LHOOK) CALL DR_HOOK('SFX_OASIS_SEND:CHECK_SFX_SEND',1,ZHOOK_HANDLE)
!
END SUBROUTINE CHECK_SFX_SEND
!
!-------------------------------------------------------------------------------
!
SUBROUTINE OUTVAR(PIN,PDIV,PWRITE)
!
IMPLICIT NONE
!
REAL, DIMENSION(:), INTENT(IN) :: PIN
REAL,               INTENT(IN) :: PDIV
!
REAL, DIMENSION(:), INTENT(OUT):: PWRITE
!
REAL(KIND=JPRB) :: ZHOOK_HANDLE
!
IF (LHOOK) CALL DR_HOOK('SFX_OASIS_SEND:OUTVAR',0,ZHOOK_HANDLE)
!
WHERE(PIN(:)/=XUNDEF)
     PWRITE(:)=PIN(:)/PDIV
ELSEWHERE
     PWRITE(:)=XUNDEF
ENDWHERE
!
IF (LHOOK) CALL DR_HOOK('SFX_OASIS_SEND:OUTVAR',1,ZHOOK_HANDLE)
!
END SUBROUTINE OUTVAR
!
!-------------------------------------------------------------------------------
#endif
!-------------------------------------------------------------------------------
!
END SUBROUTINE SFX_OASIS_SEND
