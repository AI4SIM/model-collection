!SFX_LIC Copyright 1994-2014 CNRS, Meteo-France and Universite Paul Sabatier
!SFX_LIC This is part of the SURFEX software governed by the CeCILL-C licence
!SFX_LIC version 1. See LICENSE, CeCILL-C_V1-en.txt and CeCILL-C_V1-fr.txt  
!SFX_LIC for details. version 1.
!     #########
      SUBROUTINE GET_SFXCPL_n (IM, S, U, W, &
                               HPROGRAM,KI,PRUI,PWIND,PFWSU,PFWSV,PSNET, &
                                PHEAT,PEVAP,PRAIN,PSNOW,PEVPR,PICEFLUX,  &
                                PFWSM,PPS,PHEAT_ICE,PEVAP_ICE,PSNET_ICE)  
!     ###################################################################
!
!!****  *GETSFXCPL_n* - routine to get some variables from surfex into
!                       ocean and/or a river routing model when the coupler
!                       is not in SURFEX but in ARPEGE.
!
!                       !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!                       This routine will be suppress soon.
!                       !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!
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
!!      B. Decharme      *Meteo France*
!!
!!    MODIFICATIONS
!!    -------------
!!      Original    08/2009
!!    10/2016 B. Decharme : bug surface/groundwater coupling 
!!      Modified    11/2014 : J. Pianezze - Add surface pressure coupling parameter
!-------------------------------------------------------------------------------
!
!*       0.    DECLARATIONS
!              ------------
!
!
!
!
USE MODD_SURFEX_n, ONLY : ISBA_MODEL_t
USE MODD_SEAFLUX_n, ONLY : SEAFLUX_t
USE MODD_SURF_ATM_n, ONLY : SURF_ATM_t
USE MODD_WATFLUX_n, ONLY : WATFLUX_t
!
USE MODD_SURF_PAR,   ONLY : XUNDEF
!
USE MODN_SFX_OASIS,  ONLY : LWATER
USE MODD_SFX_OASIS,  ONLY : LCPL_LAND, LCPL_CALVING, LCPL_GW, &
                            LCPL_FLOOD, LCPL_SEA, LCPL_SEAICE
!
USE MODI_GET_SFX_SEA
USE MODI_GET_SFX_LAND
USE MODI_ABOR1_SFX
USE MODI_GET_LUOUT
!
USE YOMHOOK   ,ONLY : LHOOK,   DR_HOOK
USE PARKIND1  ,ONLY : JPRB
!
USE MODI_GET_1D_MASK
!
USE MODI_GET_FRAC_n
IMPLICIT NONE
!
!*       0.1   Declarations of arguments
!              -------------------------
!
!
TYPE(ISBA_MODEL_t), INTENT(INOUT) :: IM
TYPE(SEAFLUX_t), INTENT(INOUT) :: S
TYPE(SURF_ATM_t), INTENT(INOUT) :: U
TYPE(WATFLUX_t), INTENT(INOUT) :: W
!
 CHARACTER(LEN=6),    INTENT(IN)  :: HPROGRAM
INTEGER,             INTENT(IN)  :: KI      ! number of points
!
REAL, DIMENSION(KI), INTENT(OUT) :: PRUI
REAL, DIMENSION(KI), INTENT(OUT) :: PWIND
REAL, DIMENSION(KI), INTENT(OUT) :: PFWSU
REAL, DIMENSION(KI), INTENT(OUT) :: PFWSV
REAL, DIMENSION(KI), INTENT(OUT) :: PSNET
REAL, DIMENSION(KI), INTENT(OUT) :: PHEAT
REAL, DIMENSION(KI), INTENT(OUT) :: PEVAP
REAL, DIMENSION(KI), INTENT(OUT) :: PRAIN
REAL, DIMENSION(KI), INTENT(OUT) :: PSNOW
REAL, DIMENSION(KI), INTENT(OUT) :: PEVPR
REAL, DIMENSION(KI), INTENT(OUT) :: PICEFLUX
REAL, DIMENSION(KI), INTENT(OUT) :: PFWSM
REAL, DIMENSION(KI), INTENT(OUT) :: PPS
REAL, DIMENSION(KI), INTENT(OUT) :: PHEAT_ICE
REAL, DIMENSION(KI), INTENT(OUT) :: PEVAP_ICE
REAL, DIMENSION(KI), INTENT(OUT) :: PSNET_ICE
!
!*       0.2   Declarations of local variables
!              -------------------------------
!
REAL, DIMENSION(KI)   :: ZRUNOFF    ! Cumulated Surface runoff             (kg/m2)
REAL, DIMENSION(KI)   :: ZDRAIN     ! Cumulated Deep drainage              (kg/m2)
REAL, DIMENSION(KI)   :: ZCALVING   ! Cumulated Calving flux               (kg/m2)
REAL, DIMENSION(KI)   :: ZSRCFLOOD  ! Cumulated flood freshwater flux      (kg/m2)
!
REAL, DIMENSION(KI)   :: ZSEA_FWSU  ! Cumulated zonal wind stress       (Pa.s)
REAL, DIMENSION(KI)   :: ZSEA_FWSV  ! Cumulated meridian wind stress    (Pa.s)
REAL, DIMENSION(KI)   :: ZSEA_HEAT  ! Cumulated Non solar net heat flux (J/m2)
REAL, DIMENSION(KI)   :: ZSEA_SNET  ! Cumulated Solar net heat flux     (J/m2)
REAL, DIMENSION(KI)   :: ZSEA_WIND  ! Cumulated 10m wind speed          (m)
REAL, DIMENSION(KI)   :: ZSEA_FWSM  ! Cumulated wind stress             (Pa.s)
REAL, DIMENSION(KI)   :: ZSEA_EVAP  ! Cumulated Evaporation             (kg/m2)
REAL, DIMENSION(KI)   :: ZSEA_RAIN  ! Cumulated Rainfall rate           (kg/m2)
REAL, DIMENSION(KI)   :: ZSEA_SNOW  ! Cumulated Snowfall rate           (kg/m2)
REAL, DIMENSION(KI)   :: ZSEA_EVPR  ! Cumulated Evap-Precp. rate        (kg/m2)
REAL, DIMENSION(KI)   :: ZSEA_WATF  ! Cumulated freshwater flux         (kg/m2)
REAL, DIMENSION(KI)   :: ZSEA_PRES  ! Cumulated Surface pressure        (Pa.s)
REAL, DIMENSION(KI)   :: ZSEA_LWFL  ! Cumulated long wave heat flux     (J/m2)
REAL, DIMENSION(KI)   :: ZSEA_LHFL  ! Cumulated latent heat flux        (J/m2)
REAL, DIMENSION(KI)   :: ZSEA_SHFL  ! Cumulated sensible heat flux      (J/m2)
!
REAL, DIMENSION(KI)   :: ZSEAICE_HEAT ! Cumulated Sea-ice non solar net heat flux (J/m2)
REAL, DIMENSION(KI)   :: ZSEAICE_SNET ! Cumulated Sea-ice solar net heat flux     (J/m2)
REAL, DIMENSION(KI)   :: ZSEAICE_EVAP ! Cumulated Sea-ice sublimation             (kg/m2)
!
INTEGER :: ILU, ILUOUT
!
REAL(KIND=JPRB) :: ZHOOK_HANDLE
!
!-------------------------------------------------------------------------------
IF (LHOOK) CALL DR_HOOK('GET_SFXCPL_N',0,ZHOOK_HANDLE)
!
CALL GET_LUOUT(HPROGRAM,ILUOUT)
!
!-------------------------------------------------------------------------------
! Global argument
!
IF(KI/=U%NSIZE_FULL)THEN
  WRITE(ILUOUT,*) 'size of field expected by the coupling :', KI
  WRITE(ILUOUT,*) 'size of field in SURFEX                :', U%NSIZE_FULL
  CALL ABOR1_SFX('GET_SFXCPL_N: VECTOR SIZE NOT CORRECT FOR COUPLING')
ENDIF
!
!-------------------------------------------------------------------------------
! Get variable over nature tile
!
IF(LCPL_LAND)THEN
!
! * Init land output fields
!
  ZRUNOFF  (:) = XUNDEF
  ZDRAIN   (:) = XUNDEF
  ZCALVING (:) = XUNDEF
  ZSRCFLOOD(:) = XUNDEF
!
! * Get land output fields
!       
  CALL GET_SFX_LAND(IM%O, IM%S, U, &
                    LCPL_GW,LCPL_FLOOD,LCPL_CALVING,    &
                    ZRUNOFF,ZDRAIN,ZCALVING,ZSRCFLOOD )
!
! * Assign land output fields
!        
  PRUI    (:) = ZRUNOFF (:)+ZDRAIN(:)
  PICEFLUX(:) = ZCALVING(:)
!
ENDIF
!
!-------------------------------------------------------------------------------
! Get variable over sea and water tiles and for ice
!
IF(LCPL_SEA)THEN
!
! * Init sea output fields
!
  ZSEA_FWSU (:) = XUNDEF
  ZSEA_FWSV (:) = XUNDEF
  ZSEA_HEAT (:) = XUNDEF
  ZSEA_SNET (:) = XUNDEF
  ZSEA_WIND (:) = XUNDEF
  ZSEA_FWSM (:) = XUNDEF
  ZSEA_EVAP (:) = XUNDEF
  ZSEA_RAIN (:) = XUNDEF
  ZSEA_SNOW (:) = XUNDEF
  ZSEA_WATF (:) = XUNDEF
  ZSEA_PRES (:) = XUNDEF
  ZSEA_LWFL (:) = XUNDEF
  ZSEA_LHFL (:) = XUNDEF
  ZSEA_SHFL (:) = XUNDEF
!
  ZSEAICE_HEAT (:) = XUNDEF
  ZSEAICE_SNET (:) = XUNDEF
  ZSEAICE_EVAP (:) = XUNDEF
!
! * Get sea output fields
!
  CALL GET_SFX_SEA(S, U, W, &
                   LCPL_SEAICE,LWATER,                      &
                   ZSEA_FWSU,ZSEA_FWSV,ZSEA_HEAT,ZSEA_SNET, &
                   ZSEA_WIND,ZSEA_FWSM,ZSEA_EVAP,ZSEA_RAIN, &
                   ZSEA_SNOW,ZSEA_EVPR,ZSEA_WATF,ZSEA_PRES, &
                   ZSEA_LWFL,ZSEA_LHFL,ZSEA_SHFL,           &
                   ZSEAICE_HEAT,ZSEAICE_SNET,ZSEAICE_EVAP   )
!
! * Assign sea output fields
!
  PFWSU     (:) = ZSEA_FWSU (:)
  PFWSV     (:) = ZSEA_FWSV (:)
  PSNET     (:) = ZSEA_SNET (:)
  PHEAT     (:) = ZSEA_HEAT (:)
  PEVAP     (:) = ZSEA_EVAP (:)
  PRAIN     (:) = ZSEA_RAIN (:)
  PSNOW     (:) = ZSEA_SNOW (:)
  PFWSM     (:) = ZSEA_FWSM (:)
  PHEAT_ICE (:) = ZSEAICE_HEAT (:)
  PEVAP_ICE (:) = ZSEAICE_EVAP (:)
  PSNET_ICE (:) = ZSEAICE_SNET (:)
!
ENDIF
!
!-------------------------------------------------------------------------------
IF (LHOOK) CALL DR_HOOK('GET_SFXCPL_N',1,ZHOOK_HANDLE)
!
END SUBROUTINE GET_SFXCPL_n
