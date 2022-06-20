!MNH_LIC Copyright 1994-2014 CNRS, Meteo-France and Universite Paul Sabatier
!MNH_LIC This is part of the Meso-NH software governed by the CeCILL-C licence
!MNH_LIC version 1. See LICENSE, CeCILL-C_V1-en.txt and CeCILL-C_V1-fr.txt  
!MNH_LIC for details. version 1.
!     ##########
MODULE MODI_MNH_OASIS_SEND
!     ##########
!
INTERFACE 
!
      SUBROUTINE MNH_OASIS_SEND(HPROGRAM,KI,PTIMEC,PSTEP_SURF)
!
      CHARACTER(LEN=*),      INTENT(IN) :: HPROGRAM
      INTEGER,               INTENT(IN) :: KI            ! number of points
      REAL,                  INTENT(IN) :: PTIMEC        ! Cumulated run time step (s)
      REAL,                  INTENT(IN) :: PSTEP_SURF    ! Model time step (s)
!
      END SUBROUTINE MNH_OASIS_SEND
!
END INTERFACE
!
END MODULE MODI_MNH_OASIS_SEND
!
!     ####################################################################
SUBROUTINE MNH_OASIS_SEND(HPROGRAM,KI,PTIMEC,PSTEP_SURF)
!     ####################################################################
!
!!****  *MNH_OASIS_SEND* 
!!
!!    PURPOSE
!!    -------
!!    Meso-NH driver to send coupling fields
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
!!	J. Pianezze   *LPO*	
!!
!!    MODIFICATIONS
!!    -------------
!!      Original    09/2014
!-------------------------------------------------------------------------------
!
!*       0.    DECLARATIONS
!              ------------
!
USE MODD_SURF_PAR,   ONLY : XUNDEF
!
USE MODN_SFX_OASIS,  ONLY : XTSTEP_CPL_LAND, &
                            XTSTEP_CPL_LAKE, &
                            XTSTEP_CPL_SEA , &
                            XTSTEP_CPL_WAVE, &
                            LWATER
!
USE MODD_SFX_OASIS,  ONLY : LCPL_LAND,LCPL_GW,       &
                            LCPL_FLOOD,LCPL_CALVING, &
                            LCPL_LAKE,               &
                            LCPL_SEA,LCPL_SEAICE,    &
                            LCPL_WAVE
!                           
USE MODD_MNH_SURFEX_n
!
USE MODI_GET_SFX_LAND
USE MODI_GET_SFX_LAKE
USE MODI_GET_SFX_SEA
USE MODI_GET_SFX_WAVE
!
USE MODI_GET_LUOUT
!
IMPLICIT NONE
!
!*       0.1   Declarations of arguments
!              -------------------------
!
CHARACTER(LEN=*),      INTENT(IN) :: HPROGRAM
INTEGER,               INTENT(IN) :: KI            ! number of points
REAL,                  INTENT(IN) :: PTIMEC        ! Cumulated run time step (s)
REAL,                  INTENT(IN) :: PSTEP_SURF    ! Model time step (s)
!
!
!*       0.2   Declarations of local variables
!              -------------------------------
!
REAL, DIMENSION(KI)   :: ZLAND_RUNOFF    ! Cumulated Surface runoff             (kg/m2)
REAL, DIMENSION(KI)   :: ZLAND_DRAIN     ! Cumulated Deep drainage              (kg/m2)
REAL, DIMENSION(KI)   :: ZLAND_CALVING   ! Cumulated Calving flux               (kg/m2)
REAL, DIMENSION(KI)   :: ZLAND_RECHARGE  ! Cumulated Recharge to groundwater    (kg/m2)
!
REAL, DIMENSION(KI)   :: ZLAKE_EVAP  ! Cumulated Evaporation             (kg/m2)
REAL, DIMENSION(KI)   :: ZLAKE_RAIN  ! Cumulated Rainfall rate           (kg/m2)
REAL, DIMENSION(KI)   :: ZLAKE_SNOW  ! Cumulated Snowfall rate           (kg/m2)
REAL, DIMENSION(KI)   :: ZLAKE_WATF  ! Cumulated net freshwater rate     (kg/m2)
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
REAL, DIMENSION(KI)   :: ZSEA_EVPR  ! Cumulated Evap-Precip rate        (kg/m2)
REAL, DIMENSION(KI)   :: ZSEA_WATF  ! Cumulated net freshwater rate     (kg/m2)
REAL, DIMENSION(KI)   :: ZSEA_PRES  ! Cumulated Surface pressure        (Pa.s)
REAL, DIMENSION(KI)   :: ZSEA_LWFL  ! Cumulated long wave heat flux     (J/m2)
REAL, DIMENSION(KI)   :: ZSEA_LHFL  ! Cumulated latent heat flux        (J/m2)
REAL, DIMENSION(KI)   :: ZSEA_SHFL  ! Cumulated sensible heat flux      (J/m2)
!
REAL, DIMENSION(KI)   :: ZSEAICE_HEAT ! Cumulated Sea-ice non solar net heat flux (J/m2)
REAL, DIMENSION(KI)   :: ZSEAICE_SNET ! Cumulated Sea-ice solar net heat flux     (J/m2)
REAL, DIMENSION(KI)   :: ZSEAICE_EVAP ! Cumulated Sea-ice sublimation             (kg/m2)
!
REAL, DIMENSION(KI)   :: ZWAVE_U10    ! 10m u-wind speed (m/s)
REAL, DIMENSION(KI)   :: ZWAVE_V10    ! 10m v-wind speed (m/s)
!
INTEGER               :: IDATE  ! current coupling time step (s)
INTEGER               :: ILUOUT
INTEGER               :: INKPROMA
!
LOGICAL               :: GSEND_LAND
LOGICAL               :: GSEND_LAKE
LOGICAL               :: GSEND_SEA
LOGICAL               :: GSEND_WAVE
!
REAL(KIND=JPRB) :: ZHOOK_HANDLE
!
!-------------------------------------------------------------------------------
!
!*       1.     Initialize proc by proc :
!               -------------------------
!
CALL GET_LUOUT(HPROGRAM,ILUOUT)
!
IDATE = INT(PTIMEC-PSTEP_SURF)
!
GSEND_LAND =(LCPL_LAND .AND. MOD(PTIMEC,XTSTEP_CPL_LAND)==0.0)
GSEND_LAKE =(LCPL_LAKE .AND. MOD(PTIMEC,XTSTEP_CPL_LAKE)==0.0)
GSEND_SEA  =(LCPL_SEA  .AND. MOD(PTIMEC,XTSTEP_CPL_SEA) ==0.0)
GSEND_WAVE =(LCPL_WAVE .AND. MOD(PTIMEC,XTSTEP_CPL_WAVE)==0.0)
!
!-------------------------------------------------------------------------------
!
IF(GSEND_LAND)THEN
  ZLAND_RUNOFF  (:) = XUNDEF
  ZLAND_DRAIN   (:) = XUNDEF
  ZLAND_CALVING (:) = XUNDEF
  ZLAND_RECHARGE(:) = XUNDEF
ENDIF
!
IF(GSEND_LAKE)THEN
  ZLAKE_EVAP (:) = XUNDEF
  ZLAKE_RAIN (:) = XUNDEF
  ZLAKE_SNOW (:) = XUNDEF
  ZSEA_WATF  (:) = XUNDEF  
ENDIF
!
IF(GSEND_SEA)THEN
  ZSEA_FWSU (:) = XUNDEF
  ZSEA_FWSV (:) = XUNDEF
  ZSEA_HEAT (:) = XUNDEF
  ZSEA_SNET (:) = XUNDEF
  ZSEA_WIND (:) = XUNDEF
  ZSEA_FWSM (:) = XUNDEF
  ZSEA_EVAP (:) = XUNDEF
  ZSEA_RAIN (:) = XUNDEF
  ZSEA_SNOW (:) = XUNDEF
  ZSEA_EVPR (:) = XUNDEF
  ZSEA_WATF (:) = XUNDEF
  ZSEA_PRES (:) = XUNDEF
  ZSEA_LWFL (:) = XUNDEF
  ZSEA_LHFL (:) = XUNDEF
  ZSEA_SHFL (:) = XUNDEF
  !
  ZSEAICE_HEAT (:) = XUNDEF
  ZSEAICE_SNET (:) = XUNDEF
  ZSEAICE_EVAP (:) = XUNDEF
ENDIF
!
IF(GSEND_WAVE)THEN
  ZWAVE_U10 (:) = XUNDEF
  ZWAVE_V10 (:) = XUNDEF
ENDIF
!
!-------------------------------------------------------------------------------
!
!*       2.     get local fields :
!               ------------------
!
IF(GSEND_LAND)THEN
!
! * Get river output fields
!
  CALL GET_SFX_LAND(YSURF_CUR%IM%O, YSURF_CUR%IM%S, YSURF_CUR%U,   &
                    LCPL_GW,LCPL_FLOOD,LCPL_CALVING,    &
                    ZLAND_RUNOFF (:),ZLAND_DRAIN   (:), &
                    ZLAND_CALVING(:),ZLAND_RECHARGE(:)   )
!
ENDIF
!
IF(GSEND_LAKE)THEN
!
! * Get output fields
!
  CALL GET_SFX_LAKE(YSURF_CUR%FM%F,YSURF_CUR%U,  &
                    ZLAKE_EVAP(:),ZLAKE_RAIN(:), &
                    ZLAKE_SNOW(:),ZLAKE_WATF(:)   )
!
ENDIF
!
IF(GSEND_SEA)THEN
!
! * Get sea output fields
!
  CALL GET_SFX_SEA(YSURF_CUR%SM%S, YSURF_CUR%U, YSURF_CUR%WM%W,     &
                    LCPL_SEAICE, LWATER,                            &
                    ZSEA_FWSU   (:),ZSEA_FWSV   (:),ZSEA_HEAT   (:),&
                    ZSEA_SNET   (:),ZSEA_WIND   (:),ZSEA_FWSM   (:),&
                    ZSEA_EVAP   (:),ZSEA_RAIN   (:),ZSEA_SNOW   (:),&
                    ZSEA_EVPR   (:),ZSEA_WATF   (:),ZSEA_PRES   (:),&
                    ZSEA_LWFL   (:),ZSEA_LHFL   (:),ZSEA_SHFL   (:),&
                    ZSEAICE_HEAT(:),ZSEAICE_SNET(:),ZSEAICE_EVAP(:) )
!
ENDIF
!
IF(GSEND_WAVE)THEN
!
! * Get wave output fields
!
  CALL GET_SFX_WAVE(YSURF_CUR%U,  YSURF_CUR%SM%SD%D, &
                    ZWAVE_U10(:), ZWAVE_V10(:)      )
!
ENDIF
!
!    
!-------------------------------------------------------------------------------
!
!*       3.     Send fields to OASIS proc by proc:
!               ----------------------------------
!
CALL SFX_OASIS_SEND(ILUOUT,KI,IDATE,GSEND_LAND,GSEND_LAKE,GSEND_SEA,GSEND_WAVE, &
                    ZLAND_RUNOFF,ZLAND_DRAIN,ZLAND_CALVING,ZLAND_RECHARGE,      &
                    ZLAKE_EVAP,ZLAKE_RAIN,ZLAKE_SNOW,ZLAKE_WATF,                &
                    ZSEA_FWSU,ZSEA_FWSV,ZSEA_HEAT,ZSEA_SNET,ZSEA_WIND,          &
                    ZSEA_FWSM,ZSEA_EVAP,ZSEA_RAIN,ZSEA_SNOW,                    &
                    ZSEA_EVPR,ZSEA_WATF,                                        &
                    ZSEA_PRES,ZSEA_LWFL,ZSEA_LHFL,ZSEA_SHFL,                    &
                    ZSEAICE_HEAT,ZSEAICE_SNET,ZSEAICE_EVAP,                     &
                    ZWAVE_U10, ZWAVE_V10      )
!
!-------------------------------------------------------------------------------
!
END SUBROUTINE MNH_OASIS_SEND
