!SFX_LIC Copyright 1994-2014 CNRS, Meteo-France and Universite Paul Sabatier
!SFX_LIC This is part of the SURFEX software governed by the CeCILL-C licence
!SFX_LIC version 1. See LICENSE, CeCILL-C_V1-en.txt and CeCILL-C_V1-fr.txt  
!SFX_LIC for details. version 1.
!     #########
      SUBROUTINE GET_SFX_SEA (S, U, W, &
                              OCPL_SEAICE,OWATER,                      &
                              PSEA_FWSU,PSEA_FWSV,PSEA_HEAT,PSEA_SNET, &
                              PSEA_WIND,PSEA_FWSM,PSEA_EVAP,PSEA_RAIN, &
                              PSEA_SNOW,PSEA_EVPR,PSEA_WATF,PSEA_PRES, &
                              PSEA_LWFL,PSEA_LHFL,PSEA_SHFL,           &
                              PSEAICE_HEAT,PSEAICE_SNET,PSEAICE_EVAP   )  
!     ############################################################################
!
!!****  *GET_SFX_SEA* - routine to get some variables from surfex to
!                        a oceanic general circulation model
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
!!      Original    10/2013
!!      Modified    11/2014 : J. Pianezze - Add surface pressure coupling parameter
!-------------------------------------------------------------------------------
!
!*       0.    DECLARATIONS
!              ------------
!
!
USE MODD_SEAFLUX_n, ONLY : SEAFLUX_t
USE MODD_SURF_ATM_n, ONLY : SURF_ATM_t
USE MODD_WATFLUX_n, ONLY : WATFLUX_t
!
USE MODD_SURF_PAR,   ONLY : XUNDEF
!
!
!
!
USE MODI_UNPACK_SAME_RANK
!
USE YOMHOOK   ,ONLY : LHOOK,   DR_HOOK
USE PARKIND1  ,ONLY : JPRB
!
IMPLICIT NONE
!
!*       0.1   Declarations of arguments
!              -------------------------
!
!
TYPE(SEAFLUX_t), INTENT(INOUT) :: S
TYPE(SURF_ATM_t), INTENT(INOUT) :: U
TYPE(WATFLUX_t), INTENT(INOUT) :: W
!
LOGICAL,            INTENT(IN)  :: OCPL_SEAICE ! sea-ice / ocean key
LOGICAL,            INTENT(IN)  :: OWATER      ! water included in sea smask
!
REAL, DIMENSION(:), INTENT(OUT) :: PSEA_FWSU  ! Cumulated zonal wind stress       (Pa.s)
REAL, DIMENSION(:), INTENT(OUT) :: PSEA_FWSV  ! Cumulated meridian wind stress    (Pa.s)
REAL, DIMENSION(:), INTENT(OUT) :: PSEA_HEAT  ! Cumulated Non solar net heat flux (J/m2)
REAL, DIMENSION(:), INTENT(OUT) :: PSEA_SNET  ! Cumulated Solar net heat flux     (J/m2)
REAL, DIMENSION(:), INTENT(OUT) :: PSEA_WIND  ! Cumulated 10m wind speed          (m)
REAL, DIMENSION(:), INTENT(OUT) :: PSEA_FWSM  ! Cumulated wind stress             (Pa.s)
REAL, DIMENSION(:), INTENT(OUT) :: PSEA_EVAP  ! Cumulated Evaporation             (kg/m2)
REAL, DIMENSION(:), INTENT(OUT) :: PSEA_RAIN  ! Cumulated Rainfall rate           (kg/m2)
REAL, DIMENSION(:), INTENT(OUT) :: PSEA_SNOW  ! Cumulated Snowfall rate           (kg/m2)
REAL, DIMENSION(:), INTENT(OUT) :: PSEA_EVPR  ! Cumulated Evap-Precip             (kg/m2)
REAL, DIMENSION(:), INTENT(OUT) :: PSEA_WATF  ! Cumulated Net water flux (kg/m2)
REAL, DIMENSION(:), INTENT(OUT) :: PSEA_PRES  ! Cumulated Surface pressure        (Pa.s)
REAL, DIMENSION(:), INTENT(OUT) :: PSEA_LWFL  ! Cumulated long wave heat flux     (J/m2)
REAL, DIMENSION(:), INTENT(OUT) :: PSEA_LHFL  ! Cumulated latent heat flux        (J/m2)
REAL, DIMENSION(:), INTENT(OUT) :: PSEA_SHFL  ! Cumulated sensible heat flux      (J/m2)
!
REAL, DIMENSION(:), INTENT(OUT) :: PSEAICE_HEAT ! Cumulated Sea-ice non solar net heat flux (J/m2)
REAL, DIMENSION(:), INTENT(OUT) :: PSEAICE_SNET ! Cumulated Sea-ice solar net heat flux     (J/m2)
REAL, DIMENSION(:), INTENT(OUT) :: PSEAICE_EVAP ! Cumulated Sea-ice sublimation             (kg/m2)
!
!*       0.2   Declarations of local variables
!              -------------------------------
!
REAL, DIMENSION(SIZE(PSEA_HEAT))  :: ZWIND
REAL, DIMENSION(SIZE(PSEA_HEAT))  :: ZFWSU
REAL, DIMENSION(SIZE(PSEA_HEAT))  :: ZFWSV
REAL, DIMENSION(SIZE(PSEA_HEAT))  :: ZSNET
REAL, DIMENSION(SIZE(PSEA_HEAT))  :: ZHEAT
REAL, DIMENSION(SIZE(PSEA_HEAT))  :: ZEVAP
REAL, DIMENSION(SIZE(PSEA_HEAT))  :: ZRAIN
REAL, DIMENSION(SIZE(PSEA_HEAT))  :: ZSNOW
REAL, DIMENSION(SIZE(PSEA_HEAT))  :: ZFWSM
!
REAL, DIMENSION(SIZE(PSEA_HEAT))  :: ZSNET_ICE
REAL, DIMENSION(SIZE(PSEA_HEAT))  :: ZHEAT_ICE
REAL, DIMENSION(SIZE(PSEA_HEAT))  :: ZEVAP_ICE
!
REAL(KIND=JPRB) :: ZHOOK_HANDLE
!
!-------------------------------------------------------------------------------
IF (LHOOK) CALL DR_HOOK('GET_SFX_SEA',0,ZHOOK_HANDLE)
!-------------------------------------------------------------------------------
!
!*       1.0   Initialization
!              --------------
!
PSEA_FWSU (:) = XUNDEF
PSEA_FWSV (:) = XUNDEF
PSEA_HEAT (:) = XUNDEF
PSEA_SNET (:) = XUNDEF
PSEA_WIND (:) = XUNDEF
PSEA_FWSM (:) = XUNDEF
PSEA_EVAP (:) = XUNDEF
PSEA_RAIN (:) = XUNDEF
PSEA_SNOW (:) = XUNDEF
PSEA_WATF (:) = XUNDEF
PSEA_PRES (:) = XUNDEF
PSEA_LWFL (:) = XUNDEF
PSEA_LHFL (:) = XUNDEF
PSEA_SHFL (:) = XUNDEF
!
PSEAICE_HEAT (:) = XUNDEF
PSEAICE_SNET (:) = XUNDEF
PSEAICE_EVAP (:) = XUNDEF
!
ZFWSU (:) = XUNDEF
ZFWSV (:) = XUNDEF
ZHEAT (:) = XUNDEF
ZSNET (:) = XUNDEF
ZWIND (:) = XUNDEF
ZFWSM (:) = XUNDEF
ZEVAP (:) = XUNDEF
ZRAIN (:) = XUNDEF
ZSNOW (:) = XUNDEF
!
ZHEAT_ICE (:) = XUNDEF
ZSNET_ICE (:) = XUNDEF
ZEVAP_ICE (:) = XUNDEF
!
!*       2.0   Get variable over sea
!              ---------------------
!
IF(U%NSIZE_SEA>0)THEN
!
  CALL UNPACK_SAME_RANK(U%NR_SEA(:),S%XCPL_SEA_WIND(:),PSEA_WIND(:),XUNDEF)
  CALL UNPACK_SAME_RANK(U%NR_SEA(:),S%XCPL_SEA_FWSU(:),PSEA_FWSU(:),XUNDEF)
  CALL UNPACK_SAME_RANK(U%NR_SEA(:),S%XCPL_SEA_FWSV(:),PSEA_FWSV(:),XUNDEF)
  CALL UNPACK_SAME_RANK(U%NR_SEA(:),S%XCPL_SEA_SNET(:),PSEA_SNET(:),XUNDEF)
  CALL UNPACK_SAME_RANK(U%NR_SEA(:),S%XCPL_SEA_HEAT(:),PSEA_HEAT(:),XUNDEF)
  CALL UNPACK_SAME_RANK(U%NR_SEA(:),S%XCPL_SEA_EVAP(:),PSEA_EVAP(:),XUNDEF)
  CALL UNPACK_SAME_RANK(U%NR_SEA(:),S%XCPL_SEA_RAIN(:),PSEA_RAIN(:),XUNDEF)
  CALL UNPACK_SAME_RANK(U%NR_SEA(:),S%XCPL_SEA_SNOW(:),PSEA_SNOW(:),XUNDEF)
  CALL UNPACK_SAME_RANK(U%NR_SEA(:),S%XCPL_SEA_EVPR(:),PSEA_EVPR(:),XUNDEF)
  CALL UNPACK_SAME_RANK(U%NR_SEA(:),S%XCPL_SEA_FWSM(:),PSEA_FWSM(:),XUNDEF)
  CALL UNPACK_SAME_RANK(U%NR_SEA(:),S%XCPL_SEA_PRES(:),PSEA_PRES(:),XUNDEF)
  CALL UNPACK_SAME_RANK(U%NR_SEA(:),S%XCPL_SEA_LWFL(:),PSEA_LWFL(:),XUNDEF)
  CALL UNPACK_SAME_RANK(U%NR_SEA(:),S%XCPL_SEA_LHFL(:),PSEA_LHFL(:),XUNDEF)
  CALL UNPACK_SAME_RANK(U%NR_SEA(:),S%XCPL_SEA_SHFL(:),PSEA_SHFL(:),XUNDEF)
  S%XCPL_SEA_WIND(:) = 0.0
  S%XCPL_SEA_EVAP(:) = 0.0
  S%XCPL_SEA_HEAT(:) = 0.0
  S%XCPL_SEA_SNET(:) = 0.0
  S%XCPL_SEA_FWSU(:) = 0.0
  S%XCPL_SEA_FWSV(:) = 0.0
  S%XCPL_SEA_RAIN(:) = 0.0
  S%XCPL_SEA_SNOW(:) = 0.0
  S%XCPL_SEA_EVPR(:) = 0.0
  S%XCPL_SEA_FWSM(:) = 0.0
  S%XCPL_SEA_PRES(:) = 0.0
  S%XCPL_SEA_LWFL(:) = 0.0
  S%XCPL_SEA_LHFL(:) = 0.0
  S%XCPL_SEA_SHFL(:) = 0.0
!
  IF (OCPL_SEAICE) THEN
    CALL UNPACK_SAME_RANK(U%NR_SEA(:),S%XCPL_SEAICE_SNET(:),PSEAICE_SNET(:),XUNDEF)
    CALL UNPACK_SAME_RANK(U%NR_SEA(:),S%XCPL_SEAICE_HEAT(:),PSEAICE_HEAT(:),XUNDEF)
    CALL UNPACK_SAME_RANK(U%NR_SEA(:),S%XCPL_SEAICE_EVAP(:),PSEAICE_EVAP(:),XUNDEF)
    S%XCPL_SEAICE_SNET(:) = 0.0
    S%XCPL_SEAICE_EVAP(:) = 0.0
    S%XCPL_SEAICE_HEAT(:) = 0.0  
  ENDIF
!
ENDIF
!
!*       3.0   Get variable over water without Flake
!              -------------------------------------
!
IF (OWATER.AND.U%NSIZE_WATER>0) THEN
!
  CALL UNPACK_SAME_RANK(U%NR_WATER(:),W%XCPL_WATER_WIND(:),ZWIND(:),XUNDEF)
  CALL UNPACK_SAME_RANK(U%NR_WATER(:),W%XCPL_WATER_FWSU(:),ZFWSU(:),XUNDEF)
  CALL UNPACK_SAME_RANK(U%NR_WATER(:),W%XCPL_WATER_FWSV(:),ZFWSV(:),XUNDEF)
  CALL UNPACK_SAME_RANK(U%NR_WATER(:),W%XCPL_WATER_SNET(:),ZSNET(:),XUNDEF)
  CALL UNPACK_SAME_RANK(U%NR_WATER(:),W%XCPL_WATER_HEAT(:),ZHEAT(:),XUNDEF)
  CALL UNPACK_SAME_RANK(U%NR_WATER(:),W%XCPL_WATER_EVAP(:),ZEVAP(:),XUNDEF)
  CALL UNPACK_SAME_RANK(U%NR_WATER(:),W%XCPL_WATER_RAIN(:),ZRAIN(:),XUNDEF)
  CALL UNPACK_SAME_RANK(U%NR_WATER(:),W%XCPL_WATER_SNOW(:),ZSNOW(:),XUNDEF)
  CALL UNPACK_SAME_RANK(U%NR_WATER(:),W%XCPL_WATER_FWSM(:),ZFWSM(:),XUNDEF)
!
  WHERE(U%XWATER(:)>0.0) 
    PSEA_WIND(:) = (U%XSEA(:)*PSEA_WIND(:)+U%XWATER(:)*ZWIND(:))/(U%XSEA(:)+U%XWATER(:))
    PSEA_FWSU(:) = (U%XSEA(:)*PSEA_FWSU(:)+U%XWATER(:)*ZFWSU(:))/(U%XSEA(:)+U%XWATER(:))
    PSEA_FWSV(:) = (U%XSEA(:)*PSEA_FWSV(:)+U%XWATER(:)*ZFWSV(:))/(U%XSEA(:)+U%XWATER(:))
    PSEA_SNET(:) = (U%XSEA(:)*PSEA_SNET(:)+U%XWATER(:)*ZSNET(:))/(U%XSEA(:)+U%XWATER(:))
    PSEA_HEAT(:) = (U%XSEA(:)*PSEA_HEAT(:)+U%XWATER(:)*ZHEAT(:))/(U%XSEA(:)+U%XWATER(:))
    PSEA_EVAP(:) = (U%XSEA(:)*PSEA_EVAP(:)+U%XWATER(:)*ZEVAP(:))/(U%XSEA(:)+U%XWATER(:))
    PSEA_RAIN(:) = (U%XSEA(:)*PSEA_RAIN(:)+U%XWATER(:)*ZRAIN(:))/(U%XSEA(:)+U%XWATER(:))
    PSEA_SNOW(:) = (U%XSEA(:)*PSEA_SNOW(:)+U%XWATER(:)*ZSNOW(:))/(U%XSEA(:)+U%XWATER(:))
    PSEA_FWSM(:) = (U%XSEA(:)*PSEA_FWSM(:)+U%XWATER(:)*ZFWSM(:))/(U%XSEA(:)+U%XWATER(:))
  ENDWHERE 
!
  W%XCPL_WATER_WIND(:) = 0.0
  W%XCPL_WATER_EVAP(:) = 0.0
  W%XCPL_WATER_HEAT(:) = 0.0
  W%XCPL_WATER_SNET(:) = 0.0
  W%XCPL_WATER_FWSU(:) = 0.0
  W%XCPL_WATER_FWSV(:) = 0.0
  W%XCPL_WATER_RAIN(:) = 0.0
  W%XCPL_WATER_SNOW(:) = 0.0
  W%XCPL_WATER_FWSM(:) = 0.0
!
  IF (OCPL_SEAICE) THEN
    CALL UNPACK_SAME_RANK(U%NR_WATER(:),W%XCPL_WATERICE_SNET(:),ZSNET_ICE(:),XUNDEF)
    CALL UNPACK_SAME_RANK(U%NR_WATER(:),W%XCPL_WATERICE_HEAT(:),ZHEAT_ICE(:),XUNDEF)
    CALL UNPACK_SAME_RANK(U%NR_WATER(:),W%XCPL_WATERICE_EVAP(:),ZEVAP_ICE(:),XUNDEF)
    WHERE(U%XWATER(:)>0.0)     
      PSEAICE_SNET(:) = (U%XSEA(:)*PSEAICE_SNET(:)+U%XWATER(:)*ZSNET_ICE(:))/(U%XSEA(:)+U%XWATER(:))
      PSEAICE_HEAT(:) = (U%XSEA(:)*PSEAICE_HEAT(:)+U%XWATER(:)*ZHEAT_ICE(:))/(U%XSEA(:)+U%XWATER(:))
      PSEAICE_EVAP(:) = (U%XSEA(:)*PSEAICE_EVAP(:)+U%XWATER(:)*ZEVAP_ICE(:))/(U%XSEA(:)+U%XWATER(:))
    ENDWHERE  
    W%XCPL_WATERICE_SNET(:) = 0.0
    W%XCPL_WATERICE_EVAP(:) = 0.0
    W%XCPL_WATERICE_HEAT(:) = 0.0
  ENDIF  
! 
ENDIF
!
!*       4.0   Net water flux
!              -----------------------
!
IF(U%NSIZE_SEA>0)THEN
!
  PSEA_WATF(:) = PSEA_RAIN(:) + PSEA_SNOW(:) - PSEA_EVAP(:)
!
ENDIF
!-------------------------------------------------------------------------------
IF (LHOOK) CALL DR_HOOK('GET_SFX_SEA',1,ZHOOK_HANDLE)
!-------------------------------------------------------------------------------
!
END SUBROUTINE GET_SFX_SEA
