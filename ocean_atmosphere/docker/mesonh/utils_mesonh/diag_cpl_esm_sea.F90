!SFX_LIC Copyright 1994-2014 CNRS, Meteo-France and Universite Paul Sabatier
!SFX_LIC This is part of the SURFEX software governed by the CeCILL-C licence
!SFX_LIC version 1. See LICENSE, CeCILL-C_V1-en.txt and CeCILL-C_V1-fr.txt  
!SFX_LIC for details. version 1.
!     #########
       SUBROUTINE DIAG_CPL_ESM_SEA (S, D, DI, PTSTEP, PSFTQ, PRAIN, PSNOW, &
                                    PLW, PPS, PSFTH_ICE, PSFTQ_ICE, PDIR_SW, PSCA_SW, OSIC)  
!     ###################################################################
!
!!****  *DIAG_CPL_ESM_SEA * - Computes diagnostics over sea for 
!!                Earth system model coupling or embedded seaice scheme
!!
!!    PURPOSE
!!    -------
!
!!**  METHOD
!!    ------
!!
!!    REFERENCE
!!    ---------
!!      
!!
!!    AUTHOR
!!    ------
!!     B. Decharme 
!!
!!    MODIFICATIONS
!!    -------------
!!      Original    08/2009
!!      S.Senesi    01/2014  Adapt to embedded seaice scheme (SWU and LWU 
!!                           for seaice are provided as inputs)
!!      A.Voldoire  04/2015  Add LCPL_SEAICE test
!!      Modified    11/2014 : J. Pianezze : Add surface pressure coupling parameter
!!------------------------------------------------------------------
!
USE MODD_DIAG_n, ONLY : DIAG_t
USE MODD_SEAFLUX_n, ONLY : SEAFLUX_t
!
USE MODD_CSTS,      ONLY : XSTEFAN, XLSTT
USE MODD_WATER_PAR, ONLY : XEMISWATICE
!
USE MODD_SFX_OASIS, ONLY : LCPL_SEAICE
!
USE YOMHOOK   ,ONLY : LHOOK,   DR_HOOK
USE PARKIND1  ,ONLY : JPRB
!
IMPLICIT NONE
!
!*      0.1    declarations of arguments
!
!
TYPE(SEAFLUX_t), INTENT(INOUT) :: S
TYPE(DIAG_t), INTENT(INOUT) :: D
TYPE(DIAG_t), INTENT(INOUT) :: DI
!
REAL,               INTENT(IN) :: PTSTEP    ! atmospheric time-step
REAL, DIMENSION(:), INTENT(IN) :: PSFTQ     ! water flux
REAL, DIMENSION(:), INTENT(IN) :: PRAIN     ! Rainfall
REAL, DIMENSION(:), INTENT(IN) :: PSNOW     ! Snowfall
REAL, DIMENSION(:), INTENT(IN) :: PLW       ! longwave radiation (on horizontal surf.)
REAL, DIMENSION(:), INTENT(IN) :: PPS       ! Surface pressure
REAL, DIMENSION(:), INTENT(IN) :: PSFTH_ICE ! heat flux  (W/m2)
REAL, DIMENSION(:), INTENT(IN) :: PSFTQ_ICE ! water flux (kg/m2/s)
REAL, DIMENSION(:,:),INTENT(IN):: PDIR_SW   ! direct  solar radiation (on horizontal surf.)
REAL, DIMENSION(:,:),INTENT(IN):: PSCA_SW   ! diffuse solar radiation (on horizontal surf.)
LOGICAL,            INTENT(IN) :: OSIC
!
!*      0.2    declarations of local variables
!
REAL, DIMENSION(SIZE(S%XICE_ALB)) :: ZSWU, ZTICE4
!
INTEGER                      :: ISWB ! number of SW bands
INTEGER                      :: JSWB ! loop counter on number of SW bands
INTEGER                      :: INI  ! number of points
INTEGER                      :: JI   ! loop counter on number of points
!
REAL(KIND=JPRB) :: ZHOOK_HANDLE
!
IF (LHOOK) CALL DR_HOOK('DIAG_CPL_ESM_SEA',0,ZHOOK_HANDLE)
!
!-------------------------------------------------------------------------------------
! Total or free-ice sea flux
!-------------------------------------------------------------------------------------
!
!* 10m wind speed (m)
!
S%XCPL_SEA_WIND(:) = S%XCPL_SEA_WIND(:) + PTSTEP * SQRT(D%XZON10M(:)**2+D%XMER10M(:)**2)
! 
!* wind stress (Pa.s)
!
S%XCPL_SEA_FWSU(:) = S%XCPL_SEA_FWSU(:) + PTSTEP * D%XFMU(:)
S%XCPL_SEA_FWSV(:) = S%XCPL_SEA_FWSV(:) + PTSTEP * D%XFMV(:)
S%XCPL_SEA_FWSM(:) = S%XCPL_SEA_FWSM(:) + PTSTEP * SQRT(D%XFMU(:)**2+D%XFMV(:)**2)
!
!* Solar net heat flux (J/m2)
!
S%XCPL_SEA_SNET(:) = S%XCPL_SEA_SNET(:) + PTSTEP * (D%XSWD(:) - D%XSWU(:))
!
!* Non solar heat flux (J/m2)
!
S%XCPL_SEA_HEAT(:) = S%XCPL_SEA_HEAT(:) + PTSTEP * (D%XGFLUX(:) + D%XSWU(:) - D%XSWD(:)) 
!
!* Evaporation (kg/m2)
!
S%XCPL_SEA_EVAP(:) = S%XCPL_SEA_EVAP(:) + PTSTEP * PSFTQ(:)
!
!* Precip (kg/m2)
! 
S%XCPL_SEA_RAIN(:) = S%XCPL_SEA_RAIN(:) + PTSTEP * PRAIN(:)
S%XCPL_SEA_SNOW(:) = S%XCPL_SEA_SNOW(:) + PTSTEP * PSNOW(:)
!
!* Evaporation - Precip (kg/m2)
! 
S%XCPL_SEA_EVPR(:) = S%XCPL_SEA_EVPR(:) + S%XCPL_SEA_EVAP(:) - S%XCPL_SEA_RAIN(:) - S%XCPL_SEA_SNOW(:)
!
!* Cumulated surface pressure (Pa.s)
! 
S%XCPL_SEA_PRES(:) = S%XCPL_SEA_PRES(:) + PTSTEP * PPS(:)
!
!* Cumulated long-wave heat flux (J/m2)
! 
S%XCPL_SEA_LWFL(:) = S%XCPL_SEA_LWFL(:) + PTSTEP * (D%XLWD(:) - D%XLWU(:))
!
!* Cumulated latent heat flux (J/m2)
! 
S%XCPL_SEA_LHFL(:) = S%XCPL_SEA_LHFL(:) + PTSTEP * D%XLE(:)
!
!* Cumulated sensible heat flux (J/m2)
! 
S%XCPL_SEA_SHFL(:) = S%XCPL_SEA_SHFL(:) + PTSTEP * D%XH(:)
!
!-------------------------------------------------------------------------------------
! Ice flux
!-------------------------------------------------------------------------------------
IF (LCPL_SEAICE.OR.OSIC) THEN
!
  INI  = SIZE(PDIR_SW,1)
  ISWB = SIZE(PDIR_SW,2)
!
!* Solar net heat flux (J/m2)
!
  IF (OSIC) THEN
    ZSWU(:)=DI%XSWU(:)
  ELSE
    ZSWU(:)=0.0
    DO JSWB=1,ISWB
      DO JI=1,INI
         ZSWU(JI) = ZSWU(JI) + (PDIR_SW(JI,JSWB)+PSCA_SW(JI,JSWB)) * S%XICE_ALB(JI)
      ENDDO
    ENDDO
  ENDIF
!
  S%XCPL_SEAICE_SNET(:) = S%XCPL_SEAICE_SNET(:) + PTSTEP * (D%XSWD(:) - ZSWU(:))
!
!* Non solar heat flux (J/m2)
!
  IF (OSIC) THEN
    S%XCPL_SEAICE_HEAT(:) = S%XCPL_SEAICE_HEAT(:) + PTSTEP * &
              ( PLW(:) - DI%XLWU(:) - PSFTH_ICE(:) - XLSTT*PSFTQ_ICE(:) )
  ELSE
    ZTICE4(:)=S%XTICE(:)**4
    S%XCPL_SEAICE_HEAT(:) = S%XCPL_SEAICE_HEAT(:) + PTSTEP * ( XEMISWATICE*(PLW(:)-XSTEFAN*ZTICE4(:)) &
                                                         - PSFTH_ICE(:) - XLSTT*PSFTQ_ICE(:)      ) 
  ENDIF 
!
!* Sublimation (kg/m2)
!
  S%XCPL_SEAICE_EVAP(:) = S%XCPL_SEAICE_EVAP(:) + PTSTEP * PSFTQ_ICE(:)
!
ENDIF
!
IF (LHOOK) CALL DR_HOOK('DIAG_CPL_ESM_SEA',1,ZHOOK_HANDLE)
!
!-------------------------------------------------------------------------------------
!
END SUBROUTINE DIAG_CPL_ESM_SEA
