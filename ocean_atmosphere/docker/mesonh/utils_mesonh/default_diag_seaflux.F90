!SFX_LIC Copyright 1994-2014 CNRS, Meteo-France and Universite Paul Sabatier
!SFX_LIC This is part of the SURFEX software governed by the CeCILL-C licence
!SFX_LIC version 1. See LICENSE, CeCILL-C_V1-en.txt and CeCILL-C_V1-fr.txt  
!SFX_LIC for details. version 1.
!     #########
      SUBROUTINE DEFAULT_DIAG_SEAFLUX(K2M,OSURF_BUDGET,O2M_MIN_ZS,ORAD_BUDGET,OCOEF,OSURF_VARS,&
                                  ODIAG_OCEAN,ODIAG_MISC_SEAICE,OSURF_BUDGETC,ORESET_BUDGETC,PDIAG_TSTEP,&
                                  ODIAG_OAFLUX)  
!     ########################################################################
!
!!****  *DEFAULT_DIAG_SEAFLUX* - routine to set default values for the choice of diagnostics
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
!!      V. Masson   *Meteo France*
!!
!!    MODIFICATIONS
!!    -------------
!!      Original    01/2004 
!!      Modified    09/2013 : S. Senesi : introduce ODIAG_SEAICE
!!      Modified    04/2022 : J. Pianezze : add diagnostics
!-------------------------------------------------------------------------------
!
!*       0.    DECLARATIONS
!              ------------
!
USE MODD_SURF_PAR,   ONLY : XUNDEF
!
!
USE YOMHOOK   ,ONLY : LHOOK,   DR_HOOK
USE PARKIND1  ,ONLY : JPRB
!
IMPLICIT NONE
!
!*       0.1   Declarations of arguments
!              -------------------------
!
INTEGER,  INTENT(OUT) :: K2M           ! flag for operational 2m quantities
LOGICAL,  INTENT(OUT) :: OSURF_BUDGET  ! flag for surface budget
LOGICAL,  INTENT(OUT) :: O2M_MIN_ZS
LOGICAL,  INTENT(OUT) :: ORAD_BUDGET   ! flag for radiative budget
LOGICAL,  INTENT(OUT) :: OCOEF
LOGICAL,  INTENT(OUT) :: OSURF_VARS
LOGICAL,  INTENT(OUT) :: ODIAG_OCEAN
LOGICAL,  INTENT(OUT) :: ODIAG_MISC_SEAICE
LOGICAL,  INTENT(OUT) :: OSURF_BUDGETC ! flag for cumulated surface budget
LOGICAL,  INTENT(OUT) :: ORESET_BUDGETC! flag for cumulated surface budget
REAL,     INTENT(OUT) :: PDIAG_TSTEP   ! time-step for writing
LOGICAL,  INTENT(OUT) :: ODIAG_OAFLUX  !
!
REAL(KIND=JPRB) :: ZHOOK_HANDLE
!
!*       0.2   Declarations of local variables
!              -------------------------------
!
!-------------------------------------------------------------------------------
!
IF (LHOOK) CALL DR_HOOK('DEFAULT_DIAG_SEAFLUX',0,ZHOOK_HANDLE)
!
K2M = 0
OSURF_BUDGET = .FALSE.
!
O2M_MIN_ZS   = .FALSE.
!
ORAD_BUDGET  = .FALSE.
!
OCOEF        = .FALSE.
OSURF_VARS   = .FALSE.
!
ODIAG_OCEAN  = .FALSE.
ODIAG_MISC_SEAICE = .FALSE.
!
OSURF_BUDGETC = .FALSE.
ORESET_BUDGETC= .FALSE.
!
PDIAG_TSTEP  = XUNDEF
!
ODIAG_OAFLUX  = .FALSE.
!
IF (LHOOK) CALL DR_HOOK('DEFAULT_DIAG_SEAFLUX',1,ZHOOK_HANDLE)
!
!-------------------------------------------------------------------------------
!
END SUBROUTINE DEFAULT_DIAG_SEAFLUX
