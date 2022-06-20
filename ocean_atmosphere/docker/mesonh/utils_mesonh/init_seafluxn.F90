!SFX_LIC Copyright 1994-2014 CNRS, Meteo-France and Universite Paul Sabatier
!SFX_LIC This is part of the SURFEX software governed by the CeCILL-C licence
!SFX_LIC version 1. See LICENSE, CeCILL-C_V1-en.txt and CeCILL-C_V1-fr.txt  
!SFX_LIC for details. version 1.
!     #############################################################
      SUBROUTINE INIT_SEAFLUX_n (DTCO, OREAD_BUDGETC, UG, U, GCP, SM, &
                                 HPROGRAM,HINIT,KI,KSV,KSW,                 &
                                 HSV,PCO2,PRHOA,PZENITH,PAZIM,PSW_BANDS,    &
                                 PDIR_ALB,PSCA_ALB, PEMIS,PTSRAD,PTSURF,    &
                                 KYEAR, KMONTH,KDAY,PTIME,                  &
                                 HATMFILE,HATMFILETYPE,HTEST                )  
!     #############################################################
!
!!****  *INIT_SEAFLUX_n* - routine to initialize SEAFLUX
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
!!      Original    01/2003
!!      Modified    01/2006 : sea flux parameterization.
!!                  01/2008 : coupling with 1D ocean
!!      B. Decharme 08/2009 : specific treatment for sea/ice in the Earth System Model 
!!      B. Decharme 07/2011 : read pgd+prep 
!!      B. Decharme 04/2013 : new coupling variables
!!      S. Senesi   01/2014 : introduce sea-ice model 
!!      S. Belamari 03/2014 : add NZ0 (to choose PZ0SEA formulation)
!!      R. Séférian 01/2015 : introduce interactive ocean surface albedo
!!      M.N. Bouin  03/2014 : possibility of wave parameters
!!                          ! from external source
!!      J. Pianezze 11/2014 : add wave coupling flag for wave parameters
!!      J. Pianezze 04/2022 : add OA flux diagnostics
!-------------------------------------------------------------------------------
!
!*       0.    DECLARATIONS
!              ------------
!
USE MODD_SURFEX_n, ONLY : SEAFLUX_MODEL_t
!
USE MODD_DATA_COVER_n, ONLY : DATA_COVER_t
USE MODD_SURF_ATM_GRID_n, ONLY : SURF_ATM_GRID_t
USE MODD_SURF_ATM_n, ONLY : SURF_ATM_t
USE MODD_GRID_CONF_PROJ_n, ONLY : GRID_CONF_PROJ_t
!
USE MODD_SFX_OASIS,      ONLY : LCPL_WAVE, LCPL_SEA, LCPL_SEAICE
!
USE MODD_READ_NAMELIST,  ONLY : LNAM_READ
USE MODD_CSTS,           ONLY : XTTS
USE MODD_SNOW_PAR,       ONLY : XZ0HSN
USE MODD_SURF_PAR,       ONLY : XUNDEF, NUNDEF
USE MODD_CHS_AEROSOL,    ONLY: LVARSIGI, LVARSIGJ
USE MODD_DST_SURF,       ONLY: LVARSIG_DST, NDSTMDE, NDST_MDEBEG, LRGFIX_DST
USE MODD_SLT_SURF,       ONLY: LVARSIG_SLT, NSLTMDE, NSLT_MDEBEG, LRGFIX_SLT
!
USE MODI_INIT_IO_SURF_n
USE MODI_DEFAULT_CH_DEP
!
USE MODI_DEFAULT_SEAFLUX
USE MODI_DEFAULT_DIAG_SEAFLUX
USE MODI_READ_DEFAULT_SEAFLUX_n
USE MODI_READ_SEAFLUX_CONF_n
USE MODI_READ_SEAFLUX_n
!
USE MODI_READ_OCEAN_n
!
USE MODI_DEFAULT_SEAICE
USE MODI_READ_SEAICE_n
!
USE MODI_READ_PGD_SEAFLUX_n
USE MODI_DIAG_SEAFLUX_INIT_n
USE MODI_DIAG_SEAICE_INIT_n
USE MODI_END_IO_SURF_n
USE MODI_GET_LUOUT
USE MODI_READ_SURF
USE MODI_READ_SEAFLUX_DATE
USE MODI_READ_NAM_PREP_SEAFLUX_n
USE MODI_INIT_CHEMICAL_n
USE MODI_PREP_CTRL_SEAFLUX
USE MODI_UPDATE_RAD_SEA
USE MODI_READ_SBL_n
USE MODI_ABOR1_SFX
!
USE MODI_SET_SURFEX_FILEIN
!
USE YOMHOOK   ,ONLY : LHOOK,   DR_HOOK
USE PARKIND1  ,ONLY : JPRB
!
!
IMPLICIT NONE
!
!*       0.1   Declarations of arguments
!              -------------------------
!
!
TYPE(DATA_COVER_t), INTENT(INOUT) :: DTCO
TYPE(SURF_ATM_GRID_t), INTENT(INOUT) :: UG
TYPE(SURF_ATM_t), INTENT(INOUT) :: U
TYPE(GRID_CONF_PROJ_t),INTENT(INOUT) :: GCP
TYPE(SEAFLUX_MODEL_t), INTENT(INOUT) :: SM
!
LOGICAL, INTENT(IN) :: OREAD_BUDGETC
!
CHARACTER(LEN=6),                 INTENT(IN)  :: HPROGRAM  ! program calling surf. schemes
CHARACTER(LEN=3),                 INTENT(IN)  :: HINIT     ! choice of fields to initialize
INTEGER,                          INTENT(IN)  :: KI        ! number of points
INTEGER,                          INTENT(IN)  :: KSV       ! number of scalars
INTEGER,                          INTENT(IN)  :: KSW       ! number of short-wave spectral bands
CHARACTER(LEN=6), DIMENSION(KSV), INTENT(IN)  :: HSV       ! name of all scalar variables
REAL,             DIMENSION(KI),  INTENT(IN)  :: PCO2      ! CO2 concentration (kg/m3)
REAL,             DIMENSION(KI),  INTENT(IN)  :: PRHOA     ! air density
REAL,             DIMENSION(KI),  INTENT(IN)  :: PZENITH   ! solar zenithal angle
REAL,             DIMENSION(KI),  INTENT(IN)  :: PAZIM     ! solar azimuthal angle (rad from N, clock)
REAL,             DIMENSION(KSW), INTENT(IN)  :: PSW_BANDS ! middle wavelength of each band
REAL,             DIMENSION(KI,KSW),INTENT(OUT) :: PDIR_ALB  ! direct albedo for each band
REAL,             DIMENSION(KI,KSW),INTENT(OUT) :: PSCA_ALB  ! diffuse albedo for each band
REAL,             DIMENSION(KI),  INTENT(OUT) :: PEMIS     ! emissivity
REAL,             DIMENSION(KI),  INTENT(OUT) :: PTSRAD    ! radiative temperature
REAL,             DIMENSION(KI),  INTENT(OUT) :: PTSURF    ! surface effective temperature         (K)
INTEGER,                          INTENT(IN)  :: KYEAR     ! current year (UTC)
INTEGER,                          INTENT(IN)  :: KMONTH    ! current month (UTC)
INTEGER,                          INTENT(IN)  :: KDAY      ! current day (UTC)
REAL,                             INTENT(IN)  :: PTIME     ! current time since
                                                           !  midnight (UTC, s)
!
CHARACTER(LEN=28),                INTENT(IN)  :: HATMFILE    ! atmospheric file name
CHARACTER(LEN=6),                 INTENT(IN)  :: HATMFILETYPE! atmospheric file type
CHARACTER(LEN=2),                 INTENT(IN)  :: HTEST       ! must be equal to 'OK'
!
!
!*       0.2   Declarations of local variables
!              -------------------------------
!
INTEGER           :: ILU    ! sizes of SEAFLUX arrays
INTEGER           :: ILUOUT ! unit of output listing file
INTEGER           :: IRESP  ! return code
REAL(KIND=JPRB) :: ZHOOK_HANDLE
!
!-------------------------------------------------------------------------------
!
!         Initialisation for IO
!
IF (LHOOK) CALL DR_HOOK('INIT_SEAFLUX_N',0,ZHOOK_HANDLE)
!
 CALL GET_LUOUT(HPROGRAM,ILUOUT)
!
IF (HTEST/='OK') THEN
  CALL ABOR1_SFX('INIT_SEAFLUXN: FATAL ERROR DURING ARGUMENT TRANSFER')
END IF
!
!
!         Others litlle things
!
PDIR_ALB = XUNDEF
PSCA_ALB = XUNDEF
PEMIS    = XUNDEF
PTSRAD   = XUNDEF
PTSURF   = XUNDEF
!
SM%O%LMERCATOR = .FALSE.
SM%O%LCURRENT  = .FALSE.
!
IF (LNAM_READ) THEN
 !
 !*       0.     Defaults
 !               --------
 !
 !        0.1. Hard defaults
 !      
 
 CALL DEFAULT_SEAFLUX(SM%S%XTSTEP,SM%S%XOUT_TSTEP,SM%S%CSEA_ALB,SM%S%CSEA_FLUX,SM%S%LPWG,  &
                      SM%S%LPRECIP,SM%S%LPWEBB,SM%S%NZ0,SM%S%NGRVWAVES,SM%O%LPROGSST,      &
                      SM%O%NTIME_COUPLING,SM%O%XOCEAN_TSTEP,SM%S%XICHCE,SM%S%CINTERPOL_SST,&
                      SM%S%CINTERPOL_SSS,SM%S%LWAVEWIND                            )
 CALL DEFAULT_SEAICE(HPROGRAM,                                                  &
                     SM%S%CINTERPOL_SIC,SM%S%CINTERPOL_SIT, SM%S%XFREEZING_SST, &
                     SM%S%XSEAICE_TSTEP, SM%S%XSIC_EFOLDING_TIME,               &
                     SM%S%XSIT_EFOLDING_TIME, SM%S%XCD_ICE_CST, SM%S%XSI_FLX_DRV)     
 !                     
 CALL DEFAULT_CH_DEP(SM%CHS%CCH_DRY_DEP) 
 !            
 CALL DEFAULT_DIAG_SEAFLUX(SM%SD%O%N2M,SM%SD%O%LSURF_BUDGET,SM%SD%O%L2M_MIN_ZS,&
                           SM%SD%O%LRAD_BUDGET,SM%SD%O%LCOEF,SM%SD%O%LSURF_VARS,&
                           SM%SD%GO%LDIAG_OCEAN,SM%SD%DMI%LDIAG_MISC_SEAICE,&
                           SM%SD%O%LSURF_BUDGETC,SM%SD%O%LRESET_BUDGETC,SM%SD%O%XDIAG_TSTEP,&
                           SM%SD%O%LDIAG_OAFLUX )  

ENDIF
!
!
!        0.2. Defaults from file header
!    
 CALL READ_DEFAULT_SEAFLUX_n(SM%CHS, SM%SD%GO, SM%SD%O, SM%SD%DMI, SM%O, SM%S, HPROGRAM)
!
!*       1.1    Reading of configuration:
!               -------------------------
!
 CALL READ_SEAFLUX_CONF_n(SM%CHS, SM%SD%GO, SM%SD%O, SM%SD%DMI, SM%O, SM%S, HPROGRAM)
!
SM%S%LINTERPOL_SST=.FALSE.
SM%S%LINTERPOL_SSS=.FALSE.
SM%S%LINTERPOL_SIC=.FALSE.
SM%S%LINTERPOL_SIT=.FALSE.
!
IF(LCPL_SEA)THEN 
  IF(SM%SD%O%N2M<1)THEN
     CALL ABOR1_SFX('INIT_SEAFLUX_n: N2M must be set >0 in case of LCPL_SEA')
  ENDIF
! No STT / SSS interpolation in Earth System Model
  SM%S%CINTERPOL_SST='NONE  '
  SM%S%CINTERPOL_SSS='NONE  '
  SM%S%CINTERPOL_SIC='NONE  '
  SM%S%CINTERPOL_SIT='NONE  '
ELSE
   IF(TRIM(SM%S%CINTERPOL_SST)/='NONE')THEN
      SM%S%LINTERPOL_SST=.TRUE.
   ENDIF
   IF(TRIM(SM%S%CINTERPOL_SSS)/='NONE')THEN
      SM%S%LINTERPOL_SSS=.TRUE.
   ENDIF
   IF(TRIM(SM%S%CINTERPOL_SIC)/='NONE')THEN
      SM%S%LINTERPOL_SIC=.TRUE.
   ENDIF
   IF(TRIM(SM%S%CINTERPOL_SIT)/='NONE')THEN
      SM%S%LINTERPOL_SIT=.TRUE.
   ENDIF
ENDIF
!
!*       1.     Cover fields and grid:
!               ---------------------
!* date
!
SELECT CASE (HINIT)
!
  CASE ('PGD')
!
    SM%S%TTIME%TDATE%YEAR = NUNDEF
    SM%S%TTIME%TDATE%MONTH= NUNDEF
    SM%S%TTIME%TDATE%DAY  = NUNDEF
    SM%S%TTIME%TIME       = XUNDEF
!
  CASE ('PRE')
!
    CALL PREP_CTRL_SEAFLUX(SM%SD%O,SM%SD%GO%LDIAG_OCEAN,SM%SD%DMI%LDIAG_MISC_SEAICE,ILUOUT ) 
    IF (LNAM_READ) CALL READ_NAM_PREP_SEAFLUX_n(HPROGRAM)      
    CALL READ_SEAFLUX_DATE(SM%O%LMERCATOR,HPROGRAM,HINIT,ILUOUT,HATMFILE,HATMFILETYPE,&
                           KYEAR,KMONTH,KDAY,PTIME,SM%S%TTIME)
!
  CASE DEFAULT
!
CALL INIT_IO_SURF_n(DTCO, U, HPROGRAM,'FULL  ','SURF  ','READ ')
    CALL READ_SURF(HPROGRAM,'DTCUR',SM%S%TTIME,IRESP)
    CALL END_IO_SURF_n(HPROGRAM)
!
END SELECT
!
!-----------------------------------------------------------------------------------------------------
! READ PGD FILE
!-----------------------------------------------------------------------------------------------------
!
!         Initialisation for IO
!
 CALL SET_SURFEX_FILEIN(HPROGRAM,'PGD ') ! change input file name to pgd name
!
CALL INIT_IO_SURF_n(DTCO, U, HPROGRAM,'SEA   ','SEAFLX','READ ')
!
!         Reading of the fields
!
 CALL READ_PGD_SEAFLUX_n(DTCO, SM%DTS, SM%G, SM%S, U, UG, GCP, HPROGRAM)
!
 CALL END_IO_SURF_n(HPROGRAM)
!
 CALL SET_SURFEX_FILEIN(HPROGRAM,'PREP') ! restore input file name
!-------------------------------------------------------------------------------
!
!* if only physiographic fields are to be initialized, stop here.
!
IF (HINIT/='ALL' .AND. HINIT/='SOD') THEN
  IF (LHOOK) CALL DR_HOOK('INIT_SEAFLUX_N',1,ZHOOK_HANDLE)
  RETURN
END IF
!
!-------------------------------------------------------------------------------
!
!         Initialisation for IO
!
CALL INIT_IO_SURF_n(DTCO, U, HPROGRAM,'SEA   ','SEAFLX','READ ')
!
!*       2.     Prognostic fields:
!               ----------------
!
IF(SM%S%LINTERPOL_SST.OR.SM%S%LINTERPOL_SSS.OR.SM%S%LINTERPOL_SIC.OR.SM%S%LINTERPOL_SIT)THEN
!  Initialize current Month for SST interpolation
   SM%S%TZTIME%TDATE%YEAR  = SM%S%TTIME%TDATE%YEAR
   SM%S%TZTIME%TDATE%MONTH = SM%S%TTIME%TDATE%MONTH
   SM%S%TZTIME%TDATE%DAY   = SM%S%TTIME%TDATE%DAY
   SM%S%TZTIME%TIME        = SM%S%TTIME%TIME        
ENDIF
!
 CALL READ_SEAFLUX_n(DTCO, SM%G, SM%S, U, HPROGRAM,ILUOUT)
!
IF (HINIT/='ALL') THEN
  CALL END_IO_SURF_n(HPROGRAM)
  IF (LHOOK) CALL DR_HOOK('INIT_SEAFLUX_N',1,ZHOOK_HANDLE)
  RETURN
END IF
!-------------------------------------------------------------------------------
!
!*       2.1    Ocean fields:
!               -------------
!
 CALL READ_OCEAN_n(DTCO, SM%O, SM%OR, U, HPROGRAM)
!
!-------------------------------------------------------------------------------
!
ILU = SIZE(SM%S%XCOVER,1)
!
ALLOCATE(SM%S%XSST_INI    (ILU))
SM%S%XSST_INI(:) = SM%S%XSST(:)
!
ALLOCATE(SM%S%XZ0H(ILU))
WHERE (SM%S%XSST(:)>=XTTS)
  SM%S%XZ0H(:) = SM%S%XZ0(:)
ELSEWHERE
  SM%S%XZ0H(:) = XZ0HSN
ENDWHERE
!
!-------------------------------------------------------------------------------
!
!*       3.     Specific fields when using earth system model or sea-ice scheme
!               (Sea current and Sea-ice temperature)
!               -----------------------------------------------------------------
!
IF(LCPL_SEA.OR.SM%S%LHANDLE_SIC.OR.LCPL_WAVE)THEN       
! 
  ALLOCATE(SM%S%XUMER   (ILU))
  ALLOCATE(SM%S%XVMER   (ILU))
!
  SM%S%XUMER   (:)=0.
  SM%S%XVMER   (:)=0.
!
ELSE
! 
  ALLOCATE(SM%S%XUMER   (0))
  ALLOCATE(SM%S%XVMER   (0))
!
ENDIF
!
IF(LCPL_WAVE) THEN
  ALLOCATE(SM%S%XCHARN  (ILU))
  SM%S%XCHARN  (:)=0.011
ELSE
  ALLOCATE(SM%S%XCHARN   (0))
ENDIF
!
IF(LCPL_SEAICE.OR.SM%S%LHANDLE_SIC)THEN       
  ALLOCATE(SM%S%XTICE   (ILU))
  ALLOCATE(SM%S%XICE_ALB(ILU))
  SM%S%XTICE   (:)=XUNDEF
  SM%S%XICE_ALB(:)=XUNDEF
ELSE
  ALLOCATE(SM%S%XTICE   (0))
  ALLOCATE(SM%S%XICE_ALB(0))
ENDIF
!
!-------------------------------------------------------------------------------
!
!*       4.     Seaice prognostic variables and forcings :
!
CALL READ_SEAICE_n(SM%G, SM%S, HPROGRAM,ILU,ILUOUT)
!
!-------------------------------------------------------------------------------
!
!*       5.     Albedo, emissivity and temperature fields on the mix (open sea + sea ice)
!               -----------------------------------------------------------------
!
ALLOCATE(SM%S%XEMIS    (ILU))
SM%S%XEMIS    = 0.0
!
CALL UPDATE_RAD_SEA(SM%S,PZENITH,XTTS,PDIR_ALB,PSCA_ALB,PEMIS,PTSRAD )  
!
IF (SM%S%LHANDLE_SIC) THEN
   PTSURF(:) = SM%S%XSST(:) * ( 1 - SM%S%XSIC(:)) + SM%S%XTICE(:) * SM%S%XSIC(:)
ELSE
   PTSURF(:) = SM%S%XSST(:)
ENDIF
!
!-------------------------------------------------------------------------------
!
!*       6.     SBL air fields:
!               --------------
!
 CALL READ_SBL_n(DTCO, U, SM%SB, SM%S%LSBL, HPROGRAM, "SEA   ")
!
!-------------------------------------------------------------------------------
!
!*       7.     Chemistry /dust
!               ---------
!
 CALL INIT_CHEMICAL_n(ILUOUT, KSV, HSV, SM%CHS%SVS,           &
                     SM%CHS%CCH_NAMES, SM%CHS%CAER_NAMES,     &
                     HDSTNAMES=SM%CHS%CDSTNAMES, HSLTNAMES=SM%CHS%CSLTNAMES,  &
                     HSNWNAMES=SM%CHS%CSNWNAMES     )
!
!* deposition scheme
!
IF (SM%CHS%SVS%NBEQ>0 .AND. SM%CHS%CCH_DRY_DEP=='WES89') THEN
  ALLOCATE(SM%CHS%XDEP(ILU,SM%CHS%SVS%NBEQ))
ELSE
  ALLOCATE(SM%CHS%XDEP(0,0))
END IF
!
!-------------------------------------------------------------------------------
!
!*       8.     diagnostics initialization
!               --------------------------
!
IF(.NOT.(SM%S%LHANDLE_SIC.OR.LCPL_SEAICE))THEN
  SM%SD%DMI%LDIAG_MISC_SEAICE=.FALSE.
ENDIF
!
 CALL DIAG_SEAFLUX_INIT_n(SM%SD%GO, SM%SD%O, SM%SD%D, SM%SD%DC, OREAD_BUDGETC, SM%S, &
                          HPROGRAM,ILU,KSW)
IF (SM%S%LHANDLE_SIC.OR.LCPL_SEAICE) &
        CALL DIAG_SEAICE_INIT_n(SM%SD%O, SM%SD%DI, SM%SD%DIC, SM%SD%DMI, &
                               OREAD_BUDGETC, SM%S, HPROGRAM,ILU,KSW)
                 
!
!-------------------------------------------------------------------------------
!
!         End of IO
!
 CALL END_IO_SURF_n(HPROGRAM)
IF (LHOOK) CALL DR_HOOK('INIT_SEAFLUX_N',1,ZHOOK_HANDLE)
!
!
END SUBROUTINE INIT_SEAFLUX_n
