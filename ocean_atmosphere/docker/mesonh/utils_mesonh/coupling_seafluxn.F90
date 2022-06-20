!SFX_LIC Copyright 2004-2019 CNRS, Meteo-France and Universite Paul Sabatier
!SFX_LIC This is part of the SURFEX software governed by the CeCILL-C licence
!SFX_LIC version 1. See LICENSE, CeCILL-C_V1-en.txt and CeCILL-C_V1-fr.txt
!SFX_LIC for details. version 1.
!     ###############################################################################
SUBROUTINE COUPLING_SEAFLUX_n (CHS, DTS, DGS, O, OR, G, S, DST, SLT, &
                               HPROGRAM, HCOUPLING, PTIMEC, PTSTEP, KYEAR, KMONTH, KDAY, PTIME, &
                               KI, KSV, KSW, PTSUN, PZENITH, PZENITH2, PAZIM, PZREF, PUREF,     &
                               PU, PV, PQA, PTA, PRHOA, PSV, PCO2, HSV, PRAIN, PSNOW, PLW,      &
                               PDIR_SW, PSCA_SW, PSW_BANDS, PPS, PPA, PSFTQ, PSFTH, PSFTS,      &
                               PSFCO2, PSFU, PSFV, PTRAD, PDIR_ALB, PSCA_ALB, PEMIS, PTSURF,    &
                               PZ0, PZ0H, PQSURF, PPEW_A_COEF, PPEW_B_COEF, PPET_A_COEF,        &
                               PPEQ_A_COEF, PPET_B_COEF, PPEQ_B_COEF, PZWS,  HTEST              )  
!     ###############################################################################
!
!!****  *COUPLING_SEAFLUX_n * - Driver of the WATER_FLUX scheme for sea   
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
!!     V. Masson 
!!
!!    MODIFICATIONS
!!    -------------
!!      Original    01/2004
!!      Modified    01/2006 : sea flux parameterization.
!!      Modified    09/2006 : P. Tulet Introduce Sea salt aerosol Emission/Deposition
!!      Modified    03/2009 : B. Decharme SST could change during a run => ALB and EMIS 
!!      Modified    05/2009 : V. Masson : implicitation of momentum fluxes
!!      Modified    09/2009 : B. Decharme Radiative properties at time t+1
!!      Modified    01/2010 : B. Decharme Add XTTS
!!      Modified    09/2012 : B. Decharme New wind implicitation
!!      Modified    10/2012 : P. Le Moigne CMO1D update
!!      Modified    04/2013 : P. Le Moigne Wind implicitation and SST update displaced
!!      Modified    04/2013 : B. Decharme new coupling variables
!!      Modified    01/2014 : S. Senesi : handle sea-ice cover, sea-ice model interface, 
!!                               and apply to Gelato
!!      Modified    01/2014 : S. Belamari Remove MODE_THERMOS and XLVTT
!!      Modified    05/2014 : S. Belamari New ECUME : Include salinity & atm. pressure impact 
!!      Modified    01/2015 : R. Séférian interactive ocaen surface albedo
!!      Modified    03/2014 : M.N. Bouin possibility of wave parameters from external source
!!      Modified    11/2014 : J. Pianezze : add currents for wave coupling
!!      Modified    02/2019 : S. Bielli Sea salt : significant sea wave height influences salt emission; 5 salt modes
!!      Modified    03/2019 : P. Wautelet: correct ZWS when variable not present in file
!!      Modified    03/2019 : P. Wautelet: missing use MODI_GET_LUOUT
!!---------------------------------------------------------------------
!
!
USE MODD_CH_SEAFLUX_n, ONLY : CH_SEAFLUX_t
USE MODD_DATA_SEAFLUX_n, ONLY : DATA_SEAFLUX_t
USE MODD_SURFEX_n, ONLY : SEAFLUX_DIAG_t
USE MODD_OCEAN_n, ONLY : OCEAN_t
USE MODD_OCEAN_REL_n, ONLY : OCEAN_REL_t
USE MODD_SFX_GRID_n, ONLY : GRID_t
USE MODD_SEAFLUX_n, ONLY : SEAFLUX_t
!
USE MODD_DST_n, ONLY : DST_t
USE MODD_SLT_n, ONLY : SLT_t
!
USE MODD_REPROD_OPER, ONLY : CIMPLICIT_WIND
!
USE MODD_CSTS,       ONLY : XRD, XCPD, XP00, XTT, XTTS, XTTSI, XDAY
USE MODD_SURF_PAR,   ONLY : XUNDEF
USE MODD_SFX_OASIS,  ONLY : LCPL_WAVE, LCPL_SEA, LCPL_SEAICE
USE MODD_WATER_PAR,  ONLY : XEMISWAT, XEMISWATICE
!
USE MODD_WATER_PAR, ONLY : XALBSEAICE
!
#ifdef SFX_MNH
USE MODD_FIELD_n, only: XZWS_DEFAULT
#endif
!
!
USE MODI_WATER_FLUX
USE MODI_MR98
USE MODI_ECUME_SEAFLUX
USE MODI_COARE30_SEAFLUX
USE MODI_ADD_FORECAST_TO_DATE_SURF
USE MODI_MOD1D_n
USE MODI_DIAG_INLINE_SEAFLUX_n
USE MODI_CH_AER_DEP
USE MODI_CH_DEP_WATER
USE MODI_DSLT_DEP
USE MODI_SST_UPDATE
USE MODI_INTERPOL_SST_MTH
USE MODI_UPDATE_RAD_SEA
!
USE MODE_DSLT_SURF
USE MODD_DST_SURF
USE MODD_SLT_SURF
! 
USE MODD_OCEAN_GRID,   ONLY : NOCKMIN
!
USE YOMHOOK   ,ONLY : LHOOK,   DR_HOOK
USE PARKIND1  ,ONLY : JPRB
!
USE MODI_ABOR1_SFX
!
USE MODI_COUPLING_ICEFLUX_n
USE MODI_SEAICE_GELATO1D_n
!
USE MODI_COUPLING_SLT_n
USE MODI_GET_LUOUT
!
IMPLICIT NONE
!
!*      0.1    declarations of arguments
!
!
TYPE(CH_SEAFLUX_t), INTENT(INOUT) :: CHS
TYPE(DATA_SEAFLUX_t), INTENT(INOUT) :: DTS
TYPE(SEAFLUX_DIAG_t), INTENT(INOUT) :: DGS
TYPE(OCEAN_t), INTENT(INOUT) :: O
TYPE(OCEAN_REL_t), INTENT(INOUT) :: OR
TYPE(GRID_t), INTENT(INOUT) :: G
TYPE(SEAFLUX_t), INTENT(INOUT) :: S 
TYPE(DST_t), INTENT(INOUT) :: DST
TYPE(SLT_t), INTENT(INOUT) :: SLT
!
CHARACTER(LEN=6),    INTENT(IN)  :: HPROGRAM  ! program calling surf. schemes
CHARACTER(LEN=1),    INTENT(IN)  :: HCOUPLING ! type of coupling
                                      ! 'E' : explicit
                                      ! 'I' : implicit
REAL,                INTENT(IN)  :: PTIMEC    ! current duration since start of the run (s)
INTEGER,             INTENT(IN)  :: KYEAR     ! current year (UTC)
INTEGER,             INTENT(IN)  :: KMONTH    ! current month (UTC)
INTEGER,             INTENT(IN)  :: KDAY      ! current day (UTC)
REAL,                INTENT(IN)  :: PTIME     ! current time since midnight (UTC, s)
INTEGER,             INTENT(IN)  :: KI        ! number of points
INTEGER,             INTENT(IN)  :: KSV       ! number of scalars
INTEGER,             INTENT(IN)  :: KSW       ! number of short-wave spectral bands
REAL, DIMENSION(KI), INTENT(IN)  :: PTSUN     ! solar time                    (s from midnight)
REAL,                INTENT(IN)  :: PTSTEP    ! atmospheric time-step                 (s)
REAL, DIMENSION(KI), INTENT(IN)  :: PZREF     ! height of T,q forcing                 (m)
REAL, DIMENSION(KI), INTENT(IN)  :: PUREF     ! height of wind forcing                (m)
!
REAL, DIMENSION(KI), INTENT(IN)  :: PTA       ! air temperature forcing               (K)
REAL, DIMENSION(KI), INTENT(IN)  :: PQA       ! air humidity forcing                  (kg/m3)
REAL, DIMENSION(KI), INTENT(IN)  :: PRHOA     ! air density                           (kg/m3)
REAL, DIMENSION(KI,KSV),INTENT(IN) :: PSV     ! scalar variables
!                                             ! chemistry:   first char. in HSV: '#'  (molecule/m3)
!                                             !
CHARACTER(LEN=6), DIMENSION(KSV),INTENT(IN):: HSV  ! name of all scalar variables
REAL, DIMENSION(KI), INTENT(IN)  :: PU        ! zonal wind                            (m/s)
REAL, DIMENSION(KI), INTENT(IN)  :: PV        ! meridian wind                         (m/s)
REAL, DIMENSION(KI,KSW),INTENT(IN) :: PDIR_SW ! direct  solar radiation (on horizontal surf.)
!                                             !                                       (W/m2)
REAL, DIMENSION(KI,KSW),INTENT(IN) :: PSCA_SW ! diffuse solar radiation (on horizontal surf.)
!                                             !                                       (W/m2)
REAL, DIMENSION(KSW),INTENT(IN)  :: PSW_BANDS ! mean wavelength of each shortwave band (m)
REAL, DIMENSION(KI), INTENT(IN)  :: PZENITH   ! zenithal angle at t  (radian from the vertical)
REAL, DIMENSION(KI), INTENT(IN)  :: PZENITH2  ! zenithal angle at t+1(radian from the vertical)
REAL, DIMENSION(KI), INTENT(IN)  :: PAZIM     ! azimuthal angle      (radian from North, clockwise)
REAL, DIMENSION(KI), INTENT(IN)  :: PLW       ! longwave radiation (on horizontal surf.)
!                                             !                                       (W/m2)
REAL, DIMENSION(KI), INTENT(IN)  :: PPS       ! pressure at atmospheric model surface (Pa)
REAL, DIMENSION(KI), INTENT(IN)  :: PPA       ! pressure at forcing level             (Pa)
REAL, DIMENSION(KI), INTENT(IN)  :: PZWS      ! significant sea wave                  (m)
REAL, DIMENSION(KI), INTENT(IN)  :: PCO2      ! CO2 concentration in the air          (kg/m3)
REAL, DIMENSION(KI), INTENT(IN)  :: PSNOW     ! snow precipitation                    (kg/m2/s)
REAL, DIMENSION(KI), INTENT(IN)  :: PRAIN     ! liquid precipitation                  (kg/m2/s)
!
REAL, DIMENSION(KI), INTENT(OUT) :: PSFTH     ! flux of heat                          (W/m2)
REAL, DIMENSION(KI), INTENT(OUT) :: PSFTQ     ! flux of water vapor                   (kg/m2/s)
REAL, DIMENSION(KI), INTENT(OUT) :: PSFU      ! zonal momentum flux                   (Pa)
REAL, DIMENSION(KI), INTENT(OUT) :: PSFV      ! meridian momentum flux                (Pa)
REAL, DIMENSION(KI), INTENT(OUT) :: PSFCO2    ! flux of CO2                           (m/s*kg_CO2/kg_air)
REAL, DIMENSION(KI,KSV),INTENT(OUT):: PSFTS   ! flux of scalar var.                   (kg/m2/s)
!
REAL, DIMENSION(KI), INTENT(OUT) :: PTRAD     ! radiative temperature                 (K)
REAL, DIMENSION(KI,KSW),INTENT(OUT):: PDIR_ALB! direct albedo for each spectral band  (-)
REAL, DIMENSION(KI,KSW),INTENT(OUT):: PSCA_ALB! diffuse albedo for each spectral band (-)
REAL, DIMENSION(KI), INTENT(OUT) :: PEMIS     ! emissivity                            (-)
!
REAL, DIMENSION(KI), INTENT(OUT) :: PTSURF    ! surface effective temperature         (K)
REAL, DIMENSION(KI), INTENT(OUT) :: PZ0       ! roughness length for momentum         (m)
REAL, DIMENSION(KI), INTENT(OUT) :: PZ0H      ! roughness length for heat             (m)
REAL, DIMENSION(KI), INTENT(OUT) :: PQSURF    ! specific humidity at surface          (kg/kg)
!
REAL, DIMENSION(KI), INTENT(IN) :: PPEW_A_COEF! implicit coefficients   (m2s/kg)
REAL, DIMENSION(KI), INTENT(IN) :: PPEW_B_COEF! needed if HCOUPLING='I' (m/s)
REAL, DIMENSION(KI), INTENT(IN) :: PPET_A_COEF
REAL, DIMENSION(KI), INTENT(IN) :: PPEQ_A_COEF
REAL, DIMENSION(KI), INTENT(IN) :: PPET_B_COEF
REAL, DIMENSION(KI), INTENT(IN) :: PPEQ_B_COEF
CHARACTER(LEN=2),    INTENT(IN) :: HTEST ! must be equal to 'OK'
!
!*      0.2    declarations of local variables
!     
REAL, DIMENSION(KI,KSW) :: ZDIR_ALB   ! Direct albedo at time t
REAL, DIMENSION(KI,KSW) :: ZSCA_ALB   ! Diffuse albedo at time t
!
REAL, DIMENSION(KI) :: ZEXNA      ! Exner function at forcing level
REAL, DIMENSION(KI) :: ZEXNS      ! Exner function at surface level
REAL, DIMENSION(KI) :: ZU         ! zonal wind
REAL, DIMENSION(KI) :: ZV         ! meridian wind
REAL, DIMENSION(KI) :: ZWIND      ! Wind
REAL, DIMENSION(KI) :: ZCD        ! Drag coefficient on open sea
REAL, DIMENSION(KI) :: ZCD_ICE    ! "     "          on seaice
REAL, DIMENSION(KI) :: ZCDN       ! Neutral Drag coefficient on open sea
REAL, DIMENSION(KI) :: ZCDN_ICE   ! "     "          on seaice
REAL, DIMENSION(KI) :: ZCH        ! Heat transfer coefficient on open sea
REAL, DIMENSION(KI) :: ZCH_ICE    ! "     "          on seaice
REAL, DIMENSION(KI) :: ZCE        ! Vaporization heat transfer coefficient on open sea
REAL, DIMENSION(KI) :: ZCE_ICE    ! "     "          on seaice
REAL, DIMENSION(KI) :: ZRI        ! Richardson number on open sea
REAL, DIMENSION(KI) :: ZRI_ICE    ! "     "          on seaice
REAL, DIMENSION(KI) :: ZRESA_SEA  ! aerodynamical resistance on open sea
REAL, DIMENSION(KI) :: ZRESA_SEA_ICE  ! "     "          on seaice
REAL, DIMENSION(KI) :: ZUSTAR     ! friction velocity (m/s) on open sea
REAL, DIMENSION(KI) :: ZUSTAR_ICE ! "     "          on seaice
REAL, DIMENSION(KI) :: ZZ0        ! roughness length over open sea
REAL, DIMENSION(KI) :: ZZ0_ICE    ! roughness length over seaice
REAL, DIMENSION(KI) :: ZZ0H       ! heat roughness length over open sea
REAL, DIMENSION(KI) :: ZZ0H_ICE   ! heat roughness length over seaice
REAL, DIMENSION(KI) :: ZZ0W       ! Work array for Z0 and Z0H computation
REAL, DIMENSION(KI) :: ZQSAT      ! humidity at saturation on open sea
REAL, DIMENSION(KI) :: ZQSAT_ICE  ! "     "          on seaice
!
REAL, DIMENSION(KI) :: ZSFTH      ! Heat flux for open sea (and for sea-ice points if merged)
REAL, DIMENSION(KI) :: ZSFTQ      ! Water vapor flux on open sea (and for sea-ice points if merged)
REAL, DIMENSION(KI) :: ZSFU       ! zonal      momentum flux on open sea (and for sea-ice points if merged)(Pa)
REAL, DIMENSION(KI) :: ZSFV       ! meridional momentum flux on open sea (and for sea-ice points if merged)(Pa)
!
REAL, DIMENSION(KI) :: ZSFTH_ICE  ! Heat flux on sea ice 
REAL, DIMENSION(KI) :: ZSFTQ_ICE  ! Sea-ice sublimation flux
REAL, DIMENSION(KI) :: ZSFU_ICE   ! zonal      momentum flux on seaice (Pa)
REAL, DIMENSION(KI) :: ZSFV_ICE   ! meridional momentum flux on seaice (Pa)

REAL, DIMENSION(KI) :: ZHU        ! Near surface relative humidity
REAL, DIMENSION(KI) :: ZQA        ! specific humidity (kg/kg)
REAL, DIMENSION(KI) :: ZEMIS      ! Emissivity at time t
REAL, DIMENSION(KI) :: ZTRAD      ! Radiative temperature at time t
REAL, DIMENSION(KI) :: ZHS        ! significant wave height
REAL, DIMENSION(KI) :: ZTP        ! peak period
!
REAL, DIMENSION(KI) :: ZSST       ! XSST corrected for anomalously low values (which actually are sea-ice temp)
REAL, DIMENSION(KI) :: ZMASK      ! A mask for diagnosing where seaice exists (or, for coupling_iceflux, may appear)
!
REAL                             :: ZCONVERTFACM0_SLT, ZCONVERTFACM0_DST
REAL                             :: ZCONVERTFACM3_SLT, ZCONVERTFACM3_DST
REAL                             :: ZCONVERTFACM6_SLT, ZCONVERTFACM6_DST
!
INTEGER                          :: ISIZE_WATER  ! number of points with some sea water 
INTEGER                          :: ISIZE_ICE    ! number of points with some sea ice
!
INTEGER                          :: ISWB       ! number of shortwave spectral bands
INTEGER                          :: JSWB       ! loop counter on shortwave spectral bands
!
INTEGER :: IBEG, IEND
INTEGER                          :: ISLT, IDST, JSV, IMOMENT   ! number of sea salt, dust variables
!
INTEGER :: ILUOUT
!
REAL(KIND=JPRB) :: ZHOOK_HANDLE
!-------------------------------------------------------------------------------------
! Preliminaries:
!-------------------------------------------------------------------------------------
CALL GET_LUOUT(HPROGRAM,ILUOUT)
IF (LHOOK) CALL DR_HOOK('COUPLING_SEAFLUX_N',0,ZHOOK_HANDLE)
IF (HTEST/='OK') THEN
  CALL ABOR1_SFX('COUPLING_SEAFLUXN: FATAL ERROR DURING ARGUMENT TRANSFER')        
END IF
!-------------------------------------------------------------------------------------
!
ZEXNA    (:) = XUNDEF
ZEXNS    (:) = XUNDEF
ZU       (:) = XUNDEF
ZV       (:) = XUNDEF
ZWIND    (:) = XUNDEF
ZSFTQ    (:) = XUNDEF
ZSFTH    (:) = XUNDEF
ZCD      (:) = XUNDEF    
ZCDN     (:) = XUNDEF
ZCH      (:) = XUNDEF
ZCE      (:) = XUNDEF
ZRI      (:) = XUNDEF
ZHU      (:) = XUNDEF
ZRESA_SEA(:) = XUNDEF
ZUSTAR   (:) = XUNDEF
ZZ0      (:) = XUNDEF
ZZ0H     (:) = XUNDEF
ZQSAT    (:) = XUNDEF
ZHS      (:) = XUNDEF
ZTP      (:) = XUNDEF
!
ZSFTQ_ICE(:) = XUNDEF
ZSFTH_ICE(:) = XUNDEF
ZCD_ICE  (:) = XUNDEF    
ZCDN_ICE (:) = XUNDEF
ZCH_ICE  (:) = XUNDEF
ZCE_ICE  (:) = XUNDEF
ZRI_ICE  (:) = XUNDEF
ZRESA_SEA_ICE= XUNDEF
ZUSTAR_ICE(:) = XUNDEF
ZZ0_ICE  (:) = XUNDEF
ZZ0H_ICE (:) = XUNDEF
ZQSAT_ICE(:) = XUNDEF
!
ZEMIS    (:) = XUNDEF
ZTRAD    (:) = XUNDEF
ZDIR_ALB (:,:) = XUNDEF
ZSCA_ALB (:,:) = XUNDEF
!
!-------------------------------------------------------------------------------------
!
ZEXNS(:)     = (PPS(:)/XP00)**(XRD/XCPD)
ZEXNA(:)     = (PPA(:)/XP00)**(XRD/XCPD)
!
IF(LCPL_SEA .OR. LCPL_WAVE)THEN 
  !Sea currents are taken into account
  ZU(:)=PU(:)-S%XUMER(:)
  ZV(:)=PV(:)-S%XVMER(:)
ELSE
  ZU(:)=PU(:)
  ZV(:)=PV(:)        
ENDIF
!
ZWIND(:) = SQRT(ZU(:)**2+ZV(:)**2)
!
PSFTS(:,:) = 0.
!
ZHU = 1.
!
ZQA(:) = PQA(:) / PRHOA(:)

! HS value from ECMWF file
ZHS(:) = PZWS(:)
#ifdef CPLOASIS
! HS value from WW3 if activated
IF (LCPL_WAVE) THEN
  ZHS(:)=S%XHS(:)
  ZTP(:)=S%XTP(:)
ELSE
  ZHS(:)=PZWS(:)
  ZTP(:)=S%XTP(:)
END IF
#endif
! if HS value is undef : constant value and alert message
IF (ALL(ZHS==XUNDEF)) THEN
#ifdef SFX_MNH
 ZHS(:) = XZWS_DEFAULT
 WRITE (ILUOUT,*) 'WARNING : no HS values from ECMWF or WW3, then it is initialized to a constant value of XZWS_DEFAULT m'
#else
 ZHS(:)=2.
 WRITE (ILUOUT,*) 'WARNING : no HS values from ECMWF or WW3, then it is initialized to a constant value of 2 m'
#endif
END IF
!
! - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
! Time evolution
! - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
!
S%TTIME%TIME = S%TTIME%TIME + PTSTEP
 CALL ADD_FORECAST_TO_DATE_SURF(S%TTIME%TDATE%YEAR,S%TTIME%TDATE%MONTH,S%TTIME%TDATE%DAY,S%TTIME%TIME)
!
!--------------------------------------------------------------------------------------
! Fluxes over water according to Charnock formulae
!--------------------------------------------------------------------------------------
!
IF (S%LHANDLE_SIC) THEN 
  ! Flux for sea are computed everywhere
  ISIZE_WATER = SIZE(ZMASK)
  ! Ensure freezing SST values where XSST actually has very low (sea-ice) values (old habits)
  ZSST(:)=MAX(S%XSST(:), XTTSI) 
  ! Flux over sea-ice will not be computed by next calls, but by coupling_iceflux. Hence :
  ISIZE_ICE   = 0
  ! Flux over sea-ice will be computed by coupling_iceflux anywhere sea-ice could form in one 
  ! time-step (incl. under forcing). ZMASK value is set to 1. on these points
  ZMASK(:)=0.
  WHERE ( S%XSIC(:) > 0. ) ZMASK(:)=1. 
  ! To be large, assume that seaice may form where SST is < 10C
  WHERE ( S%XSST(:) - XTTS <= 10. ) ZMASK(:)=1.
  IF (S%LINTERPOL_SIC) WHERE (S%XFSIC(:) > 0. ) ZMASK(:)=1. 
  IF (S%LINTERPOL_SIT) WHERE (S%XFSIT(:) > 0. ) ZMASK(:)=1.
ELSE
  ZSST (:) = S%XSST(:)
  ZMASK(:) = S%XSST(:) - XTTS
  ISIZE_WATER = COUNT(ZMASK(:)>=0.)
  ISIZE_ICE   = SIZE(S%XSST) - ISIZE_WATER
ENDIF
!
SELECT CASE (S%CSEA_FLUX)
CASE ('DIRECT')
CALL WATER_FLUX(S%XZ0, PTA, ZEXNA, PRHOA, ZSST, ZEXNS, ZQA, &
                PRAIN, PSNOW, XTTS, ZWIND, PZREF, PUREF,    &
                PPS, S%LHANDLE_SIC, ZQSAT,  ZSFTH, ZSFTQ,   &
                ZUSTAR, ZCD, ZCDN, ZCH, ZRI, ZRESA_SEA, ZZ0H )  
CASE ('ITERAT')
CALL MR98     (S%XZ0, PTA, ZEXNA, PRHOA, S%XSST, ZEXNS, ZQA,  &
               XTTS, ZWIND, PZREF, PUREF, PPS, ZQSAT,         &
               ZSFTH, ZSFTQ, ZUSTAR,                          &
               ZCD, ZCDN, ZCH, ZRI, ZRESA_SEA, ZZ0H           )  

CASE ('ECUME ','ECUME6')
CALL ECUME_SEAFLUX(S, DGS%O, DGS%D, ZMASK, ISIZE_WATER, ISIZE_ICE,&
              PTA, ZEXNA ,PRHOA, ZSST, ZEXNS, ZQA,         &
              PRAIN, PSNOW, ZWIND, PZREF, PUREF, PPS, PPA, &
              ZQSAT, ZSFTH, ZSFTQ, ZUSTAR,                 &
              ZCD, ZCDN, ZCH, ZCE, ZRI, ZRESA_SEA, ZZ0H    )
CASE ('COARE3')
CALL COARE30_SEAFLUX(S, DGS%O, DGS%D, ZMASK, ISIZE_WATER, ISIZE_ICE,&
              PTA, ZEXNA ,PRHOA, ZSST, ZEXNS, ZQA, PRAIN,     &
              PSNOW, ZWIND, PZREF, PUREF, PPS, ZQSAT,         &
              ZSFTH, ZSFTQ, ZUSTAR,                           &
              ZCD, ZCDN, ZCH, ZCE, ZRI, ZRESA_SEA, ZZ0H     )
END SELECT

#ifdef CPLOASIS
IF (.NOT. LCPL_WAVE) THEN
  S%XHS(:)=ZHS(:)
  S%XTP(:)=ZTP(:)
END IF
#endif

!
!-------------------------------------------------------------------------------------
!radiative properties at time t
!-------------------------------------------------------------------------------------
!
ISWB = SIZE(PSW_BANDS)
!
DO JSWB=1,ISWB
ZDIR_ALB(:,JSWB) = S%XDIR_ALB(:)
ZSCA_ALB(:,JSWB) = S%XSCA_ALB(:)
END DO
!
IF (S%LHANDLE_SIC) THEN 
ZEMIS(:) =   (1 - S%XSIC(:)) * XEMISWAT    + S%XSIC(:) * XEMISWATICE
ZTRAD(:) = (((1 - S%XSIC(:)) * XEMISWAT    * S%XSST (:)**4 + &
             S%XSIC(:)  * XEMISWATICE * S%XTICE(:)**4)/ ZEMIS(:)) ** 0.25
ELSE
ZTRAD(:) = S%XSST (:)
ZEMIS(:) = S%XEMIS(:)
END IF
!
!-------------------------------------------------------------------------------------
!Specific fields for seaice model (when using earth system model or embedded  
!seaice scheme)
!-------------------------------------------------------------------------------------
!
IF(LCPL_SEAICE.OR.S%LHANDLE_SIC)THEN
CALL COUPLING_ICEFLUX_n(KI, PTA, ZEXNA, PRHOA, S%XTICE, ZEXNS, &
             ZQA, PRAIN, PSNOW, ZWIND, PZREF, PUREF,     &
             PPS, S%XSST, XTTS, ZSFTH_ICE, ZSFTQ_ICE,    &  
             S%LHANDLE_SIC, ZMASK, ZQSAT_ICE, ZZ0_ICE,   &
             ZUSTAR_ICE, ZCD_ICE, ZCDN_ICE, ZCH_ICE,   &
             ZRI_ICE, ZRESA_SEA_ICE, ZZ0H_ICE          )
ENDIF
!
IF (S%LHANDLE_SIC) CALL COMPLEMENT_EACH_OTHER_FLUX
!
!-------------------------------------------------------------------------------------
! Momentum fluxes over sea or se-ice
!-------------------------------------------------------------------------------------
!
CALL SEA_MOMENTUM_FLUXES(ZCD, ZSFU, ZSFV)
!
! Momentum fluxes over sea-ice if embedded seaice scheme is used
!
IF (S%LHANDLE_SIC) CALL SEA_MOMENTUM_FLUXES(ZCD_ICE, ZSFU_ICE, ZSFV_ICE)
!
! CO2 flux
!
PSFCO2(:) = 0.0
!
!IF(LCPL_SEA.AND.CSEACO2=='NONE')THEN
!  PSFCO2(:) = XSEACO2(:)
!ELSEIF(CSEACO2=='CST ')THEN
!  PSFCO2 = E * deltapCO2 
!  According to Wanninkhof (medium hypothesis) : 
!  E = 1.13.10^-3 * WIND^2 CO2mol.m-2.yr-1.uatm-1
!    = 1.13.10^-3 * WIND^2 * Mco2.10^-3 * (1/365*24*3600)
!  deltapCO2 = -8.7 uatm (Table 1 half hypothesis)
PSFCO2(:) = - ZWIND(:)**2 * 1.13E-3 * 8.7 * 44.E-3 / ( 365*24*3600 )
!ENDIF
!
!-------------------------------------------------------------------------------------
! Scalar fluxes:
!-------------------------------------------------------------------------------------
!
IF (CHS%SVS%NBEQ>0.AND.(KI.GT.0)) THEN
  !
  IF (CHS%CCH_DRY_DEP == "WES89") THEN
    !
    IBEG = CHS%SVS%NSV_CHSBEG
    IEND = CHS%SVS%NSV_CHSEND
    !
    CALL CH_DEP_WATER  (ZRESA_SEA, ZUSTAR, PTA, ZTRAD,PSV(:,IBEG:IEND), &
                        CHS%SVS%CSV(IBEG:IEND), CHS%XDEP(:,1:CHS%SVS%NBEQ) )  
    !
    PSFTS(:,IBEG:IEND) = - PSV(:,IBEG:IEND) * CHS%XDEP(:,1:CHS%SVS%NBEQ)  
    !
    IF (CHS%SVS%NAEREQ > 0 ) THEN
      !
      IBEG = CHS%SVS%NSV_AERBEG
      IEND = CHS%SVS%NSV_AEREND
      !
      CALL CH_AER_DEP(PSV(:,IBEG:IEND),PSFTS(:,IBEG:IEND),ZUSTAR,ZRESA_SEA,PTA,PRHOA)   
      !  
    END IF
    !
  ELSE
    !
    IBEG = CHS%SVS%NSV_AERBEG
    IEND = CHS%SVS%NSV_AEREND
    !
    PSFTS(:,IBEG:IEND) =0.
    IF (IEND.GT.IBEG)     PSFTS(:,IBEG:IEND) =0.
    !
  ENDIF
  !
ENDIF
!
IF (CHS%SVS%NDSTEQ>0.AND.(KI.GT.0)) THEN
  !
  IBEG = CHS%SVS%NSV_DSTBEG
  IEND = CHS%SVS%NSV_DSTEND
  !
  CALL DSLT_DEP(PSV(:,IBEG:IEND), PSFTS(:,IBEG:IEND), ZUSTAR, ZRESA_SEA, PTA, &
                PRHOA, DST%XEMISSIG_DST, DST%XEMISRADIUS_DST, JPMODE_DST,     &
                XDENSITY_DST, XMOLARWEIGHT_DST, ZCONVERTFACM0_DST, ZCONVERTFACM6_DST, &
                ZCONVERTFACM3_DST, LVARSIG_DST, LRGFIX_DST, CVERMOD  )  
  !
  CALL MASSFLUX2MOMENTFLUX(         &
                PSFTS(:,IBEG:IEND), & !I/O ![kg/m2/sec] In: flux of only mass, out: flux of moments
                PRHOA,                          & !I [kg/m3] air density
                DST%XEMISRADIUS_DST,                &!I [um] emitted radius for the modes (max 3)
                DST%XEMISSIG_DST,                   &!I [-] emitted sigma for the different modes (max 3)
                NDSTMDE,                        &
                ZCONVERTFACM0_DST,              &
                ZCONVERTFACM6_DST,              &
                ZCONVERTFACM3_DST,              &
                LVARSIG_DST, LRGFIX_DST         )  
  !
ENDIF

!
IF (CHS%SVS%NSLTEQ>0.AND.(KI.GT.0)) THEN
  !
  IBEG = CHS%SVS%NSV_SLTBEG
  IEND = CHS%SVS%NSV_SLTEND
  !
  ISLT = IEND - IBEG + 1
  !
  CALL COUPLING_SLT_n(SLT, &
                      SIZE(ZUSTAR,1),           & !I [nbr] number of sea point
                      ISLT,                     & !I [nbr] number of sea salt variables
                      ZWIND,                    & !I [m/s] wind velocity
                      ZHS,                      & !I [m] significant sea wave
                      S%XSST,                   &
                      ZUSTAR,                   &
                      PSFTS(:,IBEG:IEND) )   
  !
  CALL DSLT_DEP(PSV(:,IBEG:IEND), PSFTS(:,IBEG:IEND), ZUSTAR, ZRESA_SEA, PTA, &
                PRHOA, SLT%XEMISSIG_SLT, SLT%XEMISRADIUS_SLT, JPMODE_SLT,     &
                XDENSITY_SLT, XMOLARWEIGHT_SLT, ZCONVERTFACM0_SLT, ZCONVERTFACM6_SLT, &
                ZCONVERTFACM3_SLT, LVARSIG_SLT, LRGFIX_SLT, CVERMOD  )  
  !
  CALL MASSFLUX2MOMENTFLUX(                     &
                PSFTS(:,IBEG:IEND),             & !I/O [kg/m2/sec] In: flux of only mass, out: flux of moments
                PRHOA,                          & !I [kg/m3] air density
                SLT%XEMISRADIUS_SLT,            & !I [um] emitted radius for the modes (max 3)
                SLT%XEMISSIG_SLT,               & !I [-] emitted sigma for the different modes (max 3)
                NSLTMDE,                        &
                ZCONVERTFACM0_SLT,              &
                ZCONVERTFACM6_SLT,              &
                ZCONVERTFACM3_SLT,              &
                LVARSIG_SLT, LRGFIX_SLT         ) 
  !
ENDIF
!
!-------------------------------------------------------------------------------
! Inline diagnostics at time t for SST and TRAD
!-------------------------------------------------------------------------------
!
CALL DIAG_INLINE_SEAFLUX_n(DGS%O, DGS%D, DGS%DC, DGS%DI, DGS%DIC, DGS%DMI, &
                           S, PTSTEP, PTA, ZQA, PPA, PPS, PRHOA, PU,       &
                           PV, PZREF, PUREF, ZCD, ZCDN, ZCH, ZCE, ZRI, ZHU,&
                           ZZ0H, ZQSAT, ZSFTH, ZSFTQ, ZSFU, ZSFV,          &
                           PDIR_SW, PSCA_SW, PLW, ZDIR_ALB, ZSCA_ALB,      &
                           ZEMIS, ZTRAD, PRAIN, PSNOW,                     &
                           ZCD_ICE, ZCDN_ICE, ZCH_ICE, ZCE_ICE, ZRI_ICE,   &
                           ZZ0_ICE, ZZ0H_ICE, ZQSAT_ICE, ZSFTH_ICE,        &
                           ZSFTQ_ICE, ZSFU_ICE, ZSFV_ICE)
!
!-------------------------------------------------------------------------------
! A kind of "average_flux"
!-------------------------------------------------------------------------------
!
IF (S%LHANDLE_SIC) THEN
  PSFTH  (:) = ZSFTH (:) * ( 1 - S%XSIC (:)) + ZSFTH_ICE(:) * S%XSIC(:)
  PSFTQ  (:) = ZSFTQ (:) * ( 1 - S%XSIC (:)) + ZSFTQ_ICE(:) * S%XSIC(:)
  PSFU   (:) = ZSFU  (:) * ( 1 - S%XSIC (:)) +  ZSFU_ICE(:) * S%XSIC(:)
  PSFV   (:) = ZSFV  (:) * ( 1 - S%XSIC (:)) +  ZSFV_ICE(:) * S%XSIC(:)
ELSE
  PSFTH  (:) = ZSFTH (:) 
  PSFTQ  (:) = ZSFTQ (:) 
  PSFU   (:) = ZSFU  (:) 
  PSFV   (:) = ZSFV  (:) 
ENDIF
!
!-------------------------------------------------------------------------------
! IMPOSED SSS OR INTERPOLATED SSS AT TIME t+1
!-------------------------------------------------------------------------------
!
! Daily update Sea surface salinity from monthly data
!
IF (S%LINTERPOL_SSS .AND. MOD(S%TTIME%TIME,XDAY) == 0.) THEN
  CALL INTERPOL_SST_MTH(S,'S')
  IF (ANY(S%XSSS(:)<0.0)) THEN
    CALL ABOR1_SFX('COUPLING_SEAFLUX_N: XSSS should be >=0') 
  ENDIF                      
ENDIF
!
!-------------------------------------------------------------------------------
! SEA-ICE coupling at time t+1
!-------------------------------------------------------------------------------
!
IF (S%LHANDLE_SIC) THEN
  !
  IF (S%LINTERPOL_SIC) THEN
    IF ((MOD(S%TTIME%TIME,XDAY) == 0.) .OR. (PTIMEC <= PTSTEP )) THEN
      ! Daily update Sea Ice Cover constraint from monthly data
      CALL INTERPOL_SST_MTH(S,'C')
      IF (ANY(S%XFSIC(:)>1.0).OR.ANY(S%XFSIC(:)<0.0)) THEN
        CALL ABOR1_SFX('COUPLING_SEAFLUX_N: FSIC should be >=0 and <=1') 
      ENDIF                    
    ENDIF
  ENDIF
  !
  IF (S%LINTERPOL_SIT) THEN
    IF ((MOD(S%TTIME%TIME,XDAY) == 0.) .OR. (PTIMEC <= PTSTEP )) THEN
      ! Daily update Sea Ice Thickness constraint from monthly data
      CALL INTERPOL_SST_MTH(S,'H')
      IF (ANY(S%XFSIT(:)<0.0)) THEN
        CALL ABOR1_SFX('COUPLING_SEAFLUX_N: XFSIT should be >=0') 
      ENDIF  
    ENDIF
  ENDIF
  !
  IF (S%CSEAICE_SCHEME=='GELATO') THEN
    CALL SEAICE_GELATO1D_n(S, HPROGRAM,PTIMEC, PTSTEP)
  ENDIF
  ! Update of cell-averaged albedo, emissivity and radiative 
  ! temperature is done later
ENDIF
!
!-------------------------------------------------------------------------------
! OCEANIC COUPLING, IMPOSED SST OR INTERPOLATED SST AT TIME t+1
!-------------------------------------------------------------------------------
!
IF (O%LMERCATOR) THEN
  !
  ! Update SST reference profile for relaxation purpose
  IF (DTS%LSST_DATA) THEN
    CALL SST_UPDATE(DTS, S, OR%XSEAT_REL(:,NOCKMIN+1))
    !
    ! Convert to degree C for ocean model
    OR%XSEAT_REL(:,NOCKMIN+1) = OR%XSEAT_REL(:,NOCKMIN+1) - XTT
  ENDIF
  !
  CALL MOD1D_n(DGS%GO, O, OR, G%XLAT, S, &
               HPROGRAM,PTIME,ZEMIS(:),ZDIR_ALB(:,1:KSW),ZSCA_ALB(:,1:KSW),&
               PLW(:),PSCA_SW(:,1:KSW),PDIR_SW(:,1:KSW),PSFTH(:),          &
               PSFTQ(:),PSFU(:),PSFV(:),PRAIN(:))
  !
ELSEIF(DTS%LSST_DATA) THEN 
  !
  ! Imposed SST 
  !
  CALL SST_UPDATE(DTS, S, S%XSST)
  !
ELSEIF (S%LINTERPOL_SST.AND.MOD(S%TTIME%TIME,XDAY) == 0.) THEN
   !
   ! Imposed monthly SST 
   !
   CALL INTERPOL_SST_MTH(S,'T')
   !
ENDIF
!
!-------------------------------------------------------------------------------
!Physical properties see by the atmosphere in order to close the energy budget 
!between surfex and the atmosphere. All variables should be at t+1 but very 
!difficult to do. Maybe it will be done later. However, Ts is at time t+1
!-------------------------------------------------------------------------------
!
IF (S%LHANDLE_SIC) THEN
  IF (S%CSEAICE_SCHEME/='GELATO') THEN
     S%XTICE    = S%XSST
     S%XSIC     = S%XFSIC
     S%XICE_ALB = XALBSEAICE           
  ENDIF         
  PTSURF (:) = S%XSST(:) * ( 1 - S%XSIC (:)) +   S%XTICE(:) * S%XSIC(:)
  PQSURF (:) = ZQSAT (:) * ( 1 - S%XSIC (:)) + ZQSAT_ICE(:) * S%XSIC(:)
  ZZ0W   (:) = ( 1 - S%XSIC(:) ) * 1.0/(LOG(PUREF(:)/ZZ0(:))    **2)  +  &
                     S%XSIC(:)   * 1.0/(LOG(PUREF(:)/ZZ0_ICE(:))**2)  
  PZ0    (:) = PUREF (:) * EXP ( - SQRT ( 1./  ZZ0W(:) ))
  ZZ0W   (:) = ( 1 - S%XSIC(:) ) * 1.0/(LOG(PZREF(:)/ZZ0H(:))    **2)  +  &
                     S%XSIC(:)   * 1.0/(LOG(PZREF(:)/ZZ0H_ICE(:))**2)  
  PZ0H   (:) = PZREF (:) * EXP ( - SQRT ( 1./  ZZ0W(:) ))
ELSE
  PTSURF (:) = S%XSST(:) 
  PQSURF (:) = ZQSAT (:) 
  PZ0    (:) = S%XZ0 (:) 
  PZ0H   (:) = ZZ0H  (:) 
ENDIF
!
!-------------------------------------------------------------------------------
!Radiative properties at time t+1 (see by the atmosphere) in order to close
!the energy budget between surfex and the atmosphere
!-------------------------------------------------------------------------------
!
 CALL UPDATE_RAD_SEA(S,PZENITH2,XTTS,PDIR_ALB,PSCA_ALB,PEMIS,PTRAD,PU,PV)
!
!=======================================================================================
!
IF (LHOOK) CALL DR_HOOK('COUPLING_SEAFLUX_N',1,ZHOOK_HANDLE)
!
!=======================================================================================
!
CONTAINS
!
SUBROUTINE SEA_MOMENTUM_FLUXES(PCD, PSFU, PSFV)
!
IMPLICIT NONE
!
REAL, DIMENSION(KI), INTENT(IN)  :: PCD       ! Drag coefficient (on open sea or seaice)
REAL, DIMENSION(KI), INTENT(OUT) :: PSFU      ! zonal momentum flux                   (Pa)
REAL, DIMENSION(KI), INTENT(OUT) :: PSFV      ! meridian momentum flux                (Pa)
!
REAL, DIMENSION(KI) :: ZUSTAR2    ! square of friction velocity (m2/s2)
REAL, DIMENSION(KI) :: ZWORK      ! Work array
!
REAL, DIMENSION(KI) :: ZPEW_A_COEF
REAL, DIMENSION(KI) :: ZPEW_B_COEF
!
REAL(KIND=JPRB) :: ZHOOK_HANDLE
!
IF (LHOOK) CALL DR_HOOK('COUPLING_SEAFLUX_N: SEA_MOMENTUM_FLUXES',0,ZHOOK_HANDLE)
!
IF( (LCPL_SEA .OR. LCPL_WAVE) .AND. HCOUPLING .EQ. 'E')THEN
  ZPEW_A_COEF(:)=0.0
  ZPEW_B_COEF(:)=ZWIND(:)
ELSE
  ZPEW_A_COEF(:)=PPEW_A_COEF(:)
  ZPEW_B_COEF(:)=PPEW_B_COEF(:)
ENDIF
!
ZWORK  (:) = XUNDEF
ZUSTAR2(:) = XUNDEF
!
IF(CIMPLICIT_WIND=='OLD')THEN
! old implicitation (m2/s2)
  ZUSTAR2(:) = (PCD(:)*ZWIND(:)*ZPEW_B_COEF(:)) /            &
              (1.0-PRHOA(:)*PCD(:)*ZWIND(:)*ZPEW_A_COEF(:))
ELSE
! new implicitation (m2/s2)
  ZUSTAR2(:) = (PCD(:)*ZWIND(:)*(2.*ZPEW_B_COEF(:)-ZWIND(:))) /&
              (1.0-2.0*PRHOA(:)*PCD(:)*ZWIND(:)*ZPEW_A_COEF(:))
!                   
  ZWORK(:)  = PRHOA(:)*PPEW_A_COEF(:)*ZUSTAR2(:) + ZPEW_B_COEF(:)
  ZWORK(:) = MAX(ZWORK(:),0.)
!
  WHERE(ZPEW_A_COEF(:)/= 0.)
        ZUSTAR2(:) = MAX( ( ZWORK(:) - PPEW_B_COEF(:) ) / (PRHOA(:)*ZPEW_A_COEF(:)), 0.)
  ENDWHERE
!              
ENDIF
!
PSFU = 0.
PSFV = 0.
WHERE (ZWIND(:)>0.)
  PSFU(:) = - PRHOA(:) * ZUSTAR2(:) * ZU(:) / ZWIND(:)
  PSFV(:) = - PRHOA(:) * ZUSTAR2(:) * ZV(:) / ZWIND(:)
END WHERE
!
IF (LHOOK) CALL DR_HOOK('COUPLING_SEAFLUX_N: SEA_MOMENTUM_FLUXES',1,ZHOOK_HANDLE)
!
END SUBROUTINE SEA_MOMENTUM_FLUXES
!
!=======================================================================================
!
SUBROUTINE COMPLEMENT_EACH_OTHER_FLUX
!
! Provide dummy fluxes on places with no open-sea or no sea-ice 
! Allows a smooth computing of CLS parameters in all cases while avoiding 
! having to pack arrays (in routines PARAM_CLS and CLS_TQ)
!
IMPLICIT NONE
!
REAL(KIND=JPRB) :: ZHOOK_HANDLE
!
IF (LHOOK) CALL DR_HOOK('COUPLING_SEAFLUX_N: COMPLEMENT_EACH_OTHER_FLUX',0,ZHOOK_HANDLE)
!
  WHERE (S%XSIC(:) == 1.)
     ZSFTH=ZSFTH_ICE 
     ZSFTQ=ZSFTQ_ICE 
     ZSFU=ZSFU_ICE
     ZSFV=ZSFV_ICE
     ZQSAT=ZQSAT_ICE
     ZCD=ZCD_ICE
     ZCDN=ZCDN_ICE
     ZCH=ZCH_ICE
     ZCE=ZCE_ICE
     ZRI=ZRI_ICE 
     ZZ0H=ZZ0H_ICE
  END WHERE
  WHERE (S%XSIC(:) == 0.)
     ZSFTH_ICE=ZSFTH 
     ZSFTQ_ICE=ZSFTQ 
     ZSFU_ICE=ZSFU
     ZSFV_ICE=ZSFV
     ZQSAT_ICE=ZQSAT
     ZCD_ICE=ZCD
     ZCDN_ICE=ZCDN
     ZCH_ICE=ZCH
     ZCE_ICE=ZCE
     ZRI_ICE=ZRI 
     ZZ0H_ICE=ZZ0H
  END WHERE
!
IF (LHOOK) CALL DR_HOOK('COUPLING_SEAFLUX_N: COMPLEMENT_EACH_OTHER_FLUX',1,ZHOOK_HANDLE)
!
END SUBROUTINE COMPLEMENT_EACH_OTHER_FLUX
!
!=======================================================================================
!
END SUBROUTINE COUPLING_SEAFLUX_n
