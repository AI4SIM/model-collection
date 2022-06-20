!SFX_LIC Copyright 1994-2014 CNRS, Meteo-France and Universite Paul Sabatier
!SFX_LIC This is part of the SURFEX software governed by the CeCILL-C licence
!SFX_LIC version 1. See LICENSE, CeCILL-C_V1-en.txt and CeCILL-C_V1-fr.txt  
!SFX_LIC for details. version 1.
!     #########
    SUBROUTINE ECUME_FLUX(DGO,D,PZ0SEA,PTA,PEXNA,PRHOA,PSST,PEXNS,PQA,PVMOD, &
                            PZREF,PUREF,PPS,PICHCE,OPRECIP,OPWEBB,OPWG,&
                            PQSAT,PSFTH,PSFTQ,PUSTAR,PCD,PCDN,PCH,PCE, &
                            PRI,PRESA,PRAIN,PZ0HSEA,OPERTFLUX,PPERTFLUX)  
!###############################################################################
!!
!!****  *ECUME_FLUX*
!!
!!    PURPOSE
!!    -------
!       Calculate the surface turbulent fluxes of heat, moisture, and momentum 
!       over sea surface + corrections due to rainfall & Webb effect.
!!
!!**  METHOD
!!    ------
!       The estimation of the transfer coefficients relies on the iterative 
!       computation of the scaling parameters U*/Teta*/q*. The convergence is
!       supposed to be reached in NITERFL iterations maximum.
!       The neutral transfer coefficients for momentum/temperature/humidity
!       are computed as a function of the 10m-height neutral wind speed using
!       the ECUME_v0 formulation based on the multi-campaign (POMME,FETCH,CATCH,
!       SEMAPHORE,EQUALANT) ALBATROS dataset. See  MERSEA report for more
!       details on the ECUME formulation.
!!
!!    EXTERNAL
!!    --------
!!
!!    IMPLICIT ARGUMENTS
!!    ------------------
!!
!!    REFERENCE
!!    ---------
!!      Fairall et al (1996), JGR, 3747-3764
!!      Gosnell et al (1995), JGR, 437-442
!!      Fairall et al (1996), JGR, 1295-1308
!!
!!    AUTHOR
!!    ------
!!      C. Lebeaupin  *Météo-France* (adapted from S. Belamari's code)
!!
!!    MODIFICATIONS
!!    -------------
!!      Original     15/03/2005
!!      Modified        01/2006  C. Lebeaupin (adapted from  A. Pirani's code)
!!      Modified     20/07/2009  S. Belamari
!!      Modified        08/2009  B. Decharme: limitation of Ri
!!      Modified        09/2012  B. Decharme: CD correction
!!      Modified        09/2012  B. Decharme: limitation of Ri in surface_ri.F90
!!      Modified        10/2012  P. Le Moigne: extra inputs for FLake use
!!      Modified        06/2013  B. Decharme: bug in z0 (output) computation 
!!      Modified        06/2013  J.Escobar : for REAL4/8 add EPSILON management
!!      Modified        04/2022  J. Pianezze : add diagnostics
!!
!-------------------------------------------------------------------------------

!       0.   DECLARATIONS
!            ------------
!
USE MODD_DIAG_n,    ONLY : DIAG_t, DIAG_OPTIONS_t
!
USE MODD_CSTS,       ONLY : XKARMAN, XG, XSTEFAN, XRD, XRV, &
                            XLVTT, XCL, XCPD, XCPV, XRHOLW, &
                            XTT,XP00
USE MODD_SURF_PAR,   ONLY : XUNDEF, XSURF_EPSILON                    
!
USE MODD_REPROD_OPER,  ONLY : CCHARNOCK
!
USE MODD_SURF_PAR,   ONLY : XUNDEF
USE MODD_SNOW_PAR,   ONLY : XZ0SN, XZ0HSN
USE MODD_SURF_ATM,   ONLY : XVCHRNK, XVZ0CM
!
USE MODD_WATER_PAR
!
USE MODI_WIND_THRESHOLD
USE MODI_SURFACE_RI
!
USE MODE_THERMOS
!
!
USE YOMHOOK   ,ONLY : LHOOK,   DR_HOOK
USE PARKIND1  ,ONLY : JPRB
!
IMPLICIT NONE

!       0.1. Declarations of arguments
!
TYPE(DIAG_t),         INTENT(INOUT) :: D
TYPE(DIAG_OPTIONS_t), INTENT(INOUT) :: DGO
!
REAL, DIMENSION(:), INTENT(IN)    :: PTA       ! air temperature, atm.lev (K)
REAL, DIMENSION(:), INTENT(IN)    :: PQA       ! air spec. hum., atm.lev (kg/kg)
REAL, DIMENSION(:), INTENT(IN)    :: PRHOA     ! air density, atm.lev (kg/m3)
REAL, DIMENSION(:), INTENT(IN)    :: PVMOD     ! module of wind, atm.lev (m/s)
REAL, DIMENSION(:), INTENT(IN)    :: PZREF     ! atm.level for temp./hum. (m)
REAL, DIMENSION(:), INTENT(IN)    :: PUREF     ! atm.level for wind (m)
REAL, DIMENSION(:), INTENT(IN)    :: PSST      ! Sea Surface Temperature (K)
REAL, DIMENSION(:), INTENT(IN)    :: PPS       ! air pressure at sea surf. (Pa)
REAL, DIMENSION(:), INTENT(IN)    :: PRAIN     ! precipitation rate (kg/s/m2)
REAL, DIMENSION(:), INTENT(IN)    :: PEXNA     ! Exner function at atm. level
REAL, DIMENSION(:), INTENT(IN)    :: PEXNS     ! Exner function at sea surface
REAL, DIMENSION(:), INTENT(IN)    :: PPERTFLUX ! stochastic flux perturbation pattern

REAL,               INTENT(IN)    :: PICHCE    !
LOGICAL,            INTENT(IN)    :: OPRECIP   !
LOGICAL,            INTENT(IN)    :: OPWEBB    !
LOGICAL,            INTENT(IN)    :: OPWG      !
LOGICAL,            INTENT(IN)    :: OPERTFLUX !

REAL, DIMENSION(:), INTENT(INOUT) :: PZ0SEA    ! roughness length over the ocean

! surface fluxes : latent heat, sensible heat, friction fluxes
REAL, DIMENSION(:), INTENT(OUT)   :: PSFTH     ! heat flux (W/m2)
REAL, DIMENSION(:), INTENT(OUT)   :: PSFTQ     ! water flux (kg/m2/s)
REAL, DIMENSION(:), INTENT(OUT)   :: PUSTAR    ! friction velocity (m/s)

! diagnostics
REAL, DIMENSION(:), INTENT(OUT)   :: PQSAT     ! sea surface spec. hum. (kg/kg)
REAL, DIMENSION(:), INTENT(OUT)   :: PCD       ! transfer coef. for momentum
REAL, DIMENSION(:), INTENT(OUT)   :: PCH       ! transfer coef. for temperature
REAL, DIMENSION(:), INTENT(OUT)   :: PCE       ! transfer coef. for humidity
REAL, DIMENSION(:), INTENT(OUT)   :: PCDN      ! neutral coef. for momentum
REAL, DIMENSION(:), INTENT(OUT)   :: PRESA     ! aerodynamical resistance
REAL, DIMENSION(:), INTENT(OUT)   :: PRI       ! Richardson number
REAL, DIMENSION(:), INTENT(OUT)   :: PZ0HSEA   ! heat roughness length

!       0.2. Declarations of local variables

REAL, DIMENSION(SIZE(PTA))        :: ZTAU      ! momentum flux (N/m2)
REAL, DIMENSION(SIZE(PTA))        :: ZHF       ! sensible heat flux (W/m2)
REAL, DIMENSION(SIZE(PTA))        :: ZEF       ! latent heat flux (W/m2)
REAL, DIMENSION(SIZE(PTA))        :: ZTAUR     ! momentum flx due to rain (N/m2)
REAL, DIMENSION(SIZE(PTA))        :: ZRF       ! sensible flx due to rain (W/m2)
REAL, DIMENSION(SIZE(PTA))        :: ZEFWEBB   ! Webb corr. on latent flx (W/m2)

REAL, DIMENSION(SIZE(PTA))        :: ZVMOD     ! wind intensity at atm.lev (m/s)
REAL, DIMENSION(SIZE(PTA))        :: ZQSATA    ! sat.spec.hum., atm.lev (kg/kg)
REAL, DIMENSION(SIZE(PTA))        :: ZPA       ! air pressure at atm. level (Pa)
REAL, DIMENSION(SIZE(PTA))        :: ZUSR      ! velocity scaling param. (m/s)
                                               ! =friction velocity
REAL, DIMENSION(SIZE(PTA))        :: ZTSR      ! temperature scaling param. (K)
REAL, DIMENSION(SIZE(PTA))        :: ZQSR      ! humidity scaling param. (kg/kg)
REAL, DIMENSION(SIZE(PTA))        :: ZWG       ! gustiness factor (m/s)

REAL, DIMENSION(SIZE(PTA))        :: ZUSTAR2   ! square of friction velocity
REAL, DIMENSION(SIZE(PTA))        :: ZAC       ! aerodynamical conductance
REAL, DIMENSION(SIZE(PTA))        :: ZDIRCOSZW ! orography slope cosine
                                               ! (=1 on water!)

REAL, DIMENSION(SIZE(PTA))        :: ZLV,ZLR   ! vap.heat, sea/atm level (J/kg)
REAL, DIMENSION(SIZE(PTA))        :: ZDU,ZDTH,ZDQ,ZDUWG
                                               ! vert. gradients (real atm.)
REAL, DIMENSION(SIZE(PTA))        :: ZDELTAU10N,ZDELTAT10N,ZDELTAQ10N
                                               ! vert. gradients (10-m, neutral)
REAL, DIMENSION(SIZE(PTA))        :: ZCHN,ZCEN ! neutral coef. for T,Q
REAL, DIMENSION(SIZE(PTA))        :: ZD0
REAL, DIMENSION(SIZE(PTA))        :: ZCHARN    !Charnock number
REAL, DIMENSION(SIZE(PTA))        :: ZTVSR,ZBF ! constants to compute gustiness factor
!
REAL    :: ZETV,ZRDSRV     ! thermodynamic constants
REAL    :: ZLMOU,ZLMOT     ! Obukhovs stability param. z/l for U, T/Q
REAL    :: ZPSI_U,ZPSI_T   ! PSI funct. for U, T/Q
REAL    :: ZLMOMIN,ZLMOMAX ! min/max value of Obukhovs stability parameter z/l
REAL    :: ZBETAGUST       ! gustiness factor
REAL    :: ZZBL            !atm. boundary layer depth (m)
!
REAL    :: ZCHIC,ZCHIK,ZEPS,ZLOGHS10,ZLOGTS10,ZPI,ZPIS2,ZPSIC,ZPSIK, &
             ZSQR3,ZZDQ,ZZDTH  
!
REAL    :: ZALFAC,ZCPLW,ZDQSDT,ZDTMP,ZDWAT,ZP00,ZTAC,ZWW
                                ! to compute rainfall impact & Webb correction
!
INTEGER :: NITERFL         ! maximum number of iterations (5 or 6)
INTEGER :: JLON, JJ
REAL(KIND=JPRB) :: ZHOOK_HANDLE
!
!-------------------------------------------------------------------------------
IF (LHOOK) CALL DR_HOOK('ECUME_FLUX',0,ZHOOK_HANDLE)
!
NITERFL = 10
!
!-------------------------------------------------------------------------------
!
!       1.   AUXILIARY CONSTANTS & ARRAY INITIALISATION BY UNDEFINED VALUES.
!       --------------------------------------------------------------------
!
ZLMOMIN = -200.0
ZLMOMAX = 0.25
ZP00    = 1013.25E+02
ZPIS2   = 2.0*ATAN(1.0)
ZPI     = 2.0*ZPIS2
ZSQR3   = SQRT(3.0)
ZEPS    = 1.E-8
ZETV    = XRV/XRD-1.0
ZRDSRV  = XRD/XRV
!
ZBETAGUST=1.2  ! value based on TOGA-COARE experiment
ZZBL     =600. ! Set a default value for boundary layer depth
!
ZDIRCOSZW(:)=1.
!
PCD (:) = XUNDEF
PCH (:) = XUNDEF
PCE (:) = XUNDEF
PCDN(:) = XUNDEF
ZUSR(:) = XUNDEF
ZTSR(:) = XUNDEF
ZQSR(:) = XUNDEF
ZTAU(:) = XUNDEF
ZHF (:) = XUNDEF
ZEF (:) = XUNDEF
!
PSFTH (:) = XUNDEF
PSFTQ (:) = XUNDEF
PUSTAR(:) = XUNDEF
PRESA (:) = XUNDEF
PRI   (:) = XUNDEF
!
ZWG    (:) = 0.0
ZTAUR  (:) = 0.0
ZRF    (:) = 0.0
ZEFWEBB(:) = 0.0
!
!-------------------------------------------------------------------------------
!
!       2.   INITIALISATIONS BEFORE ITERATIVE LOOP.
!       -------------------------------------------
!
ZVMOD (:) = WIND_THRESHOLD(PVMOD(:),PUREF(:))   !set a minimum value to wind
!
!       2.1. Specific humidity at saturation
!
PQSAT (:) = QSAT_SEAWATER(PSST(:),PPS(:))                       !at sea surface
ZPA   (:) = XP00*(PEXNA(:)**(XCPD/XRD))
ZQSATA(:) = QSAT(PTA(:),ZPA(:))                                 !at atm. level
!
!       2.2. Gradients at the air-sea interface
!
ZDU (:) = ZVMOD(:)              !one assumes u is measured / sea surface current
ZDTH(:) = PTA(:)/PEXNA(:)-PSST(:)/PEXNS(:)
ZDQ (:) = PQA(:)-PQSAT(:)
!
!       2.3. Initial guess
!
ZD0(:) = 1.2+6.3E-03*MAX(ZDU(:)-10.0,0.0)
!
IF (OPWG) THEN !initial guess for gustiness factor
   ZWG   (:) = 0.5
   ZDUWG (:) = SQRT(ZDU(:)**2+ZWG(:)**2)
ELSE
   ZDUWG (:) = ZDU(:)
ENDIF
!
ZDELTAU10N(:) = ZDUWG(:)
ZDELTAT10N(:) = ZDTH (:)*ZD0(:)
ZDELTAQ10N(:) = ZDQ  (:)
!
!       2.4. Latent heat of vaporisation
!
ZLV(:) = XLVTT+(XCPV-XCL)*(PSST(:)-XTT)                 !at sea surface
ZLR(:) = XLVTT+(XCPV-XCL)*(PTA(:)-XTT)                  !at atm.level
!
!       2.5. Charnock number
!
IF(CCHARNOCK=='OLD')THEN
  ZCHARN(:) = XVCHRNK
ELSE
! vary between 0.011 et 0.018 according to Chris Fairall's data as in coare3.0        
  ZCHARN(:) = MAX(0.011,MIN(0.018,0.011+0.007*(ZDUWG(:)-10.)/8.))
ENDIF
!
!-------------------------------------------------------------------------------
!
!       3.   ITERATIVE LOOP TO COMPUTE U*, T*, Q*.
!       ------------------------------------------
!
DO JJ=1,NITERFL
  DO JLON=1,SIZE(PTA)
!
!       3.1. Neutral coefficient for wind speed cdn (ECUME_v0 formulation)
!
    IF (ZDELTAU10N(JLON) <= 16.8) THEN
      PCDN(JLON) = 1.3013E-03                          &
                  + (-1.2719E-04 * ZDELTAU10N(JLON)   ) &
                  + (+1.3067E-05 * ZDELTAU10N(JLON)**2) &
                  + (-2.2261E-07 * ZDELTAU10N(JLON)**3)  
    ELSEIF (ZDELTAU10N(JLON) <= 50.0) THEN
      PCDN(JLON) = 1.3633E-03                          &
                  + (-1.3056E-04 * ZDELTAU10N(JLON)   ) &
                  + (+1.6212E-05 * ZDELTAU10N(JLON)**2) &
                  + (-4.8208E-07 * ZDELTAU10N(JLON)**3) &
                  + (+4.2684E-09 * ZDELTAU10N(JLON)**4)  
    ELSE
      PCDN(JLON) = 1.7828E-03
    ENDIF
!
!       3.2. Neutral coefficient for temperature chn (ECUME_v0 formulation)
!
    IF (ZDELTAU10N(JLON) <= 33.0) THEN
      ZCHN(JLON) = 1.2536E-03                          &
                  + (-1.2455E-04 * ZDELTAU10N(JLON)   ) &
                  + (+1.6038E-05 * ZDELTAU10N(JLON)**2) &
                  + (-4.3701E-07 * ZDELTAU10N(JLON)**3) &
                  + (+3.4517E-09 * ZDELTAU10N(JLON)**4) &
                  + (+3.5763E-12 * ZDELTAU10N(JLON)**5)  
    ELSE
      ZCHN(JLON) = 3.1374E-03
    ENDIF
!
!       3.3. Neutral coefficient for humidity cen (ECUME_v0 formulation)
!
    IF (ZDELTAU10N(JLON) <= 29.0) THEN
      ZCEN(JLON) = 1.2687E-03                          &
                  + (-1.1384E-04 * ZDELTAU10N(JLON)   ) &
                  + (+1.1467E-05 * ZDELTAU10N(JLON)**2) &
                  + (-3.9144E-07 * ZDELTAU10N(JLON)**3) &
                  + (+5.0864E-09 * ZDELTAU10N(JLON)**4)  
    ELSEIF (ZDELTAU10N(JLON) <= 33.0) THEN
      ZCEN(JLON) = -1.3526E-03                         &
                  + (+1.8229E-04 * ZDELTAU10N(JLON)   ) &
                  + (-2.6995E-06 * ZDELTAU10N(JLON)**2)  
    ELSE
      ZCEN(JLON) = 1.7232E-03
    ENDIF
    ZCEN(JLON) = ZCEN(JLON)*(1.0-PICHCE)+ZCHN(JLON)*PICHCE
!
!       3.4. Scaling parameters and roughness lenght
!
    ZUSR(JLON) = SQRT(PCDN(JLON))*ZDELTAU10N(JLON)
    ZTSR(JLON) = ZCHN(JLON)/SQRT(PCDN(JLON))*ZDELTAT10N(JLON)
    ZQSR(JLON) = ZCEN(JLON)/SQRT(PCDN(JLON))*ZDELTAQ10N(JLON)
!
!       3.5. Gustiness factor ZWG following Mondon & Redelsperger (1998)
!
    IF(OPWG) THEN
      ZTVSR(JLON)=ZTSR(JLON)*(1.0+ZETV*PQA(JLON))+ZETV*PTA(JLON)*ZQSR(JLON)
      ZBF(JLON)=MAX(0.0,-XG/PTA(JLON)*ZUSR(JLON)*ZTVSR(JLON))
      ZWG(JLON)=ZBETAGUST*(ZBF(JLON)*ZZBL)**(1./3.)
    ENDIF
!
!       3.6. Obukhovs stability param. z/l following Liu et al. (JAS, 1979)
!
! For U
    ZLMOU = PUREF(JLON)*XG*XKARMAN*(ZTSR(JLON)/(PTA(JLON)) &
       +ZETV*ZQSR(JLON)/(1.0+ZETV*PQA(JLON)))/MAX(ZUSR(JLON),ZEPS)**2  
! For T/Q
    ZLMOT = ZLMOU*PZREF(JLON)/PUREF(JLON)
    ZLMOU = MAX(MIN(ZLMOU,ZLMOMAX),ZLMOMIN)
    ZLMOT = MAX(MIN(ZLMOT,ZLMOMAX),ZLMOMIN)
!
!       3.7. Stability function psi (see Liu et al, 1979 ; Dyer and Hicks, 1970)
!            Modified to include convective form following Fairall (unpublished)
!
!   For U
    IF (ZLMOU == 0.0) THEN
      ZPSI_U = 0.0
    ELSEIF (ZLMOU > 0.0) THEN
      ZPSI_U = -7.0*ZLMOU
    ELSE
      ZCHIK  = (1.0-16.0*ZLMOU)**0.25
      ZPSIK  = 2.0*LOG((1.0+ZCHIK)/2.0) &
                +LOG((1.0+ZCHIK**2)/2.0) &
                -2.0*ATAN(ZCHIK)+ZPIS2  
      ZCHIC  = (1.0-12.87*ZLMOU)**(1.0/3.0)     !for very unstable conditions
      ZPSIC  = 1.5*LOG((ZCHIC**2+ZCHIC+1.0)/3.0)  &
                -ZSQR3*ATAN((2.0*ZCHIC+1.0)/ZSQR3) &
                +ZPI/ZSQR3  
      ZPSI_U = ZPSIC+(ZPSIK-ZPSIC)/(1.0+ZLMOU**2)
                                                !match Kansas & free-conv. forms
    ENDIF
!   For T/Q
    IF (ZLMOT == 0.0) THEN
      ZPSI_T = 0.0
    ELSEIF (ZLMOT > 0.0) THEN
      ZPSI_T = -7.0*ZLMOT
    ELSE
      ZCHIK  = (1.0-16.0*ZLMOT)**0.25
      ZPSIK  = 2.0*LOG((1.0+ZCHIK**2)/2.0)
      ZCHIC  = (1.0-12.87*ZLMOT)**(1.0/3.0)     !for very unstable conditions
      ZPSIC  = 1.5*LOG((ZCHIC**2+ZCHIC+1.0)/3.0)  &
                -ZSQR3*ATAN((2.0*ZCHIC+1.0)/ZSQR3) &
                +ZPI/ZSQR3  
      ZPSI_T = ZPSIC+(ZPSIK-ZPSIC)/(1.0+ZLMOT**2)
                                                !match Kansas & free-conv. forms
    ENDIF
!
!       3.8. Update ZDELTAU10N, ZDELTAT10N and ZDELTAQ10N
!
    ZLOGHS10 = LOG(PUREF(JLON)/10.0)
    ZLOGTS10 = LOG(PZREF(JLON)/10.0)
    ZDUWG     (JLON) = SQRT(ZDU(JLON)**2+ZWG(JLON)**2)
    ZDELTAU10N(JLON) = ZDUWG(JLON)-ZUSR(JLON)*(ZLOGHS10-ZPSI_U)/XKARMAN
    ZDELTAT10N(JLON) = ZDTH (JLON)-ZTSR(JLON)*(ZLOGTS10-ZPSI_T)/XKARMAN
    ZDELTAQ10N(JLON) = ZDQ  (JLON)-ZQSR(JLON)*(ZLOGTS10-ZPSI_T)/XKARMAN

  ENDDO
ENDDO
!
!-------------------------------------------------------------------------------
!
!       4.   COMPUTATION OF EXCHANGE COEFFICIENTS AND TURBULENT FLUXES.
!       ---------------------------------------------------------------
!
DO JLON=1,SIZE(PTA)
!
!       4.1. Exchange coefficients PCD, PCH, PCE
!
  ZZDTH = 0.5* &
           ((1.0+SIGN(1.0,ZDTH(JLON)))*MAX(ZDTH(JLON),ZEPS) &
           +(1.0-SIGN(1.0,ZDTH(JLON)))*MIN(ZDTH(JLON),-ZEPS))  
  ZZDQ  = 0.5* &
           ((1.0+SIGN(1.0,ZDQ(JLON)))*MAX(ZDQ(JLON),ZEPS)   &
           +(1.0-SIGN(1.0,ZDQ(JLON)))*MIN(ZDQ(JLON),-ZEPS))  
  PCD(JLON) = (ZUSR(JLON)/ZDUWG(JLON))**2
  PCH(JLON) = ZUSR(JLON)*ZTSR(JLON)/(ZDUWG(JLON)*ZZDTH)
  PCE(JLON) = ZUSR(JLON)*ZQSR(JLON)/(ZDUWG(JLON)*ZZDQ)
!
!       4.2. Surface turbulent fluxes
!            (ATM CONV.: ZTAU<<0 ; ZHF,ZEF<0 if atm looses heat)
!
  ZTAU(JLON) = -PRHOA(JLON)*PCD(JLON)*ZDUWG(JLON)**2
  ZHF (JLON) = -PRHOA(JLON)*XCPD*PCH(JLON)*ZDUWG(JLON)*ZDTH(JLON)
  ZEF (JLON) = -PRHOA(JLON)*ZLV(JLON)*PCE(JLON)*ZDUWG(JLON)*ZDQ(JLON)
!
!       4.3. Stochastic perturbation of turbulent fluxes

  IF( OPERTFLUX )THEN
    ZTAU(JLON) = ZTAU(JLON)* ( 1. + PPERTFLUX(JLON) / 2. )
    ZHF (JLON) = ZHF(JLON)*  ( 1. + PPERTFLUX(JLON) / 2. )
    ZEF (JLON) = ZEF(JLON)*  ( 1. + PPERTFLUX(JLON) / 2. )
  ENDIF
!
ENDDO
!
!-------------------------------------------------------------------------------
!
!       5.   COMPUTATION OF FLUX CORRECTIONS DUE TO RAINFALL.
!            (ATM conv: ZRF<0 if atm. looses heat, ZTAUR<<0)
!       -----------------------------------------------------

IF(OPRECIP) THEN
  DO JLON=1,SIZE(PTA)
!
!       5.1. Momentum flux due to rainfall (ZTAUR, N/m2)
!
! See pp3752 in FBR96.
    ZTAUR(JLON) = -PRAIN(JLON)*ZDUWG(JLON)
!
!       5.2. Sensible heat flux due to rainfall (ZRF, W/m2)
!
! See Eq.12 in GoF95, with ZCPLW as specific heat of water (J/kg/K), ZDWAT as
! water vapor diffusivity (Eq.13-3 of Pruppacher and Klett, 1978), ZDTMP as
! heat diffusivity, ZDQSDT from Clausius-Clapeyron relation and ZALFAC as
! wet-bulb factor (Eq.11 in GoF95).
    ZTAC   = PTA(JLON)-XTT
    ZCPLW  = 4224.8482+ZTAC*(-4.707+ZTAC*(0.08499 &
              +ZTAC*(1.2826E-03+ZTAC*(4.7884E-05   &
              -2.0027E-06*ZTAC))))  
    ZDWAT  = 2.11E-05*(ZP00/ZPA(JLON)) &
              *(PTA(JLON)/XTT)**1.94  
    ZDTMP  = (1.0+3.309E-03*ZTAC-1.44E-06*ZTAC**2) &
              *0.02411/(PRHOA(JLON)*XCPD)  
    ZDQSDT = ZQSATA(JLON)*ZLR(JLON)/(XRD*PTA(JLON)**2)
    ZALFAC = 1.0/(1.0+ZDQSDT*(ZLR(JLON)*ZDWAT)/(ZDTMP*XCPD))
    ZRF(JLON) = ZCPLW*PRAIN(JLON)*ZALFAC*((PSST(JLON)-PTA(JLON)) &
                 +(PQSAT(JLON)-PQA(JLON))*(ZLR(JLON)*ZDWAT)/(ZDTMP*XCPD))  

  ENDDO
ENDIF
!
!-------------------------------------------------------------------------------
!
!       6.   COMPUTATION OF WEBB CORRECTION TO LATENT HEAT FLUX (ZEFWEBB, W/m2).
!       ------------------------------------------------------------------------
!
! See Eq.21 and Eq.22 in FBR96.
IF (OPWEBB) THEN
  DO JLON=1,SIZE(PTA)
    ZWW = -(1.0+ZETV)*(PCE(JLON)*ZDUWG(JLON)*ZDQ(JLON)) &
           -(1.0+(1.0+ZETV)*PQA(JLON))*              &
           (PCH(JLON)*ZDUWG(JLON)*ZDTH(JLON))/(PTA(JLON))  
    ZEFWEBB(JLON) = PRHOA(JLON)*ZWW*ZLV(JLON)*PQA(JLON)
  ENDDO
ENDIF
!
!-------------------------------------------------------------------------------
!
!       7.   FINAL STEP : TOTAL SURFACE FLUXES AND DERIVED DIAGNOSTICS. 
!       ---------------------------------------------------------------
!
!       7.1. Richardson number
!
 CALL SURFACE_RI(PSST,PQSAT,PEXNS,PEXNA,PTA,ZQSATA, &
                PZREF,PUREF,ZDIRCOSZW,PVMOD,PRI    )  
!
!       7.2. Friction velocity which contains correction du to rain
!
ZUSTAR2(:)=-(ZTAU(:)+ZTAUR(:))/PRHOA(:)
!
IF(OPRECIP) THEN
  PCD(:)=ZUSTAR2(:)/(ZDUWG(:)**2)
ENDIF
!
PUSTAR(:)=SQRT(ZUSTAR2(:))
!
!       7.3. Aerodynamical conductance and resistance
!
ZAC(:)=PCH(:)*ZVMOD(:)
PRESA(:)=1./ MAX(ZAC(:),XSURF_EPSILON)
!
!       7.4. Total surface fluxes
!
PSFTH(:)=ZHF(:)+ZRF(:)
PSFTQ(:)=(ZEF(:)+ZEFWEBB(:))/ZLV(:)
!
!       7.5. Z0 and Z0H over water
!
PZ0SEA(:) = ZCHARN(:) * ZUSTAR2(:) / XG + XVZ0CM * PCD(:) / PCDN(:)
!
PZ0HSEA(:)=PZ0SEA(:)
!
IF (DGO%LDIAG_OAFLUX) THEN
  D%XDIAG_USTAR(:)=ZUSR (:)
  D%XDIAG_TSTAR(:)=ZTSR (:)
  D%XDIAG_QSTAR(:)=ZQSR (:)
  D%XDIAG_ZDU  (:)=ZDU  (:)
  D%XDIAG_ZDWG (:)=ZDUWG(:)
  D%XDIAG_PTA  (:)=PTA  (:)
  D%XDIAG_PEXNA(:)=PEXNA(:)
  D%XDIAG_SSTS (:)=PSST (:)
  D%XDIAG_PEXNS(:)=PEXNS(:)
  D%XDIAG_PQSAT(:)=PQSAT(:)
  D%XDIAG_PQA  (:)=PQA  (:)
  D%XDIAG_PUREF(:)=PUREF(:)
  D%XDIAG_PZREF(:)=PZREF(:)
  D%XDIAG_ZPUZ (:)=0.0
  D%XDIAG_ZPTZ (:)=0.0
  D%XDIAG_ZPQZ (:)=0.0
  D%XDIAG_PRHOA(:)=PRHOA(:)
  D%XDIAG_ZLV  (:)=ZLV  (:)
  D%XDIAG_CDN  (:)=PCDN (:)
  D%XDIAG_CHN  (:)=ZCHN (:)
  D%XDIAG_CEN  (:)=ZCEN (:)
ENDIF
!
IF (LHOOK) CALL DR_HOOK('ECUME_FLUX',1,ZHOOK_HANDLE)
!-------------------------------------------------------------------------------

END SUBROUTINE ECUME_FLUX
