!SFX_LIC Copyright 1994-2014 CNRS, Meteo-France and Universite Paul Sabatier
!SFX_LIC This is part of the SURFEX software governed by the CeCILL-C licence
!SFX_LIC version 1. See LICENSE, CeCILL-C_V1-en.txt and CeCILL-C_V1-fr.txt  
!SFX_LIC for details. version 1.
!SFX_LIC Copyright 1994-2014 CNRS, Meteo-France and Universite Paul Sabatier
!SFX_LIC This is part of the SURFEX software governed by the CeCILL-C licence
!SFX_LIC version 1. See LICENSE, CeCILL-C_V1-en.txt and CeCILL-C_V1-fr.txt  
!SFX_LIC for details. version 1.
!     #########
    SUBROUTINE ECUMEV6_FLUX(DGO,D,PZ0SEA,PTA,PEXNA,PRHOA,PSST,PSSS,PEXNS,PQA,PVMOD, &
                            PZREF,PUREF,PPS,PPA,PICHCE,OPRECIP,OPWEBB,        &
                            PQSAT,PSFTH,PSFTQ,PUSTAR,PCD,PCDN,PCH,PCE,        &
                            PRI,PRESA,PRAIN,KZ0,PZ0HSEA,OPERTFLUX,PPERTFLUX   )
!###############################################################################
!!
!!****  *ECUMEV6_FLUX*
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
!       Neutral transfer coefficients for momentum/temperature/humidity
!       are computed as a function of the 10m-height neutral wind speed using
!       the ECUME_V6 formulation based on the multi-campaign (POMME,FETCH,CATCH,
!       SEMAPHORE,EQUALANT) ALBATROS dataset.
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
!!      Modified        08/2009  B. Decharme: limitation of Ri
!!      Modified        09/2012  B. Decharme: CD correction
!!      Modified        09/2012  B. Decharme: limitation of Ri in surface_ri.F90
!!      Modified        10/2012  P. Le Moigne: extra inputs for FLake use
!!      Modified        06/2013  B. Decharme: bug in z0 (output) computation 
!!      Modified        12/2013  S. Belamari: ZRF computation updated:
!!                                1. ZP00/PPA in ZDWAT, ZLVA in ZDQSDT/ZBULB/ZRF
!!                                2. ZDWAT/ZDTMP in ZBULB/ZRF (Gosnell et al 95)
!!                                3. cool skin correction included
!!      Modified        01/2014  S. Belamari: salinity impact on latent heat of
!!                                vaporization of seawater included
!!      Modified        01/2014  S. Belamari: new formulation for pure water
!!                                specific heat (ZCPWA)
!!      Modified        01/2014  S. Belamari: 4 choices for PZ0SEA computation
!!      Modified        12/2015  S. Belamari: ECUME now provides parameterisations
!!                                for:  U10n*sqrt(CDN)          instead of CDN
!!                                      U10n*CHN/sqrt(CDN)         "       CHN
!!                                      U10n*CEN/sqrt(CDN)         "       CEN
!!      Modified        01/2016  S. Belamari: New ECUME formulation
!!      Modified        04/2022  J. Pianezze : add diagnostics
!!
!!      To be done:
!!      include gustiness computation following Mondon & Redelsperger (1998)
!!!
!-------------------------------------------------------------------------------
!!
!!    MODIFICATIONS RELATED TO SST CORRECTION COMPUTATION
!!    ---------------------------------------------------
!!      Modified        09/2013  S. Belamari: use 0.98 for the ocean emissivity
!!                                following up to date satellite measurements in
!!                                the 8-14 μm range (obtained values range from
!!                                0.98 to 0.99).
!!!
!-------------------------------------------------------------------------------
!
!       0.   DECLARATIONS
!            ------------
!
USE MODD_DIAG_n,    ONLY : DIAG_t, DIAG_OPTIONS_t
!
USE MODD_CSTS,             ONLY : XPI, XDAY, XKARMAN, XG, XP00, XSTEFAN, XRD, XRV,   &
                                  XCPD, XCPV, XCL, XTT, XLVTT
USE MODD_SURF_PAR,         ONLY : XUNDEF
USE MODD_SURF_ATM,         ONLY : XVCHRNK, XVZ0CM
USE MODD_REPROD_OPER,      ONLY : CCHARNOCK
!
USE MODE_THERMOS
USE MODI_WIND_THRESHOLD
USE MODI_SURFACE_RI
!
USE YOMHOOK,   ONLY : LHOOK,   DR_HOOK
USE PARKIND1,  ONLY : JPRB
!
USE MODI_ABOR1_SFX
!
IMPLICIT NONE
!
!       0.1. Declarations of arguments
!
TYPE(DIAG_t),         INTENT(INOUT) :: D
TYPE(DIAG_OPTIONS_t), INTENT(INOUT) :: DGO
!
REAL, DIMENSION(:), INTENT(IN)    :: PVMOD      ! module of wind at atm level (m/s)
REAL, DIMENSION(:), INTENT(IN)    :: PTA        ! air temperature at atm level (K)
REAL, DIMENSION(:), INTENT(IN)    :: PQA        ! air spec. hum. at atm level (kg/kg)
REAL, DIMENSION(:), INTENT(IN)    :: PPA        ! air pressure at atm level (Pa)
REAL, DIMENSION(:), INTENT(IN)    :: PRHOA      ! air density at atm level (kg/m3)
REAL, DIMENSION(:), INTENT(IN)    :: PEXNA      ! Exner function at atm level
REAL, DIMENSION(:), INTENT(IN)    :: PUREF      ! atm level for wind (m)
REAL, DIMENSION(:), INTENT(IN)    :: PZREF      ! atm level for temp./hum. (m)
REAL, DIMENSION(:), INTENT(IN)    :: PSSS       ! Sea Surface Salinity (g/kg)
REAL, DIMENSION(:), INTENT(IN)    :: PPS        ! air pressure at sea surface (Pa)
REAL, DIMENSION(:), INTENT(IN)    :: PEXNS      ! Exner function at sea surface
REAL, DIMENSION(:), INTENT(IN)    :: PPERTFLUX  ! stochastic flux perturbation pattern
! for correction
REAL,               INTENT(IN)    :: PICHCE    !
LOGICAL,            INTENT(IN)    :: OPRECIP   !
LOGICAL,            INTENT(IN)    :: OPWEBB    !
LOGICAL,            INTENT(IN)    :: OPERTFLUX
REAL, DIMENSION(:), INTENT(IN)    :: PRAIN     ! precipitation rate (kg/s/m2)
!
INTEGER,            INTENT(IN)    :: KZ0
!
REAL, DIMENSION(:), INTENT(INOUT) :: PSST       ! Sea Surface Temperature (K)
REAL, DIMENSION(:), INTENT(INOUT) :: PZ0SEA     ! roughness length over sea
REAL, DIMENSION(:), INTENT(OUT)   :: PZ0HSEA    ! heat roughness length over sea

! surface fluxes : latent heat, sensible heat, friction fluxes
REAL, DIMENSION(:), INTENT(OUT)   :: PUSTAR     ! friction velocity (m/s)
REAL, DIMENSION(:), INTENT(OUT)   :: PSFTH      ! heat flux (W/m2)
REAL, DIMENSION(:), INTENT(OUT)   :: PSFTQ      ! water flux (kg/m2/s)

! diagnostics
REAL, DIMENSION(:), INTENT(OUT)   :: PQSAT      ! sea surface spec. hum. (kg/kg)
REAL, DIMENSION(:), INTENT(OUT)   :: PCD        ! transfer coef. for momentum
REAL, DIMENSION(:), INTENT(OUT)   :: PCH        ! transfer coef. for temperature
REAL, DIMENSION(:), INTENT(OUT)   :: PCE        ! transfer coef. for humidity
REAL, DIMENSION(:), INTENT(OUT)   :: PCDN       ! neutral coef. for momentum
REAL, DIMENSION(:), INTENT(OUT)   :: PRI        ! Richardson number
REAL, DIMENSION(:), INTENT(OUT)   :: PRESA      ! aerodynamical resistance
!
!       0.2. Declarations of local variables
!
! specif SB
INTEGER, DIMENSION(SIZE(PTA))     :: JCV        ! convergence index
INTEGER, DIMENSION(SIZE(PTA))     :: JITER      ! nb of iterations to converge

REAL, DIMENSION(SIZE(PTA))        :: ZTAU       ! momentum flux (N/m2)
REAL, DIMENSION(SIZE(PTA))        :: ZHF        ! sensible heat flux (W/m2)
REAL, DIMENSION(SIZE(PTA))        :: ZEF        ! latent heat flux (W/m2)
REAL, DIMENSION(SIZE(PTA))        :: ZTAUR      ! momentum flx due to rain (N/m2)
REAL, DIMENSION(SIZE(PTA))        :: ZRF        ! sensible flx due to rain (W/m2)
REAL, DIMENSION(SIZE(PTA))        :: ZEFWEBB    ! Webb corr. on latent flx (W/m2)

REAL, DIMENSION(SIZE(PTA))        :: ZVMOD      ! wind intensity at atm level (m/s)
REAL, DIMENSION(SIZE(PTA))        :: ZQSATA     ! sat.spec.hum. at atm level (kg/kg)
REAL, DIMENSION(SIZE(PTA))        :: ZLVA       ! vap.heat of pure water at atm level (J/kg)
REAL, DIMENSION(SIZE(PTA))        :: ZLVS       ! vap.heat of seawater at sea surface (J/kg)
REAL, DIMENSION(SIZE(PTA))        :: ZCPA       ! specif.heat moist air (J/kg/K)
REAL, DIMENSION(SIZE(PTA))        :: ZVISA      ! kinemat.visc. of dry air (m2/s)
REAL, DIMENSION(SIZE(PTA))        :: ZDU        ! U   vert.grad. (real atm)
REAL, DIMENSION(SIZE(PTA))        :: ZDT,ZDQ    ! T,Q vert.grad. (real atm)
REAL, DIMENSION(SIZE(PTA))        :: ZDDU       ! U   vert.grad. (real atm + gust)
REAL, DIMENSION(SIZE(PTA))        :: ZDDT,ZDDQ  ! T,Q vert.grad. (real atm + WL/CS)
REAL, DIMENSION(SIZE(PTA))        :: ZUSR       ! velocity scaling param. (m/s)
                                                ! =friction velocity
REAL, DIMENSION(SIZE(PTA))        :: ZTSR       ! temperature scaling param. (K)
REAL, DIMENSION(SIZE(PTA))        :: ZQSR       ! humidity scaling param. (kg/kg)
REAL, DIMENSION(SIZE(PTA))        :: ZDELTAU10N,ZDELTAT10N,ZDELTAQ10N
                                                ! U,T,Q vert.grad. (10m, neutral atm)
REAL, DIMENSION(SIZE(PTA))        :: ZUSR0,ZTSR0,ZQSR0    ! ITERATIVE PROCESS
REAL, DIMENSION(SIZE(PTA))        :: ZDUSTO,ZDTSTO,ZDQSTO ! ITERATIVE PROCESS
REAL, DIMENSION(SIZE(PTA))        :: ZPSIU,ZPSIT! PSI funct for U, T/Q (Z0 comp)
REAL, DIMENSION(SIZE(PTA))        :: ZCHARN     ! Charnock parameter   (Z0 comp)

REAL, DIMENSION(SIZE(PTA))        :: ZUSTAR2    ! square of friction velocity
REAL, DIMENSION(SIZE(PTA))        :: ZAC        ! aerodynamical conductance
REAL, DIMENSION(SIZE(PTA))        :: ZDIRCOSZW  ! orography slope cosine
                                                ! (=1 on water!)
REAL, DIMENSION(SIZE(PTA))        :: ZPARUN,ZPARTN,ZPARQN ! neutral parameter for U,T,Q
REAL, DIMENSION(0:5)              :: ZCOEFU,ZCOEFT,ZCOEFQ

! local constants
LOGICAL :: OPCVFLX              ! to force convergence
INTEGER :: NITERMAX             ! nb of iterations to get free convergence
INTEGER :: NITERSUP             ! nb of additional iterations if OPCVFLX=.TRUE.
INTEGER :: NITERFL              ! maximum number of iterations
REAL    :: ZETV,ZRDSRV          ! thermodynamic constants
REAL    :: ZSQR3
REAL    :: ZLMOMIN,ZLMOMAX      ! min/max value of Obukhovs stability param. z/l
REAL    :: ZBTA,ZGMA            ! parameters of the stability functions
REAL    :: ZDUSR0,ZDTSR0,ZDQSR0 ! maximum gap for USR/TSR/QSR between 2 steps
REAL    :: ZP00                 ! [OPRECIP] - water vap. diffusiv.ref.press.(Pa)
REAL    :: ZUTU,ZUTT,ZUTQ       ! U10n threshold in ECUME parameterisation
REAL    :: ZCDIRU,ZCDIRT,ZCDIRQ ! coef directeur pour fonction affine U,T,Q
REAL    :: ZORDOU,ZORDOT,ZORDOQ ! ordonnee a l'origine pour fonction affine U,T,Q

INTEGER :: JJ                                   ! for ITERATIVE PROCESS
INTEGER :: JLON,JK
REAL    :: ZLMOU,ZLMOT                          ! Obukhovs param. z/l for U, T/Q
REAL    :: ZPSI_U,ZPSI_T                        ! PSI funct. for U, T/Q
REAL    :: Z0TSEA,Z0QSEA                        ! roughness length for T, Q
REAL    :: ZCHIC,ZCHIK,ZPSIC,ZPSIK,ZLOGUS10,ZLOGTS10
REAL    :: ZTAC,ZCPWA,ZDQSDT,ZDWAT,ZDTMP,ZBULB  ! [OPRECIP]
REAL    :: ZWW                                  ! [OPWEBB]

REAL(KIND=JPRB) :: ZHOOK_HANDLE
!
!-------------------------------------------------------------------------------
IF (LHOOK) CALL DR_HOOK('ECUMEV6_FLUX',0,ZHOOK_HANDLE)
!
ZDUSR0   = 1.E-06
ZDTSR0   = 1.E-06
ZDQSR0   = 1.E-09
!
NITERMAX = 5
NITERSUP = 5
OPCVFLX  = .TRUE.
!
NITERFL = NITERMAX
IF (OPCVFLX) NITERFL = NITERMAX+NITERSUP
!
ZCOEFU = (/ 1.00E-03, 3.66E-02, -1.92E-03, 2.32E-04, -7.02E-06,  6.40E-08 /)
ZCOEFT = (/ 5.36E-03, 2.90E-02, -1.24E-03, 4.50E-04, -2.06E-05,       0.0 /)
ZCOEFQ = (/ 1.00E-03, 3.59E-02, -2.87E-04,      0.0,       0.0,       0.0 /)
!
ZUTU = 40.0
ZUTT = 14.4
ZUTQ = 10.0
!
ZCDIRU = ZCOEFU(1) + 2.0*ZCOEFU(2)*ZUTU + 3.0*ZCOEFU(3)*ZUTU**2   &
                   + 4.0*ZCOEFU(4)*ZUTU**3 + 5.0*ZCOEFU(5)*ZUTU**4
ZCDIRT = ZCOEFT(1) + 2.0*ZCOEFT(2)*ZUTT + 3.0*ZCOEFT(3)*ZUTT**2   &
                   + 4.0*ZCOEFT(4)*ZUTT**3
ZCDIRQ = ZCOEFQ(1) + 2.0*ZCOEFQ(2)*ZUTQ
!
ZORDOU = ZCOEFU(0) + ZCOEFU(1)*ZUTU + ZCOEFU(2)*ZUTU**2 + ZCOEFU(3)*ZUTU**3   &
                   + ZCOEFU(4)*ZUTU**4 + ZCOEFU(5)*ZUTU**5
ZORDOT = ZCOEFT(0) + ZCOEFT(1)*ZUTT + ZCOEFT(2)*ZUTT**2 + ZCOEFT(3)*ZUTT**3   &
                   + ZCOEFT(4)*ZUTT**4
ZORDOQ = ZCOEFQ(0) + ZCOEFQ(1)*ZUTQ + ZCOEFQ(2)*ZUTQ**2
!
!-------------------------------------------------------------------------------
!
!       1.   AUXILIARY CONSTANTS & ARRAY INITIALISATION BY UNDEFINED VALUES.
!       --------------------------------------------------------------------
!
ZDIRCOSZW(:) = 1.0
!
ZETV    = XRV/XRD-1.0   !~0.61 (cf Liu et al. 1979)
ZRDSRV  = XRD/XRV       !~0.622
ZSQR3   = SQRT(3.0)
ZLMOMIN = -200.0
ZLMOMAX = 0.25
ZBTA    = 16.0
ZGMA    = 7.0           !initially =4.7, modified to 7.0 following G. Caniaux
!
ZP00    = 1013.25E+02
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
ZTAUR  (:) = 0.0
ZRF    (:) = 0.0
ZEFWEBB(:) = 0.0
!
!-------------------------------------------------------------------------------
!
!       2.   INITIALISATIONS BEFORE ITERATIVE LOOP.
!       -------------------------------------------
!
ZVMOD(:) = WIND_THRESHOLD(PVMOD(:),PUREF(:))    !set a minimum value to wind
!
!       2.0. Radiative fluxes - For warm layer & cool skin
!
!       2.0b. Warm Layer correction
!
!       2.1. Specific humidity at saturation
!
WHERE(PSSS(:)>0.0.AND.PSSS(:)/=XUNDEF)
  PQSAT (:) = QSAT_SEAWATER2(PSST(:),PPS(:),PSSS(:))    !at sea surface
ELSEWHERE
  PQSAT (:) = QSAT_SEAWATER (PSST(:),PPS(:))            !at sea surface
ENDWHERE
ZQSATA(:) = QSAT(PTA(:),PPA(:))                         !at atm level
!
!       2.2. Gradients at the air-sea interface
!
ZDU(:) = ZVMOD(:)               !one assumes u is measured / sea surface current
ZDT(:) = PTA(:)/PEXNA(:)-PSST(:)/PEXNS(:)
ZDQ(:) = PQA(:)-PQSAT(:)
!
!       2.3. Latent heat of vaporisation
!
ZLVA(:) = XLVTT+(XCPV-XCL)*(PTA (:)-XTT)                !of pure water at atm level
ZLVS(:) = XLVTT+(XCPV-XCL)*(PSST(:)-XTT)                !of pure water at sea surface
WHERE(PSSS(:)>0.0.AND.PSSS(:)/=XUNDEF)
  ZLVS(:) = ZLVS(:)*(1.0-1.00472E-3*PSSS(:))            !of seawater at sea surface
ENDWHERE
!
!       2.4. Specific heat of moist air (Businger 1982)
!
!ZCPA(:) = XCPD*(1.0+(XCPV/XCPD-1.0)*PQA(:))
ZCPA(:) = XCPD
!
!       2.4b Kinematic viscosity of dry air (Andreas 1989, CRREL Rep. 89-11)
!
ZVISA(:) = 1.326E-05*(1.0+6.542E-03*(PTA(:)-XTT)+8.301E-06*(PTA(:)-XTT)**2   &
           -4.84E-09*(PTA(:)-XTT)**3)
!
!       2.4c Coefficients for warm layer and/or cool skin correction
!
!       2.5. Initial guess
!
ZDDU(:) = ZDU(:)
ZDDT(:) = ZDT(:)
ZDDQ(:) = ZDQ(:)
ZDDU(:) = SIGN(MAX(ABS(ZDDU(:)),10.0*ZDUSR0),ZDDU(:))
ZDDT(:) = SIGN(MAX(ABS(ZDDT(:)),10.0*ZDTSR0),ZDDT(:))
ZDDQ(:) = SIGN(MAX(ABS(ZDDQ(:)),10.0*ZDQSR0),ZDDQ(:))
!
JCV (:) = -1
ZUSR(:) = 0.04*ZDDU(:)
ZTSR(:) = 0.04*ZDDT(:)
ZQSR(:) = 0.04*ZDDQ(:)
ZDELTAU10N(:) = ZDDU(:)
ZDELTAT10N(:) = ZDDT(:)
ZDELTAQ10N(:) = ZDDQ(:)
JITER(:) = 99
!
! In the following, we suppose that Richardson number PRI < XRIMAX
! If not true, Monin-Obukhov theory can't (and therefore shouldn't) be applied !
!-------------------------------------------------------------------------------
!
!       3.   ITERATIVE LOOP TO COMPUTE U*, T*, Q*.
!       ------------------------------------------
!
DO JJ=1,NITERFL
  DO JLON=1,SIZE(PTA)
!
  IF (JCV(JLON) == -1) THEN
    ZUSR0(JLON)=ZUSR(JLON)
    ZTSR0(JLON)=ZTSR(JLON)
    ZQSR0(JLON)=ZQSR(JLON)
    IF (JJ == NITERMAX+1 .OR. JJ == NITERMAX+NITERSUP) THEN
      ZDELTAU10N(JLON) = 0.5*(ZDUSTO(JLON)+ZDELTAU10N(JLON))    !forced convergence
      ZDELTAT10N(JLON) = 0.5*(ZDTSTO(JLON)+ZDELTAT10N(JLON))
      ZDELTAQ10N(JLON) = 0.5*(ZDQSTO(JLON)+ZDELTAQ10N(JLON))
      IF (JJ == NITERMAX+NITERSUP) JCV(JLON)=3
    ENDIF
    ZDUSTO(JLON) = ZDELTAU10N(JLON)
    ZDTSTO(JLON) = ZDELTAT10N(JLON)
    ZDQSTO(JLON) = ZDELTAQ10N(JLON)
!
!       3.1. Neutral parameter for wind speed (ECUME_V6 formulation)
!
    IF (ZDELTAU10N(JLON) <= ZUTU) THEN
      ZPARUN(JLON) = ZCOEFU(0) + ZCOEFU(1)*ZDELTAU10N(JLON)      &
                               + ZCOEFU(2)*ZDELTAU10N(JLON)**2   &
                               + ZCOEFU(3)*ZDELTAU10N(JLON)**3   &
                               + ZCOEFU(4)*ZDELTAU10N(JLON)**4   &
                               + ZCOEFU(5)*ZDELTAU10N(JLON)**5
    ELSE
      ZPARUN(JLON) = ZCDIRU*(ZDELTAU10N(JLON)-ZUTU) + ZORDOU
    ENDIF
    PCDN(JLON) = (ZPARUN(JLON)/ZDELTAU10N(JLON))**2
!
!       3.2. Neutral parameter for temperature (ECUME_V6 formulation)
!
    IF (ZDELTAU10N(JLON) <= ZUTT) THEN
      ZPARTN(JLON) = ZCOEFT(0) + ZCOEFT(1)*ZDELTAU10N(JLON)      &
                               + ZCOEFT(2)*ZDELTAU10N(JLON)**2   &
                               + ZCOEFT(3)*ZDELTAU10N(JLON)**3   &
                               + ZCOEFT(4)*ZDELTAU10N(JLON)**4
    ELSE
      ZPARTN(JLON) = ZCDIRT*(ZDELTAU10N(JLON)-ZUTT) + ZORDOT
    ENDIF
!
!       3.3. Neutral parameter for humidity (ECUME_V6 formulation)
!
    IF (ZDELTAU10N(JLON) <= ZUTQ) THEN
      ZPARQN(JLON) = ZCOEFQ(0) + ZCOEFQ(1)*ZDELTAU10N(JLON)      &
                               + ZCOEFQ(2)*ZDELTAU10N(JLON)**2
    ELSE
      ZPARQN(JLON) = ZCDIRQ*(ZDELTAU10N(JLON)-ZUTQ) + ZORDOQ
    ENDIF
!
!       3.4. Scaling parameters U*, T*, Q*
!
    ZUSR(JLON) = ZPARUN(JLON)
    ZTSR(JLON) = ZPARTN(JLON)*ZDELTAT10N(JLON)/ZDELTAU10N(JLON)
    ZQSR(JLON) = ZPARQN(JLON)*ZDELTAQ10N(JLON)/ZDELTAU10N(JLON)
!
!       3.4b Gustiness factor (Deardorff 1970)
!
!       3.4c Cool skin correction
!
!       3.5. Obukhovs stability param. z/l following Liu et al. (JAS, 1979)
!
! For U
    ZLMOU = PUREF(JLON)*XG*XKARMAN*(ZTSR(JLON)/PTA(JLON)   &
            +ZETV*ZQSR(JLON)/(1.0+ZETV*PQA(JLON)))/ZUSR(JLON)**2
! For T/Q
    ZLMOT = ZLMOU*(PZREF(JLON)/PUREF(JLON))
    ZLMOU = MAX(MIN(ZLMOU,ZLMOMAX),ZLMOMIN)
    ZLMOT = MAX(MIN(ZLMOT,ZLMOMAX),ZLMOMIN)
!
!       3.6. Stability function psi (see Liu et al, 1979 ; Dyer and Hicks, 1970)
!            Modified to include convective form following Fairall (unpublished)
!
! For U
    IF (ZLMOU == 0.0) THEN
      ZPSI_U = 0.0
    ELSEIF (ZLMOU > 0.0) THEN
      ZPSI_U = -ZGMA*ZLMOU
    ELSE
      ZCHIK  = (1.0-ZBTA*ZLMOU)**0.25
      ZPSIK  = 2.0*LOG((1.0+ZCHIK)/2.0)  &
                +LOG((1.0+ZCHIK**2)/2.0) &
                -2.0*ATAN(ZCHIK)+0.5*XPI
      ZCHIC  = (1.0-12.87*ZLMOU)**(1.0/3.0)       !for very unstable conditions
      ZPSIC  = 1.5*LOG((ZCHIC**2+ZCHIC+1.0)/3.0)   &
                -ZSQR3*ATAN((2.0*ZCHIC+1.0)/ZSQR3) &
                +XPI/ZSQR3
      ZPSI_U = ZPSIC+(ZPSIK-ZPSIC)/(1.0+ZLMOU**2) !match Kansas & free-conv. forms
    ENDIF
    ZPSIU(JLON) = ZPSI_U
! For T/Q
    IF (ZLMOT == 0.0) THEN
      ZPSI_T = 0.0
    ELSEIF (ZLMOT > 0.0) THEN
      ZPSI_T = -ZGMA*ZLMOT
    ELSE
      ZCHIK  = (1.0-ZBTA*ZLMOT)**0.25
      ZPSIK  = 2.0*LOG((1.0+ZCHIK**2)/2.0)
      ZCHIC  = (1.0-12.87*ZLMOT)**(1.0/3.0)       !for very unstable conditions
      ZPSIC  = 1.5*LOG((ZCHIC**2+ZCHIC+1.0)/3.0)   &
                -ZSQR3*ATAN((2.0*ZCHIC+1.0)/ZSQR3) &
                +XPI/ZSQR3
      ZPSI_T = ZPSIC+(ZPSIK-ZPSIC)/(1.0+ZLMOT**2) !match Kansas & free-conv. forms
    ENDIF
    ZPSIT(JLON) = ZPSI_T
!
!       3.7. Update air-sea gradients
!
    ZDDU(JLON) = ZDU(JLON)
    ZDDT(JLON) = ZDT(JLON)
    ZDDQ(JLON) = ZDQ(JLON)
    ZDDU(JLON) = SIGN(MAX(ABS(ZDDU(JLON)),10.0*ZDUSR0),ZDDU(JLON))
    ZDDT(JLON) = SIGN(MAX(ABS(ZDDT(JLON)),10.0*ZDTSR0),ZDDT(JLON))
    ZDDQ(JLON) = SIGN(MAX(ABS(ZDDQ(JLON)),10.0*ZDQSR0),ZDDQ(JLON))
    ZLOGUS10   = LOG(PUREF(JLON)/10.0)
    ZLOGTS10   = LOG(PZREF(JLON)/10.0)
    ZDELTAU10N(JLON) = ZDDU(JLON)-ZUSR(JLON)*(ZLOGUS10-ZPSI_U)/XKARMAN
    ZDELTAT10N(JLON) = ZDDT(JLON)-ZTSR(JLON)*(ZLOGTS10-ZPSI_T)/XKARMAN
    ZDELTAQ10N(JLON) = ZDDQ(JLON)-ZQSR(JLON)*(ZLOGTS10-ZPSI_T)/XKARMAN
    ZDELTAU10N(JLON) = SIGN(MAX(ABS(ZDELTAU10N(JLON)),10.0*ZDUSR0),   &
                            ZDELTAU10N(JLON))
    ZDELTAT10N(JLON) = SIGN(MAX(ABS(ZDELTAT10N(JLON)),10.0*ZDTSR0),   &
                            ZDELTAT10N(JLON))
    ZDELTAQ10N(JLON) = SIGN(MAX(ABS(ZDELTAQ10N(JLON)),10.0*ZDQSR0),   &
                            ZDELTAQ10N(JLON))
!
!       3.8. Test convergence for U*, T*, Q*
!
    IF (ABS(ZUSR(JLON)-ZUSR0(JLON)) < ZDUSR0 .AND.   &
        ABS(ZTSR(JLON)-ZTSR0(JLON)) < ZDTSR0 .AND.   &
        ABS(ZQSR(JLON)-ZQSR0(JLON)) < ZDQSR0) THEN
      JCV(JLON) = 1                                     !free convergence
      IF (JJ >= NITERMAX+1) JCV(JLON) = 2               !leaded convergence
    ENDIF
    JITER(JLON) = JJ
  ENDIF
!
  ENDDO
ENDDO
!
!-------------------------------------------------------------------------------
!
!       4.   COMPUTATION OF TURBULENT FLUXES AND EXCHANGE COEFFICIENTS.
!       ---------------------------------------------------------------
!
DO JLON=1,SIZE(PTA)
!
!       4.1. Surface turbulent fluxes
!            (ATM CONV.: ZTAU<<0 ; ZHF,ZEF<0 if atm looses heat)
!
  ZTAU(JLON) = -PRHOA(JLON)*ZUSR(JLON)**2
  ZHF (JLON) = -PRHOA(JLON)*ZCPA(JLON)*ZUSR(JLON)*ZTSR(JLON)
  ZEF (JLON) = -PRHOA(JLON)*ZLVS(JLON)*ZUSR(JLON)*ZQSR(JLON)
!
!       4.2. Exchange coefficients PCD, PCH, PCE
!
  PCD(JLON) = (ZUSR(JLON)/ZDDU(JLON))**2
  PCH(JLON) = (ZUSR(JLON)*ZTSR(JLON))/(ZDDU(JLON)*ZDDT(JLON))
  PCE(JLON) = (ZUSR(JLON)*ZQSR(JLON))/(ZDDU(JLON)*ZDDQ(JLON))
!
!       4.3. Stochastic perturbation of turbulent fluxes
!
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
!
IF (OPRECIP) THEN
  DO JLON=1,SIZE(PTA)
!
!       5.1. Momentum flux due to rainfall (ZTAUR, N/m2)
!
! See pp3752 in FBR96.
    ZTAUR(JLON) = -0.85*PRAIN(JLON)*PVMOD(JLON)
!
!       5.2. Sensible heat flux due to rainfall (ZRF, W/m2)
!
! See Eq.12 in GoF95 with ZCPWA as specific heat of water at atm level (J/kg/K),
! ZDQSDT from Clausius-Clapeyron relation, ZDWAT as water vapor diffusivity 
! (Eq.13-3 of Pruppacher and Klett, 1978), ZDTMP as heat diffusivity, and ZBULB
! as wet-bulb factor (Eq.11 in GoF95).
!
    ZTAC   = PTA(JLON)-XTT
    ZCPWA  = 4217.51 -3.65566*ZTAC +0.1381*ZTAC**2       &
              -2.8309E-03*ZTAC**3 +3.42061E-05*ZTAC**4   &
              -2.18107E-07*ZTAC**5 +5.74535E-10*ZTAC**6
    ZDQSDT = (ZLVA(JLON)*ZQSATA(JLON))/(XRV*PTA(JLON)**2)
    ZDWAT  = 2.11E-05*(ZP00/PPA(JLON))*(PTA(JLON)/XTT)**1.94
    ZDTMP  = (1.0+3.309E-03*ZTAC-1.44E-06*ZTAC**2)   &
              *0.02411/(PRHOA(JLON)*ZCPA(JLON))
    ZBULB  = 1.0/(1.0+ZDQSDT*(ZLVA(JLON)*ZDWAT)/(ZCPA(JLON)*ZDTMP))
    ZRF(JLON) = PRAIN(JLON)*ZCPWA*ZBULB*((PSST(JLON)-PTA(JLON))   &
                +(PQSAT(JLON)-PQA(JLON))*(ZLVA(JLON)*ZDWAT)/(ZCPA(JLON)*ZDTMP))
!
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
    ZWW = (1.0+ZETV)*(ZUSR(JLON)*ZQSR(JLON))   &
           +(1.0+(1.0+ZETV)*PQA(JLON))*(ZUSR(JLON)*ZTSR(JLON))/PTA(JLON)
    ZEFWEBB(JLON) = -PRHOA(JLON)*ZLVS(JLON)*ZWW*PQA(JLON)
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
 CALL SURFACE_RI(PSST,PQSAT,PEXNS,PEXNA,PTA,PQA,   &
                PZREF,PUREF,ZDIRCOSZW,PVMOD,PRI)
!
!       7.2. Friction velocity which contains correction due to rain
!
ZUSTAR2(:) = -(ZTAU(:)+ZTAUR(:))/PRHOA(:)       !>>0 as ZTAU<<0 & ZTAUR<=0
!
IF (OPRECIP) THEN
  PCD(:) = ZUSTAR2(:)/ZDDU(:)**2
ENDIF
!
PUSTAR(:) = SQRT(ZUSTAR2(:))                    !>>0
!
!       7.3. Aerodynamical conductance and resistance
!
ZAC  (:) = PCH(:)*ZDDU(:)
PRESA(:) = 1.0/ZAC(:)
!
!       7.4. Total surface fluxes
!
PSFTH(:) =  ZHF(:)+ZRF(:)
PSFTQ(:) = (ZEF(:)+ZEFWEBB(:))/ZLVS(:)
!
!       7.5. Charnock number
!
IF (CCHARNOCK == 'OLD') THEN
  ZCHARN(:) = XVCHRNK
ELSE            !modified for moderate wind speed as in COARE3.0
  ZCHARN(:) = MIN(0.018,MAX(0.011,0.011+(0.007/8.0)*(ZDDU(:)-10.0)))
ENDIF
!
!       7.6. Roughness lengths Z0 and Z0H over sea
!
IF (KZ0 == 0) THEN      ! ARPEGE formulation
  PZ0SEA (:) = (ZCHARN(:)/XG)*ZUSTAR2(:) + XVZ0CM*PCD(:)/PCDN(:)
  PZ0HSEA(:) = PZ0SEA (:)
ELSEIF (KZ0 == 1) THEN  ! Smith (1988) formulation
  PZ0SEA (:) = (ZCHARN(:)/XG)*ZUSTAR2(:) + 0.11*ZVISA(:)/PUSTAR(:)
  PZ0HSEA(:) = PZ0SEA (:)
ELSEIF (KZ0 == 2) THEN  ! Direct computation using the stability functions
  DO JLON=1,SIZE(PTA)
    PZ0SEA (JLON) = PUREF(JLON)/EXP(XKARMAN*ZDDU(JLON)/PUSTAR(JLON)+ZPSIU(JLON))
    Z0TSEA        = PZREF(JLON)/EXP(XKARMAN*ZDDT(JLON)/ZTSR  (JLON)+ZPSIT(JLON))
    Z0QSEA        = PZREF(JLON)/EXP(XKARMAN*ZDDQ(JLON)/ZQSR  (JLON)+ZPSIT(JLON))
    PZ0HSEA(JLON) = 0.5*(Z0TSEA+Z0QSEA)
  ENDDO
ENDIF
!
IF (DGO%LDIAG_OAFLUX) THEN
  D%XDIAG_USTAR(:)=ZUSR (:)
  D%XDIAG_TSTAR(:)=ZTSR (:)
  D%XDIAG_QSTAR(:)=ZQSR (:)
  D%XDIAG_ZDU  (:)=ZDU  (:)
  D%XDIAG_ZDWG (:)=0.0
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
  D%XDIAG_ZLV  (:)=ZLVA (:)
  D%XDIAG_CDN  (:)=PCDN (:)
  D%XDIAG_CHN  (:)=0.0
  D%XDIAG_CEN  (:)=0.0
ENDIF
!
IF (LHOOK) CALL DR_HOOK('ECUMEV6_FLUX',1,ZHOOK_HANDLE)
!
!-------------------------------------------------------------------------------
   END SUBROUTINE ECUMEV6_FLUX
