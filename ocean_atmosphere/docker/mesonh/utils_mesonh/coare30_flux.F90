!SFX_LIC Copyright 1994-2014 CNRS, Meteo-France and Universite Paul Sabatier
!SFX_LIC This is part of the SURFEX software governed by the CeCILL-C licence
!SFX_LIC version 1. See LICENSE, CeCILL-C_V1-en.txt and CeCILL-C_V1-fr.txt  
!SFX_LIC for details. version 1.
!     #########
SUBROUTINE COARE30_FLUX (S,DGO,D,PZ0SEA,PTA,PEXNA,PRHOA,PSST,PEXNS,PQA,  &
            PVMOD,PZREF,PUREF,PPS,PQSAT,PSFTH,PSFTQ,PUSTAR,PCD,PCDN,PCH,PCE,PRI,&
            PRESA,PRAIN,PZ0HSEA,PHS,PTP)  
!     #######################################################################
!
!
!!****  *COARE25_FLUX*  
!!
!!    PURPOSE
!!    -------
!      Calculate the surface fluxes of heat, moisture, and momentum over
!      sea surface with bulk algorithm COARE3.0. 
!     
!!**  METHOD
!!    ------
!      transfer coefficients were obtained using a dataset which combined COARE
!      data with those from three other ETL field experiments, and reanalysis of
!      the HEXMAX data (DeCosmos et al. 1996). 
!      ITERMAX=3 
!      Take account of the surface gravity waves on the velocity roughness and 
!      hence the momentum transfer coefficient
!        NGRVWAVES=0 no gravity waves action (Charnock) !default value
!        NGRVWAVES=1 wave age parameterization of Oost et al. 2002
!        NGRVWAVES=2 model of Taylor and Yelland 2001
!
!!    EXTERNAL
!!    --------
!!
!!    IMPLICIT ARGUMENTS
!!    ------------------ 
!!      
!!    REFERENCE
!!    ---------
!!      Fairall et al (2003), J. of Climate, vol. 16, 571-591
!!      Fairall et al (1996), JGR, 3747-3764
!!      Gosnell et al (1995), JGR, 437-442
!!      Fairall et al (1996), JGR, 1295-1308
!!      
!!    AUTHOR
!!    ------
!!     C. Lebeaupin  *Météo-France* (adapted from C. Fairall's code)
!!
!!    MODIFICATIONS
!!    -------------
!!      Original     1/06/2006
!!      B. Decharme    06/2009 limitation of Ri
!!      B. Decharme    09/2012 Bug in Ri calculation and limitation of Ri in surface_ri.F90
!!      B. Decharme    06/2013 bug in z0 (output) computation
!!      M.N. Bouin     03/2014 possibility of wave parameters from external source
!!      C. Lebeaupin   03/2014 bug if PTA=PSST and PEXNA=PEXNS: set a minimum value
!!	   	       	       add abort if no convergence
!!      C. Lebeaupin   06/2014 itermax=10 for low wind conditions (ZVMOD<=1)
!!      J. Pianezze    11/2014 add coupling wave parameters 
!!      J. Pianezze    04/2022 add diagnostics
!-------------------------------------------------------------------------------
!
!*       0.     DECLARATIONS
!               ------------
!
!
USE MODD_SEAFLUX_n, ONLY : SEAFLUX_t
USE MODD_DIAG_n,    ONLY : DIAG_t, DIAG_OPTIONS_t
!
USE MODD_CSTS,       ONLY : XKARMAN, XG, XSTEFAN, XRD, XRV, XPI, &
                            XLVTT, XCL, XCPD, XCPV, XRHOLW, XTT, &
                            XP00
USE MODD_SURF_ATM,   ONLY : XVZ0CM
!
USE MODD_SFX_OASIS,  ONLY : LCPL_WAVE
!
USE MODD_SURF_PAR,   ONLY : XUNDEF, XSURF_EPSILON
USE MODD_WATER_PAR
!
USE MODI_SURFACE_RI
USE MODI_WIND_THRESHOLD
USE MODE_COARE30_PSI
!
USE MODE_THERMOS
!
!
USE MODI_ABOR1_SFX
!
!
USE YOMHOOK   ,ONLY : LHOOK,   DR_HOOK
USE PARKIND1  ,ONLY : JPRB
!
IMPLICIT NONE
!
!*      0.1    declarations of arguments
!
!
!
TYPE(SEAFLUX_t),      INTENT(INOUT) :: S
TYPE(DIAG_t),         INTENT(INOUT) :: D
TYPE(DIAG_OPTIONS_t), INTENT(INOUT) :: DGO
!
REAL, DIMENSION(:), INTENT(IN)       :: PTA   ! air temperature at atm. level (K)
REAL, DIMENSION(:), INTENT(IN)       :: PQA   ! air humidity at atm. level (kg/kg)
REAL, DIMENSION(:), INTENT(IN)       :: PEXNA ! Exner function at atm. level
REAL, DIMENSION(:), INTENT(IN)       :: PRHOA ! air density at atm. level
REAL, DIMENSION(:), INTENT(IN)       :: PVMOD ! module of wind at atm. wind level (m/s)
REAL, DIMENSION(:), INTENT(IN)       :: PZREF ! atm. level for temp. and humidity (m)
REAL, DIMENSION(:), INTENT(IN)       :: PUREF ! atm. level for wind (m)
REAL, DIMENSION(:), INTENT(IN)       :: PSST  ! Sea Surface Temperature (K)
REAL, DIMENSION(:), INTENT(IN)       :: PEXNS ! Exner function at sea surface
REAL, DIMENSION(:), INTENT(IN)       :: PPS   ! air pressure at sea surface (Pa)
REAL, DIMENSION(:), INTENT(IN)       :: PRAIN !precipitation rate (kg/s/m2)
REAL, DIMENSION(:), INTENT(IN)       :: PHS   ! wave significant height
REAL, DIMENSION(:), INTENT(IN)       :: PTP   ! wave peak period
!
REAL, DIMENSION(:), INTENT(INOUT)    :: PZ0SEA! roughness length over the ocean
!                                                                                 
!  surface fluxes : latent heat, sensible heat, friction fluxes
REAL, DIMENSION(:), INTENT(OUT)      :: PSFTH ! heat flux (W/m2)
REAL, DIMENSION(:), INTENT(OUT)      :: PSFTQ ! water flux (kg/m2/s)
REAL, DIMENSION(:), INTENT(OUT)      :: PUSTAR! friction velocity (m/s)
!
! diagnostics
REAL, DIMENSION(:), INTENT(OUT)      :: PQSAT ! humidity at saturation
REAL, DIMENSION(:), INTENT(OUT)      :: PCD   ! heat drag coefficient
REAL, DIMENSION(:), INTENT(OUT)      :: PCDN  ! momentum drag coefficient
REAL, DIMENSION(:), INTENT(OUT)      :: PCH   ! neutral momentum drag coefficient
REAL, DIMENSION(:), INTENT(OUT)      :: PCE  !transfer coef. for latent heat flux
REAL, DIMENSION(:), INTENT(OUT)      :: PRI   ! Richardson number
REAL, DIMENSION(:), INTENT(OUT)      :: PRESA ! aerodynamical resistance
REAL, DIMENSION(:), INTENT(OUT)      :: PZ0HSEA ! heat roughness length
!
!
!*      0.2    declarations of local variables
!
REAL, DIMENSION(SIZE(PTA))      :: ZVMOD    ! wind intensity
REAL, DIMENSION(SIZE(PTA))      :: ZPA      ! Pressure at atm. level
REAL, DIMENSION(SIZE(PTA))      :: ZTA      ! Temperature at atm. level
REAL, DIMENSION(SIZE(PTA))      :: ZQASAT   ! specific humidity at saturation  at atm. level (kg/kg)
!
REAL, DIMENSION(SIZE(PTA))      :: ZO       ! rougness length ref 
REAL, DIMENSION(SIZE(PTA))      :: ZWG      ! gustiness factor (m/s)
!
REAL, DIMENSION(SIZE(PTA))      :: ZDU,ZDT,ZDQ,ZDUWG !differences
!
REAL, DIMENSION(SIZE(PTA))      :: ZUSR        !velocity scaling parameter "ustar" (m/s) = friction velocity
REAL, DIMENSION(SIZE(PTA))      :: ZTSR        !temperature sacling parameter "tstar" (degC)
REAL, DIMENSION(SIZE(PTA))      :: ZQSR        !humidity scaling parameter "qstar" (kg/kg)
!
REAL, DIMENSION(SIZE(PTA))      :: ZU10,ZT10   !vertical profils (10-m height) 
REAL, DIMENSION(SIZE(PTA))      :: ZVISA       !kinematic viscosity of dry air
REAL, DIMENSION(SIZE(PTA))      :: ZO10,ZOT10  !roughness length at 10m
REAL, DIMENSION(SIZE(PTA))      :: ZCD,ZCT,ZCC
REAL, DIMENSION(SIZE(PTA))      :: ZCD10,ZCT10 !transfer coef. at 10m
REAL, DIMENSION(SIZE(PTA))      :: ZRIBU,ZRIBCU
REAL, DIMENSION(SIZE(PTA))      :: ZETU,ZL10
!
REAL, DIMENSION(SIZE(PTA))      :: ZCHARN                      !Charnock number depends on wind module
REAL, DIMENSION(SIZE(PTA))      :: ZTWAVE,ZHWAVE,ZCWAVE,ZLWAVE !to compute gravity waves' impact
!
REAL, DIMENSION(SIZE(PTA))      :: ZZL,ZZTL!,ZZQL    !Obukhovs stability 
                                                     !param. z/l for u,T,q
REAL, DIMENSION(SIZE(PTA))      :: ZRR
REAL, DIMENSION(SIZE(PTA))      :: ZOT,ZOQ           !rougness length ref
REAL, DIMENSION(SIZE(PTA))      :: ZPUZ,ZPTZ,ZPQZ    !PHI funct. for u,T,q 
!
REAL, DIMENSION(SIZE(PTA))      :: ZBF               !constants to compute gustiness factor
!
REAL, DIMENSION(SIZE(PTA))      :: ZTAU       !momentum flux (W/m2)
REAL, DIMENSION(SIZE(PTA))      :: ZHF        !sensible heat flux (W/m2)
REAL, DIMENSION(SIZE(PTA))      :: ZEF        !latent heat flux (W/m2)
REAL, DIMENSION(SIZE(PTA))      :: ZWBAR      !diag for webb correction but not used here after
REAL, DIMENSION(SIZE(PTA))      :: ZTAUR      !momentum flux due to rain (W/m2)
REAL, DIMENSION(SIZE(PTA))      :: ZRF        !sensible heat flux due to rain (W/m2)
REAL, DIMENSION(SIZE(PTA))      :: ZCHN,ZCEN  !neutral coef. for heat and vapor
!
REAL, DIMENSION(SIZE(PTA))      :: ZLV      !latent heat constant
!
REAL, DIMENSION(SIZE(PTA))      :: ZTAC,ZDQSDT,ZDTMP,ZDWAT,ZALFAC ! for precipitation impact
REAL, DIMENSION(SIZE(PTA))      :: ZXLR                           ! vaporisation  heat  at a given temperature
REAL, DIMENSION(SIZE(PTA))      :: ZCPLW                          ! specific heat for water at a given temperature 
!
REAL, DIMENSION(SIZE(PTA))      :: ZUSTAR2  ! square of friction velocity
!
REAL, DIMENSION(SIZE(PTA))      :: ZDIRCOSZW! orography slope cosine (=1 on water!)
REAL, DIMENSION(SIZE(PTA))      :: ZAC      ! Aerodynamical conductance
!
!
INTEGER, DIMENSION(SIZE(PTA))   :: ITERMAX             ! maximum number of iterations
!
REAL    :: ZRVSRDM1,ZRDSRV,ZR2 ! thermodynamic constants
REAL    :: ZBETAGUST           !gustiness factor
REAL    :: ZZBL                !atm. boundary layer depth (m)
REAL    :: ZVISW               !m2/s kinematic viscosity of water
REAL    :: ZS                  !height of rougness length ref
REAL    :: ZCH10               !transfer coef. at 10m
!
INTEGER :: J, JLOOP    !loop indice
REAL(KIND=JPRB) :: ZHOOK_HANDLE
!
!-------------------------------------------------------------------------------
!
!       1.     Initializations
!              ---------------
!
!       1.1   Constants and parameters
!
IF (LHOOK) CALL DR_HOOK('COARE30_FLUX',0,ZHOOK_HANDLE)
!
ZRVSRDM1  = XRV/XRD-1. ! 0.607766
ZRDSRV    = XRD/XRV    ! 0.62198
ZR2       = 1.-ZRDSRV  ! pas utilisé dans cette routine
ZBETAGUST = 1.2        ! value based on TOGA-COARE experiment
ZZBL      = 600.       ! Set a default value for boundary layer depth
ZS        = 10.        ! Standard heigth =10m
ZCH10     = 0.00115
!
ZVISW     = 1.E-6
!
!       1.2   Array initialization by undefined values
!
PSFTH (:)=XUNDEF
PSFTQ (:)=XUNDEF
PUSTAR(:)=XUNDEF
!
PCD(:) = XUNDEF
PCDN(:) = XUNDEF
PCH(:) = XUNDEF
PCE(:) =XUNDEF
PRI(:) = XUNDEF
!
PRESA(:)=XUNDEF
!
!-------------------------------------------------------------------------------
!       2. INITIAL GUESS FOR THE ITERATIVE METHOD 
!          -------------------------------------
!
!       2.0     Temperature 
!
! Set a non-zero value for the temperature gradient
!
WHERE((PTA(:)*PEXNS(:)/PEXNA(:)-PSST(:))==0.) 
      ZTA(:)=PTA(:)-1E-3
ELSEWHERE
      ZTA(:)=PTA(:)      
ENDWHERE

!       2.1     Wind and humidity 
!
! Sea surface specific humidity 
!
PQSAT(:)=QSAT_SEAWATER(PSST(:),PPS(:))         
!              
! Set a minimum value to wind 
!
ZVMOD(:) = WIND_THRESHOLD(PVMOD(:),PUREF(:))
!
! Specific humidity at saturation at the atm. level 
!
ZPA(:) = XP00* (PEXNA(:)**(XCPD/XRD))
ZQASAT(:) = QSAT(ZTA(:),ZPA(:)) 
!
!
ZO(:)  = 0.0001
ZWG(:) = 0.
IF (S%LPWG) ZWG(:) = 0.5
!
ZCHARN(:) = 0.011  
!
DO J=1,SIZE(PTA)
  !
  !      2.2       initial guess
  !    
  ZDU(J) = ZVMOD(J)   !wind speed difference with surface current(=0) (m/s)
                      !initial guess for gustiness factor
  ZDT(J) = -(ZTA(J)/PEXNA(J)) + (PSST(J)/PEXNS(J)) !potential temperature difference
  ZDQ(J) = PQSAT(J)-PQA(J)                         !specific humidity difference
  !
  ZDUWG(J) = SQRT(ZDU(J)**2+ZWG(J)**2)     !wind speed difference including gustiness ZWG
  !
  !      2.3   initialization of neutral coefficients
  !
  ZU10(J)  = ZDUWG(J)*LOG(ZS/ZO(J))/LOG(PUREF(J)/ZO(J))
  ZUSR(J)  = 0.035*ZU10(J)
  ZVISA(J) = 1.326E-5*(1.+6.542E-3*(ZTA(J)-XTT)+&
             8.301E-6*(ZTA(J)-XTT)**2-4.84E-9*(ZTA(J)-XTT)**3) !Andrea (1989) CRREL Rep. 89-11
  ! 
  ZO10(J) = ZCHARN(J)*ZUSR(J)*ZUSR(J)/XG+0.11*ZVISA(J)/ZUSR(J)
  ZCD(J)  = (XKARMAN/LOG(PUREF(J)/ZO10(J)))**2  !drag coefficient
  ZCD10(J)= (XKARMAN/LOG(ZS/ZO10(J)))**2
  ZCT10(J)= ZCH10/SQRT(ZCD10(J))
  ZOT10(J)= ZS/EXP(XKARMAN/ZCT10(J))
  !
  !-------------------------------------------------------------------------------
  !             Grachev and Fairall (JAM, 1997)
  ZCT(J) = XKARMAN/LOG(PZREF(J)/ZOT10(J))      !temperature transfer coefficient
  ZCC(J) = XKARMAN*ZCT(J)/ZCD(J)               !z/L vs Rib linear coef.
  !
  ZRIBCU(J) = -PUREF(J)/(ZZBL*0.004*ZBETAGUST**3) !saturation or plateau Rib
  !ZRIBU(J) =-XG*PUREF(J)*(ZDT(J)+ZRVSRDM1*(ZTA(J)-XTT)*ZDQ)/&
  !     &((ZTA(J)-XTT)*ZDUWG(J)**2)
  ZRIBU(J)  = -XG*PUREF(J)*(ZDT(J)+ZRVSRDM1*ZTA(J)*ZDQ(J))/&
               (ZTA(J)*ZDUWG(J)**2)  
  !
  IF (ZRIBU(J)<0.) THEN
    ZETU(J) = ZCC(J)*ZRIBU(J)/(1.+ZRIBU(J)/ZRIBCU(J))    !Unstable G and F
  ELSE
    ZETU(J) = ZCC(J)*ZRIBU(J)/(1.+27./9.*ZRIBU(J)/ZCC(J))!Stable 
  ENDIF
  !
  ZL10(J) = PUREF(J)/ZETU(J) !MO length
  !
ENDDO
!
!  First guess M-O stability dependent scaling params. (u*,T*,q*) to estimate ZO and z/L (ZZL)
ZUSR(:) = ZDUWG(:)*XKARMAN/(LOG(PUREF(:)/ZO10(:))-PSIFCTU(PUREF(:)/ZL10(:)))
ZTSR(:) = -ZDT(:)*XKARMAN/(LOG(PZREF(:)/ZOT10(:))-PSIFCTT(PZREF(:)/ZL10(:)))
ZQSR(:) = -ZDQ(:)*XKARMAN/(LOG(PZREF(:)/ZOT10(:))-PSIFCTT(PZREF(:)/ZL10(:)))
!
IF (LCPL_WAVE .AND. .NOT. (ANY(S%XCHARN==0.0)) ) THEN
  ZCHARN(:) = S%XCHARN(:)
ELSE
  ZCHARN(:) = 0.011
END IF
!
ZZL(:) = 0.0
!
DO J=1,SIZE(PTA)
  !
  IF (ZETU(J)>50.) THEN
    ITERMAX(J) = 1
  ELSE
    ITERMAX(J) = 3 !number of iterations
  ENDIF
  IF (ZVMOD(J)<=1.) THEN
    ITERMAX(J) = 10
  ENDIF
  !
  IF (.NOT.LCPL_WAVE) THEN
    !then modify Charnork for high wind speeds Chris Fairall's data
    IF (ZDUWG(J)>10.) ZCHARN(J) = 0.011 + (0.018-0.011)*(ZDUWG(J)-10.)/(18-10)
    IF (ZDUWG(J)>18.) ZCHARN(J) = 0.018
  END IF
  !
  !                3.  ITERATIVE LOOP TO COMPUTE USR, TSR, QSR 
  !                -------------------------------------------
  !
  IF (S%LWAVEWIND .AND. .NOT. LCPL_WAVE) THEN
    ZHWAVE(J) = 0.018*PVMOD(J)*PVMOD(J)*(1.+0.015*PVMOD(J))
    ZTWAVE(J) = 0.729*PVMOD(J)
  ELSE 
    ZHWAVE(J) = PHS(J)
    ZTWAVE(J) = PTP(J)
    ! to avoid the nullity of HS and TP 
    IF (ZHWAVE(J) .EQ. 0.0) ZHWAVE(J) = 0.018*PVMOD(J)*PVMOD(J)*(1.+0.015*PVMOD(J))
    IF (ZTWAVE(J) .EQ. 0.0) ZTWAVE(J) = 0.729*PVMOD(J)
  ENDIF 
!
  ZCWAVE(J) = XG*ZTWAVE(J)/(2.*XPI)
  ZLWAVE(J) = ZTWAVE(J)*ZCWAVE(J)
  !
ENDDO
!
   
!
DO JLOOP=1,MAXVAL(ITERMAX) ! begin of iterative loop
  !
  DO J=1,SIZE(PTA)
    !
    IF (JLOOP.GT.ITERMAX(J)) CYCLE
    !
    IF (S%NGRVWAVES==0) THEN
      ZO(J) = ZCHARN(J)*ZUSR(J)*ZUSR(J)/XG + 0.11*ZVISA(J)/ZUSR(J) !Smith 1988
    ELSE IF (S%NGRVWAVES==1) THEN
      ZO(J) = (50./(2.*XPI))*ZLWAVE(J)*(ZUSR(J)/ZCWAVE(J))**4.5 &
              + 0.11*ZVISA(J)/ZUSR(J)                       !Oost et al. 2002  
    ELSE IF (S%NGRVWAVES==2) THEN
      ZO(J) = 1200.*ZHWAVE(J)*(ZHWAVE(J)/ZLWAVE(J))**4.5 &
              + 0.11*ZVISA(J)/ZUSR(J)                       !Taulor and Yelland 2001  
    ENDIF
    !
    ZRR(J) = ZO(J)*ZUSR(J)/ZVISA(J)
    ZOQ(J) = MIN(1.15E-4 , 5.5E-5/ZRR(J)**0.6)
    ZOT(J) = ZOQ(J)
    !
    ZZL(J) = XKARMAN * XG * PUREF(J) * &
              ( ZTSR(J)*(1.+ZRVSRDM1*PQA(J)) + ZRVSRDM1*ZTA(J)*ZQSR(J) ) / &
              ( ZTA(J)*ZUSR(J)*ZUSR(J)*(1.+ZRVSRDM1*PQA(J)) )  
    ZZTL(J)= ZZL(J)*PZREF(J)/PUREF(J)  ! for T 
!    ZZQL(J)=ZZL(J)*PZREF(J)/PUREF(J)  ! for Q
  ENDDO
  !
  ZPUZ(:) = PSIFCTU(ZZL(:))     
  ZPTZ(:) = PSIFCTT(ZZTL(:))
  !
  DO J=1,SIZE(PTA)
    !
    ! ZPQZ(J)=PSIFCTT(ZZQL(J))    
    ZPQZ(J) = ZPTZ(J)
    !
    !             3.1 scale parameters
    !
    ZUSR(J) = ZDUWG(J)*XKARMAN/(LOG(PUREF(J)/ZO(J)) -ZPUZ(J))
    ZTSR(J) = -ZDT(J)  *XKARMAN/(LOG(PZREF(J)/ZOT(J))-ZPTZ(J))
    ZQSR(J) = -ZDQ(J)  *XKARMAN/(LOG(PZREF(J)/ZOQ(J))-ZPQZ(J))
    !
    !             3.2 Gustiness factor (ZWG)
    !
    IF(S%LPWG) THEN
      ZBF(J) = -XG/ZTA(J)*ZUSR(J)*(ZTSR(J)+ZRVSRDM1*ZTA(J)*ZQSR(J))
      IF (ZBF(J)>0.) THEN
        ZWG(J) = ZBETAGUST*(ZBF(J)*ZZBL)**(1./3.)
      ELSE
        ZWG(J) = 0.2
      ENDIF
    ENDIF  
    ZDUWG(J) = SQRT(ZVMOD(J)**2 + ZWG(J)**2)
    !
  ENDDO
  !
ENDDO
!-------------------------------------------------------------------------------
!
!            4.  COMPUTE transfer coefficients PCD, PCH, ZCE and SURFACE FLUXES
!                --------------------------------------------------------------
!
ZTAU(:) = XUNDEF
ZHF(:)  = XUNDEF
ZEF(:)  = XUNDEF
!
ZWBAR(:) = 0.
ZTAUR(:) = 0.
ZRF(:)   = 0.
!
DO J=1,SIZE(PTA)
  !
  !
  !            4. transfert coefficients PCD, PCH and PCE 
  !                 and neutral PCDN, ZCHN, ZCEN 
  !
  PCD(J) = (ZUSR(J)/ZDUWG(J))**2.
  PCH(J) = ZUSR(J)*ZTSR(J)/(ZDUWG(J)*(ZTA(J)*PEXNS(J)/PEXNA(J)-PSST(J)))
  PCE(J) = ZUSR(J)*ZQSR(J)/(ZDUWG(J)*(PQA(J)-PQSAT(J)))
  !
  PCDN(J) = (XKARMAN/LOG(ZS/ZO(J)))**2.
  ZCHN(J) = (XKARMAN/LOG(ZS/ZO(J)))*(XKARMAN/LOG(ZS/ZOT(J)))
  ZCEN(J) = (XKARMAN/LOG(ZS/ZO(J)))*(XKARMAN/LOG(ZS/ZOQ(J)))
  !
  ZLV(J) = XLVTT + (XCPV-XCL)*(PSST(J)-XTT)
  !
  !            4. 2 surface fluxes 
  !
  IF (ABS(PCDN(J))>1.E-2) THEN   !!!! secure COARE3.0 CODE 
    write(*,*) 'pb PCDN in COARE30: ',PCDN(J)
    write(*,*) 'point: ',J,"/",SIZE(PTA)
    write(*,*) 'roughness: ', ZO(J)
    write(*,*) 'ustar: ',ZUSR(J)
    write(*,*) 'wind: ',ZDUWG(J)
    CALL ABOR1_SFX('COARE30: PCDN too large -> no convergence')
  ELSE
    ZTSR(J) = -ZTSR(J)
    ZQSR(J) = -ZQSR(J)
    ZTAU(J) = -PRHOA(J)*ZUSR(J)*ZUSR(J)*ZVMOD(J)/ZDUWG(J)
    ZHF(J)  =  PRHOA(J)*XCPD*ZUSR(J)*ZTSR(J)
    ZEF(J)  =  PRHOA(J)*ZLV(J)*ZUSR(J)*ZQSR(J)
    !    
    !           4.3 Contributions to surface  fluxes due to rainfall
    !
    ! SB: a priori, le facteur ZRDSRV=XRD/XRV est introduit pour
    !     adapter la formule de Clausius-Clapeyron (pour l'air
    !     sec) au cas humide.
    IF (S%LPRECIP) THEN
      ! 
      ! heat surface  fluxes
      !
      ZTAC(J)  = ZTA(J)-XTT
      !
      ZXLR(J)  = XLVTT + (XCPV-XCL)* ZTAC(J)                            ! latent heat of rain vaporization
      ZDQSDT(J)= ZQASAT(J) * ZXLR(J) / (XRD*ZTA(J)**2)                  ! Clausius-Clapeyron relation
      ZDTMP(J) = (1.0 + 3.309e-3*ZTAC(J) -1.44e-6*ZTAC(J)*ZTAC(J)) * &  !heat diffusivity
                  0.02411 / (PRHOA(J)*XCPD)
      !
      ZDWAT(J) = 2.11e-5 * (XP00/ZPA(J)) * (ZTA(J)/XTT)**1.94           ! water vapour diffusivity from eq (13.3)
      !                                                                 ! of Pruppacher and Klett (1978)      
      ZALFAC(J)= 1.0 / (1.0 + &                                         ! Eq.11 in GoF95
                   ZRDSRV*ZDQSDT(J)*ZXLR(J)*ZDWAT(J)/(ZDTMP(J)*XCPD))   ! ZALFAC=wet-bulb factor (sans dim)     
      ZCPLW(J) = 4224.8482 + ZTAC(J) * &
                              ( -4.707 + ZTAC(J) * &
                                (0.08499 + ZTAC(J) * &
                                  (1.2826e-3 + ZTAC(J) * &
                                    (4.7884e-5 - 2.0027e-6* ZTAC(J))))) ! specific heat  
      !       
      ZRF(J)   = PRAIN(J) * ZCPLW(J) * ZALFAC(J) * &                    !Eq.12 in GoF95 !SIGNE?
                   (PSST(J) - ZTA(J) + (PQSAT(J)-PQA(J))*ZXLR(J)/XCPD )
      !
      ! Momentum flux due to rainfall  
      !
      ZTAUR(J)=-0.85*(PRAIN(J) *ZVMOD(J)) !pp3752 in FBR96
      !
    ENDIF
    !
    !             4.4   Webb correction to latent heat flux
    ! 
    ZWBAR(J)=- (1./ZRDSRV)*ZUSR(J)*ZQSR(J) / (1.0+(1./ZRDSRV)*PQA(J)) &
               - ZUSR(J)*ZTSR(J)/ZTA(J)                        ! Eq.21*rhoa in FBR96    
    !
    !             4.5   friction velocity which contains correction du to rain            
    !
    ZUSTAR2(J)= - (ZTAU(J) + ZTAUR(J)) / PRHOA(J)
    PUSTAR(J) =  SQRT(ZUSTAR2(J))
    !
    !             4.6   Total surface fluxes
    !           
    PSFTH (J) =  ZHF(J) + ZRF(J)
    PSFTQ (J) =  ZEF(J) / ZLV(J)
    ! 
  ENDIF
ENDDO                      
!-------------------------------------------------------------------------------
!
!       5.  FINAL STEP : TOTAL SURFACE FLUXES AND DERIVED DIAGNOSTICS 
!           -----------
!       5.1    Richardson number
!             
!
ZDIRCOSZW(:) = 1.
 CALL SURFACE_RI(PSST,PQSAT,PEXNS,PEXNA,ZTA,ZQASAT,&
                PZREF,PUREF,ZDIRCOSZW,PVMOD,PRI   )  
!
!       5.2     Aerodynamical conductance and resistance
!             
ZAC(:) = PCH(:)*ZVMOD(:)
PRESA(:) = 1. / MAX(ZAC(:),XSURF_EPSILON)
!
!       5.3 Z0 and Z0H over sea
!
PZ0SEA(:) =  ZCHARN(:) * ZUSTAR2(:) / XG + XVZ0CM * PCD(:) / PCDN(:)
!
PZ0SEA(:) = MAX(MIN(ZO(:),0.05),10E-6)
!
PZ0HSEA(:) = PZ0SEA(:)
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
  D%XDIAG_ZPUZ (:)=ZPUZ (:)
  D%XDIAG_ZPTZ (:)=ZPTZ (:)
  D%XDIAG_ZPQZ (:)=ZPQZ (:)
  D%XDIAG_PRHOA(:)=PRHOA(:)
  D%XDIAG_ZLV  (:)=ZLV  (:)
  D%XDIAG_CDN  (:)=PCDN (:)
  D%XDIAG_CHN  (:)=ZCHN (:)
  D%XDIAG_CEN  (:)=ZCEN (:)
ENDIF
!
IF (LHOOK) CALL DR_HOOK('COARE30_FLUX',1,ZHOOK_HANDLE)
!
!-------------------------------------------------------------------------------
!
END SUBROUTINE COARE30_FLUX
