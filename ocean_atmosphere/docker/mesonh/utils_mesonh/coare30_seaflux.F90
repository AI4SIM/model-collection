!SFX_LIC Copyright 1994-2014 CNRS, Meteo-France and Universite Paul Sabatier
!SFX_LIC This is part of the SURFEX software governed by the CeCILL-C licence
!SFX_LIC version 1. See LICENSE, CeCILL-C_V1-en.txt and CeCILL-C_V1-fr.txt  
!SFX_LIC for details. version 1.
!     #########
    SUBROUTINE COARE30_SEAFLUX (S,DGO,D,PMASK,KSIZE_WATER,KSIZE_ICE,     &
                                PTA,PEXNA,PRHOA,PSST,PEXNS,PQA,     & 
                                PRAIN,PSNOW,PVMOD,PZREF,PUREF,PPS,  &
                                PQSAT,PSFTH,PSFTQ,PUSTAR,           &
                                PCD,PCDN,PCH,PCE,PRI,PRESA,PZ0HSEA )
!     ##################################################################
!
!
!!****  *COARE30_SEAFLUX*  
!!
!!    PURPOSE
!!    -------
!     
!      Calculate the sea surface fluxes with modified bulk algorithm COARE:  
!
!      Calculates the surface fluxes of heat, moisture, and momentum over
!      sea surface with the simplified COARE 3.0 bulk algorithm from Fairall et al 
!      2003
!   
!      based on water_flux computation for sea ice
!     
!!**  METHOD
!!    ------
!
!!    EXTERNAL
!!    --------
!!
!!    IMPLICIT ARGUMENTS
!!    ------------------ 
!!      
!!    REFERENCE
!!    ---------
!!     
!!    AUTHOR
!!    ------
!!     C. Lebeaupin  *Météo-France* 
!!
!!    MODIFICATIONS
!!    -------------
!!      Original     18/03/2005
!!      B. Decharme     04/2013 : Pack only input variables
!!      S. Senesi       01/2014 : When handling sea ice cover, compute open sea flux, 
!!                                and only where ice cover < 1.
!!      M.N. Bouin     03/2014 possibility of wave parameters from external source
!!      J. Pianezze    04/2022 add diagnostics
!-------------------------------------------------------------------------------
!
!*       0.     DECLARATIONS
!               ------------
!
!
!
USE MODD_SEAFLUX_n, ONLY : SEAFLUX_t
USE MODD_DIAG_n, ONLY : DIAG_t, DIAG_OPTIONS_t
!
USE MODD_SURF_PAR,   ONLY : XUNDEF
!
USE MODI_ICE_SEA_FLUX
USE MODI_COARE30_FLUX
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
TYPE(DIAG_OPTIONS_t), INTENT(INOUT) :: DGO
!
REAL, DIMENSION(:), INTENT(IN)   :: PMASK        ! Either a mask positive for open sea, or a seaice fraction
INTEGER           , INTENT(IN)   :: KSIZE_WATER  ! number of points with some sea water 
INTEGER           , INTENT(IN)   :: KSIZE_ICE    ! number of points with some sea ice
!                                    
REAL, DIMENSION(:), INTENT(IN)    :: PTA   ! air temperature at atm. level (K)
REAL, DIMENSION(:), INTENT(IN)    :: PQA   ! air humidity at atm. level (kg/kg)
REAL, DIMENSION(:), INTENT(IN)    :: PEXNA ! Exner function at atm. level
REAL, DIMENSION(:), INTENT(IN)    :: PRHOA ! air density at atm. level
REAL, DIMENSION(:), INTENT(IN)    :: PVMOD ! module of wind at atm. wind level (m/s)
REAL, DIMENSION(:), INTENT(IN)    :: PZREF ! atm. level for temp. and humidity (m)
REAL, DIMENSION(:), INTENT(IN)    :: PUREF ! atm. level for wind (m)
REAL, DIMENSION(:), INTENT(IN)    :: PSST  ! Sea Surface Temperature (K)
REAL, DIMENSION(:), INTENT(IN)    :: PEXNS ! Exner function at sea surface
REAL, DIMENSION(:), INTENT(IN)    :: PPS   ! air pressure at sea surface (Pa)
REAL, DIMENSION(:), INTENT(IN)    :: PRAIN ! precipitation rate (kg/s/m2)
REAL, DIMENSION(:), INTENT(IN)    :: PSNOW ! snow rate (kg/s/m2)
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
REAL, DIMENSION(:), INTENT(OUT)      :: PCE   !transfer coef. for latent heat flux
REAL, DIMENSION(:), INTENT(OUT)      :: PRI   ! Richardson number
REAL, DIMENSION(:), INTENT(OUT)      :: PRESA ! aerodynamical resistance
REAL, DIMENSION(:), INTENT(OUT)      :: PZ0HSEA ! heat roughness length
!
!*      0.2    declarations of local variables
!
INTEGER, DIMENSION(KSIZE_WATER) :: IR_WATER
INTEGER, DIMENSION(KSIZE_ICE)   :: IR_ICE
INTEGER                         :: J1,J2,JJ
REAL(KIND=JPRB) :: ZHOOK_HANDLE
!
!-------------------------------------------------------------------------------
!
!       1.     Create Masks for ice and water sea
!              ------------------------------------
IF (LHOOK) CALL DR_HOOK('MODI_COARE30_SEAFLUX:COARE30_SEAFLUX',0,ZHOOK_HANDLE)
!
IR_WATER(:)=0
IR_ICE(:)=0
J1=0
J2=0
!
IF (S%LHANDLE_SIC) THEN 
   ! Must compute open sea fluxes even over fully ice-covered sea, which may melt partly
   DO JJ=1,SIZE(PSST(:))
      IR_WATER(JJ)= JJ
   END DO
   ! Do not compute on sea-ice (done in coupling_iceflux)
ELSE
   ! PMASK = XSST -XTTS
   DO JJ=1,SIZE(PSST(:))
      IF (PMASK(JJ) >=0.0 ) THEN
         J1 = J1 + 1
         IR_WATER(J1)= JJ
      ELSE
         J2 = J2 + 1
         IR_ICE(J2)= JJ
      ENDIF
   END DO
ENDIF
!
!-------------------------------------------------------------------------------
!
!       2.      water sea : call to COARE30_FLUX
!              ------------------------------------------------
!
IF (KSIZE_WATER > 0 ) CALL TREAT_SURF(IR_WATER,'W')
!
!-------------------------------------------------------------------------------
!
!       3.      sea ice : call to ICE_SEA_FLUX
!              ------------------------------------
!
IF ( (KSIZE_ICE > 0 ) .AND. (.NOT. S%LHANDLE_SIC) ) CALL TREAT_SURF(IR_ICE,'I')
!
!
IF (LHOOK) CALL DR_HOOK('MODI_COARE30_SEAFLUX:COARE30_SEAFLUX',1,ZHOOK_HANDLE)
!-------------------------------------------------------------------------------
!
CONTAINS
!
SUBROUTINE TREAT_SURF(KMASK,YTYPE)
!
INTEGER, INTENT(IN), DIMENSION(:) :: KMASK
 CHARACTER(LEN=1), INTENT(IN) :: YTYPE
!
REAL, DIMENSION(SIZE(KMASK))      :: ZW_TA   ! air temperature at atm. level (K)
REAL, DIMENSION(SIZE(KMASK))      :: ZW_QA   ! air humidity at atm. level (kg/kg)
REAL, DIMENSION(SIZE(KMASK))      :: ZW_EXNA ! Exner function at atm. level
REAL, DIMENSION(SIZE(KMASK))      :: ZW_RHOA ! air density at atm. level
REAL, DIMENSION(SIZE(KMASK))      :: ZW_VMOD ! module of wind at atm. wind level (m/s)
REAL, DIMENSION(SIZE(KMASK))      :: ZW_ZREF ! atm. level for temp. and humidity (m)
REAL, DIMENSION(SIZE(KMASK))      :: ZW_UREF ! atm. level for wind (m)
REAL, DIMENSION(SIZE(KMASK))      :: ZW_SST  ! Sea Surface Temperature (K)
REAL, DIMENSION(SIZE(KMASK))      :: ZW_HS   ! wave significant height
REAL, DIMENSION(SIZE(KMASK))      :: ZW_TP   ! wave peak period
REAL, DIMENSION(SIZE(KMASK))      :: ZW_EXNS ! Exner function at sea surface
REAL, DIMENSION(SIZE(KMASK))      :: ZW_PS   ! air pressure at sea surface (Pa)
REAL, DIMENSION(SIZE(KMASK))      :: ZW_RAIN !precipitation rate (kg/s/m2)
REAL, DIMENSION(SIZE(KMASK))      :: ZW_SNOW !snow rate (kg/s/m2)
!
REAL, DIMENSION(SIZE(KMASK))      :: ZW_Z0SEA! roughness length over the ocean
!                                                                                 
!  surface fluxes : latent heat, sensible heat, friction fluxes
REAL, DIMENSION(SIZE(KMASK))      :: ZW_SFTH ! heat flux (W/m2)
REAL, DIMENSION(SIZE(KMASK))      :: ZW_SFTQ ! water flux (kg/m2/s)
REAL, DIMENSION(SIZE(KMASK))      :: ZW_USTAR! friction velocity (m/s)
!
! diagnostics
REAL, DIMENSION(SIZE(KMASK))      :: ZW_QSAT ! humidity at saturation
REAL, DIMENSION(SIZE(KMASK))      :: ZW_CD   ! heat drag coefficient
REAL, DIMENSION(SIZE(KMASK))      :: ZW_CDN  ! momentum drag coefficient
REAL, DIMENSION(SIZE(KMASK))      :: ZW_CH   ! neutral momentum drag coefficient
REAL, DIMENSION(SIZE(KMASK))      :: ZW_CE   !transfer coef. for latent heat flux
REAL, DIMENSION(SIZE(KMASK))      :: ZW_RI   ! Richardson number
REAL, DIMENSION(SIZE(KMASK))      :: ZW_RESA ! aerodynamical resistance
REAL, DIMENSION(SIZE(KMASK))      :: ZW_Z0HSEA ! heat roughness length
!
REAL(KIND=JPRB) :: ZHOOK_HANDLE
!
IF (LHOOK) CALL DR_HOOK('COARE30_SEAFLUX:TREAT_SURF',0,ZHOOK_HANDLE)
!
DO JJ=1, SIZE(KMASK)
  ZW_TA(JJ)   = PTA(KMASK(JJ))
  ZW_QA(JJ)   = PQA(KMASK(JJ))
  ZW_EXNA(JJ) = PEXNA(KMASK(JJ))
  ZW_RHOA(JJ) = PRHOA(KMASK(JJ))
  ZW_VMOD(JJ) = PVMOD(KMASK(JJ))
  ZW_ZREF(JJ) = PZREF(KMASK(JJ)) 
  ZW_UREF(JJ) = PUREF(KMASK(JJ))
  ZW_SST(JJ)  = PSST(KMASK(JJ))
  ZW_TP(JJ)   = S%XTP(KMASK(JJ))
  ZW_HS(JJ)   = S%XHS(KMASK(JJ))  
  ZW_EXNS(JJ) = PEXNS(KMASK(JJ)) 
  ZW_PS(JJ)   = PPS(KMASK(JJ))
  ZW_RAIN(JJ) = PRAIN(KMASK(JJ))
  ZW_SNOW(JJ) = PSNOW(KMASK(JJ))
  ZW_Z0SEA(JJ)= S%XZ0(KMASK(JJ))
ENDDO
!  
ZW_SFTH(:)   = XUNDEF
ZW_SFTQ(:)   = XUNDEF
ZW_USTAR(:)  = XUNDEF
ZW_QSAT(:)   = XUNDEF
ZW_CD(:)     = XUNDEF
ZW_CDN(:)    = XUNDEF
ZW_CH(:)     = XUNDEF
ZW_CE(:)     = XUNDEF
ZW_RI(:)     = XUNDEF
ZW_RESA(:)   = XUNDEF
ZW_Z0HSEA(:) = XUNDEF
!
IF (YTYPE=='W') THEN
  !
  CALL COARE30_FLUX(S,DGO,D,ZW_Z0SEA,ZW_TA,ZW_EXNA,ZW_RHOA,ZW_SST,ZW_EXNS,&
        ZW_QA,ZW_VMOD,ZW_ZREF,ZW_UREF,ZW_PS,ZW_QSAT,ZW_SFTH,ZW_SFTQ,ZW_USTAR,&
        ZW_CD,ZW_CDN,ZW_CH,ZW_CE,ZW_RI,ZW_RESA,ZW_RAIN,ZW_Z0HSEA,ZW_HS,ZW_TP)   
  !
ELSEIF ( (YTYPE=='I') .AND. (.NOT. S%LHANDLE_SIC)) THEN
  !
  CALL ICE_SEA_FLUX(ZW_Z0SEA,ZW_TA,ZW_EXNA,ZW_RHOA,ZW_SST,ZW_EXNS,ZW_QA,ZW_RAIN,ZW_SNOW,  &
         ZW_VMOD,ZW_ZREF,ZW_UREF,ZW_PS,ZW_QSAT,ZW_SFTH,ZW_SFTQ,ZW_USTAR,ZW_CD, &
         ZW_CDN,ZW_CH,ZW_RI,ZW_RESA,ZW_Z0HSEA)  
  !
ENDIF
!
DO JJ=1, SIZE(KMASK)
   S%XZ0(KMASK(JJ))  =  ZW_Z0SEA(JJ)
   PSFTH(KMASK(JJ))   =  ZW_SFTH(JJ) 
   PSFTQ(KMASK(JJ))   =  ZW_SFTQ(JJ) 
   PUSTAR(KMASK(JJ))  =  ZW_USTAR(JJ)
   PQSAT(KMASK(JJ))   =  ZW_QSAT(JJ)
   PCD(KMASK(JJ))     =  ZW_CD(JJ) 
   PCDN(KMASK(JJ))    =  ZW_CDN(JJ) 
   PCH(KMASK(JJ))     =  ZW_CH(JJ)
   PCE(KMASK(JJ))     =  ZW_CE(JJ)
   PRI(KMASK(JJ))     =  ZW_RI(JJ) 
   PRESA(KMASK(JJ))   =  ZW_RESA(JJ) 
   PZ0HSEA(KMASK(JJ)) = ZW_Z0HSEA(JJ) 
END DO
IF (LHOOK) CALL DR_HOOK('COARE30_SEAFLUX:TREAT_SURF',1,ZHOOK_HANDLE)
!
END SUBROUTINE TREAT_SURF
! 
END SUBROUTINE COARE30_SEAFLUX
