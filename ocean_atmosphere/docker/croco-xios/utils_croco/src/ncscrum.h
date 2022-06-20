! $Id: ncscrum.h 1588 2014-08-04 16:26:01Z marchesiello $
!
!======================================================================
! CROCO is a branch of ROMS developped at IRD and INRIA, in France
! The two other branches from UCLA (Shchepetkin et al)
! and Rutgers University (Arango et al) are under MIT/X style license.
! CROCO specific routines (nesting) are under CeCILL-C license.
!
! CROCO website : http://www.croco-ocean.org
!======================================================================
!
! This is include file "ncscrum.h".
! ==== == ======= ==== ============
!
!===================================================================
! indices in character array "vname", which holds variable names
!                                                and attributes.
! indxTime        time
! indxZ           free-surface
! indxUb,indxVb   vertically integrated 2D U,V-momentum components
!
! indxU,indxV     3D U- and V-momenta.
! indxT,indxS,.., indxZoo  tracers (temperature, salinity,
!                 biological tracers.
! indxSAND,indxGRAV...
! indxMUD         gravel,sand & mud sediment tracers
! indxO,indeW     omega vertical mass flux and true vertical velocity
! indxR           density anomaly
! indxbvf         Brunt Vaisala Frequency
! indxAOU  	  Apparent Oxygen Utilization
! indxWIND10      surface wind speed 10 m
! indxpCO2        partial pressure of CO2 in the ocean
! indxVisc        Horizontal viscosity coefficients
! indxDiff        Horizontal diffusivity coefficients
! indxAkv,indxAkt,indxAks  vertical viscosity/diffusivity coefficients
! indxAkk,indxAkp vertical diffusion coefficients for TKE and GLS
! indxHbl         depth of planetary boundary layer in KPP model
! indxHbbl        depth of bottom planetary boundary layer in KPP model
! indxHel         depth of euphotic layer
! indxChC         Chlorophyll/Carbon ratio
! indxTke         Turbulent kinetic energy
! indxGls         Generic length scale
! indxLsc         vertical mixing length scale
! indxHm          time evolving bathymetry
!
! indxSSH         observed sea surface height (from climatology)
! indxSUSTR,indxSVSTR  surface U-, V-momentum stress (wind forcing)
! indxBustr,indxBvstr  bottom  U-, V-momentum stress
! indxShflx       net surface heat flux.
! indxShflx_rsw   shortwave radiation flux
! indxSwflx       surface fresh water flux
! indxSST         sea surface temperature
! indxdQdSST      Q-correction coefficient dQdSST
! indxSSS         sea surface salinity
! indxQBAR        river runoff
! indxBhflx       bottom hydrothermal heat flux
! indxBwflx       bottom hydrothermal freshwater flux
!
! indxAi          fraction of cell covered by ice
! indxUi,indxVi   U,V-components of sea ice velocity
! indxHi,indxHS   depth of ice cover and depth of snow cover
! indxTIsrf       temperature of ice surface
!
! ** SEDIMENT (USGS model) **
! indxBSD,indxBSS bottom sediment grain Density and Size
!                 to be read from file if(!defined ANA_BSEDIM,
!                 && !defined SEDIMENT)
!
! indxBTHK,       sediment bed thickness, porosity, size class fractions
! indxBPOR,indxBFRA
!
! ** WAVE input data to be read from file if !defined WKB_WWAVE or OW_COUPLING
! indxWWA          wind induced wave Amplitude
! indxWWD          wind induced wave Direction
! indxWWP          wind induced wave Period
! indxWEB          wave dissipation by breaking,
! indxWED          wave dissipation by friction,
! indxWER          wave dissipation by roller breaking,
!
! ** WAVE history if WKB model or OW COUPLING or WAVE OFFLINE **
! indxHRM,indxFRQ  RMS wave height and frequency
! indxWKX,indxWKE  XI/ETA-dir wavenumber
! indxEPB,indxEPD  wave breaking and frictional dissipation
! indxWAC,indxWAR  wave action and roller action density
! indxEPR          wave roller dissipation
!
! ** MRL_WCI **
! indxSUP          wave set-up
! indxUST2D        vertically integrated stokes velocity in xi  direction
! indxVST2D        vertically integrated stokes velocity in eta direction
! indxUST          stokes velocity in xi  direction
! indxVST          stokes velocity in eta direction
! indxWST          vertical stokes velocity
! indxAkb          eddy viscosity  due to wave breaking
! indxAkw          eddy difusivity due to wave breaking
! indxKVF          vortex force
! indxCALP         surface pressure correction
! indxKAPS         Bernoulli head
!
! ** DIAGNOSTICS_UV **
!  indxMXadv,indxMYadv,indxMVadv : xi-, eta-, and s- advection terms
!  indxMCor                      : Coriolis term,
!  indxMPrsgrd                   : Pressure gradient force term
!  indxMHmix, indxMVmix          : horizontal and vertical mixinig terms
!  indxMHdiff                    : horizontal diffusion term (implicit)
!  indxMrate                     : tendency term
!  indxMBaro                     : Barotropic coupling term
!  indxMfast                     : Fast term
!  indxMBtcr                     : forth-order truncation error
!  indxMswd, indxMbdr            : surface wind & bed shear stresses (m2/s2)
!  indxMvf, indxMbrk             : vortex force & breaking body force terms
!  indxMStCo                     : Stokes-Coriolis terms
!  indxMVvf                      : vertical vortex force terms (in prsgrd.F)
!  indxMPrscrt                   : pressure correction terms (in prsgrd.F)
!  indxMsbk, indxMbwf            : surface breaking & bed wave friction (m2/s2)
!  indxMfrc                      : near-bed frictional wave streaming as body force (m2/s2)
!
! ** DIAGNOSTICS_TS **
!  indxTXadv,indxTYadv,indxTVadv : xi-, eta-, and s- advection terms
!  indxTHmix,indxTVmix           : horizontal and vertical mixinig terms
!  indxTbody                     : body force term
!  indxTrate                     : tendency term
!
! ** DIAGNOSTICS_VRT **
!  indxvrtXadv,indxvrtYadv       : xi-, eta- advection terms
!  indxvrtHdiff                  : horizontal diffusion term (implicit)
!  indxvrtCor                    : Coriolis term,
!  indxvrtPrsgrd                 : Pressure gradient force term
!  indxvrtHmix, indxvrtVmix      : horizontal and vertical mixing terms
!  indxvrtrate                   : tendency term
!  indxvrtVmix2                  : 2d/3d coupling term
!  indxvrtWind                   : Wind stress term
!  indxvrtDrag                   : Bottom drag term
!  indxvrtBaro                   : Barotropic coupling term
!  indxvrtfast                   : Fast term
!
! ** DIAGNOSTICS_EK **
!  indxekHadv,indxekHdiff        : Horizontal advection and diffusion terms
!  indxekVadv                    : Vertical advection terms
!  indxekCor                     : Coriolis term,
!  indxekPrsgrd                  : Pressure gradient force term
!  indxekHmix, indxekVmix        : horizontal and vertical mixing terms
!  indxekrate                    : tendency term
!  indxekVmix2                   : 2d/3d coupling term
!  indxekWind                    : Wind stress term
!  indxekDrag                    : Bottom drag term
!  indxekBaro                    : Barotropic coupling term
!  indxekfast                    : Fast term
!
! ** DIAGNOSTICS_PV **
!  indxpvpv                      : Potential vorticity
!  indxpvpvd                     : Potential vorticity (using alternative formulation)
!  indxpvTrhs                    : right hand side of tracer equation
!  indxpvMrhs                    : right hand side of momentum equation
!
!
! ** DIAGNOSTICS_EDDY **
!
!=======================================================================
! Output file codes
      integer filetype_his, filetype_avg
     &       ,filetype_dia, filetype_dia_avg
     &       ,filetype_diaM, filetype_diaM_avg
     &       ,filetype_diags_vrt, filetype_diags_vrt_avg
     &       ,filetype_diags_ek, filetype_diags_ek_avg
     &       ,filetype_diags_pv, filetype_diags_pv_avg
     &       ,filetype_diags_eddy_avg
     &       ,filetype_surf, filetype_surf_avg
     &       ,filetype_diabio, filetype_diabio_avg
      parameter (filetype_his=1, filetype_avg=2,
     &           filetype_dia=3, filetype_dia_avg=4,
     &           filetype_diaM=5, filetype_diaM_avg=6,
     &           filetype_diags_vrt=7, filetype_diags_vrt_avg=8,
     &           filetype_diags_ek=9, filetype_diags_ek_avg=10,
     &           filetype_diags_pv=11, filetype_diags_pv_avg=12,
     &           filetype_diags_eddy_avg=17,
     &           filetype_surf=13, filetype_surf_avg=14,
     &           filetype_diabio=15,filetype_diabio_avg=16)
!
      integer iloop, indextemp
      integer indxTime, indxZ, indxUb, indxVb
      parameter (indxTime=1, indxZ=2, indxUb=3, indxVb=4)
#ifdef MORPHODYN
      integer indxHm
      parameter (indxHm=5)
#endif
#ifdef SOLVE3D
      integer indxU, indxV
      parameter (indxU=6, indxV=7)

#ifdef TRACERS
# ifdef TEMPERATURE
      integer indxT
      parameter (indxT=indxV+1)
# endif

# ifdef SALINITY
      integer indxS
      parameter (indxS=indxV+ntrc_temp+1)
# endif
# ifdef PASSIVE_TRACER
      integer, dimension(ntrc_pas) :: indxTPAS
     & =(/(iloop,iloop=indxV+ntrc_temp+ntrc_salt+1,
     &  indxV+ntrc_temp+ntrc_salt+ntrc_pas)/)
# endif
#endif
# ifdef BIOLOGY
#  ifdef PISCES
      integer indxDIC, indxTAL, indxOXY, indxCAL, indxPO4,
     &        indxPOC, indxSIL, indxPHY, indxZOO, indxDOC,
     &        indxDIA, indxMES, indxBSI, indxFER, indxBFE,
     &        indxGOC, indxSFE, indxDFE, indxDSI, indxNFE,
     &        indxNCH, indxDCH, indxNO3, indxNH4
      parameter (indxDIC =indxV+ntrc_temp+ntrc_salt+ntrc_pas+1,
     &           indxTAL =indxDIC+1, indxOXY=indxDIC+2,
     &           indxCAL=indxDIC+3, indxPO4=indxDIC+4,
     &           indxPOC=indxDIC+5, indxSIL=indxDIC+6,
     &           indxPHY =indxDIC+7, indxZOO=indxDIC+8,
     &           indxDOC =indxDIC+9, indxDIA=indxDIC+10,
     &           indxMES =indxDIC+11, indxBSI=indxDIC+12,
     &           indxFER =indxDIC+13, indxBFE=indxDIC+14,
     &           indxGOC =indxDIC+15, indxSFE=indxDIC+16,
     &           indxDFE =indxDIC+17, indxDSI=indxDIC+18,
     &           indxNFE =indxDIC+19, indxNCH=indxDIC+20,
     &           indxDCH =indxDIC+21, indxNO3=indxDIC+22,
     &           indxNH4 =indxDIC+23)
#    ifdef key_ligand
      integer indxLGW
      parameter (indxLGW=indxDIC+24)
#     endif
#    ifdef key_pisces_quota
     integer  indxDON, indxDOP, indxPON, indxPOP, indxNPH,
     &        indxPPH, indxNDI, indxPDI, indxPIC, indxNPI,
     &        indxPPI, indxPFE, indxPCH, indxGON, indxGOP
#     ifdef key_ligand
      parameter (indxDON=indxDIC+25, indxDOP=indxDIC+26,
     &           indxPON=indxDIC+27, indxPOP=indxDIC+28,
     &           indxNPH=indxDIC+29, indxPPH=indxDIC+30,
     &           indxNDI=indxDIC+31, indxPDI=indxDIC+32,
     &           indxPIC=indxDIC+33, indxNPI=indxDIC+34,
     &           indxPPI=indxDIC+35, indxPFE=indxDIC+36,
     &           indxPCH=indxDIC+37, indxGON=indxDIC+38,
     &           indxGOP=indxDIC+39)
#     else
      parameter (indxDON=indxDIC+24, indxDOP=indxDIC+25,
     &           indxPON=indxDIC+26, indxPOP=indxDIC+27,
     &           indxNPH=indxDIC+28, indxPPH=indxDIC+29,
     &           indxNDI=indxDIC+30, indxPDI=indxDIC+31,
     &           indxPIC=indxDIC+32, indxNPI=indxDIC+33,
     &           indxPPI=indxDIC+34, indxPFE=indxDIC+35,
     &           indxPCH=indxDIC+36, indxGON=indxDIC+37,
     &           indxGOP=indxDIC+38)
#     endif
#  endif
#  elif defined BIO_NChlPZD
      integer indxNO3, indxChla,
     &        indxPhy1,indxZoo1,
     &        indxDet1
#   ifdef OXYGEN
     &      , indxO2
#   endif
      parameter (indxNO3 =indxV+ntrc_temp+ntrc_salt+ntrc_pas+1,
     &           indxChla=indxNO3+1,
     &           indxPhy1=indxNO3+2,
     &           indxZoo1=indxNO3+3,
     &           indxDet1=indxNO3+4)
#   ifdef OXYGEN
      parameter (indxO2=indxNO3+5)
#   endif
#  elif defined BIO_N2ChlPZD2
      integer indxNO3, indxNH4, indxChla,
     &        indxPhy1, indxZoo1,
     &        indxDet1, indxDet2
      parameter (indxNO3 =indxV+ntrc_temp+ntrc_salt+ntrc_pas+1,
     &           indxNH4 =indxNO3+1, indxChla=indxNO3+2,
     &           indxPhy1=indxNO3+3,
     &           indxZoo1=indxNO3+4,
     &           indxDet1=indxNO3+5, indxDet2=indxNO3+6)
#  elif defined BIO_BioEBUS
      integer indxNO3, indxNO2, indxNH4,
     &        indxPhy1, indxPhy2, indxZoo1, indxZoo2,
     &        indxDet1, indxDet2, indxDON, indxO2
      parameter (indxNO3 =indxV+ntrc_temp+ntrc_salt+ntrc_pas+1,
     &           indxNO2 =indxNO3+1,
     &           indxNH4 =indxNO3+2,
     &           indxPhy1=indxNO3+3, indxPhy2=indxNO3+4,
     &           indxZoo1=indxNO3+5, indxZoo2=indxNO3+6,
     &           indxDet1=indxNO3+7, indxDet2=indxNO3+8,
     &           indxDON =indxNO3+9, indxO2  =indxNO3+10)
#     ifdef NITROUS_OXIDE
      integer indxN2O
      parameter (indxN2O=indxNO3+11)
#     endif
#  endif
# endif /* BIOLOGY */
# ifdef SEDIMENT
      integer, dimension(NGRAV) ::  indxGRAV
     & =(/(iloop,iloop=indxV+ntrc_temp+ntrc_salt+ntrc_pas+ntrc_bio+1,
     &  indxV+ntrc_temp+ntrc_salt+ntrc_pas+ntrc_bio+NGRAV)/)
      integer, dimension(NSAND) :: indxSAND
     & =(/(iloop,iloop=indxV+ntrc_temp+ntrc_salt+ntrc_pas+ntrc_bio+1+
     &	NGRAV,
     &  indxV+ntrc_temp+ntrc_salt+ntrc_pas+ntrc_bio+NGRAV+NSAND)/)
      integer, dimension(NMUD)   :: indxMUD
     & =(/(iloop,iloop=indxV+ntrc_temp+ntrc_salt+ntrc_pas+ntrc_bio+1+
     &  NGRAV+NSAND,
     &  indxV+ntrc_temp+ntrc_salt+ntrc_pas+ntrc_bio+NGRAV+NSAND+NMUD)/)
# endif

# if(!defined ANA_BSEDIM  && !defined SEDIMENT)
      integer indxBSD, indxBSS
      parameter (indxBSD=indxV+ntrc_temp+ntrc_salt+ntrc_pas+ntrc_bio+1,
     &           indxBSS=101)
# endif

# ifdef DIAGNOSTICS_TS
      integer indxTXadv,indxTYadv,indxTVadv,
     &        indxTHmix,indxTVmix,indxTForc,indxTrate
# ifdef DIAGNOSTICS_TSVAR
     &       ,indxTVmixt
# endif
#  if defined DIAGNOSTICS_TS_MLD
     &       ,indxTXadv_mld,indxTYadv_mld,indxTVadv_mld,
     &        indxTHmix_mld,indxTVmix_mld,indxTForc_mld,indxTrate_mld,
     &        indxTentr_mld
#  endif
      parameter (indxTXadv=indxV+ntrc_temp+ntrc_salt+ntrc_pas+
     &           ntrc_bio+ntrc_sed+1,
     &           indxTYadv=indxTXadv+NT,
     &           indxTVadv=indxTYadv+NT,
     &           indxTHmix=indxTVadv+NT,
# ifdef DIAGNOSTICS_TSVAR
     &           indxTVmixt=indxTHmix+NT,
     &           indxTVmix=indxTVmixt+NT,
# else
     &           indxTVmix=indxTHmix+NT,
# endif
     &           indxTForc=indxTVmix+NT,
     &           indxTrate=indxTForc+NT
#  ifdef DIAGNOSTICS_TS_MLD
     &          ,indxTXadv_mld=indxTrate+NT,
     &           indxTYadv_mld=indxTXadv_mld+NT,
     &           indxTVadv_mld=indxTYadv_mld+NT,
     &           indxTHmix_mld=indxTVadv_mld+NT,
     &           indxTVmix_mld=indxTHmix_mld+NT,
     &           indxTForc_mld=indxTVmix_mld+NT,
     &           indxTrate_mld=indxTForc_mld+NT,
     &           indxTentr_mld=indxTrate_mld+NT
#  endif
     &                                         )
# endif
# ifdef DIAGNOSTICS_UV
      integer indxMXadv,indxMYadv,indxMVadv,indxMCor,
     &        indxMPrsgrd,indxMHmix,indxMVmix,indxMrate,
     &        indxMVmix2,indxMHdiff
      parameter (indxMXadv=indxV+ntrc_temp+ntrc_salt+
     &           ntrc_pas+ntrc_bio+ntrc_sed
     &                                                  +ntrc_diats+1,
     &           indxMYadv=indxMXadv+2,
     &           indxMVadv=indxMYadv+2,
     &           indxMCor=indxMVadv+2,
     &           indxMPrsgrd=indxMCor+2,
     &           indxMHmix=indxMPrsgrd+2,
     &           indxMVmix=indxMHmix+2,
     &           indxMVmix2=indxMVmix+2,
     &           indxMHdiff=indxMVmix2+2,
     &           indxMrate=indxMHdiff+2)
#  ifdef DIAGNOSTICS_BARO
      integer indxMBaro
      parameter (indxMBaro=indxMrate+2)
#  endif
#  ifdef M3FAST
      integer indxMfast
      parameter (indxMfast=indxMrate+4)
#  endif
# endif
# ifdef DIAGNOSTICS_VRT
      integer indxvrtXadv,indxvrtYadv,indxvrtHdiff,indxvrtCor,
     &        indxvrtPrsgrd,indxvrtHmix,indxvrtVmix,indxvrtrate,
     &        indxvrtVmix2,indxvrtWind,indxvrtDrag
      parameter (indxvrtXadv=indxV+ntrc_temp+ntrc_salt+ntrc_pas+
     &                         ntrc_bio+ntrc_sed
     &                        +ntrc_diats+ntrc_diauv+1,
     &           indxvrtYadv=indxvrtXadv+1,
     &           indxvrtHdiff=indxvrtYadv+1,
     &           indxvrtCor=indxvrtHdiff+1,
     &           indxvrtPrsgrd=indxvrtCor+1,
     &           indxvrtHmix=indxvrtPrsgrd+1,
     &           indxvrtVmix=indxvrtHmix+1,
     &           indxvrtrate=indxvrtVmix+1,
     &           indxvrtVmix2=indxvrtrate+1,
     &           indxvrtWind=indxvrtVmix2+1,
     &           indxvrtDrag=indxvrtWind+1)
#  ifdef DIAGNOSTICS_BARO
      integer indxvrtBaro
      parameter (indxvrtBaro=indxvrtDrag+1)
#  endif
#  ifdef M3FAST
      integer indxvrtfast
      parameter (indxvrtfast=indxvrtDrag+2)
#  endif
# endif
# ifdef DIAGNOSTICS_EK
      integer indxekHadv,indxekHdiff,indxekVadv,indxekCor,
     &        indxekPrsgrd,indxekHmix,indxekVmix,indxekrate,
     &        indxekvol,indxekVmix2,indxekWind,indxekDrag
      parameter (indxekHadv=indxV+ntrc_temp+ntrc_salt+ntrc_pas+
     &           ntrc_bio+ntrc_sed+
     &           ntrc_diats+ntrc_diauv+ntrc_diavrt+1,
     &           indxekHdiff=indxekHadv+1,
     &           indxekVadv=indxekHdiff+1,
     &           indxekCor=indxekVadv+1,
     &           indxekPrsgrd=indxekCor+1,
     &           indxekHmix=indxekPrsgrd+1,
     &           indxekVmix=indxekHmix+1,
     &           indxekrate=indxekVmix+1,
     &           indxekvol=indxekrate+1,
     &           indxekVmix2=indxekvol+1,
     &           indxekWind=indxekVmix2+1,
     &           indxekDrag=indxekWind+1)
#  ifdef DIAGNOSTICS_BARO
      integer indxekBaro
      parameter (indxekBaro=indxekDrag+1)
#  endif
#  ifdef M3FAST
      integer indxekfast
      parameter (indxekfast=indxekDrag+2)
#  endif
#  ifdef DIAGNOSTICS_EK_MLD
      integer indxekHadv_mld,indxekHdiff_mld,indxekVadv_mld,
     &        indxekCor_mld,indxekPrsgrd_mld,indxekHmix_mld,
     &        indxekVmix_mld,indxekrate_mld,indxekvol_mld,
     &        indxekVmix2_mld,indxekWind_mld,indxekDrag_mld
      parameter (indxekHadv_mld=indxekDrag+2,
     &           indxekHdiff_mld=indxekHadv_mld+1,
     &           indxekVadv_mld=indxekHdiff_mld+1,
     &           indxekCor_mld=indxekVadv_mld+1,
     &           indxekPrsgrd_mld=indxekCor_mld+1,
     &           indxekHmix_mld=indxekPrsgrd_mld+1,
     &           indxekVmix_mld=indxekHmix_mld+1,
     &           indxekrate_mld=indxekVmix_mld+1,
     &           indxekvol_mld=indxekrate_mld+1,
     &           indxekVmix2_mld=indxekvol_mld+1,
     &           indxekWind_mld=indxekVmix2_mld+1,
     &           indxekDrag_mld=indxekWind_mld+1)
#   ifdef DIAGNOSTICS_BARO
      integer indxekBaro_mld
      parameter (indxekBaro_mld=indxekDrag_mld+1)
#   endif
#  endif
# endif
# ifdef DIAGNOSTICS_PV
      integer indxpvMrhs,indxpvTrhs
      parameter (indxpvTrhs=indxV+ntrc_temp+ntrc_salt+ntrc_pas+
     &                ntrc_bio+ntrc_sed+
     &                ntrc_diats+ntrc_diauv+ntrc_diavrt+ntrc_diaek+1,
     &           indxpvMrhs=indxpvTrhs+2)
#  ifdef DIAGNOSTICS_PV_FULL
      integer indxpvpv,indxpvpvd
      parameter (indxpvpv=indxpvMrhs+2,
     &           indxpvpvd=indxpvpv+1)
#  endif
# endif
# if defined DIAGNOSTICS_EDDY && ! defined XIOS
      integer indxeddyzz,
     &        indxeddyuu,indxeddyvv,indxeddyuv,indxeddyub,
     &        indxeddyvb,indxeddywb,indxeddyuw,indxeddyvw,
     &        indxeddyubu,indxeddyvbv,
     &        indxeddyusu,indxeddyvsv
      parameter (indxeddyzz=indxV+ntrc_temp+ntrc_salt
     &                           +ntrc_pas+ntrc_bio+ntrc_sed
     &                           +ntrc_diats+ntrc_diauv+ntrc_diavrt
     &                           +ntrc_diaek+ntrc_diapv+400,
     &           indxeddyuu=indxeddyzz+1,
     &           indxeddyvv=indxeddyuu+1,
     &           indxeddyuv=indxeddyvv+1,
     &           indxeddyub=indxeddyuv+1,
     &           indxeddyvb=indxeddyub+1,
     &           indxeddywb=indxeddyvb+1,
     &           indxeddyuw=indxeddywb+1,
     &           indxeddyvw=indxeddyuw+1,
     &           indxeddyubu=indxeddyvw+1,
     &           indxeddyvbv=indxeddyubu+1,
     &           indxeddyusu=indxeddyvbv+1,
     &           indxeddyvsv=indxeddyusu+1)
# endif
# if defined OUTPUTS_SURFACE && ! defined XIOS
      integer indxsurft,indxsurfs,indxsurfz,indxsurfu,
     &        indxsurfv
      parameter (indxsurft=indxV+ntrc_temp+ntrc_salt
     &                           +ntrc_pas+ntrc_bio+ntrc_sed
     &                           +ntrc_diats+ntrc_diauv+ntrc_diavrt
     &                           +ntrc_diaek+ntrc_diapv+ntrc_diaeddy+400,
     &           indxsurfs=indxsurft+1,
     &           indxsurfz=indxsurfs+1,
     &           indxsurfu=indxsurfz+1,
     &           indxsurfv=indxsurfu+1)
# endif
# if defined BIOLOGY && defined DIAGNOSTICS_BIO
      integer indxbioFlux, indxbioVSink
#  if (defined BIO_NChlPZD && defined OXYGEN) || defined BIO_BioEBUS
     &        , indxGasExcFlux
#  endif
      parameter (indxbioFlux=indxV+ntrc_temp+ntrc_salt
     &                           +ntrc_pas+ntrc_bio+ntrc_sed
     &                           +ntrc_diats+ntrc_diauv+ntrc_diavrt
     &                           +ntrc_diaek+ntrc_diapv+ntrc_diaeddy
     &                                               +ntrc_surf+400)
      parameter (indxbioVSink=indxbioFlux+NumFluxTerms)

#  if (defined BIO_NChlPZD && defined OXYGEN) || defined BIO_BioEBUS
      parameter (indxGasExcFlux=indxbioFlux+NumFluxTerms+NumVSinkTerms)
#  endif
# endif /* BIOLOGY && DIAGNOSTICS_BIO */

      integer indxO, indxW, indxR, indxVisc, indxDiff, indxAkv, indxAkt
      parameter (indxO=indxV+ntrc_temp+ntrc_salt+ntrc_pas+ntrc_bio
     &                      +ntrc_sed+ntrc_substot
# ifdef MUSTANG
     &              +ntrc_subs+16
#  ifdef key_MUSTANG_specif_outputs
     &              +3*ntrc_subs +2
#   ifdef key_MUSTANG_V2
     &              +1*ntrc_subs +13
#   endif
#   ifdef key_MUSTANG_bedload
     &              +4*ntrc_subs +3
#   endif
#  endif
# endif
     &           +ntrc_diats+ntrc_diauv+ntrc_diavrt+ntrc_diaek
     &           +ntrc_diapv+ntrc_diaeddy+ntrc_surf+ntrc_diabio+1,
     &           indxW=indxO+1, indxR=indxO+2, indxVisc=indxO+3,
     &           indxDiff=indxO+4,indxAkv=indxO+5, indxAkt=indxO+6)

# ifdef BIOLOGY
#  ifdef BIO_BioEBUS
      integer indxAOU, indxWIND10
      parameter (indxAOU=indxAkv+ntrc_temp+1,
     &           indxWIND10=indxAkv+ntrc_temp+2)
#     ifdef CARBON
      integer indxpCO2
      parameter (indxpCO2=indxAkv+ntrc_temp+3)
#     endif
#  endif
# endif

# ifdef SALINITY
      integer indxAks
      parameter (indxAks=indxAkv+ntrc_temp+4)
# endif
# if defined LMD_SKPP || defined GLS_MIXING
      integer indxHbl
      parameter (indxHbl=indxAkv+ntrc_temp+5)
# endif
# ifdef LMD_BKPP
      integer indxHbbl
      parameter (indxHbbl=indxAkv+ntrc_temp+6)
# endif
# ifdef GLS_MIXING
      integer indxTke
      parameter (indxTke=indxAkv+ntrc_temp+7)
      integer indxGls
      parameter (indxGls=indxAkv+ntrc_temp+8)
      integer indxLsc
      parameter (indxLsc=indxAkv+ntrc_temp+9)
      integer indxAkk
      parameter (indxAkk=indxAkv+ntrc_temp+10)
      integer indxAkp
      parameter (indxAkp=indxAkv+ntrc_temp+11)
# endif
#endif

      integer indxSSH
#if defined BIOLOGY && !defined PISCES
      integer indxHel
# if (defined BIO_NChlPZD || defined BIO_N2ChlPZD2)
     &      , indxChC
#  ifdef OXYGEN
     &      , indxU10, indxKvO2, indxO2sat
#  endif
# endif
#endif /* BIOLOGY*/

#ifdef SOLVE3D
# if defined BIOLOGY && !defined PISCES
      parameter (indxHel=indxAkv+ntrc_temp+12)
#  if (defined BIO_NChlPZD || defined BIO_N2ChlPZD2)
      parameter (indxChC=indxHel+1)
#   ifdef OXYGEN
      parameter (indxU10=indxChC+1)
      parameter (indxKvO2=indxU10+1)
      parameter (indxO2sat=indxKvO2+1)
      parameter (indxSSH=indxO2sat+1)
#   else
      parameter (indxSSH=indxChC+1)
#   endif
#  else
      parameter (indxSSH=indxHel+1)
#  endif
# else
      parameter (indxSSH=indxAkv+ntrc_temp+12)
# endif
#else
# if defined BIOLOGY && !defined PISCES
      parameter (indxHel=indxVb+1)
#  ifdef BIO_NChlPZD
      parameter (indxChC=indxHel+1)
#   ifdef OXYGEN
      parameter (indxU10=indxChC+1)
      parameter (indxKvO2=indxU10+1)
      parameter (indxO2sat=indxKvO2+1)
      parameter (indxSSH=indxO2sat+1)
#   else
      parameter (indxSSH=indxChC+1)
#   endif
#  endif
# else
      parameter (indxSSH=indxVb+1)
# endif
#endif /* SOLVE3D */

#if defined ANA_VMIX || defined BVF_MIXING \
  || defined LMD_MIXING || defined LMD_SKPP || defined LMD_BKPP \
  || defined GLS_MIXING
      integer indxbvf
      parameter (indxbvf=indxSSH+1)
#endif

#ifdef EXACT_RESTART
      integer indxrufrc
      parameter (indxrufrc=indxSSH+300)
      integer indxrvfrc
      parameter (indxrvfrc=indxrufrc+1)
      integer indxSUSTR, indxSVSTR
# ifdef TS_MIX_ISO_FILT
      integer indxdRdx,indxdRde
# endif
# ifdef M3FAST
      integer indxru_nbq,indxrv_nbq
      integer indxru_nbq_avg2,indxrv_nbq_avg2
      integer indxqdmu_nbq,indxqdmv_nbq
      parameter (indxru_nbq=indxrvfrc+1,
     &           indxrv_nbq=indxru_nbq+1,
     &           indxru_nbq_avg2=indxrv_nbq+1,
     &           indxrv_nbq_avg2=indxru_nbq_avg2+1,
     &           indxqdmu_nbq=indxrv_nbq_avg2+1,
     &           indxqdmv_nbq=indxqdmu_nbq+1)
#  ifdef TS_MIX_ISO_FILT
      parameter (indxdRdx=indxqdmv_nbq+1,
     &           indxdRde=indxdRdx+1)
      parameter (indxSUSTR=indxdRde+1,
     &           indxSVSTR=indxdRde+2)
#  else
      parameter (indxSUSTR=indxqdmv_nbq+1,
     &           indxSVSTR=indxqdmv_nbq+2)
#  endif
# else
#  ifdef TS_MIX_ISO_FILT
      parameter (indxdRdx=indxrvfrc+1,
     &           indxdRde=indxdRdx+1)
      parameter (indxSUSTR=indxdRde+1,
     &           indxSVSTR=indxdRde+2)
#  else
      parameter (indxSUSTR=indxrvfrc+1, indxSVSTR=indxrvfrc+2)
#  endif
# endif
#else
      integer indxSUSTR, indxSVSTR
      parameter (indxSUSTR=indxSSH+2, indxSVSTR=indxSSH+3)
#endif

      integer indxTime2
      parameter (indxTime2=indxSSH+4)

#ifdef SOLVE3D
      integer indxShflx, indxShflx_rsw
      parameter (indxShflx=indxSUSTR+5)
# ifdef SALINITY
      integer indxSwflx
      parameter (indxSwflx=indxShflx+1, indxShflx_rsw=indxShflx+2)
# else
      parameter (indxShflx_rsw=indxShflx+1)
# endif
      integer indxSST, indxdQdSST
      parameter (indxSST=indxShflx_rsw+1, indxdQdSST=indxShflx_rsw+2)
# if defined SALINITY && defined SFLX_CORR
      integer indxSSS
      parameter (indxSSS=indxSST+2)
# endif
# if defined BULK_FLUX || defined OA_COUPLING
      integer indxWSPD,indxTAIR,indxRHUM,indxRADLW,indxRADSW,
     &        indxPRATE,indxUWND,indxVWND,indxPATM
      parameter (indxWSPD=indxSST+3,  indxTAIR=indxSST+4,
     &           indxRHUM=indxSST+5,  indxRADLW=indxSST+6,
     &           indxRADSW=indxSST+7, indxPRATE=indxSST+8,
     &           indxUWND=indxSST+9,  indxVWND=indxSST+10,
     &           indxPATM=indxSST+11)
      integer indxShflx_rlw,indxShflx_lat,indxShflx_sen
      parameter (indxShflx_rlw=indxSST+12,
     &           indxShflx_lat=indxSST+13, indxShflx_sen=indxSST+14)
# endif
#endif /* SOLVE3D */

      integer indxWstr
      parameter (indxWstr=indxSUSTR+23)
      integer indxUWstr
      parameter (indxUWstr=indxSUSTR+24)
      integer indxVWstr
      parameter (indxVWstr=indxSUSTR+25)
      integer indxBostr
      parameter (indxBostr=indxSUSTR+26)
      integer indxBustr, indxBvstr
      parameter (indxBustr=indxSUSTR+27,  indxBvstr=indxBustr+1)
#ifdef SOLVE3D
# ifdef SEDIMENT
      integer indxSed, indxATHK, indxBTHK, indxBPOR
      parameter (indxATHK=indxSUSTR+29,
     &           indxSed=indxSUSTR+30,
     &           indxBTHK=indxSed, indxBPOR=indxSed+1)
      integer, dimension(NST) :: indxBFRA
     & =(/(iloop,iloop=indxSed+2,indxSed+1+NST)/)
#  ifdef SUSPLOAD
      integer, dimension(NST):: indxDFLX
     & =(/(iloop,iloop=indxSed+2+NST,indxSed+1+2*NST)/)
      integer, dimension(NST) ::indxEFLX
     & =(/(iloop,iloop=indxSed+2+2*NST,indxSed+1+3*NST)/)
#   ifdef BEDLOAD
      integer, dimension(NST):: indxBDLU
     & =(/(iloop,iloop=indxSed+2+3*NST,indxSed+1+4*NST)/)
      integer, dimension(NST) ::indxBDLV
     & =(/(iloop,iloop=indxSed+2+4*NST,indxSed+1+5*NST)/)
#   endif
#  else
#   ifdef BEDLOAD
      integer, dimension(NST):: indxBDLU
     & =(/(iloop,iloop=indxSed+2+NST,indxSed+1+2*NST)/)
      integer, dimension(NST) ::indxBDLV
     & =(/(iloop,iloop=indxSed+2+2*NST,indxSed+1+3*NST)/)
#   endif
#  endif
#  if defined MIXED_BED || defined COHESIVE_BED
      integer indxBTCR
#   if defined BEDLOAD && defined SUSPLOAD
      parameter (indxBTCR=indxSed+1+5*NST+1)
#   else
      parameter (indxBTCR=indxSed+1+3*NST+1)
#   endif
#  endif
# endif
# if defined SST_SKIN && defined TEMPERATURE
      integer indxSST_skin
      parameter (indxSST_skin=indxSUSTR+41)
# endif
#endif /* SOLVE3D */

#ifdef BBL
      integer indxBBL, indxAbed, indxHrip, indxLrip, indxZbnot,
     &        indxZbapp, indxBostrw
# ifdef SEDIMENT
      parameter (indxBBL=indxSUSTR+42+6*NST,
# else
      parameter (indxBBL=indxSUSTR+42,
# endif
     &           indxAbed  =indxBBL,    indxHrip  =indxAbed+1,
     &           indxLrip  =indxAbed+2, indxZbnot =indxAbed+3,
     &           indxZbapp =indxAbed+4, indxBostrw=indxAbed+5)
# ifndef ANA_WWAVE
      integer indxWWA,indxWWD,indxWWP,indxWEB,indxWED,indxWER
#  ifdef MUSTANG
     &          ,indxWWU
#  endif
      parameter (indxWWA=indxAbed+6, indxWWD=indxWWA+1,
     &           indxWWP=indxWWA+2
#  ifdef MUSTANG
     &          ,indxWWU=indxWWA+7
#  endif
#  ifdef MRL_WCI
     &          ,indxWEB=indxWWA+3,indxWED=indxWWA+4,
     &           indxWER=indxWWA+5
#  endif
     &                             )
# endif /* !ANA_WWAVE */
# ifndef ANA_BSEDIM
# endif

#else /* BBL */

      integer indxWWA,indxWWD,indxWWP,indxWEB,indxWED,indxWER
# ifdef MUSTANG
     &          ,indxWWU
# endif
      parameter (indxWWA=indxSUSTR+42, indxWWD=indxWWA+1,
     &           indxWWP=indxWWA+2
# ifdef MUSTANG
     &          ,indxWWU=indxWWA+7
# endif
# ifdef MRL_WCI
     &          ,indxWEB=indxWWA+3,indxWED=indxWWA+4,
     &           indxWER=indxWWA+5
# endif
     &                             )
#endif  /* BBL */

#if defined MRL_WCI || defined OW_COUPLING
      integer indxSUP, indxUST2D,indxVST2D
# ifdef SEDIMENT
      parameter (indxSUP=indxSUSTR+54+6*NST,
# else
      parameter (indxSUP  =indxSUSTR+54,
# endif
     &           indxUST2D =indxSUP+1, indxVST2D=indxSUP+2)
# ifdef SOLVE3D
      integer indxUST,indxVST,indxWST,indxAkb,indxAkw,indxKVF,
     &        indxCALP,indxKAPS
      parameter (indxUST=indxSUP+3, indxVST=indxSUP+4,
     &           indxWST=indxSUP+5, indxAkb=indxSUP+6,
     &           indxAkw=indxSUP+7, indxKVF=indxSUP+8,
     &           indxCALP=indxSUP+9, indxKAPS=indxSUP+10)
# endif
# ifdef DIAGNOSTICS_UV
      integer indxMvf,indxMbrk,indxMStCo,indxMVvf,
     &        indxMPrscrt,indxMsbk,indxMbwf,indxMfrc
      parameter (indxMvf=indxKAPS+1,indxMbrk=indxMvf+2,
     &           indxMStCo=indxMvf+4,indxMVvf=indxMvf+6,
     &           indxMPrscrt=indxMvf+8,indxMsbk=indxMvf+10,
     &           indxMbwf=indxMvf+12,indxMfrc=indxMvf+14)
# endif
      integer indxHRM,indxFRQ,indxWKX,indxWKE,indxEPB
     &       ,indxEPD,indxWAC,indxWAR,indxEPR
      parameter (indxHRM=indxSUP+40,
     &           indxFRQ=indxHRM+1, indxWAC=indxHRM+2,
     &           indxWKX=indxHRM+3, indxWKE=indxHRM+4,
     &           indxEPB=indxHRM+5, indxEPD=indxHRM+6,
     &           indxWAR=indxHRM+7, indxEPR=indxHRM+8 )
#endif  /* MRL_WCI */

#ifdef PSOURCE_NCFILE
      integer indxQBAR
      parameter (indxQBAR=indxSUSTR+122)
# ifdef PSOURCE_NCFILE_TS
      integer indxTsrc
      parameter (indxTsrc=indxSUSTR+123)
# endif
#endif /* PSOURCE_NCFILE */

#ifdef DIURNAL_INPUT_SRFLX
      integer indxShflx_rswbio
      parameter (indxShflx_rswbio=indxSUSTR+124)
#endif
#if defined BHFLUX
      integer indxBhflx
      parameter (indxBhflx=indxSUSTR+131)
#endif
#if defined BWFLUX  && defined SALINTY
      integer indxBwflx
      parameter (indxBwflx=indxSUSTR+132)
#endif

#ifdef ICE
      integer indxAi
      parameter (indxAi=????)
      integer indxUi, indxVi, indxHi, indxHS, indxTIsrf
      parameter (indxUi=indxAi+1, indxVi=indxAi+2, indxHi=indxAi+3,
     &                         indxHS=indxAi+4, indxTIsrf=indxAi+5)
#endif
!
!
!===================================================================
!
!===================================================================
!
! Grid Type Codes:  r2dvar....w3hvar are codes for array types.
! ==== ==== ======  The codes are set according to the rule:
!                     horiz_grid_type+4*vert_grid_type
!    where horiz_grid_type=0,1,2,3 for RHO-,U-,V-,PSI-points
!    respectively and vert_grid_type=0 for 2D fields; 1,2 for
!    3D-RHO- and W-vertical points.

!
      integer r2dvar, u2dvar, v2dvar, p2dvar, r3dvar,
     &                u3dvar, v3dvar, p3dvar, w3dvar,
     &                pw3dvar, b3dvar
      parameter (r2dvar=0, u2dvar=1, v2dvar=2, p2dvar=3,
     & r3dvar=4, u3dvar=5, v3dvar=6, p3dvar=7, w3dvar=8,
     & pw3dvar=11, b3dvar=12)

!            Horizontal array dimensions in netCDF files.
! xi_rho     WARNING!!! In MPI code in the case of PARALLEL_FILES
! xi_u       _and_ NON-Periodicity in either XI- or ETA-direction,
! eta_rho    these depend on corresonding MPI-node indices ii,jj
! eta_v      and therefore become live variables, which are placed
!            into common block below rather than defined here as
!            parameters.
!
! Note (P. Marchesiello):
!   the remark above is now extended to periodic conditions, i.e.,
!   if PARALLEL_FILES is defined, netCDF files array dimensions are
!   always set in MPI-Setup and depend on MPI-nodes. After rejoining
!   the parallel files (ncjoin), the resulting global netCDF file has
!   the same dimension as it would have if PARALLEL_FILES was undefined.
!
      integer xi_rho,xi_u, eta_rho,eta_v
#ifndef AGRIF
# if defined MPI && defined PARALLEL_FILES
!#  ifdef EW_PERIODIC
!      parameter (xi_rho=Lm,     xi_u=Lm)
!#  endif
!#  ifdef NS_PERIODIC
!      parameter (eta_rho=Mm,    eta_v=Mm)
!#  endif
# else
      parameter (xi_rho=LLm+2,  xi_u=xi_rho-1,
     &           eta_rho=MMm+2, eta_v=eta_rho-1)
# endif
#else
# if defined MPI && defined PARALLEL_FILES
!#  ifdef EW_PERIODIC
!      common/netCDFhorizdim1/xi_rho,xi_u
!#  endif
!#  ifdef NS_PERIODIC
!      common/netCDFhorizdim2/eta_rho,eta_v
!#  endif
# else
      common/netCDFhorizdim/xi_rho,xi_u, eta_rho,eta_v
# endif
#endif /* AGRIF */
!
!====================================================================
! Naming conventions for indices, variable IDs, etc...
!
! prefix ncid_  means netCDF ID for netCDF file
!        nrec_  record number in netCDF file since initialization
!        nrpf_  maximum number of records per file  (output netCDF
!                                                       files only)
! prefix/ending rst_/_rst refers to restart  netCDF file
!               his_/_his           history
!               avg_/_avg           averages
!                    _frc           forcing
!                    _clm           climatology
!                    _qbar          river runoff
!                    _btf           hydrothermal flux
!
! endings refer to:  ___Time  time [in seconds]
!                    ___Tstep time step numbers and record numbers
!   all objects      ___Z     free-surface
!   with these       ___Ub    vertically integrated 2D U-momentum
!   endings are      ___Vb    vertically integrated 2D V-momentum
!   either
!     netCDF IDs,    ___U     3D U-momentum
!     if occur with  ___V     3D V-momentum
!     prefices rst/  ___T(NT) tracers
!     /his/avg       ___R     density anomaly
!   or               ___O     omega vertical velocity
!     parameter      ___W     true vertical velocity
!     indices, if
!     occur with     ___Akv   vertical viscosity coefficient
!     prefix indx    ___Akt   vertical T-diffusion coefficient
!     (see above).   ___Aks   vertical S-diffusion coefficient
!                    ___Hbl   depth of mixed layer LMD_SKPP.
!
! Sizes of unlimited time dimensions in netCDF files:
!
!   ntsms   surface momentum stress in current forcing file.
!   ntbulk   bulk formulation in current forcing file.
!   ntsrf   shortwave radiation flux in current forcing file.
!   ntssh   sea surface height in current climatology file.
!   ntsst   sea surface temperature in current forcing file.
!   ntsss   sea surface salinity in current forcing file.
!   ntstf   surface flux of tracers in current forcing file.
!   nttclm  tracer variables in current climatology file.
!   ntuclm  momentum variables in current climatology file.
!   ntww    wind induced wave data in current forcing file.
!   ntbulkn bulk formula variables in current forcing file.
!   ntqbar  river runoff in current forcing file.
!   ntbtf   bottom hydrothermal flux of tracer in current forcing file.
!
! vname    character array for variable names and attributes;
!=================================================================
!
      integer ncidfrc, ncidbulk, ncidclm,  ntsms
     &     , ncidqbar, ncidbtf
     &     , ntsrf,  ntssh,  ntsst, ntsss, ntuclm
     &     , ntbulk, ntqbar, ntww
#if defined WAVE_OFFLINE
      integer ncidwave
#endif

#if defined SOLVE3D && defined TRACERS
      integer nttclm(NT), ntstf(NT), nttsrc(NT)
     &       , ntbtf(NT)
#endif
      integer ncidrst, nrecrst,  nrpfrst
     &      , rstTime, rstTime2, rstTstep, rstZ,    rstUb,  rstVb
#ifdef SOLVE3D
     &                         , rstU,    rstV
# if defined TRACERS
      integer rstT(NT)
# endif
# if defined LMD_SKPP
      integer rstHbl
# endif
# ifdef LMD_BKPP
      integer rstHbbl
# endif
# if defined GLS_MIXING
      integer rstAkv,rstAkt
#  if defined SALINITY
      integer rstAks
#  endif
      integer rstTke,rstGls
# endif
# ifdef M3FAST
#  if defined LMD_MIXING || defined GLS_MIXING
      integer rstBustr, rstBvstr
#  endif
# endif
# ifdef SEDIMENT
      integer rstSed(NST+2)
# endif
# ifdef MUSTANG
      integer rstMUS(NT+3)
# endif
#endif
#ifdef EXACT_RESTART
      integer rstrufrc,rstrvfrc
# ifdef M3FAST
      integer rstru_nbq,rstrv_nbq
      integer rstru_nbq_avg2,rstrv_nbq_avg2
      integer rstqdmu_nbq,rstqdmv_nbq
# endif  /* M3FAST */
# ifdef TS_MIX_ISO_FILT
      integer rstdRdx,rstdRde
# endif
#endif
#ifdef MORPHODYN
      integer rstHm
#endif
#ifdef BBL
      integer rstBBL(2)
#endif
#ifdef WAVE_IO
      integer rstWAVE(3),hisWAVE(9)
      common /ncvars/ rstWAVE,hisWAVE
#endif
#ifdef MRL_WCI
      integer hisSUP, hisUST2D, hisVST2D
      common /ncvars/ hisSUP, hisUST2D, hisVST2D
# ifdef SOLVE3D
      integer hisUST, hisVST, hisAkb, hisAkw, hisKVF,
     &        hisCALP, hisKAPS, hisWST
      common /ncvars/ hisUST, hisVST, hisAkb, hisAkw, hisKVF,
     &        hisCALP, hisKAPS, hisWST
# endif
#endif

      integer  ncidhis, nrechis,  nrpfhis
     &      , hisTime, hisTime2, hisTstep, hisZ,    hisUb,  hisVb
     &      , hisBostr, hisWstr, hisUWstr, hisVWstr
     &      , hisBustr, hisBvstr
     &      , hisShflx, hisSwflx, hisShflx_rsw, hisBhflx, hisBwflx
# ifdef MORPHODYN
     &      , hisHm
#endif
#ifdef BBL
     &      , hisBBL(6)
#endif
#ifdef SOLVE3D
     &      , hisU,   hisV,   hisR,    hisHbl, hisHbbl
     &      , hisO,   hisW,   hisVisc, hisDiff
     &      , hisAkv, hisAkt, hisAks
# if defined ANA_VMIX || defined BVF_MIXING \
  || defined LMD_MIXING || defined LMD_SKPP || defined LMD_BKPP \
  || defined GLS_MIXING
     &      , hisbvf
# endif
# ifdef GLS_MIXING
     &      , hisTke, hisGls, hisLsc
# endif
# if defined BULK_FLUX || defined OA_COUPLING
     &      , hisShflx_rlw
     &      , hisShflx_lat,   hisShflx_sen
# endif
# ifdef SST_SKIN
     &      , hisSST_skin
# endif
# ifdef BIOLOGY
     &      , hisHel
#  ifdef BIO_NChlPZD
     &      , hisChC
#   ifdef OXYGEN
     &      , hisU10, hisKvO2, hisO2sat
#   endif
#  elif defined BIO_BioEBUS
      integer hisAOU, hisWIND10
#  endif
# endif  /* BIOLOGY */
      integer hisT(NT)
# ifdef SEDIMENT
      integer hisSed(1+NST+2
#  ifdef SUSPLOAD
     &      +2*NST
#  endif
#  ifdef BEDLOAD
     &      +2*NST
#  endif
#  if defined MIXED_BED || defined COHESIVE_BED
     &      +1
#  endif
     & )
# endif /* SEDIMENT */

# ifdef MUSTANG
      integer hisMust(ntrc_subs+6
#  ifdef key_MUSTANG_specif_outputs
     &                +3*ntrc_subs + 2
#   ifdef key_MUSTANG_V2
     &                +1*ntrc_subs + 13
#    ifdef key_MUSTANG_bedload
     &                +4*ntrc_subs + 3
#    endif
#   endif
#  endif
     &               )
# endif /* MUSTANG */

# if defined DIAGNOSTICS_TS
      integer nciddia, nrecdia, nrpfdia
     &      , diaTime, diaTime2, diaTstep
     &      , diaTXadv(NT), diaTYadv(NT), diaTVadv(NT)
     &      , diaTHmix(NT), diaTVmix(NT)
#  ifdef DIAGNOSTICS_TSVAR
     &      , diaTVmixt(NT)
#  endif
     &      , diaTForc(NT), diaTrate(NT)
#  if defined DIAGNOSTICS_TS_MLD
     &      , diaTXadv_mld(NT), diaTYadv_mld(NT), diaTVadv_mld(NT)
     &      , diaTHmix_mld(NT), diaTVmix_mld(NT)
     &      , diaTForc_mld(NT), diaTrate_mld(NT), diaTentr_mld(NT)
#  endif
# endif
# ifdef DIAGNOSTICS_UV
        integer nciddiaM, nrecdiaM, nrpfdiaM
     &      , diaTimeM,diaTime2M, diaTstepM
     &      , diaMXadv(2), diaMYadv(2), diaMVadv(2)
     &      , diaMCor(2), diaMPrsgrd(2), diaMHmix(2)
     &      , diaMHdiff(2)
     &      , diaMVmix(2), diaMVmix2(2), diaMrate(2)
#  ifdef DIAGNOSTICS_BARO
     &      , diaMBaro(2)
#  endif
#  ifdef M3FAST
     &      , diaMfast(2)
#  endif
#  ifdef MRL_WCI
     &      , diaMvf(2), diaMbrk(2), diaMStCo(2)
     &      , diaMVvf(2), diaMPrscrt(2), diaMsbk(2)
     &      , diaMbwf(2), diaMfrc(2)
#  endif
# endif
# ifdef DIAGNOSTICS_VRT
      integer nciddiags_vrt, nrecdiags_vrt, nrpfdiags_vrt
     &      , diags_vrtTime, diags_vrtTime2, diags_vrtTstep
     &      , diags_vrtXadv(2), diags_vrtYadv(2), diags_vrtHdiff(2)
     &      , diags_vrtCor(2), diags_vrtPrsgrd(2), diags_vrtHmix(2)
     &      , diags_vrtVmix(2), diags_vrtrate(2)
     &      , diags_vrtVmix2(2), diags_vrtWind(2), diags_vrtDrag(2)
#  ifdef DIAGNOSTICS_BARO
     &      , diags_vrtBaro(2)
#  endif
#  ifdef M3FAST
     &      , diags_vrtfast(2)
#  endif
# endif
# ifdef DIAGNOSTICS_EK
      integer nciddiags_ek, nrecdiags_ek, nrpfdiags_ek
     &      , diags_ekTime, diags_ekTime2, diags_ekTstep
     &      , diags_ekHadv(2), diags_ekHdiff(2),  diags_ekVadv(2)
     &      , diags_ekCor(2), diags_ekPrsgrd(2), diags_ekHmix(2)
     &      , diags_ekVmix(2), diags_ekrate(2), diags_ekvol(2)
     &      , diags_ekVmix2(2), diags_ekWind(2), diags_ekDrag(2)
#  ifdef DIAGNOSTICS_BARO
     &      , diags_ekBaro(2)
#  endif
#  ifdef M3FAST
     &      , diags_ekfast(2)
#  endif
#  ifdef DIAGNOSTICS_EK_MLD
      integer diags_ekHadv_mld(2), diags_ekHdiff_mld(2)
     &      ,  diags_ekVadv_mld(2), diags_ekCor_mld(2)
     &      , diags_ekPrsgrd_mld(2), diags_ekHmix_mld(2)
     &      , diags_ekVmix_mld(2), diags_ekrate_mld(2)
     &      , diags_ekvol_mld(2), diags_ekVmix2_mld(2)
#  endif
#  ifdef DIAGNOSTICS_BARO
     &      , diags_ekBaro_mld(2)
#  endif
# endif
# ifdef DIAGNOSTICS_PV
      integer nciddiags_pv, nrecdiags_pv, nrpfdiags_pv
     &      , diags_pvTime, diags_pvTime2, diags_pvTstep
#  ifdef DIAGNOSTICS_PV_FULL
     &      , diags_pvpv(2), diags_pvpvd(2)
#  endif
     &      , diags_pvMrhs(2), diags_pvTrhs(2)
# endif

# if defined DIAGNOSTICS_EDDY && ! defined XIOS
      integer nciddiags_eddy, nrecdiags_eddy, nrpfdiags_eddy
     &      , diags_eddyTime, diags_eddyTime2, diags_eddyTstep
     &      , diags_eddyzz(2)
     &      , diags_eddyuu(2), diags_eddyvv(2), diags_eddyuv(2)
     &      , diags_eddyub(2), diags_eddyvb(2), diags_eddywb(2)
     &      , diags_eddyuw(2), diags_eddyvw(2)
     &      , diags_eddyubu(2), diags_eddyvbv(2)
     &      , diags_eddyusu(2), diags_eddyvsv(2)
# endif

# if defined OUTPUTS_SURFACE && ! defined XIOS
      integer ncidsurf, nrecsurf, nrpfsurf
     &      , surfTime, surfTime2, surfTstep
     &      , surf_surft(2), surf_surfs(2),  surf_surfz(2)
     &      , surf_surfu(2), surf_surfv(2)
# endif
# ifdef DIAGNOSTICS_BIO
      integer nciddiabio, nrecdiabio, nrpfdiabio
     &      , diaTimebio, diaTime2bio, diaTstepbio
     &      , diabioFlux(NumFluxTerms)
     &      , diabioVSink(NumVSinkTerms)
     &      , diabioGasExc(NumGasExcTerms)
# endif

#elif defined DIAGNOSTICS_UV && defined MRL_WCI
     &      , diaMvf(2), diaMbrk(2), diaMStCo(2)
     &      , diaMVvf(2), diaMPrscrt(2), diaMsbk(2)
     &      , diaMbwf(2), diaMfrc(2)
#endif /* SOLVE3D */

#ifdef AVERAGES
      integer ncidavg, nrecavg,  nrpfavg
     &      , avgTime, avgTime2, avgTstep, avgZ, avgUb,  avgVb
     &      , avgBostr, avgWstr, avgUwstr, avgVwstr
     &      , avgBustr, avgBvstr
     &      , avgShflx, avgSwflx, avgShflx_rsw, avgBhflx, avgBwflx
# ifdef MORPHODYN
     &      , avgHm
# endif

# ifdef SOLVE3D
     &      , avgU,   avgV,   avgR,    avgHbl, avgHbbl
     &      , avgO,   avgW,   avgVisc, avgDiff
     &      , avgAkv, avgAkt, avgAks
#  if defined ANA_VMIX || defined BVF_MIXING \
 || defined LMD_MIXING || defined LMD_SKPP || defined LMD_BKPP \
 || defined GLS_MIXING
     &      , avgbvf
#  endif
#  ifdef GLS_MIXING
     &      , avgTke, avgGls, avgLsc
#  endif

#  ifdef BIOLOGY
     &      , avgHel
#   ifdef BIO_NChlPZD
     &      , avgChC
#    ifdef OXYGEN
     &      , avgU10, avgKvO2, avgO2sat
#    endif
#   elif defined BIO_BioEBUS
      integer avgAOU, avgWIND10

#  endif
# endif  /* BIOLOGY */
# if defined TRACERS
      integer avgT(NT)
# endif
#  if defined BULK_FLUX || defined OA_COUPLING
      integer avgShflx_rlw
     &      , avgShflx_lat,   avgShflx_sen
#  endif
#  ifdef SST_SKIN
      integer avgSST_skin
#  endif

#  ifdef SEDIMENT
      integer avgSed(1+NST+2
#   ifdef SUSPLOAD
     &      +2*NST
#   endif
#   ifdef BEDLOAD
     &      +2*NST
#   endif
#   if defined MIXED_BED || defined COHESIVE_BED
     &      +1
#   endif
     & )
#  endif
#  ifdef MUSTANG
      integer avgMust(ntrc_subs+6)
#  endif

# endif /* SOLVE3D */

# ifdef BBL
      integer avgBBL(6)
# endif
# ifdef WAVE_IO
      integer avgWAVE(9)
# endif
# ifdef MRL_WCI
      integer avgSUP, avgUST2D, avgVST2D
#  ifdef SOLVE3D
      integer avgUST, avgVST, avgAkb, avgAkw, avgKVF,
     &        avgCALP, avgKAPS, avgWST
#  endif
# endif
# if defined SOLVE3D && defined TRACERS
#  if defined DIAGNOSTICS_TS && defined TRACERS
      integer nciddia_avg, nrecdia_avg, nrpfdia_avg
     &      , diaTime_avg, diaTime2_avg, diaTstep_avg
     &      , diaTXadv_avg(NT), diaTYadv_avg(NT), diaTVadv_avg(NT)
     &      , diaTHmix_avg(NT), diaTVmix_avg(NT)
#  ifdef DIAGNOSTICS_TSVAR
     &      , diaTVmixt_avg(NT)
#  endif
     &      , diaTForc_avg(NT), diaTrate_avg(NT)
#   ifdef DIAGNOSTICS_TS_MLD
     &      , diaTXadv_mld_avg(NT), diaTYadv_mld_avg(NT)
     &      , diaTVadv_mld_avg(NT)
     &      , diaTHmix_mld_avg(NT), diaTVmix_mld_avg(NT)
     &      , diaTForc_mld_avg(NT), diaTrate_mld_avg(NT)
     &      , diaTentr_mld_avg(NT)
#   endif
#  endif
#  ifdef DIAGNOSTICS_UV
       integer nciddiaM_avg, nrecdiaM_avg, nrpfdiaM_avg
     &      , diaTimeM_avg, diaTime2M_avg, diaTstepM_avg
     &      , diaMXadv_avg(2), diaMYadv_avg(2), diaMVadv_avg(2)
     &      , diaMCor_avg(2), diaMPrsgrd_avg(2), diaMHmix_avg(2)
     &      , diaMHdiff_avg(2)
     &      , diaMVmix_avg(2), diaMVmix2_avg(2), diaMrate_avg(2)
#   ifdef DIAGNOSTICS_BARO
     &      , diaMBaro_avg(2)
#   endif
#   ifdef M3FAST
     &      , diaMfast_avg(2)
#   endif
#  endif
#  ifdef DIAGNOSTICS_VRT
       integer nciddiags_vrt_avg, nrecdiags_vrt_avg, nrpfdiags_vrt_avg
     &      , diags_vrtTime_avg, diags_vrtTime2_avg, diags_vrtTstep_avg
     &      , diags_vrtXadv_avg(2), diags_vrtYadv_avg(2), diags_vrtHdiff_avg(2)
     &      , diags_vrtCor_avg(2), diags_vrtPrsgrd_avg(2), diags_vrtHmix_avg(2)
     &      , diags_vrtVmix_avg(2), diags_vrtrate_avg(2)
     &      , diags_vrtVmix2_avg(2), diags_vrtWind_avg(2), diags_vrtDrag_avg(2)
#   ifdef DIAGNOSTICS_BARO
     &      , diags_vrtBaro_avg(2)
#   endif
#   ifdef M3FAST
     &      , diags_vrtfast_avg(2)
#   endif
#  endif
#  ifdef DIAGNOSTICS_EK
       integer nciddiags_ek_avg, nrecdiags_ek_avg, nrpfdiags_ek_avg
     &      , diags_ekTime_avg, diags_ekTime2_avg, diags_ekTstep_avg
     &      , diags_ekHadv_avg(2), diags_ekHdiff_avg(2), diags_ekVadv_avg(2)
     &      , diags_ekCor_avg(2), diags_ekPrsgrd_avg(2), diags_ekHmix_avg(2)
     &      , diags_ekVmix_avg(2), diags_ekrate_avg(2), diags_ekvol_avg(2)
     &      , diags_ekVmix2_avg(2), diags_ekWind_avg(2), diags_ekDrag_avg(2)
#   ifdef DIAGNOSTICS_BARO
     &      , diags_ekBaro_avg(2)
#   endif
#   ifdef M3FAST
     &      , diags_ekfast_avg(2)
#   endif
#   ifdef DIAGNOSTICS_EK_MLD
       integer diags_ekHadv_mld_avg(2), diags_ekHdiff_mld_avg(2)
     &      , diags_ekVadv_mld_avg(2), diags_ekCor_mld_avg(2)
     &      , diags_ekPrsgrd_mld_avg(2), diags_ekHmix_mld_avg(2)
     &      , diags_ekVmix_mld_avg(2), diags_ekrate_mld_avg(2)
     &      , diags_ekvol_mld_avg(2), diags_ekVmix2_mld_avg(2)
#   endif
#   ifdef DIAGNOSTICS_BARO
     &      , diags_ekBaro_mld_avg(2)
#   endif
#  endif
#  ifdef DIAGNOSTICS_PV
       integer nciddiags_pv_avg, nrecdiags_pv_avg, nrpfdiags_pv_avg
     &      , diags_pvTime_avg, diags_pvTime2_avg, diags_pvTstep_avg
#   ifdef DIAGNOSTICS_PV_FULL
     &      , diags_pvpv_avg(2), diags_pvpvd_avg(2)
#   endif
     &      , diags_pvMrhs_avg(2), diags_pvTrhs_avg(2)
#  endif
# if defined DIAGNOSTICS_EDDY && ! defined XIOS
       integer nciddiags_eddy_avg, nrecdiags_eddy_avg, nrpfdiags_eddy_avg
     &      , diags_eddyTime_avg, diags_eddyTime2_avg, diags_eddyTstep_avg
     &      , diags_eddyzz_avg(2)
     &      , diags_eddyuu_avg(2), diags_eddyvv_avg(2), diags_eddyuv_avg(2)
     &      , diags_eddyub_avg(2), diags_eddyvb_avg(2), diags_eddywb_avg(2)
     &      , diags_eddyuw_avg(2), diags_eddyvw_avg(2)
     &      , diags_eddyubu_avg(2), diags_eddyvbv_avg(2)
     &      , diags_eddyusu_avg(2), diags_eddyvsv_avg(2)
#  endif
# if defined OUTPUTS_SURFACE && ! defined XIOS
       integer ncidsurf_avg, nrecsurf_avg, nrpfsurf_avg
     &      , surfTime_avg, surfTime2_avg, surfTstep_avg
     &      , surf_surft_avg(2), surf_surfs_avg(2), surf_surfz_avg(2)
     &      , surf_surfu_avg(2), surf_surfv_avg(2)
#  endif
#  ifdef DIAGNOSTICS_BIO
      integer nciddiabio_avg, nrecdiabio_avg, nrpfdiabio_avg
     &      , diaTimebio_avg, diaTime2bio_avg, diaTstepbio_avg
     &      , diabioFlux_avg(NumFluxTerms)
     &      , diabioVSink_avg(NumVSinkTerms)
     &      , diabioGasExc_avg(NumGasExcTerms)
#  endif
# endif /* SOLVE3D */

# ifdef MRL_WCI
     &      , diaMvf_avg(2), diaMbrk_avg(2), diaMStCo_avg(2)
     &      , diaMVvf_avg(2), diaMPrscrt_avg(2), diaMsbk_avg(2)
     &      , diaMbwf_avg(2), diaMfrc_avg(2)
# endif

#endif /* AVERAGES */

#ifdef SOLVE3D
# define NWRTHIS 500+NT
#else
# define NWRTHIS 90
#endif
      logical wrthis(NWRTHIS)
#ifdef AVERAGES
     &      , wrtavg(NWRTHIS)
#endif
#ifdef DIAGNOSTICS_TS
     &      , wrtdia3D(NT+1)
     &      , wrtdia2D(NT+1)
# ifdef AVERAGES
     &      , wrtdia3D_avg(NT+1)
     &      , wrtdia2D_avg(NT+1)
# endif
#endif
#ifdef DIAGNOSTICS_UV
     &      , wrtdiaM(3)
# ifdef AVERAGES
     &      , wrtdiaM_avg(3)
# endif
#endif
#ifdef DIAGNOSTICS_VRT
     &      , wrtdiags_vrt(3)
# ifdef AVERAGES
     &      , wrtdiags_vrt_avg(3)
# endif
#endif
#ifdef DIAGNOSTICS_EK
     &      , wrtdiags_ek(3)
# ifdef AVERAGES
     &      , wrtdiags_ek_avg(3)
# endif
#endif
#ifdef DIAGNOSTICS_PV
     &      , wrtdiags_pv(NT+1)
# ifdef AVERAGES
     &      , wrtdiags_pv_avg(NT+1)
# endif
#endif
# if defined DIAGNOSTICS_EDDY && ! defined XIOS
     &      , wrtdiags_eddy(3)
# ifdef AVERAGES
     &      , wrtdiags_eddy_avg(3)
# endif
#endif
# if defined OUTPUTS_SURFACE && ! defined XIOS
     &      , wrtsurf(3)
# ifdef AVERAGES
     &      , wrtsurf_avg(3)
# endif
#endif
#ifdef DIAGNOSTICS_BIO
     &      , wrtdiabioFlux(NumFluxTerms+1)
     &      , wrtdiabioVSink(NumVSinkTerms+1)
     &      , wrtdiabioGasExc(NumGasExcTerms+1)
# ifdef AVERAGES
     &      , wrtdiabioFlux_avg(NumFluxTerms+1)
     &      , wrtdiabioVSink_avg(NumVSinkTerms+1)
     &      , wrtdiabioGasExc_avg(NumGasExcTerms+1)
# endif
#endif

      common/incscrum/
     &     ncidfrc, ncidbulk,ncidclm, ncidqbar, ncidbtf
     &     , ntsms, ntsrf, ntssh, ntsst
     &     , ntuclm, ntsss, ntbulk, ntqbar, ntww
#ifdef WAVE_OFFLINE
     &      , ncidwave
#endif

#if defined MPI && defined PARALLEL_FILES
!# ifndef EW_PERIODIC
     &      , xi_rho,  xi_u
!# endif
!# ifndef NS_PERIODIC
     &      , eta_rho, eta_v
!# endif
#endif
#if defined SOLVE3D && defined TRACERS
     &     ,  nttclm, ntstf, nttsrc, ntbtf

#endif
     &      , ncidrst, nrecrst,  nrpfrst
     &      , rstTime, rstTime2, rstTstep, rstZ,    rstUb,  rstVb
#ifdef SOLVE3D
     &                         , rstU,    rstV
# ifdef TRACERS
     & ,   rstT
# endif
# if defined LMD_SKPP
     &      , rstHbl
# endif
# ifdef LMD_BKPP
     &      , rstHbbl
# endif
# if defined GLS_MIXING
     &      , rstAkv,rstAkt
#  if defined SALINITY
     &      , rstAks
#  endif
     &      , rstTke,rstGls
# endif
# ifdef M3FAST
#  if defined GLS_MIXING || defined LMD_MIXING
     &      , rstBustr,rstBvstr
#  endif
# endif
#ifdef EXACT_RESTART
     &      , rstrufrc,rstrvfrc
# ifdef M3FAST
     &      , rstru_nbq,rstrv_nbq
     &      , rstru_nbq_avg2,rstrv_nbq_avg2
     &      , rstqdmu_nbq,rstqdmv_nbq
# endif  /* M3FAST */
# ifdef TS_MIX_ISO_FILT
     &      , rstdRdx,rstdRde
# endif
#endif

# ifdef SEDIMENT
     &                         , rstSed
# endif
# ifdef MUSTANG
     &                         , rstMUS
# endif
#endif
#ifdef MORPHODYN
     &                         , rstHm
#endif
#ifdef BBL
     &                         , rstBBL
#endif
     &      , ncidhis, nrechis,  nrpfhis
     &      , hisTime, hisTime2, hisTstep, hisZ,    hisUb,  hisVb
     &      , hisBostr, hisWstr, hisUWstr, hisVWstr
     &      , hisBustr, hisBvstr
     &      , hisShflx, hisSwflx, hisShflx_rsw
     &      , hisBhflx, hisBwflx
# ifdef MORPHODYN
     &      , hisHm
#endif
#ifdef SOLVE3D
     &      , hisU,    hisV,     hisT,    hisR
     &      , hisO,    hisW,     hisVisc, hisDiff
     &      , hisAkv,  hisAkt,   hisAks
     &      , hisHbl,  hisHbbl
# if defined ANA_VMIX || defined BVF_MIXING \
  || defined LMD_MIXING || defined LMD_SKPP || defined LMD_BKPP \
  || defined GLS_MIXING
     &      , hisbvf
# endif
# ifdef GLS_MIXING
     &      , hisTke, hisGls, hisLsc
# endif
# if defined BULK_FLUX || defined OA_COUPLING
     &      , hisShflx_rlw
     &      , hisShflx_lat, hisShflx_sen
# endif
# ifdef SST_SKIN
     &      , hisSST_skin
# endif
# ifdef BIOLOGY
     &      , hisHel
#  ifdef BIO_NChlPZD
     &      , hisChC
#   ifdef OXYGEN
     &      , hisU10, hisKvO2, hisO2sat
#   endif
#  elif defined BIO_BioEBUS
     &      , hisAOU, hisWIND10
#  endif
# endif  /* BIOLOGY */
# ifdef SEDIMENT
     &      , hisSed
# endif
# ifdef MUSTANG
     &      , hisMust
# endif
#endif
#ifdef BBL
     &      , hisBBL
#endif
#ifdef DIAGNOSTICS_TS
     &      , nciddia, nrecdia, nrpfdia
     &      , diaTime, diaTime2, diaTstep
     &      , diaTXadv, diaTYadv, diaTVadv, diaTHmix
     &      , diaTVmix, diaTForc, diaTrate
#  ifdef DIAGNOSTICS_TSVAR
     &      , diaTVmixt
#  endif
# if defined DIAGNOSTICS_TS_MLD
     &      , diaTXadv_mld, diaTYadv_mld, diaTVadv_mld, diaTHmix_mld
     &      , diaTVmix_mld, diaTForc_mld, diaTrate_mld, diaTentr_mld
# endif
# ifdef AVERAGES
     &      , nciddia_avg, nrecdia_avg, nrpfdia_avg
     &      , diaTime_avg, diaTime2_avg, diaTstep_avg
     &      , diaTXadv_avg, diaTYadv_avg, diaTVadv_avg
     &      , diaTHmix_avg, diaTVmix_avg, diaTForc_avg
#  ifdef DIAGNOSTICS_TSVAR
     &      , diaTVmixt_avg
# endif
     &      , diaTrate_avg
#  ifdef DIAGNOSTICS_TS_MLD
     &      , diaTXadv_mld_avg, diaTYadv_mld_avg, diaTVadv_mld_avg
     &      , diaTHmix_mld_avg, diaTVmix_mld_avg, diaTForc_mld_avg
     &      , diaTrate_mld_avg, diaTentr_mld_avg
#  endif
# endif
#endif
#ifdef DIAGNOSTICS_UV
     &      , nciddiaM, nrecdiaM, nrpfdiaM
     &      , diaTimeM, diaTime2M, diaTstepM
     &      , diaMXadv, diaMYadv, diaMVadv, diaMCor
     &      , diaMPrsgrd, diaMHmix, diaMVmix, diaMVmix2, diaMrate
     &      , diaMHdiff
# ifdef DIAGNOSTICS_BARO
     &      , diaMBaro
# endif
# ifdef M3FAST
     &      , diaMfast
# endif
# ifdef MRL_WCI
     &      , diaMvf, diaMbrk, diaMStCo
     &      , diaMVvf, diaMPrscrt, diaMsbk
     &      , diaMbwf, diaMfrc
# endif
# ifdef AVERAGES
     &      , nciddiaM_avg, nrecdiaM_avg, nrpfdiaM_avg
     &      , diaTimeM_avg, diaTime2M_avg, diaTstepM_avg
     &      , diaMXadv_avg, diaMYadv_avg, diaMVadv_avg
     &      , diaMCor_avg, diaMPrsgrd_avg, diaMHmix_avg
     &      , diaMHdiff_avg
     &      , diaMVmix_avg, diaMVmix2_avg, diaMrate_avg
#  ifdef DIAGNOSTICS_BARO
     &      , diaMBaro_avg
#  endif
#  ifdef M3FAST
     &      , diaMfast_avg
#  endif
#  ifdef MRL_WCI
     &      , diaMvf_avg, diaMbrk_avg, diaMStCo_avg
     &      , diaMVvf_avg, diaMPrscrt_avg, diaMsbk_avg
     &      , diaMbwf_avg,diaMfrc_avg
#  endif
# endif
#endif
#ifdef DIAGNOSTICS_VRT
     &      , nciddiags_vrt, nrecdiags_vrt, nrpfdiags_vrt
     &      , diags_vrtTime, diags_vrtTime2, diags_vrtTstep
     &      , diags_vrtXadv, diags_vrtYadv, diags_vrtHdiff, diags_vrtCor
     &      , diags_vrtPrsgrd, diags_vrtHmix, diags_vrtVmix, diags_vrtrate
     &      , diags_vrtVmix2, diags_vrtWind, diags_vrtDrag
# ifdef DIAGNOSTICS_BARO
     &      , diags_vrtBaro
# endif
# ifdef M3FAST
     &      , diags_vrtfast
# endif
# ifdef AVERAGES
     &      , nciddiags_vrt_avg, nrecdiags_vrt_avg, nrpfdiags_vrt_avg
     &      , diags_vrtTime_avg, diags_vrtTime2_avg, diags_vrtTstep_avg
     &      , diags_vrtXadv_avg, diags_vrtYadv_avg, diags_vrtHdiff_avg
     &      , diags_vrtCor_avg, diags_vrtPrsgrd_avg, diags_vrtHmix_avg
     &      , diags_vrtVmix_avg, diags_vrtrate_avg
     &      , diags_vrtVmix2_avg, diags_vrtWind_avg, diags_vrtDrag_avg
#  ifdef DIAGNOSTICS_BARO
     &      , diags_vrtBaro_avg
#  endif
#  ifdef M3FAST
     &      , diags_vrtfast_avg
#  endif
# endif
#endif
#ifdef DIAGNOSTICS_EK
     &      , nciddiags_ek, nrecdiags_ek, nrpfdiags_ek
     &      , diags_ekTime, diags_ekTime2, diags_ekTstep
     &      , diags_ekHadv, diags_ekHdiff,  diags_ekVadv
     &      , diags_ekCor, diags_ekPrsgrd, diags_ekHmix
     &      , diags_ekVmix, diags_ekrate, diags_ekvol
     &      , diags_ekVmix2, diags_ekWind, diags_ekDrag
# ifdef DIAGNOSTICS_BARO
     &      , diags_ekBaro
# endif
# ifdef M3FAST
     &      , diags_ekfast
# endif
# ifdef AVERAGES
     &      , nciddiags_ek_avg, nrecdiags_ek_avg, nrpfdiags_ek_avg
     &      , diags_ekTime_avg, diags_ekTime2_avg, diags_ekTstep_avg
     &      , diags_ekHadv_avg, diags_ekHdiff_avg, diags_ekVadv_avg
     &      , diags_ekCor_avg, diags_ekPrsgrd_avg, diags_ekHmix_avg
     &      , diags_ekVmix_avg, diags_ekrate_avg, diags_ekvol_avg
     &      , diags_ekVmix2_avg, diags_ekWind_avg, diags_ekDrag_avg
#  ifdef DIAGNOSTICS_BARO
     &      , diags_ekBaro_avg
#  endif
#  ifdef M3FAST
     &      , diags_ekfast_avg
#  endif
# endif
# ifdef DIAGNOSTICS_EK_MLD
     &      , diags_ekHadv_mld, diags_ekHdiff_mld,  diags_ekVadv_mld
     &      , diags_ekCor_mld, diags_ekPrsgrd_mld, diags_ekHmix_mld
     &      , diags_ekVmix_mld, diags_ekrate_mld, diags_ekvol_mld
     &      , diags_ekVmix2_mld
#  ifdef DIAGNOSTICS_BARO
     &      , diags_ekBaro_mld
#  endif
#  ifdef AVERAGES
     &      , diags_ekHadv_mld_avg, diags_ekHdiff_mld_avg
     &      , diags_ekVadv_mld_avg, diags_ekCor_mld_avg
     &      , diags_ekPrsgrd_mld_avg, diags_ekHmix_mld_avg
     &      , diags_ekVmix_mld_avg, diags_ekrate_mld_avg
     &      , diags_ekvol_mld_avg, diags_ekVmix2_mld_avg
#   ifdef DIAGNOSTICS_BARO
     &      , diags_ekBaro_mld_avg
#   endif
#  endif
# endif
#endif
#ifdef DIAGNOSTICS_PV
     &      , nciddiags_pv, nrecdiags_pv, nrpfdiags_pv
     &      , diags_pvTime, diags_pvTime2, diags_pvTstep
# ifdef DIAGNOSTICS_PV_FULL
     &      , diags_pvpv, diags_pvpvd
# endif
     &      , diags_pvTrhs, diags_pvMrhs
# ifdef AVERAGES
     &      , nciddiags_pv_avg, nrecdiags_pv_avg, nrpfdiags_pv_avg
     &      , diags_pvTime_avg, diags_pvTime2_avg, diags_pvTstep_avg
#  ifdef DIAGNOSTICS_PV_FULL
     &      , diags_pvpv_avg, diags_pvpvd_avg
#  endif
     &      , diags_pvTrhs_avg, diags_pvMrhs_avg
# endif
#endif
# if defined DIAGNOSTICS_EDDY && ! defined XIOS
     &      , nciddiags_eddy, nrecdiags_eddy, nrpfdiags_eddy
     &      , diags_eddyTime, diags_eddyTstep
     &      , diags_eddyzz
     &      , diags_eddyuu, diags_eddyvv, diags_eddyuv, diags_eddyub
     &      , diags_eddyvb, diags_eddywb, diags_eddyuw, diags_eddyvw
     &      , diags_eddyubu, diags_eddyvbv
     &      , diags_eddyusu, diags_eddyvsv
# ifdef AVERAGES
     &      , nciddiags_eddy_avg, nrecdiags_eddy_avg, nrpfdiags_eddy_avg
     &      , diags_eddyTime_avg, diags_eddyTime2_avg, diags_eddyTstep_avg
     &      , diags_eddyzz_avg
     &      , diags_eddyuu_avg, diags_eddyvv_avg, diags_eddyuv_avg
     &      , diags_eddyub_avg, diags_eddyvb_avg, diags_eddywb_avg
     &      , diags_eddyuw_avg, diags_eddyvw_avg
     &      , diags_eddyubu_avg, diags_eddyvbv_avg
     &      , diags_eddyusu_avg, diags_eddyvsv_avg
# endif
#endif
# if defined OUTPUTS_SURFACE && ! defined XIOS
     &      , ncidsurf, nrecsurf, nrpfsurf
     &      , surfTime, surfTime2, surfTstep
     &      , surf_surft, surf_surfs,  surf_surfz
     &      , surf_surfu, surf_surfv
# ifdef AVERAGES
     &      , ncidsurf_avg, nrecsurf_avg, nrpfsurf_avg
     &      , surfTime_avg, surfTime2_avg, surfTstep_avg
     &      , surf_surft_avg, surf_surfs_avg,  surf_surfz_avg
     &      , surf_surfu_avg, surf_surfv_avg
# endif
#endif
#ifdef DIAGNOSTICS_BIO
     &      , nciddiabio, nrecdiabio, nrpfdiabio
     &      , diaTimebio, diaTime2bio, diaTstepbio, diabioFlux
     &      , diabioVSink
     &      , diabioGasExc
# ifdef AVERAGES
     &      , nciddiabio_avg, nrecdiabio_avg, nrpfdiabio_avg
     &      , diaTimebio_avg, diaTime2bio_avg, diaTstepbio_avg
     &      , diabioFlux_avg
     &      , diabioVSink_avg
     &      , diabioGasExc_avg
# endif
#endif

#ifdef AVERAGES
     &      , ncidavg,  nrecavg,  nrpfavg
     &      , avgTime, avgTime2, avgTstep, avgZ,    avgUb,  avgVb
     &      , avgBostr, avgWstr, avgUWstr, avgVWstr
     &      , avgBustr, avgBvstr
     &      , avgShflx, avgSwflx, avgShflx_rsw
     &      , avgBhflx, avgBwflx
# ifdef MORPHODYN
     &      , avgHm
# endif
# ifdef SOLVE3D
     &      , avgU,    avgV
#  if defined TRACERS
     &      ,     avgT
#  endif
     &      ,     avgR
     &      , avgO,    avgW,     avgVisc,  avgDiff
     &      , avgAkv,  avgAkt,   avgAks
     &      , avgHbl,  avgHbbl
#  if defined ANA_VMIX || defined BVF_MIXING \
 || defined LMD_MIXING || defined LMD_SKPP || defined LMD_BKPP \
 || defined GLS_MIXING
     &      , avgbvf
#  endif
#  ifdef GLS_MIXING
     &      , avgTke, avgGls, avgLsc
#  endif
#  ifdef BIOLOGY
     &      , avgHel
#   ifdef BIO_NChlPZD
     &      , avgChC
#    ifdef OXYGEN
     &      , avgU10, avgKvO2, avgO2sat
#    endif
#   elif defined BIO_BioEBUS
     &      , avgAOU, avgWIND10
#   endif
#  endif  /* BIOLOGY */
#  if defined BULK_FLUX || defined OA_COUPLING
     &      , avgShflx_rlw
     &      , avgShflx_lat, avgShflx_sen
#  endif
#  ifdef SST_SKIN
     &      , avgSST_skin
#  endif
#  ifdef SEDIMENT
     &      , avgSed
#  endif
# endif /* SOLVE3D */

# ifdef BBL
     &      , avgBBL
# endif
# ifdef WAVE_IO
     &      , avgWAVE
# endif
# ifdef MRL_WCI
     &      , avgSUP, avgUST2D, avgVST2D
#  ifdef SOLVE3D
     &      , avgUST, avgVST, avgAkb, avgAkw
     &      , avgKVF, avgCALP, avgKAPS, avgWST
#  endif
# endif
#endif /* AVERAGES */

     &      , wrthis
#ifdef AVERAGES
     &      , wrtavg
#endif
#ifdef DIAGNOSTICS_TS
     &      , wrtdia3D
     &      , wrtdia2D
# ifdef AVERAGES
     &      , wrtdia3D_avg
     &      , wrtdia2D_avg
# endif
#endif
#ifdef DIAGNOSTICS_UV
     &      , wrtdiaM
# ifdef AVERAGES
     &      , wrtdiaM_avg
# endif
#endif
#ifdef DIAGNOSTICS_VRT
     &      , wrtdiags_vrt
# ifdef AVERAGES
     &      , wrtdiags_vrt_avg
# endif
#endif
#ifdef DIAGNOSTICS_EK
     &      , wrtdiags_ek
# ifdef AVERAGES
     &      , wrtdiags_ek_avg
# endif
#endif
#ifdef DIAGNOSTICS_PV
     &      , wrtdiags_pv
# ifdef AVERAGES
     &      , wrtdiags_pv_avg
# endif
#endif
# if defined DIAGNOSTICS_EDDY && ! defined XIOS
     &      , wrtdiags_eddy
# ifdef AVERAGES
     &      , wrtdiags_eddy_avg
# endif
#endif
# if defined OUTPUTS_SURFACE && ! defined XIOS
     &      , wrtsurf
# ifdef AVERAGES
     &      , wrtsurf_avg
# endif
#endif
#ifdef DIAGNOSTICS_BIO
     &      , wrtdiabioFlux
     &      , wrtdiabioVSink
     &      , wrtdiabioGasExc
# ifdef AVERAGES
     &      , wrtdiabioFlux_avg
     &      , wrtdiabioVSink_avg
     &      , wrtdiabioGasExc_avg
# endif
#endif
      character*80 date_str, title, start_date
      character*80 origin_date, start_date_run
      integer      start_day, start_month, start_year
     &         ,   start_hour, start_minute, start_second
     &         ,   origin_day, origin_month, origin_year
     &         ,   origin_hour, origin_minute, origin_second

      REAL(kind=8)             :: origin_date_in_sec

      character*180 ininame,  grdname,  hisname
     &         ,   rstname,  frcname,  bulkname,  usrname
     &         ,   qbarname, tsrcname
     &         ,   btfname
#ifdef AVERAGES
     &                                ,  avgname
#endif
#ifdef DIAGNOSTICS_TS
     &                                ,  dianame
# ifdef AVERAGES
     &                                ,  dianame_avg
# endif
#endif
#ifdef DIAGNOSTICS_UV
     &                                ,  dianameM
# ifdef AVERAGES
     &                                ,  dianameM_avg
# endif
#endif
#ifdef DIAGNOSTICS_VRT
     &                                ,  diags_vrtname
# ifdef AVERAGES
     &                                ,  diags_vrtname_avg
# endif
#endif
#ifdef DIAGNOSTICS_EK
     &                                ,  diags_ekname
# ifdef AVERAGES
     &                                ,  diags_ekname_avg
# endif
#endif
#ifdef DIAGNOSTICS_PV
     &                                ,  diags_pvname
# ifdef AVERAGES
     &                                ,  diags_pvname_avg
# endif
#endif
# if defined DIAGNOSTICS_EDDY && ! defined XIOS
     &                                ,  diags_eddyname
# ifdef AVERAGES
     &                                ,  diags_eddyname_avg
# endif
#endif
# if defined OUTPUTS_SURFACE && ! defined XIOS
     &                                ,  surfname
# ifdef AVERAGES
     &                                ,  surfname_avg
# endif
#endif
#ifdef DIAGNOSTICS_BIO
     &                                ,  dianamebio
# ifdef AVERAGES
     &                                ,  dianamebio_avg
# endif
#endif
#if (defined TCLIMATOLOGY  && !defined ANA_TCLIMA)\
 || (defined ZCLIMATOLOGY  && !defined ANA_SSH)\
 || (defined M2CLIMATOLOGY && !defined ANA_M2CLIMA)\
 || (defined M3CLIMATOLOGY && !defined ANA_M3CLIMA)
     &                                ,   clmname
#endif
#ifdef FRC_BRY
     &                                ,   bry_file
#endif
#if defined WKB_WWAVE && !defined ANA_BRY_WKB
     &                                ,   brywkb_file
#endif
#ifdef WAVE_OFFLINE
     &                                ,   wave_file
#endif
#ifdef ASSIMILATION
     &                     ,   aparnam,   assname
#endif
#ifdef BIOLOGY
     &                                ,   bioname
#endif
#ifdef SEDIMENT
     &                                ,   sedname
#elif defined MUSTANG
     &               ,   sedname_subst,   sedname_must
#endif
#if defined SUBSTANCE && !defined MUSTANG
     &               ,    subsname
#endif

#ifdef SOLVE3D
      character*75  vname(20, 500)
#else
      character*75  vname(20, 90)
#endif

      common /cncscrum/   date_str,   title,  start_date
     &         ,   origin_date, start_date_run
     &         ,   ininame,  grdname, hisname
     &         ,   rstname,  frcname, bulkname,  usrname
     &         ,   qbarname, tsrcname
     &         ,   btfname, origin_date_in_sec
     &         ,   start_day, start_month, start_year
     &         ,   start_hour, start_minute, start_second
     &         ,   origin_day, origin_month, origin_year
     &         ,   origin_hour, origin_minute, origin_second
#ifdef AVERAGES
     &                                ,  avgname
#endif
#ifdef DIAGNOSTICS_TS
     &                                ,  dianame
# ifdef AVERAGES
     &                                ,  dianame_avg
# endif
#endif
#ifdef DIAGNOSTICS_UV
     &                                ,  dianameM
# ifdef AVERAGES
     &                                ,  dianameM_avg
# endif
#endif
#ifdef DIAGNOSTICS_VRT
     &                                ,  diags_vrtname
# ifdef AVERAGES
     &                                ,  diags_vrtname_avg
# endif
#endif
#ifdef DIAGNOSTICS_EK
     &                                ,  diags_ekname
# ifdef AVERAGES
     &                                ,  diags_ekname_avg
# endif
#endif
#ifdef DIAGNOSTICS_PV
     &                                ,  diags_pvname
# ifdef AVERAGES
     &                                ,  diags_pvname_avg
# endif
#endif
# if defined DIAGNOSTICS_EDDY && ! defined XIOS
     &                                ,  diags_eddyname
# ifdef AVERAGES
     &                                ,  diags_eddyname_avg
# endif
#endif
# if defined OUTPUTS_SURFACE && ! defined XIOS
     &                                ,  surfname
# ifdef AVERAGES
     &                                ,  surfname_avg
# endif
#endif
#ifdef DIAGNOSTICS_BIO
     &                                ,  dianamebio
# ifdef AVERAGES
     &                                ,  dianamebio_avg
# endif
#endif
#if (defined TCLIMATOLOGY  && !defined ANA_TCLIMA)\
 || (defined ZCLIMATOLOGY  && !defined ANA_SSH)\
 || (defined M2CLIMATOLOGY && !defined ANA_M2CLIMA)\
 || (defined M3CLIMATOLOGY && !defined ANA_M3CLIMA)
     &                                ,   clmname
#endif
#ifdef FRC_BRY
     &                                ,   bry_file
#endif
#if defined WKB_WWAVE && !defined ANA_BRY_WKB
     &                                ,   brywkb_file
#endif
#ifdef WAVE_OFFLINE
     &                                ,   wave_file
#endif
#ifdef ASSIMILATION
     &                     ,   aparnam,   assname
#endif
#ifdef SEDIMENT
     &                                ,   sedname
#elif defined MUSTANG
     &               ,   sedname_subst,   sedname_must
#endif
#if defined SUBSTANCE && !defined MUSTANG
     &               ,    subsname
#endif
#ifdef BIOLOGY
     &                                ,   bioname
#endif
     &                                ,   vname
