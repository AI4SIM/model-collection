import numpy as np
import xarray as xr
import sys

mm_dryair = 28.9644E-3
mm_water  = 18.0153E-3
MvoMa  = mm_water/mm_dryair
c0  = 1.326E-5
c1=6.542E-3
c2=8.301E-6
c3=4.84E-9
alpi   = 32.62117980819471
betai  = 6295.421338904806
gami = 0.5631331575423155
alpw   = 60.2227554
betaw  = 6822.40088
gamw = 5.13926744
ip00      = 1.E-5
blk_Rgas  = 287.0596736665907
blk_Cpa   = 1004.708857833067
blk_Rvap  = 461.5249933083879
mm_dryair = 28.9644E-3
mm_water  = 18.0153E-3
r_gas     = 8.314510
g         = 9.80665
rdocpd    = blk_Rgas/blk_Cpa
MvoMa     = 0.622

#==== Physical parameters
CtoK      = 273.16
emiss_lw  = 0.985
SigmaSB   = 5.6697E-8
#==== Atmosheric parameters
lwdwn     = 300.      # W/m2
swdwn     = 150.      # W/m2
prate     = 0.      # precipitation rate (m/s)
#==== Oceanic parameters
sss       = 36.    # salinity psu
rho0      = 1024.  # Boussinesq density
cp        = 3985.  # oce heat capacity
rho0i     = 1./rho0   #
cpi       = 1./cp     #
# parameters
g           = 9.81
psurf       = 100000.

ip00        = 1.E-5
p00         = 1.E+5
MvoMa       = 0.622
rdocpd      = blk_Rgas/blk_Cpa
blk_beta    = 1.2
blk_Zabl    = 600.
vonKar      = 0.41
cpvir       = blk_Rvap/blk_Rgas - 1.
eps         = 1.E-08
blk_ZToZW   = 1.
Ch10        = 0.00115


def bulk_psiu_coare(zol):
    with np.errstate(invalid='ignore'):
        chic = np.where(
            zol < 0.,
            np.power(1.-10.15*zol,(1./3.)),
            -np.where(0.35*zol < 50., 0.35*zol, 50))

        chik = np.where(zol <= 0.,
                        np.power(1.-15.*zol,0.25),
                        np.nan)
        psik = np.where(zol <= 0.,
                        2.*np.log(0.5*(1.+chik))+np.log(0.5*(1.+chik*chik))-2.*np.arctan(chik)+0.5*np.pi,
                        np.nan)
        psic = np.where(zol <= 0.,
                        1.5*np.log( (chic*chic+chic+1.)/3.)- np.sqrt(3.)*np.arctan((2.*chic+1.)/np.sqrt(3.))+np.pi/np.sqrt(3.),
                        np.nan)
    
    return np.where(zol <= 0,
                    psic+(psik-psic)/(1.+zol*zol),
                    -( (1.+zol)+0.6667*(zol-14.28)*np.exp(chic)+8.525))


def bulk_psit_coare(zol):
    mask = zol <= 0.
    with np.errstate(invalid='ignore'):
        chic = np.where(mask,
                        np.power(1.-34.15*zol, 1./3.),
                        -np.where(0.35*zol < 50., 0.35*zol, 50))

        chik = np.where(mask,
                        np.power(1.-15.*zol,0.25),
                        np.nan)
        psik = np.where(zol <= 0.,
                        2.*np.log(0.5*(1.+chik*chik)),
                        np.nan)
        psic = np.where(zol <= 0.,
                        1.5*np.log( (chic*chic+chic+1.)/3.)-np.sqrt(3.)*np.arctan((2.*chic+1.)/np.sqrt(3.))+np.pi/np.sqrt(3.),
                        np.nan)    
    
    return np.where(zol <= 0,
                    psic+(psik-psic)/(1.+zol*zol),
                    -((1.0+2.0*zol/3.0)**1.5+0.6667*(zol-14.28)*np.exp(chic)+8.525))

def air_visc(TairC):
    cff = TairC*TairC
    return c0*(1.+c1*TairC+c2*cff-c3*cff*TairC)

def qsat(TairK, patm, coeff):
    psat = np.where(TairK <= CtoK, 
                    np.exp(alpi - betai/TairK - gami*np.log(TairK)),
                    np.exp(alpw - betaw/TairK - gamw*np.log(TairK)))
    psat = coeff * psat
    return (MvoMa*psat)/(patm+(MvoMa-1)*psat)

def exner_patm_from_tairabs(q,tairabs,z,psfc,nits):

    pair = psfc
    for _ in range(nits):
        q_sat = qsat(tairabs, pair, 1.)
        xm    =  mm_dryair + (q/q_sat) * ( mm_water - mm_dryair )
        pair  = psfc * np.exp( -g * xm * z / ( r_gas * tairabs ) )
    iexn =  (pair*ip00)**(-rdocpd)
    return iexn,pair

def spec_hum(RH,psfc,TairC):
    cff=(1.0007+3.46E-6 * 0.01*psfc)*6.1121*np.exp(17.502*TairC/(240.97+TairC))
    if(RH<2):
        cff = cff*RH
        return MvoMa*(cff/(psfc*0.01-0.378*cff))
    else:
        return 0.001*RH
    
def iteration(Wstar, Tstar, Qstar, delW, delT, TairK, Qsea, qatm, wspd0, charn, VisAir, blk_ZW):
    ZoLu    = vonKar*g*blk_ZW*(Tstar*(1.0+cpvir*qatm)+cpvir*TairK*Qstar) / (TairK*Wstar*Wstar*(1.0+cpvir*qatm)+eps)
    psi_u   = bulk_psiu_coare(ZoLu)
    psi_t   = bulk_psiu_coare(ZoLu*blk_ZToZW)
    del ZoLu
    iZoW    = g*Wstar / ( charn*Wstar*Wstar*Wstar+0.11*g*VisAir )
    logus10 = np.log(blk_ZW*iZoW)
    iZoT    = Wstar/(iZoW*VisAir)
    del iZoW
    iZoT    = np.where(8695.65 > 18181.8*(iZoT**0.6), 8695.65, 18181.8*(iZoT**0.6))
    logts10 = np.log(blk_ZW*iZoT)
    del iZoT

    cff     = vonKar/(logts10-psi_t)
    Wstar   = delW*vonKar/(logus10-psi_u)
    del logus10
    Tstar   = delT*cff
    Qstar   = (qatm-Qsea)*cff
    
    Bf=-g/TairK*Wstar*(Tstar+cpvir*TairK*Qstar)
    with np.errstate(invalid='ignore'):
        cff = np.where(Bf > 0.0,
                       np.power(blk_beta*(Bf*blk_Zabl),1./3),
                       0.2)
    del Bf
    delW  = np.sqrt(wspd0*wspd0+cff*cff)
    del cff
    
    return Wstar, Tstar, Qstar, delW, logts10, psi_t#, psi_u
    
def coare_croco(uatm,vatm,uoce,voce,TairK,qatm,blk_ZW,TseaC,nits):
    VisAir      = air_visc ( TairK - CtoK )
    Qsea        = qsat(TseaC + CtoK,psurf,0.98) # ssq_abl
    
    iexns       = (psurf*ip00)**(-rdocpd)
    iexna,patm  = exner_patm_from_tairabs(qatm,TairK,blk_ZW,psurf,nits)
    cff         = CtoK*(iexna-iexns)
    delT        = (TairK - CtoK)*iexna - TseaC*iexns + cff
    du          = uatm-uoce
    dv          = vatm-voce
    wspd0       = np.sqrt( du*du+dv*dv)
    wspd0       = np.where(wspd0 > 0.1 * min(10.,blk_ZW), wspd0, 0.1 * min(10.,blk_ZW))
    delW        = np.sqrt(wspd0*wspd0+0.25)
    
    Wstar       = 0.035*delW*np.log(10.0*10000.)/np.log(blk_ZW*10000)
    
    
    Ribcu       = - blk_ZW / ( blk_Zabl * 0.004 * blk_beta**3 )
    iZo10       = g*Wstar / (0.011*Wstar*Wstar*Wstar+0.11*g*VisAir)
    cff         = 1./( Ch10*np.log( 10.0*iZo10) )
    iZoT10      = 0.1 * np.exp( vonKar*vonKar*cff )
    CC          = np.log( blk_ZW*iZo10 )*np.log( blk_ZW*iZo10 )/ np.log( blk_ZW*iZoT10 )
    
    ZoLu         = g * blk_ZW * ( delT+cpvir*TairK*(qatm-Qsea) )/( TairK*delW*delW )
    ZoLu        = np.where(ZoLu < 0., CC*ZoLu/(1.0+ZoLu/Ribcu), CC*ZoLu/(1.0+3.0*ZoLu/CC))
    del CC
    psi_t       = bulk_psit_coare(ZoLu*blk_ZToZW)
    psi_u       = bulk_psiu_coare(ZoLu)

    Wstar       = delW*vonKar/(np.log(blk_ZW*iZo10)-psi_u)
    
    logts10     = np.log(blk_ZW*iZoT10)
    cff         = vonKar/(logts10-psi_t)
    Tstar       = delT*cff
    Qstar       = (qatm-Qsea)*cff
    charn = np.where(delW > 10.0,
                     0.011+0.125*(0.018-0.011)*(delW-10.),
                     0.011)
    charn = np.where(delW > 18.0,
                     0.018,
                     charn)
# iterative process
    for i in range(nits):
        Wstar, Tstar, Qstar, delW, logts10, psi_t = iteration(Wstar, Tstar, Qstar, delW, delT, TairK, Qsea, qatm, wspd0, charn, VisAir, blk_ZW)


        
    Cd_du     =  (Wstar/delW)**2 * delW
    Ch_du     =  vonKar/(logts10-psi_t) * np.sqrt((Wstar/delW)**2) * delW
    Ce_du     =  vonKar/(logts10-psi_t) * np.sqrt((Wstar/delW)**2) * delW

    #==== Compute fluxes from bulk parameters
    #WstarTstar = Ch_du*(TairK-CtoK-TseaC)  # Celsius * (m/s)
    #WstarQstar = Ce_du*(qatm -Qsea)
    rhoAir      = patm*(1.+qatm) / ( blk_Rgas*TairK*(1.+MvoMa*qatm) )    # rho_abl
    #hfsen      = - blk_Cpa*rhoAir*WstarTstar  # W/m2
    #Hlv        = (2.5008 - 0.0023719*TseaC)*1.0E+6
    #hflat      = - Hlv*rhoAir*WstarQstar
    #upvel      = -1.61*WstarQstar-(1.0+1.61*qatm)*WstarTstar/TairK
    #hflat      = hflat+rhoAir*Hlv*upvel*qatm # W/m2
    # Convert from W/m2 to Celsius/(m/s)
    #hflat=-hflat*rho0i*cpi
    #hfsen=-hfsen*rho0i*cpi   # Celsius * (m/s)
    # radiative fluxes
    #hflw       = rho0i*cpi*(lwdwn - emiss_lw*SigmaSB*np.power(TseaC + CtoK,4))
    #hfsw       = rho0i*cpi*swdwn
    #cff        = rhoAir*rho0i
    sustr      = rhoAir*rho0i * Cd_du * uatm    # m2/s2
    svstr      = rhoAir*rho0i * Cd_du * vatm    # m2/s2
    # net heat flux (solar + nonsolar)
    # stflx      = hfsw+hflw+hflat+hfsen
    # Salinity flux 
    
    #evap=-cp*hflat/Hlv
    #ssflx=(evap-prate)*sss   # psu * (m/s)

    #del Cd_du, Ch_du, Ce_du, Wstar, Tstar, Qstar, delW, delT, Qsea, wspd0, charn, VisAir
    return (sustr, svstr)#, svstr#Cd_du,Ch_du,Ce_du,rhoAir,Qsea

