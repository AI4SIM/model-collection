"""
Functions for COARE model bulk flux calculations.
"""
import numpy as np


MM_DRYAIR = 28.9644e-3
MM_WATER = 18.0153e-3
C0 = 1.326e-5
C1 = 6.542e-3
C2 = 8.301e-6
C3 = 4.84e-9
ALPI = 32.62117980819471
BETAI = 6295.421338904806
GAMI = 0.5631331575423155
ALPW = 60.2227554
BETAW = 6822.40088
GAMW = 5.13926744
IP00 = 1.0e-5
BLK_RGAS = 287.0596736665907
BLK_CPA = 1004.708857833067
BLK_RVAP = 461.5249933083879
MM_DRYAIR = 28.9644e-3
MM_WATER = 18.0153e-3
R_GAS = 8.314510
G = 9.80665
RDOCPD = BLK_RGAS / BLK_CPA
MVOMA = 0.622
CHARN0=0.011
CHARN1=0.018
CHARNSLOPE=0.125

# ==== Physical parameters
CTOK = 273.16
EMISS_LW = 0.985
SIGMASB = 5.6697e-8
# ==== Atmosheric parameters
LWDWN = 300.0  # W/m2
SWDWN = 150.0  # W/m2
PRATE = 0.0  # precipitation rate (m/s)

# ==== Oceanic parameters
SSS = 36.0  # salinity psu
RHO0 = 1024.0  # Boussinesq density
CP = 3985.0  # oce heat capacity
RHO_I = 1.0 / RHO0  #
CPI = 1.0 / CP  #

IP00 = 1.0e-5
P00 = 1.0e5
BLK_BETA = 1.2
BLK_ZABL = 600.0
BLK_ZW = 10
VONKAR = 0.41
CPVIR = BLK_RVAP / BLK_RGAS - 1.0
EPS = 1.0e-08
BLK_ZTOTW = 1.0
CH10 = 0.00115


def bulk_psiu_coare(zol):
    """
    Computes the velocity structure function.
    Args:
        zol : relative height z/L

    Returns:
        psi: The velocity structure function
    """
    with np.errstate(invalid="ignore"):
        chic = np.where(
            zol < 0.0,
            np.power(1.0 - 10.15 * zol, (1.0 / 3.0)),
            -np.where(0.35 * zol < 50.0, 0.35 * zol, 50),
        )

        chik = np.where(zol <= 0.0, np.power(1.0 - 15.0 * zol, 0.25), np.nan)
        psik = np.where(
            zol <= 0.0,
            2.0 * np.log(0.5 * (1.0 + chik))
            + np.log(0.5 * (1.0 + chik * chik))
            - 2.0 * np.arctan(chik)
            + 0.5 * np.pi,
            np.nan,
        )
        psic = np.where(
            zol <= 0.0,
            1.5 * np.log((chic * chic + chic + 1.0) / 3.0)
            - np.sqrt(3.0) * np.arctan((2.0 * chic + 1.0) / np.sqrt(3.0))
            + np.pi / np.sqrt(3.0),
            np.nan,
        )

    return np.where(
        zol <= 0,
        psic + (psik - psic) / (1.0 + zol * zol),
        -((1.0 + zol) + 0.6667 * (zol - 14.28) * np.exp(chic) + 8.525),
    )


def bulk_psit_coare(zol):
    """
    Computes the temperature structure function.
    Args:
        zol : relative height z/L

    Returns:
        psi: The temperature structure function
    """
    mask = zol <= 0.0
    with np.errstate(invalid="ignore"):
        chic = np.where(
            mask,
            np.power(1.0 - 34.15 * zol, 1.0 / 3.0),
            -np.where(0.35 * zol < 50.0, 0.35 * zol, 50),
        )

        chik = np.where(mask, np.power(1.0 - 15.0 * zol, 0.25), np.nan)
        psik = np.where(zol <= 0.0, 2.0 * np.log(0.5 * (1.0 + chik * chik)), np.nan)
        psic = np.where(
            zol <= 0.0,
            1.5 * np.log((chic * chic + chic + 1.0) / 3.0)
            - np.sqrt(3.0) * np.arctan((2.0 * chic + 1.0) / np.sqrt(3.0))
            + np.pi / np.sqrt(3.0),
            np.nan,
        )

    return np.where(
        zol <= 0,
        psic + (psik - psic) / (1.0 + zol * zol),
        -(
            (1.0 + 2.0 * zol / 3.0) ** 1.5
            + 0.6667 * (zol - 14.28) * np.exp(chic)
            + 8.525
        ),
    )


def air_visc(t_air_c):
    """
    Computes the air viscosity given air temperatuce.
    Args:
        t_air_c : air temperature in Celsius

    Returns:
        air_visc: Air viscosity
    """
    cff = t_air_c * t_air_c
    return C0 * (1.0 + C1 * t_air_c + C2 * cff - C3 * cff * t_air_c)


def qsat(t_air_k, patm, ratio):
    """
    Computes the saturation humidity ratio of air.
    Args:
        t_air_k : air temperature in Kelvin
        patm : atmospheric pressure in Pa
        ratio : humidity radio in %
    Returns:
        qsat: The maximum saturation humidity ratio of air
    """
    psat = np.where(
        t_air_k <= CTOK,
        np.exp(ALPI - BETAI / t_air_k - GAMI * np.log(t_air_k)),
        np.exp(ALPW - BETAW / t_air_k - GAMW * np.log(t_air_k)),
    )
    psat = ratio * psat
    return (MVOMA * psat) / (patm + (MVOMA - 1) * psat)


def exner_patm_from_tairabs(q_air, t_air_k, psfc, nits):
    """
    Computes the atmospheric pressure
    Args:
        q_air : air humidity ratio
        t_air_k : air temperature in Kelvin
        psfc : surface air pressure in Pa
        nits : number of iterations
    Returns:
        iexn: ???
        pair: surface air pressure in Pa
    """
    pair = psfc
    for _ in range(nits):
        q_sat = qsat(t_air_k, pair, 1.0)
        fraction = MM_DRYAIR + (q_air / q_sat) * (MM_WATER - MM_DRYAIR)
        pair = psfc * np.exp(-G * fraction * BLK_ZW / (R_GAS * t_air_k))
    iexn = (pair * IP00) ** (-RDOCPD)
    return iexn, pair

def coare_croco(uwnd_r, vwnd_r, t_air_k, q_atm, psurf, t_sea_c, nits):
    """
    Computes ocean atmosphere fluxes using COARE bulk parametrization
    Args:
        uwnd_r : relative u-wind speed in m.s^-1
        vwnd_r : relative u-wind speed in m.s^-1
        t_air_k : surface air temperature in Kelvin
        q_atm : surface air humidity ratio in kg.kg^-1
        psurf : surface pressure in Pa
        t_sea_c : sea surface tenperature in Celsius
        nits : number of iterations
    Returns:
        sustr: u-wind stress
        svstr: v-wind stress
        hflat: latent heat flux
        hfsen : sensible heat flux
    """
    wspd0 = np.sqrt(uwnd_r * uwnd_r + vwnd_r * vwnd_r)
    wspd0 = np.where(wspd0 > 0.1 * min(10.0, BLK_ZW), wspd0, 0.1 * min(10.0, BLK_ZW))
    iexna, patm = exner_patm_from_tairabs(q_atm, t_air_k,  psurf, nits)
    q_sat = qsat(t_sea_c + CTOK, psurf, 0.98)
    del_w = np.sqrt(wspd0 * wspd0 + 0.25)
    cff = CTOK * (iexna - (psurf * IP00) ** (-RDOCPD))
    del_t = (t_air_k - CTOK) * iexna - t_sea_c * (psurf * IP00) ** (-RDOCPD) + cff
    w_star = 0.035 * del_w * np.log(10.0 * 10000.0) / np.log(BLK_ZW * 10000)
    vis_air = air_visc(t_air_k - CTOK)
    ribcu = -BLK_ZW / (BLK_ZABL * 0.004 * BLK_BETA**3)
    izo10 = G * w_star / (CHARN0 * w_star * w_star * w_star + 0.11 * G * vis_air)
    cff = 1.0 / (CH10 * np.log(10.0 * izo10))
    izot10 = 0.1 * np.exp(VONKAR * VONKAR * cff)
    cc_ = np.log(BLK_ZW * izo10) * np.log(BLK_ZW * izo10) / np.log(BLK_ZW * izot10)
    ri_ = G * BLK_ZW * (del_t + CPVIR * t_air_k * (q_atm - q_sat)) / (t_air_k * del_w * del_w)
    zolu = np.where(
        ri_ < 0.0, cc_ * ri_ / (1.0 + ri_ / ribcu), cc_ * ri_ / (1.0 + 3.0 * ri_ / cc_)
    )
    psi_u = bulk_psiu_coare(zolu)
    logus10 = np.log(BLK_ZW * izo10)
    w_star = del_w * VONKAR / (logus10 - psi_u)
    zolt = zolu * BLK_ZTOTW
    psi_t = bulk_psit_coare(zolt)
    logts10 = np.log(BLK_ZW * izot10)
    cff = VONKAR / (logts10 - psi_t)
    t_star = del_t * cff
    q_star = (q_atm - q_sat) * cff
    charn = np.where(
        del_w > 10.0, CHARN0 + CHARNSLOPE * (CHARN1 - CHARN0) * (del_w - 10.0), CHARN0
    )
    charn = np.where(del_w > 18.0, CHARN1, charn)

    # iterative process
    for _ in range(nits):
        izow = G * w_star / (charn * w_star * w_star * w_star + 0.11 * G * vis_air)
        rr_ = w_star / (izow * vis_air)
        izot = np.where(18181.8 * (rr_**0.6) > 8695.65, 18181.8 * (rr_**0.6), 8695.65)
        zolu = (
            VONKAR
            * G
            * BLK_ZW
            * (t_star * (1.0 + CPVIR * q_atm) + CPVIR * t_air_k * q_star)
            / (t_air_k * w_star * w_star * (1.0 + CPVIR * q_atm) + EPS)
        )
        psi_u = bulk_psiu_coare(zolu)
        logus10 = np.log(BLK_ZW * izow)
        w_star = del_w * VONKAR / (logus10 - psi_u)
        zolt = zolu * BLK_ZTOTW
        psi_t = bulk_psit_coare(zolt)
        logts10 = np.log(BLK_ZW * izot)
        cff = VONKAR / (logts10 - psi_t)
        t_star = del_t * cff
        q_star = (q_atm - q_sat) * cff
        bf_ = -G / t_air_k * w_star * (t_star + CPVIR * t_air_k * q_star)
        cff = np.where(bf_ > 0.0, np.power(BLK_BETA * (bf_ * BLK_ZABL), 1.0 / 3), 0.2)
        del_w = np.sqrt(wspd0 * wspd0 + cff * cff)

    c_drag = (w_star / del_w) ** 2
    c_drag_du = c_drag * del_w
    c_heat_du = VONKAR / (logts10 - psi_t) * np.sqrt(c_drag) * del_w

    # ==== Compute fluxes from bulk parameters
    wstar_tstar = c_heat_du * (t_air_k - CTOK - t_sea_c)  # Celsius * (m/s)
    wstar_qstar = c_heat_du * (q_atm - q_sat)
    rho_air = patm * (1.0 + q_atm) / (BLK_RGAS * t_air_k * (1.0 + MVOMA * q_atm))  # rho_abl
    hfsen = -BLK_CPA * rho_air * wstar_tstar  # W/m2
    hlv = (2.5008 - 0.0023719 * t_sea_c) * 1.0e6
    hflat = -hlv * rho_air * wstar_qstar
    upvel = -1.61 * wstar_qstar - (1.0 + 1.61 * q_atm) * wstar_tstar / t_air_k
    hflat = hflat + rho_air * hlv * upvel * q_atm  # W/m2
    sustr = rho_air * c_drag_du * uwnd_r  # m2/s2
    svstr = rho_air * c_drag_du * vwnd_r  # m2/s2
    return sustr, svstr, hflat, hfsen
