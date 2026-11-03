#!/usr/bin/env python3
"""Acoustic cavitation of argon bubble in molten NaCl with the 10-reaction NaCl-vapor
chemistry of nacl_kinetics.py run kinetically inside the bubble."""

'''
USAGE INSTRUCTIONS (also in README.md):
type "python simulation/nacl.py [options]" in the terminal.


All options are optional; if no options are entered, the simulation will 
just use the defaults given below.


available options:

-------------------- BASIC OPTIONS ------------------------------------


--R0 [initial radius in microns]
radius of the bubble at ambient pressure (1atm); default 5 microns.


--f [frequency in khz]
frequency of the sound wave with which the melt is irradiated. default 20khz.


--pA [pressure amplitude in atm]
amplitude of the sound wave. default 1.2atm.


--cycles [number of cycles]
number of cycles to run. default is three.


--T0 [temperature in kelvin]
ambient melt temperature. melting point of NaCl is 1074K. default T0 is 1100K.


-------------------- MORE ADVANCED ----------------------------------


--lambda-max [number, e.g., 1e16]
rate limiter. sometimes, with particularly intense collapses, scipy's Radau
integrator can fail (i.e., it can require a timestep below what
floating point numbers can resolve). this is 
because the calculated rates of certain reactions are incredibly high.
--lambda-max caps how fast each reaction can relax towards equilibrium [1/s]; 
the equilibrium itself is unchanged, so results are unaffected as long 
as 1/λ_max is much shorter than the collapse timescale.

use case: if the simulation fails or gives an error you don't understand,
try adding --lambda-max 1e16 or, if that doesn't work, 1e14 to fix it.
this should leave the results unaffected. you can test whether it does or
not by lowering it by one order of magnitude and seeing if the results change
(e.g., if 1e16 gives you 15000K and 1e15 also gives you 15000K, you know
1e16 is sufficiently high that everything remains in equilibrium).


--ion-speedup [factor, e.g. 1e6]
speeds up the ionization rates by some factor. if the saha equilibrium
(infinite-rate limit) is desired, then >= ~1e6 should be used.
if negligible ionization is desired, <= ~1e-6 should be used.


--electron-kappa
if entered, then calculates the electron thermal conductivity using a 
crude overestimate.


--csv
if entered, the simulation will write a CSV of the results into the
output folder. it does not generate CSVs by default.


--out [output folder name; string]
output folder name; if omitted, will default to, e.g.,
"nacl_R0-[R0]um_f-[f]kHz_pA-[pA]atm"


--no-plot
if entered, the simulation will not produce plots. it
does produce plots by default.


'''





import argparse


import math
import os
import shutil
import time
import warnings
from dataclasses import dataclass, field

import numpy as np
from scipy.integrate import solve_ivp

# radau artifact
warnings.filterwarnings("ignore", message="overflow encountered in multiply",
                        category=RuntimeWarning,
                        module=r"scipy\.integrate\._ivp\.common")

import nacl_kinetics as nk

# constants
KB  = nk.KB
NAV = nk.NA_AVOGADRO
ATM = 101325.0
EV  = nk.EV
SQRT_M_ME = math.sqrt(39.948 * 1.66053906660e-27 / 9.1093837015e-31)  # ~270

T_MELT = 1074.0

# reference conditions of the diffusivities D0_1..D0_5 (fixed, not T0)
D_REF_T = 1100.0

SPECIES = ["NaCl", "Na2Cl2", "Na", "Cl", "Na+", "Cl-", "Cl+", "e-",
           "Ar", "Ar+"]
KWNAMES = ["n_NaCl", "n_Na2Cl2", "n_Na", "n_Cl", "n_Nap", "n_Clm", "n_Clp",
           "n_e", "n_Ar", "n_Arp"]
IDX = {s: i for i, s in enumerate(SPECIES)}
NSP = len(SPECIES)
IE_ = IDX["e-"]
HEAVY = np.array([k for k in range(NSP) if k != IE_])
NST = 3 + NSP + 4                # states: R, U, T, species, S_NaCl..S_Cl

STOICH = {
    "R1": {"NaCl": -2.0, "Na2Cl2": +1.0},
    "R2": {"NaCl": -1.0, "Na": +1.0, "Cl": +1.0},
    "R3": {"NaCl": -1.0, "Na+": +1.0, "Cl-": +1.0},
    "R4": {"Na": -1.0, "Na+": +1.0, "e-": +1.0},
    "R5": {"Cl": -1.0, "e-": -1.0, "Cl-": +1.0},
    "R6": {"Cl": -1.0, "Cl+": +1.0, "e-": +1.0},
    "R7": {"Na+": -1.0, "Cl-": -1.0, "Na": +1.0, "Cl": +1.0},
    "R8": {"Cl+": -1.0, "Cl-": -1.0, "Cl": +2.0},
    "R9": {"Ar": -1.0, "Ar+": +1.0, "e-": +1.0},
    "R10": {"Ar+": -1.0, "Cl-": -1.0, "Ar": +1.0, "Cl": +1.0},
}

IE_AR = nk.IE_AR


def _module_rate(fn):
    """Count Ar and Ar+ under "M" for the reactions in nacl_kinetics.py."""
    def wrapped(T, n_Ar=0.0, n_Arp=0.0, n_inert=0.0, **kw):
        return fn(T, n_inert=n_inert + n_Ar + n_Arp, **kw)
    wrapped.__name__ = "wrapped_" + fn.__name__
    return wrapped


RATEFN = {
    "R1": _module_rate(nk.rate_R1_dimerization),
    "R2": _module_rate(nk.rate_R2_dissociation),
    "R3": _module_rate(nk.rate_R3_ionpair),
    "R4": _module_rate(nk.rate_R4_Na_ionization),
    "R5": _module_rate(nk.rate_R5_Cl_attachment),
    "R6": _module_rate(nk.rate_R6_Cl_ionization),
    "R7": _module_rate(nk.rate_R7_mutual_neutralization),
    "R8": _module_rate(nk.rate_R8_ClpClm),
    "R9": nk.rate_R9_Ar_ionization,
    "R10": nk.rate_R10_ArpClm,
}
RXNS = list(STOICH)
NRX = len(RXNS)

NU = np.zeros((NRX, NSP))
for _i, _r in enumerate(RXNS):
    for _s, _v in STOICH[_r].items():
        NU[_i, IDX[_s]] = _v
NUMASK = [NU[i] != 0.0 for i in range(NRX)]

FD_REL = 1.0e-6          # relative step of the throttle's finite difference

# =========================================================================
# PER-SPECIES ENERGETICS
# =========================================================================
E0 = np.zeros(NSP)
E0[IDX["NaCl"]]   = -nk.D0_NACL * EV
E0[IDX["Na2Cl2"]] = -(2.0 * nk.D0_NACL + nk.E_DIM) * EV
E0[IDX["Na+"]]    = +nk.IE_NA * EV
E0[IDX["Cl-"]]    = -nk.EA_CL * EV
E0[IDX["Cl+"]]    = +nk.IE_CL * EV
E0[IDX["Ar+"]]    = +IE_AR * EV

_TH_NACL  = [nk.C2 * w for w in nk.NACL_FREQ]
_TH_DIMER = [nk.C2 * w for w in nk.NA2CL2_FREQS]
_LV = {s: tuple((nk.C2 * E, g) for E, g in lv) for s, lv in (
    ("Na",  nk.NA_LEVELS), ("Cl",  nk.CL_LEVELS), ("Cl+", nk.CLP_LEVELS),
    ("Ar",  nk.AR_LEVELS), ("Ar+", nk.ARP_LEVELS))}


def _elec_u_cv(T, levels):
    """Electronic excitation energy and heat capacity per particle."""
    Z = S1 = S2 = 0.0
    for th, g in levels:
        x = th / T
        if x < 600.0:
            w = g * math.exp(-x)
            Z += w
            S1 += w * th
            S2 += w * th * th
    if Z == 0.0:
        return 0.0, 0.0
    m = S1 / Z
    return KB * m, KB * (S2 / Z - m * m) / (T * T)


def _vib_u_cv(T, thetas):
    """Harmonic-oscillator energy (from the ZPE) and cv per particle."""
    u = cv = 0.0
    for th in thetas:
        x = th / T
        if x < 600.0:
            em = math.expm1(x)
            u += KB * th / em
            cv += KB * x * x * (em + 1.0) / (em * em)
    return u, cv


def uint_cvint(T):
    """Internal (rot+vib+electronic) energy and cv per particle, all species.
    Translational 3/2 kB T is NOT included."""
    u = np.zeros(NSP)
    cv = np.zeros(NSP)
    uv, cvv = _vib_u_cv(T, _TH_NACL)
    u[IDX["NaCl"]], cv[IDX["NaCl"]] = KB * T + uv, KB + cvv
    uv, cvv = _vib_u_cv(T, _TH_DIMER)
    u[IDX["Na2Cl2"]], cv[IDX["Na2Cl2"]] = 1.5 * KB * T + uv, 1.5 * KB + cvv
    for s in ("Na", "Cl", "Cl+", "Ar", "Ar+"):
        u[IDX[s]], cv[IDX[s]] = _elec_u_cv(T, _LV[s])
    return u, cv


# =========================================================================
# DENSE-GAS THERMAL CONDUCTIVITY  (Barker-Henderson diameter + Enskog,
# Bastea arXiv:cond-mat/0401153, Eqs. 1, 4c, 9b)
# =========================================================================
_EPS_AR, _RM_AR, _ALPHA_AR = 122.0 * KB, 3.85e-10, 13.2


def _u_exp6(r):
    a = _ALPHA_AR
    return _EPS_AR * (6.0 / (a - 6.0) * np.exp(a * (1.0 - r / _RM_AR))
                      - a / (a - 6.0) * (_RM_AR / r) ** 6)


def _build_d_bh():
    """Barker-Henderson diameter d(T), tabulated in ln T as a cubic spline."""
    from scipy.optimize import brentq, minimize_scalar
    from scipy.integrate import quad
    from scipy.interpolate import CubicSpline
    r_hump = minimize_scalar(lambda r: -_u_exp6(r), method="bounded",
                             bounds=(0.05 * _RM_AR, 0.8 * _RM_AR)).x
    r_zero = brentq(_u_exp6, 1.01 * r_hump, _RM_AR)
    lnT = np.linspace(math.log(50.0), math.log(3.0e5), 300)
    d = [r_hump + quad(lambda r: -math.expm1(-_u_exp6(r) / (KB * math.exp(x))),
                       r_hump, r_zero, limit=200)[0] for x in lnT]
    return CubicSpline(lnT, d), lnT[0], lnT[-1]


_D_BH, _LNT_LO, _LNT_HI = _build_d_bh()


def enskog_factor(T, n_heavy):
    d = float(_D_BH(min(max(math.log(T), _LNT_LO), _LNT_HI)))
    eta = min(n_heavy * math.pi * d ** 3 / 6.0, 0.5)
    y = 4.0 * eta
    g_c = (1.0 - 0.5 * eta) / (1.0 - eta) ** 3
    return 1.0 / g_c + 1.2 * y + 0.7574 * y * y * g_c


# =========================================================================
# MOLTEN-NaCl LIQUID AND ITS VAPOR
# =========================================================================
def p_sat_nacl(T):
    """Total saturation vapor pressure of molten NaCl [Pa] (NIST Antoine:
    Ewing & Stern 1974 for T <= 1200 K, Stull 1947 above)."""
    if T <= 1200.0:
        A, B, C = 3.56682, 5200.904, -317.409
    else:
        A, B, C = 5.07184, 8388.497, -82.638
    return 1.0e5 * 10.0 ** (A - B / (T + C))


def liquid_properties(T0):
    if T0 < T_MELT:
        raise ValueError(f"T0 = {T0} K is below the melting point of NaCl "
                         f"({T_MELT} K).")
    if T0 > 1300.0:
        print("WARNING: liquid-property fits extrapolated beyond ~1300 K.")
    rho   = 2139.3 - 0.5430 * T0
    sigma = 0.191 - 7.188e-5 * T0
    mu    = 8.931e-5 * math.exp(2641.0 / T0)
    c     = 2483.0 - 0.915 * (T0 - 273)     # Bockris and Richards, 1957
    return rho, c, mu, sigma


@dataclass
class Params:
    # Liquid (overwritten from T0 in setup())
    rho:   float = 1542.0
    c:     float = 1767.0
    mu:    float = 9.8e-4
    sigma: float = 0.112
    p0:    float = ATM
    T0:    float = 1100.0

    # Acoustic driving  p_ac(t) = -pA sin(2 pi f t + phase)
    f:     float = 20.0e3
    pA:    float = 1.2 * ATM
    phase: float = 0.0

    R0: float = 5.0e-6

    # co-volumes [m^3/particle]; estimates from b ~ 16/3 pi r^3, except for 
    # argon which is tabulated.
    b_sp: tuple = (172.1e-6 / NAV,   # NaCl
                   344.2e-6 / NAV,   # Na2Cl2
                   118.0e-6 / NAV,   # Na
                   54.1e-6 / NAV,    # Cl
                   118.0e-6 / NAV,   # Na+
                   54.1e-6 / NAV,    # Cl-
                   54.1e-6 / NAV,    # Cl+
                   0.0,              # e-
                   5.32e-29,         # Ar
                   5.32e-29)         # Ar+

    # Transport (Chapman-Enskog estimates; approximate)
    kA_sp: tuple = (0.0046, 0.0039, 0.0075, 0.0171, 0.0075, 0.0171, 0.0171,
                    0.0, 0.0177, 0.0177)
    s_k:   float = 0.7
    D0_1:  float = 0.94e-4     # NaCl-Ar
    D0_2:  float = 6.4e-5      # Na2Cl2-Ar
    D0_3:  float = 1.49e-4     # Na-Ar
    D0_4:  float = 1.65e-4     # Cl-Ar
    D0_5:  float = 8.0e-5      # Ar+-Ar

    electron_kappa: bool = False   # kappa_e = x sqrt(M_Ar/m_e) kappa (upper bound)
    lambda_max: float = float("inf")   # stiffness throttle; ~1e16 if a run stalls
    ion_speedup: float = 1.0           # multiplier on R4/R5/R6/R9

    # Derived, filled by setup()
    N_ar:    float = field(default=0.0, init=False)
    n_ref:   float = field(default=0.0, init=False)
    n1_sat:  float = field(default=0.0, init=False)
    n2_sat:  float = field(default=0.0, init=False)
    n3_sat:  float = field(default=0.0, init=False)
    n4_sat:  float = field(default=0.0, init=False)
    h1_wall: float = field(default=0.0, init=False)
    h2_wall: float = field(default=0.0, init=False)
    h3_wall: float = field(default=0.0, init=False)
    h4_wall: float = field(default=0.0, init=False)
    h5_wall: float = field(default=0.0, init=False)
    p_blake: float = field(default=0.0, init=False)
    T_peak_rhs:  float = field(default=0.0, init=False)
    clamp_hi_hit: bool = field(default=False, init=False)
    highly_dubious: float = field(default=0.0, init=False)


def setup(pr: Params):
    """Liquid properties, wall vapor densities and initial bubble contents."""
    pr.rho, pr.c, pr.mu, pr.sigma = liquid_properties(pr.T0)

    psat     = p_sat_nacl(pr.T0)
    pr.n_ref = pr.p0 / (KB * D_REF_T)

    # wall vapor split:  K1 n1^2 + n1 = n_tot  (stable root)
    n_tot = psat / (KB * pr.T0)
    K1 = nk.K_dimer(pr.T0)
    pr.n1_sat = 2.0 * n_tot / (1.0 + math.sqrt(1.0 + 4.0 * K1 * n_tot))
    pr.n2_sat = K1 * pr.n1_sat ** 2
    pr.n3_sat = pr.n4_sat = math.sqrt(nk.K_dissoc(pr.T0) * pr.n1_sat)

    ui0, _ = uint_cvint(pr.T0)
    pr.h1_wall = 2.5 * KB * pr.T0 + ui0[IDX["NaCl"]]
    pr.h2_wall = 2.5 * KB * pr.T0 + ui0[IDX["Na2Cl2"]]
    pr.h3_wall = 2.5 * KB * pr.T0 + ui0[IDX["Na"]]
    pr.h4_wall = 2.5 * KB * pr.T0 + ui0[IDX["Cl"]]
    pr.h5_wall = 2.5 * KB * pr.T0 + ui0[IDX["Ar"]]

    V0   = (4.0 / 3.0) * math.pi * pr.R0 ** 3
    p_eq = pr.p0 + 2.0 * pr.sigma / pr.R0
    if p_eq <= psat:
        raise ValueError("Initial pressure below vapor pressure, "
                         "no stable gas bubble for these parameters.")

    N1, N2 = pr.n1_sat * V0, pr.n2_sat * V0
    b = np.asarray(pr.b_sp)
    Bv = N1 * b[0] + N2 * b[1]
    pr.N_ar = ((p_eq * (V0 - Bv) - (N1 + N2) * KB * pr.T0)
               / (KB * pr.T0 + p_eq * b[IDX["Ar"]]))

    Kb = (p_eq - psat) * pr.R0 ** 3
    Rc = math.sqrt(3.0 * Kb / (2.0 * pr.sigma))
    pr.p_blake = pr.p0 - psat + 4.0 * pr.sigma / (3.0 * Rc)

    y0 = np.zeros(NST)
    y0[0], y0[1], y0[2] = pr.R0, 0.0, pr.T0
    y0[3 + IDX["NaCl"]] = N1
    y0[3 + IDX["Na2Cl2"]] = N2
    y0[3 + IDX["Na"]] = pr.n3_sat * V0
    y0[3 + IDX["Cl"]] = pr.n4_sat * V0
    y0[3 + IDX["Ar"]] = pr.N_ar
    return y0


# =========================================================================
# CHEMISTRY WITH THE STIFFNESS THROTTLE
# =========================================================================
ION_RXNS = ("R4", "R5", "R6", "R9")


def reaction_rates(T, n, lambda_max, ion_speedup=1.0):
    n = np.maximum(n, 0.0)
    kw = dict(zip(KWNAMES, n))
    rates = np.zeros(NRX)
    throttling = np.isfinite(lambda_max)

    for i, rname in enumerate(RXNS):
        fn, nu, m = RATEFN[rname], NU[i], NUMASK[i]
        mult = ion_speedup if rname in ION_RXNS else 1.0
        r0 = fn(T, **kw) * mult
        if not throttling:
            rates[i] = r0
            continue
        scale = (n[m] / np.abs(nu[m])).max()
        if scale <= 0.0:
            rates[i] = r0
            continue
        d = FD_REL * scale
        kw2 = dict(zip(KWNAMES, np.maximum(n + d * nu, 0.0)))
        lam = -(fn(T, **kw2) * mult - r0) / d
        if lam > 0.0:
            r0 *= lambda_max / (lambda_max + lam)
        rates[i] = r0
    return rates


def _bl(D, R, aU, cap):
    """Toegel et al. (2000) boundary-layer thickness, capped at R/pi."""
    d = math.sqrt(D * R / aU)
    return d if d < cap else cap


def rhs(t, y, pr: Params):
    R, U, T = y[0], y[1], y[2]
    N = np.maximum(y[3:3 + NSP], 0.0)
    if R < 1e-9:
        R = 1e-9
    if T < 5.0:
        T = 5.0

    Ne  = N[IE_]
    Nhv = float(N[HEAVY].sum())
    Ntr = Nhv + Ne
    V   = (4.0 / 3.0) * math.pi * R * R * R
    dV  = 4.0 * math.pi * R * R * U
    A   = 4.0 * math.pi * R * R

    # ---- equation of state (hard core; electrons point-like) -------------
    b = pr.b_sp
    B = float(np.dot(N, b))
    Vfree = max(V - B, 1e-6 * B)
    p_g = Ntr * KB * T / Vfree

    # ---- caloric properties ----------------------------------------------
    ui, ci = uint_cvint(T)
    Cv = 1.5 * KB * Ntr + float(N @ ci)

    # ---- transport --------------------------------------------------------
    kap_dilute = (float(np.dot(N[HEAVY], np.asarray(pr.kA_sp)[HEAVY])) / Nhv
                  * (T / 300.0) ** pr.s_k)
    kap = kap_dilute * enskog_factor(T, Nhv / V)
    if pr.electron_kappa:
        kap += SQRT_M_ME * (Ne / Nhv) * kap_dilute
    cp1 = Cv / Ntr + KB
    chi = kap / ((Ntr / V) * cp1)
    Dfac = pr.n_ref / (Nhv / V) * (T / D_REF_T) ** 0.5
    D1, D2, D3, D4, D5 = (pr.D0_1 * Dfac, pr.D0_2 * Dfac, pr.D0_3 * Dfac,
                          pr.D0_4 * Dfac, pr.D0_5 * Dfac)

    aU  = abs(U) + 1e-30
    cap = R / math.pi
    dlT = _bl(chi, R, aU, cap)

    # ---- heat conduction and vapor/atom exchange with the wall ----------
    Qdot = A * kap * (pr.T0 - T) / dlT
    dS1 = A * D1 * (pr.n1_sat - N[IDX["NaCl"]] / V)   / _bl(D1, R, aU, cap)
    dS2 = A * D2 * (pr.n2_sat - N[IDX["Na2Cl2"]] / V) / _bl(D2, R, aU, cap)
    dS3 = A * D3 * (pr.n3_sat - N[IDX["Na"]] / V)     / _bl(D3, R, aU, cap)
    dS4 = A * D4 * (pr.n4_sat - N[IDX["Cl"]] / V)     / _bl(D4, R, aU, cap)

    def dh(h_wall, s, flux):
        return h_wall - (1.5 * KB * T + ui[IDX[s]]) if flux >= 0.0 else KB * T
    dh1 = dh(pr.h1_wall, "NaCl", dS1)
    dh2 = dh(pr.h2_wall, "Na2Cl2", dS2)
    dh3 = dh(pr.h3_wall, "Na", dS3)
    dh4 = dh(pr.h4_wall, "Cl", dS4)

    # ---- ambipolar wall loss of charged species (Ar+ neutralized) --------
    Da3, Da4, Da5 = 2.0 * D3, 2.0 * D4, 2.0 * D5
    g_nap = A * Da3 * (N[IDX["Na+"]] / V) / _bl(Da3, R, aU, cap)
    g_clp = A * Da4 * (N[IDX["Cl+"]] / V) / _bl(Da4, R, aU, cap)
    g_arp = A * Da5 * (N[IDX["Ar+"]] / V) / _bl(Da5, R, aU, cap)
    g_pos = g_nap + g_clp + g_arp
    n_neg = N[IDX["Cl-"]] + N[IDX["e-"]]
    if g_pos > 0.0 and n_neg > 0.0:
        g_clm = g_pos * N[IDX["Cl-"]] / n_neg
        g_el = g_pos - g_clm               # maintains neutrality
    else:
        g_nap = g_clp = g_arp = g_clm = g_el = 0.0
    dh5 = pr.h5_wall - (1.5 * KB * T + ui[IDX["Ar"]])
    dQwall = (-(g_nap + g_clp + g_clm + g_el) * KB * T
              + g_arp * (dh5 - KB * T))

    # ---- chemistry --------------------------------------------------------
    if T > pr.T_peak_rhs:
        pr.T_peak_rhs = T
    Tch = T
    if Tch < nk.T_MIN:
        Tch = nk.T_MIN
    elif Tch > nk.T_MAX:
        Tch = nk.T_MAX
        pr.clamp_hi_hit = True
    r = reaction_rates(Tch, N / V, pr.lambda_max, pr.ion_speedup)
    dN = V * (NU.T @ r)
    dN[IDX["NaCl"]] += dS1
    dN[IDX["Na2Cl2"]] += dS2
    dN[IDX["Na"]] += dS3
    dN[IDX["Cl"]] += dS4
    dN[IDX["Na+"]] -= g_nap
    dN[IDX["Cl+"]] -= g_clp
    dN[IDX["Ar+"]] -= g_arp
    dN[IDX["Ar"]] += g_arp
    dN[IDX["Cl-"]] -= g_clm
    dN[IDX["e-"]] -= g_el
    dS3_tot = dS3 - g_nap                # total nuclei exchange, for audits
    dS4_tot = dS4 - (g_clp + g_clm)

    # ---- energy balance ---------------------------------------------------
    Uchem = V * float(r @ (NU @ (E0 + 1.5 * KB * T + ui)))
    dT = (Qdot - p_g * dV + dS1 * dh1 + dS2 * dh2
          + dS3 * dh3 + dS4 * dh4 + dQwall - Uchem) / Cv

    # ---- dp_g/dt ------------------------------------------------------------
    dVfree = dV - float(np.dot(dN, b))
    dp_g = (float(dN.sum()) * KB * T + Ntr * KB * dT) / Vfree \
        - p_g * dVfree / Vfree

    # ---- Keller-Miksis ------------------------------------------------------
    p_L   = p_g - 2.0 * pr.sigma / R - 4.0 * pr.mu * U / R
    p_inf = pr.p0 - pr.pA * math.sin(2.0 * math.pi * pr.f * t + pr.phase)
    c, rho = pr.c, pr.rho
    denom = (1.0 - U / c) * R + 4.0 * pr.mu / (rho * c)
    num = ((1.0 + U / c) * (p_L - p_inf) / rho
           + (R / (rho * c)) * (dp_g + 2.0 * pr.sigma * U / (R * R)
                                + 4.0 * pr.mu * U * U / (R * R))
           - 1.5 * U * U * (1.0 - U / (3.0 * c)))

    out = np.empty(NST)
    out[0], out[1], out[2] = U, num / denom, dT
    out[3:3 + NSP] = dN
    out[3 + NSP], out[4 + NSP] = dS1, dS2
    out[5 + NSP], out[6 + NSP] = dS3_tot, dS4_tot
    return out


def simulate(pr: Params, n_cycles=3.0, rtol=1e-8, method="Radau",
             n_uniform=12000):
    y0     = setup(pr)
    Tdrive = 1.0 / pr.f
    t_end  = n_cycles * Tdrive
    atol = [1e-14, 1e-4, 1e-3] + [1e-2] * NSP + [1e-2] * 4

    tic = time.perf_counter()
    sol = solve_ivp(rhs, (0.0, t_end), y0, args=(pr,), method=method,
                    rtol=rtol, atol=atol, dense_output=True,
                    max_step=Tdrive / 50.0)
    toc = time.perf_counter()

    t = np.union1d(np.linspace(0.0, t_end, n_uniform), sol.t)
    Y = sol.sol(t)
    res = dict(t=t, R=Y[0], U=Y[1], T=Y[2],
               N=np.maximum(Y[3:3 + NSP], 0.0),
               S1=Y[3 + NSP], S2=Y[4 + NSP], S3=Y[5 + NSP], S4=Y[6 + NSP],
               sol=sol, wall_time=toc - tic, t_end=t_end)
    if sol.status < 0:
        print("WARNING: integrator stopped early:", sol.message)
        if sol.t[-1] < Tdrive and res["T"][-2] < res["T"][-1]:
            pr.highly_dubious = 1
    return res


# ---------------------------------------------------------------- diagnostics
def gas_pressure(res, pr: Params):
    V = (4.0 / 3.0) * math.pi * res["R"] ** 3
    N = res["N"]
    return N.sum(axis=0) * KB * res["T"] / (V - np.asarray(pr.b_sp) @ N)


def conservation_audit(res):
    N, S1, S2, S3, S4 = res["N"], res["S1"], res["S2"], res["S3"], res["S4"]
    i = IDX
    Na_in = N[i["Na"]] + N[i["Na+"]] + N[i["NaCl"]] + 2 * N[i["Na2Cl2"]]
    Cl_in = N[i["Cl"]] + N[i["Cl-"]] + N[i["Cl+"]] + N[i["NaCl"]] \
        + 2 * N[i["Na2Cl2"]]
    wall = (S1 - S1[0]) + 2 * (S2 - S2[0])
    Na = Na_in - (Na_in[0] + wall + (S3 - S3[0]))
    Cl = Cl_in - (Cl_in[0] + wall + (S4 - S4[0]))
    Ar_in = N[i["Ar"]] + N[i["Ar+"]]
    Q = N[i["Na+"]] + N[i["Cl+"]] + N[i["Ar+"]] - N[i["Cl-"]] - N[i["e-"]]
    qscale = max(N[i["e-"]].max(), 1.0)
    return (np.abs(Na).max() / Na_in[0], np.abs(Cl).max() / Cl_in[0],
            np.abs(Ar_in - Ar_in[0]).max() / Ar_in[0],
            np.abs(Q).max() / qscale)


def diagnostics(res, pr: Params):
    R, U, T, N = res["R"], res["U"], res["T"], res["N"]
    p_g = gas_pressure(res, pr)
    Nv0 = N[IDX["NaCl"], 0] + 2 * N[IDX["Na2Cl2"], 0]
    dubious = 0

    imx, imn, itx, ipx = np.argmax(R), np.argmin(R), np.argmax(T), np.argmax(p_g)
    B_min = float(np.asarray(pr.b_sp) @ N[:, imn])
    R_hc = (3.0 * B_min / (4.0 * math.pi)) ** (1.0 / 3.0)
    mach = np.max(np.abs(U)) / pr.c
    eNa, eCl, eAr, eQ = conservation_audit(res)

    print("-" * 72)
    print(f"  molten NaCl:  T0 = {pr.T0:.0f} K   rho = {pr.rho:.0f} kg/m3   "
          f"c = {pr.c:.0f} m/s   p_sat = {p_sat_nacl(pr.T0):.1f} Pa")
    print(f"  R0 = {pr.R0*1e6:.3g} um   f = {pr.f/1e3:.4g} kHz   "
          f"pA = {pr.pA/ATM:.3g} atm   (Blake ~ {pr.p_blake/ATM:.2f} atm)")
    print("-" * 72)
    print(f"  R_max          = {R[imx]*1e6:9.3f} um   "
          f"(expansion ratio {R[imx]/pr.R0:6.2f})")
    print(f"  R_min          = {R[imn]*1e6:9.4f} um   "
          f"(hard-core radius {R_hc*1e6:.4f} um)")
    print(f"  peak T (core)  = {T[itx]:9.0f} K")
    print(f"  peak p_g       = {p_g[ipx]/1e9:9.3f} GPa")
    print(f"  peak |dR/dt|/c = {mach:9.3f}")
    print(f"  vapor/Ar at R_max = "
          f"{(N[IDX['NaCl'],imx]+2*N[IDX['Na2Cl2'],imx])/pr.N_ar:.2e}"
          f"   (t=0: {Nv0/pr.N_ar:.2e})")
    print(f"  conservation audit: Na {eNa:.1e}  Cl {eCl:.1e}  "
          f"Ar {eAr:.1e}  charge {eQ:.1e}")
    if pr.clamp_hi_hit:
        print("  NOTE: T exceeded the kinetics ceiling. chemistry clamped,"
              " treat the collapse peak as qualitative.")
        dubious = 1
    if mach > 1.0:
        print("  NOTE: wall Mach > 1 -- Keller-Miksis at the edge of validity.")
        dubious = 1
    if pr.pA < pr.p_blake and R[imx] / pr.R0 < 1.3:
        print("  NOTE: pA below the Blake threshold. no inertial cavitation.")
    if pr.highly_dubious == 1:
        dubious = 2
    print(f"  integration: {res['sol'].t.size} steps, "
          f"{res['sol'].nfev} RHS evals, {res['wall_time']:.1f} s")
    print("-" * 72)
    return T[itx], p_g[ipx], dubious


# --------------------------------------------------------------------- output
COLOR = {"Ar": "#7f7f7f", "Ar+": "#c71585", "NaCl": "#1f4e9c",
          "Na2Cl2": "#8e44ad", "Na": "#e8820c", "Cl": "#2e8b3d",
          "Na+": "#d62728", "Cl-": "#00a2c7", "Cl+": "#8c564b", "e-": "#000000"}
LABEL = {"NaCl": "NaCl", "Na2Cl2": "Na$_2$Cl$_2$", "Na": "Na", "Cl": "Cl",
         "Na+": "Na$^+$", "Cl-": "Cl$^-$", "Cl+": "Cl$^+$", "e-": "e$^-$",
         "Ar": "Ar", "Ar+": "Ar$^+$"}
FSAFE = {"NaCl": "NaCl", "Na2Cl2": "Na2Cl2", "Na": "Na", "Cl": "Cl",
         "Na+": "Nap", "Cl-": "Clm", "Cl+": "Clp", "e-": "e",
         "Ar": "Ar", "Ar+": "Arp"}


def make_plot(res, pr: Params, fname, n_cycles):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available -- skipping plot.")
        return
    plt.rcParams['font.size'] = 14
    t = res["t"] * 1e6
    N = res["N"]
    p_g = gas_pressure(res, pr)
    title = (f"Argon in molten NaCl ($T_0$={pr.T0:.0f} K):  "
             f"$R_0$={pr.R0*1e6:g} $\\mu$m,  f={pr.f/1e3:g} kHz,  "
             f"$p_A$={pr.pA/ATM:g} atm")

    if os.path.exists(fname):
        # only ever delete a previous output folder of this script, never an
        # arbitrary folder passed with --out
        if not os.path.isdir(os.path.join(fname, "basic")):
            raise SystemExit(f"'{fname}' already exists and is not a previous "
                             "output folder; choose another --out name.")
        shutil.rmtree(fname)
    os.makedirs(fname + "/basic/N_plots")
    os.makedirs(fname + "/analysis_plots")
    t_bound = n_cycles / pr.f * 1e6

    for y, ylab, col, log, out in (
            (res["R"] * 1e6, "R  [$\\mu$m]", "tab:blue", False, "radius"),
            (res["T"], "T  [K]", "tab:red", True, "temperature"),
            (p_g, "$p_g$ (Pa)", "tab:green", True, "pressure")):
        (plt.semilogy if log else plt.plot)(t, y, lw=1.0, color=col)
        plt.ylabel(ylab)
        plt.title(title)
        plt.xlim(0, t_bound)
        plt.xlabel("t [$\\mu$s]")
        plt.savefig(fname + f"/basic/{out}.png", dpi=300, bbox_inches="tight")
        plt.clf()

    for s in SPECIES:
        plt.semilogy(t, np.maximum(N[IDX[s]], 1.0), lw=1.0, color=COLOR[s])
        plt.ylabel(f"N({LABEL[s]})  [particles]")
        plt.title(title)
        plt.xlim(0, t_bound)
        plt.ylim(1, 1e10)
        plt.xlabel("t [$\\mu$s]")
        plt.savefig(fname + f"/basic/N_plots/N_{FSAFE[s]}.png", dpi=300,
                    bbox_inches="tight")
        plt.clf()
    plt.close("all")

    # zoomed analysis plots: t = 0 at the temperature peak
    t_pk = res["t"][np.argmax(res["T"])]
    for name, half, scale, unit, species, fs, ls in (
            ("ambipolar", 0.5e-6, 1e6, r"$\mu$s",
             [("Cl-", "green"), ("e-", "red")], 12, 10),
            ("ionization", 30e-9, 1e9, "ns",
             [("Ar+", "tab:orange"), ("Cl+", "tab:purple"),
              ("Na+", "tab:brown"), ("Cl-", "green")], 15, 13)):
        win = np.abs(res["t"] - t_pk) <= 1.5 * half
        tw = (res["t"][win] - t_pk) * scale

        plt.figure(figsize=(5.5, 4.0), dpi=300)
        ax1 = plt.gca()
        p1, = ax1.plot(tw, res["T"][win], lw=1.0, ls="--", label=r"$T$")
        ax1.set_xlabel(f"$t$ ({unit})", fontsize=fs)
        ax1.set_ylabel(r"Temperature, $T$ (K)", fontsize=fs)
        ax1.set_yscale("log")
        ax1.set_xlim(-half * scale, half * scale)
        ax1.tick_params(direction="in", which="both", top=True, labelsize=ls)
        ax1.minorticks_on()

        ax2 = ax1.twinx()
        lines = [p1]
        for s, col in species:
            p, = ax2.plot(tw, np.maximum(N[IDX[s], win], 1.0), lw=1.0,
                          color=col, label=f"$N$({LABEL[s]})")
            lines.append(p)
        if name == "ionization":
            v_tot = (N[IDX["NaCl"], win] + 2 * N[IDX["Na2Cl2"], win]
                     + 0.5 * (N[IDX["Na"], win] + N[IDX["Na+"], win]
                              + N[IDX["Cl-"], win] + N[IDX["Cl+"], win]
                              + N[IDX["Cl"], win]))
            p, = ax2.plot(tw, np.maximum(v_tot, 1.0), lw=1.0, color="red",
                          label="$N$(v,tot)")
            lines.append(p)
        ax2.set_ylabel(r"$N$ [particles]", fontsize=fs)
        ax2.set_yscale("log")
        ax2.set_ylim(1e0, 1e10)
        ax2.tick_params(direction="in", which="both", top=True, labelsize=ls)
        ax2.minorticks_on()
        ax2.legend(handles=lines, fontsize=ls)
        plt.tight_layout()
        plt.savefig(fname + f"/analysis_plots/{name}.png", dpi=300,
                    bbox_inches="tight")
        plt.close()

    # fraction of each species' thermal energy held in each mode
    T = res["T"]
    for s, rot, thetas in (("Na", 0.0, []), ("NaCl", 1.0, _TH_NACL),
                           ("Na2Cl2", 1.5, _TH_DIMER)):
        modes = {"translational": 1.5 * KB * T,
                 "rotational":    rot * KB * T,
                 "vibrational":   np.array([_vib_u_cv(Ti, thetas)[0] for Ti in T]),
                 "electronic":    np.array([_elec_u_cv(Ti, _LV[s])[0]
                                            if s in _LV else 0.0 for Ti in T])}
        total = sum(modes.values())
        plt.figure(figsize=(4, 2.8))
        for mode, u in modes.items():
            plt.plot(t, u / total, lw=1.0,
                     ls="--" if mode == "rotational" else "-", label=mode)
        plt.xlim(0, t_bound)
        plt.ylim(0, 1)
        plt.tick_params(axis='both', labelsize=8)
        plt.xlabel("t [$\\mu$s]", fontsize=8)
        plt.ylabel(f"fraction of {LABEL[s]} energy", fontsize=8)
        plt.legend(fontsize=8, ncol=2, loc="lower center",
                   bbox_to_anchor=(0.5, 1.0))
        plt.tight_layout()
        plt.savefig(fname + f"/analysis_plots/modes_{s}.png", dpi=300)
        plt.close()


def save_csv(res, folder):
    os.makedirs(folder, exist_ok=True)
    fname = os.path.join(folder, os.path.basename(os.path.normpath(folder)) + ".csv")
    cols = [res["t"], res["R"], res["U"], res["T"]] \
         + [res["N"][k] for k in range(NSP)] \
         + [res["S1"], res["S2"], res["S3"], res["S4"]]
    hdr = ("t[s] R[m] U[m/s] T[K] "
           + " ".join("N_" + s for s in SPECIES)
           + " S_NaCl S_Na2Cl2 S_Na S_Cl")
    np.savetxt(fname, np.column_stack(cols), header=hdr, fmt="%.10e")
    print(f"  data written to {fname}")


# ----------------------------------------------------------------------- main
def main(R0_=5.0, f_=20.0, pA_=1.2, cycles_=3.0, T0_=1100.0,
         lambda_max_=float('inf'), rtol_=1e-8, ion_speedup_=1.0,
         electron_kappa_=False):
    ap = argparse.ArgumentParser(
        description="Keller-Miksis argon bubble in molten NaCl with "
                    "NaCl-vapor chemistry.")
    ap.add_argument("--R0", type=float, default=R0_, help="[um]")
    ap.add_argument("--f", type=float, default=f_, help="[kHz]")
    ap.add_argument("--pA", type=float, default=pA_, help="[atm]")
    ap.add_argument("--cycles", type=float, default=cycles_)
    ap.add_argument("--T0", type=float, default=T0_, help="[K]")
    ap.add_argument("--lambda-max", type=float, default=lambda_max_,
                    dest="lambda_max", metavar="LAM",
                    help="chemistry stiffness throttle [1/s] (default inf = "
                         "unthrottled; set e.g. 1e16 if a run stalls)")
    ap.add_argument("--ion-speedup", type=float, default=ion_speedup_,
                    dest="ion_speedup")
    ap.add_argument("--electron-kappa", action=argparse.BooleanOptionalAction,
                    default=electron_kappa_, dest="electron_kappa")
    ap.add_argument("--csv", action="store_true")
    ap.add_argument("--out", default=None, help="output folder basename")
    ap.add_argument("--plot", action=argparse.BooleanOptionalAction,
                    default=True)
    args = ap.parse_args()

    pr = Params(R0=args.R0 * 1e-6, f=args.f * 1e3, pA=args.pA * ATM,
                T0=args.T0, lambda_max=args.lambda_max,
                ion_speedup=args.ion_speedup,
                electron_kappa=args.electron_kappa)
    base = args.out or (f"nacl_R0-{args.R0:g}um_f-{args.f:g}kHz"
                        f"_pA-{args.pA:g}atm"
                        + (f"_ionx{args.ion_speedup:g}"
                           if args.ion_speedup != 1.0 else "")
                        + ("_ekap" if args.electron_kappa else ""))
    if (args.plot and os.path.exists(base)
            and not os.path.isdir(os.path.join(base, "basic"))):
        raise SystemExit(f"'{base}' already exists and is not a previous "
                         "output folder; choose another --out name.")

    res = simulate(pr, n_cycles=args.cycles, rtol=rtol_)
    if args.plot:
        make_plot(res, pr, base, args.cycles)
    if args.csv:
        save_csv(res, base)
    return diagnostics(res, pr)


if __name__ == "__main__":
    main()