#!/usr/bin/env python3
"""Vapor-free Keller-Miksis argon bubble in any liquid. The bubble contains
only argon, and the only chemistry is ionization of argon (Ar <-> Ar+ + e-).

USAGE INSTRUCTIONS (also in README.md):
1. set the liquid's properties in the LIQUID PROPERTIES block below.
        no data for your liquid? see "GUIDE: LIQUIDS WITHOUT GOOD PROPERTY DATA"
        at the bottom of this file.
2. type "python simulation/novapor.py [options]" in the terminal

see below for a sample output and a list of available options.




Sample output:
#first part: parameters the simulation was run with
------------------------------------------------------------------------ 
  NaCl:  T0 = 1100 K   rho = 1542 kg/m3   c = 1726 m/s   (no vapor)
         mu = 0.985 mPa s   sigma = 0.112 N/m
  R0 = 5 um   f = 20 kHz   pA = 1.2 atm   (Blake ~ 1.09 atm)
  
#second part: results of the simulation
------------------------------------------------------------------------
  R_max          =    24.778 um   (expansion ratio   4.96)
  R_min          =    0.7259 um   (hard-core radius 0.3999 um)
  peak T (core)  =     15439 K
  peak p_g       =     0.804 GPa
  peak |dR/dt|/c =     0.221
  conservation audit: Ar 1.1e-15  charge 3.2e-26                        #should show very small
                                                                        #numbers; just makes sure
                                                                        #charge and mass are 
                                                                        #conserved

  integration: 7291 steps, 55683 RHS evals, 6.9 s                       # number of timesteps,
                                                                        #evaluations of the main
                                                                        #physics function, and 
                                                                        #runtime.
------------------------------------------------------------------------

A general rule for interpreting the results is that temperatures above 25kK should be treated with
increasing caution, as discussed in the attached report.







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
ambient liquid temperature. default is T0_DEFAULT in the liquid block.


-------------------- MORE ADVANCED ----------------------------------


--lambda-max [number, e.g., 1e16]
rate limiter. sometimes, with particularly intense collapses, scipy's Radau
integrator can fail (i.e., it can require a timestep below what
floating point numbers can resolve). this is
because the calculated argon ionization rate is incredibly high.
--lambda-max caps how fast the ionization can relax towards equilibrium [1/s];
the equilibrium itself is unchanged, so results are unaffected as long
as 1/λ_max is much shorter than the collapse timescale.

use case: if the simulation fails or gives an error you don't understand,
try adding --lambda-max 1e16 or, if that doesn't work, 1e14 to fix it.
this should leave the results unaffected. you can test whether it does or
not by lowering it by one order of magnitude and seeing if the results change
(e.g., if 1e16 gives you 15000K and 1e15 also gives you 15000K, you know
1e16 is sufficiently high that everything remains in equilibrium).


--ion-speedup [factor, e.g. 1e6]
speeds up the argon ionization rate by some factor. if the saha equilibrium
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
"novapor_R0-[R0]um_f-[f]kHz_pA-[pA]atm"


--no-plot
if entered, the simulation will not produce plots. it
does produce plots by default.


requires nacl_kinetics.py (argon data and ionization rate) in the same folder.
"""

import math

# =========================================================================
# LIQUID PROPERTIES  (edit these for your liquid; default is NaCl)
# =========================================================================
LIQUID_NAME = "NaCl"     # used in printouts and plot titles
T_MELT      = 1074.0     # melting point [K]; T0 below this raises an error.
                         # Set to None to skip the check.
T0_DEFAULT  = 1100.0     # default ambient temperature [K] (--T0 overrides)

# Temperature range [K] over which the fits below are valid. A warning is
# printed if T0 falls outside it. Set to None to skip.
FIT_RANGE   = (1074.0, 1300.0)

# Each property is a function of temperature T [K] returning SI units.
# Enter a fit, or simply return a number, e.g.   return 1550.0
# Each is evaluated once, at T0.

def density(T):            # [kg/m^3]
    return 2139.3 - 0.5430 * T

def sound_speed(T):        # [m/s]  Bockris and Richards, 1957
    return 2483.0 - 0.915 * (T - 273.0)

def viscosity(T):          # dynamic viscosity [Pa s]
    return 8.931e-5 * math.exp(2641.0 / T)

def surface_tension(T):    # [N/m]
    return 0.191 - 7.188e-5 * T

# =========================================================================
# END OF LIQUID PROPERTIES
# =========================================================================

import argparse
import os
import shutil
import time
import warnings
from dataclasses import dataclass, field

import numpy as np
from scipy.integrate import solve_ivp

warnings.filterwarnings("ignore", message="overflow encountered in multiply",
                        category=RuntimeWarning,
                        module=r"scipy\.integrate\._ivp\.common")

import nacl_kinetics as nk

KB  = nk.KB
NAV = nk.NA_AVOGADRO
ATM = 101325.0
EV  = nk.EV
SQRT_M_ME = math.sqrt(39.948 * 1.66053906660e-27 / 9.1093837015e-31)

# reference conditions of the Ar+-Ar diffusivity D0_5 (a property of argon,
# not of the liquid, so fixed whatever liquid or T0 is chosen)
D_REF_P, D_REF_T = ATM, 1100.0

SPECIES = ["Ar", "Ar+", "e-"]
KWNAMES = ["n_Ar", "n_Arp", "n_e"]
IDX = {s: i for i, s in enumerate(SPECIES)}
NSP = len(SPECIES)
IE_ = IDX["e-"]
HEAVY = np.array([k for k in range(NSP) if k != IE_])
NST = 3 + NSP                    # states: R, U, T, species

STOICH = {"R9": {"Ar": -1.0, "Ar+": +1.0, "e-": +1.0}}
RATEFN = {"R9": nk.rate_R9_Ar_ionization}
RXNS = list(STOICH)
NRX = len(RXNS)

NU = np.zeros((NRX, NSP))
for _i, _r in enumerate(RXNS):
    for _s, _v in STOICH[_r].items():
        NU[_i, IDX[_s]] = _v
NUMASK = [NU[i] != 0.0 for i in range(NRX)]

FD_REL = 1.0e-6

# =========================================================================
# PER-SPECIES ENERGETICS
# =========================================================================
E0 = np.zeros(NSP)
E0[IDX["Ar+"]] = +nk.IE_AR * EV

_LV = {s: tuple((nk.C2 * E, g) for E, g in lv) for s, lv in (
    ("Ar", nk.AR_LEVELS), ("Ar+", nk.ARP_LEVELS))}


def _elec_u_cv(T, levels):
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


def uint_cvint(T):
    """Electronic energy and cv per particle (translation excluded)."""
    u = np.zeros(NSP)
    cv = np.zeros(NSP)
    for s in ("Ar", "Ar+"):
        u[IDX[s]], cv[IDX[s]] = _elec_u_cv(T, _LV[s])
    return u, cv


# =========================================================================
# DENSE-GAS THERMAL CONDUCTIVITY (Barker-Henderson + Enskog, Bastea)
# =========================================================================
_EPS_AR, _RM_AR, _ALPHA_AR = 122.0 * KB, 3.85e-10, 13.2


def _u_exp6(r):
    a = _ALPHA_AR
    return _EPS_AR * (6.0 / (a - 6.0) * np.exp(a * (1.0 - r / _RM_AR))
                      - a / (a - 6.0) * (_RM_AR / r) ** 6)


def _build_d_bh():
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
# LIQUID
# =========================================================================
def liquid_properties(T0):
    if T_MELT is not None and T0 < T_MELT:
        raise ValueError(f"T0 = {T0} K is below the melting point of "
                         f"{LIQUID_NAME} ({T_MELT} K).")
    if FIT_RANGE is not None and not (FIT_RANGE[0] <= T0 <= FIT_RANGE[1]):
        print(f"WARNING: T0 = {T0} K is outside the property fits' range "
              f"{FIT_RANGE[0]:g}-{FIT_RANGE[1]:g} K; values are extrapolated.")
    props = {"density": density(T0), "sound_speed": sound_speed(T0),
             "viscosity": viscosity(T0), "surface_tension": surface_tension(T0)}
    for name, v in props.items():
        if not (math.isfinite(v) and v > 0.0):
            raise ValueError(f"{name}({T0} K) = {v}; must be positive.")
    return (props["density"], props["sound_speed"], props["viscosity"],
            props["surface_tension"])


@dataclass
class Params:
    p0:    float = ATM
    T0:    float = 1100.0

    f:     float = 20.0e3
    pA:    float = 1.2 * ATM
    phase: float = 0.0

    R0: float = 5.0e-6

    b_sp:  tuple = (5.32e-29, 5.32e-29, 0.0)      # Ar, Ar+, e-
    kA_sp: tuple = (0.0177, 0.0177, 0.0)
    s_k:   float = 0.7
    D0_5:  float = 8.0e-5      # Ar+-Ar diffusivity at (D_REF_P, D_REF_T)

    electron_kappa: bool = False
    lambda_max: float = float("inf")
    ion_speedup: float = 1.0           # multiplier on R9

    # liquid, filled by setup() from the property functions
    rho:   float = field(default=0.0, init=False)
    c:     float = field(default=0.0, init=False)
    mu:    float = field(default=0.0, init=False)
    sigma: float = field(default=0.0, init=False)

    N_ar:    float = field(default=0.0, init=False)
    n_ref:   float = field(default=0.0, init=False)
    h5_wall: float = field(default=0.0, init=False)
    p_blake: float = field(default=0.0, init=False)
    T_peak_rhs:  float = field(default=0.0, init=False)
    clamp_hi_hit: bool = field(default=False, init=False)
    highly_dubious: float = field(default=0.0, init=False)


def setup(pr: Params):
    pr.rho, pr.c, pr.mu, pr.sigma = liquid_properties(pr.T0)
    pr.n_ref = D_REF_P / (KB * D_REF_T)

    ui0, _ = uint_cvint(pr.T0)
    pr.h5_wall = 2.5 * KB * pr.T0 + ui0[IDX["Ar"]]

    V0   = (4.0 / 3.0) * math.pi * pr.R0 ** 3
    p_eq = pr.p0 + 2.0 * pr.sigma / pr.R0
    pr.N_ar = p_eq * V0 / (KB * pr.T0 + p_eq * pr.b_sp[IDX["Ar"]])

    Kb = p_eq * pr.R0 ** 3
    Rc = math.sqrt(3.0 * Kb / (2.0 * pr.sigma))
    pr.p_blake = pr.p0 + 4.0 * pr.sigma / (3.0 * Rc)

    y0 = np.zeros(NST)
    y0[0], y0[1], y0[2] = pr.R0, 0.0, pr.T0
    y0[3 + IDX["Ar"]] = pr.N_ar
    return y0


def reaction_rates(T, n, lambda_max, ion_speedup=1.0):
    n = np.maximum(n, 0.0)
    kw = dict(zip(KWNAMES, n))
    rates = np.zeros(NRX)
    throttling = np.isfinite(lambda_max)
    for i, rname in enumerate(RXNS):
        fn, nu, m = RATEFN[rname], NU[i], NUMASK[i]
        r0 = fn(T, **kw) * ion_speedup
        if not throttling:
            rates[i] = r0
            continue
        scale = (n[m] / np.abs(nu[m])).max()
        if scale <= 0.0:
            rates[i] = r0
            continue
        d = FD_REL * scale
        kw2 = dict(zip(KWNAMES, np.maximum(n + d * nu, 0.0)))
        lam = -(fn(T, **kw2) * ion_speedup - r0) / d
        if lam > 0.0:
            r0 *= lambda_max / (lambda_max + lam)
        rates[i] = r0
    return rates


def _bl(D, R, aU, cap):
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

    b = pr.b_sp
    B = float(np.dot(N, b))
    Vfree = max(V - B, 1e-6 * B)
    p_g = Ntr * KB * T / Vfree

    ui, ci = uint_cvint(T)
    Cv = 1.5 * KB * Ntr + float(N @ ci)

    kap_dilute = (float(np.dot(N[HEAVY], np.asarray(pr.kA_sp)[HEAVY])) / Nhv
                  * (T / 300.0) ** pr.s_k)
    kap = kap_dilute * enskog_factor(T, Nhv / V)
    if pr.electron_kappa:
        kap += SQRT_M_ME * (Ne / Nhv) * kap_dilute
    cp1 = Cv / Ntr + KB
    chi = kap / ((Ntr / V) * cp1)
    D5 = pr.D0_5 * pr.n_ref / (Nhv / V) * (T / D_REF_T) ** 0.5

    aU  = abs(U) + 1e-30
    cap = R / math.pi
    Qdot = A * kap * (pr.T0 - T) / _bl(chi, R, aU, cap)

    # ambipolar wall neutralization of Ar+ (electrons follow)
    Da5 = 2.0 * D5
    g_arp = A * Da5 * (N[IDX["Ar+"]] / V) / _bl(Da5, R, aU, cap)
    if not (g_arp > 0.0 and Ne > 0.0):
        g_arp = 0.0
    g_el = g_arp
    dh5 = pr.h5_wall - (1.5 * KB * T + ui[IDX["Ar"]])
    dQwall = -g_el * KB * T + g_arp * (dh5 - KB * T)

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
    dN[IDX["Ar+"]] -= g_arp
    dN[IDX["Ar"]] += g_arp
    dN[IDX["e-"]] -= g_el

    Uchem = V * float(r @ (NU @ (E0 + 1.5 * KB * T + ui)))
    dT = (Qdot - p_g * dV + dQwall - Uchem) / Cv

    dVfree = dV - float(np.dot(dN, b))
    dp_g = (float(dN.sum()) * KB * T + Ntr * KB * dT) / Vfree \
        - p_g * dVfree / Vfree

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
    out[3:] = dN
    return out


def simulate(pr: Params, n_cycles=3.0, rtol=1e-8, method="Radau",
             n_uniform=12000):
    y0     = setup(pr)
    Tdrive = 1.0 / pr.f
    t_end  = n_cycles * Tdrive
    atol = [1e-14, 1e-4, 1e-3] + [1e-2] * NSP

    tic = time.perf_counter()
    sol = solve_ivp(rhs, (0.0, t_end), y0, args=(pr,), method=method,
                    rtol=rtol, atol=atol, dense_output=True,
                    max_step=Tdrive / 50.0)
    toc = time.perf_counter()

    t = np.union1d(np.linspace(0.0, t_end, n_uniform), sol.t)
    Y = sol.sol(t)
    res = dict(t=t, R=Y[0], U=Y[1], T=Y[2], N=np.maximum(Y[3:], 0.0),
               sol=sol, wall_time=toc - tic, t_end=t_end)
    if sol.status < 0:
        print("WARNING: integrator stopped early:", sol.message)
        if sol.t[-1] < Tdrive and res["T"][-2] < res["T"][-1]:
            pr.highly_dubious = 1
    return res


def gas_pressure(res, pr: Params):
    V = (4.0 / 3.0) * math.pi * res["R"] ** 3
    N = res["N"]
    return N.sum(axis=0) * KB * res["T"] / (V - np.asarray(pr.b_sp) @ N)


def diagnostics(res, pr: Params):
    R, U, T, N = res["R"], res["U"], res["T"], res["N"]
    p_g = gas_pressure(res, pr)
    dubious = 0
    imx, imn, itx, ipx = np.argmax(R), np.argmin(R), np.argmax(T), np.argmax(p_g)
    B_min = float(np.asarray(pr.b_sp) @ N[:, imn])
    R_hc = (3.0 * B_min / (4.0 * math.pi)) ** (1.0 / 3.0)
    mach = np.max(np.abs(U)) / pr.c
    Ar_in = N[IDX["Ar"]] + N[IDX["Ar+"]]
    eAr = np.abs(Ar_in - Ar_in[0]).max() / Ar_in[0]
    eQ = np.abs(N[IDX["Ar+"]] - N[IDX["e-"]]).max() / max(N[IDX["e-"]].max(), 1.0)

    print("-" * 72)
    print(f"  {LIQUID_NAME}:  T0 = {pr.T0:.0f} K   rho = {pr.rho:.0f} kg/m3   "
          f"c = {pr.c:.0f} m/s   (no vapor)")
    print(f"  {' ' * len(LIQUID_NAME)}   mu = {pr.mu*1e3:.3g} mPa s   "
          f"sigma = {pr.sigma:.3g} N/m")
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
    print(f"  conservation audit: Ar {eAr:.1e}  charge {eQ:.1e}")
    if pr.clamp_hi_hit:
        print("  NOTE: T exceeded the kinetics ceiling (chemistry clamped); "
              "collapse peak is qualitative.")
        dubious = 1
    if mach > 1.0:
        print("  NOTE: wall Mach > 1, Keller-Miksis at the edge of validity.")
        dubious = 1
    if pr.pA < pr.p_blake and R[imx] / pr.R0 < 1.3:
        print("  NOTE: pA below the Blake threshold, no inertial cavitation.")
    if pr.highly_dubious == 1:
        dubious = 2
    print(f"  integration: {res['sol'].t.size} steps, "
          f"{res['sol'].nfev} RHS evals, {res['wall_time']:.1f} s")
    print("-" * 72)
    return T[itx], p_g[ipx], dubious


COLOR = {"Ar": "#7f7f7f", "Ar+": "#c71585", "e-": "#000000"}
LABEL = {"Ar": "Ar", "Ar+": "Ar$^+$", "e-": "e$^-$"}
FSAFE = {"Ar": "Ar", "Ar+": "Arp", "e-": "e"}


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
    title = (f"Argon in {LIQUID_NAME} ($T_0$={pr.T0:.0f} K):  "
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

    # zoomed ionization plot: t = 0 at the temperature peak
    t_pk = res["t"][np.argmax(res["T"])]
    half, scale = 30e-9, 1e9
    win = np.abs(res["t"] - t_pk) <= 1.5 * half
    tw = (res["t"][win] - t_pk) * scale
    plt.figure(figsize=(5.5, 4.0), dpi=300)
    ax1 = plt.gca()
    p1, = ax1.plot(tw, res["T"][win], lw=1.0, ls="--", label=r"$T$")
    ax1.set_xlabel("$t$ (ns)", fontsize=15)
    ax1.set_ylabel(r"Temperature, $T$ (K)", fontsize=15)
    ax1.set_yscale("log")
    ax1.set_xlim(-half * scale, half * scale)
    ax1.tick_params(direction="in", which="both", top=True, labelsize=13)
    ax1.minorticks_on()
    ax2 = ax1.twinx()
    p2, = ax2.plot(tw, np.maximum(N[IDX["Ar+"], win], 1.0), lw=1.0,
                   color="tab:orange", label="$N$(Ar$^+$)")
    ax2.set_ylabel(r"$N$ [particles]", fontsize=15)
    ax2.set_yscale("log")
    ax2.set_ylim(1e0, 1e10)
    ax2.tick_params(direction="in", which="both", top=True, labelsize=13)
    ax2.minorticks_on()
    ax2.legend(handles=[p1, p2], fontsize=12)
    plt.tight_layout()
    plt.savefig(fname + "/analysis_plots/ionization.png", dpi=300,
                bbox_inches="tight")
    plt.close()


def save_csv(res, folder):
    os.makedirs(folder, exist_ok=True)
    fname = os.path.join(folder, os.path.basename(os.path.normpath(folder)) + ".csv")
    cols = [res["t"], res["R"], res["U"], res["T"]] \
         + [res["N"][k] for k in range(NSP)]
    hdr = "t[s] R[m] U[m/s] T[K] " + " ".join("N_" + s for s in SPECIES)
    np.savetxt(fname, np.column_stack(cols), header=hdr, fmt="%.10e")
    print(f"  data written to {fname}")


def main(R0_=5.0, f_=20.0, pA_=1.2, cycles_=3.0, T0_=T0_DEFAULT,
         lambda_max_=float('inf'), rtol_=1e-8, ion_speedup_=1.0,
         electron_kappa_=False):
    ap = argparse.ArgumentParser(
        description=f"Keller-Miksis pure-argon bubble in {LIQUID_NAME} "
                    "(no vapor), with argon Saha ionization.")
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
    base = args.out or (f"novapor_R0-{args.R0:g}um_f-{args.f:g}kHz"
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


# =========================================================================
# GUIDE: LIQUIDS WITHOUT GOOD PROPERTY DATA
# =========================================================================
# Only the values at T0 are used, so one number per property is enough;
# a fit is only needed if you want to vary T0. Options, best first:
#
# 1. MEASURED DATA near T0. Good places to look: Janz, "Molten Salts
#    Handbook" / NIST molten-salt database (salts); Iida & Guthrie, "The
#    Physical Properties of Liquid Metals" (metals); NIST WebBook, DIPPR or
#    Yaws (molecular liquids); slag/glass atlases (oxides, silicates).
#
# 2. A WELL-STUDIED, SIMILAR LIQUID. Take a liquid of the same bonding
#    type and similar molar mass/structure (e.g. KCl or NaBr for an unusual
#    alkali halide, Cu or Ag for another noble metal, and use its properties 
#    at the same homologous temperature T0/T_melt. This should still give a 
#    good order-of-magnitude estimate of the collapse conditions.
#
# 3. ESTIMATES FROM OTHER DATA.
#    density    solid density / (1 + dV), with dV the volume change on
#               melting: ~2-6 % for most metals, ~10-25 % for alkali
#               halides (NaCl ~25 %).
#    viscosity  Andrade's formula at the melting point (metals, ~30 %;
#               rougher for simple salts):
#                   mu ~ 1.8e-7 * sqrt(M * T_melt) / V_m**(2/3)
#               with M the molar mass and V_m the liquid molar
#               volume, all in SI units.
#    sound      c = sqrt(K / rho) if the bulk modulus K of the liquid (or,
#    speed      more roughly, the solid) is known.
#
# 4. TYPICAL VALUES FOR THE CLASS of liquid, near its melting point:
#
#                         rho        c          mu           sigma
#                         [kg/m3]    [m/s]      [mPa s]      [N/m]
#    liquid metals        500-20000  1500-5000  0.5-6        0.2-2.5
#    molten salts         1500-3500  1500-3000  1-10         0.05-0.25
#      (halides, nitrates, fluorides)
#    molecular liquids    700-1500   1000-1900  0.3-5        0.02-0.07
#      (water, simple organics)
#    oxide/silicate melts 2200-4000  2000-4000  30 to >1e6   0.3-0.7
#
#    Use the middle of the range unless you know which end the liquid
#    sits at (e.g. heavy metals are dense and slow, light ones the reverse).
#
# WHICH PROPERTIES MATTER MOST
#    surface tension  sets the Blake threshold (the lowest pressure amplitude
#                     at which violent collapse occurs). It matters 
#                     most for small bubbles and pressure amplitudes
#                     close to the threshold.
#    viscosity        negligible below ~10 mPa s for micron bubbles; above
#                     ~100 mPa s it damps the collapse strongly, so
#                     viscous melts need a real value. for most ionic and
#                     metallic melts this is, however, negligible.
#    density          sets the bubble's inertia: affects R_max and collapse speed.
#    sound speed      only enters radiation damping near collapse; weakest.
#    Check the sensitivity: rerun with each uncertain property at, say,
#    half and double its estimate, and report the spread in peak T.
#
# WHEN THIS VAPOR-FREE MODEL APPLIES
#    The liquid's vapor is ignored. That is only safe if its vapor
#    pressure at T0 is low. For NaCl-like halides, p_sat of a few tens of
#    Pa already changes the peak T by ~2 %, rising to ~10-15 % at a few
#    hundred Pa for strong collapses. If p_sat is much higher, use a model
#    with vapor chemistry instead.
#
#    As a general rule, anything below 100Pa is safe, and for anything above that,
#    the simulation doesn't instantly break but will get less and less accurate.
#
#    The bubble gas is always argon here; another gas needs its own data
#    and ionization rate in place of nacl_kinetics' argon entries.