#!/usr/bin/env python3
"""Equilibrium constants and net reaction rates for hot NaCl vapor / weakly
ionized NaCl plasma in an argon bubble. Intended validity ~300 K to ~3e4 K."""

'''
You probably don't need to use this file; all simulation runs should be from nacl.py or novapor.py.

This file just has all of the partition functions/rate constants for the reactions.










REACTIONS:
    R1   2 NaCl      <-> Na2Cl2
    R2   NaCl        <-> Na + Cl
    R3   NaCl        <-> Na+ + Cl-
    R4   Na          <-> Na+ + e-        (Saha)
    R5   Cl + e-     <-> Cl-             (Saha)
    R6   Cl          <-> Cl+ + e-        (Saha)
    R7   Na+ + Cl-   <-> Na + Cl         mutual neutralization
    R8   Cl+ + Cl-   <-> Cl + Cl         mutual neutralization
    R9   Ar          <-> Ar+ + e-        (Saha)
    R10  Ar+ + Cl-   <-> Ar + Cl         mutual neutralization


'''

from math import pi, sqrt, exp, sin, cos, radians

# =========================================================================
# PHYSICAL CONSTANTS (CODATA)
# =========================================================================
KB = 1.380649e-23           # J/K
H = 6.62607015e-34          # J s
ME = 9.1093837015e-31       # kg
U = 1.66053906660e-27       # kg (atomic mass unit)
EV = 1.602176634e-19        # J
NA_AVOGADRO = 6.02214076e23
C2 = 1.4387768775           # hc/k [cm K], converts cm^-1 to K

# =========================================================================
# SPECIES DATA
# =========================================================================
# masses
M_NA = 22.98977 * U
M_CL = 35.4530 * U
M_NACL = M_NA + M_CL
M_NA2CL2 = 2.0 * M_NACL
M_NAP = M_NA - ME           # Na+
M_CLM = M_CL + ME           # Cl-
M_CLP = M_CL - ME           # Cl+
M_AR = 39.948 * U
M_ARP = M_AR - ME           # Ar+

# electronic levels: (term energy [cm^-1], degeneracy), truncated
NA_LEVELS = [(0.0, 2), (16967.6, 6), (25739.9, 2), (29172.9, 10), (30270.8, 6)]
CL_LEVELS = [(0.0, 4), (882.35, 2)]                       # 2P3/2 , 2P1/2
NAP_LEVELS = [(0.0, 1)]                                   # 1S0, next state ~33 eV
CLM_LEVELS = [(0.0, 1)]                                   # 1S0
CLP_LEVELS = [(0.0, 5), (696.0, 3), (996.5, 1), (11653.6, 5), (27878.0, 1)]
E_LEVELS = [(0.0, 2)]                                     # free electron spin
AR_LEVELS = [(0.0, 1),                                    # 3p^6 1S0
             (93143.76, 5), (93750.60, 3),                # 4s [3/2]2, [3/2]1
             (94553.67, 1), (95399.83, 3)]                # 4s'[1/2]0, [1/2]1
ARP_LEVELS = [(0.0, 4), (1431.58, 2)]                     # 3p^5 2P3/2 , 2P1/2

# NaCl (X 1Sigma+)
NACL_B = 0.2181             # cm^-1, rotational constant (NIST Chemistry WebBook)
NACL_SIGMA = 1              # symmetry number (heteronuclear)
NACL_FREQ = [364.6]         # cm^-1, vibrational fundamental

# Na2Cl2 (planar rhombus, D2h); geometry gives the moments of inertia
NA2CL2_R = 2.58e-10         # m
NA2CL2_ANGLE = 101.4        # deg, Cl-Na-Cl - nist datum
NA2CL2_SIGMA = 4            # rotational symmetry number of D2h
NA2CL2_FREQS = [274.0, 258.0, 230.0, 228.0, 134.0, 96.0]   # cm^-1, estimates


def _dimer_rot_temperatures():
    """Principal rotational temperatures [K] of the planar rhombic dimer, estimated from moments of inertia."""
    half = radians(NA2CL2_ANGLE) / 2.0
    x_cl = NA2CL2_R * sin(half)     
    y_na = NA2CL2_R * cos(half)     
    Ia = 2.0 * M_NA * y_na ** 2     # axis through the two Cl
    Ib = 2.0 * M_CL * x_cl ** 2     # axis through the two Na
    Ic = Ia + Ib                    # out-of-plane (planar molecule)
    f = H ** 2 / (8.0 * pi ** 2 * KB)
    return f / Ia, f / Ib, f / Ic


THETA_A, THETA_B, THETA_C = _dimer_rot_temperatures()

# 0 K reaction energies [eV]
D0_NACL = 4.23              # NaCl -> Na + Cl
E_DIM = 2.10                # 2 NaCl -> Na2Cl2
IE_NA = 5.139076            # Na -> Na+ + e-
EA_CL = 3.612725            # Cl- -> Cl + e-
IE_CL = 12.967632           # Cl -> Cl+ + e-
IE_AR = 15.7596119          # Ar -> Ar+ + e-
E_IONPAIR = D0_NACL + IE_NA - EA_CL      # NaCl -> Na+ + Cl-

# =========================================================================
# KINETIC PARAMETERS
# =========================================================================
D_HEAVY = 5.0e-10           # m, effective collision diameter, heavy particles
D_DIMER = 5.5e-10           # m, NaCl-NaCl (large: mu = 9 D, long-range attraction)
EFF_HEAVY_ION = 0.01        # heavy-particle impact ionization is inefficient
K_MN_300 = 1.0e-13          # m^3/s, Na+ + Cl- mutual neutralization at 300 K

# rates are evaluated at T clamped to this range; above 30 kK the model is
# not extrapolated (the callers flag when the ceiling is hit)
T_MIN, T_MAX = 300.0, 30000.0

# =========================================================================
# PARTITION FUNCTIONS
# =========================================================================


def q_trans(mass, T):
    """Translational partition function per unit volume [m^-3]."""
    return (2.0 * pi * mass * KB * T / H ** 2) ** 1.5


def q_elec(levels, T):
    """Electronic partition function from a (E[cm^-1], g) list."""
    return sum(g * exp(-C2 * E / T) for E, g in levels)


def q_rot_linear(B_cm1, sigma, T):
    """Classical rigid rotor, linear molecule."""
    return T / (sigma * C2 * B_cm1)


def q_rot_nonlinear(T, thA=None, thB=None, thC=None, sigma=NA2CL2_SIGMA):
    """Classical rigid rotor, asymmetric top."""
    thA = THETA_A if thA is None else thA
    thB = THETA_B if thB is None else thB
    thC = THETA_C if thC is None else thC
    return sqrt(pi) / sigma * sqrt(T ** 3 / (thA * thB * thC))


def q_vib(freqs_cm1, T):
    """Harmonic oscillators, energies from the zero-point level."""
    q = 1.0
    for w in freqs_cm1:
        q /= (1.0 - exp(-C2 * w / T))
    return q


# total partition functions per volume
def Q_Na(T):
    return q_trans(M_NA, T) * q_elec(NA_LEVELS, T)


def Q_Cl(T):
    return q_trans(M_CL, T) * q_elec(CL_LEVELS, T)


def Q_Nap(T):
    return q_trans(M_NAP, T) * q_elec(NAP_LEVELS, T)


def Q_Clm(T):
    return q_trans(M_CLM, T) * q_elec(CLM_LEVELS, T)


def Q_Clp(T):
    return q_trans(M_CLP, T) * q_elec(CLP_LEVELS, T)


def Q_Ar(T):
    return q_trans(M_AR, T) * q_elec(AR_LEVELS, T)


def Q_Arp(T):
    return q_trans(M_ARP, T) * q_elec(ARP_LEVELS, T)


def Q_e(T):
    return q_trans(ME, T) * 2.0


def Q_NaCl(T):
    return (q_trans(M_NACL, T) * 1.0
            * q_rot_linear(NACL_B, NACL_SIGMA, T) * q_vib(NACL_FREQ, T))


def Q_Na2Cl2(T):
    return (q_trans(M_NA2CL2, T) * 1.0
            * q_rot_nonlinear(T) * q_vib(NA2CL2_FREQS, T))


# =========================================================================
# EQUILIBRIUM CONSTANTS  (number-density units)
# =========================================================================
def K_dimer(T):
    """R1: n(Na2Cl2)/n(NaCl)^2 [m^3]."""
    return Q_Na2Cl2(T) / Q_NaCl(T) ** 2 * exp(E_DIM * EV / (KB * T))


def K_dissoc(T):
    """R2: n(Na)n(Cl)/n(NaCl) [m^-3]."""
    return Q_Na(T) * Q_Cl(T) / Q_NaCl(T) * exp(-D0_NACL * EV / (KB * T))


def K_ionpair(T):
    """R3: n(Na+)n(Cl-)/n(NaCl) [m^-3]. Equals K2*K4/K5 identically."""
    return Q_Nap(T) * Q_Clm(T) / Q_NaCl(T) * exp(-E_IONPAIR * EV / (KB * T))


def K_saha_Na(T):
    """R4: n(Na+)n(e)/n(Na) [m^-3]."""
    return Q_Nap(T) * Q_e(T) / Q_Na(T) * exp(-IE_NA * EV / (KB * T))


def K_saha_Clm(T):
    """R5: n(Cl)n(e)/n(Cl-) [m^-3]. Written in the DETACHMENT direction."""
    return Q_Cl(T) * Q_e(T) / Q_Clm(T) * exp(-EA_CL * EV / (KB * T))


def K_saha_Cl(T):
    """R6: n(Cl+)n(e)/n(Cl) [m^-3]."""
    return Q_Clp(T) * Q_e(T) / Q_Cl(T) * exp(-IE_CL * EV / (KB * T))


def K_saha_Ar(T):
    """R9: n(Ar+)n(e)/n(Ar) [m^-3]."""
    return Q_Arp(T) * Q_e(T) / Q_Ar(T) * exp(-IE_AR * EV / (KB * T))


def K_mutual_neutralization(T):
    """R7: n(Na)n(Cl)/(n(Na+)n(Cl-)) = K2/K3."""
    return K_dissoc(T) / K_ionpair(T)


def K_ClpClm(T):
    """R8: n(Cl)^2/(n(Cl+)n(Cl-)) = K5/K6."""
    return K_saha_Clm(T) / K_saha_Cl(T)


def K_ArpClm(T):
    """R10: n(Ar)n(Cl)/(n(Ar+)n(Cl-)) = K5/K9."""
    return K_saha_Clm(T) / K_saha_Ar(T)


# =========================================================================
# RATE-CONSTANT MODELS
# =========================================================================
def collision_rate(T, m1, m2, d=D_HEAVY):
    """hard-sphere collision rate constant"""
    mu = m1 * m2 / (m1 + m2)
    return pi * d * d * sqrt(8.0 * KB * T / (pi * mu))


def k_thermal(T, E_eV, m1, m2, d=D_HEAVY, eff=1):
    """line-of-centers collision theory: eff * Z * exp(-E/kT) [m^3/s]."""
    return eff * collision_rate(T, m1, m2, d) * exp(-E_eV * EV / (KB * T))


# =========================================================================
# REACTION RATES
# each returns the NET rate of the reaction as written [reactions/m^3/s];
# negative = running backwards. n_inert is any spectator acting as a
# third body. R9 and R10 also take n_Ar and n_Arp.
# =========================================================================
def third_body(n_NaCl=0.0, n_Na2Cl2=0.0, n_Na=0.0, n_Cl=0.0, n_Nap=0.0,
               n_Clm=0.0, n_Clp=0.0, n_inert=0.0):
    """total heavy-particle density."""
    return (n_NaCl + n_Na2Cl2 + n_Na + n_Cl + n_Nap + n_Clm + n_Clp + n_inert)


def rate_R1_dimerization(T, n_NaCl=0.0, n_Na2Cl2=0.0, n_Na=0.0, n_Cl=0.0,
                         n_Nap=0.0, n_Clm=0.0, n_Clp=0.0, n_e=0.0,
                         n_inert=0.0):
    """2 NaCl <-> Na2Cl2"""
    ka = collision_rate(T, M_NACL, M_NACL, D_DIMER) 
    kd = ka / K_dimer(T)                            
    return ka * n_NaCl * n_NaCl - kd * n_Na2Cl2


def rate_R2_dissociation(T, n_NaCl=0.0, n_Na2Cl2=0.0, n_Na=0.0, n_Cl=0.0,
                         n_Nap=0.0, n_Clm=0.0, n_Clp=0.0, n_e=0.0,
                         n_inert=0.0):
    """NaCl + M <-> Na + Cl + M."""
    M = third_body(n_NaCl, n_Na2Cl2, n_Na, n_Cl, n_Nap, n_Clm, n_Clp, n_inert)
    kf = k_thermal(T, D0_NACL, M_NACL, M_NACL)        # m^3/s
    kr = kf / K_dissoc(T)                             # m^6/s
    return M * (kf * n_NaCl - kr * n_Na * n_Cl)


def rate_R3_ionpair(T, n_NaCl=0.0, n_Na2Cl2=0.0, n_Na=0.0, n_Cl=0.0,
                    n_Nap=0.0, n_Clm=0.0, n_Clp=0.0, n_e=0.0, n_inert=0.0):
    """NaCl + M <-> Na+ + Cl- + M"""
    M = third_body(n_NaCl, n_Na2Cl2, n_Na, n_Cl, n_Nap, n_Clm, n_Clp, n_inert)
    kf = k_thermal(T, E_IONPAIR, M_NACL, M_NACL)
    kr = kf / K_ionpair(T)
    return M * (kf * n_NaCl - kr * n_Nap * n_Clm)


def rate_R4_Na_ionization(T, n_NaCl=0.0, n_Na2Cl2=0.0, n_Na=0.0, n_Cl=0.0,
                          n_Nap=0.0, n_Clm=0.0, n_Clp=0.0, n_e=0.0,
                          n_inert=0.0):
    """Na <-> Na+ + e-"""
    M = third_body(n_NaCl, n_Na2Cl2, n_Na, n_Cl, n_Nap, n_Clm, n_Clp, n_inert)
    K = K_saha_Na(T)
    kf = k_thermal(T, IE_NA, M_NA, M_NACL, eff=EFF_HEAVY_ION)
    fwd = kf * M * n_Na
    rev = kf * M / K * n_Nap * n_e
    return fwd - rev


def rate_R5_Cl_attachment(T, n_NaCl=0.0, n_Na2Cl2=0.0, n_Na=0.0, n_Cl=0.0,
                          n_Nap=0.0, n_Clm=0.0, n_Clp=0.0, n_e=0.0,
                          n_inert=0.0):
    """Cl + e- (+ M) <-> Cl- (+ M)"""
    M = third_body(n_NaCl, n_Na2Cl2, n_Na, n_Cl, n_Nap, n_Clm, n_Clp, n_inert)
    K = K_saha_Clm(T)                              
    kdet_M = k_thermal(T, EA_CL, M_CLM, M_NACL)       
    kdet = kdet_M * M 
    katt = kdet / K                  
    return katt * n_Cl * n_e - kdet * n_Clm


def rate_R6_Cl_ionization(T, n_NaCl=0.0, n_Na2Cl2=0.0, n_Na=0.0, n_Cl=0.0,
                          n_Nap=0.0, n_Clm=0.0, n_Clp=0.0, n_e=0.0,
                          n_inert=0.0):
    """Cl <-> Cl+ + e-."""
    M = third_body(n_NaCl, n_Na2Cl2, n_Na, n_Cl, n_Nap, n_Clm, n_Clp, n_inert)
    K = K_saha_Cl(T)
    kf = k_thermal(T, IE_CL, M_CL, M_NACL, eff=EFF_HEAVY_ION)
    kk = kf * M
    return kk * n_Cl - kk / K * n_Clp * n_e


def rate_R7_mutual_neutralization(T, n_NaCl=0.0, n_Na2Cl2=0.0, n_Na=0.0,
                                  n_Cl=0.0, n_Nap=0.0, n_Clm=0.0, n_Clp=0.0,
                                  n_e=0.0, n_inert=0.0):
    """Na+ + Cl- <-> Na + Cl."""
    kf = K_MN_300 * sqrt(300.0 / T)
    kr = kf / K_mutual_neutralization(T)
    return kf * n_Nap * n_Clm - kr * n_Na * n_Cl


def rate_R8_ClpClm(T, n_NaCl=0.0, n_Na2Cl2=0.0, n_Na=0.0, n_Cl=0.0,
                   n_Nap=0.0, n_Clm=0.0, n_Clp=0.0, n_e=0.0, n_inert=0.0):
    """Cl+ + Cl- <-> Cl + Cl."""
    kf = K_MN_300 * sqrt(300.0 / T)
    K5 = K_saha_Clm(T)
    kr = kf * K_saha_Cl(T) / K5 if K5 > 0.0 else 0.0
    return kf * n_Clp * n_Clm - kr * n_Cl * n_Cl


def rate_R9_Ar_ionization(T, n_NaCl=0.0, n_Na2Cl2=0.0, n_Na=0.0, n_Cl=0.0,
                          n_Nap=0.0, n_Clm=0.0, n_Clp=0.0, n_e=0.0,
                          n_Ar=0.0, n_Arp=0.0, n_inert=0.0):
    """Ar <-> Ar+ + e-, heavy-particle impact with argon as the bath."""
    M = third_body(n_NaCl, n_Na2Cl2, n_Na, n_Cl, n_Nap, n_Clm, n_Clp,
                   n_inert + n_Ar + n_Arp)
    K = K_saha_Ar(T)
    kf = k_thermal(T, IE_AR, M_AR, M_AR, eff=EFF_HEAVY_ION)
    kk = kf * M

    if K <= 0.0:
        return kk * n_Ar

    return kk * n_Ar - kk / K * n_Arp * n_e


def rate_R10_ArpClm(T, n_NaCl=0.0, n_Na2Cl2=0.0, n_Na=0.0, n_Cl=0.0,
                    n_Nap=0.0, n_Clm=0.0, n_Clp=0.0, n_e=0.0,
                    n_Ar=0.0, n_Arp=0.0, n_inert=0.0):
    """Ar+ + Cl- <-> Ar + Cl."""
    kf = K_MN_300 * sqrt(300.0 / T)
    K5 = K_saha_Clm(T)
    kr = kf * K_saha_Ar(T) / K5 if K5 > 0.0 else 0.0
    return kf * n_Arp * n_Clm - kr * n_Ar * n_Cl


ALL_REACTIONS = [
    ("R1  2 NaCl <-> Na2Cl2    ", rate_R1_dimerization, ("n_Na2Cl2",)),
    ("R2  NaCl <-> Na + Cl     ", rate_R2_dissociation, ("n_Na", "n_Cl")),
    ("R3  NaCl <-> Na+ + Cl-   ", rate_R3_ionpair, ("n_Nap", "n_Clm")),
    ("R4  Na <-> Na+ + e-      ", rate_R4_Na_ionization, ("n_Nap",)),
    ("R5  Cl + e- <-> Cl-      ", rate_R5_Cl_attachment, ("n_Clm",)),
    ("R6  Cl <-> Cl+ + e-      ", rate_R6_Cl_ionization, ("n_Clp",)),
    ("R7  Na+ + Cl- <-> Na + Cl", rate_R7_mutual_neutralization,
     ("n_Na", "n_Cl")),
]


# =========================================================================
# SPECIES PRODUCTION RATES
# =========================================================================
def net_production_rates(T, n_NaCl=0.0, n_Na2Cl2=0.0, n_Na=0.0, n_Cl=0.0,
                         n_Nap=0.0, n_Clm=0.0, n_Clp=0.0, n_e=0.0,
                         n_inert=0.0, include_R7=True):
    """dn/dt for every species [m^-3 s^-1], R1-R7 only."""
    kw = dict(n_NaCl=n_NaCl, n_Na2Cl2=n_Na2Cl2, n_Na=n_Na, n_Cl=n_Cl,
              n_Nap=n_Nap, n_Clm=n_Clm, n_Clp=n_Clp, n_e=n_e, n_inert=n_inert)
    r1 = rate_R1_dimerization(T, **kw)
    r2 = rate_R2_dissociation(T, **kw)
    r3 = rate_R3_ionpair(T, **kw)
    r4 = rate_R4_Na_ionization(T, **kw)
    r5 = rate_R5_Cl_attachment(T, **kw)
    r6 = rate_R6_Cl_ionization(T, **kw)
    r7 = rate_R7_mutual_neutralization(T, **kw) if include_R7 else 0.0
    return {
        "NaCl":   -2.0 * r1 - r2 - r3,
        "Na2Cl2":  r1,
        "Na":      r2 - r4 + r7,
        "Cl":      r2 - r5 - r6 + r7,
        "Na+":     r3 + r4 - r7,
        "Cl-":     r3 + r5 - r7,
        "Cl+":     r6,
        "e-":      r4 - r5 + r6,
    }