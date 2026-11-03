# Acoustic cavitation in molten NaCl (and other low-vapor-pressure melts)

Single-bubble simulations of acoustic cavitation: an argon bubble in a melt, driven by a sinusoidal sound field. Each run reports the conditions inside the bubble at collapse: peak temperature, peak pressure, maximum and minimum radius, and the argon ionization. The physics, its assumptions and its error budget are described in `theory.pdf`. This file only covers how to run the code.

## Contents

```
theory.pdf                  the report: model, validation, error estimates
simulation/
    novapor.py              vapor-free model for ANY low-vapor-pressure liquid
    nacl.py                 full model for molten NaCl, including NaCl vapor chemistry
    nacl_kinetics.py        rate and equilibrium-constant library used by both (not run directly)
```

**Which script do I want?**

- **Another salt, a metal, or any other liquid:** use `novapor.py`. It needs only four liquid properties: density, sound speed, viscosity and surface tension. The liquid's vapor is neglected, which is reasonable when its vapor pressure at the melt temperature is below about 100 Pa (see "When the vapor-free model applies" below).
- **Molten NaCl specifically:** use `nacl.py`. It includes the NaCl vapor and its chemistry, including dimerization, dissociation and ionization. It is about 5 times slower than `novapor.py`. For NaCl at 1100 K the two agree to within about 1% at the default settings.

## Requirements

- Python 3.9 or newer
- numpy and scipy
- matplotlib (optional but recommended; interpreting the simulation without plots can be difficult, particularly for troubleshooting)

```
pip install numpy scipy matplotlib
```

There is nothing to install or build. Keep `nacl_kinetics.py` in the same folder as the other two scripts.

## Quick start

From the top-level folder:

```
python simulation/novapor.py
python simulation/nacl.py
```

Both run the default case: NaCl at 1100 K, with an ambient bubble radius of 5 µm, a frequency of 20 kHz, a pressure amplitude of 1.2 atm, and 3 acoustic cycles. `novapor.py` takes about 7 s and `nacl.py` about 40 s on a laptop. The run prints a summary and writes plots to a new folder in the current directory.

Sample output (`novapor.py`, defaults):

```
------------------------------------------------------------------------
  NaCl:  T0 = 1100 K   rho = 1542 kg/m3   c = 1726 m/s   (no vapor)
         mu = 0.985 mPa s   sigma = 0.112 N/m
  R0 = 5 um   f = 20 kHz   pA = 1.2 atm   (Blake ~ 1.09 atm)
------------------------------------------------------------------------
  R_max          =    24.778 um   (expansion ratio   4.96)
  R_min          =    0.7259 um   (hard-core radius 0.3999 um)
  peak T (core)  =     15439 K
  peak p_g       =     0.804 GPa
  peak |dR/dt|/c =     0.221
  conservation audit: Ar 1.1e-15  charge 3.2e-26
  integration: 7291 steps, 55683 RHS evals, 6.9 s
------------------------------------------------------------------------
```

An explanation of the entries in this output can be found at the top of novapor.py.

## Using `novapor.py` for a different liquid

1. Open `simulation/novapor.py` and edit the **LIQUID PROPERTIES** block near the top. Set the name, the melting point, the default melt temperature T₀, and the temperature range over which your property fits are valid. Then fill in the four property functions. Each one takes T in kelvin and returns SI units. A plain number is fine, e.g. `return 7750.0`.
2. Run `python simulation/novapor.py` with whatever options you need (see below).

No data for your liquid? The **GUIDE: LIQUIDS WITHOUT GOOD PROPERTY DATA** at the bottom of `novapor.py` may help.

**Note for liquid metals.** Their surface tensions are around 1 to 2 N/m, about ten times those of molten salts. This pushes the Blake threshold significantly higher than is typical for, e.g., NaCl for smaller bubbles. For example, a 5 µm bubble in molten cobalt has a threshold of about 3.7 atm. Below the threshold, the bubble only oscillates gently. Just above the threshold, the peak temperature rises very steeply with the pressure amplitude. In that range, a 10% uncertainty in surface tension can change the peak T by an order of magnitude. Larger bubbles are much less sensitive. Either way, run the sensitivity check described in the guide.

### When the vapor-free model applies

`novapor.py` ignores the liquid's vapor. The report (Section 5) finds that for saturation vapor pressures below about 100 Pa, the vapor changes the peak temperature by less than about 10% at intermediate driving conditions. Above that, the model does not fail abruptly, but its error grows and is harder to estimate. The bubble gas is always argon.

## Options

All options are optional; anything you leave out takes its default. The same options work for both scripts.

| Option | Meaning | Default |
|---|---|---|
| `--R0` | ambient bubble radius, in µm | 5 |
| `--f` | driving frequency, in kHz | 20 |
| `--pA` | driving pressure amplitude, in atm | 1.2 |
| `--cycles` | number of acoustic cycles to simulate | 3 |
| `--T0` | melt temperature, in K | 1100 for NaCl; `T0_DEFAULT` in `novapor.py` |
| `--csv` | also write the time series (t, R, dR/dt, T, particle numbers) to a CSV file in the output folder | off |
| `--out NAME` | name of the output folder | built from the parameters, e.g. `novapor_R0-5um_f-20kHz_pA-1.2atm` |
| `--no-plot` | skip the plots | plots on |
| `--lambda-max` | caps how fast each reaction relaxes toward equilibrium, in s⁻¹; only needed if a run fails (see Troubleshooting) | no cap |
| `--ion-speedup` | multiplies the ionization rates. 10⁶ or more gives the Saha (equilibrium) limit; 10⁻⁶ or less effectively switches ionization off | 1 |
| `--electron-kappa` | adds electron heat conduction, as a deliberate overestimate | off |

Example:

```
python simulation/novapor.py --R0 10 --f 26 --pA 1.5 --cycles 5 --csv
```

## Output

Each run writes a folder like this:

```
novapor_R0-5um_f-20kHz_pA-1.2atm/
    basic/
        radius.png, temperature.png, pressure.png     against time, all cycles
        N_plots/                                      particle number of each species
    analysis_plots/
        ionization.png                                zoom on the hottest collapse
        (nacl.py also writes ambipolar.png and modes_*.png)
    novapor_R0-5um_f-20kHz_pA-1.2atm.csv               only with --csv
```

Rerunning with the same parameters overwrites that folder. If a folder with that name already exists and was *not* made by these scripts, the run stops instead of overwriting it.

## Interpreting the results

- **Accuracy.** The model is semi-quantitative. For peak temperatures of about 10 to 25 kK, the error sources that could be quantified add up to about 15% (report, Section 4). Treat results above 25 kK with increasing caution.
- **Above 30 kK** the chemistry is held at the upper limit of the kinetics model, and the output prints a note saying so. Treat such collapses as qualitative.
- **"wall Mach > 1"** means the bubble wall is moving faster than the speed of sound in the liquid. The bubble-dynamics equation is then at the edge of its validity.
- **"pA below the Blake threshold"** means no violent collapse occurred.
- **Peak T is the maximum over all simulated cycles.** Some bubbles collapse differently from one cycle to the next. A larger bubble, for instance, may alternate between strong and weak collapses. Look at `temperature.png` before comparing two runs, and keep `--cycles` the same between them.
- **Single bubble.** In multibubble fields, interactions between bubbles may make the real peak temperatures lower.

## Troubleshooting

**The run fails, stalls, or gives an error you don't understand.** Add `--lambda-max 1e16`; if that doesn't help, try `1e14`. This caps how fast the ionization reactions can relax, without changing the equilibrium they relax to. It should leave the results unchanged. To check that it does, lower the value by a factor of 10 and confirm that the peak T stays the same.

**"T0 is below the melting point"**: raise `--T0`, or check `T_MELT` in the liquid block.

**"WARNING: T0 is outside the property fits' range"**: the liquid properties are being extrapolated. This is fine for small extrapolations; otherwise, widen the fits.

## Reference

L. Hagendoorn, *Simulation of Acoustic Cavitation in Molten NaCl* (2026), included as `theory.pdf`.
