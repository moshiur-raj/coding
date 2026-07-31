#!/usr/bin/env python3
"""
Mirror-nuclei mass differences: uniform-charged-sphere model vs experiment.
================================================================================

For a mirror pair (Z, N) <-> (N, Z), A = Z + N, the nuclear mass difference is

    dm = m_nuc(N protons) - m_nuc(Z protons)
       = dE_C  -  (N - Z) (m_n - m_p) c^2                              [eV]

with the electrostatic term taken from the uniformly charged sphere,
E_C = (3/5) k Q^2 / R, and pairwise (i<j) counting Z(Z-1) rather than Z^2:

    dE_C = (3/5) * [e / (4 pi eps0)] * (N - Z)(A - 1) / R_avg          [eV]

since  N(N-1) - Z(Z-1) = (N - Z)(A - 1).

Note the single power of e: energies are reported in eV, and eV is numerically
the electrostatic potential in volts, so e/(4 pi eps0) = 1.44e6 V*fm is the
whole constant needed when R is in fm.

R is the EQUIVALENT UNIFORM-SPHERE radius, not the rms charge radius.  For a
uniform sphere  r_rms = sqrt(3/5) R,  so  R = sqrt(5/3) * r_rms.  Skipping this
conversion is worth a factor of 1.29 and is the single most common way to get
this calculation wrong.

Outputs
-------
  mirror_nuclei_results.csv   full numerical table
  mirror_nuclei_table.txt     formatted table, same content
  mirror_nuclei_chart.png     four-panel chart
"""

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.constants import e, epsilon_0, pi, physical_constants

# =============================================================================
# 1. PHYSICAL CONSTANTS  (scipy / CODATA)
# =============================================================================

# e / (4 pi eps0) in V*fm.  Energy in eV = this / R[fm] for a unit-charge pair.
K_EV_FM = e / (4.0 * pi * epsilon_0) * 1e15          # 1.43996e6  V*fm

MN_EV = physical_constants["neutron mass energy equivalent in MeV"][0] * 1e6
MP_EV = physical_constants["proton mass energy equivalent in MeV"][0] * 1e6
ME_EV = physical_constants["electron mass energy equivalent in MeV"][0] * 1e6
DNP_EV = MN_EV - MP_EV                                # 1.29333e6 eV

SPHERE = 3.0 / 5.0                                    # uniform-sphere coefficient
RMS_TO_UNIFORM = np.sqrt(5.0 / 3.0)                   # R = sqrt(5/3) * r_rms

print("Constants from scipy.constants / CODATA")
print(f"  e/(4*pi*eps0)      = {K_EV_FM:,.1f} eV*fm")
print(f"  (3/5)*e/(4pi eps0) = {SPHERE * K_EV_FM:,.1f} eV*fm   <- leading coefficient")
print(f"  m_n - m_p          = {DNP_EV:,.1f} eV")
print(f"  m_e                = {ME_EV:,.1f} eV")
print(f"  sqrt(5/3)          = {RMS_TO_UNIFORM:.6f}\n")

# =============================================================================
# 2. LOAD EXPERIMENTAL DATA
# =============================================================================

df = pd.read_csv(
    "mirror_nuclei_data.txt",
    sep=r"\s+",
    comment="#",
    na_values=["nan"],
)

# Derived nucleon numbers
df["Np"] = df["A"] - df["Zp"]        # neutrons in the parent
df["Zm"] = df["Np"]                  # protons in the mirror
df["Nm"] = df["Zp"]
df["dNZ"] = df["Np"] - df["Zp"]      # (N - Z), the isospin lever arm
df["Tz2"] = df["dNZ"]                # 2*Tz

# =============================================================================
# 3. RADII:  rms charge radius  ->  equivalent uniform-sphere radius
# =============================================================================

df["Rp"] = RMS_TO_UNIFORM * df["r_p"]
df["Rm"] = RMS_TO_UNIFORM * df["r_m"]

# R_avg: mean of whatever is measured.  When only the parent is known (the
# common case) this reduces to the parent's radius, i.e. the equal-radius
# assumption the user asked for.
df["R_avg"] = np.nanmean(np.vstack([df["Rp"].values, df["Rm"].values]), axis=0)
df["both_radii"] = df["r_m"].notna()

# =============================================================================
# 4. EXPERIMENT
# =============================================================================

# AME mass excesses are ATOMIC.  The proton-rich member drags along (N-Z) extra
# electrons, so remove them to get the NUCLEAR mass difference.
df["dm_atomic"] = (df["ME_m"] - df["ME_p"]) * 1e3                 # eV
df["dm_exp"] = df["dm_atomic"] - df["dNZ"] * ME_EV                # eV, nuclear

# The experimental Coulomb displacement energy: strip off the free-nucleon mass
# difference.  This step ASSUMES charge symmetry of the strong force -- it is a
# convention for defining dE_C, not a measurement.
df["EC_exp"] = df["dm_exp"] + df["dNZ"] * DNP_EV                  # eV

# =============================================================================
# 5. MODEL
# =============================================================================

# (a) the user's formula: one shared radius, pair counting Z(Z-1)
df["EC_form"] = SPHERE * K_EV_FM * df["dNZ"] * (df["A"] - 1) / df["R_avg"]
df["naive"] = -df["dNZ"] * DNP_EV                                  # (Z-N)(m_n-m_p)
df["dm_form"] = df["EC_form"] + df["naive"]

# (b) for reference: what Z^2 counting (no self-energy subtraction) would give
df["EC_Z2"] = (
    SPHERE * K_EV_FM * (df["Zm"] ** 2 - df["Zp"] ** 2) / df["R_avg"]
)

# (c) exact two-radius version, only where both radii are measured
df["EC_2R"] = SPHERE * K_EV_FM * (
    df["Zm"] * (df["Zm"] - 1) / df["Rm"] - df["Zp"] * (df["Zp"] - 1) / df["Rp"]
)
df["dm_2R"] = df["EC_2R"] + df["naive"]

# =============================================================================
# 6. DIAGNOSTICS
# =============================================================================

df["ratio"] = df["EC_form"] / df["EC_exp"]                # model / experiment
df["dev_pct"] = 100.0 * (df["ratio"] - 1.0)
df["resid"] = df["dm_form"] - df["dm_exp"]

# Radius that would reproduce the measured dE_C exactly, and the implied r0
df["R_fit"] = SPHERE * K_EV_FM * df["dNZ"] * (df["A"] - 1) / df["EC_exp"]
df["r0_fit"] = df["R_fit"] / df["A"] ** (1.0 / 3.0)

# Fractional weight of the electromagnetic term in the total mass difference
df["EM_frac"] = df["EC_form"] / (df["EC_form"] + np.abs(df["naive"]))

df = df.sort_values(["dNZ", "A"]).reset_index(drop=True)

# =============================================================================
# 7. TABLES
# =============================================================================

MeV = 1e-6


def fmt_table(d):
    lines = []
    head = (
        f"{'pair':>14} {'A':>3} {'N-Z':>3} {'R_avg':>6} "
        f"{'dm_exp':>9} {'dm_form':>9} {'naive':>9} {'EM term':>9} "
        f"{'EC_exp':>9} {'dev%':>7} {'r0_fit':>7}"
    )
    units = (
        f"{'':>14} {'':>3} {'':>3} {'[fm]':>6} "
        f"{'[MeV]':>9} {'[MeV]':>9} {'[MeV]':>9} {'[MeV]':>9} "
        f"{'[MeV]':>9} {'':>7} {'[fm]':>7}"
    )
    lines += [head, units, "-" * len(head)]
    last = None
    for _, r in d.iterrows():
        if last is not None and r["dNZ"] != last:
            lines.append("-" * len(head))
        last = r["dNZ"]
        pair = f"{r['parent']}/{r['mirror']}"
        star = "*" if r["both_radii"] else " "
        lines.append(
            f"{pair:>13}{star} {int(r['A']):>3} {int(r['dNZ']):>3} "
            f"{r['R_avg']:>6.3f} "
            f"{r['dm_exp']*MeV:>9.3f} {r['dm_form']*MeV:>9.3f} "
            f"{r['naive']*MeV:>9.3f} {r['EC_form']*MeV:>9.3f} "
            f"{r['EC_exp']*MeV:>9.3f} "
            f"{r['dev_pct']:>+7.1f} {r['r0_fit']:>7.3f}"
        )
    lines.append("-" * len(head))
    lines.append("* = charge radii measured for BOTH members; else parent's radius used for both.")
    return "\n".join(lines)


table = fmt_table(df)
print(table)

# --- summary statistics ------------------------------------------------------
mask_all = df["EC_exp"].notna()
mask_light = mask_all & (df["A"] <= 15)
mask_heavy = mask_all & (df["A"] > 15)

summary = []
summary.append("")
summary.append("SUMMARY")
summary.append("-" * 78)
summary.append(
    f"pairs analysed                          : {mask_all.sum()}"
)
summary.append(
    f"mean |deviation| of dE_C, all pairs     : {df.loc[mask_all,'dev_pct'].abs().mean():5.1f} %"
)
summary.append(
    f"mean |deviation| of dE_C, A <= 15       : {df.loc[mask_light,'dev_pct'].abs().mean():5.1f} %"
)
summary.append(
    f"mean |deviation| of dE_C, A >  15       : {df.loc[mask_heavy,'dev_pct'].abs().mean():5.1f} %"
)
summary.append(
    f"mean signed deviation,   A >  15        : {df.loc[mask_heavy,'dev_pct'].mean():+5.1f} %"
)
summary.append(
    f"rms residual on dm,      A >  15        : "
    f"{np.sqrt((df.loc[mask_heavy,'resid']**2).mean())*MeV:5.3f} MeV"
)
summary.append(
    f"mean fitted r0 = R_fit/A^(1/3), A > 15  : "
    f"{df.loc[mask_heavy,'r0_fit'].mean():5.3f} fm"
)
summary.append("")
summary.append("The residual is NOT the Nolen-Schiffer anomaly.  NS is a ~7% SHORTFALL of")
summary.append("careful microscopic calculations against experiment.  The raw uniform")
summary.append("sphere here runs ~3% HIGH because it omits the Coulomb exchange term, the")
summary.append("finite proton charge distribution, and the c.m. correction -- all of which")
summary.append("reduce the computed dE_C by several percent.  Apply those and this model")
summary.append("would land below experiment, which is where NS lives.")
summary.append("")
summary.append("The largest outliers are structural, not statistical: 13N, 17F and 41Sc are")
summary.append("exactly the cases where the odd proton occupies a new shell just outside a")
summary.append("closed core (12C, 16O, 40Ca).  That proton orbits well outside the charge")
summary.append("radius of the core, so it feels less repulsion than a uniformly smeared")
summary.append("fluid would predict, and the model over-shoots.  17F is the extreme case:")
summary.append("its valence 2s1/2 proton is famously halo-like.")
summary.append("")
summary.append("If the electromagnetic term is dropped entirely and only")
summary.append("(Z-N)(m_n - m_p) is kept, EVERY mirror pair comes out with the wrong")
summary.append("sign: the proton-rich member is predicted lighter, when it is heavier.")
summary.append(f"  cases with correct sign, naive term alone : "
               f"{int((np.sign(df.loc[mask_all,'naive']) == np.sign(df.loc[mask_all,'dm_exp'])).sum())}"
               f" / {int(mask_all.sum())}")
summary.append(f"  cases with correct sign, full formula     : "
               f"{int((np.sign(df.loc[mask_all,'dm_form']) == np.sign(df.loc[mask_all,'dm_exp'])).sum())}"
               f" / {int(mask_all.sum())}")

# --- equal-radius test -------------------------------------------------------
sub = df[df["both_radii"]]
summary.append("")
summary.append("EQUAL-RADIUS ASSUMPTION (pairs with both radii measured)")
summary.append("-" * 78)
summary.append(
    f"{'pair':>14} {'r_p':>7} {'r_m':>7} {'d r/r':>7} "
    f"{'EC_exp':>8} {'EC(Ravg)':>9} {'EC(2 radii)':>12}"
)
for _, r in sub.iterrows():
    summary.append(
        f"{r['parent']+'/'+r['mirror']:>14} {r['r_p']:>7.4f} {r['r_m']:>7.4f} "
        f"{100*(r['r_m']-r['r_p'])/r['r_p']:>6.1f}% "
        f"{r['EC_exp']*MeV:>8.3f} {r['EC_form']*MeV:>9.3f} {r['EC_2R']*MeV:>12.3f}"
    )
summary.append("")
summary.append("Using each member's own radius is EXACT within the model and yet agrees")
summary.append("WORSE.  The reason is error amplification, not physics.  For Fe-54/Ni-54")
summary.append("the two absolute Coulomb energies are ~118 and ~135 MeV; their difference")
summary.append("is 17 MeV.  A 1% error in either radius -- from the measurement, or from")
summary.append("the uniform-sphere idealisation itself -- moves a 130 MeV number by 1.3")
summary.append("MeV, i.e. by 7% of the answer.  Holding R fixed makes that systematic")
summary.append("cancel in the subtraction.  The equal-radius assumption is therefore not")
summary.append("just a convenience: it is the numerically stable way to use this model.")
summary.append("(The real radius difference is a genuine, measured effect -- the extra")
summary.append("proton's repulsion inflates the proton-rich partner -- and is used to")
summary.append("constrain the symmetry-energy slope L.  It is simply too small to extract")
summary.append("through a formula this crude.)")

summary_txt = "\n".join(summary)
print(summary_txt)

with open("mirror_nuclei_table.txt", "w") as f:
    f.write(__doc__ + "\n")
    f.write(table + "\n")
    f.write(summary_txt + "\n")

cols = [
    "A", "parent", "mirror", "Zp", "Zm", "dNZ", "r_p", "r_m", "R_avg",
    "dm_atomic", "dm_exp", "dm_form", "naive", "EC_form", "EC_exp",
    "EC_Z2", "EC_2R", "dm_2R", "ratio", "dev_pct", "resid", "R_fit", "r0_fit",
    "EM_frac",
]
df[cols].to_csv("mirror_nuclei_results.csv", index=False, float_format="%.6g")

# =============================================================================
# 8. CHART
# =============================================================================

plt.rcParams.update({
    "font.size": 9,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "axes.axisbelow": True,
    "figure.dpi": 140,
})

fig, ax = plt.subplots(2, 2, figsize=(12.5, 9.0))
colors = {1: "#1f77b4", 2: "#d62728", 4: "#2ca02c"}
marks = {1: "o", 2: "s", 4: "D"}

# ---- (a) universality: dE_C vs the geometric variable ----------------------
a = ax[0, 0]
x = df["dNZ"] * (df["A"] - 1) / df["R_avg"]
xs = np.linspace(0, x.max() * 1.05, 50)
a.plot(xs, SPHERE * K_EV_FM * xs * MeV, "k-", lw=1.2, zorder=1,
       label=r"$\frac{3}{5}\frac{e}{4\pi\epsilon_0}\cdot x$  (no free parameters)")
for d, g in df.groupby("dNZ"):
    a.scatter(g["dNZ"] * (g["A"] - 1) / g["R_avg"], g["EC_exp"] * MeV,
              s=42, marker=marks[d], color=colors[d], edgecolor="k",
              linewidth=0.5, zorder=3, label=f"$N-Z={d}$")
a.set_xlabel(r"$(N-Z)(A-1)\,/\,R_{\rm avg}$   [fm$^{-1}$]")
a.set_ylabel(r"Coulomb displacement energy  $\Delta E_C$  [MeV]")
a.set_title("(a)  Measured $\\Delta E_C$ against the charged-sphere variable")
a.legend(frameon=False, fontsize=8, loc="upper left")

# ---- (b) predicted vs measured mass difference -----------------------------
a = ax[0, 1]
lim = [-2.5, max(df["dm_exp"].max(), df["dm_form"].max()) * MeV * 1.08]
a.plot(lim, lim, "k-", lw=1.0, zorder=1)
for d, g in df.groupby("dNZ"):
    a.scatter(g["dm_exp"] * MeV, g["dm_form"] * MeV, s=42, marker=marks[d],
              color=colors[d], edgecolor="k", linewidth=0.5, zorder=3,
              label=f"full formula, $N-Z={d}$")
a.scatter(df["dm_exp"] * MeV, df["naive"] * MeV, s=26, marker="x",
          color="grey", zorder=2, label=r"naive $(Z-N)(m_n-m_p)$ only")
a.axhline(0, color="grey", lw=0.6, ls=":")
a.set_xlim(lim); a.set_ylim(-6, lim[1])
a.set_xlabel(r"measured $\Delta m$  [MeV]")
a.set_ylabel(r"predicted $\Delta m$  [MeV]")
a.set_title("(b)  Prediction vs experiment, with and without the EM term")
a.legend(frameon=False, fontsize=8, loc="upper left")

# ---- (c) the decomposition, stacked ----------------------------------------
a = ax[1, 0]
d1 = df[df["dNZ"] == 1].sort_values("A")
idx = np.arange(len(d1))
a.bar(idx, d1["EC_form"] * MeV, 0.62, color="#c6dbef", edgecolor="#1f77b4",
      label=r"electromagnetic  $+\frac{3}{5}\frac{e}{4\pi\epsilon_0}\frac{(N-Z)(A-1)}{R}$")
a.bar(idx, d1["naive"] * MeV, 0.62, color="#fdd0a2", edgecolor="#d94801",
      label=r"nucleon mass  $(Z-N)(m_n-m_p)$")
a.plot(idx, d1["dm_form"] * MeV, "k^-", ms=4.5, lw=1.0, label="net prediction")
a.plot(idx, d1["dm_exp"] * MeV, "kv", ms=5.5, mfc="w", label="measured")
a.axhline(0, color="k", lw=0.8)
a.set_xticks(idx)
a.set_xticklabels([f"{p.split('-')[1]}\n{p.split('-')[0]}/{m.split('-')[0]}"
                   for p, m in zip(d1["parent"], d1["mirror"])], fontsize=6.5)
a.set_ylabel(r"contribution to $\Delta m$  [MeV]")
a.set_title("(c)  The two terms competing, $N-Z=1$ series")
a.legend(frameon=False, fontsize=7.5, loc="upper left")

# ---- (d) fractional deviation vs A -----------------------------------------
a = ax[1, 1]
for d, g in df.groupby("dNZ"):
    g = g.sort_values("A")
    a.plot(g["A"], g["dev_pct"], marks[d] + "-", color=colors[d], ms=5,
           lw=0.9, mec="k", mew=0.4, label=f"$N-Z={d}$")
a.axhline(0, color="k", lw=0.9)
a.axhspan(0, 5, color="grey", alpha=0.12)
a.text(56, 4.3, "raw sphere runs ~3% high;\nexchange + finite-proton-size\ncorrections push it down",
       fontsize=6.8, color="dimgrey", ha="right", va="top")
for lab, xx in [("$^{13}$N", 13), ("$^{17}$F", 17), ("$^{41}$Sc", 41)]:
    yy = df.loc[df["A"] == xx, "dev_pct"].iloc[0]
    a.annotate(lab, (xx, yy), textcoords="offset points", xytext=(3, 5),
               fontsize=7, color="#555555")
a.set_xlabel("mass number $A$")
a.set_ylabel(r"$(\Delta E_C^{\rm model}/\Delta E_C^{\rm exp}-1)$  [%]")
a.set_title("(d)  Model minus experiment: worst where a valence proton sits outside a closed core")
a.set_xlim(0, 57)
a.legend(frameon=False, fontsize=8, loc="lower right")

fig.suptitle(
    "Mirror-nuclei mass differences: uniform charged sphere vs experiment\n"
    r"$\Delta m = \frac{3e}{20\pi\epsilon_0}\frac{(N-Z)(A-1)}{R_{\rm avg}} + (Z-N)(m_n-m_p)$",
    fontsize=11.5,
)
fig.tight_layout(rect=[0, 0, 1, 0.94])
fig.savefig("mirror_nuclei_chart.pdf", bbox_inches="tight")
print("\nwrote mirror_nuclei_results.csv, mirror_nuclei_table.txt, mirror_nuclei_chart.pdf")
