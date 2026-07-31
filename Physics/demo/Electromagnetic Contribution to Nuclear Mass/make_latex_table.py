#!/usr/bin/env python3
"""
Emit a LaTeX table from mirror_nuclei_results.csv.

Columns:  pair | A | R_avg | naive | dm_exp | dm_formula | dev
(the electromagnetic term is dropped; `naive` sits directly after R_avg)

Writes mirror_nuclei_table.tex -- a standalone, compilable document.
"""

import pandas as pd

MeV = 1e-6

df = pd.read_csv("mirror_nuclei_results.csv").sort_values(["dNZ", "A"])


def nuclide(s):
    """'He-3' -> '\\nuc{3}{He}'"""
    el, a = s.split("-")
    return rf"\nuc{{{a}}}{{{el}}}"


def num(x, nd=3, signed=True):
    """Math-mode number with a real minus sign; explicit + when signed."""
    if signed:
        return f"${x:+.{nd}f}$"
    return f"${x:.{nd}f}$"


GROUPS = {
    1: r"$N-Z = 1$ \quad ($T_z = \pm\tfrac{1}{2}$)",
    2: r"$N-Z = 2$ \quad ($T_z = \pm 1$)",
    4: r"$N-Z = 4$ \quad ($T_z = \pm 2$)",
}

rows = []
for dnz, block in df.groupby("dNZ"):
    rows.append(r"\addlinespace[2pt]")
    rows.append(rf"\multicolumn{{7}}{{l}}{{\itshape {GROUPS[dnz]}}} \\[1pt]")
    for _, r in block.iterrows():
        star = r"$^{\ast}$" if pd.notna(r["r_m"]) else ""
        pair = f"{nuclide(r['parent'])}/{nuclide(r['mirror'])}{star}"
        rows.append(
            " & ".join([
                pair,
                f"{int(r['A'])}",
                f"${r['R_avg']:.3f}$",
                num(r["naive"] * MeV, signed=False),
                num(r["dm_exp"] * MeV),
                num(r["dm_form"] * MeV),
                num(r["dev_pct"], 1),
            ])
            + r" \\"
        )

body = "\n".join(rows)

TEX = r"""\documentclass[11pt]{article}
\usepackage[a4paper,margin=2cm]{geometry}
\usepackage{booktabs}
\usepackage{amsmath}

% \nuc{15}{N} -> superscripted mass number, tight against the symbol
\newcommand{\nuc}[2]{\ensuremath{{}^{#1}\mathrm{#2}}}

\pagestyle{empty}
\begin{document}

\begin{table}[htbp]
\centering
\small
\setlength{\tabcolsep}{7pt}
\renewcommand{\arraystretch}{1.08}

\caption{Mirror-nuclei mass differences: uniformly charged sphere versus
experiment. $\Delta m = m_{\mathrm{nuc}}(N\ \mathrm{protons}) -
m_{\mathrm{nuc}}(Z\ \mathrm{protons})$, evaluated from
\[
  \Delta m \;=\; \frac{3e}{20\pi\epsilon_{0}}\,
  \frac{(N-Z)(A-1)}{R_{\mathrm{avg}}} \;+\; (Z-N)(m_{n}-m_{p}),
\]
with $R_{\mathrm{avg}} = \sqrt{5/3}\,r_{\mathrm{rms}}$ the equivalent
uniform-sphere radius. Masses are AME2020 (converted from atomic to nuclear),
charge radii from Angeli--Marinova and recent mirror-radius measurements.
The \emph{naive} column is the nucleon mass-difference term alone, i.e.\ the
prediction obtained by discarding the electrostatic energy entirely.
\emph{dev} is the fractional error of the model on the Coulomb displacement
energy.}
\label{tab:mirror}

\begin{tabular}{l c c r r r r}
\toprule
 & & $R_{\mathrm{avg}}$ & naive & $\Delta m_{\mathrm{exp}}$
 & $\Delta m_{\mathrm{formula}}$ & dev \\
pair & $A$ & [fm] & [MeV] & [MeV] & [MeV] & [\%] \\
\midrule
BODY
\bottomrule
\end{tabular}

\vspace{4pt}
{\footnotesize $^{\ast}$ charge radii measured for \emph{both} members;
otherwise the neutron-rich member's radius is used for both.\par}

\end{table}

\end{document}
"""

with open("mirror_nuclei_table.tex", "w") as f:
    f.write(TEX.replace("BODY", body))

print(f"wrote mirror_nuclei_table.tex  ({len(df)} rows)")
