# Verification scripts

Two independent implementations used to check the document, plus the logs
of every run whose numbers are quoted in it.

* `pw.py`, `scf.py` and the scripts built on them: a NumPy/SciPy reference
  that follows the document's equations and pseudocode literally.
* `jax_from_text.py`: a JAX implementation written from the text of the
  document alone (it shares no code with `pw.py`). It exists to prove that
  the document is self-contained, and it reproduces the regression numbers
  of Chapter 14 (Verification) to every quoted digit.

Neither is optimised, and both are the answer key: **reading them before
writing your own code spoils the exercise.**

Requirements: Python 3, numpy, scipy; jax for `jax_from_text.py`
(`pip install jax`, CPU is enough); optionally `colour-science` (used by
`optics_post.py` only to cross-check the analytic colour-matching functions). Run scripts from this folder. Several
scripts load `n_lda40.npy`, the converged density written by `run_main.py`.

| File | What it does |
|---|---|
| `pw.py` | The library: HGH parameters, projectors, L.S matrices, XC (LDA, PBE), smearing, Ewald, Hamiltonian application, eigensolvers |
| `scf.py` | SCF driver (Kerker/Pulay mixing, both energy expressions); run directly for Hermiticity and solver-vs-dense checks |
| `jax_from_text.py` | Independent JAX program: `small` (Table 14.1), `caseA`, `caseB`, `stage1` (empty lattice, cosine potential, parities), `jellium` |
| `checks.py` | Formula and constant checks: PW92, exchange integral, smearing identity, HGH transforms, alpha, L.S, crystal-field model, Ewald |
| `atom.py` | Radial solver: bare-ion levels and the self-consistent LDA pseudo-atom |
| `screened.py` | Screened-ion test: `radial` (oracle) and `pw 24.0 40.0 2.0` (plane-wave side) |
| `symtest.py` | [P,H]=0, [Theta,H]=0, Theta^2=-1 at the TRIM |
| `solver_compare.py` | Shows the failure of the two simpler LOBPCG variants |
| `solver_stress_test.py` | Stress test of the final solver: `python3 solver_stress_test.py 20` |
| `run_main.py` | Gold SCF, parities at the 8 TRIM, level tables, band path: `python3 run_main.py 40 8 LDA lda40` |
| `gapmin.py` | Search for the minimum direct gap between bands 12 and 13 |
| `wilson.py` | Wilson loops: `python3 wilson.py 20 24 80` (Ecut, N1, N2) |
| `ref.py` | Small reference cases (Ecut 20, 4x4x4) and mixing/smearing comparisons |
| `mix.py` | Mixing comparison (linear, Kerker, Kerker-Pulay) |
| `optics.py` | Velocity matrix elements (`test`: Hellmann-Feynman check) and bands + velocities on the irreducible wedge of a dense mesh: `python3 optics.py run 40 30` |
| `optics_post.py` | Plasma frequency, dielectric function, reflectivity, CIE colour, sum-rule and Kramers-Kronig checks: `python3 optics_post.py optics_40_30_soc.npz` |
| `optics_report.py` | Writes `../figdata/optics_spec.dat` and `../figdata/optics_numbers.tex` (every number of Chapters 12-13): `python3 optics_report.py optics_40_30_soc.npz optics_24_40_soc.npz` |
| `ev.py` | Equation-of-state scans (Birch-Murnaghan fits): `main` (LDA and PBE, Ecut 45, 8x8x8), `soc` (LDA with and without SOC, Ecut 35, 6x6x6), `cut` (not run) |
| `atom_q19.py` | LDA and PBE pseudo-atoms with the 11- and the 19-electron parameter sets (three projectors, radial GGA): `python3 atom_q19.py 0.02` |

## Eigensolvers in pw.py

- `lobpcg`: the three-line textbook version, kept to demonstrate that it diverges.
- `lobpcg_safe`: a first repair attempt; also fails (at X and W after ~30 iterations).
- `lobpcg_orth`: the version in the document (Algorithms 3 and 4). Use this one.
  Note the *two* passes of "project P against X, orthonormalize P" at the end of
  each iteration. With a single pass the solver passes every test on gold but
  loses orthogonality and diverges, some tens of iterations after convergence,
  on highly degenerate problems (cosine potential and empty lattice of Stage 1).
  This was discovered with `jax_from_text.py stage1`.

## Logs (`logs/`)

All logs were regenerated in November 2026 with the scripts as distributed.

| Log | Produced by | Quoted in |
|---|---|---|
| `checks.log` | `checks.py` | Ch. 4, 6, 7, 8 anchors; Stage 0 |
| `ref.log` | `ref.py` | Table 14.1; mixing comparison of Ch. 9 |
| `jax_small.log`, `jax_caseA_jellium.log`, `jax_stage1.log` | `jax_from_text.py` | Table 14.1, case A, Stages 1-2 |
| `mesh_symmetry.log` | `jax_from_text.py` (interactive) | Ch. 5, symmetry breaking of the shifted mesh |
| `atom.log`, `screened_radial.log`, `screened_pw.log` | `atom.py`, `screened.py` | Stage 1, items 3-4 |
| `symtest.log` | `symtest.py` | Stage 0, symmetry |
| `solver_stress.log` | `solver_stress_test.py 20` | Ch. 9, numerical stability |
| `main_lda40.log` | `run_main.py 40 8 LDA lda40` | case B, Table 14.2, parity table of Ch. 11, SCF history |
| `wilson_fine.log`, `wilson_coarse.log` | `wilson.py 20 24 80`, `wilson.py 20 24 20` | Ch. 11 (the coarse run gives the wrong count on one plane, as discussed there) |
| `gapmin.log` | `gapmin.py` | Ch. 11, direct gap |
| `optics_test.log`, `optics_16.log`, `optics_24.log`, `optics_40.log`, `optics_16_nosoc.log` | `optics.py` | Ch. 12 (velocity test), data for Ch. 12-13 |
| `ev_main.log`, `ev_soc.log` | `ev.py main`, `ev.py soc` | Ch. 14, lattice constants and bulk moduli |
| `atom_q19.log` | `atom_q19.py 0.02` | Appendix A, verification of the 19-electron tables |
| `colour_measured.log` | `optics_post.py` functions on `reference_data/*_Johnson_Christy_1972.yml` | Ch. 13, colours from measured reflectivities of Au, Ag, Cu |
| `optics_post.log` | `optics_post.py optics_40_30_soc.npz` | Ch. 12-13 numbers and checks |

`wilson_soc_N2_20.npz` holds the phases of the coarse (N2 = 20) Wilson run, the last one executed;
the figure data in `../figdata/wcc_*.dat` are from the fine (N2 = 80) run.

`reference_data/GTH_SOC_POTENTIALS` is the CP2K database file the gold
parameter tables were checked against (byte-identical to the file in the
CP2K repository, November 2026). `reference_data/*_Johnson_Christy_1972.yml`
are measured optical constants of Au, Ag and Cu from the public-domain
refractiveindex.info database.
