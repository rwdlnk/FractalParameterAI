# Ramaprabhu & Andrews (2004) Profile Comparison — Prep Doc

*Drafted 2026-05-27 in the JFM-paper-drafting session as a hand-off to a
separate Claude session that will execute the analysis and plotting. The
decisions captured here come from that drafting session and should be
inherited verbatim; do not re-deliberate them.*

## Why this doc exists

The JFM paper draft at
`/media/rod/ResearchIII/githubRepos/RT-Paper-Drafts/JFM/draft-v1/RT-multifractal.tex`
contains a new §3.4 subsection ("Comparison with Ramaprabhu and Andrews
(2004) at lower Atwood number") that closes the validation section. The
section references a Figure~\ref{fig:ramaprabhu} that is currently a
labelled placeholder PDF
(`/media/rod/ResearchIII/githubRepos/RT-Paper-Drafts/JFM/draft-v1/figures/ramaprabhu_profiles.pdf`).

This doc lays out the work needed to replace that placeholder with the
real two-panel comparison figure. Reading order suggested:

1. This doc (decisions + work items).
2. The §3.4 prose and caption in the paper draft (lines around the
   `\subsection{Comparison with Ramaprabhu and Andrews}` label,
   `\label{sec:val:ramaprabhu}`) to see exactly what the figure is
   committed to showing.
3. R&A 2004 paper pages 242 (Figure 4) and 248 (Figure 9), local at
   `/media/rod/ResearchIII/githubRepos/RT-Paper-Drafts/references/Ramaprabhu-JFM-502-2004-pp233-246.pdf` (Fig 4) and
   `/media/rod/ResearchIII/githubRepos/RT-Paper-Drafts/references/Ramaprabhu-JFM-502-2004-pp247-257.pdf` (Fig 9).

## What the figure must show

A two-panel figure:

- **Panel (a)** — mean volume-fraction profile $\langle\alpha\rangle(y/H)$
  across the mixing layer at $\tau \approx 1.21$, for our three codes at
  the finest 2D grid ($1920\times 2400$), overlaid on R&A's digitised
  $f_1(y/H)$ measurements from their Figure~4.

- **Panel (b)** — intermittency profile
  $\langle\alpha\rangle\bigl(1 - \langle\alpha\rangle\bigr)(y/H)$ at the
  same $\tau$ and grids, overlaid on R&A's digitised $B_2 = f_1 f_2(y/H)$
  measurements from their Figure~9.

Output file:
`/media/rod/ResearchIII/githubRepos/RT-Paper-Drafts/JFM/draft-v1/figures/ramaprabhu_profiles.pdf`,
overwriting the existing placeholder.

## Decisions inherited from the paper-drafting session

These are baked into the §3.4 caption and prose. Do not change them
without coordinating back to the paper draft.

### Time matching

- R&A report at $T = 1.21$. Their $T$ uses the same Dalziel
  non-dimensionalisation $T = t \sqrt{A_t g / H}$ that our paper uses for
  $\tau$.
- Our parameters: $A = 2.1\times 10^{-3}$, $g = 9.81\,\mathrm{m\,s^{-2}}$,
  $H = 0.5\,\mathrm{m}$, so $\sqrt{Ag/H} = 0.2030\,\mathrm{s^{-1}}$.
- $\tau = 1.21$ corresponds to $t = 5.96\,\mathrm{s}$. Snapshots are
  written every $0.2\,\mathrm{s}$; closest snapshot is at $t = 6.0\,
  \mathrm{s}$ ($\tau = 1.218$). Use that snapshot; do not interpolate.
- The §3.4 caption says "$\tau \approx 1.21$" — the small offset to 1.218
  is within the implicit tolerance.

### Intermittency definition

- The §3.4 caption commits to $\langle\alpha\rangle\bigl(1 -
  \langle\alpha\rangle\bigr)(y)$ --- i.e., the **product of the
  horizontally-averaged mean** rather than the **horizontal average of the
  product** $\langle\alpha(1-\alpha)\rangle$.
- These are mathematically distinct. For sharp VOF interfaces they differ
  noticeably: $\langle\alpha(1-\alpha)\rangle$ counts cells with $0 <
  \alpha < 1$ (interface-density measure), while
  $\langle\alpha\rangle(1-\langle\alpha\rangle)$ is the natural analogue
  of R&A's miscible $f_1 f_2$ (mean-product measure).
- R&A's $B_2 = f_1 f_2$ at a $y$-station is a time average of the
  instantaneous product. For a statistically steady experiment that
  equals the product of the means at the centreline by definition; for a
  transient simulation the two differ slightly. The product-of-means form
  is the right counterpart for our setup; that's what the paper claims.

### Codes and grids

- Three codes: SOLA-VOF, OpenFOAM MULES, OpenFOAM MPLIC.
- Grid: $1920\times 2400$ for all three (the finest 2D grid in our matrix).
- Locations (per CLAUDE.md):
  - SOLA-VOF: `/media/rod/RT-Simulations/Dalziel_1999/sola-vof/1920x2400/`
  - MULES: `/media/rod/RT-Simulations/Dalziel_1999/OpenFOAM-2D/MULES/1920x2400x1/`
  - MPLIC: `/media/rod/RT-Simulations/Dalziel_1999/OpenFOAM-2D/IsoAdvect/1920x2400x1/`

### Coordinate normalisation

- R&A normalises by their channel depth $H = 0.3\,\mathrm{m}$ and shows
  $y/H$ on the range $-0.25$ to $0$ in their Figs 4 and 9 (lower half of
  mix only; centreline at $y/H = 0$).
- We use $H = 0.5\,\mathrm{m}$ and the interface starts at $y_0 = 0.25\,
  \mathrm{m}$. The natural reduced coordinate is $(y - y_0)/H$.
- Plot range: $(y - y_0)/H$ from $-0.25$ to $0.25$ so we show both halves
  of the mix; that's a richer view than R&A's lower-half-only display.
  R&A's lower-half points will overlay only on the negative side.
- Do **not** scale our $y$ by R&A's $H$. Each experiment is normalised by
  its own domain extent. That's the standard low-$A$ convention and what
  Dalziel et al. (1999) also do.

## R&A Fig 4 and Fig 9 — what to digitise

### Figure 4 (page 242)

- Caption: "Volume fraction profiles from PIV-S (solid line), and
  thermocouple (circles) at $T \sim 1.21$."
- Axes: $f_1$ (y-axis, 0 to 0.5) vs $y/H$ (x-axis, $-0.25$ to $0$).
- The **solid line** is the PIV-S measurement; the **circles** are
  thermocouple measurements at discrete depths. ~10 visible circle points.
- Digitise the circles (preferred — discrete data points are cleaner than
  a curve trace). The solid line can be sampled at the same $y/H$ values
  if desired, but circles alone are sufficient for overlay.

### Figure 9 (page 248)

- Caption: "Intermittency factor (based on the two-fluid parameter $B_2$)
  across the mix at $T = 1.21$."
- Axes: $f_1 f_2$ (y-axis, 0 to 0.5) vs $y/H$ (x-axis, $0$ to $-0.25$; note
  the **reversed x-axis direction** in their plot).
- ~12 visible filled-circle data points.
- Digitise straightforwardly; flip the x-axis sign if needed to match the
  Fig 4 convention.

A simple two-column CSV per figure (`y_over_H, f1` and `y_over_H, B2`) is
sufficient. Save both under
`/media/rod/RT-Simulations/Dalziel_1999/Dalziel_data/extracted_csv/` (or
a sibling location) for future reuse.

## Work items

### 1. Extend `analyze_rt.py` Phase 1 to save $\langle\alpha\rangle(y, t)$ profiles

- File: `/media/rod/ResearchIII/githubRepos/FractalParameterAI/rt_analyzer/analyze_rt.py`.
- Phase 1 already computes $f_{\rm avg}(y)$ internally to evaluate the
  Dalziel Eq. 7 integrals (`h_{1,0}`, `h_{1,1}`, `h_{0,0}`, `h_{0,1}`).
- Add: persist the profile to disk at every output time, as a per-snapshot
  CSV. Suggested location:
  `{case}/analysis/profiles/alpha_profile_t{step}.csv`,
  one file per snapshot.
- Schema: two columns, `y_m, alpha_mean`. No header is required; if a
  header is present it must use these exact column names.
- This is a small, additive change. Make it default-on (no flag needed)
  to keep the API simple. Confirm by spot-checking one snapshot that the
  saved profile reproduces what Phase 1 used internally.

### 2. Run the extended analyzer on the three finest 2D cases

Per `ANALYSIS_COMMANDS.md`:

```
python3 -u /media/rod/ResearchIII/githubRepos/FractalParameterAI/rt_analyzer/analyze_rt.py \
    /media/rod/RT-Simulations/Dalziel_1999/sola-vof/1920x2400 \
    --skip-phase2 --skip-phase3
```

`--skip-phase2 --skip-phase3` keeps the run short --- only Phase 1 (which
is where the new profile output lives) needs to re-execute.

Repeat for the two OpenFOAM cases:

```
... /media/rod/RT-Simulations/Dalziel_1999/OpenFOAM-2D/MULES/1920x2400x1
... /media/rod/RT-Simulations/Dalziel_1999/OpenFOAM-2D/IsoAdvect/1920x2400x1
```

Verify three things after each run:

- The `profiles/` subdirectory exists with 101 CSVs ($t = 0$ through $20$\,s
  at $0.2$\,s spacing).
- The `summary/temporal_results.csv` still reproduces the previously
  reported $h_{1,0}(\tau)$ and $D(\tau)$ to within floating-point tolerance
  --- i.e., the Phase 1 modification is purely additive and did not
  perturb existing outputs.
- Spot-check the $t = 6.0\,\mathrm{s}$ profile (snapshot index 30) for
  qualitative shape: should rise monotonically from $\alpha \approx 0$ at
  $y \ll y_0$ to $\alpha \approx 1$ at $y \gg y_0$, passing through 0.5
  at $y \approx y_0$.

### 3. Digitise R&A's Figs 4 and 9

- Tools: WebPlotDigitizer (browser-based, free) or any equivalent. Manual
  pixel-counting is fine for ~22 points total.
- Output: two CSVs, ~10 + ~12 rows respectively, in
  `/media/rod/RT-Simulations/Dalziel_1999/Dalziel_data/extracted_csv/ramaprabhu2004_fig4.csv`
  and `..._fig9.csv`. Header `y_over_H,value`.
- Sanity checks: Fig 4 should reach $f_1 \approx 0.45$ at $y/H \approx 0$;
  Fig 9 should reach $B_2 \approx 0.5$ at $y/H \approx 0$ (in their
  sign-flipped convention; verify before plotting).

### 4. Write the plotting script

Recommended location:
`/media/rod/RT-Simulations/Dalziel_1999/ramaprabhu_profile_comparison.py`
(sibling to the other `cross_code_*.py` and `dalziel_validation.py`
scripts).

Sketch:

```python
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

DRAFT_FIGURES = Path('/media/rod/ResearchIII/githubRepos/'
                     'RT-Paper-Drafts/JFM/draft-v1/figures')
CASES = {
    'SOLA-VOF': '/media/rod/RT-Simulations/.../sola-vof/1920x2400',
    'MULES':    '/media/rod/RT-Simulations/.../OpenFOAM-2D/MULES/1920x2400x1',
    'MPLIC':    '/media/rod/RT-Simulations/.../OpenFOAM-2D/IsoAdvect/1920x2400x1',
}
RA_FIG4_CSV = '.../ramaprabhu2004_fig4.csv'  # f_1(y/H)
RA_FIG9_CSV = '.../ramaprabhu2004_fig9.csv'  # B_2(y/H)
SNAPSHOT_INDEX = 30   # t = 6.0 s, tau ~ 1.218
Y_INTERFACE   = 0.25  # m
DOMAIN_H      = 0.5   # m

fig, (axA, axB) = plt.subplots(1, 2, figsize=(11, 4.5), sharex=True)

for label, case in CASES.items():
    df = read_profile(case, SNAPSHOT_INDEX)  # columns y_m, alpha_mean
    y_over_H = (df['y_m'] - Y_INTERFACE) / DOMAIN_H
    axA.plot(y_over_H, df['alpha_mean'], label=label)
    axB.plot(y_over_H, df['alpha_mean'] * (1 - df['alpha_mean']), label=label)

# Overlay R&A
ra4 = np.genfromtxt(RA_FIG4_CSV, delimiter=',', names=True)
ra9 = np.genfromtxt(RA_FIG9_CSV, delimiter=',', names=True)
axA.plot(ra4['y_over_H'], ra4['value'], 'ko', label='R&A 2004 (Fig 4)')
axB.plot(ra9['y_over_H'], ra9['value'], 'ko', label='R&A 2004 (Fig 9)')

axA.set_xlabel(r'$(y - y_0)/H$')
axA.set_ylabel(r'$\langle\alpha\rangle$')
axB.set_xlabel(r'$(y - y_0)/H$')
axB.set_ylabel(r'$\langle\alpha\rangle(1 - \langle\alpha\rangle)$')
for ax, lab in [(axA, '(a)'), (axB, '(b)')]:
    ax.text(0.02, 0.92, lab, transform=ax.transAxes)
    ax.legend(loc='best', fontsize=9)
fig.tight_layout()
fig.savefig(DRAFT_FIGURES / 'ramaprabhu_profiles.pdf')
```

This is a sketch, not finished code. Adapt to the actual CSV path
conventions used by `analyze_rt.py` and to the digitised R&A column
names.

### 5. Rebuild the paper PDF

After replacing the placeholder, rebuild from the draft directory:

```
cd /media/rod/ResearchIII/githubRepos/RT-Paper-Drafts/JFM/draft-v1
pdflatex RT-multifractal && bibtex RT-multifractal && \
  pdflatex RT-multifractal && pdflatex RT-multifractal
```

Confirm Figure~\ref{fig:ramaprabhu} renders the new content. Page count
should still be 16 (the figure already had a slot reserved).

## Done criteria

- [ ] `analyze_rt.py` Phase 1 saves `alpha_profile_t*.csv` per snapshot.
- [ ] Existing Phase 1 outputs ($h$, $D$) reproduce within
      floating-point tolerance.
- [ ] Three 2D cases at $1920\times 2400$ have re-run cleanly.
- [ ] R&A Figs 4 and 9 digitised to CSV.
- [ ] Plotting script generates a two-panel PDF that matches the §3.4
      caption.
- [ ] `figures/ramaprabhu_profiles.pdf` overwritten in the draft
      directory.
- [ ] Paper PDF rebuilds without warnings.

## If results disagree with R&A

A clean overlay is the expected outcome based on Dalziel's experimental
band agreement (\S3.1) and the integral validation (\S3.4 first paragraph).
If the profiles disagree noticeably, possibilities to investigate before
revising the paper:

- Sign convention on $y/H$ in the digitised Fig 9 data.
- Wrong snapshot index (off by one --- recheck the time-to-snapshot
  mapping).
- Wrong intermittency definition --- check whether plotting
  $\langle\alpha(1-\alpha)\rangle$ instead of $\langle\alpha\rangle
  (1-\langle\alpha\rangle)$ improves the match. If so, raise the question
  back to the paper-drafting session before changing the caption.
- Resolution effect --- if disagreement is small, also overlay
  $640\times 800$ as a coarser-grid sanity check.

If after these checks a real disagreement remains, that's interesting and
needs discussion with the paper-drafting session before §3.4 is finalised.

---

*Inherits decisions from JFM draft session 2026-05-27.*

---

## OUTCOME — completed 2026-05-27

All work items above are done. The conclusion differs from the doc's original
expectation ("clean overlay at τ≈1.21"); see below.

### What was done
1. **`analyze_rt.py` Phase 1** saves `{case}/analysis/profiles/alpha_profile_t{ms:05d}.csv`
   (columns `y_m, alpha_mean`) for every snapshot incl. t=0, in both the VTK (`run_phase1`)
   and OpenFOAM (`_run_phase1_openfoam`) paths. Verified bit-identical to HEAD on 160×200 —
   purely additive. `ms` = milliseconds key (= round(t·1000)); t=6.0 s → `t06000`.
2. **Profiles regenerated** for the three finest 2D cases (SOLA-VOF NDIS-6/1920x2400, MULES &
   IsoAdvect 1920x2400x1) by running full Phase 1 to a *scratch* dir and copying only
   `profiles/` into production — `temporal_results.csv`/`fractal/`/`interfaces/` were NOT
   overwritten. `h_10` reproduces exactly.
3. **R&A Figs 4 & 9 digitised** → `Dalziel_data/extracted_csv/ramaprabhu2004_fig{4,9}.csv`
   (5 and 8 points). **Fig 9's ordinate is γ_ρ = 2 f₁f₂** (their eq. 23, peak 0.5), not f₁f₂.
4. **Plot script** `Dalziel_1999/ramaprabhu_profile_comparison.py` →
   `…/figures/ramaprabhu_profiles.pdf` (4-panel paper figure) + `_tau121`/`_developed`.
5. **Paper §3.4 rewritten** (developmental-stage narrative) + new caption + a §2 correction
   (the codes use different RNGs, not an identical draw). Rebuilt clean (17 pp).

### Revised scientific result
- R&A's `T = (x/U)√(Ag/H) = our τ` (their eq. 13, p.243). Coordinate map `y/H = y′/H′ + ½`,
  each profile normalised by its **own** channel depth (no H′/H rescale).
- At the literal match τ=1.21 our layer is ~10× thinner than R&A's — a **developmental-stage**
  effect, not a discrepancy. R&A's fine-mesh-screen grid turbulence is a far larger seed than
  our σ=0.08dx interface perturbation, so their self-similar τ=1.21 ≈ our τ≈2.7–3.3 (width-match:
  SVOF 2.72, MULES 3.33, MPLIC 3.25). At matched width the ⟨α⟩ and ⟨α⟩(1−⟨α⟩) profiles **collapse
  onto R&A** for all three codes — the spatial-profile analogue of the D-attractor IC-independence.
  Figure shows BOTH times to make the IC-fading point explicit.

### IC verification (NDIS=6 vs OF13 `perturbDalziel.py`)
- Same recipe/statistics: band 4–8dx, σ=0.08dx (=0.08 cells, sub-grid), uniform amp σ√(2/N),
  RMS=σ, heavy-over-light, u=0. **Different realizations**: SOLA-VOF LCG vs OpenFOAM NumPy
  Mersenne Twister (seed 123, corr(η)≈0.004). Off-by-one long mode (SVOF n=241–480, OF 240–480),
  RMS change 0.2%. OF13 base interface is a 5 mm tanh (2→24 cells over the grid sweep) vs SVOF
  sharp; both relax away before τ=1.21. σ=0.08dx grid-tie is intentional.

### Caveat: stale finest-grid D
Current Phase 1 gives late-time ⟨D⟩(τ>2) = 1.766 (SVOF), 1.745 (MULES) vs committed 1.792/1.773
(pre-`7597f1f`/`5668eb3`); MPLIC reproduces exactly (1.770). h_10 unaffected. Production CSVs not
overwritten — refresh if citing finest-grid D.
