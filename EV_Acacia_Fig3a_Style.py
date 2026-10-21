# =============================================================================
# EV NANO ACACIA  |  CaF2 vs AgSiNW  |  Figure 3(a)-style stacked Raman plot
#
# STYLE REFERENCE
#   Roda F., Picciolini S., Mangolini V., Gualerzi A., et al.
#   "Raman Spectroscopy Characterization of Multi-Functionalized Liposomes as
#   Drug-Delivery Systems for Neurological Disorders."
#   Nanomaterials 2023, 13, 699. doi:10.3390/nano13040699  (Figure 3a)
#
#   Fig. 3a in that paper = vertically stacked spectra (no y ticks),
#   a subtraction spectrum (sample - reference) with a zero line,
#   grey vertical bands with rotated "xxx cm-1" labels on the peaks of interest,
#   broken x-axis between the two acquired ranges (//), labels to the right.
#
# PREPROCESSING (Section 2.4 of the paper)
#   1. Polynomial baseline per spectral range (iterative modified polyfit)
#   2. Polynomial (Savitzky-Golay) smoothing
#   3. Unit vector normalisation (both ranges together)
#   4. Subtraction spectrum = AgSiNW - CaF2
#
# ONLY the two measured .txt files are used. Peak positions printed on the
# figure are read from the processed AgSiNW spectrum (local maximum near each
# marked position), not typed in by hand.
#
# HOW TO RUN:
#   pip install numpy scipy matplotlib
#   python EV_Acacia_Fig3a_Style.py  <CaF2.txt>  <AgSiNW.txt>
# =============================================================================

import sys
import csv

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter

# ----------------------------------------------------------------------------
# INPUT / OUTPUT
# ----------------------------------------------------------------------------
FILE_CAF2 = (sys.argv[1] if len(sys.argv) > 1 else
             's1-EV_nano_Acacia-dropCaF2_532nm_1800gr_10_50xLF_BC200_1s_4a.txt')
FILE_AG = (sys.argv[2] if len(sys.argv) > 2 else
           's4a-EV_nano_ACACIA-20ng-AgSiNW_532nm_1800gr_BC200_50XLF_1s_4a_10.txt')
OUT_PNG = 'R13_EV_Acacia_Fig3a_Style.png'
OUT_CSV = 'R13_EV_Acacia_PeakAssignments.csv'

# ----------------------------------------------------------------------------
# SETTINGS
# ----------------------------------------------------------------------------
# Paper plots 500-1800 | 2700-3200. Set LOW_MIN = 500 for the identical range;
# 200 keeps the 228 cm-1 band you marked.
LOW_MIN, LOW_MAX = 200, 1800
HIGH_MIN, HIGH_MAX = 2700, 3200
BASELINE_DEG = {'low': 7, 'high': 3}
SG_WINDOW, SG_POLY = 11, 3
SEARCH_HALF = 8        # cm-1 window to locate the measured maximum
BAND_HALF = 7          # half-width of the grey band (cm-1)

# Peaks marked on the manual plot + assignment.
# Sources:
#   [P]  Roda et al., Nanomaterials 2023, 13, 699 (the style paper)
#   [M]  Movasaghi Z., Rehman S., Rehman I.U. Appl. Spectrosc. Rev. 2007, 42, 493-541
#   [T]  Talari A.C.S. et al. Appl. Spectrosc. Rev. 2015, 50, 46-111
#   [C]  Czamara K. et al. J. Raman Spectrosc. 2015, 46, 4-20 (ref. 31 of [P])
#   [G]  Gualerzi A. et al. Sci. Rep. 2017, 7, 9820 (Raman of EVs)
PEAKS = [
    # (marked, class shown on plot, vibrational assignment, source)
    (228,  'Ag substrate',   'Ag-adsorbate (Ag-N / Ag-O / Ag-Cl) low-frequency mode; substrate, not EV', '[T]'),
    (519,  'Si substrate',   'Crystalline silicon optical phonon (520.7 cm-1 Si reference band); from SiNW', '[P]'),
    (612,  'Protein',        'C-C twisting of phenylalanine (~621); weak Si 2nd-order phonon may overlap', '[M][T]'),
    (795,  'Nucleic acid',   'O-P-O backbone / phosphodiester stretching (785-810)', '[M][T]'),
    (827,  'Protein/NA',     'Tyrosine Fermi doublet (828/850) / O-P-O asymmetric stretch (828)', '[M][T]'),
    (921,  'Protein/Carb.',  'C-C stretch of proline ring / glucose (920)', '[M][T]'),
    (1001, 'Protein',        'Phenylalanine symmetric ring breathing (1002-1004)', '[M][T][G]'),
    (1031, 'Protein',        'Phenylalanine C-H in-plane bending (1030-1031)', '[M][T]'),
    (1240, 'Protein',        'Amide III (1230-1250); asym. PO2- of nucleic acids may overlap', '[M][T][G]'),
    (1386, 'Lipid/Protein',  'CH3 symmetric deformation (~1380) / COO- symmetric stretch', '[M][T]'),
    (1606, 'Protein',        'Phenylalanine / tyrosine ring C=C stretch (1605-1607)', '[M][T][G]'),
    (1625, 'Protein',        'Amide I low-frequency side / Trp-Tyr C=C (1617-1630)', '[M][T]'),
    (2936, 'Protein/Lipid',  'CH3 symmetric stretching (2930-2940), mainly proteins', '[P][M][C][G]'),
]

C_AG = '#E8383D'     # red  : EV on AgSiNW (as in your manual plot)
C_CAF2 = '#3A3A3A'   # grey : EV on CaF2
C_SUB = '#7F7F7F'    # grey : subtraction (as in paper)
BAND = '#D0D0D0'


# ----------------------------------------------------------------------------
# PROCESSING
# ----------------------------------------------------------------------------
def modpoly_baseline(x, y, deg, n_iter=200):
    """Iterative modified polynomial fit (Lieber & Mahadevan-Jansen 2003)."""
    t = 2 * (x - x.mean()) / np.ptp(x)      # rescale x for a stable fit
    work = y.copy()
    for _ in range(n_iter):
        base = np.polyval(np.polyfit(t, work, deg), t)
        work = np.minimum(work, base)
    return base


def process(path):
    data = np.loadtxt(path)
    x, y = data[:, 0], data[:, 1]
    xs, ys = [], []
    for key, lo, hi in [('low', LOW_MIN, LOW_MAX), ('high', HIGH_MIN, HIGH_MAX)]:
        m = (x >= lo) & (x <= hi)
        xr, yr = x[m], y[m]
        yr = savgol_filter(yr, SG_WINDOW, SG_POLY)
        yr = yr - modpoly_baseline(xr, yr, BASELINE_DEG[key])
        xs.append(xr)
        ys.append(yr)
    # unit vector normalisation over both ranges together
    norm = np.linalg.norm(np.concatenate(ys))
    return [(xr, yr / norm) for xr, yr in zip(xs, ys)]


caf2 = process(FILE_CAF2)
ag = process(FILE_AG)

# subtraction on the AgSiNW wavenumber grid (grids differ slightly)
sub = [(xa, ya - np.interp(xa, xc, yc)) for (xa, ya), (xc, yc) in zip(ag, caf2)]

# measured peak positions from the processed AgSiNW spectrum
ag_x = np.concatenate([r[0] for r in ag])
ag_y = np.concatenate([r[1] for r in ag])
rows = []
for marked, cls, assign, ref in PEAKS:
    m = np.abs(ag_x - marked) <= SEARCH_HALF
    if not m.any():
        continue
    measured = ag_x[m][np.argmax(ag_y[m])]
    rows.append((marked, round(float(measured)), cls, assign, ref))

with open(OUT_CSV, 'w', newline='') as fh:
    w = csv.writer(fh)
    w.writerow(['Marked (cm-1)', 'Measured AgSiNW (cm-1)', 'Class', 'Assignment', 'Source'])
    w.writerows(rows)

# ----------------------------------------------------------------------------
# PLOT  (broken x-axis, stacked traces, no y ticks)
# ----------------------------------------------------------------------------
plt.rcParams.update({'font.family': 'sans-serif',
                     'font.sans-serif': ['Arial', 'Liberation Sans', 'DejaVu Sans'],
                     'font.size': 10})
fig, (axL, axR) = plt.subplots(
    1, 2, sharey=True, figsize=(7.5, 7.5),
    gridspec_kw={'width_ratios': [LOW_MAX - LOW_MIN, (HIGH_MAX - HIGH_MIN) * 1.4],
                 'wspace': 0.04})

span = max(np.ptp(r[1]) for r in ag + sub)
lab_top = 0.95 * span                # labels hang below this level
off_sub = 1.45 * span
off_ag = 2.55 * span
off_caf2 = 3.55 * span
traces = [(caf2, off_caf2, C_CAF2, 'EV Nano Acacia\nCaF$_2$'),
          (ag, off_ag, C_AG, 'EV Nano Acacia\nAgSiNW'),
          (sub, off_sub, C_SUB, 'Subtraction\nspectrum')]


def spread(pos, min_gap):
    """Shift label x-positions apart so close peaks do not overlap."""
    out = list(pos)
    for _ in range(50):
        for k in range(1, len(out)):
            if out[k] - out[k - 1] < min_gap:
                mid = (out[k] + out[k - 1]) / 2
                out[k - 1], out[k] = mid - min_gap / 2, mid + min_gap / 2
    return out


for ax, idx, lo, hi in [(axL, 0, LOW_MIN, LOW_MAX), (axR, 1, HIGH_MIN, HIGH_MAX)]:
    for spec, off, col, _ in traces:
        xr, yr = spec[idx]
        ax.plot(xr, yr + off, color=col, lw=0.8)
    ax.axhline(off_sub, color='k', lw=0.6)

    here = [r for r in rows if lo <= r[1] <= hi]
    gap = 0.024 * (LOW_MAX - LOW_MIN)          # same physical spacing on both axes
    lab_x = spread([r[1] for r in here], gap)
    for (marked, measured, cls, _, _), lx in zip(here, lab_x):
        ax.axvspan(measured - BAND_HALF, measured + BAND_HALF,
                   color=BAND, alpha=0.8, lw=0, zorder=0)
        ax.plot([measured, measured, lx], [lab_top + 0.25 * span, lab_top + 0.10 * span,
                                           lab_top], color='0.4', lw=0.5)
        ax.text(lx, lab_top - 0.03 * span, f'{measured} cm$^{{-1}}$  {cls}',
                rotation=90, ha='center', va='top', fontsize=7.5)

axL.set_xlim(LOW_MIN, LOW_MAX)
axR.set_xlim(HIGH_MIN, HIGH_MAX)
axL.set_ylim(-0.45 * span, off_caf2 + 0.9 * span)
axL.set_yticks([])
for s_ in ['top', 'right']:
    axL.spines[s_].set_visible(False)
for s_ in ['top', 'right', 'left']:
    axR.spines[s_].set_visible(False)
axR.tick_params(left=False)
axL.set_xticks([t for t in [200, 500, 1000, 1500] if t >= LOW_MIN])
axR.set_xticks([3000])

# trace labels on the right (as in the paper)
for spec, off, col, label in traces:
    axR.text(HIGH_MAX + 20, off + 0.1 * span, label, fontsize=9,
             color='k', va='center', ha='left', clip_on=False)

# // break marks on the x-axis
d = 0.012
kw = dict(color='k', clip_on=False, lw=0.8)
for ax, x0 in [(axL, 1), (axR, 0)]:
    ax.plot((x0 - d, x0 + d), (-d * 2, d * 2), transform=ax.transAxes, **kw)

axL.set_ylabel('Intensity (a.u.)', fontsize=11)
fig.text(0.47, 0.03, 'Raman shift (cm$^{-1}$)', ha='center', fontsize=11)
fig.subplots_adjust(left=0.07, right=0.80, bottom=0.09, top=0.98)
fig.savefig(OUT_PNG, dpi=600)
print(f'saved {OUT_PNG} and {OUT_CSV}')
for r in rows:
    print(r)
