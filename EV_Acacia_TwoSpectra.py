# =============================================================================
# ACACIA EVs or NVs  |  CaF2 vs Ag/SiNWs  |  two raw spectra with peak assignments
#
# Both measured spectra on one axis (raw counts: no smoothing, no baseline, no
# normalisation), broken x-axis over the 1800-2700 cm-1 gap, dotted line from
# each labelled peak up to its label (wavenumber + biomolecular class). The
# sample names are written inside the plot, next to their spectra; the y-axis
# has no numbers.
#
# The sample (EV or NV) is read from the Ag/SiNW file name ('nano' = NV).
#   EV: S2-EV_Acacia-dropCaF2 ...txt          + S2-EV_ACACIA-AgSiNW ...txt
#   NV: s1-EV_nano_Acacia-dropCaF2 ...txt     + s4a-EV_nano_ACACIA-20ng-AgSiNW ...txt
#
# Peak position = most prominent local maximum of the Ag/SiNW spectrum within
# +/-12 cm-1 of the nominal band, found on a lightly smoothed copy (Savitzky-Golay,
# 11 pts, 3rd order; used ONLY to locate the maximum). Same method as
# EV_NV_Spectra_Comparison.py, so both figures give the same numbers. The CSV
# also lists each peak's prominence relative to the noise of the spectrum.
#
# Assignment sources:
#   [P]  Roda F., Picciolini S., ... Gualerzi A., Bedoni M. Nanomaterials 2023, 13, 699
#   [M]  Movasaghi Z., Rehman S., Rehman I.U. Appl. Spectrosc. Rev. 2007, 42, 493-541
#   [T]  Talari A.C.S. et al. Appl. Spectrosc. Rev. 2015, 50, 46-111
#   [C]  Czamara K. et al. J. Raman Spectrosc. 2015, 46, 4-20
#   [G]  Gualerzi A. et al. Sci. Rep. 2017, 7, 9820
#
# HOW TO RUN (PyCharm):
#   pip install numpy scipy matplotlib
#   python EV_Acacia_TwoSpectra.py  <CaF2 file .txt>  <Ag/SiNW file .txt>
# OUTPUT: CaF2_AgSiNW_<EV|NV>.png / .tiff (600 dpi) + CaF2_AgSiNW_<EV|NV>_peaks.csv
# =============================================================================

import sys
import csv
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter, find_peaks

# ----------------------------------------------------------------------------
# INPUT / OUTPUT
# ----------------------------------------------------------------------------
FILE_CAF2 = Path(sys.argv[1] if len(sys.argv) > 1 else
                 'S2-EV_Acacia-dropCaF2_532nm_1800gr_10_50xLF_BC200_1s_4a.txt')
FILE_AG = Path(sys.argv[2] if len(sys.argv) > 2 else
               'S2-EV_ACACIA-AgSiNW_532nm_1800gr_BC200_50XLF_1s_4a_10%.txt')
TAG = 'NV' if 'nano' in FILE_AG.name.lower() else 'EV'
if ('nano' in FILE_CAF2.name.lower()) != (TAG == 'NV'):
    sys.exit('CaF2 and Ag/SiNW files are from different samples - check the file names')
OUT = f'CaF2_AgSiNW_{TAG}'

# ----------------------------------------------------------------------------
# SETTINGS
# ----------------------------------------------------------------------------
LOW_MIN, LOW_MAX = 400, 1800
HIGH_MIN, HIGH_MAX = 2700, 3100
SEARCH_HALF = 12         # cm-1 window to locate the measured maximum
SAMPLE_NAME = {'EV': 'EV Acacia', 'NV': 'Nanovesicles Acacia'}[TAG]
# counts added to the Ag/SiNW spectrum (display only) so the traces do not overlap
AG_OFFSET = {'EV': 110, 'NV': 0}[TAG]

# (nominal position, class, assignment, source)
PEAK_SETS = {
    'EV': [
        (519,  'Si substrate',  'Crystalline silicon optical phonon (520.7 cm-1 Si band); from SiNW', '[P]'),
        (831,  'Protein/NA',    'Tyrosine Fermi doublet (828/850) / O-P-O asymmetric stretch (828)', '[M][T]'),
        (1001, 'Protein',       'Phenylalanine symmetric ring breathing (1002-1004)', '[M][T][G]'),
        (1031, 'Protein',       'Phenylalanine C-H in-plane bending (1030-1034)', '[M][T]'),
        (1166, 'Protein',       'Tyrosine C-H in-plane bending (1170-1175)', '[M][T]'),
        (1355, 'Protein',       'Tryptophan / CH deformation (1340-1380); broad band', '[M][T]'),
        (1582, 'Protein',       'Phenylalanine C=C bending (1584)', '[M][T]'),
        (1604, 'Protein',       'Phenylalanine / tyrosine ring C=C stretch (1600-1607)', '[M][T][G]'),
        (1724, 'Lipid',         'C=O ester stretching of (phospho)lipids (1720-1745)', '[M][T][C]'),
        (2929, 'Protein/Lipid', 'CH2/CH3 stretching (2925-2935)', '[P][M][C][G]'),
    ],
    'NV': [
        (519,  'Si substrate',  'Crystalline silicon optical phonon (520.7 cm-1 Si band); from SiNW', '[P]'),
        (612,  'Protein',       'C-C twisting of phenylalanine (~621); weak Si 2nd-order phonon may overlap', '[M][T]'),
        (797,  'Nucleic acid',  'O-P-O backbone / phosphodiester stretching (785-810)', '[M][T]'),
        (826,  'Protein/NA',    'Tyrosine Fermi doublet (828/850) / O-P-O asymmetric stretch (828)', '[M][T]'),
        (921,  'Protein/Carb.', 'C-C stretch of proline ring / glucose (920)', '[M][T]'),
        (1001, 'Protein',       'Phenylalanine symmetric ring breathing (1002-1004)', '[M][T][G]'),
        (1031, 'Protein',       'Phenylalanine C-H in-plane bending (1030-1031)', '[M][T]'),
        (1240, 'Protein',       'Amide III (1230-1250); asym. PO2- of nucleic acids may overlap', '[M][T][G]'),
        (1386, 'Lipid/Protein', 'CH3 symmetric deformation (~1380) / COO- symmetric stretch', '[M][T]'),
        (1607, 'Protein',       'Phenylalanine / tyrosine ring C=C stretch (1605-1607)', '[M][T][G]'),
        (1624, 'Protein',       'Amide I low-frequency side / Trp-Tyr C=C (1617-1630)', '[M][T]'),
        (2936, 'Protein/Lipid', 'CH3 symmetric stretching (2930-2940), mainly proteins', '[P][M][C][G]'),
    ],
}
PEAKS = PEAK_SETS[TAG]

C_AG = '#E8383D'     # red  : on Ag/SiNWs
C_CAF2 = '#3A3A3A'   # dark grey : on CaF2
CLASS_COL = {        # label colour by class
    'Protein': '#1F6FB4', 'Protein/NA': '#6A3D9A', 'Nucleic acid': '#8E44AD',
    'Protein/Carb.': '#1F6FB4', 'Lipid/Protein': '#D35400',
    'Protein/Lipid': '#D35400', 'Lipid': '#C0392B', 'Si substrate': '#7F7F7F',
}

# ----------------------------------------------------------------------------
# DATA
# ----------------------------------------------------------------------------
caf2 = np.loadtxt(FILE_CAF2)
ag = np.loadtxt(FILE_AG)
ag_s = savgol_filter(ag[:, 1], 11, 3)          # only to locate the maxima
noise = np.std(ag[:, 1] - ag_s)
ag[:, 1] += AG_OFFSET                          # display only

rows = []
for nominal, cls, assign, ref in PEAKS:
    k = np.flatnonzero(np.abs(ag[:, 0] - nominal) <= SEARCH_HALF)
    pk, pr = find_peaks(ag_s[k], prominence=0)
    j = np.argmax(pr['prominences'])
    i = k[pk[j]]
    rows.append((nominal, round(float(ag[i, 0])), cls, assign, ref,
                 round(float(pr['prominences'][j] / noise), 1)))

with open(f'{OUT}_peaks.csv', 'w', newline='') as fh:
    w = csv.writer(fh)
    w.writerow(['Nominal (cm-1)', 'Measured Ag/SiNW (cm-1)', 'Class', 'Assignment', 'Source',
                'Prominence / noise SD'])
    w.writerows(rows)

# ----------------------------------------------------------------------------
# PLOT
# ----------------------------------------------------------------------------
# font sizes (pt) - large enough to stay readable when the figure is reduced to
# one journal column
FS_PEAK, FS_NAME, FS_AXIS, FS_TICK = 12, 15, 17, 15
plt.rcParams.update({'font.family': 'sans-serif',
                     'font.sans-serif': ['Arial', 'Liberation Sans', 'DejaVu Sans'],
                     'font.size': FS_TICK})
fig, (axL, axR) = plt.subplots(
    1, 2, sharey=True, figsize=(9, 6.5),
    gridspec_kw={'width_ratios': [LOW_MAX - LOW_MIN, HIGH_MAX - HIGH_MIN],
                 'wspace': 0.03})


def in_view(d):
    x = d[:, 0]
    return d[((x >= LOW_MIN) & (x <= LOW_MAX)) | ((x >= HIGH_MIN) & (x <= HIGH_MAX)), 1]


y_min = min(in_view(caf2).min(), in_view(ag).min())
y_max = max(in_view(caf2).max(), in_view(ag).max())
y_top = y_max + 0.70 * (y_max - y_min)       # head-room for the rotated labels
y_bot = y_min - 0.12 * (y_max - y_min)       # room for the CaF2 sample name
axL.set_ylim(y_bot, y_top)

for ax, lo, hi in [(axL, LOW_MIN, LOW_MAX), (axR, HIGH_MIN, HIGH_MAX)]:
    for d, col in [(caf2, C_CAF2), (ag, C_AG)]:
        m = (d[:, 0] >= lo) & (d[:, 0] <= hi)
        ax.plot(d[m, 0], d[m, 1], color=col, lw=0.8)
    ax.set_xlim(lo, hi)
    pos = [r[1] for r in rows]
    for k, (nominal, measured, cls, _, _, _) in enumerate(rows):
        if not lo <= measured <= hi:
            continue
        # close neighbour on the left -> put this label right of its line
        near_left = k > 0 and measured - pos[k - 1] < 45
        ha = 'left' if (near_left or measured - lo < 30) else 'right'
        t = ax.text(measured, y_top - 0.01 * (y_top - y_min), f'{measured}  {cls}',
                    rotation=90, ha=ha, va='top', fontsize=FS_PEAK, color=CLASS_COL[cls])
        fig.canvas.draw()
        label_bottom = t.get_window_extent().transformed(ax.transData.inverted()).y0
        peak_top = ag[np.abs(ag[:, 0] - measured) <= 3, 1].max()
        ax.plot([measured, measured], [peak_top + 0.01 * (y_top - y_min), label_bottom],
                color='k', ls=':', lw=1.0)
    ax.tick_params(direction='in', top=True, length=6, width=1.0, labelsize=FS_TICK)

axL.spines['right'].set_visible(False)
axR.spines['left'].set_visible(False)
axL.tick_params(labelleft=False)                     # no intensity values
axR.tick_params(left=False, right=True, labelleft=False)
axL.set_xticks([400, 600, 800, 1000, 1200, 1400, 1600])
axR.set_xticks([2800, 3000])

# sample names inside the plot: Ag/SiNW name just below its spectrum,
# CaF2 name just below its spectrum (both at the left, from x = 420)
x_lab = (420, 760)
m = (ag[:, 0] >= x_lab[0]) & (ag[:, 0] <= x_lab[1])
axL.text(x_lab[0], ag[m, 1].min() - 0.015 * (y_top - y_min), f'{SAMPLE_NAME} on Ag/SiNWs',
         color=C_AG, fontsize=FS_NAME, ha='left', va='top')
m = (caf2[:, 0] >= x_lab[0]) & (caf2[:, 0] <= x_lab[1])
axL.text(x_lab[0], caf2[m, 1].min() - 0.015 * (y_top - y_min), f'{SAMPLE_NAME} on CaF$_2$',
         color=C_CAF2, fontsize=FS_NAME, ha='left', va='top')

# // break marks
d = 0.012
kw = dict(color='k', clip_on=False, lw=1.0)
for ax, x0 in [(axL, 1), (axR, 0)]:
    dx = d * (HIGH_MAX - HIGH_MIN) / (LOW_MAX - LOW_MIN) if ax is axL else d
    for y0 in (0, 1):
        ax.plot((x0 - dx, x0 + dx), (y0 - 2 * d, y0 + 2 * d), transform=ax.transAxes, **kw)

axL.set_ylabel('Raman Intensity (a.u.)', fontsize=FS_AXIS)
fig.text(0.53, 0.015, 'Wavenumber (cm$^{-1}$)', ha='center', fontsize=FS_AXIS)
fig.subplots_adjust(left=0.07, right=0.97, bottom=0.12, top=0.97)
fig.savefig(f'{OUT}.png', dpi=600)
fig.savefig(f'{OUT}.tiff', dpi=600, pil_kwargs={'compression': 'tiff_lzw'})
for r in rows:
    print(r)
print(f'saved {OUT}.png / .tiff and {OUT}_peaks.csv')
