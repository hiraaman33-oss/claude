# =============================================================================
# EV ACACIA  |  CaF2 vs AgSiNW  |  raw spectra with peak labels on the peaks
#
# Layout follows the user's Origin plot: both raw spectra, AgSiNW drawn above
# CaF2 with a constant vertical offset (stated in the legend), broken x-axis
# over the 1800-2700 cm-1 gap, and each peak labelled directly above its maximum
# with wavenumber + assignment (rotated text, no lines).
#
# Data: only the two measured .txt files. Intensities are raw counts; the one
# change is the declared constant offset added to the AgSiNW trace for display.
# The wavenumber printed for each peak is the local maximum of the AgSiNW
# spectrum (lightly smoothed, local slope removed) within +/-8 cm-1 of the
# nominal position.
#
# Assignment sources:
#   [P]  Roda F. et al. Nanomaterials 2023, 13, 699
#   [M]  Movasaghi Z., Rehman S., Rehman I.U. Appl. Spectrosc. Rev. 2007, 42, 493-541
#   [T]  Talari A.C.S. et al. Appl. Spectrosc. Rev. 2015, 50, 46-111
#   [C]  Czamara K. et al. J. Raman Spectrosc. 2015, 46, 4-20
#
# HOW TO RUN:
#   pip install numpy scipy matplotlib
#   python EV_Acacia_Annotated.py  <CaF2.txt>  <AgSiNW.txt>  S2
# =============================================================================

import sys
import csv
import os

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter

# ----------------------------------------------------------------------------
# INPUT / OUTPUT
# ----------------------------------------------------------------------------
FILE_CAF2 = sys.argv[1]
FILE_AG = sys.argv[2]
SAMPLE = sys.argv[3] if len(sys.argv) > 3 else 'S2'
OUT_PNG = f'R14_EV_Acacia_{SAMPLE}_Annotated.png'
OUT_CSV = f'R14_EV_Acacia_{SAMPLE}_Annotated_Peaks.csv'

# ----------------------------------------------------------------------------
# SETTINGS
# ----------------------------------------------------------------------------
LOW_MIN, LOW_MAX = 200, 1800
HIGH_MIN, HIGH_MAX = 2700, 3200
AG_OFFSET = 110          # constant shift of the AgSiNW trace, display only
SEARCH_HALF = 8          # cm-1 window to locate the measured maximum

# (nominal cm-1, label text after the number, assignment, source)
# Label text follows the user's Origin plot.
# None as nominal -> label for a band region, printed as given (not measured).
PEAK_SETS = {}
PEAK_SETS['S2'] = [
    (519,  'Si peak',                        'Crystalline silicon phonon (520.7 cm-1); SiNW substrate', '[P]'),
    (836,  'Proteins',                       'Tyrosine Fermi doublet (828/850)', '[M][T]'),
    (1001, 'Proteins',                       'Phenylalanine symmetric ring breathing (1002-1004)', '[M][T]'),
    (1034, 'Proteins',                       'Phenylalanine C-H in-plane bending (1030-1034)', '[M][T]'),
    (1167, 'Protein (tyrosine C-H bend)',    'Tyrosine C-H in-plane bending (1170-1175)', '[M][T]'),
    (1355, 'Proteins',                       'Tryptophan / CH deformation (1340-1380), broad band', '[M][T]'),
    (1582, 'Proteins (Phe C=C)',             'Phenylalanine C=C bending (1584)', '[M][T]'),
    (1602, 'Proteins (Phe/Tyr ring C=C)',    'Phenylalanine / tyrosine ring C=C stretch (1600-1607)', '[M][T]'),
    (1725, 'Lipids',                         'C=O ester stretching of (phospho)lipids (1720-1745)', '[M][T][C]'),
    (2931, 'Lipids',                         'CH2/CH3 stretching (2925-2935); lipids and proteins', '[P][M][C]'),
]
# band regions labelled as a range (centre, text, assignment, source)
BANDS = {
    'S2': [(1520, '1510-1530 Lipids (Carotenoids)',
            'C=C stretching of carotenoids (1520-1525)', '[M][T]')],
}

C_AG = '#E8383D'     # red  : AgSiNW
C_CAF2 = '#3A3A3A'   # dark grey : CaF2

# ----------------------------------------------------------------------------
# DATA
# ----------------------------------------------------------------------------
caf2 = np.loadtxt(FILE_CAF2)
ag = np.loadtxt(FILE_AG)
ag_s = savgol_filter(ag[:, 1], 11, 3)


def measured_peak(nominal):
    """Local maximum near nominal, after removing a straight-line background."""
    w_ = np.abs(ag[:, 0] - nominal) <= 25
    xw, yw = ag[w_, 0], ag_s[w_]
    yw = yw - np.interp(xw, [xw[0], xw[-1]], [yw[:5].mean(), yw[-5:].mean()])
    m = np.abs(xw - nominal) <= SEARCH_HALF
    return float(xw[m][np.argmax(yw[m])])


def top_of_trace(x0, half=6):
    """Highest raw AgSiNW point (with offset) around x0 - where a label starts."""
    m = np.abs(ag[:, 0] - x0) <= half
    return ag[m, 1].max() + AG_OFFSET


rows = []
for nominal, text, assign, ref in PEAK_SETS[SAMPLE]:
    pos = measured_peak(nominal)
    rows.append((nominal, round(pos), f'{round(pos)} {text}', assign, ref))
band_rows = [(c, None, t, a, r) for c, t, a, r in BANDS.get(SAMPLE, [])]

with open(OUT_CSV, 'w', newline='') as fh:
    w = csv.writer(fh)
    w.writerow(['Nominal (cm-1)', 'Measured AgSiNW (cm-1)', 'Label', 'Assignment', 'Source'])
    w.writerows(rows + band_rows)

# ----------------------------------------------------------------------------
# PLOT
# ----------------------------------------------------------------------------
plt.rcParams.update({'font.family': 'sans-serif',
                     'font.sans-serif': ['Arial', 'Liberation Sans', 'DejaVu Sans'],
                     'font.size': 10})
fig, (axL, axR) = plt.subplots(
    1, 2, sharey=True, figsize=(9, 6.5),
    gridspec_kw={'width_ratios': [LOW_MAX - LOW_MIN, HIGH_MAX - HIGH_MIN],
                 'wspace': 0.03})

name_ag = os.path.basename(FILE_AG).split('-', 1)[-1].rsplit('.', 1)[0]
name_caf2 = os.path.basename(FILE_CAF2).split('-', 1)[-1].rsplit('.', 1)[0]

for ax, lo, hi in [(axL, LOW_MIN, LOW_MAX), (axR, HIGH_MIN, HIGH_MAX)]:
    m = (ag[:, 0] >= lo) & (ag[:, 0] <= hi)
    ax.plot(ag[m, 0], ag[m, 1] + AG_OFFSET, color=C_AG, lw=0.6,
            label=f'{name_ag}  (offset +{AG_OFFSET})')
    m = (caf2[:, 0] >= lo) & (caf2[:, 0] <= hi)
    ax.plot(caf2[m, 0], caf2[m, 1], color=C_CAF2, lw=0.6, label=name_caf2)
    ax.set_xlim(lo, hi)
    ax.tick_params(direction='in', top=True)

# peak labels: rotated text starting just above the peak maximum
pad = 6
labels = rows + band_rows
for k, (nominal, pos, text, _, _) in enumerate(labels):
    x = pos if pos is not None else nominal
    ax = axL if x <= LOW_MAX else axR
    # two close peaks: push their labels apart sideways
    ha = 'center'
    for other in labels:
        ox = other[1] if other[1] is not None else other[0]
        if ox != x and abs(ox - x) < 30:
            ha = 'right' if x < ox else 'left'
    ax.text(x, top_of_trace(x) + pad, text, rotation=90, ha=ha, va='bottom',
            fontsize=8)

all_y = np.concatenate([ag[:, 1] + AG_OFFSET, caf2[:, 1]])
axL.set_ylim(all_y.min() - 10, all_y.max() + 175)
axL.spines['right'].set_visible(False)
axR.spines['left'].set_visible(False)
axR.tick_params(left=False, right=True, labelleft=False)
axL.set_xticks([200, 400, 600, 800, 1000, 1200, 1400, 1600])
axR.set_xticks([2800, 3000, 3200])

# // break marks
d = 0.012
kw = dict(color='k', clip_on=False, lw=0.8)
k = (HIGH_MAX - HIGH_MIN) / (LOW_MAX - LOW_MIN)
for ax, x0, dx in [(axL, 1, d * k), (axR, 0, d)]:
    for y0 in (0, 1):
        ax.plot((x0 - dx, x0 + dx), (y0 - 2 * d, y0 + 2 * d), transform=ax.transAxes, **kw)

axL.set_ylabel('Raman Intensity (a.u.)', fontsize=11)
fig.text(0.5, 0.02, 'Wavenumber (cm$^{-1}$)', ha='center', fontsize=11)
axL.legend(loc='upper center', bbox_to_anchor=(0.62, 1.12), frameon=True, fontsize=7.5)
fig.subplots_adjust(left=0.08, right=0.97, bottom=0.10, top=0.88)
fig.savefig(OUT_PNG, dpi=600)
print(f'saved {OUT_PNG} and {OUT_CSV}')
for r in rows:
    print(r[:3])
