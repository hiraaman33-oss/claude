# =============================================================================
# EV NANO ACACIA  |  CaF2 vs AgSiNW  |  two raw spectra with peak assignments
#
# Layout follows the manual plot: both measured spectra on one axis (raw counts,
# no baseline / normalisation), broken x-axis over the 1800-2700 cm-1 gap,
# dotted lines at the peaks with the wavenumber and its biomolecular class.
#
# ONLY the two measured .txt files are used. The wavenumber printed for each
# line is the local maximum of the (lightly smoothed) AgSiNW spectrum within
# +/-8 cm-1 of the position marked on the manual plot.
#
# Assignment sources:
#   [P]  Roda F., Picciolini S., ... Gualerzi A., Bedoni M. Nanomaterials 2023, 13, 699
#   [M]  Movasaghi Z., Rehman S., Rehman I.U. Appl. Spectrosc. Rev. 2007, 42, 493-541
#   [T]  Talari A.C.S. et al. Appl. Spectrosc. Rev. 2015, 50, 46-111
#   [C]  Czamara K. et al. J. Raman Spectrosc. 2015, 46, 4-20
#   [G]  Gualerzi A. et al. Sci. Rep. 2017, 7, 9820
#
# HOW TO RUN:
#   pip install numpy scipy matplotlib
#   python EV_Acacia_TwoSpectra.py  <CaF2.txt>  <AgSiNW.txt>  <S1|S2>
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
SAMPLE = sys.argv[3] if len(sys.argv) > 3 else 'S1'
OUT_PNG = f'R13_EV_Acacia_{SAMPLE}_TwoSpectra.png'
OUT_CSV = f'R13_EV_Acacia_{SAMPLE}_PeakAssignments.csv'

# ----------------------------------------------------------------------------
# SETTINGS
# ----------------------------------------------------------------------------
LOW_MIN, LOW_MAX = 400, 1800
HIGH_MIN, HIGH_MAX = 2700, 3100
SEARCH_HALF = 8          # cm-1 window to locate the measured maximum

# (nominal position, class, assignment, source)
# S1: positions marked on the manual plot.
# S2: peaks clearly above noise in the S2 AgSiNW spectrum (prominence > ~3.5x noise).
PEAK_SETS = {}
PEAK_SETS['S1'] = [
    (228,  'Ag substrate',  'Ag-adsorbate (Ag-N / Ag-O / Ag-Cl) low-frequency mode; substrate, not EV', '[T]'),
    (519,  'Si substrate',  'Crystalline silicon optical phonon (520.7 cm-1 Si band); from SiNW', '[P]'),
    (612,  'Protein',       'C-C twisting of phenylalanine (~621); weak Si 2nd-order phonon may overlap', '[M][T]'),
    (795,  'Nucleic acid',  'O-P-O backbone / phosphodiester stretching (785-810)', '[M][T]'),
    (827,  'Protein/NA',    'Tyrosine Fermi doublet (828/850) / O-P-O asymmetric stretch (828)', '[M][T]'),
    (921,  'Protein/Carb.', 'C-C stretch of proline ring / glucose (920)', '[M][T]'),
    (1001, 'Protein',       'Phenylalanine symmetric ring breathing (1002-1004)', '[M][T][G]'),
    (1031, 'Protein',       'Phenylalanine C-H in-plane bending (1030-1031)', '[M][T]'),
    (1240, 'Protein',       'Amide III (1230-1250); asym. PO2- of nucleic acids may overlap', '[M][T][G]'),
    (1386, 'Lipid/Protein', 'CH3 symmetric deformation (~1380) / COO- symmetric stretch', '[M][T]'),
    (1606, 'Protein',       'Phenylalanine / tyrosine ring C=C stretch (1605-1607)', '[M][T][G]'),
    (1625, 'Protein',       'Amide I low-frequency side / Trp-Tyr C=C (1617-1630)', '[M][T]'),
    (2936, 'Protein/Lipid', 'CH3 symmetric stretching (2930-2940), mainly proteins', '[P][M][C][G]'),
]
PEAK_SETS['S2'] = [
    (519,  'Si substrate',  'Crystalline silicon optical phonon (520.7 cm-1 Si band); from SiNW', '[P]'),
    (836,  'Protein/NA',    'Tyrosine Fermi doublet (828/850) / O-P-O asymmetric stretch (828)', '[M][T]'),
    (1001, 'Protein',       'Phenylalanine symmetric ring breathing (1002-1004)', '[M][T][G]'),
    (1034, 'Protein',       'Phenylalanine C-H in-plane bending (1030-1034)', '[M][T]'),
    (1167, 'Protein',       'Tyrosine C-H in-plane bending (1170-1175)', '[M][T]'),
    (1355, 'Protein',       'Tryptophan / CH deformation (1340-1380); broad band', '[M][T]'),
    (1520, 'Carotenoid',    'C=C stretching of carotenoids (1510-1530); broad shoulder', '[M][T]'),
    (1582, 'Protein',       'Phenylalanine C=C bending (1584)', '[M][T]'),
    (1602, 'Protein',       'Phenylalanine / tyrosine ring C=C stretch (1600-1607)', '[M][T][G]'),
    (1725, 'Lipid',         'C=O ester stretching of (phospho)lipids (1720-1745)', '[M][T][C]'),
    (2931, 'Protein/Lipid', 'CH2/CH3 stretching (2925-2935)', '[P][M][C][G]'),
]
# constant vertical shift of the AgSiNW trace, display only (stated in legend)
AG_OFFSET = {'S1': 0, 'S2': 110}[SAMPLE]
# sample name used in the legend (from the raw file names)
SAMPLE_NAME = {'S1': 'EV Nano Acacia', 'S2': 'EV Acacia'}[SAMPLE]
PEAKS = PEAK_SETS[SAMPLE]

C_AG = '#E8383D'     # red  : EV on AgSiNW
C_CAF2 = '#3A3A3A'   # dark grey : EV on CaF2
CLASS_COL = {        # label colour by class
    'Protein': '#1F6FB4', 'Protein/NA': '#6A3D9A', 'Nucleic acid': '#8E44AD',
    'Protein/Carb.': '#1F6FB4', 'Lipid/Protein': '#D35400',
    'Protein/Lipid': '#D35400', 'Lipid': '#C0392B', 'Carotenoid': '#2E8B57', 'Ag substrate': '#7F7F7F', 'Si substrate': '#7F7F7F',
}

# ----------------------------------------------------------------------------
# DATA
# ----------------------------------------------------------------------------
caf2 = np.loadtxt(FILE_CAF2)
ag = np.loadtxt(FILE_AG)
ag[:, 1] += AG_OFFSET

# measured peak positions (smoothing only used to locate the maximum)
ag_s = savgol_filter(ag[:, 1], 11, 3)
rows = []
for marked, cls, assign, ref in PEAKS:
    # remove the local sloping background (straight line over +/-25 cm-1)
    w_ = np.abs(ag[:, 0] - marked) <= 25
    xw, yw = ag[w_, 0], ag_s[w_]
    yw = yw - np.interp(xw, [xw[0], xw[-1]], [yw[:5].mean(), yw[-5:].mean()])
    m = np.abs(xw - marked) <= SEARCH_HALF
    measured = xw[m][np.argmax(yw[m])]
    rows.append((marked, round(float(measured)), cls, assign, ref))

with open(OUT_CSV, 'w', newline='') as fh:
    w = csv.writer(fh)
    w.writerow(['Nominal (cm-1)', 'Measured AgSiNW (cm-1)', 'Class', 'Assignment', 'Source'])
    w.writerows(rows)

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

def in_view(d):
    x = d[:, 0]
    return d[((x >= LOW_MIN) & (x <= LOW_MAX)) | ((x >= HIGH_MIN) & (x <= HIGH_MAX)), 1]


y_min = min(in_view(caf2).min(), in_view(ag).min())
y_max = max(in_view(caf2).max(), in_view(ag).max())
y_top = y_max + 0.45 * (y_max - y_min)       # head-room for the rotated labels

for ax, lo, hi in [(axL, LOW_MIN, LOW_MAX), (axR, HIGH_MIN, HIGH_MAX)]:
    for d, col, lab in [(caf2, C_CAF2, f'{SAMPLE_NAME} - CaF$_2$'),
                        (ag, C_AG, f'{SAMPLE_NAME} - AgSiNW'
                         + (f' (+{AG_OFFSET})' if AG_OFFSET else ''))]:
        m = (d[:, 0] >= lo) & (d[:, 0] <= hi)
        ax.plot(d[m, 0], d[m, 1], color=col, lw=0.6, label=lab)
    pos = [r[1] for r in rows]
    for k, (marked, measured, cls, _, _) in enumerate(rows):
        if lo <= measured <= hi:
            ax.axvline(measured, color='k', ls=':', lw=0.8)
            # close neighbour on the left -> put this label right of its line
            near_left = k > 0 and measured - pos[k - 1] < 35
            ha = 'left' if (near_left or measured - lo < 30) else 'right'
            ax.text(measured, y_top - 0.01 * (y_top - y_min),
                    f'{measured}  {cls}', rotation=90, ha=ha, va='top',
                    fontsize=7.5, color=CLASS_COL[cls])
    ax.set_xlim(lo, hi)
    ax.tick_params(direction='in', top=True)

axL.set_ylim(y_min - 0.05 * (y_max - y_min), y_top)
axL.spines['right'].set_visible(False)
axR.spines['left'].set_visible(False)
axR.tick_params(left=False, right=True, labelleft=False)
axL.set_xticks([400, 600, 800, 1000, 1200, 1400, 1600])
axR.set_xticks([2800, 3000])

# // break marks
d = 0.012
kw = dict(color='k', clip_on=False, lw=0.8)
for ax, x0, k in [(axL, 1, (HIGH_MAX - HIGH_MIN) / (LOW_MAX - LOW_MIN)), (axR, 0, 1)]:
    dx = d * k if ax is axL else d
    for y0 in (0, 1):
        ax.plot((x0 - dx, x0 + dx), (y0 - 2 * d, y0 + 2 * d), transform=ax.transAxes, **kw)

axL.set_ylabel('Raman Intensity (a.u.)', fontsize=11)
fig.text(0.5, 0.02, 'Wavenumber (cm$^{-1}$)', ha='center', fontsize=11)
axL.legend(loc='upper center', bbox_to_anchor=(0.62, 1.12), ncol=1,
           frameon=True, fontsize=8)
fig.subplots_adjust(left=0.08, right=0.97, bottom=0.10, top=0.88)
fig.savefig(OUT_PNG, dpi=600)
print(f'saved {OUT_PNG} and {OUT_CSV}')
