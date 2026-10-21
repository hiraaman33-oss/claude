# =============================================================================
# ACACIA EVs vs NVs  |  single SERS spectra on Ag/SiNWs  |  comparison figure
#
#   S1-EV_ACACIA-AgSiNW ...txt              -> Acacia EVs  (black)
#   s4a-EV_nano_ACACIA-20ng-AgSiNW ...txt   -> Acacia NVs  (red)
#
# Raw counts are plotted exactly as recorded: no smoothing, no baseline
# correction, no normalisation.
# The NV spectrum is shifted up by a constant (OFFSET) so the two traces do not
# overlap; the y-axis therefore has no numbers, as usual for stacked spectra.
#
# Every peak label is the position of the local maximum of the spectrum within
# +/-12 cm-1 of the nominal band (found on a lightly smoothed copy, Savitzky-Golay
# 11 pts, 3rd order, used ONLY to locate the maximum; the plotted data are raw).
# Labels follow the layout of the user's original figure. The measured positions
# are written to EV_NV_Comparison_peaks.csv.
#
# HOW TO RUN (PyCharm):
#   pip install numpy scipy matplotlib
#   python EV_NV_Spectra_Comparison.py  <EV file .txt>  <NV file .txt>
# =============================================================================

import sys
import csv
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from scipy.signal import savgol_filter, find_peaks

EV_FILE = Path(sys.argv[1] if len(sys.argv) > 1 else
               'S1-EV_ACACIA-AgSiNW_532nm_1800gr_BC200_50XLF_1s_4a_10%.txt')
NV_FILE = Path(sys.argv[2] if len(sys.argv) > 2 else
               's4a-EV_nano_ACACIA-20ng-AgSiNW_532nm_1800gr_BC200_50XLF_1s_4a_10%.txt')
# which EV spectrum is used (S1 or S2), read from the EV file name
EV_TAG = 'S2' if Path(EV_FILE).name.startswith('S2') else 'S1'
OUT = 'EV_NV_Comparison' if EV_TAG == 'S1' else f'EV_NV_Comparison_{EV_TAG}'

X_MIN, X_MAX = 450, 1800
# counts added to the NV spectrum (display only); None = just enough to separate
OFFSET = {'S1': 560, 'S2': None}[EV_TAG]
# dashed comparison regions (boxes around the compared bands), per EV spectrum
BOXES = {'S1': [(775, 890), (1195, 1320), (1515, 1690)],
         'S2': [(758, 880), (962, 1070), (1148, 1184), (1335, 1405), (1550, 1658),
                (1700, 1752)]}[EV_TAG]

# nominal position, label class (None = number only), assignment (CSV), placement
# placement: 'up' / 'up-left' / 'up-right' = above the peak;
# 'pair-1' + 'pair-2' = doublet written as one label (e.g. 826/55)
PEAKS = {
    'NV': [
        (519, 'Si', 'Si substrate (crystalline silicon phonon)', 'up'),
        (797, 'Nucleic Acids', 'O-P-O backbone stretching', 'up-left'),
        (826, 'Proteins', 'Tyr Fermi doublet', 'pair-1'),
        (855, 'Proteins', 'Tyr Fermi doublet', 'pair-2'),
        (1001, 'Proteins', 'Phe ring breathing', 'up-left'),
        (1034, 'Proteins', 'Phe C-H in-plane bending', 'up-right'),
        (1241, 'Proteins', 'Amide III', 'up'),
        (1387, 'Lipids/NA', 'CH3 deformation (lipids) / nucleic-acid bases; COO- on Ag', 'up'),
        (1607, None, 'Phe/Tyr ring C=C stretching (protein)', 'up-left'),
        (1624, None, 'Tyr/Trp ring C=C stretching (protein)', 'up-right'),
    ],
    'EV_S1': [
        (1043, None, 'Phe C-H in-plane bending', 'up'),
        (1206, 'Proteins', 'Phe/Tyr C-C6H5 stretching', 'up'),
        (1284, None, 'Amide III', 'up'),
        (1380, None, 'CH3 deformation (lipids) / nucleic-acid bases; COO- on Ag', 'up'),
        (1604, None, 'Phe/Tyr ring C=C stretching', 'up-left'),
        (1628, None, 'Tyr/Trp ring C=C stretching (shoulder)', 'up-right'),
    ],
    'EV_S2': [
        (519, 'Si', 'Si substrate (crystalline silicon phonon)', 'up'),
        (836, 'Proteins', 'Tyr Fermi doublet', 'up'),
        (1001, 'Proteins', 'Phe ring breathing', 'up-left'),
        (1034, 'Proteins', 'Phe C-H in-plane bending', 'up-right'),
        (1167, 'Proteins', 'Tyr C-H bending; carotenoid C-C', 'up'),
        (1355, 'Proteins', 'Trp / CH deformation (broad)', 'up'),
        (1582, None, 'Phe C=C bending; COO- on Ag', 'up-left'),
        (1602, None, 'Phe/Tyr ring C=C stretching', 'up-right'),
        (1725, 'Lipids', 'Ester C=O stretching (lipids)', 'up'),
    ],
}
PEAKS['EV'] = PEAKS[f'EV_{EV_TAG}']
# extra class word written next to a group of numbers (as in the original figure)
GROUP_LABELS = [('NV', {'S1': 1712, 'S2': 1682}[EV_TAG], 'Proteins')]
# label text segments: (text, colour)
CLASS_SEGMENTS = {'Proteins': [('Proteins', '#7030A0')],
                  'Nucleic Acids': [('Nucleic Acids', '#1F3864')],
                  'Lipids/NA': [('Lipids/', '#2E75B6'), ('NA', 'k')],
                  'Lipids': [('Lipids', '#2E75B6')]}
# extra class word next to a group of numbers, per EV spectrum
GROUP_LABELS += {'S1': [], 'S2': [('EV', 1682, 'Proteins')]}[EV_TAG]
C_EV, C_NV = 'k', '#FF0000'

plt.rcParams.update({'font.family': 'sans-serif',
                     'font.sans-serif': ['Arial', 'Liberation Sans', 'DejaVu Sans'],
                     'font.size': 11})


def load(path):
    d = np.loadtxt(path)
    m = (d[:, 0] >= X_MIN) & (d[:, 0] <= X_MAX)
    return d[m, 0], d[m, 1]


def measure(x, y, nominal):
    """Peak position: most prominent local maximum of the smoothed spectrum within
    +/-12 cm-1 of the nominal value (works for peaks and shoulders)."""
    ys = savgol_filter(y, 11, 3)
    k = np.abs(x - nominal) <= 12
    pk, pr = find_peaks(ys[k], prominence=0)
    if len(pk):
        i = np.flatnonzero(k)[pk[np.argmax(pr['prominences'])]]
    else:
        i = np.flatnonzero(k)[np.argmax(ys[k])]
    return float(x[i])


spectra = {'EV': load(EV_FILE), 'NV': load(NV_FILE)}
if OFFSET is None:   # lowest NV point sits 60 counts above the highest EV point
    OFFSET = int(np.ceil((spectra['EV'][1].max() - spectra['NV'][1].min() + 60) / 10) * 10)

def segment_label(ax, xpos, ybase, segments, ha):
    """Vertical label made of coloured text segments stacked upwards."""
    y = ybase
    gap = 0.004 * (ax.get_ylim()[1] - ax.get_ylim()[0])   # remove the font's end padding
    for i, (txt, col) in enumerate(segments):
        t = ax.text(xpos, y, txt, rotation=90, va='bottom', ha=ha, fontsize=11, color=col)
        ax.figure.canvas.draw()
        y = t.get_window_extent().transformed(ax.transData.inverted()).y1 - gap


rows, labels = [], []
fig, ax = plt.subplots(figsize=(8.5, 6.5))
for s, col, off in [('EV', C_EV, 0), ('NV', C_NV, OFFSET)]:
    x, y = spectra[s]
    ys_plot = y                        # raw counts, plotted unchanged
    ax.plot(x, ys_plot + off, color=col, lw=0.7)
    for nominal, cls, assign, where in PEAKS[s]:
        pos = measure(x, y, nominal)
        rows.append([s, nominal, round(pos, 1), cls, assign])
        top = ys_plot[np.abs(x - pos) <= 4].max() + off
        txt = f'{pos:.1f}' if cls == 'Si' else f'{pos:.0f}'
        if where == 'pair-1':              # first band of a doublet: drawn with the second
            pair = (pos, txt, top)
            continue
        if where == 'pair-2':
            txt = f'{pair[1]}/{txt[-2:]}'           # e.g. 825/52, written as in the original
            pos, top, where = (pair[0] + pos) / 2, max(pair[2], top), 'up'
        top = ys_plot[np.abs(x - pos) <= 25].max() + off if where != 'up' else top
        dx = {'up': 0, 'up-left': -4, 'up-right': 4}[where]
        segs = [(txt, 'k')]
        if cls not in (None, 'Si'):
            segs = [(txt + '- ', 'k')] + CLASS_SEGMENTS[cls]
        labels.append((pos + dx, top + 8, segs,
                       {'up': 'center', 'up-left': 'right', 'up-right': 'left'}[where]))

y_lo = spectra['EV'][1].min() - 15
y_hi = spectra['NV'][1].max() + OFFSET + 190
for lo, hi in BOXES:
    ax.add_patch(Rectangle((lo, y_lo + 5), hi - lo, y_hi - y_lo - 20, fill=False,
                           ls='--', lw=0.8, ec='#1F3864'))
ax.set_xlim(X_MIN, X_MAX)
ax.set_ylim(y_lo, y_hi)
for lab in labels:                       # after the axis limits are fixed
    segment_label(ax, *lab)
for s_, xg, word in GROUP_LABELS:
    x, y = spectra[s_]
    off = OFFSET if s_ == 'NV' else 0
    segment_label(ax, xg, y[np.abs(x - xg) <= 60].max() + off + 8, CLASS_SEGMENTS[word], 'center')
ax.tick_params(axis='y', direction='in', length=5, labelleft=False)
ax.tick_params(axis='x', direction='in', length=5, top=False)
ax.set_xlabel('Raman Shift (cm$^{-1}$)', fontsize=13)
ax.set_ylabel('Intensity (a.u.)', fontsize=13)
# NV tag: above the frame (S1) or inside the frame, top-left (S2)
nv_xy, nv_ha, nv_va, nv_txt = {
    'S1': ((0.99, 1.012), 'right', 'bottom', 'Nanovesicles Acacia on Ag/SiNWs'),
    'S2': ((0.012, 0.975), 'left', 'top', 'Nanovesicles Acacia\non Ag/SiNWs')}[EV_TAG]
ax.text(*nv_xy, nv_txt, transform=ax.transAxes,
        ha=nv_ha, va=nv_va, color=C_NV, fontsize=11)
ev_xy, ev_txt = {'S1': ((0.006, 0.20), 'EV Acacia on Ag/SiNWs'),
                 'S2': ((0.105, 0.24), 'EV Acacia\non Ag/SiNWs')}[EV_TAG]
ax.text(*ev_xy, ev_txt, transform=ax.transAxes, ha='left', va='bottom', color=C_EV,
        fontsize=11)
fig.tight_layout()
fig.savefig(f'{OUT}.png', dpi=600)
fig.savefig(f'{OUT}.tiff', dpi=600, pil_kwargs={'compression': 'tiff_lzw'})

with open(f'{OUT}_peaks.csv', 'w', newline='') as fh:
    w = csv.writer(fh)
    w.writerow(['Spectrum', 'Nominal (cm-1)', 'Measured (cm-1)', 'Class', 'Assignment'])
    w.writerows(rows)
for r in rows:
    print(r)
