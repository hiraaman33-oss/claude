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
# Every peak label is measured from the data: the maximum of the lightly
# smoothed spectrum (Savitzky-Golay 11 pts, 3rd order; used ONLY to locate the
# maximum) after removing a straight local background over +/-30 cm-1.
# Labels follow the layout of the user's original figure. Bands rising less than
# 2.5 x the noise are flagged as weak in EV_NV_Comparison_peaks.csv (all measured
# values are written there).
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
OUT = 'EV_NV_Comparison'

X_MIN, X_MAX = 450, 1800
OFFSET = 560            # counts added to the NV spectrum (display only)
SNR_MIN = 2.5           # below this peak height / noise a band is flagged 'weak' in the CSV
BOXES = [(775, 890), (1195, 1320), (1515, 1690)]   # dashed comparison regions

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
    'EV': [
        (1043, None, 'Phe C-H in-plane bending', 'up'),
        (1206, 'Proteins', 'Phe/Tyr C-C6H5 stretching', 'up'),
        (1284, None, 'Amide III', 'up'),
        (1380, None, 'CH3 deformation (lipids) / nucleic-acid bases; COO- on Ag', 'up'),
        (1604, None, 'Phe/Tyr ring C=C stretching', 'up-left'),
        (1628, None, 'Tyr/Trp ring C=C stretching (shoulder)', 'up-right'),
    ],
}
# extra class word written next to a group of numbers (as in the original figure)
GROUP_LABELS = [('NV', 1712, 'Proteins')]
# label text segments: (text, colour)
CLASS_SEGMENTS = {'Proteins': [('Proteins', '#7030A0')],
                  'Nucleic Acids': [('Nucleic Acids', '#1F3864')],
                  'Lipids/NA': [('Lipids/', '#2E75B6'), ('NA', 'k')]}
C_EV, C_NV = 'k', '#FF0000'

plt.rcParams.update({'font.family': 'sans-serif',
                     'font.sans-serif': ['Arial', 'Liberation Sans', 'DejaVu Sans'],
                     'font.size': 11})


def load(path):
    d = np.loadtxt(path)
    m = (d[:, 0] >= X_MIN) & (d[:, 0] <= X_MAX)
    return d[m, 0], d[m, 1]


def measure(x, y, nominal, noise, shoulder=False):
    """Measured peak position, height above local line and SNR.
    Position = most prominent local maximum of the smoothed spectrum within
    +/-12 cm-1 of the nominal value (works for peaks and shoulders on a slope).
    Height = smoothed value at that position minus a straight local background
    drawn between the spectrum at -30 and +30 cm-1; for a shoulder (on the flank
    of a larger band) the prominence of the local maximum is used instead."""
    ys = savgol_filter(y, 11, 3)
    k = np.abs(x - nominal) <= 12
    pk, pr = find_peaks(ys[k], prominence=0)
    if len(pk):
        i = np.flatnonzero(k)[pk[np.argmax(pr['prominences'])]]
    else:
        i = np.flatnonzero(k)[np.argmax(ys[k])]
    w = np.abs(x - x[i]) <= 30
    xw, yw = x[w], ys[w]
    h = ys[i] - np.interp(x[i], [xw[0], xw[-1]], [yw[:6].mean(), yw[-6:].mean()])
    if shoulder:      # on the flank of a larger band: use the prominence of the local maximum
        h = float(pr['prominences'].max()) if len(pk) else 0.0
    return float(x[i]), float(h), float(h / noise)


spectra = {'EV': load(EV_FILE), 'NV': load(NV_FILE)}
noise = {s: np.std(np.diff(y)) / np.sqrt(2) for s, (x, y) in spectra.items()}

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
        sh = '(shoulder)' in assign
        pos, h, snr = measure(x, y, nominal, noise[s], shoulder=sh)
        weak = snr < SNR_MIN
        rows.append([s, nominal, round(pos, 1), round(h, 1), round(snr, 1), cls, assign,
                     'weak - tentative (~)' if weak else 'yes'])
        top = ys_plot[np.abs(x - pos) <= 4].max() + off
        txt = f'{pos:.1f}' if cls == 'Si' else f'{pos:.0f}'
        if where == 'pair-1':              # first band of a doublet: drawn with the second
            pair = (pos, txt, top, weak)
            continue
        if where == 'pair-2':
            txt = f'{pair[1]}/{txt[-2:]}'           # e.g. 825/52, written as in the original
            pos, top, weak, where = (pair[0] + pos) / 2, max(pair[2], top), pair[3] or weak, 'up'
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
ax.text(0.99, 1.012, 'Nanovesicles Acacia on Ag/SiNWs', transform=ax.transAxes,
        ha='right', va='bottom', color=C_NV, fontsize=11)
ax.text(0.006, 0.20, 'EV Acacia on Ag/SiNWs', transform=ax.transAxes,
        ha='left', va='bottom', color=C_EV, fontsize=11)
fig.tight_layout()
fig.savefig(f'{OUT}.png', dpi=600)
fig.savefig(f'{OUT}.tiff', dpi=600, pil_kwargs={'compression': 'tiff_lzw'})

with open(f'{OUT}_peaks.csv', 'w', newline='') as fh:
    w = csv.writer(fh)
    w.writerow(['Spectrum', 'Nominal (cm-1)', 'Measured (cm-1)', 'Height above local line (counts)',
                'Height / noise', 'Class', 'Assignment', 'Labelled'])
    w.writerows(rows)
print(f'noise: EV {noise["EV"]:.1f}, NV {noise["NV"]:.1f} counts')
for r in rows:
    print(r)
