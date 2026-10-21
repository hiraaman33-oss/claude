# =============================================================================
# ACACIA EVs vs NVs  |  SERS on Ag/SiNWs  |  mean +/- SD spectra, 400-1790 cm-1
#
# Uses all spectra of both data sets (6 sites x 20 spectra each) and the same
# pre-processing as the PCA-LDA / PLS-DA models (Acacia_EV_NV_Analysis.py):
#   cosmic-ray removal (modified z-score, threshold 7) -> 2 cm-1 grid ->
#   Savitzky-Golay (11 pts, 3rd order) -> ALS baseline (lambda 1e5, p 0.01) ->
#   SNV over 205-1795 cm-1
# Plotted: mean spectrum (line) +/- 1 standard deviation (shaded) of each group,
# NV shifted up by a constant (display only) so both can be labelled.
# Every label is the position of the maximum of the mean spectrum near the
# nominal band (parabolic refinement on the 2 cm-1 grid). Positions are written
# to MeanSD_400-1790_peaks.csv. Grey shadows mark the bands whose peak shift
# between EVs and NVs is consistent across all sites (MeanSD_400-1790_shifts.csv).
#
# HOW TO RUN (PyCharm):
#   pip install numpy pandas scipy matplotlib openpyxl
#   python Acacia_MeanSD_Spectra.py          (uses BASE_DIR below)
# OUTPUT: RESULTS\Fig_MeanSD_400-1790.png / .tiff (600 dpi) + peaks CSV
# =============================================================================

import sys
import csv
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import sparse
from scipy.sparse.linalg import spsolve
from scipy.signal import savgol_filter

BASE_DIR = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(r'C:\Users\Hira Aman\Downloads\CODE 005')
EV_FILE = BASE_DIR / 'EV Acacia' / 'EV_Acacia_Raman_Data.xlsx'
NV_FILE = BASE_DIR / 'Nano vesicles Acacia' / 'NV_Acacia_Raman_Data.xlsx'
OUTPUT_DIR = BASE_DIR / 'RESULTS'
OUTPUT_DIR.mkdir(exist_ok=True)

SITES = ['m1', 'm2', 'm3', 'm4', 'm5', 'm6']
X_MIN, X_MAX = 400, 1790
C_EV, C_NV = '#0B5A7E', '#B5541C'          # same colours as Figure 4
OFFSET = 8.5                               # SNV units added to the NV spectrum (display only)

# bands compared for a peak shift: (EV nominal, NV nominal). A light grey shadow is
# drawn over a band only when the shift is consistent: the six site-mean peak
# positions of EVs and of NVs do not overlap.
SHIFT_PAIRS = [(519, 519), (809, 799), (931, 927), (1001, 1001), (1161, 1169),
               (1249, 1245), (1373, 1385), (1603, 1607)]
SHADOW_PAD = 4                             # cm-1 added on each side of the two peaks

# nominal band, class, assignment (CSV), placement ('up', 'up-left', 'up-right')
PEAKS = {
    'EV': [
        (519, 'Si', 'Si substrate (crystalline silicon phonon)', 'up'),
        (809, 'Nucleic Acids', 'O-P-O backbone stretching', 'up'),
        (855, 'Proteins', 'Tyr ring breathing (Fermi doublet)', 'up'),
        (931, 'Proteins', 'C-C stretching, protein backbone', 'up'),
        (1001, 'Proteins', 'Phe ring breathing', 'up'),
        (1161, 'Proteins', 'Tyr C-H bending / C-C stretching', 'up'),
        (1249, 'Proteins', 'Amide III', 'up'),
        (1373, 'Lipids/NA', 'CH3 deformation (lipids) / nucleic-acid bases', 'up'),
        (1603, 'Proteins', 'Phe/Tyr ring C=C stretching', 'up'),
    ],
    'NV': [
        (519, 'Si', 'Si substrate (crystalline silicon phonon)', 'up'),
        (799, 'Nucleic Acids', 'O-P-O backbone stretching', 'up-left'),
        (829, 'Proteins', 'Tyr ring breathing (Fermi doublet)', 'up-right'),
        (927, 'Proteins', 'C-C stretching, protein backbone', 'up'),
        (1001, 'Proteins', 'Phe ring breathing', 'up-left'),
        (1033, 'Proteins', 'Phe C-H in-plane bending', 'up-right'),
        (1169, 'Proteins', 'Tyr C-H bending / C-C stretching', 'up'),
        (1245, 'Proteins', 'Amide III', 'up'),
        (1385, 'Lipids/NA', 'CH3 deformation (lipids) / nucleic-acid bases', 'up'),
        (1607, 'Proteins', 'Phe/Tyr ring C=C stretching', 'up'),
        (1729, 'Lipids', 'Ester C=O stretching (lipids)', 'up'),
    ],
}
CLASS_SEGMENTS = {'Proteins': [('Proteins', '#7030A0')],
                  'Nucleic Acids': [('Nucleic Acids', '#1F3864')],
                  'Lipids/NA': [('Lipids/', '#2E75B6'), ('NA', 'k')],
                  'Lipids': [('Lipids', '#2E75B6')]}

plt.rcParams.update({'font.family': 'sans-serif',
                     'font.sans-serif': ['Arial', 'Liberation Sans', 'DejaVu Sans'],
                     'font.size': 10})


# ----------------------------------------------------------------------------
# PRE-PROCESSING (identical to Acacia_EV_NV_Analysis.py, fingerprint window)
# ----------------------------------------------------------------------------
def despike(y, threshold=7.0, m=3):
    d = np.diff(y)
    med = np.median(d)
    mad = np.median(np.abs(d - med))
    if mad == 0:
        return y.copy()
    z = 0.6745 * (d - med) / mad
    spikes = np.zeros(len(y), bool)
    spikes[1:] = np.abs(z) > threshold
    out = y.copy()
    for i in np.where(spikes)[0]:
        lo, hi = max(0, i - m), min(len(y), i + m + 1)
        idx = [j for j in range(lo, hi) if not spikes[j]]
        if idx:
            out[i] = y[idx].mean()
    return out


def als_baseline(y, lam=1e5, p=0.01, niter=10):
    n = len(y)
    D = sparse.diags([1.0, -2.0, 1.0], [0, -1, -2], shape=(n, n - 2))
    DD = lam * D.dot(D.transpose())
    w = np.ones(n)
    for _ in range(niter):
        W = sparse.spdiags(w, 0, n, n)
        z = spsolve((W + DD).tocsc(), w * y)
        w = p * (y > z) + (1 - p) * (y < z)
    return z


def load_snv(path):
    grid = np.arange(205, 1796, 2.0)
    out, site_ids = [], []
    for site in SITES:
        df = pd.read_excel(path, sheet_name=site, header=2)
        x = df.iloc[:, 0].to_numpy(float)
        m = (x >= 195) & (x <= 1805)
        for y in df.iloc[:, 1:].to_numpy(float).T:
            ys = savgol_filter(np.interp(grid, x[m], despike(y[m])), 11, 3)
            c = ys - als_baseline(ys)
            out.append((c - c.mean()) / c.std())
            site_ids.append(site)
    return grid, np.array(out), np.array(site_ids)


def peak_position(g, s, lo, hi):
    """Wavenumber of the maximum in [lo, hi], refined by a parabola through 3 points."""
    m = (g >= lo) & (g <= hi)
    gg, ss = g[m], s[m]
    i = int(np.argmax(ss))
    if 0 < i < len(ss) - 1:
        a, b, c = ss[i - 1], ss[i], ss[i + 1]
        return gg[i] + 0.5 * (a - c) / (a - 2 * b + c) * (gg[1] - gg[0])
    return gg[i]


def segment_label(ax, xpos, ybase, segments, ha):
    """Vertical label made of coloured text segments stacked upwards."""
    y = ybase
    gap = 0.004 * (ax.get_ylim()[1] - ax.get_ylim()[0])
    for txt, col in segments:
        t = ax.text(xpos, y, txt, rotation=90, va='bottom', ha=ha, fontsize=10, color=col)
        ax.figure.canvas.draw()
        y = t.get_window_extent().transformed(ax.transData.inverted()).y1 - gap


# ----------------------------------------------------------------------------
# FIGURE
# ----------------------------------------------------------------------------
g, X_ev, s_ev = load_snv(EV_FILE)
_, X_nv, s_nv = load_snv(NV_FILE)
show = (g >= X_MIN) & (g <= X_MAX)
gs = g[show]

fig, ax = plt.subplots(figsize=(8.5, 6.5))
shifts = []
for ev_nom, nv_nom in SHIFT_PAIRS:
    pe = peak_position(g, X_ev.mean(0), ev_nom - 12, ev_nom + 12)
    pn = peak_position(g, X_nv.mean(0), nv_nom - 12, nv_nom + 12)
    se = [peak_position(g, X_ev[s_ev == k].mean(0), ev_nom - 12, ev_nom + 12) for k in SITES]
    sn = [peak_position(g, X_nv[s_nv == k].mean(0), nv_nom - 12, nv_nom + 12) for k in SITES]
    consistent = max(se) < min(sn) or max(sn) < min(se)
    shifts.append([round(pe, 1), round(pn, 1), round(pn - pe, 1), consistent])
    if consistent:                         # light shadow over the shifted band
        ax.axvspan(min(pe, pn) - SHADOW_PAD, max(pe, pn) + SHADOW_PAD, color='0.92', lw=0,
                   zorder=0)
rows, labels, lines = [], [], []
for s, X, col, off, name in [('EV', X_ev, C_EV, 0.0, 'EV Acacia on Ag/SiNWs'),
                             ('NV', X_nv, C_NV, OFFSET, 'Nanovesicles Acacia on Ag/SiNWs')]:
    mu, sd = X.mean(0)[show], X.std(0)[show]
    ax.fill_between(gs, mu - sd + off, mu + sd + off, color=col, alpha=0.22, lw=0)
    ax.plot(gs, mu + off, color=col, lw=1.0, label=f'{name} (mean \u00b1 SD, n = {len(X)})')
    for nominal, cls, assign, where in PEAKS[s]:
        pos = peak_position(gs, mu, nominal - 12, nominal + 12)
        k = np.abs(gs - pos) <= 4
        rows.append([s, nominal, round(pos, 1), round(float(np.interp(pos, gs, mu)), 3),
                     round(float(np.interp(pos, gs, sd)), 3), cls, assign])
        top = (mu + sd)[k].max() + off                     # label above the SD band
        if where != 'up':
            top = (mu + sd)[np.abs(gs - pos) <= 25].max() + off
        txt = f'{pos:.1f}' if cls == 'Si' else f'{pos:.0f}'
        segs = [(txt, 'k')] if cls == 'Si' else [(txt + '- ', 'k')] + CLASS_SEGMENTS[cls]
        dx = {'up': 0, 'up-left': -3, 'up-right': 3}[where]
        lines.append((pos, np.interp(pos, gs, mu) + off, top + 0.08))
        labels.append((pos + dx, top + 0.12, segs,
                       {'up': 'center', 'up-left': 'right', 'up-right': 'left'}[where]))

for xl, yb, yt in lines:                    # dotted line from the mean peak to its label
    ax.plot([xl, xl], [yb, yt], ls=(0, (1, 2)), lw=0.7, color='0.35')
ax.set_xlim(X_MIN, X_MAX)
ax.set_ylim((X_ev.mean(0) - X_ev.std(0))[show].min() - 0.3,
            (X_nv.mean(0) + X_nv.std(0))[show].max() + OFFSET + 4.3)
for lab in labels:                          # after the axis limits are fixed
    segment_label(ax, *lab)
ax.tick_params(axis='y', direction='in', length=5, labelleft=False)
ax.tick_params(axis='x', direction='in', length=5)
ax.set_xlabel('Raman Shift (cm$^{-1}$)', fontsize=13)
ax.set_ylabel('SNV Intensity (a.u.)', fontsize=13)
ax.legend(frameon=True, facecolor='white', edgecolor='none', framealpha=1, fontsize=10,
          loc='upper left', handlelength=1.5)
fig.tight_layout()
fig.savefig(OUTPUT_DIR / 'Fig_MeanSD_400-1790.png', dpi=600)
fig.savefig(OUTPUT_DIR / 'Fig_MeanSD_400-1790.tiff', dpi=600, pil_kwargs={'compression': 'tiff_lzw'})

with open(OUTPUT_DIR / 'MeanSD_400-1790_peaks.csv', 'w', newline='') as fh:
    w = csv.writer(fh)
    w.writerow(['Spectrum', 'Nominal (cm-1)', 'Measured (cm-1)', 'Mean SNV', 'SD', 'Class',
                'Assignment'])
    w.writerows(rows)
with open(OUTPUT_DIR / 'MeanSD_400-1790_shifts.csv', 'w', newline='') as fh:
    w = csv.writer(fh)
    w.writerow(['EV peak (cm-1)', 'NV peak (cm-1)', 'Shift NV-EV (cm-1)',
                'Consistent (site ranges separated, shadowed)'])
    w.writerows(shifts)
for r in rows + shifts:
    print(r)
print('Saved in:', OUTPUT_DIR)
