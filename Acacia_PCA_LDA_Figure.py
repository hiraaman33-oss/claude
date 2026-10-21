# =============================================================================
# ACACIA EVs vs NVs  |  SERS on Ag/SiNWs  |  PCA scores + LDA canonical variable
#
#   (a) PCA scores, fingerprint region (250-1790 cm-1)
#   (b) canonical variable 1 (PCA-LDA, 5 PCs), fingerprint region
#   (c) PCA scores, CH-stretching region (2760-3095 cm-1)
#   (d) canonical variable 1 (PCA-LDA, 5 PCs), CH-stretching region
# and a second figure: mean +/- SD (sample SD, n - 1) CH-stretching spectra of EVs
# and NVs (2760-3095 cm-1) with band positions and assignments.
#
# Same data, pre-processing and models as Acacia_EV_NV_Analysis.py (Figure 4,
# panels a, b, d, e):
#   cosmic-ray removal (modified z-score, threshold 7) -> 2 cm-1 grid ->
#   Savitzky-Golay (11 pts, 3rd order) -> ALS baseline (lambda 1e5, p 0.01) ->
#   SNV within each window (205-1795 and 2705-3095 cm-1)
#   PCA (5 components) on all 240 spectra; LDA on the 5 PC scores.
#   Ellipses: 2 standard deviations of the PC1/PC2 scores of each group.
#   Box plots: median, quartiles, whiskers = 1.5 x IQR; dots = single spectra.
#   p value: Welch's t-test on the 6 site means of each group.
#
# HOW TO RUN (PyCharm):
#   pip install numpy pandas scipy matplotlib openpyxl scikit-learn
#   python Acacia_PCA_LDA_Figure.py          (uses BASE_DIR below)
# OUTPUT: RESULTS\Fig_PCA_LDA.png / .tiff (600 dpi)
#         RESULTS\Fig_MeanSD_CH.png / .tiff (600 dpi) + MeanSD_CH_bands.csv
#
# Band positions in the CH figure (measured on the mean spectrum of each group):
#   'max' = maximum of the band (parabola through the 3 highest points)
#   'sh'  = shoulder: minimum of the 2nd derivative (Savitzky-Golay, 11 pts, 3rd
#           order), the standard way to locate a band without its own maximum
# A light grey shadow marks a band whose EV/NV position differs consistently:
# the positions in the 6 site-mean spectra of EVs and NVs do not overlap.
# =============================================================================

import sys
import csv
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from scipy import sparse
from scipy.sparse.linalg import spsolve
from scipy.signal import savgol_filter, find_peaks
from scipy.stats import ttest_ind
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

BASE_DIR = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(r'C:\Users\Hira Aman\Downloads\CODE 005')
EV_FILE = BASE_DIR / 'EV Acacia' / 'EV_Acacia_Raman_Data.xlsx'
NV_FILE = BASE_DIR / 'Nano vesicles Acacia' / 'NV_Acacia_Raman_Data.xlsx'
OUTPUT_DIR = BASE_DIR / 'RESULTS'
OUTPUT_DIR.mkdir(exist_ok=True)

SITES = ['m1', 'm2', 'm3', 'm4', 'm5', 'm6']
C_EV, C_NV = '#0B5A7E', '#B5541C'
COLS = {0: C_EV, 1: C_NV}
GROUPS = {0: 'EVs', 1: 'NVs'}
N_COMP = 5
# pre-processing window (raw range read, 2 cm-1 grid) and analysed region
WINDOWS = {'fingerprint': {'raw': (200, 1800), 'grid': (205, 1795), 'region': (250, 1790)},
           'ch': {'raw': (2700, 3100), 'grid': (2705, 3095), 'region': (2760, 3095)}}
# PC1 is oriented so that its loading is positive at this band (as in the text)
PC1_REF = {'fingerprint': 1580, 'ch': 2960}
TITLES = {'fingerprint': 'Fingerprint (250-1790 cm$^{-1}$)',
          'ch': 'CH-stretching (2760-3095 cm$^{-1}$)'}

plt.rcParams.update({'font.family': 'sans-serif',
                     'font.sans-serif': ['Arial', 'Liberation Sans', 'DejaVu Sans'],
                     'font.size': 9, 'axes.spines.top': False, 'axes.spines.right': False})


# ----------------------------------------------------------------------------
# PRE-PROCESSING (identical to Acacia_EV_NV_Analysis.py)
# ----------------------------------------------------------------------------
def despike(y, threshold=7.0, m=3):
    d = np.diff(y)
    med = np.median(d)
    mad = np.median(np.abs(d - med))
    if mad == 0:
        return y.copy()
    spikes = np.zeros(len(y), bool)
    spikes[1:] = np.abs(0.6745 * (d - med) / mad) > threshold
    out = y.copy()
    for i in np.where(spikes)[0]:
        idx = [j for j in range(max(0, i - m), min(len(y), i + m + 1)) if not spikes[j]]
        if idx:
            out[i] = y[idx].mean()
    return out


def als_baseline(y, lam=1e5, p=0.01, niter=10):
    n = len(y)
    D = sparse.diags([1.0, -2.0, 1.0], [0, -1, -2], shape=(n, n - 2))
    DD = lam * D.dot(D.transpose())
    w = np.ones(n)
    for _ in range(niter):
        z = spsolve((sparse.spdiags(w, 0, n, n) + DD).tocsc(), w * y)
        w = p * (y > z) + (1 - p) * (y < z)
    return z


def preprocess(x, Y, win):
    lo, hi = WINDOWS[win]['raw']
    grid = np.arange(WINDOWS[win]['grid'][0], WINDOWS[win]['grid'][1] + 1, 2.0)
    m = (x >= lo - 5) & (x <= hi + 5)
    corr = []
    for y in Y:
        ys = savgol_filter(np.interp(grid, x[m], despike(y[m])), 11, 3)
        corr.append(ys - als_baseline(ys))
    corr = np.array(corr)
    return grid, (corr - corr.mean(1, keepdims=True)) / corr.std(1, keepdims=True)


data = {w: [] for w in WINDOWS}
y, sites = [], []
for cls, path in enumerate([EV_FILE, NV_FILE]):
    for site in SITES:
        df = pd.read_excel(path, sheet_name=site, header=2)
        x, Y = df.iloc[:, 0].to_numpy(float), df.iloc[:, 1:].to_numpy(float).T
        for w in WINDOWS:
            grid, snv = preprocess(x, Y, w)
            r0, r1 = WINDOWS[w]['region']
            keep = (grid >= r0) & (grid <= r1)
            data[w].append(snv[:, keep])
            WINDOWS[w]['g'] = grid[keep]
        y += [cls] * len(Y)
        sites += [f'{GROUPS[cls]}-{site}'] * len(Y)
data = {w: np.vstack(v) for w, v in data.items()}
y, sites = np.array(y), np.array(sites)
print(f'Spectra: {len(y)} (EV {np.sum(y == 0)}, NV {np.sum(y == 1)}), sites: {len(set(sites))}')


def site_means(v):
    return {c: np.array([v[sites == s].mean() for s in sorted(set(sites[y == c]))]) for c in (0, 1)}


def stars(p):
    return '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'n.s.'


# ----------------------------------------------------------------------------
# FIGURE
# ----------------------------------------------------------------------------
fig, axs = plt.subplots(2, 2, figsize=(5.0, 5.3),
                        gridspec_kw={'width_ratios': [1.1, 0.7], 'wspace': 0.42, 'hspace': 0.45})
letters = iter('abcd')
for row, w in enumerate(['fingerprint', 'ch']):
    X, g = data[w], WINDOWS[w]['g']
    pca = PCA(n_components=N_COMP, svd_solver='full').fit(X)
    T = pca.transform(X)
    T[:, 0] *= np.sign(pca.components_[0][np.argmin(np.abs(g - PC1_REF[w]))])
    ev1, ev2 = pca.explained_variance_ratio_[:2] * 100

    # --- PCA scores with 2-SD ellipses
    ax = axs[row, 0]
    for c in (0, 1):
        P = T[y == c, :2]
        ax.scatter(P[:, 0], P[:, 1], s=6, color=COLS[c], lw=0, alpha=0.85, label=GROUPS[c])
        vals, vecs = np.linalg.eigh(np.cov(P.T))
        ang = np.degrees(np.arctan2(vecs[1, 1], vecs[0, 1]))
        ax.add_patch(Ellipse(P.mean(0), 4 * np.sqrt(vals[1]), 4 * np.sqrt(vals[0]), angle=ang,
                             fc='none', ec=COLS[c], lw=1.0))
    ax.axhline(0, color='k', lw=0.5)
    ax.axvline(0, color='k', lw=0.5)
    ax.set_xlabel(f'PC1 ({ev1:.1f}%)')
    ax.set_ylabel(f'PC2 ({ev2:.1f}%)')
    ylo, yhi = ax.get_ylim()
    ax.set_ylim(ylo, yhi + 0.28 * (yhi - ylo))
    ax.legend(frameon=False, fontsize=7.5, loc='upper left', ncol=2, handletextpad=0.1,
              columnspacing=0.8, markerscale=1.5)
    ax.set_title(TITLES[w], fontsize=8.5, loc='left')

    # --- canonical variable 1 (LDA on the 5 PC scores)
    ld = LinearDiscriminantAnalysis().fit(T, y).transform(T).ravel()
    ld *= np.sign(ld[y == 1].mean())
    ax = axs[row, 1]
    rng = np.random.default_rng(0)          # horizontal jitter of the dots only
    for c in (0, 1):
        v = ld[y == c]
        ax.boxplot(v, positions=[c], widths=0.5, showfliers=False, patch_artist=True,
                   boxprops=dict(fc=COLS[c], alpha=0.30, ec=COLS[c]), medianprops=dict(color=COLS[c]),
                   whiskerprops=dict(color=COLS[c]), capprops=dict(color=COLS[c]))
        ax.scatter(c + rng.uniform(-0.15, 0.15, len(v)), v, s=3, color=COLS[c], lw=0, alpha=0.8)
    sm = site_means(ld)
    p = ttest_ind(sm[0], sm[1], equal_var=False).pvalue
    top = ld.max() + 0.08 * np.ptp(ld)
    ax.plot([0, 1], [top, top], color='k', lw=0.6)
    ax.text(0.5, top, stars(p), ha='center', va='bottom', fontsize=8)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(['EVs', 'NVs'])
    ax.set_ylabel('Canonical variable 1')
    ax.set_ylim(ld.min() - 0.18 * np.ptp(ld), top + 0.15 * np.ptp(ld))
    ax.text(0.5, 0.02, "Welch's t-test on site means\n(n = 6), " + stars(p) + f' p = {p:.1e}',
            transform=ax.transAxes, ha='center', va='bottom', fontsize=6)
    print(f'{w}: PC1 {ev1:.1f} %, PC2 {ev2:.1f} %, LD1 Welch p (site means) = {p:.2e}')

    for ax in axs[row]:
        ax.text(-0.30 if ax is axs[row, 1] else -0.22, 1.10, f'({next(letters)})',
                transform=ax.transAxes, fontsize=10, va='bottom')
fig.subplots_adjust(left=0.13, right=0.98, bottom=0.09, top=0.93)
fig.savefig(OUTPUT_DIR / 'Fig_PCA_LDA.png', dpi=600)
fig.savefig(OUTPUT_DIR / 'Fig_PCA_LDA.tiff', dpi=600, pil_kwargs={'compression': 'tiff_lzw'})
print('Saved in:', OUTPUT_DIR)


# ============================================================================
# CH-STRETCHING REGION: mean +/- SD spectra with band positions
# ============================================================================
CH_OFFSET = 5.5                            # SNV units added to the NV spectrum (display only)
# bands labelled (and shadowed) in the figure; all bands are written to the CSV
CH_SHOW = [2935]                           # main CH band only
# (nominal band, search half-width, 'max' or 'sh', class, assignment)
CH_BANDS = [
    (2853, 10, 'sh', 'Lipids', 'CH2 symmetric stretching'),
    (2880, 12, 'sh', 'Lipids', 'CH2 asymmetric stretching'),
    (2935, 12, 'max', 'Proteins', 'CH3 symmetric stretching'),
    (2960, 10, 'sh', 'Proteins/Lipids', 'CH3 asymmetric stretching'),
    (3065, 12, 'max', 'Proteins', 'Aromatic =C-H stretching'),
]
CLASS_SEGMENTS = {'Proteins': [('Proteins', '#7030A0')],
                  'Lipids': [('Lipids', '#2E75B6')],
                  'Proteins/Lipids': [('Proteins/', '#7030A0'), ('Lipids', '#2E75B6')]}


def band_position(gr, s, nominal, half, kind):
    """'max': most prominent local maximum within +/-half of the nominal band;
    'sh': most prominent local minimum of the 2nd derivative there.
    Refined by a parabola through 3 points. None = no such band (not resolved)."""
    if kind == 'sh':
        s = -savgol_filter(s, 11, 3, deriv=2)
    m = np.flatnonzero(np.abs(gr - nominal) <= half)
    pk, pr = find_peaks(s[m], prominence=0)
    if not len(pk):
        return None
    i = m[pk[np.argmax(pr['prominences'])]]
    a, b, c = s[i - 1], s[i], s[i + 1]
    return gr[i] + 0.5 * (a - c) / (a - 2 * b + c) * (gr[1] - gr[0])


def segment_label(ax, xpos, ybase, segments):
    """Vertical label made of coloured text segments stacked upwards."""
    yy = ybase
    gap = 0.004 * (ax.get_ylim()[1] - ax.get_ylim()[0])
    for txt, col in segments:
        t = ax.text(xpos, yy, txt, rotation=90, va='bottom', ha='center', fontsize=10, color=col)
        ax.figure.canvas.draw()
        yy = t.get_window_extent().transformed(ax.transData.inverted()).y1 - gap


plt.rcParams.update({'font.size': 10, 'axes.spines.top': True, 'axes.spines.right': True})
g, X = WINDOWS['ch']['g'], data['ch']
fig, ax = plt.subplots(figsize=(6.5, 6.0))
rows, labels, lines = [], [], []
for nominal, half, kind, cls, assign in CH_BANDS:      # consistent EV/NV differences
    pos = {c: band_position(g, X[y == c].mean(0), nominal, half, kind) for c in (0, 1)}
    site_pos = {c: [band_position(g, X[sites == k].mean(0), nominal, half, kind)
                    for k in sorted(set(sites[y == c]))] for c in (0, 1)}
    resolved = None not in list(pos.values()) + site_pos[0] + site_pos[1]
    consistent = resolved and ((max(site_pos[0]) < min(site_pos[1])) or
                               (max(site_pos[1]) < min(site_pos[0])))
    if consistent and nominal in CH_SHOW:
        ax.axvspan(min(pos.values()) - 4, max(pos.values()) + 4, color='0.92', lw=0, zorder=0)
    for c in (0, 1):
        sp = [v for v in site_pos[c] if v is not None]
        rows.append([GROUPS[c], nominal, 'maximum' if kind == 'max' else 'shoulder (2nd derivative)',
                     'not resolved' if pos[c] is None else round(pos[c], 1),
                     f'{min(sp):.1f}-{max(sp):.1f} ({len(sp)} of 6 sites)' if sp else 'not resolved',
                     round(pos[1] - pos[0], 1) if resolved else '', consistent, cls, assign])
for c, off, name in [(0, 0.0, 'EV Acacia on Ag/SiNWs'), (1, CH_OFFSET, 'Nanovesicles Acacia on Ag/SiNWs')]:
    mu, sd = X[y == c].mean(0), X[y == c].std(0, ddof=1)
    ax.fill_between(g, mu - sd + off, mu + sd + off, color=COLS[c], alpha=0.22, lw=0)
    ax.plot(g, mu + off, color=COLS[c], lw=1.0, label=f'{name} (mean \u00b1 SD, n = {np.sum(y == c)})')
    for r in rows:
        if r[0] != GROUPS[c] or r[3] == 'not resolved' or r[1] not in CH_SHOW:
            continue
        pos = r[3]
        top = (mu + sd)[np.abs(g - pos) <= 6].max() + off
        txt = f'{pos:.0f}' + (' (sh)' if r[2] != 'maximum' else '')
        lines.append((pos, np.interp(pos, g, mu) + off, top + 0.08))
        labels.append((pos, top + 0.12, [(txt + '- ', 'k')] + CLASS_SEGMENTS[r[7]]))
for xl, yb, yt in lines:                    # dotted line from the mean spectrum to its label
    ax.plot([xl, xl], [yb, yt], ls=(0, (1, 2)), lw=0.7, color='0.35')
ax.set_xlim(2760, 3095)
ax.set_ylim((X[y == 0].mean(0) - X[y == 0].std(0, ddof=1)).min() - 0.3,
            (X[y == 1].mean(0) + X[y == 1].std(0, ddof=1)).max() + CH_OFFSET + 4.4)
for lab in labels:                          # after the axis limits are fixed
    segment_label(ax, *lab)
ax.tick_params(axis='y', direction='in', length=5, labelleft=False)
ax.tick_params(axis='x', direction='in', length=5)
ax.set_xlabel('Raman Shift (cm$^{-1}$)', fontsize=13)
ax.set_ylabel('Intensity (a.u.)', fontsize=13)
ax.legend(frameon=True, facecolor='white', edgecolor='none', framealpha=1, fontsize=9,
          loc='upper left', handlelength=1.5)
fig.tight_layout()
fig.savefig(OUTPUT_DIR / 'Fig_MeanSD_CH.png', dpi=600)
fig.savefig(OUTPUT_DIR / 'Fig_MeanSD_CH.tiff', dpi=600, pil_kwargs={'compression': 'tiff_lzw'})
with open(OUTPUT_DIR / 'MeanSD_CH_bands.csv', 'w', newline='') as fh:
    wr = csv.writer(fh)
    wr.writerow(['Group', 'Nominal (cm-1)', 'Method', 'Position (cm-1)', '6 sites (min-max)',
                 'Shift NV-EV (cm-1)', 'Consistent (shadowed)', 'Class', 'Assignment'])
    wr.writerows(rows)
for r in rows:
    print(r)
