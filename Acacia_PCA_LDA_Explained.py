# =============================================================================
# ACACIA EVs vs NVs  |  SERS on Ag/SiNWs  |  PCA + LDA figure, step by step
#
# Makes ONE figure with four large panels (panels c and f of the old Figure 4,
# the mean spectra, are removed):
#   (a) PCA score plot, fingerprint region      (250-1790 cm-1)
#   (b) LDA canonical variable 1, fingerprint region
#   (c) PCA score plot, CH-stretching region    (2760-3095 cm-1)
#   (d) LDA canonical variable 1, CH-stretching region
#
# Everything is calculated from your original data:
#   CODE 005\EV Acacia\EV_Acacia_Raman_Data.xlsx              (120 spectra)
#   CODE 005\Nano vesicles Acacia\NV_Acacia_Raman_Data.xlsx   (120 spectra)
# Each workbook has 6 sheets (m1-m6 = 6 measurement sites), 20 spectra per sheet.
#
# The script is written in numbered steps; each step prints what it did, so you
# can follow it in the PyCharm "Run" window. The same calculations are used in
# Acacia_EV_NV_Analysis.py, so the numbers are identical to Figure 4.
#
# HOW TO RUN (PyCharm):
#   1. pip install numpy pandas scipy matplotlib openpyxl scikit-learn
#   2. check BASE_DIR below, then press Run
# OUTPUT (in CODE 005\RESULTS):
#   Fig_PCA_LDA_large.png / .tiff        the figure (600 dpi)
#   PCA_LDA_scores_fingerprint.csv       PC1-PC5 and LD1 score of every spectrum
#   PCA_LDA_scores_CH.csv                (open in Excel to check any point)
#   PCA_LDA_summary.txt                  all numbers printed by the script
# =============================================================================

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from scipy import sparse
from scipy.sparse.linalg import spsolve
from scipy.signal import savgol_filter
from scipy.stats import ttest_ind
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

# ----------------------------------------------------------------------------
# SETTINGS
# ----------------------------------------------------------------------------
BASE_DIR = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(r'C:\Users\Hira Aman\Downloads\CODE 005')
EV_FILE = BASE_DIR / 'EV Acacia' / 'EV_Acacia_Raman_Data.xlsx'
NV_FILE = BASE_DIR / 'Nano vesicles Acacia' / 'NV_Acacia_Raman_Data.xlsx'
OUTPUT_DIR = BASE_DIR / 'RESULTS'
OUTPUT_DIR.mkdir(exist_ok=True)
SHOW_WINDOW = len(sys.argv) <= 1     # open the figure in PyCharm when run without arguments
if not SHOW_WINDOW:
    matplotlib.use('Agg')

SITES = ['m1', 'm2', 'm3', 'm4', 'm5', 'm6']
GROUPS = {0: 'EVs', 1: 'NVs'}               # class 0 = EVs, class 1 = NVs
COLS = {0: '#0B5A7E', 1: '#B5541C'}         # same colours as Figure 4
N_COMP = 5                                  # number of principal components given to LDA
# spectral windows: raw range read, 2 cm-1 grid, region used in PCA
WINDOWS = {'fingerprint': {'raw': (200, 1800), 'grid': (205, 1795), 'region': (250, 1790)},
           'ch': {'raw': (2700, 3100), 'grid': (2705, 3095), 'region': (2760, 3095)}}
# PCA does not fix the sign of a component; PC1 is turned so that its loading is
# positive at this band (only flips the plot left-right, values are unchanged)
PC1_REF = {'fingerprint': 1580, 'ch': 2960}
TITLES = {'fingerprint': 'Fingerprint (250-1790 cm$^{-1}$)',
          'ch': 'CH-stretching (2760-3095 cm$^{-1}$)'}
FILE_TAG = {'fingerprint': 'fingerprint', 'ch': 'CH'}

plt.rcParams.update({'font.family': 'sans-serif',
                     'font.sans-serif': ['Arial', 'Liberation Sans', 'DejaVu Sans'],
                     'font.size': 12, 'axes.spines.top': False, 'axes.spines.right': False})
SUMMARY = []


def report(*a):
    s = ' '.join(str(v) for v in a)
    print(s)
    SUMMARY.append(s)


# ============================================================================
# STEP 1 - read the raw spectra and pre-process them
# ============================================================================
# For every spectrum, separately in each window:
#   a) cosmic-ray spikes removed (modified z-score of the point-to-point
#      differences > 7 -> replaced by the mean of the neighbouring points)
#   b) interpolated onto a 2 cm-1 grid (same x values for all spectra)
#   c) Savitzky-Golay smoothing, 11 points, 3rd-order polynomial
#   d) background removed: asymmetric least squares baseline (lambda 1e5, p 0.01)
#   e) SNV: (spectrum - its mean) / its standard deviation, so every spectrum has
#      mean 0 and SD 1 and differences in total signal strength are removed
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


report('STEP 1 - reading and pre-processing the spectra')
data = {w: [] for w in WINDOWS}
y, sites, names = [], [], []
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
        names += [f'{GROUPS[cls]}-{site}-{c}' for c in df.columns[1:]]
        report(f'  {path.name:28s} sheet {site}: {len(Y)} spectra')
data = {w: np.vstack(v) for w, v in data.items()}
y, sites = np.array(y), np.array(sites)
report(f'  total: {len(y)} spectra (EVs {np.sum(y == 0)}, NVs {np.sum(y == 1)}), '
       f'{len(set(sites))} measurement sites')
for w in WINDOWS:
    report(f'  {w}: {data[w].shape[1]} wavenumber points '
           f'({WINDOWS[w]["g"][0]:.0f}-{WINDOWS[w]["g"][-1]:.0f} cm-1, every 2 cm-1)')


def site_means(v):
    """Mean of a value over the 20 spectra of each site -> 6 numbers per group."""
    return {c: np.array([v[sites == s].mean() for s in sorted(set(sites[y == c]))]) for c in (0, 1)}


def stars(p):
    return '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'n.s.'


# ============================================================================
# STEPS 2-4 for each region, then the figure
# ============================================================================
fig, axs = plt.subplots(2, 2, figsize=(10, 9),
                        gridspec_kw={'width_ratios': [1.35, 1], 'wspace': 0.32, 'hspace': 0.42})
letters = iter('abcd')
for row, w in enumerate(['fingerprint', 'ch']):
    X, g = data[w], WINDOWS[w]['g']
    report(f'\n========== {TITLES[w].replace("$^{-1}$", "-1")} ==========')

    # ------------------------------------------------------------------------
    # STEP 2 - PCA (panels a and c)
    # ------------------------------------------------------------------------
    # PCA is fitted on all 240 SNV spectra together, WITHOUT the EV/NV labels.
    # It finds the directions (principal components) along which the spectra
    # differ most. Each spectrum gets a score on each PC; the score plot shows
    # PC1 against PC2. The % on each axis is the share of the total variance
    # of the spectra explained by that PC.
    pca = PCA(n_components=N_COMP, svd_solver='full').fit(X)
    T = pca.transform(X)                                     # scores, 240 x 5
    sign = np.sign(pca.components_[0][np.argmin(np.abs(g - PC1_REF[w]))])
    T[:, 0] *= sign
    var = pca.explained_variance_ratio_ * 100
    report('STEP 2 - PCA on all 240 spectra (labels not used)')
    report('  explained variance: ' + ', '.join(f'PC{i + 1} {v:.1f} %' for i, v in enumerate(var))
           + f'  (PC1-PC{N_COMP} together {var.sum():.1f} %)')
    for c in (0, 1):
        report(f'  {GROUPS[c]}: mean PC1 score {T[y == c, 0].mean():+.2f}, '
               f'mean PC2 score {T[y == c, 1].mean():+.2f}')

    # ------------------------------------------------------------------------
    # STEP 3 - LDA on the 5 PC scores (panels b and d)
    # ------------------------------------------------------------------------
    # LDA uses the labels: it finds the one combination of PC1-PC5 that best
    # separates EVs from NVs (largest difference between the group means
    # relative to the spread within the groups). With two groups there is one
    # such axis, "canonical variable 1" (LD1). Each spectrum's LD1 value is a
    # dot in panel b/d. LDA is fitted here on all spectra, to DISPLAY the
    # separation; the classification accuracy of the paper comes from the
    # cross-validation in Acacia_EV_NV_Analysis.py (whole sites left out).
    lda = LinearDiscriminantAnalysis().fit(T, y)
    ld = lda.transform(T).ravel()
    ld *= np.sign(ld[y == 1].mean())                         # NVs on the positive side
    report('STEP 3 - LDA on PC1-PC5 (labels used), canonical variable 1 = LD1')
    for c in (0, 1):
        q1, med, q3 = np.percentile(ld[y == c], [25, 50, 75])
        report(f'  {GROUPS[c]}: LD1 median {med:+.2f} (quartiles {q1:+.2f} to {q3:+.2f}), '
               f'range {ld[y == c].min():+.2f} to {ld[y == c].max():+.2f}')

    # ------------------------------------------------------------------------
    # STEP 4 - test on site means (text in panels b and d)
    # ------------------------------------------------------------------------
    # The 20 spectra of one site are not independent, so the test uses one
    # number per site: the mean LD1 of its 20 spectra (6 EV sites, 6 NV sites).
    # Welch's t-test (no equal-variance assumption) compares the two sets of 6.
    sm = site_means(ld)
    p = ttest_ind(sm[0], sm[1], equal_var=False).pvalue
    report('STEP 4 - Welch t-test on the 6 site means of LD1')
    for c in (0, 1):
        report(f'  {GROUPS[c]} site means: ' + ', '.join(f'{v:+.2f}' for v in sm[c]))
    report(f'  p = {p:.2e}  ({stars(p)})')

    # save every score so any dot in the figure can be checked in Excel
    out = pd.DataFrame({'spectrum': names, 'group': [GROUPS[c] for c in y], 'site': sites})
    for i in range(N_COMP):
        out[f'PC{i + 1}'] = T[:, i]
    out['LD1'] = ld
    out.to_csv(OUTPUT_DIR / f'PCA_LDA_scores_{FILE_TAG[w]}.csv', index=False)

    # ------------------------------------------------------------------------
    # PANEL: PCA score plot with 2-SD ellipses
    # ------------------------------------------------------------------------
    # ellipse = 2 standard deviations of each group's PC1/PC2 scores
    # (axes from the eigenvectors/eigenvalues of the group's 2 x 2 covariance)
    ax = axs[row, 0]
    for c in (0, 1):
        P = T[y == c, :2]
        ax.scatter(P[:, 0], P[:, 1], s=14, color=COLS[c], lw=0, alpha=0.85, label=GROUPS[c])
        vals, vecs = np.linalg.eigh(np.cov(P.T))
        ang = np.degrees(np.arctan2(vecs[1, 1], vecs[0, 1]))
        ax.add_patch(Ellipse(P.mean(0), 4 * np.sqrt(vals[1]), 4 * np.sqrt(vals[0]), angle=ang,
                             fc='none', ec=COLS[c], lw=1.4))
    ax.axhline(0, color='k', lw=0.6)
    ax.axvline(0, color='k', lw=0.6)
    ax.set_xlabel(f'PC1 ({var[0]:.1f}%)')
    ax.set_ylabel(f'PC2 ({var[1]:.1f}%)')
    ylo, yhi = ax.get_ylim()
    ax.set_ylim(ylo, yhi + 0.28 * (yhi - ylo))
    ax.legend(frameon=False, fontsize=11, loc='upper left', ncol=2, handletextpad=0.1,
              columnspacing=1.0, markerscale=1.6)
    ax.set_title(TITLES[w], fontsize=12, loc='left')

    # ------------------------------------------------------------------------
    # PANEL: LD1 box plot
    # ------------------------------------------------------------------------
    # box = 25th-75th percentile, line = median, whiskers = most extreme value
    # within 1.5 x box height; dots = all 120 spectra of the group (spread
    # sideways at random only so they do not cover each other)
    ax = axs[row, 1]
    rng = np.random.default_rng(0)
    for c in (0, 1):
        v = ld[y == c]
        ax.boxplot(v, positions=[c], widths=0.5, showfliers=False, patch_artist=True,
                   boxprops=dict(fc=COLS[c], alpha=0.30, ec=COLS[c]),
                   medianprops=dict(color=COLS[c], lw=1.6),
                   whiskerprops=dict(color=COLS[c]), capprops=dict(color=COLS[c]))
        ax.scatter(c + rng.uniform(-0.15, 0.15, len(v)), v, s=8, color=COLS[c], lw=0, alpha=0.8)
    top = ld.max() + 0.08 * np.ptp(ld)
    ax.plot([0, 1], [top, top], color='k', lw=0.8)
    ax.text(0.5, top, stars(p), ha='center', va='bottom', fontsize=12)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(['EVs', 'NVs'])
    ax.set_ylabel('Canonical variable 1')
    ax.set_ylim(ld.min() - 0.20 * np.ptp(ld), top + 0.15 * np.ptp(ld))
    ax.text(0.5, 0.02, "Welch's t-test on site means\n(n = 6), " + stars(p) + f' p = {p:.1e}',
            transform=ax.transAxes, ha='center', va='bottom', fontsize=9)

    for ax in axs[row]:
        ax.text(-0.27 if ax is axs[row, 1] else -0.18, 1.08, f'({next(letters)})',
                transform=ax.transAxes, fontsize=14, va='bottom')

fig.subplots_adjust(left=0.09, right=0.98, bottom=0.07, top=0.94)
fig.savefig(OUTPUT_DIR / 'Fig_PCA_LDA_large.png', dpi=600)
fig.savefig(OUTPUT_DIR / 'Fig_PCA_LDA_large.tiff', dpi=600, pil_kwargs={'compression': 'tiff_lzw'})
(OUTPUT_DIR / 'PCA_LDA_summary.txt').write_text('\n'.join(SUMMARY) + '\n', encoding='utf-8')
print(f'\nSaved in: {OUTPUT_DIR}')
if SHOW_WINDOW:
    plt.show()
