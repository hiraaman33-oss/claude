# =============================================================================
# ACACIA EVs vs NVs  |  SERS on Ag/SiNWs  |  complete analysis (one file)
#
# Raw data (one sheet per measurement site m1-m6, 20 raw spectra per site):
#   CODE 005/EV Acacia/EV_Acacia_Raman_Data.xlsx
#   CODE 005/Nano vesicles Acacia/NV_Acacia_Raman_Data.xlsx
#
# PRE-PROCESSING (manuscript, "Data analysis"), per spectral window
#   fingerprint 200-1800 cm-1, CH-stretching 2700-3100 cm-1:
#   1. cosmic-ray removal: modified z-score on first differences, threshold 7
#   2. interpolation onto a 2 cm-1 grid (205-1795 and 2705-3095 cm-1)
#   3. Savitzky-Golay smoothing, 11 points, 3rd order
#   4. asymmetric least squares baseline, lambda = 1e5, p = 0.01, 10 iterations
#   5. SNV normalisation within the window
#
# MODELS, for each analysed window
#   Fingerprint (250-1790), CH-stretch (2760-3095), Whole range (both windows),
#   220-240 cm-1 band:
#   - PCA (5 PCs, pooled spectra) + LDA                         = PCA-LDA
#   - PLS-DA, 5 latent variables, autoscaled, threshold 0.5     = PLS-DA
#   - stratified 5-fold cross-validation grouped by measurement site
#     (StratifiedGroupKFold, shuffle=True, random_state=0)
#
# OUTPUTS (folder RESULTS inside CODE 005)
#   accuracy_summary.csv / R1_Accuracy_Summary.png
#   per window: confusion matrices, LD1 histogram, PCA scores, PCA loadings
#   Table_S1_bands.csv (Welch's t-test on site means, Bonferroni)
#   Fig3_Acacia_MeanSpectra.png/.tiff, Fig4_Acacia_PCA_LDA.png/.tiff (600 dpi)
#   (Fig. 4: (a,d) PCA scores, (b,e) LD1 box plots, (c,f) mean +/- SD spectra with
#    zoomed insets of the peak shifts measured from the data)
#   results_log.txt (every number used)
#
# HOW TO RUN (PyCharm Terminal):
#   pip install numpy scipy pandas openpyxl scikit-learn matplotlib
#   python Acacia_EV_NV_Analysis.py
# (or pass another CODE 005 folder:  python Acacia_EV_NV_Analysis.py "D:\path\CODE 005")
# =============================================================================

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from scipy import sparse
from scipy.sparse.linalg import spsolve
from scipy.signal import savgol_filter
from scipy.stats import ttest_ind
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.cross_decomposition import PLSRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import confusion_matrix

warnings.filterwarnings('ignore')

# ----------------------------------------------------------------------------
# PATHS
# ----------------------------------------------------------------------------
BASE_DIR = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(r'C:\Users\Hira Aman\Downloads\CODE 005')
EV_FILE = BASE_DIR / 'EV Acacia' / 'EV_Acacia_Raman_Data.xlsx'
NV_FILE = BASE_DIR / 'Nano vesicles Acacia' / 'NV_Acacia_Raman_Data.xlsx'
OUTPUT_DIR = BASE_DIR / 'RESULTS'
OUTPUT_DIR.mkdir(exist_ok=True)

# ----------------------------------------------------------------------------
# SETTINGS
# ----------------------------------------------------------------------------
SITES = ['m1', 'm2', 'm3', 'm4', 'm5', 'm6']
NAMES = {0: 'EV Acacia', 1: 'NV Acacia'}
C_EV, C_NV = '#0B5A7E', '#B5541C'
COLS = {0: C_EV, 1: C_NV}
N_COMP = 5
CV_SEED = 0

WINDOWS = {
    'fingerprint': {'raw': (200, 1800), 'grid': (205, 1795)},
    'ch':          {'raw': (2700, 3100), 'grid': (2705, 3095)},
}
# analysed windows: (label, [(pre-processing window, lo, hi), ...])
ANALYSES = {
    'Fingerprint': ('Fingerprint (250-1790 cm-1)', [('fingerprint', 250, 1790)]),
    'CH': ('CH-stretch (2760-3095 cm-1)', [('ch', 2760, 3095)]),
    'Whole': ('Whole range - all data (200-3200 cm-1)',
              [('fingerprint', 205, 1795), ('ch', 2705, 3095)]),
    'Band220': ('220-240 cm-1 band', [('fingerprint', 220, 240)]),
}
TABLE_S1 = [  # (band, lo, hi, window)
    ('519', 510, 530, 'fingerprint'), ('826', 820, 835, 'fingerprint'),
    ('855', 850, 862, 'fingerprint'), ('1001', 995, 1008, 'fingerprint'),
    ('1031', 1025, 1040, 'fingerprint'), ('1160', 1150, 1175, 'fingerprint'),
    ('1245', 1235, 1255, 'fingerprint'), ('1380', 1340, 1420, 'fingerprint'),
    ('1520', 1505, 1535, 'fingerprint'), ('1580', 1570, 1590, 'fingerprint'),
    ('1605', 1598, 1615, 'fingerprint'), ('1725', 1710, 1735, 'fingerprint'),
    ('2850', 2840, 2860, 'ch'), ('2880', 2870, 2890, 'ch'), ('2930', 2920, 2940, 'ch'),
    ('2960', 2950, 2970, 'ch'), ('3060', 3050, 3075, 'ch'),
]

plt.rcParams.update({'font.family': 'sans-serif',
                     'font.sans-serif': ['Arial', 'Liberation Sans', 'DejaVu Sans'],
                     'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False})
LOG = []


def log(*a):
    s = ' '.join(str(x) for x in a)
    print(s)
    LOG.append(s)


def save(fig, name, paper=False):
    """Analysis plots: 300 dpi PNG. Paper figures: fixed size, 600 dpi PNG + TIFF."""
    if paper:
        fig.savefig(OUTPUT_DIR / f'{name}.png', dpi=600)
        fig.savefig(OUTPUT_DIR / f'{name}.tiff', dpi=600, pil_kwargs={'compression': 'tiff_lzw'})
    else:
        fig.savefig(OUTPUT_DIR / f'{name}.png', dpi=300, bbox_inches='tight')
    plt.close(fig)


# ============================================================================
# 1. LOAD + PRE-PROCESS
# ============================================================================
def load_workbook(path):
    out = []
    for site in SITES:
        df = pd.read_excel(path, sheet_name=site, header=2)
        out.append((site, df.iloc[:, 0].to_numpy(float), df.iloc[:, 1:].to_numpy(float).T))
    return out


def despike(y, threshold=7.0, m=3):
    """Whitaker & Hayes modified z-score on first differences."""
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
    """Asymmetric least squares baseline (Eilers & Boelens)."""
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
    snv = (corr - corr.mean(1, keepdims=True)) / corr.std(1, keepdims=True)
    return grid, corr, snv


log('=' * 70)
log('ACACIA EVs vs NVs - analysis log')
log('EV file:', EV_FILE)
log('NV file:', NV_FILE)
log('=' * 70)

data = {w: {'corr': [], 'snv': []} for w in WINDOWS}
labels, sites = [], []
for cls, path in enumerate([EV_FILE, NV_FILE]):
    for site, x, Y in load_workbook(path):
        for w in WINDOWS:
            grid, corr, snv = preprocess(x, Y, w)
            data[w]['grid'] = grid
            data[w]['corr'].append(corr)
            data[w]['snv'].append(snv)
        labels += [cls] * len(Y)
        sites += [f'{NAMES[cls]}-{site}'] * len(Y)
for w in WINDOWS:
    data[w]['corr'] = np.vstack(data[w]['corr'])
    data[w]['snv'] = np.vstack(data[w]['snv'])
y = np.array(labels)
sites = np.array(sites)
log(f'Spectra: {len(y)} (EV {np.sum(y == 0)}, NV {np.sum(y == 1)}), sites: {len(set(sites))}')


def get_matrix(key):
    """Wavenumbers and SNV matrix of an analysed window (concatenated if needed)."""
    gs, xs = [], []
    for w, lo, hi in ANALYSES[key][1]:
        g = data[w]['grid']
        m = (g >= lo) & (g <= hi)
        gs.append(g[m])
        xs.append(data[w]['snv'][:, m])
    return np.concatenate(gs), np.hstack(xs)


# ============================================================================
# 2. MODELS
# ============================================================================
folds = list(StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=CV_SEED)
             .split(np.zeros(len(y)), y, sites))


def cv_pca_lda(X):
    n = min(N_COMP, X.shape[1])
    T = PCA(n_components=n).fit_transform(X)
    pred = np.empty(len(y), int)
    for tr, te in folds:
        pred[te] = LinearDiscriminantAnalysis().fit(T[tr], y[tr]).predict(T[te])
    return pred


def cv_plsda(X):
    n = min(N_COMP, X.shape[1])
    pred = np.empty(len(y), int)
    for tr, te in folds:
        sc = StandardScaler().fit(X[tr])
        pls = PLSRegression(n_components=n, scale=False).fit(sc.transform(X[tr]), y[tr])
        pred[te] = (pls.predict(sc.transform(X[te])).ravel() > 0.5).astype(int)
    return pred


def site_means(v):
    return {c: np.array([v[sites == s].mean() for s in sorted(set(sites[y == c]))]) for c in (0, 1)}


def plot_confusion(cm, title, name):
    fig, ax = plt.subplots(figsize=(4.6, 4.2))
    ax.imshow(cm, cmap='Blues')
    for i in range(2):
        for j in range(2):
            ax.text(j, i, cm[i, j], ha='center', va='center', fontsize=16, fontweight='bold',
                    color='white' if cm[i, j] > cm.max() / 2 else 'black')
    ax.set_xticks([0, 1])
    ax.set_xticklabels([NAMES[0], NAMES[1]], rotation=20, ha='right')
    ax.set_yticks([0, 1])
    ax.set_yticklabels([NAMES[0], NAMES[1]])
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    ax.set_title(title, fontweight='bold')
    save(fig, name)


acc_rows = []
for key, (label, _) in ANALYSES.items():
    g, X = get_matrix(key)
    log(f'\n--- {label}  ({X.shape[1]} variables) ---')
    pca = PCA(n_components=min(N_COMP, X.shape[1])).fit(X)
    v1, v2 = pca.explained_variance_ratio_[:2] * 100
    log(f'PCA: PC1 {v1:.1f} %, PC2 {v2:.1f} %')
    for model, pred in [('PCA+LDA', cv_pca_lda(X)), ('PLS-DA', cv_plsda(X))]:
        acc = (pred == y).mean() * 100
        cm = confusion_matrix(y, pred)
        acc_rows.append({'Window': label, 'Model': model, 'Accuracy (%)': round(acc, 1)})
        log(f'{model}: {acc:.1f} %   EV correct {cm[0, 0]}/120, NV correct {cm[1, 1]}/120')
        tag = 'PCA-LDA' if model == 'PCA+LDA' else 'PLS-DA'
        plot_confusion(cm, f'{tag} confusion matrix - {label}\n({acc:.1f}% accuracy)',
                       f'{key}_ConfusionMatrix_{tag}')
        if model == 'PCA+LDA':
            acc_lda = acc

    # LD1 histogram (PCA + LDA fitted on all spectra, descriptive)
    T = pca.transform(X)
    ld = LinearDiscriminantAnalysis().fit(T, y).transform(T).ravel()
    ld *= np.sign(ld[y == 1].mean())
    fig, ax = plt.subplots(figsize=(7, 4.6))
    bins = np.histogram_bin_edges(ld, 24)
    for c in (0, 1):
        ax.hist(ld[y == c], bins=bins, color=COLS[c], alpha=0.6, ec='white', label=NAMES[c])
    ax.set_xlabel('LD1 score')
    ax.set_ylabel('Count')
    ax.grid(ls='--', alpha=0.3)
    ax.legend(frameon=False)
    ax.set_title(f'LDA discriminant score - {label} ({acc_lda:.1f}% CV accuracy)', fontweight='bold')
    save(fig, f'{key}_LDA_DiscriminantScore')

    # PCA scores + loadings
    fig, ax = plt.subplots(figsize=(6, 5))
    for c in (0, 1):
        P = T[y == c, :2]
        ax.scatter(P[:, 0], P[:, 1], s=18, color=COLS[c], alpha=0.8, ec='white', lw=0.4,
                   label=NAMES[c])
        vals, vecs = np.linalg.eigh(np.cov(P.T))
        ang = np.degrees(np.arctan2(vecs[1, 1], vecs[0, 1]))
        ax.add_patch(Ellipse(P.mean(0), 4 * np.sqrt(vals[1]), 4 * np.sqrt(vals[0]), angle=ang,
                             fc=COLS[c], ec=COLS[c], alpha=0.15))
    ax.set_xlabel(f'PC1 ({v1:.1f}% variance)')
    ax.set_ylabel(f'PC2 ({v2:.1f}% variance)')
    ax.grid(ls='--', alpha=0.3)
    ax.legend(frameon=False)
    ax.set_title(f'PCA scores - {label}', fontweight='bold')
    save(fig, f'{key}_PCA_Scores')

    fig, ax = plt.subplots(figsize=(9, 4.6))
    xs = np.arange(len(g)) if key == 'Whole' else g      # Whole: two windows side by side
    ax.plot(xs, pca.components_[0], color=C_EV, lw=1.4, label=f'PC1 ({v1:.1f}% variance)')
    ax.plot(xs, pca.components_[1], color=C_NV, lw=1.4, label=f'PC2 ({v2:.1f}% variance)')
    ax.axhline(0, color='grey', lw=0.8)
    if key == 'Whole':
        cut = np.argmax(np.diff(g) > 10) + 1
        ax.axvline(cut - 0.5, color='k', ls=':', lw=0.8)
        ticks = [i for i, v in enumerate(g) if v in (400, 800, 1200, 1600, 2800, 3000)]
        ax.set_xticks(ticks)
        ax.set_xticklabels([int(g[i]) for i in ticks])
    ax.set_xlabel('Raman shift (cm$^{-1}$)')
    ax.set_ylabel('Loading (a.u.)')
    ax.grid(ls='--', alpha=0.3)
    ax.legend(frameon=False)
    ax.set_title(f'PCA loadings - {label}', fontweight='bold')
    save(fig, f'{key}_PCA_Loadings')

acc_df = pd.DataFrame(acc_rows)
acc_df.to_csv(OUTPUT_DIR / 'accuracy_summary.csv', index=False)

# accuracy bar chart
fig, ax = plt.subplots(figsize=(10, 5.5))
labels_w = [ANALYSES[k][0] for k in ANALYSES]
xpos = np.arange(len(labels_w))
for k, (model, col) in enumerate([('PCA+LDA', C_EV), ('PLS-DA', C_NV)]):
    vals = [acc_df[(acc_df.Window == lw) & (acc_df.Model == model)]['Accuracy (%)'].iloc[0]
            for lw in labels_w]
    bars = ax.bar(xpos + (k - 0.5) * 0.35, vals, 0.35, color=col, label=model.replace('+', ' + '))
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 1, f'{v:.1f}%', ha='center', fontsize=9)
ax.axhline(50, color='grey', ls=':', lw=1)
ax.text(len(labels_w) - 0.5, 51.5, 'chance level', color='grey', ha='right', fontsize=8)
ax.set_xticks(xpos)
ax.set_xticklabels([lw.replace(' (', '\n(') for lw in labels_w], fontsize=9)
ax.set_ylabel('Site-grouped cross-validated accuracy (%)')
ax.set_ylim(0, 110)
ax.grid(axis='y', ls='--', alpha=0.3)
ax.legend(frameon=False, loc='lower right')
ax.set_title('Classification accuracy by spectral window and model', fontweight='bold')
save(fig, 'R1_Accuracy_Summary')

# ============================================================================
# 3. BAND STATISTICS (Table S1) + I2960/I2930
# ============================================================================
rows = []
for band, lo, hi, w in TABLE_S1:
    g = data[w]['grid']
    v = data[w]['snv'][:, (g >= lo) & (g <= hi)].mean(1)
    sm = site_means(v)
    p = ttest_ind(sm[0], sm[1], equal_var=False).pvalue
    rows.append({'Band (cm-1)': band, 'Interval': f'{lo}-{hi}', 'window': w,
                 'EV mean': sm[0].mean(), 'EV SD': sm[0].std(ddof=1),
                 'NV mean': sm[1].mean(), 'NV SD': sm[1].std(ddof=1),
                 'p (Welch)': p, 'p (Bonferroni)': min(1.0, p * len(TABLE_S1))})
bands = pd.DataFrame(rows)
bands.round(4).to_csv(OUTPUT_DIR / 'Table_S1_bands.csv', index=False)
log('\n--- Table S1 (site means, n = 6; Welch; Bonferroni x17) ---')
for _, b in bands.iterrows():
    log(f"{b['Band (cm-1)']:>5} ({b['Interval']})  EV {b['EV mean']:6.2f} +/- {b['EV SD']:.2f}   "
        f"NV {b['NV mean']:6.2f} +/- {b['NV SD']:.2f}   p = {b['p (Bonferroni)']:.3g}")
g, C = data['ch']['grid'], data['ch']['corr']
ratio = C[:, (g >= 2950) & (g <= 2970)].mean(1) / C[:, (g >= 2920) & (g <= 2940)].mean(1)
rm = site_means(ratio)
log(f'I2960/I2930: EV {rm[0].mean():.2f} +/- {rm[0].std(ddof=1):.2f}, '
    f'NV {rm[1].mean():.2f} +/- {rm[1].std(ddof=1):.2f}')


# ============================================================================
# 4. PAPER FIGURES 3 AND 4 (layout of Roda et al., Nanomaterials 2023, Fig. 3)
# ============================================================================
def spread(pos, gap):
    out = list(pos)
    for _ in range(100):
        for i in range(1, len(out)):
            if out[i] - out[i - 1] < gap:
                mid = (out[i] + out[i - 1]) / 2
                out[i - 1], out[i] = mid - gap / 2, mid + gap / 2
    return out


def region(w, lo, hi):
    g = data[w]['grid']
    m = (g >= lo) & (g <= hi)
    return g[m], data[w]['snv'][:, m]


sig = bands[bands['p (Bonferroni)'] < 0.05]
REG = {'fingerprint': (250, 1790), 'ch': (2760, 3095)}
plt.rcParams.update({'font.size': 9, 'axes.spines.top': True, 'axes.spines.right': True})

# --- Figure 3: stacked means +/- SD, subtraction spectrum, significant bands
fig = plt.figure(figsize=(7.0, 6.2))
gs = fig.add_gridspec(1, 2, width_ratios=[1790 - 250, (3095 - 2760) * 1.6], wspace=0.04)
axL, axR = fig.add_subplot(gs[0]), fig.add_subplot(gs[1])
means = {}
for w in REG:
    g, X = region(w, *REG[w])
    means[w] = {c: (X[y == c].mean(0), X[y == c].std(0)) for c in (0, 1)}
    means[w]['grid'] = g
span = max(np.ptp(means[w][c][0]) + 2 * means[w][c][1].max() for w in REG for c in (0, 1))
OFF = {0: 2.05 * span, 1: 1.10 * span, 'sub': 0.42 * span}
for ax, w in [(axL, 'fingerprint'), (axR, 'ch')]:
    g = means[w]['grid']
    for c in (0, 1):
        m, s = means[w][c]
        ax.fill_between(g, m - s + OFF[c], m + s + OFF[c], color=COLS[c], alpha=0.18, lw=0)
        ax.plot(g, m + OFF[c], color=COLS[c], lw=0.9)
    ax.plot(g, means[w][0][0] - means[w][1][0] + OFF['sub'], color='#7F7F7F', lw=0.9)
    ax.axhline(OFF['sub'], color='k', lw=0.5)
    sw = sig[sig.window == w]
    gap = 34 if w == 'fingerprint' else 34 / 1.6
    y_band = OFF['sub'] - 0.30 * span
    for (_, b), lx in zip(sw.iterrows(), spread(list(sw['Band (cm-1)'].astype(float)), gap)):
        lo, hi = map(float, b['Interval'].split('-'))
        c = (lo + hi) / 2
        ax.axvspan(lo, hi, color='#D3D3D3', alpha=0.7, lw=0, zorder=0)
        ax.plot([c, c, lx], [y_band, y_band - 0.06 * span, y_band - 0.14 * span], color='0.35', lw=0.5)
        ax.text(lx, y_band - 0.17 * span, f"{b['Band (cm-1)']} cm$^{{-1}}$",
                rotation=90, ha='center', va='top', fontsize=7.5)
axL.set_xlim(250, 1790)
axR.set_xlim(2760, 3095)
for ax in (axL, axR):
    ax.set_ylim(-0.62 * span, OFF[0] + 1.0 * span)
    ax.set_yticks([])
    for s_ in ['top', 'right']:
        ax.spines[s_].set_visible(False)
axR.spines['left'].set_visible(False)
axL.set_xticks([500, 1000, 1500])
axR.set_xticks([2800, 3000])
axL.set_ylabel('Intensity (a.u.)', fontsize=10)
for key, txt in [(0, 'Acacia EVs'), (1, 'Acacia NVs'), ('sub', 'Subtraction\nspectrum\n(EVs - NVs)')]:
    axR.text(3110, OFF[key] + 0.15 * span, txt, fontsize=8.5, va='center', ha='left', clip_on=False)
d = 0.015
k = (3095 - 2760) * 1.6 / (1790 - 250)
axL.plot((1 - d * k, 1 + d * k), (-d * 1.5, d * 1.5), transform=axL.transAxes, color='k', clip_on=False, lw=0.8)
axR.plot((-d, d), (-d * 1.5, d * 1.5), transform=axR.transAxes, color='k', clip_on=False, lw=0.8)
fig.text(0.47, 0.025, 'Raman shift (cm$^{-1}$)', ha='center', fontsize=10)
fig.subplots_adjust(left=0.06, right=0.82, bottom=0.10, top=0.98)
save(fig, 'Fig3_Acacia_MeanSpectra', paper=True)


# --- Figure 4: scores, LD1 box plots, mean +/- SD spectra with peak shifts
def stars(p):
    return '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'n.s.'


SPEC_RANGE = {'fingerprint': (400, 1790), 'ch': (2760, 3095)}   # x-range of panels (c), (f)
# bands checked for a peak shift: (search lo, hi, inset lo, hi, inset box in axes fraction)
SHIFT_BANDS = {'fingerprint': [(1340, 1420, 1330, 1430, [0.05, 0.50, 0.25, 0.36]),
                               (1590, 1625, 1585, 1625, [0.38, 0.50, 0.25, 0.36])],
               'ch': [(2900, 2960, 2905, 2965, [0.06, 0.50, 0.30, 0.36])]}


def peak_position(g, s, lo, hi):
    """Wavenumber of the maximum in [lo, hi], refined by a parabola through 3 points."""
    m = (g >= lo) & (g <= hi)
    gg, ss = g[m], s[m]
    i = int(np.argmax(ss))
    if 0 < i < len(ss) - 1:
        a, b, c = ss[i - 1], ss[i], ss[i + 1]
        return gg[i] + 0.5 * (a - c) / (a - 2 * b + c) * (gg[1] - gg[0])
    return gg[i]


def band_shift(w, lo, hi):
    """EV and NV peak positions of the mean spectra, and whether the six site-mean
    peak positions of EVs and NVs are fully separated (consistent shift)."""
    g, X = region(w, *REG[w])
    pe = peak_position(g, X[y == 0].mean(0), lo, hi)
    pn = peak_position(g, X[y == 1].mean(0), lo, hi)
    se = [peak_position(g, X[sites == s].mean(0), lo, hi) for s in sorted(set(sites[y == 0]))]
    sn = [peak_position(g, X[sites == s].mean(0), lo, hi) for s in sorted(set(sites[y == 1]))]
    return pe, pn, max(se) < min(sn) or max(sn) < min(se)


fig, axs = plt.subplots(2, 3, figsize=(7.5, 5.3),
                        gridspec_kw={'width_ratios': [1.1, 0.7, 1.6], 'wspace': 0.42, 'hspace': 0.45})
letters = iter('abcdef')
for row, (w, title, ref) in enumerate([('fingerprint', 'Fingerprint (250-1790 cm$^{-1}$)', 1580),
                                       ('ch', 'CH-stretching (2760-3095 cm$^{-1}$)', 2960)]):
    g, X = region(w, *REG[w])
    pca = PCA(n_components=N_COMP).fit(X)
    T = pca.transform(X)
    sgn = np.sign(pca.components_[0][np.argmin(np.abs(g - ref))])   # orientation as in the text
    T[:, 0] *= sgn
    ev1, ev2 = pca.explained_variance_ratio_[:2] * 100
    ax = axs[row, 0]
    for c in (0, 1):
        P = T[y == c, :2]
        ax.scatter(P[:, 0], P[:, 1], s=6, color=COLS[c], lw=0, alpha=0.85, label=NAMES[c].split()[0] + 's')
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
    ax.set_title(title, fontsize=8.5, loc='left')

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
    log(f'Figure 4 {w}: LD1 site means, Welch p = {p:.2e}')

    # --- mean +/- SD spectra (same SNV spectra as used in the models)
    ax = axs[row, 2]
    show = (g >= SPEC_RANGE[w][0]) & (g <= SPEC_RANGE[w][1])
    for c in (0, 1):
        m, s = X[y == c].mean(0), X[y == c].std(0)
        ax.fill_between(g[show], (m - s)[show], (m + s)[show], color=COLS[c], alpha=0.2, lw=0)
        ax.plot(g[show], m[show], color=COLS[c], lw=0.9, label=NAMES[c].split()[0] + 's')
    ax.set_xlim(*SPEC_RANGE[w])
    ax.set_xlabel('Raman shift (cm$^{-1}$)')
    ax.set_ylabel('SNV intensity (a.u.)')
    ax.legend(frameon=False, fontsize=7.5, loc='upper right', handlelength=1.2)
    # zoomed insets with the measured peak shift (arrow drawn at true scale)
    for lo, hi, zlo, zhi, box in SHIFT_BANDS[w]:
        pe, pn, consistent = band_shift(w, lo, hi)
        log(f'Peak shift {lo}-{hi}: EV {pe:.1f}, NV {pn:.1f}, shift {pn - pe:+.1f} cm-1, '
            f'site ranges separated: {consistent}')
        if not consistent:
            continue
        ins = ax.inset_axes(box)
        z = (g >= zlo) & (g <= zhi)
        tops = []
        for c in (0, 1):
            m, s = X[y == c].mean(0), X[y == c].std(0)
            ins.fill_between(g[z], (m - s)[z], (m + s)[z], color=COLS[c], alpha=0.2, lw=0)
            ins.plot(g[z], m[z], color=COLS[c], lw=0.9)
            tops.append((m + s)[z].max())
        ylo_i = min((X[y == c].mean(0) - X[y == c].std(0))[z].min() for c in (0, 1))
        ya = max(tops) + 0.10 * (max(tops) - ylo_i)
        for pk, c in [(pe, 0), (pn, 1)]:          # dotted lines at the measured maxima
            ins.vlines(pk, ylo_i, ya, color=COLS[c], ls=':', lw=0.8)
        ins.annotate('', xy=(pn, ya), xytext=(pe, ya),
                     arrowprops=dict(arrowstyle='->', color='k', lw=0.9, shrinkA=0, shrinkB=0,
                                     mutation_scale=7))
        ins.text((pe + pn) / 2, ya + 0.07 * (max(tops) - ylo_i), f'{pn - pe:+.0f} cm$^{{-1}}$',
                 ha='center', va='bottom', fontsize=6.5)
        ins.set_xlim(zlo, zhi)
        ins.set_ylim(ylo_i, ya + 0.32 * (max(tops) - ylo_i))
        ins.set_yticks([])
        ins.set_xticks([zlo + 10, zhi - 10])
        ins.tick_params(labelsize=6, length=2, pad=1)
        ins.set_title(f'{pe:.0f} \u2192 {pn:.0f}', fontsize=6.5, pad=2)
        for s_ in ['left', 'top', 'right']:      # no frame: only the wavenumber axis
            ins.spines[s_].set_visible(False)
        ax.add_patch(plt.Rectangle((zlo, ax.get_ylim()[0]), zhi - zlo,
                                   ax.get_ylim()[1] - ax.get_ylim()[0],
                                   color='0.92', lw=0, zorder=0))
    for ax in axs[row]:
        for s_ in ['top', 'right']:
            ax.spines[s_].set_visible(False)
        ax.text(-0.30 if ax is axs[row, 1] else -0.22, 1.10, f'({next(letters)})',
                transform=ax.transAxes, fontsize=10, va='bottom')
fig.subplots_adjust(left=0.08, right=0.99, bottom=0.09, top=0.93)
save(fig, 'Fig4_Acacia_PCA_LDA', paper=True)

log(f'\nCross-validation: StratifiedGroupKFold(5, shuffle=True, random_state={CV_SEED}), grouped by site.')
log(f'All outputs saved in: {OUTPUT_DIR}')
(OUTPUT_DIR / 'results_log.txt').write_text('\n'.join(LOG) + '\n', encoding='utf-8')
