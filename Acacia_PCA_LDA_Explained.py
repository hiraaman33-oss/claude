# =============================================================================
# ACACIA EVs vs NVs  |  SERS on Ag/SiNWs  |  PCA + LDA figure, step by step
#
# Makes ONE figure with four large panels (panels c and f of the old Figure 4,
# the mean spectra, are removed):
#   (a) PCA score plot, fingerprint region      (250-1790 cm-1)
#   (b) LDA canonical variable 1, fingerprint region
#   (c) PCA score plot, CH-stretching region    (2760-3095 cm-1)
#   (d) LDA canonical variable 1, CH-stretching region
# and further figures/tables (all from the same data):
#   STEP 5  cross-validated confusion matrices (PCA-LDA and PLS-DA)
#   STEP 6  PCA loadings: which bands drive PC1 and PC2
#   STEP 7  difference spectrum NV - EV with its site-to-site uncertainty
#   STEP 8  exact label-shuffling (permutation) test over all 924 ways of
#           calling 6 of the 12 sites "EV"
#   STEP 9  RESULTS_for_manuscript.txt: the actual results written as text,
#           every number filled in from this run
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
#   Fig_ConfusionMatrices.png / .tiff    cross-validated confusion matrices (600 dpi)
#   CV_accuracy.csv                      accuracy, sensitivity, specificity
#   Fig_Loadings.png / .tiff             PC1 and PC2 loadings, both regions
#   Fig_Difference.png / .tiff           NV - EV difference spectra, both regions
#   Fig_Permutation.png / .tiff          shuffled-label accuracies vs real accuracy
#   Loadings_main_bands.csv, Difference_main_bands.csv, Permutation_test.csv
#   PCA_LDA_summary.txt                  all numbers printed by the script
#   RESULTS_for_manuscript.txt           the results in words (STEP 9)
# The permutation test (STEP 8) repeats the whole cross-validation 924 times per
# region and takes several minutes; set N_PERM_MODE = 'skip' below to leave it out.
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
from sklearn.cross_decomposition import PLSRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import confusion_matrix
from itertools import combinations

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
CV_SEED = 0                                 # random split of the sites into 5 folds
N_PERM_MODE = 'exact'                       # 'exact' = all 924 site labelings, 'skip' = no test
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


def save_figure(fig_, name):
    """Save PNG + TIFF (600 dpi). If Windows refuses to overwrite a file (usually
    because the old image is still open in Photos / PyCharm / another viewer),
    the figure is saved under a new name instead of stopping the script."""
    for ext, kw in [('png', {}), ('tiff', {'pil_kwargs': {'compression': 'tiff_lzw'}})]:
        target = OUTPUT_DIR / f'{name}.{ext}'
        for k in range(1, 50):
            try:
                fig_.savefig(target, dpi=600, **kw)
                break
            except OSError as err:
                print(f'  could not write {target.name} ({err.strerror}) - is it open in '
                      f'another program? Saving as a new file instead.')
                target = OUTPUT_DIR / f'{name}_{k}.{ext}'
        print(f'  saved {target}')


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
RES = {}                                    # results of every step, used in STEP 9
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
    # separation; the classification accuracy comes from the cross-validation
    # in STEP 5 (whole sites left out).
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
    load = pca.components_[:2].copy()
    load[0] *= sign                                          # same orientation as the scores
    RES[w] = {'var': var, 'p_ld1': p, 'ld_median': {c: np.median(ld[y == c]) for c in (0, 1)},
              'pc1_mean': {c: T[y == c, 0].mean() for c in (0, 1)}, 'loadings': load}

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
save_figure(fig, 'Fig_PCA_LDA_large')


# ============================================================================
# STEP 5 - cross-validated classification and confusion matrices
# ============================================================================
# Panels b and d only SHOW the separation (LDA fitted on all spectra). Whether
# a NEW spectrum can be classified correctly is tested by cross-validation:
#   - the 12 sites (6 EV, 6 NV) are split into 5 folds; all 20 spectra of a site
#     always stay together, so a site is never in training and test at once
#   - the model is trained on 4 folds and predicts the spectra of the 5th fold;
#     repeated 5 times, so every spectrum is predicted once, by a model that
#     never saw its site
#   PCA-LDA: PCA scores of all spectra (unsupervised, no labels), LDA trained on
#            the training folds only (as in Acacia_EV_NV_Analysis.py)
#   PLS-DA : autoscaling + PLS (5 latent variables) trained on the training folds
#            only; predicted value > 0.5 = NV
# Confusion matrix: rows = true group, columns = predicted group.
#   sensitivity = NVs correctly called NV, specificity = EVs correctly called EV
#   (NV is taken as the "positive" class)
folds = list(StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=CV_SEED)
             .split(np.zeros(len(y)), y, sites))


def cv_pca_lda(X, refit=False, labels=None, folds_=None, T=None):
    """refit=True: PCA is also re-fitted inside each training fold (stricter check).
    labels / folds_ / T are only given in the permutation test (STEP 8)."""
    labels = y if labels is None else labels
    folds_ = folds if folds_ is None else folds_
    if T is None:
        T = PCA(n_components=N_COMP, svd_solver='full').fit_transform(X)
    pred = np.empty(len(labels), int)
    for tr, te in folds_:
        if refit:
            pca_f = PCA(n_components=N_COMP, svd_solver='full').fit(X[tr])
            Ttr, Tte = pca_f.transform(X[tr]), pca_f.transform(X[te])
        else:
            Ttr, Tte = T[tr], T[te]
        pred[te] = LinearDiscriminantAnalysis().fit(Ttr, labels[tr]).predict(Tte)
    return pred


def cv_plsda(X, labels=None, folds_=None):
    labels = y if labels is None else labels
    folds_ = folds if folds_ is None else folds_
    pred = np.empty(len(labels), int)
    for tr, te in folds_:
        sc = StandardScaler().fit(X[tr])
        pls = PLSRegression(n_components=N_COMP, scale=False).fit(sc.transform(X[tr]), labels[tr])
        pred[te] = (pls.predict(sc.transform(X[te])).ravel() > 0.5).astype(int)
    return pred


report('\nSTEP 5 - cross-validation (5 folds, whole sites left out)')
for k, (tr, te) in enumerate(folds, 1):
    report(f'  fold {k}: test sites ' + ', '.join(sorted(set(sites[te]))))
cv_rows = []
fig2, axs2 = plt.subplots(2, 2, figsize=(8, 7.6))
letters = iter('abcd')
for row, w in enumerate(['fingerprint', 'ch']):
    X = data[w]
    strict = (cv_pca_lda(X, refit=True) == y).mean() * 100
    for col, (model, pred) in enumerate([('PCA-LDA', cv_pca_lda(X)), ('PLS-DA', cv_plsda(X))]):
        cm = confusion_matrix(y, pred)
        acc = (pred == y).mean() * 100
        sens, spec = cm[1, 1] / cm[1].sum() * 100, cm[0, 0] / cm[0].sum() * 100
        cv_rows.append({'Region': TITLES[w].replace('$^{-1}$', '-1'), 'Model': model,
                        'Accuracy (%)': round(acc, 1), 'Sensitivity NV (%)': round(sens, 1),
                        'Specificity EV (%)': round(spec, 1),
                        'EV correct': f'{cm[0, 0]}/{cm[0].sum()}',
                        'NV correct': f'{cm[1, 1]}/{cm[1].sum()}'})
        RES[w].setdefault('cv', {})[model] = {'acc': acc, 'sens': sens, 'spec': spec, 'cm': cm}
        report(f'  {w:11s} {model}: accuracy {acc:.1f} %, sensitivity {sens:.1f} %, '
               f'specificity {spec:.1f} %  (EV {cm[0, 0]}/120, NV {cm[1, 1]}/120)')
        ax = axs2[row, col]
        ax.imshow(cm, cmap='Blues', vmin=0, vmax=120)
        for i in range(2):
            for j in range(2):
                ax.text(j, i, f'{cm[i, j]}\n({cm[i, j] / cm[i].sum() * 100:.1f}%)', ha='center',
                        va='center', fontsize=13, color='white' if cm[i, j] > 60 else 'black')
        ax.set_xticks([0, 1])
        ax.set_xticklabels(['EVs', 'NVs'])
        ax.set_yticks([0, 1])
        ax.set_yticklabels(['EVs', 'NVs'])
        ax.set_xlabel('Predicted')
        ax.set_ylabel('True')
        ax.set_title(f'{model} - {"Fingerprint" if w == "fingerprint" else "CH-stretching"}\n'
                     f'accuracy {acc:.1f}%', fontsize=12)
        ax.text(-0.32, 1.12, f'({next(letters)})', transform=ax.transAxes, fontsize=14, va='bottom')
    report(f'  {w:11s} PCA-LDA with PCA re-fitted in each fold (stricter check): {strict:.1f} %')
    RES[w]['strict'] = strict
pd.DataFrame(cv_rows).to_csv(OUTPUT_DIR / 'CV_accuracy.csv', index=False)
fig2.tight_layout()
save_figure(fig2, 'Fig_ConfusionMatrices')


def main_bands(g, curve, n, min_gap=20):
    """The n largest local extrema (by absolute value) of a curve, at least
    min_gap cm-1 apart. Returns [(wavenumber, value), ...] sorted by wavenumber."""
    i = np.flatnonzero((np.abs(curve[1:-1]) >= np.abs(curve[:-2])) &
                       (np.abs(curve[1:-1]) >= np.abs(curve[2:]))) + 1
    chosen = []
    for k in i[np.argsort(-np.abs(curve[i]))]:
        if all(abs(g[k] - g[j]) >= min_gap for j in chosen):
            chosen.append(k)
        if len(chosen) == n:
            break
    return [(g[k], curve[k]) for k in sorted(chosen)]


N_BANDS = {'fingerprint': 8, 'ch': 4}       # how many bands are labelled per region


# ============================================================================
# STEP 6 - PCA loadings: which bands make PC1 and PC2
# ============================================================================
# A loading tells how much each wavenumber contributes to a PC. A spectrum with
# a high PC1 score is relatively strong at the bands with positive PC1 loading
# and weak at the bands with negative loading. Because the group means of the
# PC1 scores have opposite signs (STEP 2), the PC1 loading shows which bands are
# relatively stronger in EVs and which in NVs (relative, because of SNV).
report('\nSTEP 6 - PCA loadings (largest PC1 bands)')
fig3, axs3 = plt.subplots(1, 2, figsize=(12, 4.6), gridspec_kw={'width_ratios': [3, 1.3]})
load_rows = []
for ax, w, letter in zip(axs3, ['fingerprint', 'ch'], 'ab'):
    g, L, var = WINDOWS[w]['g'], RES[w]['loadings'], RES[w]['var']
    hi_group = GROUPS[0] if RES[w]['pc1_mean'][0] > 0 else GROUPS[1]   # group on the PC1 > 0 side
    lo_group = GROUPS[1] if hi_group == GROUPS[0] else GROUPS[0]
    ax.axhline(0, color='k', lw=0.6)
    ax.plot(g, L[0], color='#0B5A7E', lw=1.2, label=f'PC1 ({var[0]:.1f}%)')
    ax.plot(g, L[1], color='#B5541C', lw=1.0, alpha=0.8, label=f'PC2 ({var[1]:.1f}%)')
    bands = main_bands(g, L[0], N_BANDS[w])
    RES[w]['load_bands'] = bands
    span = np.ptp(np.r_[L[0], L[1]])
    for wn, v in bands:
        ax.text(wn, v + (0.03 if v > 0 else -0.03) * span, f'{wn:.0f}', ha='center',
                va='bottom' if v > 0 else 'top', fontsize=9, rotation=90)
        load_rows.append({'Region': w, 'Band (cm-1)': round(wn), 'PC1 loading': round(v, 4),
                          'Relatively stronger in': hi_group if v > 0 else lo_group})
    report(f'  {w}: PC1 > 0 = {hi_group} side.  Main PC1 bands: ' +
           ', '.join(f'{wn:.0f} ({"+" if v > 0 else "-"}, {hi_group if v > 0 else lo_group})'
                     for wn, v in bands))
    ax.set_xlim(g[0], g[-1])
    ax.set_ylim(L[:2].min() - 0.18 * span, L[:2].max() + 0.22 * span)
    ax.set_xlabel('Raman shift (cm$^{-1}$)')
    ax.set_ylabel('Loading (a.u.)')
    ax.set_title(f'{TITLES[w]}\nPC1 > 0: {hi_group} side', fontsize=11, loc='left')
    ax.legend(frameon=False, fontsize=10, loc='lower left')
    ax.text(-0.10 if w == 'fingerprint' else -0.25, 1.06, f'({letter})', transform=ax.transAxes,
            fontsize=14, va='bottom')
fig3.tight_layout()
save_figure(fig3, 'Fig_Loadings')
pd.DataFrame(load_rows).to_csv(OUTPUT_DIR / 'Loadings_main_bands.csv', index=False)


# ============================================================================
# STEP 7 - difference spectrum NV - EV
# ============================================================================
# 1. mean SNV spectrum of each site (20 spectra)  -> 6 EV and 6 NV site spectra
# 2. difference = mean of the 6 NV site spectra - mean of the 6 EV site spectra
# 3. uncertainty from the site-to-site variation: SE = sqrt(SD_EV^2/6 + SD_NV^2/6)
#    (SD over the 6 site spectra); the shaded band is difference +/- 2 SE.
#    Where the band does not include zero, the difference is consistent
#    across sites. Positive = relatively stronger in NVs, negative = in EVs.
report('\nSTEP 7 - difference spectrum NV - EV (site means, +/- 2 SE)')
fig4, axs4 = plt.subplots(1, 2, figsize=(12, 4.6), gridspec_kw={'width_ratios': [3, 1.3]})
diff_rows = []
for ax, w, letter in zip(axs4, ['fingerprint', 'ch'], 'ab'):
    X, g = data[w], WINDOWS[w]['g']
    S = {c: np.array([X[sites == s].mean(0) for s in sorted(set(sites[y == c]))]) for c in (0, 1)}
    d = S[1].mean(0) - S[0].mean(0)
    se = np.sqrt(S[0].var(0, ddof=1) / len(S[0]) + S[1].var(0, ddof=1) / len(S[1]))
    sig = np.abs(d) > 2 * se
    ax.axhline(0, color='k', lw=0.6)
    ax.fill_between(g, d - 2 * se, d + 2 * se, color='0.75', lw=0, label='$\\pm$2 SE (6 + 6 sites)')
    ax.plot(g, d, color='k', lw=1.0, label='NV $-$ EV')
    bands = [b for b in main_bands(g, np.where(sig, d, 0), N_BANDS[w]) if b[1] != 0]
    RES[w]['diff_bands'] = bands
    RES[w]['diff_sig_frac'] = sig.mean() * 100
    span = np.ptp(np.r_[d - 2 * se, d + 2 * se])
    for wn, v in bands:
        ax.text(wn, v + (0.04 if v > 0 else -0.04) * span, f'{wn:.0f}', ha='center',
                va='bottom' if v > 0 else 'top', fontsize=9, rotation=90)
        diff_rows.append({'Region': w, 'Band (cm-1)': round(wn), 'NV - EV (SNV)': round(v, 3),
                          '2 SE': round(float(np.interp(wn, g, 2 * se)), 3),
                          'Relatively stronger in': GROUPS[1] if v > 0 else GROUPS[0]})
    report(f'  {w}: |difference| > 2 SE at {sig.mean() * 100:.0f} % of the wavenumbers. '
           'Largest consistent differences: ' +
           ', '.join(f'{wn:.0f} ({"NVs" if v > 0 else "EVs"} higher)' for wn, v in bands))
    ax.set_xlim(g[0], g[-1])
    ax.set_ylim((d - 2 * se).min() - 0.15 * span, (d + 2 * se).max() + 0.2 * span)
    ax.set_xlabel('Raman shift (cm$^{-1}$)')
    ax.set_ylabel('$\\Delta$ SNV intensity, NV $-$ EV')
    ax.set_title(TITLES[w], fontsize=11, loc='left')
    ax.legend(frameon=False, fontsize=10, loc='lower left' if w == 'fingerprint' else 'upper left')
    ax.text(-0.10 if w == 'fingerprint' else -0.25, 1.06, f'({letter})', transform=ax.transAxes,
            fontsize=14, va='bottom')
fig4.tight_layout()
save_figure(fig4, 'Fig_Difference')
pd.DataFrame(diff_rows).to_csv(OUTPUT_DIR / 'Difference_main_bands.csv', index=False)


# ============================================================================
# STEP 8 - exact label-shuffling (permutation) test
# ============================================================================
# Question: could an accuracy this high appear by chance, given that only 12
# sites were measured? Every one of the 924 ways of calling 6 of the 12 sites
# "EV" (and the other 6 "NV") is tried; for each, the complete cross-validation
# of STEP 5 is repeated (new folds, new models). The real labeling is one of the
# 924. p = (number of labelings with accuracy >= real accuracy) / 924.
# The smallest possible p with 6 + 6 sites is 1/924 = 0.0011 (or 2/924 = 0.0022
# if the mirror labeling, EV<->NV swapped, scores the same).
site_ids = np.array(sorted(set(sites)))
perm_rows = []
if N_PERM_MODE == 'exact':
    report('\nSTEP 8 - exact permutation test (924 site labelings x 2 models x 2 regions)')
    labelings = list(combinations(range(len(site_ids)), 6))
    fig5, axs5 = plt.subplots(2, 2, figsize=(9, 7))
    letters = iter('abcd')
    for row, w in enumerate(['fingerprint', 'ch']):
        X = data[w]
        T_all = PCA(n_components=N_COMP, svd_solver='full').fit_transform(X)
        accs = {'PCA-LDA': [], 'PLS-DA': []}
        for k, ev_sites in enumerate(labelings):
            lab_site = dict(zip(site_ids, [0 if i in ev_sites else 1 for i in range(len(site_ids))]))
            yp = np.array([lab_site[s] for s in sites])
            fp = list(StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=CV_SEED)
                      .split(np.zeros(len(yp)), yp, sites))
            accs['PCA-LDA'].append((cv_pca_lda(X, labels=yp, folds_=fp, T=T_all) == yp).mean() * 100)
            accs['PLS-DA'].append((cv_plsda(X, labels=yp, folds_=fp) == yp).mean() * 100)
            if (k + 1) % 100 == 0:
                print(f'  {w}: {k + 1}/{len(labelings)} labelings done')
        for col, model in enumerate(['PCA-LDA', 'PLS-DA']):
            a = np.array(accs[model])
            obs = RES[w]['cv'][model]['acc']
            p_perm = np.mean(a >= obs - 1e-9)
            RES[w].setdefault('perm', {})[model] = {'p': p_perm, 'median': np.median(a),
                                                    'p95': np.percentile(a, 95), 'n': len(a)}
            perm_rows.append({'Region': w, 'Model': model, 'Real accuracy (%)': round(obs, 1),
                              'Shuffled: median (%)': round(np.median(a), 1),
                              'Shuffled: 95th percentile (%)': round(np.percentile(a, 95), 1),
                              'Shuffled: maximum (%)': round(a.max(), 1),
                              'Labelings >= real': int(np.sum(a >= obs - 1e-9)),
                              'Labelings tested': len(a), 'p (exact)': round(p_perm, 4)})
            report(f'  {w:11s} {model}: real {obs:.1f} %; shuffled median {np.median(a):.1f} %, '
                   f'95th percentile {np.percentile(a, 95):.1f} %, max {a.max():.1f} %; '
                   f'{np.sum(a >= obs - 1e-9)} of {len(a)} labelings >= real -> p = {p_perm:.4f}')
            ax = axs5[row, col]
            ax.hist(a, bins=np.arange(0, 102.5, 2.5), color='0.6', edgecolor='white')
            ax.axvline(obs, color='#B5541C', lw=2)
            ax.text(obs - 1, ax.get_ylim()[1] * 0.92, f'real\n{obs:.1f}%', ha='right', va='top',
                    color='#B5541C', fontsize=10)
            ax.set_xlim(0, 100)
            ax.set_xlabel('Cross-validated accuracy (%)')
            ax.set_ylabel('Number of site labelings')
            ax.set_title(f'{model} - {"Fingerprint" if w == "fingerprint" else "CH-stretching"}'
                         f'\np = {p_perm:.4f} ({len(a)} labelings)', fontsize=11)
            ax.text(-0.18, 1.08, f'({next(letters)})', transform=ax.transAxes, fontsize=14,
                    va='bottom')
    fig5.tight_layout()
    save_figure(fig5, 'Fig_Permutation')
    pd.DataFrame(perm_rows).to_csv(OUTPUT_DIR / 'Permutation_test.csv', index=False)
else:
    report('\nSTEP 8 - permutation test skipped (N_PERM_MODE = "skip")')


# ============================================================================
# STEP 9 - the results in words (every number comes from this run)
# ============================================================================
def region_name(w):
    return 'fingerprint region (250-1790 cm-1)' if w == 'fingerprint' else \
        'CH-stretching region (2760-3095 cm-1)'


txt = ['RESULTS - Acacia EVs vs NVs on Ag/SiNWs (generated by Acacia_PCA_LDA_Explained.py)',
       '=' * 78, '',
       f'Data: {np.sum(y == 0)} EV and {np.sum(y == 1)} NV SERS spectra, 6 measurement sites per '
       f'group, 20 spectra per site. Pre-processing: cosmic-ray removal, 2 cm-1 grid, '
       f'Savitzky-Golay (11 pts, 3rd order), ALS baseline, SNV.', '']
for w in ['fingerprint', 'ch']:
    r = RES[w]
    v = r['var']
    pc1_side = GROUPS[0] if r['pc1_mean'][0] > 0 else GROUPS[1]
    txt += [f'--- {region_name(w).upper()} ---', '',
            f'1. Unsupervised PCA. PC1 and PC2 explain {v[0]:.1f}% and {v[1]:.1f}% of the variance '
            f'(PC1-PC5: {v[:N_COMP].sum():.1f}%). The groups separate mainly along PC1 '
            f'(mean PC1 score: EVs {r["pc1_mean"][0]:+.2f}, NVs {r["pc1_mean"][1]:+.2f}), '
            f'although the labels were not used.',
            '   PC1 loading, main bands (cm-1; relatively stronger in): ' +
            ', '.join(f'{wn:.0f} ({pc1_side if val > 0 else ({GROUPS[0], GROUPS[1]} - {pc1_side}).pop()})'
                      for wn, val in r['load_bands']) + '.',
            '',
            f'2. Difference spectrum NV - EV (site means, +/- 2 SE): the difference exceeds 2 SE '
            f'at {r["diff_sig_frac"]:.0f}% of the wavenumbers. Largest consistent differences: ' +
            ', '.join(f'{wn:.0f} ({"NVs" if val > 0 else "EVs"} higher)'
                      for wn, val in r['diff_bands']) + '.',
            '   (Relative intensities after SNV normalisation, not absolute amounts.)', '',
            f'3. PCA-LDA (5 PCs), displayed on all spectra: median canonical variable 1 '
            f'{r["ld_median"][0]:+.2f} (EVs) vs {r["ld_median"][1]:+.2f} (NVs); Welch t-test on '
            f'the 6 + 6 site means p = {r["p_ld1"]:.1e}. Descriptive only (LDA fitted on the same '
            f'spectra).', '',
            '4. Classification of new sites (5-fold cross-validation, whole sites left out):']
    for model in ['PCA-LDA', 'PLS-DA']:
        c = r['cv'][model]
        txt.append(f'   {model}: accuracy {c["acc"]:.1f}% ({c["cm"][0, 0] + c["cm"][1, 1]}/'
                   f'{c["cm"].sum()} spectra), sensitivity (NV) {c["sens"]:.1f}%, specificity (EV) '
                   f'{c["spec"]:.1f}%; EVs {c["cm"][0, 0]}/120, NVs {c["cm"][1, 1]}/120 correct.')
    txt.append(f'   Stricter check (PCA re-fitted inside each training fold): PCA-LDA '
               f'{r["strict"]:.1f}%.')
    if 'perm' in r:
        txt += ['', '5. Exact permutation test (all 924 assignments of 6 of the 12 sites to "EV"):']
        for model in ['PCA-LDA', 'PLS-DA']:
            q = r['perm'][model]
            txt.append(f'   {model}: shuffled labels give a median accuracy of {q["median"]:.1f}% '
                       f'(95th percentile {q["p95"]:.1f}%); p = {q["p"]:.4f}.')
    txt.append('')
txt += ['--- WHAT THESE RESULTS DO AND DO NOT SHOW ---',
        '* The EV and NV spectra differ reproducibly: a model trained on some sites classifies the '
        'spectra of sites it has never seen with the accuracies above, far above the accuracies '
        'obtained with shuffled labels.',
        '* The independent unit is the measurement site (6 per group), not the spectrum (120 per '
        'group). With 6 + 6 sites the smallest possible permutation p is 2/924 = 0.0022 (the real '
        'labeling and its EV<->NV mirror image give the same accuracy); p = 0.0022 means that no '
        'other way of splitting the 12 sites reaches the real accuracy.',
        '* If all EV spectra come from one preparation and all NV spectra from another, the '
        'comparison cannot separate EV-vs-NV biology from preparation or measurement-session '
        'effects; independent replicate preparations would address this.',
        '* After SNV, band intensities are relative (composition), not absolute amounts.']
(OUTPUT_DIR / 'RESULTS_for_manuscript.txt').write_text('\n'.join(txt) + '\n', encoding='utf-8')
print('\n' + '\n'.join(txt))

(OUTPUT_DIR / 'PCA_LDA_summary.txt').write_text('\n'.join(SUMMARY) + '\n', encoding='utf-8')
print(f'\nSaved in: {OUTPUT_DIR}')
if SHOW_WINDOW:
    plt.show()
