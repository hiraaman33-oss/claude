# =============================================================================
# GRAPHENE EFFECT ANALYSIS  |  SSY Raman Spectroscopy
# Book2.xlsx  |  Sheet: 4 - Normalised  |  100 spectra  |  450-1800 cm-1
#
# SCIENTIFIC QUESTION
#   Does adding a graphene layer change the SSY Raman signal,
#   and is the effect consistent across both substrates (PDMS, SiO2/Si)?
#
# PREPROCESSING (on top of the already-normalised data)
#   1. Savitzky-Golay smoothing      (window=7, poly=2)
#   2. ALS baseline correction       (removes substrate fluorescence background)
#   3. SNV normalisation             (removes intensity scale differences between sessions)
#
# ANALYSES
#   A. Mean spectra + difference spectra  : mean(Gr-X) - mean(X) per substrate
#   B. Graphene band intensities          : D-band (~1350) and G-band (~1580) box plots + stats
#   C. Binary PLS-DA : Gr (50) vs no-Gr (50)   - 3 LVs, 5-fold CV + LOO + perm (999)
#   D. PDMS-only     : Gr-PDMS (25) vs PDMS (25)
#   E. SiO2-only     : Gr-SiO2 (25) vs SiO2 (25)
#   F. VIP comparison: all 3 binary models side by side
#   G. Accuracy summary
#
# OUTPUTS -> GrapheneEffect_Analysis folder (see OUTPUT_DIR below)
#
# HOW TO RUN (PyCharm Terminal):
#   pip install pandas openpyxl scikit-learn scipy matplotlib
# =============================================================================

import os
import re
import subprocess
import warnings

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from scipy.stats import mannwhitneyu
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve

from sklearn.cross_decomposition import PLSRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import (
    StratifiedKFold, LeaveOneOut, cross_val_predict, permutation_test_score
)
from sklearn.metrics import confusion_matrix, accuracy_score, ConfusionMatrixDisplay
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, ClassifierMixin

warnings.filterwarnings('ignore')

# =============================================================================
# 0.  SETTINGS
# =============================================================================
DATA_PATH      = r"C:\Users\Hira Aman\Desktop\PROF_DOMENICO'S"
FILE_NAME      = "Book2.xlsx"
SHEET_NAME     = "4 - Normalised"

WAVENUMBER_MIN = 450
WAVENUMBER_MAX = 1800

SG_WINDOW   = 7
SG_POLY     = 2
ALS_LAMBDA  = 1e5      # baseline smoothness (higher = smoother)
ALS_P       = 0.01     # asymmetry (small = baseline stays below peaks)
ALS_NITER   = 15

N_LV_BINARY = 3        # latent variables for binary PLS-DA models
N_PERM      = 999
RANDOM_STATE = 42

D_BAND_RANGE = (1320, 1380)   # D-band integration window (cm-1)
G_BAND_RANGE = (1560, 1620)   # G-band integration window (cm-1)

OUTPUT_DIR = os.path.join(DATA_PATH, "GrapheneEffect_Analysis")

# Robust directory creation
try:
    os.makedirs(OUTPUT_DIR, exist_ok=True)
except Exception:
    pass
if not os.path.isdir(OUTPUT_DIR):
    try:
        subprocess.run('cmd /c mkdir "%s"' % OUTPUT_DIR,
                       shell=True, check=True, capture_output=True)
    except Exception as _e:
        raise RuntimeError(
            "\n[ERROR] Cannot create output folder:\n  %s\n"
            "Create it manually in Windows Explorer.\nError: %s" % (OUTPUT_DIR, _e))
print("[INFO] Output folder ready : %s" % OUTPUT_DIR)

# =============================================================================
# 1.  FILE LOADER
# =============================================================================
def load_xlsx(file_path, sheet):
    try:
        df = pd.read_excel(file_path, sheet_name=sheet,
                           header=0, index_col=0, engine='openpyxl')
        print("[INFO] Read engine : openpyxl")
        return df
    except Exception as e:
        raise RuntimeError(
            "\n[ERROR] Cannot read file: %s\nDetails: %s" % (file_path, e))

# =============================================================================
# 2.  CLASS MAPPING
# =============================================================================
def assign_class(col_name):
    s = col_name.lower()
    if   re.search(r'gr[-_]?sio2|gr[-_]?si',     s): return 'SSY-Gr-SiO2/Si'
    elif re.search(r'gr\w*[-_]pdms|gr[-_]?pdms', s): return 'SSY-Gr-PDMS'
    elif re.search(r'sio2[-_]?si',                s): return 'SSY-SiO2/Si'
    elif re.search(r'pdms',                       s): return 'SSY-PDMS'
    else:                                              return None

# =============================================================================
# 3.  LOAD DATA
# =============================================================================
print("\n" + "=" * 65)
print("  GRAPHENE EFFECT ANALYSIS  |  SSY Raman  |  450-1800 cm-1")
print("=" * 65)

file_path = os.path.join(DATA_PATH, FILE_NAME)
print("\n[INFO] File : %s" % file_path)

raw = load_xlsx(file_path, SHEET_NAME)
print("[INFO] Loaded : %d wavenumber rows x %d sample columns" % raw.shape)

wavenumbers_all = raw.index.astype(float).values
col_names_all   = raw.columns.tolist()
X_all           = raw.values.T.astype(float)

crop_mask   = ((wavenumbers_all >= WAVENUMBER_MIN) &
               (wavenumbers_all <= WAVENUMBER_MAX))
wavenumbers = wavenumbers_all[crop_mask]
X_crop      = X_all[:, crop_mask]

assert X_crop.shape[1] > 0,         "ERROR: No points in specified range!"
assert np.isnan(X_crop).sum() == 0, "ERROR: NaN values in data!"
print("[INFO] Cropped : %.1f - %.1f cm-1  (%d points)" % (
    wavenumbers[0], wavenumbers[-1], X_crop.shape[1]))

classes_raw = np.array([assign_class(c) for c in col_names_all])
valid_mask  = np.array([c is not None for c in classes_raw])
X_crop      = X_crop[valid_mask]
classes     = classes_raw[valid_mask]
col_names   = [col_names_all[i] for i in range(len(col_names_all)) if valid_mask[i]]

print("\n[INFO] Total valid samples : %d" % len(classes))
for cls, cnt in zip(*np.unique(classes, return_counts=True)):
    print("         %-25s : %d" % (cls, cnt))

# =============================================================================
# 4.  PREPROCESSING  :  SG  ->  ALS baseline  ->  SNV
# =============================================================================
def als_baseline(y, lam=ALS_LAMBDA, p=ALS_P, niter=ALS_NITER):
    L = len(y)
    D = diags([1, -2, 1], [0, 1, 2], shape=(L - 2, L))
    H = lam * (D.T @ D)
    w = np.ones(L)
    z = y.copy()
    for _ in range(niter):
        W = diags(w)
        z = spsolve(W + H, w * y)
        w = p * (y > z) + (1 - p) * (y <= z)
    return z

def snv(X):
    mu = X.mean(axis=1, keepdims=True)
    sd = X.std(axis=1, keepdims=True)
    sd[sd == 0] = 1
    return (X - mu) / sd

print("\n[PRE] Step 1/3 : SG smoothing ...")
X_sg = savgol_filter(X_crop, window_length=SG_WINDOW, polyorder=SG_POLY, axis=1)

print("[PRE] Step 2/3 : ALS baseline correction (lam=%.0e, p=%.3f) ..." % (ALS_LAMBDA, ALS_P))
X_als = np.array([x - als_baseline(x) for x in X_sg])

print("[PRE] Step 3/3 : SNV normalisation ...")
X_pp  = snv(X_als)

print("[PRE] Preprocessing complete. Shape : %s" % str(X_pp.shape))

# =============================================================================
# 5.  HELPER FUNCTIONS
# =============================================================================
def compute_vip(fitted_pls):
    T_ = fitted_pls.x_scores_
    W_ = fitted_pls.x_weights_
    Q_ = fitted_pls.y_loadings_
    p  = W_.shape[0]
    SS = np.sum(T_**2, axis=0) * np.sum(Q_**2, axis=0)
    Wn = W_ / np.linalg.norm(W_, axis=0)
    return np.sqrt(p * np.sum(SS * Wn**2, axis=1) / np.sum(SS))

class PLSDAClf(BaseEstimator, ClassifierMixin):
    def __init__(self, n_components=3):
        self.n_components = n_components
    def fit(self, X, y):
        self.classes_ = np.unique(y)
        Yd = pd.get_dummies(pd.Series(y)).values.astype(float)
        self._pls = PLSRegression(n_components=self.n_components,
                                  scale=False, max_iter=10000)
        self._pls.fit(X, Yd)
        return self
    def predict(self, X):
        return np.argmax(self._pls.predict(X), axis=1)

def save_fig(fig, fname):
    fig.savefig(os.path.join(OUTPUT_DIR, fname), dpi=150, bbox_inches='tight')
    plt.close(fig)
    print('[SAVED] %s' % fname)

try:
    _trapz = np.trapezoid          # NumPy >= 2.0
except AttributeError:
    _trapz = np.trapz              # NumPy < 2.0

def band_intensity(X, wavenumbers, band_range):
    mask = (wavenumbers >= band_range[0]) & (wavenumbers <= band_range[1])
    return _trapz(X[:, mask], wavenumbers[mask], axis=1)

def sig_stars(p):
    if   p < 0.001: return '***'
    elif p < 0.01:  return '**'
    elif p < 0.05:  return '*'
    else:           return 'ns'

def run_binary_plsda(X, y, label_0, label_1, n_lv=N_LV_BINARY, n_perm=N_PERM):
    """Fit full model + 5-fold CV + LOO-CV + permutation test. Returns dict."""
    cv5 = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    loo = LeaveOneOut()
    pipe = Pipeline([('sc', StandardScaler()),
                     ('pls', PLSDAClf(n_lv))])

    # Full model (for scores + VIP)
    sc_full = StandardScaler().fit(X)
    Xs      = sc_full.transform(X)
    Yd      = pd.get_dummies(pd.Series(y)).values.astype(float)
    pls_full = PLSRegression(n_components=n_lv, scale=False, max_iter=10000)
    pls_full.fit(Xs, Yd)
    T_full = pls_full.x_scores_
    vip    = compute_vip(pls_full)

    # CV
    y_5f   = cross_val_predict(pipe, X, y, cv=cv5)
    y_loo  = cross_val_predict(pipe, X, y, cv=loo)
    acc_5f = accuracy_score(y, y_5f)
    acc_loo= accuracy_score(y, y_loo)
    cm_5f  = confusion_matrix(y, y_5f)
    cm_loo = confusion_matrix(y, y_loo)

    # Permutation
    obs, perm_sc, pval = permutation_test_score(
        pipe, X, y, scoring='accuracy', cv=cv5,
        n_permutations=n_perm, random_state=RANDOM_STATE, n_jobs=-1)

    return dict(T=T_full, vip=vip, y=y,
                y_5f=y_5f, y_loo=y_loo,
                acc_5f=acc_5f, acc_loo=acc_loo,
                cm_5f=cm_5f, cm_loo=cm_loo,
                obs=obs, perm_sc=perm_sc, pval=pval,
                labels=[label_0, label_1])

# =============================================================================
# 6.  PER-CLASS DATA SUBSETS
# =============================================================================
idx = {cls: np.where(classes == cls)[0] for cls in np.unique(classes)}

X_GrPDMS  = X_pp[idx['SSY-Gr-PDMS']]
X_PDMS    = X_pp[idx['SSY-PDMS']]
X_GrSiO2  = X_pp[idx['SSY-Gr-SiO2/Si']]
X_SiO2    = X_pp[idx['SSY-SiO2/Si']]

PALETTE = {'SSY-PDMS': '#1f77b4', 'SSY-Gr-PDMS': '#ff7f0e',
           'SSY-SiO2/Si': '#2ca02c', 'SSY-Gr-SiO2/Si': '#d62728'}
C_GR  = '#d62728'    # graphene groups (red family)
C_NGR = '#1f77b4'    # no-graphene groups (blue family)
C_PDMS  = '#ff7f0e'
C_SIO2  = '#2ca02c'

# =============================================================================
# 7A.  MEAN SPECTRA (all 4 classes)
# =============================================================================
print("\n[PLOT] Mean spectra ...")
fig, ax = plt.subplots(figsize=(11, 4.5))
for cls in ['SSY-PDMS', 'SSY-Gr-PDMS', 'SSY-SiO2/Si', 'SSY-Gr-SiO2/Si']:
    m = X_pp[idx[cls]].mean(axis=0)
    s = X_pp[idx[cls]].std(axis=0)
    lbl = cls.replace('SSY-', '')
    ls  = '--' if 'Gr' in cls else '-'
    ax.plot(wavenumbers, m, color=PALETTE[cls], lw=1.5, ls=ls, label=lbl)
    ax.fill_between(wavenumbers, m - s, m + s, color=PALETTE[cls], alpha=0.10)

for band_c, (lo, hi), bname in [('purple', D_BAND_RANGE, 'D-band'),
                                  ('green',  G_BAND_RANGE, 'G-band')]:
    ax.axvspan(lo, hi, alpha=0.12, color=band_c, label='%s (%d-%d cm-1)' % (bname, lo, hi))

ax.set_xlim(WAVENUMBER_MIN, WAVENUMBER_MAX)
ax.set_xlabel('Wavenumber (cm$^{-1}$)', fontsize=12)
ax.set_ylabel('Intensity (SNV a.u.)', fontsize=12)
ax.set_title('Mean Raman Spectra ± 1 SD  |  ALS + SNV preprocessed  |  n=25 per class',
             fontsize=11, fontweight='bold')
ax.legend(fontsize=8, ncol=3); ax.grid(True, alpha=0.2)
fig.tight_layout()
save_fig(fig, '01_Mean_Spectra.png')

# =============================================================================
# 7B.  DIFFERENCE SPECTRA  :  mean(Gr-X) - mean(X)
# =============================================================================
print("[PLOT] Difference spectra ...")
diff_PDMS = X_GrPDMS.mean(axis=0) - X_PDMS.mean(axis=0)
diff_SiO2 = X_GrSiO2.mean(axis=0) - X_SiO2.mean(axis=0)

fig, ax = plt.subplots(figsize=(11, 4.5))
ax.axhline(0, color='k', lw=0.8, ls='-')
ax.plot(wavenumbers, diff_PDMS, color='#ff7f0e', lw=1.6,
        label='Gr-PDMS − PDMS')
ax.plot(wavenumbers, diff_SiO2, color='#d62728',  lw=1.6, ls='--',
        label='Gr-SiO2/Si − SiO2/Si')

ax.fill_between(wavenumbers, diff_PDMS, 0,
                where=(diff_PDMS > 0), alpha=0.15, color='#ff7f0e')
ax.fill_between(wavenumbers, diff_PDMS, 0,
                where=(diff_PDMS < 0), alpha=0.15, color='#ff7f0e')
ax.fill_between(wavenumbers, diff_SiO2, 0,
                where=(diff_SiO2 > 0), alpha=0.10, color='#d62728')
ax.fill_between(wavenumbers, diff_SiO2, 0,
                where=(diff_SiO2 < 0), alpha=0.10, color='#d62728')

for band_c, (lo, hi), bname in [('purple', D_BAND_RANGE, 'D-band'),
                                  ('green',  G_BAND_RANGE, 'G-band')]:
    ax.axvspan(lo, hi, alpha=0.12, color=band_c)
    mid = (lo + hi) / 2
    ax.text(mid, ax.get_ylim()[1] if ax.get_ylim()[1] != 1 else 0.5,
            bname, ha='center', fontsize=9, color=band_c,
            fontweight='bold')

# Re-annotate after plot is drawn
ymax = max(np.abs(diff_PDMS).max(), np.abs(diff_SiO2).max())
for band_c, (lo, hi), bname in [('purple', D_BAND_RANGE, 'D-band'),
                                  ('green',  G_BAND_RANGE, 'G-band')]:
    mid = (lo + hi) / 2
    ax.text(mid, ymax * 1.02, bname, ha='center', fontsize=9,
            color=band_c, fontweight='bold')

ax.set_xlim(WAVENUMBER_MIN, WAVENUMBER_MAX)
ax.set_xlabel('Wavenumber (cm$^{-1}$)', fontsize=12)
ax.set_ylabel('Δ Intensity (SNV a.u.)', fontsize=12)
ax.set_title('Difference Spectra : Graphene Effect\n'
             'Positive = graphene increases intensity  |  Negative = graphene suppresses',
             fontsize=11, fontweight='bold')
ax.legend(fontsize=10); ax.grid(True, alpha=0.2)
fig.tight_layout()
save_fig(fig, '02_Difference_Spectra.png')

# =============================================================================
# 8.  GRAPHENE BAND INTENSITY BOX PLOTS  (D-band & G-band)
# =============================================================================
print("[PLOT] Graphene band intensities ...")
d_ints = {cls: band_intensity(X_pp[idx[cls]], wavenumbers, D_BAND_RANGE)
          for cls in np.unique(classes)}
g_ints = {cls: band_intensity(X_pp[idx[cls]], wavenumbers, G_BAND_RANGE)
          for cls in np.unique(classes)}

ORDER = ['SSY-PDMS', 'SSY-Gr-PDMS', 'SSY-SiO2/Si', 'SSY-Gr-SiO2/Si']
XLABELS = ['PDMS', 'Gr-PDMS', 'SiO2/Si', 'Gr-SiO2/Si']

fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))
for ax, band_ints, band_name, band_range in zip(
        axes,
        [d_ints, g_ints],
        ['D-band', 'G-band'],
        [D_BAND_RANGE, G_BAND_RANGE]):

    data   = [band_ints[cls] for cls in ORDER]
    colors = [PALETTE[cls] for cls in ORDER]

    bp = ax.boxplot(data, patch_artist=True, widths=0.5,
                    medianprops=dict(color='black', lw=2))
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color); patch.set_alpha(0.7)

    # Jitter points
    for i, (d, color) in enumerate(zip(data, colors), 1):
        jitter = np.random.default_rng(42).uniform(-0.12, 0.12, len(d))
        ax.scatter(i + jitter, d, color=color, s=20, alpha=0.6, zorder=3)

    # Stats brackets: PDMS vs Gr-PDMS
    stat1, p1 = mannwhitneyu(data[0], data[1], alternative='two-sided')
    stat2, p2 = mannwhitneyu(data[2], data[3], alternative='two-sided')
    ymax = max(max(d.max() for d in data), 0)
    ygap = (max(d.max() for d in data) - min(d.min() for d in data)) * 0.08
    for x1, x2, p, ypos in [(1, 2, p1, ymax + ygap),
                              (3, 4, p2, ymax + ygap * 2.5)]:
        ax.plot([x1, x1, x2, x2], [ypos, ypos+ygap*0.5,
                ypos+ygap*0.5, ypos], color='k', lw=1.2)
        ax.text((x1+x2)/2, ypos+ygap*0.5, sig_stars(p),
                ha='center', va='bottom', fontsize=12)

    ax.set_xticks(range(1, 5))
    ax.set_xticklabels(XLABELS, rotation=15, ha='right', fontsize=10)
    ax.set_ylabel('Integrated intensity (SNV a.u.)', fontsize=11)
    ax.set_title('%s  (%d–%d cm$^{-1}$)\n'
                 'PDMS: p=%.3f %s  |  SiO2/Si: p=%.3f %s' % (
                     band_name, band_range[0], band_range[1],
                     p1, sig_stars(p1), p2, sig_stars(p2)),
                 fontsize=10, fontweight='bold')
    ax.grid(True, alpha=0.2, axis='y')

fig.suptitle('Graphene Band Intensities : D-band & G-band\n'
             '* p<0.05  ** p<0.01  *** p<0.001  ns = not significant',
             fontsize=12, fontweight='bold')
fig.tight_layout()
save_fig(fig, '03_Graphene_Band_Boxplots.png')

# =============================================================================
# 8B.  D/G RATIO
# =============================================================================
print("[PLOT] D/G ratio ...")
dg_ratio = {}
for cls in np.unique(classes):
    d = d_ints[cls]
    g = g_ints[cls]
    g_safe = np.where(np.abs(g) < 1e-9, 1e-9, g)
    dg_ratio[cls] = d / g_safe

fig, ax = plt.subplots(figsize=(7, 5))
data_dg = [dg_ratio[cls] for cls in ORDER]
bp = ax.boxplot(data_dg, patch_artist=True, widths=0.5,
                medianprops=dict(color='black', lw=2))
for patch, color in zip(bp['boxes'], [PALETTE[c] for c in ORDER]):
    patch.set_facecolor(color); patch.set_alpha(0.7)
for i, (d, color) in enumerate(zip(data_dg, [PALETTE[c] for c in ORDER]), 1):
    jitter = np.random.default_rng(42).uniform(-0.12, 0.12, len(d))
    ax.scatter(i + jitter, d, color=color, s=25, alpha=0.6, zorder=3)
ax.set_xticks(range(1, 5))
ax.set_xticklabels(XLABELS, rotation=15, ha='right', fontsize=10)
ax.set_ylabel('D/G intensity ratio', fontsize=12)
ax.set_title('D/G Raman Band Ratio per Class\n'
             'Higher ratio = more disorder in graphene layer',
             fontsize=11, fontweight='bold')
ax.grid(True, alpha=0.2, axis='y')
fig.tight_layout()
save_fig(fig, '04_DG_Ratio.png')

# =============================================================================
# 9.  BINARY PLS-DA — helper plot functions
# =============================================================================
def plot_binary_scores(res, title_prefix, fname_scores, fname_conf, fname_perm):
    T = res['T']
    y = res['y']
    labs = res['labels']
    y_5f = res['y_5f']
    y_loo = res['y_loo']
    cols = [C_NGR, C_GR]   # 0=no-Gr, 1=Gr

    # -- Score plot: full model left, 5-fold CV right --
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    for ax, cv_pred, subtitle in zip(
            axes,
            [None, y_5f],
            ['Full model (re-sub)', '5-fold CV predictions']):
        for yi, lab, col in zip([0, 1], labs, cols):
            mask = y == yi
            ec = []
            if cv_pred is not None:
                ec = ['red' if cv_pred[i] != y[i] else 'black'
                      for i in np.where(mask)[0]]
            else:
                ec = ['black'] * mask.sum()
            ax.scatter(T[mask, 0], T[mask, 1], c=col, s=70,
                       edgecolors=ec, linewidths=0.8, alpha=0.85,
                       label=lab, zorder=3)
        ax.axhline(0, color='grey', lw=0.7, ls='--')
        ax.axvline(0, color='grey', lw=0.7, ls='--')
        ax.set_xlabel('LV1', fontsize=11)
        ax.set_ylabel('LV2', fontsize=11)
        ax.legend(fontsize=9); ax.grid(True, alpha=0.25)
        title_line2 = subtitle
        if cv_pred is not None:
            n_wrong = int((y != cv_pred).sum())
            title_line2 += '  |  acc=%.1f%%  |  %d misclassified' % (
                res['acc_5f'] * 100, n_wrong)
            ax.scatter([], [], c='white', edgecolors='red',
                       linewidths=1.5, s=60, label='Misclassified')
            ax.legend(fontsize=8)
        ax.set_title('%s Scores\n%s' % (title_prefix, title_line2),
                     fontsize=10, fontweight='bold')
    fig.suptitle('n=%d  |  %d LVs  |  LOO acc=%.1f%%' % (
        len(y), N_LV_BINARY, res['acc_loo'] * 100), fontsize=10)
    fig.tight_layout()
    save_fig(fig, fname_scores)

    # -- Confusion matrices (5-fold + LOO side by side) --
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    for ax, cm, subtitle, cmap in zip(
            axes,
            [res['cm_5f'], res['cm_loo']],
            ['5-fold CV  acc=%.1f%%' % (res['acc_5f']*100),
             'LOO-CV  acc=%.1f%%'   % (res['acc_loo']*100)],
            ['Blues', 'Purples']):
        ConfusionMatrixDisplay(cm, display_labels=labs).plot(
            ax=ax, colorbar=False, cmap=cmap, xticks_rotation=0)
        ax.set_title('%s\n%s' % (title_prefix, subtitle),
                     fontsize=10, fontweight='bold')
    fig.tight_layout()
    save_fig(fig, fname_conf)

    # -- Permutation test --
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.hist(res['perm_sc'], bins=30, color='lightsteelblue',
            edgecolor='white', label='Permuted accuracy')
    ax.axvline(res['obs'], color='crimson', lw=2.5,
               label='Observed: %.1f%%   p=%.4f' % (res['obs']*100, res['pval']))
    ax.axvline(0.5, color='grey', lw=1.2, ls=':',
               label='Chance level (50%)')
    ax.set_xlabel('5-fold CV accuracy', fontsize=12)
    ax.set_ylabel('Count', fontsize=12)
    ax.set_title('Permutation Test (%d perm.)  |  %s' % (N_PERM, title_prefix),
                 fontsize=10, fontweight='bold')
    ax.legend(fontsize=9); ax.grid(True, alpha=0.25)
    fig.tight_layout()
    save_fig(fig, fname_perm)

def plot_binary_vip(res, wavenumbers, title_prefix, fname):
    vip = res['vip']
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.plot(wavenumbers, vip, color='steelblue', lw=1.0)
    ax.fill_between(wavenumbers, vip, 1, where=(vip > 1),
                    alpha=0.25, color='crimson', label='VIP > 1')
    ax.axhline(1, color='crimson', ls='--', lw=1.0, label='VIP = 1')
    for band_c, (lo, hi), bname in [('purple', D_BAND_RANGE, 'D-band'),
                                      ('green',  G_BAND_RANGE, 'G-band')]:
        ax.axvspan(lo, hi, alpha=0.15, color=band_c, label=bname)
    ax.set_xlim(WAVENUMBER_MIN, WAVENUMBER_MAX)
    ax.set_xlabel('Wavenumber (cm$^{-1}$)', fontsize=12)
    ax.set_ylabel('VIP score', fontsize=12)
    ax.set_title('VIP Scores  |  %s\nVIP > 1 = important for graphene discrimination' %
                 title_prefix, fontsize=11, fontweight='bold')
    ax.legend(fontsize=9); ax.grid(True, alpha=0.2)
    fig.tight_layout()
    save_fig(fig, fname)

# =============================================================================
# 10. ANALYSIS C — Binary : Gr (50) vs no-Gr (50)
# =============================================================================
print("\n[MODEL C] Binary PLS-DA : Gr vs no-Gr (50 vs 50) ...")
X_bin = np.vstack([X_GrPDMS, X_GrSiO2, X_PDMS, X_SiO2])
y_bin = np.array([1]*25 + [1]*25 + [0]*25 + [0]*25)

res_bin = run_binary_plsda(X_bin, y_bin, label_0='no-Gr', label_1='Gr')
print("  5-fold CV : %.1f%%   LOO : %.1f%%   p=%.4f [%s]" % (
    res_bin['acc_5f']*100, res_bin['acc_loo']*100, res_bin['pval'],
    'SIGNIFICANT' if res_bin['pval'] < 0.05 else 'NOT significant'))

plot_binary_scores(res_bin, 'PLS-DA  Gr vs no-Gr  (n=100)',
                   '05_Binary_GrNoGr_Scores.png',
                   '07_Binary_GrNoGr_Confusion.png',
                   '08_Binary_GrNoGr_Permutation.png')
plot_binary_vip(res_bin, wavenumbers, 'Gr vs no-Gr  (all substrates)',
                '06_Binary_GrNoGr_VIP.png')

# =============================================================================
# 11. ANALYSIS D — PDMS-only binary : Gr-PDMS (25) vs PDMS (25)
# =============================================================================
print("\n[MODEL D] Binary PLS-DA : Gr-PDMS vs PDMS (25 vs 25) ...")
X_pdms_bin = np.vstack([X_GrPDMS, X_PDMS])
y_pdms_bin = np.array([1]*25 + [0]*25)

res_pdms = run_binary_plsda(X_pdms_bin, y_pdms_bin,
                             label_0='PDMS', label_1='Gr-PDMS')
print("  5-fold CV : %.1f%%   LOO : %.1f%%   p=%.4f [%s]" % (
    res_pdms['acc_5f']*100, res_pdms['acc_loo']*100, res_pdms['pval'],
    'SIGNIFICANT' if res_pdms['pval'] < 0.05 else 'NOT significant'))

plot_binary_scores(res_pdms, 'PLS-DA  Gr-PDMS vs PDMS  (n=50)',
                   '09_PDMS_Binary_Scores.png',
                   '11_PDMS_Binary_Confusion.png',
                   '12_PDMS_Binary_Permutation.png')
plot_binary_vip(res_pdms, wavenumbers, 'Gr-PDMS vs PDMS  (PDMS substrate only)',
                '10_PDMS_Binary_VIP.png')

# =============================================================================
# 12. ANALYSIS E — SiO2-only binary : Gr-SiO2 (25) vs SiO2 (25)
# =============================================================================
print("\n[MODEL E] Binary PLS-DA : Gr-SiO2/Si vs SiO2/Si (25 vs 25) ...")
X_sio2_bin = np.vstack([X_GrSiO2, X_SiO2])
y_sio2_bin = np.array([1]*25 + [0]*25)

res_sio2 = run_binary_plsda(X_sio2_bin, y_sio2_bin,
                              label_0='SiO2/Si', label_1='Gr-SiO2/Si')
print("  5-fold CV : %.1f%%   LOO : %.1f%%   p=%.4f [%s]" % (
    res_sio2['acc_5f']*100, res_sio2['acc_loo']*100, res_sio2['pval'],
    'SIGNIFICANT' if res_sio2['pval'] < 0.05 else 'NOT significant'))

plot_binary_scores(res_sio2, 'PLS-DA  Gr-SiO2/Si vs SiO2/Si  (n=50)',
                   '13_SiO2_Binary_Scores.png',
                   '15_SiO2_Binary_Confusion.png',
                   '16_SiO2_Binary_Permutation.png')
plot_binary_vip(res_sio2, wavenumbers, 'Gr-SiO2/Si vs SiO2/Si  (SiO2/Si substrate only)',
                '14_SiO2_Binary_VIP.png')

# =============================================================================
# 13. ANALYSIS F — VIP COMPARISON  (all 3 binary models on same plot)
# =============================================================================
print("\n[PLOT] VIP comparison ...")
fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
panel_info = [
    (res_bin,  'Gr vs no-Gr  (all 100 spectra)',    'steelblue'),
    (res_pdms, 'Gr-PDMS vs PDMS  (n=50)',           '#ff7f0e'),
    (res_sio2, 'Gr-SiO2/Si vs SiO2/Si  (n=50)',    '#d62728'),
]
for ax, (res, title, col) in zip(axes, panel_info):
    vip = res['vip']
    ax.plot(wavenumbers, vip, color=col, lw=1.0)
    ax.fill_between(wavenumbers, vip, 1, where=(vip > 1),
                    alpha=0.30, color=col)
    ax.axhline(1, color='grey', ls='--', lw=0.8)
    for band_c, (lo, hi), bname in [('purple', D_BAND_RANGE, 'D'),
                                      ('green',  G_BAND_RANGE, 'G')]:
        ax.axvspan(lo, hi, alpha=0.12, color=band_c)
        ax.text((lo+hi)/2, ax.get_ylim()[1] if ax.get_ylim()[1] != 1 else 2,
                bname, ha='center', fontsize=8, color=band_c, fontweight='bold')
    ax.set_ylabel('VIP', fontsize=10)
    acc_txt = '5fold=%.0f%%  LOO=%.0f%%' % (res['acc_5f']*100, res['acc_loo']*100)
    ax.set_title('%s    [%s]' % (title, acc_txt), fontsize=10, fontweight='bold')
    ax.set_xlim(WAVENUMBER_MIN, WAVENUMBER_MAX)
    ax.grid(True, alpha=0.2)

axes[-1].set_xlabel('Wavenumber (cm$^{-1}$)', fontsize=12)
axes[0].set_title(panel_info[0][1] + '    [%s]' % (
    '5fold=%.0f%%  LOO=%.0f%%' % (res_bin['acc_5f']*100, res_bin['acc_loo']*100)),
    fontsize=10, fontweight='bold')
fig.suptitle('VIP Score Comparison — Which Raman Bands Drive Graphene Discrimination?\n'
             'Shaded = VIP > 1  |  Purple = D-band  |  Green = G-band',
             fontsize=12, fontweight='bold')
fig.tight_layout()
save_fig(fig, '17_VIP_Comparison.png')

# =============================================================================
# 14. ANALYSIS G — ACCURACY SUMMARY
# =============================================================================
print("[PLOT] Accuracy summary ...")
fig, ax = plt.subplots(figsize=(9, 5.5))
models   = ['Gr vs\nno-Gr\n(n=100)', 'Gr-PDMS vs\nPDMS\n(n=50)',
            'Gr-SiO2/Si vs\nSiO2/Si\n(n=50)']
accs_5f  = [res_bin['acc_5f']*100,  res_pdms['acc_5f']*100,  res_sio2['acc_5f']*100]
accs_loo = [res_bin['acc_loo']*100, res_pdms['acc_loo']*100, res_sio2['acc_loo']*100]

x = np.arange(3); w = 0.28
b1 = ax.bar(x - w/2, accs_5f,  w, color='steelblue',  edgecolor='k', lw=0.8, label='5-fold CV')
b2 = ax.bar(x + w/2, accs_loo, w, color='darkorange', edgecolor='k', lw=0.8, label='LOO-CV')

for bar in list(b1) + list(b2):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.8,
            '%.0f%%' % bar.get_height(),
            ha='center', va='bottom', fontsize=10, fontweight='bold')

ax.axhline(50,  color='red',  ls=':', lw=1.5, label='Chance level (50%)')
ax.axhline(100, color='grey', ls=':', lw=1)
ax.set_xticks(x); ax.set_xticklabels(models, fontsize=10)
ax.set_ylabel('Classification Accuracy (%)', fontsize=12)
ax.set_ylim(0, 118)
ax.set_title('Binary PLS-DA Accuracy : Can graphene be distinguished from no-graphene?\n'
             'ALS + SNV preprocessed  |  %d LVs' % N_LV_BINARY,
             fontsize=11, fontweight='bold')

# Add p-value annotations below bars
for xi, res in zip(x, [res_bin, res_pdms, res_sio2]):
    ax.text(xi, 3, 'p=%.4f\n%s' % (res['pval'],
            '★ sig.' if res['pval'] < 0.05 else 'n.s.'),
            ha='center', va='bottom', fontsize=8, color='darkred')

ax.legend(fontsize=10); ax.grid(True, alpha=0.2, axis='y')
fig.tight_layout()
save_fig(fig, '18_Accuracy_Summary.png')

# =============================================================================
# 15. SAVE RESULTS TO EXCEL
# =============================================================================
print("\n[EXCEL] Saving results ...")
out_xlsx = os.path.join(OUTPUT_DIR, 'GrapheneEffect_Results.xlsx')

with pd.ExcelWriter(out_xlsx, engine='openpyxl') as writer:

    # --- Summary ---
    rows = [
        ('Input file',            FILE_NAME),
        ('Sheet',                 SHEET_NAME),
        ('Total samples',         100),
        ('Spectral window cm-1',  '%d-%d' % (WAVENUMBER_MIN, WAVENUMBER_MAX)),
        ('Spectral points',       X_crop.shape[1]),
        ('Preprocessing',         'SG(w=%d,p=%d) + ALS(lam=%.0e,p=%.2f) + SNV' % (
                                   SG_WINDOW, SG_POLY, ALS_LAMBDA, ALS_P)),
        ('PLS-DA latent vars',    N_LV_BINARY),
        ('', ''),
        ('--- MODEL C : Gr vs no-Gr (n=100) ---', ''),
        ('5-fold CV accuracy (%)', '%.2f' % (res_bin['acc_5f']*100)),
        ('LOO-CV accuracy (%)',    '%.2f' % (res_bin['acc_loo']*100)),
        ('Permutation p-value',   '%.4f' % res_bin['pval']),
        ('Permutation result',    'Significant' if res_bin['pval']<0.05 else 'NOT significant'),
        ('', ''),
        ('--- MODEL D : Gr-PDMS vs PDMS (n=50) ---', ''),
        ('5-fold CV accuracy (%)', '%.2f' % (res_pdms['acc_5f']*100)),
        ('LOO-CV accuracy (%)',    '%.2f' % (res_pdms['acc_loo']*100)),
        ('Permutation p-value',   '%.4f' % res_pdms['pval']),
        ('Permutation result',    'Significant' if res_pdms['pval']<0.05 else 'NOT significant'),
        ('', ''),
        ('--- MODEL E : Gr-SiO2/Si vs SiO2/Si (n=50) ---', ''),
        ('5-fold CV accuracy (%)', '%.2f' % (res_sio2['acc_5f']*100)),
        ('LOO-CV accuracy (%)',    '%.2f' % (res_sio2['acc_loo']*100)),
        ('Permutation p-value',   '%.4f' % res_sio2['pval']),
        ('Permutation result',    'Significant' if res_sio2['pval']<0.05 else 'NOT significant'),
        ('', ''),
        ('--- BAND INTENSITY STATS ---', ''),
    ]
    for band_name, band_ints, band_range in [
            ('D-band (%d-%d cm-1)' % D_BAND_RANGE, d_ints, D_BAND_RANGE),
            ('G-band (%d-%d cm-1)' % G_BAND_RANGE, g_ints, G_BAND_RANGE)]:
        rows.append((band_name, ''))
        for cls in ORDER:
            rows.append(('  %s mean' % cls.replace('SSY-',''),
                         '%.4f' % band_ints[cls].mean()))
        _, p1 = mannwhitneyu(band_ints['SSY-PDMS'],    band_ints['SSY-Gr-PDMS'],    alternative='two-sided')
        _, p2 = mannwhitneyu(band_ints['SSY-SiO2/Si'], band_ints['SSY-Gr-SiO2/Si'], alternative='two-sided')
        rows.append(('  PDMS vs Gr-PDMS p-value',       '%.4f %s' % (p1, sig_stars(p1))))
        rows.append(('  SiO2/Si vs Gr-SiO2/Si p-value', '%.4f %s' % (p2, sig_stars(p2))))
        rows.append(('', ''))

    pd.DataFrame(rows, columns=['Metric', 'Value']).to_excel(
        writer, sheet_name='Summary', index=False)

    # --- VIP scores ---
    vip_df = pd.DataFrame({
        'Wavenumber_cm1':    wavenumbers,
        'VIP_GrVsNoGr':      res_bin['vip'],
        'VIP_GrPDMS_vsPDMS': res_pdms['vip'],
        'VIP_GrSiO2_vsSiO2': res_sio2['vip'],
    })
    vip_df.to_excel(writer, sheet_name='VIP_Scores', index=False)

    # --- Band intensities ---
    band_rows = []
    for cls in ORDER:
        for i in range(25):
            band_rows.append({
                'Class':      cls,
                'Sample_idx': i+1,
                'D_band_int': d_ints[cls][i],
                'G_band_int': g_ints[cls][i],
                'DG_ratio':   dg_ratio[cls][i],
            })
    pd.DataFrame(band_rows).to_excel(writer, sheet_name='Band_Intensities', index=False)

    # --- Confusion matrices ---
    labels_bin  = ['no-Gr', 'Gr']
    labels_pdms = ['PDMS', 'Gr-PDMS']
    labels_sio2 = ['SiO2/Si', 'Gr-SiO2/Si']
    for sheet, cm, labs in [
            ('CM_GrNoGr_5fold',  res_bin['cm_5f'],   labels_bin),
            ('CM_GrNoGr_LOO',    res_bin['cm_loo'],   labels_bin),
            ('CM_PDMS_5fold',    res_pdms['cm_5f'],   labels_pdms),
            ('CM_PDMS_LOO',      res_pdms['cm_loo'],  labels_pdms),
            ('CM_SiO2_5fold',    res_sio2['cm_5f'],   labels_sio2),
            ('CM_SiO2_LOO',      res_sio2['cm_loo'],  labels_sio2)]:
        pd.DataFrame(cm, index=pd.Index(labs, name='True\\Pred'),
                     columns=labs).to_excel(writer, sheet_name=sheet)

    # --- Permutation scores ---
    pd.DataFrame({
        'Perm_GrNoGr':  res_bin['perm_sc'],
        'Perm_PDMS':    res_pdms['perm_sc'],
        'Perm_SiO2':    res_sio2['perm_sc'],
    }).to_excel(writer, sheet_name='Permutation_Scores', index=False)

    # --- Difference spectra ---
    pd.DataFrame({
        'Wavenumber_cm1':       wavenumbers,
        'Diff_GrPDMS_vs_PDMS':  diff_PDMS,
        'Diff_GrSiO2_vs_SiO2':  diff_SiO2,
    }).to_excel(writer, sheet_name='Difference_Spectra', index=False)

print('[SAVED] %s' % out_xlsx)

# =============================================================================
# 16. FINAL CONSOLE SUMMARY
# =============================================================================
print('\n' + '=' * 65)
print('  GRAPHENE EFFECT ANALYSIS — FINAL SUMMARY')
print('=' * 65)
print('  Preprocessing : SG + ALS baseline correction + SNV')
print()
print('  %-35s  %10s  %10s  %10s' % ('MODEL', '5-fold CV', 'LOO-CV', 'p-value'))
print('  ' + '-' * 70)
for label, res in [('Gr vs no-Gr (n=100)',        res_bin),
                   ('Gr-PDMS vs PDMS (n=50)',      res_pdms),
                   ('Gr-SiO2/Si vs SiO2/Si (n=50)', res_sio2)]:
    sig = '★' if res['pval'] < 0.05 else ''
    print('  %-35s  %9.1f%%  %9.1f%%  %.4f %s' % (
        label, res['acc_5f']*100, res['acc_loo']*100, res['pval'], sig))
print()
print('  Graphene band intensities (Mann-Whitney):')
print('  %-35s  %10s  %10s' % ('Comparison', 'D-band', 'G-band'))
print('  ' + '-' * 58)
for cls_base, cls_gr in [('SSY-PDMS', 'SSY-Gr-PDMS'),
                          ('SSY-SiO2/Si', 'SSY-Gr-SiO2/Si')]:
    _, pd_ = mannwhitneyu(d_ints[cls_base], d_ints[cls_gr], alternative='two-sided')
    _, pg_ = mannwhitneyu(g_ints[cls_base], g_ints[cls_gr], alternative='two-sided')
    lbl = cls_base.replace('SSY-','') + ' vs Gr'
    print('  %-35s  %6s(%.3f)  %6s(%.3f)' % (
        lbl, sig_stars(pd_), pd_, sig_stars(pg_), pg_))
print()
print('  All outputs saved to : %s' % OUTPUT_DIR)
print('=' * 65)
