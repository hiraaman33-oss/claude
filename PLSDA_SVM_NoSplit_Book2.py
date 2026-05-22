"""
=============================================================================
PLS-DA + SVM-RBF  |  NO TRAIN/TEST SPLIT  |  ALL 100 SPECTRA
SSY Raman Spectroscopy  |  Four Substrate Classes
SPECTRAL WINDOW : 450 - 1800 cm-1

DATASET
  - Book2.xlsx  |  Sheet: "4 - Normalised"
  - 100 spectra across four substrate classes (balanced, 25 each)
      SSY-PDMS       : 25
      SSY-SiO2/Si    : 25
      SSY-Gr-SiO2/Si : 25
      SSY-Gr-PDMS    : 25  (GrSL-PDMS label treated as SSY-Gr-PDMS)

STRATEGY
  - ALL 100 spectra used for model building and validation
  - VIP selection  : 3-LV PLS-DA on all 100 -> top-1800 VIP
  - PLS-DA (5 LVs) : 5-fold CV  +  LOO-CV  +  permutation test (999)
  - SVM-RBF        : 5-fold CV  +  LOO-CV
  - No data leakage : StandardScaler fitted inside CV pipeline each fold

OUTPUTS -> C:/Users/Hira Aman/Desktop/PROF_DOMENICO'S/PLSDA_SVM_NoSplit_Book2/
  01_PLSDA_Scores_LV1_LV2.png
  02_PLSDA_Scores_LV1_LV3.png
  03_PLSDA_Scores_LV2_LV3.png
  04_PLSDA_Scores_3D.png
  05_PLSDA_Confusion_5fold.png
  06_PLSDA_Confusion_LOO.png
  07_PLSDA_VIP_Scores.png
  08_PLSDA_Loadings.png
  09_PLSDA_Permutation.png
  10_SVM_Cluster_5fold.png
  11_SVM_Cluster_LOO.png
  12_SVM_Confusion_5fold.png
  13_SVM_Confusion_LOO.png
  14_Accuracy_Comparison.png
  15_Recall_Comparison.png
  16_Mean_Spectra.png
  Results_NoSplit_Book2.xlsx

HOW TO RUN (PyCharm Terminal):
  pip install pandas openpyxl scikit-learn scipy matplotlib
=============================================================================
"""

import os
import re
import subprocess
import warnings

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D                    # noqa

from sklearn.cross_decomposition import PLSRegression
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import (
    StratifiedKFold, LeaveOneOut, cross_val_predict,
    permutation_test_score
)
from sklearn.metrics import (
    confusion_matrix, accuracy_score, ConfusionMatrixDisplay
)
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from scipy.signal import savgol_filter

warnings.filterwarnings('ignore')

# =============================================================================
# 0.  SETTINGS
# =============================================================================
DATA_PATH      = r"C:\Users\Hira Aman\Desktop\PROF_DOMENICO'S"
FILE_NAME      = "Book2.xlsx"
SHEET_NAME     = "4 - Normalised"

WAVENUMBER_MIN = 450
WAVENUMBER_MAX = 1800

SG_WINDOW      = 7
SG_POLY        = 2
VIP_TOP_N      = 1800
N_COMPONENTS   = 5
N_PERM         = 999
RANDOM_STATE   = 42

OUTPUT_DIR = os.path.join(DATA_PATH, "PLSDA_SVM_NoSplit_Book2")

# Robust directory creation (handles apostrophes in Windows paths)
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
            "Please create it manually in Windows Explorer.\n"
            "Error: %s" % (OUTPUT_DIR, _e))
if not os.path.isdir(OUTPUT_DIR):
    raise RuntimeError(
        "\n[ERROR] Output folder still missing:\n  %s\n"
        "Create it manually and re-run." % OUTPUT_DIR)
print("[INFO] Output folder ready : %s" % OUTPUT_DIR)

# =============================================================================
# 1.  FILE LOADER  (xlsx via openpyxl)
# =============================================================================
def load_xlsx(file_path, sheet):
    try:
        df = pd.read_excel(file_path, sheet_name=sheet,
                           header=0, index_col=0, engine='openpyxl')
        print("[INFO] Read engine : openpyxl")
        return df
    except Exception as e:
        raise RuntimeError(
            "\n[ERROR] Cannot read file: %s\n"
            "Run:  pip install openpyxl\nThen re-run.\nDetails: %s" % (file_path, e))

# =============================================================================
# 2.  CLASS MAPPING
#     Class is determined solely by the substrate keyword in the column name.
#     Order matters: check the more-specific Gr-* patterns first.
# =============================================================================
def assign_class(col_name):
    s = col_name.lower()
    if   re.search(r'gr[-_]?sio2|gr[-_]?si',     s): return 'SSY-Gr-SiO2/Si'
    elif re.search(r'gr\w*[-_]pdms|gr[-_]?pdms', s): return 'SSY-Gr-PDMS'   # covers GrSL-PDMS
    elif re.search(r'sio2[-_]?si',                s): return 'SSY-SiO2/Si'
    elif re.search(r'pdms',                       s): return 'SSY-PDMS'
    else:                                              return None

# =============================================================================
# 3.  LOAD DATA
# =============================================================================
print("\n" + "=" * 65)
print("  PLS-DA + SVM-RBF  |  NO SPLIT  |  ALL 100 SPECTRA  |  450-1800 cm-1")
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
print("[INFO] Cropped : %.4f - %.4f cm-1  (%d points)  OK" % (
    wavenumbers[0], wavenumbers[-1], X_crop.shape[1]))

classes_raw = np.array([assign_class(c) for c in col_names_all])
bad_mask    = np.array([c is None for c in classes_raw])
if bad_mask.any():
    print("\n[WARNING] %d unrecognised column(s) — excluded:" % bad_mask.sum())
    for c in np.array(col_names_all)[bad_mask]:
        print("  ", c)

valid_mask = ~bad_mask
X_crop     = X_crop[valid_mask]
classes    = classes_raw[valid_mask]
col_names  = [col_names_all[i] for i in range(len(col_names_all)) if valid_mask[i]]

le = LabelEncoder()
y  = le.fit_transform(classes)

print("\n[INFO] Total valid samples : %d" % len(classes))
# Per-class counts (used later for recall denominators)
cls_counts = {}
for cls, cnt in zip(*np.unique(classes, return_counts=True)):
    cls_counts[cls] = int(cnt)
    print("         %-25s : %d" % (cls, cnt))
print("\n[INFO] Class encoding : %s" % str(
    dict(zip(le.classes_, le.transform(le.classes_)))))

# =============================================================================
# 4.  PRE-PROCESSING  (SG smooth on all 100 spectra)
# =============================================================================
X_sm = savgol_filter(X_crop, window_length=SG_WINDOW,
                     polyorder=SG_POLY, axis=1)
print("\n[INFO] SG smoothing applied (window=%d, poly=%d)" % (SG_WINDOW, SG_POLY))

# =============================================================================
# 5.  VIP HELPER
# =============================================================================
def compute_vip(fitted_pls):
    T_ = fitted_pls.x_scores_
    W_ = fitted_pls.x_weights_
    Q_ = fitted_pls.y_loadings_
    p  = W_.shape[0]
    SS = np.sum(T_**2, axis=0) * np.sum(Q_**2, axis=0)
    Wn = W_ / np.linalg.norm(W_, axis=0)
    return np.sqrt(p * np.sum(SS * Wn**2, axis=1) / np.sum(SS))

# =============================================================================
# 6.  VIP SELECTION  (3-LV PLS-DA on all 100 spectra)
# =============================================================================
print("\n[VIP] Initial 3-LV PLS-DA on all 100 spectra -> top-%d VIP ..." % VIP_TOP_N)

sc_init  = StandardScaler().fit(X_sm)
Xs_init  = sc_init.transform(X_sm)
Yd_init  = pd.get_dummies(pd.Series(y)).values.astype(float)
pls_init = PLSRegression(n_components=3, scale=False, max_iter=10000)
pls_init.fit(Xs_init, Yd_init)
vip_all  = compute_vip(pls_init)

n_vip_gt1 = int((vip_all > 1).sum())
top_idx   = np.argsort(vip_all)[-VIP_TOP_N:]
wn_sel    = wavenumbers[top_idx]
X_sel     = X_sm[:, top_idx]               # (100, VIP_TOP_N)

print("[VIP] VIP > 1 : %d / %d variables" % (n_vip_gt1, len(vip_all)))
print("[VIP] Top-%d VIP selected  (%.1f - %.1f cm-1)" % (
    VIP_TOP_N, wn_sel.min(), wn_sel.max()))

# =============================================================================
# 7.  HELPER CLASSIFIER
# =============================================================================
class PLSDAClf(BaseEstimator, ClassifierMixin):
    """PLS-DA wrapped as sklearn classifier (scaler is OUTSIDE in pipeline)."""
    def __init__(self, n_components=5):
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

# CV strategies
cv5 = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
loo = LeaveOneOut()

# =============================================================================
# 8.  FIT FULL PLS-DA ON ALL 100 (for score / loading visualisation only)
# =============================================================================
print("\n[PLS-DA] Fitting full model on all 100 spectra (%d LVs, top-%d vars) ..." % (
    N_COMPONENTS, VIP_TOP_N))

sc_full   = StandardScaler().fit(X_sel)
X_scaled  = sc_full.transform(X_sel)
Yd_full   = pd.get_dummies(pd.Series(y)).values.astype(float)
pls_full  = PLSRegression(n_components=N_COMPONENTS, scale=False, max_iter=10000)
pls_full.fit(X_scaled, Yd_full)

T_all    = pls_full.x_scores_      # (100, 5)
P_load   = pls_full.x_loadings_    # (VIP_TOP_N, 5)

# R2X cumulative
SS_tot  = np.sum(X_scaled ** 2)
r2x_cum = []
for lv in range(1, N_COMPONENTS + 1):
    Xrec = pls_full.x_scores_[:, :lv] @ pls_full.x_loadings_[:, :lv].T
    r2x_cum.append(1 - np.sum((X_scaled - Xrec) ** 2) / SS_tot)
print("[PLS-DA] R2X cumulative : " + "  ".join(
    ["LV%d=%.1f%%" % (i+1, r*100) for i, r in enumerate(r2x_cum)]))

resub_acc = accuracy_score(y, np.argmax(pls_full.predict(X_scaled), axis=1))
print("[PLS-DA] Re-substitution accuracy : %.1f%%" % (resub_acc * 100))

# =============================================================================
# 9.  PLS-DA CV  (5-fold + LOO)  — scaler INSIDE pipeline (no leakage)
# =============================================================================
clf_pipe_pls = Pipeline([
    ('sc',    StandardScaler()),
    ('plsda', PLSDAClf(N_COMPONENTS))
])

print("\n[PLS-DA] 5-fold CV on all 100 spectra ...")
y_pls_5f   = cross_val_predict(clf_pipe_pls, X_sel, y, cv=cv5)
acc_pls_5f = accuracy_score(y, y_pls_5f)
cm_pls_5f  = confusion_matrix(y, y_pls_5f)
print("[PLS-DA] 5-fold CV accuracy : %.1f%%" % (acc_pls_5f * 100))

print("[PLS-DA] LOO-CV on all 100 spectra  (100 folds) ...")
y_pls_loo   = cross_val_predict(clf_pipe_pls, X_sel, y, cv=loo)
acc_pls_loo = accuracy_score(y, y_pls_loo)
cm_pls_loo  = confusion_matrix(y, y_pls_loo)
print("[PLS-DA] LOO-CV accuracy        : %.1f%%" % (acc_pls_loo * 100))

# =============================================================================
# 10. PLS-DA PERMUTATION TEST
# =============================================================================
print("\n[PLS-DA] Permutation test (%d permutations, 5-fold CV) ..." % N_PERM)
obs_score, perm_scores, p_val = permutation_test_score(
    clf_pipe_pls, X_sel, y,
    scoring='accuracy', cv=cv5,
    n_permutations=N_PERM,
    random_state=RANDOM_STATE, n_jobs=-1)
print("[PLS-DA] Observed : %.2f%%   p = %.4f  [%s]" % (
    obs_score * 100, p_val,
    'SIGNIFICANT (p<0.05)' if p_val < 0.05 else 'NOT significant'))

# =============================================================================
# 11. SVM-RBF CV  (5-fold + LOO)
# =============================================================================
svm_pipe = Pipeline([
    ('sc',  StandardScaler()),
    ('svm', SVC(kernel='rbf', C=10, gamma='scale',
                probability=True, random_state=RANDOM_STATE))
])

print("\n[SVM-RBF] 5-fold CV on all 100 spectra ...")
y_svm_5f   = cross_val_predict(svm_pipe, X_sel, y, cv=cv5)
acc_svm_5f = accuracy_score(y, y_svm_5f)
cm_svm_5f  = confusion_matrix(y, y_svm_5f)
print("[SVM-RBF] 5-fold CV accuracy : %.1f%%" % (acc_svm_5f * 100))

print("[SVM-RBF] LOO-CV on all 100 spectra  (100 folds) ...")
y_svm_loo   = cross_val_predict(svm_pipe, X_sel, y, cv=loo)
acc_svm_loo = accuracy_score(y, y_svm_loo)
cm_svm_loo  = confusion_matrix(y, y_svm_loo)
print("[SVM-RBF] LOO-CV accuracy        : %.1f%%" % (acc_svm_loo * 100))

# =============================================================================
# 12. PCA FOR SVM CLUSTER VISUALISATION
# =============================================================================
pca_viz = PCA(n_components=2, random_state=RANDOM_STATE)
Z_all   = pca_viz.fit_transform(X_scaled)   # (100, 2)
print("\n[PCA-viz] PC1=%.1f%%  PC2=%.1f%% explained variance" % (
    pca_viz.explained_variance_ratio_[0] * 100,
    pca_viz.explained_variance_ratio_[1] * 100))

# =============================================================================
# 13. COLOUR / MARKER SCHEME
# =============================================================================
PALETTE = {
    'SSY-PDMS'       : '#1f77b4',
    'SSY-Gr-PDMS'    : '#ff7f0e',
    'SSY-SiO2/Si'    : '#2ca02c',
    'SSY-Gr-SiO2/Si' : '#d62728',
}
MARKER = {
    'SSY-PDMS'       : 'o',
    'SSY-Gr-PDMS'    : 's',
    'SSY-SiO2/Si'    : '^',
    'SSY-Gr-SiO2/Si' : 'D',
}
TITLE_SUF = "VIP-top-%d | %d LVs | %d-%d cm-1" % (
    VIP_TOP_N, N_COMPONENTS, WAVENUMBER_MIN, WAVENUMBER_MAX)

def lv_var(i):
    return (r2x_cum[i] - (r2x_cum[i-1] if i > 0 else 0)) * 100

def save_fig(fig, fname):
    fig.savefig(os.path.join(OUTPUT_DIR, fname), dpi=150, bbox_inches='tight')
    plt.close(fig)
    print('[SAVED] %s' % fname)

# =============================================================================
# 14. PLS-DA SCORE PLOTS
# =============================================================================
def scatter_scores(ax, lv_x, lv_y, cv_pred=None):
    for cls in le.classes_:
        idx = classes == cls
        edge_colors = []
        if cv_pred is not None:
            for i in np.where(idx)[0]:
                edge_colors.append('red' if y[i] != cv_pred[i] else 'black')
        else:
            edge_colors = ['black'] * idx.sum()
        ax.scatter(T_all[idx, lv_x], T_all[idx, lv_y],
                   c=PALETTE[cls], marker=MARKER[cls], s=75,
                   edgecolors=edge_colors, linewidths=0.8,
                   alpha=0.85, label=cls, zorder=3)
    ax.axhline(0, color='grey', lw=0.7, ls='--')
    ax.axvline(0, color='grey', lw=0.7, ls='--')
    ax.legend(fontsize=8, framealpha=0.85, ncol=2)
    ax.grid(True, alpha=0.25)
    if cv_pred is not None:
        n_wrong = int((y != cv_pred).sum())
        ax.scatter([], [], c='white', edgecolors='red',
                   linewidths=1.5, s=60, label='Misclassified (%d)' % n_wrong)
        ax.legend(fontsize=7.5, framealpha=0.85, ncol=2)

# Fig 01 — LV1 vs LV2
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
for ax, pred, subtitle in zip(
        axes,
        [None,        y_pls_5f],
        ['Full model (re-sub)', '5-fold CV predictions']):
    scatter_scores(ax, 0, 1, cv_pred=pred)
    ax.set_xlabel('LV1 (%.1f%% var.)' % lv_var(0), fontsize=11)
    ax.set_ylabel('LV2 (%.1f%% var.)' % lv_var(1), fontsize=11)
    ax.set_title('PLS-DA Scores LV1 vs LV2\n%s' % subtitle,
                 fontsize=10, fontweight='bold')
fig.suptitle(TITLE_SUF + '  |  n=100', fontsize=11, fontweight='bold')
fig.tight_layout()
save_fig(fig, '01_PLSDA_Scores_LV1_LV2.png')

# Fig 02 — LV1 vs LV3
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
for ax, pred, subtitle in zip(
        axes,
        [None,       y_pls_loo],
        ['Full model (re-sub)', 'LOO-CV predictions']):
    scatter_scores(ax, 0, 2, cv_pred=pred)
    ax.set_xlabel('LV1 (%.1f%% var.)' % lv_var(0), fontsize=11)
    ax.set_ylabel('LV3 (%.1f%% var.)' % lv_var(2), fontsize=11)
    ax.set_title('PLS-DA Scores LV1 vs LV3\n%s' % subtitle,
                 fontsize=10, fontweight='bold')
fig.suptitle(TITLE_SUF + '  |  n=100', fontsize=11, fontweight='bold')
fig.tight_layout()
save_fig(fig, '02_PLSDA_Scores_LV1_LV3.png')

# Fig 03 — LV2 vs LV3
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
for ax, pred, subtitle in zip(
        axes,
        [None,       y_pls_5f],
        ['Full model (re-sub)', '5-fold CV predictions']):
    scatter_scores(ax, 1, 2, cv_pred=pred)
    ax.set_xlabel('LV2 (%.1f%% var.)' % lv_var(1), fontsize=11)
    ax.set_ylabel('LV3 (%.1f%% var.)' % lv_var(2), fontsize=11)
    ax.set_title('PLS-DA Scores LV2 vs LV3\n%s' % subtitle,
                 fontsize=10, fontweight='bold')
fig.suptitle(TITLE_SUF + '  |  n=100', fontsize=11, fontweight='bold')
fig.tight_layout()
save_fig(fig, '03_PLSDA_Scores_LV2_LV3.png')

# Fig 04 — 3D scores
fig = plt.figure(figsize=(9, 7))
ax  = fig.add_subplot(111, projection='3d')
for cls in le.classes_:
    idx = classes == cls
    ax.scatter(T_all[idx,0], T_all[idx,1], T_all[idx,2],
               c=PALETTE[cls], marker=MARKER[cls], s=60,
               edgecolors='k', linewidths=0.4, alpha=0.85, label=cls)
ax.set_xlabel('LV1 (%.1f%%)' % lv_var(0), fontsize=9)
ax.set_ylabel('LV2 (%.1f%%)' % lv_var(1), fontsize=9)
ax.set_zlabel('LV3 (%.1f%%)' % lv_var(2), fontsize=9)
ax.set_title('PLS-DA 3D Scores  |  ' + TITLE_SUF, fontsize=10, fontweight='bold')
ax.legend(fontsize=8, ncol=2)
fig.tight_layout()
save_fig(fig, '04_PLSDA_Scores_3D.png')

# =============================================================================
# 15. PLS-DA CONFUSION MATRICES
# =============================================================================
def plot_confusion(cm, labels, title, fname, cmap):
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    ConfusionMatrixDisplay(cm, display_labels=labels).plot(
        ax=ax, colorbar=False, cmap=cmap, xticks_rotation=30)
    ax.set_title(title, fontsize=11, fontweight='bold')
    fig.tight_layout()
    save_fig(fig, fname)

plot_confusion(cm_pls_5f,  le.classes_,
    'PLS-DA  5-fold CV  (n=100)\nAccuracy = %.1f%%' % (acc_pls_5f  * 100),
    '05_PLSDA_Confusion_5fold.png', 'Blues')

plot_confusion(cm_pls_loo, le.classes_,
    'PLS-DA  LOO-CV  (n=100)\nAccuracy = %.1f%%'   % (acc_pls_loo * 100),
    '06_PLSDA_Confusion_LOO.png',   'Purples')

# =============================================================================
# 16. VIP SCORES
# =============================================================================
sel_mask = np.zeros(len(wavenumbers), dtype=bool)
sel_mask[top_idx] = True

fig, ax = plt.subplots(figsize=(11, 4))
ax.plot(wavenumbers, vip_all, color='steelblue', lw=0.9, zorder=2)
ax.fill_between(wavenumbers, vip_all, 1, where=(vip_all > 1),
                alpha=0.25, color='crimson',
                label='VIP > 1  (%d bands)' % n_vip_gt1, zorder=1)
ax.axhline(1, color='crimson', ls='--', lw=1.0, label='VIP = 1 threshold')
ax.fill_between(wavenumbers, 0, 0.06 * vip_all.max(), where=sel_mask,
                alpha=0.20, color='green',
                label='Top-%d selected' % VIP_TOP_N)
ax.set_xlim(WAVENUMBER_MIN, WAVENUMBER_MAX)
ax.set_xlabel('Wavenumber (cm-1)', fontsize=12)
ax.set_ylabel('VIP score', fontsize=12)
ax.set_title('Variable Importance in Projection  |  ' + TITLE_SUF,
             fontsize=11, fontweight='bold')
ax.legend(fontsize=9); ax.grid(True, alpha=0.25)
fig.tight_layout()
save_fig(fig, '07_PLSDA_VIP_Scores.png')

# =============================================================================
# 17. LOADINGS  LV1 / LV2 / LV3
# =============================================================================
fig, axes = plt.subplots(3, 1, figsize=(11, 8), sharex=True)
for i, (ax, col) in enumerate(zip(axes,
                                   ['steelblue', 'darkorange', 'seagreen'])):
    ax.stem(wn_sel, P_load[:, i], linefmt=col, markerfmt=' ', basefmt='k-')
    ax.axhline(0, color='k', lw=0.5)
    ax.set_ylabel('Loading LV%d' % (i+1), fontsize=10)
    ax.set_xlim(WAVENUMBER_MIN, WAVENUMBER_MAX)
    ax.grid(True, alpha=0.2)
    for pk in np.argsort(np.abs(P_load[:, i]))[-5:]:
        ax.annotate('%.0f' % wn_sel[pk],
                    xy=(wn_sel[pk], P_load[pk, i]), fontsize=7,
                    ha='center', xytext=(0, 5), textcoords='offset points')
axes[-1].set_xlabel('Wavenumber (cm-1)', fontsize=12)
axes[0].set_title('PLS-DA Loadings  LV1 / LV2 / LV3  |  top-%d vars  |  %s' % (
    VIP_TOP_N, TITLE_SUF), fontsize=10, fontweight='bold')
fig.tight_layout()
save_fig(fig, '08_PLSDA_Loadings.png')

# =============================================================================
# 18. PERMUTATION TEST
# =============================================================================
fig, ax = plt.subplots(figsize=(7.5, 4.5))
ax.hist(perm_scores, bins=35, color='lightsteelblue',
        edgecolor='white', label='Permuted accuracy')
ax.axvline(obs_score, color='crimson', lw=2.5,
           label='Observed: %.1f%%   p=%.4f' % (obs_score*100, p_val))
ax.axvline(0.25, color='grey', lw=1.2, ls=':',
           label='Chance level (25%)')
ax.set_xlabel('5-fold CV accuracy', fontsize=12)
ax.set_ylabel('Count', fontsize=12)
ax.set_title('Permutation Test (%d perm.)  |  PLS-DA  |  ' % N_PERM + TITLE_SUF,
             fontsize=10, fontweight='bold')
ax.legend(fontsize=9); ax.grid(True, alpha=0.25)
fig.tight_layout()
save_fig(fig, '09_PLSDA_Permutation.png')

# =============================================================================
# 19. SVM-RBF CLUSTER PLOTS  (PCA space)
# =============================================================================
def scatter_cluster_pca(ax, Z, true_cls, cv_pred, title):
    for cls in le.classes_:
        idx = np.where(true_cls == cls)[0]
        for i in idx:
            wrong = (le.transform([true_cls[i]])[0] != cv_pred[i])
            ec    = 'red' if wrong else 'black'
            lw    = 2.0   if wrong else 0.6
            ax.scatter(Z[i, 0], Z[i, 1],
                       c=PALETTE[cls], marker=MARKER[cls], s=75,
                       edgecolors=ec, linewidths=lw, alpha=0.88, zorder=3)

    handles = [plt.scatter([], [], c=PALETTE[c], marker=MARKER[c],
                           s=55, edgecolors='k', label=c)
               for c in le.classes_]
    handles += [
        plt.scatter([], [], c='white', marker='o', s=55,
                    edgecolors='red',   linewidths=2.0, label='Misclassified'),
        plt.scatter([], [], c='white', marker='o', s=55,
                    edgecolors='black', linewidths=0.6, label='Correct'),
    ]
    ax.legend(handles=handles, fontsize=7.5, framealpha=0.85)
    ax.set_xlabel('PC1 (%.1f%%)' % (pca_viz.explained_variance_ratio_[0]*100),
                  fontsize=10)
    ax.set_ylabel('PC2 (%.1f%%)' % (pca_viz.explained_variance_ratio_[1]*100),
                  fontsize=10)
    n_wrong = int((le.transform(true_cls) != cv_pred).sum())
    ax.set_title('%s\n%d misclassified / %d' % (title, n_wrong, len(y)),
                 fontsize=10, fontweight='bold')
    ax.grid(True, alpha=0.25)

# Fig 10 — SVM 5-fold CV cluster
fig, ax = plt.subplots(figsize=(8, 6.5))
scatter_cluster_pca(ax, Z_all, classes, y_svm_5f,
    'SVM-RBF Cluster Plot  |  5-fold CV  |  acc=%.1f%%  |  n=100' % (acc_svm_5f*100))
fig.suptitle(TITLE_SUF, fontsize=10)
fig.tight_layout()
save_fig(fig, '10_SVM_Cluster_5fold.png')

# Fig 11 — SVM LOO-CV cluster
fig, ax = plt.subplots(figsize=(8, 6.5))
scatter_cluster_pca(ax, Z_all, classes, y_svm_loo,
    'SVM-RBF Cluster Plot  |  LOO-CV  |  acc=%.1f%%  |  n=100' % (acc_svm_loo*100))
fig.suptitle(TITLE_SUF, fontsize=10)
fig.tight_layout()
save_fig(fig, '11_SVM_Cluster_LOO.png')

# =============================================================================
# 20. SVM CONFUSION MATRICES
# =============================================================================
plot_confusion(cm_svm_5f,  le.classes_,
    'SVM-RBF  5-fold CV  (n=100)\nAccuracy = %.1f%%' % (acc_svm_5f  * 100),
    '12_SVM_Confusion_5fold.png', 'Blues')

plot_confusion(cm_svm_loo, le.classes_,
    'SVM-RBF  LOO-CV  (n=100)\nAccuracy = %.1f%%'   % (acc_svm_loo * 100),
    '13_SVM_Confusion_LOO.png',   'Purples')

# =============================================================================
# 21. ACCURACY COMPARISON BAR CHART
# =============================================================================
fig, ax = plt.subplots(figsize=(8, 5.5))

methods   = ['5-fold CV', 'LOO-CV']
pls_accs  = [acc_pls_5f * 100, acc_pls_loo * 100]
svm_accs  = [acc_svm_5f * 100, acc_svm_loo * 100]

x = np.arange(len(methods))
w = 0.30
b1 = ax.bar(x - w/2, pls_accs, w, color='steelblue',  edgecolor='k',
            linewidth=0.8, label='PLS-DA')
b2 = ax.bar(x + w/2, svm_accs, w, color='darkorange', edgecolor='k',
            linewidth=0.8, label='SVM-RBF')

for bar in list(b1) + list(b2):
    ax.text(bar.get_x() + bar.get_width()/2,
            bar.get_height() + 0.8,
            '%.1f%%' % bar.get_height(),
            ha='center', va='bottom', fontsize=10, fontweight='bold')

ax.axhline(100, color='grey', ls=':', lw=1)
ax.set_xticks(x)
ax.set_xticklabels(methods, fontsize=12)
ax.set_ylabel('Accuracy (%)', fontsize=12)
ax.set_ylim(0, 118)
ax.set_title('Overall Accuracy  |  PLS-DA vs SVM-RBF\n'
             'All 100 spectra  |  ' + TITLE_SUF,
             fontsize=11, fontweight='bold')
ax.legend(fontsize=11)
ax.grid(True, alpha=0.2, axis='y')
fig.tight_layout()
save_fig(fig, '14_Accuracy_Comparison.png')

# =============================================================================
# 22. PER-CLASS RECALL COMPARISON  (2-panel: 5-fold / LOO)
#     Uses actual per-class counts (dataset is unbalanced)
# =============================================================================
# row sums of confusion matrix = actual counts per class (matches le.classes_ order)
n_per_cls_arr = cm_pls_5f.sum(axis=1)   # same for all CMs since all use same y

fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), sharey=True)
panel_data = [
    ('5-fold CV  (n=100)', cm_pls_5f, cm_svm_5f),
    ('LOO-CV  (n=100)',    cm_pls_loo, cm_svm_loo),
]
x = np.arange(len(le.classes_)); w = 0.30
for ax, (ptitle, cm_pls, cm_svm) in zip(axes, panel_data):
    rec_pls = [cm_pls[i, i] / n_per_cls_arr[i] * 100 for i in range(len(le.classes_))]
    rec_svm = [cm_svm[i, i] / n_per_cls_arr[i] * 100 for i in range(len(le.classes_))]
    b1 = ax.bar(x - w/2, rec_pls, w, color='steelblue',  edgecolor='k',
                linewidth=0.7, label='PLS-DA')
    b2 = ax.bar(x + w/2, rec_svm, w, color='darkorange', edgecolor='k',
                linewidth=0.7, label='SVM-RBF')
    for bar in list(b1) + list(b2):
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + 1.5,
                '%.0f%%' % bar.get_height(),
                ha='center', va='bottom', fontsize=9, fontweight='bold')
    ax.axhline(100, color='grey', ls=':', lw=1)
    ax.set_xticks(x)
    # Show class label + n on x-axis
    xlabels = ['%s\n(n=%d)' % (c.replace('SSY-', ''), n_per_cls_arr[i])
               for i, c in enumerate(le.classes_)]
    ax.set_xticklabels(xlabels, rotation=20, ha='right', fontsize=8)
    ax.set_title(ptitle, fontsize=11, fontweight='bold')
    ax.set_ylim(0, 122)
    ax.grid(True, alpha=0.2, axis='y')
    ax.legend(fontsize=10)

axes[0].set_ylabel('Recall (%)', fontsize=12)
fig.suptitle('Per-Class Recall  |  PLS-DA vs SVM-RBF  |  ' + TITLE_SUF,
             fontsize=12, fontweight='bold')
fig.tight_layout()
save_fig(fig, '15_Recall_Comparison.png')

# =============================================================================
# 23. MEAN SPECTRA  +/- 1 SD
# =============================================================================
fig, ax = plt.subplots(figsize=(10, 4.5))
for cls in le.classes_:
    idx = classes == cls
    m   = X_sm[idx].mean(axis=0)
    s   = X_sm[idx].std(axis=0)
    ax.plot(wavenumbers, m, color=PALETTE[cls], lw=1.5,
            label='%s (n=%d)' % (cls, idx.sum()))
    ax.fill_between(wavenumbers, m - s, m + s,
                    color=PALETTE[cls], alpha=0.12)
ax.set_xlim(WAVENUMBER_MIN, WAVENUMBER_MAX)
ax.set_xlabel('Wavenumber (cm-1)', fontsize=12)
ax.set_ylabel('Intensity (a.u.)', fontsize=12)
ax.set_title('Mean Raman Spectra +/- 1 SD  (all 100 spectra)  |  %d-%d cm-1' % (
    WAVENUMBER_MIN, WAVENUMBER_MAX), fontsize=11, fontweight='bold')
ax.legend(fontsize=9); ax.grid(True, alpha=0.25)
fig.tight_layout()
save_fig(fig, '16_Mean_Spectra.png')

# =============================================================================
# 24. SAVE ALL RESULTS TO EXCEL
# =============================================================================
out_xlsx = os.path.join(OUTPUT_DIR, 'Results_NoSplit_Book2.xlsx')

with pd.ExcelWriter(out_xlsx, engine='openpyxl') as writer:

    # --- Summary ---
    rows = [
        ('Input file',                    FILE_NAME),
        ('Sheet',                         SHEET_NAME),
        ('Total samples',                 len(y)),
        ('Spectral window (cm-1)',        '%d-%d' % (WAVENUMBER_MIN, WAVENUMBER_MAX)),
        ('Spectral points in window',     X_crop.shape[1]),
        ('SG window / poly',              '%d/%d' % (SG_WINDOW, SG_POLY)),
        ('VIP > 1 (initial model)',       n_vip_gt1),
        ('VIP top-N selected',            VIP_TOP_N),
        ('N latent variables (PLS-DA)',   N_COMPONENTS),
        ('', ''),
        ('--- Class counts ---',          ''),
    ]
    for cls in le.classes_:
        rows.append(('  ' + cls, cls_counts[cls]))
    rows += [
        ('', ''),
        ('--- PLS-DA ---',                ''),
        ('Re-substitution accuracy (%)',  '%.2f' % (resub_acc     * 100)),
        ('5-fold CV accuracy (%)',        '%.2f' % (acc_pls_5f    * 100)),
        ('LOO-CV accuracy (%)',           '%.2f' % (acc_pls_loo   * 100)),
        ('Permutation p-value',           '%.4f' % p_val),
        ('Permutation result',
         'Significant (p<0.05)' if p_val < 0.05 else 'NOT significant'),
        ('', ''),
        ('--- SVM-RBF ---',               ''),
        ('5-fold CV accuracy (%)',        '%.2f' % (acc_svm_5f  * 100)),
        ('LOO-CV accuracy (%)',           '%.2f' % (acc_svm_loo * 100)),
        ('SVM kernel',                    'RBF'),
        ('SVM C',                         10),
        ('SVM gamma',                     'scale'),
    ]
    for i, r2 in enumerate(r2x_cum, 1):
        rows.append(('R2X cumulative LV%d (%%)' % i, '%.4f' % (r2 * 100)))
    pd.DataFrame(rows, columns=['Metric', 'Value']).to_excel(
        writer, sheet_name='Summary', index=False)

    # --- Per-class accuracy table ---
    cls_rows = []
    for i, cls in enumerate(le.classes_):
        n = n_per_cls_arr[i]
        cls_rows.append({
            'Class':                 cls,
            'N_samples':             int(n),
            'PLSDA_5fold_Correct':   int(cm_pls_5f[i, i]),
            'PLSDA_5fold_Recall_%':  '%.1f' % (cm_pls_5f[i, i] / n * 100),
            'PLSDA_LOO_Correct':     int(cm_pls_loo[i, i]),
            'PLSDA_LOO_Recall_%':    '%.1f' % (cm_pls_loo[i, i] / n * 100),
            'SVM_5fold_Correct':     int(cm_svm_5f[i, i]),
            'SVM_5fold_Recall_%':    '%.1f' % (cm_svm_5f[i, i] / n * 100),
            'SVM_LOO_Correct':       int(cm_svm_loo[i, i]),
            'SVM_LOO_Recall_%':      '%.1f' % (cm_svm_loo[i, i] / n * 100),
        })
    pd.DataFrame(cls_rows).to_excel(
        writer, sheet_name='Per_Class_Accuracy', index=False)

    # --- All predictions ---
    pd.DataFrame({
        'Label':               col_names,
        'True_Class':          classes,
        'PLSDA_5fold_Pred':    le.inverse_transform(y_pls_5f),
        'PLSDA_5fold_Correct': (classes == le.inverse_transform(y_pls_5f)),
        'PLSDA_LOO_Pred':      le.inverse_transform(y_pls_loo),
        'PLSDA_LOO_Correct':   (classes == le.inverse_transform(y_pls_loo)),
        'SVM_5fold_Pred':      le.inverse_transform(y_svm_5f),
        'SVM_5fold_Correct':   (classes == le.inverse_transform(y_svm_5f)),
        'SVM_LOO_Pred':        le.inverse_transform(y_svm_loo),
        'SVM_LOO_Correct':     (classes == le.inverse_transform(y_svm_loo)),
        'LV1': T_all[:, 0], 'LV2': T_all[:, 1],
        'LV3': T_all[:, 2], 'PC1': Z_all[:, 0], 'PC2': Z_all[:, 1],
    }).to_excel(writer, sheet_name='All_Predictions', index=False)

    # --- Confusion matrices ---
    for sheet, cm in [
        ('CM_PLSDA_5fold',  cm_pls_5f),
        ('CM_PLSDA_LOO',    cm_pls_loo),
        ('CM_SVM_5fold',    cm_svm_5f),
        ('CM_SVM_LOO',      cm_svm_loo),
    ]:
        pd.DataFrame(cm,
                     index=pd.Index(le.classes_, name='True\Predicted'),
                     columns=le.classes_).to_excel(writer, sheet_name=sheet)

    # --- VIP scores ---
    pd.DataFrame({
        'Wavenumber_cm1':    wavenumbers,
        'VIP_Score':         vip_all,
        'VIP_gt_1':          vip_all > 1,
        'Selected_Top%d' % VIP_TOP_N: sel_mask,
    }).to_excel(writer, sheet_name='VIP_Scores', index=False)

    # --- Loadings ---
    load_df = pd.DataFrame(
        P_load,
        columns=['Loading_LV%d' % (i+1) for i in range(N_COMPONENTS)])
    load_df.insert(0, 'Wavenumber_cm1', wn_sel)
    load_df.to_excel(writer, sheet_name='Loadings', index=False)

    # --- Permutation ---
    pd.DataFrame({
        'Permutation_Index': np.arange(1, N_PERM + 1),
        'Permuted_Accuracy': perm_scores,
    }).to_excel(writer, sheet_name='Permutation_Test', index=False)

print('\n[SAVED] %s' % out_xlsx)

# =============================================================================
# 25. FINAL CONSOLE SUMMARY
# =============================================================================
print('\n' + '=' * 65)
print('  FINAL RESULTS SUMMARY  (No train/test split)')
print('=' * 65)
print('  File             : %s  [%s]' % (FILE_NAME, SHEET_NAME))
print('  Samples          : %d  (all used)' % len(y))
print('  Spectral window  : %d-%d cm-1  |  %d points' % (
    WAVENUMBER_MIN, WAVENUMBER_MAX, X_crop.shape[1]))
print('  VIP > 1          : %d / %d   ->  top-%d selected' % (
    n_vip_gt1, len(vip_all), VIP_TOP_N))
print()
print('  Class distribution:')
for cls in le.classes_:
    print('    %-25s : %d' % (cls, cls_counts[cls]))
print()
print('  %-10s  %14s  %14s' % ('MODEL', '5-fold CV', 'LOO-CV'))
print('  ' + '-' * 42)
print('  %-10s  %13.1f%%  %13.1f%%' % ('PLS-DA',  acc_pls_5f*100, acc_pls_loo*100))
print('  %-10s  %13.1f%%  %13.1f%%' % ('SVM-RBF', acc_svm_5f*100, acc_svm_loo*100))
print()
print('  PLS-DA permutation p-value : %.4f  [%s]' % (
    p_val, 'SIGNIFICANT' if p_val < 0.05 else 'NOT significant'))
print()
print('  Per-class recall  (PLS-DA):')
print('  %-25s  %6s  %12s  %10s' % ('Class', 'n', '5-fold CV', 'LOO-CV'))
print('  ' + '-' * 58)
for i, cls in enumerate(le.classes_):
    n = n_per_cls_arr[i]
    print('  %-25s  %4d  %10.0f%%  %8.0f%%' % (
        cls, n,
        cm_pls_5f[i, i] / n * 100,
        cm_pls_loo[i, i] / n * 100))
print()
print('  Per-class recall  (SVM-RBF):')
print('  %-25s  %6s  %12s  %10s' % ('Class', 'n', '5-fold CV', 'LOO-CV'))
print('  ' + '-' * 58)
for i, cls in enumerate(le.classes_):
    n = n_per_cls_arr[i]
    print('  %-25s  %4d  %10.0f%%  %8.0f%%' % (
        cls, n,
        cm_svm_5f[i, i] / n * 100,
        cm_svm_loo[i, i] / n * 100))
print()
print('  R2X cumulative (PLS-DA):')
for i, r2 in enumerate(r2x_cum, 1):
    print('    LV%d: %.2f%%' % (i, r2 * 100))
print()
print('  All outputs saved to : %s' % OUTPUT_DIR)
print('=' * 65)
