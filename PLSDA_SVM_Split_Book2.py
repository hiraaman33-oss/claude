"""
=============================================================================
PLS-DA + SVM-RBF  |  68 TRAIN / 32 TEST SPLIT  |  NO DATA LEAKAGE
SSY Raman Spectroscopy  |  Four Substrate Classes
SPECTRAL WINDOW : 450 - 1800 cm-1

DATASET
  - Book2.xlsx  |  Sheet: "4 - Normalised"
  - 100 spectra  ->  68 train  +  32 test  (stratified, 17+8 per class)
      SSY-PDMS       : 17 train + 8 test
      SSY-SiO2/Si    : 17 train + 8 test
      SSY-Gr-SiO2/Si : 17 train + 8 test
      SSY-Gr-PDMS    : 17 train + 8 test

LEAKAGE PREVENTION (every step)
  - Stratified random split FIRST, before any fitting
  - VIP selection  : 3-LV PLS-DA fitted on TRAIN ONLY -> top-1000 VIP
  - StandardScaler : fitted on TRAIN ONLY, applied to both sets
  - PLS-DA / SVM   : 5-fold CV + LOO-CV on TRAIN ONLY
  - Permutation    : on TRAIN SET only
  - All CV pipelines : StandardScaler fitted inside each fold
  - Test set       : untouched until final evaluation

OUTPUTS -> C:/Users/Hira Aman/Desktop/PROF_DOMENICO'S/PLSDA_SVM_Split_Book2/
  01_PLSDA_Scores_LV1_LV2.png        10_PLSDA_Permutation.png
  02_PLSDA_Scores_LV1_LV3.png        11_SVM_Cluster_5fold_Train.png
  03_PLSDA_Scores_LV2_LV3.png        12_SVM_Cluster_LOO_Train.png
  04_PLSDA_Scores_3D.png             13_SVM_Cluster_Test.png
  05_PLSDA_Confusion_5fold_Train.png  14_SVM_Confusion_5fold_Train.png
  06_PLSDA_Confusion_LOO_Train.png    15_SVM_Confusion_LOO_Train.png
  07_PLSDA_Confusion_Test.png         16_SVM_Confusion_Test.png
  08_PLSDA_VIP_Scores.png             17_Accuracy_Comparison.png
  09_PLSDA_Loadings.png               18_Recall_Comparison.png
                                      19_Mean_Spectra.png
  Results_Split_Book2.xlsx

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
    permutation_test_score, train_test_split
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
VIP_TOP_N      = 1000
N_COMPONENTS   = 5
N_PERM         = 999
RANDOM_STATE   = 42
TEST_SIZE      = 0.32      # 32 test / 68 train

OUTPUT_DIR = os.path.join(DATA_PATH, "PLSDA_SVM_Split_Book2")

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
            "\n[ERROR] Cannot read file: %s\n"
            "Run:  pip install openpyxl\nThen re-run.\nDetails: %s" % (file_path, e))

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
print("\n" + "=" * 68)
print("  PLS-DA + SVM-RBF  |  68 TRAIN / 32 TEST  |  NO DATA LEAKAGE")
print("=" * 68)

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
for cls, cnt in zip(*np.unique(classes, return_counts=True)):
    print("         %-25s : %d" % (cls, int(cnt)))
print("\n[INFO] Class encoding : %s" % str(
    dict(zip(le.classes_, le.transform(le.classes_)))))

# =============================================================================
# 4.  PRE-PROCESSING  (SG smooth — deterministic, no fitted parameters)
# =============================================================================
X_sm = savgol_filter(X_crop, window_length=SG_WINDOW,
                     polyorder=SG_POLY, axis=1)
print("\n[INFO] SG smoothing applied (window=%d, poly=%d)" % (SG_WINDOW, SG_POLY))

# =============================================================================
# 5.  STRATIFIED TRAIN / TEST SPLIT  —  STEP 1 BEFORE ANY FITTING
# =============================================================================
idx_all = np.arange(len(y))
idx_train, idx_test = train_test_split(
    idx_all, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE)
idx_train = np.sort(idx_train)
idx_test  = np.sort(idx_test)

X_train_sm  = X_sm[idx_train]
X_test_sm   = X_sm[idx_test]
y_train     = y[idx_train]
y_test      = y[idx_test]
cls_train   = classes[idx_train]
cls_test    = classes[idx_test]
names_train = [col_names[i] for i in idx_train]
names_test  = [col_names[i] for i in idx_test]

print("\n[SPLIT] Train : %d  |  Test : %d" % (len(y_train), len(y_test)))
print("        %-25s  %6s  %6s" % ("Class", "Train", "Test"))
for cls in le.classes_:
    print("        %-25s  %5d  %5d" % (
        cls, int((cls_train == cls).sum()), int((cls_test == cls).sum())))

# =============================================================================
# 6.  VIP HELPER
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
# 7.  VIP SELECTION — fitted on TRAIN ONLY
# =============================================================================
print("\n[VIP] 3-LV PLS-DA on TRAIN (%d spectra) -> top-%d VIP ..." % (
    len(y_train), VIP_TOP_N))

sc_vip       = StandardScaler().fit(X_train_sm)          # train only
Xs_vip       = sc_vip.transform(X_train_sm)
Yd_vip       = pd.get_dummies(pd.Series(y_train)).values.astype(float)
pls_vip      = PLSRegression(n_components=3, scale=False, max_iter=10000)
pls_vip.fit(Xs_vip, Yd_vip)
vip_all      = compute_vip(pls_vip)

n_vip_gt1    = int((vip_all > 1).sum())
top_idx      = np.argsort(vip_all)[-VIP_TOP_N:]          # indices into wavenumbers
wn_sel       = wavenumbers[top_idx]

X_train_sel  = X_train_sm[:, top_idx]                    # (68, 1000)
X_test_sel   = X_test_sm[:,  top_idx]                    # (32, 1000) — same columns

print("[VIP] VIP > 1 : %d / %d variables" % (n_vip_gt1, len(vip_all)))
print("[VIP] Top-%d selected  (%.1f - %.1f cm-1)" % (
    VIP_TOP_N, wn_sel.min(), wn_sel.max()))

# =============================================================================
# 8.  HELPER CLASSIFIER
# =============================================================================
class PLSDAClf(BaseEstimator, ClassifierMixin):
    """PLS-DA wrapped as sklearn classifier. Scaler lives outside in Pipeline."""
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

cv5 = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
loo = LeaveOneOut()

# =============================================================================
# 9.  FULL PLS-DA ON TRAIN  (for score / loading visualisation only)
# =============================================================================
print("\n[PLS-DA] Fitting full model on TRAIN (%d LVs, top-%d vars) ..." % (
    N_COMPONENTS, VIP_TOP_N))

sc_full      = StandardScaler().fit(X_train_sel)          # train only
X_tr_sc      = sc_full.transform(X_train_sel)
X_te_sc      = sc_full.transform(X_test_sel)              # transform test

Yd_full      = pd.get_dummies(pd.Series(y_train)).values.astype(float)
pls_full     = PLSRegression(n_components=N_COMPONENTS, scale=False, max_iter=10000)
pls_full.fit(X_tr_sc, Yd_full)

T_train      = pls_full.x_scores_                         # (68, 5)
T_test_raw   = pls_full.transform(X_te_sc)
T_test       = T_test_raw[0] if isinstance(T_test_raw, tuple) else T_test_raw
P_load       = pls_full.x_loadings_                       # (1000, 5)

SS_tot  = np.sum(X_tr_sc ** 2)
r2x_cum = []
for lv in range(1, N_COMPONENTS + 1):
    Xrec = pls_full.x_scores_[:, :lv] @ pls_full.x_loadings_[:, :lv].T
    r2x_cum.append(1 - np.sum((X_tr_sc - Xrec) ** 2) / SS_tot)
print("[PLS-DA] R2X cumulative (train) : " + "  ".join(
    ["LV%d=%.1f%%" % (i+1, r*100) for i, r in enumerate(r2x_cum)]))

# =============================================================================
# 10. PLS-DA CV ON TRAIN  (5-fold + LOO)
# =============================================================================
clf_pipe_pls = Pipeline([('sc', StandardScaler()), ('plsda', PLSDAClf(N_COMPONENTS))])

print("\n[PLS-DA] 5-fold CV on TRAIN (%d spectra) ..." % len(y_train))
y_pls_5f    = cross_val_predict(clf_pipe_pls, X_train_sel, y_train, cv=cv5)
acc_pls_5f  = accuracy_score(y_train, y_pls_5f)
cm_pls_5f   = confusion_matrix(y_train, y_pls_5f)
print("[PLS-DA] 5-fold CV accuracy (train) : %.1f%%" % (acc_pls_5f * 100))

print("[PLS-DA] LOO-CV on TRAIN (%d folds) ..." % len(y_train))
y_pls_loo   = cross_val_predict(clf_pipe_pls, X_train_sel, y_train, cv=loo)
acc_pls_loo = accuracy_score(y_train, y_pls_loo)
cm_pls_loo  = confusion_matrix(y_train, y_pls_loo)
print("[PLS-DA] LOO-CV accuracy (train)    : %.1f%%" % (acc_pls_loo * 100))

# =============================================================================
# 11. PLS-DA FINAL EVALUATION ON TEST SET
# =============================================================================
print("\n[PLS-DA] Final evaluation on TEST SET (%d spectra) ..." % len(y_test))
clf_pipe_pls.fit(X_train_sel, y_train)
y_pls_test   = clf_pipe_pls.predict(X_test_sel)
acc_pls_test = accuracy_score(y_test, y_pls_test)
cm_pls_test  = confusion_matrix(y_test, y_pls_test)
print("[PLS-DA] Test set accuracy          : %.1f%%" % (acc_pls_test * 100))

# =============================================================================
# 12. PLS-DA PERMUTATION TEST  (on TRAIN)
# =============================================================================
print("\n[PLS-DA] Permutation test (%d perm., 5-fold CV, TRAIN) ..." % N_PERM)
obs_score, perm_scores, p_val = permutation_test_score(
    clf_pipe_pls, X_train_sel, y_train,
    scoring='accuracy', cv=cv5,
    n_permutations=N_PERM, random_state=RANDOM_STATE, n_jobs=-1)
print("[PLS-DA] Observed : %.2f%%   p = %.4f  [%s]" % (
    obs_score * 100, p_val,
    'SIGNIFICANT (p<0.05)' if p_val < 0.05 else 'NOT significant'))

# =============================================================================
# 13. SVM-RBF CV ON TRAIN  +  TEST EVALUATION
# =============================================================================
svm_pipe = Pipeline([
    ('sc',  StandardScaler()),
    ('svm', SVC(kernel='rbf', C=10, gamma='scale',
                probability=True, random_state=RANDOM_STATE))
])

print("\n[SVM-RBF] 5-fold CV on TRAIN (%d spectra) ..." % len(y_train))
y_svm_5f    = cross_val_predict(svm_pipe, X_train_sel, y_train, cv=cv5)
acc_svm_5f  = accuracy_score(y_train, y_svm_5f)
cm_svm_5f   = confusion_matrix(y_train, y_svm_5f)
print("[SVM-RBF] 5-fold CV accuracy (train) : %.1f%%" % (acc_svm_5f * 100))

print("[SVM-RBF] LOO-CV on TRAIN (%d folds) ..." % len(y_train))
y_svm_loo   = cross_val_predict(svm_pipe, X_train_sel, y_train, cv=loo)
acc_svm_loo = accuracy_score(y_train, y_svm_loo)
cm_svm_loo  = confusion_matrix(y_train, y_svm_loo)
print("[SVM-RBF] LOO-CV accuracy (train)    : %.1f%%" % (acc_svm_loo * 100))

print("\n[SVM-RBF] Final evaluation on TEST SET (%d spectra) ..." % len(y_test))
svm_pipe.fit(X_train_sel, y_train)
y_svm_test   = svm_pipe.predict(X_test_sel)
acc_svm_test = accuracy_score(y_test, y_svm_test)
cm_svm_test  = confusion_matrix(y_test, y_svm_test)
print("[SVM-RBF] Test set accuracy          : %.1f%%" % (acc_svm_test * 100))

# =============================================================================
# 14. PCA FOR CLUSTER VISUALISATION  (fitted on train only)
# =============================================================================
pca_viz = PCA(n_components=2, random_state=RANDOM_STATE)
pca_viz.fit(X_tr_sc)
Z_train = pca_viz.transform(X_tr_sc)
Z_test  = pca_viz.transform(X_te_sc)
print("\n[PCA-viz] PC1=%.1f%%  PC2=%.1f%% explained variance (train)" % (
    pca_viz.explained_variance_ratio_[0] * 100,
    pca_viz.explained_variance_ratio_[1] * 100))

# =============================================================================
# 15. COLOUR / MARKER SCHEME
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
TITLE_SUF = "VIP-top-%d | %d LVs | %d-%d cm-1 | 68 train / 32 test" % (
    VIP_TOP_N, N_COMPONENTS, WAVENUMBER_MIN, WAVENUMBER_MAX)

def lv_var(i):
    return (r2x_cum[i] - (r2x_cum[i-1] if i > 0 else 0)) * 100

def save_fig(fig, fname):
    fig.savefig(os.path.join(OUTPUT_DIR, fname), dpi=150, bbox_inches='tight')
    plt.close(fig)
    print('[SAVED] %s' % fname)

# =============================================================================
# 16. PLS-DA SCORE PLOTS
#     Train = filled markers  |  Test = star markers
# =============================================================================
def scatter_scores_split(ax, lv_x, lv_y, cv_pred_train=None):
    for cls in le.classes_:
        idx = cls_train == cls
        ec = []
        if cv_pred_train is not None:
            for i in np.where(idx)[0]:
                ec.append('red' if y_train[i] != cv_pred_train[i] else 'black')
        else:
            ec = ['black'] * int(idx.sum())
        ax.scatter(T_train[idx, lv_x], T_train[idx, lv_y],
                   c=PALETTE[cls], marker=MARKER[cls], s=70,
                   edgecolors=ec, linewidths=0.8, alpha=0.85,
                   label='Train: %s' % cls, zorder=3)
    for cls in le.classes_:
        idx = cls_test == cls
        ax.scatter(T_test[idx, lv_x], T_test[idx, lv_y],
                   c=PALETTE[cls], marker='*', s=160,
                   edgecolors='k', linewidths=0.6, alpha=0.95,
                   label='Test: %s' % cls, zorder=4)
    ax.axhline(0, color='grey', lw=0.7, ls='--')
    ax.axvline(0, color='grey', lw=0.7, ls='--')
    ax.legend(fontsize=6.5, framealpha=0.85, ncol=2)
    ax.grid(True, alpha=0.25)

# Fig 01
fig, axes = plt.subplots(1, 2, figsize=(15, 6))
for ax, cv_pred, subtitle in zip(axes,
        [None, y_pls_5f],
        ['Full train model', '5-fold CV (train) + test projected']):
    scatter_scores_split(ax, 0, 1, cv_pred_train=cv_pred)
    ax.set_xlabel('LV1 (%.1f%% var.)' % lv_var(0), fontsize=11)
    ax.set_ylabel('LV2 (%.1f%% var.)' % lv_var(1), fontsize=11)
    ax.set_title('PLS-DA Scores LV1 vs LV2\n%s' % subtitle, fontsize=10, fontweight='bold')
fig.suptitle(TITLE_SUF, fontsize=10, fontweight='bold')
fig.tight_layout()
save_fig(fig, '01_PLSDA_Scores_LV1_LV2.png')

# Fig 02
fig, axes = plt.subplots(1, 2, figsize=(15, 6))
for ax, cv_pred, subtitle in zip(axes,
        [None, y_pls_loo],
        ['Full train model', 'LOO-CV (train) + test projected']):
    scatter_scores_split(ax, 0, 2, cv_pred_train=cv_pred)
    ax.set_xlabel('LV1 (%.1f%% var.)' % lv_var(0), fontsize=11)
    ax.set_ylabel('LV3 (%.1f%% var.)' % lv_var(2), fontsize=11)
    ax.set_title('PLS-DA Scores LV1 vs LV3\n%s' % subtitle, fontsize=10, fontweight='bold')
fig.suptitle(TITLE_SUF, fontsize=10, fontweight='bold')
fig.tight_layout()
save_fig(fig, '02_PLSDA_Scores_LV1_LV3.png')

# Fig 03
fig, axes = plt.subplots(1, 2, figsize=(15, 6))
for ax, cv_pred, subtitle in zip(axes,
        [None, y_pls_5f],
        ['Full train model', '5-fold CV (train) + test projected']):
    scatter_scores_split(ax, 1, 2, cv_pred_train=cv_pred)
    ax.set_xlabel('LV2 (%.1f%% var.)' % lv_var(1), fontsize=11)
    ax.set_ylabel('LV3 (%.1f%% var.)' % lv_var(2), fontsize=11)
    ax.set_title('PLS-DA Scores LV2 vs LV3\n%s' % subtitle, fontsize=10, fontweight='bold')
fig.suptitle(TITLE_SUF, fontsize=10, fontweight='bold')
fig.tight_layout()
save_fig(fig, '03_PLSDA_Scores_LV2_LV3.png')

# Fig 04 — 3D
fig = plt.figure(figsize=(9, 7))
ax  = fig.add_subplot(111, projection='3d')
for cls in le.classes_:
    idx = cls_train == cls
    ax.scatter(T_train[idx,0], T_train[idx,1], T_train[idx,2],
               c=PALETTE[cls], marker=MARKER[cls], s=55,
               edgecolors='k', linewidths=0.4, alpha=0.85, label='Train: %s' % cls)
for cls in le.classes_:
    idx = cls_test == cls
    ax.scatter(T_test[idx,0], T_test[idx,1], T_test[idx,2],
               c=PALETTE[cls], marker='*', s=130,
               edgecolors='k', linewidths=0.4, alpha=0.95, label='Test: %s' % cls)
ax.set_xlabel('LV1 (%.1f%%)' % lv_var(0), fontsize=9)
ax.set_ylabel('LV2 (%.1f%%)' % lv_var(1), fontsize=9)
ax.set_zlabel('LV3 (%.1f%%)' % lv_var(2), fontsize=9)
ax.set_title('PLS-DA 3D Scores  |  ' + TITLE_SUF, fontsize=9, fontweight='bold')
ax.legend(fontsize=6.5, ncol=2)
fig.tight_layout()
save_fig(fig, '04_PLSDA_Scores_3D.png')

# =============================================================================
# 17. PLS-DA CONFUSION MATRICES
# =============================================================================
def plot_confusion(cm, labels, title, fname, cmap):
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    ConfusionMatrixDisplay(cm, display_labels=labels).plot(
        ax=ax, colorbar=False, cmap=cmap, xticks_rotation=30)
    ax.set_title(title, fontsize=11, fontweight='bold')
    fig.tight_layout()
    save_fig(fig, fname)

plot_confusion(cm_pls_5f,   le.classes_,
    'PLS-DA  5-fold CV  (TRAIN n=%d)\nAccuracy = %.1f%%' % (len(y_train), acc_pls_5f*100),
    '05_PLSDA_Confusion_5fold_Train.png', 'Blues')

plot_confusion(cm_pls_loo,  le.classes_,
    'PLS-DA  LOO-CV  (TRAIN n=%d)\nAccuracy = %.1f%%' % (len(y_train), acc_pls_loo*100),
    '06_PLSDA_Confusion_LOO_Train.png', 'Purples')

plot_confusion(cm_pls_test, le.classes_,
    'PLS-DA  TEST SET  (n=%d)\nAccuracy = %.1f%%' % (len(y_test), acc_pls_test*100),
    '07_PLSDA_Confusion_Test.png', 'Greens')

# =============================================================================
# 18. VIP SCORES
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
                alpha=0.20, color='green', label='Top-%d selected' % VIP_TOP_N)
ax.set_xlim(WAVENUMBER_MIN, WAVENUMBER_MAX)
ax.set_xlabel('Wavenumber (cm-1)', fontsize=12)
ax.set_ylabel('VIP score', fontsize=12)
ax.set_title('VIP Scores (fitted on TRAIN only)  |  ' + TITLE_SUF,
             fontsize=10, fontweight='bold')
ax.legend(fontsize=9); ax.grid(True, alpha=0.25)
fig.tight_layout()
save_fig(fig, '08_PLSDA_VIP_Scores.png')

# =============================================================================
# 19. LOADINGS  LV1 / LV2 / LV3
# =============================================================================
fig, axes = plt.subplots(3, 1, figsize=(11, 8), sharex=True)
for i, (ax, col) in enumerate(zip(axes, ['steelblue', 'darkorange', 'seagreen'])):
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
axes[0].set_title('PLS-DA Loadings  LV1/LV2/LV3  |  top-%d vars  |  %s' % (
    VIP_TOP_N, TITLE_SUF), fontsize=10, fontweight='bold')
fig.tight_layout()
save_fig(fig, '09_PLSDA_Loadings.png')

# =============================================================================
# 20. PERMUTATION TEST
# =============================================================================
fig, ax = plt.subplots(figsize=(7.5, 4.5))
ax.hist(perm_scores, bins=35, color='lightsteelblue',
        edgecolor='white', label='Permuted accuracy')
ax.axvline(obs_score, color='crimson', lw=2.5,
           label='Observed: %.1f%%   p=%.4f' % (obs_score*100, p_val))
ax.axvline(0.25, color='grey', lw=1.2, ls=':', label='Chance level (25%)')
ax.set_xlabel('5-fold CV accuracy (train)', fontsize=12)
ax.set_ylabel('Count', fontsize=12)
ax.set_title('Permutation Test (%d perm.)  |  TRAIN only  |  ' % N_PERM + TITLE_SUF,
             fontsize=9, fontweight='bold')
ax.legend(fontsize=9); ax.grid(True, alpha=0.25)
fig.tight_layout()
save_fig(fig, '10_PLSDA_Permutation.png')

# =============================================================================
# 21. SVM CLUSTER PLOTS  (PCA space)
# =============================================================================
def scatter_cluster(ax, mode, cv_pred, test_pred, title):
    """mode='train' colours train CV errors; mode='test' colours test errors."""
    if mode == 'train':
        for cls in le.classes_:
            idx = np.where(cls_train == cls)[0]
            for i in idx:
                wrong = (y_train[i] != cv_pred[i])
                ax.scatter(Z_train[i, 0], Z_train[i, 1],
                           c=PALETTE[cls], marker=MARKER[cls], s=70,
                           edgecolors='red' if wrong else 'black',
                           linewidths=2.0 if wrong else 0.6, alpha=0.88, zorder=3)
        for cls in le.classes_:
            idx = cls_test == cls
            ax.scatter(Z_test[idx, 0], Z_test[idx, 1],
                       c=PALETTE[cls], marker='*', s=140,
                       edgecolors='k', linewidths=0.5, alpha=0.95, zorder=4,
                       label='Test: %s' % cls)
    else:
        for cls in le.classes_:
            idx = np.where(cls_train == cls)[0]
            ax.scatter(Z_train[idx, 0], Z_train[idx, 1],
                       c=PALETTE[cls], marker=MARKER[cls], s=60,
                       edgecolors='black', linewidths=0.5, alpha=0.40, zorder=2)
        for cls in le.classes_:
            idx = np.where(cls_test == cls)[0]
            for i in idx:
                wrong = (y_test[i] != test_pred[i])
                ax.scatter(Z_test[i, 0], Z_test[i, 1],
                           c=PALETTE[cls], marker='*', s=160,
                           edgecolors='red' if wrong else 'black',
                           linewidths=2.5 if wrong else 0.8, alpha=0.95, zorder=4)

    handles  = [plt.scatter([], [], c=PALETTE[c], marker=MARKER[c],
                            s=50, edgecolors='k', label='Train: %s' % c)
                for c in le.classes_]
    handles += [plt.scatter([], [], c=PALETTE[c], marker='*',
                            s=80, edgecolors='k', label='Test: %s' % c)
                for c in le.classes_]
    handles += [
        plt.scatter([], [], c='white', marker='o', s=50,
                    edgecolors='red',   linewidths=2.0, label='Misclassified'),
        plt.scatter([], [], c='white', marker='o', s=50,
                    edgecolors='black', linewidths=0.6, label='Correct'),
    ]
    ax.legend(handles=handles, fontsize=7, framealpha=0.85)
    ax.set_xlabel('PC1 (%.1f%%)' % (pca_viz.explained_variance_ratio_[0]*100), fontsize=10)
    ax.set_ylabel('PC2 (%.1f%%)' % (pca_viz.explained_variance_ratio_[1]*100), fontsize=10)
    ax.set_title(title, fontsize=10, fontweight='bold')
    ax.grid(True, alpha=0.25)

fig, ax = plt.subplots(figsize=(8, 6.5))
scatter_cluster(ax, 'train', y_svm_5f, y_svm_test,
    'SVM-RBF  |  5-fold CV (TRAIN n=%d)  acc=%.1f%%\n'
    'Train: red=misclassified  |  Stars=test (projected)' % (len(y_train), acc_svm_5f*100))
fig.suptitle(TITLE_SUF, fontsize=9); fig.tight_layout()
save_fig(fig, '11_SVM_Cluster_5fold_Train.png')

fig, ax = plt.subplots(figsize=(8, 6.5))
scatter_cluster(ax, 'train', y_svm_loo, y_svm_test,
    'SVM-RBF  |  LOO-CV (TRAIN n=%d)  acc=%.1f%%\n'
    'Train: red=misclassified  |  Stars=test (projected)' % (len(y_train), acc_svm_loo*100))
fig.suptitle(TITLE_SUF, fontsize=9); fig.tight_layout()
save_fig(fig, '12_SVM_Cluster_LOO_Train.png')

fig, ax = plt.subplots(figsize=(8, 6.5))
scatter_cluster(ax, 'test', y_svm_5f, y_svm_test,
    'SVM-RBF  |  TEST SET (n=%d)  acc=%.1f%%\n'
    'Stars=test: red=misclassified  |  Faded=train (reference)' % (len(y_test), acc_svm_test*100))
fig.suptitle(TITLE_SUF, fontsize=9); fig.tight_layout()
save_fig(fig, '13_SVM_Cluster_Test.png')

# =============================================================================
# 22. SVM CONFUSION MATRICES
# =============================================================================
plot_confusion(cm_svm_5f,   le.classes_,
    'SVM-RBF  5-fold CV  (TRAIN n=%d)\nAccuracy = %.1f%%' % (len(y_train), acc_svm_5f*100),
    '14_SVM_Confusion_5fold_Train.png', 'Blues')

plot_confusion(cm_svm_loo,  le.classes_,
    'SVM-RBF  LOO-CV  (TRAIN n=%d)\nAccuracy = %.1f%%' % (len(y_train), acc_svm_loo*100),
    '15_SVM_Confusion_LOO_Train.png', 'Purples')

plot_confusion(cm_svm_test, le.classes_,
    'SVM-RBF  TEST SET  (n=%d)\nAccuracy = %.1f%%' % (len(y_test), acc_svm_test*100),
    '16_SVM_Confusion_Test.png', 'Greens')

# =============================================================================
# 23. ACCURACY COMPARISON  (3 evaluation modes)
# =============================================================================
fig, ax = plt.subplots(figsize=(9, 5.5))
methods  = ['5-fold CV\n(train n=68)', 'LOO-CV\n(train n=68)', 'Test set\n(n=32)']
pls_accs = [acc_pls_5f*100, acc_pls_loo*100, acc_pls_test*100]
svm_accs = [acc_svm_5f*100, acc_svm_loo*100, acc_svm_test*100]

x = np.arange(len(methods)); w = 0.30
b1 = ax.bar(x - w/2, pls_accs, w, color='steelblue',  edgecolor='k', linewidth=0.8, label='PLS-DA')
b2 = ax.bar(x + w/2, svm_accs, w, color='darkorange', edgecolor='k', linewidth=0.8, label='SVM-RBF')
for bar in list(b1) + list(b2):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.8,
            '%.1f%%' % bar.get_height(),
            ha='center', va='bottom', fontsize=10, fontweight='bold')
ax.axhline(100, color='grey', ls=':', lw=1)
ax.set_xticks(x); ax.set_xticklabels(methods, fontsize=11)
ax.set_ylabel('Accuracy (%)', fontsize=12); ax.set_ylim(0, 118)
ax.set_title('Overall Accuracy  |  PLS-DA vs SVM-RBF\n' + TITLE_SUF,
             fontsize=10, fontweight='bold')
ax.legend(fontsize=11); ax.grid(True, alpha=0.2, axis='y')
fig.tight_layout()
save_fig(fig, '17_Accuracy_Comparison.png')

# =============================================================================
# 24. PER-CLASS RECALL  (3-panel)
# =============================================================================
n_per_train = cm_pls_5f.sum(axis=1)
n_per_test  = cm_pls_test.sum(axis=1)

fig, axes = plt.subplots(1, 3, figsize=(17, 5.5), sharey=True)
panel_data = [
    ('5-fold CV  (TRAIN n=%d)' % len(y_train), cm_pls_5f,   cm_svm_5f,   n_per_train),
    ('LOO-CV  (TRAIN n=%d)'    % len(y_train), cm_pls_loo,  cm_svm_loo,  n_per_train),
    ('TEST SET  (n=%d)'        % len(y_test),  cm_pls_test, cm_svm_test, n_per_test),
]
x = np.arange(len(le.classes_)); w = 0.30
for ax, (ptitle, cm_p, cm_s, n_per) in zip(axes, panel_data):
    rec_pls = [cm_p[i, i] / n_per[i] * 100 for i in range(len(le.classes_))]
    rec_svm = [cm_s[i, i] / n_per[i] * 100 for i in range(len(le.classes_))]
    b1 = ax.bar(x - w/2, rec_pls, w, color='steelblue',  edgecolor='k', linewidth=0.7, label='PLS-DA')
    b2 = ax.bar(x + w/2, rec_svm, w, color='darkorange', edgecolor='k', linewidth=0.7, label='SVM-RBF')
    for bar in list(b1) + list(b2):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1.5,
                '%.0f%%' % bar.get_height(),
                ha='center', va='bottom', fontsize=9, fontweight='bold')
    ax.axhline(100, color='grey', ls=':', lw=1)
    ax.set_xticks(x)
    ax.set_xticklabels(['%s\n(n=%d)' % (c.replace('SSY-', ''), n_per[i])
                        for i, c in enumerate(le.classes_)],
                       rotation=20, ha='right', fontsize=8)
    ax.set_title(ptitle, fontsize=10, fontweight='bold')
    ax.set_ylim(0, 122); ax.grid(True, alpha=0.2, axis='y'); ax.legend(fontsize=9)
axes[0].set_ylabel('Recall (%)', fontsize=12)
fig.suptitle('Per-Class Recall  |  PLS-DA vs SVM-RBF  |  ' + TITLE_SUF,
             fontsize=11, fontweight='bold')
fig.tight_layout()
save_fig(fig, '18_Recall_Comparison.png')

# =============================================================================
# 25. MEAN SPECTRA  (all 100 spectra for reference)
# =============================================================================
fig, ax = plt.subplots(figsize=(10, 4.5))
for cls in le.classes_:
    idx = classes == cls
    m, s = X_sm[idx].mean(axis=0), X_sm[idx].std(axis=0)
    ax.plot(wavenumbers, m, color=PALETTE[cls], lw=1.5,
            label='%s (n=%d)' % (cls, idx.sum()))
    ax.fill_between(wavenumbers, m - s, m + s, color=PALETTE[cls], alpha=0.12)
ax.set_xlim(WAVENUMBER_MIN, WAVENUMBER_MAX)
ax.set_xlabel('Wavenumber (cm-1)', fontsize=12)
ax.set_ylabel('Intensity (a.u.)', fontsize=12)
ax.set_title('Mean Raman Spectra +/- 1 SD  (all 100 spectra)  |  %d-%d cm-1' % (
    WAVENUMBER_MIN, WAVENUMBER_MAX), fontsize=11, fontweight='bold')
ax.legend(fontsize=9); ax.grid(True, alpha=0.25)
fig.tight_layout()
save_fig(fig, '19_Mean_Spectra.png')

# =============================================================================
# 26. EXCEL RESULTS
# =============================================================================
out_xlsx = os.path.join(OUTPUT_DIR, 'Results_Split_Book2.xlsx')

with pd.ExcelWriter(out_xlsx, engine='openpyxl') as writer:

    # Summary
    rows = [
        ('Input file',                   FILE_NAME),
        ('Sheet',                        SHEET_NAME),
        ('Total samples',                len(y)),
        ('Train samples',                len(y_train)),
        ('Test samples',                 len(y_test)),
        ('Spectral window (cm-1)',       '%d-%d' % (WAVENUMBER_MIN, WAVENUMBER_MAX)),
        ('Spectral points in window',    X_crop.shape[1]),
        ('SG window / poly',             '%d/%d' % (SG_WINDOW, SG_POLY)),
        ('VIP > 1 (train model)',        n_vip_gt1),
        ('VIP top-N selected',           VIP_TOP_N),
        ('N latent variables (PLS-DA)',  N_COMPONENTS),
        ('Random state',                 RANDOM_STATE),
        ('', ''),
        ('--- Class distribution ---',   ''),
    ]
    for cls in le.classes_:
        rows.append(('  %s  (train/test)' % cls,
                     '%d / %d' % ((cls_train==cls).sum(), (cls_test==cls).sum())))
    rows += [
        ('', ''),
        ('--- PLS-DA ---',                    ''),
        ('5-fold CV accuracy % (train)',      '%.2f' % (acc_pls_5f  *100)),
        ('LOO-CV accuracy % (train)',         '%.2f' % (acc_pls_loo *100)),
        ('Test set accuracy %',              '%.2f' % (acc_pls_test*100)),
        ('Permutation p-value (train)',       '%.4f' % p_val),
        ('Permutation result',
         'Significant (p<0.05)' if p_val < 0.05 else 'NOT significant'),
        ('', ''),
        ('--- SVM-RBF ---',                   ''),
        ('5-fold CV accuracy % (train)',      '%.2f' % (acc_svm_5f  *100)),
        ('LOO-CV accuracy % (train)',         '%.2f' % (acc_svm_loo *100)),
        ('Test set accuracy %',              '%.2f' % (acc_svm_test*100)),
        ('SVM kernel',                        'RBF'),
        ('SVM C',                             10),
        ('SVM gamma',                         'scale'),
    ]
    for i, r2 in enumerate(r2x_cum, 1):
        rows.append(('R2X cumulative LV%d (%%)' % i, '%.4f' % (r2*100)))
    pd.DataFrame(rows, columns=['Metric', 'Value']).to_excel(
        writer, sheet_name='Summary', index=False)

    # Per-class accuracy
    cls_rows = []
    for i, cls in enumerate(le.classes_):
        nt, ne = n_per_train[i], n_per_test[i]
        cls_rows.append({
            'Class':               cls,
            'N_train':             int(nt),
            'N_test':              int(ne),
            'PLSDA_5fold_train_%': '%.1f' % (cm_pls_5f[i,i]  /nt*100),
            'PLSDA_LOO_train_%':   '%.1f' % (cm_pls_loo[i,i] /nt*100),
            'PLSDA_Test_%':        '%.1f' % (cm_pls_test[i,i]/ne*100),
            'SVM_5fold_train_%':   '%.1f' % (cm_svm_5f[i,i]  /nt*100),
            'SVM_LOO_train_%':     '%.1f' % (cm_svm_loo[i,i] /nt*100),
            'SVM_Test_%':          '%.1f' % (cm_svm_test[i,i]/ne*100),
        })
    pd.DataFrame(cls_rows).to_excel(writer, sheet_name='Per_Class_Accuracy', index=False)

    # Train predictions
    pd.DataFrame({
        'Label':               names_train,
        'True_Class':          cls_train,
        'PLSDA_5fold_Pred':    le.inverse_transform(y_pls_5f),
        'PLSDA_5fold_Correct': cls_train == le.inverse_transform(y_pls_5f),
        'PLSDA_LOO_Pred':      le.inverse_transform(y_pls_loo),
        'PLSDA_LOO_Correct':   cls_train == le.inverse_transform(y_pls_loo),
        'SVM_5fold_Pred':      le.inverse_transform(y_svm_5f),
        'SVM_5fold_Correct':   cls_train == le.inverse_transform(y_svm_5f),
        'SVM_LOO_Pred':        le.inverse_transform(y_svm_loo),
        'SVM_LOO_Correct':     cls_train == le.inverse_transform(y_svm_loo),
        'LV1': T_train[:,0], 'LV2': T_train[:,1],
        'LV3': T_train[:,2], 'PC1': Z_train[:,0], 'PC2': Z_train[:,1],
    }).to_excel(writer, sheet_name='Train_Predictions', index=False)

    # Test predictions
    pd.DataFrame({
        'Label':              names_test,
        'True_Class':         cls_test,
        'PLSDA_Test_Pred':    le.inverse_transform(y_pls_test),
        'PLSDA_Test_Correct': cls_test == le.inverse_transform(y_pls_test),
        'SVM_Test_Pred':      le.inverse_transform(y_svm_test),
        'SVM_Test_Correct':   cls_test == le.inverse_transform(y_svm_test),
        'LV1': T_test[:,0], 'LV2': T_test[:,1],
        'LV3': T_test[:,2], 'PC1': Z_test[:,0], 'PC2': Z_test[:,1],
    }).to_excel(writer, sheet_name='Test_Predictions', index=False)

    # Confusion matrices
    for sheet, cm in [
        ('CM_PLSDA_5fold_Train', cm_pls_5f),
        ('CM_PLSDA_LOO_Train',   cm_pls_loo),
        ('CM_PLSDA_Test',        cm_pls_test),
        ('CM_SVM_5fold_Train',   cm_svm_5f),
        ('CM_SVM_LOO_Train',     cm_svm_loo),
        ('CM_SVM_Test',          cm_svm_test),
    ]:
        pd.DataFrame(cm,
                     index=pd.Index(le.classes_, name='True\\Predicted'),
                     columns=le.classes_).to_excel(writer, sheet_name=sheet)

    # VIP scores
    pd.DataFrame({
        'Wavenumber_cm1':         wavenumbers,
        'VIP_Score':              vip_all,
        'VIP_gt_1':               vip_all > 1,
        'Selected_Top%d' % VIP_TOP_N: sel_mask,
    }).to_excel(writer, sheet_name='VIP_Scores', index=False)

    # Loadings
    load_df = pd.DataFrame(P_load,
        columns=['Loading_LV%d' % (i+1) for i in range(N_COMPONENTS)])
    load_df.insert(0, 'Wavenumber_cm1', wn_sel)
    load_df.to_excel(writer, sheet_name='Loadings', index=False)

    # Permutation
    pd.DataFrame({
        'Permutation_Index': np.arange(1, N_PERM + 1),
        'Permuted_Accuracy': perm_scores,
    }).to_excel(writer, sheet_name='Permutation_Test', index=False)

    # Train/test split record
    pd.DataFrame({
        'Sample_Label': col_names,
        'Class':        classes,
        'Split':        ['Train' if i in set(idx_train) else 'Test'
                         for i in range(len(y))],
    }).to_excel(writer, sheet_name='TrainTest_Split', index=False)

print('\n[SAVED] %s' % out_xlsx)

# =============================================================================
# 27. FINAL CONSOLE SUMMARY
# =============================================================================
print('\n' + '=' * 68)
print('  FINAL RESULTS  |  68 Train / 32 Test  |  No Data Leakage')
print('=' * 68)
print('  File            : %s  [%s]' % (FILE_NAME, SHEET_NAME))
print('  Samples         : %d total  (Train=%d, Test=%d)' % (
    len(y), len(y_train), len(y_test)))
print('  Spectral window : %d-%d cm-1  |  %d points' % (
    WAVENUMBER_MIN, WAVENUMBER_MAX, X_crop.shape[1]))
print('  VIP > 1 (train) : %d / %d  ->  top-%d selected' % (
    n_vip_gt1, len(vip_all), VIP_TOP_N))
print()
print('  Class distribution (train / test):')
for cls in le.classes_:
    print('    %-25s : %d / %d' % (
        cls, (cls_train==cls).sum(), (cls_test==cls).sum()))
print()
print('  %-10s  %18s  %16s  %12s' % (
    'MODEL', '5-fold CV (train)', 'LOO-CV (train)', 'Test set'))
print('  ' + '-' * 62)
print('  %-10s  %17.1f%%  %15.1f%%  %11.1f%%' % (
    'PLS-DA', acc_pls_5f*100, acc_pls_loo*100, acc_pls_test*100))
print('  %-10s  %17.1f%%  %15.1f%%  %11.1f%%' % (
    'SVM-RBF', acc_svm_5f*100, acc_svm_loo*100, acc_svm_test*100))
print()
print('  Permutation p-value (train) : %.4f  [%s]' % (
    p_val, 'SIGNIFICANT' if p_val < 0.05 else 'NOT significant'))
print()
print('  Per-class recall  (PLS-DA):')
print('  %-25s  %6s  %6s  %14s  %12s  %10s' % (
    'Class', 'nTrain', 'nTest', '5-fold CV', 'LOO-CV', 'Test'))
print('  ' + '-' * 76)
for i, cls in enumerate(le.classes_):
    nt, ne = n_per_train[i], n_per_test[i]
    print('  %-25s  %5d  %5d  %13.0f%%  %11.0f%%  %8.0f%%' % (
        cls, nt, ne,
        cm_pls_5f[i,i]/nt*100, cm_pls_loo[i,i]/nt*100, cm_pls_test[i,i]/ne*100))
print()
print('  Per-class recall  (SVM-RBF):')
print('  %-25s  %6s  %6s  %14s  %12s  %10s' % (
    'Class', 'nTrain', 'nTest', '5-fold CV', 'LOO-CV', 'Test'))
print('  ' + '-' * 76)
for i, cls in enumerate(le.classes_):
    nt, ne = n_per_train[i], n_per_test[i]
    print('  %-25s  %5d  %5d  %13.0f%%  %11.0f%%  %8.0f%%' % (
        cls, nt, ne,
        cm_svm_5f[i,i]/nt*100, cm_svm_loo[i,i]/nt*100, cm_svm_test[i,i]/ne*100))
print()
print('  R2X cumulative (PLS-DA, train):')
for i, r2 in enumerate(r2x_cum, 1):
    print('    LV%d: %.2f%%' % (i, r2*100))
print()
print('  All outputs saved to : %s' % OUTPUT_DIR)
print('=' * 68)
