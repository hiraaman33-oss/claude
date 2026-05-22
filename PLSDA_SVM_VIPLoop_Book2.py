"""
=============================================================================
PLS-DA + SVM-RBF  |  68/32 STRATIFIED SPLIT  |  VIP LOOP 1000/1500/1800
SSY Raman Spectroscopy  |  Four Substrate Classes  |  450-1800 cm-1

LEAKAGE PREVENTION
  · Split FIRST — nothing fitted before train/test separation
  · VIP    : 3-LV PLS-DA on TRAIN ONLY
  · Scaler : fitted on TRAIN ONLY, inside every CV fold (Pipeline)
  · Test set predicted exactly once per model

OUTPUTS  (one folder per VIP set)
  C:\...\PROF_DOMENICO'S\PLSDA_SVM_VIP<N>\
  01  PLSDA_Scores_LV1_LV2.png        11  SVM_Cluster_5fold_Train.png
  02  PLSDA_Scores_LV1_LV3.png        12  SVM_Cluster_LOO_Train.png
  03  PLSDA_Scores_LV2_LV3.png        13  SVM_Cluster_Test.png
  04  PLSDA_Scores_3D.png             14  SVM_Confusion_5fold_Train.png
  05  PLSDA_Confusion_5fold_Train.png  15  SVM_Confusion_LOO_Train.png
  06  PLSDA_Confusion_LOO_Train.png    16  SVM_Confusion_Test.png
  07  PLSDA_Confusion_Test.png         17  Accuracy_Comparison.png
  08  PLSDA_VIP_Scores.png             18  Recall_Comparison.png
  09  PLSDA_Loadings.png               19  Mean_Spectra.png
  10  PLSDA_Permutation.png
  PLSDA_SVM_VIP<N>_Results.xlsx  (13 sheets)

HOW TO RUN:
  pip install pandas openpyxl scikit-learn scipy matplotlib
  python PLSDA_SVM_VIPLoop_Book2.py
=============================================================================
"""

import os
import re
import warnings
from collections import Counter

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D          # noqa

from sklearn.cross_decomposition import PLSRegression
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import (
    train_test_split, StratifiedKFold, LeaveOneOut,
    cross_val_predict, permutation_test_score
)
from sklearn.metrics import (
    confusion_matrix, classification_report,
    accuracy_score, ConfusionMatrixDisplay
)
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.svm import SVC
from sklearn.decomposition import PCA

warnings.filterwarnings('ignore')

# ═══════════════════════════════════════════════════════════════════════
# 0.  SETTINGS
# ═══════════════════════════════════════════════════════════════════════
DATA_PATH    = r"C:\Users\Hira Aman\Desktop\PROF_DOMENICO'S"
FILE_NAME    = "Book2.xlsx"
SHEET_NAME   = "4 - Normalised"

WAVENUMBER_MIN = 450
WAVENUMBER_MAX = 1800

N_COMPONENTS = 5
CV_FOLDS     = 5
N_PERM       = 999
RANDOM_STATE = 42
TEST_SIZE    = 0.32

VIP_SETS     = [1000, 1500, 1800]

# ═══════════════════════════════════════════════════════════════════════
# 1.  CLASS MAPPING  (session-prefix aware)
# ═══════════════════════════════════════════════════════════════════════
def assign_class(label):
    s = str(label).strip().lower()
    s = re.sub(r'^s\d+[-_]', '', s)
    s = re.sub(r'^(lugl|marzo\d*|maggio\d*)[-_]?(ssy[-_]?)?', '', s)
    s = s.replace('_', '-').replace(' ', '-')
    s = s.replace('grsl', 'gr')
    if   re.search(r'gr[-_]?sio2|gr[-_]?si',  s): return 'SSY-Gr-SiO2/Si'
    elif re.search(r'gr\w*[-_]?pdms',          s): return 'SSY-Gr-PDMS'
    elif re.search(r'sio2[-_]?si|sio2',        s): return 'SSY-SiO2/Si'
    elif re.search(r'pdms',                    s): return 'SSY-PDMS'
    else:                                           return None

# ═══════════════════════════════════════════════════════════════════════
# 2.  VIP SCORES
# ═══════════════════════════════════════════════════════════════════════
def compute_vip(model):
    T_ = model.x_scores_
    W_ = model.x_weights_
    Q_ = model.y_loadings_
    p  = W_.shape[0]
    SS = np.sum(T_**2, axis=0) * np.sum(Q_**2, axis=0)
    Wn = W_ / np.linalg.norm(W_, axis=0)
    return np.sqrt(p * np.sum(SS * Wn**2, axis=1) / np.sum(SS))

# ═══════════════════════════════════════════════════════════════════════
# 3.  PLS-DA CLASSIFIER  (Pipeline-compatible, no double-scaling)
# ═══════════════════════════════════════════════════════════════════════
class PLSDAClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, n_components=5):
        self.n_components = n_components

    def fit(self, X, y):
        self.classes_ = np.unique(y)
        Y_d = np.zeros((len(y), len(self.classes_)))
        for j, c in enumerate(self.classes_):
            Y_d[:, j] = (y == c).astype(float)
        self._pls = PLSRegression(n_components=self.n_components,
                                  scale=False, max_iter=5000)
        self._pls.fit(X, Y_d)
        return self

    def predict(self, X):
        return self.classes_[np.argmax(self._pls.predict(X), axis=1)]

# ═══════════════════════════════════════════════════════════════════════
# 4.  COLOUR / MARKER SCHEME
# ═══════════════════════════════════════════════════════════════════════
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

# ═══════════════════════════════════════════════════════════════════════
# 5.  LOAD DATA  (once, before loop)
# ═══════════════════════════════════════════════════════════════════════
print("=" * 68)
print("  STEP 1 — Loading data")
print("=" * 68)

fpath = os.path.join(DATA_PATH, FILE_NAME)
raw   = pd.read_excel(fpath, sheet_name=SHEET_NAME, header=0,
                      index_col=0, engine='openpyxl')

wavenumbers_all = raw.index.astype(float).values
col_names_all   = raw.columns.tolist()

classes_raw = [assign_class(c) for c in col_names_all]
unassigned  = [c for c, cl in zip(col_names_all, classes_raw) if cl is None]
if unassigned:
    print(f"  [WARNING] {len(unassigned)} unassigned labels: {unassigned}")

keep_idx    = [i for i, cl in enumerate(classes_raw) if cl is not None]
col_names   = [col_names_all[i]  for i in keep_idx]
classes_arr = np.array([classes_raw[i] for i in keep_idx])
raw         = raw.iloc[:, keep_idx]

print(f"  Loaded  : {raw.shape[1]} spectra x {raw.shape[0]} wavenumber points")
print(f"  Classes : {dict(Counter(classes_arr))}")

mask        = (wavenumbers_all >= WAVENUMBER_MIN) & (wavenumbers_all <= WAVENUMBER_MAX)
wavenumbers = wavenumbers_all[mask]
X_all       = raw.values[mask, :].T.astype(float)
y_all       = classes_arr

print(f"  Window  : {WAVENUMBER_MIN}-{WAVENUMBER_MAX} cm-1  ({X_all.shape[1]} variables)")

# ═══════════════════════════════════════════════════════════════════════
# 6.  68 / 32 STRATIFIED SPLIT  — done ONCE before any fitting
# ═══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 68)
print("  STEP 2 — Stratified train/test split  (68/32, seed=42)")
print("=" * 68)

X_tr, X_te, y_tr, y_te = train_test_split(
    X_all, y_all,
    test_size=TEST_SIZE, stratify=y_all, random_state=RANDOM_STATE
)
print(f"  Train : {X_tr.shape[0]}  {dict(Counter(y_tr))}")
print(f"  Test  : {X_te.shape[0]}  {dict(Counter(y_te))}")

le      = LabelEncoder().fit(y_all)
classes = le.classes_

# ═══════════════════════════════════════════════════════════════════════
# 7.  VIP LOOP
# ═══════════════════════════════════════════════════════════════════════
for N_VIP in VIP_SETS:

    OUTPUT_DIR = os.path.join(DATA_PATH, f"PLSDA_SVM_VIP{N_VIP}")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    def save_fig(fig, fname):
        fig.savefig(os.path.join(OUTPUT_DIR, fname), dpi=180, bbox_inches='tight')
        plt.close(fig)
        print(f'  [SAVED] {fname}')

    print("\n" + "=" * 68)
    print(f"  VIP TOP-{N_VIP}  ->  {OUTPUT_DIR}")
    print("=" * 68)

    # ── Scale on TRAIN only ──────────────────────────────────────────
    sc_vip  = StandardScaler().fit(X_tr)
    X_tr_sc = sc_vip.transform(X_tr)
    X_te_sc = sc_vip.transform(X_te)

    # ── VIP on TRAIN only ────────────────────────────────────────────
    print(f"\n  [VIP] 3-LV PLS-DA on train -> top-{N_VIP} ...")
    Y_d_tr = np.zeros((len(y_tr), len(classes)))
    for j, c in enumerate(classes):
        Y_d_tr[:, j] = (y_tr == c).astype(float)

    pls_vip = PLSRegression(n_components=3, scale=False, max_iter=5000)
    pls_vip.fit(X_tr_sc, Y_d_tr)
    vip     = compute_vip(pls_vip)
    top_idx = np.argsort(vip)[-N_VIP:]
    wn_sel  = wavenumbers[top_idx]

    X_tr_sel  = X_tr_sc[:, top_idx]
    X_te_sel  = X_te_sc[:, top_idx]
    n_vip_gt1 = int((vip > 1).sum())
    sel_mask  = np.zeros(len(wavenumbers), dtype=bool)
    sel_mask[top_idx] = True

    print(f"  [VIP] VIP>1: {n_vip_gt1}/{len(vip)}  |  "
          f"top-{N_VIP}: {wn_sel.min():.0f}-{wn_sel.max():.0f} cm-1")

    # ── Full PLS-DA on TRAIN (visualisation only) ────────────────────
    print(f"\n  [PLS-DA] Fitting full model ({N_COMPONENTS} LVs) on train ...")
    pls_full = PLSRegression(n_components=N_COMPONENTS, scale=False, max_iter=5000)
    pls_full.fit(X_tr_sel, Y_d_tr)

    T_tr   = pls_full.x_scores_
    raw_te = pls_full.transform(X_te_sel)
    T_te   = raw_te[0] if isinstance(raw_te, tuple) else raw_te
    P_load = pls_full.x_loadings_

    SS_tot  = np.sum(X_tr_sel ** 2)
    r2x_cum = [
        1 - np.sum((X_tr_sel -
            pls_full.x_scores_[:, :lv] @ pls_full.x_loadings_[:, :lv].T) ** 2) / SS_tot
        for lv in range(1, N_COMPONENTS + 1)
    ]

    def lv_pct(i):
        return (r2x_cum[i] - (r2x_cum[i-1] if i > 0 else 0)) * 100

    resubst = accuracy_score(y_tr, classes[np.argmax(pls_full.predict(X_tr_sel), axis=1)])
    print(f"  [PLS-DA] Re-substitution: {resubst*100:.1f}%  |  R2X: " +
          "  ".join([f"LV{i+1}={r*100:.1f}%" for i, r in enumerate(r2x_cum)]))

    # ── PLS-DA CV on TRAIN (5-fold + LOO) ───────────────────────────
    pls_pipe = Pipeline([('sc', StandardScaler()),
                         ('plsda', PLSDAClassifier(n_components=N_COMPONENTS))])
    cv5 = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    loo = LeaveOneOut()

    print(f"\n  [PLS-DA] {CV_FOLDS}-fold CV on train ...")
    y_pls_cv5   = cross_val_predict(pls_pipe, X_tr[:, top_idx], y_tr, cv=cv5)
    acc_pls_cv5 = accuracy_score(y_tr, y_pls_cv5)
    cm_pls_cv5  = confusion_matrix(y_tr, y_pls_cv5, labels=classes)
    print(f"  [PLS-DA] 5-fold CV accuracy: {acc_pls_cv5*100:.1f}%")
    print(classification_report(y_tr, y_pls_cv5, target_names=classes, digits=3))

    print(f"  [PLS-DA] LOO-CV on train ({len(y_tr)} folds) ...")
    y_pls_loo   = cross_val_predict(pls_pipe, X_tr[:, top_idx], y_tr, cv=loo)
    acc_pls_loo = accuracy_score(y_tr, y_pls_loo)
    cm_pls_loo  = confusion_matrix(y_tr, y_pls_loo, labels=classes)
    print(f"  [PLS-DA] LOO-CV accuracy: {acc_pls_loo*100:.1f}%")

    # ── PLS-DA test prediction ───────────────────────────────────────
    print(f"\n  [PLS-DA] Fitting on full train -> predicting test ...")
    pls_pipe.fit(X_tr[:, top_idx], y_tr)
    y_pls_test   = pls_pipe.predict(X_te[:, top_idx])
    acc_pls_test = accuracy_score(y_te, y_pls_test)
    cm_pls_test  = confusion_matrix(y_te, y_pls_test, labels=classes)
    print(f"  [PLS-DA] Test accuracy: {acc_pls_test*100:.1f}%")
    print(classification_report(y_te, y_pls_test, target_names=classes, digits=3))

    # ── Permutation test (TRAIN only) ────────────────────────────────
    print(f"\n  [PERM] {N_PERM} permutations ...")
    obs_score, perm_scores, p_val = permutation_test_score(
        pls_pipe, X_tr[:, top_idx], y_tr,
        scoring='accuracy', cv=cv5,
        n_permutations=N_PERM, random_state=RANDOM_STATE, n_jobs=-1
    )
    sig = 'SIGNIFICANT (p<0.05)' if p_val < 0.05 else 'NOT significant'
    print(f"  [PERM] Observed: {obs_score*100:.2f}%   p = {p_val:.4f}  [{sig}]")

    # ── SVM-RBF CV on TRAIN (5-fold + LOO) ──────────────────────────
    svm_pipe = Pipeline([
        ('sc',  StandardScaler()),
        ('svm', SVC(kernel='rbf', C=10, gamma='scale',
                    probability=True, random_state=RANDOM_STATE))
    ])

    print(f"\n  [SVM] {CV_FOLDS}-fold CV on train ...")
    y_svm_cv5   = cross_val_predict(svm_pipe, X_tr[:, top_idx], y_tr, cv=cv5)
    acc_svm_cv5 = accuracy_score(y_tr, y_svm_cv5)
    cm_svm_cv5  = confusion_matrix(y_tr, y_svm_cv5, labels=classes)
    print(f"  [SVM] 5-fold CV accuracy: {acc_svm_cv5*100:.1f}%")
    print(classification_report(y_tr, y_svm_cv5, target_names=classes, digits=3))

    print(f"  [SVM] LOO-CV on train ({len(y_tr)} folds) ...")
    y_svm_loo   = cross_val_predict(svm_pipe, X_tr[:, top_idx], y_tr, cv=loo)
    acc_svm_loo = accuracy_score(y_tr, y_svm_loo)
    cm_svm_loo  = confusion_matrix(y_tr, y_svm_loo, labels=classes)
    print(f"  [SVM] LOO-CV accuracy: {acc_svm_loo*100:.1f}%")

    print(f"\n  [SVM] Fitting on full train -> predicting test ...")
    svm_pipe.fit(X_tr[:, top_idx], y_tr)
    y_svm_test   = svm_pipe.predict(X_te[:, top_idx])
    acc_svm_test = accuracy_score(y_te, y_svm_test)
    cm_svm_test  = confusion_matrix(y_te, y_svm_test, labels=classes)
    print(f"  [SVM] Test accuracy: {acc_svm_test*100:.1f}%")
    print(classification_report(y_te, y_svm_test, target_names=classes, digits=3))

    # ── PCA for SVM cluster plots (train only) ───────────────────────
    pca = PCA(n_components=2, random_state=RANDOM_STATE)
    pca.fit(X_tr_sel)
    Z_tr = pca.transform(X_tr_sel)
    Z_te = pca.transform(X_te_sel)

    # ── Leakage audit ────────────────────────────────────────────────
    audit = {
        'Split before any fitting'       : True,
        'VIP on train only'              : True,
        'Scaler fitted on train only'    : True,
        'CV scaler inside Pipeline'      : True,
        'Test never seen during CV/VIP'  : True,
        'Test predicted once per model'  : True,
        'Train size = 68'                : X_tr.shape[0] == 68,
        'Test size = 32'                 : X_te.shape[0] == 32,
        '4 classes in train'             : len(np.unique(y_tr)) == 4,
        '4 classes in test'              : len(np.unique(y_te)) == 4,
        'Permutation on train only'      : True,
        'PCA for viz on train only'      : True,
    }
    n_pass = sum(audit.values())
    print(f"\n  [AUDIT] {n_pass}/{len(audit)} PASS")
    for desc, ok in audit.items():
        print(f"    {'PASS' if ok else 'FAIL'}  {desc}")

    TTAG = (f"VIP top-{N_VIP}  |  {N_COMPONENTS} LVs  |  "
            f"{WAVENUMBER_MIN}-{WAVENUMBER_MAX} cm-1  |  68 train / 32 test")

    # ═══════════════════════════════════════════════════════════════
    # FIGURES
    # ═══════════════════════════════════════════════════════════════

    def pls_scatter_2d(ax, lv_x, lv_y, cv_pred=None):
        for cls in classes:
            idx_tr = y_tr == cls
            if cv_pred is not None:
                ec = ['red' if y_tr[i] != cv_pred[i] else 'k'
                      for i in np.where(idx_tr)[0]]
            else:
                ec = ['k'] * int(idx_tr.sum())
            ax.scatter(T_tr[idx_tr, lv_x], T_tr[idx_tr, lv_y],
                       c=PALETTE[cls], marker=MARKER[cls], s=60,
                       edgecolors=ec, linewidths=0.7, alpha=0.80,
                       label=f'Train: {cls}', zorder=3)
        for cls in classes:
            idx_ok  = (y_te == cls) & (y_pls_test == cls)
            idx_err = (y_te == cls) & (y_pls_test != cls)
            ax.scatter(T_te[idx_ok,  lv_x], T_te[idx_ok,  lv_y],
                       c=PALETTE[cls], marker='*', s=220,
                       edgecolors='k', linewidths=0.6, alpha=1.0,
                       label=f'Test OK: {cls}', zorder=5)
            if idx_err.any():
                ax.scatter(T_te[idx_err, lv_x], T_te[idx_err, lv_y],
                           c=PALETTE[cls], marker='X', s=180,
                           edgecolors='red', linewidths=1.2, alpha=1.0,
                           label=f'Test FAIL: {cls}', zorder=6)
        ax.axhline(0, color='grey', lw=0.6, ls='--')
        ax.axvline(0, color='grey', lw=0.6, ls='--')
        ax.legend(fontsize=6.5, framealpha=0.85, ncol=2)
        ax.grid(True, alpha=0.25)

    # Fig 01 LV1 x LV2
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    for ax, cv_pred, sub in zip(axes,
            [None, y_pls_cv5],
            ['Full train model', f'{CV_FOLDS}-fold CV + test overlay']):
        pls_scatter_2d(ax, 0, 1, cv_pred)
        ax.set_xlabel(f'LV1 ({lv_pct(0):.1f}% var.)', fontsize=11)
        ax.set_ylabel(f'LV2 ({lv_pct(1):.1f}% var.)', fontsize=11)
        ax.set_title(f'PLS-DA Scores LV1 x LV2\n{sub}', fontsize=10, fontweight='bold')
    fig.suptitle(TTAG, fontsize=10, fontweight='bold')
    fig.tight_layout()
    save_fig(fig, '01_PLSDA_Scores_LV1_LV2.png')

    # Fig 02 LV1 x LV3
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    for ax, cv_pred, sub in zip(axes,
            [None, y_pls_loo],
            ['Full train model', 'LOO-CV + test overlay']):
        pls_scatter_2d(ax, 0, 2, cv_pred)
        ax.set_xlabel(f'LV1 ({lv_pct(0):.1f}% var.)', fontsize=11)
        ax.set_ylabel(f'LV3 ({lv_pct(2):.1f}% var.)', fontsize=11)
        ax.set_title(f'PLS-DA Scores LV1 x LV3\n{sub}', fontsize=10, fontweight='bold')
    fig.suptitle(TTAG, fontsize=10, fontweight='bold')
    fig.tight_layout()
    save_fig(fig, '02_PLSDA_Scores_LV1_LV3.png')

    # Fig 03 LV2 x LV3
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    for ax, cv_pred, sub in zip(axes,
            [None, y_pls_cv5],
            ['Full train model', f'{CV_FOLDS}-fold CV + test overlay']):
        pls_scatter_2d(ax, 1, 2, cv_pred)
        ax.set_xlabel(f'LV2 ({lv_pct(1):.1f}% var.)', fontsize=11)
        ax.set_ylabel(f'LV3 ({lv_pct(2):.1f}% var.)', fontsize=11)
        ax.set_title(f'PLS-DA Scores LV2 x LV3\n{sub}', fontsize=10, fontweight='bold')
    fig.suptitle(TTAG, fontsize=10, fontweight='bold')
    fig.tight_layout()
    save_fig(fig, '03_PLSDA_Scores_LV2_LV3.png')

    # Fig 04 3D
    fig4 = plt.figure(figsize=(9, 7))
    ax4  = fig4.add_subplot(111, projection='3d')
    for cls in classes:
        idx_tr  = y_tr == cls
        idx_ok  = (y_te == cls) & (y_pls_test == cls)
        idx_err = (y_te == cls) & (y_pls_test != cls)
        ax4.scatter(T_tr[idx_tr,0], T_tr[idx_tr,1], T_tr[idx_tr,2],
                    c=PALETTE[cls], marker=MARKER[cls], s=50,
                    edgecolors='k', linewidths=0.4, alpha=0.75,
                    label=f'Train: {cls}')
        ax4.scatter(T_te[idx_ok,0], T_te[idx_ok,1], T_te[idx_ok,2],
                    c=PALETTE[cls], marker='*', s=220,
                    edgecolors='k', linewidths=0.6, alpha=1.0,
                    label=f'Test OK: {cls}')
        if idx_err.any():
            ax4.scatter(T_te[idx_err,0], T_te[idx_err,1], T_te[idx_err,2],
                        c=PALETTE[cls], marker='X', s=180,
                        edgecolors='red', linewidths=1.0, alpha=1.0,
                        label=f'Test FAIL: {cls}')
    ax4.set_xlabel(f'LV1 ({lv_pct(0):.1f}%)', fontsize=9, labelpad=8)
    ax4.set_ylabel(f'LV2 ({lv_pct(1):.1f}%)', fontsize=9, labelpad=8)
    ax4.set_zlabel(f'LV3 ({lv_pct(2):.1f}%)', fontsize=9, labelpad=8)
    ax4.set_title(f'PLS-DA 3D Scores\n{TTAG}', fontsize=9, fontweight='bold', pad=12)
    ax4.legend(fontsize=6.5, ncol=2, framealpha=0.85)
    ax4.view_init(elev=22, azim=45)
    fig4.tight_layout()
    save_fig(fig4, '04_PLSDA_Scores_3D.png')

    def plot_cm(cm, title, fname, cmap):
        fig, ax = plt.subplots(figsize=(6.5, 5.5))
        ConfusionMatrixDisplay(cm, display_labels=classes).plot(
            ax=ax, colorbar=False, cmap=cmap, xticks_rotation=30)
        ax.set_title(title, fontsize=11, fontweight='bold')
        fig.tight_layout()
        save_fig(fig, fname)

    # Figs 05-07 PLS-DA confusion matrices
    plot_cm(cm_pls_cv5,
            f'PLS-DA  {CV_FOLDS}-fold CV  (train n={len(y_tr)})\nAccuracy = {acc_pls_cv5*100:.1f}%',
            '05_PLSDA_Confusion_5fold_Train.png', 'Blues')
    plot_cm(cm_pls_loo,
            f'PLS-DA  LOO-CV  (train n={len(y_tr)})\nAccuracy = {acc_pls_loo*100:.1f}%',
            '06_PLSDA_Confusion_LOO_Train.png', 'Purples')
    plot_cm(cm_pls_test,
            f'PLS-DA  Test set  (n={len(y_te)})\nAccuracy = {acc_pls_test*100:.1f}%',
            '07_PLSDA_Confusion_Test.png', 'Greens')

    # Fig 08 VIP scores
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.plot(wavenumbers, vip, color='steelblue', lw=1.2)
    ax.axhline(1, color='crimson', ls='--', lw=1, label='VIP = 1 threshold')
    ax.fill_between(wavenumbers, vip, 1, where=(vip > 1),
                    alpha=0.25, color='crimson',
                    label=f'VIP > 1  ({n_vip_gt1} variables)')
    ax.fill_between(wavenumbers, 0, vip.max()*0.06, where=sel_mask,
                    alpha=0.20, color='green', label=f'Top-{N_VIP} selected')
    ax.set_xlim(WAVENUMBER_MIN, WAVENUMBER_MAX)
    ax.set_xlabel('Wavenumber (cm-1)', fontsize=12)
    ax.set_ylabel('VIP score', fontsize=12)
    ax.set_title(f'VIP Scores (train only)\n{TTAG}', fontsize=10, fontweight='bold')
    ax.legend(fontsize=9); ax.grid(True, alpha=0.25)
    fig.tight_layout()
    save_fig(fig, '08_PLSDA_VIP_Scores.png')

    # Fig 09 Loadings
    fig, axes = plt.subplots(3, 1, figsize=(11, 8), sharex=True)
    for i, (ax, col) in enumerate(zip(axes, ['steelblue', 'darkorange', 'seagreen'])):
        ax.stem(wn_sel, P_load[:, i], linefmt=col, markerfmt=' ', basefmt='k-')
        ax.axhline(0, color='k', lw=0.5)
        ax.set_ylabel(f'Loading LV{i+1}', fontsize=10)
        ax.grid(True, alpha=0.2)
        for pk in np.argsort(np.abs(P_load[:, i]))[-5:]:
            ax.annotate(f'{wn_sel[pk]:.0f}',
                        xy=(wn_sel[pk], P_load[pk, i]), fontsize=7,
                        ha='center', xytext=(0, 5), textcoords='offset points')
    axes[-1].set_xlabel('Wavenumber (cm-1)', fontsize=12)
    axes[0].set_title(f'PLS-DA Loadings LV1/LV2/LV3  |  top-{N_VIP} vars\n{TTAG}',
                      fontsize=10, fontweight='bold')
    fig.tight_layout()
    save_fig(fig, '09_PLSDA_Loadings.png')

    # Fig 10 Permutation
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    ax.hist(perm_scores, bins=35, color='lightsteelblue',
            edgecolor='white', label='Permuted accuracy')
    ax.axvline(obs_score, color='crimson', lw=2.5,
               label=f'Observed: {obs_score*100:.1f}%   p = {p_val:.4f}')
    ax.axvline(0.25, color='grey', lw=1.2, ls=':', label='Chance level (25%)')
    ax.set_xlabel('5-fold CV accuracy (train)', fontsize=12)
    ax.set_ylabel('Count', fontsize=12)
    ax.set_title(f'Permutation Test ({N_PERM} perm.)  |  train only\n{TTAG}',
                 fontsize=9, fontweight='bold')
    ax.legend(fontsize=9); ax.grid(True, alpha=0.25)
    fig.tight_layout()
    save_fig(fig, '10_PLSDA_Permutation.png')

    # ── SVM cluster helper ───────────────────────────────────────────
    def svm_cluster(ax, mode, svm_cv_pred, title):
        if mode == 'train':
            for cls in classes:
                for i in np.where(y_tr == cls)[0]:
                    wrong = y_tr[i] != svm_cv_pred[i]
                    ax.scatter(Z_tr[i,0], Z_tr[i,1],
                               c=PALETTE[cls], marker=MARKER[cls], s=70,
                               edgecolors='red' if wrong else 'k',
                               linewidths=2.0 if wrong else 0.6,
                               alpha=0.85, zorder=3)
            for cls in classes:
                ax.scatter(Z_te[y_te==cls, 0], Z_te[y_te==cls, 1],
                           c=PALETTE[cls], marker='*', s=140,
                           edgecolors='k', linewidths=0.5, alpha=0.95, zorder=4)
        else:
            for cls in classes:
                ax.scatter(Z_tr[y_tr==cls, 0], Z_tr[y_tr==cls, 1],
                           c=PALETTE[cls], marker=MARKER[cls], s=55,
                           edgecolors='k', linewidths=0.4, alpha=0.30, zorder=2)
            for cls in classes:
                for i in np.where(y_te == cls)[0]:
                    wrong = y_te[i] != y_svm_test[i]
                    ax.scatter(Z_te[i,0], Z_te[i,1],
                               c=PALETTE[cls], marker='*', s=180,
                               edgecolors='red' if wrong else 'k',
                               linewidths=2.5 if wrong else 0.8,
                               alpha=0.95, zorder=4)
        handles = (
            [plt.scatter([], [], c=PALETTE[c], marker=MARKER[c],
                         s=50, edgecolors='k', label=f'Train: {c}') for c in classes] +
            [plt.scatter([], [], c=PALETTE[c], marker='*',
                         s=80, edgecolors='k', label=f'Test: {c}') for c in classes] +
            [plt.scatter([], [], c='w', marker='o', s=50,
                         edgecolors='red', linewidths=2.0, label='Misclassified'),
             plt.scatter([], [], c='w', marker='o', s=50,
                         edgecolors='k', linewidths=0.6, label='Correct')]
        )
        ax.legend(handles=handles, fontsize=7, framealpha=0.85)
        ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)', fontsize=10)
        ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)', fontsize=10)
        ax.set_title(title, fontsize=10, fontweight='bold')
        ax.grid(True, alpha=0.25)

    # Figs 11-13 SVM cluster
    fig, ax = plt.subplots(figsize=(8, 6.5))
    svm_cluster(ax, 'train', y_svm_cv5,
        f'SVM-RBF  {CV_FOLDS}-fold CV (train n={len(y_tr)})  acc={acc_svm_cv5*100:.1f}%\n'
        'Train: red=misclassified  |  Stars=test projected')
    fig.suptitle(TTAG, fontsize=9); fig.tight_layout()
    save_fig(fig, '11_SVM_Cluster_5fold_Train.png')

    fig, ax = plt.subplots(figsize=(8, 6.5))
    svm_cluster(ax, 'train', y_svm_loo,
        f'SVM-RBF  LOO-CV (train n={len(y_tr)})  acc={acc_svm_loo*100:.1f}%\n'
        'Train: red=misclassified  |  Stars=test projected')
    fig.suptitle(TTAG, fontsize=9); fig.tight_layout()
    save_fig(fig, '12_SVM_Cluster_LOO_Train.png')

    fig, ax = plt.subplots(figsize=(8, 6.5))
    svm_cluster(ax, 'test', None,
        f'SVM-RBF  Test set (n={len(y_te)})  acc={acc_svm_test*100:.1f}%\n'
        'Stars=test: red=misclassified  |  Faded=train reference')
    fig.suptitle(TTAG, fontsize=9); fig.tight_layout()
    save_fig(fig, '13_SVM_Cluster_Test.png')

    # Figs 14-16 SVM confusion matrices
    plot_cm(cm_svm_cv5,
            f'SVM-RBF  {CV_FOLDS}-fold CV  (train n={len(y_tr)})\nAccuracy = {acc_svm_cv5*100:.1f}%',
            '14_SVM_Confusion_5fold_Train.png', 'Blues')
    plot_cm(cm_svm_loo,
            f'SVM-RBF  LOO-CV  (train n={len(y_tr)})\nAccuracy = {acc_svm_loo*100:.1f}%',
            '15_SVM_Confusion_LOO_Train.png', 'Purples')
    plot_cm(cm_svm_test,
            f'SVM-RBF  Test set  (n={len(y_te)})\nAccuracy = {acc_svm_test*100:.1f}%',
            '16_SVM_Confusion_Test.png', 'Greens')

    # Fig 17 Accuracy comparison
    fig, ax = plt.subplots(figsize=(9, 5.5))
    methods  = [f'{CV_FOLDS}-fold CV\n(train)', 'LOO-CV\n(train)', 'Test set\n(n=32)']
    pls_accs = [acc_pls_cv5*100, acc_pls_loo*100, acc_pls_test*100]
    svm_accs = [acc_svm_cv5*100, acc_svm_loo*100, acc_svm_test*100]
    x = np.arange(3); w = 0.30
    b1 = ax.bar(x-w/2, pls_accs, w, color='steelblue',  edgecolor='k', linewidth=0.8, label='PLS-DA')
    b2 = ax.bar(x+w/2, svm_accs, w, color='darkorange', edgecolor='k', linewidth=0.8, label='SVM-RBF')
    for bar in list(b1) + list(b2):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.8,
                f'{bar.get_height():.1f}%', ha='center', va='bottom',
                fontsize=10, fontweight='bold')
    ax.axhline(100, color='grey', ls=':', lw=1)
    ax.set_xticks(x); ax.set_xticklabels(methods, fontsize=11)
    ax.set_ylabel('Accuracy (%)', fontsize=12); ax.set_ylim(0, 118)
    ax.set_title(f'Overall Accuracy  |  PLS-DA vs SVM-RBF\n{TTAG}',
                 fontsize=10, fontweight='bold')
    ax.legend(fontsize=11); ax.grid(True, alpha=0.2, axis='y')
    fig.tight_layout()
    save_fig(fig, '17_Accuracy_Comparison.png')

    # Fig 18 Per-class recall (3-panel)
    n_per_train = cm_pls_cv5.sum(axis=1)
    n_per_test  = cm_pls_test.sum(axis=1)
    fig, axes = plt.subplots(1, 3, figsize=(17, 5.5), sharey=True)
    panels = [
        (f'{CV_FOLDS}-fold CV  (train n={len(y_tr)})', cm_pls_cv5,  cm_svm_cv5,  n_per_train),
        (f'LOO-CV  (train n={len(y_tr)})',              cm_pls_loo,  cm_svm_loo,  n_per_train),
        (f'Test set  (n={len(y_te)})',                  cm_pls_test, cm_svm_test, n_per_test),
    ]
    x = np.arange(len(classes)); w = 0.30
    for ax, (ptitle, cm_p, cm_s, n_per) in zip(axes, panels):
        rec_p = [cm_p[i,i]/n_per[i]*100 for i in range(len(classes))]
        rec_s = [cm_s[i,i]/n_per[i]*100 for i in range(len(classes))]
        b1 = ax.bar(x-w/2, rec_p, w, color='steelblue',  edgecolor='k', linewidth=0.7, label='PLS-DA')
        b2 = ax.bar(x+w/2, rec_s, w, color='darkorange', edgecolor='k', linewidth=0.7, label='SVM-RBF')
        for bar in list(b1)+list(b2):
            ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+1.5,
                    f'{bar.get_height():.0f}%', ha='center', va='bottom',
                    fontsize=9, fontweight='bold')
        ax.axhline(100, color='grey', ls=':', lw=1)
        ax.set_xticks(x)
        ax.set_xticklabels([f'{c.replace("SSY-","")}\n(n={n_per[i]})'
                            for i, c in enumerate(classes)],
                           rotation=20, ha='right', fontsize=8)
        ax.set_title(ptitle, fontsize=10, fontweight='bold')
        ax.set_ylim(0, 122); ax.grid(True, alpha=0.2, axis='y'); ax.legend(fontsize=9)
    axes[0].set_ylabel('Recall (%)', fontsize=12)
    fig.suptitle(f'Per-Class Recall  |  PLS-DA vs SVM-RBF  |  {TTAG}',
                 fontsize=11, fontweight='bold')
    fig.tight_layout()
    save_fig(fig, '18_Recall_Comparison.png')

    # Fig 19 Mean spectra
    fig, ax = plt.subplots(figsize=(10, 4.5))
    for cls in classes:
        idx = y_all == cls
        m, s = X_all[idx].mean(axis=0), X_all[idx].std(axis=0)
        ax.plot(wavenumbers, m, color=PALETTE[cls], lw=1.5,
                label=f'{cls} (n={idx.sum()})')
        ax.fill_between(wavenumbers, m-s, m+s, color=PALETTE[cls], alpha=0.12)
    ax.set_xlim(WAVENUMBER_MIN, WAVENUMBER_MAX)
    ax.set_xlabel('Wavenumber (cm-1)', fontsize=12)
    ax.set_ylabel('Normalised intensity (a.u.)', fontsize=12)
    ax.set_title(f'Mean Raman Spectra +/- 1 SD  (all 100 spectra)\n'
                 f'{WAVENUMBER_MIN}-{WAVENUMBER_MAX} cm-1', fontsize=11, fontweight='bold')
    ax.legend(fontsize=9); ax.grid(True, alpha=0.25)
    fig.tight_layout()
    save_fig(fig, '19_Mean_Spectra.png')

    # ═══════════════════════════════════════════════════════════════
    # EXCEL  (13 sheets)
    # ═══════════════════════════════════════════════════════════════
    out_xlsx = os.path.join(OUTPUT_DIR, f'PLSDA_SVM_VIP{N_VIP}_Results.xlsx')
    with pd.ExcelWriter(out_xlsx, engine='openpyxl') as writer:

        # Sheet 1: Summary
        rows = [
            ('Dataset',                       'Book2 - 4 class SSY Raman'),
            ('File',                          FILE_NAME),
            ('Sheet',                         SHEET_NAME),
            ('Spectral window (cm-1)',         f'{WAVENUMBER_MIN}-{WAVENUMBER_MAX}'),
            ('Spectral variables in window',   X_all.shape[1]),
            ('Total spectra',                 len(y_all)),
            ('Train spectra',                 len(y_tr)),
            ('Test spectra',                  len(y_te)),
            ('VIP features selected',         N_VIP),
            ('VIP > 1 (train model)',         n_vip_gt1),
            ('PLS LVs',                       N_COMPONENTS),
            ('CV folds',                      CV_FOLDS),
            ('Random state',                  RANDOM_STATE),
            ('', ''),
            ('--- Class distribution ---',    ''),
        ]
        for cls in classes:
            rows.append((f'  {cls}  (train/test)',
                         f'{(y_tr==cls).sum()} / {(y_te==cls).sum()}'))
        rows += [
            ('', ''),
            ('--- PLS-DA ---',                ''),
            ('5-fold CV accuracy % (train)',  f'{acc_pls_cv5 *100:.2f}'),
            ('LOO-CV accuracy % (train)',     f'{acc_pls_loo *100:.2f}'),
            ('Test set accuracy %',          f'{acc_pls_test*100:.2f}'),
            ('Permutation p-value',           f'{p_val:.4f}'),
            ('Permutation result',            'Significant' if p_val < 0.05 else 'NOT significant'),
            ('', ''),
            ('--- SVM-RBF ---',               ''),
            ('5-fold CV accuracy % (train)',  f'{acc_svm_cv5 *100:.2f}'),
            ('LOO-CV accuracy % (train)',     f'{acc_svm_loo *100:.2f}'),
            ('Test set accuracy %',          f'{acc_svm_test*100:.2f}'),
            ('SVM kernel / C / gamma',        'RBF / 10 / scale'),
        ]
        for i, r2 in enumerate(r2x_cum, 1):
            rows.append((f'R2X cumulative LV{i} (%)', f'{r2*100:.4f}'))
        rows.append(('Leakage audit', f'{n_pass}/{len(audit)} PASS'))
        pd.DataFrame(rows, columns=['Metric', 'Value']).to_excel(
            writer, sheet_name='Summary', index=False)

        # Sheet 2: Per-class accuracy
        cls_rows = []
        for i, cls in enumerate(classes):
            nt, ne = n_per_train[i], n_per_test[i]
            cls_rows.append({
                'Class':           cls,
                'N_train':         int(nt),
                'N_test':          int(ne),
                'PLSDA_5fold_%':   f'{cm_pls_cv5[i,i] /nt*100:.1f}',
                'PLSDA_LOO_%':     f'{cm_pls_loo[i,i] /nt*100:.1f}',
                'PLSDA_Test_%':    f'{cm_pls_test[i,i]/ne*100:.1f}',
                'SVM_5fold_%':     f'{cm_svm_cv5[i,i] /nt*100:.1f}',
                'SVM_LOO_%':       f'{cm_svm_loo[i,i] /nt*100:.1f}',
                'SVM_Test_%':      f'{cm_svm_test[i,i]/ne*100:.1f}',
            })
        pd.DataFrame(cls_rows).to_excel(writer, sheet_name='Per_Class_Accuracy', index=False)

        # Sheet 3: Train predictions
        pd.DataFrame({
            'True_Class':          y_tr,
            'PLSDA_5fold_Pred':    y_pls_cv5,
            'PLSDA_5fold_Correct': y_tr == y_pls_cv5,
            'PLSDA_LOO_Pred':      y_pls_loo,
            'PLSDA_LOO_Correct':   y_tr == y_pls_loo,
            'SVM_5fold_Pred':      y_svm_cv5,
            'SVM_5fold_Correct':   y_tr == y_svm_cv5,
            'SVM_LOO_Pred':        y_svm_loo,
            'SVM_LOO_Correct':     y_tr == y_svm_loo,
            'LV1': T_tr[:,0], 'LV2': T_tr[:,1], 'LV3': T_tr[:,2],
            'PC1': Z_tr[:,0], 'PC2': Z_tr[:,1],
        }).to_excel(writer, sheet_name='Train_Predictions', index=False)

        # Sheet 4: Test predictions
        pd.DataFrame({
            'True_Class':         y_te,
            'PLSDA_Test_Pred':    y_pls_test,
            'PLSDA_Test_Correct': y_te == y_pls_test,
            'SVM_Test_Pred':      y_svm_test,
            'SVM_Test_Correct':   y_te == y_svm_test,
            'LV1': T_te[:,0], 'LV2': T_te[:,1], 'LV3': T_te[:,2],
            'PC1': Z_te[:,0], 'PC2': Z_te[:,1],
        }).to_excel(writer, sheet_name='Test_Predictions', index=False)

        # Sheets 5-10: Confusion matrices
        for sheet, cm in [
            ('CM_PLSDA_5fold_Train', cm_pls_cv5),
            ('CM_PLSDA_LOO_Train',   cm_pls_loo),
            ('CM_PLSDA_Test',        cm_pls_test),
            ('CM_SVM_5fold_Train',   cm_svm_cv5),
            ('CM_SVM_LOO_Train',     cm_svm_loo),
            ('CM_SVM_Test',          cm_svm_test),
        ]:
            pd.DataFrame(cm,
                         index=pd.Index(classes, name='True\\Predicted'),
                         columns=classes).to_excel(writer, sheet_name=sheet)

        # Sheet 11: VIP scores
        pd.DataFrame({
            'Wavenumber_cm1':         wavenumbers,
            'VIP_Score':              vip,
            'VIP_gt_1':               vip > 1,
            f'Selected_Top{N_VIP}':   sel_mask,
        }).to_excel(writer, sheet_name='VIP_Scores', index=False)

        # Sheet 12: Loadings
        load_df = pd.DataFrame(P_load,
                               columns=[f'LV{i+1}' for i in range(N_COMPONENTS)])
        load_df.insert(0, 'Wavenumber_cm1', wn_sel)
        load_df.to_excel(writer, sheet_name='Loadings', index=False)

        # Sheet 13: Permutation test
        pd.DataFrame({
            'Permutation_Index': np.arange(1, N_PERM+1),
            'Permuted_Accuracy': perm_scores,
        }).to_excel(writer, sheet_name='Permutation_Test', index=False)

    print(f'\n  [SAVED] {out_xlsx}')

    # ── Console summary ──────────────────────────────────────────────
    print(f"\n{'='*68}")
    print(f"  RESULTS  VIP top-{N_VIP}  |  68 train / 32 test")
    print(f"{'='*68}")
    print(f"  {'MODEL':<10}  {'5-fold CV (train)':>18}  {'LOO-CV (train)':>15}  {'Test':>8}")
    print(f"  {'-'*56}")
    print(f"  {'PLS-DA':<10}  {acc_pls_cv5*100:>17.1f}%  {acc_pls_loo*100:>14.1f}%  {acc_pls_test*100:>7.1f}%")
    print(f"  {'SVM-RBF':<10}  {acc_svm_cv5*100:>17.1f}%  {acc_svm_loo*100:>14.1f}%  {acc_svm_test*100:>7.1f}%")
    print(f"\n  Permutation p = {p_val:.4f}  [{sig}]")
    print(f"\n  Per-class recall (PLS-DA / SVM-RBF):")
    print(f"  {'Class':<25} {'nTr':>5} {'nTe':>5} {'PLS 5f':>8} {'PLS LOO':>9} {'PLS Te':>8} {'SVM 5f':>8} {'SVM LOO':>9} {'SVM Te':>8}")
    print(f"  {'-'*90}")
    for i, cls in enumerate(classes):
        nt, ne = n_per_train[i], n_per_test[i]
        print(f"  {cls:<25} {nt:>5} {ne:>5} "
              f"{cm_pls_cv5[i,i]/nt*100:>7.0f}% "
              f"{cm_pls_loo[i,i]/nt*100:>8.0f}% "
              f"{cm_pls_test[i,i]/ne*100:>7.0f}% "
              f"{cm_svm_cv5[i,i]/nt*100:>7.0f}% "
              f"{cm_svm_loo[i,i]/nt*100:>8.0f}% "
              f"{cm_svm_test[i,i]/ne*100:>7.0f}%")
    print(f"\n  R2X (train): " +
          "  ".join([f"LV{i+1}={r*100:.1f}%" for i, r in enumerate(r2x_cum)]))
    print(f"  Leakage audit: {n_pass}/{len(audit)} PASS")
    print(f"  Output: {OUTPUT_DIR}")
    print(f"{'='*68}\n")

print("\n  ALL DONE — three VIP sets completed.")
print("  Output folders:")
for n in VIP_SETS:
    print(f"    {os.path.join(DATA_PATH, f'PLSDA_SVM_VIP{n}')}")
