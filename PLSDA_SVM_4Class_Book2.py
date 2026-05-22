"""
=============================================================================
PLS-DA + SVM-RBF  |  4-CLASS  |  Book2.xlsx  |  68/32 STRATIFIED SPLIT
=============================================================================
File   : Book2.xlsx   (C:/Users/Hira Aman/Desktop/PROF_DOMENICO'S)
Sheet  : 4 - Normalised

Four classes (25 each, 3 sessions):
  1. SSY-PDMS          2. SSY-Gr-PDMS
  3. SSY-SiO2/Si       4. SSY-Gr-SiO2/Si

Train / Test split : 68 train (all 4 classes, 17 each)
                     32 test  -> GRAPHENE ONLY kept (n=16)
                                 SSY-Gr-PDMS (8) + SSY-Gr-SiO2/Si (8)
                                 Non-graphene test (n=16) discarded

VIP sets      : top-1000 / top-1500 / top-1800  (one output folder each)
Models        : PLS-DA  (5 LVs)  +  SVM-RBF  (C=10, gamma=scale)
Validation    : 5-fold CV  +  LOO-CV  (both on TRAIN only)
               External test on graphene set (n=16)
Permutations  : 999  (train only)
Spectral win  : 450-1800 cm-1

OUTPUTS  ->  PROF_DOMENICO'S\PLSDA_SVM_4Class_VIP<N>\
  01  PLSDA_Scores_LV1_LV2.png       10  SVM_Cluster_5fold_Train.png
  02  PLSDA_Scores_LV1_LV3.png       11  SVM_Cluster_LOO_Train.png
  03  PLSDA_Scores_3D.png             12  SVM_Cluster_Test_Graphene.png
  04  PLSDA_Confusion_5fold_Train.png 13  SVM_Confusion_5fold_Train.png
  05  PLSDA_Confusion_LOO_Train.png   14  SVM_Confusion_LOO_Train.png
  06  PLSDA_Confusion_Test_Graph.png  15  SVM_Confusion_Test_Graphene.png
  07  PLSDA_VIP_Scores.png            16  Accuracy_Comparison.png
  08  PLSDA_Loadings.png              17  Recall_Comparison.png
  09  PLSDA_Permutation.png           18  Mean_Spectra.png
  PLSDA_SVM_4Class_VIP<N>_Results.xlsx  (13 sheets)

HOW TO RUN:
  pip install pandas openpyxl scikit-learn scipy matplotlib
  python PLSDA_SVM_4Class_Book2.py
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

VIP_SETS         = [1000, 1500, 1800]
GRAPHENE_CLASSES = ['SSY-Gr-PDMS', 'SSY-Gr-SiO2/Si']

# ═══════════════════════════════════════════════════════════════════════
# 1.  CLASS MAPPING  (session-prefix aware)
# ═══════════════════════════════════════════════════════════════════════
def assign_class(label):
    s = str(label).strip().lower()
    s = re.sub(r'^s\d+[-_]', '', s)
    s = re.sub(r'^(lugl|marzo\d*|maggio\d*)[-_]?(ssy[-_]?)?', '', s)
    s = s.replace('_', '-').replace(' ', '-')
    s = s.replace('grsl', 'gr')
    if   re.search(r'gr[-_]?sio2', s): return 'SSY-Gr-SiO2/Si'
    elif re.search(r'gr[-_]?pdms', s): return 'SSY-Gr-PDMS'
    elif re.search(r'sio2',        s): return 'SSY-SiO2/Si'
    elif re.search(r'pdms',        s): return 'SSY-PDMS'
    else:                               return None

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
# 5.  LOAD DATA
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
    print(f"  [WARNING] {len(unassigned)} unassigned: {unassigned}")

keep_idx    = [i for i, cl in enumerate(classes_raw) if cl is not None]
col_names   = [col_names_all[i]  for i in keep_idx]
classes_arr = [classes_raw[i]    for i in keep_idx]
raw         = raw.iloc[:, keep_idx]

print(f"  Loaded  : {raw.shape[1]} spectra x {raw.shape[0]} wavenumber points")
print(f"  Classes : {dict(Counter(classes_arr))}")

mask        = (wavenumbers_all >= WAVENUMBER_MIN) & (wavenumbers_all <= WAVENUMBER_MAX)
wavenumbers = wavenumbers_all[mask]
X_all       = raw.values[mask, :].T.astype(float)
y_all       = np.array(classes_arr)

print(f"  Window  : {WAVENUMBER_MIN}-{WAVENUMBER_MAX} cm-1  ({X_all.shape[1]} variables)")

# ═══════════════════════════════════════════════════════════════════════
# 6.  68 / 32 STRATIFIED SPLIT  — done ONCE before any fitting
# ═══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 68)
print("  STEP 2 — Train/Test split  (68/32 stratified, seed=42)")
print("=" * 68)

X_tr, X_te_all, y_tr, y_te_all = train_test_split(
    X_all, y_all,
    test_size=0.32, stratify=y_all, random_state=RANDOM_STATE
)

# Keep GRAPHENE ONLY in test set — discard non-graphene test spectra
graph_mask = np.isin(y_te_all, GRAPHENE_CLASSES)
X_te       = X_te_all[graph_mask]
y_te       = y_te_all[graph_mask]

print(f"  Train (all 4 classes)          : {X_tr.shape[0]}  {dict(Counter(y_tr))}")
print(f"  Test  (graphene only, KEPT)    : {X_te.shape[0]}  {dict(Counter(y_te))}")
print(f"  Test  (non-graphene, DISCARDED): {X_te_all.shape[0]-X_te.shape[0]}  "
      f"{dict(Counter(y_te_all[~graph_mask]))}")

le      = LabelEncoder().fit(y_all)
classes = le.classes_

# ═══════════════════════════════════════════════════════════════════════
# 7.  VIP LOOP
# ═══════════════════════════════════════════════════════════════════════
for N_VIP in VIP_SETS:

    OUTPUT_DIR = os.path.join(DATA_PATH, f"PLSDA_SVM_4Class_VIP{N_VIP}")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    def save_fig(fig, fname):
        fig.savefig(os.path.join(OUTPUT_DIR, fname), dpi=180, bbox_inches='tight')
        plt.close(fig)
        print(f'  [SAVED] {fname}')

    print("\n" + "=" * 68)
    print(f"  VIP TOP-{N_VIP}  ->  {OUTPUT_DIR}")
    print("=" * 68)

    # ── Scale on TRAIN only ──────────────────────────────────────────
    scaler  = StandardScaler().fit(X_tr)
    X_tr_sc = scaler.transform(X_tr)
    X_te_sc = scaler.transform(X_te)

    # ── VIP on TRAIN only ────────────────────────────────────────────
    print(f"\n  [VIP] PLS-DA on train -> top-{N_VIP} ...")
    Y_d_tr = np.zeros((len(y_tr), len(classes)))
    for j, c in enumerate(classes):
        Y_d_tr[:, j] = (y_tr == c).astype(float)

    pls_vip = PLSRegression(n_components=N_COMPONENTS, scale=False, max_iter=5000)
    pls_vip.fit(X_tr_sc, Y_d_tr)
    vip     = compute_vip(pls_vip)
    top_idx = np.argsort(vip)[-N_VIP:]
    wn_sel  = wavenumbers[top_idx]

    X_tr_sel  = X_tr_sc[:, top_idx]   # (68, N_VIP) — scaled, VIP-selected
    X_te_sel  = X_te_sc[:, top_idx]   # (16, N_VIP)
    n_vip_gt1 = int((vip > 1).sum())
    sel_mask  = np.zeros(len(wavenumbers), dtype=bool)
    sel_mask[top_idx] = True

    print(f"  [VIP] VIP>1: {n_vip_gt1}/{len(vip)}  |  "
          f"top-{N_VIP}: {wn_sel.min():.0f}-{wn_sel.max():.0f} cm-1")

    # ── Full PLS-DA on TRAIN (for visualisation) ─────────────────────
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
    print(f"  [PLS-DA] Re-substitution: {resubst*100:.1f}%")
    print(f"  [PLS-DA] R2X: " +
          "  ".join([f"LV{i+1}={r*100:.1f}%" for i, r in enumerate(r2x_cum)]))

    # ── PLS-DA TEST prediction (graphene only) ───────────────────────
    print(f"\n  [PLS-DA] Predicting graphene test set (n={len(y_te)}) ...")
    pls_full_pipe = Pipeline([('sc', StandardScaler()),
                               ('plsda', PLSDAClassifier(n_components=N_COMPONENTS))])
    pls_full_pipe.fit(X_tr[:, top_idx], y_tr)
    y_pls_test   = pls_full_pipe.predict(X_te[:, top_idx])
    acc_pls_test = accuracy_score(y_te, y_pls_test)
    cm_pls_test  = confusion_matrix(y_te, y_pls_test, labels=classes)
    print(f"  [PLS-DA] Test accuracy (graphene): {acc_pls_test*100:.1f}%")
    print(classification_report(y_te, y_pls_test,
                                 labels=GRAPHENE_CLASSES,
                                 target_names=GRAPHENE_CLASSES, digits=3))

    # ── PLS-DA 5-fold CV on TRAIN ────────────────────────────────────
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

    # ── PLS-DA LOO-CV on TRAIN ───────────────────────────────────────
    print(f"  [PLS-DA] LOO-CV on train ({len(y_tr)} folds) ...")
    y_pls_loo   = cross_val_predict(pls_pipe, X_tr[:, top_idx], y_tr, cv=loo)
    acc_pls_loo = accuracy_score(y_tr, y_pls_loo)
    cm_pls_loo  = confusion_matrix(y_tr, y_pls_loo, labels=classes)
    print(f"  [PLS-DA] LOO-CV accuracy: {acc_pls_loo*100:.1f}%")

    # ── Permutation test (TRAIN only) ────────────────────────────────
    print(f"\n  [PERM] {N_PERM} permutations (train only) ...")
    obs_score, perm_scores, p_val = permutation_test_score(
        pls_pipe, X_tr[:, top_idx], y_tr,
        scoring='accuracy', cv=cv5,
        n_permutations=N_PERM, random_state=RANDOM_STATE, n_jobs=-1
    )
    sig = 'SIGNIFICANT (p<0.05)' if p_val < 0.05 else 'NOT significant'
    print(f"  [PERM] Observed: {obs_score*100:.2f}%   p = {p_val:.4f}  [{sig}]")

    # ── SVM-RBF 5-fold CV + LOO-CV + TEST ────────────────────────────
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

    print(f"\n  [SVM] Predicting graphene test set (n={len(y_te)}) ...")
    svm_pipe.fit(X_tr[:, top_idx], y_tr)
    y_svm_test   = svm_pipe.predict(X_te[:, top_idx])
    acc_svm_test = accuracy_score(y_te, y_svm_test)
    cm_svm_test  = confusion_matrix(y_te, y_svm_test, labels=classes)
    print(f"  [SVM] Test accuracy (graphene): {acc_svm_test*100:.1f}%")
    print(classification_report(y_te, y_svm_test,
                                 labels=GRAPHENE_CLASSES,
                                 target_names=GRAPHENE_CLASSES, digits=3))

    # ── PCA for SVM cluster plots (fitted on TRAIN only) ─────────────
    pca = PCA(n_components=2, random_state=RANDOM_STATE)
    pca.fit(X_tr_sel)
    Z_tr = pca.transform(X_tr_sel)
    Z_te = pca.transform(X_te_sel)

    TTAG = (f"VIP top-{N_VIP}  |  {N_COMPONENTS} LVs  |  "
            f"{WAVENUMBER_MIN}-{WAVENUMBER_MAX} cm-1  |  "
            f"68 train / {len(y_te)} graphene test")

    # ═════════════════════════════════════════════════════════════════
    # FIGURES
    # ═════════════════════════════════════════════════════════════════

    # ── helper: 2-D PLS-DA score scatter ────────────────────────────
    def pls_scatter_2d(ax, lv_x, lv_y, cv_pred=None):
        for cls in classes:
            idx_tr = y_tr == cls
            if cv_pred is not None:
                ec = ['red' if y_tr[i] != cv_pred[i] else 'k'
                      for i in np.where(idx_tr)[0]]
            else:
                ec = ['k'] * int(idx_tr.sum())
            ax.scatter(T_tr[idx_tr, lv_x], T_tr[idx_tr, lv_y],
                       c=PALETTE[cls], marker=MARKER[cls], s=55,
                       edgecolors=ec, linewidths=0.7, alpha=0.75,
                       label=f'Train: {cls}', zorder=3)
        for cls in GRAPHENE_CLASSES:
            idx_ok  = (y_te == cls) & (y_pls_test == cls)
            idx_err = (y_te == cls) & (y_pls_test != cls)
            ax.scatter(T_te[idx_ok,  lv_x], T_te[idx_ok,  lv_y],
                       c=PALETTE[cls], marker='*', s=220,
                       edgecolors='k', linewidths=0.6, alpha=1.0,
                       label=f'Test-Gr OK: {cls}', zorder=5)
            if idx_err.any():
                ax.scatter(T_te[idx_err, lv_x], T_te[idx_err, lv_y],
                           c=PALETTE[cls], marker='X', s=180,
                           edgecolors='red', linewidths=1.2, alpha=1.0,
                           label=f'Test-Gr FAIL: {cls}', zorder=6)
        ax.axhline(0, color='grey', lw=0.6, ls='--')
        ax.axvline(0, color='grey', lw=0.6, ls='--')
        ax.legend(fontsize=6.5, framealpha=0.85, ncol=2)
        ax.grid(True, alpha=0.25)

    # Fig 01 LV1 x LV2
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    for ax, cv_pred, sub in zip(axes,
            [None, y_pls_cv5],
            ['Full train model', f'{CV_FOLDS}-fold CV (train) + graphene test overlay']):
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
            ['Full train model', 'LOO-CV (train) + graphene test overlay']):
        pls_scatter_2d(ax, 0, 2, cv_pred)
        ax.set_xlabel(f'LV1 ({lv_pct(0):.1f}% var.)', fontsize=11)
        ax.set_ylabel(f'LV3 ({lv_pct(2):.1f}% var.)', fontsize=11)
        ax.set_title(f'PLS-DA Scores LV1 x LV3\n{sub}', fontsize=10, fontweight='bold')
    fig.suptitle(TTAG, fontsize=10, fontweight='bold')
    fig.tight_layout()
    save_fig(fig, '02_PLSDA_Scores_LV1_LV3.png')

    # Fig 03 3D
    fig3 = plt.figure(figsize=(9, 7))
    ax3  = fig3.add_subplot(111, projection='3d')
    for cls in classes:
        idx = y_tr == cls
        ax3.scatter(T_tr[idx,0], T_tr[idx,1], T_tr[idx,2],
                    c=PALETTE[cls], marker=MARKER[cls], s=50,
                    edgecolors='k', linewidths=0.4, alpha=0.75,
                    label=f'Train: {cls}')
    for cls in GRAPHENE_CLASSES:
        idx_ok  = (y_te == cls) & (y_pls_test == cls)
        idx_err = (y_te == cls) & (y_pls_test != cls)
        ax3.scatter(T_te[idx_ok,0],  T_te[idx_ok,1],  T_te[idx_ok,2],
                    c=PALETTE[cls], marker='*', s=220,
                    edgecolors='k', linewidths=0.6, alpha=1.0,
                    label=f'Test-Gr OK: {cls}')
        if idx_err.any():
            ax3.scatter(T_te[idx_err,0], T_te[idx_err,1], T_te[idx_err,2],
                        c=PALETTE[cls], marker='X', s=180,
                        edgecolors='red', linewidths=1.0, alpha=1.0,
                        label=f'Test-Gr FAIL: {cls}')
    ax3.set_xlabel(f'LV1 ({lv_pct(0):.1f}%)', fontsize=9, labelpad=8)
    ax3.set_ylabel(f'LV2 ({lv_pct(1):.1f}%)', fontsize=9, labelpad=8)
    ax3.set_zlabel(f'LV3 ({lv_pct(2):.1f}%)', fontsize=9, labelpad=8)
    ax3.set_title(f'PLS-DA 3D Scores\nTrain=all 4 classes  |  Test=graphene only\n{TTAG}',
                  fontsize=9, fontweight='bold', pad=12)
    ax3.legend(fontsize=6.5, ncol=2, framealpha=0.85)
    ax3.view_init(elev=22, azim=45)
    fig3.tight_layout()
    save_fig(fig3, '03_PLSDA_Scores_3D.png')

    # ── confusion matrix helper ───────────────────────────────────────
    def plot_cm(cm, title, fname, cmap):
        fig, ax = plt.subplots(figsize=(6.5, 5.5))
        ConfusionMatrixDisplay(cm, display_labels=classes).plot(
            ax=ax, colorbar=False, cmap=cmap, xticks_rotation=30)
        ax.set_title(title, fontsize=10, fontweight='bold')
        fig.tight_layout()
        save_fig(fig, fname)

    # Figs 04-06 PLS-DA confusion matrices
    plot_cm(cm_pls_cv5,
            f'PLS-DA  {CV_FOLDS}-fold CV  (train n={len(y_tr)})\nAccuracy = {acc_pls_cv5*100:.1f}%',
            '04_PLSDA_Confusion_5fold_Train.png', 'Blues')
    plot_cm(cm_pls_loo,
            f'PLS-DA  LOO-CV  (train n={len(y_tr)})\nAccuracy = {acc_pls_loo*100:.1f}%',
            '05_PLSDA_Confusion_LOO_Train.png', 'Purples')
    plot_cm(cm_pls_test,
            f'PLS-DA  Test Set — Graphene Only  (n={len(y_te)})\n'
            f'Accuracy = {acc_pls_test*100:.1f}%  (non-graphene rows = 0)',
            '06_PLSDA_Confusion_Test_Graphene.png', 'Greens')

    # Fig 07 VIP scores
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
    save_fig(fig, '07_PLSDA_VIP_Scores.png')

    # Fig 08 Loadings LV1/LV2/LV3
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
    save_fig(fig, '08_PLSDA_Loadings.png')

    # Fig 09 Permutation
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
    save_fig(fig, '09_PLSDA_Permutation.png')

    # ── SVM cluster helper ────────────────────────────────────────────
    def svm_cluster(ax, mode, svm_cv_pred, title):
        """
        mode='train': train points coloured by CV error; test stars as reference.
        mode='test' : train faded; test stars coloured by prediction error.
        """
        if mode == 'train':
            for cls in classes:
                for i in np.where(y_tr == cls)[0]:
                    wrong = y_tr[i] != svm_cv_pred[i]
                    ax.scatter(Z_tr[i, 0], Z_tr[i, 1],
                               c=PALETTE[cls], marker=MARKER[cls], s=70,
                               edgecolors='red' if wrong else 'k',
                               linewidths=2.0 if wrong else 0.6,
                               alpha=0.85, zorder=3)
            for cls in GRAPHENE_CLASSES:
                ax.scatter(Z_te[y_te == cls, 0], Z_te[y_te == cls, 1],
                           c=PALETTE[cls], marker='*', s=160,
                           edgecolors='k', linewidths=0.5,
                           alpha=0.95, zorder=4)
        else:
            for cls in classes:
                ax.scatter(Z_tr[y_tr == cls, 0], Z_tr[y_tr == cls, 1],
                           c=PALETTE[cls], marker=MARKER[cls], s=55,
                           edgecolors='k', linewidths=0.4,
                           alpha=0.30, zorder=2)
            for cls in GRAPHENE_CLASSES:
                for i in np.where(y_te == cls)[0]:
                    wrong = y_te[i] != y_svm_test[i]
                    ax.scatter(Z_te[i, 0], Z_te[i, 1],
                               c=PALETTE[cls], marker='*', s=200,
                               edgecolors='red' if wrong else 'k',
                               linewidths=2.5 if wrong else 0.8,
                               alpha=0.95, zorder=4)

        handles = (
            [plt.scatter([], [], c=PALETTE[c], marker=MARKER[c],
                         s=50, edgecolors='k', label=f'Train: {c}')
             for c in classes] +
            [plt.scatter([], [], c=PALETTE[c], marker='*',
                         s=80, edgecolors='k', label=f'Test-Gr: {c}')
             for c in GRAPHENE_CLASSES] +
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

    # Figs 10-12 SVM cluster plots
    fig, ax = plt.subplots(figsize=(8, 6.5))
    svm_cluster(ax, 'train', y_svm_cv5,
        f'SVM-RBF  {CV_FOLDS}-fold CV (train n={len(y_tr)})  acc={acc_svm_cv5*100:.1f}%\n'
        'Train: red=misclassified  |  Stars=graphene test (reference)')
    fig.suptitle(TTAG, fontsize=9); fig.tight_layout()
    save_fig(fig, '10_SVM_Cluster_5fold_Train.png')

    fig, ax = plt.subplots(figsize=(8, 6.5))
    svm_cluster(ax, 'train', y_svm_loo,
        f'SVM-RBF  LOO-CV (train n={len(y_tr)})  acc={acc_svm_loo*100:.1f}%\n'
        'Train: red=misclassified  |  Stars=graphene test (reference)')
    fig.suptitle(TTAG, fontsize=9); fig.tight_layout()
    save_fig(fig, '11_SVM_Cluster_LOO_Train.png')

    fig, ax = plt.subplots(figsize=(8, 6.5))
    svm_cluster(ax, 'test', None,
        f'SVM-RBF  Graphene Test (n={len(y_te)})  acc={acc_svm_test*100:.1f}%\n'
        'Stars=test: red=misclassified  |  Faded=train (reference)')
    fig.suptitle(TTAG, fontsize=9); fig.tight_layout()
    save_fig(fig, '12_SVM_Cluster_Test_Graphene.png')

    # Figs 13-15 SVM confusion matrices
    plot_cm(cm_svm_cv5,
            f'SVM-RBF  {CV_FOLDS}-fold CV  (train n={len(y_tr)})\nAccuracy = {acc_svm_cv5*100:.1f}%',
            '13_SVM_Confusion_5fold_Train.png', 'Blues')
    plot_cm(cm_svm_loo,
            f'SVM-RBF  LOO-CV  (train n={len(y_tr)})\nAccuracy = {acc_svm_loo*100:.1f}%',
            '14_SVM_Confusion_LOO_Train.png', 'Purples')
    plot_cm(cm_svm_test,
            f'SVM-RBF  Graphene Test  (n={len(y_te)})\n'
            f'Accuracy = {acc_svm_test*100:.1f}%  (non-graphene rows = 0)',
            '15_SVM_Confusion_Test_Graphene.png', 'Greens')

    # Fig 16 Accuracy comparison
    fig, ax = plt.subplots(figsize=(9, 5.5))
    methods  = [f'{CV_FOLDS}-fold CV\n(train)', 'LOO-CV\n(train)',
                f'Test-Graphene\n(n={len(y_te)})']
    pls_accs = [acc_pls_cv5*100, acc_pls_loo*100, acc_pls_test*100]
    svm_accs = [acc_svm_cv5*100, acc_svm_loo*100, acc_svm_test*100]
    x = np.arange(3); w = 0.30
    b1 = ax.bar(x-w/2, pls_accs, w, color='steelblue',  edgecolor='k', lw=0.8, label='PLS-DA')
    b2 = ax.bar(x+w/2, svm_accs, w, color='darkorange', edgecolor='k', lw=0.8, label='SVM-RBF')
    for bar in list(b1)+list(b2):
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
    save_fig(fig, '16_Accuracy_Comparison.png')

    # Fig 17 Per-class recall (3-panel: 5-fold / LOO / Test)
    n_per_tr = cm_pls_cv5.sum(axis=1)
    n_per_te = cm_pls_test.sum(axis=1)
    fig, axes = plt.subplots(1, 3, figsize=(17, 5.5), sharey=True)
    panels = [
        (f'{CV_FOLDS}-fold CV  (train n={len(y_tr)})', cm_pls_cv5,  cm_svm_cv5,  n_per_tr),
        (f'LOO-CV  (train n={len(y_tr)})',              cm_pls_loo,  cm_svm_loo,  n_per_tr),
        (f'Test-Graphene  (n={len(y_te)})',             cm_pls_test, cm_svm_test, n_per_te),
    ]
    x_cls = np.arange(len(classes)); w = 0.30
    for ax, (ptitle, cm_p, cm_s, n_per) in zip(axes, panels):
        rec_p = [cm_p[i,i]/n_per[i]*100 if n_per[i] > 0 else 0
                 for i in range(len(classes))]
        rec_s = [cm_s[i,i]/n_per[i]*100 if n_per[i] > 0 else 0
                 for i in range(len(classes))]
        b1 = ax.bar(x_cls-w/2, rec_p, w, color='steelblue',  edgecolor='k', lw=0.7, label='PLS-DA')
        b2 = ax.bar(x_cls+w/2, rec_s, w, color='darkorange', edgecolor='k', lw=0.7, label='SVM-RBF')
        for bar in list(b1)+list(b2):
            if bar.get_height() > 0:
                ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+1.5,
                        f'{bar.get_height():.0f}%', ha='center', va='bottom',
                        fontsize=9, fontweight='bold')
        ax.axhline(100, color='grey', ls=':', lw=1)
        ax.set_xticks(x_cls)
        ax.set_xticklabels([f'{c.replace("SSY-","")}\n(n={n_per[i]})'
                            for i, c in enumerate(classes)],
                           rotation=20, ha='right', fontsize=8)
        ax.set_title(ptitle, fontsize=10, fontweight='bold')
        ax.set_ylim(0, 122); ax.grid(True, alpha=0.2, axis='y'); ax.legend(fontsize=9)
    axes[0].set_ylabel('Recall (%)', fontsize=12)
    fig.suptitle(f'Per-Class Recall  |  PLS-DA vs SVM-RBF  |  {TTAG}\n'
                 f'(Test panel: graphene classes only — non-graphene n=0)',
                 fontsize=10, fontweight='bold')
    fig.tight_layout()
    save_fig(fig, '17_Recall_Comparison.png')

    # Fig 18 Mean spectra (all 100)
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
    save_fig(fig, '18_Mean_Spectra.png')

    # ═════════════════════════════════════════════════════════════════
    # EXCEL  (13 sheets)
    # ═════════════════════════════════════════════════════════════════
    out_xlsx = os.path.join(OUTPUT_DIR, f'PLSDA_SVM_4Class_VIP{N_VIP}_Results.xlsx')
    with pd.ExcelWriter(out_xlsx, engine='openpyxl') as writer:

        # Sheet 1: Summary
        rows = [
            ('Dataset',                          'Book2 - 4 class SSY Raman'),
            ('File',                             FILE_NAME),
            ('Sheet',                            SHEET_NAME),
            ('Spectral window (cm-1)',            f'{WAVENUMBER_MIN}-{WAVENUMBER_MAX}'),
            ('Spectral variables',               X_all.shape[1]),
            ('Total spectra',                    len(y_all)),
            ('Train spectra (all 4 classes)',    len(y_tr)),
            ('Test spectra (graphene only)',      len(y_te)),
            ('Test discarded (non-graphene)',     X_te_all.shape[0]-X_te.shape[0]),
            ('VIP features selected',            N_VIP),
            ('VIP > 1',                          n_vip_gt1),
            ('PLS LVs',                          N_COMPONENTS),
            ('CV folds',                         CV_FOLDS),
            ('Random state',                     RANDOM_STATE),
            ('', ''),
            ('--- Class distribution ---',       ''),
        ]
        for cls in classes:
            rows.append((f'  {cls}  (train)',  int((y_tr==cls).sum())))
        for cls in GRAPHENE_CLASSES:
            rows.append((f'  {cls}  (test)',   int((y_te==cls).sum())))
        rows += [
            ('', ''),
            ('--- PLS-DA ---',                   ''),
            ('5-fold CV accuracy % (train)',     f'{acc_pls_cv5 *100:.2f}'),
            ('LOO-CV accuracy % (train)',        f'{acc_pls_loo *100:.2f}'),
            ('Test accuracy % (graphene)',       f'{acc_pls_test*100:.2f}'),
            ('Permutation p-value',              f'{p_val:.4f}'),
            ('Permutation result',               'Significant' if p_val<0.05 else 'NOT significant'),
            ('', ''),
            ('--- SVM-RBF ---',                  ''),
            ('5-fold CV accuracy % (train)',     f'{acc_svm_cv5 *100:.2f}'),
            ('LOO-CV accuracy % (train)',        f'{acc_svm_loo *100:.2f}'),
            ('Test accuracy % (graphene)',       f'{acc_svm_test*100:.2f}'),
            ('SVM kernel / C / gamma',           'RBF / 10 / scale'),
        ]
        for i, r2 in enumerate(r2x_cum, 1):
            rows.append((f'R2X cumulative LV{i} (%)', f'{r2*100:.4f}'))
        pd.DataFrame(rows, columns=['Metric', 'Value']).to_excel(
            writer, sheet_name='Summary', index=False)

        # Sheet 2: Per-class accuracy
        cls_rows = []
        for i, cls in enumerate(classes):
            nt = n_per_tr[i]
            ne = n_per_te[i]
            cls_rows.append({
                'Class':           cls,
                'N_train':         int(nt),
                'N_test':          int(ne),
                'PLSDA_5fold_%':   f'{cm_pls_cv5[i,i]/nt*100:.1f}' if nt>0 else '-',
                'PLSDA_LOO_%':     f'{cm_pls_loo[i,i]/nt*100:.1f}' if nt>0 else '-',
                'PLSDA_Test_%':    f'{cm_pls_test[i,i]/ne*100:.1f}' if ne>0 else '-',
                'SVM_5fold_%':     f'{cm_svm_cv5[i,i]/nt*100:.1f}' if nt>0 else '-',
                'SVM_LOO_%':       f'{cm_svm_loo[i,i]/nt*100:.1f}' if nt>0 else '-',
                'SVM_Test_%':      f'{cm_svm_test[i,i]/ne*100:.1f}' if ne>0 else '-',
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

        # Sheet 4: Test predictions (graphene only)
        pd.DataFrame({
            'True_Class':         y_te,
            'PLSDA_Test_Pred':    y_pls_test,
            'PLSDA_Test_Correct': y_te == y_pls_test,
            'SVM_Test_Pred':      y_svm_test,
            'SVM_Test_Correct':   y_te == y_svm_test,
            'LV1': T_te[:,0], 'LV2': T_te[:,1], 'LV3': T_te[:,2],
            'PC1': Z_te[:,0], 'PC2': Z_te[:,1],
        }).to_excel(writer, sheet_name='Test_Predictions_Graphene', index=False)

        # Sheets 5-10: Confusion matrices
        for sheet, cm in [
            ('CM_PLSDA_5fold_Train',    cm_pls_cv5),
            ('CM_PLSDA_LOO_Train',      cm_pls_loo),
            ('CM_PLSDA_Test_Graphene',  cm_pls_test),
            ('CM_SVM_5fold_Train',      cm_svm_cv5),
            ('CM_SVM_LOO_Train',        cm_svm_loo),
            ('CM_SVM_Test_Graphene',    cm_svm_test),
        ]:
            pd.DataFrame(cm,
                         index=pd.Index(classes, name='True\\Predicted'),
                         columns=classes).to_excel(writer, sheet_name=sheet)

        # Sheet 11: VIP scores
        pd.DataFrame({
            'Wavenumber_cm1':       wavenumbers,
            'VIP_Score':            vip,
            'VIP_gt_1':             vip > 1,
            f'Selected_Top{N_VIP}': sel_mask,
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
    print(f"  RESULTS  VIP top-{N_VIP}")
    print(f"{'='*68}")
    print(f"  Train: {len(y_tr)} spectra (all 4 classes)")
    print(f"  Test : {len(y_te)} spectra (GRAPHENE ONLY)")
    print(f"\n  {'MODEL':<10} {'5-fold CV (tr)':>15} {'LOO-CV (tr)':>13} {'Test-Gr':>10}")
    print(f"  {'-'*52}")
    print(f"  {'PLS-DA':<10} {acc_pls_cv5*100:>14.1f}% {acc_pls_loo*100:>12.1f}% {acc_pls_test*100:>9.1f}%")
    print(f"  {'SVM-RBF':<10} {acc_svm_cv5*100:>14.1f}% {acc_svm_loo*100:>12.1f}% {acc_svm_test*100:>9.1f}%")
    print(f"\n  Permutation p = {p_val:.4f}  [{sig}]")
    print(f"  Output: {OUTPUT_DIR}")
    print(f"{'='*68}\n")

print("\n  ALL DONE — three VIP sets completed.")
for n in VIP_SETS:
    print(f"    {os.path.join(DATA_PATH, f'PLSDA_SVM_4Class_VIP{n}')}")
