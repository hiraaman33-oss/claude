"""
=============================================================================
PLS-DA  –  4-CLASS  –  Book2.xlsx  –  68 / 32 STRATIFIED SPLIT
=============================================================================
File   : Book2.xlsx   (C:/Users/Hira Aman/Desktop/PROF_DOMENICO'S)
Sheet  : 4 - Normalised
Layout : Row 0 = header (sample names); Column 0 = Wavenumber_cm-1
         Each COLUMN is one spectrum  (100 spectra total)

Four classes (25 measurements each, 3 sessions):
  1. SSY-PDMS
  2. SSY-Gr-PDMS        (GrSL-PDMS = same class)
  3. SSY-SiO2/Si
  4. SSY-Gr-SiO2/Si     (GrSL-SiO2/Si = same class)

Sessions:
  ·  lugl   → Luglio  2025  (July)
  ·  marzo  → Marzo   2026  (March)
  ·  maggio → Maggio  2026  (May)

Train / Test split : 68 / 32  stratified (random_state=42)
  Train  : all 68 spectra  (4 classes, 17 each)
  Test   : GRAPHENE ONLY — SSY-Gr-PDMS (8) + SSY-Gr-SiO2/Si (8) = 16
           Non-graphene test spectra (SSY-PDMS + SSY-SiO2/Si = 16) discarded

VIP feature sets   : top-1000 / top-1500 / top-1800  (run all three)
CV                 : 5-fold stratified (train only)
Permutations       : 999
Spectral window    : 450 – 1800 cm-1

OUTPUTS  → PROF_DOMENICO'S\PLSDA_4Class_Book2_VIP<N>\
  01_Scores_LV1_LV2_train+test.png
  02_Scores_LV1_LV3_train+test.png
  03_Scores_3D_LV1_LV2_LV3.png
  04_Confusion_Matrix_CV.png
  05_Confusion_Matrix_Test_Graphene.png
  06_VIP_Scores.png
  07_Loadings_LV1_LV2.png
  08_Permutation_Test.png
  09_Mean_Spectra.png
  PLSDA_4Class_Book2_VIP<N>_Results.xlsx

HOW TO RUN (PyCharm Terminal):
  pip install pandas openpyxl scikit-learn scipy matplotlib
  python PLSDA_4Class_Book2_6832.py
=============================================================================
"""

import os, re
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D       # noqa
from sklearn.cross_decomposition import PLSRegression
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import (train_test_split, cross_val_predict,
                                     StratifiedKFold, permutation_test_score)
from sklearn.metrics import (confusion_matrix, classification_report,
                             accuracy_score, ConfusionMatrixDisplay)
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, ClassifierMixin
import warnings
warnings.filterwarnings('ignore')

# ═══════════════════════════════════════════════════════════════════════
# 0.  USER SETTINGS  ← only section you need to edit
# ═══════════════════════════════════════════════════════════════════════
DATA_PATH      = r"C:\Users\Hira Aman\Desktop\PROF_DOMENICO'S"
FILE_NAME      = "Book2.xlsx"
SHEET_NAME     = "4 - Normalised"

WAVENUMBER_MIN = 450        # cm-1
WAVENUMBER_MAX = 1800       # cm-1

N_COMPONENTS   = 5          # PLS latent variables
CV_FOLDS       = 5          # stratified k-fold
N_PERM         = 999        # permutation test repeats
RANDOM_STATE   = 42

VIP_SETS       = [1000, 1500, 1800]

# Graphene classes kept for test evaluation; non-graphene test discarded
GRAPHENE_CLASSES = ['SSY-Gr-PDMS', 'SSY-Gr-SiO2/Si']

# ═══════════════════════════════════════════════════════════════════════
# 1.  COLUMN-NAME → CLASS MAPPING
# ═══════════════════════════════════════════════════════════════════════
def assign_class(label: str):
    s = str(label).strip().lower()
    s = re.sub(r'^s\d+[-_]', '', s)
    s = re.sub(r'^(lugl|marzo\d*|maggio\d*)[-_]?(ssy[-_]?)?', '', s)
    s = s.replace('_', '-').replace(' ', '-')
    s = s.replace('grsl', 'gr')

    if   re.search(r'gr[-_]?sio2', s): return 'SSY-Gr-SiO2/Si'
    elif re.search(r'gr[-_]?pdms', s): return 'SSY-Gr-PDMS'
    elif re.search(r'sio2',        s): return 'SSY-SiO2/Si'
    elif re.search(r'pdms',        s): return 'SSY-PDMS'
    else:
        return None

# ═══════════════════════════════════════════════════════════════════════
# 2.  VIP SCORE FUNCTION
# ═══════════════════════════════════════════════════════════════════════
def vip_scores(model):
    T_ = model.x_scores_
    W_ = model.x_weights_
    Q_ = model.y_loadings_
    p  = W_.shape[0]
    SS = np.sum(T_**2, axis=0) * np.sum(Q_**2, axis=0)
    Wn = W_ / np.linalg.norm(W_, axis=0)
    return np.sqrt(p * np.sum(SS * Wn**2, axis=1) / np.sum(SS))

# ═══════════════════════════════════════════════════════════════════════
# 3.  PLS-DA CLASSIFIER WRAPPER
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
raw   = pd.read_excel(fpath, sheet_name=SHEET_NAME, header=0, index_col=0,
                      engine='openpyxl')

wavenumbers_all = raw.index.astype(float).values
col_names       = raw.columns.tolist()

classes_raw = [assign_class(c) for c in col_names]
unassigned  = [c for c, cl in zip(col_names, classes_raw) if cl is None]
if unassigned:
    print(f"  [WARNING] {len(unassigned)} unassigned labels: {unassigned[:5]}")

keep_idx = [i for i, cl in enumerate(classes_raw) if cl is not None]
col_names   = [col_names[i]   for i in keep_idx]
classes_raw = [classes_raw[i] for i in keep_idx]
raw         = raw.iloc[:, keep_idx]

from collections import Counter
print(f"  Loaded  : {raw.shape[1]} spectra x {raw.shape[0]} wavenumber points")
print(f"  Classes : {dict(Counter(classes_raw))}")

# ═══════════════════════════════════════════════════════════════════════
# 6.  CROP TO SPECTRAL WINDOW
# ═══════════════════════════════════════════════════════════════════════
mask = (wavenumbers_all >= WAVENUMBER_MIN) & (wavenumbers_all <= WAVENUMBER_MAX)
wavenumbers = wavenumbers_all[mask]
X_all       = raw.values[mask, :].T.astype(float)
y_all       = np.array(classes_raw)

print(f"  Spectral window : {WAVENUMBER_MIN}-{WAVENUMBER_MAX} cm-1  "
      f"({X_all.shape[1]} variables)")

# ═══════════════════════════════════════════════════════════════════════
# 7.  68 / 32 STRATIFIED SPLIT  ← DONE FIRST, BEFORE ANY FIT
# ═══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 68)
print("  STEP 2 — Train/Test split  (68/32 stratified)")
print("=" * 68)

X_tr, X_te_all, y_tr, y_te_all = train_test_split(
    X_all, y_all,
    test_size    = 0.32,
    stratify     = y_all,
    random_state = RANDOM_STATE,
)

# ── Filter test to GRAPHENE ONLY — discard non-graphene test spectra ──
graph_mask = np.isin(y_te_all, GRAPHENE_CLASSES)
X_te       = X_te_all[graph_mask]
y_te       = y_te_all[graph_mask]

print(f"  Train (all 4 classes) : {X_tr.shape[0]}  {dict(Counter(y_tr))}")
print(f"  Test  (all 32, before filter) : {X_te_all.shape[0]}  {dict(Counter(y_te_all))}")
print(f"  Test  (graphene only, KEPT)   : {X_te.shape[0]}   {dict(Counter(y_te))}")
print(f"  Test  (non-graphene, DISCARDED): "
      f"{X_te_all.shape[0] - X_te.shape[0]}  "
      f"{dict(Counter(y_te_all[~graph_mask]))}")

le = LabelEncoder().fit(y_all)

# ═══════════════════════════════════════════════════════════════════════
# 8.  LOOP OVER VIP FEATURE-SET SIZES
# ═══════════════════════════════════════════════════════════════════════
for N_VIP in VIP_SETS:

    OUTPUT_DIR = os.path.join(DATA_PATH, f"PLSDA_4Class_Book2_VIP{N_VIP}")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("\n" + "=" * 68)
    print(f"  VIP TOP-{N_VIP}")
    print("=" * 68)

    # ── 8a. Scale TRAIN only ─────────────────────────────────────────
    scaler   = StandardScaler()
    X_tr_sc  = scaler.fit_transform(X_tr)
    X_te_sc  = scaler.transform(X_te)          # graphene test only

    # ── 8b. VIP on TRAIN only ────────────────────────────────────────
    print(f"\n  STEP 3 — VIP selection on training set only (top-{N_VIP})")
    pls_vip  = PLSRegression(n_components=N_COMPONENTS, scale=False, max_iter=5000)
    Y_tr_d   = pd.get_dummies(pd.Series(y_tr)).values.astype(float)
    pls_vip.fit(X_tr_sc, Y_tr_d)
    vip      = vip_scores(pls_vip)
    top_idx  = np.argsort(vip)[-N_VIP:]
    wavenumbers_sel = wavenumbers[top_idx]

    X_tr_sel = X_tr_sc[:, top_idx]
    X_te_sel = X_te_sc[:, top_idx]
    print(f"  VIP > 1 : {np.sum(vip > 1)} / {len(vip)} variables total")
    print(f"  Selected top-{N_VIP} from {len(vip)} variables")

    # ── 8c. Final PLS-DA on TRAIN ────────────────────────────────────
    print(f"\n  STEP 4 — Fitting PLS-DA ({N_COMPONENTS} LVs) on training set")
    pls_final = PLSRegression(n_components=N_COMPONENTS, scale=False, max_iter=5000)
    pls_final.fit(X_tr_sel, Y_tr_d)

    T_tr = pls_final.x_scores_
    P    = pls_final.x_loadings_

    y_tr_pred = le.classes_[np.argmax(pls_final.predict(X_tr_sel), axis=1)]
    resubst   = accuracy_score(y_tr, y_tr_pred)
    print(f"  Re-substitution accuracy (train) : {resubst*100:.1f} %")

    # ── 8d. External test prediction (GRAPHENE ONLY) ─────────────────
    print(f"\n  STEP 5 — External test prediction  (graphene only, n={len(y_te)})")
    T_te_raw  = pls_final.transform(X_te_sel)
    T_te      = T_te_raw[0] if isinstance(T_te_raw, tuple) else T_te_raw
    y_te_pred = le.classes_[np.argmax(pls_final.predict(X_te_sel), axis=1)]
    test_acc  = accuracy_score(y_te, y_te_pred)
    print(f"  Test accuracy (graphene, n={len(y_te)}) : {test_acc*100:.1f} %")
    print(f"\n  Test classification report (graphene only):")
    print(classification_report(y_te, y_te_pred,
                                 labels=GRAPHENE_CLASSES,
                                 target_names=GRAPHENE_CLASSES, digits=3))

    # ── 8e. 5-fold CV on TRAIN ───────────────────────────────────────
    print(f"\n  STEP 6 — {CV_FOLDS}-fold stratified CV on training set")
    cv_pipe = Pipeline([
        ('sc',    StandardScaler()),
        ('plsda', PLSDAClassifier(n_components=N_COMPONENTS))
    ])
    cv      = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True,
                               random_state=RANDOM_STATE)
    y_cv    = cross_val_predict(cv_pipe, X_tr[:, top_idx], y_tr, cv=cv)
    cv_acc  = accuracy_score(y_tr, y_cv)
    print(f"  CV accuracy ({CV_FOLDS}-fold, train) : {cv_acc*100:.1f} %")
    print(f"\n  CV classification report (all 4 classes):")
    print(classification_report(y_tr, y_cv,
                                 target_names=le.classes_, digits=3))

    # ── 8f. Permutation test (TRAIN) ─────────────────────────────────
    print(f"\n  STEP 7 — Permutation test ({N_PERM} permutations, train only)")
    obs_score, perm_scores, p_val = permutation_test_score(
        cv_pipe, X_tr[:, top_idx], y_tr,
        scoring='accuracy', cv=cv,
        n_permutations=N_PERM,
        random_state=RANDOM_STATE, n_jobs=-1,
    )
    sig = 'SIGNIFICANT (p<0.05)' if p_val < 0.05 else 'NOT significant'
    print(f"  Observed CV accuracy : {obs_score*100:.2f} %")
    print(f"  Permutation p-value  : {p_val:.4f}  [{sig}]")

    # ── 8g. R²X ──────────────────────────────────────────────────────
    SS_tot = np.sum(X_tr_sel ** 2)
    r2x    = []
    for lv in range(1, N_COMPONENTS + 1):
        Xrec = pls_final.x_scores_[:, :lv] @ pls_final.x_loadings_[:, :lv].T
        r2x.append(1 - np.sum((X_tr_sel - Xrec)**2) / SS_tot)

    def lv_pct(i):
        return (r2x[i] - (r2x[i-1] if i > 0 else 0)) * 100

    print("\n  Cumulative R2X (train):")
    for i, r in enumerate(r2x, 1):
        print(f"    LV{i}: {r*100:.2f} %")

    # ── Confusion matrices ────────────────────────────────────────────
    cm_cv   = confusion_matrix(y_tr, y_cv,    labels=le.classes_)
    # 4-column CM: rows = graphene true classes, cols = all 4 predicted
    cm_test = confusion_matrix(y_te, y_te_pred, labels=le.classes_)

    title_tag = (f"VIP top-{N_VIP}  |  CV={cv_acc*100:.1f}%  "
                 f"|  Test(Gr)={test_acc*100:.1f}%  |  p={p_val:.3f}")

    # ═════════════════════════════════════════════════════════════════
    # FIGURES
    # ═════════════════════════════════════════════════════════════════

    def score_scatter_2d(ax, lv_x, lv_y):
        # Train — all 4 classes
        for cls in le.classes_:
            idx_tr = y_tr == cls
            ax.scatter(T_tr[idx_tr, lv_x], T_tr[idx_tr, lv_y],
                       c=PALETTE[cls], marker=MARKER[cls], s=55,
                       edgecolors='k', lw=0.4, alpha=0.75,
                       label=f'{cls} (train)', zorder=3)
        # Test — graphene only (correct = star, wrong = X)
        for cls in GRAPHENE_CLASSES:
            idx_ok  = (y_te == cls) & (y_te_pred == cls)
            idx_err = (y_te == cls) & (y_te_pred != cls)
            ax.scatter(T_te[idx_ok,  lv_x], T_te[idx_ok,  lv_y],
                       c=PALETTE[cls], marker='*', s=220,
                       edgecolors='k', lw=0.6, alpha=1.0,
                       label=f'{cls} (test-Gr OK)', zorder=5)
            if idx_err.any():
                ax.scatter(T_te[idx_err, lv_x], T_te[idx_err, lv_y],
                           c=PALETTE[cls], marker='X', s=180,
                           edgecolors='red', lw=1.0, alpha=1.0,
                           label=f'{cls} (test-Gr FAIL)', zorder=6)
        ax.axhline(0, color='grey', lw=0.6, ls='--')
        ax.axvline(0, color='grey', lw=0.6, ls='--')
        ax.legend(fontsize=7, framealpha=0.8, ncol=2)
        ax.grid(True, alpha=0.25)

    # Fig 1 – LV1 vs LV2
    fig, ax = plt.subplots(figsize=(8, 6))
    score_scatter_2d(ax, 0, 1)
    ax.set_xlabel(f'LV1  ({lv_pct(0):.1f}% var.)', fontsize=12)
    ax.set_ylabel(f'LV2  ({lv_pct(1):.1f}% var.)', fontsize=12)
    ax.set_title(f'PLS-DA Scores  LV1 x LV2\n{title_tag}',
                 fontsize=10, fontweight='bold')
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, '01_Scores_LV1_LV2.png'),
                dpi=180, bbox_inches='tight')
    plt.close(fig); print('[SAVED] 01_Scores_LV1_LV2.png')

    # Fig 2 – LV1 vs LV3
    fig, ax = plt.subplots(figsize=(8, 6))
    score_scatter_2d(ax, 0, 2)
    ax.set_xlabel(f'LV1  ({lv_pct(0):.1f}% var.)', fontsize=12)
    ax.set_ylabel(f'LV3  ({lv_pct(2):.1f}% var.)', fontsize=12)
    ax.set_title(f'PLS-DA Scores  LV1 x LV3\n{title_tag}',
                 fontsize=10, fontweight='bold')
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, '02_Scores_LV1_LV3.png'),
                dpi=180, bbox_inches='tight')
    plt.close(fig); print('[SAVED] 02_Scores_LV1_LV3.png')

    # Fig 3 – 3D  LV1 x LV2 x LV3
    fig3 = plt.figure(figsize=(9, 7))
    ax3  = fig3.add_subplot(111, projection='3d')
    for cls in le.classes_:
        idx_tr = y_tr == cls
        ax3.scatter(T_tr[idx_tr,0], T_tr[idx_tr,1], T_tr[idx_tr,2],
                    c=PALETTE[cls], marker=MARKER[cls], s=50,
                    edgecolors='k', lw=0.4, alpha=0.75,
                    label=f'{cls} (train)', zorder=3)
    for cls in GRAPHENE_CLASSES:
        idx_ok  = (y_te == cls) & (y_te_pred == cls)
        idx_err = (y_te == cls) & (y_te_pred != cls)
        ax3.scatter(T_te[idx_ok,0], T_te[idx_ok,1], T_te[idx_ok,2],
                    c=PALETTE[cls], marker='*', s=220,
                    edgecolors='k', lw=0.6, alpha=1.0,
                    label=f'{cls} (test-Gr OK)', zorder=5)
        if idx_err.any():
            ax3.scatter(T_te[idx_err,0], T_te[idx_err,1], T_te[idx_err,2],
                        c=PALETTE[cls], marker='X', s=180,
                        edgecolors='red', lw=1.0, alpha=1.0,
                        label=f'{cls} (test-Gr FAIL)', zorder=6)
    ax3.set_xlabel(f'LV1 ({lv_pct(0):.1f}%)', fontsize=9, labelpad=8)
    ax3.set_ylabel(f'LV2 ({lv_pct(1):.1f}%)', fontsize=9, labelpad=8)
    ax3.set_zlabel(f'LV3 ({lv_pct(2):.1f}%)', fontsize=9, labelpad=8)
    ax3.set_title(f'PLS-DA 3D Scores  (train=all 4 classes, test=graphene only)\n{title_tag}',
                  fontsize=9, fontweight='bold', pad=12)
    ax3.legend(fontsize=7, loc='upper left', framealpha=0.85)
    ax3.grid(True, alpha=0.2)
    ax3.view_init(elev=22, azim=45)
    fig3.tight_layout()
    fig3.savefig(os.path.join(OUTPUT_DIR, '03_Scores_3D.png'),
                 dpi=180, bbox_inches='tight')
    plt.close(fig3); print('[SAVED] 03_Scores_3D.png')

    # Fig 4 – Confusion matrix CV  (all 4 classes, train)
    fig, ax = plt.subplots(figsize=(6, 5))
    ConfusionMatrixDisplay(cm_cv, display_labels=le.classes_).plot(
        ax=ax, colorbar=False, cmap='Blues', xticks_rotation=30)
    ax.set_title(f'Confusion Matrix – {CV_FOLDS}-fold CV (train, all 4 classes)\n'
                 f'Accuracy = {cv_acc*100:.1f}%',
                 fontsize=10, fontweight='bold')
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, '04_Confusion_Matrix_CV.png'),
                dpi=180, bbox_inches='tight')
    plt.close(fig); print('[SAVED] 04_Confusion_Matrix_CV.png')

    # Fig 5 – Confusion matrix Test  (graphene only, 4-col to show any mis-pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    ConfusionMatrixDisplay(cm_test, display_labels=le.classes_).plot(
        ax=ax, colorbar=False, cmap='Oranges', xticks_rotation=30)
    ax.set_title(f'Confusion Matrix – External Test (graphene only, n={len(y_te)})\n'
                 f'Accuracy = {test_acc*100:.1f}%  '
                 f'(non-graphene rows = 0, discarded)',
                 fontsize=9, fontweight='bold')
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, '05_Confusion_Matrix_Test_Graphene.png'),
                dpi=180, bbox_inches='tight')
    plt.close(fig); print('[SAVED] 05_Confusion_Matrix_Test_Graphene.png')

    # Fig 6 – VIP scores
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(wavenumbers, vip, color='steelblue', lw=1.2)
    ax.axhline(1, color='red', ls='--', lw=1, label='VIP = 1 threshold')
    ax.fill_between(wavenumbers, vip, 1, where=(vip > 1),
                    alpha=0.25, color='red',
                    label=f'VIP > 1  ({np.sum(vip>1)} variables)')
    ax.axvline(wavenumbers[np.argsort(vip)[-1]], color='gold', ls=':', lw=1.5,
               label=f'Top VIP @ {wavenumbers[np.argsort(vip)[-1]]:.0f} cm-1')
    ax.set_xlabel('Wavenumber (cm-1)', fontsize=12)
    ax.set_ylabel('VIP score',         fontsize=12)
    ax.set_title(f'Variable Importance in Projection (VIP)\n'
                 f'Computed on training set only  |  VIP top-{N_VIP}',
                 fontsize=11, fontweight='bold')
    ax.set_xlim(WAVENUMBER_MIN, WAVENUMBER_MAX)
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, '06_VIP_Scores.png'),
                dpi=180, bbox_inches='tight')
    plt.close(fig); print('[SAVED] 06_VIP_Scores.png')

    # Fig 7 – Loadings LV1 & LV2
    P_full = pls_final.x_loadings_
    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    for i, ax in enumerate(axes):
        ax.plot(wavenumbers_sel, P_full[:, i],
                color=['steelblue', 'darkorange'][i], lw=1.2)
        ax.axhline(0, color='k', lw=0.7)
        ax.set_ylabel(f'Loading LV{i+1}', fontsize=11)
        ax.grid(True, alpha=0.3)
    axes[-1].set_xlabel('Wavenumber (cm-1)', fontsize=12)
    axes[0].set_title(f'PLS-DA Loadings  LV1 & LV2  (VIP top-{N_VIP})',
                      fontsize=12, fontweight='bold')
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, '07_Loadings_LV1_LV2.png'),
                dpi=180, bbox_inches='tight')
    plt.close(fig); print('[SAVED] 07_Loadings_LV1_LV2.png')

    # Fig 8 – Permutation test
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.hist(perm_scores, bins=35, color='lightsteelblue', edgecolor='white',
            label='Permuted accuracy')
    ax.axvline(obs_score, color='crimson', lw=2,
               label=f'Observed: {obs_score*100:.1f}%  (p = {p_val:.4f})')
    ax.set_xlabel('Cross-validated accuracy (train)', fontsize=12)
    ax.set_ylabel('Count',                            fontsize=12)
    ax.set_title(f'Permutation Test  (n = {N_PERM})  |  train only\n'
                 f'Chance level = 25%  (4 classes)',
                 fontsize=11, fontweight='bold')
    ax.legend(fontsize=10); ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, '08_Permutation_Test.png'),
                dpi=180, bbox_inches='tight')
    plt.close(fig); print('[SAVED] 08_Permutation_Test.png')

    # Fig 9 – Mean spectra
    fig, ax = plt.subplots(figsize=(10, 4.5))
    for cls in le.classes_:
        ax.plot(wavenumbers, X_all[y_all == cls].mean(axis=0),
                color=PALETTE[cls], label=cls, lw=1.5)
    ax.set_xlabel('Wavenumber (cm-1)', fontsize=12)
    ax.set_ylabel('Normalised intensity (a.u.)', fontsize=12)
    ax.set_title('Mean Raman spectra per class (normalised, all 100)',
                 fontsize=12, fontweight='bold')
    ax.set_xlim(WAVENUMBER_MIN, WAVENUMBER_MAX)
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, '09_Mean_Spectra.png'),
                dpi=180, bbox_inches='tight')
    plt.close(fig); print('[SAVED] 09_Mean_Spectra.png')

    # ═════════════════════════════════════════════════════════════════
    # EXCEL RESULTS FILE
    # ═════════════════════════════════════════════════════════════════
    out_xlsx = os.path.join(OUTPUT_DIR,
                            f'PLSDA_4Class_Book2_VIP{N_VIP}_Results.xlsx')
    with pd.ExcelWriter(out_xlsx, engine='openpyxl') as writer:

        # Summary
        pd.DataFrame({
            'Metric': [
                'Dataset', 'File', 'Sheet', 'Spectral window', 'Spectra total',
                'Train (all 4 classes)', 'Test (graphene only)',
                'Test discarded (non-graphene)',
                'VIP features', 'CV folds',
                'CV accuracy (train)', 'Test accuracy (graphene)',
                'Permutation p-value', 'Permutation result',
                'R2X LV1', 'R2X LV2', 'R2X LV3',
            ],
            'Value': [
                'Book2 – 4 class SSY Raman', FILE_NAME, SHEET_NAME,
                f'{WAVENUMBER_MIN}-{WAVENUMBER_MAX} cm-1',
                len(y_all),
                f'{X_tr.shape[0]} spectra  {dict(Counter(y_tr))}',
                f'{X_te.shape[0]} spectra  {dict(Counter(y_te))}',
                f'{X_te_all.shape[0] - X_te.shape[0]} spectra (discarded)',
                N_VIP, CV_FOLDS,
                f'{cv_acc*100:.2f} %',
                f'{test_acc*100:.2f} %',
                f'{p_val:.4f}',
                'Significant (p<0.05)' if p_val < 0.05 else 'NOT significant',
                f'{r2x[0]*100:.2f} %',
                f'{r2x[1]*100:.2f} %',
                f'{r2x[2]*100:.2f} %',
            ],
        }).to_excel(writer, sheet_name='Summary', index=False)

        # CV predictions (train, all 4 classes)
        pd.DataFrame({
            'True':      y_tr,
            'CV_Pred':   y_cv,
            'Correct':   y_tr == y_cv,
        }).to_excel(writer, sheet_name='CV_Predictions', index=False)

        # Test predictions (graphene only)
        pd.DataFrame({
            'True':       y_te,
            'Test_Pred':  y_te_pred,
            'Correct':    y_te == y_te_pred,
            'LV1': T_te[:, 0], 'LV2': T_te[:, 1], 'LV3': T_te[:, 2],
        }).to_excel(writer, sheet_name='Test_Predictions_Graphene', index=False)

        # Confusion matrices
        pd.DataFrame(cm_cv,
                     index=pd.Index(le.classes_, name='True\\Predicted'),
                     columns=le.classes_).to_excel(writer, sheet_name='CM_CV_Train')
        pd.DataFrame(cm_test,
                     index=pd.Index(le.classes_, name='True\\Predicted'),
                     columns=le.classes_).to_excel(writer, sheet_name='CM_Test_Graphene')

        # VIP scores
        pd.DataFrame({'Wavenumber_cm1': wavenumbers, 'VIP': vip}
                     ).to_excel(writer, sheet_name='VIP_Scores', index=False)

        # Loadings
        load_df = pd.DataFrame(P_full,
                               columns=[f'LV{i+1}' for i in range(N_COMPONENTS)])
        load_df.insert(0, 'Wavenumber_cm1', wavenumbers_sel)
        load_df.to_excel(writer, sheet_name='Loadings', index=False)

        # PLS-DA scores train
        score_df = pd.DataFrame(T_tr,
                                columns=[f'LV{i+1}' for i in range(N_COMPONENTS)])
        score_df.insert(0, 'Class', y_tr)
        score_df.to_excel(writer, sheet_name='Scores_Train', index=False)

        # PLS-DA scores test (graphene)
        score_te = pd.DataFrame(T_te,
                                columns=[f'LV{i+1}' for i in range(N_COMPONENTS)])
        score_te.insert(0, 'Class', y_te)
        score_te.to_excel(writer, sheet_name='Scores_Test_Graphene', index=False)

        # Permutation
        pd.DataFrame({'Permuted_Accuracy': perm_scores}
                     ).to_excel(writer, sheet_name='Permutation_Test', index=False)

    print(f'\n[SAVED] {out_xlsx}')

    # ── Final summary ────────────────────────────────────────────────
    print("\n" + "=" * 68)
    print(f"  RESULTS SUMMARY  –  VIP top-{N_VIP}")
    print("=" * 68)
    print(f"  Train  : {X_tr.shape[0]} spectra (all 4 classes)")
    print(f"  Test   : {X_te.shape[0]} spectra (GRAPHENE ONLY — discarded non-graphene)")
    print(f"  CV accuracy (train, 5-fold) : {cv_acc*100:.2f} %")
    print(f"  Test accuracy (graphene)    : {test_acc*100:.2f} %")
    print(f"  Permutation p               : {p_val:.4f}  [{sig}]")
    print(f"\n  Per-class recall (test, graphene only):")
    for cls in GRAPHENE_CLASSES:
        i  = list(le.classes_).index(cls)
        n  = cm_test[i].sum()
        if n > 0:
            print(f"    {cls:<25} : {cm_test[i,i]}/{n}  "
                  f"({cm_test[i,i]/n*100:.0f}%)")
    print(f"\n  Output folder : {OUTPUT_DIR}")
    print("=" * 68)

print("\n\n  ALL DONE — three VIP sets completed.")
