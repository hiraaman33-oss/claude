"""
=============================================================================
PLS-DA + SVM-RBF  |  NO TRAIN/TEST SPLIT  |  ALL 80 SPECTRA
SSY Raman Spectroscopy  |  Four Substrate Classes
SPECTRAL WINDOW : 450 - 1800 cm-1

MODELS    : PLS-DA  (5 LVs, 5-fold CV only)
            SVM-RBF (C=10, gamma=scale, 5-fold CV only)
VIP       : ALL variables with VIP > 1  (3-LV PLS-DA on all 80 spectra)
OUTPUTS   -> C:/Users/Hira Aman/Desktop/PROF_DOMENICOS/PLSDA_SVM_VIPgt1/

  01_PLSDA_Scores_LV1_LV2.png
  02_PLSDA_Scores_LV1_LV3.png
  03_PLSDA_Scores_LV2_LV3.png
  04_PLSDA_Scores_3D.png
  05_PLSDA_Accuracy_Matrix_5fold.png
  06_PLSDA_Recall_Matrix_5fold.png
  07_PLSDA_VIP_Scores.png
  08_PLSDA_Loadings.png
  09_PLSDA_Permutation.png
  10_SVM_Cluster_5fold.png
  11_SVM_Accuracy_Matrix_5fold.png
  12_SVM_Recall_Matrix_5fold.png
  13_Accuracy_Comparison.png
  14_Per_Class_Recall_5fold.png
  15_Mean_Spectra.png
  Results_5fold.xlsx

HOW TO RUN (PyCharm Terminal):
  pip install pandas openpyxl scikit-learn scipy matplotlib xlrd
=============================================================================
"""

import os
import re
import subprocess
import tempfile
import warnings

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D          # noqa: F401

from sklearn.cross_decomposition import PLSRegression
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import StratifiedKFold, cross_val_predict, permutation_test_score
from sklearn.metrics import confusion_matrix, accuracy_score, ConfusionMatrixDisplay
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
FILE_NAME      = "Book1.xls"
SHEET_NAME     = "Sheet1"

WAVENUMBER_MIN = 450
WAVENUMBER_MAX = 1800
SG_WINDOW      = 7
SG_POLY        = 2
N_COMPONENTS   = 5
CV_FOLDS       = 5
N_PERM         = 999
RANDOM_STATE   = 42

OUTPUT_DIR = os.path.join(DATA_PATH, "PLSDA_SVM_VIPgt1")

try:
    os.makedirs(OUTPUT_DIR, exist_ok=True)
except Exception:
    pass
if not os.path.isdir(OUTPUT_DIR):
    try:
        subprocess.run('cmd /c mkdir "%s"' % OUTPUT_DIR,
                       shell=True, check=True, capture_output=True)
    except Exception as _e:
        raise RuntimeError("Cannot create folder: %s\n%s" % (OUTPUT_DIR, _e))
print("[INFO] Output folder : %s" % OUTPUT_DIR)

# =============================================================================
# 1.  FILE LOADER
# =============================================================================
def load_xls(file_path, sheet):
    try:
        df = pd.read_excel(file_path, sheet_name=sheet,
                           header=0, index_col=0, engine='xlrd')
        print("[INFO] Engine: xlrd"); return df
    except Exception as e1:
        print("[INFO] xlrd failed (%s)" % e1.__class__.__name__)
    try:
        import win32com.client
        tmp = file_path.replace('.xls', '_tmp.xlsx')
        xl  = win32com.client.Dispatch("Excel.Application")
        xl.Visible = False
        wb  = xl.Workbooks.Open(file_path)
        wb.SaveAs(tmp, FileFormat=51); wb.Close(); xl.Quit()
        df = pd.read_excel(tmp, sheet_name=sheet,
                           header=0, index_col=0, engine='openpyxl')
        os.remove(tmp)
        print("[INFO] Engine: Excel COM"); return df
    except Exception as e2:
        print("[INFO] Excel COM failed (%s)" % e2.__class__.__name__)
    try:
        import shutil
        lo = shutil.which("soffice") or shutil.which("libreoffice")
        if lo:
            tmp_dir = tempfile.mkdtemp()
            subprocess.run([lo, "--headless", "--convert-to", "xlsx",
                            "--outdir", tmp_dir, file_path],
                           check=True, capture_output=True, timeout=60)
            xlsx = os.path.join(tmp_dir,
                os.path.basename(file_path).replace('.xls', '.xlsx'))
            df = pd.read_excel(xlsx, sheet_name=sheet,
                               header=0, index_col=0, engine='openpyxl')
            print("[INFO] Engine: LibreOffice"); return df
    except Exception as e3:
        print("[INFO] LibreOffice failed (%s)" % e3.__class__.__name__)
    raise RuntimeError("Cannot read .xls — run: pip install xlrd")

# =============================================================================
# 2.  CLASS MAPPING
# =============================================================================
def assign_class(col_name):
    s = col_name.lower()
    s = re.sub(r'^s\d+[-_]', '', s)
    s = s.replace('grsl', 'gr')
    s = re.sub(r'^(lugl|marzo\d*)[-_](ssy[-_]?)?', '', s)
    s = s.replace('_', '-').replace(' ', '-')
    if   re.search(r'gr[-_]?sio2', s): return 'SSY-Gr-SiO2/Si'
    elif re.search(r'gr[-_]?pdms', s): return 'SSY-Gr-PDMS'
    elif re.search(r'sio2',        s): return 'SSY-SiO2/Si'
    elif re.search(r'pdms',        s): return 'SSY-PDMS'
    else:                               return None

# =============================================================================
# 3.  LOAD & CROP DATA
# =============================================================================
print("\n" + "=" * 65)
print("  PLS-DA + SVM-RBF  |  5-FOLD CV ONLY  |  80 SPECTRA")
print("=" * 65)

raw             = load_xls(os.path.join(DATA_PATH, FILE_NAME), SHEET_NAME)
wavenumbers_all = raw.index.astype(float).values
col_names_all   = raw.columns.tolist()
X_all           = raw.values.T.astype(float)

crop_mask   = (wavenumbers_all >= WAVENUMBER_MIN) & (wavenumbers_all <= WAVENUMBER_MAX)
wavenumbers = wavenumbers_all[crop_mask]
X_crop      = X_all[:, crop_mask]
print("[INFO] Window: %d-%d cm-1  (%d points)" % (WAVENUMBER_MIN, WAVENUMBER_MAX, X_crop.shape[1]))

classes_raw = np.array([assign_class(c) for c in col_names_all])
valid_mask  = classes_raw != None
X_crop      = X_crop[valid_mask]
classes     = classes_raw[valid_mask]
col_names   = [col_names_all[i] for i in range(len(col_names_all)) if valid_mask[i]]

le = LabelEncoder()
y  = le.fit_transform(classes)

print("[INFO] Samples: %d" % len(classes))
for cls, cnt in zip(*np.unique(classes, return_counts=True)):
    print("  %-25s : %d" % (cls, cnt))

# =============================================================================
# 4.  PRE-PROCESSING
# =============================================================================
X_sm = savgol_filter(X_crop, window_length=SG_WINDOW, polyorder=SG_POLY, axis=1)
print("[INFO] SG smoothing (window=%d, poly=%d)" % (SG_WINDOW, SG_POLY))

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
# 6.  VIP SELECTION
# =============================================================================
print("\n[VIP] 3-LV PLS-DA -> all variables with VIP > 1 ...")
sc_init  = StandardScaler().fit(X_sm)
Xs_init  = sc_init.transform(X_sm)
Yd_init  = pd.get_dummies(pd.Series(y)).values.astype(float)
pls_init = PLSRegression(n_components=3, scale=False, max_iter=10000)
pls_init.fit(Xs_init, Yd_init)
vip_all        = compute_vip(pls_init)
top_idx        = np.where(vip_all > 1)[0]
n_vip_selected = len(top_idx)
wn_sel         = wavenumbers[top_idx]
X_sel          = X_sm[:, top_idx]
print("[VIP] VIP > 1: %d/%d  |  range: %.0f-%.0f cm-1" % (
    n_vip_selected, len(vip_all), wn_sel.min(), wn_sel.max()))

# =============================================================================
# 7.  PLS-DA CLASSIFIER
# =============================================================================
class PLSDAClf(BaseEstimator, ClassifierMixin):
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

cv5 = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)

# =============================================================================
# 8.  FULL PLS-DA MODEL (visualisation)
# =============================================================================
sc_full  = StandardScaler().fit(X_sel)
X_scaled = sc_full.transform(X_sel)
Yd_full  = pd.get_dummies(pd.Series(y)).values.astype(float)
pls_full = PLSRegression(n_components=N_COMPONENTS, scale=False, max_iter=10000)
pls_full.fit(X_scaled, Yd_full)

T_all  = pls_full.x_scores_
P_load = pls_full.x_loadings_

SS_tot  = np.sum(X_scaled ** 2)
r2x_cum = [1 - np.sum((X_scaled -
            pls_full.x_scores_[:, :lv] @ pls_full.x_loadings_[:, :lv].T) ** 2) / SS_tot
           for lv in range(1, N_COMPONENTS + 1)]

resub_acc = accuracy_score(y, np.argmax(pls_full.predict(X_scaled), axis=1))
print("[PLS-DA] Re-substitution: %.1f%%" % (resub_acc * 100))
print("[PLS-DA] R2X: " + "  ".join(["LV%d=%.1f%%" % (i+1, r*100) for i, r in enumerate(r2x_cum)]))

# =============================================================================
# 9.  PLS-DA 5-FOLD CV
# =============================================================================
pls_pipe = Pipeline([('sc', StandardScaler()), ('plsda', PLSDAClf(N_COMPONENTS))])
print("\n[PLS-DA] %d-fold CV ..." % CV_FOLDS)
y_pls_5f   = cross_val_predict(pls_pipe, X_sel, y, cv=cv5)
acc_pls_5f = accuracy_score(y, y_pls_5f)
cm_pls_5f  = confusion_matrix(y, y_pls_5f)
print("[PLS-DA] %d-fold CV accuracy: %.1f%%" % (CV_FOLDS, acc_pls_5f * 100))

# =============================================================================
# 10. PERMUTATION TEST
# =============================================================================
print("\n[PLS-DA] Permutation test (%d perm.) ..." % N_PERM)
obs_score, perm_scores, p_val = permutation_test_score(
    pls_pipe, X_sel, y,
    scoring='accuracy', cv=cv5,
    n_permutations=N_PERM, random_state=RANDOM_STATE, n_jobs=-1)
sig_str = 'SIGNIFICANT (p<0.05)' if p_val < 0.05 else 'NOT significant'
print("[PLS-DA] Observed: %.2f%%   p=%.4f  [%s]" % (obs_score*100, p_val, sig_str))

# =============================================================================
# 11. SVM-RBF 5-FOLD CV
# =============================================================================
svm_pipe = Pipeline([
    ('sc',  StandardScaler()),
    ('svm', SVC(kernel='rbf', C=10, gamma='scale',
                probability=True, random_state=RANDOM_STATE))
])
print("\n[SVM] %d-fold CV ..." % CV_FOLDS)
y_svm_5f   = cross_val_predict(svm_pipe, X_sel, y, cv=cv5)
acc_svm_5f = accuracy_score(y, y_svm_5f)
cm_svm_5f  = confusion_matrix(y, y_svm_5f)
print("[SVM] %d-fold CV accuracy: %.1f%%" % (CV_FOLDS, acc_svm_5f * 100))

# =============================================================================
# 12. PCA FOR SVM CLUSTER PLOT
# =============================================================================
pca_viz = PCA(n_components=2, random_state=RANDOM_STATE)
Z_all   = pca_viz.fit_transform(X_scaled)
print("[PCA] PC1=%.1f%%  PC2=%.1f%%" % (
    pca_viz.explained_variance_ratio_[0]*100,
    pca_viz.explained_variance_ratio_[1]*100))

# =============================================================================
# 13. COLOURS / MARKERS / HELPERS
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

TTAG = "VIP > 1 (%d vars)  |  %d LVs  |  %d-%d cm-1  |  n=80" % (
    n_vip_selected, N_COMPONENTS, WAVENUMBER_MIN, WAVENUMBER_MAX)

def lv_var(i):
    return (r2x_cum[i] - (r2x_cum[i-1] if i > 0 else 0)) * 100

def save_fig(fig, fname):
    fig.savefig(os.path.join(OUTPUT_DIR, fname),
                dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print('[SAVED] %s' % fname)

n_per_cls = len(y) // len(le.classes_)   # 20

# =============================================================================
# 14. PLS-DA SCORE PLOTS  (no text overlap: legend outside, suptitle padded)
# =============================================================================
def scatter_scores(ax, lv_x, lv_y, cv_pred=None):
    for cls in le.classes_:
        idx = classes == cls
        ec  = []
        if cv_pred is not None:
            for i in np.where(idx)[0]:
                ec.append('red' if y[i] != cv_pred[i] else 'black')
        else:
            ec = ['black'] * int(idx.sum())
        ax.scatter(T_all[idx, lv_x], T_all[idx, lv_y],
                   c=PALETTE[cls], marker=MARKER[cls], s=70,
                   edgecolors=ec, linewidths=0.9,
                   alpha=0.85, label=cls, zorder=3)
    ax.axhline(0, color='grey', lw=0.7, ls='--')
    ax.axvline(0, color='grey', lw=0.7, ls='--')
    ax.grid(True, alpha=0.25)
    if cv_pred is not None:
        n_wrong = int((y != cv_pred).sum())
        ax.scatter([], [], c='white', edgecolors='red',
                   linewidths=1.5, s=60, label='Misclassified (%d)' % n_wrong)
    # Legend placed OUTSIDE the axes (bottom, horizontal) to avoid overlap
    ax.legend(fontsize=8, framealpha=0.9,
              loc='upper center', bbox_to_anchor=(0.5, -0.14),
              ncol=3, borderaxespad=0)

# Fig 01 — LV1 x LV2
fig, axes = plt.subplots(1, 2, figsize=(15, 6.5))
fig.suptitle('PLS-DA Score Plot  LV1 x LV2\n' + TTAG,
             fontsize=11, fontweight='bold', y=1.01)
for ax, pred, sub in zip(axes,
        [None,       y_pls_5f],
        ['Full model (re-substitution)', '%d-fold CV predictions' % CV_FOLDS]):
    scatter_scores(ax, 0, 1, cv_pred=pred)
    ax.set_xlabel('LV1  (%.1f%% var.)' % lv_var(0), fontsize=11, labelpad=6)
    ax.set_ylabel('LV2  (%.1f%% var.)' % lv_var(1), fontsize=11, labelpad=6)
    ax.set_title(sub, fontsize=10, fontweight='bold', pad=8)
fig.tight_layout(rect=[0, 0.06, 1, 1])
save_fig(fig, '01_PLSDA_Scores_LV1_LV2.png')

# Fig 02 — LV1 x LV3
fig, axes = plt.subplots(1, 2, figsize=(15, 6.5))
fig.suptitle('PLS-DA Score Plot  LV1 x LV3\n' + TTAG,
             fontsize=11, fontweight='bold', y=1.01)
for ax, pred, sub in zip(axes,
        [None,       y_pls_5f],
        ['Full model (re-substitution)', '%d-fold CV predictions' % CV_FOLDS]):
    scatter_scores(ax, 0, 2, cv_pred=pred)
    ax.set_xlabel('LV1  (%.1f%% var.)' % lv_var(0), fontsize=11, labelpad=6)
    ax.set_ylabel('LV3  (%.1f%% var.)' % lv_var(2), fontsize=11, labelpad=6)
    ax.set_title(sub, fontsize=10, fontweight='bold', pad=8)
fig.tight_layout(rect=[0, 0.06, 1, 1])
save_fig(fig, '02_PLSDA_Scores_LV1_LV3.png')

# Fig 03 — LV2 x LV3
fig, axes = plt.subplots(1, 2, figsize=(15, 6.5))
fig.suptitle('PLS-DA Score Plot  LV2 x LV3\n' + TTAG,
             fontsize=11, fontweight='bold', y=1.01)
for ax, pred, sub in zip(axes,
        [None,       y_pls_5f],
        ['Full model (re-substitution)', '%d-fold CV predictions' % CV_FOLDS]):
    scatter_scores(ax, 1, 2, cv_pred=pred)
    ax.set_xlabel('LV2  (%.1f%% var.)' % lv_var(1), fontsize=11, labelpad=6)
    ax.set_ylabel('LV3  (%.1f%% var.)' % lv_var(2), fontsize=11, labelpad=6)
    ax.set_title(sub, fontsize=10, fontweight='bold', pad=8)
fig.tight_layout(rect=[0, 0.06, 1, 1])
save_fig(fig, '03_PLSDA_Scores_LV2_LV3.png')

# Fig 04 — 3D
fig = plt.figure(figsize=(9, 7.5))
ax  = fig.add_subplot(111, projection='3d')
for cls in le.classes_:
    idx = classes == cls
    ax.scatter(T_all[idx,0], T_all[idx,1], T_all[idx,2],
               c=PALETTE[cls], marker=MARKER[cls], s=55,
               edgecolors='k', linewidths=0.4, alpha=0.85, label=cls)
ax.set_xlabel('LV1 (%.1f%%)' % lv_var(0), fontsize=9, labelpad=10)
ax.set_ylabel('LV2 (%.1f%%)' % lv_var(1), fontsize=9, labelpad=10)
ax.set_zlabel('LV3 (%.1f%%)' % lv_var(2), fontsize=9, labelpad=10)
ax.set_title('PLS-DA 3D Scores\n' + TTAG, fontsize=10, fontweight='bold', pad=14)
ax.legend(fontsize=8, ncol=2, loc='upper left',
          bbox_to_anchor=(0.0, 0.95), framealpha=0.85)
ax.view_init(elev=22, azim=45)
fig.tight_layout()
save_fig(fig, '04_PLSDA_Scores_3D.png')

# =============================================================================
# 15. CONFUSION MATRIX HELPERS
# =============================================================================
def plot_accuracy_matrix(cm, labels, title, fname, cmap):
    """
    Raw-count confusion matrix.
    Overall accuracy = trace(CM) / total.
    Each cell = number of samples.
    """
    fig, ax = plt.subplots(figsize=(7, 6))
    disp = ConfusionMatrixDisplay(cm, display_labels=labels)
    disp.plot(ax=ax, colorbar=True, cmap=cmap,
              xticks_rotation=25, values_format='d')
    ax.set_title(title, fontsize=11, fontweight='bold', pad=12)
    ax.set_xlabel('Predicted Class', fontsize=10, labelpad=8)
    ax.set_ylabel('True Class', fontsize=10, labelpad=8)
    # Fix: ensure tick labels do not clip
    ax.tick_params(axis='x', pad=4)
    ax.tick_params(axis='y', pad=4)
    fig.tight_layout()
    save_fig(fig, fname)


def plot_recall_matrix(cm, labels, title, fname, cmap):
    """
    Row-normalised confusion matrix — diagonal = Recall per class.
    Formula: CM_norm[i,j] = CM[i,j] / sum(CM[i,:])
    Each cell shows: normalised value  AND  raw count in brackets.
    """
    n_cls    = len(labels)
    row_sums = cm.sum(axis=1, keepdims=True).astype(float)
    cm_norm  = cm.astype(float) / row_sums

    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    im = ax.imshow(cm_norm, interpolation='nearest',
                   cmap=cmap, vmin=0, vmax=1)
    cbar = plt.colorbar(im, ax=ax, fraction=0.045, pad=0.05)
    cbar.set_label('Recall  (fraction)', fontsize=10, labelpad=8)
    cbar.ax.tick_params(labelsize=9)

    ticks = np.arange(n_cls)
    ax.set_xticks(ticks)
    ax.set_xticklabels(labels, rotation=25, ha='right', fontsize=9)
    ax.set_yticks(ticks)
    ax.set_yticklabels(labels, fontsize=9)

    # Cell text — value on first line, count on second
    for i in range(n_cls):
        for j in range(n_cls):
            val   = cm_norm[i, j]
            count = int(cm[i, j])
            color = 'white' if val > 0.55 else 'black'
            ax.text(j, i - 0.12, '%.2f' % val,
                    ha='center', va='center',
                    color=color, fontsize=11, fontweight='bold')
            ax.text(j, i + 0.22, '(%d)' % count,
                    ha='center', va='center',
                    color=color, fontsize=8)

    ax.set_ylabel('True Class', fontsize=11, labelpad=8)
    ax.set_xlabel('Predicted Class', fontsize=11, labelpad=8)
    ax.set_title(title, fontsize=10, fontweight='bold', pad=12)

    # Per-class recall summary inside a text box (no axis overlap)
    summary = '\n'.join(['%s : %.0f%%' % (lbl.replace('SSY-',''), cm_norm[i,i]*100)
                         for i, lbl in enumerate(labels)])
    ax.text(0.99, 0.99, 'Recall per class:\n' + summary,
            transform=ax.transAxes,
            fontsize=8, va='top', ha='right',
            bbox=dict(boxstyle='round,pad=0.4', fc='lightyellow',
                      ec='grey', alpha=0.85))

    fig.tight_layout()
    save_fig(fig, fname)

# =============================================================================
# 16. PLS-DA CONFUSION MATRICES  (Fig 05 Accuracy, Fig 06 Recall)
# =============================================================================
plot_accuracy_matrix(
    cm_pls_5f, le.classes_,
    'PLS-DA  Accuracy Matrix  |  %d-fold CV  (n=%d)\n'
    'Cells = sample counts   |   Overall accuracy = %.1f%%'
    % (CV_FOLDS, len(y), acc_pls_5f * 100),
    '05_PLSDA_Accuracy_Matrix_5fold.png', 'Blues')

plot_recall_matrix(
    cm_pls_5f, le.classes_,
    'PLS-DA  Recall Matrix  |  %d-fold CV  (n=%d)\n'
    'Diagonal = Recall per class   |   Formula: TP / (TP + FN)'
    % (CV_FOLDS, len(y)),
    '06_PLSDA_Recall_Matrix_5fold.png', 'Blues')

# =============================================================================
# 17. VIP SCORES
# =============================================================================
sel_mask = np.zeros(len(wavenumbers), dtype=bool)
sel_mask[top_idx] = True

fig, ax = plt.subplots(figsize=(11, 4.5))
ax.plot(wavenumbers, vip_all, color='steelblue', lw=1.0, zorder=2)
ax.fill_between(wavenumbers, vip_all, 1, where=(vip_all > 1),
                alpha=0.25, color='crimson',
                label='VIP > 1  (%d vars selected)' % n_vip_selected)
ax.axhline(1, color='crimson', ls='--', lw=1.0, label='VIP = 1 threshold')
ax.set_xlim(WAVENUMBER_MIN, WAVENUMBER_MAX)
ax.set_xlabel('Wavenumber (cm$^{-1}$)', fontsize=12, labelpad=6)
ax.set_ylabel('VIP score', fontsize=12, labelpad=6)
ax.set_title('Variable Importance in Projection\n' + TTAG,
             fontsize=11, fontweight='bold', pad=10)
ax.legend(fontsize=9, loc='upper right', framealpha=0.9)
ax.grid(True, alpha=0.25)
fig.tight_layout()
save_fig(fig, '07_PLSDA_VIP_Scores.png')

# =============================================================================
# 18. LOADINGS  LV1/LV2/LV3
# =============================================================================
fig, axes = plt.subplots(3, 1, figsize=(11, 9), sharex=True)
fig.suptitle('PLS-DA Loadings  LV1 / LV2 / LV3  |  VIP > 1 (%d vars)\n%s'
             % (n_vip_selected, TTAG), fontsize=10, fontweight='bold', y=1.01)
for i, (ax, col) in enumerate(zip(axes, ['steelblue', 'darkorange', 'seagreen'])):
    ax.stem(wn_sel, P_load[:, i], linefmt=col, markerfmt=' ', basefmt='k-')
    ax.axhline(0, color='k', lw=0.5)
    ax.set_ylabel('Loading LV%d' % (i+1), fontsize=10)
    ax.set_xlim(WAVENUMBER_MIN, WAVENUMBER_MAX)
    ax.grid(True, alpha=0.2)
    # Annotate top-5 peaks — offset upward to avoid overlap with stems
    for pk in np.argsort(np.abs(P_load[:, i]))[-5:]:
        yoff = 8 if P_load[pk, i] >= 0 else -12
        ax.annotate('%.0f' % wn_sel[pk],
                    xy=(wn_sel[pk], P_load[pk, i]),
                    xytext=(0, yoff), textcoords='offset points',
                    fontsize=7, ha='center', color='black',
                    arrowprops=dict(arrowstyle='-', lw=0.4, color='grey'))
axes[-1].set_xlabel('Wavenumber (cm$^{-1}$)', fontsize=12, labelpad=6)
fig.tight_layout(rect=[0, 0, 1, 0.99])
save_fig(fig, '08_PLSDA_Loadings.png')

# =============================================================================
# 19. PERMUTATION TEST
# =============================================================================
fig, ax = plt.subplots(figsize=(8, 5))
ax.hist(perm_scores, bins=35, color='lightsteelblue',
        edgecolor='white', label='Permuted accuracy')
ax.axvline(obs_score, color='crimson', lw=2.5,
           label='Observed: %.1f%%   p=%.4f' % (obs_score*100, p_val))
ax.axvline(0.25, color='grey', lw=1.2, ls=':', label='Chance level (25%)')
ax.set_xlabel('%d-fold CV accuracy' % CV_FOLDS, fontsize=12, labelpad=6)
ax.set_ylabel('Count', fontsize=12, labelpad=6)
ax.set_title('Permutation Test  (%d permutations)  |  PLS-DA\n%s'
             % (N_PERM, TTAG), fontsize=10, fontweight='bold', pad=10)
ax.legend(fontsize=9, loc='upper left', framealpha=0.9)
ax.grid(True, alpha=0.25)
fig.tight_layout()
save_fig(fig, '09_PLSDA_Permutation.png')

# =============================================================================
# 20. SVM CLUSTER PLOT  (Fig 10)
# =============================================================================
fig, ax = plt.subplots(figsize=(9, 7))
for cls in le.classes_:
    idx = np.where(classes == cls)[0]
    for i in idx:
        wrong = (le.transform([classes[i]])[0] != y_svm_5f[i])
        ax.scatter(Z_all[i, 0], Z_all[i, 1],
                   c=PALETTE[cls], marker=MARKER[cls], s=75,
                   edgecolors='red' if wrong else 'black',
                   linewidths=2.0 if wrong else 0.6,
                   alpha=0.88, zorder=3)

handles = ([plt.scatter([], [], c=PALETTE[c], marker=MARKER[c],
                        s=55, edgecolors='k', linewidths=0.6, label=c)
            for c in le.classes_] +
           [plt.scatter([], [], c='white', marker='o', s=55,
                        edgecolors='red', linewidths=2.0, label='Misclassified'),
            plt.scatter([], [], c='white', marker='o', s=55,
                        edgecolors='black', linewidths=0.6, label='Correct')])
n_wrong_svm = int((le.transform(classes) != y_svm_5f).sum())
ax.legend(handles=handles, fontsize=8,
          loc='upper center', bbox_to_anchor=(0.5, -0.10),
          ncol=3, framealpha=0.9, borderaxespad=0)
ax.set_xlabel('PC1  (%.1f%% var.)' % (pca_viz.explained_variance_ratio_[0]*100),
              fontsize=11, labelpad=6)
ax.set_ylabel('PC2  (%.1f%% var.)' % (pca_viz.explained_variance_ratio_[1]*100),
              fontsize=11, labelpad=6)
ax.set_title('SVM-RBF  Cluster Plot  |  %d-fold CV  |  acc=%.1f%%\n'
             '%d misclassified / %d  |  %s'
             % (CV_FOLDS, acc_svm_5f*100, n_wrong_svm, len(y), TTAG),
             fontsize=10, fontweight='bold', pad=10)
ax.grid(True, alpha=0.25)
fig.tight_layout(rect=[0, 0.08, 1, 1])
save_fig(fig, '10_SVM_Cluster_5fold.png')

# =============================================================================
# 21. SVM CONFUSION MATRICES  (Fig 11 Accuracy, Fig 12 Recall)
# =============================================================================
plot_accuracy_matrix(
    cm_svm_5f, le.classes_,
    'SVM-RBF  Accuracy Matrix  |  %d-fold CV  (n=%d)\n'
    'Cells = sample counts   |   Overall accuracy = %.1f%%'
    % (CV_FOLDS, len(y), acc_svm_5f * 100),
    '11_SVM_Accuracy_Matrix_5fold.png', 'Oranges')

plot_recall_matrix(
    cm_svm_5f, le.classes_,
    'SVM-RBF  Recall Matrix  |  %d-fold CV  (n=%d)\n'
    'Diagonal = Recall per class   |   Formula: TP / (TP + FN)'
    % (CV_FOLDS, len(y)),
    '12_SVM_Recall_Matrix_5fold.png', 'Oranges')

# =============================================================================
# 22. ACCURACY COMPARISON  (Fig 13)
# =============================================================================
fig, ax = plt.subplots(figsize=(7, 5.5))
methods  = ['PLS-DA', 'SVM-RBF']
accs     = [acc_pls_5f * 100, acc_svm_5f * 100]
colors   = ['steelblue', 'darkorange']
bars = ax.bar(methods, accs, color=colors, edgecolor='k', linewidth=0.9,
              width=0.45, zorder=3)
for bar, val in zip(bars, accs):
    ax.text(bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 1.2,
            '%.1f%%' % val,
            ha='center', va='bottom', fontsize=13, fontweight='bold')
ax.axhline(100, color='grey', ls=':', lw=1)
ax.set_ylabel('Accuracy  (%)', fontsize=12, labelpad=6)
ax.set_ylim(0, 120)
ax.set_title('Overall Accuracy  |  PLS-DA vs SVM-RBF\n'
             '%d-fold CV  |  ' % CV_FOLDS + TTAG,
             fontsize=11, fontweight='bold', pad=10)
ax.grid(True, alpha=0.2, axis='y', zorder=0)
ax.tick_params(axis='x', labelsize=12)
fig.tight_layout()
save_fig(fig, '13_Accuracy_Comparison.png')

# =============================================================================
# 23. PER-CLASS RECALL  (Fig 14)
# =============================================================================
fig, ax = plt.subplots(figsize=(10, 5.5))
rec_pls = [cm_pls_5f[i, i] / n_per_cls * 100 for i in range(len(le.classes_))]
rec_svm = [cm_svm_5f[i, i] / n_per_cls * 100 for i in range(len(le.classes_))]
x = np.arange(len(le.classes_))
w = 0.32
b1 = ax.bar(x - w/2, rec_pls, w, color='steelblue',  edgecolor='k',
            linewidth=0.8, label='PLS-DA', zorder=3)
b2 = ax.bar(x + w/2, rec_svm, w, color='darkorange', edgecolor='k',
            linewidth=0.8, label='SVM-RBF', zorder=3)
for bar in list(b1) + list(b2):
    if bar.get_height() > 0:
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 1.5,
                '%.0f%%' % bar.get_height(),
                ha='center', va='bottom', fontsize=10, fontweight='bold')
ax.axhline(100, color='grey', ls=':', lw=1)
ax.set_xticks(x)
ax.set_xticklabels([c.replace('SSY-', '') + '\n(n=%d)' % n_per_cls
                    for c in le.classes_],
                   fontsize=10)
ax.set_ylabel('Recall  (%)', fontsize=12, labelpad=6)
ax.set_ylim(0, 125)
ax.set_title('Per-Class Recall  |  PLS-DA vs SVM-RBF\n'
             '%d-fold CV  |  ' % CV_FOLDS + TTAG,
             fontsize=11, fontweight='bold', pad=10)
ax.legend(fontsize=11, loc='upper right', framealpha=0.9)
ax.grid(True, alpha=0.2, axis='y', zorder=0)
fig.tight_layout()
save_fig(fig, '14_Per_Class_Recall_5fold.png')

# =============================================================================
# 24. MEAN SPECTRA  (Fig 15)
# =============================================================================
fig, ax = plt.subplots(figsize=(10, 5))
for cls in le.classes_:
    idx = classes == cls
    m   = X_sm[idx].mean(axis=0)
    s   = X_sm[idx].std(axis=0)
    ax.plot(wavenumbers, m, color=PALETTE[cls], lw=1.6,
            label='%s  (n=%d)' % (cls, idx.sum()))
    ax.fill_between(wavenumbers, m - s, m + s, color=PALETTE[cls], alpha=0.12)
ax.set_xlim(WAVENUMBER_MIN, WAVENUMBER_MAX)
ax.set_xlabel('Wavenumber (cm$^{-1}$)', fontsize=12, labelpad=6)
ax.set_ylabel('Intensity (a.u.)', fontsize=12, labelpad=6)
ax.set_title('Mean Raman Spectra  +/- 1 SD  (all 80 spectra)\n%d-%d cm-1'
             % (WAVENUMBER_MIN, WAVENUMBER_MAX),
             fontsize=11, fontweight='bold', pad=10)
ax.legend(fontsize=9, loc='upper right', framealpha=0.9)
ax.grid(True, alpha=0.25)
fig.tight_layout()
save_fig(fig, '15_Mean_Spectra.png')

# =============================================================================
# 25. EXCEL RESULTS
# =============================================================================
out_xlsx = os.path.join(OUTPUT_DIR, 'Results_5fold.xlsx')
with pd.ExcelWriter(out_xlsx, engine='openpyxl') as writer:

    # Summary
    rows = [
        ('File',                          FILE_NAME),
        ('Total samples',                 len(y)),
        ('Spectral window (cm-1)',        '%d-%d' % (WAVENUMBER_MIN, WAVENUMBER_MAX)),
        ('SG smoothing window/poly',      '%d/%d' % (SG_WINDOW, SG_POLY)),
        ('VIP > 1 selected',              n_vip_selected),
        ('PLS-DA latent variables',       N_COMPONENTS),
        ('CV folds',                      CV_FOLDS),
        ('', ''),
        ('--- PLS-DA ---',                ''),
        ('Re-substitution accuracy (%)',  '%.2f' % (resub_acc   * 100)),
        ('%d-fold CV accuracy (%%)' % CV_FOLDS, '%.2f' % (acc_pls_5f * 100)),
        ('Permutation p-value',           '%.4f' % p_val),
        ('Permutation result',            'Significant' if p_val < 0.05 else 'NOT significant'),
        ('', ''),
        ('--- SVM-RBF ---',               ''),
        ('%d-fold CV accuracy (%%)' % CV_FOLDS, '%.2f' % (acc_svm_5f * 100)),
        ('SVM kernel / C / gamma',        'RBF / 10 / scale'),
    ]
    for i, r2 in enumerate(r2x_cum, 1):
        rows.append(('R2X cumulative LV%d (%%)' % i, '%.4f' % (r2 * 100)))
    pd.DataFrame(rows, columns=['Metric', 'Value']).to_excel(
        writer, sheet_name='Summary', index=False)

    # Per-class recall
    cls_rows = []
    for i, cls in enumerate(le.classes_):
        cls_rows.append({
            'Class':               cls,
            'N_per_class':         n_per_cls,
            'PLSDA_5fold_TP':      int(cm_pls_5f[i, i]),
            'PLSDA_5fold_Recall%': '%.1f' % (cm_pls_5f[i, i] / n_per_cls * 100),
            'SVM_5fold_TP':        int(cm_svm_5f[i, i]),
            'SVM_5fold_Recall%':   '%.1f' % (cm_svm_5f[i, i] / n_per_cls * 100),
        })
    pd.DataFrame(cls_rows).to_excel(
        writer, sheet_name='Per_Class_Recall', index=False)

    # Predictions
    pd.DataFrame({
        'Label':               col_names,
        'True_Class':          classes,
        'PLSDA_5fold_Pred':    le.inverse_transform(y_pls_5f),
        'PLSDA_5fold_Correct': (classes == le.inverse_transform(y_pls_5f)),
        'SVM_5fold_Pred':      le.inverse_transform(y_svm_5f),
        'SVM_5fold_Correct':   (classes == le.inverse_transform(y_svm_5f)),
        'LV1': T_all[:, 0], 'LV2': T_all[:, 1],
        'LV3': T_all[:, 2], 'PC1': Z_all[:, 0], 'PC2': Z_all[:, 1],
    }).to_excel(writer, sheet_name='All_Predictions', index=False)

    # Accuracy CMs (raw counts)
    for sheet, cm in [('CM_Accuracy_PLSDA', cm_pls_5f),
                      ('CM_Accuracy_SVM',   cm_svm_5f)]:
        pd.DataFrame(cm,
                     index=pd.Index(le.classes_, name='True\\Predicted'),
                     columns=le.classes_).to_excel(writer, sheet_name=sheet)

    # Recall CMs (row-normalised)
    for sheet, cm in [('CM_Recall_PLSDA', cm_pls_5f),
                      ('CM_Recall_SVM',   cm_svm_5f)]:
        cm_n = cm.astype(float) / cm.sum(axis=1, keepdims=True)
        pd.DataFrame(cm_n,
                     index=pd.Index(le.classes_, name='True\\Predicted'),
                     columns=le.classes_).to_excel(writer, sheet_name=sheet)

    # VIP
    pd.DataFrame({
        'Wavenumber_cm1': wavenumbers,
        'VIP_Score':      vip_all,
        'VIP_gt_1':       vip_all > 1,
        'Selected_VIPgt1': sel_mask,
    }).to_excel(writer, sheet_name='VIP_Scores', index=False)

    # Loadings
    ld = pd.DataFrame(P_load,
                      columns=['LV%d' % (i+1) for i in range(N_COMPONENTS)])
    ld.insert(0, 'Wavenumber_cm1', wn_sel)
    ld.to_excel(writer, sheet_name='Loadings', index=False)

    # Permutation
    pd.DataFrame({
        'Permutation_Index': np.arange(1, N_PERM + 1),
        'Permuted_Accuracy': perm_scores,
    }).to_excel(writer, sheet_name='Permutation_Test', index=False)

print('\n[SAVED] %s' % out_xlsx)

# =============================================================================
# 26. CONSOLE SUMMARY
# =============================================================================
print('\n' + '=' * 65)
print('  RESULTS  |  %d-fold CV  |  VIP > 1 (%d vars)' % (CV_FOLDS, n_vip_selected))
print('=' * 65)
print('  %-10s   %d-fold CV' % ('MODEL', CV_FOLDS))
print('  ' + '-' * 28)
print('  %-10s   %.1f%%' % ('PLS-DA',  acc_pls_5f * 100))
print('  %-10s   %.1f%%' % ('SVM-RBF', acc_svm_5f * 100))
print()
print('  Permutation p = %.4f  [%s]' % (p_val, sig_str))
print()
print('  Per-class Recall  (TP / (TP+FN)):')
print('  %-25s   %8s   %8s' % ('Class', 'PLS-DA', 'SVM-RBF'))
print('  ' + '-' * 46)
for i, cls in enumerate(le.classes_):
    print('  %-25s   %7.0f%%   %7.0f%%' % (
        cls,
        cm_pls_5f[i, i] / n_per_cls * 100,
        cm_svm_5f[i, i] / n_per_cls * 100))
print()
print('  15 figures + Excel -> %s' % OUTPUT_DIR)
print('=' * 65)
