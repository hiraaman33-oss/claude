"""
=============================================================================
PLS-DA + SVM-RBF  |  BINARY CLASSIFICATION  |  TRAINING (60) / TEST (20)
SSY Raman Spectroscopy
SPECTRAL WINDOW : 450 - 1800 cm-1

BINARY CLASSES
  Class 1  PDMS    : SSY-PDMS  +  SSY-Gr-PDMS     (40 spectra total)
  Class 2  SiO2/Si : SSY-SiO2/Si + SSY-Gr-SiO2/Si (40 spectra total)

SPLIT  (Gr-only test set — sessions ignored)
  Training : 60 spectra  (10 Gr-PDMS + 20 PDMS + 10 Gr-SiO2/Si + 20 SiO2/Si)
  Test     : 20 spectra  — Gr ONLY  (10 Gr-PDMS + 10 Gr-SiO2/Si)
             randomly selected from 20 Gr per class; remaining Gr go to train

VALIDATION
  PLS-DA  :  5-fold CV on training  +  external test
  SVM-RBF :  5-fold CV on training  +  external test
  Per-class Accuracy  = (TP + TN) / N_total
  Per-class Recall    = TP / (TP + FN)

OUTPUTS -> C:/Users/Hira Aman/Desktop/PROF_DOMENICO'S/PLSDA_SVM_Binary/
  01_PLSDA_Scores_LV1_LV2.png
  02_PLSDA_Scores_LV1_LV3.png
  03_PLSDA_Scores_3D.png
  04_PLSDA_Confusion_5fold.png
  05_PLSDA_Confusion_Test.png
  06_PLSDA_VIP_Scores.png
  07_PLSDA_Loadings.png
  08_PLSDA_Permutation.png
  09_PLSDA_Cluster_5fold.png
  10_SVM_Cluster_5fold.png
  11_SVM_Cluster_Test.png
  12_SVM_Confusion_5fold.png
  13_SVM_Confusion_Test.png
  14_Accuracy_Comparison.png
  15_Recall_Comparison.png
  16_Mean_Spectra.png
  Results_Binary.xlsx

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
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
from mpl_toolkits.mplot3d import Axes3D                    # noqa

from sklearn.cross_decomposition import PLSRegression
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import (
    StratifiedKFold, cross_val_predict,
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
FILE_NAME      = "Book1.xls"
SHEET_NAME     = "Sheet1"

WAVENUMBER_MIN = 450
WAVENUMBER_MAX = 1800

SG_WINDOW      = 7
SG_POLY        = 2
VIP_TOP_N      = 1500
N_COMPONENTS   = 3          # binary PLS-DA: 3 LVs sufficient
N_PERM         = 999
RANDOM_STATE   = 42

OUTPUT_DIR = os.path.join(DATA_PATH, "PLSDA_SVM_Binary")

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
# 1.  ROBUST FILE LOADER
# =============================================================================
def load_xls(file_path, sheet):
    try:
        df = pd.read_excel(file_path, sheet_name=sheet,
                           header=0, index_col=0, engine='xlrd')
        print("[INFO] Read engine : xlrd")
        return df
    except Exception as e1:
        print("[INFO] xlrd unavailable (%s), trying Excel COM ..." % e1.__class__.__name__)
    try:
        import win32com.client
        tmp = file_path.replace('.xls', '_tmp.xlsx')
        xl  = win32com.client.Dispatch("Excel.Application")
        xl.Visible = False
        wb  = xl.Workbooks.Open(file_path)
        wb.SaveAs(tmp, FileFormat=51)
        wb.Close(); xl.Quit()
        df = pd.read_excel(tmp, sheet_name=sheet,
                           header=0, index_col=0, engine='openpyxl')
        os.remove(tmp)
        print("[INFO] Read engine : Excel COM")
        return df
    except Exception as e2:
        print("[INFO] Excel COM unavailable (%s), trying LibreOffice ..." % e2.__class__.__name__)
    try:
        import shutil
        lo = shutil.which("soffice") or shutil.which("libreoffice")
        if lo:
            tmp_dir = tempfile.mkdtemp()
            subprocess.run([lo, "--headless", "--convert-to", "xlsx",
                            "--outdir", tmp_dir, file_path],
                           check=True, capture_output=True, timeout=60)
            xlsx = os.path.join(
                tmp_dir,
                os.path.basename(file_path).replace('.xls', '.xlsx'))
            df = pd.read_excel(xlsx, sheet_name=sheet,
                               header=0, index_col=0, engine='openpyxl')
            print("[INFO] Read engine : LibreOffice")
            return df
    except Exception as e3:
        print("[INFO] LibreOffice unavailable (%s)" % e3.__class__.__name__)
    raise RuntimeError(
        "\n[ERROR] Cannot read the .xls file.\n"
        "Run:  pip install xlrd   then re-run.")

# =============================================================================
# 2.  BINARY CLASS MAPPING  +  Gr / no-Gr FLAG
#     PDMS    = SSY-PDMS  +  SSY-Gr-PDMS      (anything with 'pdms')
#     SiO2/Si = SSY-SiO2/Si + SSY-Gr-SiO2/Si  (anything with 'sio2')
# =============================================================================
def assign_class(col_name):
    s = col_name.lower()
    s = re.sub(r'^s\d+[-_]', '', s)
    s = s.replace('grsl', 'gr')
    s = re.sub(r'^(lugl|marzo\d*)[-_](ssy[-_]?)?', '', s)
    s = s.replace('_', '-').replace(' ', '-')
    if   re.search(r'sio2', s): return 'SiO2/Si'
    elif re.search(r'pdms', s): return 'PDMS'
    else:                        return None

def has_gr(col_name):
    s = col_name.lower()
    return bool(re.search(r'[-_]gr[-_]|[-_]gr$|grsl', s))

# =============================================================================
# 3.  LOAD DATA
# =============================================================================
print("\n" + "=" * 65)
print("  PLS-DA + SVM-RBF  |  BINARY  |  Raman  |  450-1800 cm-1")
print("  Class 1: PDMS  (PDMS + Gr-PDMS)")
print("  Class 2: SiO2/Si  (SiO2/Si + Gr-SiO2/Si)")
print("=" * 65)

file_path = os.path.join(DATA_PATH, FILE_NAME)
print("\n[INFO] File : %s" % file_path)

raw = load_xls(file_path, SHEET_NAME)
print("[INFO] Loaded : %d wavenumber rows x %d sample columns" % raw.shape)

wavenumbers_all = raw.index.astype(float).values
col_names_all   = raw.columns.tolist()
X_all           = raw.values.T.astype(float)

crop_mask   = ((wavenumbers_all >= WAVENUMBER_MIN) &
               (wavenumbers_all <= WAVENUMBER_MAX))
wavenumbers = wavenumbers_all[crop_mask]
X_crop      = X_all[:, crop_mask]

assert X_crop.shape[1] > 0,          "ERROR: No spectral points in range!"
assert np.isnan(X_crop).sum() == 0,  "ERROR: NaN values in data!"
print("[INFO] Cropped : %.1f - %.1f cm-1  (%d points)  OK" % (
    wavenumbers[0], wavenumbers[-1], X_crop.shape[1]))

classes_raw = np.array([assign_class(c) for c in col_names_all])
bad_mask    = np.array([c is None for c in classes_raw])
if bad_mask.any():
    print("\n[WARNING] %d unrecognised column(s) — skipped" % bad_mask.sum())

valid_mask = ~bad_mask
X_crop     = X_crop[valid_mask]
classes    = classes_raw[valid_mask]
col_names  = [col_names_all[i] for i in range(len(col_names_all)) if valid_mask[i]]

le = LabelEncoder()          # PDMS=0, SiO2/Si=1  (alphabetical)
y  = le.fit_transform(classes)

print("\n[INFO] Total valid samples : %d" % len(classes))
for cls, cnt in zip(*np.unique(classes, return_counts=True)):
    print("         %-12s : %d" % (cls, cnt))
print("[INFO] Class encoding : %s" % str(
    dict(zip(le.classes_, le.transform(le.classes_)))))

assert len(le.classes_) == 2, "Expected exactly 2 classes, got %d" % len(le.classes_)

# =============================================================================
# 4.  SPLIT  — Gr-only test set
#
#   Data per class (40 each):  20 Gr  +  20 no-Gr
#
#   Step 1 — Exclude 5 no-Gr per class (10 total) — not used anywhere
#   Step 2 — Test : 5 Gr per class (10 total, Gr spectra only)
#   Step 3 — Train: remaining 30 per class (60 total)
#             = 15 Gr + 15 no-Gr per class
# =============================================================================
print("\n[SPLIT] Gr-only test split ...")

gr_mask = np.array([has_gr(c) for c in col_names])

pdms_gr_idx    = np.where((classes == 'PDMS')    &  gr_mask)[0]   # 20
pdms_nogr_idx  = np.where((classes == 'PDMS')    & ~gr_mask)[0]   # 20
sio2_gr_idx    = np.where((classes == 'SiO2/Si') &  gr_mask)[0]   # 20
sio2_nogr_idx  = np.where((classes == 'SiO2/Si') & ~gr_mask)[0]   # 20

rng = np.random.default_rng(RANDOM_STATE)

# Test: 5 Gr per class  →  10 Gr-only test spectra
pdms_gr_test  = rng.choice(pdms_gr_idx,   size=5, replace=False)
sio2_gr_test  = rng.choice(sio2_gr_idx,   size=5, replace=False)
test_idx      = np.sort(np.concatenate([pdms_gr_test, sio2_gr_test]))

# Exclude: 5 no-Gr per class  →  10 spectra dropped entirely
pdms_nogr_excl = rng.choice(pdms_nogr_idx, size=5, replace=False)
sio2_nogr_excl = rng.choice(sio2_nogr_idx, size=5, replace=False)
excl_idx       = np.concatenate([pdms_nogr_excl, sio2_nogr_excl])

# Train: everything else  →  60 spectra
all_idx   = np.arange(len(y))
train_idx = np.sort(np.setdiff1d(all_idx, np.concatenate([test_idx, excl_idx])))

assert len(train_idx) == 60, "Train size = %d (expected 60)" % len(train_idx)
assert len(test_idx)  == 10, "Test size  = %d (expected 10)"  % len(test_idx)
assert len(excl_idx)  == 10, "Excl size  = %d (expected 10)"  % len(excl_idx)

X_train     = X_crop[train_idx]
X_test      = X_crop[test_idx]
y_train     = y[train_idx]
y_test      = y[test_idx]
cls_train   = classes[train_idx]
cls_test    = classes[test_idx]
names_train = [col_names[i] for i in train_idx]
names_test  = [col_names[i] for i in test_idx]

print("[SPLIT] Training : %d spectra  (15 Gr + 15 no-Gr per class)" % len(y_train))
for cls, cnt in zip(*np.unique(cls_train, return_counts=True)):
    print("           %-12s: %d" % (cls, cnt))
print("[SPLIT] Test     : %d spectra  (Gr only: 5 per class)" % len(y_test))
for cls, cnt in zip(*np.unique(cls_test, return_counts=True)):
    print("           %-12s: %d" % (cls, cnt))
print("[SPLIT] Excluded : %d no-Gr spectra (5 per class, not used)" % len(excl_idx))

# =============================================================================
# 5.  PRE-PROCESSING  (Savitzky-Golay smooth)
# =============================================================================
X_train_sm = savgol_filter(X_train, window_length=SG_WINDOW, polyorder=SG_POLY, axis=1)
X_test_sm  = savgol_filter(X_test,  window_length=SG_WINDOW, polyorder=SG_POLY, axis=1)
print("\n[INFO] SG smoothing applied (window=%d, poly=%d)" % (SG_WINDOW, SG_POLY))

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
# 7.  VIP SELECTION  (3-LV PLS-DA on TRAINING only — no leakage)
# =============================================================================
print("\n[VIP] 3-LV PLS-DA on training -> top-%d VIP ..." % VIP_TOP_N)

sc_init  = StandardScaler().fit(X_train_sm)
Xs_init  = sc_init.transform(X_train_sm)
Yd_init  = pd.get_dummies(pd.Series(y_train)).values.astype(float)
pls_init = PLSRegression(n_components=3, scale=False, max_iter=10000)
pls_init.fit(Xs_init, Yd_init)
vip_all  = compute_vip(pls_init)

n_vip_gt1   = int((vip_all > 1).sum())
top_n       = min(VIP_TOP_N, len(vip_all))
top_idx     = np.argsort(vip_all)[-top_n:]
wn_sel      = wavenumbers[top_idx]
X_train_sel = X_train_sm[:, top_idx]
X_test_sel  = X_test_sm[:,  top_idx]

print("[VIP] VIP > 1 : %d / %d variables" % (n_vip_gt1, len(vip_all)))
print("[VIP] Top-%d selected  (%.1f - %.1f cm-1)" % (
    top_n, wn_sel.min(), wn_sel.max()))

# =============================================================================
# 8.  HELPER CLASSIFIER
# =============================================================================
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
    def transform(self, X):
        return self._pls.transform(X)[0]

cv5 = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

sc_final = StandardScaler().fit(X_train_sel)
Xtr_fin  = sc_final.transform(X_train_sel)
Xte_fin  = sc_final.transform(X_test_sel)

# =============================================================================
# 9.  PLS-DA BINARY FINAL MODEL
# =============================================================================
print("\n[PLS-DA] Fitting final model (%d LVs, top-%d vars) ..." % (N_COMPONENTS, top_n))

Yd_fin    = pd.get_dummies(pd.Series(y_train)).values.astype(float)
pls_final = PLSRegression(n_components=N_COMPONENTS, scale=False, max_iter=10000)
pls_final.fit(Xtr_fin, Yd_fin)

T_train = pls_final.x_scores_           # (60, N_COMPONENTS)
T_test  = pls_final.transform(Xte_fin)  # (20, N_COMPONENTS)
P_load  = pls_final.x_loadings_         # (top_n, N_COMPONENTS)

# R2X cumulative
SS_tot  = np.sum(Xtr_fin ** 2)
r2x_cum = []
for lv in range(1, N_COMPONENTS + 1):
    Xrec = pls_final.x_scores_[:, :lv] @ pls_final.x_loadings_[:, :lv].T
    r2x_cum.append(1 - np.sum((Xtr_fin - Xrec) ** 2) / SS_tot)
print("[PLS-DA] R2X cumulative : " + "  ".join(
    ["LV%d=%.1f%%" % (i+1, r*100) for i, r in enumerate(r2x_cum)]))

train_acc_resub = accuracy_score(y_train, np.argmax(pls_final.predict(Xtr_fin), axis=1))
print("[PLS-DA] Training re-sub accuracy : %.1f%%" % (train_acc_resub * 100))

# 5-fold CV
clf_pipe_pls = Pipeline([('sc', StandardScaler()), ('plsda', PLSDAClf(N_COMPONENTS))])
print("[PLS-DA] 5-fold CV on training set ...")
y_pls_cv   = cross_val_predict(clf_pipe_pls, X_train_sel, y_train, cv=cv5)
acc_pls_cv = accuracy_score(y_train, y_pls_cv)
cm_pls_cv  = confusion_matrix(y_train, y_pls_cv)
print("[PLS-DA] 5-fold CV accuracy : %.1f%%" % (acc_pls_cv * 100))

# External test
y_pls_test   = np.argmax(pls_final.predict(Xte_fin), axis=1)
acc_pls_test = accuracy_score(y_test, y_pls_test)
cm_pls_test  = confusion_matrix(y_test, y_pls_test)
print("[PLS-DA] External test accuracy : %.1f%%" % (acc_pls_test * 100))

# Permutation test
print("[PLS-DA] Permutation test (%d permutations) ..." % N_PERM)
obs_score, perm_scores, p_val = permutation_test_score(
    clf_pipe_pls, X_train_sel, y_train,
    scoring='accuracy', cv=cv5,
    n_permutations=N_PERM,
    random_state=RANDOM_STATE, n_jobs=-1)
print("[PLS-DA] Observed : %.2f%%   p = %.4f  [%s]" % (
    obs_score * 100, p_val,
    'SIGNIFICANT (p<0.05)' if p_val < 0.05 else 'NOT significant'))

# =============================================================================
# 10. SVM-RBF BINARY
# =============================================================================
svm_pipe = Pipeline([
    ('sc',  StandardScaler()),
    ('svm', SVC(kernel='rbf', C=10, gamma='scale',
                probability=True, random_state=RANDOM_STATE))
])

print("\n[SVM-RBF] 5-fold CV on training set ...")
y_svm_cv   = cross_val_predict(svm_pipe, X_train_sel, y_train, cv=cv5)
acc_svm_cv = accuracy_score(y_train, y_svm_cv)
cm_svm_cv  = confusion_matrix(y_train, y_svm_cv)
print("[SVM-RBF] 5-fold CV accuracy : %.1f%%" % (acc_svm_cv * 100))

svm_pipe.fit(X_train_sel, y_train)
y_svm_test   = svm_pipe.predict(X_test_sel)
acc_svm_test = accuracy_score(y_test, y_svm_test)
cm_svm_test  = confusion_matrix(y_test, y_svm_test)
print("[SVM-RBF] External test accuracy : %.1f%%" % (acc_svm_test * 100))

# =============================================================================
# 11. PCA FOR CLUSTER VISUALISATION  (fit on TRAIN only)
# =============================================================================
pca_viz = PCA(n_components=2, random_state=RANDOM_STATE)
sc_viz  = StandardScaler().fit(X_train_sel)
pca_viz.fit(sc_viz.transform(X_train_sel))
Z_train = pca_viz.transform(sc_viz.transform(X_train_sel))
Z_test  = pca_viz.transform(sc_viz.transform(X_test_sel))
print("\n[PCA] PC1=%.1f%%  PC2=%.1f%% explained variance" % (
    pca_viz.explained_variance_ratio_[0]*100,
    pca_viz.explained_variance_ratio_[1]*100))

# =============================================================================
# 12. COLOUR / MARKER SCHEME
# =============================================================================
PALETTE = {
    'PDMS'    : '#1f77b4',   # blue
    'SiO2/Si' : '#d62728',   # red
}
MARKER_TR = {
    'PDMS'    : 'o',
    'SiO2/Si' : 's',
}
CLASS_LABELS = le.classes_    # ['PDMS', 'SiO2/Si']

TITLE_SUF = "VIP-top-%d | %d LVs | %d-%d cm⁻¹" % (
    top_n, N_COMPONENTS, WAVENUMBER_MIN, WAVENUMBER_MAX)

def lv_var(i):
    return (r2x_cum[i] - (r2x_cum[i-1] if i > 0 else 0)) * 100

def save_fig(fig, fname):
    fig.savefig(os.path.join(OUTPUT_DIR, fname), dpi=150, bbox_inches='tight')
    plt.close(fig)
    print('[SAVED] %s' % fname)

# =============================================================================
# HELPER: Per-class Accuracy and Recall
#   Accuracy_i = (TP_i + TN_i) / N_total
#   Recall_i   = TP_i  / (TP_i + FN_i)
# =============================================================================
def per_class_metrics(cm):
    N = cm.sum()
    acc_list, rec_list = [], []
    for i in range(len(cm)):
        TP = cm[i, i]
        FP = cm[:, i].sum() - TP
        FN = cm[i, :].sum() - TP
        TN = N - TP - FP - FN
        acc_list.append((TP + TN) / N * 100)
        denom = TP + FN
        rec_list.append(TP / denom * 100 if denom > 0 else 0.0)
    return np.array(acc_list), np.array(rec_list)

# =============================================================================
# HELPER: Confusion matrix figure with per-class stats table (GridSpec)
# =============================================================================
def plot_confusion_with_stats(cm, labels, title, fname, cmap, overall_acc):
    acc_list, rec_list = per_class_metrics(cm)
    short = [l for l in labels]    # already short for binary

    fig = plt.figure(figsize=(7, 8))
    gs  = GridSpec(2, 1, figure=fig, height_ratios=[3, 0.9], hspace=0.45)
    ax_cm  = fig.add_subplot(gs[0])
    ax_tbl = fig.add_subplot(gs[1])

    ConfusionMatrixDisplay(cm, display_labels=labels).plot(
        ax=ax_cm, colorbar=False, cmap=cmap, xticks_rotation=15)
    ax_cm.set_title('%s\nOverall Accuracy = %.1f%%' % (title, overall_acc * 100),
                    fontsize=11, fontweight='bold')

    ax_tbl.axis('off')
    col_labels = ['Class', 'Accuracy (%)', 'Recall (%)']
    table_data = [[short[i], '%.1f' % acc_list[i], '%.1f' % rec_list[i]]
                  for i in range(len(labels))]
    tbl = ax_tbl.table(cellText=table_data, colLabels=col_labels,
                       loc='center', cellLoc='center')
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(10)
    tbl.scale(1, 2.0)
    for j in range(3):
        tbl[0, j].set_facecolor('#404040')
        tbl[0, j].set_text_props(color='white', fontweight='bold')
    for i, cls in enumerate(labels):
        tbl[i+1, 0].set_facecolor(PALETTE.get(cls, '#cccccc'))
        tbl[i+1, 0].set_text_props(color='white', fontweight='bold')
        tbl[i+1, 1].set_facecolor('#f0f4ff')
        tbl[i+1, 2].set_facecolor('#fff4f0')
    save_fig(fig, fname)

# =============================================================================
# HELPER: Stats annotation box for cluster plots
# =============================================================================
def add_stats_box(ax, cm_main, label_main, cm_sec=None, label_sec=None):
    acc_m, rec_m = per_class_metrics(cm_main)
    lines = [label_main,
             '%-10s  %5s  %6s' % ('Class', 'Acc', 'Recall'),
             '-' * 26]
    for i, cls in enumerate(CLASS_LABELS):
        lines.append('%-10s  %4.1f%%  %5.1f%%' % (cls, acc_m[i], rec_m[i]))
    if cm_sec is not None:
        acc_s, rec_s = per_class_metrics(cm_sec)
        lines += ['', label_sec,
                  '%-10s  %5s  %6s' % ('Class', 'Acc', 'Recall'),
                  '-' * 26]
        for i, cls in enumerate(CLASS_LABELS):
            lines.append('%-10s  %4.1f%%  %5.1f%%' % (cls, acc_s[i], rec_s[i]))
    ax.text(0.02, 0.02, '\n'.join(lines), transform=ax.transAxes,
            fontsize=7.5, verticalalignment='bottom', fontfamily='monospace',
            bbox=dict(boxstyle='round,pad=0.4', facecolor='white',
                      alpha=0.90, edgecolor='#888888'))

# =============================================================================
# 13. PLS-DA SCORE PLOTS  (LV space, train + test)
# =============================================================================
def scatter_lv(ax, lv_x, lv_y, xlabel, ylabel, title):
    for cls in CLASS_LABELS:
        idx_tr = cls_train == cls
        idx_te = cls_test  == cls
        ax.scatter(T_train[idx_tr, lv_x], T_train[idx_tr, lv_y],
                   c=PALETTE[cls], marker=MARKER_TR[cls], s=75,
                   edgecolors='k', linewidths=0.6, alpha=0.80,
                   label='%s (train)' % cls, zorder=3)
        ax.scatter(T_test[idx_te, lv_x], T_test[idx_te, lv_y],
                   c=PALETTE[cls], marker='*', s=300,
                   edgecolors='k', linewidths=1.0, alpha=1.00,
                   label='%s (test)' % cls, zorder=5)
    ax.axhline(0, color='grey', lw=0.7, ls='--')
    ax.axvline(0, color='grey', lw=0.7, ls='--')
    ax.set_xlabel(xlabel, fontsize=11); ax.set_ylabel(ylabel, fontsize=11)
    ax.set_title(title, fontsize=10, fontweight='bold')
    ax.legend(fontsize=8.5, framealpha=0.85)
    ax.grid(True, alpha=0.25)

# Fig 01 — LV1 vs LV2
fig, ax = plt.subplots(figsize=(8, 6))
scatter_lv(ax, 0, 1,
    'LV1 (%.1f%% var.)' % lv_var(0),
    'LV2 (%.1f%% var.)' % lv_var(1),
    'PLS-DA Scores  LV1 vs LV2  |  ' + TITLE_SUF +
    '\nFilled = train  |  Stars = test')
fig.tight_layout()
save_fig(fig, '01_PLSDA_Scores_LV1_LV2.png')

# Fig 02 — LV1 vs LV3
fig, ax = plt.subplots(figsize=(8, 6))
scatter_lv(ax, 0, 2,
    'LV1 (%.1f%% var.)' % lv_var(0),
    'LV3 (%.1f%% var.)' % lv_var(2),
    'PLS-DA Scores  LV1 vs LV3  |  ' + TITLE_SUF +
    '\nFilled = train  |  Stars = test')
fig.tight_layout()
save_fig(fig, '02_PLSDA_Scores_LV1_LV3.png')

# Fig 03 — 3D scores
fig = plt.figure(figsize=(9, 7))
ax  = fig.add_subplot(111, projection='3d')
for cls in CLASS_LABELS:
    itr = cls_train == cls; ite = cls_test == cls
    ax.scatter(T_train[itr,0], T_train[itr,1], T_train[itr,2],
               c=PALETTE[cls], marker=MARKER_TR[cls], s=55,
               edgecolors='k', linewidths=0.4, alpha=0.80, label='%s (train)' % cls)
    ax.scatter(T_test[ite,0], T_test[ite,1], T_test[ite,2],
               c=PALETTE[cls], marker='*', s=220,
               edgecolors='k', linewidths=0.8, alpha=1.00, label='%s (test)' % cls)
ax.set_xlabel('LV1 (%.1f%%)' % lv_var(0), fontsize=9)
ax.set_ylabel('LV2 (%.1f%%)' % lv_var(1), fontsize=9)
ax.set_zlabel('LV3 (%.1f%%)' % lv_var(2), fontsize=9)
ax.set_title('PLS-DA 3D Scores  |  Binary  |  ' + TITLE_SUF,
             fontsize=10, fontweight='bold')
ax.legend(fontsize=8)
fig.tight_layout()
save_fig(fig, '03_PLSDA_Scores_3D.png')

# =============================================================================
# 14. PLS-DA CONFUSION MATRICES WITH PER-CLASS STATS TABLE
# =============================================================================
plot_confusion_with_stats(
    cm_pls_cv, CLASS_LABELS,
    'PLS-DA  5-fold CV  (train n=60)',
    '04_PLSDA_Confusion_5fold.png', 'Blues', acc_pls_cv)

plot_confusion_with_stats(
    cm_pls_test, CLASS_LABELS,
    'PLS-DA  External Test  (Gr only, n=10)',
    '05_PLSDA_Confusion_Test.png', 'Oranges', acc_pls_test)

# =============================================================================
# 15. VIP SCORES
# =============================================================================
sel_mask = np.zeros(len(wavenumbers), dtype=bool)
sel_mask[top_idx] = True

fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(wavenumbers, vip_all, color='steelblue', lw=1.0, zorder=2)
ax.fill_between(wavenumbers, vip_all, 1, where=(vip_all > 1),
                alpha=0.25, color='crimson',
                label='VIP > 1  (%d bands)' % n_vip_gt1, zorder=1)
ax.axhline(1, color='crimson', ls='--', lw=1.0, label='VIP = 1 threshold')
ax.fill_between(wavenumbers, 0, 0.07*vip_all.max(), where=sel_mask,
                alpha=0.18, color='green',
                label='Top-%d selected' % top_n)
ax.set_xlim(WAVENUMBER_MIN, WAVENUMBER_MAX)
ax.set_xlabel('Wavenumber (cm⁻¹)', fontsize=12)
ax.set_ylabel('VIP score', fontsize=12)
ax.set_title('Variable Importance in Projection  |  Binary  |  ' + TITLE_SUF,
             fontsize=11, fontweight='bold')
ax.legend(fontsize=9); ax.grid(True, alpha=0.25)
fig.tight_layout()
save_fig(fig, '06_PLSDA_VIP_Scores.png')

# =============================================================================
# 16. LOADINGS  LV1 / LV2 / LV3
# =============================================================================
fig, axes = plt.subplots(N_COMPONENTS, 1, figsize=(11, 3*N_COMPONENTS), sharex=True)
colors_load = ['steelblue', 'darkorange', 'seagreen']
for i, (ax, col) in enumerate(zip(axes, colors_load[:N_COMPONENTS])):
    ax.stem(wn_sel, P_load[:, i], linefmt=col, markerfmt=' ', basefmt='k-')
    ax.axhline(0, color='k', lw=0.5)
    ax.set_ylabel('Loading LV%d' % (i+1), fontsize=10)
    ax.set_xlim(WAVENUMBER_MIN, WAVENUMBER_MAX)
    ax.grid(True, alpha=0.2)
    for pk in np.argsort(np.abs(P_load[:, i]))[-5:]:
        ax.annotate('%.0f' % wn_sel[pk],
                    xy=(wn_sel[pk], P_load[pk, i]), fontsize=7,
                    ha='center', xytext=(0, 5), textcoords='offset points')
axes[-1].set_xlabel('Wavenumber (cm⁻¹)', fontsize=12)
axes[0].set_title('PLS-DA Loadings  |  Binary  |  ' + TITLE_SUF,
                  fontsize=10, fontweight='bold')
fig.tight_layout()
save_fig(fig, '07_PLSDA_Loadings.png')

# =============================================================================
# 17. PERMUTATION TEST
# =============================================================================
fig, ax = plt.subplots(figsize=(7.5, 4.5))
ax.hist(perm_scores, bins=35, color='lightsteelblue', edgecolor='white',
        label='Permuted accuracy')
ax.axvline(obs_score, color='crimson', lw=2.5,
           label='Observed: %.1f%%   p=%.4f' % (obs_score*100, p_val))
ax.axvline(0.50, color='grey', lw=1.2, ls=':', label='Chance level (50%)')
ax.set_xlabel('5-fold CV accuracy', fontsize=12)
ax.set_ylabel('Count', fontsize=12)
ax.set_title('Permutation Test (%d perm.)  |  PLS-DA Binary  |  ' % N_PERM + TITLE_SUF,
             fontsize=10, fontweight='bold')
ax.legend(fontsize=9); ax.grid(True, alpha=0.25)
fig.tight_layout()
save_fig(fig, '08_PLSDA_Permutation.png')

# =============================================================================
# 18. PLS-DA CLUSTER PLOT  (LV1 vs LV2)
#     Train = 5-fold CV coloured   Test = external coloured   Red edge = wrong
# =============================================================================
fig, ax = plt.subplots(figsize=(9, 7))

for cls in CLASS_LABELS:
    for i in np.where(cls_train == cls)[0]:
        wrong = (le.transform([cls_train[i]])[0] != y_pls_cv[i])
        ax.scatter(T_train[i, 0], T_train[i, 1],
                   c=PALETTE[cls], marker=MARKER_TR[cls], s=75,
                   edgecolors='red' if wrong else 'black',
                   linewidths=2.0 if wrong else 0.6, alpha=0.82, zorder=3)
    for i in np.where(cls_test == cls)[0]:
        wrong = (le.transform([cls_test[i]])[0] != y_pls_test[i])
        ax.scatter(T_test[i, 0], T_test[i, 1],
                   c=PALETTE[cls], marker='*', s=300,
                   edgecolors='red' if wrong else 'black',
                   linewidths=2.0 if wrong else 1.0, alpha=1.00, zorder=5)

handles  = [plt.scatter([], [], c=PALETTE[c], marker=MARKER_TR[c],
                        s=60, edgecolors='k', label=c) for c in CLASS_LABELS]
handles += [plt.scatter([], [], c='grey', marker='*', s=200,
                        edgecolors='k', label='Test samples'),
            Line2D([0],[0], marker='o', color='w', markerfacecolor='white',
                   markeredgecolor='red', markeredgewidth=2.0, markersize=9,
                   label='Misclassified'),
            Line2D([0],[0], marker='o', color='w', markerfacecolor='white',
                   markeredgecolor='black', markeredgewidth=0.6, markersize=9,
                   label='Correct')]
ax.legend(handles=handles, fontsize=9, framealpha=0.85, loc='upper right')
ax.axhline(0, color='grey', lw=0.7, ls='--')
ax.axvline(0, color='grey', lw=0.7, ls='--')
ax.set_xlabel('LV1 (%.1f%% var.)' % lv_var(0), fontsize=11)
ax.set_ylabel('LV2 (%.1f%% var.)' % lv_var(1), fontsize=11)
nw_tr = int((le.transform(cls_train) != y_pls_cv).sum())
nw_te = int((le.transform(cls_test)  != y_pls_test).sum())
ax.set_title(
    'PLS-DA Binary Cluster  |  5-fold CV (train) + External Test (Gr only)\n'
    'Train: %d wrong/60  CV acc=%.1f%%     Test: %d wrong/10  acc=%.1f%%' % (
        nw_tr, acc_pls_cv*100, nw_te, acc_pls_test*100),
    fontsize=10, fontweight='bold')
ax.grid(True, alpha=0.25)
add_stats_box(ax, cm_pls_cv, '5-fold CV (train)', cm_pls_test, 'External Test')
fig.suptitle(TITLE_SUF, fontsize=10)
fig.tight_layout()
save_fig(fig, '09_PLSDA_Cluster_5fold.png')

# =============================================================================
# 19. SVM-RBF CLUSTER PLOTS  (PCA space)
# =============================================================================
def scatter_svm(ax, pred_tr, pred_te, title, cm_cv_arg, cm_te_arg):
    for cls in CLASS_LABELS:
        for i in np.where(cls_train == cls)[0]:
            wrong = (le.transform([cls_train[i]])[0] != pred_tr[i])
            ax.scatter(Z_train[i, 0], Z_train[i, 1],
                       c=PALETTE[cls], marker=MARKER_TR[cls], s=75,
                       edgecolors='red' if wrong else 'black',
                       linewidths=1.8 if wrong else 0.6, alpha=0.85, zorder=3)
        for i in np.where(cls_test == cls)[0]:
            wrong = (le.transform([cls_test[i]])[0] != pred_te[i])
            ax.scatter(Z_test[i, 0], Z_test[i, 1],
                       c=PALETTE[cls], marker='*', s=300,
                       edgecolors='red' if wrong else 'black',
                       linewidths=2.0 if wrong else 1.0, alpha=1.00, zorder=5)

    handles  = [plt.scatter([], [], c=PALETTE[c], marker=MARKER_TR[c],
                            s=60, edgecolors='k', label=c) for c in CLASS_LABELS]
    handles += [plt.scatter([], [], c='grey', marker='*', s=200,
                            edgecolors='k', label='Test samples'),
                Line2D([0],[0], marker='o', color='w', markerfacecolor='white',
                       markeredgecolor='red', markeredgewidth=2.0, markersize=9,
                       label='Misclassified'),
                Line2D([0],[0], marker='o', color='w', markerfacecolor='white',
                       markeredgecolor='black', markeredgewidth=0.6, markersize=9,
                       label='Correct')]
    ax.legend(handles=handles, fontsize=9, framealpha=0.85, loc='upper right')
    ax.set_xlabel('PC1 (%.1f%%)' % (pca_viz.explained_variance_ratio_[0]*100), fontsize=11)
    ax.set_ylabel('PC2 (%.1f%%)' % (pca_viz.explained_variance_ratio_[1]*100), fontsize=11)
    ax.set_title(title, fontsize=10, fontweight='bold')
    ax.grid(True, alpha=0.25)
    add_stats_box(ax, cm_cv_arg, '5-fold CV (train)', cm_te_arg, 'External Test')

# Fig 10 — SVM 5-fold + test
fig, ax = plt.subplots(figsize=(9, 7))
nw_tr = int((le.transform(cls_train) != y_svm_cv).sum())
nw_te = int((le.transform(cls_test)  != y_svm_test).sum())
scatter_svm(ax, y_svm_cv, y_svm_test,
    'SVM-RBF Binary Cluster  |  5-fold CV (train) + External Test (Gr only)\n'
    'Train: %d wrong/60  CV acc=%.1f%%     Test: %d wrong/10  acc=%.1f%%' % (
        nw_tr, acc_svm_cv*100, nw_te, acc_svm_test*100),
    cm_svm_cv, cm_svm_test)
fig.suptitle(TITLE_SUF, fontsize=10)
fig.tight_layout()
save_fig(fig, '10_SVM_Cluster_5fold.png')

# Fig 11 — SVM test-only cluster
fig, ax = plt.subplots(figsize=(9, 7))
nw_te = int((le.transform(cls_test) != y_svm_test).sum())
scatter_svm(ax, le.transform(cls_train), y_svm_test,
    'SVM-RBF Binary Cluster  |  External Test focus  (Gr only, n=10)\n'
    'Test: %d wrong/10  |  Test accuracy = %.1f%%' % (nw_te, acc_svm_test*100),
    cm_svm_cv, cm_svm_test)
fig.suptitle(TITLE_SUF, fontsize=10)
fig.tight_layout()
save_fig(fig, '11_SVM_Cluster_Test.png')

# =============================================================================
# 20. SVM CONFUSION MATRICES WITH PER-CLASS STATS TABLE
# =============================================================================
plot_confusion_with_stats(
    cm_svm_cv, CLASS_LABELS,
    'SVM-RBF  5-fold CV  (train n=60)',
    '12_SVM_Confusion_5fold.png', 'Blues', acc_svm_cv)

plot_confusion_with_stats(
    cm_svm_test, CLASS_LABELS,
    'SVM-RBF  External Test  (Gr only, n=10)',
    '13_SVM_Confusion_Test.png', 'Oranges', acc_svm_test)

# =============================================================================
# 21. ACCURACY COMPARISON  (3-panel)
#     P1: Overall accuracy  P2: Per-class accuracy CV  P3: Per-class accuracy Test
# =============================================================================
pls_acc_cv_v,  pls_rec_cv_v  = per_class_metrics(cm_pls_cv)
svm_acc_cv_v,  svm_rec_cv_v  = per_class_metrics(cm_svm_cv)
pls_acc_te_v,  pls_rec_te_v  = per_class_metrics(cm_pls_test)
svm_acc_te_v,  svm_rec_te_v  = per_class_metrics(cm_svm_test)

fig, axes = plt.subplots(1, 3, figsize=(18, 6))
x_ov = np.arange(2); w_ov = 0.28
groups   = ['5-fold CV\n(train n=60)', 'External Test\n(Gr only, n=10)']
xc = np.arange(len(CLASS_LABELS)); wc = 0.30

# Panel 1: Overall accuracy
ax = axes[0]
b1 = ax.bar(x_ov - w_ov/2, [acc_pls_cv*100, acc_pls_test*100], w_ov,
            color='steelblue', edgecolor='k', lw=0.8, label='PLS-DA')
b2 = ax.bar(x_ov + w_ov/2, [acc_svm_cv*100, acc_svm_test*100], w_ov,
            color='darkorange', edgecolor='k', lw=0.8, label='SVM-RBF')
for bar in list(b1)+list(b2):
    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.8,
            '%.1f%%' % bar.get_height(), ha='center', va='bottom',
            fontsize=11, fontweight='bold')
ax.axhline(100, color='grey', ls=':', lw=1)
ax.set_xticks(x_ov); ax.set_xticklabels(groups, fontsize=10)
ax.set_ylabel('Accuracy (%)', fontsize=12); ax.set_ylim(0, 118)
ax.set_title('Overall Accuracy', fontsize=11, fontweight='bold')
ax.legend(fontsize=10); ax.grid(True, alpha=0.2, axis='y')

# Panel 2: Per-class accuracy (5-fold CV)
ax = axes[1]
b1 = ax.bar(xc-wc/2, pls_acc_cv_v, wc, color='steelblue',  edgecolor='k', lw=0.7, label='PLS-DA')
b2 = ax.bar(xc+wc/2, svm_acc_cv_v, wc, color='darkorange', edgecolor='k', lw=0.7, label='SVM-RBF')
for bar in list(b1)+list(b2):
    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.6,
            '%.1f%%' % bar.get_height(), ha='center', va='bottom',
            fontsize=9, fontweight='bold')
ax.axhline(100, color='grey', ls=':', lw=1)
ax.set_xticks(xc); ax.set_xticklabels(CLASS_LABELS, fontsize=10)
ax.set_ylabel('Accuracy (%)', fontsize=12); ax.set_ylim(0, 118)
ax.set_title('Per-Class Accuracy\n5-fold CV (train)', fontsize=11, fontweight='bold')
ax.legend(fontsize=10); ax.grid(True, alpha=0.2, axis='y')

# Panel 3: Per-class accuracy (External Test)
ax = axes[2]
b1 = ax.bar(xc-wc/2, pls_acc_te_v, wc, color='steelblue',  edgecolor='k', lw=0.7, label='PLS-DA')
b2 = ax.bar(xc+wc/2, svm_acc_te_v, wc, color='darkorange', edgecolor='k', lw=0.7, label='SVM-RBF')
for bar in list(b1)+list(b2):
    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.6,
            '%.1f%%' % bar.get_height(), ha='center', va='bottom',
            fontsize=9, fontweight='bold')
ax.axhline(100, color='grey', ls=':', lw=1)
ax.set_xticks(xc); ax.set_xticklabels(CLASS_LABELS, fontsize=10)
ax.set_ylabel('Accuracy (%)', fontsize=12); ax.set_ylim(0, 118)
ax.set_title('Per-Class Accuracy\nExternal Test (Gr only)', fontsize=11, fontweight='bold')
ax.legend(fontsize=10); ax.grid(True, alpha=0.2, axis='y')

fig.suptitle('Accuracy  |  PDMS vs SiO2/Si  |  PLS-DA vs SVM-RBF  |  ' + TITLE_SUF,
             fontsize=11, fontweight='bold')
fig.tight_layout()
save_fig(fig, '14_Accuracy_Comparison.png')

# =============================================================================
# 22. RECALL COMPARISON  (3-panel)
#     P1: Macro recall  P2: Per-class recall CV  P3: Per-class recall Test
# =============================================================================
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

# Panel 1: Macro recall
ax = axes[0]
b1 = ax.bar(x_ov-w_ov/2, [pls_rec_cv_v.mean(), pls_rec_te_v.mean()], w_ov,
            color='steelblue', edgecolor='k', lw=0.8, label='PLS-DA')
b2 = ax.bar(x_ov+w_ov/2, [svm_rec_cv_v.mean(), svm_rec_te_v.mean()], w_ov,
            color='darkorange', edgecolor='k', lw=0.8, label='SVM-RBF')
for bar in list(b1)+list(b2):
    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.8,
            '%.1f%%' % bar.get_height(), ha='center', va='bottom',
            fontsize=11, fontweight='bold')
ax.axhline(100, color='grey', ls=':', lw=1)
ax.set_xticks(x_ov); ax.set_xticklabels(groups, fontsize=10)
ax.set_ylabel('Recall (%)', fontsize=12); ax.set_ylim(0, 118)
ax.set_title('Macro-Average Recall', fontsize=11, fontweight='bold')
ax.legend(fontsize=10); ax.grid(True, alpha=0.2, axis='y')

# Panel 2: Per-class recall (5-fold CV)
ax = axes[1]
b1 = ax.bar(xc-wc/2, pls_rec_cv_v, wc, color='steelblue',  edgecolor='k', lw=0.7, label='PLS-DA')
b2 = ax.bar(xc+wc/2, svm_rec_cv_v, wc, color='darkorange', edgecolor='k', lw=0.7, label='SVM-RBF')
for bar in list(b1)+list(b2):
    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+1.5,
            '%.0f%%' % bar.get_height(), ha='center', va='bottom',
            fontsize=9, fontweight='bold')
ax.axhline(100, color='grey', ls=':', lw=1)
ax.set_xticks(xc); ax.set_xticklabels(CLASS_LABELS, fontsize=10)
ax.set_ylabel('Recall (%)', fontsize=12); ax.set_ylim(0, 122)
ax.set_title('Per-Class Recall\n5-fold CV (train)', fontsize=11, fontweight='bold')
ax.legend(fontsize=10); ax.grid(True, alpha=0.2, axis='y')

# Panel 3: Per-class recall (External Test)
ax = axes[2]
b1 = ax.bar(xc-wc/2, pls_rec_te_v, wc, color='steelblue',  edgecolor='k', lw=0.7, label='PLS-DA')
b2 = ax.bar(xc+wc/2, svm_rec_te_v, wc, color='darkorange', edgecolor='k', lw=0.7, label='SVM-RBF')
for bar in list(b1)+list(b2):
    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+1.5,
            '%.0f%%' % bar.get_height(), ha='center', va='bottom',
            fontsize=9, fontweight='bold')
ax.axhline(100, color='grey', ls=':', lw=1)
ax.set_xticks(xc); ax.set_xticklabels(CLASS_LABELS, fontsize=10)
ax.set_ylabel('Recall (%)', fontsize=12); ax.set_ylim(0, 122)
ax.set_title('Per-Class Recall\nExternal Test (Gr only)', fontsize=11, fontweight='bold')
ax.legend(fontsize=10); ax.grid(True, alpha=0.2, axis='y')

fig.suptitle('Recall  |  PDMS vs SiO2/Si  |  PLS-DA vs SVM-RBF  |  ' + TITLE_SUF,
             fontsize=12, fontweight='bold')
fig.tight_layout()
save_fig(fig, '15_Recall_Comparison.png')

# =============================================================================
# 23. MEAN SPECTRA  ± 1 SD  (binary classes over all 80 spectra)
# =============================================================================
X_all_sm = savgol_filter(X_crop, window_length=SG_WINDOW, polyorder=SG_POLY, axis=1)

fig, ax = plt.subplots(figsize=(10, 5))
for cls in CLASS_LABELS:
    idx = classes == cls
    m = X_all_sm[idx].mean(axis=0)
    s = X_all_sm[idx].std(axis=0)
    ax.plot(wavenumbers, m, color=PALETTE[cls], lw=2.0, label='%s (n=%d)' % (cls, idx.sum()))
    ax.fill_between(wavenumbers, m-s, m+s, color=PALETTE[cls], alpha=0.15)
ax.set_xlim(WAVENUMBER_MIN, WAVENUMBER_MAX)
ax.set_xlabel('Wavenumber (cm⁻¹)', fontsize=12)
ax.set_ylabel('Intensity (a.u.)', fontsize=12)
ax.set_title('Mean Raman Spectra ± 1 SD  |  Binary Classes  |  %d–%d cm⁻¹' % (
    WAVENUMBER_MIN, WAVENUMBER_MAX), fontsize=11, fontweight='bold')
ax.legend(fontsize=10); ax.grid(True, alpha=0.25)
fig.tight_layout()
save_fig(fig, '16_Mean_Spectra.png')

# =============================================================================
# 24. SAVE ALL RESULTS TO EXCEL
# =============================================================================
out_xlsx = os.path.join(OUTPUT_DIR, 'Results_Binary.xlsx')

with pd.ExcelWriter(out_xlsx, engine='openpyxl') as writer:

    # Summary
    rows = [
        ('Input file',                         FILE_NAME),
        ('Binary Class 1 (PDMS)',              'PDMS + Gr-PDMS  (n=40 total)'),
        ('Binary Class 2 (SiO2/Si)',           'SiO2/Si + Gr-SiO2/Si  (n=40 total)'),
        ('Spectral window (cm-1)',              '%d-%d' % (WAVENUMBER_MIN, WAVENUMBER_MAX)),
        ('Spectral points in window',           X_crop.shape[1]),
        ('SG window / poly',                    '%d/%d' % (SG_WINDOW, SG_POLY)),
        ('VIP > 1 (initial model)',             n_vip_gt1),
        ('VIP top-N selected',                  top_n),
        ('N latent variables (PLS-DA)',         N_COMPONENTS),
        ('Total samples',                       len(y)),
        ('Training set',                        '%d (15 Gr-PDMS + 15 PDMS + 15 Gr-SiO2/Si + 15 SiO2/Si)' % len(y_train)),
        ('Test set',                            '%d — Gr only (5 Gr-PDMS + 5 Gr-SiO2/Si)' % len(y_test)),
        ('Excluded (no-Gr)',                    '10 spectra excluded entirely (5 per class)'),
        ('', ''),
        ('--- PLS-DA ---',                      ''),
        ('PLS-DA Train re-sub acc (%)',         '%.2f' % (train_acc_resub * 100)),
        ('PLS-DA 5-fold CV acc (%)',            '%.2f' % (acc_pls_cv   * 100)),
        ('PLS-DA External test acc (%)',        '%.2f' % (acc_pls_test * 100)),
        ('PLS-DA Permutation p-value',          '%.4f' % p_val),
        ('PLS-DA Permutation result',
         'Significant (p<0.05)' if p_val < 0.05 else 'NOT significant'),
        ('', ''),
        ('--- SVM-RBF ---',                     ''),
        ('SVM-RBF 5-fold CV acc (%)',           '%.2f' % (acc_svm_cv   * 100)),
        ('SVM-RBF External test acc (%)',       '%.2f' % (acc_svm_test * 100)),
        ('SVM kernel', 'RBF'), ('SVM C', 10), ('SVM gamma', 'scale'),
    ]
    for i, r2 in enumerate(r2x_cum, 1):
        rows.append(('R2X cumulative LV%d (%%)' % i, '%.4f' % (r2 * 100)))
    pd.DataFrame(rows, columns=['Metric', 'Value']).to_excel(
        writer, sheet_name='Summary', index=False)

    # Per-class Accuracy & Recall
    cls_rows = []
    for i, cls in enumerate(CLASS_LABELS):
        cls_rows.append({
            'Class':                 cls,
            'N_Train':               int(cm_pls_cv[i,:].sum()),
            'N_Test':                int(cm_pls_test[i,:].sum()),
            'PLSDA_CV_Accuracy_%':   '%.2f' % pls_acc_cv_v[i],
            'PLSDA_CV_Recall_%':     '%.2f' % pls_rec_cv_v[i],
            'PLSDA_Test_Accuracy_%': '%.2f' % pls_acc_te_v[i],
            'PLSDA_Test_Recall_%':   '%.2f' % pls_rec_te_v[i],
            'SVM_CV_Accuracy_%':     '%.2f' % svm_acc_cv_v[i],
            'SVM_CV_Recall_%':       '%.2f' % svm_rec_cv_v[i],
            'SVM_Test_Accuracy_%':   '%.2f' % svm_acc_te_v[i],
            'SVM_Test_Recall_%':     '%.2f' % svm_rec_te_v[i],
        })
    pd.DataFrame(cls_rows).to_excel(
        writer, sheet_name='Per_Class_Metrics', index=False)

    # CV predictions (train)
    pd.DataFrame({
        'Label':               names_train,
        'True_Class':          cls_train,
        'PLSDA_5fold_Pred':    le.inverse_transform(y_pls_cv),
        'PLSDA_5fold_Correct': (cls_train == le.inverse_transform(y_pls_cv)),
        'SVM_5fold_Pred':      le.inverse_transform(y_svm_cv),
        'SVM_5fold_Correct':   (cls_train == le.inverse_transform(y_svm_cv)),
        'LV1': T_train[:,0], 'LV2': T_train[:,1],
        'LV3': T_train[:,2], 'PC1': Z_train[:,0], 'PC2': Z_train[:,1],
    }).to_excel(writer, sheet_name='Train_CV_Predictions', index=False)

    # Test predictions
    pd.DataFrame({
        'Label':              names_test,
        'True_Class':         cls_test,
        'PLSDA_Test_Pred':    le.inverse_transform(y_pls_test),
        'PLSDA_Test_Correct': (cls_test == le.inverse_transform(y_pls_test)),
        'SVM_Test_Pred':      le.inverse_transform(y_svm_test),
        'SVM_Test_Correct':   (cls_test == le.inverse_transform(y_svm_test)),
        'LV1': T_test[:,0], 'LV2': T_test[:,1],
        'LV3': T_test[:,2], 'PC1': Z_test[:,0], 'PC2': Z_test[:,1],
    }).to_excel(writer, sheet_name='Test_Predictions', index=False)

    # Confusion matrices
    for sheet, cm in [
        ('CM_PLSDA_5fold', cm_pls_cv),
        ('CM_PLSDA_Test',  cm_pls_test),
        ('CM_SVM_5fold',   cm_svm_cv),
        ('CM_SVM_Test',    cm_svm_test),
    ]:
        pd.DataFrame(cm,
                     index=pd.Index(CLASS_LABELS, name='True\\Predicted'),
                     columns=CLASS_LABELS).to_excel(writer, sheet_name=sheet)

    # VIP scores
    pd.DataFrame({
        'Wavenumber_cm1':         wavenumbers,
        'VIP_Score':              vip_all,
        'VIP_gt_1':               vip_all > 1,
        'Selected_Top%d' % top_n: sel_mask,
    }).to_excel(writer, sheet_name='VIP_Scores', index=False)

    # Loadings
    load_df = pd.DataFrame(P_load,
        columns=['Loading_LV%d' % (i+1) for i in range(N_COMPONENTS)])
    load_df.insert(0, 'Wavenumber_cm1', wn_sel)
    load_df.to_excel(writer, sheet_name='Loadings', index=False)

    # Permutation
    pd.DataFrame({
        'Permutation_Index': np.arange(1, N_PERM+1),
        'Permuted_Accuracy': perm_scores,
    }).to_excel(writer, sheet_name='Permutation_Test', index=False)

print('\n[SAVED] %s' % out_xlsx)

# =============================================================================
# 25. FINAL CONSOLE SUMMARY
# =============================================================================
print('\n' + '=' * 68)
print('  BINARY RESULTS  |  PDMS vs SiO2/Si  |  5-fold CV + External Test')
print('=' * 68)
print('  Spectral window  : %d-%d cm-1   |   %d points' % (
    WAVENUMBER_MIN, WAVENUMBER_MAX, X_crop.shape[1]))
print('  VIP > 1          : %d / %d   ->   top-%d selected' % (
    n_vip_gt1, len(vip_all), top_n))
print('  Split            : 60 train  |  10 test (Gr only: 5+5)  |  10 excluded (no-Gr)')
print()
print('  %-10s  %20s  %18s' % ('MODEL', '5-fold CV (train)', 'External Test'))
print('  ' + '-' * 52)
print('  %-10s  %19.1f%%  %17.1f%%' % ('PLS-DA',  acc_pls_cv*100,  acc_pls_test*100))
print('  %-10s  %19.1f%%  %17.1f%%' % ('SVM-RBF', acc_svm_cv*100,  acc_svm_test*100))
print()
print('  PLS-DA permutation p-value : %.4f  [%s]' % (
    p_val, 'SIGNIFICANT' if p_val < 0.05 else 'NOT significant'))
print()
print('  Per-class Accuracy & Recall — 5-fold CV (train):')
print('  %-12s  %9s  %7s  |  %9s  %7s' % (
    'Class', 'PLS Acc', 'PLS Rec', 'SVM Acc', 'SVM Rec'))
print('  ' + '-' * 54)
for i, cls in enumerate(CLASS_LABELS):
    print('  %-12s  %8.1f%%  %6.1f%%  |  %8.1f%%  %6.1f%%' % (
        cls, pls_acc_cv_v[i], pls_rec_cv_v[i], svm_acc_cv_v[i], svm_rec_cv_v[i]))
print()
print('  Per-class Accuracy & Recall — External Test:')
print('  %-12s  %9s  %7s  |  %9s  %7s' % (
    'Class', 'PLS Acc', 'PLS Rec', 'SVM Acc', 'SVM Rec'))
print('  ' + '-' * 54)
for i, cls in enumerate(CLASS_LABELS):
    print('  %-12s  %8.1f%%  %6.1f%%  |  %8.1f%%  %6.1f%%' % (
        cls, pls_acc_te_v[i], pls_rec_te_v[i], svm_acc_te_v[i], svm_rec_te_v[i]))
print()
print('  R2X cumulative (PLS-DA):')
for i, r2 in enumerate(r2x_cum, 1):
    print('    LV%d: %.2f%%' % (i, r2*100))
print()
print('  All outputs saved to : %s' % OUTPUT_DIR)
print('=' * 68)
