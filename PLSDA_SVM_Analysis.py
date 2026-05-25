"""
=============================================================================
PLS-DA + SVM-RBF  |  TRAINING (60) / TEST (20) SPLIT
SSY Raman Spectroscopy  |  Four Substrate Classes
SPECTRAL WINDOW : 450 - 1800 cm-1

CROSS-VALIDATION STRATEGY (no data leakage)
  PLS-DA  :  5-fold CV on training set  +  external test evaluation
  SVM-RBF :  5-fold CV on training set  +  external test evaluation
  Per-class Accuracy  = (TP + TN) / N_total
  Per-class Recall    = TP / (TP + FN)

OUTPUTS -> C:/Users/Hira Aman/Desktop/PROF_DOMENICO'S/PLSDA_SVM_Results/
  01_PLSDA_Scores_LV1_LV2.png
  02_PLSDA_Scores_LV1_LV3.png
  03_PLSDA_Scores_3D.png
  04_PLSDA_Confusion_5fold.png
  05_PLSDA_Confusion_Test.png
  06_PLSDA_VIP_Scores.png
  07_PLSDA_Loadings.png
  08_PLSDA_Permutation.png
  09_PLSDA_Cluster_5fold.png
  10_SVM_Cluster_5fold_Train.png
  11_SVM_Cluster_Test.png
  12_SVM_Confusion_5fold.png
  13_SVM_Confusion_Test.png
  14_Accuracy_Comparison.png
  15_Recall_Comparison.png
  16_Mean_Spectra.png
  Results.xlsx

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
    StratifiedKFold, cross_val_predict, permutation_test_score
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
VIP_TOP_N      = 1000
N_COMPONENTS   = 5
N_PERM         = 999
RANDOM_STATE   = 42

OUTPUT_DIR = os.path.join(DATA_PATH, "PLSDA_SVM_Results")

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
        "In PyCharm Terminal run:   pip install xlrd\nThen re-run.")

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
# 3.  LOAD DATA
# =============================================================================
print("=" * 65)
print("  PLS-DA + SVM-RBF  |  SSY Raman  |  450-1800 cm-1")
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

assert X_crop.shape[1] > 0,          "ERROR: No points in specified range!"
assert np.isnan(X_crop).sum() == 0,  "ERROR: NaN values found in data!"
print("[INFO] Cropped : %.4f - %.4f cm-1  (%d points)  OK" % (
    wavenumbers[0], wavenumbers[-1], X_crop.shape[1]))

classes_raw = np.array([assign_class(c) for c in col_names_all])
bad_mask    = np.array([c is None for c in classes_raw])
if bad_mask.any():
    print("\n[WARNING] %d unrecognised column(s)" % bad_mask.sum())

valid_mask = ~bad_mask
X_crop     = X_crop[valid_mask]
classes    = classes_raw[valid_mask]
col_names  = [col_names_all[i] for i in range(len(col_names_all)) if valid_mask[i]]

le = LabelEncoder()
y  = le.fit_transform(classes)

print("\n[INFO] Valid samples : %d" % len(classes))
for cls, cnt in zip(*np.unique(classes, return_counts=True)):
    print("         %-25s : %d" % (cls, cnt))
print("\n[INFO] Class encoding : %s" % str(
    dict(zip(le.classes_, le.transform(le.classes_)))))

# =============================================================================
# 4.  STRATIFIED TRAIN / TEST SPLIT  (60 train / 20 test)
# =============================================================================
print("\n[SPLIT] Building stratified train/test split ...")
np.random.seed(RANDOM_STATE)

train_list = []
test_list  = []

for cls in le.classes_:
    cls_mask   = np.where(classes == cls)[0]
    lugl_mask  = np.array([i for i in cls_mask if 'lugl'  in col_names[i].lower()])
    marzo_mask = np.array([i for i in cls_mask if 'marzo' in col_names[i].lower()])
    np.random.shuffle(lugl_mask)
    np.random.shuffle(marzo_mask)
    test_list  += list(lugl_mask[:2])  + list(marzo_mask[:3])
    train_list += list(lugl_mask[2:])  + list(marzo_mask[3:])

train_idx = np.array(sorted(train_list))
test_idx  = np.array(sorted(test_list))

assert len(train_idx) == 60, "ERROR: training set has %d samples!" % len(train_idx)
assert len(test_idx)  == 20, "ERROR: test set has %d samples!"     % len(test_idx)
assert len(set(train_list) & set(test_list)) == 0, "ERROR: train/test overlap!"

X_train     = X_crop[train_idx]
X_test      = X_crop[test_idx]
y_train     = y[train_idx]
y_test      = y[test_idx]
cls_train   = classes[train_idx]
cls_test    = classes[test_idx]
names_train = [col_names[i] for i in train_idx]
names_test  = [col_names[i] for i in test_idx]

print("[SPLIT] Training set : %d spectra" % len(y_train))
for cls, cnt in zip(*np.unique(cls_train, return_counts=True)):
    lc = sum(1 for n in names_train if assign_class(n)==cls and 'lugl'  in n.lower())
    mc = sum(1 for n in names_train if assign_class(n)==cls and 'marzo' in n.lower())
    print("           %-25s: %d  (luglio=%d, marzo=%d)" % (cls, cnt, lc, mc))
print("\n[SPLIT] Test set : %d spectra" % len(y_test))
for cls, cnt in zip(*np.unique(cls_test, return_counts=True)):
    lc = sum(1 for n in names_test if assign_class(n)==cls and 'lugl'  in n.lower())
    mc = sum(1 for n in names_test if assign_class(n)==cls and 'marzo' in n.lower())
    print("           %-25s: %d  (luglio=%d, marzo=%d)" % (cls, cnt, lc, mc))

# =============================================================================
# 5.  PRE-PROCESSING  (SG smooth)
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
# 7.  VIP SELECTION  (fit on TRAINING only — no leakage)
# =============================================================================
print("\n[VIP] Initial PLS-DA on training set -> top-%d VIP ..." % VIP_TOP_N)

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
print("[VIP] Top-%d VIP selected  (%.1f - %.1f cm-1)" % (
    top_n, wn_sel.min(), wn_sel.max()))

# =============================================================================
# 8.  HELPER CLASSIFIER
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
    def transform(self, X):
        return self._pls.transform(X)[0]

cv5 = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

sc_final = StandardScaler().fit(X_train_sel)
Xtr_fin  = sc_final.transform(X_train_sel)
Xte_fin  = sc_final.transform(X_test_sel)

# =============================================================================
# 9.  PLS-DA FINAL MODEL + CV
# =============================================================================
print("\n[PLS-DA] Fitting final model (%d LVs, top-%d vars) ..." % (N_COMPONENTS, top_n))

Yd_fin    = pd.get_dummies(pd.Series(y_train)).values.astype(float)
pls_final = PLSRegression(n_components=N_COMPONENTS, scale=False, max_iter=10000)
pls_final.fit(Xtr_fin, Yd_fin)

T_train = pls_final.x_scores_           # (60, 5)
T_test  = pls_final.transform(Xte_fin)  # (20, 5)
P_load  = pls_final.x_loadings_         # (top_n, 5)

SS_tot  = np.sum(Xtr_fin ** 2)
r2x_cum = []
for lv in range(1, N_COMPONENTS + 1):
    Xrec = pls_final.x_scores_[:, :lv] @ pls_final.x_loadings_[:, :lv].T
    r2x_cum.append(1 - np.sum((Xtr_fin - Xrec) ** 2) / SS_tot)
print("[PLS-DA] R2X cumulative : " + "  ".join(
    ["LV%d=%.1f%%" % (i+1, r*100) for i, r in enumerate(r2x_cum)]))

train_acc_resub = accuracy_score(y_train, np.argmax(pls_final.predict(Xtr_fin), axis=1))
print("[PLS-DA] Training re-sub accuracy : %.1f%%" % (train_acc_resub * 100))

clf_pipe_pls = Pipeline([('sc', StandardScaler()), ('plsda', PLSDAClf(N_COMPONENTS))])

print("[PLS-DA] 5-fold CV on training set ...")
y_pls_5fold = cross_val_predict(clf_pipe_pls, X_train_sel, y_train, cv=cv5)
acc_pls_5f  = accuracy_score(y_train, y_pls_5fold)
cm_pls_5f   = confusion_matrix(y_train, y_pls_5fold)
print("[PLS-DA] 5-fold CV accuracy : %.1f%%" % (acc_pls_5f * 100))

y_pls_test   = np.argmax(pls_final.predict(Xte_fin), axis=1)
acc_pls_test = accuracy_score(y_test, y_pls_test)
cm_pls_test  = confusion_matrix(y_test, y_pls_test)
print("[PLS-DA] External test accuracy : %.1f%%" % (acc_pls_test * 100))

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
# 10. SVM-RBF + CV
# =============================================================================
svm_pipe = Pipeline([
    ('sc',  StandardScaler()),
    ('svm', SVC(kernel='rbf', C=10, gamma='scale',
                probability=True, random_state=RANDOM_STATE))
])

print("\n[SVM-RBF] 5-fold CV on training set ...")
y_svm_5fold = cross_val_predict(svm_pipe, X_train_sel, y_train, cv=cv5)
acc_svm_5f  = accuracy_score(y_train, y_svm_5fold)
cm_svm_5f   = confusion_matrix(y_train, y_svm_5fold)
print("[SVM-RBF] 5-fold CV accuracy : %.1f%%" % (acc_svm_5f * 100))

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
Xtr_viz = sc_viz.transform(X_train_sel)
Xte_viz = sc_viz.transform(X_test_sel)
pca_viz.fit(Xtr_viz)
Z_train = pca_viz.transform(Xtr_viz)
Z_test  = pca_viz.transform(Xte_viz)
print("\n[PCA-viz] PC1=%.1f%%  PC2=%.1f%% explained variance" % (
    pca_viz.explained_variance_ratio_[0]*100,
    pca_viz.explained_variance_ratio_[1]*100))

# =============================================================================
# 12. COLOUR / MARKER SCHEME
# =============================================================================
PALETTE = {
    'SSY-PDMS'       : '#1f77b4',
    'SSY-Gr-PDMS'    : '#ff7f0e',
    'SSY-SiO2/Si'    : '#2ca02c',
    'SSY-Gr-SiO2/Si' : '#d62728',
}
MARKER_TR = {
    'SSY-PDMS'       : 'o',
    'SSY-Gr-PDMS'    : 's',
    'SSY-SiO2/Si'    : '^',
    'SSY-Gr-SiO2/Si' : 'D',
}
TITLE_SUF = "VIP-top-%d | %d LVs | %d-%d cm-1" % (
    top_n, N_COMPONENTS, WAVENUMBER_MIN, WAVENUMBER_MAX)

def lv_var(i):
    return (r2x_cum[i] - (r2x_cum[i-1] if i > 0 else 0)) * 100

def save_fig(fig, fname):
    fig.savefig(os.path.join(OUTPUT_DIR, fname), dpi=150, bbox_inches='tight')
    plt.close(fig)
    print('[SAVED] %s' % fname)

# =============================================================================
# HELPER: Per-class Accuracy and Recall from confusion matrix
#   Accuracy_i = (TP_i + TN_i) / N_total
#   Recall_i   = TP_i / (TP_i + FN_i)
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
# HELPER: Confusion matrix + per-class stats table  (GridSpec layout)
# =============================================================================
def plot_confusion_with_stats(cm, labels, title, fname, cmap, overall_acc):
    acc_list, rec_list = per_class_metrics(cm)
    short_labels = [l.replace('SSY-', '') for l in labels]

    fig = plt.figure(figsize=(8, 9))
    gs  = GridSpec(2, 1, figure=fig, height_ratios=[3, 1.1], hspace=0.45)
    ax_cm  = fig.add_subplot(gs[0])
    ax_tbl = fig.add_subplot(gs[1])

    ConfusionMatrixDisplay(cm, display_labels=labels).plot(
        ax=ax_cm, colorbar=False, cmap=cmap, xticks_rotation=30)
    ax_cm.set_title('%s\nOverall Accuracy = %.1f%%' % (title, overall_acc * 100),
                    fontsize=11, fontweight='bold')

    ax_tbl.axis('off')
    col_labels = ['Class', 'Accuracy (%)', 'Recall (%)']
    table_data = [
        [short_labels[i], '%.1f' % acc_list[i], '%.1f' % rec_list[i]]
        for i in range(len(labels))
    ]
    tbl = ax_tbl.table(cellText=table_data, colLabels=col_labels,
                       loc='center', cellLoc='center')
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9.5)
    tbl.scale(1, 1.8)
    for j in range(len(col_labels)):
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
#         Shows per-class Accuracy & Recall; optionally two sets (CV + Test)
# =============================================================================
def add_stats_box(ax, cm_main, label_main, cm_second=None, label_second=None):
    acc_m, rec_m = per_class_metrics(cm_main)
    short = [c.replace('SSY-', '') for c in le.classes_]
    lines = [label_main]
    lines.append('%-14s  %5s  %6s' % ('Class', 'Acc', 'Recall'))
    lines.append('-' * 30)
    for i in range(len(le.classes_)):
        lines.append('%-14s  %4.1f%%  %5.1f%%' % (short[i], acc_m[i], rec_m[i]))
    if cm_second is not None:
        acc_s, rec_s = per_class_metrics(cm_second)
        lines += ['', label_second,
                  '%-14s  %5s  %6s' % ('Class', 'Acc', 'Recall'),
                  '-' * 30]
        for i in range(len(le.classes_)):
            lines.append('%-14s  %4.1f%%  %5.1f%%' % (short[i], acc_s[i], rec_s[i]))
    ax.text(0.02, 0.02, '\n'.join(lines), transform=ax.transAxes,
            fontsize=6.8, verticalalignment='bottom', fontfamily='monospace',
            bbox=dict(boxstyle='round,pad=0.4', facecolor='white',
                      alpha=0.88, edgecolor='#888888'))

# =============================================================================
# 13. PLS-DA SCORE PLOTS  (train filled markers, test stars)
# =============================================================================
def scatter_train_test_scores(ax, lv_x, lv_y, title):
    for cls in le.classes_:
        idx_tr = cls_train == cls
        idx_te = cls_test  == cls
        ax.scatter(T_train[idx_tr, lv_x], T_train[idx_tr, lv_y],
                   c=PALETTE[cls], marker=MARKER_TR[cls], s=70,
                   edgecolors='k', linewidths=0.5, alpha=0.80,
                   label=cls + ' (train)', zorder=3)
        ax.scatter(T_test[idx_te, lv_x], T_test[idx_te, lv_y],
                   c=PALETTE[cls], marker='*', s=280,
                   edgecolors='k', linewidths=1.0, alpha=1.00,
                   label=cls + ' (test)', zorder=5)
    ax.axhline(0, color='grey', lw=0.7, ls='--')
    ax.axvline(0, color='grey', lw=0.7, ls='--')
    ax.set_title(title, fontsize=10, fontweight='bold')
    ax.legend(fontsize=7, framealpha=0.85, ncol=2)
    ax.grid(True, alpha=0.25)

# Fig 01 — LV1 vs LV2
fig, ax = plt.subplots(figsize=(8, 6))
scatter_train_test_scores(ax, 0, 1,
    'PLS-DA Scores  LV1 vs LV2\nFilled=train | Stars=test | ' + TITLE_SUF)
ax.set_xlabel('LV1 (%.1f%% var.)' % lv_var(0), fontsize=12)
ax.set_ylabel('LV2 (%.1f%% var.)' % lv_var(1), fontsize=12)
fig.tight_layout()
save_fig(fig, '01_PLSDA_Scores_LV1_LV2.png')

# Fig 02 — LV1 vs LV3
fig, ax = plt.subplots(figsize=(8, 6))
scatter_train_test_scores(ax, 0, 2,
    'PLS-DA Scores  LV1 vs LV3\nFilled=train | Stars=test | ' + TITLE_SUF)
ax.set_xlabel('LV1 (%.1f%% var.)' % lv_var(0), fontsize=12)
ax.set_ylabel('LV3 (%.1f%% var.)' % lv_var(2), fontsize=12)
fig.tight_layout()
save_fig(fig, '02_PLSDA_Scores_LV1_LV3.png')

# Fig 03 — 3D scores
fig = plt.figure(figsize=(9, 7))
ax  = fig.add_subplot(111, projection='3d')
for cls in le.classes_:
    itr = cls_train == cls
    ite = cls_test  == cls
    ax.scatter(T_train[itr,0], T_train[itr,1], T_train[itr,2],
               c=PALETTE[cls], marker=MARKER_TR[cls], s=55,
               edgecolors='k', linewidths=0.4, alpha=0.80, label=cls+' (train)')
    ax.scatter(T_test[ite,0],  T_test[ite,1],  T_test[ite,2],
               c=PALETTE[cls], marker='*', s=220,
               edgecolors='k', linewidths=0.8, alpha=1.00, label=cls+' (test)')
ax.set_xlabel('LV1 (%.1f%%)' % lv_var(0), fontsize=9)
ax.set_ylabel('LV2 (%.1f%%)' % lv_var(1), fontsize=9)
ax.set_zlabel('LV3 (%.1f%%)' % lv_var(2), fontsize=9)
ax.set_title('PLS-DA 3D Scores  |  ' + TITLE_SUF, fontsize=10, fontweight='bold')
ax.legend(fontsize=7, ncol=2)
fig.tight_layout()
save_fig(fig, '03_PLSDA_Scores_3D.png')

# =============================================================================
# 14. PLS-DA CONFUSION MATRICES WITH PER-CLASS STATS TABLE
# =============================================================================
plot_confusion_with_stats(
    cm_pls_5f, le.classes_,
    'PLS-DA  5-fold CV  (train n=60)',
    '04_PLSDA_Confusion_5fold.png', 'Blues', acc_pls_5f)

plot_confusion_with_stats(
    cm_pls_test, le.classes_,
    'PLS-DA  External Test  (n=20)',
    '05_PLSDA_Confusion_Test.png', 'Oranges', acc_pls_test)

# =============================================================================
# 15. VIP SCORES
# =============================================================================
sel_mask = np.zeros(len(wavenumbers), dtype=bool)
sel_mask[top_idx] = True

fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(wavenumbers, vip_all, color='steelblue', lw=0.9, zorder=2)
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
ax.set_title('VIP Scores  |  ' + TITLE_SUF, fontsize=11, fontweight='bold')
ax.legend(fontsize=9); ax.grid(True, alpha=0.25)
fig.tight_layout()
save_fig(fig, '06_PLSDA_VIP_Scores.png')

# =============================================================================
# 16. LOADINGS  LV1 / LV2 / LV3
# =============================================================================
fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
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
axes[-1].set_xlabel('Wavenumber (cm⁻¹)', fontsize=12)
axes[0].set_title('PLS-DA Loadings  LV1/LV2/LV3  |  ' + TITLE_SUF,
                  fontsize=10, fontweight='bold')
fig.tight_layout()
save_fig(fig, '07_PLSDA_Loadings.png')

# =============================================================================
# 17. PERMUTATION TEST
# =============================================================================
fig, ax = plt.subplots(figsize=(7, 4.5))
ax.hist(perm_scores, bins=35, color='lightsteelblue', edgecolor='white',
        label='Permuted accuracy')
ax.axvline(obs_score, color='crimson', lw=2.5,
           label='Observed: %.1f%%   p=%.4f' % (obs_score*100, p_val))
ax.axvline(0.25, color='grey', lw=1.2, ls=':', label='Chance (25%)')
ax.set_xlabel('5-fold CV accuracy', fontsize=12)
ax.set_ylabel('Count', fontsize=12)
ax.set_title('Permutation Test (%d perm.)  |  PLS-DA  |  ' % N_PERM + TITLE_SUF,
             fontsize=10, fontweight='bold')
ax.legend(fontsize=9); ax.grid(True, alpha=0.25)
fig.tight_layout()
save_fig(fig, '08_PLSDA_Permutation.png')

# =============================================================================
# 18. PLS-DA CLUSTER PLOT  (LV1 vs LV2 space)
#     Train = 5-fold CV predictions   Test = external predictions
#     Stats box shows per-class Accuracy & Recall for both sets
# =============================================================================
fig, ax = plt.subplots(figsize=(9, 7))

for cls in le.classes_:
    for i in np.where(cls_train == cls)[0]:
        wrong = (le.transform([cls_train[i]])[0] != y_pls_5fold[i])
        ax.scatter(T_train[i, 0], T_train[i, 1],
                   c=PALETTE[cls], marker=MARKER_TR[cls], s=70,
                   edgecolors='red' if wrong else 'black',
                   linewidths=2.0 if wrong else 0.5, alpha=0.82, zorder=3)
    for i in np.where(cls_test == cls)[0]:
        wrong = (le.transform([cls_test[i]])[0] != y_pls_test[i])
        ax.scatter(T_test[i, 0], T_test[i, 1],
                   c=PALETTE[cls], marker='*', s=280,
                   edgecolors='red' if wrong else 'black',
                   linewidths=2.0 if wrong else 1.0, alpha=1.00, zorder=5)

handles = [plt.scatter([], [], c=PALETTE[c], marker=MARKER_TR[c],
                       s=55, edgecolors='k', label=c)
           for c in le.classes_]
handles += [
    plt.scatter([], [], c='grey', marker='*', s=200, edgecolors='k',
                label='Test samples'),
    Line2D([0],[0], marker='o', color='w', markerfacecolor='white',
           markeredgecolor='red', markeredgewidth=2.0, markersize=8,
           label='Misclassified'),
    Line2D([0],[0], marker='o', color='w', markerfacecolor='white',
           markeredgecolor='black', markeredgewidth=0.6, markersize=8,
           label='Correct'),
]
ax.legend(handles=handles, fontsize=7.5, framealpha=0.85, loc='upper right')
ax.axhline(0, color='grey', lw=0.7, ls='--')
ax.axvline(0, color='grey', lw=0.7, ls='--')
ax.set_xlabel('LV1 (%.1f%% var.)' % lv_var(0), fontsize=10)
ax.set_ylabel('LV2 (%.1f%% var.)' % lv_var(1), fontsize=10)
nw_tr = int((le.transform(cls_train) != y_pls_5fold).sum())
nw_te = int((le.transform(cls_test)  != y_pls_test).sum())
ax.set_title(
    'PLS-DA Cluster  |  5-fold CV (train) + External Test  |  ' + TITLE_SUF + '\n'
    'Train: %d wrong / 60  (CV acc=%.1f%%)     Test: %d wrong / 20  (acc=%.1f%%)' % (
        nw_tr, acc_pls_5f*100, nw_te, acc_pls_test*100),
    fontsize=9.5, fontweight='bold')
ax.grid(True, alpha=0.25)
add_stats_box(ax, cm_pls_5f, '5-fold CV (train)', cm_pls_test, 'External Test')
fig.tight_layout()
save_fig(fig, '09_PLSDA_Cluster_5fold.png')

# =============================================================================
# 19. SVM CLUSTER PLOTS  (PCA space)
# =============================================================================
def scatter_svm_cluster(ax, pred_tr, pred_te, title, cm_cv, cm_te,
                        stats_label_cv, stats_label_te):
    for cls in le.classes_:
        for i in np.where(cls_train == cls)[0]:
            wrong = (le.transform([cls_train[i]])[0] != pred_tr[i])
            ax.scatter(Z_train[i, 0], Z_train[i, 1],
                       c=PALETTE[cls], marker=MARKER_TR[cls], s=70,
                       edgecolors='red' if wrong else 'black',
                       linewidths=1.8 if wrong else 0.5, alpha=0.85, zorder=3)
        for i in np.where(cls_test == cls)[0]:
            wrong = (le.transform([cls_test[i]])[0] != pred_te[i])
            ax.scatter(Z_test[i, 0], Z_test[i, 1],
                       c=PALETTE[cls], marker='*', s=280,
                       edgecolors='red' if wrong else 'black',
                       linewidths=2.0 if wrong else 1.0, alpha=1.00, zorder=5)

    handles = [plt.scatter([], [], c=PALETTE[c], marker=MARKER_TR[c],
                           s=55, edgecolors='k', label=c)
               for c in le.classes_]
    handles += [
        plt.scatter([], [], c='grey', marker='*', s=200, edgecolors='k',
                    label='Test samples'),
        Line2D([0],[0], marker='o', color='w', markerfacecolor='white',
               markeredgecolor='red', markeredgewidth=2.0, markersize=8,
               label='Misclassified'),
        Line2D([0],[0], marker='o', color='w', markerfacecolor='white',
               markeredgecolor='black', markeredgewidth=0.6, markersize=8,
               label='Correct'),
    ]
    ax.legend(handles=handles, fontsize=7.5, framealpha=0.85, loc='upper right')
    ax.set_xlabel('PC1 (%.1f%%)' % (pca_viz.explained_variance_ratio_[0]*100), fontsize=10)
    ax.set_ylabel('PC2 (%.1f%%)' % (pca_viz.explained_variance_ratio_[1]*100), fontsize=10)
    ax.set_title(title, fontsize=9.5, fontweight='bold')
    ax.grid(True, alpha=0.25)
    add_stats_box(ax, cm_cv, stats_label_cv, cm_te, stats_label_te)

# Fig 10 — SVM 5-fold CV train + test
fig, ax = plt.subplots(figsize=(9, 7))
nw_tr = int((le.transform(cls_train) != y_svm_5fold).sum())
nw_te = int((le.transform(cls_test)  != y_svm_test).sum())
scatter_svm_cluster(
    ax, y_svm_5fold, y_svm_test,
    'SVM-RBF Cluster  |  5-fold CV (train) + External Test  |  ' + TITLE_SUF + '\n'
    'Train: %d wrong / 60  (CV acc=%.1f%%)     Test: %d wrong / 20  (acc=%.1f%%)' % (
        nw_tr, acc_svm_5f*100, nw_te, acc_svm_test*100),
    cm_svm_5f, cm_svm_test,
    '5-fold CV (train)', 'External Test')
fig.tight_layout()
save_fig(fig, '10_SVM_Cluster_5fold_Train.png')

# Fig 11 — SVM test-only cluster (train shown correctly positioned, edges all black)
fig, ax = plt.subplots(figsize=(9, 7))
nw_te = int((le.transform(cls_test) != y_svm_test).sum())
scatter_svm_cluster(
    ax, le.transform(cls_train), y_svm_test,
    'SVM-RBF Cluster  |  External Test  (n=20)  |  ' + TITLE_SUF + '\n'
    'Test: %d wrong / 20  |  Test accuracy = %.1f%%' % (nw_te, acc_svm_test*100),
    cm_svm_5f, cm_svm_test,
    '5-fold CV (train)', 'External Test')
fig.tight_layout()
save_fig(fig, '11_SVM_Cluster_Test.png')

# =============================================================================
# 20. SVM CONFUSION MATRICES WITH PER-CLASS STATS TABLE
# =============================================================================
plot_confusion_with_stats(
    cm_svm_5f, le.classes_,
    'SVM-RBF  5-fold CV  (train n=60)',
    '12_SVM_Confusion_5fold.png', 'Blues', acc_svm_5f)

plot_confusion_with_stats(
    cm_svm_test, le.classes_,
    'SVM-RBF  External Test  (n=20)',
    '13_SVM_Confusion_Test.png', 'Oranges', acc_svm_test)

# =============================================================================
# 21. ACCURACY COMPARISON  (3-panel)
#     P1: Overall accuracy | P2: Per-class accuracy CV | P3: Per-class accuracy Test
# =============================================================================
pls_acc_cv, pls_rec_cv = per_class_metrics(cm_pls_5f)
svm_acc_cv, svm_rec_cv = per_class_metrics(cm_svm_5f)
pls_acc_te, pls_rec_te = per_class_metrics(cm_pls_test)
svm_acc_te, svm_rec_te = per_class_metrics(cm_svm_test)
short_cls = [c.replace('SSY-', '') for c in le.classes_]

fig, axes = plt.subplots(1, 3, figsize=(20, 6))

# Panel 1: Overall accuracy
ax = axes[0]
x_ov = np.arange(2); w_ov = 0.28
groups = ['5-fold CV\n(train n=60)', 'External Test\n(n=20)']
b1 = ax.bar(x_ov - w_ov/2, [acc_pls_5f*100, acc_pls_test*100], w_ov,
            color='steelblue', edgecolor='k', lw=0.7, label='PLS-DA')
b2 = ax.bar(x_ov + w_ov/2, [acc_svm_5f*100, acc_svm_test*100], w_ov,
            color='darkorange', edgecolor='k', lw=0.7, label='SVM-RBF')
for bar in list(b1) + list(b2):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.8,
            '%.1f%%' % bar.get_height(), ha='center', va='bottom',
            fontsize=10, fontweight='bold')
ax.axhline(100, color='grey', ls=':', lw=1)
ax.set_xticks(x_ov); ax.set_xticklabels(groups, fontsize=10)
ax.set_ylabel('Accuracy (%)', fontsize=12); ax.set_ylim(0, 118)
ax.set_title('Overall Accuracy', fontsize=11, fontweight='bold')
ax.legend(fontsize=10); ax.grid(True, alpha=0.2, axis='y')

# Panel 2: Per-class accuracy (5-fold CV train)
ax = axes[1]
xc = np.arange(len(le.classes_)); wc = 0.30
b1 = ax.bar(xc - wc/2, pls_acc_cv, wc, color='steelblue',  edgecolor='k', lw=0.7, label='PLS-DA')
b2 = ax.bar(xc + wc/2, svm_acc_cv, wc, color='darkorange', edgecolor='k', lw=0.7, label='SVM-RBF')
for bar in list(b1) + list(b2):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
            '%.1f%%' % bar.get_height(), ha='center', va='bottom',
            fontsize=7.5, fontweight='bold')
ax.axhline(100, color='grey', ls=':', lw=1)
ax.set_xticks(xc); ax.set_xticklabels(short_cls, rotation=20, ha='right', fontsize=9)
ax.set_ylabel('Accuracy (%)', fontsize=12); ax.set_ylim(0, 118)
ax.set_title('Per-Class Accuracy\n5-fold CV (train)', fontsize=11, fontweight='bold')
ax.legend(fontsize=10); ax.grid(True, alpha=0.2, axis='y')

# Panel 3: Per-class accuracy (External Test)
ax = axes[2]
b1 = ax.bar(xc - wc/2, pls_acc_te, wc, color='steelblue',  edgecolor='k', lw=0.7, label='PLS-DA')
b2 = ax.bar(xc + wc/2, svm_acc_te, wc, color='darkorange', edgecolor='k', lw=0.7, label='SVM-RBF')
for bar in list(b1) + list(b2):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
            '%.1f%%' % bar.get_height(), ha='center', va='bottom',
            fontsize=7.5, fontweight='bold')
ax.axhline(100, color='grey', ls=':', lw=1)
ax.set_xticks(xc); ax.set_xticklabels(short_cls, rotation=20, ha='right', fontsize=9)
ax.set_ylabel('Accuracy (%)', fontsize=12); ax.set_ylim(0, 118)
ax.set_title('Per-Class Accuracy\nExternal Test', fontsize=11, fontweight='bold')
ax.legend(fontsize=10); ax.grid(True, alpha=0.2, axis='y')

fig.suptitle('Accuracy  |  PLS-DA vs SVM-RBF  |  ' + TITLE_SUF,
             fontsize=12, fontweight='bold')
fig.tight_layout()
save_fig(fig, '14_Accuracy_Comparison.png')

# =============================================================================
# 22. RECALL COMPARISON  (3-panel)
#     P1: Macro recall | P2: Per-class recall CV | P3: Per-class recall Test
# =============================================================================
fig, axes = plt.subplots(1, 3, figsize=(20, 6))

# Panel 1: Macro recall
ax = axes[0]
b1 = ax.bar(x_ov - w_ov/2,
            [pls_rec_cv.mean(), pls_rec_te.mean()], w_ov,
            color='steelblue', edgecolor='k', lw=0.7, label='PLS-DA')
b2 = ax.bar(x_ov + w_ov/2,
            [svm_rec_cv.mean(), svm_rec_te.mean()], w_ov,
            color='darkorange', edgecolor='k', lw=0.7, label='SVM-RBF')
for bar in list(b1) + list(b2):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.8,
            '%.1f%%' % bar.get_height(), ha='center', va='bottom',
            fontsize=10, fontweight='bold')
ax.axhline(100, color='grey', ls=':', lw=1)
ax.set_xticks(x_ov); ax.set_xticklabels(groups, fontsize=10)
ax.set_ylabel('Recall (%)', fontsize=12); ax.set_ylim(0, 118)
ax.set_title('Macro-Average Recall', fontsize=11, fontweight='bold')
ax.legend(fontsize=10); ax.grid(True, alpha=0.2, axis='y')

# Panel 2: Per-class recall (5-fold CV)
ax = axes[1]
b1 = ax.bar(xc - wc/2, pls_rec_cv, wc, color='steelblue',  edgecolor='k', lw=0.7, label='PLS-DA')
b2 = ax.bar(xc + wc/2, svm_rec_cv, wc, color='darkorange', edgecolor='k', lw=0.7, label='SVM-RBF')
for bar in list(b1) + list(b2):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1.5,
            '%.0f%%' % bar.get_height(), ha='center', va='bottom',
            fontsize=7.5, fontweight='bold')
ax.axhline(100, color='grey', ls=':', lw=1)
ax.set_xticks(xc); ax.set_xticklabels(short_cls, rotation=20, ha='right', fontsize=9)
ax.set_ylabel('Recall (%)', fontsize=12); ax.set_ylim(0, 122)
ax.set_title('Per-Class Recall\n5-fold CV (train)', fontsize=11, fontweight='bold')
ax.legend(fontsize=10); ax.grid(True, alpha=0.2, axis='y')

# Panel 3: Per-class recall (External Test)
ax = axes[2]
b1 = ax.bar(xc - wc/2, pls_rec_te, wc, color='steelblue',  edgecolor='k', lw=0.7, label='PLS-DA')
b2 = ax.bar(xc + wc/2, svm_rec_te, wc, color='darkorange', edgecolor='k', lw=0.7, label='SVM-RBF')
for bar in list(b1) + list(b2):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1.5,
            '%.0f%%' % bar.get_height(), ha='center', va='bottom',
            fontsize=7.5, fontweight='bold')
ax.axhline(100, color='grey', ls=':', lw=1)
ax.set_xticks(xc); ax.set_xticklabels(short_cls, rotation=20, ha='right', fontsize=9)
ax.set_ylabel('Recall (%)', fontsize=12); ax.set_ylim(0, 122)
ax.set_title('Per-Class Recall\nExternal Test', fontsize=11, fontweight='bold')
ax.legend(fontsize=10); ax.grid(True, alpha=0.2, axis='y')

fig.suptitle('Recall  |  PLS-DA vs SVM-RBF  |  ' + TITLE_SUF,
             fontsize=12, fontweight='bold')
fig.tight_layout()
save_fig(fig, '15_Recall_Comparison.png')

# =============================================================================
# 23. MEAN SPECTRA  ± 1 SD
# =============================================================================
X_all_sm = savgol_filter(X_crop, window_length=SG_WINDOW, polyorder=SG_POLY, axis=1)
fig, ax = plt.subplots(figsize=(10, 4.5))
for cls in le.classes_:
    idx = classes == cls
    m   = X_all_sm[idx].mean(axis=0)
    s   = X_all_sm[idx].std(axis=0)
    ax.plot(wavenumbers, m, color=PALETTE[cls], lw=1.5, label=cls)
    ax.fill_between(wavenumbers, m-s, m+s, color=PALETTE[cls], alpha=0.12)
ax.set_xlim(WAVENUMBER_MIN, WAVENUMBER_MAX)
ax.set_xlabel('Wavenumber (cm⁻¹)', fontsize=12)
ax.set_ylabel('Intensity (a.u.)', fontsize=12)
ax.set_title('Mean Raman Spectra ± 1 SD  (all 80 spectra)  |  %d–%d cm⁻¹' % (
    WAVENUMBER_MIN, WAVENUMBER_MAX), fontsize=11, fontweight='bold')
ax.legend(fontsize=9); ax.grid(True, alpha=0.25)
fig.tight_layout()
save_fig(fig, '16_Mean_Spectra.png')

# =============================================================================
# 24. SAVE ALL RESULTS TO EXCEL
# =============================================================================
out_xlsx = os.path.join(OUTPUT_DIR, 'Results.xlsx')

with pd.ExcelWriter(out_xlsx, engine='openpyxl') as writer:

    # Summary
    rows = [
        ('Input file',                          FILE_NAME),
        ('Spectral window (cm-1)',               '%d-%d' % (WAVENUMBER_MIN, WAVENUMBER_MAX)),
        ('Spectral points in window',            X_crop.shape[1]),
        ('SG window / poly',                     '%d/%d' % (SG_WINDOW, SG_POLY)),
        ('VIP > 1 (initial model)',              n_vip_gt1),
        ('VIP top-N selected',                   top_n),
        ('N latent variables (PLS-DA)',          N_COMPONENTS),
        ('Total samples',                        len(y)),
        ('Training set size',                    len(y_train)),
        ('Test set size',                        len(y_test)),
        ('', ''),
        ('--- PLS-DA ---',                       ''),
        ('PLS-DA Train re-sub acc (%)',          '%.2f' % (train_acc_resub * 100)),
        ('PLS-DA 5-fold CV acc (%)',             '%.2f' % (acc_pls_5f  * 100)),
        ('PLS-DA External test acc (%)',         '%.2f' % (acc_pls_test * 100)),
        ('PLS-DA Permutation p-value',           '%.4f' % p_val),
        ('PLS-DA Permutation result',
         'Significant (p<0.05)' if p_val < 0.05 else 'NOT significant'),
        ('', ''),
        ('--- SVM-RBF ---',                      ''),
        ('SVM-RBF 5-fold CV acc (%)',            '%.2f' % (acc_svm_5f  * 100)),
        ('SVM-RBF External test acc (%)',        '%.2f' % (acc_svm_test * 100)),
        ('SVM kernel', 'RBF'), ('SVM C', 10), ('SVM gamma', 'scale'),
    ]
    for i, r2 in enumerate(r2x_cum, 1):
        rows.append(('PLS-DA R2X cumulative LV%d (%%)' % i, '%.4f' % (r2 * 100)))
    pd.DataFrame(rows, columns=['Metric', 'Value']).to_excel(
        writer, sheet_name='Summary', index=False)

    # Per-class Accuracy & Recall
    cls_rows = []
    for i, cls in enumerate(le.classes_):
        cls_rows.append({
            'Class':                  cls,
            'N_Train':                int(cm_pls_5f[i, :].sum()),
            'N_Test':                 int(cm_pls_test[i, :].sum()),
            'PLSDA_CV_Accuracy_%':    '%.2f' % pls_acc_cv[i],
            'PLSDA_CV_Recall_%':      '%.2f' % pls_rec_cv[i],
            'PLSDA_Test_Accuracy_%':  '%.2f' % pls_acc_te[i],
            'PLSDA_Test_Recall_%':    '%.2f' % pls_rec_te[i],
            'SVM_CV_Accuracy_%':      '%.2f' % svm_acc_cv[i],
            'SVM_CV_Recall_%':        '%.2f' % svm_rec_cv[i],
            'SVM_Test_Accuracy_%':    '%.2f' % svm_acc_te[i],
            'SVM_Test_Recall_%':      '%.2f' % svm_rec_te[i],
        })
    pd.DataFrame(cls_rows).to_excel(
        writer, sheet_name='Per_Class_Metrics', index=False)

    # Train/Test assignment
    assign_rows = []
    for i, (name, cls) in enumerate(zip(col_names, classes)):
        assign_rows.append({
            'Original_Label': name,
            'True_Class':     cls,
            'Split':          'TRAIN' if i in train_list else 'TEST',
            'Session':        'Luglio-2025' if 'lugl' in name.lower() else 'Marzo-2026',
        })
    pd.DataFrame(assign_rows).to_excel(
        writer, sheet_name='Train_Test_Assignment', index=False)

    # CV predictions (train)
    pd.DataFrame({
        'Label':               names_train,
        'True_Class':          cls_train,
        'PLSDA_5fold_Pred':    le.inverse_transform(y_pls_5fold),
        'PLSDA_5fold_Correct': (cls_train == le.inverse_transform(y_pls_5fold)),
        'SVM_5fold_Pred':      le.inverse_transform(y_svm_5fold),
        'SVM_5fold_Correct':   (cls_train == le.inverse_transform(y_svm_5fold)),
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
        ('CM_PLSDA_5fold', cm_pls_5f),
        ('CM_PLSDA_Test',  cm_pls_test),
        ('CM_SVM_5fold',   cm_svm_5f),
        ('CM_SVM_Test',    cm_svm_test),
    ]:
        pd.DataFrame(cm,
                     index=pd.Index(le.classes_, name='True\\Predicted'),
                     columns=le.classes_).to_excel(writer, sheet_name=sheet)

    # VIP scores
    pd.DataFrame({
        'Wavenumber_cm1':          wavenumbers,
        'VIP_Score':               vip_all,
        'VIP_gt_1':                vip_all > 1,
        'Selected_Top%d' % top_n:  sel_mask,
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
print('\n' + '=' * 72)
print('  FINAL RESULTS  |  Train/Test Split  |  5-fold CV + External Test')
print('=' * 72)
print('  Spectral window  : %d-%d cm-1   |   %d points' % (
    WAVENUMBER_MIN, WAVENUMBER_MAX, X_crop.shape[1]))
print('  VIP > 1          : %d / %d   ->   top-%d selected' % (
    n_vip_gt1, len(vip_all), top_n))
print()
print('  %-10s  %20s  %18s' % ('MODEL', '5-fold CV (train)', 'External Test'))
print('  ' + '-' * 52)
print('  %-10s  %19.1f%%  %17.1f%%' % ('PLS-DA',  acc_pls_5f*100, acc_pls_test*100))
print('  %-10s  %19.1f%%  %17.1f%%' % ('SVM-RBF', acc_svm_5f*100, acc_svm_test*100))
print()
print('  PLS-DA permutation p-value : %.4f  [%s]' % (
    p_val, 'SIGNIFICANT' if p_val < 0.05 else 'NOT significant'))
print()
print('  Per-class Accuracy & Recall — 5-fold CV (train):')
print('  %-25s  %9s  %7s  |  %9s  %7s' % (
    'Class', 'PLS Acc', 'PLS Rec', 'SVM Acc', 'SVM Rec'))
print('  ' + '-' * 66)
for i, cls in enumerate(le.classes_):
    print('  %-25s  %8.1f%%  %6.1f%%  |  %8.1f%%  %6.1f%%' % (
        cls, pls_acc_cv[i], pls_rec_cv[i], svm_acc_cv[i], svm_rec_cv[i]))
print()
print('  Per-class Accuracy & Recall — External Test:')
print('  %-25s  %9s  %7s  |  %9s  %7s' % (
    'Class', 'PLS Acc', 'PLS Rec', 'SVM Acc', 'SVM Rec'))
print('  ' + '-' * 66)
for i, cls in enumerate(le.classes_):
    print('  %-25s  %8.1f%%  %6.1f%%  |  %8.1f%%  %6.1f%%' % (
        cls, pls_acc_te[i], pls_rec_te[i], svm_acc_te[i], svm_rec_te[i]))
print()
print('  R2X cumulative (PLS-DA):')
for i, r2 in enumerate(r2x_cum, 1):
    print('    LV%d: %.2f%%' % (i, r2*100))
print()
print('  All outputs saved to : %s' % OUTPUT_DIR)
print('=' * 72)
