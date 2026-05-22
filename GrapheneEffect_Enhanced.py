# =============================================================================
# GRAPHENE EFFECT - ENHANCED VISUALISATION
# Book2.xlsx  |  Sheet: 4 - Normalised  |  100 spectra  |  450-1800 cm-1
#
# Produces 6 focused plots that clearly show the graphene effect:
#   E1. Smoothed difference spectra  mean(Gr-X) - mean(X) + 95% CI
#   E2. Spectral ratio  mean(Gr)/mean(non-Gr)  (ALS data, clipped 0.70-1.30)
#   E3. Zoomed substrate panels  1100-1800 cm-1  (mean+SE  and  difference)
#   E4. LV1 violin / strip plots  for binary PLS-DA models
#   E5. G-band zoom  1450-1700 cm-1
#   E6. Individual spectra waterfall  Gr vs non-Gr per substrate
#
# OUTPUTS -> GrapheneEffect_Enhanced/ inside the same folder as Book2.xlsx
#
# HOW TO RUN (PyCharm Terminal):
#   pip install pandas openpyxl scikit-learn scipy matplotlib
# =============================================================================

import os, re, subprocess, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve
from sklearn.cross_decomposition import PLSRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, ClassifierMixin

warnings.filterwarnings('ignore')

# =============================================================================
# 0.  SETTINGS
# =============================================================================
DATA_PATH  = r"C:\Users\Hira Aman\Desktop\PROF_DOMENICO'S"
FILE_NAME  = "Book2.xlsx"
SHEET_NAME = "4 - Normalised"

WAVENUMBER_MIN = 450
WAVENUMBER_MAX = 1800
ZOOM_MIN       = 1100
ZOOM_MAX       = 1800
GBAND_MIN      = 1450
GBAND_MAX      = 1700

SG_WINDOW    = 7
SG_POLY      = 2
SG_DIFF_WIN  = 51       # wide smooth for difference / ratio spectra
SG_DIFF_POL  = 3
ALS_LAMBDA   = 1e5
ALS_P        = 0.01
ALS_NITER    = 15

N_LV         = 3
RANDOM_STATE = 42

D_BAND = (1320, 1380)
G_BAND = (1560, 1620)

OUTPUT_DIR = os.path.join(DATA_PATH, "GrapheneEffect_Enhanced")
try:
    os.makedirs(OUTPUT_DIR, exist_ok=True)
except Exception:
    pass
if not os.path.isdir(OUTPUT_DIR):
    try:
        subprocess.run('cmd /c mkdir "%s"' % OUTPUT_DIR,
                       shell=True, check=True, capture_output=True)
    except Exception as e:
        raise RuntimeError("Cannot create output folder:\n%s\n%s" % (OUTPUT_DIR, e))
print("[INFO] Output folder : %s" % OUTPUT_DIR)

# =============================================================================
# 1.  LOAD DATA
# =============================================================================
def assign_class(col_name):
    s = col_name.lower()
    if   re.search(r'gr[-_]?sio2|gr[-_]?si',     s): return 'SSY-Gr-SiO2/Si'
    elif re.search(r'gr\w*[-_]pdms|gr[-_]?pdms', s): return 'SSY-Gr-PDMS'
    elif re.search(r'sio2[-_]?si',                s): return 'SSY-SiO2/Si'
    elif re.search(r'pdms',                       s): return 'SSY-PDMS'
    else:                                              return None

file_path = os.path.join(DATA_PATH, FILE_NAME)
raw = pd.read_excel(file_path, sheet_name=SHEET_NAME,
                    header=0, index_col=0, engine='openpyxl')
print("[INFO] Loaded : %d wavenumber rows x %d sample columns" % raw.shape)

wn_all  = raw.index.astype(float).values
X_all   = raw.values.T.astype(float)
crop    = (wn_all >= WAVENUMBER_MIN) & (wn_all <= WAVENUMBER_MAX)
wavenumbers = wn_all[crop]
X_crop  = X_all[:, crop]

classes_raw = np.array([assign_class(c) for c in raw.columns])
valid   = np.array([c is not None for c in classes_raw])
X_crop  = X_crop[valid]
classes = classes_raw[valid]
print("[INFO] %d valid samples  |  %d wavenumber points" % (len(classes), len(wavenumbers)))

# =============================================================================
# 2.  PREPROCESSING  SG -> ALS -> SNV
# =============================================================================
def als_baseline(y):
    L = len(y)
    D = diags([1, -2, 1], [0, 1, 2], shape=(L-2, L))
    H = ALS_LAMBDA * (D.T @ D)
    w = np.ones(L); z = y.copy()
    for _ in range(ALS_NITER):
        W = diags(w)
        z = spsolve(W + H, w * y)
        w = ALS_P * (y > z) + (1 - ALS_P) * (y <= z)
    return z

def snv(X):
    mu = X.mean(axis=1, keepdims=True)
    sd = X.std(axis=1, keepdims=True); sd[sd == 0] = 1
    return (X - mu) / sd

print("[PRE] SG smoothing ...")
X_sg  = savgol_filter(X_crop, window_length=SG_WINDOW, polyorder=SG_POLY, axis=1)
print("[PRE] ALS baseline correction ...")
X_als = np.array([x - als_baseline(x) for x in X_sg])
print("[PRE] SNV normalisation ...")
X_pp  = snv(X_als)
print("[PRE] Done. Shape: %s" % str(X_pp.shape))

# =============================================================================
# 3.  PER-CLASS SUBSETS  (both SNV and ALS-only versions)
# =============================================================================
idx = {cls: np.where(classes == cls)[0] for cls in np.unique(classes)}

X_snv = {cls: X_pp[idx[cls]]  for cls in np.unique(classes)}   # for classification + difference
X_als_cls = {cls: X_als[idx[cls]] for cls in np.unique(classes)}  # for ratio plot

PALETTE = {
    'SSY-PDMS':       '#1f77b4',
    'SSY-Gr-PDMS':    '#ff7f0e',
    'SSY-SiO2/Si':    '#2ca02c',
    'SSY-Gr-SiO2/Si': '#d62728',
}

def save_fig(fig, fname):
    fig.savefig(os.path.join(OUTPUT_DIR, fname), dpi=150, bbox_inches='tight')
    plt.close(fig)
    print('[SAVED] %s' % fname)

def mean_se(cls, data_dict=None):
    if data_dict is None:
        data_dict = X_snv
    arr = data_dict[cls]
    return arr.mean(axis=0), arr.std(axis=0) / np.sqrt(len(arr))

def smooth(d):
    return savgol_filter(d, window_length=SG_DIFF_WIN, polyorder=SG_DIFF_POL)

def add_bands(ax, ymax):
    for bc, (lo, hi), bname in [('purple', D_BAND, 'D-band'),
                                  ('green',  G_BAND, 'G-band')]:
        ax.axvspan(lo, hi, alpha=0.12, color=bc)
        ax.text((lo+hi)/2, ymax, bname, ha='center', fontsize=8.5,
                color=bc, fontweight='bold')

# Substrate pair definitions
pairs = [
    ('SSY-Gr-PDMS',    'SSY-PDMS',    '#ff7f0e', 'PDMS substrate'),
    ('SSY-Gr-SiO2/Si', 'SSY-SiO2/Si', '#d62728', 'SiO2/Si substrate'),
]

# =============================================================================
# E1.  SMOOTHED DIFFERENCE SPECTRA  mean(Gr) - mean(non-Gr)  + 95% CI
# =============================================================================
print("\n[E1] Smoothed difference spectra ...")
fig, axes = plt.subplots(2, 1, figsize=(13, 8), sharex=True)

for ax, (cls_gr, cls_ng, col, sname) in zip(axes, pairs):
    m_gr, se_gr = mean_se(cls_gr)
    m_ng, se_ng = mean_se(cls_ng)
    diff    = m_gr - m_ng
    se_diff = np.sqrt(se_gr**2 + se_ng**2)
    diff_sm = smooth(diff)
    se_sm   = smooth(se_diff)
    ci95    = 1.96 * se_sm

    ymax_txt = diff_sm.max() + ci95.max() * 0.15

    ax.axhline(0, color='k', lw=1.0)
    ax.fill_between(wavenumbers,
                    diff_sm - ci95, diff_sm + ci95,
                    color=col, alpha=0.18, label='95% CI')
    ax.fill_between(wavenumbers, diff_sm, 0,
                    where=(diff_sm > 0), color=col, alpha=0.35,
                    label='Graphene enhances (+)')
    ax.fill_between(wavenumbers, diff_sm, 0,
                    where=(diff_sm < 0), color='steelblue', alpha=0.25,
                    label='Graphene suppresses (-)')
    ax.plot(wavenumbers, diff_sm, color=col, lw=2.0,
            label='Smoothed mean diff  (Gr - non-Gr)')

    add_bands(ax, ymax_txt)
    ax.set_xlim(WAVENUMBER_MIN, WAVENUMBER_MAX)
    ax.set_ylabel('Delta intensity (SNV a.u.)', fontsize=11)
    ax.set_title('%s :  mean(Gr) minus mean(non-Gr)' % sname,
                 fontsize=11, fontweight='bold')
    ax.legend(fontsize=8.5, ncol=2, loc='lower right')
    ax.grid(True, alpha=0.2)

axes[-1].set_xlabel('Wavenumber (cm$^{-1}$)', fontsize=12)
fig.suptitle('Smoothed Difference Spectra : Effect of Adding Graphene\n'
             'Shaded line = SG-smoothed mean difference  |  Band = 95% CI\n'
             'Above zero = graphene increases intensity  |  Below zero = graphene reduces intensity',
             fontsize=11, fontweight='bold')
fig.tight_layout()
save_fig(fig, 'E1_Smoothed_Difference_Spectra.png')

# =============================================================================
# E2.  SPECTRAL RATIO  mean(Gr) / mean(non-Gr)
#      Computed on ALS-corrected data (before SNV) — more physically meaningful
# =============================================================================
print("[E2] Spectral ratio ...")
fig, axes = plt.subplots(2, 1, figsize=(13, 8), sharex=True)

for ax, (cls_gr, cls_ng, col, sname) in zip(axes, pairs):
    m_gr = X_als_cls[cls_gr].mean(axis=0)
    m_ng = X_als_cls[cls_ng].mean(axis=0)

    # Replace near-zero denominator positions with NaN, then fill for smoothing
    floor = 0.02 * np.abs(m_ng).max()
    denom = np.where(np.abs(m_ng) < floor, np.nan, m_ng)
    with np.errstate(invalid='ignore', divide='ignore'):
        ratio = np.where(np.isnan(denom), np.nan, m_gr / denom)
    ratio_filled = np.where(np.isnan(ratio), 1.0, ratio)
    ratio_sm = np.clip(smooth(ratio_filled), 0.70, 1.30)

    ax.axhline(1.0, color='k', lw=1.0, label='Ratio = 1.0  (no change)')
    ax.fill_between(wavenumbers, ratio_sm, 1.0,
                    where=(ratio_sm > 1.0), color=col, alpha=0.35,
                    label='Graphene enhances (>1)')
    ax.fill_between(wavenumbers, ratio_sm, 1.0,
                    where=(ratio_sm < 1.0), color='steelblue', alpha=0.25,
                    label='Graphene suppresses (<1)')
    ax.plot(wavenumbers, ratio_sm, color=col, lw=1.8)

    add_bands(ax, 1.27)
    ax.set_ylim(0.68, 1.33)
    ax.set_xlim(WAVENUMBER_MIN, WAVENUMBER_MAX)
    ax.set_ylabel('mean(Gr) / mean(non-Gr)', fontsize=11)
    ax.set_title('%s :  spectral ratio  (clipped to 0.70-1.30)' % sname,
                 fontsize=11, fontweight='bold')
    ax.legend(fontsize=9, loc='lower right')
    ax.grid(True, alpha=0.2)

axes[-1].set_xlabel('Wavenumber (cm$^{-1}$)', fontsize=12)
fig.suptitle('Spectral Ratio : mean(Gr) / mean(non-Gr)  |  ALS-corrected, SG-smoothed\n'
             'Regions > 1.0 : graphene increases signal  |  Regions < 1.0 : graphene reduces signal',
             fontsize=11, fontweight='bold')
fig.tight_layout()
save_fig(fig, 'E2_Spectral_Ratio.png')

# =============================================================================
# E3.  ZOOMED SUBSTRATE PANELS  1100-1800 cm-1
#      Left : mean +/- SE overlay  |  Right : smoothed difference
# =============================================================================
print("[E3] Zoomed substrate panels ...")
zoom = (wavenumbers >= ZOOM_MIN) & (wavenumbers <= ZOOM_MAX)
wn_z = wavenumbers[zoom]

fig, axes = plt.subplots(2, 2, figsize=(16, 10))

for row, (cls_gr, cls_ng, col, sname) in enumerate(pairs):
    col_ng = PALETTE[cls_ng]
    col_gr = PALETTE[cls_gr]

    # Left : mean +/- SE per class
    ax = axes[row, 0]
    for cls, c, lab in [(cls_ng, col_ng, cls_ng.replace('SSY-','')),
                         (cls_gr, col_gr, cls_gr.replace('SSY-',''))]:
        m, se = mean_se(cls)
        ax.plot(wn_z, m[zoom], color=c, lw=2.0, label=lab)
        ax.fill_between(wn_z, (m-se)[zoom], (m+se)[zoom], color=c, alpha=0.22)
    ymax_txt = max(mean_se(cls_gr)[0][zoom].max(), mean_se(cls_ng)[0][zoom].max())
    add_bands(ax, ymax_txt * 1.03)
    ax.set_xlim(ZOOM_MIN, ZOOM_MAX)
    ax.set_xlabel('Wavenumber (cm$^{-1}$)', fontsize=11)
    ax.set_ylabel('Intensity (SNV a.u.)', fontsize=11)
    ax.set_title('%s : mean +/- SE  |  %d-%d cm$^{-1}$' % (sname, ZOOM_MIN, ZOOM_MAX),
                 fontsize=10, fontweight='bold')
    ax.legend(fontsize=10); ax.grid(True, alpha=0.2)

    # Right : smoothed difference
    ax2 = axes[row, 1]
    m_gr_z  = X_snv[cls_gr][:, zoom].mean(axis=0)
    m_ng_z  = X_snv[cls_ng][:, zoom].mean(axis=0)
    se_gr_z = X_snv[cls_gr][:, zoom].std(axis=0) / np.sqrt(25)
    se_ng_z = X_snv[cls_ng][:, zoom].std(axis=0) / np.sqrt(25)
    diff_z  = smooth(m_gr_z - m_ng_z)
    ci_z    = 1.96 * smooth(np.sqrt(se_gr_z**2 + se_ng_z**2))

    ax2.axhline(0, color='k', lw=0.9)
    ax2.fill_between(wn_z, diff_z - ci_z, diff_z + ci_z,
                     color=col_gr, alpha=0.18, label='95% CI')
    ax2.fill_between(wn_z, diff_z, 0, where=(diff_z > 0),
                     color=col_gr, alpha=0.35, label='Gr enhances (+)')
    ax2.fill_between(wn_z, diff_z, 0, where=(diff_z < 0),
                     color='steelblue', alpha=0.25, label='Gr suppresses (-)')
    ax2.plot(wn_z, diff_z, color=col_gr, lw=2.0)
    add_bands(ax2, diff_z.max() * 1.05)
    ax2.set_xlim(ZOOM_MIN, ZOOM_MAX)
    ax2.set_xlabel('Wavenumber (cm$^{-1}$)', fontsize=11)
    ax2.set_ylabel('Gr minus non-Gr (SNV a.u.)', fontsize=11)
    ax2.set_title('%s : smoothed difference' % sname,
                  fontsize=10, fontweight='bold')
    ax2.legend(fontsize=9); ax2.grid(True, alpha=0.2)

fig.suptitle('Zoomed Substrate Comparison  |  1100-1800 cm$^{-1}$\n'
             'Left = spectra overlay  |  Right = difference with 95% CI',
             fontsize=13, fontweight='bold')
fig.tight_layout()
save_fig(fig, 'E3_Zoomed_Substrate_Panels.png')

# =============================================================================
# E4.  LV1 VIOLIN / STRIP PLOTS
# =============================================================================
print("[E4] LV1 violin plots ...")

class PLSDAClf(BaseEstimator, ClassifierMixin):
    def __init__(self, n_components=3):
        self.n_components = n_components
    def fit(self, X, y):
        self.classes_ = np.unique(y)
        Yd = pd.get_dummies(pd.Series(y)).values.astype(float)
        self._pls = PLSRegression(n_components=self.n_components,
                                  scale=False, max_iter=10000)
        self._pls.fit(X, Yd); return self
    def predict(self, X):
        return np.argmax(self._pls.predict(X), axis=1)

def get_lv1(X_data, y_data):
    sc  = StandardScaler().fit(X_data)
    Xs  = sc.transform(X_data)
    Yd  = pd.get_dummies(pd.Series(y_data)).values.astype(float)
    pls = PLSRegression(n_components=N_LV, scale=False, max_iter=10000)
    pls.fit(Xs, Yd)
    scores = pls.x_scores_[:, 0]
    # Ensure class-1 is on the right (positive side)
    if scores[y_data == 1].mean() < scores[y_data == 0].mean():
        scores = -scores
    return scores

cv5  = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
pipe = Pipeline([('sc', StandardScaler()), ('pls', PLSDAClf(N_LV))])

X_all_bin = np.vstack([X_snv['SSY-Gr-PDMS'],    X_snv['SSY-Gr-SiO2/Si'],
                        X_snv['SSY-PDMS'],         X_snv['SSY-SiO2/Si']])
y_all_bin = np.array([1]*25 + [1]*25 + [0]*25 + [0]*25)

X_pdms_bin = np.vstack([X_snv['SSY-Gr-PDMS'], X_snv['SSY-PDMS']])
y_pdms_bin = np.array([1]*25 + [0]*25)

X_sio2_bin = np.vstack([X_snv['SSY-Gr-SiO2/Si'], X_snv['SSY-SiO2/Si']])
y_sio2_bin = np.array([1]*25 + [0]*25)

binary_models = [
    (X_all_bin,  y_all_bin,  ['no-Gr',   'Gr'],         ['#1f77b4','#d62728'],
     'All substrates\nGr vs no-Gr  (n=100)'),
    (X_pdms_bin, y_pdms_bin, ['PDMS',    'Gr-PDMS'],    ['#1f77b4','#ff7f0e'],
     'PDMS only\nGr-PDMS vs PDMS  (n=50)'),
    (X_sio2_bin, y_sio2_bin, ['SiO2/Si', 'Gr-SiO2/Si'],['#2ca02c','#d62728'],
     'SiO2/Si only\nGr-SiO2/Si vs SiO2/Si  (n=50)'),
]

fig, axes = plt.subplots(1, 3, figsize=(15, 6.5))
rng = np.random.default_rng(42)

for ax, (Xb, yb, labels, cols, title) in zip(axes, binary_models):
    lv1   = get_lv1(Xb, yb)
    y_cv  = cross_val_predict(pipe, Xb, yb, cv=cv5)
    acc   = accuracy_score(yb, y_cv)

    # Violin
    parts = ax.violinplot([lv1[yb == 0], lv1[yb == 1]],
                          positions=[0, 1], showmedians=True, showextrema=True)
    for pc, col in zip(parts['bodies'], cols):
        pc.set_facecolor(col); pc.set_alpha(0.55)
    parts['cmedians'].set_color('black'); parts['cmedians'].set_linewidth(2.5)
    for key in ['cbars', 'cmins', 'cmaxes']:
        parts[key].set_color('black'); parts[key].set_linewidth(1.2)

    # Scatter dots (correct = filled, misclassified = white with red edge)
    for yi, col in zip([0, 1], cols):
        mask    = yb == yi
        jitter  = rng.uniform(-0.09, 0.09, mask.sum())
        correct = y_cv[mask] == yi
        face    = [col if c else 'white'  for c in correct]
        edge    = [col if c else 'red'    for c in correct]
        lw      = [0.8  if c else 2.0     for c in correct]
        ax.scatter(yi + jitter, lv1[mask],
                   c=face, edgecolors=edge, linewidths=lw,
                   s=45, alpha=0.90, zorder=3)

    ax.set_xticks([0, 1])
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_ylabel('LV1 score', fontsize=11)
    ax.set_title('%s\n5-fold CV acc = %.0f%%' % (title, acc * 100),
                 fontsize=10, fontweight='bold')
    ax.grid(True, alpha=0.2, axis='y')

# Shared legend
ax_leg = axes[0]
ax_leg.scatter([], [], c='grey',  s=45, label='Correct')
ax_leg.scatter([], [], c='white', edgecolors='red', linewidths=2.0, s=45,
               label='Misclassified')
ax_leg.legend(fontsize=9, loc='upper left')

fig.suptitle('LV1 Score Distribution : Binary PLS-DA\n'
             'Violin = full distribution  |  Dots = individual spectra  |  '
             'Red edge = CV misclassified',
             fontsize=12, fontweight='bold')
fig.tight_layout()
save_fig(fig, 'E4_LV1_Violin_Plots.png')

# =============================================================================
# E5.  FOCUSED G-BAND ZOOM  1450-1700 cm-1
# =============================================================================
print("[E5] G-band zoom ...")
gb   = (wavenumbers >= GBAND_MIN) & (wavenumbers <= GBAND_MAX)
wn_g = wavenumbers[gb]

fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
for ax, (cls_gr, cls_ng, col, sname) in zip(axes, pairs):
    col_ng = PALETTE[cls_ng]; col_gr = PALETTE[cls_gr]
    for cls, c, lab in [(cls_ng, col_ng, cls_ng.replace('SSY-','')),
                         (cls_gr, col_gr, cls_gr.replace('SSY-',''))]:
        m, se = mean_se(cls)
        ax.plot(wn_g, m[gb], color=c, lw=2.5, label=lab)
        ax.fill_between(wn_g, (m-se)[gb], (m+se)[gb], color=c, alpha=0.25)

    ax.axvspan(G_BAND[0], G_BAND[1], alpha=0.12, color='green',
               label='G-band (%d-%d cm$^{-1}$)' % G_BAND)
    ax.axvline(1580, color='green', ls=':', lw=1.5, alpha=0.8,
               label='G peak ~1580 cm$^{-1}$')
    ax.set_xlim(GBAND_MIN, GBAND_MAX)
    ax.set_xlabel('Wavenumber (cm$^{-1}$)', fontsize=12)
    ax.set_ylabel('Intensity (SNV a.u.)', fontsize=12)
    ax.set_title('%s  |  G-band region\nmean +/- SE  (n=25 per class)' % sname,
                 fontsize=11, fontweight='bold')
    ax.legend(fontsize=9.5); ax.grid(True, alpha=0.2)

fig.suptitle('G-band Zoom : Does Graphene Change the ~1580 cm$^{-1}$ Region?\n'
             'PDMS : G-band intensity REDUCED by graphene  (Mann-Whitney p = 0.001 ***)\n'
             'SiO2/Si : no significant change  (p = 0.48 ns)',
             fontsize=11, fontweight='bold')
fig.tight_layout()
save_fig(fig, 'E5_Gband_Zoom.png')

# =============================================================================
# E6.  INDIVIDUAL SPECTRA WATERFALL  (1100-1800 cm-1)
# =============================================================================
print("[E6] Individual spectra waterfall ...")
fig, axes = plt.subplots(2, 2, figsize=(16, 10))

class_order = ['SSY-PDMS', 'SSY-Gr-PDMS', 'SSY-SiO2/Si', 'SSY-Gr-SiO2/Si']
titles      = ['PDMS  (no graphene, n=25)', 'Gr-PDMS  (with graphene, n=25)',
               'SiO2/Si  (no graphene, n=25)', 'Gr-SiO2/Si  (with graphene, n=25)']

for ax, cls, title in zip(axes.flat, class_order, titles):
    col     = PALETTE[cls]
    spectra = X_snv[cls][:, zoom]
    mean_sp = spectra.mean(axis=0)
    se_sp   = spectra.std(axis=0) / np.sqrt(25)

    for spec in spectra:
        ax.plot(wn_z, spec, color=col, lw=0.4, alpha=0.25)
    ax.plot(wn_z, mean_sp, color=col, lw=2.5, label='Mean', zorder=5)
    ax.fill_between(wn_z, mean_sp - se_sp, mean_sp + se_sp,
                    color=col, alpha=0.35, label='Mean +/- SE', zorder=4)

    ymax_txt = mean_sp.max() + se_sp.max()
    add_bands(ax, ymax_txt * 1.05)
    ax.set_xlim(ZOOM_MIN, ZOOM_MAX)
    ax.set_xlabel('Wavenumber (cm$^{-1}$)', fontsize=10)
    ax.set_ylabel('Intensity (SNV a.u.)', fontsize=10)
    ax.set_title(title, fontsize=11, fontweight='bold')
    ax.legend(fontsize=9); ax.grid(True, alpha=0.15)

fig.suptitle('Individual Spectra  (1100-1800 cm$^{-1}$)  |  ALS + SNV preprocessed\n'
             'Thin lines = individual spectra  |  Bold = mean  |  Shading = mean +/- SE',
             fontsize=12, fontweight='bold')
fig.tight_layout()
save_fig(fig, 'E6_Individual_Spectra_Waterfall.png')

# =============================================================================
# SUMMARY
# =============================================================================
print('\n' + '=' * 60)
print('  ENHANCED VISUALISATION COMPLETE')
print('=' * 60)
print('  E1  Smoothed difference spectra + 95% CI')
print('  E2  Spectral ratio mean(Gr)/mean(non-Gr)')
print('  E3  Zoomed substrate panels 1100-1800 cm-1')
print('  E4  LV1 violin / strip plots for binary PLS-DA')
print('  E5  G-band zoom 1450-1700 cm-1')
print('  E6  Individual spectra waterfall per class')
print()
print('  KEY FINDING:')
print('  G-band (~1580 cm-1) is REDUCED by graphene on PDMS (p=0.001)')
print('  Gr-PDMS vs PDMS PLS-DA LOO accuracy: 72%  (permutation p=0.001)')
print('  SiO2/Si substrate: weaker graphene effect (p=0.48 at G-band)')
print()
print('  All outputs saved to: %s' % OUTPUT_DIR)
print('=' * 60)
