# =============================================================================
# ACACIA EVs vs NANOVESICLES (NVs) on Ag/SiNWs  |  ALL FIGURES, TABLES AND MODEL
# DETAILS FOR THE PAPER  |  one script, original data only
#
# INPUT (nothing else is used)
#   CODE 005\EV Acacia\EV_Acacia_Raman_Data.xlsx              120 SERS spectra
#   CODE 005\Nano vesicles Acacia\NV_Acacia_Raman_Data.xlsx   120 SERS spectra
#       (6 sheets m1-m6 = 6 measurement sites, 20 spectra per sheet)
#   CODE 005\Single spectra\  (for Figure 1 only; copy these 4 .txt files there)
#       S2-EV_Acacia-dropCaF2_532nm_1800gr_10_50xLF_BC200_1s_4a.txt
#       S2-EV_ACACIA-AgSiNW_532nm_1800gr_BC200_50XLF_1s_4a_10%.txt
#       s1-EV_nano_Acacia-dropCaF2_532nm_1800gr_10_50xLF_BC200_1s_4a.txt
#       s4a-EV_nano_ACACIA-20ng-AgSiNW_532nm_1800gr_BC200_50XLF_1s_4a_10%.txt
#     (if the folder is missing, Figure 1 is skipped and everything else runs)
#
# OUTPUT (CODE 005\PAPER, figures as PNG + TIFF, 600 dpi)
#   Fig1a_EV_CaF2_vs_AgSiNW, Fig1b_NV_CaF2_vs_AgSiNW   raw single spectra
#   Fig2a_MeanSD_Fingerprint, Fig2b_MeanSD_CH          mean +/- SD, band labels
#   Fig3_PCA_LDA                PCA scores + LDA canonical variable (a-d)
#   Fig4_Loadings_Difference    PC1/PC2 loadings (a, b), NV - EV difference (c, d)
#   Fig5_ConfusionMatrices      cross-validated, PCA-LDA and PLS-DA (a-d)
#   FigS1_Permutation           exact label-shuffling test (a-d)
#   Table1_Model_performance.csv     accuracy, sensitivity, specificity, p
#   TableS1_Band_statistics.csv      17 bands, Welch on site means, Bonferroni
#   TableS2_Fold_results.csv         every cross-validation fold
#   TableS3_Peak_positions.csv       labelled peaks of Fig 1 and Fig 2
#   TableS4_Peak_shifts.csv          EV/NV peak positions per site, shifts
#   TableS5_Main_bands.csv           main loading and difference bands
#   Scores_fingerprint.csv, Scores_CH.csv   PC1-PC5 and LD1 of every spectrum
#   Model_details.txt                every setting of every model + results
#   RESULTS_for_manuscript.txt       the results in words, numbers filled in
#
# HOW TO RUN (PyCharm)
#   1. pip install numpy pandas scipy matplotlib openpyxl scikit-learn
#   2. check BASE_DIR below, press Run. Takes a few minutes (the permutation
#      test repeats the whole cross-validation 924 times per region; set
#      N_PERM_MODE = 'skip' to leave it out).
# =============================================================================

import sys
import csv
from pathlib import Path
from itertools import combinations

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from scipy import sparse
from scipy.sparse.linalg import spsolve
from scipy.signal import savgol_filter, find_peaks
from scipy.stats import ttest_ind
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.cross_decomposition import PLSRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import confusion_matrix
import sklearn

# ----------------------------------------------------------------------------
# SETTINGS
# ----------------------------------------------------------------------------
BASE_DIR = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(r'C:\Users\Hira Aman\Downloads\CODE 005')
EV_FILE = BASE_DIR / 'EV Acacia' / 'EV_Acacia_Raman_Data.xlsx'
NV_FILE = BASE_DIR / 'Nano vesicles Acacia' / 'NV_Acacia_Raman_Data.xlsx'
SINGLE_DIR = BASE_DIR / 'Single spectra'
OUTPUT_DIR = BASE_DIR / 'PAPER'
OUTPUT_DIR.mkdir(exist_ok=True)
SHOW_WINDOW = len(sys.argv) <= 1            # show the figures in PyCharm at the end
if not SHOW_WINDOW:
    matplotlib.use('Agg')

SITES = ['m1', 'm2', 'm3', 'm4', 'm5', 'm6']
GROUPS = {0: 'EVs', 1: 'NVs'}               # class 0 = EVs, class 1 = NVs
COLS = {0: '#0B5A7E', 1: '#B5541C'}
# pre-processing
DESPIKE_Z, DESPIKE_M = 7.0, 3               # modified z-score threshold, neighbours replaced
SG_WINDOW, SG_ORDER = 11, 3                 # Savitzky-Golay smoothing
ALS_LAM, ALS_P, ALS_ITER = 1e5, 0.01, 10    # asymmetric least squares baseline
WINDOWS = {'fingerprint': {'raw': (200, 1800), 'grid': (205, 1795), 'region': (250, 1790)},
           'ch': {'raw': (2700, 3100), 'grid': (2705, 3095), 'region': (2760, 3095)}}
# models
N_COMP = 5                                  # PCs given to LDA / latent variables of PLS-DA
PLS_THRESHOLD = 0.5                         # PLS-DA: predicted value > 0.5 = NV
N_FOLDS, CV_SEED = 5, 0                     # site-grouped cross-validation
N_PERM_MODE = 'exact'                       # 'exact' = all 924 site labelings, 'skip'
PC1_REF = {'fingerprint': 1580, 'ch': 2960} # PC1 sign: loading positive at this band
TITLES = {'fingerprint': 'Fingerprint (250-1790 cm$^{-1}$)',
          'ch': 'CH-stretching (2760-3095 cm$^{-1}$)'}
PLAIN = {'fingerprint': 'Fingerprint (250-1790 cm-1)', 'ch': 'CH-stretching (2760-3095 cm-1)'}
SHORT = {'fingerprint': 'Fingerprint', 'ch': 'CH-stretching'}

plt.rcParams.update({'font.family': 'sans-serif',
                     'font.sans-serif': ['Arial', 'Liberation Sans', 'DejaVu Sans'],
                     'font.size': 10})
LOG = []


def report(*a):
    s = ' '.join(str(v) for v in a)
    print(s)
    LOG.append(s)


def save_figure(fig_, name):
    """PNG + TIFF, 600 dpi. If Windows cannot overwrite a file (usually because it
    is open in another program) the figure is saved under a new name."""
    for ext, kw in [('png', {}), ('tiff', {'pil_kwargs': {'compression': 'tiff_lzw'}})]:
        target = OUTPUT_DIR / f'{name}.{ext}'
        for k in range(1, 50):
            try:
                fig_.savefig(target, dpi=600, **kw)
                break
            except OSError as err:
                print(f'  could not write {target.name} ({err.strerror}) - open in another '
                      f'program? Saving as a new file instead.')
                target = OUTPUT_DIR / f'{name}_{k}.{ext}'
    report(f'  saved {name}')


def segment_label(ax, xpos, ybase, segments, ha='center', fontsize=10):
    """Vertical label made of coloured text segments stacked upwards."""
    yy = ybase
    gap = 0.004 * (ax.get_ylim()[1] - ax.get_ylim()[0])
    for txt, col in segments:
        t = ax.text(xpos, yy, txt, rotation=90, va='bottom', ha=ha, fontsize=fontsize, color=col)
        ax.figure.canvas.draw()
        yy = t.get_window_extent().transformed(ax.transData.inverted()).y1 - gap


def parabola_peak(g, s, i):
    """Refine the position of point i by a parabola through i-1, i, i+1."""
    a, b, c = s[i - 1], s[i], s[i + 1]
    return g[i] + 0.5 * (a - c) / (a - 2 * b + c) * (g[1] - g[0])


def peak_position(g, s, lo, hi):
    """Maximum in [lo, hi], refined by a parabola."""
    m = np.flatnonzero((g >= lo) & (g <= hi))
    i = m[int(np.argmax(s[m]))]
    return parabola_peak(g, s, i) if 0 < i < len(g) - 1 else g[i]


def stars(p):
    return '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'n.s.'


# ============================================================================
# 1. DATA AND PRE-PROCESSING
# ============================================================================
# every spectrum, separately in each window:
#   cosmic-ray removal -> 2 cm-1 grid -> Savitzky-Golay -> ALS baseline -> SNV
def despike(y, threshold=DESPIKE_Z, m=DESPIKE_M):
    d = np.diff(y)
    med = np.median(d)
    mad = np.median(np.abs(d - med))
    if mad == 0:
        return y.copy()
    spikes = np.zeros(len(y), bool)
    spikes[1:] = np.abs(0.6745 * (d - med) / mad) > threshold
    out = y.copy()
    for i in np.where(spikes)[0]:
        idx = [j for j in range(max(0, i - m), min(len(y), i + m + 1)) if not spikes[j]]
        if idx:
            out[i] = y[idx].mean()
    return out


def als_baseline(y, lam=ALS_LAM, p=ALS_P, niter=ALS_ITER):
    n = len(y)
    D = sparse.diags([1.0, -2.0, 1.0], [0, -1, -2], shape=(n, n - 2))
    DD = lam * D.dot(D.transpose())
    w = np.ones(n)
    for _ in range(niter):
        z = spsolve((sparse.spdiags(w, 0, n, n) + DD).tocsc(), w * y)
        w = p * (y > z) + (1 - p) * (y < z)
    return z


def preprocess(x, Y, win):
    lo, hi = WINDOWS[win]['raw']
    grid = np.arange(WINDOWS[win]['grid'][0], WINDOWS[win]['grid'][1] + 1, 2.0)
    m = (x >= lo - 5) & (x <= hi + 5)
    corr = []
    for yy in Y:
        ys = savgol_filter(np.interp(grid, x[m], despike(yy[m])), SG_WINDOW, SG_ORDER)
        corr.append(ys - als_baseline(ys))
    corr = np.array(corr)
    return grid, (corr - corr.mean(1, keepdims=True)) / corr.std(1, keepdims=True)


report('1. DATA')
full = {w: [] for w in WINDOWS}             # SNV on the whole pre-processing window
y, sites, names = [], [], []
for cls, path in enumerate([EV_FILE, NV_FILE]):
    for site in SITES:
        df = pd.read_excel(path, sheet_name=site, header=2)
        x, Y = df.iloc[:, 0].to_numpy(float), df.iloc[:, 1:].to_numpy(float).T
        for w in WINDOWS:
            grid, snv = preprocess(x, Y, w)
            full[w].append(snv)
            WINDOWS[w]['grid_full'] = grid
        y += [cls] * len(Y)
        sites += [f'{GROUPS[cls]}-{site}'] * len(Y)
        names += [f'{GROUPS[cls]}-{site}-{c}' for c in df.columns[1:]]
    report(f'  {path.name}: {len(SITES)} sites x 20 = {sum(1 for c in y if c == cls)} spectra')
full = {w: np.vstack(v) for w, v in full.items()}
y, sites = np.array(y), np.array(sites)
data = {}                                   # analysis regions used by all models
for w in WINDOWS:
    gf = WINDOWS[w]['grid_full']
    keep = (gf >= WINDOWS[w]['region'][0]) & (gf <= WINDOWS[w]['region'][1])
    data[w] = full[w][:, keep]
    WINDOWS[w]['g'] = gf[keep]
    report(f'  {PLAIN[w]}: {data[w].shape[1]} wavenumbers (2 cm-1 grid)')
SITE_IDS = sorted(set(sites))


def site_means(v):
    """Mean of a value over the 20 spectra of each site -> 6 numbers per group."""
    return {c: np.array([v[sites == s].mean() for s in sorted(set(sites[y == c]))]) for c in (0, 1)}


RES = {w: {} for w in WINDOWS}              # every result, used for the tables and the text


# ============================================================================
# 2. FIGURE 1 - single raw spectra, CaF2 vs Ag/SiNWs (EV and NV)
# ============================================================================
# raw counts (no smoothing, no baseline, no normalisation); peak = most prominent
# local maximum within +/-12 cm-1 of the nominal band on a lightly smoothed copy
SINGLE_FILES = {'EV': ('s2-ev_acacia-dropcaf2', 's2-ev_acacia-agsinw'),
                'NV': ('s1-ev_nano_acacia-dropcaf2', 's4a-ev_nano_acacia-20ng-agsinw')}
SINGLE_NAME = {'EV': 'EV Acacia', 'NV': 'Nanovesicles Acacia'}
SINGLE_OFFSET = {'EV': 110, 'NV': 0}        # counts added to the Ag/SiNW trace (display only)
SINGLE_PEAKS = {
    'EV': [(519, 'Si substrate', 'Si optical phonon'), (831, 'Protein/NA', 'Tyr Fermi doublet / O-P-O'),
           (1001, 'Protein', 'Phe ring breathing'), (1031, 'Protein', 'Phe C-H in-plane bending'),
           (1166, 'Protein', 'Tyr C-H in-plane bending'), (1355, 'Protein', 'Trp / CH deformation'),
           (1582, 'Protein', 'Phe C=C bending'), (1604, 'Protein', 'Phe/Tyr ring C=C stretching'),
           (1724, 'Lipid', 'C=O ester stretching'), (2929, 'Protein/Lipid', 'CH2/CH3 stretching')],
    'NV': [(519, 'Si substrate', 'Si optical phonon'), (612, 'Protein', 'Phe C-C twisting'),
           (797, 'Nucleic acid', 'O-P-O backbone stretching'),
           (826, 'Protein/NA', 'Tyr Fermi doublet / O-P-O'), (921, 'Protein/Carb.', 'C-C proline / glucose'),
           (1001, 'Protein', 'Phe ring breathing'), (1031, 'Protein', 'Phe C-H in-plane bending'),
           (1240, 'Protein', 'Amide III'), (1386, 'Lipid/Protein', 'CH3 deformation / COO-'),
           (1607, 'Protein', 'Phe/Tyr ring C=C stretching'), (1624, 'Protein', 'Trp/Tyr C=C, amide I side'),
           (2936, 'Protein/Lipid', 'CH3 symmetric stretching')]}
SINGLE_COL = {'Protein': '#1F6FB4', 'Protein/NA': '#6A3D9A', 'Nucleic acid': '#8E44AD',
              'Protein/Carb.': '#1F6FB4', 'Lipid/Protein': '#D35400', 'Protein/Lipid': '#D35400',
              'Lipid': '#C0392B', 'Si substrate': '#7F7F7F'}
peak_table = []


def find_single(prefix):
    if not SINGLE_DIR.exists():
        return None
    hits = [f for f in SINGLE_DIR.glob('*.txt') if f.name.lower().startswith(prefix)]
    return hits[0] if len(hits) == 1 else None


def figure1(tag):
    f_caf2, f_ag = (find_single(p) for p in SINGLE_FILES[tag])
    if f_caf2 is None or f_ag is None:
        report(f'  Figure 1 {tag}: files not found in {SINGLE_DIR} - skipped')
        return
    caf2, ag = np.loadtxt(f_caf2), np.loadtxt(f_ag)
    ag_s = savgol_filter(ag[:, 1], 11, 3)              # only to locate the maxima
    noise = np.std(ag[:, 1] - ag_s)
    ag[:, 1] += SINGLE_OFFSET[tag]
    rows = []
    for nominal, cls, assign in SINGLE_PEAKS[tag]:
        k = np.flatnonzero(np.abs(ag[:, 0] - nominal) <= 12)
        pk, pr = find_peaks(ag_s[k], prominence=0)
        j = np.argmax(pr['prominences'])
        rows.append((round(float(ag[k[pk[j]], 0])), cls))
        peak_table.append({'Figure': f'1 ({tag}, single spectrum on Ag/SiNWs)', 'Group': tag,
                           'Nominal (cm-1)': nominal, 'Measured (cm-1)': round(float(ag[k[pk[j]], 0])),
                           'Class': cls, 'Assignment': assign,
                           'Prominence / noise SD': round(float(pr['prominences'][j] / noise), 1)})
    lo1, hi1, lo2, hi2 = 400, 1800, 2700, 3100
    fig, (axL, axR) = plt.subplots(1, 2, sharey=True, figsize=(9, 6.5),
                                   gridspec_kw={'width_ratios': [hi1 - lo1, hi2 - lo2], 'wspace': 0.03})

    def in_view(d):
        xx = d[:, 0]
        return d[((xx >= lo1) & (xx <= hi1)) | ((xx >= lo2) & (xx <= hi2)), 1]

    y_min = min(in_view(caf2).min(), in_view(ag).min())
    y_max = max(in_view(caf2).max(), in_view(ag).max())
    y_top = y_max + 0.45 * (y_max - y_min)
    axL.set_ylim(y_min - 0.12 * (y_max - y_min), y_top)
    for ax, lo, hi in [(axL, lo1, hi1), (axR, lo2, hi2)]:
        for d, col in [(caf2, '#3A3A3A'), (ag, '#E8383D')]:
            m = (d[:, 0] >= lo) & (d[:, 0] <= hi)
            ax.plot(d[m, 0], d[m, 1], color=col, lw=0.6)
        ax.set_xlim(lo, hi)
        pos = [r[0] for r in rows]
        for k, (measured, cls) in enumerate(rows):
            if not lo <= measured <= hi:
                continue
            near_left = k > 0 and measured - pos[k - 1] < 35
            ha = 'left' if (near_left or measured - lo < 30) else 'right'
            t = ax.text(measured, y_top - 0.01 * (y_top - y_min), f'{measured}  {cls}', rotation=90,
                        ha=ha, va='top', fontsize=7.5, color=SINGLE_COL[cls])
            fig.canvas.draw()
            bottom = t.get_window_extent().transformed(ax.transData.inverted()).y0
            peak_top = ag[np.abs(ag[:, 0] - measured) <= 3, 1].max()
            ax.plot([measured, measured], [peak_top + 0.01 * (y_top - y_min), bottom],
                    color='k', ls=':', lw=0.8)
        ax.tick_params(direction='in', top=True)
    axL.spines['right'].set_visible(False)
    axR.spines['left'].set_visible(False)
    axL.tick_params(labelleft=False)
    axR.tick_params(left=False, right=True, labelleft=False)
    axL.set_xticks([400, 600, 800, 1000, 1200, 1400, 1600])
    axR.set_xticks([2800, 3000])
    for d_, col, txt in [(ag, '#E8383D', f'{SINGLE_NAME[tag]} on Ag/SiNWs'),
                         (caf2, '#3A3A3A', f'{SINGLE_NAME[tag]} on CaF$_2$')]:
        m = (d_[:, 0] >= 420) & (d_[:, 0] <= 760)
        axL.text(420, d_[m, 1].min() - 0.015 * (y_top - y_min), txt, color=col, fontsize=10,
                 ha='left', va='top')
    dd = 0.012
    for ax, x0 in [(axL, 1), (axR, 0)]:
        dx = dd * (hi2 - lo2) / (hi1 - lo1) if ax is axL else dd
        for y0 in (0, 1):
            ax.plot((x0 - dx, x0 + dx), (y0 - 2 * dd, y0 + 2 * dd), transform=ax.transAxes,
                    color='k', clip_on=False, lw=0.8)
    axL.set_ylabel('Raman Intensity (a.u.)', fontsize=11)
    fig.text(0.5, 0.02, 'Wavenumber (cm$^{-1}$)', ha='center', fontsize=11)
    fig.subplots_adjust(left=0.06, right=0.97, bottom=0.10, top=0.97)
    save_figure(fig, f'Fig1{"a" if tag == "EV" else "b"}_{tag}_CaF2_vs_AgSiNW')


report('\n2. FIGURE 1 - single spectra on CaF2 and Ag/SiNWs')
for tag in ['EV', 'NV']:
    figure1(tag)


# ============================================================================
# 3. FIGURE 2 - mean +/- SD spectra (all 120 + 120 spectra, sample SD n - 1)
# ============================================================================
# label = maximum of the group mean spectrum near the nominal band (parabola);
# grey shadow = EV/NV shift that is consistent: the peak positions of the 6 EV
# and 6 NV site-mean spectra do not overlap
CLASS_SEGMENTS = {'Proteins': [('Proteins', '#7030A0')],
                  'Nucleic Acids': [('Nucleic Acids', '#1F3864')],
                  'Lipids/NA': [('Lipids/', '#2E75B6'), ('NA', 'k')],
                  'Lipids': [('Lipids', '#2E75B6')],
                  'Proteins/Lipids': [('Proteins/', '#7030A0'), ('Lipids', '#2E75B6')]}
FP_PEAKS = {
    0: [(519, 'Si', 'Si substrate (crystalline silicon phonon)', 'up'),
        (809, 'Nucleic Acids', 'O-P-O backbone stretching', 'up'),
        (855, 'Proteins', 'Tyr ring breathing (Fermi doublet)', 'up'),
        (931, 'Proteins', 'C-C stretching, protein backbone', 'up'),
        (1001, 'Proteins', 'Phe ring breathing', 'up'),
        (1161, 'Proteins', 'Tyr C-H bending / C-C stretching', 'up'),
        (1249, 'Proteins', 'Amide III', 'up'),
        (1373, 'Lipids/NA', 'CH3 deformation (lipids) / nucleic-acid bases', 'up'),
        (1603, 'Proteins', 'Phe/Tyr ring C=C stretching', 'up')],
    1: [(519, 'Si', 'Si substrate (crystalline silicon phonon)', 'up'),
        (799, 'Nucleic Acids', 'O-P-O backbone stretching', 'up-left'),
        (829, 'Proteins', 'Tyr ring breathing (Fermi doublet)', 'up-right'),
        (927, 'Proteins', 'C-C stretching, protein backbone', 'up'),
        (1001, 'Proteins', 'Phe ring breathing', 'up-left'),
        (1033, 'Proteins', 'Phe C-H in-plane bending', 'up-right'),
        (1169, 'Proteins', 'Tyr C-H bending / C-C stretching', 'up'),
        (1245, 'Proteins', 'Amide III', 'up'),
        (1385, 'Lipids/NA', 'CH3 deformation (lipids) / nucleic-acid bases', 'up'),
        (1607, 'Proteins', 'Phe/Tyr ring C=C stretching', 'up'),
        (1729, 'Lipids', 'Ester C=O stretching (lipids)', 'up')]}
FP_SHIFT_PAIRS = [(519, 519), (809, 799), (931, 927), (1001, 1001), (1161, 1169),
                  (1249, 1245), (1373, 1385), (1603, 1607)]
# CH region: (nominal, half-width, 'max' or 'sh' = shoulder, class, assignment)
CH_BANDS = [(2853, 10, 'sh', 'Lipids', 'CH2 symmetric stretching'),
            (2880, 12, 'sh', 'Lipids', 'CH2 asymmetric stretching'),
            (2935, 12, 'max', 'Proteins', 'CH3 symmetric stretching'),
            (2960, 10, 'sh', 'Proteins/Lipids', 'CH3 asymmetric stretching'),
            (3065, 12, 'max', 'Proteins', 'Aromatic =C-H stretching')]
CH_SHOW = [2935]                            # labelled in Fig 2b (all bands go to Table S4)
NAME_LONG = {0: 'EV Acacia on Ag/SiNWs', 1: 'Nanovesicles Acacia on Ag/SiNWs'}
shift_table = []


def band_position(gr, s, nominal, half, kind):
    """'max': most prominent local maximum within +/-half; 'sh': most prominent
    minimum of the 2nd derivative (shoulder). None = not resolved."""
    if kind == 'sh':
        s = -savgol_filter(s, 11, 3, deriv=2)
    m = np.flatnonzero(np.abs(gr - nominal) <= half)
    pk, pr = find_peaks(s[m], prominence=0)
    if not len(pk):
        return None
    return parabola_peak(gr, s, m[pk[np.argmax(pr['prominences'])]])


report('\n3. FIGURE 2 - mean +/- SD spectra')
# --- Fig 2a, fingerprint 400-1790 cm-1
g = WINDOWS['fingerprint']['grid_full']
Xf = full['fingerprint']
show = (g >= 400) & (g <= 1790)
gs = g[show]
fig, ax = plt.subplots(figsize=(8.5, 6.5))
for ev_nom, nv_nom in FP_SHIFT_PAIRS:
    pe = peak_position(g, Xf[y == 0].mean(0), ev_nom - 12, ev_nom + 12)
    pn = peak_position(g, Xf[y == 1].mean(0), nv_nom - 12, nv_nom + 12)
    se = [peak_position(g, Xf[sites == f'EVs-{k}'].mean(0), ev_nom - 12, ev_nom + 12) for k in SITES]
    sn = [peak_position(g, Xf[sites == f'NVs-{k}'].mean(0), nv_nom - 12, nv_nom + 12) for k in SITES]
    consistent = max(se) < min(sn) or max(sn) < min(se)
    shift_table.append({'Region': 'fingerprint', 'Method': 'maximum', 'EV (cm-1)': round(pe, 1),
                        'NV (cm-1)': round(pn, 1), 'Shift NV-EV (cm-1)': round(pn - pe, 1),
                        'EV 6 sites': f'{min(se):.1f}-{max(se):.1f}',
                        'NV 6 sites': f'{min(sn):.1f}-{max(sn):.1f}', 'Consistent': consistent})
    if consistent:
        ax.axvspan(min(pe, pn) - 4, max(pe, pn) + 4, color='0.92', lw=0, zorder=0)
labels, lines = [], []
for c, off in [(0, 0.0), (1, 8.5)]:
    mu, sd = Xf[y == c].mean(0)[show], Xf[y == c].std(0, ddof=1)[show]
    ax.fill_between(gs, mu - sd + off, mu + sd + off, color=COLS[c], alpha=0.22, lw=0)
    ax.plot(gs, mu + off, color=COLS[c], lw=1.0,
            label=f'{NAME_LONG[c]} (mean \u00b1 SD, n = {np.sum(y == c)})')
    for nominal, cls, assign, where in FP_PEAKS[c]:
        pos = peak_position(gs, mu, nominal - 12, nominal + 12)
        peak_table.append({'Figure': '2a (mean spectrum)', 'Group': GROUPS[c], 'Nominal (cm-1)': nominal,
                           'Measured (cm-1)': round(pos, 1), 'Class': cls, 'Assignment': assign,
                           'Mean SNV': round(float(np.interp(pos, gs, mu)), 3),
                           'SD': round(float(np.interp(pos, gs, sd)), 3)})
        top = (mu + sd)[np.abs(gs - pos) <= (4 if where == 'up' else 25)].max() + off
        txt = f'{pos:.1f}' if cls == 'Si' else f'{pos:.0f}'
        segs = [(txt, 'k')] if cls == 'Si' else [(txt + '- ', 'k')] + CLASS_SEGMENTS[cls]
        dx = {'up': 0, 'up-left': -3, 'up-right': 3}[where]
        lines.append((pos, np.interp(pos, gs, mu) + off, top + 0.08))
        labels.append((pos + dx, top + 0.12, segs,
                       {'up': 'center', 'up-left': 'right', 'up-right': 'left'}[where]))
for xl, yb, yt in lines:
    ax.plot([xl, xl], [yb, yt], ls=(0, (1, 2)), lw=0.7, color='0.35')
ax.set_xlim(400, 1790)
ax.set_ylim((Xf[y == 0].mean(0) - Xf[y == 0].std(0, ddof=1))[show].min() - 0.3,
            (Xf[y == 1].mean(0) + Xf[y == 1].std(0, ddof=1))[show].max() + 8.5 + 4.3)
for lab in labels:
    segment_label(ax, *lab)
ax.tick_params(axis='y', direction='in', length=5, labelleft=False)
ax.tick_params(axis='x', direction='in', length=5)
ax.set_xlabel('Raman Shift (cm$^{-1}$)', fontsize=13)
ax.set_ylabel('Intensity (a.u.)', fontsize=13)
ax.legend(frameon=True, facecolor='white', edgecolor='none', framealpha=1, fontsize=10,
          loc='upper left', handlelength=1.5)
fig.tight_layout()
save_figure(fig, 'Fig2a_MeanSD_Fingerprint')

# --- Fig 2b, CH-stretching 2760-3095 cm-1
g, X = WINDOWS['ch']['g'], data['ch']
fig, ax = plt.subplots(figsize=(6.5, 6.0))
ch_rows, labels, lines = [], [], []
for nominal, half, kind, cls, assign in CH_BANDS:
    pos = {c: band_position(g, X[y == c].mean(0), nominal, half, kind) for c in (0, 1)}
    site_pos = {c: [band_position(g, X[sites == k].mean(0), nominal, half, kind)
                    for k in sorted(set(sites[y == c]))] for c in (0, 1)}
    resolved = None not in list(pos.values()) + site_pos[0] + site_pos[1]
    consistent = bool(resolved and ((max(site_pos[0]) < min(site_pos[1])) or
                                    (max(site_pos[1]) < min(site_pos[0]))))
    sp = {c: [v for v in site_pos[c] if v is not None] for c in (0, 1)}
    shift_table.append({'Region': 'ch', 'Method': 'maximum' if kind == 'max' else 'shoulder (2nd derivative)',
                        'EV (cm-1)': 'not resolved' if pos[0] is None else round(pos[0], 1),
                        'NV (cm-1)': 'not resolved' if pos[1] is None else round(pos[1], 1),
                        'Shift NV-EV (cm-1)': round(pos[1] - pos[0], 1) if resolved else '',
                        'EV 6 sites': f'{min(sp[0]):.1f}-{max(sp[0]):.1f} ({len(sp[0])}/6)' if sp[0] else '',
                        'NV 6 sites': f'{min(sp[1]):.1f}-{max(sp[1]):.1f} ({len(sp[1])}/6)' if sp[1] else '',
                        'Consistent': consistent})
    if consistent and nominal in CH_SHOW:
        ax.axvspan(min(pos.values()) - 4, max(pos.values()) + 4, color='0.92', lw=0, zorder=0)
    for c in (0, 1):
        if pos[c] is not None:
            peak_table.append({'Figure': '2b (mean spectrum)', 'Group': GROUPS[c], 'Nominal (cm-1)': nominal,
                               'Measured (cm-1)': round(pos[c], 1), 'Class': cls, 'Assignment': assign})
        ch_rows.append((c, nominal, kind, pos[c], cls))
for c, off in [(0, 0.0), (1, 5.5)]:
    mu, sd = X[y == c].mean(0), X[y == c].std(0, ddof=1)
    ax.fill_between(g, mu - sd + off, mu + sd + off, color=COLS[c], alpha=0.22, lw=0)
    ax.plot(g, mu + off, color=COLS[c], lw=1.0,
            label=f'{NAME_LONG[c]} (mean \u00b1 SD, n = {np.sum(y == c)})')
    for cc, nominal, kind, pos, cls in ch_rows:
        if cc != c or pos is None or nominal not in CH_SHOW:
            continue
        top = (mu + sd)[np.abs(g - pos) <= 6].max() + off
        txt = f'{pos:.0f}' + (' (sh)' if kind == 'sh' else '')
        lines.append((pos, np.interp(pos, g, mu) + off, top + 0.08))
        labels.append((pos, top + 0.12, [(txt + '- ', 'k')] + CLASS_SEGMENTS[cls]))
for xl, yb, yt in lines:
    ax.plot([xl, xl], [yb, yt], ls=(0, (1, 2)), lw=0.7, color='0.35')
ax.set_xlim(2760, 3095)
ax.set_ylim((X[y == 0].mean(0) - X[y == 0].std(0, ddof=1)).min() - 0.3,
            (X[y == 1].mean(0) + X[y == 1].std(0, ddof=1)).max() + 5.5 + 4.4)
for lab in labels:
    segment_label(ax, *lab)
ax.tick_params(axis='y', direction='in', length=5, labelleft=False)
ax.tick_params(axis='x', direction='in', length=5)
ax.set_xlabel('Raman Shift (cm$^{-1}$)', fontsize=13)
ax.set_ylabel('Intensity (a.u.)', fontsize=13)
ax.legend(frameon=True, facecolor='white', edgecolor='none', framealpha=1, fontsize=9,
          loc='upper left', handlelength=1.5)
fig.tight_layout()
save_figure(fig, 'Fig2b_MeanSD_CH')
for r in shift_table:
    if r['Consistent']:
        report(f"  consistent shift ({r['Region']}): {r['EV (cm-1)']} -> {r['NV (cm-1)']} cm-1 "
               f"({r['Shift NV-EV (cm-1)']:+})")


# ============================================================================
# 4. FIGURE 3 - PCA scores and LDA canonical variable
# ============================================================================
# PCA: all 240 SNV spectra, labels NOT used. LDA: the 5 PC scores with labels,
# fitted on all spectra (display of the separation; the classification result is
# the cross-validation of section 6). p: Welch's t-test on the 6 + 6 site means.
report('\n4. FIGURE 3 - PCA and LDA')
plt.rcParams.update({'font.size': 12, 'axes.spines.top': False, 'axes.spines.right': False})
fig, axs = plt.subplots(2, 2, figsize=(10, 9),
                        gridspec_kw={'width_ratios': [1.35, 1], 'wspace': 0.32, 'hspace': 0.42})
letters = iter('abcd')
for row, w in enumerate(['fingerprint', 'ch']):
    X, g = data[w], WINDOWS[w]['g']
    pca = PCA(n_components=N_COMP, svd_solver='full').fit(X)
    T = pca.transform(X)
    sign = np.sign(pca.components_[0][np.argmin(np.abs(g - PC1_REF[w]))])
    T[:, 0] *= sign
    var = pca.explained_variance_ratio_ * 100
    lda = LinearDiscriminantAnalysis().fit(T, y)
    ld = lda.transform(T).ravel()
    ld *= np.sign(ld[y == 1].mean())
    sm = site_means(ld)
    p = ttest_ind(sm[0], sm[1], equal_var=False).pvalue
    loadings = pca.components_[:2].copy()
    loadings[0] *= sign
    RES[w].update(var=var, T=T, ld=ld, p_ld1=p, loadings=loadings,
                  pc1_mean={c: T[y == c, 0].mean() for c in (0, 1)},
                  ld_median={c: np.median(ld[y == c]) for c in (0, 1)}, ld_site=sm)
    report(f'  {PLAIN[w]}: PC1 {var[0]:.1f} %, PC2 {var[1]:.1f} %, PC1-5 {var.sum():.1f} %; '
           f'LD1 site means p = {p:.2e}')
    scores = pd.DataFrame({'spectrum': names, 'group': [GROUPS[c] for c in y], 'site': sites})
    for i in range(N_COMP):
        scores[f'PC{i + 1}'] = T[:, i]
    scores['LD1'] = ld
    scores.to_csv(OUTPUT_DIR / f'Scores_{"fingerprint" if w == "fingerprint" else "CH"}.csv', index=False)

    ax = axs[row, 0]                        # PCA scores, 2-SD ellipses
    for c in (0, 1):
        P = T[y == c, :2]
        ax.scatter(P[:, 0], P[:, 1], s=14, color=COLS[c], lw=0, alpha=0.85, label=GROUPS[c])
        vals, vecs = np.linalg.eigh(np.cov(P.T))
        ang = np.degrees(np.arctan2(vecs[1, 1], vecs[0, 1]))
        ax.add_patch(Ellipse(P.mean(0), 4 * np.sqrt(vals[1]), 4 * np.sqrt(vals[0]), angle=ang,
                             fc='none', ec=COLS[c], lw=1.4))
    ax.axhline(0, color='k', lw=0.6)
    ax.axvline(0, color='k', lw=0.6)
    ax.set_xlabel(f'PC1 ({var[0]:.1f}%)')
    ax.set_ylabel(f'PC2 ({var[1]:.1f}%)')
    ylo, yhi = ax.get_ylim()
    ax.set_ylim(ylo, yhi + 0.28 * (yhi - ylo))
    ax.legend(frameon=False, fontsize=11, loc='upper left', ncol=2, handletextpad=0.1,
              columnspacing=1.0, markerscale=1.6)
    ax.set_title(TITLES[w], fontsize=12, loc='left')

    ax = axs[row, 1]                        # LD1 box plot
    rng = np.random.default_rng(0)
    for c in (0, 1):
        v = ld[y == c]
        ax.boxplot(v, positions=[c], widths=0.5, showfliers=False, patch_artist=True,
                   boxprops=dict(fc=COLS[c], alpha=0.30, ec=COLS[c]),
                   medianprops=dict(color=COLS[c], lw=1.6),
                   whiskerprops=dict(color=COLS[c]), capprops=dict(color=COLS[c]))
        ax.scatter(c + rng.uniform(-0.15, 0.15, len(v)), v, s=8, color=COLS[c], lw=0, alpha=0.8)
    top = ld.max() + 0.08 * np.ptp(ld)
    ax.plot([0, 1], [top, top], color='k', lw=0.8)
    ax.text(0.5, top, stars(p), ha='center', va='bottom', fontsize=12)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(['EVs', 'NVs'])
    ax.set_ylabel('Canonical variable 1')
    ax.set_ylim(ld.min() - 0.20 * np.ptp(ld), top + 0.15 * np.ptp(ld))
    ax.text(0.5, 0.02, "Welch's t-test on site means\n(n = 6), " + stars(p) + f' p = {p:.1e}',
            transform=ax.transAxes, ha='center', va='bottom', fontsize=9)
    for a in axs[row]:
        a.text(-0.27 if a is axs[row, 1] else -0.18, 1.08, f'({next(letters)})',
               transform=a.transAxes, fontsize=14, va='bottom')
fig.subplots_adjust(left=0.09, right=0.98, bottom=0.07, top=0.94)
save_figure(fig, 'Fig3_PCA_LDA')


# ============================================================================
# 5. FIGURE 4 - PC loadings and NV - EV difference spectrum
# ============================================================================
# loadings: positive PC1 loading = band relatively stronger in the group with
# positive PC1 scores. Difference: mean of the 6 NV site spectra - mean of the 6
# EV site spectra; shaded = +/- 2 SE from the site-to-site variation.
def main_bands(g, curve, n, min_gap=20):
    """n largest local extrema (absolute value), at least min_gap cm-1 apart."""
    i = np.flatnonzero((np.abs(curve[1:-1]) >= np.abs(curve[:-2])) &
                       (np.abs(curve[1:-1]) >= np.abs(curve[2:]))) + 1
    chosen = []
    for k in i[np.argsort(-np.abs(curve[i]))]:
        if all(abs(g[k] - g[j]) >= min_gap for j in chosen):
            chosen.append(k)
        if len(chosen) == n:
            break
    return [(g[k], curve[k]) for k in sorted(chosen)]


N_BANDS = {'fingerprint': 8, 'ch': 4}
report('\n5. FIGURE 4 - loadings and difference spectrum')
band_rows = []
fig, axs = plt.subplots(2, 2, figsize=(12, 9), gridspec_kw={'width_ratios': [3, 1.3]})
for col, w in enumerate(['fingerprint', 'ch']):
    g, L, var = WINDOWS[w]['g'], RES[w]['loadings'], RES[w]['var']
    hi_g = GROUPS[0] if RES[w]['pc1_mean'][0] > 0 else GROUPS[1]
    lo_g = GROUPS[1] if hi_g == GROUPS[0] else GROUPS[0]
    ax = axs[0, col]
    ax.axhline(0, color='k', lw=0.6)
    ax.plot(g, L[0], color='#0B5A7E', lw=1.2, label=f'PC1 ({var[0]:.1f}%)')
    ax.plot(g, L[1], color='#B5541C', lw=1.0, alpha=0.8, label=f'PC2 ({var[1]:.1f}%)')
    bands = main_bands(g, L[0], N_BANDS[w])
    RES[w]['load_bands'] = [(wn, v, hi_g if v > 0 else lo_g) for wn, v in bands]
    span = np.ptp(L[:2])
    for wn, v in bands:
        ax.text(wn, v + (0.03 if v > 0 else -0.03) * span, f'{wn:.0f}', ha='center',
                va='bottom' if v > 0 else 'top', fontsize=9, rotation=90)
        band_rows.append({'Source': 'PC1 loading', 'Region': w, 'Band (cm-1)': round(wn),
                          'Value': round(v, 4), 'Relatively stronger in': hi_g if v > 0 else lo_g})
    ax.set_xlim(g[0], g[-1])
    ax.set_ylim(L[:2].min() - 0.18 * span, L[:2].max() + 0.22 * span)
    ax.set_xlabel('Raman shift (cm$^{-1}$)')
    ax.set_ylabel('Loading (a.u.)')
    ax.set_title(f'{TITLES[w]}\nPC1 > 0: {hi_g} side', fontsize=11, loc='left')
    ax.legend(frameon=False, fontsize=10, loc='lower left')

    X = data[w]
    S = {c: np.array([X[sites == s].mean(0) for s in sorted(set(sites[y == c]))]) for c in (0, 1)}
    d = S[1].mean(0) - S[0].mean(0)
    se = np.sqrt(S[0].var(0, ddof=1) / len(S[0]) + S[1].var(0, ddof=1) / len(S[1]))
    sig = np.abs(d) > 2 * se
    ax = axs[1, col]
    ax.axhline(0, color='k', lw=0.6)
    ax.fill_between(g, d - 2 * se, d + 2 * se, color='0.75', lw=0, label='$\\pm$2 SE (6 + 6 sites)')
    ax.plot(g, d, color='k', lw=1.0, label='NV $-$ EV')
    bands = [b for b in main_bands(g, np.where(sig, d, 0), N_BANDS[w]) if b[1] != 0]
    RES[w]['diff_bands'] = bands
    RES[w]['diff_sig_frac'] = sig.mean() * 100
    span = np.ptp(np.r_[d - 2 * se, d + 2 * se])
    for wn, v in bands:
        ax.text(wn, v + (0.04 if v > 0 else -0.04) * span, f'{wn:.0f}', ha='center',
                va='bottom' if v > 0 else 'top', fontsize=9, rotation=90)
        band_rows.append({'Source': 'NV - EV difference', 'Region': w, 'Band (cm-1)': round(wn),
                          'Value': round(v, 3), 'Relatively stronger in': GROUPS[1] if v > 0 else GROUPS[0]})
    ax.set_xlim(g[0], g[-1])
    ax.set_ylim((d - 2 * se).min() - 0.15 * span, (d + 2 * se).max() + 0.2 * span)
    ax.set_xlabel('Raman shift (cm$^{-1}$)')
    ax.set_ylabel('$\\Delta$ SNV intensity, NV $-$ EV')
    ax.set_title(TITLES[w], fontsize=11, loc='left')
    ax.legend(frameon=False, fontsize=10, loc='lower left' if w == 'fingerprint' else 'upper left')
    report(f'  {PLAIN[w]}: PC1 > 0 = {hi_g}; difference > 2 SE at {sig.mean() * 100:.0f} % '
           f'of wavenumbers')
for a, letter in zip([axs[0, 0], axs[0, 1], axs[1, 0], axs[1, 1]], 'abcd'):
    a.text(-0.10 if a in (axs[0, 0], axs[1, 0]) else -0.25, 1.06, f'({letter})',
           transform=a.transAxes, fontsize=14, va='bottom')
fig.tight_layout()
save_figure(fig, 'Fig4_Loadings_Difference')


# ============================================================================
# 6. FIGURE 5 - cross-validated classification (confusion matrices)
# ============================================================================
# 12 sites split into 5 folds, all 20 spectra of a site together; train on 4
# folds, predict the 5th; every spectrum predicted once by a model that never
# saw its site.
#   PCA-LDA: PCA (5 PCs) on all spectra without labels; LDA on the training folds
#   PLS-DA : autoscaling + PLS (5 latent variables) on the training folds;
#            predicted value > 0.5 = NV
def make_folds(labels):
    return list(StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=CV_SEED)
                .split(np.zeros(len(labels)), labels, sites))


def cv_pca_lda(X, labels, folds_, T=None, refit=False):
    if T is None:
        T = PCA(n_components=N_COMP, svd_solver='full').fit_transform(X)
    pred = np.empty(len(labels), int)
    for tr, te in folds_:
        if refit:                           # stricter: PCA from the training folds only
            pf = PCA(n_components=N_COMP, svd_solver='full').fit(X[tr])
            Ttr, Tte = pf.transform(X[tr]), pf.transform(X[te])
        else:
            Ttr, Tte = T[tr], T[te]
        pred[te] = LinearDiscriminantAnalysis().fit(Ttr, labels[tr]).predict(Tte)
    return pred


def cv_plsda(X, labels, folds_):
    pred = np.empty(len(labels), int)
    for tr, te in folds_:
        sc = StandardScaler().fit(X[tr])
        pls = PLSRegression(n_components=N_COMP, scale=False).fit(sc.transform(X[tr]), labels[tr])
        pred[te] = (pls.predict(sc.transform(X[te])).ravel() > PLS_THRESHOLD).astype(int)
    return pred


report('\n6. FIGURE 5 - cross-validation')
folds = make_folds(y)
for k, (tr, te) in enumerate(folds, 1):
    report(f'  fold {k}: test sites ' + ', '.join(sorted(set(sites[te]))))
fold_rows = []
fig, axs = plt.subplots(2, 2, figsize=(8, 7.6))
letters = iter('abcd')
for row, w in enumerate(['fingerprint', 'ch']):
    X = data[w]
    RES[w]['strict'] = (cv_pca_lda(X, y, folds, refit=True) == y).mean() * 100
    for col, (model, pred) in enumerate([('PCA-LDA', cv_pca_lda(X, y, folds)),
                                         ('PLS-DA', cv_plsda(X, y, folds))]):
        cm = confusion_matrix(y, pred)
        acc = (pred == y).mean() * 100
        sens, spec = cm[1, 1] / cm[1].sum() * 100, cm[0, 0] / cm[0].sum() * 100
        RES[w].setdefault('cv', {})[model] = {'acc': acc, 'sens': sens, 'spec': spec, 'cm': cm}
        for k, (tr, te) in enumerate(folds, 1):
            fold_rows.append({'Region': PLAIN[w], 'Model': model, 'Fold': k,
                              'Test sites': ', '.join(sorted(set(sites[te]))),
                              'Test spectra': len(te), 'Correct': int(np.sum(pred[te] == y[te])),
                              'Accuracy (%)': round(np.mean(pred[te] == y[te]) * 100, 1)})
        report(f'  {PLAIN[w]} {model}: accuracy {acc:.1f} %, sensitivity {sens:.1f} %, '
               f'specificity {spec:.1f} %')
        ax = axs[row, col]
        ax.imshow(cm, cmap='Blues', vmin=0, vmax=120)
        for i in range(2):
            for j in range(2):
                ax.text(j, i, f'{cm[i, j]}\n({cm[i, j] / cm[i].sum() * 100:.1f}%)', ha='center',
                        va='center', fontsize=13, color='white' if cm[i, j] > 60 else 'black')
        ax.set_xticks([0, 1])
        ax.set_xticklabels(['EVs', 'NVs'])
        ax.set_yticks([0, 1])
        ax.set_yticklabels(['EVs', 'NVs'])
        ax.set_xlabel('Predicted')
        ax.set_ylabel('True')
        ax.set_title(f'{model} - {SHORT[w]}\naccuracy {acc:.1f}%', fontsize=12)
        ax.text(-0.32, 1.12, f'({next(letters)})', transform=ax.transAxes, fontsize=14, va='bottom')
    report(f'  {PLAIN[w]} PCA-LDA, PCA re-fitted in each fold: {RES[w]["strict"]:.1f} %')
fig.tight_layout()
save_figure(fig, 'Fig5_ConfusionMatrices')
pd.DataFrame(fold_rows).to_csv(OUTPUT_DIR / 'TableS2_Fold_results.csv', index=False)


# ============================================================================
# 7. FIGURE S1 - exact label-shuffling (permutation) test
# ============================================================================
# all 924 ways of calling 6 of the 12 sites "EV"; for each, the whole
# cross-validation is repeated. p = labelings with accuracy >= real / 924.
# The real labeling and its EV<->NV mirror give the same accuracy, so the
# smallest possible p is 2/924 = 0.0022.
if N_PERM_MODE == 'exact':
    report('\n7. FIGURE S1 - permutation test (924 site labelings)')
    labelings = list(combinations(range(len(SITE_IDS)), 6))
    fig, axs = plt.subplots(2, 2, figsize=(9, 7))
    letters = iter('abcd')
    for row, w in enumerate(['fingerprint', 'ch']):
        X = data[w]
        T_all = PCA(n_components=N_COMP, svd_solver='full').fit_transform(X)
        accs = {'PCA-LDA': [], 'PLS-DA': []}
        for k, ev_sites in enumerate(labelings):
            lab = dict(zip(SITE_IDS, [0 if i in ev_sites else 1 for i in range(len(SITE_IDS))]))
            yp = np.array([lab[s] for s in sites])
            fp = make_folds(yp)
            accs['PCA-LDA'].append((cv_pca_lda(X, yp, fp, T=T_all) == yp).mean() * 100)
            accs['PLS-DA'].append((cv_plsda(X, yp, fp) == yp).mean() * 100)
            if (k + 1) % 100 == 0:
                print(f'  {w}: {k + 1}/{len(labelings)}')
        for col, model in enumerate(['PCA-LDA', 'PLS-DA']):
            a = np.array(accs[model])
            obs = RES[w]['cv'][model]['acc']
            n_ge = int(np.sum(a >= obs - 1e-9))
            RES[w].setdefault('perm', {})[model] = {'p': n_ge / len(a), 'median': np.median(a),
                                                    'p95': np.percentile(a, 95), 'n_ge': n_ge,
                                                    'n': len(a)}
            report(f'  {PLAIN[w]} {model}: shuffled median {np.median(a):.1f} %, 95th percentile '
                   f'{np.percentile(a, 95):.1f} %; {n_ge}/{len(a)} >= real -> p = {n_ge / len(a):.4f}')
            ax = axs[row, col]
            ax.hist(a, bins=np.arange(0, 102.5, 2.5), color='0.6', edgecolor='white')
            ax.axvline(obs, color='#B5541C', lw=2)
            ax.text(obs - 1, ax.get_ylim()[1] * 0.92, f'real\n{obs:.1f}%', ha='right', va='top',
                    color='#B5541C', fontsize=10)
            ax.set_xlim(0, 100)
            ax.set_xlabel('Cross-validated accuracy (%)')
            ax.set_ylabel('Number of site labelings')
            ax.set_title(f'{model} - {SHORT[w]}\np = {n_ge / len(a):.4f} ({len(a)} labelings)',
                         fontsize=11)
            ax.text(-0.18, 1.08, f'({next(letters)})', transform=ax.transAxes, fontsize=14,
                    va='bottom')
    fig.tight_layout()
    save_figure(fig, 'FigS1_Permutation')
else:
    report('\n7. permutation test skipped (N_PERM_MODE = "skip")')


# ============================================================================
# 8. TABLES
# ============================================================================
report('\n8. TABLES')
perf = []
for w in ['fingerprint', 'ch']:
    for model in ['PCA-LDA', 'PLS-DA']:
        c = RES[w]['cv'][model]
        row = {'Region': PLAIN[w], 'Model': model, 'Accuracy (%)': round(c['acc'], 1),
               'Sensitivity NV (%)': round(c['sens'], 1), 'Specificity EV (%)': round(c['spec'], 1),
               'EVs correct': f"{c['cm'][0, 0]}/{c['cm'][0].sum()}",
               'NVs correct': f"{c['cm'][1, 1]}/{c['cm'][1].sum()}"}
        if 'perm' in RES[w]:
            row['Permutation p (924 labelings)'] = round(RES[w]['perm'][model]['p'], 4)
        perf.append(row)
pd.DataFrame(perf).to_csv(OUTPUT_DIR / 'Table1_Model_performance.csv', index=False)

TABLE_S1 = [('519', 510, 530, 'fingerprint'), ('826', 820, 835, 'fingerprint'),
            ('855', 850, 862, 'fingerprint'), ('1001', 995, 1008, 'fingerprint'),
            ('1031', 1025, 1040, 'fingerprint'), ('1160', 1150, 1175, 'fingerprint'),
            ('1245', 1235, 1255, 'fingerprint'), ('1380', 1340, 1420, 'fingerprint'),
            ('1520', 1505, 1535, 'fingerprint'), ('1580', 1570, 1590, 'fingerprint'),
            ('1605', 1598, 1615, 'fingerprint'), ('1725', 1710, 1735, 'fingerprint'),
            ('2850', 2840, 2860, 'ch'), ('2880', 2870, 2890, 'ch'), ('2930', 2920, 2940, 'ch'),
            ('2960', 2950, 2970, 'ch'), ('3060', 3050, 3075, 'ch')]
s1 = []
for band, lo, hi, w in TABLE_S1:            # mean SNV intensity in the interval, per site
    gf = WINDOWS[w]['grid_full']
    v = full[w][:, (gf >= lo) & (gf <= hi)].mean(1)
    sm = site_means(v)
    p = ttest_ind(sm[0], sm[1], equal_var=False).pvalue
    s1.append({'Band (cm-1)': band, 'Interval (cm-1)': f'{lo}-{hi}',
               'EV mean': round(sm[0].mean(), 3), 'EV SD': round(sm[0].std(ddof=1), 3),
               'NV mean': round(sm[1].mean(), 3), 'NV SD': round(sm[1].std(ddof=1), 3),
               'Higher in': GROUPS[0] if sm[0].mean() > sm[1].mean() else GROUPS[1],
               'p (Welch)': float(f'{p:.3g}'), 'p (Bonferroni x17)': float(f'{min(1.0, p * 17):.3g}')})
S1 = pd.DataFrame(s1)
S1.to_csv(OUTPUT_DIR / 'TableS1_Band_statistics.csv', index=False)
pd.DataFrame(peak_table).to_csv(OUTPUT_DIR / 'TableS3_Peak_positions.csv', index=False)
pd.DataFrame(shift_table).to_csv(OUTPUT_DIR / 'TableS4_Peak_shifts.csv', index=False)
pd.DataFrame(band_rows).to_csv(OUTPUT_DIR / 'TableS5_Main_bands.csv', index=False)
report('  Table 1, Tables S1-S5 and the score tables saved')


# ============================================================================
# 9. MODEL DETAILS AND RESULTS TEXT
# ============================================================================
md = ['MODEL DETAILS - Acacia EVs vs NVs on Ag/SiNWs (generated by Acacia_Paper_All.py)',
      '=' * 78,
      f'Software: Python {sys.version.split()[0]}, numpy {np.__version__}, scikit-learn '
      f'{sklearn.__version__}, matplotlib {matplotlib.__version__}',
      '', 'DATA',
      f'  {np.sum(y == 0)} EV and {np.sum(y == 1)} NV SERS spectra; 6 measurement sites per group, '
      f'20 spectra per site (sites = independent units).',
      '', 'PRE-PROCESSING (each spectrum, each window separately)',
      f'  1. cosmic-ray removal: modified z-score of first differences > {DESPIKE_Z:g}; '
      f'spike replaced by the mean of non-spike points within +/-{DESPIKE_M}',
      '  2. linear interpolation onto a 2 cm-1 grid: 205-1795 cm-1 (fingerprint), 2705-3095 cm-1 (CH)',
      f'  3. Savitzky-Golay smoothing: {SG_WINDOW} points, polynomial order {SG_ORDER}',
      f'  4. baseline: asymmetric least squares, lambda = {ALS_LAM:g}, p = {ALS_P}, {ALS_ITER} iterations',
      '  5. SNV within the window (mean 0, SD 1 per spectrum)',
      '  analysis regions: 250-1790 cm-1 (770 points) and 2760-3095 cm-1 (168 points)',
      '', 'PCA',
      f'  {N_COMP} components, exact SVD (svd_solver = "full"), mean-centred, all 240 spectra, '
      f'labels not used. PC1 sign: loading positive at {PC1_REF["fingerprint"]} / '
      f'{PC1_REF["ch"]} cm-1.',
      '', 'PCA-LDA',
      f'  LDA (scikit-learn LinearDiscriminantAnalysis, default SVD solver, equal-covariance) on '
      f'PC1-PC{N_COMP}; priors = class proportions (0.5/0.5).',
      '  Fig 3 b/d: LDA fitted on all spectra (display). Canonical variable 1 oriented NVs > 0.',
      '  Statistics: Welch t-test on the 6 + 6 site means of canonical variable 1.',
      '', 'PLS-DA',
      f'  autoscaling (StandardScaler, fitted on training folds), PLS regression with {N_COMP} latent '
      f'variables on y = 0 (EV) / 1 (NV), class NV if prediction > {PLS_THRESHOLD}.',
      '', 'VALIDATION',
      f'  StratifiedGroupKFold: {N_FOLDS} folds, shuffle = True, random_state = {CV_SEED}, groups = '
      f'measurement sites (a site is never in training and test at the same time).']
for k, (tr, te) in enumerate(folds, 1):
    md.append(f'    fold {k}: test = ' + ', '.join(sorted(set(sites[te]))) + f' ({len(te)} spectra)')
md += ['  Stricter check: PCA re-fitted inside every training fold.',
       '  Permutation test: all 924 assignments of 6 of the 12 sites to "EV", full '
       'cross-validation repeated for each; p = labelings with accuracy >= real / 924.',
       '  Sensitivity = NVs correctly classified; specificity = EVs correctly classified.',
       '', 'RESULTS']
for w in ['fingerprint', 'ch']:
    r = RES[w]
    md.append(f'  {PLAIN[w]}: explained variance PC1-PC{N_COMP} = ' +
              ', '.join(f'{v:.1f}' for v in r['var']) + f' % (sum {r["var"].sum():.1f} %)')
    for model in ['PCA-LDA', 'PLS-DA']:
        c = r['cv'][model]
        line = (f'    {model}: accuracy {c["acc"]:.1f} %, sensitivity {c["sens"]:.1f} %, specificity '
                f'{c["spec"]:.1f} %, confusion [[{c["cm"][0, 0]} {c["cm"][0, 1]}] '
                f'[{c["cm"][1, 0]} {c["cm"][1, 1]}]]')
        if 'perm' in r:
            line += f', permutation p = {r["perm"][model]["p"]:.4f}'
        md.append(line)
    md.append(f'    PCA-LDA with PCA re-fitted per fold: {r["strict"]:.1f} %; '
              f'LD1 site-mean Welch p = {r["p_ld1"]:.2e}')
(OUTPUT_DIR / 'Model_details.txt').write_text('\n'.join(md) + '\n', encoding='utf-8')

other = {GROUPS[0]: GROUPS[1], GROUPS[1]: GROUPS[0]}
txt = ['RESULTS - Acacia EVs vs NVs on Ag/SiNWs (generated by Acacia_Paper_All.py)', '=' * 78, '']
cons = [r for r in shift_table if r['Consistent']]
txt += ['Mean spectra (Fig 2). Peak shifts consistent across all 6 + 6 sites: ' +
        ', '.join(f"{r['EV (cm-1)']} -> {r['NV (cm-1)']} cm-1 ({r['Shift NV-EV (cm-1)']:+})" for r in cons) +
        ' (EV -> NV).', '']
sig_bands = S1[S1['p (Bonferroni x17)'] < 0.05]
txt += [f'Band statistics (Table S1, Welch on site means, Bonferroni x17): {len(sig_bands)} of '
        f'{len(S1)} bands differ (p < 0.05): ' +
        ', '.join(f"{b['Band (cm-1)']} ({b['Higher in']} higher)" for _, b in sig_bands.iterrows()) + '.', '']
for w in ['fingerprint', 'ch']:
    r = RES[w]
    v = r['var']
    txt += [f'--- {PLAIN[w].upper()} ---',
            f'PCA (Fig 3): PC1 {v[0]:.1f}%, PC2 {v[1]:.1f}% (PC1-PC5 {v.sum():.1f}%). EVs and NVs separate '
            f'along PC1 (mean scores EVs {r["pc1_mean"][0]:+.2f}, NVs {r["pc1_mean"][1]:+.2f}) although '
            f'the labels were not used.',
            'PC1 loading bands (Fig 4): ' +
            ', '.join(f'{wn:.0f} ({grp} higher)' for wn, _, grp in r['load_bands']) + '.',
            f'Difference spectrum NV - EV (Fig 4): > 2 SE at {r["diff_sig_frac"]:.0f}% of the wavenumbers; '
            'largest: ' + ', '.join(f'{wn:.0f} ({"NVs" if val > 0 else "EVs"} higher)'
                                    for wn, val in r['diff_bands']) + '.',
            f'Canonical variable 1 (Fig 3, display): median {r["ld_median"][0]:+.2f} (EVs) vs '
            f'{r["ld_median"][1]:+.2f} (NVs); Welch on site means p = {r["p_ld1"]:.1e}.']
    for model in ['PCA-LDA', 'PLS-DA']:
        c = r['cv'][model]
        line = (f'{model} (Fig 5, sites left out): accuracy {c["acc"]:.1f}% '
                f'({c["cm"][0, 0] + c["cm"][1, 1]}/{c["cm"].sum()}), sensitivity {c["sens"]:.1f}%, '
                f'specificity {c["spec"]:.1f}%')
        if 'perm' in r:
            q = r['perm'][model]
            line += (f'; shuffled labels: median {q["median"]:.1f}%, 95th percentile {q["p95"]:.1f}%, '
                     f'p = {q["p"]:.4f}')
        txt.append(line + '.')
    txt += [f'PCA-LDA with PCA re-fitted per fold: {r["strict"]:.1f}%.', '']
txt += ['--- LIMITATIONS TO STATE ---',
        '* Independent units are the 6 + 6 measurement sites, not the 240 spectra; the smallest '
        'possible permutation p is 2/924 = 0.0022.',
        '* If all EV spectra come from one preparation and all NV spectra from another, '
        'preparation or measurement-session effects cannot be separated from EV-vs-NV biology.',
        '* After SNV, intensities are relative (composition), not absolute amounts.',
        '* Fig 3 b/d (LDA on all spectra) shows the separation; the classification performance is '
        'the cross-validation (Fig 5, Table 1).']
(OUTPUT_DIR / 'RESULTS_for_manuscript.txt').write_text('\n'.join(txt) + '\n', encoding='utf-8')
(OUTPUT_DIR / 'Run_log.txt').write_text('\n'.join(LOG) + '\n', encoding='utf-8')
print('\n' + '\n'.join(md) + '\n\n' + '\n'.join(txt))
print(f'\nAll outputs saved in: {OUTPUT_DIR}')
if SHOW_WINDOW:
    plt.show()
