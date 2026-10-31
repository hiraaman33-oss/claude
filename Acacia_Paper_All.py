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
#   Fig2_label_positions.csv    every Fig 2 label position (from the data)
#   Fig3_PCA_LDA                PCA scores + LDA canonical variable (a-d)
#   Fig4_Loadings_Difference    PC1/PC2 loadings (a, b), NV - EV difference (c, d)
#   Fig5_ConfusionMatrices      cross-validated, PCA-LDA and PLS-DA (a-d)
#   FigS1_Permutation           exact label-shuffling test (a-d)
#   Fig2c_PeakFit_CH_GL         CH-region band fitting, EV and NV, Gaussian-Lorentzian sum
#   FigS2_PeakFit_CH            the same fitting with all 3 line shapes (robustness)
#   Table1_Model_performance.csv     accuracy, sensitivity, specificity, p
#   TableS1_Band_statistics.csv      17 bands, Welch on site means, Bonferroni
#   TableS2_Fold_results.csv         every cross-validation fold
#   TableS3_Peak_positions.csv       labelled peaks of Fig 1 and Fig 2
#   TableS4_Peak_shifts.csv          EV/NV peak positions per site, shifts
#   TableS5_Main_bands.csv           main loading and difference bands
#   TableS6_PeakFit_CH.csv           fitted bands: position, FWHM, % area, fit quality (R2, AIC, BIC)
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
from scipy.optimize import least_squares
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
PCA_PER_FOLD = True                         # PCA-LDA: True = PCA fitted on the training folds only
                                            # (main result); False = PCA once on all 240 spectra
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
    simple = lambda n: n.lower().replace(' ', '_').replace('%', '')   # 'S2-EV Acacia ... 10%' -> 's2-ev_acacia ... 10'
    hits = [f for f in SINGLE_DIR.iterdir() if f.is_file() and f.suffix.lower() in ('.txt', '')
            and simple(f.name).startswith(prefix)]
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
    fs_peak, fs_name, fs_axis, fs_tick = 12, 15, 17, 15     # readable at one-column width
    fig, (axL, axR) = plt.subplots(1, 2, sharey=True, figsize=(9, 6.5),
                                   gridspec_kw={'width_ratios': [hi1 - lo1, hi2 - lo2], 'wspace': 0.03})

    def in_view(d):
        xx = d[:, 0]
        return d[((xx >= lo1) & (xx <= hi1)) | ((xx >= lo2) & (xx <= hi2)), 1]

    y_min = min(in_view(caf2).min(), in_view(ag).min())
    y_max = max(in_view(caf2).max(), in_view(ag).max())
    y_top = y_max + 0.70 * (y_max - y_min)
    axL.set_ylim(y_min - 0.12 * (y_max - y_min), y_top)
    for ax, lo, hi in [(axL, lo1, hi1), (axR, lo2, hi2)]:
        for d, col in [(caf2, '#3A3A3A'), (ag, '#E8383D')]:
            m = (d[:, 0] >= lo) & (d[:, 0] <= hi)
            ax.plot(d[m, 0], d[m, 1], color=col, lw=0.8)
        ax.set_xlim(lo, hi)
        pos = [r[0] for r in rows]
        for k, (measured, cls) in enumerate(rows):
            if not lo <= measured <= hi:
                continue
            near_left = k > 0 and measured - pos[k - 1] < 45
            ha = 'left' if (near_left or measured - lo < 30) else 'right'
            t = ax.text(measured, y_top - 0.01 * (y_top - y_min), f'{measured}  {cls}', rotation=90,
                        ha=ha, va='top', fontsize=fs_peak, color=SINGLE_COL[cls])
            fig.canvas.draw()
            bottom = t.get_window_extent().transformed(ax.transData.inverted()).y0
            peak_top = ag[np.abs(ag[:, 0] - measured) <= 3, 1].max()
            ax.plot([measured, measured], [peak_top + 0.01 * (y_top - y_min), bottom],
                    color='k', ls=':', lw=1.0)
        ax.tick_params(direction='in', top=True, length=6, width=1.0, labelsize=fs_tick)
    axL.spines['right'].set_visible(False)
    axR.spines['left'].set_visible(False)
    axL.tick_params(labelleft=False)
    axR.tick_params(left=False, right=True, labelleft=False)
    axL.set_xticks([400, 600, 800, 1000, 1200, 1400, 1600])
    axR.set_xticks([2800, 3000])
    for d_, col, txt in [(ag, '#E8383D', f'{SINGLE_NAME[tag]} on Ag/SiNWs'),
                         (caf2, '#3A3A3A', f'{SINGLE_NAME[tag]} on CaF$_2$')]:
        m = (d_[:, 0] >= 420) & (d_[:, 0] <= 760)
        axL.text(420, d_[m, 1].min() - 0.015 * (y_top - y_min), txt, color=col, fontsize=fs_name,
                 ha='left', va='top')
    dd = 0.012
    for ax, x0 in [(axL, 1), (axR, 0)]:
        dx = dd * (hi2 - lo2) / (hi1 - lo1) if ax is axL else dd
        for y0 in (0, 1):
            ax.plot((x0 - dx, x0 + dx), (y0 - 2 * dd, y0 + 2 * dd), transform=ax.transAxes,
                    color='k', clip_on=False, lw=1.0)
    axL.set_ylabel('Raman Intensity (a.u.)', fontsize=fs_axis)
    fig.text(0.53, 0.015, 'Wavenumber (cm$^{-1}$)', ha='center', fontsize=fs_axis)
    fig.subplots_adjust(left=0.07, right=0.97, bottom=0.12, top=0.97)
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
# bands labelled in Fig 2b, per group (all bands go to Table S4): the CH2 shoulder
# is found in both groups (2nd derivative, mean spectrum and all 6 sites of each
# group): weaker in EVs (~2874 cm-1), stronger and shifted in NVs (~2883 cm-1)
CH_SHOW = {0: [2880, 2935], 1: [2880, 2935]}
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
plt.close(fig)     # positions above -> Tables S3/S4; the figure itself is drawn in section 3c

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
    if consistent and nominal in CH_SHOW[0] and nominal in CH_SHOW[1]:
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
        if cc != c or pos is None or nominal not in CH_SHOW[c]:
            continue
        top = (mu + sd)[np.abs(g - pos) <= 6].max() + off
        txt = f'{pos:.0f}'
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
plt.close(fig)     # drawn in section 3c
for r in shift_table:
    if r['Consistent']:
        report(f"  consistent shift ({r['Region']}): {r['EV (cm-1)']} -> {r['NV (cm-1)']} cm-1 "
               f"({r['Shift NV-EV (cm-1)']:+})")


# ----------------------------------------------------------------------------
# FIGURE S2 - band fitting of the CH-stretching region (mean spectra)
# ----------------------------------------------------------------------------
# The mean SNV spectrum of each group (2780-3095 cm-1) is described as a straight
# baseline + 5 bands (I-V, starting values below). Each band has a free height,
# position (+/-12 cm-1) and FWHM (12-60 cm-1); least-squares fit (scipy). Three
# line shapes are compared, as is usual for band fitting:
#   G     Gaussian
#   GL    Gaussian-Lorentzian sum (pseudo-Voigt), one mixing fraction per spectrum
#   L     Lorentzian
# Overlapping bands do not have a unique decomposition, so the % areas depend on
# the line shape; conclusions should rest on what is the same for all three.
# Residual = data - fit, in % of the fitted band height.
FIT_RANGE = (2780, 3095)
FIT_BANDS = [(2850, 'I', 'CH$_2$ sym.'), (2880, 'II', 'CH$_2$ asym.'), (2935, 'III', 'CH$_3$ sym.'),
             (2965, 'IV', 'CH$_3$ asym.'), (3065, 'V', '=C-H arom.')]
FIT_SHAPES = {'G': 'Gaussian', 'GL': 'Gaussian-Lorentzian sum', 'L': 'Lorentzian'}


def band_profile(x, c, w, eta):
    return eta / (1 + 4 * ((x - c) / w) ** 2) + (1 - eta) * np.exp(-4 * np.log(2) * ((x - c) / w) ** 2)


def band_area(h, w, eta):
    return h * w * (eta * np.pi / 2 + (1 - eta) * np.sqrt(np.pi / (4 * np.log(2))))


def fit_bands(x, yv, shape):
    nb = len(FIT_BANDS)
    p0, lo, hi = [yv.min(), 0.0], [-5, -5], [5, 5]
    for c, _, _ in FIT_BANDS:
        p0 += [1.0, c, 30]
        lo += [0, c - 12, 12]
        hi += [20, c + 12, 60]
    if shape == 'GL':
        p0, lo, hi = p0 + [0.5], lo + [0], hi + [1]

    def model(p):
        eta = {'G': 0.0, 'L': 1.0}.get(shape, p[-1])
        base = p[0] + p[1] * (x - 2940) / 100
        comps = [p[2 + 3 * k] * band_profile(x, p[3 + 3 * k], p[4 + 3 * k], eta) for k in range(nb)]
        return base, comps, eta

    r = least_squares(lambda p: model(p)[0] + sum(model(p)[1]) - yv, p0, bounds=(lo, hi))
    base, comps, eta = model(r.x)
    return r.x, base, comps, eta


# ============================================================================
# 3c. FIGURE 2a / 2b - final figures: data-derived band labels, large fonts
# ============================================================================
# Label positions from the group mean spectra: peak = most prominent local
# maximum in the window; shoulder = most prominent minimum of the 2nd
# derivative (Savitzky-Golay 11, order 3); parabola-refined, rounded.
# Labels are placed clear of the curves, the SD bands, the legend and the frame.
report('\n3c. FIGURE 2a / 2b - band labels')
F2_GRID = {w: WINDOWS[w]['grid_full'] for w in WINDOWS}
OFFSET = {'fingerprint': 9.0, 'ch': 5.5}       # SNV units added to the NV spectrum
TOP_ROOM = {'fingerprint': 4.8, 'ch': 4.4}     # space above the NV spectrum for labels
# font sizes (pt): band labels, axis titles, tick numbers
LABEL_FS, AXIS_FS, TICK_FS = 10, 13, 10

# class colours of the label text
PURPLE, BLUE, DARKBLUE = '#7030A0', '#2E75B6', '#1F3864'
GOLD, GREEN, GREY = '#9C6500', '#3A7D22', '#404040'
CLASS = {'Proteins': [('Proteins', PURPLE)],
         'Proteins (COO$^{-}$)': [('Proteins (COO$^{-}$)', PURPLE)],
         'Lipids': [('Lipids', BLUE)],
         'Nucleic Acids': [('Nucleic Acids', DARKBLUE)],
         'Lipids/NA': [('Lipids/', BLUE), ('NA', 'k')],
         'Lipids/Proteins': [('Lipids/', BLUE), ('Proteins', PURPLE)],
         'Proteins/Lipids': [('Proteins/', PURPLE), ('Lipids', BLUE)],
         # functional-group labels (LABEL_SET = 'revised')
         'Carotenoids/Phe': [('Carotenoids/', GOLD), ('Phe', PURPLE)],
         'Carotenoids': [('Carotenoids', GOLD)],
         'COO- (sym.)': [('COO$^{-}$ (sym.)', GREEN)],
         'Aromatic C=C': [('Arom. C=C', PURPLE)],
         'Amide III': [('Amide III', PURPLE)],
         'O-P-O': [('O\u2013P\u2013O', DARKBLUE)],
         'Ester C=O': [('Ester C=O', BLUE)],
         'CH2 (lipids)': [('CH$_2$ (lipids)', BLUE)],
         'CH3': [('CH$_3$', GREY)],
         # Acacia labels (LABEL_SET = 'acacia')
         'Phe': [('Phe', PURPLE)],
         'Tyr': [('Tyr', PURPLE)],
         'Phenolics': [('Phenolics', GOLD)],
         'Phenolics/Tyr': [('Phenolics/', GOLD), ('Tyr', PURPLE)],
         'Polysacch./Tyr': [('Polysacch./', GREEN), ('Tyr', PURPLE)],
         'C-COO-': [('C\u2013COO$^{-}$', GREEN)],
         # general class names (LABEL_SET = 'general')
         'Phenolics/Proteins': [('Phenolics/', GOLD), ('Proteins', PURPLE)],
         'Polysacch./Proteins': [('Polysacch./', GREEN), ('Proteins', PURPLE)],
         None: []}

# ----------------------------------------------------------------------------
# LABELS: (region, group 0=EV 1=NV, kind 'peak'/'shoulder', window lo, hi,
#          text format, class, required)
# text format: {p} = position; '~' and 'sh' are written as given
# required=False: label only if a local extremum exists in the window
# ----------------------------------------------------------------------------
LABELS = [
    # --- Fig 2a, fingerprint (exactly the labels of the approved figure)
    ('fingerprint', 0, 'peak', 507, 531, '{p:.1f}', None, True),          # Si 518.5
    ('fingerprint', 0, 'peak', 797, 821, '{p}', 'Nucleic Acids', True),   # 809
    ('fingerprint', 0, 'peak', 843, 867, '{p}', 'Proteins', True),        # 855
    ('fingerprint', 0, 'peak', 919, 943, '{p}', 'Proteins', True),        # 932
    ('fingerprint', 0, 'peak', 989, 1013, '{p}', 'Proteins', True),       # 1002
    ('fingerprint', 0, 'peak', 1149, 1173, '{p}', 'Proteins', True),      # 1162
    ('fingerprint', 0, 'peak', 1237, 1261, '{p}', 'Proteins', True),      # 1249
    ('fingerprint', 0, 'peak', 1361, 1385, '{p}', 'Lipids/NA', True),     # 1373
    ('fingerprint', 0, 'peak', 1591, 1615, '{p}', 'Proteins', True),      # 1603
    ('fingerprint', 1, 'peak', 507, 531, '{p:.1f}', None, True),          # Si 518.1
    ('fingerprint', 1, 'peak', 787, 811, '{p}', 'Nucleic Acids', True),   # 799
    ('fingerprint', 1, 'peak', 817, 841, '{p}', 'Proteins', True),        # 829
    ('fingerprint', 1, 'peak', 915, 939, '{p}', 'Proteins', True),        # 927
    ('fingerprint', 1, 'peak', 989, 1013, '{p}', 'Proteins', True),       # 1001
    ('fingerprint', 1, 'peak', 1021, 1045, '{p}', 'Proteins', True),      # 1033
    ('fingerprint', 1, 'peak', 1157, 1181, '{p}', 'Proteins', True),      # 1169
    ('fingerprint', 1, 'peak', 1233, 1257, '{p}', 'Proteins', True),      # 1244
    ('fingerprint', 1, 'peak', 1373, 1397, '{p}', 'Lipids/NA', True),     # 1385
    ('fingerprint', 1, 'peak', 1595, 1619, '{p}', 'Proteins', True),      # 1607
    ('fingerprint', 1, 'peak', 1717, 1741, '{p}', 'Lipids', True),        # 1728
    # --- Fig 2b, CH region: main CH3 band
    ('ch', 0, 'peak', 2923, 2947, '{p}', 'Proteins', True),               # 2933
    ('ch', 1, 'peak', 2923, 2947, '{p}', 'Proteins', True),               # 2937
]
# ON/OFF SWITCHES for optional labels (True = label shown, False = not shown)
SHOW_CH_SHOULDERS = True     # Fig 2b: CH2 lipid shoulder, EV 2874 / NV 2883 (+ grey shadow)
SHOW_EXTRA_PEAKS = False     # Fig 2a: further peaks present in the mean spectra at all sites
if SHOW_CH_SHOULDERS:
    LABELS += [('ch', 0, 'shoulder', 2868, 2892, '{p}', 'Lipids', True),    # 2874
               ('ch', 1, 'shoulder', 2868, 2892, '{p}', 'Lipids', True)]    # 2883
if SHOW_EXTRA_PEAKS:
    LABELS += [('fingerprint', 0, 'peak', 627, 651, '{p}', 'Proteins', True),        # 639 Tyr
               ('fingerprint', 0, 'peak', 757, 781, '{p}', 'Nucleic Acids', True),   # 769
               ('fingerprint', 0, 'peak', 1061, 1085, '{p}', 'Lipids', True),        # 1073 C-C / C-O
               ('fingerprint', 1, 'peak', 655, 679, '{p}', 'Nucleic Acids', True),   # 667 G/T ring
               ('fingerprint', 1, 'peak', 1063, 1087, '{p}', 'Lipids', True),        # 1075 C-C / C-O
               ('fingerprint', 1, 'peak', 1275, 1299, '{p}', 'Proteins', True)]      # 1287 amide III
# WHICH LABEL WORDS (positions are the same in all three):
#   'acacia'   = assignments for an Acacia flower sample: phenolics/flavonoids, uronic-acid
#                carboxylates and cell-wall polysaccharides besides proteins and lipids;
#                no carotenoids, because the strongest carotenoid band (~1520 cm-1) is
#                absent at all 6 EV sites although the ~1160 cm-1 band is clear
#   'revised'  = functional groups with carotenoids at 1001/1160 (manuscript v7 text)
#   'original' = the first version (Proteins / Lipids / Lipids/NA / Nucleic Acids)
#   'general'  = class names only (Proteins, Nucleic Acids, Lipids, Phenolics, Polysacch.)
#                for an Acacia sample; vibrations and references in the band table (Word)
LABEL_SET = 'general'
REVISED = {('fingerprint', 797): 'O-P-O', ('fingerprint', 787): 'O-P-O',            # 809 / 799
           ('fingerprint', 989): 'Carotenoids/Phe',                                 # 1002 / 1001
           ('fingerprint', 1149): 'Carotenoids', ('fingerprint', 1157): 'Carotenoids',  # 1162 / 1169
           ('fingerprint', 1237): 'Amide III', ('fingerprint', 1233): 'Amide III',  # 1249 / 1244
           ('fingerprint', 1361): 'COO- (sym.)', ('fingerprint', 1373): 'COO- (sym.)',  # 1373 / 1385
           ('fingerprint', 1591): 'Aromatic C=C', ('fingerprint', 1595): 'Aromatic C=C',  # 1603 / 1607
           ('fingerprint', 1717): 'Ester C=O',                                      # 1728
           ('ch', 2923): 'CH3', ('ch', 2868): 'CH2 (lipids)'}                       # 2933/2937, 2874/2883
ACACIA = {('fingerprint', 797): 'O-P-O', ('fingerprint', 787): 'O-P-O',           # 809 / 799
          ('fingerprint', 843): 'Polysacch./Tyr',                                  # 855 (EV)
          ('fingerprint', 817): 'Tyr',                                             # 829 (NV)
          ('fingerprint', 919): 'C-COO-', ('fingerprint', 915): 'C-COO-',          # 932 / 927
          ('fingerprint', 989): 'Phe', ('fingerprint', 1021): 'Phe',               # 1002/1001, 1033
          ('fingerprint', 1149): 'Phenolics/Tyr', ('fingerprint', 1157): 'Phenolics/Tyr',  # 1162 / 1169
          ('fingerprint', 1237): 'Amide III', ('fingerprint', 1233): 'Amide III',  # 1249 / 1244
          ('fingerprint', 1361): 'COO- (sym.)', ('fingerprint', 1373): 'COO- (sym.)',  # 1373 / 1385
          ('fingerprint', 1591): 'Phenolics', ('fingerprint', 1595): 'Phenolics',  # 1603 / 1607 (minor Phe)
          ('fingerprint', 1717): 'Ester C=O',                                      # 1728
          ('ch', 2923): 'CH3', ('ch', 2868): 'CH2 (lipids)'}                       # 2933/2937, 2874/2883
GENERAL = {('fingerprint', 843): 'Polysacch./Proteins',                          # 855 (EV)
           ('fingerprint', 1149): 'Phenolics/Proteins', ('fingerprint', 1157): 'Phenolics/Proteins',  # 1162 / 1169
           ('fingerprint', 1361): 'Polysacch./Proteins', ('fingerprint', 1373): 'Polysacch./Proteins',  # 1373 / 1385
           ('fingerprint', 1591): 'Phenolics', ('fingerprint', 1595): 'Phenolics'}  # 1603 / 1607 (minor proteins)
if LABEL_SET in ('revised', 'acacia', 'general'):
    NEW = {'revised': REVISED, 'acacia': ACACIA, 'general': GENERAL}[LABEL_SET]
    LABELS = [lab[:6] + (NEW.get((lab[0], lab[3]), lab[6]),) + lab[7:] for lab in LABELS]
# grey shadows: bands compared between EVs and NVs (EV window, NV window, kind);
# drawn only if the shift is consistent at all sites (site ranges do not overlap)
SHIFTS = {'fingerprint': [((507, 531), (507, 531), 'peak'), ((797, 821), (787, 811), 'peak'),
                          ((919, 943), (915, 939), 'peak'), ((989, 1013), (989, 1013), 'peak'),
                          ((1149, 1173), (1157, 1181), 'peak'), ((1237, 1261), (1233, 1257), 'peak'),
                          ((1361, 1385), (1373, 1397), 'peak'), ((1591, 1615), (1595, 1619), 'peak')],
          'ch': [((2923, 2947), (2923, 2947), 'peak')]}
if SHOW_CH_SHOULDERS:
    SHIFTS['ch'].insert(0, ((2868, 2892), (2868, 2892), 'shoulder'))


def f2_band_position(g, s, lo, hi, kind):
    """Most prominent local maximum ('peak') or 2nd-derivative minimum ('shoulder')
    inside [lo, hi], refined by a parabola. None if there is none."""
    if kind == 'shoulder':
        s = -savgol_filter(s, 11, 3, deriv=2)
    m = np.flatnonzero((g >= lo) & (g <= hi))
    pk, pr = find_peaks(s[m], prominence=0)
    if not len(pk):
        return None
    i = m[pk[np.argmax(pr['prominences'])]]
    a, b, c = s[i - 1], s[i], s[i + 1]
    return g[i] + 0.5 * (a - c) / (a - 2 * b + c) * (g[1] - g[0])


f2_rows = []
both1728 = all(f2_band_position(F2_GRID['fingerprint'], full['fingerprint'][y == c].mean(0), 1715, 1740,
                             'peak') is not None for c in (0, 1))
for region, c, kind, lo, hi, fmt, cls, required in LABELS:
    pos = f2_band_position(F2_GRID[region], full[region][y == c].mean(0), lo, hi, kind)
    if required == 'both1728':
        draw = both1728
        note = '' if both1728 else 'not drawn: no local maximum in both EV and NV'
    elif pos is None:
        draw, note = False, 'not drawn: no local extremum in window'
    else:
        draw, note = True, ''
    text = None
    if pos is not None:
        text = fmt.format(p=pos) if '.1f' in fmt else fmt.format(p=int(round(pos)))
    f2_rows.append({'Figure': '2a' if region == 'fingerprint' else '2b', 'Group': GROUPS[c],
                 'Kind': kind, 'Window (cm-1)': f'{lo}-{hi}',
                 'Position (cm-1)': '' if pos is None else round(pos, 1),
                 'Label': '' if text is None else (text + ('- ' + cls.replace('$^{-}$', '-') if cls else '')),
                 'Drawn': draw, 'Note': note, '_region': region, '_c': c, '_pos': pos, '_text': text,
                 '_cls': cls})
table = pd.DataFrame(f2_rows)
f2_show = table.drop(columns=[k for k in table.columns if k.startswith('_')])
pd.set_option('display.width', 200)
pd.set_option('display.max_columns', 20)
print('\nLABEL POSITIONS (from the mean spectra)')
print(f2_show.to_string(index=False))
f2_show.to_csv(OUTPUT_DIR / 'Fig2_label_positions.csv', index=False)


def shift_position(g, s, lo, hi, kind):
    """Band position used for the grey shift shadows (same method as Table S4):
    peak = maximum in the window, refined by a parabola (also valid at the window
    edge); shoulder = 2nd-derivative minimum."""
    if kind == 'shoulder':
        return f2_band_position(g, s, lo, hi, kind)
    m = np.flatnonzero((g >= lo) & (g <= hi))
    i = m[int(np.argmax(s[m]))]
    a, b, c = s[i - 1], s[i], s[i + 1]
    return g[i] + 0.5 * (a - c) / (a - 2 * b + c) * (g[1] - g[0])


def site_positions(region, c, lo, hi, kind):
    return [shift_position(F2_GRID[region], full[region][sites == f'{GROUPS[c]}-{k}'].mean(0), lo, hi, kind)
            for k in SITES]


# ============================================================================
# 3. FIGURES
# ============================================================================
def f2_spread(xs, widths, x_min, x_max):
    """Move label centres apart until neighbouring labels do not overlap."""
    xs = list(xs)
    for _ in range(200):
        moved = False
        for i in range(1, len(xs)):
            need = (widths[i - 1] + widths[i]) / 2 + 2
            if xs[i] - xs[i - 1] < need:
                mid = (xs[i] + xs[i - 1]) / 2
                xs[i - 1], xs[i] = mid - need / 2, mid + need / 2
                moved = True
        xs = [min(max(v, x_min), x_max) for v in xs]
        if not moved:
            break
    return xs


def f2_draw_label(ax, x, ybase, segments, column_dx=0.0):
    """Vertical label, coloured segments stacked upwards; returns (texts, top).
    column_dx > 0: two-column label (number in the first column, class words in a
    second column right of it), used only where a one-column label does not fit."""
    yy = ybase
    yr = ax.get_ylim()[1] - ax.get_ylim()[0]
    texts, top = [], ybase
    for k, (txt, col) in enumerate(segments):
        if column_dx and k == 1:                       # start the second column
            x, yy = x + column_dx, ybase
        xx = x - column_dx / 2 if column_dx and k == 0 else x
        t = ax.text(xx, yy, txt, rotation=90, va='bottom', ha='center', fontsize=LABEL_FS, color=col)
        texts.append(t)
        ax.figure.canvas.draw()
        y1 = t.get_window_extent().transformed(ax.transData.inverted()).y1
        top = max(top, y1)
        # a space after the 'xxxx-' part, the coloured class words joined directly
        yy = y1 + (0.008 * yr if txt.endswith('-') else -0.004 * yr)
    return texts, top


def f2_make_figure(region, xlim, figsize, legend_fs, name):
    g = F2_GRID[region]
    X = full[region]
    off = OFFSET[region]
    show_ = (g >= xlim[0]) & (g <= xlim[1])
    gs = g[show_]
    fig, ax = plt.subplots(figsize=figsize)
    # grey shadows for consistent shifts
    for (ev_w, nv_w, kind) in SHIFTS[region]:
        pe = shift_position(g, X[y == 0].mean(0), *ev_w, kind)
        pn = shift_position(g, X[y == 1].mean(0), *nv_w, kind)
        se = site_positions(region, 0, *ev_w, kind)
        sn = site_positions(region, 1, *nv_w, kind)
        if None in [pe, pn] + se + sn:
            continue
        if max(se) < min(sn) or max(sn) < min(se):
            ax.axvspan(min(pe, pn) - 4, max(pe, pn) + 4, color='0.92', lw=0, zorder=0)
    stats = {}
    for c in (0, 1):
        mu, sd = X[y == c].mean(0)[show_], X[y == c].std(0, ddof=1)[show_]
        o = off if c == 1 else 0.0
        stats[c] = (mu, sd, o)
        ax.fill_between(gs, mu - sd + o, mu + sd + o, color=COLS[c], alpha=0.22, lw=0)
        ax.plot(gs, mu + o, color=COLS[c], lw=1.0,
                label=f'{NAME_LONG[c]} (mean \u00b1 SD, n = {np.sum(y == c)})')
    ax.set_xlim(*xlim)
    ax.set_ylim((X[y == 0].mean(0) - X[y == 0].std(0, ddof=1))[show_].min() - 0.3,
                (X[y == 1].mean(0) + X[y == 1].std(0, ddof=1))[show_].max() + off + TOP_ROOM[region])
    ax.tick_params(axis='y', direction='in', length=5, labelleft=False)
    ax.tick_params(axis='x', direction='in', length=5, labelsize=TICK_FS)
    ax.set_xlabel('Raman Shift (cm$^{-1}$)', fontsize=AXIS_FS)
    ax.set_ylabel('Intensity (a.u.)', fontsize=AXIS_FS)
    ax.legend(frameon=True, facecolor='white', edgecolor='none', framealpha=1, fontsize=legend_fs,
              loc='upper left', handlelength=1.5)
    fig.tight_layout()
    fig.canvas.draw()
    # label width in data units (= height of one rotated text line)
    probe = ax.text(0, 0, '0', rotation=90, fontsize=LABEL_FS)
    fig.canvas.draw()
    bb = probe.get_window_extent().transformed(ax.transData.inverted())
    w_label = bb.x1 - bb.x0
    probe.remove()
    leg_box = ax.get_legend().get_window_extent().transformed(ax.transData.inverted())
    tops = []
    for c in (0, 1):
        mu, sd, o = stats[c]
        sel = table[(table._region == region) & (table._c == c) & table.Drawn].sort_values('_pos')
        if sel.empty:
            continue
        xp = sel._pos.to_numpy(float)
        xl = f2_spread(xp, [w_label] * len(xp), xlim[0] + w_label, xlim[1] - w_label)
        mu1, sd1, o1 = stats[1]
        y_top_axis = ax.get_ylim()[1]

        def base_at(x_peak, x_lab):
            """Label base: just above the SD band under the label (and above the peak
            if the label sits at the peak)."""
            span = np.abs(gs - x_lab) <= w_label
            if abs(x_lab - x_peak) <= w_label:
                span |= np.abs(gs - x_peak) <= 4
            return (mu + sd)[span].max() + o + 0.15

        def leader(x_peak, y_peak, x_end, base):
            """Dotted line from the band on the mean spectrum to the label."""
            if abs(x_end - x_peak) < 1:
                ax.plot([x_peak, x_peak], [y_peak, base], ls=(0, (1, 2)), lw=0.7, color='0.35')
            else:
                ax.plot([x_peak, x_end], [y_peak, base], ls=(0, (1, 2)), lw=0.7, color='0.35')

        def ceiling(x_lab):
            """Highest y a label may reach: below the NV band (for EV labels), the
            legend and the frame."""
            k = np.abs(gs - x_lab) <= w_label
            lim = y_top_axis - 0.1
            if leg_box.x0 - w_label <= x_lab <= leg_box.x1 + w_label:
                lim = min(lim, leg_box.y0 - 0.1)
            if c == 0:
                lim = min(lim, (mu1 - sd1)[k].min() + o1 - 0.1)
            return lim

        for j, ((_, r), x_peak) in enumerate(zip(sel.iterrows(), xp)):
            cls = r._cls if isinstance(r._cls, str) else None     # pandas stores None as NaN
            segs = [(r._text + '-' if cls else r._text, 'k')] + CLASS[cls]
            texts, top = f2_draw_label(ax, xl[j], base_at(x_peak, xl[j]) + 0.05, segs)
            length = top - base_at(x_peak, xl[j])
            if top > ceiling(xl[j]):
                # label would touch the other spectrum: move it sideways (smallest move
                # that fits, without coming closer than one label width to a neighbour)
                for t in texts:
                    t.remove()
                others = [v for i, v in enumerate(xl) if i != j]
                for dx in sorted(np.arange(-120, 121, 2), key=abs):
                    xc = xl[j] + dx
                    if any(abs(xc - v) < w_label + 2 for v in others):
                        continue
                    if base_at(x_peak, xc) + length < ceiling(xc):
                        xl[j] = xc
                        break
                else:
                    # still too tall: write the label in two columns (same font)
                    two_col = w_label * 0.5
                    texts, top = f2_draw_label(ax, xl[j], base_at(x_peak, xl[j]) + 0.05, segs, two_col)
                    length2 = top - base_at(x_peak, xl[j])
                    for t in texts:
                        t.remove()
                    for dx in sorted(np.arange(-120, 121, 2), key=abs):
                        xc = xl[j] + dx
                        if any(abs(xc - v) < 1.5 * w_label + 2 for v in others):
                            continue
                        if base_at(x_peak, xc) + length2 < ceiling(xc):
                            xl[j] = xc
                            break
                    else:
                        print(f'  WARNING {name}: no free place for the label at {x_peak:.0f} cm-1')
                    texts, top = f2_draw_label(ax, xl[j], base_at(x_peak, xl[j]) + 0.05, segs, two_col)
                    x_lab, base = xl[j], base_at(x_peak, xl[j])
                    y_peak = np.interp(x_peak, gs, mu) + o
                    tops.append((c, x_lab, base, top))
                    leader(x_peak, y_peak, x_lab - two_col / 2, base)
                    continue
                texts, top = f2_draw_label(ax, xl[j], base_at(x_peak, xl[j]) + 0.05, segs)
            x_lab, base = xl[j], base_at(x_peak, xl[j])
            y_peak = np.interp(x_peak, gs, mu) + o
            tops.append((c, x_lab, base, top))
            # dotted leader: up from the mean spectrum, then to the label
            leader(x_peak, y_peak, x_lab, base)
    # check: EV labels must stay below the NV spectrum (mean - SD)
    mu1, sd1, o1 = stats[1]
    for c, x_lab, base, top in tops:
        if c == 0:
            k = np.abs(gs - x_lab) <= w_label
            if top > (mu1 - sd1)[k].min() + o1:
                print(f'  WARNING {name}: EV label at {x_lab:.0f} cm-1 reaches the NV spectrum')
    save_figure(fig, name)


print()
plt.rcParams.update({'font.size': TICK_FS})
f2_make_figure('fingerprint', (400, 1790), (8.5, 6.5), 10, 'Fig2a_MeanSD_Fingerprint')
f2_make_figure('ch', (2760, 3095), (6.5, 6.0), 9, 'Fig2b_MeanSD_CH')
plt.rcParams.update({'font.size': 10})


report('\n3b. FIGURE S2 - CH band fitting')
g = WINDOWS['ch']['g']
fr = (g >= FIT_RANGE[0]) & (g <= FIT_RANGE[1])
gx = g[fr]
fit_rows = []
FITS = {}                                   # (group, line shape) -> fitted curves
plt.rcParams.update({'font.size': 13})
fig = plt.figure(figsize=(15, 9.4))
outer = fig.add_gridspec(2, 3, hspace=0.38, wspace=0.28)
letters = iter('abcdef')
for row, c in enumerate((0, 1)):
    yv = data['ch'][y == c].mean(0)[fr]
    for col, shape in enumerate(FIT_SHAPES):
        p, base, comps, eta = fit_bands(gx, yv, shape)
        total = base + sum(comps)
        height = (total - base).max()
        res = (yv - total) / height * 100
        r2 = 1 - np.sum((yv - total) ** 2) / np.sum((yv - yv.mean()) ** 2)
        # AIC / BIC (lower = better; penalise the extra mixing parameter of GL)
        n_par, rss = len(p), np.sum((yv - total) ** 2)
        aic = len(yv) * np.log(rss / len(yv)) + 2 * n_par
        bic = len(yv) * np.log(rss / len(yv)) + n_par * np.log(len(yv))
        FITS[(c, shape)] = (p, base, comps, eta, res, height)
        areas = np.array([band_area(p[2 + 3 * k], p[4 + 3 * k], eta) for k in range(len(FIT_BANDS))])
        for k, (nominal, roman, assign) in enumerate(FIT_BANDS):
            fit_rows.append({'Group': GROUPS[c], 'Line shape': FIT_SHAPES[shape], 'Band': roman,
                             'Assignment': assign.replace('$_2$', '2').replace('$_3$', '3'),
                             'Position (cm-1)': round(p[3 + 3 * k], 1), 'FWHM (cm-1)': round(p[4 + 3 * k], 1),
                             'FWHM at limit (60)': p[4 + 3 * k] > 59.9,
                             'Area (% of CH bands)': round(areas[k] / areas.sum() * 100, 1),
                             'GL Lorentzian fraction': round(eta, 2), 'R2': round(r2, 4),
                             'Max |residual| (%)': round(np.abs(res).max(), 1),
                             'AIC': round(aic, 1), 'BIC': round(bic, 1)})
        report(f'  {GROUPS[c]} {shape}: R2 {r2:.4f}, max|res| {np.abs(res).max():.1f} %; band II at '
               f'{p[3 + 3]:.0f} cm-1, FWHM {p[4 + 3]:.0f}, {areas[1] / areas.sum() * 100:.1f} % of area')
        inner = outer[row, col].subgridspec(2, 1, height_ratios=[1, 3.2], hspace=0.05)
        ax_r = fig.add_subplot(inner[0])
        ax = fig.add_subplot(inner[1], sharex=ax_r)
        ax_r.plot(gx, res, color='k', lw=1.0)
        ax_r.axhline(0, color='k', ls='--', lw=0.6)
        ax_r.set_ylim(-8, 8)
        ax_r.set_yticks([-8, 0, 8])
        ax_r.set_yticklabels(['-8%', '0', '8%'], fontsize=11)
        ax_r.set_ylabel('Residuals', fontsize=12)
        ax_r.tick_params(labelbottom=False)
        ax_r.set_title(f'{NAME_LONG[c].split(" on")[0]}: {FIT_SHAPES[shape]} ({shape})', fontsize=13)
        ax.plot(gx, yv - base, color='k', lw=2.2, label='data')
        ax.plot(gx, total - base, color='0.6', lw=1.4, label='fit')
        for k, comp in enumerate(comps):
            ax.plot(gx, comp, color='k', lw=0.9)
            ax.text(p[3 + 3 * k], comp.max() + 0.04 * height, FIT_BANDS[k][1], ha='center',
                    va='bottom', fontsize=14, family='serif')
        ax.set_xlim(*FIT_RANGE)
        ax.set_ylim(0, height * 1.12)
        ax.set_xlabel('Raman shift (cm$^{-1}$)')
        ax.set_ylabel('Intensity (a.u.)')
        ax.tick_params(axis='y', labelleft=False)
        ax_r.text(-0.16, 1.25, f'({next(letters)})', transform=ax_r.transAxes, fontsize=17, va='bottom')
fig.subplots_adjust(left=0.05, right=0.99, bottom=0.07, top=0.93)
save_figure(fig, 'FigS2_PeakFit_CH')

# main figure: Gaussian-Lorentzian sum only (best or equal-best fit for both groups
# by AIC/BIC; for NVs the fitted Lorentzian fraction is 0, i.e. Gaussian)
fig = plt.figure(figsize=(12.5, 7.0))
outer = fig.add_gridspec(1, 2, wspace=0.18)
for col, c in enumerate((0, 1)):
    yv = data['ch'][y == c].mean(0)[fr]
    p, base, comps, eta, res, height = FITS[(c, 'GL')]
    total = base + sum(comps)
    inner = outer[col].subgridspec(2, 1, height_ratios=[1, 3.4], hspace=0.05)
    ax_r = fig.add_subplot(inner[0])
    ax = fig.add_subplot(inner[1], sharex=ax_r)
    ax_r.plot(gx, res, color='k', lw=1.0)
    ax_r.axhline(0, color='k', ls='--', lw=0.6)
    ax_r.set_ylim(-8, 8)
    ax_r.set_yticks([-8, 0, 8])
    ax_r.set_yticklabels(['-8%', '0', '8%'], fontsize=12)
    ax_r.set_ylabel('Residuals', fontsize=14)
    ax_r.tick_params(labelbottom=False)
    ax_r.set_title(f'{NAME_LONG[c]}', fontsize=15)
    ax.plot(gx, yv - base, color='k', lw=2.2, label='Mean spectrum')
    ax.plot(gx, total - base, color='0.6', lw=1.4, label=f'GL fit (Lorentzian fraction {eta:.2f})')
    for k, comp in enumerate(comps):
        ax.plot(gx, comp, color='k', lw=0.9)
        ax.text(p[3 + 3 * k], comp.max() + 0.04 * height, f'{FIT_BANDS[k][1]}\n{p[3 + 3 * k]:.0f}',
                ha='center', va='bottom', fontsize=13, family='serif', linespacing=0.95)
    ax.set_xlim(*FIT_RANGE)
    ax.set_ylim(0, height * 1.22)
    ax.set_xlabel('Raman shift (cm$^{-1}$)')
    ax.set_ylabel('Intensity (a.u.)')
    ax.tick_params(axis='y', labelleft=False)
    ax.legend(frameon=False, fontsize=12, loc='upper right')
    ax_r.text(-0.10, 1.25, f'({"ab"[col]})', transform=ax_r.transAxes, fontsize=17, va='bottom')
fig.subplots_adjust(left=0.05, right=0.99, bottom=0.10, top=0.90)
save_figure(fig, 'Fig2c_PeakFit_CH_GL')
plt.rcParams.update({'font.size': 10})


# ============================================================================
# 4. FIGURE 3 - PCA scores and cross-validated LDA scores
# ============================================================================
# (a, c) PCA: all 240 SNV spectra, labels NOT used; 2-SD ellipse per group and the
#        6 site means per group (larger markers).
# (b, d) Cross-validated LDA: StratifiedGroupKFold(5, shuffle, seed 0), groups = sites;
#        in each fold PCA(5) + LDA are fitted on the training spectra only and the
#        test spectra are projected on the LDA axis (out-of-fold scores). Each fold's
#        axis is shifted so that 0 is that fold's LDA decision boundary and oriented
#        so that NVs are positive; a spectrum below 0 is classified as EV.
# The all-data LDA (canonical variable 1) is still computed for the score tables.
report('\n4. FIGURE 3 - PCA and cross-validated LDA')
F3_FS = {'label': 9, 'tick': 8, 'legend': 7.5, 'letter': 10, 'title': 8.5}
F3_LW = 0.8
RC_BEFORE_F3 = plt.rcParams.copy()
plt.rcParams.update({'font.family': 'sans-serif', 'font.sans-serif': ['Arial', 'Liberation Sans', 'DejaVu Sans'],
                     'font.size': F3_FS['tick'], 'axes.spines.top': False, 'axes.spines.right': False,
                     'axes.linewidth': F3_LW, 'xtick.major.width': F3_LW, 'ytick.major.width': F3_LW,
                     'lines.linewidth': F3_LW, 'axes.grid': False})


def cv_lda_scores(X):
    """Out-of-fold LDA scores (PCA(5) + LDA fitted on the training folds only).
    Returns scores (0 = decision boundary, NVs positive) and the LDA predictions."""
    folds_ = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=CV_SEED
                                  ).split(np.zeros(len(y)), y, sites)
    score, pred = np.empty(len(y)), np.empty(len(y), int)
    for tr, te in folds_:
        pf = PCA(n_components=N_COMP, svd_solver='full').fit(X[tr])
        lda_ = LinearDiscriminantAnalysis().fit(pf.transform(X[tr]), y[tr])
        z_tr = lda_.transform(pf.transform(X[tr])).ravel()
        z_te = lda_.transform(pf.transform(X[te])).ravel()
        # decision_function is a linear function of the LDA score: find where it is 0
        a, b = np.polyfit(z_tr, lda_.decision_function(pf.transform(X[tr])), 1)
        orient = 1.0 if np.mean(z_tr[y[tr] == 1]) > -b / a else -1.0   # NV training mean positive
        score[te] = orient * (z_te + b / a)
        pred[te] = lda_.predict(pf.transform(X[te]))
    return score, pred


fig, axs = plt.subplots(2, 2, figsize=(17 / 2.54, 14 / 2.54),
                        gridspec_kw={'width_ratios': [1.6, 1], 'wspace': 0.30, 'hspace': 0.45})
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


    ax = axs[row, 0]                        # PCA scores, 2-SD ellipses, site means
    for c in (0, 1):
        P = T[y == c, :2]
        ax.scatter(P[:, 0], P[:, 1], s=8, color=COLS[c], lw=0, alpha=0.7, label=GROUPS[c], zorder=2)
        vals, vecs = np.linalg.eigh(np.cov(P.T))
        ang = np.degrees(np.arctan2(vecs[1, 1], vecs[0, 1]))
        ax.add_patch(Ellipse(P.mean(0), 4 * np.sqrt(vals[1]), 4 * np.sqrt(vals[0]), angle=ang,
                             fc='none', ec=COLS[c], lw=F3_LW, zorder=3))
        site_xy = np.array([T[sites == s, :2].mean(0) for s in sorted(set(sites[y == c]))])
        ax.scatter(site_xy[:, 0], site_xy[:, 1], s=40, marker='o' if c == 0 else 's', color=COLS[c],
                   edgecolors='k', linewidths=F3_LW, zorder=4)
    ax.axhline(0, color='k', lw=0.5, zorder=1)
    ax.axvline(0, color='k', lw=0.5, zorder=1)
    ax.set_xlabel(f'PC1 ({var[0]:.1f}%)', fontsize=F3_FS['label'])
    ax.set_ylabel(f'PC2 ({var[1]:.1f}%)', fontsize=F3_FS['label'])
    ax.tick_params(labelsize=F3_FS['tick'], width=F3_LW)
    ylo, yhi = ax.get_ylim()
    ax.set_ylim(ylo, yhi + 0.25 * (yhi - ylo))
    handles = [plt.Line2D([], [], ls='', marker='o', ms=4, color=COLS[0], alpha=0.7, label=GROUPS[0]),
               plt.Line2D([], [], ls='', marker='o', ms=4, color=COLS[1], alpha=0.7, label=GROUPS[1]),
               plt.Line2D([], [], ls='', marker='o', ms=6, mfc='0.75', mec='k', mew=F3_LW,
                          label='site means')]
    ax.legend(handles=handles, frameon=True, facecolor='white', edgecolor='none', framealpha=1,
              fontsize=F3_FS['legend'], loc='upper left', ncol=3,
              handletextpad=0.2, columnspacing=0.9, borderaxespad=0.2)
    ax.set_title(TITLES[w], fontsize=F3_FS['title'], loc='left', pad=3)

    ax = axs[row, 1]                        # cross-validated LDA scores
    cvs, cvp = cv_lda_scores(X)
    wrong = {c: int(np.sum((cvs[y == c] > 0) != (c == 1))) for c in (0, 1)}
    RES[w].update(cv_ld=cvs, cv_ld_wrong=wrong, cv_ld_median={c: np.median(cvs[y == c]) for c in (0, 1)},
                  cv_ld_pred_errors={c: int(np.sum(cvp[y == c] != c)) for c in (0, 1)})
    rng = np.random.default_rng(0)
    for c in (0, 1):
        v = cvs[y == c]
        ax.boxplot(v, positions=[c], widths=0.5, whis=1.5, showfliers=False, patch_artist=True,
                   boxprops=dict(fc=COLS[c], alpha=0.30, ec=COLS[c], lw=F3_LW),
                   medianprops=dict(color=COLS[c], lw=1.2),
                   whiskerprops=dict(color=COLS[c], lw=F3_LW), capprops=dict(color=COLS[c], lw=F3_LW))
        ax.scatter(c + rng.uniform(-0.15, 0.15, len(v)), v, s=5, color=COLS[c], lw=0, alpha=0.7, zorder=3)
    ax.axhline(0, color='k', ls='--', lw=F3_LW, zorder=1)
    ax.set_xticks([0, 1])
    ax.set_xticklabels([GROUPS[0], GROUPS[1]], fontsize=F3_FS['label'])
    ax.set_xlim(-0.6, 1.6)
    ax.set_ylabel('Cross-validated LDA score', fontsize=F3_FS['label'])
    ax.tick_params(labelsize=F3_FS['tick'], width=F3_LW)
    report(f'  {PLAIN[w]}: PC1 {var[0]:.1f} %, PC2 {var[1]:.1f} %; cross-validated LDA (PCA re-fitted per '
           f'fold): wrong side of 0: {wrong[0]} of {np.sum(y == 0)} EVs, {wrong[1]} of {np.sum(y == 1)} NVs '
           f'(LDA predictions: {RES[w]["cv_ld_pred_errors"][0]} EVs, {RES[w]["cv_ld_pred_errors"][1]} NVs '
           f'misclassified)')
    for a in axs[row]:
        a.text(-0.30 if a is axs[row, 1] else -0.17, 1.10, f'({next(letters)})', transform=a.transAxes,
               fontsize=F3_FS['letter'], fontweight='bold', va='bottom', ha='left')
fig.subplots_adjust(left=0.085, right=0.985, bottom=0.085, top=0.93)


def save_figure_rgb(fig_, name):
    """PNG and LZW-compressed RGB TIFF, 600 dpi (new name if the old file is open)."""
    import io
    from PIL import Image
    buf = io.BytesIO()
    fig_.savefig(buf, dpi=600, format='png')
    img = Image.open(buf).convert('RGB')
    for ext, kw in [('png', {}), ('tiff', {'compression': 'tiff_lzw'})]:
        target = OUTPUT_DIR / f'{name}.{ext}'
        for k in range(1, 50):
            try:
                img.save(target, dpi=(600, 600), **kw)
                break
            except OSError:
                target = OUTPUT_DIR / f'{name}_{k}.{ext}'
        report(f'  saved {target.name}')


save_figure_rgb(fig, 'Fig3_PCA_LDA')
plt.rcParams.update(RC_BEFORE_F3)          # the later figures keep their own settings
plt.rcParams.update({'font.size': 16, 'axes.spines.top': False, 'axes.spines.right': False,
                     'axes.linewidth': 1.2})


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
fig, axs = plt.subplots(2, 2, figsize=(14, 10.5), gridspec_kw={'width_ratios': [3, 1.3]})
for col, w in enumerate(['fingerprint', 'ch']):
    g, L, var = WINDOWS[w]['g'], RES[w]['loadings'], RES[w]['var']
    hi_g = GROUPS[0] if RES[w]['pc1_mean'][0] > 0 else GROUPS[1]
    lo_g = GROUPS[1] if hi_g == GROUPS[0] else GROUPS[0]
    ax = axs[0, col]
    ax.axhline(0, color='k', lw=0.6)
    ax.plot(g, L[0], color='#0B5A7E', lw=1.6, label=f'PC1 ({var[0]:.1f}%)')
    ax.plot(g, L[1], color='#B5541C', lw=1.4, alpha=0.8, label=f'PC2 ({var[1]:.1f}%)')
    bands = main_bands(g, L[0], N_BANDS[w])
    RES[w]['load_bands'] = [(wn, v, hi_g if v > 0 else lo_g) for wn, v in bands]
    span = np.ptp(L[:2])
    for wn, v in bands:
        ax.text(wn, v + (0.03 if v > 0 else -0.03) * span, f'{wn:.0f}', ha='center',
                va='bottom' if v > 0 else 'top', fontsize=13, rotation=90)
        band_rows.append({'Source': 'PC1 loading', 'Region': w, 'Band (cm-1)': round(wn),
                          'Value': round(v, 4), 'Relatively stronger in': hi_g if v > 0 else lo_g})
    ax.set_xlim(g[0], g[-1])
    ax.set_ylim(L[:2].min() - 0.18 * span, L[:2].max() + (0.22 if w == 'fingerprint' else 0.55) * span)
    ax.set_xlabel('Raman shift (cm$^{-1}$)')
    ax.set_ylabel('Loading (a.u.)')
    ax.set_title(f'{TITLES[w]}\nPC1 > 0: {hi_g} side', fontsize=15, loc='left')
    ax.legend(frameon=False, fontsize=14 if w == 'fingerprint' else 12,
              loc='lower left' if w == 'fingerprint' else 'upper right')

    X = data[w]
    S = {c: np.array([X[sites == s].mean(0) for s in sorted(set(sites[y == c]))]) for c in (0, 1)}
    d = S[1].mean(0) - S[0].mean(0)
    se = np.sqrt(S[0].var(0, ddof=1) / len(S[0]) + S[1].var(0, ddof=1) / len(S[1]))
    sig = np.abs(d) > 2 * se
    ax = axs[1, col]
    ax.axhline(0, color='k', lw=0.6)
    ax.fill_between(g, d - 2 * se, d + 2 * se, color='0.75', lw=0, label='$\\pm$2 SE (6 + 6 sites)')
    ax.plot(g, d, color='k', lw=1.4, label='NV $-$ EV')
    bands = [b for b in main_bands(g, np.where(sig, d, 0), N_BANDS[w]) if b[1] != 0]
    RES[w]['diff_bands'] = bands
    RES[w]['diff_sig_frac'] = sig.mean() * 100
    span = np.ptp(np.r_[d - 2 * se, d + 2 * se])
    for wn, v in bands:
        ax.text(wn, v + (0.04 if v > 0 else -0.04) * span, f'{wn:.0f}', ha='center',
                va='bottom' if v > 0 else 'top', fontsize=13, rotation=90)
        band_rows.append({'Source': 'NV - EV difference', 'Region': w, 'Band (cm-1)': round(wn),
                          'Value': round(v, 3), 'Relatively stronger in': GROUPS[1] if v > 0 else GROUPS[0]})
    ax.set_xlim(g[0], g[-1])
    ax.set_ylim((d - 2 * se).min() - 0.15 * span, (d + 2 * se).max() + (0.2 if w == 'fingerprint' else 0.5) * span)
    ax.set_xlabel('Raman shift (cm$^{-1}$)')
    ax.set_ylabel('$\\Delta$ SNV intensity, NV $-$ EV')
    ax.set_title(TITLES[w], fontsize=15, loc='left')
    ax.legend(frameon=False, fontsize=14 if w == 'fingerprint' else 12,
              loc='lower left' if w == 'fingerprint' else 'upper right')
    report(f'  {PLAIN[w]}: PC1 > 0 = {hi_g}; difference > 2 SE at {sig.mean() * 100:.0f} % '
           f'of wavenumbers')
for a, letter in zip([axs[0, 0], axs[0, 1], axs[1, 0], axs[1, 1]], 'abcd'):
    a.text(-0.10 if a in (axs[0, 0], axs[1, 0]) else -0.25, 1.06, f'({letter})',
           transform=a.transAxes, fontsize=18, va='bottom')
fig.tight_layout()
save_figure(fig, 'Fig4_Loadings_Difference')


# ============================================================================
# 6. FIGURE 5 - cross-validated classification (confusion matrices)
# ============================================================================
# 12 sites split into 5 folds, all 20 spectra of a site together; train on 4
# folds, predict the 5th; every spectrum predicted once by a model that never
# saw its site.
#   PCA-LDA: PCA (5 PCs) + LDA fitted on the training folds (PCA_PER_FOLD = True;
#            False: PCA once on all spectra without labels, LDA on the training folds)
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


MAIN_TXT = 'PCA fitted on the training folds only' if PCA_PER_FOLD else 'PCA fitted once on all spectra'
ALT_TXT = 'PCA fitted once on all spectra' if PCA_PER_FOLD else 'PCA re-fitted in each training fold'
report('\n6. FIGURE 5 - cross-validation (PCA-LDA: ' + MAIN_TXT + ')')
folds = make_folds(y)
for k, (tr, te) in enumerate(folds, 1):
    report(f'  fold {k}: test sites ' + ', '.join(sorted(set(sites[te]))))
fold_rows = []
fig, axs = plt.subplots(2, 2, figsize=(9.5, 9))
letters = iter('abcd')
for row, w in enumerate(['fingerprint', 'ch']):
    X = data[w]
    RES[w]['alt'] = (cv_pca_lda(X, y, folds, refit=not PCA_PER_FOLD) == y).mean() * 100
    for col, (model, pred) in enumerate([('PCA-LDA', cv_pca_lda(X, y, folds, refit=PCA_PER_FOLD)),
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
                        va='center', fontsize=18, color='white' if cm[i, j] > 60 else 'black')
        ax.set_xticks([0, 1])
        ax.set_xticklabels(['EVs', 'NVs'])
        ax.set_yticks([0, 1])
        ax.set_yticklabels(['EVs', 'NVs'])
        ax.set_xlabel('Predicted')
        ax.set_ylabel('True')
        ax.set_title(f'{model} - {SHORT[w]}\naccuracy {acc:.1f}%', fontsize=16)
        ax.text(-0.36, 1.12, f'({next(letters)})', transform=ax.transAxes, fontsize=18, va='bottom')
    report(f'  {PLAIN[w]} PCA-LDA, comparison with {ALT_TXT}: {RES[w]["alt"]:.1f} %')
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
    fig, axs = plt.subplots(2, 2, figsize=(11, 9))
    letters = iter('abcd')
    for row, w in enumerate(['fingerprint', 'ch']):
        X = data[w]
        T_all = PCA(n_components=N_COMP, svd_solver='full').fit_transform(X)
        accs = {'PCA-LDA': [], 'PLS-DA': []}
        for k, ev_sites in enumerate(labelings):
            lab = dict(zip(SITE_IDS, [0 if i in ev_sites else 1 for i in range(len(SITE_IDS))]))
            yp = np.array([lab[s] for s in sites])
            fp = make_folds(yp)
            accs['PCA-LDA'].append((cv_pca_lda(X, yp, fp, T=T_all, refit=PCA_PER_FOLD) == yp).mean() * 100)
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
                    color='#B5541C', fontsize=14)
            ax.set_xlim(0, 100)
            ax.set_xlabel('Cross-validated accuracy (%)')
            ax.set_ylabel('Number of site labelings')
            ax.set_title(f'{model} - {SHORT[w]}\np = {n_ge / len(a):.4f} ({len(a)} labelings)',
                         fontsize=15)
            ax.text(-0.18, 1.08, f'({next(letters)})', transform=ax.transAxes, fontsize=18,
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
pd.DataFrame(fit_rows).to_csv(OUTPUT_DIR / 'TableS6_PeakFit_CH.csv', index=False)
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
      f'  {N_COMP} components, exact SVD (svd_solver = "full"), mean-centred, labels not used. '
      f'Figure 3 a/c: all 240 spectra. PCA-LDA classification: ' + MAIN_TXT + '. '
      f'PC1 sign: loading positive at {PC1_REF["fingerprint"]} / '
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
md += ['  Comparison: PCA-LDA with ' + ALT_TXT + '.',
       '  Permutation test: all 924 assignments of 6 of the 12 sites to "EV", full '
       'cross-validation repeated for each; p = labelings with accuracy >= real / 924.',
       '  Sensitivity = NVs correctly classified; specificity = EVs correctly classified.',
       '', 'CH BAND FITTING (Figure S2, Table S6)',
       f'  mean spectrum of each group, {FIT_RANGE[0]}-{FIT_RANGE[1]} cm-1; straight baseline + '
       f'{len(FIT_BANDS)} bands (start ' + ', '.join(str(b[0]) for b in FIT_BANDS) + ' cm-1), '
       'position +/-12 cm-1, FWHM 12-60 cm-1, non-linear least squares (scipy least_squares);',
       '  line shapes: Gaussian, Gaussian-Lorentzian sum (one mixing fraction per spectrum), Lorentzian.']
for r_ in [r for r in fit_rows if r['Band'] == 'II']:
    md.append(f"    band II, {r_['Group']}, {r_['Line shape']}: {r_['Position (cm-1)']} cm-1, FWHM "
              f"{r_['FWHM (cm-1)']}{' (limit)' if r_['FWHM at limit (60)'] else ''}, "
              f"{r_['Area (% of CH bands)']} % of area, R2 {r_['R2']}")
md += ['', 'RESULTS']
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
    md.append(f'    PCA-LDA with {ALT_TXT} (comparison): {r["alt"]:.1f} %; '
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
            f'Cross-validated LDA score (Fig 3; PCA + LDA fitted on the training folds only): median '
            f'{r["cv_ld_median"][0]:+.2f} (EVs) vs {r["cv_ld_median"][1]:+.2f} (NVs); on the wrong side of '
            f'the decision boundary: {r["cv_ld_wrong"][0]} of 120 EVs and {r["cv_ld_wrong"][1]} of 120 NVs.']
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
    txt += [f'PCA-LDA (Fig 5, Table 1) uses {MAIN_TXT}; with {ALT_TXT}: {r["alt"]:.1f}%.', '']
txt += ['--- LIMITATIONS TO STATE ---',
        '* Independent units are the 6 + 6 measurement sites, not the 240 spectra; the smallest '
        'possible permutation p is 2/924 = 0.0022.',
        '* If all EV spectra come from one preparation and all NV spectra from another, '
        'preparation or measurement-session effects cannot be separated from EV-vs-NV biology.',
        '* After SNV, intensities are relative (composition), not absolute amounts.',
        '* Fig 3 b/d shows the out-of-fold LDA scores of the PCA-LDA model (PCA + LDA fitted on the '
        'training folds); with PCA_PER_FOLD = True the spectra below 0 are exactly the errors in '
        'Fig 5 / Table 1.']
(OUTPUT_DIR / 'RESULTS_for_manuscript.txt').write_text('\n'.join(txt) + '\n', encoding='utf-8')
(OUTPUT_DIR / 'Run_log.txt').write_text('\n'.join(LOG) + '\n', encoding='utf-8')
print('\n' + '\n'.join(md) + '\n\n' + '\n'.join(txt))
print(f'\nAll outputs saved in: {OUTPUT_DIR}')
if SHOW_WINDOW:
    plt.show()
