# ============================================================
# GrapheneRaw_Analysis.py
# Graphene Enhancement Analysis from Raw Raman Spectra
# 100 raw .txt files, 4 classes x 25 spectra each
#
# Scientific goal: Determine if graphene improves SSY Raman spectra
# Strategy: Raw spectra -> ALS baseline -> Net Raman signal
#           -> Enhancement Factor, SNR, Reproducibility, Fingerprint
#
# INPUT : Nuova cartella\ folder (100 .txt files, tab-separated)
# OUTPUT: GrapheneRaw_Results\ folder (figures + Excel report)
# ============================================================

import os
import re
import glob
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Patch
from scipy.signal import savgol_filter, find_peaks
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve
from scipy.stats import wilcoxon, mannwhitneyu, ttest_1samp
from openpyxl import Workbook
from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
from openpyxl.utils import get_column_letter

# ─────────────────────────────────────────────────────────────
# USER SETTINGS  -- adjust paths if needed
# ─────────────────────────────────────────────────────────────
RAW_DIR = r"C:\Users\Hira Aman\Desktop\PROF_DOMENICO'S\Nuova cartella"
OUT_DIR = r"C:\Users\Hira Aman\Desktop\PROF_DOMENICO'S\GrapheneRaw_Results"

# ─────────────────────────────────────────────────────────────
# PREPROCESSING PARAMETERS
# ─────────────────────────────────────────────────────────────
SG_WIN  = 11        # Savitzky-Golay window (odd)
SG_ORD  = 2         # Savitzky-Golay polynomial order
ALS_LAM = 1e5       # ALS smoothness
ALS_P   = 0.01      # ALS asymmetry
ALS_NITER = 15      # ALS iterations

# ─────────────────────────────────────────────────────────────
# SPECTRAL REGIONS (cm-1)
# ─────────────────────────────────────────────────────────────
SSY_PEAKS = {        # SSY characteristic peaks
    '~1150': (1130, 1170),
    '~1229': (1215, 1245),
    '~1386': (1370, 1405),
    '~1476': (1460, 1490),
    '~1500': (1490, 1515),
    '~1596': (1580, 1620),
}
GR_D_BAND   = (1320, 1380)   # graphene D-band
GR_G_BAND   = (1555, 1610)   # graphene G-band
NOISE_REGION = (420,  600)   # quiet region for noise estimation
WN_FULL     = (400,  2000)   # full spectral range
WN_PLOT     = (900,  1800)   # Raman fingerprint window for plots

# ─────────────────────────────────────────────────────────────
# COLORS
# ─────────────────────────────────────────────────────────────
CLR = {
    'SSY-PDMS':       '#1565C0',  # dark blue
    'SSY-Gr-PDMS':    '#C62828',  # dark red
    'SSY-SiO2/Si':    '#2E7D32',  # dark green
    'SSY-Gr-SiO2/Si': '#E65100',  # dark orange
}
CLASS_ORDER = ['SSY-PDMS', 'SSY-Gr-PDMS', 'SSY-SiO2/Si', 'SSY-Gr-SiO2/Si']

# ─────────────────────────────────────────────────────────────
# HELPER FUNCTIONS
# ─────────────────────────────────────────────────────────────

def als_baseline(y, lam=ALS_LAM, p=ALS_P, niter=ALS_NITER):
    L = len(y)
    D = diags([1, -2, 1], [0, 1, 2], shape=(L-2, L), dtype=float)
    H = lam * (D.T @ D)
    w = np.ones(L)
    z = y.copy()
    for _ in range(niter):
        W = diags(w)
        z = spsolve((W + H).tocsc(), w * y)
        w = p * (y > z) + (1 - p) * (y <= z)
    return z


def preprocess(raw_int):
    smoothed  = savgol_filter(raw_int, SG_WIN, SG_ORD)
    baseline  = als_baseline(smoothed)
    net_raman = smoothed - baseline
    return smoothed, baseline, net_raman


def peak_area(wn, spectrum, lo, hi):
    mask = (wn >= lo) & (wn <= hi)
    if mask.sum() == 0:
        return 0.0
    try:
        return np.trapezoid(spectrum[mask], wn[mask])
    except AttributeError:
        return np.trapz(spectrum[mask], wn[mask])


def peak_height(wn, spectrum, lo, hi):
    mask = (wn >= lo) & (wn <= hi)
    if mask.sum() == 0:
        return 0.0
    return float(np.clip(spectrum[mask], 0, None).max())


def assign_class(filename):
    s = os.path.basename(filename).lower()
    if re.search(r'gr[-_]sio2|gr[-_]si', s):              return 'SSY-Gr-SiO2/Si'
    if re.search(r'grsl[-_]pdms|gr[-_]pdms|gr\w*[-_]pdms', s): return 'SSY-Gr-PDMS'
    if re.search(r'sio2[-_]si', s):                        return 'SSY-SiO2/Si'
    if re.search(r'pdms', s):                              return 'SSY-PDMS'
    return None


def noise_rms(wn, net_raman):
    mask = (wn >= NOISE_REGION[0]) & (wn <= NOISE_REGION[1])
    return float(np.std(net_raman[mask])) if mask.sum() > 0 else 1.0


def rsd_percent(values):
    v = np.array(values)
    if v.mean() == 0:
        return np.nan
    return 100.0 * v.std(ddof=1) / v.mean()


def sig_stars(p):
    if p < 0.001: return '***'
    if p < 0.01:  return '**'
    if p < 0.05:  return '*'
    return 'ns'


def save_fig(fig, name):
    path = os.path.join(OUT_DIR, name)
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved: {name}')


# ─────────────────────────────────────────────────────────────
# LOAD ALL SPECTRA
# ─────────────────────────────────────────────────────────────

print('=' * 60)
print('GrapheneRaw_Analysis.py  -- Loading spectra...')
print('=' * 60)

os.makedirs(OUT_DIR, exist_ok=True)

txt_files = sorted(glob.glob(os.path.join(RAW_DIR, '*.txt')))
if not txt_files:
    raise FileNotFoundError(f'No .txt files found in:\n  {RAW_DIR}\nCheck RAW_DIR path.')

# Storage: per class lists of (wn, raw, smoothed, baseline, net_raman)
data = {cls: [] for cls in CLASS_ORDER}
wn_ref = None   # common wavenumber axis (first file)

for fpath in txt_files:
    cls = assign_class(fpath)
    if cls is None:
        print(f'  [SKIP] Cannot classify: {os.path.basename(fpath)}')
        continue
    try:
        arr = np.loadtxt(fpath)
        wn  = arr[:, 0]
        raw = arr[:, 1]
    except Exception as e:
        print(f'  [SKIP] Read error in {os.path.basename(fpath)}: {e}')
        continue

    if wn_ref is None:
        wn_ref = wn
    elif len(wn) != len(wn_ref):
        # Interpolate to common axis
        raw = np.interp(wn_ref, wn, raw)
        wn  = wn_ref

    smoothed, baseline, net = preprocess(raw)
    data[cls].append({'file': os.path.basename(fpath),
                      'raw': raw, 'smoothed': smoothed,
                      'baseline': baseline, 'net': net})

print()
for cls in CLASS_ORDER:
    print(f'  {cls}: {len(data[cls])} spectra loaded')
print()

# Build matrices: shape (n_spectra, n_wavenumbers)
mat = {}
for cls in CLASS_ORDER:
    mat[cls] = {
        'raw':      np.vstack([d['raw']      for d in data[cls]]),
        'smoothed': np.vstack([d['smoothed'] for d in data[cls]]),
        'baseline': np.vstack([d['baseline'] for d in data[cls]]),
        'net':      np.vstack([d['net']      for d in data[cls]]),
    }

# ─────────────────────────────────────────────────────────────
# FIG 1 : MEAN RAW SPECTRA (full range)
# ─────────────────────────────────────────────────────────────

print('Generating figures...')

fig, axes = plt.subplots(2, 2, figsize=(14, 9), sharey=False)
fig.suptitle('Mean Raw Raman Spectra — All Classes\n(400–2000 cm⁻¹, no preprocessing)',
             fontsize=13, fontweight='bold')
axes = axes.flatten()

for ax, cls in zip(axes, CLASS_ORDER):
    m  = mat[cls]['raw'].mean(axis=0)
    sd = mat[cls]['raw'].std(axis=0)
    ax.plot(wn_ref, m, color=CLR[cls], lw=1.5, label='Mean')
    ax.fill_between(wn_ref, m-sd, m+sd, color=CLR[cls], alpha=0.20, label='±1 SD')
    ax.set_title(cls, color=CLR[cls], fontweight='bold')
    ax.set_xlabel('Raman Shift (cm⁻¹)')
    ax.set_ylabel('Intensity (counts)')
    ax.set_xlim(WN_FULL)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

plt.tight_layout()
save_fig(fig, 'R1_MeanRawSpectra_AllClasses.png')

# ─────────────────────────────────────────────────────────────
# FIG 2 : MEAN NET RAMAN (ALS-corrected) — fingerprint window
# ─────────────────────────────────────────────────────────────

fig, ax = plt.subplots(figsize=(12, 6))
ax.set_title('Mean Net Raman Signal After ALS Baseline Correction\n(Fluorescence removed — true molecular signal)',
             fontsize=12, fontweight='bold')

for cls in CLASS_ORDER:
    m   = mat[cls]['net'].mean(axis=0)
    sd  = mat[cls]['net'].std(axis=0)
    mask = (wn_ref >= WN_PLOT[0]) & (wn_ref <= WN_PLOT[1])
    ax.plot(wn_ref[mask], m[mask], color=CLR[cls], lw=2, label=cls)
    ax.fill_between(wn_ref[mask], (m-sd)[mask], (m+sd)[mask],
                    color=CLR[cls], alpha=0.12)

# Mark SSY peaks
peak_centers = [1150, 1229, 1386, 1476, 1500, 1596]
ymax = ax.get_ylim()[1]
for pc in peak_centers:
    ax.axvline(pc, color='grey', lw=0.7, ls='--', alpha=0.5)
    ax.text(pc, ymax*0.97, f'{pc}', ha='center', va='top', fontsize=7, color='grey')

# Mark graphene bands
ax.axvspan(GR_D_BAND[0], GR_D_BAND[1], color='purple', alpha=0.07, label='Gr D-band')
ax.axvspan(GR_G_BAND[0], GR_G_BAND[1], color='brown',  alpha=0.07, label='Gr G-band')

ax.set_xlabel('Raman Shift (cm⁻¹)', fontsize=11)
ax.set_ylabel('Net Intensity (counts, baseline-removed)', fontsize=11)
ax.set_xlim(WN_PLOT)
ax.legend(fontsize=9, ncol=2)
ax.grid(alpha=0.3)
plt.tight_layout()
save_fig(fig, 'R2_MeanNetRaman_Fingerprint.png')

# ─────────────────────────────────────────────────────────────
# FIG 3 : FLUORESCENCE BACKGROUND COMPARISON
# ─────────────────────────────────────────────────────────────

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle('Fluorescence Background Comparison\n(ALS baseline = background estimate)',
             fontsize=12, fontweight='bold')

pairs = [('SSY-PDMS', 'SSY-Gr-PDMS'), ('SSY-SiO2/Si', 'SSY-Gr-SiO2/Si')]
pair_titles = ['PDMS Substrate', 'SiO2/Si Substrate']

for ax, (cls_ng, cls_gr), title in zip(axes, pairs, pair_titles):
    mask = (wn_ref >= WN_PLOT[0]) & (wn_ref <= WN_PLOT[1])
    for cls in [cls_ng, cls_gr]:
        m  = mat[cls]['baseline'].mean(axis=0)
        sd = mat[cls]['baseline'].std(axis=0)
        ax.plot(wn_ref[mask], m[mask], color=CLR[cls], lw=2, label=cls)
        ax.fill_between(wn_ref[mask], (m-sd)[mask], (m+sd)[mask],
                        color=CLR[cls], alpha=0.15)

    # annotation: mean background level
    bkg_ng = mat[cls_ng]['baseline'][:, mask].mean()
    bkg_gr = mat[cls_gr]['baseline'][:, mask].mean()
    pct = 100 * (bkg_ng - bkg_gr) / bkg_ng
    sign = 'lower' if pct > 0 else 'higher'
    ax.set_title(f'{title}\nGraphene background {abs(pct):.0f}% {sign}',
                 fontsize=10)
    ax.set_xlabel('Raman Shift (cm⁻¹)')
    ax.set_ylabel('Background (counts)')
    ax.set_xlim(WN_PLOT)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

plt.tight_layout()
save_fig(fig, 'R3_FluorescenceBackground.png')

# ─────────────────────────────────────────────────────────────
# FIG 4 : NET SIGNAL PAIR PANELS (Gr vs no-Gr, per substrate)
# ─────────────────────────────────────────────────────────────

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle('Net Raman Signal: Graphene vs No-Graphene\n(After ALS background removal)',
             fontsize=12, fontweight='bold')

for ax, (cls_ng, cls_gr), title in zip(axes, pairs, pair_titles):
    mask = (wn_ref >= WN_PLOT[0]) & (wn_ref <= WN_PLOT[1])
    for cls in [cls_ng, cls_gr]:
        m  = mat[cls]['net'].mean(axis=0)
        sd = mat[cls]['net'].std(axis=0)
        lw = 2.5 if 'Gr' in cls else 1.5
        ls = '-'  if 'Gr' in cls else '--'
        ax.plot(wn_ref[mask], m[mask], color=CLR[cls], lw=lw, ls=ls, label=cls)
        ax.fill_between(wn_ref[mask], (m-sd)[mask], (m+sd)[mask],
                        color=CLR[cls], alpha=0.12)

    ax.axvspan(GR_D_BAND[0], GR_D_BAND[1], color='purple', alpha=0.07, label='Gr D-band')
    ax.axvspan(GR_G_BAND[0], GR_G_BAND[1], color='brown',  alpha=0.07, label='Gr G-band')

    for pc in [1229, 1386, 1476, 1596]:
        ax.axvline(pc, color='grey', lw=0.6, ls=':', alpha=0.6)

    ax.set_title(title, fontsize=11, fontweight='bold')
    ax.set_xlabel('Raman Shift (cm⁻¹)')
    ax.set_ylabel('Net Intensity (counts)')
    ax.set_xlim(WN_PLOT)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

plt.tight_layout()
save_fig(fig, 'R4_NetRaman_PairComparison.png')

# ─────────────────────────────────────────────────────────────
# COMPUTE PER-SPECTRUM METRICS
# ─────────────────────────────────────────────────────────────

metrics = {cls: {'peak_heights': {}, 'snr': [], 'bkg_mean': []} for cls in CLASS_ORDER}

for cls in CLASS_ORDER:
    for lbl in SSY_PEAKS:
        metrics[cls]['peak_heights'][lbl] = []

    for i, d in enumerate(data[cls]):
        wn  = wn_ref
        net = mat[cls]['net'][i]
        bkg = mat[cls]['baseline'][i]

        noise = noise_rms(wn, net)
        if noise < 0.01:
            noise = 0.01

        # Peak heights and SNR
        snr_list = []
        for lbl, (lo, hi) in SSY_PEAKS.items():
            ph = peak_height(wn, net, lo, hi)
            metrics[cls]['peak_heights'][lbl].append(ph)
            snr_list.append(ph / noise)

        metrics[cls]['snr'].append(np.mean(snr_list))
        metrics[cls]['bkg_mean'].append(float(bkg.mean()))

# ─────────────────────────────────────────────────────────────
# FIG 5 : SNR COMPARISON (box + strip plot)
# ─────────────────────────────────────────────────────────────

fig, ax = plt.subplots(figsize=(9, 6))
ax.set_title('Signal-to-Noise Ratio per Spectrum\n(SNR = mean peak height / noise RMS)',
             fontsize=12, fontweight='bold')

positions = np.arange(len(CLASS_ORDER))
for i, cls in enumerate(CLASS_ORDER):
    vals = metrics[cls]['snr']
    bp   = ax.boxplot(vals, positions=[i], widths=0.5,
                      patch_artist=True, notch=False,
                      boxprops=dict(facecolor=CLR[cls], alpha=0.5),
                      medianprops=dict(color='black', lw=2),
                      whiskerprops=dict(color=CLR[cls]),
                      capprops=dict(color=CLR[cls]),
                      flierprops=dict(marker='o', color=CLR[cls], alpha=0.5))
    jitter = np.random.default_rng(42).uniform(-0.15, 0.15, len(vals))
    ax.scatter([i + j for j in jitter], vals, color=CLR[cls],
               s=30, zorder=5, alpha=0.7, edgecolors='white', lw=0.5)

# Statistical comparison: Gr vs no-Gr per substrate
for (cls_ng, cls_gr) in pairs:
    i_ng = CLASS_ORDER.index(cls_ng)
    i_gr = CLASS_ORDER.index(cls_gr)
    vals_ng = metrics[cls_ng]['snr']
    vals_gr = metrics[cls_gr]['snr']
    stat, p = mannwhitneyu(vals_gr, vals_ng, alternative='greater')
    stars = sig_stars(p)
    y_top = max(max(vals_ng), max(vals_gr)) * 1.05
    x_mid = (i_ng + i_gr) / 2
    ax.plot([i_ng, i_gr], [y_top, y_top], 'k-', lw=1)
    ax.text(x_mid, y_top * 1.01, f'p={p:.3f} {stars}', ha='center', fontsize=9)

ax.set_xticks(positions)
ax.set_xticklabels([c.replace('SSY-', '') for c in CLASS_ORDER], fontsize=9)
ax.set_ylabel('Mean SNR (peak / noise)', fontsize=11)
ax.grid(axis='y', alpha=0.3)
ax.set_ylim(bottom=0)
plt.tight_layout()
save_fig(fig, 'R5_SNR_Comparison.png')

# ─────────────────────────────────────────────────────────────
# FIG 6 : FLUORESCENCE BACKGROUND BOXPLOT
# ─────────────────────────────────────────────────────────────

fig, ax = plt.subplots(figsize=(9, 6))
ax.set_title('Fluorescence Background Level per Spectrum\n(Mean ALS baseline — lower = cleaner spectrum)',
             fontsize=12, fontweight='bold')

for i, cls in enumerate(CLASS_ORDER):
    vals = metrics[cls]['bkg_mean']
    ax.boxplot(vals, positions=[i], widths=0.5,
               patch_artist=True,
               boxprops=dict(facecolor=CLR[cls], alpha=0.5),
               medianprops=dict(color='black', lw=2),
               whiskerprops=dict(color=CLR[cls]),
               capprops=dict(color=CLR[cls]),
               flierprops=dict(marker='o', color=CLR[cls], alpha=0.5))
    jitter = np.random.default_rng(42).uniform(-0.15, 0.15, len(vals))
    ax.scatter([i + j for j in jitter], vals, color=CLR[cls],
               s=30, zorder=5, alpha=0.7, edgecolors='white', lw=0.5)

# Annotation: % difference
for cls_ng, cls_gr in pairs:
    i_ng = CLASS_ORDER.index(cls_ng)
    i_gr = CLASS_ORDER.index(cls_gr)
    m_ng = np.mean(metrics[cls_ng]['bkg_mean'])
    m_gr = np.mean(metrics[cls_gr]['bkg_mean'])
    pct  = 100 * (m_ng - m_gr) / m_ng
    stat, p = mannwhitneyu(metrics[cls_ng]['bkg_mean'],
                           metrics[cls_gr]['bkg_mean'], alternative='two-sided')
    y_top = max(max(metrics[cls_ng]['bkg_mean']),
                max(metrics[cls_gr]['bkg_mean'])) * 1.05
    ax.plot([i_ng, i_gr], [y_top, y_top], 'k-', lw=1)
    direction = 'lower' if pct > 0 else 'higher'
    ax.text((i_ng+i_gr)/2, y_top*1.01,
            f'Gr {abs(pct):.0f}% {direction}\np={p:.3f}{sig_stars(p)}',
            ha='center', fontsize=8)

ax.set_xticks(range(len(CLASS_ORDER)))
ax.set_xticklabels([c.replace('SSY-', '') for c in CLASS_ORDER], fontsize=9)
ax.set_ylabel('Mean Background (counts)', fontsize=11)
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
save_fig(fig, 'R6_Background_Boxplot.png')

# ─────────────────────────────────────────────────────────────
# FIG 7 : ENHANCEMENT FACTOR (Gr / no-Gr) per SSY PEAK
# ─────────────────────────────────────────────────────────────

fig, axes = plt.subplots(1, 2, figsize=(13, 6))
fig.suptitle('Graphene Enhancement Factor at SSY Peaks\n'
             'EF = mean net peak height (Gr) / mean net peak height (no-Gr)',
             fontsize=12, fontweight='bold')

peak_labels = list(SSY_PEAKS.keys())
x = np.arange(len(peak_labels))

for ax, (cls_ng, cls_gr), title in zip(axes, pairs, pair_titles):
    ef_vals   = []
    ef_ci     = []  # 95% CI from bootstrapping

    m_ng_arr = np.array([metrics[cls_ng]['peak_heights'][lbl] for lbl in peak_labels])
    m_gr_arr = np.array([metrics[cls_gr]['peak_heights'][lbl] for lbl in peak_labels])

    rng = np.random.default_rng(42)
    for k in range(len(peak_labels)):
        ng_vals = np.array(metrics[cls_ng]['peak_heights'][peak_labels[k]])
        gr_vals = np.array(metrics[cls_gr]['peak_heights'][peak_labels[k]])
        # Bootstrap EF
        n_boot = 2000
        ef_boot = []
        for _ in range(n_boot):
            ng_b = rng.choice(ng_vals, size=len(ng_vals), replace=True)
            gr_b = rng.choice(gr_vals, size=len(gr_vals), replace=True)
            ef_boot.append(gr_b.mean() / (ng_b.mean() + 1e-9))
        ef_boot = np.array(ef_boot)
        ef_vals.append(np.nanmean(ef_boot))
        ci_lo, ci_hi = np.nanpercentile(ef_boot, [2.5, 97.5])
        ef_ci.append((ci_lo, ci_hi))

    # Point estimates
    ef_point = [(np.mean(metrics[cls_gr]['peak_heights'][lbl]) /
                 (np.mean(metrics[cls_ng]['peak_heights'][lbl]) + 1e-9))
                for lbl in peak_labels]

    bars = ax.bar(x, ef_point,
                  color=[CLR[cls_gr] if ef > 1 else '#9E9E9E' for ef in ef_point],
                  alpha=0.75, edgecolor='black', lw=0.8)

    # Error bars from bootstrap CI
    ci_lo_arr = [ef_ci[k][0] for k in range(len(peak_labels))]
    ci_hi_arr = [ef_ci[k][1] for k in range(len(peak_labels))]
    ax.errorbar(x, ef_point,
                yerr=[np.array(ef_point) - np.array(ci_lo_arr),
                      np.array(ci_hi_arr) - np.array(ef_point)],
                fmt='none', color='black', capsize=4, lw=1.5)

    # Reference line EF=1
    ax.axhline(1.0, color='black', lw=1.5, ls='--', label='EF = 1 (no change)')

    # Annotate EF values
    for xi, ef in zip(x, ef_point):
        ax.text(xi, ef + 0.02, f'{ef:.2f}', ha='center', va='bottom', fontsize=9)

    ax.set_xticks(x)
    ax.set_xticklabels(peak_labels, fontsize=9, rotation=20)
    ax.set_ylabel('Enhancement Factor (EF)', fontsize=11)
    ax.set_title(f'{title}\n{cls_gr} / {cls_ng}', fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim(bottom=0)

plt.tight_layout()
save_fig(fig, 'R7_EnhancementFactor_SSYpeaks.png')

# ─────────────────────────────────────────────────────────────
# FIG 8 : REPRODUCIBILITY — RSD% per class per peak
# ─────────────────────────────────────────────────────────────

fig, ax = plt.subplots(figsize=(12, 6))
ax.set_title('Spectral Reproducibility: RSD% at SSY Peaks\n'
             '(Lower RSD% = more reproducible / consistent spectra)',
             fontsize=12, fontweight='bold')

n_cls  = len(CLASS_ORDER)
n_pk   = len(peak_labels)
width  = 0.18
offsets = np.linspace(-(n_cls-1)*width/2, (n_cls-1)*width/2, n_cls)

for i, cls in enumerate(CLASS_ORDER):
    rsds = [rsd_percent(metrics[cls]['peak_heights'][lbl]) for lbl in peak_labels]
    ax.bar(x + offsets[i], rsds, width=width,
           color=CLR[cls], alpha=0.75, edgecolor='black', lw=0.5, label=cls)

ax.set_xticks(x)
ax.set_xticklabels(peak_labels, fontsize=10)
ax.set_ylabel('RSD% (coefficient of variation)', fontsize=11)
ax.legend(fontsize=8, ncol=2)
ax.axhline(20, color='grey', lw=1, ls='--', alpha=0.6, label='20% threshold')
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
save_fig(fig, 'R8_Reproducibility_RSD.png')

# ─────────────────────────────────────────────────────────────
# FIG 9 : GRAPHENE FINGERPRINT DETECTION
# ─────────────────────────────────────────────────────────────

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle('Graphene Fingerprint: D-band & G-band Detection\n'
             '(Peaks appear only in graphene-coated substrates)',
             fontsize=12, fontweight='bold')

zoom_lo, zoom_hi = 1200, 1700

for ax, (cls_ng, cls_gr), title in zip(axes, pairs, pair_titles):
    mask = (wn_ref >= zoom_lo) & (wn_ref <= zoom_hi)
    for cls in [cls_ng, cls_gr]:
        m  = mat[cls]['net'].mean(axis=0)
        sd = mat[cls]['net'].std(axis=0)
        lw = 2.5 if 'Gr' in cls else 1.5
        ls = '-'  if 'Gr' in cls else '--'
        ax.plot(wn_ref[mask], m[mask], color=CLR[cls], lw=lw, ls=ls, label=cls)
        ax.fill_between(wn_ref[mask], (m-sd)[mask], (m+sd)[mask],
                        color=CLR[cls], alpha=0.10)

    ax.axvspan(GR_D_BAND[0], GR_D_BAND[1], color='purple', alpha=0.12,
               label='Gr D-band (~1350 cm⁻¹)')
    ax.axvspan(GR_G_BAND[0], GR_G_BAND[1], color='brown',  alpha=0.12,
               label='Gr G-band (~1582 cm⁻¹)')

    # D-band comparison
    d_ng = peak_height(wn_ref, mat[cls_ng]['net'].mean(axis=0), *GR_D_BAND)
    d_gr = peak_height(wn_ref, mat[cls_gr]['net'].mean(axis=0), *GR_D_BAND)
    g_ng = peak_height(wn_ref, mat[cls_ng]['net'].mean(axis=0), *GR_G_BAND)
    g_gr = peak_height(wn_ref, mat[cls_gr]['net'].mean(axis=0), *GR_G_BAND)

    ax.set_title(f'{title}\nD-band: no-Gr={d_ng:.0f}, Gr={d_gr:.0f} | '
                 f'G-band: no-Gr={g_ng:.0f}, Gr={g_gr:.0f}',
                 fontsize=9)
    ax.set_xlabel('Raman Shift (cm⁻¹)')
    ax.set_ylabel('Net Intensity (counts)')
    ax.set_xlim(zoom_lo, zoom_hi)
    ax.legend(fontsize=7, ncol=1)
    ax.grid(alpha=0.3)

plt.tight_layout()
save_fig(fig, 'R9_GrapheneFingerprint_DGbands.png')

# ─────────────────────────────────────────────────────────────
# FIG 10 : DIFFERENCE SPECTRA (Gr minus no-Gr)
# ─────────────────────────────────────────────────────────────

fig, axes = plt.subplots(2, 1, figsize=(12, 9))
fig.suptitle('Graphene Contribution: Difference Spectra\n'
             '(Gr − no-Gr net signal: positive = graphene enhances; negative = suppresses)',
             fontsize=12, fontweight='bold')

for ax, (cls_ng, cls_gr), title in zip(axes, pairs, pair_titles):
    mask = (wn_ref >= WN_PLOT[0]) & (wn_ref <= WN_PLOT[1])
    m_ng = mat[cls_ng]['net'].mean(axis=0)
    m_gr = mat[cls_gr]['net'].mean(axis=0)
    diff = m_gr - m_ng

    # Uncertainty via propagation
    se_ng = mat[cls_ng]['net'].std(axis=0) / np.sqrt(len(data[cls_ng]))
    se_gr = mat[cls_gr]['net'].std(axis=0) / np.sqrt(len(data[cls_gr]))
    se_diff = np.sqrt(se_ng**2 + se_gr**2)

    color = CLR[cls_gr]
    ax.plot(wn_ref[mask], diff[mask], color=color, lw=2, label='Gr − no-Gr difference')
    ax.fill_between(wn_ref[mask],
                    (diff - 1.96*se_diff)[mask],
                    (diff + 1.96*se_diff)[mask],
                    color=color, alpha=0.20, label='95% CI')
    ax.axhline(0, color='black', lw=1, ls='--')
    ax.axhspan(0, diff[mask].max()*1.5, color='green', alpha=0.03)
    ax.axhspan(diff[mask].min()*1.5, 0, color='red',   alpha=0.03)

    ax.axvspan(GR_D_BAND[0], GR_D_BAND[1], color='purple', alpha=0.10, label='Gr D-band')
    ax.axvspan(GR_G_BAND[0], GR_G_BAND[1], color='brown',  alpha=0.10, label='Gr G-band')

    for pc in [1229, 1386, 1476, 1596]:
        ax.axvline(pc, color='grey', lw=0.6, ls=':', alpha=0.7)

    ax.set_title(f'{title}: {cls_gr} − {cls_ng}', fontsize=10, fontweight='bold')
    ax.set_xlabel('Raman Shift (cm⁻¹)')
    ax.set_ylabel('ΔNet Intensity (counts)')
    ax.set_xlim(WN_PLOT)
    ax.legend(fontsize=8, ncol=2)
    ax.grid(alpha=0.3)
    ax.text(0.02, 0.95, '▲ Graphene ENHANCES', transform=ax.transAxes,
            color='green', fontsize=8, va='top')
    ax.text(0.02, 0.05, '▼ Graphene SUPPRESSES', transform=ax.transAxes,
            color='red', fontsize=8, va='bottom')

plt.tight_layout()
save_fig(fig, 'R10_DifferenceSpectra_GrEffect.png')

# ─────────────────────────────────────────────────────────────
# FIG 11 : INDIVIDUAL SPECTRA WATERFALL (per class)
# ─────────────────────────────────────────────────────────────

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Individual Net Raman Spectra — All 25 per Class\n'
             '(Waterfall offset; shows within-class variability)',
             fontsize=12, fontweight='bold')
axes = axes.flatten()

mask = (wn_ref >= WN_PLOT[0]) & (wn_ref <= WN_PLOT[1])
for ax, cls in zip(axes, CLASS_ORDER):
    n = len(data[cls])
    offset_step = mat[cls]['net'][:, mask].max() * 0.8 / max(n, 1)
    for i in range(n):
        y = mat[cls]['net'][i, mask] + i * offset_step
        ax.plot(wn_ref[mask], y, color=CLR[cls], lw=0.8, alpha=0.7)
    ax.set_title(cls, color=CLR[cls], fontweight='bold')
    ax.set_xlabel('Raman Shift (cm⁻¹)')
    ax.set_ylabel('Net Intensity (offset)')
    ax.set_xlim(WN_PLOT)
    ax.grid(alpha=0.2)

plt.tight_layout()
save_fig(fig, 'R11_Waterfall_AllSpectra.png')

# ─────────────────────────────────────────────────────────────
# FIG 12 : SUMMARY DASHBOARD
# ─────────────────────────────────────────────────────────────

fig = plt.figure(figsize=(16, 12))
fig.suptitle('GRAPHENE EFFECT ON SSY RAMAN SPECTRA — SUMMARY DASHBOARD',
             fontsize=14, fontweight='bold', y=0.99)

gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.45, wspace=0.40)

# Panel A: Mean net spectra overlay (PDMS pair)
ax_a = fig.add_subplot(gs[0, :2])
mask = (wn_ref >= WN_PLOT[0]) & (wn_ref <= WN_PLOT[1])
for cls in ['SSY-PDMS', 'SSY-Gr-PDMS']:
    m = mat[cls]['net'].mean(axis=0)
    ls = '--' if 'Gr' not in cls else '-'
    ax_a.plot(wn_ref[mask], m[mask], color=CLR[cls], lw=2, ls=ls, label=cls)
ax_a.axvspan(GR_D_BAND[0], GR_D_BAND[1], color='purple', alpha=0.09)
ax_a.axvspan(GR_G_BAND[0], GR_G_BAND[1], color='brown',  alpha=0.09)
ax_a.set_title('A  Net Raman: PDMS Pair', fontsize=10, fontweight='bold')
ax_a.set_xlabel('cm⁻¹'); ax_a.set_ylabel('Net Counts')
ax_a.legend(fontsize=8); ax_a.grid(alpha=0.3); ax_a.set_xlim(WN_PLOT)

# Panel B: Mean net spectra (SiO2 pair)
ax_b = fig.add_subplot(gs[1, :2])
for cls in ['SSY-SiO2/Si', 'SSY-Gr-SiO2/Si']:
    m = mat[cls]['net'].mean(axis=0)
    ls = '--' if 'Gr' not in cls else '-'
    ax_b.plot(wn_ref[mask], m[mask], color=CLR[cls], lw=2, ls=ls, label=cls)
ax_b.axvspan(GR_D_BAND[0], GR_D_BAND[1], color='purple', alpha=0.09)
ax_b.axvspan(GR_G_BAND[0], GR_G_BAND[1], color='brown',  alpha=0.09)
ax_b.set_title('B  Net Raman: SiO2/Si Pair', fontsize=10, fontweight='bold')
ax_b.set_xlabel('cm⁻¹'); ax_b.set_ylabel('Net Counts')
ax_b.legend(fontsize=8); ax_b.grid(alpha=0.3); ax_b.set_xlim(WN_PLOT)

# Panel C: EF bar chart (both pairs)
ax_c = fig.add_subplot(gs[0:2, 2])
bar_width = 0.35
ef_pdms  = [np.mean(metrics['SSY-Gr-PDMS']['peak_heights'][lbl]) /
             (np.mean(metrics['SSY-PDMS']['peak_heights'][lbl]) + 1e-9)
             for lbl in peak_labels]
ef_sio2  = [np.mean(metrics['SSY-Gr-SiO2/Si']['peak_heights'][lbl]) /
             (np.mean(metrics['SSY-SiO2/Si']['peak_heights'][lbl]) + 1e-9)
             for lbl in peak_labels]
xi = np.arange(len(peak_labels))
ax_c.barh(xi - bar_width/2, ef_pdms, height=bar_width,
          color=CLR['SSY-Gr-PDMS'], alpha=0.75, label='Gr-PDMS/PDMS')
ax_c.barh(xi + bar_width/2, ef_sio2, height=bar_width,
          color=CLR['SSY-Gr-SiO2/Si'], alpha=0.75, label='Gr-SiO2/PDMS')
ax_c.axvline(1.0, color='black', lw=1.5, ls='--')
ax_c.set_yticks(xi); ax_c.set_yticklabels(peak_labels, fontsize=8)
ax_c.set_xlabel('Enhancement Factor')
ax_c.set_title('C  EF per SSY Peak', fontsize=10, fontweight='bold')
ax_c.legend(fontsize=7); ax_c.grid(axis='x', alpha=0.3)

# Panel D: SNR comparison
ax_d = fig.add_subplot(gs[2, 0])
snr_means = [np.mean(metrics[cls]['snr']) for cls in CLASS_ORDER]
snr_sems  = [np.std(metrics[cls]['snr'], ddof=1)/np.sqrt(len(metrics[cls]['snr']))
              for cls in CLASS_ORDER]
bars = ax_d.bar(range(len(CLASS_ORDER)), snr_means, yerr=snr_sems,
                color=[CLR[c] for c in CLASS_ORDER], alpha=0.75,
                edgecolor='black', lw=0.8, capsize=4)
ax_d.set_xticks(range(len(CLASS_ORDER)))
ax_d.set_xticklabels([c.replace('SSY-', '') for c in CLASS_ORDER],
                      fontsize=7, rotation=15)
ax_d.set_ylabel('Mean SNR')
ax_d.set_title('D  Signal-to-Noise', fontsize=10, fontweight='bold')
ax_d.grid(axis='y', alpha=0.3)

# Panel E: Background comparison
ax_e = fig.add_subplot(gs[2, 1])
bkg_means = [np.mean(metrics[cls]['bkg_mean']) for cls in CLASS_ORDER]
bkg_sems  = [np.std(metrics[cls]['bkg_mean'], ddof=1)/np.sqrt(len(metrics[cls]['bkg_mean']))
              for cls in CLASS_ORDER]
ax_e.bar(range(len(CLASS_ORDER)), bkg_means, yerr=bkg_sems,
         color=[CLR[c] for c in CLASS_ORDER], alpha=0.75,
         edgecolor='black', lw=0.8, capsize=4)
ax_e.set_xticks(range(len(CLASS_ORDER)))
ax_e.set_xticklabels([c.replace('SSY-', '') for c in CLASS_ORDER],
                      fontsize=7, rotation=15)
ax_e.set_ylabel('Background (counts)')
ax_e.set_title('E  Fluorescence Bkg', fontsize=10, fontweight='bold')
ax_e.grid(axis='y', alpha=0.3)

# Panel F: RSD% mean across peaks
ax_f = fig.add_subplot(gs[2, 2])
rsd_mean_all = []
for cls in CLASS_ORDER:
    rsds = [rsd_percent(metrics[cls]['peak_heights'][lbl]) for lbl in peak_labels]
    rsd_mean_all.append(np.nanmean(rsds))
ax_f.bar(range(len(CLASS_ORDER)), rsd_mean_all,
         color=[CLR[c] for c in CLASS_ORDER], alpha=0.75,
         edgecolor='black', lw=0.8)
ax_f.set_xticks(range(len(CLASS_ORDER)))
ax_f.set_xticklabels([c.replace('SSY-', '') for c in CLASS_ORDER],
                      fontsize=7, rotation=15)
ax_f.set_ylabel('Mean RSD%')
ax_f.set_title('F  Reproducibility', fontsize=10, fontweight='bold')
ax_f.axhline(20, color='grey', lw=1, ls='--', alpha=0.7)
ax_f.grid(axis='y', alpha=0.3)

save_fig(fig, 'R12_SummaryDashboard.png')

# ─────────────────────────────────────────────────────────────
# STATISTICAL TESTS — Mann-Whitney U (all 25 vs 25)
# ─────────────────────────────────────────────────────────────

print('\nStatistical Tests (Mann-Whitney U, two-sided):\n')
stat_results = []
for cls_ng, cls_gr in pairs:
    print(f'  {cls_gr} vs {cls_ng}:')
    for lbl in peak_labels:
        vals_ng = metrics[cls_ng]['peak_heights'][lbl]
        vals_gr = metrics[cls_gr]['peak_heights'][lbl]
        ef = np.mean(vals_gr) / (np.mean(vals_ng) + 1e-9)
        stat, p = mannwhitneyu(vals_gr, vals_ng, alternative='two-sided')
        stars = sig_stars(p)
        print(f'    {lbl}: EF={ef:.3f}, p={p:.4f} {stars}')
        stat_results.append({'pair': f'{cls_gr} / {cls_ng}',
                              'peak': lbl, 'EF': ef, 'p': p, 'sig': stars,
                              'mean_Gr': np.mean(vals_gr), 'mean_noGr': np.mean(vals_ng)})
    # SNR
    snr_ng = metrics[cls_ng]['snr']
    snr_gr = metrics[cls_gr]['snr']
    stat, p_snr = mannwhitneyu(snr_gr, snr_ng, alternative='two-sided')
    ef_snr = np.mean(snr_gr) / (np.mean(snr_ng) + 1e-9)
    print(f'    SNR: EF={ef_snr:.3f}, p={p_snr:.4f} {sig_stars(p_snr)}')
    print()

# ─────────────────────────────────────────────────────────────
# EXCEL REPORT
# ─────────────────────────────────────────────────────────────

print('Writing Excel report...')
wb = Workbook()

# ── Sheet 1: Summary ──
ws = wb.active
ws.title = 'Summary'
hdr_fill = PatternFill('solid', fgColor='1565C0')
hdr_font = Font(bold=True, color='FFFFFF')
border = Border(bottom=Side(style='thin'))

ws['A1'] = 'GRAPHENE EFFECT ON SSY RAMAN — ANALYSIS SUMMARY'
ws['A1'].font = Font(bold=True, size=14)
ws.merge_cells('A1:H1')

headers = ['Substrate Pair', 'Peak (cm⁻¹)', 'Mean no-Gr (counts)',
           'Mean Gr (counts)', 'Enhancement Factor', 'p-value',
           'Significance', 'Interpretation']
for col, h in enumerate(headers, 1):
    cell = ws.cell(row=3, column=col, value=h)
    cell.font = hdr_font
    cell.fill = hdr_fill
    cell.alignment = Alignment(horizontal='center')

row = 4
for r in stat_results:
    ef = r['EF']
    if ef > 1.1:   interp = 'ENHANCED by graphene'
    elif ef < 0.9: interp = 'Suppressed by graphene'
    else:          interp = 'No significant change'
    values = [r['pair'], r['peak'], f"{r['mean_noGr']:.1f}",
              f"{r['mean_Gr']:.1f}", f"{ef:.3f}", f"{r['p']:.4f}",
              r['sig'], interp]
    for col, v in enumerate(values, 1):
        ws.cell(row=row, column=col, value=v)
    row += 1

for col in range(1, 9):
    ws.column_dimensions[get_column_letter(col)].width = 22

# ── Sheet 2: SNR ──
ws2 = wb.create_sheet('SNR')
ws2['A1'] = 'Signal-to-Noise Ratio per Spectrum'
ws2['A1'].font = Font(bold=True, size=12)
ws2.merge_cells('A1:E1')
ws2.append([])
ws2.append(['Class', 'Mean SNR', 'Std SNR', 'Min SNR', 'Max SNR'])
for col in range(1, 6):
    ws2.cell(row=3, column=col).font = Font(bold=True)
for cls in CLASS_ORDER:
    vals = metrics[cls]['snr']
    ws2.append([cls, f'{np.mean(vals):.2f}', f'{np.std(vals):.2f}',
                f'{np.min(vals):.2f}', f'{np.max(vals):.2f}'])

# ── Sheet 3: Reproducibility ──
ws3 = wb.create_sheet('Reproducibility')
ws3['A1'] = 'Reproducibility: RSD% at Each SSY Peak'
ws3['A1'].font = Font(bold=True, size=12)
ws3.merge_cells('A1:G1')
ws3.append([])
ws3.append(['Class'] + peak_labels + ['Mean RSD%'])
for col in range(1, len(peak_labels)+3):
    ws3.cell(row=3, column=col).font = Font(bold=True)
for cls in CLASS_ORDER:
    rsds = [rsd_percent(metrics[cls]['peak_heights'][lbl]) for lbl in peak_labels]
    ws3.append([cls] + [f'{r:.1f}%' for r in rsds] + [f'{np.nanmean(rsds):.1f}%'])

# ── Sheet 4: Raw EF values ──
ws4 = wb.create_sheet('Peak_Heights')
ws4['A1'] = 'Per-Spectrum Peak Heights (net counts after ALS baseline)'
ws4['A1'].font = Font(bold=True, size=12)
ws4.merge_cells('A1:H1')
ws4.append([])
header_row = ['Class', 'Spectrum file'] + peak_labels
ws4.append(header_row)
for col in range(1, len(header_row)+1):
    ws4.cell(row=3, column=col).font = Font(bold=True)
for cls in CLASS_ORDER:
    for i, d in enumerate(data[cls]):
        row_vals = [cls, d['file']]
        for lbl in peak_labels:
            row_vals.append(f"{metrics[cls]['peak_heights'][lbl][i]:.2f}")
        ws4.append(row_vals)

wb.save(os.path.join(OUT_DIR, 'GrapheneRaw_Results.xlsx'))
print('  Saved: GrapheneRaw_Results.xlsx')

# ─────────────────────────────────────────────────────────────
# CONSOLE SUMMARY
# ─────────────────────────────────────────────────────────────

print()
print('=' * 60)
print('SCIENTIFIC CONCLUSIONS')
print('=' * 60)
for cls_ng, cls_gr in pairs:
    print(f'\n{cls_gr.replace("SSY-","")} vs {cls_ng.replace("SSY-","")}:')
    bkg_ng = np.mean(metrics[cls_ng]['bkg_mean'])
    bkg_gr = np.mean(metrics[cls_gr]['bkg_mean'])
    snr_ng = np.mean(metrics[cls_ng]['snr'])
    snr_gr = np.mean(metrics[cls_gr]['snr'])
    pct_bkg = 100*(bkg_ng - bkg_gr)/bkg_ng
    pct_snr = 100*(snr_gr - snr_ng)/snr_ng
    print(f'  Fluorescence background: {pct_bkg:+.1f}% (Gr vs no-Gr)')
    print(f'  Mean SNR change:         {pct_snr:+.1f}%')
    for lbl in peak_labels:
        ef = (np.mean(metrics[cls_gr]['peak_heights'][lbl]) /
              (np.mean(metrics[cls_ng]['peak_heights'][lbl]) + 1e-9))
        print(f'  EF at {lbl}: {ef:.2f}', end='')
        if ef > 1.05: print(' [ENHANCED]')
        elif ef < 0.95: print(' [suppressed]')
        else: print(' [no change]')
print()
print(f'Output saved to: {OUT_DIR}')
print('Done.')
