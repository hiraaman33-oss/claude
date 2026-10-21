# =============================================================================
# ACACIA EVs vs NVs  |  Figures 3 and 4 in the layout of Roda et al.
# (Nanomaterials 2023, 13, 699, Figure 3)
#
#   Figure 3: mean SERS spectra of EVs and NVs (+/- SD), stacked, with the
#             difference spectrum (EVs - NVs); grey bands = bands of Table S1
#             that differ significantly (Welch's t-test on site means,
#             Bonferroni-corrected p < 0.05); broken axis fingerprint // CH.
#   Figure 4: PCA score plots (2 SD ellipses), LD1 box plots with test on
#             site means, PC1/PC2 loadings - fingerprint and CH-stretching.
#
# Every value comes from EV_Acacia_Raman_Data.xlsx and NV_Acacia_Raman_Data.xlsx
# processed by acacia_pipeline.py. All numbers used in the figures are also
# written to Acacia_Fig3_Fig4_values.txt.
#
# HOW TO RUN:
#   pip install numpy scipy pandas openpyxl scikit-learn matplotlib
#   python Acacia_Fig3_Fig4.py  EV_Acacia_Raman_Data.xlsx  NV_Acacia_Raman_Data.xlsx
# =============================================================================

import sys
import warnings

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from scipy.stats import ttest_ind

import acacia_pipeline as ap

warnings.filterwarnings('ignore', category=FutureWarning)

EV_FILE = sys.argv[1] if len(sys.argv) > 1 else 'EV_Acacia_Raman_Data.xlsx'
NV_FILE = sys.argv[2] if len(sys.argv) > 2 else 'NV_Acacia_Raman_Data.xlsx'
OUT_FIG3 = 'Fig3_Acacia_MeanSpectra'
OUT_FIG4 = 'Fig4_Acacia_PCA_LDA'
OUT_TXT = 'Acacia_Fig3_Fig4_values.txt'

C_EV = '#0B5A7E'     # same palette as the manuscript figures
C_NV = '#B5541C'
C_SUB = '#7F7F7F'
BAND = '#D3D3D3'
NAMES = {0: 'EVs', 1: 'NVs'}
COLS = {0: C_EV, 1: C_NV}

plt.rcParams.update({'font.family': 'sans-serif',
                     'font.sans-serif': ['Arial', 'Liberation Sans', 'DejaVu Sans'],
                     'font.size': 9, 'axes.linewidth': 0.8,
                     'xtick.direction': 'out', 'ytick.direction': 'out'})

# ----------------------------------------------------------------------------
# DATA
# ----------------------------------------------------------------------------
data, y, sites = ap.build_dataset(EV_FILE, NV_FILE)
bands = ap.band_table(data, y, sites)
sig = bands[bands.p_bonf < 0.05]
folds = ap.cv_folds(y, sites)
log = []

# ============================================================================
# FIGURE 3 - stacked mean spectra + difference spectrum
# ============================================================================
fig = plt.figure(figsize=(7.0, 6.2))
gs = fig.add_gridspec(1, 2, width_ratios=[1790 - 250, (3095 - 2760) * 1.6], wspace=0.04)
axL, axR = fig.add_subplot(gs[0]), fig.add_subplot(gs[1], sharey=None)

means = {}
for w in ['fingerprint', 'ch']:
    g, X = ap.region(data, w)
    means[w] = {c: (X[y == c].mean(0), X[y == c].std(0)) for c in (0, 1)}
    means[w]['grid'] = g

# common vertical scale: offsets from the largest mean+SD span
span = max(np.ptp(means[w][c][0]) + 2 * means[w][c][1].max()
           for w in ['fingerprint', 'ch'] for c in (0, 1))
OFF = {0: 2.05 * span, 1: 1.10 * span, 'sub': 0.42 * span}


def spread(pos, gap):
    """Shift label positions apart so neighbouring labels do not overlap."""
    out = list(pos)
    for _ in range(100):
        for i in range(1, len(out)):
            if out[i] - out[i - 1] < gap:
                mid = (out[i] + out[i - 1]) / 2
                out[i - 1], out[i] = mid - gap / 2, mid + gap / 2
    return out

for ax, w in [(axL, 'fingerprint'), (axR, 'ch')]:
    g = means[w]['grid']
    for c in (0, 1):
        m, s = means[w][c]
        ax.fill_between(g, m - s + OFF[c], m + s + OFF[c], color=COLS[c], alpha=0.18, lw=0)
        ax.plot(g, m + OFF[c], color=COLS[c], lw=0.9)
    diff = means[w][0][0] - means[w][1][0]
    ax.plot(g, diff + OFF['sub'], color=C_SUB, lw=0.9)
    ax.axhline(OFF['sub'], color='k', lw=0.5)
    # grey bands: significant Table S1 bands in this window
    sw = sig[sig.window == w]
    gap = 34 if w == 'fingerprint' else 34 / 1.6      # same distance on paper
    y_band = OFF['sub'] - 0.30 * span
    for (_, b), lx in zip(sw.iterrows(), spread(list(sw.band.astype(float)), gap)):
        c = (b.lo + b.hi) / 2
        ax.axvspan(b.lo, b.hi, ymin=0.0, color=BAND, alpha=0.7, lw=0, zorder=0)
        ax.plot([c, c, lx], [y_band, y_band - 0.06 * span, y_band - 0.14 * span],
                color='0.35', lw=0.5)
        ax.text(lx, y_band - 0.17 * span, f'{b.band} cm$^{{-1}}$',
                rotation=90, ha='center', va='top', fontsize=7.5)

axL.set_xlim(250, 1790)
axR.set_xlim(2760, 3095)
for ax in (axL, axR):
    ax.set_ylim(-0.62 * span, OFF[0] + 1.0 * span)
    ax.set_yticks([])
    for s_ in ['top', 'right']:
        ax.spines[s_].set_visible(False)
axR.spines['left'].set_visible(False)
axL.set_xticks([500, 1000, 1500])
axR.set_xticks([2800, 3000])
axL.set_ylabel('Intensity (a.u.)', fontsize=10)

# trace labels on the right, as in Roda et al.
for key, txt in [(0, 'Acacia EVs'), (1, 'Acacia NVs'), ('sub', 'Subtraction\nspectrum\n(EVs - NVs)')]:
    axR.text(3110, OFF[key] + 0.15 * span, txt, fontsize=8.5, va='center', ha='left',
             clip_on=False)

# // break marks on the x-axis
d = 0.015
k = (3095 - 2760) * 1.6 / (1790 - 250)
axL.plot((1 - d * k, 1 + d * k), (-d * 1.5, d * 1.5), transform=axL.transAxes,
         color='k', clip_on=False, lw=0.8)
axR.plot((-d, d), (-d * 1.5, d * 1.5), transform=axR.transAxes,
         color='k', clip_on=False, lw=0.8)
fig.text(0.47, 0.025, 'Raman shift (cm$^{-1}$)', ha='center', fontsize=10)
fig.subplots_adjust(left=0.06, right=0.82, bottom=0.10, top=0.98)
for ext in ('png', 'tiff'):
    fig.savefig(f'{OUT_FIG3}.{ext}', dpi=600,
                pil_kwargs={'compression': 'tiff_lzw'} if ext == 'tiff' else None)
plt.close(fig)

log.append('FIGURE 3')
log.append('Grey bands = Table S1 bands with Bonferroni-corrected p < 0.05 (Welch, site means, n = 6):')
for _, b in bands.iterrows():
    log.append(f'  {b.band:>5} ({b.lo}-{b.hi})  EV {b.ev_mean:6.2f} +/- {b.ev_sd:.2f}   '
               f'NV {b.nv_mean:6.2f} +/- {b.nv_sd:.2f}   p_bonf = {b.p_bonf:.3g}'
               + ('   *' if b.p_bonf < 0.05 else ''))
r = ap.ch_ratio(data, y, sites)
log.append(f'  I2960/I2930: EV {r[0].mean():.2f} +/- {r[0].std(ddof=1):.2f}, '
           f'NV {r[1].mean():.2f} +/- {r[1].std(ddof=1):.2f}')


# ============================================================================
# FIGURE 4 - scores, LD1 box plots, loadings
# ============================================================================
def ellipse_2sd(ax, P, color):
    cov = np.cov(P.T)
    vals, vecs = np.linalg.eigh(cov)
    ang = np.degrees(np.arctan2(vecs[1, 1], vecs[0, 1]))
    w_, h_ = 2 * 2 * np.sqrt(vals[1]), 2 * 2 * np.sqrt(vals[0])
    ax.add_patch(Ellipse(P.mean(0), w_, h_, angle=ang, fc='none', ec=color, lw=1.0))


def stars(p):
    return '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'n.s.'


fig, axs = plt.subplots(2, 3, figsize=(7.2, 5.2),
                        gridspec_kw={'width_ratios': [1.15, 0.75, 1.35],
                                     'wspace': 0.42, 'hspace': 0.45})
letters = iter('abcdef')
log.append('\nFIGURE 4')
for row, (w, title) in enumerate([('fingerprint', 'Fingerprint (250-1790 cm$^{-1}$)'),
                                  ('ch', 'CH-stretching (2760-3095 cm$^{-1}$)')]):
    g, X = ap.region(data, w)
    pca, T = ap.pca_fit(X)
    # the sign of a PC is arbitrary: orient PC1 as described in the text,
    # i.e. positive loading at ~1580 cm-1 (fingerprint) / ~2960 cm-1 (CH)
    ref = 1580 if w == 'fingerprint' else 2960
    sgn = np.sign(pca.components_[0][np.argmin(np.abs(g - ref))])
    T[:, 0] *= sgn
    load1 = pca.components_[0] * sgn
    load2 = pca.components_[1]
    ev1, ev2 = pca.explained_variance_ratio_[:2] * 100

    # --- scores
    ax = axs[row, 0]
    for c in (0, 1):
        P = T[y == c, :2]
        ax.scatter(P[:, 0], P[:, 1], s=6, color=COLS[c], lw=0, alpha=0.85, label=NAMES[c])
        ellipse_2sd(ax, P, COLS[c])
    ax.axhline(0, color='k', lw=0.5)
    ax.axvline(0, color='k', lw=0.5)
    ax.set_xlabel(f'PC1 ({ev1:.1f}%)')
    ax.set_ylabel(f'PC2 ({ev2:.1f}%)')
    ylo, yhi = ax.get_ylim()
    ax.set_ylim(ylo, yhi + 0.28 * (yhi - ylo))          # room for the legend
    ax.legend(frameon=False, fontsize=7.5, loc='upper left', ncol=2,
              handletextpad=0.1, columnspacing=0.8, markerscale=1.5)
    ax.set_title(title, fontsize=8.5, loc='left')

    # --- LD1 box plot (PCA 5 PCs + LDA fitted on all spectra, descriptive)
    ld = ap.lda_scores(X, y)
    ld *= np.sign(ld[y == 1].mean())          # NVs positive, as in Roda et al.
    ax = axs[row, 1]
    rng = np.random.default_rng(0)            # jitter of the dots only
    for c in (0, 1):
        v = ld[y == c]
        ax.boxplot(v, positions=[c], widths=0.5, showfliers=False, patch_artist=True,
                   boxprops=dict(fc=COLS[c], alpha=0.30, ec=COLS[c]),
                   medianprops=dict(color=COLS[c]), whiskerprops=dict(color=COLS[c]),
                   capprops=dict(color=COLS[c]))
        ax.scatter(c + rng.uniform(-0.15, 0.15, len(v)), v, s=3, color=COLS[c], lw=0, alpha=0.8)
    sm = ap.site_means(ld, y, sites)
    p = ttest_ind(sm[0], sm[1], equal_var=False).pvalue
    top = ld.max() + 0.08 * np.ptp(ld)
    ax.plot([0, 1], [top, top], color='k', lw=0.6)
    ax.text(0.5, top, stars(p), ha='center', va='bottom', fontsize=8)
    ax.set_xticks([0, 1])
    ax.set_xticklabels([NAMES[0], NAMES[1]])
    ax.set_ylabel('Canonical variable 1')
    ax.set_ylim(ld.min() - 0.18 * np.ptp(ld), top + 0.15 * np.ptp(ld))
    ax.text(0.5, 0.02, "Welch's t-test on site means\n(n = 6), " + stars(p) + f' p = {p:.1e}',
            transform=ax.transAxes, ha='center', va='bottom', fontsize=6)

    # --- loadings
    ax = axs[row, 2]
    ax.plot(g, load1, color=C_EV, lw=0.8, label='PC1')
    ax.plot(g, load2, color=C_NV, lw=0.8, label='PC2')
    ax.axhline(0, color='k', lw=0.5)
    ax.set_xlim(g[0], g[-1])
    ax.set_xlabel('Raman shift (cm$^{-1}$)')
    ax.set_ylabel('Loading (a.u.)')
    ax.legend(frameon=False, fontsize=7.5, loc='upper left', handlelength=1.2)

    for ax in axs[row]:
        for s_ in ['top', 'right']:
            ax.spines[s_].set_visible(False)
        ax.text(-0.30 if ax is axs[row, 1] else -0.22, 1.10, f'({next(letters)})',
                transform=ax.transAxes, fontsize=10, va='bottom')

    # numbers behind the panels
    def cnt(pred):
        return (((pred == 0) & (y == 0)).sum(), ((pred == 1) & (y == 1)).sum())
    lda_pred = ap.cv_pca_lda(X, y, sites, folds)
    pls_pred = ap.cv_plsda(X, y, folds)
    log.append(f'{w}: PC1 {ev1:.1f}%  PC2 {ev2:.1f}%')
    log.append(f'  LD1 site means EV {np.round(sm[0], 2)}  NV {np.round(sm[1], 2)}  '
               f'Welch p = {p:.2e} ({stars(p)})')
    for name, pred in [('PCA-LDA', lda_pred), ('PLS-DA', pls_pred)]:
        e, n_ = cnt(pred)
        log.append(f'  {name}: {(pred == y).mean() * 100:.1f}%  (EVs {e}/120, NVs {n_}/120)')

fig.subplots_adjust(left=0.08, right=0.99, bottom=0.09, top=0.93)
for ext in ('png', 'tiff'):
    fig.savefig(f'{OUT_FIG4}.{ext}', dpi=600,
                pil_kwargs={'compression': 'tiff_lzw'} if ext == 'tiff' else None)
plt.close(fig)

log.append('\nCross-validation: StratifiedGroupKFold(5, shuffle=True, random_state=0), '
           'grouped by measurement site.')
open(OUT_TXT, 'w').write('\n'.join(log) + '\n')
print('\n'.join(log))
