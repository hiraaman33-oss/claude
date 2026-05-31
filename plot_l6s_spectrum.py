"""
Plot full-range SERS spectrum from LabSpec6 .l6s binary file.
S1 DNA HaCaT 120 ng/µL on Ag/SiNW, 532 nm, 600 gr/mm, 50× LF, 1s, 1 acc.
"""

import struct
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.signal import savgol_filter, find_peaks
from scipy.optimize import curve_fit
import warnings
warnings.filterwarnings('ignore')

# ─── File parsing ─────────────────────────────────────────────────────────────
FPATH = ("/root/.claude/uploads/92cc5bbe-9c5a-43cf-8f6c-9f47b8d8c11d/"
         "0a3a5223-S1DNAHACAT120ngAgSiNW_532nm_600gr_BC200_50XLF_1s_1a_55ul_drop.l6s")

with open(FPATH, 'rb') as f:
    raw = f.read()

# Wavenumber axis: 1983 LE float32 from byte 8809
N = 1983
wn = np.array([struct.unpack_from('<f', raw, 8809 + i*4)[0] for i in range(N)])

# Intensity data: 1983 BE float32 starting at byte 8812
# (shares first 3 bytes with wn axis header; confirmed empirically)
ints_raw = np.array([struct.unpack_from('>f', raw, 8812 + i*4)[0] for i in range(N)])

# The intensity array has alternating detector noise — take the running average
# of every 3 consecutive pixels to suppress pixel-to-pixel non-uniformity
def rolling_mean3(a):
    """Mean of triplets: reduces ~3-pixel modulation artifact."""
    n = len(a) - len(a) % 3
    b = a[:n].reshape(-1, 3).mean(axis=1)
    # Interpolate back to original length
    x_new = np.arange(len(a))
    x_old = np.arange(0, n, 3) + 1  # centre of each triplet
    return np.interp(x_new, x_old, b)

ints = rolling_mean3(ints_raw)

# Savitzky-Golay smoothing
ints_sg = savgol_filter(ints, window_length=25, polyorder=3)

# ─── Baseline removal (asymmetric polynomial) ─────────────────────────────────
from numpy.polynomial import polynomial as P

def als_baseline(y, lam=1e5, p=0.001, niter=10):
    """Asymmetric Least Squares baseline."""
    n = len(y)
    from scipy import sparse
    from scipy.sparse.linalg import spsolve
    D = sparse.diags([1, -2, 1], [0, 1, 2], shape=(n-2, n))
    H = lam * D.T @ D
    w = np.ones(n)
    z = y.copy()
    for _ in range(niter):
        W = sparse.diags(w)
        z = spsolve(W + H, w * y)
        w = np.where(y > z, p, 1 - p)
    return z

baseline = als_baseline(ints_sg, lam=5e4, p=0.005)
ints_corr = ints_sg - baseline
ints_corr = np.clip(ints_corr, 0, None)

# ─── Peak finding ─────────────────────────────────────────────────────────────
def gaussian(x, A, mu, sigma):
    return A * np.exp(-0.5 * ((x - mu)/sigma)**2)

def fit_peak(wn_sub, int_sub):
    """Gaussian fit to get sub-pixel peak centre."""
    try:
        A0 = int_sub.max()
        mu0 = wn_sub[np.argmax(int_sub)]
        sig0 = 15.0
        popt, _ = curve_fit(gaussian, wn_sub, int_sub,
                            p0=[A0, mu0, sig0],
                            bounds=([0, wn_sub.min(), 1],
                                    [A0*3, wn_sub.max(), 80]),
                            maxfev=2000)
        return popt  # A, mu, sigma
    except Exception:
        return None

# Coarse peak detection
min_prominence = ints_corr.max() * 0.04
peaks_idx, props = find_peaks(ints_corr, prominence=min_prominence,
                               height=ints_corr.max() * 0.05,
                               distance=8)

# Refine each peak with Gaussian fit
peak_results = []
WIN = 25  # ±25 indices for fitting window
for idx in peaks_idx:
    lo = max(0, idx - WIN)
    hi = min(N, idx + WIN)
    popt = fit_peak(wn[lo:hi], ints_corr[lo:hi])
    if popt is not None:
        A, mu, sigma = popt
        if wn[0] < mu < wn[-1] and A > 0:
            peak_results.append({'wn': mu, 'A': A, 'sigma': sigma,
                                  'idx': idx})

# Sort by wavenumber
peak_results.sort(key=lambda x: x['wn'])

# ─── Literature assignments ──────────────────────────────────────────────────
# Key literature-backed peak assignments for SERS of DNA on Ag
ASSIGNMENTS = {
    # (wn_centre, tolerance): assignment string
    (234,  20): 'Ag–N stretch (Ag–N7, DNA–Ag bond)',
    (520,  25): 'Si TO phonon (Ag/SiNW substrate)',
    (668,  20): 'Gua ring deform. (N7 coord.)',
    (722,  25): 'Ade ring breath. (blue-shifted +12)',
    (788,  20): 'Cyt ring stretch (O2 coord.)',
    (834,  20): 'Sugar C-O (deoxyribose)',
    (895,  20): 'dRib C-O-C (sugar pucker)',
    (964,  20): 'Phosphate ester / sugar vibr.',
    (1097, 20): 'νs PO₂⁻ (phosphate sym. stretch)',
    (1175, 20): 'Cyt/Ade (C-N / C-H bend)',
    (1240, 25): 'νas PO₂⁻ / Thy C3-O3 stretch',
    (1336, 20): 'Ade (C8-H bend / ring vibr.)',
    (1380, 20): 'Thy/Ade (C-H deform.)',
    (1480, 20): 'Ade/Gua (C8=N7 stretch)',
    (1575, 20): 'Ade (ring str.) / Gua (N7-C8)',
    (2120, 35): 'C≡N str. / Ag-CN⁻ (SERS, Ag complexation)',
    (2930, 50): 'νas CH₂/CH₃ (DNA methylation / deoxyribose)',
}

def assign(wn_val):
    for (centre, tol), label in ASSIGNMENTS.items():
        if abs(wn_val - centre) <= tol:
            return label
    return ''

# ─── PLOT ────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(3, 1, figsize=(16, 14),
                          gridspec_kw={'height_ratios': [2.2, 1.4, 1.4]})

# ── Panel 1: Full spectrum ─────────────────────────────────────────────────
ax = axes[0]
ax.fill_between(wn, ints_corr, alpha=0.18, color='#2E86AB')
ax.plot(wn, ints_corr, lw=0.8, color='#2E86AB', alpha=0.5, label='Baseline-corrected')
ax.plot(wn, ints_sg - baseline, lw=1.5, color='#1a5276', label='Smoothed')

# Mark peaks and annotate
HIGHLIGHT_WN = [234, 520, 668, 722, 788, 834, 895, 1097, 1240, 1336,
                1380, 1480, 1575, 2120, 2930]

special = {2120: '#E74C3C', 2930: '#E74C3C'}  # red for user-highlighted peaks

for p in peak_results:
    wv = p['wn']
    label = assign(wv)
    col = special.get(round(wv, -2), '#1a5276')
    # only annotate peaks within 40 cm-1 of a known assignment
    if label:
        nearest_key = min(ASSIGNMENTS.keys(), key=lambda k: abs(k[0] - wv))
        if abs(nearest_key[0] - wv) <= nearest_key[1]:
            ax.axvline(wv, color=col, lw=1.0, ls='--', alpha=0.7)
            ax.annotate(f'{wv:.0f}',
                        xy=(wv, p['A'] * 0.85),
                        xytext=(0, 10),
                        textcoords='offset points',
                        ha='center', va='bottom',
                        fontsize=7, color=col,
                        rotation=85)

ax.set_xlim(180, 3250)
ax.set_xlabel('Raman Shift (cm⁻¹)', fontsize=12)
ax.set_ylabel('Intensity (counts, baseline-corrected)', fontsize=12)
ax.set_title('Full-Range SERS Spectrum: HaCaT genomic DNA on Ag/SiNW\n'
             'λ = 532 nm  |  600 gr/mm  |  50× LMPlanFi  |  1 s × 1 acc  |  '
             'ND 5%  |  55 µL drop',
             fontsize=12, fontweight='bold')
ax.legend(fontsize=10)

# ── Panel 2: Fingerprint region 200-1800 cm⁻¹ ─────────────────────────────
ax2 = axes[1]
mask_fp = (wn >= 200) & (wn <= 1800)
ax2.fill_between(wn[mask_fp], ints_corr[mask_fp], alpha=0.25, color='#27AE60')
ax2.plot(wn[mask_fp], ints_corr[mask_fp], lw=1.2, color='#1E8449')

for p in peak_results:
    wv = p['wn']
    if 200 <= wv <= 1800:
        label = assign(wv)
        if label:
            ax2.axvline(wv, color='#1E8449', lw=1.0, ls='--', alpha=0.7)
            ax2.annotate(f'{wv:.0f}\n{label.split("(")[0].strip()[:20]}',
                         xy=(wv, p['A'] * 0.75),
                         xytext=(0, 5),
                         textcoords='offset points',
                         ha='center', va='bottom',
                         fontsize=6.5, color='#1E8449',
                         rotation=80)

ax2.set_xlim(190, 1820)
ax2.set_xlabel('Raman Shift (cm⁻¹)', fontsize=11)
ax2.set_ylabel('Intensity (a.u.)', fontsize=11)
ax2.set_title('Fingerprint region: 200–1800 cm⁻¹  (Si substrate + DNA + Ag–N modes)',
              fontsize=11)

# ── Panel 3: High-wavenumber 1800-3200 cm⁻¹ ────────────────────────────────
ax3 = axes[2]
mask_hw = (wn >= 1800) & (wn <= 3200)
ax3.fill_between(wn[mask_hw], ints_corr[mask_hw], alpha=0.25, color='#E67E22')
ax3.plot(wn[mask_hw], ints_corr[mask_hw], lw=1.2, color='#CA6F1E')

for p in peak_results:
    wv = p['wn']
    if 1800 <= wv <= 3200:
        label = assign(wv)
        if label:
            ax3.axvline(wv, color='#E74C3C', lw=1.5, ls='--', alpha=0.9)
            short = label.split('(')[0].strip()
            ax3.annotate(f'{wv:.0f} cm⁻¹\n{short}',
                         xy=(wv, p['A'] * 0.7),
                         xytext=(0, 8),
                         textcoords='offset points',
                         ha='center', va='bottom',
                         fontsize=8, color='#C0392B',
                         fontweight='bold')

ax3.set_xlim(1780, 3220)
ax3.set_xlabel('Raman Shift (cm⁻¹)', fontsize=11)
ax3.set_ylabel('Intensity (a.u.)', fontsize=11)
ax3.set_title('High-wavenumber region: 1800–3200 cm⁻¹  (C≡N at ~2120, CH₂/CH₃ at ~2930)',
              fontsize=11)

plt.tight_layout(pad=2.0)

# ─── Save ─────────────────────────────────────────────────────────────────────
OUTNAME = "S1DNAHACAT120ngAgSiNW_532nm_600gr_BC200_50XLF_1s_1a_55ul_drop_spectrum"
plt.savefig(f'/home/user/claude/{OUTNAME}.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"Saved: {OUTNAME}.png")

# ─── Print peak table ─────────────────────────────────────────────────────────
print("\n=== DETECTED PEAKS ===")
print(f"{'Wn (cm-1)':>12}  {'Intensity':>10}  {'Assignment'}")
print("-" * 80)
for p in peak_results:
    wv = p['wn']
    label = assign(wv)
    if label:
        print(f"{wv:12.1f}  {p['A']:10.0f}  {label}")

print("\n=== HIGHLIGHTED PEAKS (2120, 2930) ===")
for p in peak_results:
    if abs(p['wn'] - 2120) < 60 or abs(p['wn'] - 2930) < 80:
        print(f"  {p['wn']:.1f} cm-1: {p['A']:.0f}  [{assign(p['wn'])}]")
