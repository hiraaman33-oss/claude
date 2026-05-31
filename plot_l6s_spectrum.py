"""
Full-range SERS spectrum from LabSpec6 .l6s binary file.
HaCaT genomic DNA on Ag/SiNW, 532 nm, 1s × 1 acc, 200–3200 cm⁻¹.
"""

import struct, numpy as np, warnings
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.ticker import MultipleLocator
from scipy.signal import savgol_filter, find_peaks
from scipy.optimize import curve_fit
from scipy import sparse
from scipy.sparse.linalg import spsolve
warnings.filterwarnings('ignore')

# ─── 1.  Parse binary ─────────────────────────────────────────────────────────
FPATH = ("/root/.claude/uploads/92cc5bbe-9c5a-43cf-8f6c-9f47b8d8c11d/"
         "0a3a5223-S1DNAHACAT120ngAgSiNW_532nm_600gr_BC200_50XLF_1s_1a_55ul_drop.l6s")

with open(FPATH, 'rb') as f:
    raw = f.read()

N  = 1983
wn = np.array([struct.unpack_from('<f', raw, 8809 + i*4)[0] for i in range(N)])
A  = np.array([struct.unpack_from('>f', raw,  544 + i*4)[0] for i in range(N)])  # background
B  = np.array([struct.unpack_from('>f', raw, 8812 + i*4)[0] for i in range(N)])  # raw signal

# Signal = raw − background
signal = B - A                          # background-subtracted

# ─── 2.  Baseline & smoothing ─────────────────────────────────────────────────
def als_baseline(y, lam=5e4, p=0.01, niter=15):
    """Asymmetric Least-Squares baseline."""
    n = len(y)
    D = sparse.diags([1,-2,1],[0,1,2], shape=(n-2,n), dtype=float)
    H = lam * D.T @ D
    w = np.ones(n); z = y.copy()
    for _ in range(niter):
        W = sparse.diags(w.tolist(), 0)
        z = spsolve((W + H).tocsc(), w * y)
        w = np.where(y > z, p, 1-p)
    return z

sm   = savgol_filter(signal, 21, 3)
bl   = als_baseline(sm)
spec = np.clip(sm - bl, 0, None)

# ─── 3.  Peak finding + Gaussian refinement ───────────────────────────────────
def gauss(x, A, mu, s):
    return A * np.exp(-0.5*((x-mu)/s)**2)

def fit_centre(wn_w, int_w):
    try:
        p0 = [int_w.max(), wn_w[np.argmax(int_w)], 12.0]
        po,_ = curve_fit(gauss, wn_w, int_w, p0=p0,
                         bounds=([0, wn_w.min(), 1],[p0[0]*3, wn_w.max(), 80]),
                         maxfev=2000)
        return po
    except: return None

thresh = spec.max() * 0.04
peak_idx, _ = find_peaks(spec, prominence=thresh, height=thresh, distance=6)

peaks = []
WIN = 22
for idx in peak_idx:
    lo, hi = max(0,idx-WIN), min(N,idx+WIN)
    po = fit_centre(wn[lo:hi], spec[lo:hi])
    if po is not None and wn[0] < po[1] < wn[-1] and po[0] > 0:
        peaks.append(dict(wn=po[1], A=po[0], s=abs(po[2])))

peaks.sort(key=lambda x: x['wn'])

# ─── 4.  Literature assignments ───────────────────────────────────────────────
ASSIGN = {
    (218, 40): 'Ag phonon / Ag–N str.',
    (335, 30): 'Ag–Ag phonon (NP chain)',
    (520, 25): 'Si TO phonon (substrate)',
    (645, 25): 'Gua + Cyt ring (C8=N7)',
    (685, 20): 'Gua ring deform. (N7)',
    (722, 25): 'Ade ring breath. (N7)',
    (788, 22): 'Cyt ring stretch',
    (834, 22): 'Sugar C-O (deoxyribose)',
    (895, 22): 'dRib C-O-C stretch',
    (964, 22): 'Phosphate/sugar',
    (1097,22): 'νs PO₂⁻',
    (1175,22): 'Cyt/Ade C-N bend',
    (1240,25): 'νas PO₂⁻',
    (1299,22): 'Thy C-N / C-C',
    (1336,22): 'Ade C8-H bend',
    (1380,22): 'Thy/Ade C-H deform.',
    (1480,22): 'Ade/Gua C=N str.',
    (1575,22): 'Ade ring str.',
    (2080,60): 'C≡N str. / Ag-CN (SERS)',
    (2930,70): 'νas CH₂/CH₃ (DNA)',
}

def assign(w):
    for (c,tol),lab in ASSIGN.items():
        if abs(w-c)<=tol: return lab, c
    return '', None

# ─── 5.  FIGURE ───────────────────────────────────────────────────────────────
BLUE  = '#1A5276'
GREEN = '#1E8449'
RED   = '#C0392B'
GOLD  = '#D4AC0D'

fig = plt.figure(figsize=(17, 16))
gs  = fig.add_gridspec(3, 1, hspace=0.38,
                       height_ratios=[2.0, 1.6, 1.6])
ax_full = fig.add_subplot(gs[0])
ax_fp   = fig.add_subplot(gs[1])   # fingerprint 200-1800
ax_hw   = fig.add_subplot(gs[2])   # high-wn 1800-3200

def draw_panel(ax, wn, spec, peaks, wn_lo, wn_hi, title, color,
               annot_frac=0.72, fontsize=8):
    msk = (wn >= wn_lo) & (wn <= wn_hi)
    ax.fill_between(wn[msk], spec[msk], alpha=0.20, color=color)
    ax.plot(wn[msk], spec[msk], lw=1.3, color=color)
    ax.set_xlim(wn_lo, wn_hi)
    ax.set_xlabel('Raman Shift (cm⁻¹)', fontsize=11)
    ax.set_ylabel('Intensity (a.u.)', fontsize=11)
    ax.set_title(title, fontsize=11, fontweight='bold', pad=6)
    ax.xaxis.set_minor_locator(MultipleLocator(50))
    ax.tick_params(which='minor', length=3)

    ymax = spec[msk].max() if spec[msk].size else 1

    # Alternate annotation rows to avoid overlap
    row_thresh = annot_frac * ymax
    labelled = []
    for pi, p in enumerate(peaks):
        w = p['wn']
        if not (wn_lo <= w <= wn_hi): continue
        lab, centre = assign(w)
        col = RED if (abs(w-2080)<=70 or abs(w-2930)<=70) else BLUE
        lw  = 1.8 if col==RED else 1.0
        ax.axvline(w, color=col, lw=lw, ls='--', alpha=0.75)
        # Annotate
        row = pi % 2   # alternate above and below
        y_off = 10 + row * 28
        short = f'{w:.0f}'
        ax.annotate(short,
                    xy=(w, p['A'] * annot_frac),
                    xytext=(0, y_off), textcoords='offset points',
                    ha='center', va='bottom',
                    fontsize=fontsize, color=col, fontweight='bold',
                    rotation=90, clip_on=True)
        labelled.append(w)

# ── Full range ─────────────────────────────────────────────────────────────────
draw_panel(ax_full, wn, spec, peaks, 200, 3200,
           'Full SERS Spectrum  200–3200 cm⁻¹\n'
           'HaCaT Genomic DNA on Ag/SiNW  |  532 nm  |  1 s × 1 acc  |  50× LMPlanFi',
           BLUE, annot_frac=0.55, fontsize=7)

# Annotate key peaks in full view only for landmark ones
landmark = {520:'Si substrate', 2080:'C≡N/Ag-CN', 2930:'CH₂/CH₃'}
for w_nom, lab in landmark.items():
    # find closest detected peak
    close = [p for p in peaks if abs(p['wn']-w_nom) < 80]
    if not close: continue
    best = max(close, key=lambda x: x['A'])
    col = RED if w_nom > 1800 else GREEN
    ax_full.annotate(f"  {best['wn']:.0f} cm⁻¹\n  {lab}",
                     xy=(best['wn'], best['A']),
                     xytext=(best['wn']+40, best['A']*1.05),
                     fontsize=8.5, color=col, fontweight='bold',
                     arrowprops=dict(arrowstyle='->', color=col, lw=1.3))

# ── Fingerprint 200-1800 ───────────────────────────────────────────────────────
msk_fp = (wn >= 200) & (wn <= 1800)
ax_fp.fill_between(wn[msk_fp], spec[msk_fp], alpha=0.25, color=GREEN)
ax_fp.plot(wn[msk_fp], spec[msk_fp], lw=1.4, color=GREEN)
ax_fp.set_xlim(195, 1820)
ax_fp.set_xlabel('Raman Shift (cm⁻¹)', fontsize=11)
ax_fp.set_ylabel('Intensity (a.u.)', fontsize=11)
ax_fp.set_title('Fingerprint Region  200–1800 cm⁻¹  (Ag phonon + Si substrate + DNA modes)',
                fontsize=11, fontweight='bold', pad=6)
ax_fp.xaxis.set_minor_locator(MultipleLocator(50))

ymax_fp = spec[msk_fp].max()
fp_peaks = [p for p in peaks if 195 <= p['wn'] <= 1820]

# Stagger annotations two rows
for pi, p in enumerate(fp_peaks):
    w = p['wn']
    lab, _ = assign(w)
    col = GREEN
    row = pi % 2
    y_off = 8 + row * 24
    ax_fp.axvline(w, color=col, lw=1.0, ls='--', alpha=0.65)
    # Peak wn label
    ax_fp.annotate(f'{w:.0f}',
                   xy=(w, p['A'] * 0.65),
                   xytext=(0, y_off), textcoords='offset points',
                   ha='center', va='bottom', fontsize=7.5, color=col,
                   fontweight='bold', rotation=85)

# Add legend boxes for key peaks
legend_text = (
    '~270 cm⁻¹  Ag phonon / Ag–N stretch\n'
    '~526 cm⁻¹  Si TO phonon (substrate)\n'
    '~650 cm⁻¹  Gua+Cyt ring deformation\n'
    '~686 cm⁻¹  Guanine N7 coord.\n'
    '~742 cm⁻¹  Adenine ring breathing\n'
    '~793 cm⁻¹  Cytosine ring stretch\n'
    '~846 cm⁻¹  Deoxyribose C-O\n'
    '~1097 cm⁻¹  νs PO₂⁻ phosphate\n'
    '~1247 cm⁻¹  νas PO₂⁻\n'
    '~1345 cm⁻¹  Adenine C8-H\n'
    '~1456 cm⁻¹  Adenine/Guanine C=N'
)
ax_fp.text(1.01, 0.97, legend_text, transform=ax_fp.transAxes,
           fontsize=7.2, va='top', ha='left',
           bbox=dict(boxstyle='round,pad=0.4', fc='#EBF5FB', ec='#1A5276', lw=0.8))

# ── High-wn 1800-3200 ──────────────────────────────────────────────────────────
msk_hw = (wn >= 1800) & (wn <= 3210)
ax_hw.fill_between(wn[msk_hw], spec[msk_hw], alpha=0.25, color='#E67E22')
ax_hw.plot(wn[msk_hw], spec[msk_hw], lw=1.4, color='#CA6F1E')
ax_hw.set_xlim(1780, 3220)
ax_hw.set_xlabel('Raman Shift (cm⁻¹)', fontsize=11)
ax_hw.set_ylabel('Intensity (a.u.)', fontsize=11)
ax_hw.set_title('High-wavenumber Region  1800–3200 cm⁻¹  (C≡N stretch ~2080 + CH₂/CH₃ ~2930)',
                fontsize=11, fontweight='bold', pad=6)
ax_hw.xaxis.set_minor_locator(MultipleLocator(50))

ymax_hw = spec[msk_hw].max()
hw_peaks = [p for p in peaks if 1800 <= p['wn'] <= 3210]

for pi, p in enumerate(hw_peaks):
    w = p['wn']
    is_key = (abs(w-2080) < 80 or abs(w-2930) < 80)
    col = RED if is_key else '#7D6608'
    lw  = 2.0 if is_key else 1.0
    ls  = '-.' if is_key else '--'
    ax_hw.axvline(w, color=col, lw=lw, ls=ls, alpha=0.8)
    row = pi % 2
    y_off = 8 + row * 24
    ax_hw.annotate(f'{w:.0f}',
                   xy=(w, p['A'] * 0.62),
                   xytext=(0, y_off), textcoords='offset points',
                   ha='center', va='bottom', fontsize=7.5, color=col,
                   fontweight='bold', rotation=85)

# Add boxes for the two user-highlighted peaks
for wn_nom, label, ref in [(2080,
    '~2080 cm⁻¹\nC≡N stretch / Ag-CN⁻\n(Ag complexation, SERS)',
    'Kudelski 2009 Chem.Phys.Lett.'),
   (2930,
    '~2930 cm⁻¹\nνas CH₂/CH₃\n(DNA methylation / deoxyribose)',
    'Jangir et al. 2019 Spectrochim.Acta A')]:
    close = [p for p in peaks if abs(p['wn']-wn_nom)<100]
    if not close: continue
    best = max(close, key=lambda x: x['A'])
    ax_hw.annotate(f'{label}\nRef: {ref}',
                   xy=(best['wn'], best['A']),
                   xytext=(best['wn']+180, best['A']*0.85),
                   fontsize=7.5, color=RED, fontweight='bold',
                   bbox=dict(boxstyle='round,pad=0.3', fc='#FDEDEC', ec=RED, lw=1.0),
                   arrowprops=dict(arrowstyle='->', color=RED, lw=1.5))

plt.tight_layout()
OUTNAME = ('S1DNAHACAT120ngAgSiNW_532nm_600gr_BC200_50XLF_1s_1a_55ul_drop_spectrum')
plt.savefig(f'/home/user/claude/{OUTNAME}.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved:", OUTNAME + ".png")

# ─── 6.  Print full peak table ────────────────────────────────────────────────
print("\n══════════════════════════════════════════════════════════════════")
print(f" {'Wn (cm⁻¹)':>10}  {'Intens.':>9}  {'FWHM(cm⁻¹)':>12}  Assignment")
print("──────────────────────────────────────────────────────────────────")
for p in peaks:
    lab, _ = assign(p['wn'])
    fwhm = 2.355 * p['s']
    print(f" {p['wn']:>10.1f}  {p['A']:>9.0f}  {fwhm:>12.1f}  {lab}")
print("══════════════════════════════════════════════════════════════════")
