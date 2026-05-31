"""
Full-range SERS spectrum from LabSpec6 .l6s binary file.
HaCaT genomic DNA on Ag/SiNW, 532 nm, 1s x 1 acc, 200-3200 cm-1.
Publication-quality readability.
"""

import struct, numpy as np, warnings
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from scipy.signal import savgol_filter, find_peaks
from scipy.optimize import curve_fit
from scipy import sparse
from scipy.sparse.linalg import spsolve
warnings.filterwarnings('ignore')

# ─── 1.  Parse binary ────────────────────────────────────────────────────────
FPATH = ("/root/.claude/uploads/92cc5bbe-9c5a-43cf-8f6c-9f47b8d8c11d/"
         "0a3a5223-S1DNAHACAT120ngAgSiNW_532nm_600gr_BC200_50XLF_1s_1a_55ul_drop.l6s")
with open(FPATH, 'rb') as f:
    raw = f.read()

N  = 1983
wn = np.array([struct.unpack_from('<f', raw, 8809 + i*4)[0] for i in range(N)])
A  = np.array([struct.unpack_from('>f', raw,  544 + i*4)[0] for i in range(N)])
B  = np.array([struct.unpack_from('>f', raw, 8812 + i*4)[0] for i in range(N)])
signal = B - A

# ─── 2.  Baseline & smoothing ────────────────────────────────────────────────
def als_baseline(y, lam=5e4, p=0.01, niter=15):
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

# ─── 3.  Interpolation helper ────────────────────────────────────────────────
def spec_at(w_target):
    """Return interpolated intensity at a given wavenumber."""
    return float(np.interp(w_target, wn, spec))

# ─── 4.  Manually-defined key peaks with assignments ────────────────────────
# Format: (nominal_wn, search_tol, wavenumber_label, bond_assignment, colour)
# Find the actual peak within ±tol of nominal, use highest intensity
PEAKS = [
    # LW region
    (273,  50, '273 cm⁻¹', 'Ag–N stretch\n(metal–DNA bond)',        '#A04000'),
    (515,  20, '515 cm⁻¹', 'Si TO phonon\n(SiNW substrate)',         '#7B5B00'),
    # DNA fingerprint
    (648,  30, '648 cm⁻¹', 'Gua+Cyt\nring deform.',                  '#1A6B35'),
    (684,  20, '684 cm⁻¹', 'Guanine N7\ncoordination',               '#1A6B35'),
    (730,  35, '730 cm⁻¹', 'Adenine\nring breathing',                '#1A6B35'),
    (790,  25, '790 cm⁻¹', 'Cytosine\nring stretch',                 '#1A6B35'),
    (845,  30, '845 cm⁻¹', 'Deoxyribose\nC–O stretch',               '#1A6B35'),
    (970,  30, '970 cm⁻¹', 'Phosphate /\nsugar backbone',            '#1A6B35'),
    (1097, 30, '1097 cm⁻¹','νs PO₂⁻\nphosphate',                    '#1A6B35'),
    (1175, 30, '1175 cm⁻¹','Cyt/Ade\nC–N bend',                     '#1A6B35'),
    (1240, 30, '1240 cm⁻¹','νas PO₂⁻\nphosphate',                   '#1A6B35'),
    (1336, 30, '1336 cm⁻¹','Adenine\nC8–H bend',                    '#1A6B35'),
    (1380, 30, '1380 cm⁻¹','Thy/Ade\nC–H deform.',                  '#1A6B35'),
    (1480, 30, '1480 cm⁻¹','Ade/Gua\nC=N stretch',                  '#1A6B35'),
    (1575, 30, '1575 cm⁻¹','Adenine\nring stretch',                  '#1A6B35'),
    # HW region
    (2080, 80, '2080 cm⁻¹','C≡N stretch\n(Ag–CN complex)',           '#8B0000'),
    (2930, 80, '2930 cm⁻¹','νas CH₂/CH₃\n(DNA methylation\nmarker)','#8B0000'),
]

def find_peak_in_range(wn_nom, tol):
    """Return (actual_wn, intensity) of highest point within ±tol."""
    msk = (wn >= wn_nom - tol) & (wn <= wn_nom + tol)
    if not msk.any():
        return wn_nom, spec_at(wn_nom)
    idx = np.argmax(spec[msk])
    return float(wn[msk][idx]), float(spec[msk][idx])

resolved = []
for (nom, tol, wn_lbl, bond_lbl, col) in PEAKS:
    actual_wn, actual_int = find_peak_in_range(nom, tol)
    resolved.append(dict(nom=nom, wn=actual_wn, A=actual_int,
                         wn_lbl=wn_lbl, bond=bond_lbl, col=col))

# ─── 5.  Figure ──────────────────────────────────────────────────────────────
plt.rcParams.update({'font.family': 'DejaVu Sans'})

fig = plt.figure(figsize=(17, 22))
fig.patch.set_facecolor('white')
gs = fig.add_gridspec(4, 1, hspace=0.60,
                      height_ratios=[1.6, 1.6, 2.2, 1.6],
                      left=0.07, right=0.97, top=0.96, bottom=0.04)

ax0 = fig.add_subplot(gs[0])
ax1 = fig.add_subplot(gs[1])
ax2 = fig.add_subplot(gs[2])
ax3 = fig.add_subplot(gs[3])

def style_ax(ax):
    ax.set_facecolor('#FAFAFA')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(axis='x', which='major', labelsize=11, length=5)
    ax.tick_params(axis='x', which='minor', length=3)
    ax.set_xlabel('Raman Shift (cm⁻¹)', fontsize=12, labelpad=5)
    ax.set_ylabel('Intensity (a.u.)', fontsize=12, labelpad=5)
    ax.set_yticks([])

# ─── Panel 0: Overview ───────────────────────────────────────────────────────
C_SPEC = '#1A3A5C'
msk = (wn >= 200) & (wn <= 3200)
ax0.fill_between(wn[msk], spec[msk], alpha=0.18, color=C_SPEC)
ax0.plot(wn[msk], spec[msk], lw=1.4, color=C_SPEC)
ax0.axvspan(200, 620, alpha=0.13, color='#E67E22', zorder=0)
ax0.axvspan(2300, 3200, alpha=0.13, color='#2980B9', zorder=0)

ymax0 = spec[msk].max()

# 4 landmark annotations on overview only
for pk in [p for p in resolved if p['nom'] in (273, 515, 2080, 2930)]:
    w, y = pk['wn'], pk['A']
    col = pk['col']
    ax0.annotate('', xy=(w, y + ymax0*0.04),
                 xytext=(w, y + ymax0*0.22),
                 arrowprops=dict(arrowstyle='->', color=col, lw=1.8))
    ax0.text(w, y + ymax0*0.24,
             f"{pk['wn_lbl']}\n{pk['bond'].split(chr(10))[0]}",
             ha='center', va='bottom', fontsize=10, color=col, fontweight='bold',
             bbox=dict(boxstyle='round,pad=0.3', fc='white', ec=col, lw=1.2, alpha=0.95))

# Region label boxes
ax0.text(340, ymax0*0.78, 'LW Region\n125–550 cm⁻¹', ha='center', va='center',
         fontsize=10, color='#784212', style='italic',
         bbox=dict(boxstyle='round,pad=0.3', fc='#FFF3CD', ec='#E67E22', lw=1.0))
ax0.text(2750, ymax0*0.78, 'HW Region\n2300–3400 cm⁻¹', ha='center', va='center',
         fontsize=10, color='#1A5276', style='italic',
         bbox=dict(boxstyle='round,pad=0.3', fc='#D6EAF8', ec='#2980B9', lw=1.0))

ax0.set_xlim(190, 3210)
ax0.set_ylim(-ymax0*0.05, ymax0*1.50)
ax0.set_title('Overview — Full SERS Spectrum  200–3200 cm⁻¹\n'
              'HaCaT Genomic DNA on Ag/SiNW  |  532 nm  |  1 s × 1 acc  |  50× LMPlanFi',
              fontsize=12, fontweight='bold', pad=8)
ax0.xaxis.set_major_locator(MultipleLocator(200))
ax0.xaxis.set_minor_locator(MultipleLocator(50))
style_ax(ax0)

# ─── Panel 1: LW region 200–620 ──────────────────────────────────────────────
msk1 = (wn >= 200) & (wn <= 620)
ax1.fill_between(wn[msk1], spec[msk1], alpha=0.28, color='#E67E22')
ax1.plot(wn[msk1], spec[msk1], lw=2.0, color='#A04000')
ax1.axvspan(200, 620, alpha=0.06, color='#E67E22')

ymax1 = spec[msk1].max()
lw_pks = [p for p in resolved if 200 <= p['nom'] <= 620]

# Two clear annotations for LW: alternating levels
y_levels1 = [0.30, 0.62]
for i, pk in enumerate(lw_pks):
    w, y = pk['wn'], pk['A']
    col = pk['col']
    ax1.axvline(w, color=col, lw=1.5, ls='--', alpha=0.7)
    yt = ymax1 * y_levels1[i % 2]
    # Connector line
    ax1.annotate('', xy=(w, min(y + ymax1*0.04, yt - ymax1*0.06)),
                 xytext=(w, yt),
                 arrowprops=dict(arrowstyle='-', color=col, lw=1.0, alpha=0.6))
    ax1.text(w, yt, f"{pk['wn_lbl']}\n{pk['bond']}",
             ha='center', va='bottom', fontsize=10.5, color=col, fontweight='bold',
             bbox=dict(boxstyle='round,pad=0.3', fc='white', ec=col, lw=1.2, alpha=0.95))

ax1.set_xlim(195, 625)
ax1.set_ylim(-ymax1*0.05, ymax1*1.62)
ax1.set_title('Low-wavenumber (LW) Region  200–620 cm⁻¹\n'
              'Ag–N stretch (metal–DNA bond)  +  Si TO phonon (substrate)',
              fontsize=11, fontweight='bold', pad=6)
ax1.xaxis.set_major_locator(MultipleLocator(50))
ax1.xaxis.set_minor_locator(MultipleLocator(10))
style_ax(ax1)

# ─── Panel 2: DNA Fingerprint 600–1800 ───────────────────────────────────────
C_FP = '#1A6B35'
msk2 = (wn >= 600) & (wn <= 1800)
ax2.fill_between(wn[msk2], spec[msk2], alpha=0.22, color=C_FP)
ax2.plot(wn[msk2], spec[msk2], lw=1.8, color=C_FP)

ymax2 = spec[msk2].max()
fp_pks = [p for p in resolved if 600 <= p['nom'] <= 1800]

# 4 stagger levels to spread 13 annotations clearly
y_lev2 = [0.12, 0.35, 0.58, 0.81]
for i, pk in enumerate(fp_pks):
    w, y_sp = pk['wn'], pk['A']
    col = C_FP
    ax2.axvline(w, color=col, lw=1.0, ls='--', alpha=0.55)
    yt = ymax2 * y_lev2[i % 4]
    # Thin connector
    if y_sp < yt - ymax2*0.04:
        ax2.annotate('', xy=(w, y_sp + ymax2*0.02), xytext=(w, yt - ymax2*0.02),
                     arrowprops=dict(arrowstyle='-', color=col, lw=0.7, alpha=0.4))
    ax2.text(w, yt, f"{pk['wn_lbl']}\n{pk['bond']}",
             ha='center', va='bottom', fontsize=9.2, color=col, fontweight='bold',
             bbox=dict(boxstyle='round,pad=0.22', fc='white', ec=col, lw=0.9, alpha=0.94))

ax2.set_xlim(590, 1820)
ax2.set_ylim(-ymax2*0.04, ymax2*1.75)
ax2.set_title('DNA Fingerprint Region  600–1800 cm⁻¹\n'
              'Nucleobase vibrations (Ade, Gua, Cyt, Thy)  |  Phosphate backbone (PO₂⁻)',
              fontsize=11, fontweight='bold', pad=6)
ax2.xaxis.set_major_locator(MultipleLocator(100))
ax2.xaxis.set_minor_locator(MultipleLocator(25))
style_ax(ax2)

# ─── Panel 3: HW region 1800–3200 ────────────────────────────────────────────
C_HW = '#8B0000'
msk3 = (wn >= 1800) & (wn <= 3200)
ax3.fill_between(wn[msk3], spec[msk3], alpha=0.22, color='#2980B9')
ax3.plot(wn[msk3], spec[msk3], lw=2.0, color=C_HW)
ax3.axvspan(2300, 3200, alpha=0.07, color='#2980B9')

ymax3 = spec[msk3].max()
hw_pks = [p for p in resolved if 1800 <= p['nom'] <= 3200]

# Two clearly separated annotations for HW
y_lev3 = [0.28, 0.64]
for i, pk in enumerate(hw_pks):
    w, y_sp = pk['wn'], pk['A']
    col = C_HW
    ax3.axvline(w, color=col, lw=2.0, ls='-.', alpha=0.85)
    yt = ymax3 * y_lev3[i % 2]
    ax3.annotate('', xy=(w, y_sp + ymax3*0.03),
                 xytext=(w, yt - ymax3*0.04),
                 arrowprops=dict(arrowstyle='->', color=col, lw=1.6))
    ax3.text(w, yt, f"{pk['wn_lbl']}\n{pk['bond']}",
             ha='center', va='bottom', fontsize=10.5, color=col, fontweight='bold',
             bbox=dict(boxstyle='round,pad=0.32', fc='white', ec=col, lw=1.5, alpha=0.97))

ax3.set_xlim(1780, 3220)
ax3.set_ylim(-ymax3*0.05, ymax3*1.60)
ax3.set_title('High-wavenumber (HW) Region  1800–3200 cm⁻¹\n'
              'C≡N stretch ~2080 cm⁻¹  |  CH₂/CH₃ DNA methylation marker ~2930 cm⁻¹',
              fontsize=11, fontweight='bold', pad=6)
ax3.xaxis.set_major_locator(MultipleLocator(100))
ax3.xaxis.set_minor_locator(MultipleLocator(25))
style_ax(ax3)

# ─── Footer ──────────────────────────────────────────────────────────────────
fig.text(0.50, 0.014,
         'Sample: HaCaT genomic DNA (120 ng/µL, 5 µL drop) on Ag/SiNW  |  '
         'λ = 532 nm  |  600 gr/mm grating  |  50× LMPlanFi  |  1 s × 1 acc\n'
         'Processing: B−A background subtraction, SG smoothing (w=21, poly=3), '
         'ALS baseline correction  |  '
         'Peak assignments: Lancia et al. Sci. Rep. 13, 11370 (2023)',
         ha='center', va='bottom', fontsize=9, color='#444444', style='italic')

OUTNAME = 'S1DNAHACAT120ngAgSiNW_532nm_600gr_BC200_50XLF_1s_1a_55ul_drop_spectrum'
plt.savefig(f'/home/user/claude/{OUTNAME}.png', dpi=150, bbox_inches='tight',
            facecolor='white')
plt.close()
print("Saved:", OUTNAME + ".png")
