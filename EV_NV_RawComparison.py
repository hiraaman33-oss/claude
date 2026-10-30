# =============================================================================
# ACACIA EVs vs NANOVESICLES (NVs) on Ag/SiNWs  |  raw single-spectrum comparison
#
# One plot, both raw spectra stacked (EV below, NV above), broken x-axis
# 400-1800 | 2700-3100 cm-1, same colours as the other paper figures
# (EV blue #0B5A7E, NV red-orange #B5541C), large fonts for print.
#
# RAW DATA: counts exactly as recorded - no smoothing, no baseline, no
# normalisation. The NV spectrum is shifted up by a constant (display only), so
# the y-axis has no numbers.
#
# INPUT: the folder below (C:\Users\Hira Aman\Downloads\comparison ev nv).
#   Every .txt file (2 columns: wavenumber, counts) is read. A file is
#   NV if its name contains 'nano', otherwise EV; CaF2 files are ignored.
#   If a group has more than one Ag/SiNW file, the one named in EV_PICK / NV_PICK
#   is used (default: S2 for EV, s4a for NV, the best-matching BC200 pair);
#   all files found are listed in the Run window.
#
# BAND LABELS: the same bands, classes, label style and peak-finding as Figure 1
#   (CaF2 vs Ag/SiNWs): most prominent local maximum within +/-12 cm-1 of each
#   nominal band, located on a lightly smoothed copy (Savitzky-Golay 11 pts /
#   order 3; the plotted line is raw), so the numbers are identical to Figure 1.
#   The table (with prominence / noise) is printed and saved as
#   EV_NV_RawComparison_peaks.csv.
#
# HOW TO RUN (PyCharm): pip install numpy scipy matplotlib, check DATA_DIR, Run.
# OUTPUT (in DATA_DIR): EV_NV_RawComparison.png / .tiff (600 dpi) + peaks CSV
# =============================================================================

import sys
import csv
from pathlib import Path

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter, find_peaks

DATA_DIR = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(r'C:\Users\Hira Aman\Downloads\comparison ev nv')
if len(sys.argv) > 1:
    matplotlib.use('Agg')
EV_PICK = 's2'        # start of the EV file name to use if there are several EV files
NV_PICK = 's4a'       # start of the NV file name to use if there are several NV files

C_EV, C_NV = '#0B5A7E', '#B5541C'
NAME = {'EV': 'EV Acacia on Ag/SiNWs', 'NV': 'Nanovesicles Acacia on Ag/SiNWs'}
LOW, HIGH = (400, 1800), (2700, 3100)
FS_LABEL, FS_NAME, FS_AXIS, FS_TICK = 13, 15, 17, 15

# bands of Figure 1, per spectrum: (nominal cm-1, class, assignment)
PEAKS = {
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
# label colour by class (as in Figure 1)
CLASS_COL = {'Protein': '#1F6FB4', 'Protein/NA': '#6A3D9A', 'Nucleic acid': '#8E44AD',
             'Protein/Carb.': '#1F6FB4', 'Lipid/Protein': '#D35400', 'Protein/Lipid': '#D35400',
             'Lipid': '#C0392B', 'Si substrate': '#7F7F7F'}

plt.rcParams.update({'font.family': 'sans-serif',
                     'font.sans-serif': ['Arial', 'Liberation Sans', 'DejaVu Sans'],
                     'font.size': FS_TICK})

# ----------------------------------------------------------------------------
# 1. FILES
# ----------------------------------------------------------------------------
files = sorted(f for f in DATA_DIR.iterdir() if f.is_file() and f.suffix.lower() in ('.txt', ''))
print(f'Files in {DATA_DIR}:')
groups = {'EV': [], 'NV': []}
for f in files:
    n = f.name.lower()
    kind = 'NV' if 'nano' in n else 'EV'
    caf2 = 'caf2' in n
    print(f'  {f.name}  ->  {kind}{" (CaF2, ignored)" if caf2 else ""}')
    if not caf2:
        groups[kind].append(f)
chosen = {}
for kind, pick in [('EV', EV_PICK), ('NV', NV_PICK)]:
    cand = groups[kind]
    if not cand:
        sys.exit(f'No {kind} Ag/SiNW file found in {DATA_DIR}')
    if len(cand) > 1:
        sel = [f for f in cand if f.name.lower().startswith(pick.lower() + '-')]
        if len(sel) != 1:
            sys.exit(f'Several {kind} files found; set {kind}_PICK to the start of the file name to use')
        cand = sel
    chosen[kind] = cand[0]
    print(f'{kind} spectrum used: {cand[0].name}')

spec = {k: np.loadtxt(f) for k, f in chosen.items()}

# ----------------------------------------------------------------------------
# 2. BAND POSITIONS FROM THE DATA
# ----------------------------------------------------------------------------
rows, labels = [], {'EV': [], 'NV': []}
for kind, d in spec.items():
    x, yv = d[:, 0], d[:, 1]
    ys = savgol_filter(yv, 11, 3)              # only to locate the maxima
    noise = np.std(yv - ys)
    for nominal, cls, assign in PEAKS[kind]:
        k = np.flatnonzero(np.abs(x - nominal) <= 12)
        pk, pr = find_peaks(ys[k], prominence=0)
        j = int(np.argmax(pr['prominences']))
        pos = int(round(float(x[k[pk[j]]])))
        snr = pr['prominences'][j] / noise
        rows.append([kind, nominal, pos, round(float(snr), 1), cls, assign])
        labels[kind].append((pos, cls))
print('\nBAND POSITIONS (raw spectra, same method as Figure 1)')
print(f'{"Spectrum":8s} {"Nominal":>7s} {"Measured":>8s} {"Prom/noise":>10s}  Class')
for r in rows:
    print(f'{r[0]:8s} {r[1]:7d} {r[2]:8d} {r[3]:10.1f}  {r[4]}')
with open(DATA_DIR / 'EV_NV_RawComparison_peaks.csv', 'w', newline='') as fh:
    w = csv.writer(fh)
    w.writerow(['Spectrum', 'Nominal (cm-1)', 'Measured (cm-1)', 'Prominence / noise SD', 'Class',
                'Assignment'])
    w.writerows(rows)


# ----------------------------------------------------------------------------
# 3. FIGURE: one plot, broken x-axis
# ----------------------------------------------------------------------------
def in_view(d):
    x = d[:, 0]
    return ((x >= LOW[0]) & (x <= LOW[1])) | ((x >= HIGH[0]) & (x <= HIGH[1]))


ev, nv = spec['EV'], spec['NV']
ev_v, nv_v = ev[in_view(ev), 1], nv[in_view(nv), 1]
OFFSET = int(np.ceil((ev_v.max() - nv_v.min() + 0.08 * np.ptp(ev_v)) / 10) * 10)   # first guess


def put_label(fig, ax, x, y0, cls, text):
    """Vertical label 'position  class' in the class colour (as in Figure 1); returns top."""
    h = ax.text(x, y0, f'{text}  {cls}', rotation=90, ha='center', va='bottom', fontsize=FS_LABEL,
                color=CLASS_COL[cls])
    fig.canvas.draw()
    return h.get_window_extent().transformed(ax.transData.inverted()).y1


def draw(offset):
    """Draw the figure; return it and how much the NV spectrum must move up so
    that no EV label reaches the NV spectrum or the NV sample name."""
    shift = {'EV': 0, 'NV': offset}
    y_lo, y_hi = ev_v.min(), nv_v.max() + offset
    span = y_hi - y_lo
    fig, (axL, axR) = plt.subplots(1, 2, sharey=True, figsize=(11, 8),
                                   gridspec_kw={'width_ratios': [LOW[1] - LOW[0], HIGH[1] - HIGH[0]],
                                                'wspace': 0.03})
    axL.set_ylim(y_lo - 0.10 * span, y_hi + 0.45 * span)
    for ax, (lo, hi) in [(axL, LOW), (axR, HIGH)]:
        for kind, col in [('EV', C_EV), ('NV', C_NV)]:
            d = spec[kind]
            m = (d[:, 0] >= lo) & (d[:, 0] <= hi)
            ax.plot(d[m, 0], d[m, 1] + shift[kind], color=col, lw=0.8)
        ax.set_xlim(lo, hi)
        ax.tick_params(direction='in', length=6, width=1.0, labelsize=FS_TICK)
    axL.spines['right'].set_visible(False)
    axR.spines['left'].set_visible(False)
    axL.tick_params(labelleft=False)
    axR.tick_params(left=False, right=True, labelleft=False)
    axL.set_xticks([400, 600, 800, 1000, 1200, 1400, 1600])
    axR.set_xticks([2800, 3000])
    axL.set_ylabel('Raman Intensity (a.u.)', fontsize=FS_AXIS)
    fig.text(0.53, 0.015, 'Raman Shift (cm$^{-1}$)', ha='center', fontsize=FS_AXIS)
    fig.subplots_adjust(left=0.06, right=0.98, bottom=0.11, top=0.97)
    fig.canvas.draw()
    # sample names inside the plot, at the left, just below each spectrum
    name_box = None
    for kind, col in [('EV', C_EV), ('NV', C_NV)]:
        d = spec[kind]
        m = (d[:, 0] >= 410) & (d[:, 0] <= 760)
        t = axL.text(415, d[m, 1].min() + shift[kind] - 0.015 * span, NAME[kind], color=col,
                     fontsize=FS_NAME, ha='left', va='top')
        if kind == 'NV':
            fig.canvas.draw()
            name_box = t.get_window_extent().transformed(axL.transData.inverted())
    need = 0.0
    for kind in ['EV', 'NV']:
        d = spec[kind]
        for ax, (lo, hi) in [(axL, LOW), (axR, HIGH)]:
            sel = sorted((p, c) for p, c in labels[kind] if lo <= p <= hi)
            if not sel:
                continue
            probe = ax.text(0, 0, '0', rotation=90, fontsize=FS_LABEL)
            fig.canvas.draw()
            bb = probe.get_window_extent().transformed(ax.transData.inverted())
            wl = bb.x1 - bb.x0
            probe.remove()
            xl = [p for p, _ in sel]
            for _ in range(200):                     # spread overlapping labels
                moved = False
                for i in range(1, len(xl)):
                    if xl[i] - xl[i - 1] < wl * 1.1:
                        mid = (xl[i] + xl[i - 1]) / 2
                        xl[i - 1], xl[i] = mid - wl * 0.55, mid + wl * 0.55
                        moved = True
                if not moved:
                    break
            for (p, cls), xlab in zip(sel, xl):
                near = (d[:, 0] >= min(p, xlab) - wl) & (d[:, 0] <= max(p, xlab) + wl)
                base = d[near, 1].max() + shift[kind] + 0.03 * span
                text = f'{p}'
                top = put_label(fig, ax, xlab, base, cls, text)
                y_peak = d[np.argmin(np.abs(d[:, 0] - p)), 1] + shift[kind]
                ax.plot([p, xlab], [y_peak + 0.01 * span, base - 0.005 * span], ls=(0, (1, 2)),
                        lw=0.9, color='0.3')
                if kind == 'EV':                      # EV label must stay below NV trace / name
                    k = (nv[:, 0] >= xlab - wl) & (nv[:, 0] <= xlab + wl)
                    ceiling = nv[k, 1].min() + offset - 0.02 * span
                    if ax is axL and name_box.x0 - wl <= xlab <= name_box.x1 + wl:
                        ceiling = min(ceiling, name_box.y0 - 0.01 * span)
                    need = max(need, top - ceiling)
    dd = 0.012
    for ax, x0 in [(axL, 1), (axR, 0)]:
        dx = dd * (HIGH[1] - HIGH[0]) / (LOW[1] - LOW[0]) if ax is axL else dd
        for y0 in (0, 1):
            ax.plot((x0 - dx, x0 + dx), (y0 - 2 * dd, y0 + 2 * dd), transform=ax.transAxes,
                    color='k', clip_on=False, lw=1.0)
    return fig, need


for _ in range(8):                                    # raise NV until nothing overlaps
    fig, need = draw(OFFSET)
    if need <= 0:
        break
    plt.close(fig)
    OFFSET = int(np.ceil((OFFSET + need * 1.05) / 10) * 10)
fig.savefig(DATA_DIR / 'EV_NV_RawComparison.png', dpi=600)
fig.savefig(DATA_DIR / 'EV_NV_RawComparison.tiff', dpi=600, pil_kwargs={'compression': 'tiff_lzw'})
print(f'\nNV spectrum shifted up by {OFFSET} counts (display only)')
print(f'Saved in: {DATA_DIR}')
if len(sys.argv) <= 1:
    plt.show()
