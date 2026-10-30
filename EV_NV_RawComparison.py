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
# BAND LABELS COME FROM THE DATA: for each band window below, the most prominent
#   local maximum of the spectrum (located on a lightly smoothed copy,
#   Savitzky-Golay 11 pts / order 3; the plotted line is raw) is labelled only if
#   its prominence is at least MIN_SNR x the noise SD of that spectrum.
#   The table of all positions is printed and saved as EV_NV_RawComparison_peaks.csv.
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
MIN_SNR = 3.0         # label a band only if prominence >= 3 x noise SD
FS_LABEL, FS_NAME, FS_AXIS, FS_TICK = 13, 15, 17, 15

PURPLE, BLUE, DARKBLUE = '#7030A0', '#2E75B6', '#1F3864'
CLASS = {'Proteins': [('Proteins', PURPLE)], 'Lipids': [('Lipids', BLUE)],
         'Nucleic Acids': [('Nucleic Acids', DARKBLUE)],
         'Lipids/NA': [('Lipids/', BLUE), ('NA', 'k')],
         'Proteins/Lipids': [('Proteins/', PURPLE), ('Lipids', BLUE)], None: []}
# band windows (lo, hi), class, assignment
BANDS = [((507, 531), None, 'Si substrate (optical phonon)'),
         ((790, 812), 'Nucleic Acids', 'O-P-O backbone stretching'),
         ((818, 842), 'Proteins', 'Tyr Fermi doublet'),
         ((848, 862), 'Proteins', 'Tyr Fermi doublet'),
         ((912, 930), 'Proteins', 'C-C stretching (Pro) / glucose'),
         ((993, 1010), 'Proteins', 'Phe ring breathing'),
         ((1024, 1042), 'Proteins', 'Phe C-H in-plane bending'),
         ((1155, 1178), 'Proteins', 'Tyr C-H bending'),
         ((1230, 1255), 'Proteins', 'Amide III'),
         ((1340, 1365), 'Proteins', 'Trp / CH deformation'),
         ((1370, 1395), 'Lipids/NA', 'CH3 deformation / nucleic-acid bases'),
         ((1574, 1590), 'Proteins', 'Phe C=C bending'),
         ((1596, 1612), 'Proteins', 'Phe/Tyr ring C=C stretching'),
         ((1616, 1632), 'Proteins', 'Tyr/Trp ring C=C stretching'),
         ((1715, 1740), 'Lipids', 'C=O ester stretching'),
         ((2920, 2945), 'Proteins/Lipids', 'CH3 / CH2 stretching')]

plt.rcParams.update({'font.family': 'sans-serif',
                     'font.sans-serif': ['Arial', 'Liberation Sans', 'DejaVu Sans'],
                     'font.size': FS_TICK})

# ----------------------------------------------------------------------------
# 1. FILES
# ----------------------------------------------------------------------------
files = sorted(DATA_DIR.glob('*.txt'))
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
    for (lo, hi), cls, assign in BANDS:
        k = np.flatnonzero((x >= lo) & (x <= hi))
        if len(k) < 3:
            continue
        pk, pr = find_peaks(ys[k], prominence=0)
        if not len(pk):
            rows.append([kind, f'{lo}-{hi}', '', '', cls, assign, 'no local maximum'])
            continue
        j = int(np.argmax(pr['prominences']))
        pos, snr = x[k[pk[j]]], pr['prominences'][j] / noise
        ok = snr >= MIN_SNR
        rows.append([kind, f'{lo}-{hi}', round(float(pos), 1), round(float(snr), 1), cls, assign,
                     'labelled' if ok else f'not labelled (< {MIN_SNR:g} x noise)'])
        if ok:
            labels[kind].append((pos, cls))
print('\nBAND POSITIONS (raw spectra)')
print(f'{"Spectrum":8s} {"Window":10s} {"Position":>9s} {"Prom/noise":>10s}  {"Class":16s} Result')
for r in rows:
    print(f'{r[0]:8s} {r[1]:10s} {str(r[2]):>9s} {str(r[3]):>10s}  {str(r[4]):16s} {r[6]}')
with open(DATA_DIR / 'EV_NV_RawComparison_peaks.csv', 'w', newline='') as fh:
    w = csv.writer(fh)
    w.writerow(['Spectrum', 'Window (cm-1)', 'Position (cm-1)', 'Prominence / noise SD', 'Class',
                'Assignment', 'Result'])
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
    """Vertical label: black number + coloured class, stacked upwards; returns top."""
    yy = y0
    yr = np.diff(ax.get_ylim())[0]
    top = y0
    for t, col in [(text + ('- ' if cls else ''), 'k')] + CLASS[cls]:
        h = ax.text(x, yy, t, rotation=90, ha='center', va='bottom', fontsize=FS_LABEL, color=col)
        fig.canvas.draw()
        top = h.get_window_extent().transformed(ax.transData.inverted()).y1
        yy = top - 0.004 * yr
    return top


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
                text = f'{p:.1f}' if cls is None else f'{p:.0f}'
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
