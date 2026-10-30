# =============================================================================
# ACACIA EVs and NANOVESICLES (NVs)  |  CaF2 vs Ag/SiNWs  |  SERS enhancement
#
# Message of the figure: on CaF2 (no enhancement) the vesicles give no Raman
# bands; on Ag/SiNWs the bands of proteins, nucleic acids and lipids appear.
#
#   (a) EV Acacia:           on CaF2 (grey)  and on Ag/SiNWs (blue)
#   (b) Nanovesicles Acacia: on CaF2 (grey)  and on Ag/SiNWs (red-orange)
#
# RAW DATA: counts exactly as recorded - no smoothing, no baseline, no
#   normalisation. All four spectra are drawn on the SAME count scale (shared
#   y-axis); each spectrum is only moved up by a constant so they do not overlap.
#   A 50-count scale bar shows the size of bands and noise.
# BAND LABELS (Ag/SiNW spectra): the same bands and the same peak-finding as
#   Figure 1 - most prominent local maximum within +/-12 cm-1 of the nominal band,
#   located on a lightly smoothed copy (Savitzky-Golay 11 / 3; plotted line raw).
# ENHANCEMENT TABLE: for every labelled band, the band prominence divided by the
#   noise SD of the spectrum, on Ag/SiNWs and on CaF2 (same window). Printed and
#   saved as CaF2_vs_AgSiNW_bands.csv.
#
# INPUT: DATA_DIR with the four .txt files (names as recorded):
#   S2-EV_Acacia-dropCaF2 ...      S2-EV_ACACIA-AgSiNW ...
#   s1-EV_nano_Acacia-dropCaF2 ... s4a-EV_nano_ACACIA-20ng-AgSiNW ...
# HOW TO RUN (PyCharm): pip install numpy scipy matplotlib, check DATA_DIR, Run.
# OUTPUT (in DATA_DIR): CaF2_vs_AgSiNW_Enhancement.png / .tiff  (two panels, 600 dpi)
#                       CaF2_vs_AgSiNW_OneGraph.png / .tiff     (all four in one graph)
#                       CaF2_vs_AgSiNW_bands.csv
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
FILES = {('EV', 'CaF2'): 's2-ev_acacia-dropcaf2', ('EV', 'Ag'): 's2-ev_acacia-agsinw',
         ('NV', 'CaF2'): 's1-ev_nano_acacia-dropcaf2', ('NV', 'Ag'): 's4a-ev_nano_acacia-20ng-agsinw'}
COL = {'EV': '#0B5A7E', 'NV': '#B5541C', 'CaF2': '#555555'}
NAME = {'EV': 'EV Acacia', 'NV': 'Nanovesicles Acacia'}
LOW, HIGH = (400, 1800), (2700, 3100)
FS_LABEL, FS_NAME, FS_AXIS, FS_TICK, FS_PANEL = 12, 14, 17, 15, 18
LABEL_TEXT = 'none'   # 'none': dotted line only; 'name': 'Protein'; 'position+name': '1002  Protein'
MARK_LEN = 0.05      # length of the dotted peak marker when LABEL_TEXT = 'none' (fraction of plot height)
GAP = 25              # counts between the top of the CaF2 trace and the bottom of the Ag trace

PEAKS = {
    'EV': [(519, 'Si substrate'), (831, 'Protein/NA'), (1001, 'Protein'), (1031, 'Protein'),
           (1166, 'Protein'), (1355, 'Protein'), (1582, 'Protein'), (1604, 'Protein'),
           (1724, 'Lipid'), (2929, 'Protein/Lipid')],
    'NV': [(519, 'Si substrate'), (612, 'Protein'), (797, 'Nucleic acid'), (826, 'Protein/NA'),
           (921, 'Protein/Carb.'), (1001, 'Protein'), (1031, 'Protein'), (1240, 'Protein'),
           (1386, 'Lipid/Protein'), (1607, 'Protein'), (1624, 'Protein'), (2936, 'Protein/Lipid')]}
CLASS_COL = {'Protein': '#1F6FB4', 'Protein/NA': '#6A3D9A', 'Nucleic acid': '#8E44AD',
             'Protein/Carb.': '#1F6FB4', 'Lipid/Protein': '#D35400', 'Protein/Lipid': '#D35400',
             'Lipid': '#C0392B', 'Si substrate': '#7F7F7F'}

plt.rcParams.update({'font.family': 'sans-serif',
                     'font.sans-serif': ['Arial', 'Liberation Sans', 'DejaVu Sans'],
                     'font.size': FS_TICK})

# ----------------------------------------------------------------------------
# 1. FILES
# ----------------------------------------------------------------------------
def simple(name):
    """lower case, spaces -> '_', '%' removed: 'S2-EV Acacia-dropCaF2_..._10%' -> 's2-ev_acacia-dropcaf2_..._10'"""
    return name.lower().replace(' ', '_').replace('%', '')


spec = {}
for key, prefix in FILES.items():
    hits = [f for f in DATA_DIR.iterdir() if f.is_file() and f.suffix.lower() in ('.txt', '')
            and simple(f.name).startswith(prefix)]
    if len(hits) != 1:
        sys.exit(f'Expected one file starting with "{prefix}" in {DATA_DIR}, found {len(hits)}')
    spec[key] = np.loadtxt(hits[0])
    print(f'{key[0]} on {"CaF2" if key[1] == "CaF2" else "Ag/SiNWs"}: {hits[0].name}')


def view(d):
    x = d[:, 0]
    return ((x >= LOW[0]) & (x <= LOW[1])) | ((x >= HIGH[0]) & (x <= HIGH[1]))


# ----------------------------------------------------------------------------
# 2. BANDS: positions on Ag/SiNWs, prominence / noise on both substrates
# ----------------------------------------------------------------------------
def band(d, nominal):
    x, yv = d[:, 0], d[:, 1]
    ys = savgol_filter(yv, 11, 3)
    noise = np.std(yv - ys)
    k = np.flatnonzero(np.abs(x - nominal) <= 12)
    pk, pr = find_peaks(ys[k], prominence=0)
    if not len(pk):
        return None, 0.0
    j = int(np.argmax(pr['prominences']))
    return int(round(float(x[k[pk[j]]]))), float(pr['prominences'][j] / noise)


rows, labels = [], {'EV': [], 'NV': []}
print(f'\n{"Sample":8s} {"Band":>6s} {"Class":15s} {"Ag/SiNWs: pos":>13s} {"S/N":>5s}   {"CaF2: S/N":>9s}')
for s in ['EV', 'NV']:
    for nominal, cls in PEAKS[s]:
        pos, snr_ag = band(spec[(s, 'Ag')], nominal)
        _, snr_caf = band(spec[(s, 'CaF2')], nominal)
        labels[s].append((pos, cls))
        rows.append([NAME[s], nominal, cls, pos, round(snr_ag, 1), round(snr_caf, 1)])
        print(f'{s:8s} {nominal:6d} {cls:15s} {pos:13d} {snr_ag:5.1f}   {snr_caf:9.1f}')
with open(DATA_DIR / 'CaF2_vs_AgSiNW_bands.csv', 'w', newline='') as fh:
    w = csv.writer(fh)
    w.writerow(['Sample', 'Nominal (cm-1)', 'Class', 'Position on Ag/SiNWs (cm-1)',
                'Prominence/noise on Ag/SiNWs', 'Prominence/noise on CaF2 (same window)'])
    w.writerows(rows)

# ----------------------------------------------------------------------------
# 3. FIGURE: two panels, shared count scale, broken x-axis
# ----------------------------------------------------------------------------
shift = {}
for s in ['EV', 'NV']:
    caf, ag = spec[(s, 'CaF2')], spec[(s, 'Ag')]
    shift[(s, 'CaF2')] = 0.0
    shift[(s, 'Ag')] = caf[view(caf), 1].max() - ag[view(ag), 1].min() + GAP
lo_all = min(spec[(s, 'CaF2')][view(spec[(s, 'CaF2')]), 1].min() for s in ['EV', 'NV'])
hi_all = max(spec[(s, 'Ag')][view(spec[(s, 'Ag')]), 1].max() + shift[(s, 'Ag')] for s in ['EV', 'NV'])
span = hi_all - lo_all
ylim = (lo_all - 0.10 * span, hi_all + (0.40 if LABEL_TEXT != 'none' else 0.15) * span)  # room for labels

fig = plt.figure(figsize=(11, 13))
gs = fig.add_gridspec(2, 2, width_ratios=[LOW[1] - LOW[0], HIGH[1] - HIGH[0]], wspace=0.03,
                      hspace=0.12)
for row, s in enumerate(['EV', 'NV']):
    axL = fig.add_subplot(gs[row, 0])
    axR = fig.add_subplot(gs[row, 1], sharey=axL)
    axL.set_ylim(*ylim)
    for ax, (lo, hi) in [(axL, LOW), (axR, HIGH)]:
        for sub, col in [('CaF2', COL['CaF2']), ('Ag', COL[s])]:
            d = spec[(s, sub)]
            m = (d[:, 0] >= lo) & (d[:, 0] <= hi)
            ax.plot(d[m, 0], d[m, 1] + shift[(s, sub)], color=col, lw=0.8)
        ax.set_xlim(lo, hi)
        ax.tick_params(direction='in', length=6, width=1.0, labelsize=FS_TICK)
    fig.canvas.draw()
    # band labels on the Ag/SiNW spectrum
    ag = spec[(s, 'Ag')]
    for ax, (lo, hi) in [(axL, LOW), (axR, HIGH)]:
        sel = sorted((p, c) for p, c in labels[s] if lo <= p <= hi)
        if not sel:
            continue
        probe = ax.text(0, 0, '0', rotation=90, fontsize=FS_LABEL)
        fig.canvas.draw()
        bb = probe.get_window_extent().transformed(ax.transData.inverted())
        wl = bb.x1 - bb.x0
        probe.remove()
        xl = [p for p, _ in sel]
        for _ in range(200):
            moved = False
            for i in range(1, len(xl)):
                if xl[i] - xl[i - 1] < wl * 1.1:
                    mid = (xl[i] + xl[i - 1]) / 2
                    xl[i - 1], xl[i] = mid - wl * 0.55, mid + wl * 0.55
                    moved = True
            if not moved:
                break
        if LABEL_TEXT == 'none':              # dotted marker above each peak, no text
            for p, _ in sel:
                y_peak = ag[np.argmin(np.abs(ag[:, 0] - p)), 1] + shift[(s, 'Ag')]
                ax.plot([p, p], [y_peak + 0.01 * span, y_peak + (0.01 + 2 * MARK_LEN) * span],  # half-height panels
                        ls=(0, (1, 2)), lw=1.2, color='0.3')
            continue
        for (p, cls), xlab in zip(sel, xl):
            near = (ag[:, 0] >= min(p, xlab) - wl) & (ag[:, 0] <= max(p, xlab) + wl)
            base = ag[near, 1].max() + shift[(s, 'Ag')] + 0.03 * span
            ax.text(xlab, base, (f'{p}  {cls}' if LABEL_TEXT == 'position+name' else cls), rotation=90, ha='center', va='bottom',
                    fontsize=FS_LABEL, color=CLASS_COL[cls])
            y_peak = ag[np.argmin(np.abs(ag[:, 0] - p)), 1] + shift[(s, 'Ag')]
            ax.plot([p, xlab], [y_peak + 0.01 * span, base - 0.005 * span], ls=(0, (1, 2)), lw=0.9,
                    color='0.3')
    # sample names inside the panel, below each trace
    for sub, col, txt in [('Ag', COL[s], f'{NAME[s]} on Ag/SiNWs'),
                          ('CaF2', COL['CaF2'], f'{NAME[s]} on CaF$_2$')]:
        d = spec[(s, sub)]
        m = (d[:, 0] >= 410) & (d[:, 0] <= 900)
        axL.text(415, d[m, 1].min() + shift[(s, sub)] - 0.012 * span, txt, color=col,
                 fontsize=FS_NAME, ha='left', va='top')
    # 50-count scale bar (right panel), so band size and noise can be judged
    xb = 3075
    yb = ylim[1] - 0.12 * (ylim[1] - ylim[0]) - 50     # top-right corner, clear of the traces
    axR.plot([xb, xb], [yb, yb + 50], color='k', lw=2.0)
    axR.text(xb - 10, yb + 25, '50\ncounts', ha='right', va='center', fontsize=FS_NAME - 2)
    # axes cosmetics, break marks
    axL.spines['right'].set_visible(False)
    axR.spines['left'].set_visible(False)
    axL.tick_params(labelleft=False)
    axR.tick_params(left=False, right=True, labelleft=False)
    axL.set_xticks([400, 600, 800, 1000, 1200, 1400, 1600])
    axR.set_xticks([2800, 3000])
    if row == 0:
        axL.tick_params(labelbottom=False)
        axR.tick_params(labelbottom=False)
    dd = 0.012
    for ax, x0 in [(axL, 1), (axR, 0)]:
        dx = dd * (HIGH[1] - HIGH[0]) / (LOW[1] - LOW[0]) if ax is axL else dd
        for y0 in (0, 1):
            ax.plot((x0 - dx, x0 + dx), (y0 - 2 * dd * 0.5, y0 + 2 * dd * 0.5), transform=ax.transAxes,
                    color='k', clip_on=False, lw=1.0)
    axL.set_ylabel('Raman Intensity (a.u.)', fontsize=FS_AXIS)
    axL.text(-0.07, 1.0, f'({"ab"[row]})', transform=axL.transAxes, fontsize=FS_PANEL,
             va='top', ha='right')
fig.text(0.53, 0.02, 'Raman Shift (cm$^{-1}$)', ha='center', fontsize=FS_AXIS)
fig.subplots_adjust(left=0.08, right=0.98, bottom=0.06, top=0.99)
fig.savefig(DATA_DIR / 'CaF2_vs_AgSiNW_Enhancement.png', dpi=600)
fig.savefig(DATA_DIR / 'CaF2_vs_AgSiNW_Enhancement.tiff', dpi=600, pil_kwargs={'compression': 'tiff_lzw'})
plt.close(fig)


# ============================================================================
# 4. ONE-GRAPH VERSION: all four spectra in one plot, same count scale
#    bottom -> top: EV on CaF2, EV on Ag/SiNWs, NV on CaF2, NV on Ag/SiNWs
# ============================================================================
ORDER = [('EV', 'CaF2'), ('EV', 'Ag'), ('NV', 'CaF2'), ('NV', 'Ag')]


def label_positions(ax, s, fig):
    """x positions of the labels of sample s, spread so they do not overlap."""
    out = {}
    for lo, hi in [LOW, HIGH]:
        sel = sorted((p, c) for p, c in labels[s] if lo <= p <= hi)
        if not sel:
            continue
        a = ax[0] if (lo, hi) == LOW else ax[1]
        probe = a.text(0, 0, '0', rotation=90, fontsize=FS_LABEL)
        fig.canvas.draw()
        bb = probe.get_window_extent().transformed(a.transData.inverted())
        wl = bb.x1 - bb.x0
        probe.remove()
        xl = [p for p, _ in sel]
        for _ in range(200):
            moved = False
            for i in range(1, len(xl)):
                if xl[i] - xl[i - 1] < wl * 1.1:
                    mid = (xl[i] + xl[i - 1]) / 2
                    xl[i - 1], xl[i] = mid - wl * 0.55, mid + wl * 0.55
                    moved = True
            if not moved:
                break
        out[(lo, hi)] = (a, sel, xl, wl)
    return out


def draw_one(label_room):
    """Draw the single graph; return the figure and how much more space the EV
    labels need so that they stay below the NV-on-CaF2 trace and its name."""
    off, top = {}, None
    for key in ORDER:                          # stack the traces with fixed gaps
        d = spec[key]
        yv = d[view(d), 1]
        if top is None:
            off[key] = 0.0
        else:
            gap = label_room if key == ('NV', 'CaF2') else GAP
            off[key] = top + gap - yv.min()
        top = yv.max() + off[key]
    y_lo = spec[ORDER[0]][view(spec[ORDER[0]]), 1].min()
    span = top - y_lo
    fig, (axL, axR) = plt.subplots(1, 2, sharey=True, figsize=(11, 12),
                                   gridspec_kw={'width_ratios': [LOW[1] - LOW[0], HIGH[1] - HIGH[0]],
                                                'wspace': 0.03})
    head = 0.28 if LABEL_TEXT != 'none' else 0.14          # room above the top trace
    axL.set_ylim(y_lo - 0.06 * span, top + head * span)
    for ax, (lo, hi) in [(axL, LOW), (axR, HIGH)]:
        for s, sub in ORDER:
            d = spec[(s, sub)]
            m = (d[:, 0] >= lo) & (d[:, 0] <= hi)
            ax.plot(d[m, 0], d[m, 1] + off[(s, sub)], color=COL['CaF2'] if sub == 'CaF2' else COL[s],
                    lw=0.8)
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
    fig.subplots_adjust(left=0.07, right=0.98, bottom=0.07, top=0.98)
    fig.canvas.draw()
    # names below each trace
    nv_caf_name = None
    for s, sub in ORDER:
        d = spec[(s, sub)]
        m = (d[:, 0] >= 410) & (d[:, 0] <= 900)
        txt = f'{NAME[s]} on ' + ('CaF$_2$' if sub == 'CaF2' else 'Ag/SiNWs')
        t = axL.text(415, d[m, 1].min() + off[(s, sub)] - 0.008 * span, txt,
                     color=COL['CaF2'] if sub == 'CaF2' else COL[s], fontsize=FS_NAME, ha='left', va='top')
        if (s, sub) == ('NV', 'CaF2'):
            fig.canvas.draw()
            nv_caf_name = t.get_window_extent().transformed(axL.transData.inverted())
    need = 0.0
    for s in ['EV', 'NV']:
        ag = spec[(s, 'Ag')]
        o = off[(s, 'Ag')]
        for (lo, hi), (a, sel, xl, wl) in label_positions((axL, axR), s, fig).items():
            for (p, cls), xlab in zip(sel, xl):
                if LABEL_TEXT == 'none':          # dotted marker above each peak, no text
                    y_peak = ag[np.argmin(np.abs(ag[:, 0] - p)), 1] + o
                    y_top = y_peak + (0.005 + MARK_LEN) * span
                    a.plot([p, p], [y_peak + 0.005 * span, y_top], ls=(0, (1, 2)), lw=1.2, color='0.3')
                    if s == 'EV':
                        nc = spec[('NV', 'CaF2')]
                        k = (nc[:, 0] >= p - wl) & (nc[:, 0] <= p + wl)
                        need = max(need, y_top - (nc[k, 1].min() + off[('NV', 'CaF2')] - 0.01 * span))
                    continue
                near = (ag[:, 0] >= min(p, xlab) - wl) & (ag[:, 0] <= max(p, xlab) + wl)
                base = ag[near, 1].max() + o + 0.015 * span
                h = a.text(xlab, base, (f'{p}  {cls}' if LABEL_TEXT == 'position+name' else cls), rotation=90, ha='center', va='bottom',
                           fontsize=FS_LABEL, color=CLASS_COL[cls])
                y_peak = ag[np.argmin(np.abs(ag[:, 0] - p)), 1] + o
                a.plot([p, xlab], [y_peak + 0.005 * span, base - 0.003 * span], ls=(0, (1, 2)),
                       lw=0.9, color='0.3')
                if s == 'EV':                 # EV labels must stay below NV on CaF2
                    fig.canvas.draw()
                    y1 = h.get_window_extent().transformed(a.transData.inverted()).y1
                    nc = spec[('NV', 'CaF2')]
                    k = (nc[:, 0] >= xlab - wl) & (nc[:, 0] <= xlab + wl)
                    ceiling = nc[k, 1].min() + off[('NV', 'CaF2')] - 0.01 * span
                    if a is axL and nv_caf_name.x0 - wl <= xlab <= nv_caf_name.x1 + wl:
                        ceiling = min(ceiling, nv_caf_name.y0 - 0.005 * span)
                    need = max(need, y1 - ceiling)
    # 50-count scale bar, top right
    xb = 3075
    yb = top + (head - 0.08) * span - 50
    axR.plot([xb, xb], [yb, yb + 50], color='k', lw=2.0)
    axR.text(xb - 10, yb + 25, '50\ncounts', ha='right', va='center', fontsize=FS_NAME - 2)
    dd = 0.012
    for ax, x0 in [(axL, 1), (axR, 0)]:
        dx = dd * (HIGH[1] - HIGH[0]) / (LOW[1] - LOW[0]) if ax is axL else dd
        for y0 in (0, 1):
            ax.plot((x0 - dx, x0 + dx), (y0 - dd, y0 + dd), transform=ax.transAxes, color='k',
                    clip_on=False, lw=1.0)
    return fig, need, off


room = 80.0 if LABEL_TEXT != 'none' else float(GAP)
for _ in range(10):                            # enlarge the space for the EV labels until they fit
    fig, need, off_one = draw_one(room)
    if need <= 0:
        break
    plt.close(fig)
    room += need * 1.1 + 5
fig.savefig(DATA_DIR / 'CaF2_vs_AgSiNW_OneGraph.png', dpi=600)
fig.savefig(DATA_DIR / 'CaF2_vs_AgSiNW_OneGraph.tiff', dpi=600, pil_kwargs={'compression': 'tiff_lzw'})
print('One-graph version: vertical shifts (display only): ' +
      ', '.join(f'{s} on {"CaF2" if sub == "CaF2" else "Ag/SiNWs"} +{off_one[(s, sub)]:.0f}'
                for s, sub in ORDER))

for s in ['EV', 'NV']:
    print(f'{NAME[s]} on Ag/SiNWs shifted up by {shift[(s, "Ag")]:.0f} counts (display only)')
print(f'Saved in: {DATA_DIR}')
if len(sys.argv) <= 1:
    plt.show()
