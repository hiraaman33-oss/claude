# =============================================================================
# ACACIA EVs vs NANOVESICLES (NVs) on Ag/SiNWs  |  FIGURE 2a and 2b
# mean +/- SD SERS spectra with band labels
#
#   Fig2a_MeanSD_Fingerprint   400-1790 cm-1
#   Fig2b_MeanSD_CH            2760-3095 cm-1
#
# Same data, pre-processing and colours as Acacia_Paper_All.py; larger fonts, lines
# and offsets for print, and the band labels listed in LABELS below.
#
# DATA: CODE 005\EV Acacia\EV_Acacia_Raman_Data.xlsx and
#       CODE 005\Nano vesicles Acacia\NV_Acacia_Raman_Data.xlsx
#       (6 sites x 20 spectra per group = 120 + 120 spectra)
# PRE-PROCESSING (every spectrum): cosmic-ray removal (modified z-score > 7) ->
#   2 cm-1 grid -> Savitzky-Golay (11 points, order 3) -> ALS baseline
#   (lambda 1e5, p 0.01, 10 iterations) -> SNV over 205-1795 / 2705-3095 cm-1.
# PLOTTED: mean of each group +/- 1 sample SD (n - 1); NV shifted up for display.
#
# LABEL POSITIONS COME FROM THE DATA (mean spectrum of the group):
#   peak     : most prominent local maximum of the mean spectrum inside the window
#   shoulder : most prominent local minimum of the 2nd derivative (Savitzky-Golay,
#              window 11, polyorder 3) of the mean spectrum inside the window
#   both refined by a parabola through 3 points, then rounded to an integer.
#   A label marked 'if present' is drawn only if such a local extremum exists.
# All positions are printed as a table and saved to Fig2_label_positions.csv.
# Grey shadows: EV/NV band shifts that are consistent at all 6 + 6 sites.
#
# HOW TO RUN (PyCharm):
#   pip install numpy pandas scipy matplotlib openpyxl
#   check BASE_DIR, press Run.  Output: CODE 005\PAPER (PNG + TIFF, 600 dpi)
# =============================================================================

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from scipy import sparse
from scipy.sparse.linalg import spsolve
from scipy.signal import savgol_filter, find_peaks

BASE_DIR = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(r'C:\Users\Hira Aman\Downloads\CODE 005')
EV_FILE = BASE_DIR / 'EV Acacia' / 'EV_Acacia_Raman_Data.xlsx'
NV_FILE = BASE_DIR / 'Nano vesicles Acacia' / 'NV_Acacia_Raman_Data.xlsx'
OUTPUT_DIR = BASE_DIR / 'PAPER'
OUTPUT_DIR.mkdir(exist_ok=True)
if len(sys.argv) > 1:
    matplotlib.use('Agg')

SITES = ['m1', 'm2', 'm3', 'm4', 'm5', 'm6']
GROUPS = {0: 'EVs', 1: 'NVs'}
COLS = {0: '#0B5A7E', 1: '#B5541C'}
NAME_LONG = {0: 'EV Acacia on Ag/SiNWs', 1: 'Nanovesicles Acacia on Ag/SiNWs'}
WINDOWS = {'fingerprint': {'raw': (200, 1800), 'grid': (205, 1795)},
           'ch': {'raw': (2700, 3100), 'grid': (2705, 3095)}}
OFFSET = {'fingerprint': 12.0, 'ch': 6.5}      # SNV units added to the NV spectrum
TOP_ROOM = {'fingerprint': 7.0, 'ch': 6.0}     # space above the NV spectrum for labels
# font sizes (pt), chosen to stay readable when the figure is reduced to one
# journal column
LABEL_FS, AXIS_FS, TICK_FS = 13, 17, 15

# class colours of the label text
PURPLE, BLUE, DARKBLUE = '#7030A0', '#2E75B6', '#1F3864'
CLASS = {'Proteins': [('Proteins', PURPLE)],
         'Proteins (COO$^{-}$)': [('Proteins (COO$^{-}$)', PURPLE)],
         'Lipids': [('Lipids', BLUE)],
         'Nucleic Acids': [('Nucleic Acids', DARKBLUE)],
         'Lipids/NA': [('Lipids/', BLUE), ('NA', 'k')],
         'Lipids/Proteins': [('Lipids/', BLUE), ('Proteins', PURPLE)],
         'Proteins/Lipids': [('Proteins/', PURPLE), ('Lipids', BLUE)],
         None: []}

# ----------------------------------------------------------------------------
# LABELS: (region, group 0=EV 1=NV, kind 'peak'/'shoulder', window lo, hi,
#          text format, class, required)
# text format: {p} = position; '~' and 'sh' are written as given
# required=False: label only if a local extremum exists in the window
# ----------------------------------------------------------------------------
LABELS = [
    # --- Fig 2a, fingerprint: labels of the current version
    ('fingerprint', 0, 'peak', 507, 531, '{p:.1f}', None, True),          # Si 518.5 (one decimal, as before)
    ('fingerprint', 0, 'peak', 797, 821, '{p}', 'Nucleic Acids', True),
    ('fingerprint', 0, 'peak', 843, 867, '{p}', 'Proteins', True),
    ('fingerprint', 0, 'peak', 919, 943, '{p}', 'Proteins', True),
    ('fingerprint', 0, 'peak', 989, 1013, '{p}', 'Proteins', True),
    ('fingerprint', 0, 'peak', 1149, 1173, '{p}', 'Proteins', True),
    ('fingerprint', 0, 'peak', 1237, 1261, '{p}', 'Proteins', True),
    ('fingerprint', 0, 'peak', 1361, 1385, '{p}', 'Lipids/NA', True),
    ('fingerprint', 0, 'peak', 1591, 1615, '{p}', 'Proteins', True),
    ('fingerprint', 1, 'peak', 507, 531, '{p:.1f}', None, True),
    ('fingerprint', 1, 'peak', 787, 811, '{p}', 'Nucleic Acids', True),
    ('fingerprint', 1, 'peak', 817, 841, '{p}', 'Proteins', True),
    ('fingerprint', 1, 'peak', 915, 939, '{p}', 'Proteins', True),
    ('fingerprint', 1, 'peak', 989, 1013, '{p}', 'Proteins', True),
    ('fingerprint', 1, 'peak', 1021, 1045, '{p}', 'Proteins', True),
    ('fingerprint', 1, 'peak', 1157, 1181, '{p}', 'Proteins', True),
    ('fingerprint', 1, 'peak', 1233, 1257, '{p}', 'Proteins', True),
    ('fingerprint', 1, 'peak', 1373, 1397, '{p}', 'Lipids/NA', True),
    ('fingerprint', 1, 'peak', 1595, 1619, '{p}', 'Proteins', True),
    # --- Fig 2a, new labels
    ('fingerprint', 0, 'shoulder', 1560, 1585, '~{p} sh', 'Proteins (COO$^{-}$)', True),
    ('fingerprint', 1, 'shoulder', 1620, 1640, '~{p} sh', None, True),
    ('fingerprint', 0, 'shoulder', 1490, 1510, '~{p}', None, True),
    ('fingerprint', 0, 'peak', 822, 835, '{p}', 'Proteins', True),       # weak Tyr band
    ('fingerprint', 1, 'peak', 850, 860, '{p}', 'Proteins', False),      # weak Tyr band, if present
    ('fingerprint', 0, 'peak', 1715, 1740, '{p}', 'Lipids', 'both1728'),  # only if present in EV AND NV
    ('fingerprint', 1, 'peak', 1715, 1740, '{p}', 'Lipids', 'both1728'),
    # --- Fig 2b, CH region
    ('ch', 0, 'shoulder', 2868, 2892, '{p}', 'Lipids/Proteins', True),   # 2874 (current label, new class)
    ('ch', 1, 'shoulder', 2868, 2892, '{p}', 'Lipids/Proteins', True),   # 2883
    ('ch', 0, 'peak', 2923, 2947, '{p}', 'Proteins', True),              # 2933
    ('ch', 1, 'peak', 2923, 2947, '{p}', 'Proteins', True),              # 2937
    ('ch', 1, 'shoulder', 2950, 2975, '~{p} sh', 'Proteins/Lipids', True),
    ('ch', 0, 'shoulder', 2845, 2860, '~{p}', 'Lipids', True),
    # no labels above 3000 cm-1
]
# grey shadows: bands compared between EVs and NVs (EV window, NV window, kind);
# drawn only if the shift is consistent at all sites (site ranges do not overlap)
SHIFTS = {'fingerprint': [((507, 531), (507, 531), 'peak'), ((797, 821), (787, 811), 'peak'),
                          ((919, 943), (915, 939), 'peak'), ((989, 1013), (989, 1013), 'peak'),
                          ((1149, 1173), (1157, 1181), 'peak'), ((1237, 1261), (1233, 1257), 'peak'),
                          ((1361, 1385), (1373, 1397), 'peak'), ((1591, 1615), (1595, 1619), 'peak')],
          'ch': [((2868, 2892), (2868, 2892), 'shoulder'), ((2923, 2947), (2923, 2947), 'peak')]}

plt.rcParams.update({'font.family': 'sans-serif',
                     'font.sans-serif': ['Arial', 'Liberation Sans', 'DejaVu Sans'],
                     'font.size': TICK_FS})


# ============================================================================
# 1. DATA AND PRE-PROCESSING (identical to Acacia_Paper_All.py)
# ============================================================================
def despike(y, threshold=7.0, m=3):
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


def als_baseline(y, lam=1e5, p=0.01, niter=10):
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
        ys = savgol_filter(np.interp(grid, x[m], despike(yy[m])), 11, 3)
        corr.append(ys - als_baseline(ys))
    corr = np.array(corr)
    return grid, (corr - corr.mean(1, keepdims=True)) / corr.std(1, keepdims=True)


data = {w: [] for w in WINDOWS}
grids, y, sites = {}, [], []
for cls, path in enumerate([EV_FILE, NV_FILE]):
    for site in SITES:
        df = pd.read_excel(path, sheet_name=site, header=2)
        x, Y = df.iloc[:, 0].to_numpy(float), df.iloc[:, 1:].to_numpy(float).T
        for w in WINDOWS:
            grids[w], snv = preprocess(x, Y, w)
            data[w].append(snv)
        y += [cls] * len(Y)
        sites += [f'{GROUPS[cls]}-{site}'] * len(Y)
data = {w: np.vstack(v) for w, v in data.items()}
y, sites = np.array(y), np.array(sites)
print(f'{len(y)} spectra read (EVs {np.sum(y == 0)}, NVs {np.sum(y == 1)})')


# ============================================================================
# 2. BAND POSITIONS FROM THE DATA
# ============================================================================
def band_position(g, s, lo, hi, kind):
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


rows = []
both1728 = all(band_position(grids['fingerprint'], data['fingerprint'][y == c].mean(0), 1715, 1740,
                             'peak') is not None for c in (0, 1))
for region, c, kind, lo, hi, fmt, cls, required in LABELS:
    pos = band_position(grids[region], data[region][y == c].mean(0), lo, hi, kind)
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
    rows.append({'Figure': '2a' if region == 'fingerprint' else '2b', 'Group': GROUPS[c],
                 'Kind': kind, 'Window (cm-1)': f'{lo}-{hi}',
                 'Position (cm-1)': '' if pos is None else round(pos, 1),
                 'Label': '' if text is None else (text + ('- ' + cls.replace('$^{-}$', '-') if cls else '')),
                 'Drawn': draw, 'Note': note, '_region': region, '_c': c, '_pos': pos, '_text': text,
                 '_cls': cls})
table = pd.DataFrame(rows)
show = table.drop(columns=[k for k in table.columns if k.startswith('_')])
pd.set_option('display.width', 200)
pd.set_option('display.max_columns', 20)
print('\nLABEL POSITIONS (from the mean spectra)')
print(show.to_string(index=False))
show.to_csv(OUTPUT_DIR / 'Fig2_label_positions.csv', index=False)


def site_positions(region, c, lo, hi, kind):
    return [band_position(grids[region], data[region][sites == f'{GROUPS[c]}-{k}'].mean(0), lo, hi, kind)
            for k in SITES]


# ============================================================================
# 3. FIGURES
# ============================================================================
def spread(xs, widths, x_min, x_max):
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


def draw_label(ax, x, ybase, segments, column_dx=0.0):
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


def make_figure(region, xlim, figsize, legend_fs, name):
    g = grids[region]
    X = data[region]
    off = OFFSET[region]
    show_ = (g >= xlim[0]) & (g <= xlim[1])
    gs = g[show_]
    fig, ax = plt.subplots(figsize=figsize)
    # grey shadows for consistent shifts
    for (ev_w, nv_w, kind) in SHIFTS[region]:
        pe = band_position(g, X[y == 0].mean(0), *ev_w, kind)
        pn = band_position(g, X[y == 1].mean(0), *nv_w, kind)
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
        ax.plot(gs, mu + o, color=COLS[c], lw=1.4,
                label=f'{NAME_LONG[c]} (mean \u00b1 SD, n = {np.sum(y == c)})')
    ax.set_xlim(*xlim)
    ax.set_ylim((X[y == 0].mean(0) - X[y == 0].std(0, ddof=1))[show_].min() - 0.3,
                (X[y == 1].mean(0) + X[y == 1].std(0, ddof=1))[show_].max() + off + TOP_ROOM[region])
    ax.tick_params(axis='y', direction='in', length=6, width=1.0, labelleft=False)
    ax.tick_params(axis='x', direction='in', length=6, width=1.0, labelsize=TICK_FS)
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
        xl = spread(xp, [w_label] * len(xp), xlim[0] + w_label, xlim[1] - w_label)
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
            texts, top = draw_label(ax, xl[j], base_at(x_peak, xl[j]) + 0.05, segs)
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
                    texts, top = draw_label(ax, xl[j], base_at(x_peak, xl[j]) + 0.05, segs, two_col)
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
                    texts, top = draw_label(ax, xl[j], base_at(x_peak, xl[j]) + 0.05, segs, two_col)
                    x_lab, base = xl[j], base_at(x_peak, xl[j])
                    y_peak = np.interp(x_peak, gs, mu) + o
                    tops.append((c, x_lab, base, top))
                    leader(x_peak, y_peak, x_lab - two_col / 2, base)
                    continue
                texts, top = draw_label(ax, xl[j], base_at(x_peak, xl[j]) + 0.05, segs)
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
    for ext, kw in [('png', {}), ('tiff', {'pil_kwargs': {'compression': 'tiff_lzw'}})]:
        fig.savefig(OUTPUT_DIR / f'{name}.{ext}', dpi=600, **kw)
    print(f'saved {name}.png / .tiff')


print()
make_figure('fingerprint', (400, 1790), (8.5, 7.8), 13, 'Fig2a_MeanSD_Fingerprint')
make_figure('ch', (2760, 3095), (7.0, 6.5), 12, 'Fig2b_MeanSD_CH')
print(f'\nAll outputs saved in: {OUTPUT_DIR}')
if len(sys.argv) <= 1:
    plt.show()
