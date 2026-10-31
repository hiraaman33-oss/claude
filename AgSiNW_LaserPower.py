# =============================================================================
# Bare Ag/SiNW substrate at increasing laser power (532 nm, 1800 gr/mm, 50x LWD,
# 1 s, 1 accumulation): raw spectra, same style as the paper figures.
#
# INPUT: DATA_DIR with the spectra of the bare substrate, one .txt file per laser
#        power, named as recorded, e.g.
#        S7-AgSiNW_532nm_1800gr_BC200_50XLF_1s_1a_100%.txt  (or without % / .txt)
#        Every file whose name contains "AgSiNW" and ends with _<power> or
#        _<power>% is used; "2-5" is read as 2.5 %.
# HOW TO RUN (PyCharm): pip install numpy scipy matplotlib, check DATA_DIR, Run.
# OUTPUT (in DATA_DIR): AgSiNW_LaserPower.png / .tiff (600 dpi) and
#        AgSiNW_LaserPower_Si_band.csv (Si band position and height per power).
# =============================================================================
import re
import sys
from pathlib import Path

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter

DATA_DIR = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(r'C:\Users\Hira Aman\Downloads\AGSINW TEST')
if len(sys.argv) > 1:
    matplotlib.use('Agg')

STACK = True          # True : spectra shifted upwards (low power at the bottom), power written
                      #        on each spectrum, 50-count scale bar (style of the paper figures)
                      # False: spectra on one counts axis, as recorded, with a legend
GAP = 30              # counts between stacked spectra (STACK = True)
XLIM = (400, 2000)
FIGSIZE = (8.5, 6.5)  # inches, same as Fig 2a
FS_LEGEND, FS_AXIS, FS_TICK = 10, 13, 10   # same as Fig 2a/2b

plt.rcParams.update({'font.family': 'sans-serif',
                     'font.sans-serif': ['Arial', 'Liberation Sans', 'DejaVu Sans'],
                     'font.size': FS_TICK})


# ----------------------------------------------------------------------------
# 1. FILES: laser power from the end of the file name
# ----------------------------------------------------------------------------
def laser_power(name):
    stem = name[:-4] if name.lower().endswith('.txt') else name
    m = re.search(r'_([0-9]+(?:[-.,][0-9]+)?)%?$', stem)
    return float(re.sub('[-,]', '.', m.group(1))) if m else None


spectra = []
for f in sorted(DATA_DIR.iterdir()):
    if f.is_file() and f.suffix.lower() in ('.txt', '') and 'agsinw' in f.name.lower():
        p = laser_power(f.name)
        if p is None:
            print(f'skipped (no laser power at the end of the name): {f.name}')
            continue
        spectra.append((p, f.name, np.loadtxt(f)))
if not spectra:
    sys.exit(f'No Ag/SiNW spectra found in {DATA_DIR}')
spectra.sort(key=lambda s: s[0])
for p, name, d in spectra:
    print(f'{p:g} %  ->  {name}  ({len(d)} points, {d[0, 0]:.0f}-{d[-1, 0]:.0f} cm-1)')


# ----------------------------------------------------------------------------
# 2. Si band (first-order phonon, ~520 cm-1): position and height per power
#    (position: maximum of a Savitzky-Golay (11, 3) copy within 505-535 cm-1)
# ----------------------------------------------------------------------------
rows = []
for p, name, d in spectra:
    x, y = d[:, 0], d[:, 1]
    k = (x >= 505) & (x <= 535)
    ys = savgol_filter(y, 11, 3)
    i = np.flatnonzero(k)[np.argmax(ys[k])]
    base = np.median(y[(x >= 560) & (x <= 640)])
    rows.append((p, x[i], ys[i], ys[i] - base))
with open(DATA_DIR / 'AgSiNW_LaserPower_Si_band.csv', 'w') as fh:
    fh.write('Laser power (%),Si band position (cm-1),Si band height (counts),'
             'Si band above background (counts)\n')
    for r in rows:
        fh.write(f'{r[0]:g},{r[1]:.1f},{r[2]:.0f},{r[3]:.0f}\n')
print('\nLaser power   Si band (cm-1)   height above background (counts)')
for r in rows:
    print(f'{r[0]:>9g} %   {r[1]:>12.1f}   {r[3]:>10.0f}')

# ----------------------------------------------------------------------------
# 3. FIGURE
# ----------------------------------------------------------------------------
# one grey hue, light (low power) -> dark (high power)
greys = plt.get_cmap('Greys')(np.linspace(0.40, 0.95, len(spectra)))
fig, ax = plt.subplots(figsize=FIGSIZE)
offset, top = 0.0, None
handles = []
for (p, name, d), col in zip(spectra, greys):
    x, y = d[:, 0], d[:, 1]
    m = (x >= XLIM[0]) & (x <= XLIM[1])
    if STACK and top is not None:
        offset = top + GAP - y[m].min()
    h, = ax.plot(x[m], y[m] + offset, color=col, lw=0.6, label=f'{p:g}%')
    handles.append(h)
    if STACK:                                  # power written above the right end of the spectrum
        r = m & (x >= XLIM[1] - 250)
        label_y = (y[r] + offset).max() + 4
        ax.text(XLIM[1] - 10, label_y, f'{p:g}%', ha='right', va='bottom', fontsize=FS_LEGEND)
    top = (y[m] + offset).max()
ax.set_xlim(*XLIM)
ax.set_xlabel('Raman Shift (cm$^{-1}$)', fontsize=FS_AXIS)
ax.tick_params(axis='x', direction='in', length=5, labelsize=FS_TICK)
if STACK:
    ax.set_ylabel('Intensity (a.u.)', fontsize=FS_AXIS)
    ax.tick_params(axis='y', direction='in', length=5, labelleft=False)
    y0, y1 = ax.get_ylim()
    xb, yb = XLIM[1] - 40, y1 - 0.08 * (y1 - y0) - 50          # 50-count scale bar
    ax.plot([xb, xb], [yb, yb + 50], color='k', lw=1.5)
    ax.text(xb - 15, yb + 25, '50\ncounts', ha='right', va='center', fontsize=FS_LEGEND)
    ax.text(XLIM[1] - 10, label_y + 0.05 * (y1 - y0), 'Laser power', ha='right', va='bottom',
            fontsize=FS_LEGEND)                                 # heading above the power labels
else:
    ax.set_ylabel('Intensity (counts)', fontsize=FS_AXIS)
    ax.tick_params(axis='y', direction='in', length=5, labelsize=FS_TICK)
    ax.legend(handles=handles[::-1], title='Laser power', frameon=True, facecolor='white',
              edgecolor='none', framealpha=1, fontsize=FS_LEGEND, title_fontsize=FS_LEGEND,
              loc='upper right')
fig.tight_layout()


def save_figure(fig, path, **kw):
    """Save; if Windows keeps the old file open, save under a new name."""
    try:
        fig.savefig(path, **kw)
        return path
    except OSError:
        alt = path.with_name(path.stem + '_1' + path.suffix)
        fig.savefig(alt, **kw)
        print(f'{path.name} is open in another program; saved as {alt.name}')
        return alt


save_figure(fig, DATA_DIR / 'AgSiNW_LaserPower.png', dpi=600)
save_figure(fig, DATA_DIR / 'AgSiNW_LaserPower.tiff', dpi=600, pil_kwargs={'compression': 'tiff_lzw'})
print(f'\nSaved in: {DATA_DIR}')
if len(sys.argv) <= 1:
    plt.show()
