# =============================================================================
# ACACIA EVs vs NVs  |  independent check of Acacia_MeanSD_Spectra.py
#
# 1. Standard deviation: the data are read again with a different Excel reader
#    (openpyxl), pre-processed with separately written code (banded ALS solver),
#    and mean / SD are computed point by point with an explicit formula.
# 2. Peak positions, three independent methods for every labelled band:
#    A  parabola through the 3 highest points of the mean spectrum (as in the figure)
#    B  least-squares fit of a Gaussian + straight line to the mean spectrum
#       (+/-10 cm-1; +/-16 cm-1 for the broad 1373/1385 and 1603/1607 bands)
#    C  raw data at full instrument resolution (no interpolation, no smoothing,
#       no ALS): mean of the 120 despiked raw spectra, local straight-line baseline
#       removed, centroid of the top of the band (points above 80 % of its height)
#    plus the spread of the peak position over the 6 measurement sites.
#
# HOW TO RUN:  python Acacia_MeanSD_verify.py  "<CODE 005 folder>"
# OUTPUT: RESULTS\MeanSD_verification.txt
# =============================================================================

import sys
from pathlib import Path

import numpy as np
import openpyxl
from scipy.linalg import solve_banded
from scipy.signal import savgol_filter
from scipy.optimize import curve_fit

BASE_DIR = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(r'C:\Users\Hira Aman\Downloads\CODE 005')
FILES = {'EV': BASE_DIR / 'EV Acacia' / 'EV_Acacia_Raman_Data.xlsx',
         'NV': BASE_DIR / 'Nano vesicles Acacia' / 'NV_Acacia_Raman_Data.xlsx'}
SITES = ['m1', 'm2', 'm3', 'm4', 'm5', 'm6']
sys.argv = sys.argv[:1] + [str(BASE_DIR)]
ns = {}
exec(Path(__file__).with_name('Acacia_MeanSD_Spectra.py').read_text().split('# FIGURE')[0], ns)
PEAKS, peak_position = ns['PEAKS'], ns['peak_position']
report = []


def out(s=''):
    print(s)
    report.append(s)


def read(path):
    wb = openpyxl.load_workbook(path, read_only=True, data_only=True)
    Y, site = [], []
    for s in SITES:
        a = np.array([[np.nan if v is None else v for v in r]
                      for r in wb[s].iter_rows(min_row=4, values_only=True)], float)
        a = a[~np.isnan(a[:, 0])]
        x = a[:, 0]
        Y += list(a[:, 1:].T)
        site += [s] * (a.shape[1] - 1)
    return x, np.array(Y), np.array(site)


def despike2(y, thr=7.0, m=3):
    d = np.diff(y)
    mad = np.median(np.abs(d - np.median(d)))
    z = np.zeros(len(y))
    z[1:] = 0.6745 * (d - np.median(d)) / mad
    bad = np.abs(z) > thr
    o = y.copy()
    for i in np.flatnonzero(bad):
        nb = [j for j in range(max(0, i - m), min(len(y), i + m + 1)) if not bad[j]]
        o[i] = y[nb].mean()
    return o


def als2(y, lam=1e5, p=0.01, niter=10):
    n = len(y)
    D = np.diff(np.eye(n), 2, axis=0)
    H = lam * D.T @ D
    ab = np.zeros((5, n))
    for k in range(-2, 3):
        ab[2 - k, max(0, k):n + min(0, k)] = np.diagonal(H, k)
    w = np.ones(n)
    for _ in range(niter):
        ab2 = ab.copy()
        ab2[2] += w
        z = solve_banded((2, 2), ab2, w * y)
        w = np.where(y > z, p, 1 - p)
    return z


def gauss_line(x, a, x0, s, b0, b1):
    return a * np.exp(-0.5 * ((x - x0) / s) ** 2) + b0 + b1 * (x - x0)


grid = np.arange(205, 1796, 2.0)
show = (grid >= 400) & (grid <= 1790)
out('ACACIA EVs vs NVs - check of the mean +/- SD figure (400-1790 cm-1)')
out('=' * 72)
for tag, path in FILES.items():
    x, Y, site = read(path)
    m = (x >= 195) & (x <= 1805)
    D = np.array([despike2(y[m]) for y in Y])                      # despiked, raw axis
    P = []
    for d in D:
        s = savgol_filter(np.interp(grid, x[m], d), 11, 3)
        c = s - als2(s)
        P.append((c - c.mean()) / c.std())
    P = np.array(P)
    n = len(P)
    mu = P.sum(0) / n                                              # explicit formulas
    sd0 = np.sqrt(((P - mu) ** 2).sum(0) / n)
    sd1 = np.sqrt(((P - mu) ** 2).sum(0) / (n - 1))
    g_ref, X_ref, _ = ns['load_snv'](path)
    out(f'\n{tag}: {n} spectra ({", ".join(f"{k}: {(site == k).sum()}" for k in SITES)})')
    out(f'  mean: max |difference| to the figure script  {np.abs(mu - X_ref.mean(0)).max():.1e}')
    out(f'  SD (n-1): max |difference| to the figure script  '
        f'{np.abs(sd1 - X_ref.std(0, ddof=1)).max():.1e}')
    out(f'  SD over 400-1790: median {np.median(sd1[show]):.3f}, range {sd1[show].min():.3f}'
        f'-{sd1[show].max():.3f} SNV units; SD(n-1)/SD(n) = {np.sqrt(n / (n - 1)):.4f}')

    xr = x[m]
    raw_mean = D.mean(0)
    out(f'  raw point spacing {np.diff(xr).min():.2f}-{np.diff(xr).max():.2f} cm-1')
    out(f'  {"band":>6} {"A parab.":>9} {"B Gauss":>9} {"C raw":>8} {"max|A-B|,|A-C|":>15}'
        f' {"6 sites (min-max)":>19}  label')
    for nominal, cls, _, _ in PEAKS[tag]:
        A = peak_position(grid, mu, nominal - 12, nominal + 12)
        half = 10 if nominal not in (1373, 1385, 1603, 1607) else 16   # broad bands: wider
        w = np.abs(grid - A) <= half
        p0 = [mu[w].max() - mu[w].min(), A, 5, mu[w].min(), 0]
        Bp = curve_fit(gauss_line, grid[w], mu[w], p0=p0, maxfev=20000,
                       bounds=([0, A - 8, 1, -np.inf, -np.inf], [np.inf, A + 8, 40, np.inf, np.inf]))[0][1]
        k = np.abs(xr - A) <= 12                                   # raw resolution
        xx, yy = xr[k], raw_mean[k]
        base = np.interp(xx, [xx[0], xx[-1]], [yy[0], yy[-1]])
        h = yy - base
        top = h >= 0.8 * h.max()
        C = (xx[top] * h[top]).sum() / h[top].sum()
        sp = [peak_position(grid, mu * 0 + P[site == k2].mean(0), nominal - 12, nominal + 12)
              for k2 in SITES]
        lab = f'{A:.1f}' if cls == 'Si' else f'{A:.0f}'
        dev = max(abs(A - Bp), abs(A - C))
        out(f'  {nominal:>6} {A:9.1f} {Bp:9.1f} {C:8.1f} {dev:15.1f} '
            f'{min(sp):9.1f}-{max(sp):<9.1f}  {lab}')

(BASE_DIR / 'RESULTS').mkdir(exist_ok=True)
(BASE_DIR / 'RESULTS' / 'MeanSD_verification.txt').write_text('\n'.join(report) + '\n',
                                                              encoding='utf-8')
