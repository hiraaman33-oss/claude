"""
Fit the carbon D and G bands in the AgSiNW (SERS) spectra of EV Acacia and
annotate them against the CaF2 reference.

Model (1100-1800 cm-1, after rubber-band baseline): Lorentzian D (~1350),
Gaussian D3 amorphous-carbon band (~1500, Sadezky et al. 2005), Lorentzian G
(~1590), plus a linear offset.

Usage:  python3 EV_Acacia_DG_Bands.py <single_dir> <caf2_dir> <map_dir>
"""
import glob
import os
import re
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from scipy.signal import savgol_filter
from scipy.spatial import ConvexHull

SINGLE_DIR, CAF2_DIR, MAP_DIR = sys.argv[1:4]
OUT = "EV_Acacia_DG_Bands"
os.makedirs(OUT, exist_ok=True)
G = np.arange(1100, 1800, 1.0)


def read_l6s(path):
    d = open(path, "rb").read()
    runs = []
    for off in range(4):
        a = np.frombuffer(d[off:len(d) - (len(d) - off) % 4], "<f4")
        ok = np.isfinite(a) & (np.abs(a) > 1e-3) & (np.abs(a) < 1e7)
        i = 0
        while i < len(a):
            if ok[i]:
                j = i
                while j < len(a) and ok[j]:
                    j += 1
                if j - i > 1000:
                    runs.append((off + 4 * i, a[i:j].astype(float)))
                i = j
            else:
                i += 1
    wn = [r for o, r in runs if np.all(np.diff(r) > 0) and 50 < r[0] < 5000][0]
    y = [r for o, r in sorted(runs, key=lambda t: t[0]) if len(r) == len(wn) and not np.all(np.diff(r) > 0)][0]
    o = np.argsort(wn)
    return wn[o], y[o]


def rubberband(x, y):
    v = ConvexHull(np.column_stack([x, y])).vertices
    v = np.roll(v, -v.argmin())
    v = v[:v.argmax() + 1]
    return np.interp(x, x[v], y[v])


def prep(wn, y):
    yy = savgol_filter(np.interp(G, wn, y), 9, 2)
    yy = yy - rubberband(G, yy)
    return yy / yy.max()


def lor(x, a, c, w):
    return a * (w / 2) ** 2 / ((x - c) ** 2 + (w / 2) ** 2)


def gau(x, a, c, w):
    return a * np.exp(-4 * np.log(2) * (x - c) ** 2 / w ** 2)


def model(x, aD, cD, wD, a3, c3, w3, aG, cG, wG, b0, b1):
    return lor(x, aD, cD, wD) + gau(x, a3, c3, w3) + lor(x, aG, cG, wG) + b0 + b1 * (x - 1450)


P0 = [0.6, 1360, 150, 0.2, 1500, 120, 1, 1595, 70, 0, 0]
LO = [0, 1320, 40, 0, 1450, 50, 0, 1560, 20, -1, -1e-2]
HI = [5, 1400, 350, 5, 1550, 250, 5, 1620, 200, 1, 1e-2]


def fit(y):
    p, _ = curve_fit(model, G, y, p0=P0, bounds=(LO, HI), maxfev=20000)
    yfit = model(G, *p)
    aD, cD, wD, a3, c3, w3, aG, cG, wG = p[:9]
    area = lambda a, w: a * w * np.pi / 2
    return p, dict(D_pos=round(cD, 1), D_FWHM=round(wD, 1), G_pos=round(cG, 1), G_FWHM=round(wG, 1),
                   D3_pos=round(c3, 1), ID_IG_height=round(aD / aG, 2),
                   AD_AG_area=round(area(aD, wD) / area(aG, wG), 2),
                   R2=round(1 - np.sum((y - yfit) ** 2) / np.sum((y - y.mean()) ** 2), 3))


spectra = {}
for f in sorted(glob.glob(os.path.join(SINGLE_DIR, "*.l6s"))):
    spectra[re.search(r"-(S\d+)-", f).group(1)] = prep(*read_l6s(f))
maps = []
for f in sorted(glob.glob(os.path.join(MAP_DIR, "*.txt"))):
    d = np.genfromtxt(f, comments="#", delimiter="\t", encoding="latin1")
    wn = d[0, 2:]
    o = np.argsort(wn)
    maps += [prep(wn[o], r[2:][o]) for r in d[1:]]
spectra = {"Mean of 100 map spectra": np.mean(maps, axis=0), **spectra}
caf2 = np.mean([prep(*read_l6s(f)) for f in glob.glob(os.path.join(CAF2_DIR, "*.l6s"))], axis=0)

rows, fits = [], {}
for n, y in spectra.items():
    p, r = fit(y)
    fits[n] = p
    rows.append(dict(spectrum=n, **r))
res = pd.DataFrame(rows)
res.to_csv(f"{OUT}/DG_fit_results.csv", index=False)
print(res.to_string(index=False))

# ------------------------------------------------------------------ figure
n = len(spectra) + 1
fig, axs = plt.subplots(n, 1, figsize=(10, 2.6 * n), sharex=True)
for ax, (name, y) in zip(axs, spectra.items()):
    p = fits[name]
    ax.plot(G, y, color="0.2", lw=1.1, label=f"{name} (SERS, AgSiNW)")
    ax.plot(G, model(G, *p), color="k", ls="--", lw=0.9, label="fit")
    base = p[9] + p[10] * (G - 1450)
    ax.fill_between(G, base, lor(G, *p[0:3]) + base, color="#1f77b4", alpha=0.35, label=f"D band {p[1]:.0f}")
    ax.fill_between(G, base, gau(G, *p[3:6]) + base, color="#bbbbbb", alpha=0.5, label=f"D3 (amorphous) {p[4]:.0f}")
    ax.fill_between(G, base, lor(G, *p[6:9]) + base, color="#d62728", alpha=0.35, label=f"G band {p[7]:.0f}")
    r = res[res.spectrum == name].iloc[0]
    ax.text(0.01, 0.92, f"I$_D$/I$_G$ = {r.ID_IG_height}   A$_D$/A$_G$ = {r.AD_AG_area}   R² = {r.R2}",
            transform=ax.transAxes, fontsize=8, va="top")
    ax.legend(fontsize=7, loc="upper right")
ax = axs[-1]
ax.plot(G, caf2, color="#b03a2e", lw=1.2, label="EV Acacia on CaF2 (normal Raman, mean of 5)")
ax.legend(fontsize=7, loc="upper right")
for ax in axs:
    for c, lab in [(1350, "D"), (1456, "1456"), (1590, "G"), (1726, "1726")]:
        ax.axvline(c, color="k", ls=":", lw=0.7)
    ax.set_yticks([])
axs[0].set_title("Carbon D and G bands in the SERS spectra vs EV Acacia on CaF2\n"
                 "(dotted lines: D ≈1350, 1456 CH$_2$, G ≈1590, 1726 C=O)")
axs[-1].set_xlabel("Raman shift (cm$^{-1}$)")
fig.tight_layout()
fig.savefig(f"{OUT}/DG_bands_plot.png", dpi=200, bbox_inches="tight")
print(f"Outputs written to {OUT}/")
