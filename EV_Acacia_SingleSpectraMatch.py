"""
EV Acacia – compare single AgSiNW spectra (S1, S2) with the CaF2 reference
spectra and with every map pixel, focusing on the 1456 and 1726 cm-1 bands.

Usage:  python3 EV_Acacia_SingleSpectraMatch.py <single_dir> <caf2_dir> <map_dir>
  single_dir : S1/S2 spectra (.l6s or 2-column .txt; one file per name is used)
  caf2_dir   : EV Acacia on CaF2 reference spectra (.l6s)
  map_dir    : LabRAM map exports (*.txt)
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
from scipy.signal import savgol_filter
from scipy.spatial import ConvexHull

SINGLE_DIR, CAF2_DIR, MAP_DIR = sys.argv[1:4]
OUT = "EV_Acacia_SingleSpectraMatch"
os.makedirs(OUT, exist_ok=True)
TARGETS = [1456, 1726]
TOL = 8
G = np.arange(700, 1800, 1.0)
G_MASK = ~((G >= 920) & (G <= 990))  # Si 2nd order in AgSiNW spectra


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


def read_any(path):
    if path.endswith(".l6s"):
        return read_l6s(path)
    d = np.loadtxt(path)
    o = np.argsort(d[:, 0])
    return d[o, 0], d[o, 1]


def measure(wn, sp, c, half=45):
    """Band position and SNR above a local linear baseline (0 if no maximum within ±TOL)."""
    w = (wn > c - half) & (wn < c + half)
    x, y = wn[w], savgol_filter(sp[w], 9, 2)
    lft, rgt = x < c - half + 12, x > c + half - 12
    xl, yl = x[lft][np.argmin(y[lft])], y[lft].min()
    xr, yr = x[rgt][np.argmin(y[rgt])], y[rgt].min()
    h = y - (yl + (yr - yl) * (x - xl) / (xr - xl))
    core = np.abs(x - c) <= 12
    i = np.argmax(np.where(core, h, -np.inf))
    dr = np.diff(sp[w])
    sigma = 1.4826 * np.median(np.abs(dr - np.median(dr))) / np.sqrt(2)
    snr = h[i] / sigma if abs(x[i] - c) <= TOL else 0.0
    return round(float(x[i]), 1), round(float(snr), 1)


def rubberband(x, y):
    v = ConvexHull(np.column_stack([x, y])).vertices
    v = np.roll(v, -v.argmin())
    v = v[:v.argmax() + 1]
    return np.interp(x, x[v], y[v])


def prep(wn, y):
    yy = savgol_filter(np.interp(G, wn, y), 9, 2)
    yy = yy - rubberband(G, yy)
    return yy / yy[G_MASK].max()


def r(a, b, m=G_MASK):
    return round(float(np.corrcoef(a[m], b[m])[0, 1]), 3)


def win_r(a, b):
    w = G_MASK & (((G >= 1400) & (G <= 1510)) | ((G >= 1670) & (G <= 1780)))
    return r(a, b, w)


# ------------------------------------------------------------------ load
singles = {}
for f in sorted(glob.glob(os.path.join(SINGLE_DIR, "*"))):
    name = re.search(r"-(S\d+)-", os.path.basename(f)).group(1)
    if name not in singles or f.endswith(".l6s"):
        singles[name] = read_any(f)
caf2 = {re.search(r"-(s\d+)-", os.path.basename(f)).group(1): read_l6s(f)
        for f in sorted(glob.glob(os.path.join(CAF2_DIR, "*.l6s")))}
ref = np.mean([prep(*v) for v in caf2.values()], axis=0)

maps = {}
for f in sorted(glob.glob(os.path.join(MAP_DIR, "*.txt"))):
    m = re.search(r"-(m\d+)-", os.path.basename(f)).group(1)
    d = np.genfromtxt(f, comments="#", delimiter="\t", encoding="latin1")
    wn = d[0, 2:]
    o = np.argsort(wn)
    for i, row in enumerate(d[1:]):
        maps[f"{m}_px{i:02d}"] = (wn[o], row[2:][o], i + 1, row[0], row[1])

# ------------------------------------------------------------------ S1/S2 and CaF2 bands + similarity
rows = []
for name, (wn, y) in list(caf2.items()) + list(singles.items()):
    row = dict(spectrum=name, type="CaF2 reference" if name.startswith("s") else "AgSiNW single")
    for c in TARGETS:
        row[f"{c}_pos"], row[f"{c}_SNR"] = measure(wn, y, c)
    p = prep(wn, y)
    row["r_vs_CaF2_fingerprint"] = r(p, ref)
    row["r_vs_CaF2_1456_1726_windows"] = win_r(p, ref)
    rows.append(row)
summary = pd.DataFrame(rows)
summary.to_csv(f"{OUT}/S1_S2_vs_CaF2.csv", index=False)
print(summary.to_string(index=False))

# ------------------------------------------------------------------ which map pixels look like S1 / S2
S = {n: prep(*v) for n, v in singles.items()}
mrows = []
for sid, (wn, y, no, x, yy) in maps.items():
    p = prep(wn, y)
    row = dict(spectrum=sid, map=sid[:2], spectrum_no_1based=no, X_um=round(x, 3), Y_um=round(yy, 3))
    for c in TARGETS:
        row[f"{c}_pos"], row[f"{c}_SNR"] = measure(wn, y, c)
    for n, s in S.items():
        row[f"r_vs_{n}"] = r(p, s)
    row["r_vs_CaF2"] = r(p, ref)
    mrows.append(row)
mdf = pd.DataFrame(mrows)
mdf.to_csv(f"{OUT}/map_pixels_vs_S1_S2_CaF2.csv", index=False)
for n in S:
    print(f"\nMap pixels most similar to {n}:")
    print(mdf.sort_values(f"r_vs_{n}", ascending=False).head(5)[
        ["spectrum", "spectrum_no_1based", "X_um", "Y_um", f"r_vs_{n}", "1456_SNR", "1726_SNR"]].to_string(index=False))
mean_map = np.mean([prep(v[0], v[1]) for v in maps.values()], axis=0)
for n, s in S.items():
    print(f"r({n}, mean of all 100 map spectra) = {r(s, mean_map)}")

# ------------------------------------------------------------------ plot
show = [("EV Acacia on CaF2 (mean s4,s5,s8,s9,s10)", ref, "#b03a2e")]
cols = ["#1f77b4", "#2ca02c", "#9467bd", "#ff7f0e"]
for k, n in enumerate(S):
    s = summary[summary.spectrum == n].iloc[0]
    show.append((f"{n} (AgSiNW) – r vs CaF2 {s['r_vs_CaF2_fingerprint']}, 1456: {s['1456_pos']} cm$^{{-1}}$ "
                 f"SNR {s['1456_SNR']}, 1726: {s['1726_pos']} cm$^{{-1}}$ SNR {s['1726_SNR']}", S[n], cols[k]))
show.append(("Mean of all 100 map spectra (m1–m5)", mean_map, "0.4"))
fig, ax = plt.subplots(1, 3, figsize=(18, 2.1 * len(show) + 1), gridspec_kw=dict(width_ratios=[3, 1, 1]))
for k, (lab, y, col) in enumerate(show[::-1]):
    for a, (lo, hi) in zip(ax, [(700, 1800), (1400, 1510), (1670, 1780)]):
        w = (G >= lo) & (G <= hi)
        yy = np.where(G_MASK, y, np.nan)[w]
        yy = yy - np.nanmin(yy)
        a.plot(G[w], yy / np.nanmax(yy) + 1.25 * k, color=col, lw=1.3)
    ax[0].text(705, 1.25 * k + 1.04, lab, fontsize=8, color=col)
for a, t, (lo, hi) in zip(ax, ["Fingerprint", "1456 cm$^{-1}$ zoom", "1726 cm$^{-1}$ zoom"],
                          [(700, 1800), (1400, 1510), (1670, 1780)]):
    for c in TARGETS:
        a.axvline(c, color="k", ls=":", lw=0.8)
    a.set_xlim(lo, hi)
    a.set_yticks([])
    a.set_title(t)
    a.set_xlabel("Raman shift (cm$^{-1}$)")
fig.suptitle("EV Acacia: single AgSiNW spectra S1, S2 vs CaF2 reference")
fig.tight_layout()
fig.savefig(f"{OUT}/S1_S2_vs_CaF2_plot.png", dpi=200, bbox_inches="tight")
print(f"\nOutputs written to {OUT}/")
