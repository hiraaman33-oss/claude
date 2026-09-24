"""
EV nano Acacia on AgSiNW (SERS) vs EV Acacia on CaF2.

Inputs
  single_dir : single spectra (2-column .txt, e.g. s3, s4, s4a)
  map_dir    : LabSpec map exports (*.txt); a map may cover only the fingerprint
               (e.g. m1, 200-1800) or only the CH region (e.g. m6a, 2700-3100)
  caf2_dir   : EV Acacia on CaF2 reference spectra (.l6s)

For every spectrum that covers the fingerprint: 1456/1726 band SNR (local linear
baseline, max within ±8 cm-1), Pearson r vs CaF2, and a carbon D/D3/G fit.
For every spectrum that covers the CH region: r vs CaF2 over 2800-3050 and the
CH band positions.

Usage:  python3 EV_NanoAcacia_Analysis.py <single_dir> <map_dir> <caf2_dir>
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

SINGLE_DIR, MAP_DIR, CAF2_DIR = sys.argv[1:4]
OUT = "EV_NanoAcacia_Analysis"
os.makedirs(OUT, exist_ok=True)
TARGETS = [1456, 1726]
TOL = 8
FP = np.arange(700, 1800, 1.0)
FP_MASK = ~((FP >= 920) & (FP <= 990))       # Si 2nd order on AgSiNW
CH = np.arange(2800, 3050, 1.0)
DG = np.arange(1100, 1800, 1.0)


# ------------------------------------------------------------------ readers
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


def load_map(path):
    """Header row of wavenumbers (with or without leading empty X/Y fields), then X, Y, intensities."""
    rows = [l.rstrip("\n\r").split("\t") for l in open(path, encoding="latin1")
            if not l.startswith("#") and l.strip()]
    wn = np.array([float(v) for v in rows[0] if v.strip()])
    data = np.array([[float(v) for v in r if v.strip()] for r in rows[1:]])
    o = np.argsort(wn)
    wn, spec = wn[o], data[:, 2:][:, o]
    wn, idx = np.unique(wn, return_index=True)
    return wn, data[:, :2], spec[:, idx]


def name_of(path):
    return re.search(r"-([A-Za-z]+\d+[a-z]?)-", os.path.basename(path)).group(1)


# ------------------------------------------------------------------ processing
def rubberband(x, y):
    v = ConvexHull(np.column_stack([x, y])).vertices
    v = np.roll(v, -v.argmin())
    v = v[:v.argmax() + 1]
    return np.interp(x, x[v], y[v])


def prep(wn, y, grid):
    yy = savgol_filter(np.interp(grid, wn, y), 9, 2)
    yy = yy - rubberband(grid, yy)
    return yy / (yy.max() + 1e-12)


def covers(wn, grid):
    return wn.min() <= grid.min() and wn.max() >= grid.max()


def measure(wn, sp, c, half=45):
    w = (wn > c - half) & (wn < c + half)
    x, y = wn[w], savgol_filter(sp[w], 9, 2)
    lft, rgt = x < c - half + 12, x > c + half - 12
    xl, yl = x[lft][np.argmin(y[lft])], y[lft].min()
    xr, yr = x[rgt][np.argmin(y[rgt])], y[rgt].min()
    h = y - (yl + (yr - yl) * (x - xl) / (xr - xl))
    i = np.argmax(np.where(np.abs(x - c) <= 12, h, -np.inf))
    dr = np.diff(sp[w])
    sigma = 1.4826 * np.median(np.abs(dr - np.median(dr))) / np.sqrt(2)
    return round(float(x[i]), 1), round(float(h[i] / sigma), 1) if abs(x[i] - c) <= TOL else 0.0


def r(a, b, m=None):
    m = np.ones_like(a, bool) if m is None else m
    return round(float(np.corrcoef(a[m], b[m])[0, 1]), 3)


def lor(x, a, c, w):
    return a * (w / 2) ** 2 / ((x - c) ** 2 + (w / 2) ** 2)


def gau(x, a, c, w):
    return a * np.exp(-4 * np.log(2) * (x - c) ** 2 / w ** 2)


def dg_model(x, aD, cD, wD, a3, c3, w3, aG, cG, wG, b0, b1):
    return lor(x, aD, cD, wD) + gau(x, a3, c3, w3) + lor(x, aG, cG, wG) + b0 + b1 * (x - 1450)


def dg_fit(y):
    p, _ = curve_fit(dg_model, DG, y, p0=[0.6, 1360, 150, 0.2, 1500, 120, 1, 1595, 70, 0, 0],
                     bounds=([0, 1320, 40, 0, 1450, 50, 0, 1560, 20, -1, -1e-2],
                             [5, 1400, 350, 5, 1550, 250, 5, 1620, 200, 1, 1e-2]), maxfev=20000)
    res = y - dg_model(DG, *p)
    return p, dict(D_pos=round(p[1], 1), G_pos=round(p[7], 1), ID_IG=round(p[0] / p[6], 2),
                   R2_DG=round(1 - np.sum(res ** 2) / np.sum((y - y.mean()) ** 2), 3))


# ------------------------------------------------------------------ load
caf2 = [read_l6s(f) for f in sorted(glob.glob(os.path.join(CAF2_DIR, "*.l6s")))]
ref_fp = np.mean([prep(wn, y, FP) for wn, y in caf2], axis=0)
ref_ch = np.mean([prep(wn, y, CH) for wn, y in caf2], axis=0)
ref_dg = np.mean([prep(wn, y, DG) for wn, y in caf2], axis=0)

spectra = []  # (id, kind, wn, y, extra)
for f in sorted(glob.glob(os.path.join(SINGLE_DIR, "*.txt"))):
    d = np.loadtxt(f)
    o = np.argsort(d[:, 0])
    spectra.append((name_of(f), "single", d[o, 0], d[o, 1], {}))
maps = {}
for f in sorted(glob.glob(os.path.join(MAP_DIR, "*.txt"))):
    m = name_of(f)
    wn, xy, spec = load_map(f)
    maps[m] = (wn, spec)
    for i, (p, y) in enumerate(zip(xy, spec)):
        spectra.append((f"{m}_px{i:02d}", f"map {m}", wn, y,
                        dict(spectrum_no_1based=i + 1, X_um=round(p[0], 3), Y_um=round(p[1], 3))))

# ------------------------------------------------------------------ per-spectrum metrics
rows = []
for sid, kind, wn, y, extra in spectra:
    row = dict(spectrum=sid, type=kind, **extra)
    if covers(wn, FP):
        for c in TARGETS:
            row[f"{c}_pos"], row[f"{c}_SNR"] = measure(wn, y, c)
        row["r_vs_CaF2_fingerprint"] = r(prep(wn, y, FP), ref_fp, FP_MASK)
    if covers(wn, CH):
        row["r_vs_CaF2_CH"] = r(prep(wn, y, CH), ref_ch)
        pch = prep(wn, y, CH)
        row["CH_max_pos"] = float(CH[np.argmax(pch)])
    rows.append(row)
df = pd.DataFrame(rows)
df.to_csv(f"{OUT}/nano_all_spectra_metrics.csv", index=False)

ref_rows = []
for f, (wn, y) in zip(sorted(glob.glob(os.path.join(CAF2_DIR, "*.l6s"))), caf2):
    row = dict(spectrum=name_of(f), type="CaF2 reference")
    for c in TARGETS:
        row[f"{c}_pos"], row[f"{c}_SNR"] = measure(wn, y, c)
    row["CH_max_pos"] = float(CH[np.argmax(prep(wn, y, CH))])
    ref_rows.append(row)

pd.set_option("display.width", 250)
print("CaF2 references:\n", pd.DataFrame(ref_rows).to_string(index=False))
cols = ["spectrum", "type", "1456_pos", "1456_SNR", "1726_pos", "1726_SNR", "r_vs_CaF2_fingerprint",
        "r_vs_CaF2_CH", "CH_max_pos"]
print("\nSingle spectra:\n", df[df.type == "single"][[c for c in cols if c in df]].to_string(index=False))
for m in maps:
    sub = df[df.type == f"map {m}"].dropna(axis=1, how="all")
    print(f"\nMap {m}: {len(sub)} spectra, range {maps[m][0].min():.0f}-{maps[m][0].max():.0f} cm-1")
    if "1456_SNR" in sub:
        both = sub[(sub["1456_SNR"] >= 3) & (sub["1726_SNR"] >= 3)]
        print(f"  spectra with 1456 SNR>=3: {(sub['1456_SNR'] >= 3).sum()}, with 1726 SNR>=3: "
              f"{(sub['1726_SNR'] >= 3).sum()}, with both: {len(both)}")
        print(sub.sort_values("r_vs_CaF2_fingerprint", ascending=False).head(5)[
            [c for c in cols + ["spectrum_no_1based", "X_um", "Y_um"] if c in sub]].to_string(index=False))
    if "r_vs_CaF2_CH" in sub:
        print(f"  r vs CaF2 (CH region): mean {sub['r_vs_CaF2_CH'].mean():.3f}, "
              f"max {sub['r_vs_CaF2_CH'].max():.3f} ({sub.loc[sub['r_vs_CaF2_CH'].idxmax(), 'spectrum']})")

# ------------------------------------------------------------------ D/G fits
fp_sets = {sid: (wn, y) for sid, kind, wn, y, _ in spectra if kind == "single"}
for m, (wn, spec) in maps.items():
    if covers(wn, DG):
        fp_sets[f"{m} mean"] = (wn, spec.mean(axis=0))
dg_rows = []
for n, (wn, y) in fp_sets.items():
    _, res = dg_fit(prep(wn, y, DG))
    dg_rows.append(dict(spectrum=n, **res))
dgdf = pd.DataFrame(dg_rows)
dgdf.to_csv(f"{OUT}/nano_DG_fit.csv", index=False)
print("\nCarbon D/G fit (1100-1800):\n", dgdf.to_string(index=False))

# ------------------------------------------------------------------ figure
fp_show = [("EV Acacia on CaF2 (mean of 5)", prep(*caf2[0], FP) * 0 + ref_fp, "#b03a2e")]
palette = ["#1f77b4", "#2ca02c", "#9467bd", "#ff7f0e", "#8c564b", "#17becf"]
k = 0
for n, (wn, y) in fp_sets.items():
    s = df[df.spectrum == n]
    lab = n if s.empty else (f"{n} – r {s['r_vs_CaF2_fingerprint'].iloc[0]}, 1456 SNR {s['1456_SNR'].iloc[0]}, "
                             f"1726 SNR {s['1726_SNR'].iloc[0]}")
    fp_show.append((f"{lab} (SERS)", prep(wn, y, FP), palette[k % len(palette)]))
    k += 1
ch_show = [("CaF2", ref_ch, "#b03a2e")]
for sid, kind, wn, y, _ in spectra:
    if kind == "single" and covers(wn, CH):
        ch_show.append((sid, prep(wn, y, CH), palette[len(ch_show) - 1]))
for m, (wn, spec) in maps.items():
    if covers(wn, CH):
        ch_show.append((f"{m} mean", prep(wn, spec.mean(axis=0), CH), "0.35"))

fig = plt.figure(figsize=(20, 1.5 * len(fp_show) + 2))
gs = fig.add_gridspec(1, 4, width_ratios=[3, 1, 1, 1.4])
axs = [fig.add_subplot(gs[0, i]) for i in range(4)]
for j, (lab, y, col) in enumerate(fp_show[::-1]):
    for a, (lo, hi) in zip(axs[:3], [(700, 1800), (1400, 1510), (1670, 1780)]):
        w = (FP >= lo) & (FP <= hi)
        yy = np.where(FP_MASK, y, np.nan)[w]
        yy = yy - np.nanmin(yy)
        a.plot(FP[w], yy / np.nanmax(yy) + 1.25 * j, color=col, lw=1.2)
    axs[0].text(705, 1.25 * j + 1.04, lab, fontsize=7.5, color=col)
for j, (lab, y, col) in enumerate(ch_show[::-1]):
    axs[3].plot(CH, y - y.min() + 1.2 * j, color=col, lw=1.2)
    axs[3].text(2805, 1.2 * j + 0.95, lab, fontsize=7.5, color=col)
for a, t, (lo, hi) in zip(axs, ["Fingerprint", "1456 zoom", "1726 zoom", "CH stretch"],
                          [(700, 1800), (1400, 1510), (1670, 1780), (2800, 3050)]):
    marks = TARGETS if hi <= 1800 else [2850, 2885, 2935]
    for c in marks:
        a.axvline(c, color="k", ls=":", lw=0.7)
    a.set_xlim(lo, hi)
    a.set_yticks([])
    a.set_title(t)
    a.set_xlabel("Raman shift (cm$^{-1}$)")
fig.suptitle("EV nano Acacia on AgSiNW (SERS) vs EV Acacia on CaF2")
fig.tight_layout()
fig.savefig(f"{OUT}/nano_vs_CaF2_plot.png", dpi=200, bbox_inches="tight")
print(f"\nOutputs written to {OUT}/")
