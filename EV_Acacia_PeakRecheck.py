"""
EV Acacia – direct re-check of the 1456 and 1726 cm-1 bands in every map spectrum.

Works on the raw spectra (light SG smoothing only). Each band is measured above a
local linear baseline (c ± 45 cm-1) and expressed as SNR against the raw
point-to-point noise. Spectrum numbers are given 0-based (pixel) and 1-based
(order of the spectrum in the exported file, as seen in Origin/LabSpec).

Usage:  python3 EV_Acacia_PeakRecheck.py <map_dir> <reference_dir>
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

MAP_DIR = sys.argv[1] if len(sys.argv) > 1 else "."
REF_DIR = sys.argv[2] if len(sys.argv) > 2 else MAP_DIR
OUT = "EV_Acacia_PeakMatch"
os.makedirs(OUT, exist_ok=True)
TARGETS = [1456, 1726]
G = np.arange(700, 1800, 1.0)


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


def measure(wn, sp, c, half=45):
    """Peak position, SNR and height of band c above a local linear baseline."""
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
    is_max = abs(x[i] - c) <= 11.5  # a real maximum, not the window edge
    return x[i], h[i] / sigma, is_max


def rubberband(x, y):
    v = ConvexHull(np.column_stack([x, y])).vertices
    v = np.roll(v, -v.argmin())
    v = v[:v.argmax() + 1]
    return np.interp(x, x[v], y[v])


def display(wn, y):
    yy = savgol_filter(np.interp(G, wn, y), 9, 2)
    return yy - rubberband(G, yy)


refs = [read_l6s(f) for f in sorted(glob.glob(os.path.join(REF_DIR, "*.l6s")))]
ref_rows = []
for (wn, y), f in zip(refs, sorted(glob.glob(os.path.join(REF_DIR, "*.l6s")))):
    r = dict(reference=re.search(r"-(s\d+)-", f).group(1))
    for c in TARGETS:
        p, snr, _ = measure(wn, y, c)
        r[f"{c}_pos"], r[f"{c}_SNR"] = round(p, 1), round(snr, 1)
    ref_rows.append(r)
print("CaF2 references:\n", pd.DataFrame(ref_rows).to_string(index=False))

rows, disp = [], {}
for f in sorted(glob.glob(os.path.join(MAP_DIR, "*.txt"))):
    m = re.search(r"-(m\d+)-", os.path.basename(f)).group(1)
    d = np.genfromtxt(f, comments="#", delimiter="\t", encoding="latin1")
    wn = d[0, 2:]
    o = np.argsort(wn)
    wn = wn[o]
    for i, r in enumerate(d[1:]):
        sp = r[2:][o]
        sid = f"{m}_px{i:02d}"
        row = dict(spectrum=sid, map=m, pixel_0based=i, spectrum_no_1based=i + 1,
                   X_um=round(r[0], 3), Y_um=round(r[1], 3))
        for c in TARGETS:
            p, snr, ok = measure(wn, sp, c)
            row[f"{c}_pos"], row[f"{c}_SNR"] = round(p, 1), round(snr, 1) if ok else 0.0
        disp[sid] = display(wn, sp)
        rows.append(row)
df = pd.DataFrame(rows)
TOL = 8  # a band only counts if its maximum is within ±8 cm-1 of the CaF2 position
for c in TARGETS:
    df.loc[(df[f"{c}_pos"] - c).abs() > TOL, f"{c}_SNR"] = 0.0
df["both_min_SNR"] = df[["1456_SNR", "1726_SNR"]].min(axis=1)
df = df.sort_values("both_min_SNR", ascending=False)
df.to_csv(f"{OUT}/recheck_1456_1726_all_spectra.csv", index=False)
print(df.head(10).to_string(index=False))

best1456 = df.sort_values("1456_SNR", ascending=False).iloc[0]
best1726 = df.sort_values("1726_SNR", ascending=False).iloc[0]
bestboth = df.iloc[0]
print("\nStrongest 1456:", best1456.spectrum, "| strongest 1726:", best1726.spectrum,
      "| best with both:", bestboth.spectrum)

ref = np.mean([display(*r) / display(*r)[(G > 1380) & (G < 1780)].max() for r in refs], axis=0)
show = [("EV Acacia on CaF2 (mean s4,s5,s8,s9,s10)", ref, "#b03a2e")]
for lab, row, col in [("strongest 1456", best1456, "#1f77b4"), ("strongest 1726", best1726, "#9467bd"),
                      ("best with both (weak)", bestboth, "#2ca02c")]:
    show.append((f"{row['spectrum']} (spectrum #{row['spectrum_no_1based']} in {row['map']}, X={row['X_um']}, "
                 f"Y={row['Y_um']} µm) – {lab}: 1456 SNR {row['1456_SNR']}, 1726 SNR {row['1726_SNR']}",
                 disp[row["spectrum"]], col))
fig, ax = plt.subplots(1, 3, figsize=(18, 8), gridspec_kw=dict(width_ratios=[3, 1, 1]))
for k, (lab, y, col) in enumerate(show[::-1]):
    for a, (lo, hi) in zip(ax, [(700, 1800), (1400, 1510), (1670, 1780)]):
        w = (G >= lo) & (G <= hi)
        yy = y[w] - y[w].min()
        a.plot(G[w], yy / yy.max() + 1.2 * k, color=col, lw=1.3)
    ax[0].text(705, 1.2 * k + 1.03, lab, fontsize=8, color=col)
for a, t, (lo, hi) in zip(ax, ["Fingerprint", "1456 cm$^{-1}$ zoom", "1726 cm$^{-1}$ zoom"],
                          [(700, 1800), (1400, 1510), (1670, 1780)]):
    for c in TARGETS:
        a.axvline(c, color="k", ls=":", lw=0.8)
    a.set_xlim(lo, hi)
    a.set_title(t)
    a.set_yticks([])
    a.set_xlabel("Raman shift (cm$^{-1}$)")
fig.suptitle("Re-check of 1456 and 1726 cm$^{-1}$ in all 100 map spectra vs EV Acacia on CaF2")
fig.tight_layout()
fig.savefig(f"{OUT}/recheck_1456_1726_plot.png", dpi=200, bbox_inches="tight")
print(f"Outputs written to {OUT}/")
