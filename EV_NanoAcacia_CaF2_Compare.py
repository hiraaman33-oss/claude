"""
EV nano Acacia on CaF2 vs EV Acacia on CaF2 – band-by-band comparison.

Both measured as normal Raman on CaF2 (532 nm, 1800 gr, 100x, 60 s x 4).
Peaks are found on each spectrum after a rubber-band baseline (400-1800 and
2750-3100 cm-1, SG 11/2 smoothing). Every band found in either sample is then
measured in all spectra as SNR above a local linear baseline (±30 cm-1 window)
against the raw point-to-point noise, after SG 21/2 smoothing so that single
noise spikes are not counted. A band counts as present at SNR >= 3.

Usage:  python3 EV_NanoAcacia_CaF2_Compare.py <acacia_caf2_dir (.l6s)> <nano_caf2_dir (.txt/.l6s)>
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
from scipy.signal import find_peaks, savgol_filter
from scipy.spatial import ConvexHull

ACACIA_DIR, NANO_DIR = sys.argv[1:3]
OUT = "EV_NanoAcacia_CaF2_Compare"
os.makedirs(OUT, exist_ok=True)
SEGS = [(400, 1800), (2750, 3100)]
GRID = np.concatenate([np.arange(lo, hi, 1.0) for lo, hi in SEGS])
PRESENT = 3.0


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
    return wn, y


def read_any(path):
    if path.endswith(".l6s"):
        wn, y = read_l6s(path)
    else:
        d = np.loadtxt(path)
        wn, y = d[:, 0], d[:, 1]
    o = np.argsort(wn)
    return wn[o], y[o]


def rubberband(x, y):
    v = ConvexHull(np.column_stack([x, y])).vertices
    v = np.roll(v, -v.argmin())
    v = v[:v.argmax() + 1]
    return np.interp(x, x[v], y[v])


def noise(wn, y):
    """Robust raw point-to-point noise over the fingerprint."""
    d = np.diff(y[(wn > 400) & (wn < 1800)])
    return 1.4826 * np.median(np.abs(d - np.median(d))) / np.sqrt(2)


def prep(wn, y):
    out = np.empty_like(GRID)
    for lo, hi in SEGS:
        g = (GRID >= lo) & (GRID < hi)
        yy = savgol_filter(np.interp(GRID[g], wn, y), 11, 2)
        out[g] = yy - rubberband(GRID[g], yy)
    return out


def band_snr(wn, y, c, sigma, half=30):
    """Max within c±8 above a line through the window-edge minima, in units of sigma."""
    w = (wn > c - half) & (wn < c + half)
    # SG 21/2 (about 10 cm-1): a band must be several points wide, single-pixel noise spikes do not count
    x, yy = wn[w], savgol_filter(y[w], 21, 2)
    lft, rgt = x < c - half + 10, x > c + half - 10
    xl, yl = x[lft][np.argmin(yy[lft])], yy[lft].min()
    xr, yr = x[rgt][np.argmin(yy[rgt])], yy[rgt].min()
    h = yy - (yl + (yr - yl) * (x - xl) / (xr - xl))
    core = np.abs(x - c) <= 10
    i = np.argmax(np.where(core, h, -np.inf))
    is_max = 0 < i < len(x) - 1 and h[i] >= h[i - 1] and h[i] >= h[i + 1]
    return round(float(x[i]), 1), round(float(h[i] / sigma), 1) if is_max else 0.0


def name_of(p):
    m = re.search(r"-(s\d+[a-z]?)-", os.path.basename(p))
    return m.group(1) if m else os.path.basename(p)


acacia = {f"Acacia {name_of(f)}": read_any(f) for f in sorted(glob.glob(os.path.join(ACACIA_DIR, "*.l6s")))}
nano = {f"nano {name_of(f)}": read_any(f) for f in sorted(glob.glob(os.path.join(NANO_DIR, "*")))}
allspec = {**acacia, **nano}
sig = {n: noise(*v) for n, v in allspec.items()}

# ------------------------------------------------------------------ band list from both samples
def peaks_of(wn, y, sigma):
    p = prep(wn, y)
    found = []
    for lo, hi in SEGS:
        g = (GRID >= lo) & (GRID < hi)
        idx, _ = find_peaks(p[g], prominence=4 * sigma, distance=12)
        found += list(GRID[g][idx])
    return found


cands, src = [], []
for n, (wn, y) in allspec.items():
    p = peaks_of(wn, y, sig[n])
    cands += p
    src += [n in nano] * len(p)
o = np.argsort(cands)
cands, src = np.array(cands)[o], np.array(src)[o]
clusters = []  # cluster candidates within 10 cm-1: (positions, from-nano flags)
for c, fn in zip(cands, src):
    if clusters and c - clusters[-1][0][-1] <= 10:
        clusters[-1][0].append(c)
        clusters[-1][1].append(fn)
    else:
        clusters.append(([c], [fn]))
# keep a band if it is found in >= 2 Acacia spectra or in the nano spectrum;
# position = median of the Acacia peaks (or of the nano peak if nano-only)
bands = []
for pos, fn in clusters:
    pos, fn = np.array(pos), np.array(fn)
    if (~fn).sum() >= 2:
        bands.append(round(float(np.median(pos[~fn]))))
    elif fn.any():
        bands.append(round(float(np.median(pos[fn]))))

rows = []
for c in bands:
    row = {"band_cm-1": c}
    for n, (wn, y) in allspec.items():
        pos, snr = band_snr(wn, y, c, sig[n])
        row[f"{n} SNR"] = snr
        row[f"{n} pos"] = pos
    a_snr = [row[f"{n} SNR"] for n in acacia]
    n_snr = [row[f"{n} SNR"] for n in nano]
    a_present = sum(s >= PRESENT for s in a_snr)
    n_present = sum(s >= PRESENT for s in n_snr)
    row["Acacia present (of 5)"] = a_present
    row["nano present"] = f"{n_present}/{len(nano)}"
    row["Acacia mean SNR"] = round(float(np.mean(a_snr)), 1)
    shift = max(abs(row[f"{n} pos"] - c) for n in nano)
    if a_present >= 3 and n_present == len(nano):
        verdict = "both" if shift <= 5 else f"both, nano shifted {shift:+.0f}".replace("+", "")
    elif a_present >= 3:
        verdict = "MISSING in nano"
    elif n_present == len(nano):
        verdict = "only in nano"
    else:
        verdict = "weak/uncertain in both"
    row["verdict"] = verdict
    rows.append(row)
df = pd.DataFrame(rows)
df.to_csv(f"{OUT}/band_by_band_comparison.csv", index=False)
show = ["band_cm-1", "Acacia present (of 5)", "Acacia mean SNR"] + [f"{n} SNR" for n in nano] + \
       [f"{n} pos" for n in nano] + ["verdict"]
pd.set_option("display.width", 250)
print(df[show].to_string(index=False))

# ------------------------------------------------------------------ whole-spectrum similarity
P = {n: prep(*v) for n, v in allspec.items()}
aca_mean = np.mean([P[n] / P[n].max() for n in acacia], axis=0)
seg = {"fingerprint 400-1800": (GRID < 1800), "CH 2750-3100": (GRID >= 2750)}
sim = []
for n in allspec:
    row = {"spectrum": n}
    for lab, m in seg.items():
        row[f"r vs Acacia mean ({lab})"] = round(float(np.corrcoef(P[n][m], aca_mean[m])[0, 1]), 3)
    sim.append(row)
simdf = pd.DataFrame(sim)
simdf.to_csv(f"{OUT}/similarity.csv", index=False)
print("\n", simdf.to_string(index=False))
print("\nNoise sigma (counts):", {k: round(v, 1) for k, v in sig.items()})

# ------------------------------------------------------------------ figure
fig, axs = plt.subplots(1, 2, figsize=(18, 7), gridspec_kw=dict(width_ratios=[3, 1]))
curves = [("EV Acacia on CaF2 (mean of 5)", aca_mean, "#b03a2e")] + \
         [(f"EV {n} on CaF2", P[n] / P[n].max(), "#1f4e79") for n in nano]
for k, (lab, y, col) in enumerate(curves[::-1]):
    for a, (lo, hi) in zip(axs, SEGS):
        g = (GRID >= lo) & (GRID < hi)
        yy = y[g] / y[g].max()
        a.plot(GRID[g], yy + 1.15 * k, color=col, lw=1.2)
    axs[0].text(405, 1.15 * k + 0.95, lab, color=col, fontsize=9)
colors = {"both": "#2ca02c", "shifted": "#ff7f0e", "MISSING in nano": "#d62728", "only in nano": "#1f77b4",
          "weak/uncertain in both": "#999999"}
for _, rw in df.iterrows():
    a = axs[0] if rw["band_cm-1"] < 1800 else axs[1]
    col = colors["shifted"] if "shifted" in rw.verdict else colors[rw.verdict]
    a.axvline(rw["band_cm-1"], color=col, ls=":", lw=1)
    a.text(rw["band_cm-1"], 1.0, str(rw["band_cm-1"]), transform=a.get_xaxis_transform(), rotation=90,
           fontsize=7, va="bottom", ha="center", color=col)
for a, (lo, hi) in zip(axs, SEGS):
    a.set_xlim(lo, hi)
    a.set_yticks([])
    a.set_xlabel("Raman shift (cm$^{-1}$)")
handles = [plt.Line2D([], [], color=c, ls=":", label=l) for l, c in colors.items()]
axs[0].legend(handles=handles, fontsize=8, loc="upper left", bbox_to_anchor=(0, -0.08), ncol=4)
fig.suptitle("EV nano Acacia vs EV Acacia, both on CaF2 (normal Raman) – band-by-band", y=1.06)
fig.tight_layout()
fig.savefig(f"{OUT}/nano_vs_acacia_CaF2_bands.png", dpi=200, bbox_inches="tight")
print(f"Outputs written to {OUT}/")

# ------------------------------------------------------------------ zoom check on raw data (light smoothing only)
zooms = [(1400, 1520, [1455, 1466]), (1660, 1790, [1727]), (1060, 1140, [1095, 1118]), (2800, 3050, [2885, 2937])]
fig, axs = plt.subplots(1, len(zooms), figsize=(18, 4.5))
for a, (lo, hi, marks) in zip(axs, zooms):
    for k, (n, (wn, y)) in enumerate([(n, allspec[n]) for n in list(acacia)[:1]] + list(nano.items())):
        w = (wn >= lo) & (wn <= hi)
        yy = savgol_filter(y[w], 7, 2)
        raw = y[w]
        base = np.interp(wn[w], [wn[w][0], wn[w][-1]], [yy[:15].min(), yy[-15:].min()])
        col = "#b03a2e" if n in acacia else "#1f4e79"
        a.plot(wn[w], (raw - base) / sig[n], color=col, lw=0.4, alpha=0.35)
        a.plot(wn[w], (yy - base) / sig[n], color=col, lw=1.3, label=f"EV {n} on CaF2")
    a.axhline(3, color="k", ls="--", lw=0.7)
    for c in marks:
        a.axvline(c, color="k", ls=":", lw=0.8)
        a.text(c, 1.0, str(c), transform=a.get_xaxis_transform(), ha="center", va="bottom", fontsize=8)
    a.set_xlim(lo, hi)
    a.set_xlabel("Raman shift (cm$^{-1}$)")
axs[0].set_ylabel("signal / noise  (dashed: 3σ)")
axs[0].legend(fontsize=7)
fig.suptitle("Raw-data zoom (thin: raw, thick: SG 7/2), in units of each spectrum's noise σ", y=1.04)
fig.tight_layout()
fig.savefig(f"{OUT}/zoom_raw_check.png", dpi=200, bbox_inches="tight")
