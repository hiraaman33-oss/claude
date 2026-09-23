"""
EV Acacia – find map spectra carrying the 1456 and 1726 cm-1 bands of the CaF2
reference and rank them by similarity to the CaF2 spectra.

Peak presence is re-checked with 10 independent detection variants (different
baseline, smoothing, tolerance, threshold, raw data, 2nd derivative, band fit).
A spectrum only counts as having a peak when it passes the check in every
variant it is required to pass. The CaF2 references are run through the same
checks as a positive control.

Usage:  python3 EV_Acacia_PeakMatch.py <map_dir> <reference_dir>
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
from scipy import sparse
from scipy.optimize import curve_fit
from scipy.signal import medfilt, savgol_filter
from scipy.sparse.linalg import spsolve
from scipy.spatial import ConvexHull

MAP_DIR = sys.argv[1] if len(sys.argv) > 1 else "."
REF_DIR = sys.argv[2] if len(sys.argv) > 2 else MAP_DIR
OUT = "EV_Acacia_PeakMatch"
os.makedirs(OUT, exist_ok=True)

TARGETS = [1456, 1726]
FP = np.arange(400, 1800, 1.0)          # fingerprint grid
CH = np.arange(2750, 3100, 1.0)          # CH-stretch grid
EXCLUDE = [(500, 545), (920, 990)]       # Si bands in the AgSiNW maps
FP_MASK = np.ones_like(FP, bool)
for lo, hi in EXCLUDE:
    FP_MASK &= ~((FP >= lo) & (FP <= hi))


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
    return wn, y


def load_map(path):
    d = np.genfromtxt(path, comments="#", delimiter="\t", encoding="latin1")
    wn, xy, spec = d[0, 2:], d[1:, :2], d[1:, 2:]
    o = np.argsort(wn)
    wn, spec = wn[o], spec[:, o]
    wn, idx = np.unique(wn, return_index=True)
    return wn, xy, spec[:, idx]


# ------------------------------------------------------------------ preprocessing
def despike(y, k=7, thr=6):
    m = medfilt(y, k)
    r = y - m
    mad = np.median(np.abs(r - np.median(r))) + 1e-9
    y = y.copy()
    bad = np.abs(r) > thr * 1.4826 * mad
    y[bad] = m[bad]
    return y


def als(y, lam=1e5, p=0.01, niter=10):
    L = len(y)
    D = sparse.diags([1.0, -2.0, 1.0], [0, -1, -2], shape=(L, L - 2))
    DD = lam * D.dot(D.T)
    w = np.ones(L)
    for _ in range(niter):
        z = spsolve((sparse.spdiags(w, 0, L, L) + DD).tocsc(), w * y)
        w = p * (y > z) + (1 - p) * (y < z)
    return z


def rubberband(x, y):
    v = ConvexHull(np.column_stack([x, y])).vertices
    v = np.roll(v, -v.argmin())
    v = v[:v.argmax() + 1]
    return np.interp(x, x[v], y[v])


def on_grid(wn, y, grid):
    return np.interp(grid, wn, despike(y))


def noise_sigma(raw):
    """Robust point-to-point noise of the unsmoothed spectrum."""
    d = np.diff(raw)[FP_MASK[:-1] & FP_MASK[1:]]
    return 1.4826 * np.median(np.abs(d - np.median(d))) / np.sqrt(2) + 1e-9


def base_rb_als(y):
    y = y - rubberband(FP, y)
    return y - als(y, lam=1e7, p=0.001)


# ------------------------------------------------------------------ peak checks
def local_peak(y, c, tol, sigma, k):
    """Local maximum within c±tol that is not at the window edge, with height
    above a local linear baseline (anchors at c±40) > k*sigma."""
    w = (FP >= c - tol) & (FP <= c + tol)
    idx = np.where(w)[0]
    i = idx[np.argmax(y[idx])]
    if i in (idx[0], idx[-1]):
        return False, FP[i], 0.0
    la = (FP >= c - 45) & (FP <= c - 30)
    ra = (FP >= c + 30) & (FP <= c + 45)
    xl, yl = FP[la].mean(), np.min(y[la])
    xr, yr = FP[ra].mean(), np.min(y[ra])
    base = yl + (yr - yl) * (FP[i] - xl) / (xr - xl)
    h = y[i] - base
    return h > k * sigma, FP[i], h / sigma


def lorentz(x, a, x0, g, b0, b1):
    return a * g ** 2 / ((x - x0) ** 2 + g ** 2) + b0 + b1 * (x - x0)


def fit_peak(y, c, sigma):
    w = (FP >= c - 40) & (FP <= c + 40)
    x, yy = FP[w], y[w]
    try:
        p, _ = curve_fit(lorentz, x, yy, p0=[max(yy.max() - yy.min(), 1e-6), c, 8, yy.min(), 0],
                         bounds=([0, c - 15, 2, -np.inf, -np.inf], [np.inf, c + 15, 35, np.inf, np.inf]),
                         maxfev=5000)
    except Exception:
        return False, np.nan, 0.0
    a, x0, g = p[:3]
    return (a > 3 * sigma) and (2.5 <= 2 * g <= 60) and abs(x0 - c) <= 12, x0, a / sigma


def check_all(raw_fp):
    """Run the 10 detection variants for both target bands."""
    sigma = noise_sigma(raw_fp)
    sg11 = savgol_filter(raw_fp, 11, 3)
    variants = {
        "V1 rubberband+ALS, SG11, ±10, 3σ": (base_rb_als(sg11), 10, 3),
        "V2 rubberband+ALS, SG11, ±15, 3σ": (base_rb_als(sg11), 15, 3),
        "V3 rubberband+ALS, SG7, ±10, 3σ": (base_rb_als(savgol_filter(raw_fp, 7, 3)), 10, 3),
        "V4 rubberband+ALS, SG21, ±10, 3σ": (base_rb_als(savgol_filter(raw_fp, 21, 3)), 10, 3),
        "V5 ALS only (λ1e5), SG11, ±10, 3σ": (sg11 - als(sg11), 10, 3),
        "V6 raw (no smoothing), local baseline, ±10, 3σ": (raw_fp, 10, 3),
        "V7 rubberband+ALS, SG11, ±10, 5σ (strict)": (base_rb_als(sg11), 10, 5),
        "V8 rubberband+ALS, SG11, ±6, 3σ (tight position)": (base_rb_als(sg11), 6, 3),
    }
    res = {}
    for c in TARGETS:
        for name, (y, tol, k) in variants.items():
            ok, pos, snr = local_peak(y, c, tol, sigma, k)
            res[(c, name)] = (ok, pos, snr)
        # V9: 2nd-derivative minimum (band curvature) at the target
        d2 = -savgol_filter(raw_fp, 15, 3, deriv=2)
        s2 = 1.4826 * np.median(np.abs(d2 - np.median(d2))) + 1e-12
        w = (FP >= c - 10) & (FP <= c + 10)
        i = np.where(w)[0][np.argmax(d2[w])]
        res[(c, "V9 2nd-derivative band, ±10, 3σ")] = (d2[i] > 3 * s2, FP[i], d2[i] / s2)
        # V10: Lorentzian band fit on raw data
        res[(c, "V10 Lorentzian fit on raw, ±12, 3σ")] = fit_peak(raw_fp, c, sigma)
    return res


VARIANT_NAMES = None


def summarise(res):
    global VARIANT_NAMES
    VARIANT_NAMES = sorted({n for _, n in res}, key=lambda s: int(s.split()[0][1:]))
    row = {}
    for c in TARGETS:
        passes = [bool(res[(c, n)][0]) for n in VARIANT_NAMES]
        row[f"{c}_checks_passed"] = int(sum(passes))
        row[f"{c}_pos_cm-1"] = round(float(res[(c, VARIANT_NAMES[0])][1]), 1)
        row[f"{c}_SNR"] = round(float(res[(c, VARIANT_NAMES[0])][2]), 1)
        for n, p in zip(VARIANT_NAMES, passes):
            row[f"{c} {n.split()[0]}"] = p
    return row


# ------------------------------------------------------------------ similarity to CaF2
def proc_for_match(wn, y):
    fp = base_rb_als(savgol_filter(on_grid(wn, y, FP), 11, 3))
    ch = savgol_filter(on_grid(wn, y, CH), 11, 3)
    ch = ch - rubberband(CH, ch)
    return fp, ch


def zc(a, mask=None):
    a = a[mask] if mask is not None else a
    a = a - a.mean()
    return a / (np.linalg.norm(a) + 1e-12)


def pearson(a, b, mask=None):
    return float(zc(a, mask) @ zc(b, mask))


# ------------------------------------------------------------------ load references
refs = {}
for f in sorted(glob.glob(os.path.join(REF_DIR, "*.l6s")), key=lambda f: int(re.search(r"-s(\d+)-", f).group(1))):
    name = re.search(r"-(s\d+)-", os.path.basename(f)).group(1)
    wn, y = read_l6s(f)
    refs[name] = dict(raw_fp=on_grid(wn, y, FP), proc=proc_for_match(wn, y))

ref_rows = []
for n, r in refs.items():
    row = dict(spectrum=n, **summarise(check_all(r["raw_fp"])))
    ref_rows.append(row)
ref_df = pd.DataFrame(ref_rows)
ref_df.to_csv(f"{OUT}/positive_control_CaF2_peak_checks.csv", index=False)
print("Positive control – CaF2 references (checks passed out of 10):")
print(ref_df[["spectrum", "1456_checks_passed", "1456_pos_cm-1", "1456_SNR",
              "1726_checks_passed", "1726_pos_cm-1", "1726_SNR"]].to_string(index=False))

ref_fp = np.mean([r["proc"][0] / r["proc"][0][FP_MASK].max() for r in refs.values()], axis=0)
ref_ch = np.mean([r["proc"][1] / r["proc"][1].max() for r in refs.values()], axis=0)
band_win = ((FP >= 1400) & (FP <= 1500)) | ((FP >= 1680) & (FP <= 1780))

# ------------------------------------------------------------------ maps
rows, store = [], {}
for f in sorted(glob.glob(os.path.join(MAP_DIR, "*.txt"))):
    base = os.path.basename(f)
    mname = re.search(r"-(m\d+)-", base).group(1)
    wn, xy, spec = load_map(f)
    for i, (p, y) in enumerate(zip(xy, spec)):
        sid = f"{mname}_px{i:02d}"
        raw_fp = on_grid(wn, y, FP)
        fp, ch = proc_for_match(wn, y)
        store[sid] = (raw_fp, fp, ch)
        row = dict(spectrum=sid, map=mname, pixel=i, X_um=round(p[0], 3), Y_um=round(p[1], 3),
                   map_file=re.sub(r"^[0-9a-f]{8}-", "", base))
        row.update(summarise(check_all(raw_fp)))
        # similarity to the mean CaF2 spectrum
        row["r_fingerprint"] = round(pearson(fp, ref_fp, FP_MASK), 4)
        row["r_CH"] = round(pearson(ch, ref_ch), 4)
        row["r_1456_1726_windows"] = round(pearson(fp, ref_fp, band_win), 4)
        row["r_per_ref_min"] = round(min(pearson(fp, r["proc"][0], FP_MASK) for r in refs.values()), 4)
        # intensity ratio I1726/I1456 vs CaF2
        rows.append(row)
df = pd.DataFrame(rows)

ref_ratio = np.mean([r["proc"][0][np.abs(FP - 1726) <= 8].max() / r["proc"][0][np.abs(FP - 1456) <= 8].max()
                     for r in refs.values()])
df["I1726/I1456"] = [round(store[s][1][np.abs(FP - 1726) <= 8].max() /
                           max(store[s][1][np.abs(FP - 1456) <= 8].max(), 1e-9), 3) for s in df.spectrum]
df["both_peaks_10of10"] = (df["1456_checks_passed"] == 10) & (df["1726_checks_passed"] == 10)
df["both_peaks_min_checks"] = df[["1456_checks_passed", "1726_checks_passed"]].min(axis=1)
df["match_score"] = (0.4 * df["r_1456_1726_windows"] + 0.4 * df["r_fingerprint"] + 0.2 * df["r_CH"]).round(4)
df = df.sort_values(["both_peaks_min_checks", "match_score"], ascending=[False, False]).reset_index(drop=True)
df.to_csv(f"{OUT}/peak_checks_and_match_all_spectra.csv", index=False)

print(f"\nCaF2 mean I1726/I1456 = {ref_ratio:.3f}")
print("\nDistribution of min(checks passed for 1456, 1726) over 100 map spectra:")
print(df["both_peaks_min_checks"].value_counts().sort_index(ascending=False).to_string())
cols = ["spectrum", "X_um", "Y_um", "1456_checks_passed", "1456_pos_cm-1", "1456_SNR",
        "1726_checks_passed", "1726_pos_cm-1", "1726_SNR", "I1726/I1456",
        "r_1456_1726_windows", "r_fingerprint", "r_CH", "match_score"]
print("\nTop 15 (sorted by peak evidence, then match score):")
print(df[cols].head(15).to_string(index=False))

# ------------------------------------------------------------------ stability of the pick (10 re-rankings)
cand = df[df["both_peaks_10of10"]].copy()
if cand.empty:
    cand = df[df["both_peaks_min_checks"] == df["both_peaks_min_checks"].max()].copy()
weights = [(0.4, 0.4, 0.2), (1, 0, 0), (0, 1, 0), (0.5, 0.5, 0), (0.34, 0.33, 0.33),
           (0.6, 0.3, 0.1), (0.2, 0.6, 0.2), (0.3, 0.3, 0.4), (0.5, 0.25, 0.25), (0.25, 0.5, 0.25)]
picks = []
for a, b_, c in weights:
    s = a * cand["r_1456_1726_windows"] + b_ * cand["r_fingerprint"] + c * cand["r_CH"]
    picks.append(cand.loc[s.idxmax(), "spectrum"])
stab = pd.Series(picks).value_counts()
print("\nRe-ranking the qualifying spectra 10 times with different score weights – winner counts:")
print(stab.to_string())
best = stab.index[0]
pd.DataFrame({"weights(win,fp,CH)": [str(w) for w in weights], "winner": picks}).to_csv(
    f"{OUT}/stability_10_rerankings.csv", index=False)

# ------------------------------------------------------------------ figures
b = df[df.spectrum == best].iloc[0]
raw_fp, fp, ch = store[best]


def u(y, m=None):
    return y / (np.abs(y[m] if m is not None else y).max() + 1e-12)


fig = plt.figure(figsize=(13, 9))
gs = fig.add_gridspec(2, 3, height_ratios=[1.3, 1], width_ratios=[2.2, 1, 1])
a0 = fig.add_subplot(gs[0, :])
yb = u(fp, FP_MASK); yr = u(ref_fp, FP_MASK)
yb[~FP_MASK] = np.nan
a0.plot(FP, yr + 1.1, color="#b03a2e", lw=1.3, label="EV Acacia on CaF2 (mean of s4, s5, s8, s9, s10)")
a0.plot(FP, yb, color="#1f4e79", lw=1.3, label=f"{best} (AgSiNW map)")
for c in TARGETS:
    a0.axvspan(c - 10, c + 10, color="#f4d03f", alpha=0.35)
    a0.text(c, 2.2, f"{c}", ha="center", fontsize=10, fontweight="bold")
a0.set_xlim(400, 1800); a0.set_ylim(-0.15, 2.35)
a0.set_xlabel("Raman shift (cm$^{-1}$)"); a0.set_ylabel("Normalised intensity (offset)")
a0.set_title(f"Best match with 1456 & 1726 cm$^{{-1}}$ peaks: {best}  (X={b['X_um']} µm, Y={b['Y_um']} µm)\n"
             f"match score {b['match_score']:.3f}  |  r(1456/1726 windows) {b['r_1456_1726_windows']:.3f}  |  "
             f"r(fingerprint) {b['r_fingerprint']:.3f}  |  peak checks passed: 1456 {b['1456_checks_passed']}/10, "
             f"1726 {b['1726_checks_passed']}/10", fontsize=10)
a0.legend(loc="upper left", fontsize=9)
for j, c in enumerate(TARGETS):
    a = fig.add_subplot(gs[1, 1 + j])
    w = (FP >= c - 60) & (FP <= c + 60)
    a.plot(FP[w], u(ref_fp[w]), color="#b03a2e", lw=1.3, label="CaF2")
    a.plot(FP[w], u(fp[w]), color="#1f4e79", lw=1.3, label=best)
    a.axvline(c, color="k", ls=":", lw=0.8)
    a.set_title(f"{c} cm$^{{-1}}$ zoom  (map peak at {b[f'{c}_pos_cm-1']}, SNR {b[f'{c}_SNR']})", fontsize=9)
    a.set_xlabel("Raman shift (cm$^{-1}$)")
    if j == 0:
        a.legend(fontsize=8)
a = fig.add_subplot(gs[1, 0])
top = df.head(12).iloc[::-1]
colors = ["#1f4e79" if s == best else ("#7fa7cf" if q else "#c8c8c8")
          for s, q in zip(top.spectrum, top.both_peaks_10of10)]
a.barh(top.spectrum, top.match_score, color=colors)
for yv, (s, sc, n1, n2) in enumerate(zip(top.spectrum, top.match_score, top["1456_checks_passed"],
                                        top["1726_checks_passed"])):
    a.text(sc + 0.005, yv, f"{sc:.3f}  ({n1}/10, {n2}/10)", va="center", fontsize=7)
a.set_xlim(0, max(top.match_score.max() * 1.35, 0.1))
a.set_xlabel("match score to CaF2")
a.set_title("Top 12 by peak evidence – (1456, 1726 checks passed of 10)\ndark blue = selected best match", fontsize=9)
a.tick_params(axis="y", labelsize=7)
fig.tight_layout()
fig.savefig(f"{OUT}/best_score_match_plot.png", dpi=200, bbox_inches="tight")

# score maps: min checks and match score
maps = sorted(df["map"].unique())
fig, axs = plt.subplots(2, len(maps), figsize=(3.2 * len(maps) + 1, 6.2), constrained_layout=True)
for j, mname in enumerate(maps):
    sub = df[df["map"] == mname]
    xs, ys = np.unique(sub.X_um), np.unique(sub.Y_um)
    for k, (col, cmap, vmin, vmax) in enumerate([("both_peaks_min_checks", "Greens", 0, 10),
                                                  ("match_score", "viridis", df.match_score.min(),
                                                   df.match_score.max())]):
        Z = np.full((len(ys), len(xs)), np.nan)
        for _, r in sub.iterrows():
            Z[np.searchsorted(ys, r.Y_um), np.searchsorted(xs, r.X_um)] = r[col]
        im = axs[k, j].imshow(Z, origin="lower", cmap=cmap, vmin=vmin, vmax=vmax,
                              extent=[xs[0], xs[-1], ys[0], ys[-1]], aspect="auto")
        if mname == b["map"]:
            axs[k, j].plot(b.X_um, b.Y_um, "r*", ms=14)
        axs[k, j].set_title(f"{mname} – {'peak checks (min of 2)' if k == 0 else 'match score'}", fontsize=9)
        if j == len(maps) - 1:
            fig.colorbar(im, ax=axs[k, :], shrink=0.9)
fig.savefig(f"{OUT}/peak_and_score_maps.png", dpi=200, bbox_inches="tight")

pd.DataFrame({"wavenumber_cm-1": FP, f"{best}_raw_despiked": raw_fp, f"{best}_processed": fp,
              "CaF2_mean_processed_norm": ref_fp}).to_csv(f"{OUT}/best_match_{best}.csv", index=False)
print(f"\nBEST: {best}  ({b['map_file']}, pixel {b['pixel']}, X={b['X_um']}, Y={b['Y_um']})")
print(f"Outputs written to {OUT}/")
