"""
EV Acacia – match Raman map pixels (AgSiNW) to reference spectra on CaF2
=======================================================================
1. Reads the CaF2 reference spectra (LabSpec 6 .l6s binary, or 2-col txt/csv).
2. Reads all LabRAM map exports (*.txt, m1..m5).
3. Applies identical preprocessing (despike, SG smooth, ALS baseline, mask
   substrate bands, vector normalisation) to everything.
4. Scores every map pixel against every reference and against the mean CaF2
   reference (Pearson r of spectra + Pearson r of 1st derivatives).

Usage:  python3 EV_Acacia_CaF2_Match.py <map_dir> <reference_dir>
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
from scipy.signal import medfilt, savgol_filter
from scipy.sparse.linalg import spsolve

MAP_DIR = sys.argv[1] if len(sys.argv) > 1 else "."
REF_DIR = sys.argv[2] if len(sys.argv) > 2 else MAP_DIR
OUT = "EV_Acacia_CaF2_Match"
os.makedirs(OUT, exist_ok=True)

SEGMENTS = [(400, 1800), (2750, 3100)]
GRID = np.concatenate([np.arange(lo, hi, 1.0) for lo, hi in SEGMENTS])
# Si substrate (1st / 2nd order) in the maps; CaF2 322 cm-1 lies below 400
EXCLUDE = [(500, 545), (920, 990)]
MASK = np.ones_like(GRID, bool)
for lo, hi in EXCLUDE:
    MASK &= ~((GRID >= lo) & (GRID <= hi))

EV_BANDS = {1003: "Phe", 1250: "Amide III", 1300: "CH2 twist", 1445: "CH2/CH3 def",
            1655: "Amide I", 2850: "CH2 sym", 2885: "CH2 asym", 2935: "CH3"}


# ------------------------------------------------------------------ readers
def read_l6s(path):
    """LabSpec 6 single spectrum: first float32 run of N points is intensity,
    the monotonic float32 run of the same length is the Raman-shift axis."""
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
    axis = [r for o, r in runs if np.all(np.diff(r) > 0) and 50 < r[0] < 5000]
    wn = axis[0]
    inten = [r for o, r in sorted(runs, key=lambda t: t[0]) if len(r) == len(wn) and not np.all(np.diff(r) > 0)]
    return wn, inten[0]


def read_two_col(path):
    d = pd.read_csv(path, sep=None, engine="python", comment="#", header=None).apply(
        pd.to_numeric, errors="coerce").dropna().values
    return d[:, 0], d[:, 1]


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


def als_baseline(y, lam=1e5, p=0.01, niter=10):
    L = len(y)
    D = sparse.diags([1.0, -2.0, 1.0], [0, -1, -2], shape=(L, L - 2))
    DD = lam * D.dot(D.T)
    w = np.ones(L)
    for _ in range(niter):
        z = spsolve((sparse.spdiags(w, 0, L, L) + DD).tocsc(), w * y)
        w = p * (y > z) + (1 - p) * (y < z)
    return z


def rubberband(x, y):
    """Lower convex-hull baseline (robust under strong, wide bands)."""
    from scipy.spatial import ConvexHull
    pts = np.column_stack([x, y])
    v = ConvexHull(pts).vertices
    v = np.roll(v, -v.argmin())
    v = v[:v.argmax() + 1]
    return np.interp(x, x[v], y[v])


def preprocess(wn, y):
    o = np.argsort(wn)
    wn, y = wn[o], despike(np.asarray(y, float)[o])
    out = np.empty_like(GRID)
    for lo, hi in SEGMENTS:
        g = (GRID >= lo) & (GRID < hi)
        yi = savgol_filter(np.interp(GRID[g], wn, y), 11, 3)
        yi = yi - rubberband(GRID[g], yi)
        out[g] = yi - als_baseline(yi, lam=1e7, p=0.001)  # remove residual curvature
    return out


SEG_ID = np.select([(GRID >= lo) & (GRID < hi) for lo, hi in SEGMENTS], range(len(SEGMENTS)))[MASK]


def zn(a):
    """Mask substrate bands, then mean-centre and unit-normalise each spectral
    region separately (fingerprint and CH region weighted equally); the dot
    product of two outputs is the average Pearson r over the regions."""
    a = np.atleast_2d(a)[:, MASK].astype(float)
    out = np.empty_like(a)
    for s in range(len(SEGMENTS)):
        c = a[:, SEG_ID == s] - a[:, SEG_ID == s].mean(1, keepdims=True)
        out[:, SEG_ID == s] = c / (np.linalg.norm(c, axis=1, keepdims=True) + 1e-12) / np.sqrt(len(SEGMENTS))
    return out


def dzn(a):
    return zn(np.gradient(np.atleast_2d(a), axis=1))


def unit(y):
    return y / (np.abs(y[MASK]).max() + 1e-12)


# ------------------------------------------------------------------ load references
ref_files = sorted(glob.glob(os.path.join(REF_DIR, "*.l6s")) + glob.glob(os.path.join(REF_DIR, "*.csv")),
                   key=lambda f: int(re.search(r"-s(\d+)-", f).group(1)) if re.search(r"-s(\d+)-", f) else 0)
refs, ref_names = [], []
for f in ref_files:
    wn, y = read_l6s(f) if f.endswith(".l6s") else read_two_col(f)
    m = re.search(r"-(s\d+)-", os.path.basename(f))
    name = m.group(1) if m else os.path.basename(f)
    ref_names.append(name)
    refs.append(preprocess(wn, y))
    pd.DataFrame({"wavenumber_cm-1": wn, "intensity_raw": y}).to_csv(f"{OUT}/ref_{name}_raw.csv", index=False)
refs = np.array(refs)
ref_mean = np.mean([unit(r) for r in refs], axis=0)
Zr = zn(refs)
ref_rr = pd.DataFrame(Zr @ Zr.T, index=ref_names, columns=ref_names).round(3)
print("CaF2 references:", ref_names)
print("Reference-to-reference Pearson r:\n", ref_rr)
ref_rr.to_csv(f"{OUT}/reference_correlation.csv")

# ------------------------------------------------------------------ load maps
rows, P = [], []
for f in sorted(glob.glob(os.path.join(MAP_DIR, "*.txt"))):
    base = os.path.basename(f)
    m = re.search(r"-(m\d+)-", base)
    mname = m.group(1) if m else base
    clean = re.sub(r"^[0-9a-f]{8}-", "", base)  # strip upload prefix
    wn, xy, spec = load_map(f)
    for i, (p, y) in enumerate(zip(xy, spec)):
        rows.append(dict(spectrum=f"{mname}_px{i:02d}", map=mname, pixel=i, X_um=round(p[0], 3),
                         Y_um=round(p[1], 3), map_file=clean))
        P.append(preprocess(wn, y))
info = pd.DataFrame(rows)
P = np.array(P)
Zp, dZp = zn(P), dzn(P)
print(f"Loaded {len(info)} map spectra from {info['map'].nunique()} maps")

# ------------------------------------------------------------------ scoring
for k, n in enumerate(ref_names):
    info[f"r_{n}"] = Zp @ Zr[k]
info["r_meanCaF2"] = Zp @ zn(ref_mean)[0]
info["dr_meanCaF2"] = dZp @ dzn(ref_mean)[0]
rcols = [f"r_{n}" for n in ref_names]
info["r_best_single_ref"] = info[rcols].max(axis=1)
info["best_single_ref"] = info[rcols].idxmax(axis=1).str[2:]
info["r_avg_over_refs"] = info[rcols].mean(axis=1)

# Band-profile similarity: relative intensities at the bands present in the CaF2 spectra
CAF2_BANDS = [850, 1003, 1080, 1125, 1300, 1450, 1655, 1740, 2850, 2885, 2935]


def band_profile(y):
    v = np.array([y[(GRID >= c - 10) & (GRID <= c + 10)].max() for c in CAF2_BANDS])
    return np.clip(v, 0, None) / (np.linalg.norm(np.clip(v, 0, None)) + 1e-12)


bp_ref = band_profile(ref_mean)
info["band_cos_meanCaF2"] = np.array([band_profile(y) for y in P]) @ bp_ref
# ranking score: whole-spectrum shape (band-profile cosine kept as supporting column)
info["score"] = 0.6 * info["r_meanCaF2"] + 0.4 * info["dr_meanCaF2"]
info = info.round(4)
rank = info.sort_values("score", ascending=False).reset_index()
rank.insert(0, "rank", np.arange(1, len(rank) + 1))
rank.drop(columns="index").to_csv(f"{OUT}/ranking_vs_CaF2.csv", index=False)

per_ref = pd.DataFrame([dict(reference=n, best_map_spectrum=info.loc[info[f"r_{n}"].idxmax(), "spectrum"],
                             r=info[f"r_{n}"].max()) for n in ref_names])
per_ref.to_csv(f"{OUT}/best_match_per_reference.csv", index=False)
per_map = info.loc[info.groupby("map")["score"].idxmax(), ["map", "spectrum", "X_um", "Y_um", "score",
                                                             "r_meanCaF2", "dr_meanCaF2", "band_cos_meanCaF2"]]
per_map.to_csv(f"{OUT}/best_match_per_map.csv", index=False)

bi = rank.loc[0, "index"]
b = info.loc[bi]
print("\nTop 10 vs mean CaF2 reference:")
print(rank[["rank", "spectrum", "X_um", "Y_um", "score", "r_meanCaF2", "dr_meanCaF2", "band_cos_meanCaF2",
            "r_best_single_ref", "best_single_ref"]].head(10).to_string(index=False))
print("\nBest map spectrum for each CaF2 reference:\n", per_ref.to_string(index=False))
print("\nBest spectrum in each map:\n", per_map.to_string(index=False))

pd.DataFrame({"wavenumber_cm-1": GRID, f"{b['spectrum']}_processed": P[bi],
              "meanCaF2_processed_norm": ref_mean}).to_csv(f"{OUT}/best_match_{b['spectrum']}.csv", index=False)


# ------------------------------------------------------------------ figures
def seg_plot(ax, y, off=0.0, **kw):
    kw.setdefault("color", ax._get_lines.get_next_color())
    for j, (lo, hi) in enumerate(SEGMENTS):
        g = (GRID >= lo) & (GRID < hi)
        yy = y.copy().astype(float)
        yy[~MASK] = np.nan
        ax.plot(GRID[g], yy[g] + off, **(kw if j == 0 else {k: v for k, v in kw.items() if k != "label"}))


def mark_bands(ax):
    for c, n in EV_BANDS.items():
        ax.axvline(c, color="#999", ls=":", lw=0.7)
        ax.text(c, 1.0, f"{c}", transform=ax.get_xaxis_transform(), rotation=90, fontsize=7, va="top", ha="right")


# Fig 1 – all CaF2 reference spectra
fig, ax = plt.subplots(figsize=(11, 6))
for k, n in enumerate(ref_names):
    seg_plot(ax, unit(refs[k]), off=1.1 * k, lw=1, label=n)
seg_plot(ax, unit(ref_mean), off=1.1 * len(ref_names), color="k", lw=1.4, label="mean CaF2")
mark_bands(ax)
ax.set_title("EV Acacia on CaF2 – all reference spectra (processed, offset)")
ax.set_xlabel("Raman shift (cm$^{-1}$)"); ax.set_ylabel("Normalised intensity (offset)")
ax.legend(fontsize=8, loc="upper right")
fig.savefig(f"{OUT}/01_CaF2_reference_spectra.png", dpi=200, bbox_inches="tight")

# Fig 2 – best match overlay + top 2-5
fig, ax = plt.subplots(figsize=(11, 6))
seg_plot(ax, unit(ref_mean), off=1.2, color="#b03a2e", lw=1.3, label="EV Acacia on CaF2 (mean of refs)")
seg_plot(ax, unit(P[bi]), color="#1f4e79", lw=1.3,
         label=f"Best: {b['spectrum']} (r={b['r_meanCaF2']:.3f}, deriv r={b['dr_meanCaF2']:.3f})")
for k in range(1, 5):
    idx = rank.loc[k, "index"]
    seg_plot(ax, unit(P[idx]), off=-1.2 - 0.0 * k, color="0.6", lw=0.6,
             label="ranks 2–5 (offset)" if k == 1 else None)
mark_bands(ax)
ax.set_title(f"Best map spectrum vs EV Acacia on CaF2: {b['spectrum']}  (X={b['X_um']}, Y={b['Y_um']} µm)")
ax.set_xlabel("Raman shift (cm$^{-1}$)"); ax.set_ylabel("Normalised intensity (offset)")
ax.legend(fontsize=8, loc="upper right")
fig.savefig(f"{OUT}/02_best_match_overlay.png", dpi=200, bbox_inches="tight")

# Fig 3 – score maps
maps = sorted(info["map"].unique())
fig, axs = plt.subplots(1, len(maps), figsize=(3 * len(maps) + 1, 3.2), constrained_layout=True)
for a, mname in zip(np.atleast_1d(axs), maps):
    sub = info[info["map"] == mname]
    xs, ys = np.unique(sub["X_um"]), np.unique(sub["Y_um"])
    Z = np.full((len(ys), len(xs)), np.nan)
    for _, r in sub.iterrows():
        Z[np.searchsorted(ys, r["Y_um"]), np.searchsorted(xs, r["X_um"])] = r["score"]
    im = a.imshow(Z, origin="lower", cmap="viridis", vmin=info["score"].min(), vmax=info["score"].max(),
                  extent=[xs[0], xs[-1], ys[0], ys[-1]], aspect="auto")
    if mname == b["map"]:
        a.plot(b["X_um"], b["Y_um"], "r*", ms=14)
    a.set_title(mname); a.set_xlabel("X (µm)")
np.atleast_1d(axs)[0].set_ylabel("Y (µm)")
fig.colorbar(im, ax=axs, label="match score vs CaF2")
fig.savefig(f"{OUT}/03_score_maps.png", dpi=200, bbox_inches="tight")

# Fig 4 – correlation heat-map pixels x references
fig, a = plt.subplots(figsize=(6, 12))
H = rank[rcols + ["r_meanCaF2"]].values
im = a.imshow(H, aspect="auto", cmap="magma")
a.set_xticks(range(H.shape[1])); a.set_xticklabels(ref_names + ["mean"], rotation=45)
a.set_yticks(range(0, len(rank), 5)); a.set_yticklabels(rank["spectrum"].iloc[::5], fontsize=6)
a.set_title("Pearson r: map spectra (ranked) vs CaF2 refs")
fig.colorbar(im, ax=a)
fig.savefig(f"{OUT}/04_correlation_matrix.png", dpi=200, bbox_inches="tight")
print(f"\nOutputs written to {OUT}/")
