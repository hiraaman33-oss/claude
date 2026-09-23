"""
EV Acacia – Raman map spectrum matching
=======================================
Loads LabRAM map exports (m1..m5, AgSiNW substrate, 532 nm), preprocesses every
pixel spectrum and finds the single spectrum that best matches a reference.

Reference:
  * If REF_FILE is set (2-column txt/csv: wavenumber, intensity – e.g. EV Acacia
    on CaF2), each pixel is scored by correlation to it.
  * Otherwise a band-template of EV marker bands is used, combined with the
    dataset consensus (medoid) spectrum.

Usage:  python3 EV_Acacia_MapMatch.py <map_dir> [reference_file]
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
REF_FILE = sys.argv[2] if len(sys.argv) > 2 else None
OUT = "EV_Acacia_Match"
os.makedirs(OUT, exist_ok=True)

# Common grid: fingerprint + high-wavenumber windows
GRID = np.concatenate([np.arange(400, 1800, 1.0), np.arange(2750, 3100, 1.0)])
# Substrate bands to exclude (Si 1st/2nd order, CaF2 322 is below 400 anyway)
EXCLUDE = [(500, 545), (920, 990)]
MASK = np.ones_like(GRID, bool)
for lo, hi in EXCLUDE:
    MASK &= ~((GRID >= lo) & (GRID <= hi))

# EV marker bands (cm-1) – proteins, lipids, nucleic acids
EV_BANDS = {
    "Phe ring breathing": 1003, "Phe / C-H in-plane": 1031,
    "PO2- / C-C lipid": 1087, "C-N / C-C stretch": 1127,
    "Amide III": 1250, "CH2 twist (lipid)": 1300,
    "CH2/CH3 deformation": 1445, "Amide I / C=C": 1655,
    "CH2 sym stretch (lipid)": 2850, "CH2 asym stretch": 2885,
    "CH3 stretch (protein)": 2935,
}


def load_map(path):
    d = np.genfromtxt(path, comments="#", delimiter="\t", encoding="latin1")
    wn = d[0, 2:]
    xy = d[1:, :2]
    spec = d[1:, 2:]
    order = np.argsort(wn)
    wn, spec = wn[order], spec[:, order]
    wn, idx = np.unique(wn, return_index=True)  # stitched-window overlaps
    return wn, xy, spec[:, idx]


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
        W = sparse.spdiags(w, 0, L, L)
        z = spsolve((W + DD).tocsc(), w * y)
        w = p * (y > z) + (1 - p) * (y < z)
    return z


def preprocess(wn, y):
    y = despike(y)
    out = np.empty_like(GRID)
    for lo, hi in [(400, 1800), (2750, 3100)]:
        g = (GRID >= lo) & (GRID < hi)
        s = (wn >= lo - 20) & (wn <= hi + 20)
        yi = np.interp(GRID[g], wn[s], y[s])
        yi = savgol_filter(yi, 11, 3)
        out[g] = yi - als_baseline(yi)
    return out


def raw_on_grid(wn, y):
    return np.interp(GRID, wn, despike(y))


def vnorm(a):
    a = a * MASK
    return a / (np.linalg.norm(a, axis=-1, keepdims=True) + 1e-12)


def band_snr(y):
    """Signal at each EV band relative to the local noise of the processed spectrum."""
    noise = np.std(np.diff(y[(GRID > 1800 - 150) & (GRID < 1800)])) / np.sqrt(2) + 1e-9
    out = {}
    for name, c in EV_BANDS.items():
        w = (GRID >= c - 8) & (GRID <= c + 8)
        out[name] = y[w].max() / noise
    return out


def load_reference(path):
    d = pd.read_csv(path, sep=None, engine="python", comment="#", header=None).apply(
        pd.to_numeric, errors="coerce").dropna().values
    wn, y = d[:, 0], d[:, 1]
    o = np.argsort(wn)
    return preprocess(wn[o], y[o])


# ---------------------------------------------------------------- load all maps
rows, proc, raw = [], [], []
for f in sorted(glob.glob(os.path.join(MAP_DIR, "*.txt"))):
    m = re.search(r"-(m\d+)-", os.path.basename(f))
    mname = m.group(1) if m else os.path.basename(f)
    wn, xy, spec = load_map(f)
    for i, (p, y) in enumerate(zip(xy, spec)):
        rows.append(dict(map=mname, pixel=i, X_um=p[0], Y_um=p[1], file=os.path.basename(f)))
        proc.append(preprocess(wn, y))
        raw.append(raw_on_grid(wn, y))
info = pd.DataFrame(rows)
P, R = np.array(proc), np.array(raw)
N = vnorm(P)
print(f"Loaded {len(info)} spectra from {info['map'].nunique()} maps")

# ---------------------------------------------------------------- scoring
corr_all = N @ N.T
info["consensus_r"] = (corr_all.sum(1) - 1) / (len(N) - 1)  # mean r to all others
snr = pd.DataFrame([band_snr(y) for y in P])
info["EV_bands_detected"] = (snr > 5).sum(axis=1)
info["EV_band_SNR_mean"] = snr.mean(axis=1).round(2)

if REF_FILE:
    ref = load_reference(REF_FILE)
    rn = vnorm(ref[None])[0]
    info["ref_r"] = N @ rn
    dref = vnorm(np.gradient(ref)[None])[0]
    info["ref_deriv_r"] = vnorm(np.gradient(P, axis=1)) @ dref
    info["score"] = 0.6 * info["ref_r"] + 0.4 * info["ref_deriv_r"]
    ref_label = f"Reference: {os.path.basename(REF_FILE)}"
else:
    ref = None
    # rank-combine: representativeness + EV band evidence
    info["score"] = (info["consensus_r"].rank(pct=True) + info["EV_bands_detected"].rank(pct=True)
                     + info["EV_band_SNR_mean"].rank(pct=True)) / 3
    ref_label = "No reference supplied – EV band template + consensus"

info = pd.concat([info, snr.round(1).add_prefix("SNR ")], axis=1)
rank = info.sort_values("score", ascending=False)
best = rank.index[0]
b = info.loc[best]
print(ref_label)
print(rank[["map", "pixel", "X_um", "Y_um", "score", "consensus_r",
            "EV_bands_detected", "EV_band_SNR_mean"]].head(10).to_string())

# ---------------------------------------------------------------- outputs
rank.to_csv(f"{OUT}/ranking_all_spectra.csv", index=False)
pd.DataFrame({"wavenumber_cm-1": GRID, "raw_despiked": R[best], "processed": P[best]}).to_csv(
    f"{OUT}/best_match_{b['map']}_px{int(b['pixel'])}.csv", index=False)

fig, ax = plt.subplots(2, 1, figsize=(11, 8), gridspec_kw=dict(height_ratios=[3, 2]))
for seg in [(400, 1800), (2750, 3100)]:
    g = (GRID >= seg[0]) & (GRID < seg[1])
    ax[0].plot(GRID[g], N[best][g] if ref is None else P[best][g] / np.abs(P[best]).max(),
               color="#1f4e79", lw=1.2, label=None)
    if ref is not None:
        ax[0].plot(GRID[g], ref[g] / np.abs(ref).max() + 0.6, color="#b03a2e", lw=1.2)
    for k in rank.index[1:5]:
        ax[0].plot(GRID[g], N[k][g] - 0.08, color="0.7", lw=0.6)
for name, c in EV_BANDS.items():
    ax[0].axvline(c, color="#999", ls=":", lw=0.7)
    ax[0].text(c, ax[0].get_ylim()[1], f"{c}", rotation=90, fontsize=7, va="top", ha="right")
for lo, hi in EXCLUDE:
    ax[0].axvspan(lo, hi, color="#eee")
ax[0].set_title(f"Best match: {b['map']} pixel {int(b['pixel'])} (X={b['X_um']:.2f}, Y={b['Y_um']:.2f} µm)"
                f" — {ref_label}", fontsize=10)
ax[0].set_xlabel("Raman shift (cm$^{-1}$)"); ax[0].set_ylabel("Normalised intensity")
ax[0].plot([], [], color="#1f4e79", label="best match")
if ref is not None:
    ax[0].plot([], [], color="#b03a2e", label="reference (offset)")
ax[0].plot([], [], color="0.7", label="ranks 2–5 (offset)")
ax[0].legend(fontsize=8)

maps = sorted(info["map"].unique())
for j, mname in enumerate(maps):
    sub = info[info["map"] == mname]
    xs, ys = np.unique(sub["X_um"]), np.unique(sub["Y_um"])
    Z = np.full((len(ys), len(xs)), np.nan)
    for _, r in sub.iterrows():
        Z[np.searchsorted(ys, r["Y_um"]), np.searchsorted(xs, r["X_um"])] = r["score"]
    a = fig.add_axes([0.06 + j * 0.175, 0.07, 0.14, 0.25])
    im = a.imshow(Z, origin="lower", cmap="viridis", vmin=info["score"].min(), vmax=info["score"].max(),
                  extent=[xs[0], xs[-1], ys[0], ys[-1]], aspect="auto")
    if mname == b["map"]:
        a.plot(b["X_um"], b["Y_um"], "r*", ms=14)
    a.set_title(mname, fontsize=9); a.tick_params(labelsize=7)
ax[1].axis("off")
fig.colorbar(im, cax=fig.add_axes([0.94, 0.07, 0.012, 0.25]), label="match score")
fig.savefig(f"{OUT}/best_match_overlay_and_scoremaps.png", dpi=200, bbox_inches="tight")

fig, a = plt.subplots(figsize=(11, 4))
mean = N.mean(0); sd = N.std(0)
for seg in [(400, 1800), (2750, 3100)]:
    g = (GRID >= seg[0]) & (GRID < seg[1])
    a.fill_between(GRID[g], (mean - sd)[g], (mean + sd)[g], color="#cfe2f3")
    a.plot(GRID[g], mean[g], color="#1f4e79", lw=1, label=None)
    a.plot(GRID[g], N[best][g], color="#b03a2e", lw=1)
a.set_title("All 100 spectra: mean ± SD (blue) vs best match (red)")
a.set_xlabel("Raman shift (cm$^{-1}$)")
fig.savefig(f"{OUT}/all_spectra_mean_vs_best.png", dpi=200, bbox_inches="tight")
print(f"Outputs written to {OUT}/")
