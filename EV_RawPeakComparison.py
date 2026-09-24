"""
Raw-data peak comparison (no smoothing, no baseline correction, no normalisation).

Groups
  A_CaF2   : EV Acacia on CaF2 (.l6s)          A_SERS : EV Acacia on AgSiNW (maps + single spectra)
  N_CaF2   : EV nano Acacia on CaF2 (.txt)     N_SERS : EV nano Acacia on AgSiNW (maps + single spectra)

Peak finding on the raw counts: scipy find_peaks with prominence >= K_SIGMA * sigma,
where sigma is the raw point-to-point noise (1.4826*MAD(diff)/sqrt2) of that spectrum,
prominence window ±WLEN_CM cm-1, minimum spacing MIN_DIST_CM cm-1. For each peak:
  position     raw maximum (cm-1) – sampling ±half the point spacing
  intensity    raw counts at the maximum
  prominence   height above the higher of the two flanking minima (counts)  -> SNR = prom/sigma
  FWHM         width at half prominence, linear interpolation between raw points
A real Raman band cannot be narrower than the 4.5 cm-1 instrument resolution (about 10
data points). Maxima with FWHM < 3.5 cm-1, or 3.5-4.5 cm-1 with SNR < 7, are noise excursions
or cosmic spikes: they are rejected and listed in rejected_narrow_maxima.csv. Maxima of
3.5-4.5 cm-1 with SNR >= 7 are kept and flagged "narrow (near resolution) - verify".
Shoulders that are not separate maxima in the raw data cannot be found this way.

Best-spectrum choice (all on raw data)
  CaF2 groups : the replicate with the highest minimum SNR over its group's shared bands
  SERS groups : the spectrum that reproduces the largest fraction of the chosen CaF2
                spectrum's bands that lie inside its measured range (a raw peak within
                ±MATCH_CM), ties broken by more bands matched, fewer extra peaks, then by
                total SNR of the matched peaks.

Usage:  python3 EV_RawPeakComparison.py <acacia_caf2> <acacia_maps> <acacia_single>
                                        <nano_caf2> <nano_maps> <nano_single>
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
from scipy.signal import find_peaks, peak_widths

A_CAF2, A_MAPS, A_SINGLE, N_CAF2, N_MAPS, N_SINGLE = sys.argv[1:7]
OUT = "EV_RawPeakComparison"
os.makedirs(OUT, exist_ok=True)
K_SIGMA = 5.0
WLEN_CM = 150.0
MIN_DIST_CM = 5.0
MATCH_CM = 8.0
RESOLUTION = 4.5
NARROW_MIN = 3.5   # sharp real bands measured just under the resolution (noise at half height) ...
NARROW_SNR = 7.0   # ... are kept, flagged, only if they are this strong
SEGS = [(300, 1800), (2750, 3100)]

# Tentative literature assignments (range lo, hi, name). First match wins.
ASSIGN = [
    (315, 330, "CaF2 lattice mode (substrate)", "CaF2"),
    (510, 525, "Si optical phonon (substrate)", "SERS"),
    (615, 625, "Phe C-C twist"),
    (638, 648, "Tyr C-C twist"),
    (755, 765, "Trp ring breathing"),
    (775, 790, "Nucleic acid ring breathing / O-P-O"),
    (820, 835, "Tyr (Fermi doublet) / O-P-O stretch"),
    (840, 860, "Tyr ring breathing / C-C stretch (proline, polysaccharide C-O-C)"),
    (870, 882, "C-C-N+ (choline, lipid) / hydroxyproline"),
    (895, 910, "C-C stretch / C-O-C (saccharide)"),
    (920, 945, "C-C stretch protein backbone (alpha-helix)"),
    (945, 990, "Si 2nd-order phonon (substrate)", "SERS"),
    (998, 1008, "Phe ring breathing"),
    (1028, 1036, "Phe C-H in-plane bend"),
    (1055, 1070, "C-C stretch (lipid, trans chains)"),
    (1075, 1100, "PO2- symmetric stretch / C-C (gauche lipid) / C-O (carbohydrate)"),
    (1115, 1135, "C-C stretch (lipid, trans) / C-N stretch (protein)"),
    (1150, 1162, "C-C / C-N stretch (protein) / carotenoid C-C"),
    (1165, 1180, "Tyr C-H bend"),
    (1200, 1215, "Tyr / Phe C-C6H5 stretch"),
    (1230, 1290, "Amide III"),
    (1290, 1310, "CH2 twist (lipid)"),
    (1330, 1350, "CH3CH2 wagging (protein, nucleic acids)"),
    (1350, 1400, "COO- symmetric stretch / CH3 bend / carbon D band", "SERS"),
    (1350, 1400, "COO- symmetric stretch / CH3 bend"),
    (1435, 1475, "CH2/CH3 deformation (scissoring)"),
    (1515, 1530, "Carotenoid C=C stretch"),
    (1540, 1575, "Amide II / Trp"),
    (1575, 1595, "COO- asymmetric stretch / carbon G band / Phe", "SERS"),
    (1575, 1595, "COO- asymmetric stretch / Phe"),
    (1595, 1615, "Phe / Tyr ring C=C stretch / carbon G band", "SERS"),
    (1595, 1615, "Phe / Tyr ring C=C stretch"),
    (1640, 1675, "Amide I / C=C stretch (lipid)"),
    (1710, 1750, "C=O stretch (ester, lipids)"),
    (2840, 2860, "CH2 symmetric stretch (lipid)"),
    (2870, 2900, "CH2 asymmetric stretch (lipid)"),
    (2920, 2950, "CH3 symmetric stretch (protein) / CH stretch"),
    (2955, 2990, "CH3 asymmetric stretch"),
    (3000, 3020, "=C-H stretch (unsaturated lipid)"),
    (3050, 3070, "Aromatic C-H stretch"),
]


def assign(pos, substrate):
    """substrate: 'CaF2' or 'SERS'; entries with a substrate tag only apply there."""
    for e in ASSIGN:
        lo, hi, name = e[:3]
        only = e[3] if len(e) > 3 else None
        if lo <= pos <= hi and (only is None or only == substrate):
            return name
    return "unassigned"


REJECTED = []


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


def read_single(path):
    if path.endswith(".l6s"):
        wn, y = read_l6s(path)
    else:
        d = np.loadtxt(path)
        wn, y = d[:, 0], d[:, 1]
    o = np.argsort(wn)
    wn, y = wn[o], y[o]
    wn, idx = np.unique(wn, return_index=True)
    return wn, y[idx]


def load_map(path):
    rows = [l.rstrip("\n\r").split("\t") for l in open(path, encoding="latin1")
            if not l.startswith("#") and l.strip()]
    wn = np.array([float(v) for v in rows[0] if v.strip()])
    data = np.array([[float(v) for v in r if v.strip()] for r in rows[1:]])
    o = np.argsort(wn)
    wn, spec = wn[o], data[:, 2:][:, o]
    wn, idx = np.unique(wn, return_index=True)
    return wn, data[:, :2], spec[:, idx]


def tag(path):
    m = re.search(r"-([A-Za-z]+\d+[a-z]?)-", os.path.basename(path))
    return m.group(1) if m else os.path.basename(path)


def clean(path):
    return re.sub(r"^[0-9a-f]{8}-", "", os.path.basename(path))


def collect_singles(d):
    out = {}
    for f in sorted(glob.glob(os.path.join(d, "*"))):
        n = tag(f)
        if n in out and not f.endswith(".l6s"):
            continue  # identical .txt/.l6s pair: keep the .l6s
        out[n] = dict(file=clean(f), wn_y=read_single(f), label=n)
    return out


def collect_maps(d):
    out = {}
    for f in sorted(glob.glob(os.path.join(d, "*.txt"))):
        m = tag(f)
        wn, xy, spec = load_map(f)
        for i, (p, y) in enumerate(zip(xy, spec)):
            out[f"{m}_px{i:02d}"] = dict(file=clean(f), wn_y=(wn, y),
                                         label=f"{m} spectrum #{i + 1} (X={p[0]:.2f}, Y={p[1]:.2f} µm)")
    return out


# ------------------------------------------------------------------ raw peak finding
def noise_sigma(wn, y):
    m = (wn > 400) & (wn < 1800) if wn.max() > 1000 else np.ones_like(wn, bool)
    d = np.diff(y[m])
    return 1.4826 * np.median(np.abs(d - np.median(d))) / np.sqrt(2)


def raw_peaks(wn, y, substrate, who=""):
    sigma = noise_sigma(wn, y)
    step = np.median(np.diff(wn))
    rows = []
    for lo, hi in SEGS:
        m = (wn >= lo) & (wn <= hi)
        if m.sum() < 50:
            continue
        x, yy = wn[m], y[m]
        idx, props = find_peaks(yy, prominence=K_SIGMA * sigma, wlen=int(WLEN_CM / step),
                                distance=max(1, int(MIN_DIST_CM / step)))
        if len(idx) == 0:
            continue
        w = peak_widths(yy, idx, rel_height=0.5, prominence_data=(props["prominences"],
                                                                 props["left_bases"], props["right_bases"]))
        for k, i in enumerate(idx):
            if i < 3 or i > len(x) - 4:  # edge of the measured window: not a resolved peak
                continue
            fwhm = float(np.interp(w[3][k], np.arange(len(x)), x) - np.interp(w[2][k], np.arange(len(x)), x))
            row = dict(position_cm1=round(float(x[i]), 1), raw_counts=round(float(yy[i]), 1),
                       prominence_counts=round(float(props["prominences"][k]), 1),
                       SNR=round(float(props["prominences"][k] / sigma), 1), FWHM_cm1=round(fwhm, 1),
                       assignment=assign(float(x[i]), substrate))
            row["flag"] = ""
            if fwhm < RESOLUTION:
                if fwhm >= NARROW_MIN and row["SNR"] >= NARROW_SNR:
                    row["flag"] = "narrow (near resolution) - verify"
                else:
                    if who:
                        REJECTED.append(dict(spectrum=who, **row))
                    continue
            rows.append(row)
    return pd.DataFrame(rows), sigma, step


def covered(wn, c):
    return wn.min() + 10 < c < wn.max() - 10


def match(ref_df, cand_df, wn):
    """Reference bands reproduced by a candidate within ±MATCH_CM (only bands inside its range)."""
    if cand_df.empty:
        return 0, 0, 0.0, 0
    pos = cand_df.position_cm1.values
    matched, snr_sum, used = 0, 0.0, set()
    testable = 0
    for c in ref_df.position_cm1.values:
        if not covered(wn, c):
            continue
        testable += 1
        d = np.abs(pos - c)
        j = int(np.argmin(d))
        if d[j] <= MATCH_CM:
            matched += 1
            snr_sum += cand_df.SNR.values[j]
            used.add(j)
    return matched, len(pos) - len(used), snr_sum, testable


def best_caf2(group):
    peaks = {n: raw_peaks(*v["wn_y"], "CaF2") for n, v in group.items()}
    if len(group) == 1:
        n = next(iter(group))
        return n, peaks, pd.DataFrame([dict(spectrum=n, note="only spectrum")])
    # bands shared by at least half of the replicates
    allpos = np.sort(np.concatenate([p[0].position_cm1.values for p in peaks.values()]))
    shared, cl = [], []
    for c in allpos:
        if cl and c - cl[-1] > MATCH_CM:
            if len(cl) >= len(group) / 2:
                shared.append(np.median(cl))
            cl = []
        cl.append(c)
    if len(cl) >= len(group) / 2:
        shared.append(np.median(cl))
    rows = []
    for n, (df, sigma, _) in peaks.items():
        snrs = []
        for c in shared:
            d = np.abs(df.position_cm1.values - c) if not df.empty else np.array([99])
            snrs.append(df.SNR.values[np.argmin(d)] if d.min() <= MATCH_CM else 0.0)
        found = [v for v in snrs if v > 0]
        rows.append(dict(spectrum=n, n_bands=len(df), shared_bands=len(shared),
                         shared_bands_found=len(found),
                         mean_SNR_shared=round(float(np.mean(found)) if found else 0.0, 1),
                         noise_sigma=round(sigma, 1)))
    t = pd.DataFrame(rows).sort_values(["shared_bands_found", "mean_SNR_shared"], ascending=False)
    return t.iloc[0].spectrum, peaks, t


def best_sers(group, ref_df):
    rows = []
    for n, v in group.items():
        wn, y = v["wn_y"]
        df, sigma, _ = raw_peaks(wn, y, "SERS")
        m, extra, snr, testable = match(ref_df, df, wn)
        rows.append(dict(spectrum=n, label=v["label"], range=f"{wn.min():.0f}-{wn.max():.0f}",
                         ref_bands_testable=testable, ref_bands_matched=m,
                         extra_peaks=extra, matched_SNR_sum=round(snr, 1)))
    t = pd.DataFrame(rows)
    t["fraction_matched"] = (t.ref_bands_matched / t.ref_bands_testable.clip(lower=1)).round(3)
    t = t.sort_values(["fraction_matched", "ref_bands_matched", "extra_peaks", "matched_SNR_sum"],
                      ascending=[False, False, True, False])
    return t.iloc[0].spectrum, t


# ------------------------------------------------------------------ run
groups = {
    "A_CaF2": {k: v for k, v in collect_singles(A_CAF2).items()},
    "N_CaF2": {k: v for k, v in collect_singles(N_CAF2).items()},
    "A_SERS": {**collect_maps(A_MAPS), **collect_singles(A_SINGLE)},
    "N_SERS": {**collect_maps(N_MAPS), **collect_singles(N_SINGLE)},
}
title = {"A_CaF2": "EV Acacia on CaF2", "N_CaF2": "EV nano Acacia on CaF2",
         "A_SERS": "EV Acacia on AgSiNW", "N_SERS": "EV nano Acacia on AgSiNW"}

chosen, tables = {}, {}
for g in ["A_CaF2", "N_CaF2"]:
    n, peaks, t = best_caf2(groups[g])
    chosen[g] = n
    tables[g] = raw_peaks(*groups[g][n]["wn_y"], "CaF2", who=f"{title[g]} {n}")[0]
    t.to_csv(f"{OUT}/selection_{g}.csv", index=False)
    print(f"\n{title[g]} – replicate ranking:\n{t.to_string(index=False)}\n -> chosen: {n}")
for g, ref in [("A_SERS", "A_CaF2"), ("N_SERS", "N_CaF2")]:
    n, t = best_sers(groups[g], tables[ref])
    chosen[g] = n
    tables[g] = raw_peaks(*groups[g][n]["wn_y"], "SERS", who=f"{title[g]} {n}")[0]
    t.to_csv(f"{OUT}/selection_{g}.csv", index=False)
    print(f"\n{title[g]} – top 5 by bands of {title[ref]} reproduced:\n{t.head(5).to_string(index=False)}"
          f"\n -> chosen: {n}")

# ------------------------------------------------------------------ axis calibration check (raw maxima)
def raw_max(wn, y, lo, hi):
    m = (wn >= lo) & (wn <= hi)
    return float(wn[m][np.argmax(y[m])]) if m.sum() else np.nan


cal = []
for g, (lo, hi, ref, lab) in {"A_CaF2": (310, 335, 321.6, "CaF2"), "N_CaF2": (310, 335, 321.6, "CaF2"),
                              "A_SERS": (505, 535, 520.7, "Si"), "N_SERS": (505, 535, 520.7, "Si")}.items():
    pos = [raw_max(*v["wn_y"], lo, hi) for v in groups[g].values() if v["wn_y"][0].min() < lo]
    if pos:
        cal.append(dict(group=title[g], band=f"{lab} {ref}", n=len(pos), median_raw_max=round(np.median(pos), 1),
                        offset=round(np.median(pos) - ref, 1), spread_IQR=round(float(np.subtract(*np.percentile(pos, [75, 25]))), 1)))
cal = pd.DataFrame(cal)
cal.to_csv(f"{OUT}/axis_calibration_check.csv", index=False)
print("\nAxis calibration check (raw maxima of substrate bands):\n", cal.to_string(index=False))

all_rows = []
for g in groups:
    v = groups[g][chosen[g]]
    df = tables[g].copy()
    df.insert(0, "spectrum", v["label"])
    df.insert(0, "group", title[g])
    df.insert(2, "file", v["file"])
    all_rows.append(df)
    sigma = noise_sigma(*v["wn_y"])
    step = np.median(np.diff(v["wn_y"][0]))
    print(f"\n=== {title[g]}: {v['label']}  ({v['file']})  noise σ = {sigma:.1f} counts, "
          f"point spacing {step:.2f} cm-1 ===")
    print(df.drop(columns=["group", "spectrum", "file"]).to_string(index=False))
peaks_all = pd.concat(all_rows, ignore_index=True)
peaks_all.to_csv(f"{OUT}/raw_peaks_chosen_spectra.csv", index=False)
pd.DataFrame(REJECTED).to_csv(f"{OUT}/rejected_narrow_maxima.csv", index=False)
print(f"\nRejected narrow maxima (FWHM < {RESOLUTION} cm-1, noise/spikes): {len(REJECTED)} "
      f"(see rejected_narrow_maxima.csv)")


# ------------------------------------------------------------------ pairwise band matching
def pair_table(g1, g2):
    a, b = tables[g1], tables[g2]
    wa, wb = groups[g1][chosen[g1]]["wn_y"][0], groups[g2][chosen[g2]]["wn_y"][0]
    rows, used = [], set()
    for _, r in a.iterrows():
        d = np.abs(b.position_cm1.values - r.position_cm1) if not b.empty else np.array([99.0])
        j = int(np.argmin(d))
        if d[j] <= MATCH_CM:
            used.add(j)
            rb = b.iloc[j]
            rows.append({f"{title[g1]} (cm-1)": r.position_cm1, f"{title[g2]} (cm-1)": rb.position_cm1,
                         "shift (cm-1)": round(rb.position_cm1 - r.position_cm1, 1),
                         f"SNR {title[g1]}": r.SNR, f"SNR {title[g2]}": rb.SNR,
                         "status": "in both", "assignment": r.assignment})
        else:
            near = f"; nearest {title[g2]} band at {b.position_cm1.values[j]:.1f} (Δ{d[j]:.1f})" \
                if (not b.empty and d[j] <= 15) else ""
            rows.append({f"{title[g1]} (cm-1)": r.position_cm1, f"{title[g2]} (cm-1)": np.nan, "shift (cm-1)": np.nan,
                         f"SNR {title[g1]}": r.SNR, f"SNR {title[g2]}": np.nan,
                         "status": (f"only in {title[g1]}" if covered(wb, r.position_cm1) else
                                    f"outside {title[g2]} range") + near, "assignment": r.assignment})
    for j, rb in b.reset_index(drop=True).iterrows():
        if j not in used:
            rows.append({f"{title[g1]} (cm-1)": np.nan, f"{title[g2]} (cm-1)": rb.position_cm1, "shift (cm-1)": np.nan,
                         f"SNR {title[g1]}": np.nan, f"SNR {title[g2]}": rb.SNR,
                         "status": f"only in {title[g2]}" if covered(wa, rb.position_cm1) else
                         f"outside {title[g1]} range", "assignment": rb.assignment})
    t = pd.DataFrame(rows)
    key = t[f"{title[g1]} (cm-1)"].fillna(t[f"{title[g2]} (cm-1)"])
    return t.assign(_k=key).sort_values("_k").drop(columns="_k")


pairs = [("A_CaF2", "A_SERS", "1_Acacia_CaF2_vs_AgSiNW"), ("N_CaF2", "N_SERS", "2_nanoAcacia_CaF2_vs_AgSiNW"),
         ("A_CaF2", "N_CaF2", "3a_Acacia_vs_nanoAcacia_CaF2"), ("A_SERS", "N_SERS", "3b_Acacia_vs_nanoAcacia_AgSiNW")]
pd.set_option("display.width", 260)
pd.set_option("display.max_colwidth", 70)
for g1, g2, name in pairs:
    t = pair_table(g1, g2)
    t.to_csv(f"{OUT}/{name}_band_table.csv", index=False)
    print(f"\n##### {title[g1]} vs {title[g2]} #####\n{t.to_string(index=False)}")


# ------------------------------------------------------------------ plots (raw counts only)
COLORS = {"A_CaF2": "#b03a2e", "A_SERS": "#1f4e79", "N_CaF2": "#7b3294", "N_SERS": "#1a9641"}


def short(a):
    return a.split(" / ")[0].split(" (")[0]


def plot_pair(g1, g2, name):
    specs = [g1, g2]
    segs = [s for s in SEGS if any(groups[g][chosen[g]]["wn_y"][0].max() > s[0] + 50 and
                                   groups[g][chosen[g]]["wn_y"][0].min() < s[1] - 50 for g in specs)]
    fig, axs = plt.subplots(2, len(segs), figsize=(22, 11), squeeze=False,
                            gridspec_kw=dict(width_ratios=[s[1] - s[0] for s in segs]))
    for r, g in enumerate(specs):
        wn, y = groups[g][chosen[g]]["wn_y"]
        pk = tables[g]
        for c, (lo, hi) in enumerate(segs):
            a = axs[r, c]
            m = (wn >= lo) & (wn <= hi)
            if m.sum() < 10:
                a.text(0.5, 0.5, "not measured in this range", ha="center", transform=a.transAxes)
                a.set_yticks([])
                continue
            a.plot(wn[m], y[m], color=COLORS[g], lw=0.7)
            ymin, ymax = y[m].min(), y[m].max()
            a.set_ylim(ymin - 0.02 * (ymax - ymin), ymax + 0.45 * (ymax - ymin))
            for _, p in pk[(pk.position_cm1 >= lo) & (pk.position_cm1 <= hi)].iterrows():
                a.plot([p.position_cm1] * 2, [p.raw_counts + 0.02 * (ymax - ymin), p.raw_counts + 0.08 * (ymax - ymin)],
                       color="k", lw=0.7)
                a.text(p.position_cm1, p.raw_counts + 0.09 * (ymax - ymin),
                       f"{p.position_cm1:.1f} {short(p.assignment)}", rotation=90, fontsize=6.5,
                       ha="center", va="bottom")
            a.set_xlim(lo, hi)
            if c == 0:
                a.set_ylabel("raw intensity (counts)")
            if r == 1:
                a.set_xlabel("Raman shift (cm$^{-1}$)")
        axs[r, 0].set_title(f"{title[g]} – {groups[g][chosen[g]]['label']}  [raw data, no processing]",
                            loc="left", fontsize=10, color=COLORS[g])
    fig.suptitle(f"{title[g1]} vs {title[g2]} – peaks: raw maximum, prominence ≥ {K_SIGMA:.0f}σ "
                 f"(position ±{np.median(np.diff(groups[g1][chosen[g1]]['wn_y'][0])) / 2:.2f} cm$^{{-1}}$ sampling, "
                 f"resolution {RESOLUTION} cm$^{{-1}}$)", fontsize=11)
    fig.tight_layout()
    fig.savefig(f"{OUT}/{name}.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


for g1, g2, name in pairs:
    plot_pair(g1, g2, name)
print(f"\nOutputs written to {OUT}/")
