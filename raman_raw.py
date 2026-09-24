"""Shared helpers for raw-data Raman peak work (no smoothing, no baseline correction)."""
import re
import os

import numpy as np
import pandas as pd
from scipy.signal import find_peaks, peak_widths

RESOLUTION = 4.5    # cm-1, instrument spectral resolution (LabRAM HR, 1800 gr/mm)
NARROW_MIN = 3.5    # sharp bands measured just under the resolution are kept (flagged) ...
NARROW_SNR = 7.0    # ... only if at least this strong

# Tentative literature assignments: (lo, hi, name[, substrate]) - first match wins.
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
    (1615, 1625, "Tyr / Trp ring C=C stretch"),
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


def short(name):
    return name.split(" / ")[0].split(" (")[0]


# ------------------------------------------------------------------ readers
def read_l6s(path):
    """LabSpec 6 single spectrum: the monotonic float32 run is the axis, the first
    float32 run of the same length that is not monotonic is the intensity."""
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


def tag(path):
    m = re.search(r"-([A-Za-z]+\d+[a-z]?)-", os.path.basename(path))
    return m.group(1) if m else os.path.basename(path)


def clean(path):
    return re.sub(r"^[0-9a-f]{8}-", "", os.path.basename(path))


# ------------------------------------------------------------------ raw peak finding
def noise_sigma(wn, y, lo=400, hi=1800):
    """Robust raw point-to-point noise: 1.4826*MAD(diff)/sqrt2."""
    m = (wn > lo) & (wn < hi)
    d = np.diff(y[m])
    return 1.4826 * np.median(np.abs(d - np.median(d))) / np.sqrt(2)


def raw_peaks(wn, y, substrate, lo, hi, k_sigma=5.0, wlen_cm=150.0, min_dist_cm=5.0, sigma=None):
    """Peaks on raw counts: prominence >= k_sigma*sigma within ±wlen_cm/2, FWHM >= resolution
    (or >= NARROW_MIN with SNR >= NARROW_SNR, flagged). Returns (accepted, rejected)."""
    sigma = noise_sigma(wn, y) if sigma is None else sigma
    step = np.median(np.diff(wn))
    m = (wn >= lo) & (wn <= hi)
    x, yy = wn[m], y[m]
    idx, props = find_peaks(yy, prominence=k_sigma * sigma, wlen=int(wlen_cm / step),
                            distance=max(1, int(min_dist_cm / step)))
    acc, rej = [], []
    if len(idx):
        w = peak_widths(yy, idx, rel_height=0.5,
                        prominence_data=(props["prominences"], props["left_bases"], props["right_bases"]))
        for k, i in enumerate(idx):
            if i < 3 or i > len(x) - 4:
                continue
            fwhm = float(np.interp(w[3][k], np.arange(len(x)), x) - np.interp(w[2][k], np.arange(len(x)), x))
            row = dict(position_cm1=round(float(x[i]), 1), raw_counts=round(float(yy[i]), 1),
                       prominence_counts=round(float(props["prominences"][k]), 1),
                       SNR=round(float(props["prominences"][k] / sigma), 1), FWHM_cm1=round(fwhm, 1),
                       assignment=assign(float(x[i]), substrate), flag="")
            if fwhm < RESOLUTION:
                if fwhm >= NARROW_MIN and row["SNR"] >= NARROW_SNR:
                    row["flag"] = "narrow (near resolution) - verify"
                else:
                    rej.append(row)
                    continue
            acc.append(row)
    cols = ["position_cm1", "raw_counts", "prominence_counts", "SNR", "FWHM_cm1", "assignment", "flag"]
    return pd.DataFrame(acc, columns=cols), pd.DataFrame(rej, columns=cols)


def local_check(wn, y, pos, sigma, win=6.0, flank=(15.0, 35.0), k=3.0):
    """Independent check of one peak on raw data: the raw maximum within ±win of pos must be
    within one data point of pos, and its height above the straight line joining the minima of
    the two flanks [pos-flank1, pos-flank0] and [pos+flank0, pos+flank1] must be >= k*sigma."""
    step = np.median(np.diff(wn))
    m = (wn >= pos - win) & (wn <= pos + win)
    if m.sum() < 3:
        return False, np.nan, 0.0
    xm = wn[m][np.argmax(y[m])]
    lm = (wn >= pos - flank[1]) & (wn <= pos - flank[0])
    rm = (wn >= pos + flank[0]) & (wn <= pos + flank[1])
    if lm.sum() < 3 or rm.sum() < 3:
        return False, float(xm), 0.0
    xl, yl = wn[lm][np.argmin(y[lm])], y[lm].min()
    xr, yr = wn[rm][np.argmin(y[rm])], y[rm].min()
    h = y[m].max() - (yl + (yr - yl) * (xm - xl) / (xr - xl))
    return bool(abs(xm - pos) <= 1.01 * step and h >= k * sigma), round(float(xm), 1), round(float(h / sigma), 1)


# ------------------------------------------------------------------ noise-calibrated local test
CALIB_WINDOWS = [(5.0, (12.0, 30.0)), (8.0, (20.0, 40.0)), (12.0, (35.0, 60.0))]


def calib_score(wn, y, c, sigma):
    """Best local_check height/sigma around c over several window sizes, re-centred on the raw
    maximum it finds (second pass, ±3 cm-1). Returns (position of raw maximum, score)."""
    best = (np.nan, 0.0)
    for win, fl in CALIB_WINDOWS:
        _, xm, _ = local_check(wn, y, c, sigma, win=win, flank=fl, k=-np.inf)
        if np.isnan(xm):
            continue
        _, xm2, s2 = local_check(wn, y, xm, sigma, win=3.0, flank=fl, k=-np.inf)
        if s2 > best[1]:
            best = (xm2, s2)
    return best


def noise_threshold(wn, y, sigma, region=(600.0, 720.0), n=300, pct=99.0, seed=0):
    """pct-th percentile of calib_score at random positions in a band-free region: the score
    that pure noise reaches with the same test."""
    rng = np.random.default_rng(seed)
    return float(np.percentile([calib_score(wn, y, c, sigma)[1] for c in rng.uniform(*region, n)], pct))
