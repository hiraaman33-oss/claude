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
    that pure noise reaches with the same test (windows containing a cosmic ray excluded)."""
    from scipy.signal import medfilt
    rng = np.random.default_rng(seed)
    # cosmic rays in the reference region (points > 5 sigma above the running median of 7 points)
    # are not part of the noise the test must beat: skip positions whose window contains one
    m = (wn >= region[0] - 20) & (wn <= region[1] + 20)
    rays = wn[m][(y[m] - medfilt(y[m], 7)) > 5.0 * sigma]
    maxwin = max(w for w, _ in CALIB_WINDOWS)
    scores = []
    for c in rng.uniform(*region, n):
        if len(rays) and np.min(np.abs(rays - c)) <= maxwin + 1.0:
            continue
        scores.append(calib_score(wn, y, c, sigma)[1])
    return float(np.percentile(scores, pct))


def checked_peaks(wn, y, substrate, lo, hi):
    """Raw peaks plus two independent re-checks (local_check; repeat search with other settings)."""
    sigma = noise_sigma(wn, y)
    acc, rej = raw_peaks(wn, y, substrate, lo, hi, sigma=sigma)
    alt, _ = raw_peaks(wn, y, substrate, lo, hi, k_sigma=4.0, wlen_cm=100.0, min_dist_cm=4.0, sigma=sigma)
    c1, c1pos, c1snr, c2 = [], [], [], []
    for _, p in acc.iterrows():
        f0 = max(15.0, p.FWHM_cm1)
        ok, xm, snr = local_check(wn, y, p.position_cm1, sigma, flank=(f0, f0 + 20.0))
        c1.append(ok)
        c1pos.append(xm)
        c1snr.append(snr)
        c2.append(bool((not alt.empty) and np.min(np.abs(alt.position_cm1.values - p.position_cm1)) <= 2.0))
    acc = acc.assign(check1_local=c1, check1_pos=c1pos, check1_SNR=c1snr, check2_repeat=c2)
    acc["confirmed"] = acc.check1_local & acc.check2_repeat
    return acc, rej, sigma


# ------------------------------------------------------------------ noise-calibrated peak list (replicate-aware)
def spike_like(wn, y, xm, height):
    """A cosmic-ray spike is 1-2 data points wide: its maximum stands far above the mean of the
    neighbouring points (±1-2 points). For a real band (FWHM >= 4.5 cm-1, ~10 points) the
    neighbours are almost as high. Spike-like if the drop exceeds 60 % of the band height, or if
    only 1-2 contiguous points lie above half of the band height."""
    i = int(np.argmin(np.abs(wn - xm)))
    nb = np.r_[y[max(i - 2, 0):i], y[i + 1:i + 3]]
    if height <= 0:
        return False
    if (y[i] - nb.mean()) > 0.6 * height:
        return True
    # also spike-like if only 1-2 contiguous points lie above half height (a band of FWHM >= 3.5 cm-1
    # spans >= ~8 points at 0.44 cm-1 spacing)
    half = y[i] - 0.5 * height
    lo_i = i
    while lo_i - 1 >= 0 and y[lo_i - 1] >= half:
        lo_i -= 1
    hi_i = i
    while hi_i + 1 < len(y) and y[hi_i + 1] >= half:
        hi_i += 1
    return bool(hi_i - lo_i + 1 <= 2)


def visible_peaks(spectra, name, lo, hi, substrate="SERS", step=1.0, merge_cm=8.0):
    """Noise-calibrated peak list for spectra[name]["wn_y"]; the other entries of `spectra` are
    replicates used to check reproduction. Returns (DataFrame, noise threshold)."""
    wn, y = spectra[name]["wn_y"]
    sigma = noise_sigma(wn, y)
    thr = noise_threshold(wn, y, sigma)
    others = {}
    for o, v in spectra.items():
        if o != name:
            ow, oy = v["wn_y"]
            osg = noise_sigma(ow, oy)
            others[o] = (ow, oy, osg, noise_threshold(ow, oy, osg))
    own_pk, _, _ = checked_peaks(wn, y, substrate, lo, hi)
    own_ok = own_pk[own_pk.confirmed]  # bands passing checks 1+2 in this spectrum
    own_conf, own_fwhm = own_ok.position_cm1.values, own_ok.FWHM_cm1.values
    grid = np.arange(lo + 40, hi - 40, step)
    sc = np.array([calib_score(wn, y, c, sigma) for c in grid], dtype=float)  # (position, score)

    def reproduced(xm):
        rep_in = []
        for o, (ow, oy, osg, othr) in others.items():
            ox, osc = calib_score(ow, oy, xm, osg)
            if osc > othr and abs(ox - xm) <= 3.0:
                rep_in.append(f"{o} {ox:.1f}")
        return rep_in

    def reps(xm):
        return {r.split()[0] for r in reproduced(xm)}

    # each contiguous stretch of the score profile above the noise threshold is one band (its
    # strongest maximum); a secondary maximum inside the stretch (>= merge_cm away) is kept as a
    # separate band only if another spectrum reproduces it (independent evidence of a shoulder)
    above = sc[:, 1] > thr
    rows = []
    k = 0
    while k < len(grid):
        if not above[k]:
            k += 1
            continue
        e = k
        while e + 1 < len(grid) and above[e + 1]:
            e += 1
        seg = sc[k:e + 1]
        # candidate bands in the stretch: distinct raw maxima found by calib_score, best score each
        cands = {}
        for xm, s_ in seg:
            cands[xm] = max(s_, cands.get(xm, 0.0))
        cands = sorted(cands.items(), key=lambda t: -t[1])
        rep = {xm: reps(xm) for xm, _ in cands}
        own = {}  # FWHM of the checks-1+2 band at this maximum (None if there is none)
        for xm, _ in cands:
            d = np.abs(own_conf - xm) if len(own_conf) else np.array([99.0])
            own[xm] = float(own_fwhm[d.argmin()]) if d.min() <= 2.0 else None
        # main band: strongest reproduced maximum, else strongest maximum
        main = next(((xm, s_) for xm, s_ in cands if rep[xm]), cands[0])
        chosen = [main]
        for xm, s_ in cands:  # further bands: >= merge_cm apart and reproduced or confirmed by checks 1+2
            near = min(chosen, key=lambda c: abs(xm - c[0]))
            dist = abs(xm - near[0])
            # a reproduced maximum is a separate band (shoulder) only if one replicate shows BOTH it and
            # the neighbouring band; otherwise replicates just scatter around one broad band.
            # An own (checks 1+2) band counts if it is narrower than its distance.
            separate_rep = bool(rep[xm] & rep.get(near[0], set()))
            if dist >= merge_cm and (separate_rep or (own[xm] is not None and own[xm] < dist)):
                chosen.append((xm, s_))
        for xm, s_ in chosen:
            rep_in = reproduced(xm)
            spike = spike_like(wn, y, xm, s_ * sigma)
            rows.append(dict(position_cm1=round(float(xm), 1), raw_counts=round(float(y[np.argmin(np.abs(wn - xm))]), 1),
                             calib_score=round(float(s_), 1), noise_threshold=round(thr, 1),
                             band_extent_cm1=f"{grid[k]:.0f}-{grid[e]:.0f}",
                             reproduced_in="; ".join(rep_in), assignment=assign(xm, substrate),
                             flag=("spike-like, reproduced" if rep_in else
                                   "spike-like, not reproduced: possible cosmic ray") if spike
                             else ("" if rep_in else ("not reproduced in replicates" if others
                                                      else "no replicate available")),
                             confirmed=(not spike) or bool(rep_in)))
        k = e + 1
    out = pd.DataFrame(rows).drop_duplicates("position_cm1").sort_values("position_cm1").reset_index(drop=True)
    # bands closer than merge_cm (a band split where the score dipped under the threshold): keep the
    # kept/reproduced/stronger one
    keep = []
    for _, r in out.iterrows():
        if keep and r.position_cm1 - keep[-1].position_cm1 < merge_cm:
            prev = keep[-1]
            rank = lambda q: (bool(q.confirmed), bool(q.reproduced_in), q.calib_score)  # noqa: E731
            if rank(r) > rank(prev):
                keep[-1] = r
            continue
        keep.append(r)
    out = pd.DataFrame(keep).reset_index(drop=True)
    out["SNR"] = out.calib_score
    out["FWHM_cm1"] = np.nan  # not measurable reliably on raw data for weak bands; see band_extent_cm1
    return out, thr


# compact slide labels (full assignments are in the CSV files)
SLIDE = {"C-C stretch protein backbone": "C-C backbone", "CH2/CH3 deformation": "CH$_2$/CH$_3$",
         "COO- symmetric stretch": "COO$^-$ sym.", "COO- asymmetric stretch": "COO$^-$ asym.",
         "Si optical phonon": "Si", "Tyr ring breathing": "Tyr", "Phe ring breathing": "Phe",
         "Tyr C-H bend": "Tyr", "C=O stretch": "C=O ester", "C-C stretch": "C-C", "Tyr": "Tyr", "Phe": "Phe",
         "unassigned": "n.a.", "PO2- symmetric stretch": "PO$_2^-$ / C-O", "CH3CH2 wagging": "CH$_3$CH$_2$ wag",
         "CH2 twist": "CH$_2$ twist", "Amide II": "Amide II", "Tyr / Trp ring C=C stretch": "Tyr/Trp", "Si 2nd-order phonon": "Si 2nd order", "Carotenoid C=C stretch": "Carotenoid"}


def slide_label(assignment):
    if assignment in SLIDE:
        return SLIDE[assignment]
    sh = short(assignment)
    return SLIDE.get(sh, sh)


def draw(ax, wn, y, pk, lo, hi, color, shared_pos, label_fs=10, headroom=0.9, title=None, shared_color="#2ca02c"):
    """Raw spectrum with labelled confirmed peaks (labels pushed apart, ticks at true positions)."""
    m = (wn >= lo) & (wn <= hi)
    ax.plot(wn[m], y[m], color=color, lw=0.9)
    ymin, ymax = y[m].min(), y[m].max()
    rng = ymax - ymin
    ax.set_ylim(ymin - 0.03 * rng, ymax + headroom * rng)
    for p in shared_pos:
        if lo <= p <= hi:
            ax.axvline(p, color=shared_color, ls="--", lw=1.0, alpha=0.8)
    sel = pk[(pk.confirmed) & (pk.position_cm1 >= lo) & (pk.position_cm1 <= hi)].sort_values("position_cm1")
    gap = 0.014 * (hi - lo) * (label_fs / 11.0)  # minimum x-distance between rotated labels
    xs = []
    for x in sel.position_cm1:  # push labels apart left-to-right; ticks stay at the true position
        xs.append(max(x, xs[-1] + gap) if xs else x)
    for (_, p), xt in zip(sel.iterrows(), xs):
        top = p.raw_counts + 0.03 * rng
        ax.plot([p.position_cm1, p.position_cm1, xt], [top, top + 0.04 * rng, top + 0.06 * rng], color="k", lw=0.9)
        ax.text(xt, top + 0.07 * rng, f"{p.position_cm1:.1f} {slide_label(p.assignment)}",
                rotation=90, fontsize=label_fs, ha="center", va="bottom", clip_on=True)
    if title:
        ax.text(0.01, 0.97, title, transform=ax.transAxes, ha="left", va="top", color=color, fontsize=14,
                fontweight="bold", bbox=dict(facecolor="white", edgecolor="none", alpha=0.85, pad=2))
    ax.set_xlim(lo, hi)
    ax.tick_params(direction="in", length=5)
