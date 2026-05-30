"""
plot_clean_sers.py
==================
Loads SERS maps m1-m4, finds the cleanest DNA+HaCaT spectrum
from all 100 spectra, plots it (200-3200 cm-1) with every peak labeled.

HOW TO RUN IN PyCharm:
  1. Edit FOLDER below (folder containing all 4 map .txt files).
  2. Run with Shift+F10.
  3. Figure saved to that folder as: clean_sers_spectrum.png

REQUIREMENTS:
  pip install numpy scipy matplotlib
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")          # remove this line to get an interactive window
import matplotlib.pyplot as plt
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve
from scipy.signal import find_peaks

# ======================================================================
#  EDIT THIS PATH ONLY
FOLDER = r"C:\Users\Hira Aman\Desktop\RAMAN MAP 1"
#  EDIT THIS PATH ONLY
# ======================================================================

FILE_M1 = os.path.join(FOLDER, "m1-DNA-HACAT1-20ng-AgSiNW_532nm_600gr_BC200_50XLF_05s_4a_2-5%_5ul_drop-center.txt")
FILE_M2 = os.path.join(FOLDER, "m2-DNA-HACAT1-20ng-AgSiNW_532nm_600gr_BC200_50XLF_05s_4a_2-5%_5ul_drop-edge.txt")
FILE_M3 = os.path.join(FOLDER, "m3-DNA-HACAT1-20ng-AgSiNW_532nm_600gr_BC200_50XLF_05s_4a_2-5%_5ul_drop-center.txt")
FILE_M4 = os.path.join(FOLDER, "m4-DNA-HACAT1-20ng-AgSiNW_532nm_600gr_BC200_50XLF_05s_4a_2-5%_5ul_drop-center.txt")

MAP_FILES = {"m1": FILE_M1, "m2": FILE_M2, "m3": FILE_M3, "m4": FILE_M4}

# Known DNA + HaCaT Raman band positions (literature, cm-1)
KNOWN_BANDS = [650, 720, 752, 782, 828, 932, 1001, 1092,
               1178, 1240, 1310, 1375, 1425, 1484, 1575, 1640,
               2870, 2930, 2960]
KNOWN_TOL = 18    # cm-1 window for scoring

# Spectral region colours and labels
REGIONS = {
    (200,  400):  ("#EBF5FB", "200-400"),
    (400,  900):  ("#FEF9E7", "DNA backbone\n400-900"),
    (900,  1300): ("#EAFAF1", "Fingerprint\n900-1300"),
    (1300, 1800): ("#FDF2F8", "Nucleobases\n1300-1800"),
    (1800, 2800): ("#F8F9FA", "Silent region\n1800-2800"),
    (2800, 3200): ("#EBF5FB", "CH stretch\n2800-3200"),
}

# Assignment table
ASSIGN = {
    (400,  460):  "Ring deformation / sugar",
    (460,  510):  "Sugar-phosphate backbone",
    (510,  560):  "Backbone C-C stretch",
    (560,  615):  "G/C ring deformation",
    (615,  680):  "G/T ring breathing",
    (680,  710):  "Guanine ring breathing",
    (710,  750):  "Adenine ring breathing",
    (750,  800):  "Cytosine + O-P-O backbone",
    (800,  860):  "Backbone O-P-O stretch",
    (860,  920):  "Backbone C-C stretch",
    (920,  970):  "Sugar C-O stretch",
    (970,  1020): "Phenylalanine / ring",
    (1020, 1075): "C-N stretch / ring",
    (1075, 1115): "PO4 symmetric stretch",
    (1115, 1145): "PO4 / C-N",
    (1145, 1200): "C-N (cytosine/adenine)",
    (1200, 1270): "Amide III / thymine",
    (1270, 1340): "A/T/G in-plane",
    (1340, 1400): "T/G C-H deformation",
    (1400, 1455): "A/G ring stretch",
    (1455, 1510): "A/C C=N stretch",
    (1510, 1595): "G/A ring C=C / C=N",
    (1595, 1680): "Amide I / C=C stretch",
    (1680, 1800): "C=O stretch (nucleobase)",
    (2800, 2900): "CH2 symmetric stretch (HaCaT)",
    (2900, 2970): "CH3 symmetric stretch (HaCaT)",
    (2970, 3100): "CH2 asymmetric stretch (HaCaT)",
}


def get_assignment(wn_val):
    for (lo, hi), label in ASSIGN.items():
        if lo <= wn_val < hi:
            return label
    return "---"


# ======================================================================
#  DATA LOADING
# ======================================================================

def load_map(filepath):
    """Load a LabSpec6 map txt file. Returns wavenumber array and 2D spectra array."""
    rows = []
    with open(filepath, "r", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.replace(",", ".").split("\t")
            if len(parts) < 3:
                parts = line.replace(",", ".").split()
            try:
                rows.append([float(v) for v in parts])
            except ValueError:
                continue
    wavenumbers = np.array(rows[0])
    n_pts = len(wavenumbers)
    spectra = []
    for row in rows[1:]:
        arr = np.array(row)
        if len(arr) == n_pts + 2:
            spectra.append(arr[2:])
        elif len(arr) == n_pts:
            spectra.append(arr)
        else:
            spectra.append(arr[-n_pts:])
    return wavenumbers, np.array(spectra)


# ======================================================================
#  SIGNAL PROCESSING
# ======================================================================

def als_baseline(y, lam=1e5, p=0.005, n_iter=15):
    """Asymmetric Least Squares baseline correction."""
    L = len(y)
    D = diags([1, -2, 1], [0, 1, 2], shape=(L - 2, L), dtype=float)
    w = np.ones(L)
    for _ in range(n_iter):
        W = diags(w, 0, shape=(L, L), dtype=float)
        z = spsolve((W + lam * D.T.dot(D)).tocsr(), w * y)
        w = p * (y > z) + (1 - p) * (y <= z)
    return z


def process_spectrum(wn, sp_raw):
    """ALS baseline correction then min-max normalisation."""
    z  = als_baseline(sp_raw)
    bc = np.clip(sp_raw - z, 0, None)
    sp_max = bc.max()
    if sp_max <= 0:
        return bc, 0.0
    sp_n = bc / sp_max
    return sp_n, sp_max


def find_spectrum_peaks(wn, sp_n, height=0.05, prominence=0.04):
    """Detect genuine Raman peaks above noise floor."""
    step     = float(np.diff(wn).mean())
    min_dist = max(5, int(15 / step))
    idxs, _  = find_peaks(sp_n, height=height, prominence=prominence,
                           distance=min_dist)
    return wn[idxs], sp_n[idxs]


def score_spectrum(wn, sp_raw):
    """Score spectrum quality: SNR multiplied by matched known DNA bands."""
    sp_n, sp_max = process_spectrum(wn, sp_raw)
    if sp_max <= 0:
        return -1, sp_n, sp_max

    # SNR: signal max vs noise in silent region 2050-2550 cm-1
    noise_mask = (wn > 2050) & (wn < 2550)
    z_full     = als_baseline(sp_raw)
    bc_full    = np.clip(sp_raw - z_full, 0, None)
    noise_std  = bc_full[noise_mask].std()
    if noise_std < 1:
        noise_std = 1
    snr = sp_max / noise_std

    # Count how many known DNA+HaCaT bands are present
    pk_wns, _ = find_spectrum_peaks(wn, sp_n)
    n_matched = 0
    for k in KNOWN_BANDS:
        if pk_wns.size > 0 and np.abs(pk_wns - k).min() < KNOWN_TOL:
            n_matched += 1

    # Penalise artifact peaks in biologically silent region 1800-2600 cm-1
    artifact_mask = (pk_wns > 1800) & (pk_wns < 2600)
    n_artifacts   = int(artifact_mask.sum())

    score = n_matched * snr / (1 + n_artifacts * 2)
    return score, sp_n, sp_max


# ======================================================================
#  MAIN
# ======================================================================

def main():
    matplotlib.rcParams.update({"font.family": "DejaVu Sans",
                                 "figure.facecolor": "white"})

    print("=" * 60)
    print("Scanning all 100 SERS spectra for the cleanest signal ...")
    print("=" * 60)

    best_score  = -1
    best_map    = ""
    best_sp_idx = 0
    best_wn     = None
    best_sp_n   = None
    best_sp_max = 0.0
    best_pk_wns = None
    best_pk_int = None

    for map_name, filepath in MAP_FILES.items():
        print(f"  Loading {map_name} ...")
        wn, spectra = load_map(filepath)
        for i, sp_raw in enumerate(spectra):
            score, sp_n, sp_max = score_spectrum(wn, sp_raw)
            if score > best_score:
                pk_wns, pk_ints = find_spectrum_peaks(wn, sp_n)
                best_score  = score
                best_map    = map_name
                best_sp_idx = i + 1
                best_wn     = wn
                best_sp_n   = sp_n
                best_sp_max = sp_max
                best_pk_wns = pk_wns
                best_pk_int = pk_ints

    print(f"\n  Best spectrum : {best_map} -- spectrum {best_sp_idx}")
    print(f"  Raw max count : {best_sp_max:.0f}")

    n_bio = sum(1 for k in KNOWN_BANDS
                if best_pk_wns.size > 0 and
                   np.abs(best_pk_wns - k).min() < KNOWN_TOL)
    print(f"  Peaks matched : {n_bio}/19 known DNA+HaCaT bands")
    print(f"  Peaks detected: {len(best_pk_wns)} total")

    # Print peak table to console
    print("\n  #   Position (cm-1)  Norm. intensity  Assignment")
    print("  " + "-" * 62)
    for j, (w, h) in enumerate(zip(best_pk_wns, best_pk_int), 1):
        print(f"  {j:<4}{w:<20.1f}{h:<17.4f}{get_assignment(w)}")

    # ------------------------------------------------------------------
    #  FIGURE
    # ------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(18, 7))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    # Shaded spectral regions
    for (lo, hi), (color, label) in REGIONS.items():
        ax.axvspan(lo, hi, color=color, alpha=0.55, lw=0)
        mid = (lo + hi) / 2
        if hi - lo > 150:
            ax.text(mid, 1.28, label, fontsize=7.5, ha="center", va="bottom",
                    color="grey", style="italic")

    # Spectrum fill and line
    mask = (best_wn >= 200) & (best_wn <= 3200)
    ax.fill_between(best_wn[mask], best_sp_n[mask], alpha=0.15, color="#1A5276")
    ax.plot(best_wn[mask], best_sp_n[mask], color="#1A5276", lw=1.5, zorder=5)

    # Peak markers and staggered labels
    pks = [(w, h) for w, h in zip(best_pk_wns, best_pk_int) if 200 < w < 3200]
    pks.sort(key=lambda x: x[0])
    label_y   = [1.10, 1.18, 1.10, 1.18]
    prev_x    = -9999
    level_idx = 0
    for w, h in pks:
        gap = w - prev_x
        level_idx = (level_idx + 1) % 4 if gap < 45 else 0
        y_lbl = label_y[level_idx]
        ax.plot(w, h, "o", color="#1A5276", ms=5.5, zorder=8,
                markeredgecolor="white", markeredgewidth=0.6)
        ax.plot([w, w], [h + 0.02, y_lbl - 0.04],
                color="#1A5276", lw=0.7, alpha=0.55)
        rot = 90 if gap < 60 else 0
        ax.text(w, y_lbl, f"{w:.0f}", fontsize=7.5, ha="center", va="bottom",
                color="#1A5276", fontweight="bold", rotation=rot)
        ax.axvline(w, color="#1A5276", lw=0.45, ls="--", alpha=0.25, zorder=3)
        prev_x = w

    ax.set_xlim(200, 3200)
    ax.set_ylim(-0.05, 1.50)
    ax.set_xlabel("Raman shift (cm⁻¹)", fontsize=13, fontweight="bold")
    ax.set_ylabel("Normalised intensity (a.u.)", fontsize=13, fontweight="bold")

    title_line1 = (f"DNA + HaCaT cells  |  SERS  |  {best_map} spectrum {best_sp_idx}"
                   f"  |  AgSiNW substrate  |  532 nm  |  2.5% laser power  |  0.5 s x 4 acc.")
    title_line2 = (f"Cleanest signal from 100 spectra (m1-m4)  "
                   f"ALS baseline corrected  {len(pks)} peaks detected")
    ax.set_title(title_line1 + "\n" + title_line2, fontsize=11, fontweight="bold")

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(labelsize=10)
    ax.grid(True, alpha=0.10, lw=0.5)

    out_path = os.path.join(FOLDER, "clean_sers_spectrum.png")
    fig.savefig(out_path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"\n  Figure saved: {out_path}")
    print("Done.")


if __name__ == "__main__":
    main()
