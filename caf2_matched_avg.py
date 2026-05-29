"""
caf2_matched_avg.py
===================
Loads the 21 SERS map spectra that match all three CaF2 reference peaks
(780 / 1483 / 1575 cm-1 within ±5 cm-1), averages them, and produces a
publication-quality comparison figure against the CaF2 reference.

HOW TO RUN:
  1. pip install numpy matplotlib scipy   (once, in PyCharm terminal)
  2. Verify FOLDER path below is correct.
  3. Press Run.
  4. Output image saved to FOLDER as  caf2_matched_spectra_avg.png
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve

# ─────────────────────────────────────────────────────────────────────────────
#  EDIT ONLY THIS SECTION
# ─────────────────────────────────────────────────────────────────────────────
FOLDER = r"C:\Users\Hira Aman\Desktop\RAMAN MAP 1"

FILE_CAF2 = os.path.join(FOLDER, "S1-DNA-HACAT1-20ng-CaF2_532nm_600gr_BC50_100X_10s_4a_100%.txt")
FILE_M1   = os.path.join(FOLDER, "m1-DNA-HACAT1-20ng-AgSiNW_532nm_600gr_BC200_50XLF_05s_4a_2-5%_5ul_drop-center.txt")
FILE_M2   = os.path.join(FOLDER, "m2-DNA-HACAT1-20ng-AgSiNW_532nm_600gr_BC200_50XLF_05s_4a_2-5%_5ul_drop-edge.txt")
FILE_M3   = os.path.join(FOLDER, "m3-DNA-HACAT1-20ng-AgSiNW_532nm_600gr_BC200_50XLF_05s_4a_2-5%_5ul_drop-center.txt")
FILE_M4   = os.path.join(FOLDER, "m4-DNA-HACAT1-20ng-AgSiNW_532nm_600gr_BC200_50XLF_05s_4a_2-5%_5ul_drop-center.txt")

# Spectrum numbers (1-indexed) that match all 3 CaF2 peaks simultaneously
SELECTED = {
    "m1": [16, 18, 20, 21],
    "m2": [1, 6, 8, 9, 11, 12, 18, 25],
    "m3": [2, 4, 9, 14, 19],
    "m4": [3, 7, 10, 12],
}
# ─────────────────────────────────────────────────────────────────────────────

# CaF2 exact peak positions confirmed from the reference file
CAF2_PEAKS  = [780.32, 1483.39, 1575.11]
PEAK_NAMES  = ["~780 cm⁻¹", "~1483 cm⁻¹", "~1575 cm⁻¹"]
ZOOM_PAD    = 45   # cm-1 padding around each zoom window


# ── LOADERS ──────────────────────────────────────────────────────────────────

def load_caf2(filepath):
    """2-column tab-separated: wavenumber, intensity."""
    wn, sp = [], []
    with open(filepath, "r", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.replace(",", ".").split()
            if len(parts) < 2:
                continue
            try:
                # handle optional leading row-index column
                if len(parts) >= 3:
                    try:
                        w, i = float(parts[1]), float(parts[2])
                    except ValueError:
                        w, i = float(parts[0]), float(parts[1])
                else:
                    w, i = float(parts[0]), float(parts[1])
                wn.append(w)
                sp.append(i)
            except ValueError:
                continue
    wn = np.array(wn, dtype=float)
    sp = np.array(sp, dtype=float)
    order = np.argsort(wn)
    return wn[order], sp[order]


def load_map_selected(filepath, spectrum_numbers):
    """
    Load LabSpec6 exported map .txt.
    Row 0 = wavenumber axis (1989 fields).
    Rows 1+ = [X, Y, intensity×1989]  (X,Y may or may not be present).
    spectrum_numbers : list of 1-based indices to extract.
    Returns (wn, list_of_spectra).
    """
    data_rows = []
    with open(filepath, "r", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.replace(",", ".").split("\t")
            if len(parts) < 3:
                parts = line.replace(",", ".").split()
            try:
                data_rows.append([float(v) for v in parts])
            except ValueError:
                continue

    if len(data_rows) < 2:
        raise ValueError(f"File has fewer than 2 data rows: {filepath}")

    wn = np.array(data_rows[0], dtype=float)
    n_wn = len(wn)

    spectra_out = []
    for sp_num in spectrum_numbers:
        row_idx = sp_num   # data_rows[0]=wn axis, data_rows[1]=spectrum 1, etc.
        if row_idx >= len(data_rows):
            raise IndexError(
                f"Spectrum #{sp_num} requested but file only has "
                f"{len(data_rows) - 1} spectra: {filepath}"
            )
        arr = np.array(data_rows[row_idx], dtype=float)
        if len(arr) == n_wn + 2:       # X, Y prepended
            spectra_out.append(arr[2:])
        elif len(arr) == n_wn:          # no X,Y
            spectra_out.append(arr)
        else:                           # fallback
            spectra_out.append(arr[-n_wn:])

    return wn, spectra_out


# ── PROCESSING ───────────────────────────────────────────────────────────────

def als_baseline(y, lam=1e5, p=0.005, n_iter=15):
    """Asymmetric Least Squares baseline (Eilers & Boelens 2005)."""
    L = len(y)
    D = diags([1, -2, 1], [0, 1, 2], shape=(L - 2, L), dtype=float)
    w = np.ones(L)
    for _ in range(n_iter):
        W = diags(w, 0, shape=(L, L), dtype=float)
        Z = W + lam * D.T.dot(D)
        z = spsolve(Z.tocsr(), w * y)
        w = p * (y > z) + (1 - p) * (y <= z)
    return z


def baseline_correct(wn, sp, lo=395, hi=1810):
    """Restrict to [lo, hi] cm-1, ALS-correct, clip negatives."""
    mask = (wn >= lo) & (wn <= hi)
    wn_r = wn[mask]
    sp_r = sp[mask]
    bl   = als_baseline(sp_r)
    return wn_r, np.clip(sp_r - bl, 0, None)


def minmax(sp):
    lo, hi = sp.min(), sp.max()
    if hi == lo:
        return np.zeros_like(sp)
    return (sp - lo) / (hi - lo)


def find_local_max(wn, sp, center, tol=10):
    """Return (peak_wn, peak_intensity) of local max within [center-tol, center+tol]."""
    mask = (wn >= center - tol) & (wn <= center + tol)
    if mask.sum() == 0:
        return np.nan, np.nan
    idx = int(np.argmax(sp[mask]))
    wn_w = wn[mask]
    sp_w = sp[mask]
    return float(wn_w[idx]), float(sp_w[idx])


# ── FIGURE ───────────────────────────────────────────────────────────────────

def make_figure(wn_avg, sp_avg_n,
                wn_c,   sp_c_n,
                all_wn_list, all_sp_list,
                out_path):
    """
    Top panel  : full 400-1800 cm-1 — individual (faded grey) + average (black) + CaF2 (orange).
    Bottom row : 3 zoom panels, one per CaF2 peak, with exact position labels.
    """
    matplotlib.rcParams.update({
        "font.family":       "DejaVu Sans",
        "figure.facecolor":  "white",
        "axes.spines.top":   False,
        "axes.spines.right": False,
    })

    fig = plt.figure(figsize=(16, 12))
    gs  = fig.add_gridspec(
        2, 3,
        height_ratios=[2.2, 1.5],
        hspace=0.52, wspace=0.30,
        left=0.07, right=0.97,
        top=0.91,  bottom=0.07,
    )
    ax_main = fig.add_subplot(gs[0, :])          # spans all 3 columns
    ax_z    = [fig.add_subplot(gs[1, k]) for k in range(3)]

    # ── Main panel ────────────────────────────────────────────────────────────
    # individual spectra (interpolated to avg wavenumber axis)
    for wn_i, sp_i in zip(all_wn_list, all_sp_list):
        # restrict to same range as avg
        mask_i = (wn_i >= 400) & (wn_i <= 1800)
        ax_main.plot(wn_i[mask_i], sp_i[mask_i],
                     color="#AAAAAA", lw=0.6, alpha=0.45, zorder=1)

    # CaF2 reference
    mask_c = (wn_c >= 400) & (wn_c <= 1800)
    ax_main.plot(wn_c[mask_c], sp_c_n[mask_c],
                 color="#E67E22", lw=1.8, alpha=0.9, label="CaF₂ reference", zorder=3)

    # average SERS
    mask_a = (wn_avg >= 400) & (wn_avg <= 1800)
    ax_main.plot(wn_avg[mask_a], sp_avg_n[mask_a],
                 color="black", lw=2.2, label="Average SERS (n = 21)", zorder=4)

    # CaF2 peak markers + shift annotation
    y_top = 1.05
    for caf2_pos, pname in zip(CAF2_PEAKS, PEAK_NAMES):
        ax_main.axvline(caf2_pos, color="#E67E22", lw=1.0, ls="--", alpha=0.7, zorder=2)

        sers_wn, sers_int = find_local_max(wn_avg[mask_a], sp_avg_n[mask_a], caf2_pos, tol=10)
        shift = sers_wn - caf2_pos if not np.isnan(sers_wn) else np.nan

        label_text = (
            f"CaF₂: {caf2_pos:.0f}\n"
            f"SERS: {sers_wn:.0f}\n"
            f"Δ = {shift:+.1f} cm⁻¹"
        ) if not np.isnan(sers_wn) else f"CaF₂: {caf2_pos:.0f}"

        ax_main.text(
            caf2_pos, y_top + 0.02, label_text,
            fontsize=8, ha="center", va="bottom",
            color="#C0392B",
            bbox=dict(boxstyle="round,pad=0.25", fc="white",
                      ec="#E67E22", lw=0.8, alpha=0.92),
            zorder=5,
        )
        if not np.isnan(sers_wn):
            ax_main.axvline(sers_wn, color="black", lw=0.8, ls=":", alpha=0.6, zorder=2)

    ax_main.set_xlim(400, 1800)
    ax_main.set_ylim(-0.06, y_top + 0.28)
    ax_main.set_xlabel("Raman shift (cm⁻¹)", fontsize=11)
    ax_main.set_ylabel("Normalised intensity (a.u.)", fontsize=11)
    ax_main.set_title(
        "Average of 21 SERS map spectra vs CaF₂ reference\n"
        "(ALS baseline-corrected · min-max normalised · "
        "spectra selected where all three CaF₂ peaks are present within ±5 cm⁻¹)",
        fontsize=10, fontweight="bold",
    )
    ax_main.legend(fontsize=9, loc="upper left", framealpha=0.9)
    ax_main.grid(True, alpha=0.12, lw=0.5)

    # ── Zoom panels ───────────────────────────────────────────────────────────
    zoom_titles = [
        f"Zoom: {CAF2_PEAKS[0]:.0f} cm⁻¹ region",
        f"Zoom: {CAF2_PEAKS[1]:.0f} cm⁻¹ region",
        f"Zoom: {CAF2_PEAKS[2]:.0f} cm⁻¹ region",
    ]
    for k, (ax_z_k, caf2_pos, ztitle) in enumerate(
            zip(ax_z, CAF2_PEAKS, zoom_titles)):

        lo = caf2_pos - ZOOM_PAD
        hi = caf2_pos + ZOOM_PAD

        # individual
        for wn_i, sp_i in zip(all_wn_list, all_sp_list):
            mi = (wn_i >= lo) & (wn_i <= hi)
            ax_z_k.plot(wn_i[mi], sp_i[mi],
                        color="#BBBBBB", lw=0.7, alpha=0.5)

        # CaF2
        mc = (wn_c >= lo) & (wn_c <= hi)
        ax_z_k.plot(wn_c[mc], sp_c_n[mc],
                    color="#E67E22", lw=1.8, alpha=0.9, label="CaF₂")

        # average
        ma = (wn_avg >= lo) & (wn_avg <= hi)
        ax_z_k.plot(wn_avg[ma], sp_avg_n[ma],
                    color="black", lw=2.0, label="Avg SERS")

        # CaF2 reference marker
        ax_z_k.axvline(caf2_pos, color="#E67E22", lw=1.0, ls="--",
                       alpha=0.8, label=f"CaF₂ {caf2_pos:.0f}")

        # SERS peak marker
        sers_wn, sers_int = find_local_max(wn_avg[ma], sp_avg_n[ma],
                                           caf2_pos, tol=12)
        shift = sers_wn - caf2_pos if not np.isnan(sers_wn) else np.nan

        if not np.isnan(sers_wn):
            ax_z_k.axvline(sers_wn, color="black", lw=1.0, ls=":",
                           alpha=0.85)
            y_max_local = sp_avg_n[ma].max() if ma.sum() > 0 else 1.0
            ax_z_k.annotate(
                f"CaF₂: {caf2_pos:.1f}\nSERS: {sers_wn:.1f}\nΔ={shift:+.1f} cm⁻¹",
                xy=(sers_wn, sers_int),
                xytext=(sers_wn + 8, sers_int + 0.08),
                fontsize=7.5,
                color="#C0392B",
                fontweight="bold",
                arrowprops=dict(arrowstyle="->", color="#C0392B", lw=0.8),
                bbox=dict(boxstyle="round,pad=0.2", fc="white",
                          ec="#C0392B", lw=0.7, alpha=0.92),
            )

        ax_z_k.set_xlim(lo, hi)
        ax_z_k.set_title(ztitle, fontsize=9, fontweight="bold")
        ax_z_k.set_xlabel("Raman shift (cm⁻¹)", fontsize=8)
        if k == 0:
            ax_z_k.set_ylabel("Norm. intensity", fontsize=8)
        ax_z_k.legend(fontsize=7, loc="upper right", framealpha=0.85)
        ax_z_k.grid(True, alpha=0.12, lw=0.5)
        ax_z_k.tick_params(labelsize=7)

    fig.savefig(out_path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Figure saved: {out_path}")


# ── MAIN ─────────────────────────────────────────────────────────────────────

def main():
    print("=" * 65)
    print("CaF2 Matched Spectra — Average Plot")
    print("=" * 65)

    # -- Load CaF2 reference
    print("\nLoading CaF2 reference ...")
    wn_c, sp_c = load_caf2(FILE_CAF2)
    wn_c_bc, sp_c_bc = baseline_correct(wn_c, sp_c)
    sp_c_n = minmax(sp_c_bc)
    print(f"  OK: {len(wn_c)} pts, {wn_c[0]:.1f} – {wn_c[-1]:.1f} cm-1")

    # -- Load selected spectra from each map
    map_files = {
        "m1": FILE_M1,
        "m2": FILE_M2,
        "m3": FILE_M3,
        "m4": FILE_M4,
    }

    all_wn_bc = []   # wavenumber arrays after BLC (may differ slightly per map)
    all_sp_n  = []   # normalised BLC spectra
    labels    = []   # e.g. "m2_spectrum_11"

    for mlab, fpath in map_files.items():
        sp_nums = SELECTED.get(mlab, [])
        if not sp_nums:
            continue
        print(f"\nLoading {mlab}: spectra {sp_nums} ...")
        wn_map, sp_list = load_map_selected(fpath, sp_nums)
        for sp_num, sp_raw in zip(sp_nums, sp_list):
            wn_bc, sp_bc = baseline_correct(wn_map, sp_raw)
            sp_n = minmax(sp_bc)
            all_wn_bc.append(wn_bc)
            all_sp_n.append(sp_n)
            labels.append(f"{mlab}_spectrum_{sp_num:02d}")
            print(f"    {mlab}_spectrum_{sp_num:02d}  loaded OK")

    print(f"\nTotal spectra loaded: {len(all_sp_n)}")

    # -- Build common wavenumber axis for averaging (use first map's BLC axis)
    # All maps share essentially the same wavenumber range post-BLC (400-1800)
    # Interpolate everything onto the first spectrum's axis
    wn_ref = all_wn_bc[0]
    sp_interp = []
    for wn_i, sp_i in zip(all_wn_bc, all_sp_n):
        if np.allclose(wn_i, wn_ref, atol=0.1):
            sp_interp.append(sp_i)
        else:
            sp_interp.append(np.interp(wn_ref, wn_i, sp_i))

    sp_stack = np.vstack(sp_interp)          # shape: (21, N)
    sp_avg   = sp_stack.mean(axis=0)
    sp_avg_n = minmax(sp_avg)                # normalise the average

    # -- Report peak positions
    print("\n  CaF2 peak positions vs average SERS:")
    print(f"  {'CaF2 (cm-1)':>14}  {'SERS avg (cm-1)':>16}  {'Shift':>8}")
    print(f"  {'-'*45}")
    for caf2_pos in CAF2_PEAKS:
        sers_wn, _ = find_local_max(wn_ref, sp_avg_n, caf2_pos, tol=10)
        shift = sers_wn - caf2_pos if not np.isnan(sers_wn) else np.nan
        print(f"  {caf2_pos:>14.2f}  {sers_wn:>16.2f}  {shift:>+8.2f}")

    # -- Make figure
    out_png = os.path.join(FOLDER, "caf2_matched_spectra_avg.png")
    print(f"\nGenerating figure ...")
    make_figure(
        wn_ref, sp_avg_n,
        wn_c_bc, sp_c_n,
        all_wn_bc, all_sp_n,
        out_png,
    )

    # -- Also save CSV of the average spectrum
    out_csv = os.path.join(FOLDER, "caf2_matched_spectra_avg.csv")
    header = "wavenumber_cm-1,avg_SERS_normalised"
    np.savetxt(out_csv,
               np.column_stack([wn_ref, sp_avg_n]),
               delimiter=",", header=header, comments="")
    print(f"  CSV saved:    {out_csv}")

    print("\nDone.\n")


if __name__ == "__main__":
    main()
