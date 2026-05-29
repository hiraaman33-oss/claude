"""
caf2_peak_match.py
==================
Compare SERS map spectra (m2, m3, m4) against CaF2 reference.
No baseline correction — only min-max normalisation.
Reports whether CaF2 peaks at 781, 1483, 1572 cm-1 appear in each
map spectrum (tolerance ±10 cm-1) and produces figures.

Edit FOLDER and filenames below, then run.
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# ── FILE PATHS ──────────────────────────────────────────────────────────────
FOLDER = r"C:\Users\Hira Aman\Desktop\RAMAN MAP 1"

FILE_CAF2 = os.path.join(FOLDER, "S1-DNA-HACAT1-20ng-CaF2_532nm_600gr_BC50_100X_10s_4a_100%.txt")
FILE_M2   = os.path.join(FOLDER, "m2-DNA-HACAT1-20ng-AgSiNW_532nm_600gr_BC200_50XLF_05s_4a_2-5%_5ul_drop-edge.txt")
FILE_M3   = os.path.join(FOLDER, "m3-DNA-HACAT1-20ng-AgSiNW_532nm_600gr_BC200_50XLF_05s_4a_2-5%_5ul_drop-center.txt")
FILE_M4   = os.path.join(FOLDER, "m4-DNA-HACAT1-20ng-AgSiNW_532nm_600gr_BC200_50XLF_05s_4a_2-5%_5ul_drop-center.txt")

OUT_DIR = os.path.join(FOLDER, "caf2_peak_match")

# ── ANALYSIS SETTINGS ────────────────────────────────────────────────────────
TARGET_PEAKS = [781, 1483, 1572]   # cm-1 — CaF2 reference peaks to search for
PEAK_TOL     = 10                  # ±cm-1 search window around each target

ZOOM_PAD     = 35                  # extra cm-1 padding around each peak window in plots

# ── LOADERS ──────────────────────────────────────────────────────────────────

def load_txt_caf2(filepath):
    """Load 2-column (or 3-column) CaF2 reference txt: wavenumber, intensity."""
    wn, intensity = [], []
    with open(filepath, "r", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.replace(",", ".").split()
            if len(parts) >= 2:
                try:
                    # 3-column variant has row index as first column
                    if len(parts) >= 3:
                        try:
                            w = float(parts[1])
                            i = float(parts[2])
                        except ValueError:
                            w = float(parts[0])
                            i = float(parts[1])
                    else:
                        w = float(parts[0])
                        i = float(parts[1])
                    wn.append(w)
                    intensity.append(i)
                except ValueError:
                    continue
    wn = np.array(wn, dtype=float)
    intensity = np.array(intensity, dtype=float)
    # sort by wavenumber just in case
    order = np.argsort(wn)
    return wn[order], intensity[order]


def load_txt_map(filepath):
    """
    Load LabSpec6 exported map .txt.
    Header lines start with '#'.  First data row = wavenumber axis.
    Subsequent rows: col[0]=X, col[1]=Y, col[2:] = intensities.
    Returns (wn_array, spectra_array[N, M]).
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
        raise ValueError(f"Not enough data rows in {filepath}")

    wn      = np.array(data_rows[0], dtype=float)
    spectra = []
    for row in data_rows[1:]:
        arr = np.array(row, dtype=float)
        # first two columns are X, Y coordinates
        if len(arr) == len(wn) + 2:
            spectra.append(arr[2:])
        elif len(arr) == len(wn):
            spectra.append(arr)
        else:
            # best-effort: take last len(wn) values
            spectra.append(arr[-len(wn):])

    return wn, np.array(spectra, dtype=float)


# ── NORMALISATION ─────────────────────────────────────────────────────────────

def minmax_norm(spectrum):
    lo, hi = spectrum.min(), spectrum.max()
    if hi == lo:
        return np.zeros_like(spectrum)
    return (spectrum - lo) / (hi - lo)


# ── PEAK SEARCH ───────────────────────────────────────────────────────────────

def find_peak_in_window(wn, spectrum, center, tol):
    """
    Within [center-tol, center+tol], return (found, actual_wn, intensity, shift).
    'found' is True only if the local max is a genuine local peak
    (higher than its immediate neighbours at ±5 cm-1).
    """
    mask = (wn >= center - tol) & (wn <= center + tol)
    if mask.sum() == 0:
        return False, np.nan, np.nan, np.nan

    wn_w  = wn[mask]
    sp_w  = spectrum[mask]
    idx_local = int(np.argmax(sp_w))
    actual_wn = float(wn_w[idx_local])
    intensity = float(sp_w[idx_local])
    shift     = actual_wn - center

    # require the local max intensity ≥ 5 % of global spectrum max
    # (avoids reporting noise bumps)
    threshold = 0.05 * spectrum.max()
    found = intensity >= threshold

    return found, actual_wn, intensity, shift


def search_peaks(wn, spectra, targets=TARGET_PEAKS, tol=PEAK_TOL):
    """
    For each spectrum and each target peak, run find_peak_in_window.
    Returns dict: results[peak_cm][spectrum_idx] = (found, actual_wn, intensity, shift)
    """
    results = {p: {} for p in targets}
    for i, sp in enumerate(spectra):
        sp_norm = minmax_norm(sp)
        for p in targets:
            results[p][i] = find_peak_in_window(wn, sp_norm, p, tol)
    return results


# ── FIGURES ───────────────────────────────────────────────────────────────────

MAP_COLORS  = {"m2": "#1f77b4", "m3": "#2ca02c", "m4": "#d62728"}
CAF2_COLOR  = "#ff7f0e"


def _plot_zoom_panel(ax, wn, sp_norm, wn_caf2, sp_caf2_norm,
                     center, tol, pad, map_label, sp_idx,
                     found, actual_wn, shift):
    lo, hi = center - tol - pad, center + tol + pad
    m_m = (wn     >= lo) & (wn     <= hi)
    m_c = (wn_caf2 >= lo) & (wn_caf2 <= hi)

    color = MAP_COLORS.get(map_label, "steelblue")

    ax.plot(wn_caf2[m_c], sp_caf2_norm[m_c],
            color=CAF2_COLOR, lw=1.4, alpha=0.9, label="CaF₂ ref")
    ax.plot(wn[m_m], sp_norm[m_m],
            color=color, lw=1.2, alpha=0.85,
            label=f"{map_label} sp#{sp_idx + 1}")

    # mark the target window
    ax.axvspan(center - tol, center + tol, alpha=0.08, color="grey")
    ax.axvline(center, color="grey", lw=0.8, ls="--", alpha=0.5)

    if found:
        ax.axvline(actual_wn, color=color, lw=1.2, ls=":", alpha=0.9)
        ax.text(actual_wn, ax.get_ylim()[1] if ax.get_ylim()[1] > 0 else 1.0,
                f"{actual_wn:.0f}\n({shift:+.1f})",
                fontsize=7, ha="center", va="top", color=color,
                bbox=dict(boxstyle="round,pad=0.15", fc="white",
                          ec=color, lw=0.6, alpha=0.9))

    ax.set_xlim(lo, hi)
    ax.set_xlabel("Raman shift (cm⁻¹)", fontsize=8)
    ax.tick_params(labelsize=7)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def plot_peak_overview(wn_caf2, sp_caf2_norm,
                       map_data,   # dict: {"m2": (wn, spectra), ...}
                       results,    # from search_peaks, per map
                       target_peak,
                       out_dir):
    """
    One figure per target peak: rows = map (m2/m3/m4), cols = 25 spectra.
    Green border = found, red border = absent.
    """
    tol = PEAK_TOL
    pad = ZOOM_PAD
    ncols = 5
    nrows_per_map = 5   # 25 spectra in 5×5

    for map_label, (wn, spectra) in map_data.items():
        n = len(spectra)
        fig_ncols = ncols
        fig_nrows = nrows_per_map
        fig, axes = plt.subplots(fig_nrows, fig_ncols,
                                 figsize=(fig_ncols * 3.2, fig_nrows * 2.8),
                                 sharex=True, sharey=True)
        fig.suptitle(
            f"CaF₂ peak {target_peak} cm⁻¹  (±{tol} cm⁻¹)  —  {map_label}\n"
            f"Green border = peak found  |  Red border = absent  |  "
            f"Orange = CaF₂  |  Colour = SERS map",
            fontsize=10, fontweight="bold", y=1.01
        )

        res_map = results[map_label][target_peak]

        for idx, ax in enumerate(axes.flat):
            if idx >= n:
                ax.set_visible(False)
                continue

            sp_norm = minmax_norm(spectra[idx])
            found, actual_wn, intensity, shift = res_map[idx]

            _plot_zoom_panel(ax, wn, sp_norm,
                             wn_caf2, sp_caf2_norm,
                             target_peak, tol, pad,
                             map_label, idx,
                             found, actual_wn, shift)

            border_color = "#27ae60" if found else "#c0392b"
            for spine in ax.spines.values():
                spine.set_edgecolor(border_color)
                spine.set_linewidth(1.8)

            ax.set_title(f"sp #{idx + 1}", fontsize=7, pad=2)
            if idx == 0:
                ax.legend(fontsize=6, loc="upper left",
                          framealpha=0.7, handlelength=1)

        fig.tight_layout(rect=[0, 0, 1, 0.98])
        fname = os.path.join(out_dir,
                             f"peak_{target_peak}_{map_label}_grid.png")
        fig.savefig(fname, dpi=140, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        print(f"  Saved: {fname}")


def plot_summary_per_peak(wn_caf2, sp_caf2_norm,
                          map_data, all_results,
                          target_peak, out_dir):
    """
    One combined figure: 3 rows (m2/m3/m4), 25 columns (spectra).
    Very compact — just shows presence/absence at a glance.
    """
    tol = PEAK_TOL
    ncols = 25
    nrows = 3
    map_labels = list(map_data.keys())

    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(ncols * 1.8, nrows * 2.2),
                             sharex=True, sharey=False)
    fig.suptitle(
        f"CaF₂ peak {target_peak} cm⁻¹  |  All maps × all spectra\n"
        f"Green = found  |  Red = absent",
        fontsize=11, fontweight="bold"
    )

    for row_i, map_label in enumerate(map_labels):
        wn, spectra = map_data[map_label]
        res_map = all_results[map_label][target_peak]
        color = MAP_COLORS.get(map_label, "steelblue")

        for col_i in range(ncols):
            ax = axes[row_i, col_i]
            if col_i >= len(spectra):
                ax.set_visible(False)
                continue

            sp_norm = minmax_norm(spectra[col_i])
            found, actual_wn, intensity, shift = res_map[col_i]

            lo = target_peak - tol - ZOOM_PAD
            hi = target_peak + tol + ZOOM_PAD
            m_m = (wn      >= lo) & (wn      <= hi)
            m_c = (wn_caf2 >= lo) & (wn_caf2 <= hi)

            ax.plot(wn_caf2[m_c], sp_caf2_norm[m_c],
                    color=CAF2_COLOR, lw=0.8, alpha=0.7)
            ax.plot(wn[m_m], sp_norm[m_m],
                    color=color, lw=0.9)

            if found:
                ax.axvline(actual_wn, color=color, lw=0.9, ls=":")
                ax.text(0.5, 0.96, f"{actual_wn:.0f}",
                        transform=ax.transAxes,
                        fontsize=5.5, ha="center", va="top", color=color)

            ax.axvspan(target_peak - tol, target_peak + tol,
                       alpha=0.07, color="grey")

            border_color = "#27ae60" if found else "#c0392b"
            for spine in ax.spines.values():
                spine.set_edgecolor(border_color)
                spine.set_linewidth(1.6)

            ax.set_xlim(lo, hi)
            ax.tick_params(labelsize=5, length=2)
            if col_i == 0:
                ax.set_ylabel(map_label, fontsize=8, fontweight="bold")
            if row_i == 0:
                ax.set_title(f"#{col_i + 1}", fontsize=6, pad=1)

    fig.tight_layout(h_pad=0.3, w_pad=0.1)
    fname = os.path.join(out_dir, f"peak_{target_peak}_summary.png")
    fig.savefig(fname, dpi=130, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Saved: {fname}")


def plot_full_spectra_overview(wn_caf2, sp_caf2_norm,
                               map_data, all_results, out_dir):
    """
    One figure: full normalised spectrum (400-1800 cm-1) for every map spectrum.
    Vertical lines at each target peak position; dot = found, X = absent.
    """
    tol = PEAK_TOL
    ncols = 5
    nrows = 5

    for map_label, (wn, spectra) in map_data.items():
        color = MAP_COLORS.get(map_label, "steelblue")
        fig, axes = plt.subplots(nrows, ncols,
                                 figsize=(ncols * 3.6, nrows * 2.4),
                                 sharex=True)
        fig.suptitle(
            f"{map_label}  — full normalised spectra (400–1800 cm⁻¹)\n"
            f"Vertical lines at CaF₂ peaks: "
            + "  |  ".join(f"{p} cm⁻¹" for p in TARGET_PEAKS)
            + "   (● found, ✕ absent)",
            fontsize=10, fontweight="bold", y=1.01
        )

        mask_full_c = (wn_caf2 >= 400) & (wn_caf2 <= 1800)
        mask_full_m = (wn >= 400) & (wn <= 1800)

        for idx, ax in enumerate(axes.flat):
            if idx >= len(spectra):
                ax.set_visible(False)
                continue

            sp_norm = minmax_norm(spectra[idx])

            ax.plot(wn_caf2[mask_full_c], sp_caf2_norm[mask_full_c],
                    color=CAF2_COLOR, lw=0.8, alpha=0.6, label="CaF₂")
            ax.plot(wn[mask_full_m], sp_norm[mask_full_m],
                    color=color, lw=1.0, label=f"{map_label}")

            for p in TARGET_PEAKS:
                res = all_results[map_label][p][idx]
                found, actual_wn = res[0], res[1]
                ls  = "-"  if found else "--"
                alp = 0.85 if found else 0.30
                ax.axvline(p, color="grey", lw=0.8, ls=ls, alpha=alp)
                if found:
                    ax.plot(actual_wn, sp_norm[np.argmin(np.abs(wn - actual_wn))],
                            "o", ms=4, color=color, zorder=5)
                    ax.text(actual_wn, -0.07,
                            f"●{actual_wn:.0f}",
                            fontsize=5.5, ha="center", va="top",
                            color=color, rotation=60)
                else:
                    ax.text(p, -0.07, f"✕{p}",
                            fontsize=5.5, ha="center", va="top",
                            color="#c0392b", rotation=60)

            ax.set_xlim(400, 1800)
            ax.set_ylim(-0.25, 1.15)
            ax.set_title(f"sp #{idx + 1}", fontsize=7, pad=2)
            ax.tick_params(labelsize=6)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            if idx == 0:
                ax.legend(fontsize=6, loc="upper right", framealpha=0.7)

        fig.tight_layout(rect=[0, 0, 1, 0.97])
        fname = os.path.join(out_dir, f"{map_label}_full_spectra_overview.png")
        fig.savefig(fname, dpi=130, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        print(f"  Saved: {fname}")


# ── TEXT REPORT ───────────────────────────────────────────────────────────────

def print_and_save_report(all_results, map_data, out_dir):
    lines = []
    lines.append("=" * 70)
    lines.append("CaF2 PEAK MATCH REPORT")
    lines.append(f"Target peaks: {TARGET_PEAKS} cm-1   tolerance: ±{PEAK_TOL} cm-1")
    lines.append("Method: min-max normalisation only (no baseline correction)")
    lines.append("=" * 70)

    for p in TARGET_PEAKS:
        lines.append(f"\n{'─' * 60}")
        lines.append(f"  CaF2 peak: {p} cm-1")
        lines.append(f"{'─' * 60}")
        for map_label in map_data:
            res_map = all_results[map_label][p]
            found_list = [(i, res_map[i]) for i in sorted(res_map)
                          if res_map[i][0]]
            absent_count = sum(1 for i in res_map if not res_map[i][0])
            lines.append(f"\n  {map_label}:  "
                         f"{len(found_list)}/25 spectra show this peak")
            if found_list:
                lines.append("    Spectrum  | Actual wn (cm-1) | Shift (cm-1) | Intensity (norm)")
                lines.append("    " + "-" * 56)
                for sp_idx, (found, actual_wn, intensity, shift) in found_list:
                    lines.append(
                        f"    #{sp_idx + 1:2d}       |"
                        f"  {actual_wn:7.2f}         |"
                        f"  {shift:+6.2f}       |"
                        f"  {intensity:.4f}"
                    )
            else:
                lines.append("    No spectrum shows this peak within tolerance.")

    lines.append("\n" + "=" * 70)
    lines.append("SUMMARY TABLE  (Y = found, . = absent)")
    lines.append(f"{'Map':<5} {'Peak':>6}  " +
                 "  ".join(f"sp{i+1:02d}" for i in range(25)))
    lines.append("─" * 160)
    for map_label in map_data:
        for p in TARGET_PEAKS:
            res_map = all_results[map_label][p]
            row = "  ".join("Y " if res_map[i][0] else ". " for i in range(25))
            lines.append(f"{map_label:<5} {p:>6}  {row}")
    lines.append("=" * 70)

    report_text = "\n".join(lines)
    print(report_text)

    report_path = os.path.join(out_dir, "caf2_peak_match_report.txt")
    with open(report_path, "w", encoding="utf-8") as fh:
        fh.write(report_text)
    print(f"\n  Report saved: {report_path}")
    return report_text


# ── MAIN ─────────────────────────────────────────────────────────────────────

def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    matplotlib.rcParams.update({"font.family": "DejaVu Sans",
                                 "figure.facecolor": "white"})

    print("Loading CaF2 reference ...")
    wn_caf2, sp_caf2 = load_txt_caf2(FILE_CAF2)
    sp_caf2_norm = minmax_norm(sp_caf2)
    print(f"  CaF2: {len(wn_caf2)} points, "
          f"{wn_caf2[0]:.1f} – {wn_caf2[-1]:.1f} cm-1")

    print("Loading map files ...")
    map_data = {}
    for label, path in [("m2", FILE_M2), ("m3", FILE_M3), ("m4", FILE_M4)]:
        wn, spectra = load_txt_map(path)
        print(f"  {label}: {len(spectra)} spectra × {spectra.shape[1]} pts, "
              f"wn {wn[0]:.1f}–{wn[-1]:.1f} cm-1")
        map_data[label] = (wn, spectra)

    print("\nSearching for CaF2 peaks in all map spectra ...")
    all_results = {}
    for map_label, (wn, spectra) in map_data.items():
        all_results[map_label] = search_peaks(wn, spectra)
        for p in TARGET_PEAKS:
            n_found = sum(1 for v in all_results[map_label][p].values() if v[0])
            print(f"  {map_label}  {p} cm-1: found in {n_found}/25 spectra")

    print("\nGenerating figures ...")

    # 1. Per-peak summary grid (3 rows × 25 cols)
    for p in TARGET_PEAKS:
        plot_summary_per_peak(wn_caf2, sp_caf2_norm,
                              map_data, all_results, p, OUT_DIR)

    # 2. Per-map per-peak 5×5 grid (detailed zoom)
    for p in TARGET_PEAKS:
        plot_peak_overview(wn_caf2, sp_caf2_norm,
                           map_data, all_results, p, OUT_DIR)

    # 3. Full spectrum overview per map
    plot_full_spectra_overview(wn_caf2, sp_caf2_norm,
                               map_data, all_results, OUT_DIR)

    # 4. Text report
    print("\n")
    print_and_save_report(all_results, map_data, OUT_DIR)

    print(f"\nAll outputs saved to: {OUT_DIR}")


if __name__ == "__main__":
    main()
