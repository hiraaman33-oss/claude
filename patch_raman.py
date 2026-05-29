"""Patch raman_analysis.py: replace functions 11+12 with zoomed 3-panel versions."""
with open('/home/user/claude/raman_analysis.py', 'r', encoding='utf-8') as f:
    lines = f.readlines()

# Lines are 1-indexed; we replace lines 709..963 (inclusive) = indices 708..962
KEEP_BEFORE = 708   # keep lines[0:708]  (lines 1–708)
KEEP_AFTER  = 963   # keep lines[963:]   (lines 964 onward)

NEW_BLOCK = r'''
# -----------------------------------------------------------------------------
#  11.  PEAK DETECTION  (automatic, scipy.signal)
# -----------------------------------------------------------------------------
def _baseline_correct_display(wn, spectrum):
    """Baseline-correct a spectrum for display; returns (wn_region, bc_spectrum)."""
    mask  = (wn >= 395) & (wn <= 1810)
    wn_r  = wn[mask]
    sp_sg = savgol_filter(spectrum[mask], window_length=11, polyorder=3)
    bline = als_baseline(sp_sg, lam=1e6, p=0.005, n_iter=15)
    return wn_r, np.clip(sp_sg - bline, 0, None)


def detect_peaks(wn, spectrum,
                 distance_cm=45, height_pct=0.06, prominence_pct=0.05):
    """
    Find peaks in the ALS-baseline-corrected spectrum (400-1800 cm-1).
    Returns sorted array of peak wavenumbers.
    """
    from scipy.signal import find_peaks as _fp
    wn_r, sp_bc = _baseline_correct_display(wn, spectrum)
    if sp_bc.max() < 1e-6:
        return np.array([])
    step     = float(np.diff(wn_r).mean())
    dist_pts = max(3, int(distance_cm / step))
    idxs, _ = _fp(sp_bc,
                  height     = height_pct    * sp_bc.max(),
                  distance   = dist_pts,
                  prominence = prominence_pct * sp_bc.max())
    return wn_r[idxs]


def detect_peaks_caf2(wn, spectrum):
    """Stricter peak detection for CaF2 (few real Raman bands in 400-1800)."""
    return detect_peaks(wn, spectrum,
                        distance_cm=60, height_pct=0.12, prominence_pct=0.10)


# -----------------------------------------------------------------------------
#  12.  INDIVIDUAL SPECTRUM vs CaF2  --  3-PANEL ZOOMED FIGURES
# -----------------------------------------------------------------------------
_ZOOM_REGIONS = [(400, 750), (750, 1250), (1250, 1800)]
_ZOOM_LABELS  = ["400 - 750 cm⁻¹", "750 - 1250 cm⁻¹", "1250 - 1800 cm⁻¹"]


def _annotate_panel(ax, wn_m, sp_m, wn_c, sp_c,
                    peaks_map, peaks_caf2, lo, hi, peak_tol=22.0):
    """
    Add dotted vertical lines + wavenumber labels to one zoomed panel.
    Red bold '?' label  => peak in SERS but NOT in CaF2.
    Grey label          => peak shared with CaF2 (substrate/instrument).
    """
    y_m   = sp_m[(wn_m >= lo) & (wn_m <= hi)]
    y_c   = sp_c[(wn_c >= lo) & (wn_c <= hi)]
    ymax  = max(y_m.max() if y_m.size else 0, y_c.max() if y_c.size else 0)
    yspan = ymax if ymax > 0 else 1.0

    used_x = []

    # SERS peaks
    for wn_p in sorted(peaks_map):
        if not (lo <= wn_p <= hi):
            continue
        if any(abs(wn_p - u) < peak_tol * 0.6 for u in used_x):
            continue
        used_x.append(wn_p)

        in_caf2 = any(abs(wn_p - c) < peak_tol for c in peaks_caf2)
        color  = "#C0392B" if not in_caf2 else "#555555"
        weight = "bold"    if not in_caf2 else "normal"
        marker = "?\n"     if not in_caf2 else ""

        i_loc  = np.argmin(np.abs(wn_m - wn_p))
        y_peak = sp_m[i_loc]

        ax.axvline(wn_p, color=color, lw=0.9, ls="--", alpha=0.75)
        ax.text(wn_p, y_peak + yspan * 0.06,
                f"{marker}{int(round(wn_p))}",
                fontsize=9, ha="center", va="bottom",
                color=color, fontweight=weight,
                bbox=dict(boxstyle="round,pad=0.15", fc="white",
                          ec=color, lw=0.5, alpha=0.85))

    # CaF2-only peaks
    for wn_p in sorted(peaks_caf2):
        if not (lo <= wn_p <= hi):
            continue
        if any(abs(wn_p - u) < peak_tol * 0.6 for u in used_x):
            continue
        used_x.append(wn_p)

        i_loc  = np.argmin(np.abs(wn_c - wn_p))
        y_peak = sp_c[i_loc]

        ax.axvline(wn_p, color="#999999", lw=0.7, ls="--", alpha=0.60)
        ax.text(wn_p, y_peak + yspan * 0.04,
                f"{int(round(wn_p))}",
                fontsize=7.5, ha="center", va="bottom",
                color="#777777",
                bbox=dict(boxstyle="round,pad=0.10", fc="white",
                          ec="none", alpha=0.7))


def plot_spectrum_vs_caf2(wn_map, sp_map, wn_caf2, sp_caf2,
                          peaks_map, peaks_caf2,
                          spectrum_idx, out_dir, peak_tol=22.0):
    """
    Three-panel zoomed figure per map spectrum.

    Both spectra displayed baseline-corrected so Raman peaks rise from zero.
    Panel 1: 400-750 cm-1   Panel 2: 750-1250 cm-1   Panel 3: 1250-1800 cm-1
    Black = SERS.  Red = CaF2.  Red bold '?' = SERS-specific peak.
    """
    matplotlib.rcParams.update({"font.family": "DejaVu Sans",
                                 "figure.facecolor": "white",
                                 "axes.titlesize": 11,
                                 "axes.titleweight": "bold"})

    wn_m, sp_m = _baseline_correct_display(wn_map,  sp_map)
    wn_c, sp_c = _baseline_correct_display(wn_caf2, sp_caf2)

    fig, axes = plt.subplots(3, 1, figsize=(16, 13),
                             gridspec_kw={"hspace": 0.55,
                                          "left": 0.06, "right": 0.98,
                                          "top": 0.93, "bottom": 0.06})
    fig.suptitle(
        f"Spectrum #{spectrum_idx + 1}  vs  CaF₂ reference  "
        f"(ALS baseline-corrected)\n"
        f"Red bold ? = SERS-specific peak  |  Grey = shared with CaF₂",
        fontsize=12, fontweight="bold"
    )

    for ax, (lo, hi), rlabel in zip(axes, _ZOOM_REGIONS, _ZOOM_LABELS):
        m_m = (wn_m >= lo) & (wn_m <= hi)
        m_c = (wn_c >= lo) & (wn_c <= hi)
        y_m = sp_m[m_m];  y_c = sp_c[m_c]

        if y_m.size == 0 and y_c.size == 0:
            ax.set_visible(False); continue

        ax.fill_between(wn_m[m_m], y_m, alpha=0.13, color="black")
        ax.plot(wn_m[m_m], y_m, color="black",   lw=1.7,
                label=f"SERS #{spectrum_idx + 1}")
        ax.fill_between(wn_c[m_c], y_c, alpha=0.13, color="#C0392B")
        ax.plot(wn_c[m_c], y_c, color="#C0392B", lw=1.3, alpha=0.9,
                label="CaF₂ reference")

        ymax_local = max(y_m.max() if y_m.size else 0,
                         y_c.max() if y_c.size else 0)
        ax.set_ylim(-ymax_local * 0.05, ymax_local * 1.55)

        _annotate_panel(ax, wn_m, sp_m, wn_c, sp_c,
                        peaks_map, peaks_caf2, lo, hi, peak_tol)

        ax.set_xlim(lo, hi)
        ax.set_xlabel("Raman shift (cm⁻¹)", fontsize=10)
        ax.set_ylabel("Intensity (a.u.)", fontsize=10)
        ax.set_title(rlabel)
        ax.legend(fontsize=9, loc="upper right", framealpha=0.9)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(labelsize=9)
        ax.grid(True, alpha=0.15, lw=0.5)

    os.makedirs(out_dir, exist_ok=True)
    outfile = os.path.join(out_dir, f"spectrum_{spectrum_idx + 1:02d}_vs_caf2.png")
    fig.savefig(outfile, dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return outfile


def plot_all_individual(wn, spectra, wn_caf2, sp_caf2, out_dir):
    """
    One 3-panel zoomed figure per map spectrum + summary grid.
    """
    matplotlib.rcParams.update({"font.family": "DejaVu Sans",
                                 "figure.facecolor": "white"})

    print("\n  Detecting CaF₂ reference peaks (strict thresholds) ...")
    peaks_caf2 = detect_peaks_caf2(wn_caf2, sp_caf2)
    print(f"  CaF₂ peaks: {len(peaks_caf2)}  "
          f"[{', '.join(str(int(w)) for w in sorted(peaks_caf2))}]")

    n = len(spectra)
    print(f"\n  Plotting {n} individual spectra vs CaF₂ ...")

    all_peaks_map = []
    for i, sp in enumerate(spectra):
        pk = detect_peaks(wn, sp)
        all_peaks_map.append(pk)
        outfile = plot_spectrum_vs_caf2(
            wn, sp, wn_caf2, sp_caf2, pk, peaks_caf2, i, out_dir)
        sers_only = sum(1 for w in pk
                        if not any(abs(w - c) < 22 for c in peaks_caf2))
        print(f"    [{i+1:2d}/{n}]  {len(pk)} peaks  ({sers_only} SERS-specific)"
              f"  -> {os.path.basename(outfile)}")

    # Summary grid (baseline-corrected)
    ncols = 5
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(ncols * 4.0, nrows * 3.0),
                             sharex=True)
    fig.suptitle(
        "All SERS Map Spectra vs CaF₂  (baseline-corrected, 400-1800 cm⁻¹)\n"
        "Black = SERS  |  Red = CaF₂  |  red ? = SERS-specific",
        fontsize=11, fontweight="bold", y=1.01)

    wn_c_bc, sp_c_bc = _baseline_correct_display(wn_caf2, sp_caf2)
    m_c_g = (wn_c_bc >= 400) & (wn_c_bc <= 1800)

    for i in range(ncols * nrows):
        ax = list(axes.flat)[i]
        if i >= n:
            ax.set_visible(False); continue

        wn_m_bc, sp_m_bc = _baseline_correct_display(wn, spectra[i])
        m_m_g = (wn_m_bc >= 400) & (wn_m_bc <= 1800)

        ax.plot(wn_m_bc[m_m_g], sp_m_bc[m_m_g], color="black",   lw=0.9)
        ax.plot(wn_c_bc[m_c_g], sp_c_bc[m_c_g], color="#C0392B", lw=0.7, alpha=0.8)

        ymax_g = max(sp_m_bc[m_m_g].max(), sp_c_bc[m_c_g].max())

        for wn_p in all_peaks_map[i]:
            if not (400 <= wn_p <= 1800):
                continue
            in_caf2 = any(abs(wn_p - c) < 22 for c in peaks_caf2)
            color  = "#C0392B" if not in_caf2 else "#888888"
            marker = "?" if not in_caf2 else ""
            ax.axvline(wn_p, color=color, lw=0.5, ls=":", alpha=0.8)
            ax.text(wn_p, ymax_g * 1.05,
                    f"{marker}{int(round(wn_p))}",
                    fontsize=4.5, ha="center", va="bottom",
                    color=color, rotation=90)

        ax.set_xlim(400, 1800)
        ax.set_ylim(-ymax_g * 0.05, ymax_g * 1.6)
        ax.set_title(f"#{i + 1}", fontsize=8, pad=2)
        ax.tick_params(labelsize=6)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.tight_layout()
    grid_file = os.path.join(out_dir, "summary_grid_all_spectra.png")
    fig.savefig(grid_file, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"\n  Summary grid -> {grid_file}")

    # SERS-unique peak frequency table
    sers_unique = [wn_p for pk in all_peaks_map for wn_p in pk
                   if 400 <= wn_p <= 1800
                   and not any(abs(wn_p - c) < 22 for c in peaks_caf2)]
    if sers_unique:
        bins = np.arange(400, 1820, 25)
        counts, edges = np.histogram(np.array(sers_unique), bins=bins)
        hot = np.where(counts > 0)[0]
        print("\n  SERS-unique peak frequency  (cm-1 : n spectra):")
        for idx_b in sorted(hot, key=lambda x: -counts[x]):
            c  = int(counts[idx_b])
            wc = 0.5 * (edges[idx_b] + edges[idx_b + 1])
            print(f"    {wc:6.0f} cm-1 : {c:2d}  {'#' * c}")

    print(f"\n  All files saved to: {out_dir}\n")

'''

new_lines = (
    lines[:KEEP_BEFORE]
    + [NEW_BLOCK]
    + lines[KEEP_AFTER:]
)

with open('/home/user/claude/raman_analysis.py', 'w', encoding='utf-8') as f:
    f.writelines(new_lines)

print("Patched successfully.")
