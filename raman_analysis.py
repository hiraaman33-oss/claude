# =============================================================================
#  Raman Map Analysis Pipeline  —  v2  (spike removal + baseline correction)
#  File  : raman_analysis.py
#  Usage : Open in PyCharm and press  ▶  Run
#
#  Pipeline:
#   1.  Load LabSpec 6 .l6m map file
#   2.  Per-spectrum cosmic-ray / spike removal  (modified Z-score, 2nd deriv)
#   3.  Average all cleaned spectra
#   4.  Asymmetric Least Squares (ALS) baseline correction
#   5.  Wavelet denoising  (db8, universal threshold)
#   6.  Min-Max normalisation  →  [0, 1]
#   7.  Standard Normal Variate (SNV)
#   8.  Publication-quality 6-panel figure  +  zoomed fingerprint with labels
#
#  Install once (PyCharm terminal):
#   pip install numpy matplotlib PyWavelets scipy
# =============================================================================

import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyArrowPatch
import pywt
from scipy import sparse
from scipy.sparse.linalg import spsolve
from scipy.signal import savgol_filter

# ─────────────────────────────────────────────────────────────────────────────
#  FILE PATH
# ─────────────────────────────────────────────────────────────────────────────
FILE_PATH = (
    r"C:\Users\Hira Aman\Desktop\RAMAN MAP 1"
    r"\m1-DNA-HACAT1-20ng-AgSiNW_532nm_600gr_BC200_50XLF_05s_4a_2-5%_5ul_drop-center.l6m"
)

# ─────────────────────────────────────────────────────────────────────────────
#  KNOWN RAMAN / SERS PEAKS  (DNA + HACAT + AgNW, 532 nm excitation)
# ─────────────────────────────────────────────────────────────────────────────
PEAK_LABELS = {
    382:  "Ag–N",
    521:  "Si",
    662:  "G (ring)",
    785:  "DNA backbone",
    1000: "Phe",
    1090: "PO₄⁻",
    1175: "C–H bend",
    1340: "G (C–N)",
    1484: "A/C (C=N)",
    1575: "G/A (ring)",
    2850: "CH₂ sym",
    2930: "CH₂ asym",
    3060: "ArC–H",
    3130: "C–H str",
}

# ─────────────────────────────────────────────────────────────────────────────
#  1.  LOAD  .l6m
# ─────────────────────────────────────────────────────────────────────────────
def _find_wn_axis(arr):
    """Scan float32 array for monotonically-increasing Raman shift axis."""
    i = 0
    while i < len(arr) - 50:
        chunk = arr[i:i+50]
        if not np.all(np.isfinite(chunk)):
            i += 1; continue
        d = np.diff(chunk)
        if np.all(d > 0) and 0.3 < float(d.mean()) < 10.0 and 100 < float(chunk[0]) < 3800:
            end = i + 50
            while end < len(arr) and np.isfinite(arr[end]):
                seg = arr[i:end+1]
                dd  = np.diff(seg)
                if np.all(dd > 0) and float(dd.std()) < 2.0:
                    end += 1
                else:
                    break
            return i, end - i
        i += 1
    return None, None


def load_l6m(filepath):
    if not os.path.isfile(filepath):
        raise FileNotFoundError(f"File not found:\n  {filepath}")
    with open(filepath, "rb") as fh:
        raw = fh.read()
    if not raw[:8].startswith(b"LabSpec"):
        raise ValueError("Not a LabSpec 6 file.")
    n_aligned = (len(raw) // 4) * 4
    arr = np.frombuffer(raw[:n_aligned], dtype=np.float32).copy()

    wn_idx, n_wn = _find_wn_axis(arr)
    if wn_idx is None:
        raise RuntimeError("Could not locate wavenumber axis.")
    wavenumbers = arr[wn_idx:wn_idx+n_wn].astype(np.float64)

    total_floats = wn_idx
    n_spectra    = total_floats // n_wn
    leftover     = total_floats  % n_wn
    spectra_raw  = arr[leftover:leftover + n_spectra*n_wn].reshape(n_spectra, n_wn).astype(np.float64)

    spectra_raw[~np.isfinite(spectra_raw)]      = np.nan
    spectra_raw[spectra_raw > 1e7]              = np.nan
    spectra_raw[spectra_raw < 0]                = np.nan

    print(f"\n{'─'*55}")
    print(f"  Spectra : {n_spectra} pixels × {n_wn} points")
    print(f"  Range   : {wavenumbers[0]:.1f} – {wavenumbers[-1]:.1f} cm⁻¹")
    print(f"  Step    : {np.diff(wavenumbers).mean():.3f} cm⁻¹/ch")
    print(f"{'─'*55}\n")
    return wavenumbers, spectra_raw


# ─────────────────────────────────────────────────────────────────────────────
#  2.  SPIKE REMOVAL  (per spectrum, before averaging)
# ─────────────────────────────────────────────────────────────────────────────
def remove_spikes(spectrum, threshold=5.0, window=3):
    """
    Detect and interpolate over cosmic-ray spikes.

    Algorithm (Whitaker & Hayes 2018, adapted):
      • Compute the modified Z-score of the second derivative.
      • Points exceeding `threshold` are flagged as spikes.
      • Flagged points are replaced by linear interpolation from
        their nearest non-spike neighbours.
    """
    sp = spectrum.copy()
    x  = np.arange(len(sp))

    # Second derivative
    d2     = np.concatenate([[0], np.diff(np.diff(sp)), [0]])
    median = np.median(d2)
    mad    = np.median(np.abs(d2 - median))
    mz     = 0.6745 * np.abs(d2 - median) / (mad + 1e-10)

    spike_mask = mz > threshold

    # Dilate mask by `window` to catch shoulder pixels
    from scipy.ndimage import binary_dilation
    spike_mask = binary_dilation(spike_mask, iterations=window)

    if spike_mask.any() and (~spike_mask).sum() > 10:
        sp[spike_mask] = np.interp(
            x[spike_mask], x[~spike_mask], sp[~spike_mask]
        )
    return sp, spike_mask


def clean_all_spectra(spectra, wn, threshold=5.0):
    """Apply spike removal to every valid spectrum."""
    cleaned = []
    n_total_spikes = 0
    for i, sp in enumerate(spectra):
        if not np.isfinite(sp).all():
            continue
        sp_clean, mask = remove_spikes(sp, threshold=threshold)
        n_total_spikes += int(mask.sum())
        cleaned.append(sp_clean)
    print(f"  Spike removal: {len(cleaned)} valid spectra, "
          f"{n_total_spikes} spike points removed total.")
    return np.array(cleaned)


# ─────────────────────────────────────────────────────────────────────────────
#  3.  AVERAGE
# ─────────────────────────────────────────────────────────────────────────────
def compute_average(spectra):
    avg = np.mean(spectra, axis=0)
    print(f"  Average computed from {len(spectra)} spectra.")
    return avg


# ─────────────────────────────────────────────────────────────────────────────
#  4.  BASELINE CORRECTION  —  Asymmetric Least Squares (ALS)
# ─────────────────────────────────────────────────────────────────────────────
def als_baseline(y, lam=1e6, p=0.005, n_iter=15):
    """
    Asymmetric Least Squares baseline estimator (Eilers & Boelens 2005).

    Parameters
    ----------
    lam    : smoothness of the baseline (larger → smoother)
    p      : asymmetry (small p → baseline stays below peaks)
    n_iter : number of re-weighting iterations
    """
    L  = len(y)
    # Second-difference matrix  (L-2 x L)
    D  = sparse.diags([1, -2, 1], [0, 1, 2], shape=(L-2, L), dtype=float)
    DT = D.T                              # (L x L-2)
    w  = np.ones(L)
    for _ in range(n_iter):
        W   = sparse.diags(w, 0, shape=(L, L), dtype=float)
        Z   = W + lam * DT.dot(D)        # (L x L)
        z   = spsolve(Z.tocsr(), w * y)
        w   = p * (y > z) + (1 - p) * (y <= z)
    return z


def subtract_baseline(spectrum):
    baseline = als_baseline(spectrum)
    corrected = spectrum - baseline
    corrected = np.clip(corrected, 0, None)   # no negative intensities
    return corrected, baseline


# ─────────────────────────────────────────────────────────────────────────────
#  5.  WAVELET DENOISING
# ─────────────────────────────────────────────────────────────────────────────
def wavelet_denoise(spectrum, wavelet="db8", level=5, mode="soft"):
    n      = len(spectrum)
    coeffs = pywt.wavedec(spectrum, wavelet, level=min(level, pywt.dwt_max_level(n, wavelet)))
    sigma  = np.median(np.abs(coeffs[-1])) / 0.6745
    thr    = sigma * np.sqrt(2.0 * np.log(n))
    coeffs_t = [coeffs[0]] + [pywt.threshold(c, thr, mode) for c in coeffs[1:]]
    return pywt.waverec(coeffs_t, wavelet)[:n]


# ─────────────────────────────────────────────────────────────────────────────
#  6.  MIN-MAX NORMALISATION
# ─────────────────────────────────────────────────────────────────────────────
def minmax_normalize(spectrum):
    lo, hi = spectrum.min(), spectrum.max()
    if np.isclose(hi, lo):
        return np.zeros_like(spectrum)
    return (spectrum - lo) / (hi - lo)


# ─────────────────────────────────────────────────────────────────────────────
#  7.  SNV
# ─────────────────────────────────────────────────────────────────────────────
def snv(spectrum):
    mu  = np.mean(spectrum)
    std = np.std(spectrum)
    if np.isclose(std, 0.0):
        return np.zeros_like(spectrum)
    return (spectrum - mu) / std


# ─────────────────────────────────────────────────────────────────────────────
#  8.  FIGURE  —  publication quality
# ─────────────────────────────────────────────────────────────────────────────

STYLE = {
    "raw"      : ("#7B7B7B", 0.35, 0.8),   # (colour, alpha, lw)
    "avg_raw"  : ("#C0392B", 1.0,  1.8),
    "cleaned"  : ("#2980B9", 1.0,  1.6),
    "baseline" : ("#E67E22", 1.0,  1.4),
    "denoised" : ("#27AE60", 1.0,  2.0),
    "norm"     : ("#8E44AD", 1.0,  2.0),
    "snv"      : ("#1A252F", 1.0,  2.0),
}


def _axis_style(ax, xlim=None, ylabel="Intensity", grid=True):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_xlabel("Raman shift (cm⁻¹)", fontsize=10)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.tick_params(labelsize=9)
    if xlim:
        ax.set_xlim(xlim)
    if grid:
        ax.grid(True, alpha=0.2, lw=0.5)


def _annotate_peaks(ax, wn, spectrum, peak_dict, tol=25, min_height=0.05):
    """Draw vertical dashed lines + labels for known peaks that appear in spectrum."""
    ymax = spectrum.max()
    used_x = []
    for center, label in sorted(peak_dict.items()):
        mask = (wn >= center - tol) & (wn <= center + tol)
        if not mask.any():
            continue
        local_max_idx = np.argmax(spectrum[mask])
        local_wn  = wn[mask][local_max_idx]
        local_int = spectrum[mask][local_max_idx]
        if local_int < min_height * ymax:
            continue
        # avoid crowding
        if any(abs(local_wn - u) < 60 for u in used_x):
            continue
        used_x.append(local_wn)
        ax.axvline(local_wn, color="gray", lw=0.8, ls="--", alpha=0.6)
        ax.text(local_wn, local_int * 1.04, f"{int(local_wn)}\n{label}",
                fontsize=7.5, ha="center", va="bottom",
                color="#2C3E50", rotation=0,
                bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.7))


def plot_full_pipeline(wn, raw_spectra, cleaned_spectra,
                       avg_raw, avg_cleaned, baseline,
                       avg_bc, avg_denoised, avg_norm, avg_snv):
    """Six-panel figure covering every processing stage."""

    matplotlib.rcParams.update({
        "font.family": "DejaVu Sans",
        "axes.titlesize": 11,
        "axes.titleweight": "bold",
        "figure.facecolor": "white",
    })

    fig = plt.figure(figsize=(18, 13))
    fig.suptitle(
        "SERS Raman Map — Complete Pre-processing Pipeline\n"
        "DNA + HACAT cells on AgSiNW  |  532 nm  |  600 gr/mm  |  50×LF  |  0.5 s  |  4 acc.",
        fontsize=13, fontweight="bold", y=0.99
    )

    gs = gridspec.GridSpec(3, 2, hspace=0.55, wspace=0.32,
                           left=0.07, right=0.97, top=0.93, bottom=0.06)
    axes = [fig.add_subplot(gs[r, c]) for r in range(3) for c in range(2)]

    # ── Panel 1: All raw spectra + average ──────────────────────────────
    ax = axes[0]
    for sp in raw_spectra:
        ax.plot(wn, sp, color=STYLE["raw"][0], alpha=STYLE["raw"][1],
                lw=STYLE["raw"][2])
    ax.plot(wn, avg_raw, color=STYLE["avg_raw"][0], lw=STYLE["avg_raw"][2],
            label=f"Average (n={len(raw_spectra)})")
    ax.set_title("① Raw spectra  (all map pixels)")
    ax.legend(fontsize=9, loc="upper right")
    _axis_style(ax, ylabel="Counts")

    # ── Panel 2: Before/after spike removal ─────────────────────────────
    ax = axes[1]
    ax.plot(wn, avg_raw,     color=STYLE["avg_raw"][0], lw=1.4,
            alpha=0.55, ls="--", label="Before spike removal")
    ax.plot(wn, avg_cleaned, color=STYLE["cleaned"][0], lw=STYLE["cleaned"][2],
            label="After spike removal")
    ax.set_title("② Cosmic-ray / spike removal  (modified Z-score, 2nd deriv.)")
    ax.legend(fontsize=9, loc="upper right")
    _axis_style(ax, ylabel="Counts")

    # ── Panel 3: Baseline + corrected ───────────────────────────────────
    ax = axes[2]
    ax.plot(wn, avg_cleaned,  color=STYLE["avg_raw"][0],   lw=1.4,
            alpha=0.6, label="Spike-cleaned average")
    ax.plot(wn, baseline,     color=STYLE["baseline"][0],  lw=1.8,
            ls="--", label="ALS baseline")
    ax.plot(wn, avg_bc,       color=STYLE["denoised"][0],  lw=1.8,
            label="Baseline-corrected")
    ax.set_title("③ Asymmetric Least Squares (ALS) baseline correction")
    ax.legend(fontsize=9, loc="upper right")
    _axis_style(ax, ylabel="Counts")

    # ── Panel 4: Denoised — full range ──────────────────────────────────
    ax = axes[3]
    ax.plot(wn, avg_bc,       color=STYLE["denoised"][0], lw=1.2,
            alpha=0.55, ls="--", label="Before denoising")
    ax.plot(wn, avg_denoised, color=STYLE["norm"][0],     lw=2.0,
            label="After wavelet denoising (db8, L=5)")
    ax.set_title("④ Wavelet denoising  (Daubechies-8, soft universal threshold)")
    ax.legend(fontsize=9, loc="upper right")
    _axis_style(ax, ylabel="Counts")

    # ── Panel 5: Normalised — fingerprint region ZOOMED with labels ─────
    ax = axes[4]
    fp_lo, fp_hi = 400, 1800
    mask_fp = (wn >= fp_lo) & (wn <= fp_hi)
    ax.fill_between(wn[mask_fp], avg_norm[mask_fp],
                    alpha=0.18, color=STYLE["norm"][0])
    ax.plot(wn[mask_fp], avg_norm[mask_fp],
            color=STYLE["norm"][0], lw=2.2, label="Min-Max normalised")
    _annotate_peaks(ax, wn[mask_fp], avg_norm[mask_fp], PEAK_LABELS,
                    tol=30, min_height=0.08)
    ax.set_title("⑤ Min-Max normalised  —  fingerprint region  (400–1800 cm⁻¹)")
    ax.legend(fontsize=9, loc="upper right")
    _axis_style(ax, xlim=(fp_lo, fp_hi), ylabel="Normalised intensity (a.u.)")

    # ── Panel 6: SNV — fingerprint region ZOOMED with labels ────────────
    ax = axes[5]
    ax.fill_between(wn[mask_fp], avg_snv[mask_fp],
                    alpha=0.18, color=STYLE["snv"][0])
    ax.plot(wn[mask_fp], avg_snv[mask_fp],
            color=STYLE["snv"][0], lw=2.2, label="SNV spectrum")
    ax.axhline(0, color="black", lw=0.6, ls="--", alpha=0.4)
    _annotate_peaks(ax, wn[mask_fp], avg_snv[mask_fp], PEAK_LABELS,
                    tol=30, min_height=0.08)
    ax.set_title("⑥ Standard Normal Variate (SNV)  —  fingerprint region")
    ax.legend(fontsize=9, loc="upper right")
    _axis_style(ax, xlim=(fp_lo, fp_hi), ylabel="SNV units (σ)")

    plt.savefig("raman_pipeline_output.png", dpi=180, bbox_inches="tight",
                facecolor="white")
    print("  Saved → raman_pipeline_output.png")
    plt.show()


def plot_final_zoomed(wn, avg_norm, avg_snv):
    """
    Large two-panel figure of the final spectrum zoomed into
    fingerprint (400–1800 cm⁻¹) and C-H stretch (2700–3200 cm⁻¹).
    """
    matplotlib.rcParams.update({"font.family": "DejaVu Sans"})

    fig, axes = plt.subplots(1, 2, figsize=(16, 6),
                             gridspec_kw={"wspace": 0.35})
    fig.suptitle(
        "Final SNV-pre-processed Raman Spectrum  —  Key Peak Regions",
        fontsize=14, fontweight="bold"
    )

    regions = [
        (axes[0], 400,  1800, "Fingerprint region  (400–1800 cm⁻¹)"),
        (axes[1], 2700, 3200, "C–H stretch region  (2700–3200 cm⁻¹)"),
    ]

    for ax, lo, hi, title in regions:
        mask = (wn >= lo) & (wn <= hi)
        y    = avg_snv[mask]
        x    = wn[mask]

        ax.fill_between(x, y, y.min(), alpha=0.15, color="#1A252F")
        ax.plot(x, y, color="#1A252F", lw=2.2)
        _annotate_peaks(ax, x, y, PEAK_LABELS, tol=30, min_height=0.05)
        ax.set_title(title, fontsize=11, fontweight="bold", pad=8)
        _axis_style(ax, xlim=(lo, hi), ylabel="SNV units (σ)")

    plt.savefig("raman_final_zoomed.png", dpi=180, bbox_inches="tight",
                facecolor="white")
    print("  Saved → raman_final_zoomed.png")
    plt.show()


# ─────────────────────────────────────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────────────────────────────────────
def main():
    print("\n" + "="*55)
    print("  RAMAN MAP ANALYSIS PIPELINE  v2")
    print("="*55)

    # 1. Load
    wn, spectra = load_l6m(FILE_PATH)

    # Keep only fully-finite spectra for processing
    raw_spectra = np.array([spectra[i] for i in range(len(spectra))
                            if np.isfinite(spectra[i]).all()
                            and spectra[i].max() < 1e6])
    print(f"  Valid spectra loaded: {len(raw_spectra)}")

    avg_raw = np.mean(raw_spectra, axis=0)

    # 2. Spike removal (per spectrum)
    print("Step 2 – Spike removal …")
    cleaned_spectra = clean_all_spectra(raw_spectra, wn, threshold=5.0)
    avg_cleaned = np.mean(cleaned_spectra, axis=0)

    # 3. Average already computed above; smooth with Savitzky-Golay first
    #    to stabilise ALS
    avg_sg = savgol_filter(avg_cleaned, window_length=11, polyorder=3)

    # 4. Baseline correction
    print("Step 3 – ALS baseline correction …")
    avg_bc_raw, baseline = subtract_baseline(avg_sg)

    # 5. Wavelet denoise
    print("Step 4 – Wavelet denoising …")
    avg_denoised = wavelet_denoise(avg_bc_raw, wavelet="db8", level=5, mode="soft")
    avg_denoised = np.clip(avg_denoised, 0, None)

    # 6. Normalise
    print("Step 5 – Min-Max normalising …")
    avg_norm = minmax_normalize(avg_denoised)

    # 7. SNV
    print("Step 6 – SNV …")
    avg_snv = snv(avg_norm)

    # Summary
    print(f"\n{'─'*55}")
    print(f"  Raw avg    : max={avg_raw.max():.1f}  mean={avg_raw.mean():.1f}")
    print(f"  After spikes: max={avg_cleaned.max():.1f}")
    print(f"  After baseline: max={avg_bc_raw.max():.1f}")
    print(f"  Denoised   : max={avg_denoised.max():.1f}")
    print(f"  Normalised : max={avg_norm.max():.4f}  min={avg_norm.min():.4f}")
    print(f"  SNV        : mean={avg_snv.mean():.4f}  std={avg_snv.std():.4f}")
    print(f"{'─'*55}\n")

    # Save CSV
    header = "wavenumber_cm-1,raw_avg,spike_removed,baseline_corrected,denoised,normalized,snv"
    out_arr = np.column_stack([wn, avg_raw, avg_cleaned, avg_bc_raw,
                               avg_denoised, avg_norm, avg_snv])
    np.savetxt("raman_processed_spectrum.csv", out_arr,
               delimiter=",", header=header, comments="")
    print("  CSV saved → raman_processed_spectrum.csv")

    # Figures
    print("  Generating figures …")
    plot_full_pipeline(wn, raw_spectra, cleaned_spectra,
                       avg_raw, avg_cleaned, baseline,
                       avg_bc_raw, avg_denoised, avg_norm, avg_snv)
    plot_final_zoomed(wn, avg_norm, avg_snv)
    print("\n  Done.\n")


if __name__ == "__main__":
    main()
