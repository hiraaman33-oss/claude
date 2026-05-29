# =============================================================================
#  Raman Map Analysis Pipeline
#  File  : raman_analysis.py
#  Usage : Open in PyCharm and press  ▶  Run
#
#  Steps :
#   1.  Load LabSpec 6 .l6m map file  (custom binary parser)
#   2.  Compute the average Raman spectrum across all valid pixels
#   3.  Wavelet-based noise filtering  (PyWavelets, Daubechies db8)
#   4.  Min-Max normalisation  [0, 1]
#   5.  Standard Normal Variate (SNV) pre-processing
#   6.  Four-panel figure with every processing stage
#
#  Dependencies (install once in PyCharm terminal):
#   pip install numpy matplotlib PyWavelets scipy
# =============================================================================

import os
import struct
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import pywt

# ─────────────────────────────────────────────────────────────────────────────
# 0.  FILE PATH  –  change this to match your machine
# ─────────────────────────────────────────────────────────────────────────────
FILE_PATH = (
    r"C:\Users\Hira Aman\Desktop\RAMAN MAP 1"
    r"\m1DNAHACAT120ngAgSiNW_532nm_600gr_BC200_50XLF_05s_4a_25_5ul_dropcenter.l6m"
)

# ─────────────────────────────────────────────────────────────────────────────
# 1.  LOAD  .l6m  (LabSpec 6 binary format)
# ─────────────────────────────────────────────────────────────────────────────

def _find_wavenumber_axis(arr_f32):
    """
    Scan a float32 array for a monotonically-increasing linear ramp
    whose values fall in the Raman shift window (100 – 4000 cm⁻¹).
    Returns (start_index, length) of the axis, or (None, None).
    """
    n = len(arr_f32)
    i = 0
    while i < n - 50:
        chunk = arr_f32[i : i + 50]
        if not np.all(np.isfinite(chunk)):
            i += 1
            continue
        d = np.diff(chunk)
        if (np.all(d > 0)
                and 0.3 < float(d.mean()) < 10.0
                and 100.0 < float(chunk[0]) < 3800.0):
            # Extend rightward while the ramp continues
            end = i + 50
            while end < n and np.isfinite(arr_f32[end]):
                seg = arr_f32[i : end + 1]
                dd  = np.diff(seg)
                if np.all(dd > 0) and float(dd.std()) < 2.0:
                    end += 1
                else:
                    break
            return i, end - i
        i += 1
    return None, None


def load_l6m(filepath):
    """
    Parse a LabSpec 6 .l6m Raman map file.

    Returns
    -------
    wavenumbers : ndarray, shape (n_wn,)
        Raman shift axis in cm⁻¹.
    spectra : ndarray, shape (n_spectra, n_wn)
        Raw intensity matrix (counts).  NaN for invalid pixels.
    """
    if not os.path.isfile(filepath):
        raise FileNotFoundError(f"File not found:\n  {filepath}")

    with open(filepath, "rb") as fh:
        raw = fh.read()

    if not raw[:8].startswith(b"LabSpec"):
        raise ValueError("Not a LabSpec 6 file (bad magic bytes).")

    # ── Align to 4-byte boundary and view as float32 ─────────────────────
    n_aligned = (len(raw) // 4) * 4
    arr = np.frombuffer(raw[:n_aligned], dtype=np.float32).copy()

    # ── Locate the wavenumber axis ────────────────────────────────────────
    wn_idx, n_wn = _find_wavenumber_axis(arr)
    if wn_idx is None:
        raise RuntimeError(
            "Could not locate the wavenumber axis in the file.\n"
            "Please contact the developer or export the data as CSV from LabSpec 6."
        )
    wn_byte   = wn_idx * 4
    wavenumbers = arr[wn_idx : wn_idx + n_wn].astype(np.float64)

    # ── Locate spectral data block ────────────────────────────────────────
    #   The intensity block directly precedes the wavenumber axis.
    #   n_spectra × n_wn float32 values end at wn_byte.
    total_intensity_floats = wn_idx          # floats before the wn axis
    n_spectra = total_intensity_floats // n_wn
    leftover  = total_intensity_floats  % n_wn

    if n_spectra == 0:
        raise RuntimeError(
            f"Unexpected layout: {total_intensity_floats} floats before the "
            f"wavenumber axis ({n_wn} pts) gives 0 complete spectra."
        )

    data_start_idx = leftover             # skip any leading partial data
    data_end_idx   = data_start_idx + n_spectra * n_wn

    spectra_raw = arr[data_start_idx:data_end_idx].reshape(n_spectra, n_wn).astype(np.float64)

    # Mark clearly non-physical values as NaN
    MAX_REALISTIC_COUNTS = 1e7
    spectra_raw[~np.isfinite(spectra_raw)]        = np.nan
    spectra_raw[spectra_raw > MAX_REALISTIC_COUNTS] = np.nan
    spectra_raw[spectra_raw < 0]                  = np.nan

    print(f"\n{'─'*55}")
    print(f"  File    : {os.path.basename(filepath)}")
    print(f"  Spectra : {n_spectra} pixels × {n_wn} wavenumber points")
    print(f"  Range   : {wavenumbers[0]:.1f} – {wavenumbers[-1]:.1f} cm⁻¹")
    step = float(np.diff(wavenumbers).mean())
    print(f"  Step    : {step:.3f} cm⁻¹/channel")
    n_bad = int(np.sum(np.any(np.isnan(spectra_raw), axis=1)))
    print(f"  Bad px  : {n_bad} (will be excluded from average)")
    print(f"{'─'*55}\n")

    return wavenumbers, spectra_raw


# ─────────────────────────────────────────────────────────────────────────────
# 2.  AVERAGE SPECTRUM
# ─────────────────────────────────────────────────────────────────────────────

def compute_average_spectrum(spectra):
    """
    Mean over the spatial (pixel) axis, ignoring NaN values.

    Returns
    -------
    avg : ndarray, shape (n_wn,)
    n_valid : int   – number of pixels that contributed to each channel
    """
    avg     = np.nanmean(spectra, axis=0)
    n_valid = int(np.sum(np.all(np.isfinite(spectra), axis=1)))
    print(f"  Average computed from {n_valid} / {spectra.shape[0]} valid spectra.")
    return avg, n_valid


# ─────────────────────────────────────────────────────────────────────────────
# 3.  WAVELET-BASED NOISE FILTERING
# ─────────────────────────────────────────────────────────────────────────────

def wavelet_denoise(spectrum,
                    wavelet    = "db8",
                    level      = None,
                    mode       = "soft",
                    threshold_rule = "universal"):
    """
    Remove high-frequency noise via discrete wavelet thresholding.

    Parameters
    ----------
    spectrum        : 1-D array of intensities
    wavelet         : PyWavelets wavelet name  (default: Daubechies-8)
    level           : decomposition depth; None → automatic (log₂ based)
    mode            : 'soft' (smoother) or 'hard' thresholding
    threshold_rule  : 'universal'  σ√(2 ln N)   (global noise estimate)
                      'bayes'      level-adaptive  (better for mixed noise)

    Returns
    -------
    denoised : 1-D array, same length as spectrum
    """
    n = len(spectrum)
    if level is None:
        level = min(pywt.dwt_max_level(n, wavelet), 6)

    # Decompose
    coeffs = pywt.wavedec(spectrum, wavelet, level=level)

    # Estimate noise σ from the finest detail coefficients (MAD estimator)
    finest = coeffs[-1]
    sigma  = np.median(np.abs(finest)) / 0.6745      # robust σ estimate

    # Threshold each detail sub-band
    thresholded = [coeffs[0]]                        # keep approximation unchanged
    for j, c in enumerate(coeffs[1:], start=1):
        if threshold_rule == "universal":
            # Donoho-Johnstone universal threshold
            thr = sigma * np.sqrt(2.0 * np.log(n))
        else:
            # Level-adaptive (BayesShrink approximation)
            sigma_j = np.median(np.abs(c)) / 0.6745
            thr     = sigma_j ** 2 / max(sigma, 1e-10)

        thresholded.append(pywt.threshold(c, thr, mode=mode))

    denoised = pywt.waverec(thresholded, wavelet)

    # waverec can add one extra sample; trim to original length
    return denoised[:n]


# ─────────────────────────────────────────────────────────────────────────────
# 4.  MIN-MAX NORMALISATION
# ─────────────────────────────────────────────────────────────────────────────

def minmax_normalize(spectrum):
    """
    Scale every intensity to [0, 1].

        x_norm = (x − min) / (max − min)

    This makes spectra from different measurements directly comparable
    in terms of relative band heights.
    """
    lo, hi = spectrum.min(), spectrum.max()
    if np.isclose(hi, lo):
        return np.zeros_like(spectrum)
    return (spectrum - lo) / (hi - lo)


# ─────────────────────────────────────────────────────────────────────────────
# 5.  STANDARD NORMAL VARIATE  (SNV)
# ─────────────────────────────────────────────────────────────────────────────

def snv(spectrum):
    """
    Standard Normal Variate pre-processing.

        x_snv = (x − mean(x)) / std(x)

    Centres the spectrum to zero mean and scales it to unit variance.
    Corrects for multiplicative scatter and baseline differences that
    arise from sample-to-sample path-length or particle-size variations.
    """
    mu  = np.mean(spectrum)
    std = np.std(spectrum)
    if np.isclose(std, 0.0):
        return np.zeros_like(spectrum)
    return (spectrum - mu) / std


# ─────────────────────────────────────────────────────────────────────────────
# 6.  PLOTTING
# ─────────────────────────────────────────────────────────────────────────────

def plot_pipeline(wavenumbers, avg_raw, avg_denoised, avg_norm, avg_snv, n_valid):
    """Four-panel figure showing every stage of the processing pipeline."""

    fig = plt.figure(figsize=(14, 10))
    fig.suptitle(
        "Raman Map Analysis Pipeline\n"
        "m1DNAHACAT120ng AgSiNW | 532 nm | 600 gr/mm | 50×LF | 0.5 s",
        fontsize=13, fontweight="bold", y=0.98
    )

    gs   = gridspec.GridSpec(2, 2, hspace=0.42, wspace=0.32)
    axes = [fig.add_subplot(gs[r, c]) for r in range(2) for c in range(2)]

    panels = [
        (avg_raw,      "Step 1 – Average raw spectrum",
         f"Average of {n_valid} spectra", "Intensity (counts)", "tab:blue"),
        (avg_denoised, "Step 2 – Wavelet denoised  (db8, universal threshold)",
         "Soft-thresholded DWT coefficients", "Intensity (counts)", "tab:green"),
        (avg_norm,     "Step 3 – Min-Max normalised",
         "x_norm = (x − min) / (max − min)", "Normalised intensity (a.u.)", "tab:orange"),
        (avg_snv,      "Step 4 – SNV pre-processed",
         "x_snv = (x − μ) / σ   |   removes scatter differences", "SNV units (σ)", "tab:red"),
    ]

    for ax, (y, title, subtitle, ylabel, colour) in zip(axes, panels):
        ax.plot(wavenumbers, y, color=colour, lw=1.0, alpha=0.9)
        ax.set_title(title, fontsize=10, fontweight="bold", pad=6)
        ax.set_xlabel("Raman shift (cm⁻¹)", fontsize=9)
        ax.set_ylabel(ylabel, fontsize=9)
        ax.text(0.02, 0.97, subtitle,
                transform=ax.transAxes, fontsize=7.5, va="top",
                color="dimgray", style="italic")
        ax.axhline(0, color="black", lw=0.4, ls="--", alpha=0.4)
        ax.tick_params(labelsize=8)
        ax.set_xlim(wavenumbers[0], wavenumbers[-1])
        ax.grid(True, alpha=0.25, lw=0.5)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig("raman_pipeline_output.png", dpi=150, bbox_inches="tight")
    print("  Figure saved → raman_pipeline_output.png")
    plt.show()


def plot_overlay(wavenumbers, avg_denoised, avg_norm, avg_snv):
    """
    Overlay of denoised, normalised, and SNV spectra on a single axis
    (useful for direct visual comparison of shapes).
    """
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(wavenumbers, avg_denoised / avg_denoised.max(),
            label="Denoised (scaled to 1)", color="steelblue", lw=1.2, alpha=0.85)
    ax.plot(wavenumbers, avg_norm,
            label="Min-Max normalised", color="darkorange", lw=1.2, alpha=0.85)
    # SNV can be negative; shift it for visual overlay
    snv_shifted = (avg_snv - avg_snv.min()) / (avg_snv.max() - avg_snv.min())
    ax.plot(wavenumbers, snv_shifted,
            label="SNV (shifted to [0,1] for display)", color="firebrick", lw=1.2, alpha=0.85)
    ax.set_xlabel("Raman shift (cm⁻¹)", fontsize=11)
    ax.set_ylabel("Relative intensity", fontsize=11)
    ax.set_title("Spectral comparison after each pre-processing stage", fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(wavenumbers[0], wavenumbers[-1])
    plt.tight_layout()
    plt.savefig("raman_overlay_comparison.png", dpi=150, bbox_inches="tight")
    print("  Overlay figure saved → raman_overlay_comparison.png")
    plt.show()


# ─────────────────────────────────────────────────────────────────────────────
# 7.  MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("\n" + "="*55)
    print("  RAMAN MAP ANALYSIS PIPELINE")
    print("="*55)

    # ── 1. Load ──────────────────────────────────────────────────────────
    wavenumbers, spectra = load_l6m(FILE_PATH)

    # ── 2. Average ───────────────────────────────────────────────────────
    print("Step 2 – Computing average spectrum …")
    avg_raw, n_valid = compute_average_spectrum(spectra)

    # ── 3. Wavelet denoise ───────────────────────────────────────────────
    print("Step 3 – Wavelet denoising …")
    avg_denoised = wavelet_denoise(
        avg_raw,
        wavelet        = "db8",     # Daubechies-8: good for smooth Raman peaks
        level          = 5,         # 5 levels of decomposition
        mode           = "soft",    # soft thresholding → smoother result
        threshold_rule = "universal"
    )

    # ── 4. Normalise ─────────────────────────────────────────────────────
    print("Step 4 – Min-Max normalising …")
    avg_norm = minmax_normalize(avg_denoised)

    # ── 5. SNV ───────────────────────────────────────────────────────────
    print("Step 5 – SNV pre-processing …")
    avg_snv = snv(avg_norm)

    # ── 6. Print summary ─────────────────────────────────────────────────
    print(f"\n{'─'*55}")
    print("  Summary of processed average spectrum")
    print(f"{'─'*55}")
    print(f"  Raw       : mean={avg_raw.mean():.2f},  max={avg_raw.max():.2f}")
    print(f"  Denoised  : mean={avg_denoised.mean():.2f},  max={avg_denoised.max():.2f}")
    print(f"  Normalised: min={avg_norm.min():.4f},  max={avg_norm.max():.4f}")
    print(f"  SNV       : mean={avg_snv.mean():.4f},  std={avg_snv.std():.4f}")
    print(f"{'─'*55}\n")

    # ── 7. Save processed spectrum to CSV ────────────────────────────────
    output_csv = "raman_processed_spectrum.csv"
    header = "wavenumber_cm-1,raw_avg,denoised,normalized,snv"
    out_arr = np.column_stack([wavenumbers, avg_raw, avg_denoised, avg_norm, avg_snv])
    np.savetxt(output_csv, out_arr, delimiter=",", header=header, comments="")
    print(f"  Processed spectrum saved → {output_csv}")

    # ── 8. Plot ──────────────────────────────────────────────────────────
    print("  Generating figures …")
    plot_pipeline(wavenumbers, avg_raw, avg_denoised, avg_norm, avg_snv, n_valid)
    plot_overlay(wavenumbers, avg_denoised, avg_norm, avg_snv)

    print("\n  Done.\n")


if __name__ == "__main__":
    main()
