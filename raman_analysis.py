# =============================================================================
#  Raman Map Analysis Pipeline  —  v3  (spike removal + baseline + comparison)
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
#   9.  Load LabSpec 6 .l6s single-spectrum file and apply same pipeline
#  10.  Comparison plot: map-average vs single spectrum, peak-to-peak
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
#  FILE PATHS
# ─────────────────────────────────────────────────────────────────────────────
FILE_PATH = (
    r"C:\Users\Hira Aman\Desktop\RAMAN MAP 1"
    r"\m1-DNA-HACAT1-20ng-AgSiNW_532nm_600gr_BC200_50XLF_05s_4a_2-5%_5ul_drop-center.l6m"
)

FILE_PATH_L6S = (
    r"C:\Users\Hira Aman\Desktop\RAMAN MAP 1"
    r"\S1-DNA-HACAT1-20ng-CaF2_532nm_600gr_BC50_100X_10s_4a_100%.l6s"
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
#  1b.  LOAD  .l6s  (single spectrum, LabSpec 6)
# ─────────────────────────────────────────────────────────────────────────────
def load_l6s(filepath):
    """
    Load a LabSpec 6 single-spectrum .l6s file.

    The file stores float32 values at byte offset +1 from the file start
    (i.e. the first float32 occupies bytes 1-4, the second bytes 5-8, etc.).
    This function tries all 4 possible byte alignments to locate the
    monotonically-increasing wavenumber axis (400–1800 cm⁻¹), then reads
    the intensity block that immediately precedes it in the file.
    """
    if not os.path.isfile(filepath):
        raise FileNotFoundError(f"File not found:\n  {filepath}")
    with open(filepath, "rb") as fh:
        raw = fh.read()
    if not raw[:8].startswith(b"LabSpec"):
        raise ValueError("Not a LabSpec 6 file.")

    # Try byte offsets 0, 1, 2, 3 to handle non-standard alignment
    best = (None, None, 0, 0)   # (offset, wn_idx_in_arr, n_wn, arr)
    for boff in range(4):
        chunk = raw[boff:]
        n_aligned = (len(chunk) // 4) * 4
        arr = np.frombuffer(chunk[:n_aligned], dtype=np.float32).copy()
        wn_idx, n_wn = _find_wn_axis_l6s(arr)
        if wn_idx is not None and n_wn > best[2]:
            best = (boff, wn_idx, n_wn, arr)

    boff, wn_idx, n_wn, arr = best
    if wn_idx is None:
        raise RuntimeError("Could not locate wavenumber axis in .l6s file.")

    wavenumbers = arr[wn_idx:wn_idx + n_wn].astype(np.float64)
    intensity   = _find_intensity_block(arr, wn_idx, n_wn)

    print(f"\n{'─'*55}")
    print(f"  Single spectrum  ×  {n_wn} points  (byte offset {boff})")
    print(f"  Range   : {wavenumbers[0]:.1f} – {wavenumbers[-1]:.1f} cm⁻¹")
    print(f"  Step    : {np.diff(wavenumbers).mean():.3f} cm⁻¹/ch")
    print(f"  Max counts : {intensity.max():.1f}")
    print(f"{'─'*55}\n")
    return wavenumbers, intensity


def _find_wn_axis_l6s(arr):
    """Find the longest monotonic float32 run in the 400–1800 cm⁻¹ range."""
    best = (None, 0)
    i = 0
    while i < len(arr) - 20:
        v = float(arr[i])
        if 390 < v < 420 and np.isfinite(arr[i]):
            j = i
            while (j < len(arr) - 1
                   and np.isfinite(arr[j + 1])
                   and float(arr[j + 1]) > float(arr[j])
                   and float(arr[j + 1]) < 1810
                   and float(arr[j + 1]) - float(arr[j]) < 5.0):
                j += 1
            length = j - i + 1
            if length > best[1]:
                best = (i, length)
            i = j + 1
        else:
            i += 1
    return best[0], best[1]


def _find_intensity_block(arr, wn_idx, n_wn):
    """
    Extract the intensity block immediately before the wavenumber axis.
    Falls back to scanning all blocks before wn_idx for the one with the
    largest dynamic range that contains plausible photon counts (0–10^7).
    """
    candidate_start = wn_idx - n_wn
    if candidate_start >= 0:
        block = arr[candidate_start:candidate_start + n_wn].astype(np.float64)
        if np.all(np.isfinite(block)) and np.all(block >= 0) and block.max() < 1e7:
            return block

    best_block = None
    best_score = -1
    for start in range(0, wn_idx - n_wn + 1):
        block = arr[start:start + n_wn].astype(np.float64)
        if not (np.all(np.isfinite(block)) and np.all(block >= 0) and block.max() < 1e7):
            continue
        score = float(block.max()) - float(block.min())
        if score > best_score:
            best_score = score
            best_block = block
    if best_block is None:
        raise RuntimeError("Could not locate intensity block in .l6s file.")
    return best_block


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
#  9.  PREPROCESS SINGLE SPECTRUM  (same pipeline as map average)
# ─────────────────────────────────────────────────────────────────────────────
def preprocess_single(wn, spectrum):
    """Apply the full pipeline to a single spectrum and return SNV result."""
    sp, _ = remove_spikes(spectrum, threshold=5.0, window=3)
    sp = savgol_filter(sp, window_length=11, polyorder=3)
    sp_bc, _ = subtract_baseline(sp)
    sp_dn = wavelet_denoise(sp_bc, wavelet="db8", level=5, mode="soft")
    sp_dn = np.clip(sp_dn, 0, None)
    sp_nm = minmax_normalize(sp_dn)
    sp_snv = snv(sp_nm)
    return sp_snv


# ─────────────────────────────────────────────────────────────────────────────
#  10.  COMPARISON PLOT  —  map average vs single .l6s spectrum
# ─────────────────────────────────────────────────────────────────────────────
def plot_comparison(wn_map, snv_map, wn_l6s, snv_l6s,
                    label_map="Map average (AgSiNW, 50×LF, 0.5 s)",
                    label_l6s="Single spectrum (CaF₂, 100×, 10 s)"):
    """
    Side-by-side + overlay comparison of the two SNV-processed spectra.

    Two panels:
      • Left  : both spectra overlaid with an offset for clarity; dotted
                vertical lines mark every known peak; colour-coded labels
                show whether each peak appears in BOTH (green), MAP only
                (orange) or L6S only (blue).
      • Right : difference spectrum (map – l6s) after interpolating to a
                common wavenumber grid.
    """
    matplotlib.rcParams.update({
        "font.family": "DejaVu Sans",
        "axes.titlesize": 11,
        "axes.titleweight": "bold",
        "figure.facecolor": "white",
    })

    # ── Interpolate both spectra onto a common grid ──────────────────────────
    wn_min = max(wn_map.min(), wn_l6s.min())
    wn_max = min(wn_map.max(), wn_l6s.max())
    wn_common = np.arange(wn_min, wn_max, 1.0)

    interp_map = np.interp(wn_common, wn_map, snv_map)
    interp_l6s = np.interp(wn_common, wn_l6s, snv_l6s)

    # Offset the map trace so both are visible without overlap
    offset = max(abs(interp_l6s.max()), abs(interp_map.max())) * 1.6
    map_shifted = interp_map + offset

    fig, axes = plt.subplots(1, 2, figsize=(18, 7),
                             gridspec_kw={"width_ratios": [2, 1], "wspace": 0.30})
    fig.suptitle(
        "Peak-to-Peak Comparison: SERS Map Average vs Single Spectrum\n"
        "532 nm  |  600 gr/mm  |  SNV pre-processed",
        fontsize=13, fontweight="bold", y=1.01
    )

    ax = axes[0]

    # Plot both traces
    ax.plot(wn_common, interp_l6s, color="#2471A3", lw=2.0,
            label=label_l6s, zorder=3)
    ax.plot(wn_common, map_shifted, color="#C0392B", lw=2.0,
            label=label_map, zorder=3)

    # Bracket label showing the offset
    ax.annotate("", xy=(wn_common[-1] + 30, 0),
                xytext=(wn_common[-1] + 30, offset),
                arrowprops=dict(arrowstyle="<->", color="gray", lw=1.2))
    ax.text(wn_common[-1] + 40, offset / 2, "offset",
            fontsize=8, color="gray", va="center", rotation=90)

    # ── Peak annotations ─────────────────────────────────────────────────────
    tol = 30      # ±30 cm⁻¹ window for peak detection
    min_h = 0.06  # minimum relative height to be considered "present"

    used_x = []
    for center, plabel in sorted(PEAK_LABELS.items()):
        if not (wn_min <= center <= wn_max):
            continue
        mask = (wn_common >= center - tol) & (wn_common <= center + tol)
        if not mask.any():
            continue

        # Detect local peak in each spectrum
        seg_map = interp_map[mask]
        seg_l6s = interp_l6s[mask]
        peak_map = seg_map.max()
        peak_l6s = seg_l6s.max()

        in_map = peak_map > min_h * abs(interp_map).max()
        in_l6s = peak_l6s > min_h * abs(interp_l6s).max()

        if not (in_map or in_l6s):
            continue

        # Find the actual wavenumber of the stronger peak
        if in_map and in_l6s:
            peak_wn = wn_common[mask][np.argmax(seg_map)]
            color   = "#1E8449"   # green  – in both
            marker  = "▲"
        elif in_map:
            peak_wn = wn_common[mask][np.argmax(seg_map)]
            color   = "#E67E22"   # orange – map only
            marker  = "●"
        else:
            peak_wn = wn_common[mask][np.argmax(seg_l6s)]
            color   = "#2471A3"   # blue   – l6s only
            marker  = "■"

        # Avoid x-crowding
        if any(abs(peak_wn - u) < 55 for u in used_x):
            continue
        used_x.append(peak_wn)

        # Dotted vertical line spanning both traces
        ax.axvline(peak_wn, color=color, lw=1.0, ls=":", alpha=0.85, zorder=2)

        # Label on the map trace (top)
        y_label_map = map_shifted[mask][np.argmax(seg_map)] if in_map else map_shifted[mask].max()
        ax.text(peak_wn, y_label_map + abs(map_shifted).max() * 0.06,
                f"{int(peak_wn)}\n{plabel}",
                fontsize=7.0, ha="center", va="bottom", color=color,
                bbox=dict(boxstyle="round,pad=0.15", fc="white", ec=color,
                          alpha=0.85, lw=0.6))

    # ── Legend for colour coding ─────────────────────────────────────────────
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color="#2471A3", lw=2, label=label_l6s),
        Line2D([0], [0], color="#C0392B", lw=2, label=f"{label_map}  (offset)"),
        Line2D([0], [0], color="#1E8449", lw=1.5, ls=":",
               label="Peak in BOTH  ▲"),
        Line2D([0], [0], color="#E67E22", lw=1.5, ls=":",
               label="Map only  ●"),
        Line2D([0], [0], color="#2471A3", lw=1.5, ls=":",
               label="Single-spectrum only  ■"),
    ]
    ax.legend(handles=legend_elements, fontsize=8.5, loc="upper right",
              framealpha=0.9)

    ax.set_xlabel("Raman shift (cm⁻¹)", fontsize=10)
    ax.set_ylabel("SNV units (σ)", fontsize=10)
    ax.set_title("Spectral comparison with peak labels", fontsize=11)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(labelsize=9)
    ax.grid(True, alpha=0.15, lw=0.5)
    ax.set_xlim(wn_common[0] - 20, wn_common[-1] + 80)

    # ── Right panel: difference spectrum ────────────────────────────────────
    ax2 = axes[1]
    diff = interp_map - interp_l6s
    ax2.fill_between(wn_common, diff, 0,
                     where=diff > 0, alpha=0.35, color="#E67E22",
                     label="Map > Single")
    ax2.fill_between(wn_common, diff, 0,
                     where=diff < 0, alpha=0.35, color="#2471A3",
                     label="Single > Map")
    ax2.plot(wn_common, diff, color="#2C3E50", lw=1.4)
    ax2.axhline(0, color="black", lw=0.8, ls="--", alpha=0.5)

    ax2.set_xlabel("Raman shift (cm⁻¹)", fontsize=10)
    ax2.set_ylabel("Δ SNV (map − single)", fontsize=10)
    ax2.set_title("Difference spectrum", fontsize=11)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)
    ax2.tick_params(labelsize=9)
    ax2.grid(True, alpha=0.15, lw=0.5)
    ax2.legend(fontsize=8.5, loc="upper right", framealpha=0.9)

    plt.savefig("raman_comparison.png", dpi=180, bbox_inches="tight",
                facecolor="white")
    print("  Saved → raman_comparison.png")
    plt.show()


# ─────────────────────────────────────────────────────────────────────────────
#  11.  PEAK DETECTION  (automatic, scipy.signal)
# ─────────────────────────────────────────────────────────────────────────────
def detect_peaks(wn, spectrum, distance_cm=35, height_pct=0.04, prominence_pct=0.03):
    """
    Find peaks in a spectrum over 400–1800 cm⁻¹.

    Works on the ALS-baseline-corrected signal so broad fluorescence
    background does not swamp low Raman peaks.  Returns an array of
    peak wavenumbers sorted by ascending wavenumber.
    """
    from scipy.signal import find_peaks as _fp

    mask = (wn >= 395) & (wn <= 1810)
    wn_r   = wn[mask]
    sp_sg  = savgol_filter(spectrum[mask], window_length=11, polyorder=3)
    bline  = als_baseline(sp_sg, lam=1e5, p=0.01, n_iter=10)
    sp_bc  = np.clip(sp_sg - bline, 0, None)

    if sp_bc.max() < 1e-6:
        return np.array([])

    step         = float(np.diff(wn_r).mean())
    dist_pts     = max(3, int(distance_cm / step))
    height_thr   = height_pct    * sp_bc.max()
    prom_thr     = prominence_pct * sp_bc.max()

    idxs, _ = _fp(sp_bc, height=height_thr, distance=dist_pts,
                  prominence=prom_thr)
    return wn_r[idxs]


# ─────────────────────────────────────────────────────────────────────────────
#  12.  INDIVIDUAL SPECTRUM vs CaF₂ COMPARISON PLOTS
# ─────────────────────────────────────────────────────────────────────────────
def _sg_display(spectrum, wl=9, po=3):
    """Light Savitzky-Golay smooth for display (preserves counts scale)."""
    return savgol_filter(spectrum, window_length=wl, polyorder=po)


def plot_spectrum_vs_caf2(wn_map, sp_map, wn_caf2, sp_caf2,
                          peaks_map, peaks_caf2,
                          spectrum_idx, out_dir,
                          peak_tol=22.0):
    """
    One-panel figure: SERS spectrum (black) vs CaF₂ reference (red).

    Dotted vertical lines mark every detected peak.
    Wavenumber label + '?' → peak present in SERS but NOT in CaF₂.
    Plain wavenumber label  → peak present in BOTH (substrate / instrument).
    """
    matplotlib.rcParams.update({"font.family": "DejaVu Sans",
                                 "figure.facecolor": "white"})

    # Restrict display to fingerprint region
    fp = (400, 1800)
    m_m   = (wn_map  >= fp[0]) & (wn_map  <= fp[1])
    m_c   = (wn_caf2 >= fp[0]) & (wn_caf2 <= fp[1])

    sp_m_d = _sg_display(sp_map)
    sp_c_d = _sg_display(sp_caf2)

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(wn_map[m_m],   sp_m_d[m_m],  color="black", lw=1.3,
            label=f"SERS map spectrum #{spectrum_idx + 1}")
    ax.plot(wn_caf2[m_c],  sp_c_d[m_c],  color="#C0392B", lw=1.0, alpha=0.85,
            label="CaF₂ reference")

    ymax_m = sp_m_d[m_m].max()
    ymax_c = sp_c_d[m_c].max()
    ymax   = max(ymax_m, ymax_c)
    ymin   = min(sp_m_d[m_m].min(), sp_c_d[m_c].min())
    y_span = ymax - ymin

    used_x = []

    # ── annotate SERS peaks ──────────────────────────────────────────────────
    for wn_p in sorted(peaks_map):
        if not (fp[0] <= wn_p <= fp[1]):
            continue
        if any(abs(wn_p - u) < peak_tol * 0.7 for u in used_x):
            continue
        used_x.append(wn_p)

        in_caf2 = any(abs(wn_p - c) < peak_tol for c in peaks_caf2)

        # height at this wavenumber in the smoothed map spectrum
        i_local = np.argmin(np.abs(wn_map - wn_p))
        y_val   = sp_m_d[i_local]

        color  = "#C0392B" if not in_caf2 else "#555555"
        marker = "?\n" if not in_caf2 else ""
        lw_vl  = 1.0 if not in_caf2 else 0.7

        ax.axvline(wn_p, color=color, lw=lw_vl, ls=":", alpha=0.85)
        ax.text(wn_p,
                y_val + y_span * 0.04,
                f"{marker}{int(round(wn_p))}",
                fontsize=7.5, ha="center", va="bottom",
                color=color,
                fontweight="bold" if not in_caf2 else "normal",
                bbox=dict(boxstyle="round,pad=0.1", fc="white",
                          ec="none", alpha=0.7))

    # ── annotate CaF₂-only peaks ────────────────────────────────────────────
    for wn_p in sorted(peaks_caf2):
        if not (fp[0] <= wn_p <= fp[1]):
            continue
        if any(abs(wn_p - u) < peak_tol * 0.7 for u in used_x):
            continue
        used_x.append(wn_p)

        i_local = np.argmin(np.abs(wn_caf2 - wn_p))
        y_val   = sp_c_d[i_local]

        ax.axvline(wn_p, color="#888888", lw=0.6, ls=":", alpha=0.6)
        ax.text(wn_p,
                y_val + y_span * 0.02,
                f"{int(round(wn_p))}",
                fontsize=6.5, ha="center", va="bottom",
                color="#888888", alpha=0.85)

    ax.set_xlim(fp[0] - 10, fp[1] + 10)
    ax.set_ylim(ymin - y_span * 0.08, ymax + y_span * 0.30)
    ax.set_xlabel("Raman shift (cm⁻¹)", fontsize=10)
    ax.set_ylabel("Intensity (counts)", fontsize=10)
    ax.set_title(
        f"Spectrum #{spectrum_idx + 1}  vs  CaF₂ reference   "
        f"|   ? = SERS-specific peak (not in CaF₂)",
        fontsize=10, fontweight="bold"
    )
    ax.legend(fontsize=9, loc="upper right", framealpha=0.9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(labelsize=9)
    ax.grid(True, alpha=0.12, lw=0.5)

    os.makedirs(out_dir, exist_ok=True)
    outfile = os.path.join(out_dir, f"spectrum_{spectrum_idx + 1:02d}_vs_caf2.png")
    fig.savefig(outfile, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return outfile


def plot_all_individual(wn, spectra, wn_caf2, sp_caf2, out_dir):
    """
    Generate one figure per map spectrum (spectrum_01 … spectrum_N) plus a
    summary grid showing all spectra together.
    """
    matplotlib.rcParams.update({"font.family": "DejaVu Sans",
                                 "figure.facecolor": "white"})

    print(f"\n  Detecting CaF₂ reference peaks …")
    peaks_caf2 = detect_peaks(wn_caf2, sp_caf2)
    print(f"  CaF₂ peaks found: {len(peaks_caf2)}  "
          f"({', '.join(str(int(w)) for w in peaks_caf2[:10])}{'…' if len(peaks_caf2)>10 else ''})")

    n = len(spectra)
    print(f"\n  Plotting {n} individual spectra vs CaF₂ …")

    # ── Individual figures ────────────────────────────────────────────────────
    all_peaks_map = []
    for i, sp in enumerate(spectra):
        pk = detect_peaks(wn, sp)
        all_peaks_map.append(pk)
        outfile = plot_spectrum_vs_caf2(
            wn, sp, wn_caf2, sp_caf2,
            pk, peaks_caf2, i, out_dir
        )
        print(f"    [{i+1:2d}/{n}]  {len(pk)} peaks  → {os.path.basename(outfile)}")

    # ── Summary grid ─────────────────────────────────────────────────────────
    ncols = 5
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(ncols * 4.5, nrows * 3.2),
                             sharex=True)
    fig.suptitle(
        "All SERS Map Spectra vs CaF₂ Reference  (400–1800 cm⁻¹)\n"
        "Black = SERS  |  Red = CaF₂  |  ? = SERS-specific peak",
        fontsize=12, fontweight="bold", y=1.01
    )

    sp_caf2_sg = _sg_display(sp_caf2)
    m_c = (wn_caf2 >= 400) & (wn_caf2 <= 1800)

    axes_flat = axes.flat if nrows > 1 else [axes] if ncols == 1 else list(axes.flat)
    for i in range(ncols * nrows):
        ax = list(axes.flat)[i]
        if i >= n:
            ax.set_visible(False)
            continue

        sp_sg = _sg_display(spectra[i])
        m_m   = (wn >= 400) & (wn <= 1800)

        ax.plot(wn[m_m],      sp_sg[m_m],      color="black", lw=0.9)
        ax.plot(wn_caf2[m_c], sp_caf2_sg[m_c], color="#C0392B", lw=0.7, alpha=0.8)

        ymax = max(sp_sg[m_m].max(), sp_caf2_sg[m_c].max())
        ymin = min(sp_sg[m_m].min(), sp_caf2_sg[m_c].min())

        for wn_p in all_peaks_map[i]:
            if not (400 <= wn_p <= 1800):
                continue
            in_caf2 = any(abs(wn_p - c) < 22 for c in peaks_caf2)
            color = "#C0392B" if not in_caf2 else "#888888"
            ax.axvline(wn_p, color=color, lw=0.5, ls=":", alpha=0.8)
            marker = "?" if not in_caf2 else ""
            ax.text(wn_p, ymax + (ymax - ymin) * 0.05,
                    f"{marker}{int(round(wn_p))}",
                    fontsize=5.5, ha="center", va="bottom",
                    color=color, rotation=90)

        ax.set_xlim(400, 1800)
        ax.set_ylim(ymin - (ymax - ymin) * 0.05,
                    ymax + (ymax - ymin) * 0.45)
        ax.set_title(f"#{i + 1}", fontsize=8, pad=2)
        ax.tick_params(labelsize=6)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    for ax in list(axes.flat)[n:]:
        ax.set_visible(False)

    fig.tight_layout()
    grid_file = os.path.join(out_dir, "summary_grid_all_spectra.png")
    fig.savefig(grid_file, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"\n  Summary grid saved → {grid_file}")

    # ── SERS-unique peak frequency table ─────────────────────────────────────
    # Collect all SERS-unique peaks across all spectra; bin into 20-cm⁻¹ slots
    sers_unique = []
    for pk_map in all_peaks_map:
        for wn_p in pk_map:
            if not (400 <= wn_p <= 1800):
                continue
            if not any(abs(wn_p - c) < 22 for c in peaks_caf2):
                sers_unique.append(wn_p)

    if sers_unique:
        sers_unique = np.array(sers_unique)
        bins = np.arange(400, 1820, 20)
        counts, edges = np.histogram(sers_unique, bins=bins)
        hot = np.where(counts > 0)[0]
        print(f"\n  SERS-unique peak frequency (wavenumber : # spectra showing it):")
        for idx_b in hot:
            c = int(counts[idx_b])
            wn_c = 0.5 * (edges[idx_b] + edges[idx_b + 1])
            bar = "█" * c
            print(f"    {wn_c:6.0f} cm⁻¹ : {c:2d}  {bar}")

    print(f"\n  All files saved to: {out_dir}\n")


# ─────────────────────────────────────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────────────────────────────────────
def main():
    print("\n" + "="*55)
    print("  RAMAN MAP ANALYSIS PIPELINE  v3")
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

    # ── Load and process single .l6s spectrum ────────────────────────────────
    print("\n" + "="*55)
    print("  SINGLE SPECTRUM (.l6s)  PROCESSING")
    print("="*55)
    wn_s, raw_s = load_l6s(FILE_PATH_L6S)
    print("  Applying preprocessing pipeline to single spectrum …")
    snv_s = preprocess_single(wn_s, raw_s)

    # ── Comparison plot (SNV overlay) ─────────────────────────────────────────
    print("  Generating comparison figure …")
    plot_comparison(wn, avg_snv, wn_s, snv_s)

    # ── Individual spectra vs CaF₂  ───────────────────────────────────────────
    print("\n" + "="*55)
    print("  INDIVIDUAL SPECTRA vs CaF₂  (400–1800 cm⁻¹)")
    print("="*55)
    # Save to the same folder as the input .l6m file
    out_dir = os.path.join(os.path.dirname(FILE_PATH), "individual_spectra")
    print(f"  Output folder: {out_dir}")
    # Use spike-cleaned spectra so cosmic rays don't become false peaks
    plot_all_individual(wn, cleaned_spectra, wn_s, raw_s, out_dir)

    print("\n  Done.\n")


if __name__ == "__main__":
    main()
