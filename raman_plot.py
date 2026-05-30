import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, savgol_filter
import os

# ── FILE PATH ──────────────────────────────────────────────────────────────────
FILE_PATH = r"C:\Users\Hira Aman\Desktop\RAMAN MAP 1\S1-DNA-HACAT1-20ng-CaF2_532nm_600gr_BC50_100X_10s_4a_100%.txt"

# ── LOAD DATA ─────────────────────────────────────────────────────────────────
if not os.path.exists(FILE_PATH):
    raise FileNotFoundError(f"File not found:\n{FILE_PATH}\nCheck the path and filename.")

wavenumber, intensity = [], []
with open(FILE_PATH, encoding="utf-8", errors="ignore") as f:
    for line in f:
        line = line.strip().replace("\r", "")
        parts = line.split("\t")
        if len(parts) == 2:
            try:
                wavenumber.append(float(parts[0]))
                intensity.append(float(parts[1]))
            except ValueError:
                pass

wn = np.array(wavenumber)
raw = np.array(intensity)
print(f"Loaded {len(wn)} points | Range: {wn.min():.1f} – {wn.max():.1f} cm⁻¹")

# ── SMOOTHING ─────────────────────────────────────────────────────────────────
smoothed = savgol_filter(raw, window_length=11, polyorder=3)

# ── BASELINE ESTIMATE (low-wavenumber flat region) ────────────────────────────
baseline = np.mean(raw[(wn > 400) & (wn < 650)])
print(f"Baseline estimate: {baseline:.1f} counts")

# ── PEAK DETECTION ────────────────────────────────────────────────────────────
peaks, props = find_peaks(
    smoothed,
    height=baseline + 12,
    prominence=8,
    distance=8
)

print(f"\nDetected {len(peaks)} peaks:")
print(f"  {'cm⁻¹':>8}  {'Intensity':>10}  {'Above BL':>9}")
for p in peaks:
    print(f"  {wn[p]:>8.1f}  {smoothed[p]:>10.1f}  {smoothed[p]-baseline:>9.1f}")

# ── DNA BAND REFERENCE TABLE ──────────────────────────────────────────────────
dna_assignments = {
    666:  "G ring def.",
    724:  "A ring breath.",
    782:  "O-P-O / Thy+Cyt",
    806:  "Deoxyribose C-C",
    865:  "Backbone",
    912:  "Backbone C-C/C-O",
    1002: "Phe / ring",
    1058: "Backbone C-O",
    1097: "νs PO₂⁻",
    1179: "Cyt C-H / Thy",
    1242: "νas PO₂⁻",
    1295: "Cyt + Ade",
    1326: "Ade + Gua",
    1367: "Thy + Ade",
    1406: "Deoxyribose CH₂",
    1483: "Ade + Gua ring",
    1574: "Ade + Gua in-plane",
    1684: "Gua C=O / Thy C=O",
}

def nearest_assignment(peak_wn, ref_dict, tolerance=20):
    best, best_dist = None, tolerance
    for ref_wn, label in ref_dict.items():
        dist = abs(peak_wn - ref_wn)
        if dist < best_dist:
            best, best_dist = label, dist
    return best

# ── PLOT ──────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(14, 6))

ax.plot(wn, smoothed, color="#1f77b4", linewidth=1.4, label="Smoothed")
ax.plot(wn, raw, color="#aec7e8", linewidth=0.6, alpha=0.6, label="Raw")
ax.axhline(baseline, color="gray", linewidth=0.8, linestyle="--", alpha=0.5, label=f"Baseline ≈ {baseline:.0f}")

# Alternate label positions above/below to avoid overlap
for i, p in enumerate(peaks):
    x = wn[p]
    y = smoothed[p]
    label = nearest_assignment(x, dna_assignments) or ""
    offset_y = 18 if i % 2 == 0 else 30

    ax.annotate(
        f"{x:.0f}\n{label}",
        xy=(x, y),
        xytext=(0, offset_y),
        textcoords="offset points",
        ha="center",
        va="bottom",
        fontsize=7,
        color="#d62728",
        arrowprops=dict(arrowstyle="-", color="#d62728", lw=0.8),
    )
    ax.plot(x, y, "v", color="#d62728", markersize=5)

ax.set_xlabel("Raman Shift (cm⁻¹)", fontsize=12)
ax.set_ylabel("Intensity (counts)", fontsize=12)
ax.set_title(
    "SERS Spectrum — DNA HaCaT (20 ng/µL) on CaF₂\n"
    "532 nm | 100X | 10 s × 4 acc | 600 gr/mm",
    fontsize=12
)
ax.set_xlim(wn.min(), wn.max())
ax.legend(fontsize=9)
ax.grid(True, alpha=0.25, linestyle="--")

plt.tight_layout()
plt.savefig("raman_DNA_HaCaT_CaF2.png", dpi=200, bbox_inches="tight")
print("\nPlot saved: raman_DNA_HaCaT_CaF2.png")
plt.show()
