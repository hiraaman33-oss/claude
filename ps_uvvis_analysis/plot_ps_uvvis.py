"""UV-Vis analysis of 20 nm polystyrene (PS) nanoparticles in aqueous SDS.

Reads PerkinElmer Lambda 950 .asc files and produces the standard plots used in
nanoplastic UV-Vis papers. Usage:
    python3 plot_ps_uvvis.py <folder_with_asc_files> <output_folder>
"""
import sys, glob, os, re, csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter

src, out = sys.argv[1], sys.argv[2]
os.makedirs(out, exist_ok=True)
plt.rcParams.update({"font.size": 11, "axes.spines.top": False,
                     "axes.spines.right": False, "figure.dpi": 150})


def read_asc(path):
    lines = open(path, encoding="latin-1").read().splitlines()
    i = lines.index("#DATA") + 1
    d = np.array([[float(v) for v in l.split()] for l in lines[i:] if l.strip()])
    d = d[np.argsort(d[:, 0])]          # ascending wavelength
    return d[:, 0], d[:, 1]


def conc_from_name(name):
    """D3_10ug_ml -> 10 ug/mL ; D5_0_5ug_ml -> 0.5 ug/mL ; D6_1ng_ml -> 0.001 ug/mL"""
    m = re.search(r"D\d+_([\d_]+?)(ug|ng)_ml", name)
    val = float(m.group(1).replace("_", "."))
    return val if m.group(2) == "ug" else val / 1000.0


spectra = []
for f in sorted(glob.glob(os.path.join(src, "*.asc"))):
    wl, A = read_asc(f)
    name = re.sub(r"^[0-9a-f]{8}-", "", os.path.basename(f)).replace(".Sample.asc", "")
    spectra.append(dict(name=name, c=conc_from_name(name), wl=wl, A=A))
spectra.sort(key=lambda s: (-s["c"], s["name"]))
wl = spectra[0]["wl"]

label = lambda s: f"{s['name']}  ({s['c']:g} µg/mL)"
colors = {10: "#c0392b", 0.5: "#2471a3", 0.001: "#7f8c8d"}
col = lambda s: colors.get(s["c"], "k")

# replicate groups
groups = {}
for s in spectra:
    groups.setdefault(s["c"], []).append(s["A"])
mean = {c: np.mean(v, axis=0) for c, v in groups.items()}
sd = {c: (np.std(v, axis=0, ddof=1) if len(v) > 1 else None) for c, v in groups.items()}
blank = mean[0.001]                      # 1 ng/mL triplicate used as quasi-blank
base_mask = (wl >= 380) & (wl <= 400)

# ---------- 1. raw spectra ----------
fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
for s in spectra:
    ls = "-" if s["c"] != 0.001 else "--"
    for a in ax:
        a.plot(s["wl"], s["A"], ls, color=col(s), lw=1.3, label=label(s))
ax[0].set_title("Raw absorbance spectra")
ax[1].set_title("Zoom (y-axis log)")
ax[1].set_yscale("log")
for a in ax:
    a.set_xlabel("Wavelength (nm)"); a.set_ylabel("Absorbance (A.U.)")
ax[0].legend(fontsize=8)
fig.tight_layout(); fig.savefig(f"{out}/1_raw_spectra.png"); plt.close(fig)

# ---------- 2. baseline (offset) corrected ----------
fig, ax = plt.subplots(figsize=(7, 4.5))
for s in spectra:
    ax.plot(wl, s["A"] - s["A"][base_mask].mean(), color=col(s), lw=1.3, label=label(s))
ax.axvspan(380, 400, color="0.9", zorder=0, label="baseline region 380–400 nm")
ax.set_xlabel("Wavelength (nm)"); ax.set_ylabel("ΔA (offset-corrected)")
ax.set_title("Spectra after subtracting the 380–400 nm offset")
ax.legend(fontsize=8); fig.tight_layout(); fig.savefig(f"{out}/2_baseline_corrected.png"); plt.close(fig)

# ---------- 3. difference spectra vs 1 ng/mL (quasi-blank: water+SDS+cuvette) ----------
fig, ax = plt.subplots(figsize=(7, 4.5))
for c in (10, 0.5):
    ax.plot(wl, mean[c] - blank, color=colors[c], lw=1.5, label=f"{c:g} µg/mL − 1 ng/mL")
ax.fill_between(wl, -3 * sd[0.001], 3 * sd[0.001], color="0.8", label="±3σ of 1 ng/mL replicates")
ax.axhline(0, color="k", lw=0.6)
for x, t in [(262, "262"), (254, "254"), (220, "~220")]:
    ax.axvline(x, color="0.6", ls=":", lw=0.8); ax.text(x, ax.get_ylim()[1] * 0.95, t, ha="center", fontsize=8)
ax.set_xlim(200, 320)
ax.set_xlabel("Wavelength (nm)"); ax.set_ylabel("ΔA vs. 1 ng/mL")
ax.set_title("Difference spectra (PS signal above SDS/water background)")
ax.legend(fontsize=8); fig.tight_layout(); fig.savefig(f"{out}/3_difference_spectra.png"); plt.close(fig)

# ---------- 4. derivative spectra (resolve PS phenyl fine structure) ----------
fig, ax = plt.subplots(2, 1, figsize=(7, 7), sharex=True)
for c in (10, 0.5, 0.001):
    y = savgol_filter(mean[c], 9, 3)
    d1 = savgol_filter(mean[c], 9, 3, deriv=1, delta=1.0)
    d2 = savgol_filter(mean[c], 11, 3, deriv=2, delta=1.0)
    ax[0].plot(wl, d1, color=colors[c], label=f"{c:g} µg/mL")
    ax[1].plot(wl, d2, color=colors[c], label=f"{c:g} µg/mL")
ax[0].set_ylabel("dA/dλ (nm⁻¹)"); ax[1].set_ylabel("d²A/dλ² (nm⁻²)")
ax[1].set_xlabel("Wavelength (nm)"); ax[0].set_xlim(235, 300)
ax[0].set_ylim(-0.004, 0.003); ax[1].set_ylim(-0.0015, 0.0015)
ax[0].set_title("1st and 2nd derivative (Savitzky–Golay), 235–300 nm")
ax[0].legend(fontsize=8); fig.tight_layout(); fig.savefig(f"{out}/4_derivative_spectra.png"); plt.close(fig)

# ---------- 5. replicate reproducibility of 1 ng/mL ----------
fig, ax = plt.subplots(figsize=(7, 4.5))
for A, s in zip(groups[0.001], [s for s in spectra if s["c"] == 0.001]):
    ax.plot(wl, A, lw=0.9, alpha=0.7, label=s["name"])
ax.plot(wl, blank, "k", lw=1.6, label="mean")
ax.fill_between(wl, blank - sd[0.001], blank + sd[0.001], color="k", alpha=0.15, label="±1 SD")
ax.set_xlabel("Wavelength (nm)"); ax.set_ylabel("Absorbance (A.U.)")
ax.set_title("1 ng/mL replicates (n = 3): mean ± SD")
ax.legend(fontsize=8); fig.tight_layout(); fig.savefig(f"{out}/5_replicates_1ngml.png"); plt.close(fig)

# ---------- 6. log–log extinction (scattering exponent) ----------
fig, ax = plt.subplots(figsize=(6, 4.5))
fit = (wl >= 280) & (wl <= 400)
slopes = {}
for c in (10, 0.5, 0.001):
    ax.loglog(wl, mean[c], "o", ms=2, color=colors[c], label=f"{c:g} µg/mL")
    k, b = np.polyfit(np.log10(wl[fit]), np.log10(mean[c][fit]), 1)
    slopes[c] = k
    ax.loglog(wl[fit], 10 ** (b + k * np.log10(wl[fit])), "-", color=colors[c], lw=1.5)
    ax.text(330, mean[c][wl == 330][0] * 1.06, f"slope = {k:.2f}", color=colors[c], fontsize=8)
ax.set_xlabel("Wavelength (nm, log)"); ax.set_ylabel("Absorbance (log)")
ax.set_title("log A vs log λ (fit 280–400 nm)\nRayleigh scattering would give slope ≈ −4")
ax.legend(fontsize=8); fig.tight_layout(); fig.savefig(f"{out}/6_loglog_scattering.png"); plt.close(fig)

# ---------- 7. calibration plot (A vs concentration) ----------
lams = [262, 225, 220]
fig, ax = plt.subplots(figsize=(6, 4.5))
rows = []
for lam, mk in zip(lams, "osD"):
    i = np.where(wl == lam)[0][0]
    cs = sorted(mean)
    ys = [mean[c][i] - mean[c][base_mask].mean() for c in cs]
    es = [(sd[c][i] if sd[c] is not None else 0) for c in cs]
    ax.errorbar(cs, ys, yerr=es, fmt=mk + "-", capsize=3, label=f"{lam} nm")
    rows += [(lam, c, y, e) for c, y, e in zip(cs, ys, es)]
ax.set_xscale("log"); ax.set_xlabel("PS concentration (µg/mL, log)")
ax.set_ylabel("ΔA (offset-corrected)")
ax.set_title("Absorbance vs concentration\n(only 3 levels – add more standards for a real calibration)")
ax.legend(fontsize=8); fig.tight_layout(); fig.savefig(f"{out}/7_calibration.png"); plt.close(fig)

# ---------- exports ----------
with open(f"{out}/all_spectra.csv", "w", newline="") as fh:
    w = csv.writer(fh)
    w.writerow(["wavelength_nm"] + [s["name"] for s in spectra])
    for j in range(len(wl)):
        w.writerow([wl[j]] + [f"{s['A'][j]:.6f}" for s in spectra])

i260 = np.where(wl == 262)[0][0]; i220 = np.where(wl == 220)[0][0]
with open(f"{out}/summary.txt", "w") as fh:
    fh.write("Offset-corrected ΔA (A_λ - mean A_380-400)\n")
    for lam, c, y, e in rows:
        fh.write(f"  {lam} nm  {c:g} ug/mL : {y:.4f} ± {e:.4f}\n")
    fh.write("\nDifference vs 1 ng/mL mean:\n")
    for c in (10, 0.5):
        fh.write(f"  {c:g} ug/mL : dA262 = {mean[c][i260]-blank[i260]:.4f}, dA220 = {mean[c][i220]-blank[i220]:.4f}\n")
    fh.write(f"\n1 ng/mL replicate SD: 262 nm {sd[0.001][i260]:.4f}, 220 nm {sd[0.001][i220]:.4f}\n")
    fh.write(f"3*SD (approx. detection threshold in A): 262 nm {3*sd[0.001][i260]:.4f}, 220 nm {3*sd[0.001][i220]:.4f}\n")
    fh.write("\nlog-log slopes (280-400 nm): " + ", ".join(f"{c:g} ug/mL: {k:.2f}" for c, k in slopes.items()) + "\n")
print(open(f"{out}/summary.txt").read())
