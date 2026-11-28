"""Publication figures: UV-Vis of 20 nm polystyrene (PS) nanoparticles.

Stock: 25 mg/mL PS in water with 200 ppm SDS, serially diluted in ultrapure water.
Usage:  python3 paper_figures.py <folder_with_asc_files> <output_folder>
Each figure is written as 600-dpi TIFF, vector PDF and PNG.
"""
import sys, glob, os, re, csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker
from scipy.signal import savgol_filter

src, out = sys.argv[1], sys.argv[2]
os.makedirs(out, exist_ok=True)

MM = 1 / 25.4                       # mm -> inch
ONE_COL, TWO_COL = 85 * MM, 174 * MM
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Liberation Sans", "DejaVu Sans"],
    "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
    "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "xtick.minor.width": 0.4, "ytick.minor.width": 0.4,
    "xtick.direction": "in", "ytick.direction": "in",
    "xtick.top": True, "ytick.right": True,
    "lines.linewidth": 1.2, "legend.frameon": False,
    "pdf.fonttype": 42, "ps.fonttype": 42, "savefig.dpi": 600,
})
# validated categorical slots (blue, orange, aqua) + line style as 2nd encoding
STYLE = {10: dict(color="#257de0", ls="-"),
         0.5: dict(color="#f47230", ls="-"),
         0.001: dict(color="#00b894", ls="--")}
NAME = {10: "10 µg/mL", 0.5: "0.5 µg/mL", 0.001: "1 ng/mL"}
TEXT2 = "#555555"
HC_EV = 1239.84193                  # eV·nm


def save(fig, name):
    for ext in ("tiff", "pdf", "png"):
        kw = dict(pil_kwargs={"compression": "tiff_lzw"}) if ext == "tiff" else {}
        fig.savefig(f"{out}/{name}.{ext}", bbox_inches="tight", **kw)
    plt.close(fig)


def panel(ax, letter):
    ax.text(-0.16, 1.02, f"({letter})", transform=ax.transAxes,
            fontsize=9, fontweight="bold", va="bottom")


def read_asc(path):
    lines = open(path, encoding="latin-1").read().splitlines()
    i = lines.index("#DATA") + 1
    d = np.array([[float(v) for v in l.split()] for l in lines[i:] if l.strip()])
    d = d[np.argsort(d[:, 0])]
    return d[:, 0], d[:, 1]


def conc(name):
    m = re.search(r"D\d+_([\d_]+?)(ug|ng)_ml", name)
    v = float(m.group(1).replace("_", "."))
    return v if m.group(2) == "ug" else v / 1000


groups = {}
for f in sorted(glob.glob(os.path.join(src, "*.asc"))):
    wl, A = read_asc(f)
    groups.setdefault(conc(os.path.basename(f)), []).append(A)
C = sorted(groups, reverse=True)
mean = {c: np.mean(v, 0) for c, v in groups.items()}
sd = {c: np.std(v, 0, ddof=1) if len(v) > 1 else np.zeros_like(wl) for c, v in groups.items()}
n = {c: len(v) for c, v in groups.items()}
blank, blank_sd = mean[0.001], sd[0.001]
ps = mean[10] - blank                       # PS-only spectrum (10 µg/mL)
ok = wl >= 205                              # A > 2 below 205 nm: outside linear range
idx = lambda lam: int(np.where(wl == lam)[0][0])

# ================= Figure 1: spectra, PS spectrum, derivative =================
fig, ax = plt.subplots(1, 3, figsize=(TWO_COL, 55 * MM),
                       gridspec_kw=dict(wspace=0.42))
a = ax[0]
for c in C:
    lab = NAME[c] + (f" (n = {n[c]})" if n[c] > 1 else "")
    a.plot(wl[ok], mean[c][ok], label=lab, **STYLE[c])
    if n[c] > 1:
        a.fill_between(wl[ok], (mean[c] - sd[c])[ok], (mean[c] + sd[c])[ok],
                       color=STYLE[c]["color"], alpha=0.25, lw=0)
a.set_yscale("log"); a.set_xlim(205, 400); a.set_ylim(0.03, 2.5)
a.set_xlabel("Wavelength (nm)"); a.set_ylabel("Absorbance")
a.legend(loc="upper right", handlelength=2.2); panel(a, "a")

a = ax[1]
a.plot(wl[ok], ps[ok], color=STYLE[10]["color"])
a.axhline(0, color="0.4", lw=0.5)
a.set_xlim(205, 320); a.set_ylim(-0.05, 1.6)
a.set_xlabel("Wavelength (nm)"); a.set_ylabel("ΔA (10 µg/mL − 1 ng/mL)")
a.annotate("~212 nm\nshoulder", xy=(212, ps[idx(212)]), xytext=(228, 1.30),
           fontsize=7, color=TEXT2, arrowprops=dict(arrowstyle="-", lw=0.5, color=TEXT2))
ins = a.inset_axes([0.45, 0.30, 0.52, 0.42])
z = (wl >= 240) & (wl <= 290)
ins.plot(wl[z], ps[z], color=STYLE[10]["color"], lw=1)
for p in (253, 262, 269):
    ins.axvline(p, color="0.6", lw=0.4, ls=":")
ins.set_ylim(0.012, 0.068); ins.tick_params(labelsize=6)
ins.set_xticks([250, 270]); ins.tick_params(axis="x", labelsize=6, pad=1.5)
for p in (253, 262, 269):
    ins.text(p - 0.5, 0.015, f"{p}", rotation=90, fontsize=5.5, color=TEXT2, va="bottom", ha="right")
ins.set_title("phenyl B-band (π→π*)", fontsize=6.5, pad=2)
panel(a, "b")

a = ax[2]
for c in C:
    d2 = savgol_filter(mean[c], 11, 3, deriv=2)
    z = (wl >= 240) & (wl <= 290)
    a.plot(wl[z], d2[z] * 1e4, label=NAME[c], **STYLE[c])
d2 = savgol_filter(mean[10], 11, 3, deriv=2)
for p in (262, 269):
    a.annotate(f"{p} nm", xy=(p, d2[idx(p)] * 1e4), xytext=(p + 2.5, d2[idx(p)] * 1e4 - 0.6),
               ha="left", fontsize=7, color=TEXT2,
               arrowprops=dict(arrowstyle="-", lw=0.5, color=TEXT2))
a.axhline(0, color="0.4", lw=0.5)
a.set_xlim(240, 290); a.set_ylim(-5.5, 6.5)
a.set_xlabel("Wavelength (nm)"); a.set_ylabel(r"d$^2$A/dλ$^2$ (×10$^{-4}$ nm$^{-2}$)")
a.legend(loc="upper left"); panel(a, "c")
save(fig, "Fig1_UVVis_spectra")

# ================= Figure 2: energy (eV) and Tauc plots =================
E = HC_EV / wl
alpha = 2.303 * ps / 1.0                     # cm^-1, path length 1 cm, 10 µg/mL
alpha = np.clip(alpha, 0, None)


def tauc_fit(y, lo=5.2, hi=5.9, npts=8):
    """Steepest linear segment of the absorption edge (sliding window, R² ≥ 0.99)."""
    cand = np.where((E >= lo) & (E <= hi) & ok)[0]
    best = None
    for s0 in range(len(cand) - npts + 1):
        j = cand[s0:s0 + npts]
        k, b = np.polyfit(E[j], y[j], 1)
        r2 = np.corrcoef(E[j], y[j])[0, 1] ** 2
        if r2 >= 0.99 and (best is None or k > best[1]):
            best = (-b / k, k, b, j, r2)
    Eg, k, b, j, r2 = best
    m = np.zeros_like(E, bool); m[j] = True
    return Eg, k, b, m, r2


fig, ax = plt.subplots(1, 3, figsize=(TWO_COL, 55 * MM), gridspec_kw=dict(wspace=0.45))
a = ax[0]
for c in C:
    a.plot(E[ok], mean[c][ok], label=NAME[c], **STYLE[c])
a.set_yscale("log"); a.set_xlim(E[ok].min(), E[ok].max()); a.set_ylim(0.03, 2.5)
a.set_xlabel("Photon energy (eV)"); a.set_ylabel("Absorbance")
top = a.secondary_xaxis("top", functions=(lambda e: HC_EV / e, lambda l: HC_EV / l))
top.set_xticks([300, 250, 220]); top.set_xlabel("Wavelength (nm)", fontsize=7)
top.tick_params(labelsize=6)
a.legend(loc="upper left"); panel(a, "a")

results = {}
for a, (expo, lab, key) in zip(ax[1:], [(2, r"($\alpha h\nu$)$^{2}$ (eV cm$^{-1}$)$^2$", "direct"),
                                       (0.5, r"($\alpha h\nu$)$^{1/2}$ (eV cm$^{-1}$)$^{1/2}$", "indirect")]):
    y = (alpha * E) ** expo
    Eg, k, b, m, r2 = tauc_fit(y)
    r = (E >= 4.6) & ok
    a.plot(E[r], y[r], "o", ms=2.2, color=STYLE[10]["color"], mfc="none", mew=0.6)
    xx = np.linspace(Eg, E[m].max() + 0.1, 50)
    a.plot(xx, k * xx + b, color="#191919", lw=0.9)
    a.plot(E[m], y[m], "o", ms=2.2, color=STYLE[10]["color"])
    a.axhline(0, color="0.4", lw=0.5)
    a.set_xlim(4.6, 6.1); a.set_ylim(-0.03 * y[r].max(), 1.05 * y[r].max())
    a.set_xlabel(r"Photon energy, $h\nu$ (eV)"); a.set_ylabel(lab)
    a.text(0.04, 0.92, f"{key} allowed\n$E_g$ = {Eg:.2f} eV\n$R^2$ = {r2:.3f}",
           transform=a.transAxes, va="top", fontsize=7)
    results[key] = (Eg, r2, HC_EV / E[m].max(), HC_EV / E[m].min())
    panel(a, "b" if key == "direct" else "c")
save(fig, "Fig2_energy_Tauc")

# ================= Figure 3: concentration response =================
fig, ax = plt.subplots(1, 2, figsize=(TWO_COL * 0.72, 52 * MM), gridspec_kw=dict(wspace=0.5))
rows = []
for a, lam, letter in zip(ax, (220, 262), "ab"):
    i = idx(lam)
    cs = np.array(sorted(mean))
    y = np.array([mean[c][i] - blank[i] for c in cs])
    e = np.array([sd[c][i] for c in cs])
    lod = 3 * blank_sd[i]
    a.axhspan(-lod, lod, color="0.85", lw=0, label="±3σ blank (LOD)")
    a.axhline(0, color="0.4", lw=0.5)
    for c, yy, ee in zip(cs, y, e):
        a.errorbar(c, yy, yerr=ee if ee > 0 else None, fmt="o", ms=4,
                   color=STYLE[c]["color"], capsize=2, elinewidth=0.6)
        a.annotate(NAME[c], (c, yy), xytext=(-6 if c == 10 else 4, 5), textcoords="offset points",
                   ha="right" if c == 10 else "left", fontsize=6.5, color=TEXT2)
        rows.append((lam, c, yy, ee, lod))
    a.set_xscale("log"); a.set_xlim(4e-4, 40)
    a.set_xlabel("PS concentration (µg/mL)")
    a.set_ylabel(f"ΔA at {lam} nm (vs. 1 ng/mL)")
    a.set_ylim(-0.08 * y.max(), 1.2 * y.max())
    a.legend(loc="upper left"); panel(a, letter)
save(fig, "Fig3_concentration_response")

# ================= Figure S1: blank replicates =================
fig, (a, a2) = plt.subplots(2, 1, figsize=(ONE_COL, 95 * MM), sharex=True,
                            gridspec_kw=dict(height_ratios=[2, 1], hspace=0.12))
for k, A in enumerate(groups[0.001]):
    a.plot(wl[ok], A[ok], lw=0.7, color=["#257de0", "#f47230", "#00b894"][k], label=f"Replicate {k+1}")
a.plot(wl[ok], blank[ok], color="#191919", lw=1.2, label="Mean")
a.set_ylabel("Absorbance")
a.legend(loc="upper right"); panel(a, "a")
a2.plot(wl[ok], blank_sd[ok] * 1e3, color="#191919", lw=0.8)
a2.set_xlim(205, 400); a2.set_ylim(0, 1.6)
a2.set_xlabel("Wavelength (nm)"); a2.set_ylabel("SD (×10$^{-3}$)")
panel(a2, "b")
save(fig, "FigS1_blank_replicates")

# ================= Figure S2: log–log extinction =================
fig, a = plt.subplots(figsize=(ONE_COL, 60 * MM))
fit = (wl >= 280) & (wl <= 400)
slopes = {}
for c in C:
    a.loglog(wl[ok], mean[c][ok], label=NAME[c], **STYLE[c])
    k, b = np.polyfit(np.log10(wl[fit]), np.log10(mean[c][fit]), 1)
    slopes[c] = k
a.set_xlabel("Wavelength (nm)"); a.set_ylabel("Absorbance")
a.set_xticks([210, 250, 300, 400]); a.set_xticklabels(["210", "250", "300", "400"])
a.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
a.text(0.97, 0.95, "slope 280–400 nm\n" + "\n".join(f"{NAME[c]}: {slopes[c]:.2f}" for c in C),
       transform=a.transAxes, ha="right", va="top", fontsize=6.5)
a.legend(loc="center right")
save(fig, "FigS2_loglog_extinction")

# ================= numbers for the text =================
with open(f"{out}/figure_values.csv", "w", newline="") as fh:
    w = csv.writer(fh)
    w.writerow(["wavelength_nm", "PS_ug_per_mL", "dA_vs_blank", "SD", "LOD_3sigma"])
    for r in rows:
        w.writerow([r[0], r[1], f"{r[2]:.5f}", f"{r[3]:.5f}", f"{r[4]:.5f}"])
with open(f"{out}/figure_values.txt", "w") as fh:
    for key, (Eg, r2, l1, l2) in results.items():
        fh.write(f"Tauc {key}: Eg = {Eg:.2f} eV (R2 = {r2:.3f}, fit {l1:.0f}-{l2:.0f} nm)\n")
    for c in C:
        fh.write(f"log-log slope 280–400 nm, {NAME[c]}: {slopes[c]:.2f}\n")
    for r in rows:
        fh.write(f"{r[0]} nm, {r[1]:g} ug/mL: dA = {r[2]:.4f} ± {r[3]:.4f} (LOD 3σ = {r[4]:.4f})\n")
print(open(f"{out}/figure_values.txt").read())
