"""
DNA HaCaT SERS Comparison Plots
================================
Part 1: Individual HIGH/MODERATE spectra vs CaF2 reference (raw overlay)
Part 2: Mean spectrum per map vs CaF2 reference with DNA band markers
Part 3: Zoom-in panels on key DNA band regions
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.signal import savgol_filter, find_peaks
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve
from scipy.optimize import curve_fit
import os, warnings
warnings.filterwarnings("ignore")

# ── PATHS ─────────────────────────────────────────────────────────────────────
BASE = "/root/.claude/uploads/f4cf220b-abe6-4680-b18d-91dd4043b59c"
OUT  = "/home/user/claude"

FILES = {
    "caf2": f"{BASE}/7896f9bf-S1DNAHACAT120ngCaF2_532nm_600gr_BC50_100X_10s_4a_100.txt",
    "M1":   f"{BASE}/fcfe8db1-m1DNAHACAT120ngAgSiNW_532nm_600gr_BC200_50XLF_05s_4a_25_5ul_dropcenter.txt",
    "M2":   f"{BASE}/a1702c49-m2DNAHACAT120ngAgSiNW_532nm_600gr_BC200_50XLF_05s_4a_25_5ul_dropedge.txt",
    "M3":   f"{BASE}/c13a713d-m3DNAHACAT120ngAgSiNW_532nm_600gr_BC200_50XLF_05s_4a_25_5ul_dropcenter.txt",
    "M4":   f"{BASE}/fe9514dc-m4DNAHACAT120ngAgSiNW_532nm_600gr_BC200_50XLF_05s_4a_25_5ul_dropcenter.txt",
}

# ── DNA BAND REFERENCE TABLE ───────────────────────────────────────────────────
DNA_BANDS = [
    ("G ring def.",         668.1),
    ("A ring breath.",      722.1),
    ("O-P-O / Thy+Cyt",    782.7),
    ("Deoxyribose C-C",     806.0),
    ("Backbone C-C/C-O",   912.9),
    ("Phe / ring",         999.1),
    ("νs PO₂⁻",          1097.5),
    ("νas PO₂⁻",         1239.4),
    ("Cyt + Ade",         1293.8),
    ("Ade + Gua",         1330.4),
    ("Thy + Ade",         1370.6),
    ("Ade+Gua ring",      1481.2),
    ("Ade+Gua in-plane",  1573.3),
    ("Gua/Thy C=O",       1684.8),
]

# Zoom regions: (label, wn_min, wn_max)
ZOOM_REGIONS = [
    ("Region A  650–850 cm⁻¹\n(G ring, A ring, O-P-O, Deoxyribose)", 640,  860),
    ("Region B  870–1120 cm⁻¹\n(Backbone, Phe, νs PO₂⁻)",           860, 1130),
    ("Region C  1200–1420 cm⁻¹\n(νas PO₂⁻, Cyt+Ade, Ade+Gua, Thy+Ade)", 1190, 1430),
    ("Region D  1440–1720 cm⁻¹\n(Ade+Gua ring/plane, Gua/Thy C=O)", 1430, 1730),
]

# ── LOADERS ───────────────────────────────────────────────────────────────────
def load_single(path):
    data = []
    with open(path, encoding="utf-8", errors="ignore") as f:
        for line in f:
            p = line.strip().replace("\r", "").split("\t")
            if len(p) == 2:
                try: data.append((float(p[0]), float(p[1])))
                except: pass
    return np.array([d[0] for d in data]), np.array([d[1] for d in data])

def load_map(path):
    with open(path, encoding="utf-8", errors="ignore") as f:
        lines = f.readlines()
    wn = None; wn_row = 0
    for i, l in enumerate(lines):
        p = l.strip().replace("\r", "").split("\t")
        try:
            v = [float(x) for x in p if x.strip()]
            if len(v) > 100: wn = np.array(v); wn_row = i; break
        except: pass
    spectra, pos = [], []
    for line in lines[wn_row+1:]:
        p = line.strip().replace("\r", "").split("\t")
        if len(p) < 5: continue
        try:
            x, y = float(p[0]), float(p[1])
            v = np.array([float(z) for z in p[2:] if z.strip()])
            if len(v) == len(wn): pos.append((x, y)); spectra.append(v)
        except: pass
    return wn, np.array(spectra), pos

# ── PROCESSING ────────────────────────────────────────────────────────────────
def airpls(y, lam=1e5, imax=20):
    N = len(y); D = diags([1,-2,1],[0,1,2],shape=(N-2,N)); H = lam*D.T.dot(D)
    w = np.ones(N)
    for _ in range(imax):
        Z = spsolve(diags(w,0)+H, w*y); d = y-Z; dn = d[d<0]
        if len(dn)==0: break
        m = dn.mean(); s = max(dn.std(),1e-9)
        w = np.exp(np.clip(np.where(d<0,2*(d-m)/s,0),-500,500)); w[w>1]=1
    return Z

def process(wn, sp):
    sm = savgol_filter(sp, 11, 3)
    bl = airpls(sm)
    return np.clip(sm - bl, 0, None)

def norm_max(sp, wn=None, lo=650, hi=1750):
    if wn is not None:
        m = (wn>=lo)&(wn<=hi)
        peak = sp[m].max() if m.any() else sp.max()
    else:
        peak = sp.max()
    return sp / max(peak, 1e-9)

def cosine_sim(a, b):
    na = np.linalg.norm(a); nb = np.linalg.norm(b)
    if na<1e-9 or nb<1e-9: return 0.0
    return float(np.dot(a,b)/(na*nb))

# ── LOAD & PROCESS ALL DATA ───────────────────────────────────────────────────
print("Loading CaF2 reference...")
wn_ref, raw_ref = load_single(FILES["caf2"])
sp_ref_bc = process(wn_ref, raw_ref)
sp_ref_n  = norm_max(sp_ref_bc, wn_ref)

print("Loading maps...")
map_data = {}
for mname in ["M1","M2","M3","M4"]:
    wn_m, sps, pos = load_map(FILES[mname])
    # process all spectra
    bc_list = []
    for sp in sps:
        bc_list.append(process(wn_m, sp))
    map_data[mname] = {"wn": wn_m, "raw": sps, "bc": np.array(bc_list), "pos": pos}
    print(f"  {mname}: {len(pos)} spectra")

# ── COMPUTE COSINE SIMILARITY FOR ALL SPECTRA ─────────────────────────────────
WN_LO, WN_HI = 650.0, 1750.0
mask_ref = (wn_ref >= WN_LO) & (wn_ref <= WN_HI)
ref_dna_n = norm_max(sp_ref_bc[mask_ref])
wn_ref_dna = wn_ref[mask_ref]

results = []
for mname, d in map_data.items():
    wn_m = d["wn"]
    mask_m = (wn_m >= WN_LO) & (wn_m <= WN_HI)
    wn_dna = wn_m[mask_m]
    for i, bc in enumerate(d["bc"]):
        sp_dna = bc[mask_m]
        sp_n = norm_max(sp_dna)
        sp_interp = np.interp(wn_ref_dna, wn_dna, sp_n)
        cos = cosine_sim(sp_interp, ref_dna_n)
        if cos >= 0.30:   conf = "HIGH"
        elif cos >= 0.20: conf = "MODERATE"
        elif cos >= 0.10: conf = "LOW"
        else:             conf = "NOISE"
        results.append({"map": mname, "idx": i, "cos": cos, "conf": conf,
                        "pos": d["pos"][i]})

results.sort(key=lambda r: r["cos"], reverse=True)
high_mod = [r for r in results if r["conf"] in ("HIGH","MODERATE")]
print(f"\nHIGH+MODERATE spectra: {len(high_mod)}")

# ── STYLE SETTINGS ────────────────────────────────────────────────────────────
MAP_COLORS = {"M1":"#E84040","M2":"#F5A623","M3":"#2E86AB","M4":"#27AE60"}
MAP_STYLE  = {"M1":"-","M2":"--","M3":"-.","M4":":"}
REF_COLOR  = "#222222"
BAND_COLOR = "#8B00FF"   # purple dotted lines for DNA band positions

plt.rcParams.update({
    "font.family":"DejaVu Sans","font.size":10,
    "axes.titlesize":11,"axes.labelsize":10,
    "xtick.labelsize":9,"ytick.labelsize":9,
    "legend.fontsize":8,"figure.dpi":180,
})

def add_band_markers(ax, wn_lo, wn_hi, ymax, label_offset=0.05, short=False):
    """Draw vertical dotted lines + labels for DNA bands in the given wn range."""
    for label, wn_pos in DNA_BANDS:
        if wn_lo <= wn_pos <= wn_hi:
            ax.axvline(wn_pos, color=BAND_COLOR, linewidth=0.8, linestyle=":",
                       alpha=0.7, zorder=1)
            if not short:
                ax.text(wn_pos, ymax*(0.97-label_offset),
                        f"{wn_pos:.0f}", ha="center", va="top",
                        fontsize=6.5, color=BAND_COLOR, rotation=90,
                        bbox=dict(boxstyle="round,pad=0.1",fc="white",ec="none",alpha=0.7))

def set_spine(ax):
    for sp in ax.spines.values(): sp.set_linewidth(0.7)
    ax.tick_params(width=0.7)

# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 1: CaF2 reference spectrum with all DNA band labels
# ══════════════════════════════════════════════════════════════════════════════
print("\nPlot 1: CaF2 reference...")
fig, ax = plt.subplots(figsize=(14, 5))
mask = (wn_ref >= 400) & (wn_ref <= 1800)

ax.plot(wn_ref[mask], raw_ref[mask], color="#AEC7E8", lw=0.8, alpha=0.7, label="Raw")
ax.plot(wn_ref[mask], sp_ref_bc[mask], color=REF_COLOR, lw=1.4, label="Baseline-corrected")

ymax = sp_ref_bc[mask].max() * 1.25
add_band_markers(ax, 400, 1800, ymax)

# annotate peaks
for label, wn_pos in DNA_BANDS:
    if 400 <= wn_pos <= 1800:
        m = (wn_ref >= wn_pos-10) & (wn_ref <= wn_pos+10)
        I = sp_ref_bc[m].max() if m.any() else 0
        if I > 1:
            ax.annotate(f"{wn_pos:.0f}\n{label}", xy=(wn_pos, I),
                        xytext=(0, 14), textcoords="offset points",
                        ha="center", va="bottom", fontsize=6.5, color=BAND_COLOR,
                        arrowprops=dict(arrowstyle="-", color=BAND_COLOR, lw=0.6))

ax.set_xlim(400, 1800)
ax.set_ylim(-2, ymax)
ax.set_xlabel("Raman Shift (cm⁻¹)")
ax.set_ylabel("Intensity (counts, baseline-corrected)")
ax.set_title("CaF₂ Reference — DNA HaCaT 20 ng/µL | 532 nm | 600 gr/mm | 10 s × 4 acc\n"
             "Purple dotted lines: DNA band reference positions")
ax.legend(loc="upper right")
ax.grid(True, alpha=0.2, linestyle="--", lw=0.5)
set_spine(ax)
plt.tight_layout()
fig.savefig(f"{OUT}/Fig1_CaF2_Reference.png", bbox_inches="tight")
plt.close(fig)
print(f"  Saved Fig1_CaF2_Reference.png")

# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 2: Best spectra per map (top 3 per map) vs CaF2 — INDIVIDUAL PANELS
# ══════════════════════════════════════════════════════════════════════════════
print("Plot 2: Best spectra per map vs CaF2...")

# Gather top 3 per map (HIGH preference, then MODERATE)
top_per_map = {}
for mname in ["M1","M2","M3","M4"]:
    mr = [r for r in results if r["map"]==mname]
    top_per_map[mname] = mr[:3]

fig, axes = plt.subplots(4, 1, figsize=(14, 20), sharex=False)
fig.suptitle("Best SERS Spectra vs CaF₂ Reference — DNA HaCaT on Ag/SiNW\n"
             "532 nm | 50X LF | 0.5 s × 4 acc | 600 gr/mm  |  Black = CaF₂ ref",
             fontsize=12, fontweight="bold", y=1.005)

for ax_i, (mname, ax) in enumerate(zip(["M1","M2","M3","M4"], axes)):
    d = map_data[mname]
    wn_m = d["wn"]
    mask_m = (wn_m >= 400) & (wn_m <= 1800)

    # CaF2 reference (normalised to 1)
    ref_plot = norm_max(sp_ref_bc, wn_ref, 650, 1750)
    ref_mask = (wn_ref >= 400) & (wn_ref <= 1800)
    ax.plot(wn_ref[ref_mask], ref_plot[ref_mask],
            color=REF_COLOR, lw=1.2, alpha=0.85, linestyle="--", label="CaF₂ ref (norm.)", zorder=5)

    ymax = ref_plot[ref_mask].max()
    offset_step = 0.25

    # Plot top 3 spectra for this map
    for k, r in enumerate(top_per_map[mname]):
        sp_bc = d["bc"][r["idx"]]
        sp_n  = norm_max(sp_bc, wn_m, 650, 1750)
        offset = (len(top_per_map[mname]) - k - 1) * offset_step
        conf_sym = "★" if r["conf"]=="HIGH" else "◆" if r["conf"]=="MODERATE" else "•"
        label = (f"{mname}-S{r['idx']+1}  {conf_sym}{r['conf']}  "
                 f"cos={r['cos']:.4f}  ({r['pos'][0]:.1f},{r['pos'][1]:.1f}) µm")
        ax.plot(wn_m[mask_m], sp_n[mask_m] + offset,
                color=MAP_COLORS[mname], lw=1.1, alpha=0.85+0.05*k,
                linestyle=["-","--","-."][k], label=label, zorder=4)
        ymax = max(ymax, sp_n[mask_m].max() + offset)

    add_band_markers(ax, 400, 1800, ymax*1.15, label_offset=0.03)

    # Offset guide lines (faint)
    for k in range(len(top_per_map[mname])):
        off = k * offset_step
        if off > 0:
            ax.axhline(off, color="gray", lw=0.4, linestyle=":", alpha=0.5)

    ax.set_xlim(400, 1800)
    ax.set_ylim(-0.05, ymax*1.22)
    ax.set_ylabel("Normalised Intensity + offset")
    ax.set_xlabel("Raman Shift (cm⁻¹)" if ax_i == 3 else "")
    title_loc = "drop-center" if mname != "M2" else "drop-edge"
    ax.set_title(f"{mname}  ({title_loc})  —  top 3 spectra vs CaF₂", fontsize=10)
    ax.legend(loc="upper left", fontsize=7.5, framealpha=0.85)
    ax.grid(True, alpha=0.15, linestyle="--", lw=0.5)
    set_spine(ax)

plt.tight_layout()
fig.savefig(f"{OUT}/Fig2_Best_Spectra_vs_CaF2.png", bbox_inches="tight", dpi=180)
plt.close(fig)
print(f"  Saved Fig2_Best_Spectra_vs_CaF2.png")

# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 3: Mean spectrum per map vs CaF2 — single overlay panel
# ══════════════════════════════════════════════════════════════════════════════
print("Plot 3: Mean spectra vs CaF2...")

fig, ax = plt.subplots(figsize=(14, 6))

ref_n = norm_max(sp_ref_bc, wn_ref, 650, 1750)
ref_m = (wn_ref >= 400) & (wn_ref <= 1800)
ax.plot(wn_ref[ref_m], ref_n[ref_m],
        color=REF_COLOR, lw=2.0, linestyle="--", label="CaF₂ ref", zorder=10, alpha=0.9)

ymax = ref_n[ref_m].max()
offset_step = 0.30
for k, mname in enumerate(["M1","M2","M3","M4"]):
    d = map_data[mname]
    wn_m = d["wn"]
    mask_m = (wn_m >= 400) & (wn_m <= 1800)
    # Mean of all baseline-corrected spectra
    mean_bc = d["bc"].mean(axis=0)
    mean_n  = norm_max(mean_bc, wn_m, 650, 1750)
    # SD band
    std_bc  = d["bc"].std(axis=0)
    std_n   = std_bc / max(mean_bc[(wn_m>=650)&(wn_m<=1750)].max(), 1e-9)

    offset = k * offset_step
    label_loc = "drop-edge" if mname=="M2" else "drop-center"
    label = f"{mname} ({label_loc})  mean of {len(d['pos'])} spectra  [+{offset:.2f}]"
    color = MAP_COLORS[mname]
    ax.fill_between(wn_m[mask_m],
                    mean_n[mask_m]+offset - std_n[mask_m]*0.5,
                    mean_n[mask_m]+offset + std_n[mask_m]*0.5,
                    color=color, alpha=0.15, zorder=2)
    ax.plot(wn_m[mask_m], mean_n[mask_m]+offset,
            color=color, lw=1.5, linestyle=MAP_STYLE[mname],
            label=label, zorder=5)
    ymax = max(ymax, (mean_n[mask_m]+offset).max())

ymax_plot = ymax * 1.22
add_band_markers(ax, 400, 1800, ymax_plot, label_offset=0.04)

ax.set_xlim(400, 1800)
ax.set_ylim(-0.05, ymax_plot)
ax.set_xlabel("Raman Shift (cm⁻¹)")
ax.set_ylabel("Normalised Intensity (offset per map)")
ax.set_title("Mean SERS Spectra (M1–M4) vs CaF₂ Reference — DNA HaCaT 20 ng/µL on Ag/SiNW\n"
             "Shaded band = ±½ SD  |  Purple dotted = CaF₂ DNA band positions  |  Black dashed = CaF₂ ref",
             fontsize=10)
ax.legend(loc="upper left", fontsize=8.5, framealpha=0.9)
ax.grid(True, alpha=0.18, linestyle="--", lw=0.5)
set_spine(ax)
plt.tight_layout()
fig.savefig(f"{OUT}/Fig3_Mean_Spectra_vs_CaF2.png", bbox_inches="tight", dpi=180)
plt.close(fig)
print(f"  Saved Fig3_Mean_Spectra_vs_CaF2.png")

# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 4: Zoom-in on 4 DNA band regions — mean spectra + CaF2
# ══════════════════════════════════════════════════════════════════════════════
print("Plot 4: Zoom-in regions (mean spectra)...")

fig, axes = plt.subplots(2, 2, figsize=(16, 10))
axes = axes.flatten()
fig.suptitle("Zoom-in: DNA Band Regions — Mean SERS Spectra vs CaF₂ Reference\n"
             "DNA HaCaT 20 ng/µL on Ag/SiNW  |  Shaded = ±½ SD per map",
             fontsize=11, fontweight="bold")

# Pre-compute mean/std per map and CaF2 on a common grid
common_wn = np.linspace(400, 1800, 2800)
ref_common = np.interp(common_wn, wn_ref, sp_ref_bc)
ref_common_n = ref_common / max(ref_common[(common_wn>=650)&(common_wn<=1750)].max(), 1e-9)

map_means_n = {}
map_stds_n  = {}
for mname, d in map_data.items():
    wn_m = d["wn"]
    interp_bcs = np.array([np.interp(common_wn, wn_m, bc) for bc in d["bc"]])
    mn = interp_bcs.mean(axis=0)
    sd = interp_bcs.std(axis=0)
    peak = max(mn[(common_wn>=650)&(common_wn<=1750)].max(), 1e-9)
    map_means_n[mname] = mn / peak
    map_stds_n[mname]  = sd / peak

for ax_i, (zlabel, z_lo, z_hi) in enumerate(ZOOM_REGIONS):
    ax = axes[ax_i]
    zmask = (common_wn >= z_lo) & (common_wn <= z_hi)
    wn_z = common_wn[zmask]

    ref_z = ref_common_n[zmask]
    all_vals = [ref_z]

    # CaF2 reference
    ax.plot(wn_z, ref_z, color=REF_COLOR, lw=2.2, linestyle="--",
            label="CaF₂ ref", zorder=10, alpha=0.9)

    for k, mname in enumerate(["M1","M2","M3","M4"]):
        mn_z = map_means_n[mname][zmask]
        sd_z = map_stds_n[mname][zmask]
        color = MAP_COLORS[mname]
        ax.fill_between(wn_z, mn_z-sd_z*0.5, mn_z+sd_z*0.5,
                        color=color, alpha=0.18, zorder=2)
        ax.plot(wn_z, mn_z, color=color, lw=1.5,
                linestyle=MAP_STYLE[mname],
                label=f"{mname}", zorder=5)
        all_vals.append(mn_z + sd_z*0.5)

    ymax_z = np.concatenate(all_vals).max() * 1.28

    # DNA band markers in this zoom window
    bands_in_zone = [(lbl, wn_p) for lbl, wn_p in DNA_BANDS if z_lo <= wn_p <= z_hi]
    for lbl, wn_p in bands_in_zone:
        ax.axvline(wn_p, color=BAND_COLOR, lw=0.9, linestyle=":", alpha=0.8, zorder=1)
        ax.text(wn_p, ymax_z*0.98, f"{wn_p:.0f}\n{lbl}",
                ha="center", va="top", fontsize=7, color=BAND_COLOR,
                rotation=0 if len(bands_in_zone)<=4 else 90,
                bbox=dict(boxstyle="round,pad=0.15",fc="white",ec="none",alpha=0.8))

    ax.set_xlim(z_lo, z_hi)
    ax.set_ylim(-0.03, ymax_z)
    ax.set_title(zlabel, fontsize=9.5)
    ax.set_xlabel("Raman Shift (cm⁻¹)")
    ax.set_ylabel("Normalised Intensity")
    ax.legend(loc="upper right", fontsize=7.5, framealpha=0.9, ncol=2)
    ax.grid(True, alpha=0.18, linestyle="--", lw=0.5)
    set_spine(ax)

plt.tight_layout()
fig.savefig(f"{OUT}/Fig4_ZoomIn_Regions.png", bbox_inches="tight", dpi=180)
plt.close(fig)
print(f"  Saved Fig4_ZoomIn_Regions.png")

# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 5: Individual HIGH-confidence spectra (raw vs processed) for best hotspots
# ══════════════════════════════════════════════════════════════════════════════
print("Plot 5: Raw vs processed for top HIGH-confidence spectra...")

# Take the single best spectrum from each map
best_per_map = {}
for mname in ["M1","M2","M3","M4"]:
    mr = [r for r in results if r["map"]==mname]
    best_per_map[mname] = mr[0]

fig, axes = plt.subplots(2, 2, figsize=(16, 10))
axes = axes.flatten()
fig.suptitle("Raw vs Baseline-Corrected SERS Spectra — Best Hotspot per Map\n"
             "DNA HaCaT 20 ng/µL on Ag/SiNW | 532 nm | 50X LF | 0.5 s × 4 acc",
             fontsize=11, fontweight="bold")

for ax_i, mname in enumerate(["M1","M2","M3","M4"]):
    ax = axes[ax_i]
    r  = best_per_map[mname]
    d  = map_data[mname]
    wn_m = d["wn"]
    raw  = d["raw"][r["idx"]]
    bc   = d["bc"][r["idx"]]
    mask_m = (wn_m >= 400) & (wn_m <= 1800)

    raw_n = raw / max(raw[mask_m].max(), 1e-9)
    bc_n  = bc  / max(bc[(wn_m>=650)&(wn_m<=1750)].max(), 1e-9)

    ax.plot(wn_m[mask_m], raw_n[mask_m],
            color="#AAAAAA", lw=0.8, alpha=0.7, label="Raw")
    ax.plot(wn_m[mask_m], bc_n[mask_m],
            color=MAP_COLORS[mname], lw=1.4, label="Baseline-corrected")

    # CaF2 reference (dotted)
    ref_n2 = sp_ref_bc / max(sp_ref_bc[(wn_ref>=650)&(wn_ref<=1750)].max(), 1e-9)
    ref_m2 = (wn_ref >= 400) & (wn_ref <= 1800)
    ax.plot(wn_ref[ref_m2], ref_n2[ref_m2]*0.8,
            color=REF_COLOR, lw=1.0, linestyle=":", alpha=0.7, label="CaF₂ ref (×0.8)")

    ymax_ax = max(bc_n[mask_m].max(), raw_n[mask_m].max()) * 1.3
    add_band_markers(ax, 400, 1800, ymax_ax, label_offset=0.04, short=True)

    loc_str = "drop-edge" if mname=="M2" else "drop-center"
    conf_sym = "★" if r["conf"]=="HIGH" else "◆"
    ax.set_title(f"{mname}-S{r['idx']+1}  {conf_sym}{r['conf']}  cos={r['cos']:.4f}  "
                 f"({r['pos'][0]:.1f},{r['pos'][1]:.1f}) µm  [{loc_str}]",
                 fontsize=9.5)
    ax.set_xlim(400, 1800)
    ax.set_ylim(-0.05, ymax_ax)
    ax.set_xlabel("Raman Shift (cm⁻¹)")
    ax.set_ylabel("Normalised Intensity")
    ax.legend(loc="upper right", fontsize=8, framealpha=0.9)
    ax.grid(True, alpha=0.18, linestyle="--", lw=0.5)
    set_spine(ax)

plt.tight_layout()
fig.savefig(f"{OUT}/Fig5_Raw_vs_Processed_Best.png", bbox_inches="tight", dpi=180)
plt.close(fig)
print(f"  Saved Fig5_Raw_vs_Processed_Best.png")

# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 6: 4-panel zoom per map — best spectrum vs CaF2 in each region
# ══════════════════════════════════════════════════════════════════════════════
print("Plot 6: Per-map zoom panels (best spectrum in each region)...")

for mname in ["M1","M2","M3","M4"]:
    r = best_per_map[mname]
    d = map_data[mname]
    wn_m = d["wn"]
    bc   = d["bc"][r["idx"]]
    bc_n = bc / max(bc[(wn_m>=650)&(wn_m<=1750)].max(), 1e-9)
    ref_n3 = sp_ref_bc / max(sp_ref_bc[(wn_ref>=650)&(wn_ref<=1750)].max(), 1e-9)

    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    loc_str = "drop-edge" if mname=="M2" else "drop-center"
    fig.suptitle(f"{mname}-S{r['idx']+1} ({loc_str})  vs  CaF₂ reference — Zoom by DNA Band Region\n"
                 f"cos={r['cos']:.4f}  |  conf={r['conf']}  |  pos=({r['pos'][0]:.1f},{r['pos'][1]:.1f}) µm",
                 fontsize=10.5, fontweight="bold")

    for ax_i, (zlabel, z_lo, z_hi) in enumerate(ZOOM_REGIONS):
        ax = axes[ax_i]
        mz = (wn_m >= z_lo) & (wn_m <= z_hi)
        rz = (wn_ref >= z_lo) & (wn_ref <= z_hi)

        ax.plot(wn_ref[rz], ref_n3[rz], color=REF_COLOR, lw=1.8,
                linestyle="--", alpha=0.85, label="CaF₂ ref", zorder=5)
        ax.plot(wn_m[mz], bc_n[mz], color=MAP_COLORS[mname], lw=1.5,
                label=f"{mname}-S{r['idx']+1}", zorder=4)

        all_y = np.concatenate([ref_n3[rz], bc_n[mz]])
        ymax_z = all_y.max() * 1.35 if all_y.size else 1.2

        bands_in_zone = [(lbl, wn_p) for lbl, wn_p in DNA_BANDS if z_lo <= wn_p <= z_hi]
        for lbl, wn_p in bands_in_zone:
            ax.axvline(wn_p, color=BAND_COLOR, lw=1.0, linestyle=":", alpha=0.8)
            ax.text(wn_p, ymax_z*0.99, f"{wn_p:.0f}", ha="center", va="top",
                    fontsize=7.5, color=BAND_COLOR, fontweight="bold")
            ax.text(wn_p, ymax_z*0.87, lbl, ha="center", va="top",
                    fontsize=6, color=BAND_COLOR, rotation=0)

        ax.set_xlim(z_lo, z_hi)
        ax.set_ylim(-0.03, ymax_z)
        short_label = zlabel.split("\n")[0]
        ax.set_title(short_label, fontsize=8.5)
        ax.set_xlabel("Raman Shift (cm⁻¹)")
        if ax_i == 0: ax.set_ylabel("Normalised Intensity")
        ax.legend(fontsize=7.5, framealpha=0.9)
        ax.grid(True, alpha=0.18, linestyle="--", lw=0.5)
        set_spine(ax)

    plt.tight_layout()
    fname = f"{OUT}/Fig6_{mname}_ZoomRegions.png"
    fig.savefig(fname, bbox_inches="tight", dpi=180)
    plt.close(fig)
    print(f"  Saved Fig6_{mname}_ZoomRegions.png")

# ══════════════════════════════════════════════════════════════════════════════
print("\n✓ All figures saved to:", OUT)
print("Files created:")
for fn in sorted(os.listdir(OUT)):
    if fn.startswith("Fig") and fn.endswith(".png"):
        size_kb = os.path.getsize(f"{OUT}/{fn}") // 1024
        print(f"  {fn}  ({size_kb} KB)")
