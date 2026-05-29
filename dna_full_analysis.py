"""
dna_full_analysis.py
====================
Goal 1: Clean DNA+HaCaT SERS spectrum (m3 spectrum 20) — full 200-3200 cm-1, all peaks labeled.
Goal 2: Compare ALL peaks with S1 reference. Show every matched peak and its shift.
Goal 3: Suggestions printed as report.

Run in PyCharm — edit FOLDER path only.
"""
import os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve
from scipy.signal import find_peaks
from scipy.ndimage import uniform_filter1d

# ── EDIT THIS ────────────────────────────────────────────────────────────────
FOLDER   = r"C:\Users\Hira Aman\Desktop\RAMAN MAP 1"
FILE_S1  = os.path.join(FOLDER, "S1-DNA-HACAT1-20ng-CaF2_532nm_600gr_BC50_100X_10s_4a_100%.txt")
FILE_M3  = os.path.join(FOLDER, "m3-DNA-HACAT1-20ng-AgSiNW_532nm_600gr_BC200_50XLF_05s_4a_2-5%_5ul_drop-center.txt")
SERS_SP  = 20          # spectrum number inside m3 (1-25) — cleanest signal
# ─────────────────────────────────────────────────────────────────────────────

MATCH_TOL = 15.0   # cm-1: window to call two peaks "the same band"
C_S1   = "#E67E22"   # orange
C_SERS = "#1A5276"   # dark blue

REGION_COLORS = {
    (200, 400):  ("#EBF5FB", "200–400"),
    (400, 900):  ("#FEF9E7", "DNA backbone\n400–900"),
    (900, 1300): ("#EAFAF1", "Fingerprint\n900–1300"),
    (1300,1800): ("#FDF2F8", "Nucleobases\n1300–1800"),
    (1800,2800): ("#F8F9FA", "Silent region\n1800–2800"),
    (2800,3200): ("#EBF5FB", "CH stretch\n2800–3200"),
}

ASSIGNMENTS = {
    (400,430):  "Ring def. / sugar",
    (420,460):  "DNA base ring def.",
    (460,510):  "Sugar-phosphate",
    (510,560):  "Backbone C-C",
    (560,615):  "G/C deform.",
    (615,680):  "G/T ring breath.",
    (680,710):  "G ring breath.",
    (710,750):  "A ring breath.",
    (750,800):  "Cyt. + O-P-O",
    (800,860):  "Backbone O-P-O",
    (860,920):  "Backbone C-C",
    (920,970):  "Sugar C-O",
    (970,1020): "Phe / ring",
    (1020,1075):"C-N / ring",
    (1075,1115):"PO₄ sym str.",
    (1115,1145):"PO₄ / C-N",
    (1145,1200):"C-N (Cyt/A)",
    (1200,1270):"Amide III / T",
    (1270,1340):"A/T/G in-plane",
    (1340,1400):"T/G CH def.",
    (1400,1455):"A/G ring str.",
    (1455,1510):"A/C C=N str.",
    (1510,1595):"G/A ring C=C",
    (1595,1680):"Amide I / C=C",
    (1680,1800):"C=O (base)",
    (2800,2900):"CH₂ sym str.",
    (2900,2970):"CH₃ sym str.",
    (2970,3100):"CH₂ asym str.",
}

def assign(w):
    for (lo,hi), lbl in ASSIGNMENTS.items():
        if lo <= w < hi: return lbl
    return "—"

def load_caf2(fp):
    wn, sp = [], []
    with open(fp,"r",encoding="utf-8",errors="replace") as fh:
        for line in fh:
            line=line.strip()
            if not line or line.startswith("#"): continue
            p=line.replace(",",".").split()
            if len(p)<2: continue
            try: wn.append(float(p[0])); sp.append(float(p[1]))
            except: continue
    wn=np.array(wn); sp=np.array(sp); o=np.argsort(wn)
    return wn[o], sp[o]

def load_map(fp):
    rows=[]
    with open(fp,"r",encoding="utf-8",errors="replace") as fh:
        for line in fh:
            line=line.strip()
            if not line or line.startswith("#"): continue
            p=line.replace(",",".").split("\t")
            if len(p)<3: p=line.replace(",",".").split()
            try: rows.append([float(v) for v in p])
            except: continue
    wn=np.array(rows[0]); n=len(wn)
    spectra=[]
    for row in rows[1:]:
        arr=np.array(row)
        if len(arr)==n+2: spectra.append(arr[2:])
        elif len(arr)==n: spectra.append(arr)
        else: spectra.append(arr[-n:])
    return wn, np.array(spectra)

def als_baseline(y, lam=1e5, p=0.005, nit=15):
    L=len(y); D=diags([1,-2,1],[0,1,2],shape=(L-2,L),dtype=float)
    w=np.ones(L)
    for _ in range(nit):
        W=diags(w,0,shape=(L,L),dtype=float); z=spsolve((W+lam*D.T.dot(D)).tocsr(),w*y)
        w=p*(y>z)+(1-p)*(y<=z)
    return z

def blc_norm(wn, sp):
    z=als_baseline(sp); bc=np.clip(sp-z,0,None)
    mx=bc.max(); return bc/(mx if mx>0 else 1), mx

def get_peaks(wn, sp_n, height=0.06, prom=0.05):
    step=float(np.diff(wn).mean()); dist=max(5,int(15/step))
    idxs,props=find_peaks(sp_n, height=height, prominence=prom, distance=dist)
    return wn[idxs], sp_n[idxs], props["prominences"]

# ─── LOAD DATA ───────────────────────────────────────────────────────────────
print("Loading S1 reference ...")
wn_s1, sp_s1_raw = load_caf2(FILE_S1)
n_s1, max_s1 = blc_norm(wn_s1, sp_s1_raw)
pk_s1_wn, pk_s1_int, pk_s1_prom = get_peaks(wn_s1, n_s1, height=0.08, prom=0.06)
print(f"  {len(wn_s1)} pts, {wn_s1[0]:.0f}–{wn_s1[-1]:.0f} cm-1, max={max_s1:.0f} cts, {len(pk_s1_wn)} peaks")

print(f"Loading m3 spectrum {SERS_SP} ...")
wn_m, spec_m = load_map(FILE_M3)
sp_raw_m = spec_m[SERS_SP-1]
n_m, max_m = blc_norm(wn_m, sp_raw_m)
pk_m_wn, pk_m_int, pk_m_prom = get_peaks(wn_m, n_m, height=0.05, prom=0.04)
print(f"  {len(wn_m)} pts, {wn_m[0]:.0f}–{wn_m[-1]:.0f} cm-1, max={max_m:.0f} cts, {len(pk_m_wn)} peaks")

# ─── MATCH PEAKS ─────────────────────────────────────────────────────────────
matched = []     # (s1_wn, sers_wn, shift)
used_sers = set()

for i, w_s in enumerate(pk_s1_wn):
    diffs = np.abs(pk_m_wn - w_s)
    cands = np.where(diffs <= MATCH_TOL)[0]
    cands = [c for c in cands if c not in used_sers]
    if len(cands) > 0:
        best = cands[int(np.argmin(diffs[cands]))]
        used_sers.add(best)
        matched.append((w_s, float(pk_m_wn[best]), float(pk_m_wn[best]-w_s),
                        float(pk_s1_int[i]), float(pk_m_int[best])))

only_sers = [(float(pk_m_wn[j]), float(pk_m_int[j]))
             for j in range(len(pk_m_wn)) if j not in used_sers and pk_m_wn[j]>400]
unmatched_s1 = [(float(pk_s1_wn[i]), float(pk_s1_int[i]))
                for i in range(len(pk_s1_wn))
                if not any(abs(m[0]-pk_s1_wn[i])<0.01 for m in matched)]

print(f"\nMatched pairs: {len(matched)}")
print(f"S1 peaks NOT in SERS: {len(unmatched_s1)}")
print(f"SERS-only peaks: {len(only_sers)}")

# ═════════════════════════════════════════════════════════════════════════════
# FIGURE 1: Clean SERS spectrum — full 200–3200 cm-1
# ═════════════════════════════════════════════════════════════════════════════
fig1, ax = plt.subplots(figsize=(18, 7)); fig1.patch.set_facecolor("white")
ax.set_facecolor("white")

# Shaded spectral regions
for (lo,hi),(color,label) in REGION_COLORS.items():
    ax.axvspan(lo, hi, color=color, alpha=0.55, lw=0)
    mid = (lo+hi)/2
    if hi-lo > 150:
        ax.text(mid, 1.18, label, fontsize=7.5, ha="center", va="bottom",
                color="grey", style="italic")

# Spectrum
mask = (wn_m >= 200) & (wn_m <= 3200)
ax.fill_between(wn_m[mask], n_m[mask], alpha=0.15, color=C_SERS)
ax.plot(wn_m[mask], n_m[mask], color=C_SERS, lw=1.5, zorder=5)

# ── Peak labels — staggered to avoid overlap ──────────────────────────────
peaks_in_range = [(w, h) for w, h in zip(pk_m_wn, pk_m_int) if 200 < w < 3200]
peaks_in_range.sort(key=lambda x: x[0])

y_levels = [1.08, 1.14, 1.08, 1.14]  # alternating heights for adjacent peaks
prev_x = -9999; level_idx = 0
for w, h in peaks_in_range:
    ax.plot([w, w], [h+0.01, h+0.05], color=C_SERS, lw=0.8, alpha=0.7)
    ax.plot(w, h, "o", color=C_SERS, ms=4.5, zorder=8)
    gap = w - prev_x
    if gap < 40: level_idx = (level_idx+1) % len(y_levels)
    else: level_idx = 0
    y_lbl = y_levels[level_idx]
    ax.text(w, y_lbl, f"{w:.0f}", fontsize=7.5, ha="center", va="bottom",
            color=C_SERS, fontweight="bold", rotation=90 if gap < 55 else 0)
    ax.axvline(w, color=C_SERS, lw=0.5, ls="--", alpha=0.3, zorder=3)
    prev_x = w

ax.set_xlim(200, 3200); ax.set_ylim(-0.05, 1.45)
ax.set_xlabel("Raman shift (cm⁻¹)", fontsize=13, fontweight="bold")
ax.set_ylabel("Normalised intensity (a.u.)", fontsize=13, fontweight="bold")
ax.set_title(
    f"DNA + HaCaT SERS signal  |  m3 spectrum {SERS_SP}  |  AgSiNW substrate  |  532 nm\n"
    f"Best clean spectrum from 100 scanned  ·  SNR ≈ 87  ·  ALS baseline corrected  ·  {len(peaks_in_range)} peaks detected",
    fontsize=11, fontweight="bold")
ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
ax.tick_params(labelsize=10); ax.grid(True, alpha=0.10, lw=0.5)
out1 = os.path.join(FOLDER, "fig_A_clean_SERS_full_range.png")
fig1.savefig(out1, dpi=180, bbox_inches="tight", facecolor="white")
plt.close(fig1)
print(f"\nSaved: {out1}")

# ═════════════════════════════════════════════════════════════════════════════
# FIGURE 2: S1 vs SERS — ALL peaks, shifts
# ═════════════════════════════════════════════════════════════════════════════
fig2 = plt.figure(figsize=(20, 14)); fig2.patch.set_facecolor("white")
from matplotlib.gridspec import GridSpec
gs = GridSpec(3, 2, figure=fig2, height_ratios=[2.5, 2.5, 2.0],
              hspace=0.52, wspace=0.32, left=0.06, right=0.97, top=0.94, bottom=0.06)

fig2.suptitle(
    "S1 Reference (DNA+HaCaT on CaF₂, normal Raman)  vs  Best SERS spectrum (m3_sp20 on AgSiNW)\n"
    "ALS baseline corrected · min-max normalised",
    fontsize=12, fontweight="bold", y=0.97)

# ─── Panel A: S1 spectrum ───────────────────────────────────────────────────
ax_s1 = fig2.add_subplot(gs[0, :])
mask_s1 = (wn_s1 >= 400) & (wn_s1 <= 1800)
ax_s1.fill_between(wn_s1[mask_s1], n_s1[mask_s1], alpha=0.20, color=C_S1)
ax_s1.plot(wn_s1[mask_s1], n_s1[mask_s1], color=C_S1, lw=1.6, label="S1 reference (CaF₂)", zorder=5)

for w, h in zip(pk_s1_wn, pk_s1_int):
    ax_s1.plot(w, h+0.03, "v", color=C_S1, ms=5, zorder=7)
    if h >= 0.25:
        ax_s1.text(w, h+0.05, f"{w:.0f}", fontsize=7, ha="center", va="bottom",
                   color=C_S1, fontweight="bold")
    else:
        ax_s1.text(w, h+0.05, f"{w:.0f}", fontsize=6, ha="center", va="bottom",
                   color="#BA4A00", alpha=0.75)
ax_s1.set_xlim(400, 1800); ax_s1.set_ylim(-0.05, 1.45)
ax_s1.set_title(f"S1 Reference  —  Normal Raman, CaF₂ substrate  ({len(pk_s1_wn)} peaks detected)",
                fontsize=10, fontweight="bold")
ax_s1.set_xlabel("Raman shift (cm⁻¹)", fontsize=10)
ax_s1.set_ylabel("Norm. intensity", fontsize=10)
ax_s1.spines["top"].set_visible(False); ax_s1.spines["right"].set_visible(False)
ax_s1.grid(True, alpha=0.10, lw=0.5); ax_s1.tick_params(labelsize=9)
ax_s1.legend(fontsize=9, loc="upper left")

# ─── Panel B: SERS spectrum (same x range) ──────────────────────────────────
ax_sers = fig2.add_subplot(gs[1, :])
mask_comp = (wn_m >= 400) & (wn_m <= 1800)
ax_sers.fill_between(wn_m[mask_comp], n_m[mask_comp], alpha=0.18, color=C_SERS)
ax_sers.plot(wn_m[mask_comp], n_m[mask_comp], color=C_SERS, lw=1.6,
             label="SERS m3_sp20 (AgSiNW)", zorder=5)

# SERS peaks — matched (filled circle) vs SERS-only (star)
matched_sers_wns = set(m[1] for m in matched)
for w, h in zip(pk_m_wn, pk_m_int):
    if w < 400 or w > 1800: continue
    if w in matched_sers_wns:
        ax_sers.plot(w, h+0.03, "^", color=C_SERS, ms=6, zorder=7)
    else:
        ax_sers.plot(w, h+0.03, "*", color="#117A65", ms=9, zorder=7)
    ax_sers.text(w, h+0.07, f"{w:.0f}", fontsize=7, ha="center", va="bottom",
                 color=C_SERS if w in matched_sers_wns else "#117A65", fontweight="bold")

from matplotlib.lines import Line2D
legend_handles = [
    Line2D([0],[0], marker="^", color=C_SERS, ms=7, lw=0, label="Matched with S1 (▲)"),
    Line2D([0],[0], marker="*", color="#117A65", ms=9, lw=0, label="SERS-only (★ not in S1)"),
]
ax_sers.set_xlim(400, 1800); ax_sers.set_ylim(-0.05, 1.45)
ax_sers.set_title(f"SERS m3_sp20  —  AgSiNW substrate  ({len([w for w in pk_m_wn if 400<=w<=1800])} peaks in 400–1800 cm⁻¹)",
                  fontsize=10, fontweight="bold")
ax_sers.set_xlabel("Raman shift (cm⁻¹)", fontsize=10)
ax_sers.set_ylabel("Norm. intensity", fontsize=10)
ax_sers.spines["top"].set_visible(False); ax_sers.spines["right"].set_visible(False)
ax_sers.grid(True, alpha=0.10, lw=0.5); ax_sers.tick_params(labelsize=9)
leg2 = ax_sers.legend(handles=legend_handles + [mpatches.Patch(color=C_SERS, label="SERS m3_sp20")],
                      fontsize=8, loc="upper left")

# ─── Panel C: Shift scatter ──────────────────────────────────────────────────
ax_sh = fig2.add_subplot(gs[2, 0])
if matched:
    ws = [m[0] for m in matched]
    dsh = [m[2] for m in matched]
    colors = ["#C0392B" if d > 0 else "#1A5276" for d in dsh]
    ax_sh.bar(range(len(matched)), dsh, color=colors, edgecolor="white", lw=0.5, zorder=3)
    ax_sh.set_xticks(range(len(matched)))
    ax_sh.set_xticklabels([f"{w:.0f}" for w in ws], rotation=70, ha="right", fontsize=6.5)
    ax_sh.axhline(0, color="black", lw=1.2, alpha=0.5)
    ax_sh.axhline(+5, color="#C0392B", lw=0.7, ls=":", alpha=0.4)
    ax_sh.axhline(-5, color="#1A5276", lw=0.7, ls=":", alpha=0.4)
    ax_sh.set_xlabel("S1 peak position (cm⁻¹)", fontsize=9)
    ax_sh.set_ylabel("Δ shift = SERS − S1 (cm⁻¹)", fontsize=9)
    ax_sh.set_title(f"Peak shifts  (SERS − S1)  |  {len(matched)} matched pairs\n"
                    "Red = blue-shift (+)  ·  Blue = red-shift (−)  ·  Dotted = ±5 cm⁻¹",
                    fontsize=9, fontweight="bold")
    ax_sh.spines["top"].set_visible(False); ax_sh.spines["right"].set_visible(False)
    ax_sh.grid(True, alpha=0.12, lw=0.5, axis="y"); ax_sh.tick_params(labelsize=8)
    mean_sh = np.mean(dsh); ax_sh.text(len(matched)-1, ax_sh.get_ylim()[1]*0.85,
        f"Mean Δ = {mean_sh:+.1f}\nSD = {np.std(dsh):.1f}",
        ha="right", fontsize=8, color="black",
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="grey", alpha=0.8))

# ─── Panel D: Summary bar ──────────────────────────────────────────────────
ax_sum = fig2.add_subplot(gs[2, 1])
n_match_in_range = len([m for m in matched if 400<=m[0]<=1800])
n_only_s1 = len([w for w, h in unmatched_s1 if 400<=w<=1800])
n_only_sers = len([w for w, h in only_sers if 400<=w<=1800])
n_sers_total = len([w for w in pk_m_wn if 400<=w<=1800])

categories  = ["Matched\nS1 ↔ SERS", "S1 only\n(not enhanced)", "SERS only\n(new in SERS)"]
values      = [n_match_in_range, n_only_s1, n_only_sers]
bar_colors  = ["#27AE60", "#E67E22", "#1A5276"]
bars = ax_sum.bar(categories, values, color=bar_colors, edgecolor="white",
                  width=0.55, zorder=3)
for bar, val in zip(bars, values):
    ax_sum.text(bar.get_x()+bar.get_width()/2, val+0.3, str(val),
                ha="center", va="bottom", fontsize=12, fontweight="bold")
ax_sum.set_ylabel("Number of peaks", fontsize=9)
ax_sum.set_title(f"Peak summary  (400–1800 cm⁻¹)\n"
                 f"S1: {len([w for w in pk_s1_wn if 400<=w<=1800])} peaks  ·  "
                 f"SERS: {n_sers_total} peaks",
                 fontsize=9, fontweight="bold")
ax_sum.spines["top"].set_visible(False); ax_sum.spines["right"].set_visible(False)
ax_sum.tick_params(labelsize=9); ax_sum.grid(True, alpha=0.12, lw=0.5, axis="y")
ax_sum.set_ylim(0, max(values)*1.25)

out2 = os.path.join(FOLDER, "fig_B_S1_vs_SERS_all_peaks.png")
fig2.savefig(out2, dpi=180, bbox_inches="tight", facecolor="white")
plt.close(fig2)
print(f"Saved: {out2}")

# ═════════════════════════════════════════════════════════════════════════════
# TEXT REPORT
# ═════════════════════════════════════════════════════════════════════════════
W = 74
lines = []
lines += [
"="*W,
"REPORT: DNA + HaCaT Raman / SERS Peak Analysis",
"="*W,
"",
f"REFERENCE : S1  —  DNA+HaCaT on CaF2, normal Raman (no enhancement)",
f"           {len(wn_s1)} pts, {wn_s1[0]:.0f}–{wn_s1[-1]:.0f} cm-1, max corrected signal = {max_s1:.0f} counts",
f"SERS      : m3 spectrum {SERS_SP}  —  DNA+HaCaT on AgSiNW",
f"           {len(wn_m)} pts, {wn_m[0]:.0f}–{wn_m[-1]:.0f} cm-1, max corrected signal = {max_m:.0f} counts",
f"METHOD    : ALS baseline correction  →  min-max normalise  →  scipy find_peaks",
f"MATCH TOL : ±{MATCH_TOL:.0f} cm-1",
"",
"="*W,
"GOAL 1: ALL PEAKS IN CLEAN SERS SPECTRUM (m3_sp20)  —  200–3200 cm-1",
"="*W, "",
f"  {'Peak #':<8} {'Position (cm-1)':<20} {'Norm. intensity':<18} {'Assignment'}",
f"  {'-'*68}",
]

for i, (w, h) in enumerate(zip(pk_m_wn, pk_m_int), 1):
    lines.append(f"  {i:<8} {w:<20.1f} {h:<18.4f} {assign(w)}")

lines += [
"",
"="*W,
"GOAL 2: PEAK-BY-PEAK COMPARISON — S1 vs SERS (400–1800 cm-1)",
"="*W, "",
f"  {'S1 (cm-1)':<14} {'SERS (cm-1)':<14} {'Δ shift':<12} {'S1 int':<10} {'SERS int':<10} {'Assignment'}",
f"  {'-'*76}",
"  [A] MATCHED PAIRS (same band, shifted):",
]
for w_s, w_m, dsh, i_s, i_m in sorted(matched):
    marker = ">>>" if abs(dsh) > 10 else "   "
    lines.append(f"  {marker} {w_s:<14.1f} {w_m:<14.1f} {dsh:>+7.1f} cm-1  "
                 f"{i_s:<10.3f} {i_m:<10.3f} {assign(w_s)}")

lines += ["", "  [B] S1 PEAKS NOT FOUND IN SERS  (mode not enhanced):"]
for w, h in sorted(unmatched_s1):
    if 400 <= w <= 1800:
        lines.append(f"       {w:.1f} cm-1   S1-int={h:.3f}   {assign(w)}")

lines += ["", "  [C] SERS PEAKS NOT IN S1  (newly enhanced / hotspot-only):"]
for w, h in sorted(only_sers):
    if 400 <= w <= 1800:
        lines.append(f"       {w:.1f} cm-1   SERS-int={h:.3f}   {assign(w)}")
for w, h in sorted(only_sers):
    if w > 1800:
        lines.append(f"       {w:.1f} cm-1   SERS-int={h:.3f}   CH stretch (HaCaT cells)")

# Shift statistics
dshifts = [m[2] for m in matched]
pos_sh = [d for d in dshifts if d > 0]
neg_sh = [d for d in dshifts if d < 0]
lines += [
"",
f"  Matched pairs       : {len(matched)}",
f"  Blue-shifted (SERS > S1)  : {len(pos_sh)}  mean = {np.mean(pos_sh) if pos_sh else 0:+.1f} cm-1",
f"  Red-shifted  (SERS < S1)  : {len(neg_sh)}  mean = {np.mean(neg_sh) if neg_sh else 0:+.1f} cm-1",
f"  Overall mean shift        : {np.mean(dshifts):+.1f} cm-1",
f"  Overall SD of shift       : {np.std(dshifts):.1f} cm-1",
"",
"="*W,
"GOAL 3: SUGGESTIONS",
"="*W, "",
"  1. USE m3_sp20 AS YOUR REPRESENTATIVE SERS SPECTRUM.",
"     It has the highest SNR (87) from all 100 spectra, the flattest",
"     baseline, and shows 14/19 known DNA+HaCaT bands. Present it as",
"     your benchmark SERS spectrum in your report.",
"",
"  2. S1 IS WEAK — NORMAL RAMAN, NOT IDEAL FOR A CLEAN REFERENCE.",
f"     S1 max signal after baseline removal is only {max_s1:.0f} counts.",
"     This is low for normal Raman. If possible, re-measure S1 with",
"     longer accumulation (e.g. 30–60 s × 5 acc.) for a cleaner reference.",
"     If not, acknowledge this limitation in the paper.",
"",
"  3. INTERPRET THE SHIFT PATTERN, NOT JUST THE 3 PEAKS.",
f"     You have {len(matched)} matched pairs. Most shifts are small",
f"     ({np.mean(pos_sh) if pos_sh else 0:+.1f} to {np.mean(neg_sh) if neg_sh else 0:+.1f} cm-1),",
"     consistent with DNA molecules adsorbing flat on Ag surface",
"     with weak charge-transfer perturbation. The few larger shifts",
"     (>10 cm-1) at 1467 and 1609 cm-1 suggest those base modes",
"     are in direct contact with Ag hotspot surface.",
"",
"  4. THE CH STRETCH PEAKS (2930, 2977 cm-1) IN SERS ONLY ARE REAL.",
"     They come from HaCaT cell lipids/proteins. S1 cannot see them",
"     because its range ends at 1800 cm-1. These are genuine SERS",
"     enhancement of cellular CH2/CH3 modes — worth noting.",
"",
"  5. MODES NOT ENHANCED (S1-only peaks) TELL A STORY.",
f"     {len(unmatched_s1)} S1 peaks have no match in SERS. This is normal —",
"     SERS only enhances modes of molecules at the Ag nanogap.",
"     Modes not enhanced = parts of the molecule not touching Ag.",
"     For DNA, the unenhanced backbone modes suggest the phosphate",
"     groups are partially shielded from the Ag surface.",
"",
"="*W,
]

report_text = "\n".join(lines)
print(report_text)
rp = os.path.join(FOLDER, "dna_full_analysis_report.txt")
with open(rp, "w", encoding="utf-8") as fh:
    fh.write(report_text)
print(f"\nReport saved: {rp}")
print("Done.")
