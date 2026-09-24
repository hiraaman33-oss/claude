"""
EV Acacia: best AgSiNW (SERS) spectrum vs EV Acacia on CaF2 – raw data, 400-1800 cm-1,
slide-ready figures (16:9).

1. Peaks on raw counts (no smoothing, no baseline correction) with raman_raw.raw_peaks.
2. Best AgSiNW spectrum = the one reproducing the largest fraction of the CaF2 bands in
   400-1800 cm-1 (raw peak within ±MATCH_CM), ties -> more matched, then total SNR.
3. Every peak is re-checked twice, independently:
     check 1: raman_raw.local_check – raw maximum within ±6 cm-1 is the same data point and
              its height above a straight line between the flank minima is >= 3 sigma;
     check 2: peak search repeated with different settings (4 sigma, ±50 cm-1 prominence
              window, 4 cm-1 spacing) must find the peak again within ±2 cm-1.
   Only peaks passing both checks are labelled on the figures; all are listed in the CSV.

Usage:  python3 EV_Acacia_PPT_Compare.py <caf2_dir> <agsinw_dir>
"""
import glob
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from raman_raw import clean, local_check, noise_sigma, raw_peaks, read_single, short, tag

CAF2_DIR, SERS_DIR = sys.argv[1:3]
OUT = "EV_Acacia_PPT_Compare"
os.makedirs(OUT, exist_ok=True)
LO, HI = 400, 1800
MATCH_CM = 8.0
ZOOMS = [(780, 1150), (1150, 1500), (1500, 1800)]
C_CAF2, C_SERS, C_SHARED = "#b03a2e", "#1f4e79", "#2ca02c"


def load(d):
    out = {}
    for f in sorted(glob.glob(os.path.join(d, "*"))):
        out[tag(f)] = dict(file=clean(f), wn_y=read_single(f))
    return out


def checked_peaks(wn, y, substrate):
    sigma = noise_sigma(wn, y)
    acc, rej = raw_peaks(wn, y, substrate, LO, HI, sigma=sigma)
    alt, _ = raw_peaks(wn, y, substrate, LO, HI, k_sigma=4.0, wlen_cm=100.0, min_dist_cm=4.0, sigma=sigma)
    c1, c1pos, c1snr, c2 = [], [], [], []
    for _, p in acc.iterrows():
        f0 = max(15.0, p.FWHM_cm1)
        ok, xm, snr = local_check(wn, y, p.position_cm1, sigma, flank=(f0, f0 + 20.0))
        c1.append(ok)
        c1pos.append(xm)
        c1snr.append(snr)
        c2.append(bool((not alt.empty) and np.min(np.abs(alt.position_cm1.values - p.position_cm1)) <= 2.0))
    acc = acc.assign(check1_local=c1, check1_pos=c1pos, check1_SNR=c1snr, check2_repeat=c2)
    acc["confirmed"] = acc.check1_local & acc.check2_repeat
    return acc, rej, sigma


caf2, sers = load(CAF2_DIR), load(SERS_DIR)
ref_name = next(iter(caf2))  # the CaF2 spectrum supplied
wn_r, y_r = caf2[ref_name]["wn_y"]
ref_pk, ref_rej, ref_sigma = checked_peaks(wn_r, y_r, "CaF2")

# ------------------------------------------------------------------ choose the best AgSiNW spectrum
sel, cand = [], {}
for n, v in sers.items():
    wn, y = v["wn_y"]
    pk, rej, sigma = checked_peaks(wn, y, "SERS")
    cand[n] = (pk, rej, sigma)
    conf = pk[pk.confirmed]
    ref_conf = ref_pk[ref_pk.confirmed]
    matched, snr_sum = 0, 0.0
    for c in ref_conf.position_cm1:
        if conf.empty:
            break
        d = np.abs(conf.position_cm1.values - c)
        if d.min() <= MATCH_CM:
            matched += 1
            snr_sum += conf.SNR.values[d.argmin()]
    sel.append(dict(spectrum=n, file=v["file"], range=f"{wn.min():.0f}-{wn.max():.0f}",
                    confirmed_peaks=len(conf), CaF2_bands=len(ref_conf), CaF2_bands_matched=matched,
                    fraction=round(matched / max(len(ref_conf), 1), 3), matched_SNR_sum=round(snr_sum, 1),
                    noise_sigma=round(sigma, 1)))
sel = pd.DataFrame(sel).sort_values(["fraction", "CaF2_bands_matched", "matched_SNR_sum"], ascending=False)
sel.to_csv(f"{OUT}/selection_AgSiNW.csv", index=False)
best = sel.iloc[0].spectrum
wn_s, y_s = sers[best]["wn_y"]
s_pk, s_rej, s_sigma = cand[best]
print("AgSiNW candidates:\n", sel.to_string(index=False), f"\n-> chosen: {best}")

pd.set_option("display.width", 250)
pd.set_option("display.max_colwidth", 60)
for lab, pk, sg, f in [(f"EV Acacia on CaF2 ({ref_name})", ref_pk, ref_sigma, caf2[ref_name]["file"]),
                       (f"EV Acacia on AgSiNW ({best})", s_pk, s_sigma, sers[best]["file"])]:
    print(f"\n=== {lab}  {f}  noise sigma {sg:.1f} counts ===")
    print(pk.to_string(index=False))

# ------------------------------------------------------------------ band table
rc, sc = ref_pk[ref_pk.confirmed].reset_index(drop=True), s_pk[s_pk.confirmed].reset_index(drop=True)
rows, used = [], set()
for _, r in rc.iterrows():
    d = np.abs(sc.position_cm1.values - r.position_cm1) if not sc.empty else np.array([99.0])
    j = int(d.argmin())
    if d[j] <= MATCH_CM and j not in used:
        used.add(j)
        s = sc.iloc[j]
        rows.append(dict(CaF2_cm1=r.position_cm1, AgSiNW_cm1=s.position_cm1,
                         shift_cm1=round(s.position_cm1 - r.position_cm1, 1), SNR_CaF2=r.SNR, SNR_AgSiNW=s.SNR,
                         FWHM_CaF2=r.FWHM_cm1, FWHM_AgSiNW=s.FWHM_cm1, status="in both",
                         band_CaF2=r.assignment, band_AgSiNW=s.assignment))
    else:
        rows.append(dict(CaF2_cm1=r.position_cm1, AgSiNW_cm1=np.nan, shift_cm1=np.nan, SNR_CaF2=r.SNR,
                         SNR_AgSiNW=np.nan, FWHM_CaF2=r.FWHM_cm1, FWHM_AgSiNW=np.nan,
                         status="only on CaF2" + (f" (nearest AgSiNW {sc.position_cm1.values[j]:.1f}, "
                                                  f"Δ{d[j]:.1f})" if (not sc.empty and d[j] <= 15) else ""),
                         band_CaF2=r.assignment, band_AgSiNW=""))
for j, s in sc.iterrows():
    if j not in used:
        rows.append(dict(CaF2_cm1=np.nan, AgSiNW_cm1=s.position_cm1, shift_cm1=np.nan, SNR_CaF2=np.nan,
                         SNR_AgSiNW=s.SNR, FWHM_CaF2=np.nan, FWHM_AgSiNW=s.FWHM_cm1, status="only on AgSiNW",
                         band_CaF2="", band_AgSiNW=s.assignment))
tab = pd.DataFrame(rows)
tab = tab.assign(_k=tab.CaF2_cm1.fillna(tab.AgSiNW_cm1)).sort_values("_k").drop(columns="_k")
tab.to_csv(f"{OUT}/band_table_400-1800.csv", index=False)
print("\n##### Band table (confirmed peaks only) #####\n", tab.to_string(index=False))

allpk = pd.concat([ref_pk.assign(spectrum=f"CaF2 {ref_name}"), s_pk.assign(spectrum=f"AgSiNW {best}")])
allpk.to_csv(f"{OUT}/peaks_with_checks.csv", index=False)
pd.concat([ref_rej.assign(spectrum=f"CaF2 {ref_name}"), s_rej.assign(spectrum=f"AgSiNW {best}")]).to_csv(
    f"{OUT}/rejected_narrow_maxima.csv", index=False)
shared = tab[tab.status == "in both"][["CaF2_cm1", "AgSiNW_cm1"]].values


# ------------------------------------------------------------------ figures (raw counts only)
plt.rcParams.update({"font.size": 13, "axes.linewidth": 1.1})

# compact slide labels (full assignments are in the CSV files)
SLIDE = {"C-C stretch protein backbone": "C-C backbone", "CH2/CH3 deformation": "CH$_2$/CH$_3$",
         "COO- symmetric stretch": "COO$^-$ sym.", "COO- asymmetric stretch": "COO$^-$ asym.",
         "Si optical phonon": "Si (substrate)", "Tyr ring breathing": "Tyr", "Phe ring breathing": "Phe",
         "Tyr C-H bend": "Tyr", "C=O stretch": "C=O ester", "C-C stretch": "C-C", "Tyr": "Tyr", "Phe": "Phe",
         "unassigned": "n.a."}


def slide_label(assignment):
    sh = short(assignment)
    return SLIDE.get(sh, sh)


def draw(ax, wn, y, pk, lo, hi, color, shared_pos, label_fs=10, headroom=0.9, title=None):
    m = (wn >= lo) & (wn <= hi)
    ax.plot(wn[m], y[m], color=color, lw=0.9)
    ymin, ymax = y[m].min(), y[m].max()
    rng = ymax - ymin
    ax.set_ylim(ymin - 0.03 * rng, ymax + headroom * rng)
    for p in shared_pos:
        if lo <= p <= hi:
            ax.axvline(p, color=C_SHARED, ls="--", lw=1.0, alpha=0.8)
    for _, p in pk[(pk.confirmed) & (pk.position_cm1 >= lo) & (pk.position_cm1 <= hi)].iterrows():
        top = p.raw_counts + 0.03 * rng
        ax.plot([p.position_cm1] * 2, [top, top + 0.06 * rng], color="k", lw=0.9)
        ax.text(p.position_cm1, top + 0.07 * rng, f"{p.position_cm1:.1f} {slide_label(p.assignment)}",
                rotation=90, fontsize=label_fs, ha="center", va="bottom", clip_on=True)
    if title:
        ax.text(0.01, 0.97, title, transform=ax.transAxes, ha="left", va="top", color=color, fontsize=14,
                fontweight="bold", bbox=dict(facecolor="white", edgecolor="none", alpha=0.85, pad=2))
    ax.set_xlim(lo, hi)
    ax.tick_params(direction="in", length=5)


# (1) 400-1800 comparison, 16:9 slide
fig, axs = plt.subplots(2, 1, figsize=(13.33, 7.5), sharex=True)
draw(axs[0], wn_r, y_r, ref_pk, LO, HI, C_CAF2, shared[:, 0] if len(shared) else [], label_fs=11, headroom=1.3,
     title=f"EV Acacia on CaF$_2$ ({ref_name})")
draw(axs[1], wn_s, y_s, s_pk, LO, HI, C_SERS, shared[:, 1] if len(shared) else [], label_fs=11, headroom=1.3,
     title=f"EV Acacia on AgSiNW ({best})")
for a in axs:
    a.set_ylabel("Intensity (counts)")
axs[1].set_xlabel("Raman shift (cm$^{-1}$)")
fig.text(0.99, 0.005, "Raw data, no smoothing/baseline correction. Green dashed: band present on both substrates "
         f"(±{MATCH_CM:.0f} cm$^{{-1}}$).", ha="right", fontsize=9, color="0.35")
fig.tight_layout(rect=(0, 0.02, 1, 1))
fig.savefig(f"{OUT}/EV_Acacia_CaF2_vs_AgSiNW_400-1800.png", dpi=300)
plt.close(fig)

# (2) zoomed regions, 16:9 slide
fig, axs = plt.subplots(2, len(ZOOMS), figsize=(13.33, 7.5))
for c, (lo, hi) in enumerate(ZOOMS):
    draw(axs[0, c], wn_r, y_r, ref_pk, lo, hi, C_CAF2, shared[:, 0] if len(shared) else [], label_fs=11, headroom=1.6)
    draw(axs[1, c], wn_s, y_s, s_pk, lo, hi, C_SERS, shared[:, 1] if len(shared) else [], label_fs=11, headroom=1.6)
    axs[1, c].set_xlabel("Raman shift (cm$^{-1}$)")
    axs[0, c].set_title(f"{lo}–{hi} cm$^{{-1}}$", fontsize=13)
axs[0, 0].set_ylabel(f"CaF$_2$ ({ref_name})\nIntensity (counts)", color=C_CAF2)
axs[1, 0].set_ylabel(f"AgSiNW ({best})\nIntensity (counts)", color=C_SERS)
fig.text(0.99, 0.005, "Raw data. Labelled peaks passed both independent re-checks.", ha="right", fontsize=9, color="0.35")
fig.tight_layout(rect=(0, 0.02, 1, 1))
fig.savefig(f"{OUT}/EV_Acacia_CaF2_vs_AgSiNW_zoom.png", dpi=300)
plt.close(fig)
print(f"\nOutputs written to {OUT}/")
