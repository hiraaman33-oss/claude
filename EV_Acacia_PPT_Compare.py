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
4. AgSiNW spectrum, noise-calibrated search (catches weak/sharp bands visible by eye):
   raman_raw.calib_score is evaluated every 1 cm-1; each local maximum (one per ±8 cm-1) must
   exceed the 99th percentile that pure noise reaches with the same test in a band-free region
   (600-720 cm-1) of that spectrum. Each contiguous above-threshold stretch is one band, placed
   at its strongest maximum that another AgSiNW spectrum reproduces within ±3 cm-1 (else its
   strongest maximum); a further maximum >= 8 cm-1 away counts as a separate band only if it is
   reproduced, or passed checks 1+2 in this spectrum with a FWHM smaller than its distance. A spike-like maximum (1-2 points wide) is only kept if
   reproduced (otherwise: possible cosmic ray).

Usage:  python3 EV_Acacia_PPT_Compare.py <caf2_dir> <agsinw_dir> ["sample name"]
        (sample name defaults to "EV Acacia"; output goes to <sample_name>_PPT_Compare/)
"""
import glob
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from raman_raw import checked_peaks as _checked_peaks
from raman_raw import clean, draw, read_single, tag, visible_peaks

CAF2_DIR, SERS_DIR = sys.argv[1:3]
SAMPLE = sys.argv[3] if len(sys.argv) > 3 else "EV Acacia"
TAG = SAMPLE.replace(" ", "_")
OUT = f"{TAG}_PPT_Compare"
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
    return _checked_peaks(wn, y, substrate, LO, HI)


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


s_pk, s_thr = visible_peaks(sers, best, LO, HI)
print(f"\nnoise-calibrated threshold for {best}: {s_thr:.1f}")

pd.set_option("display.width", 250)
pd.set_option("display.max_colwidth", 60)
for lab, pk, sg, f in [(f"{SAMPLE} on CaF2 ({ref_name})", ref_pk, ref_sigma, caf2[ref_name]["file"]),
                       (f"{SAMPLE} on AgSiNW ({best})", s_pk, s_sigma, sers[best]["file"])]:
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


# (1) 400-1800 comparison, 16:9 slide
fig, axs = plt.subplots(2, 1, figsize=(13.33, 7.5), sharex=True)
draw(axs[0], wn_r, y_r, ref_pk, LO, HI, C_CAF2, shared[:, 0] if len(shared) else [], label_fs=11, headroom=1.3,
     title=f"{SAMPLE} on CaF$_2$ ({ref_name})")
draw(axs[1], wn_s, y_s, s_pk, LO, HI, C_SERS, shared[:, 1] if len(shared) else [], label_fs=11, headroom=1.3,
     title=f"{SAMPLE} on AgSiNW ({best})")
for a in axs:
    a.set_ylabel("Intensity (counts)")
axs[1].set_xlabel("Raman shift (cm$^{-1}$)")
fig.text(0.99, 0.005, "Raw data, no smoothing/baseline correction. Green dashed: band present on both substrates "
         f"(±{MATCH_CM:.0f} cm$^{{-1}}$).", ha="right", fontsize=9, color="0.35")
fig.tight_layout(rect=(0, 0.02, 1, 1))
fig.savefig(f"{OUT}/{TAG}_CaF2_vs_AgSiNW_400-1800.png", dpi=300)
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
fig.savefig(f"{OUT}/{TAG}_CaF2_vs_AgSiNW_zoom.png", dpi=300)
plt.close(fig)
print(f"\nOutputs written to {OUT}/")
