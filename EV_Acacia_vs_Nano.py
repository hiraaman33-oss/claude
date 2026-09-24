"""
EV Acacia vs EV nano Acacia on the same substrate (AgSiNW or CaF2) – raw data, 400-1800 cm-1,
slide-ready.

Peaks: raman_raw.visible_peaks (noise-calibrated local test on raw counts; broad bands merged;
spike-like maxima kept only if reproduced by a replicate of the same sample). Replicates are all
spectra in the same folder (e.g. AgSiNW: EV Acacia S1, S2; EV nano Acacia s3, s4, s4a).
Bands are matched between the two samples within ±MATCH_CM.

Usage:  python3 EV_Acacia_vs_Nano.py <acacia_dir> <nano_dir> [acacia_name] [nano_name] [AgSiNW|CaF2]
        (defaults S2, s4, AgSiNW - the AgSiNW spectra chosen against CaF2 in EV_Acacia_PPT_Compare.py)
"""
import glob
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from raman_raw import clean, draw, read_single, tag, visible_peaks

A_DIR, N_DIR = sys.argv[1:3]
A_NAME = sys.argv[3] if len(sys.argv) > 3 else "S2"
N_NAME = sys.argv[4] if len(sys.argv) > 4 else "s4"
SUBSTRATE = sys.argv[5] if len(sys.argv) > 5 else "AgSiNW"
SUB = "SERS" if SUBSTRATE == "AgSiNW" else "CaF2"  # assignment context in raman_raw
SUB_TEX = "CaF$_2$" if SUBSTRATE == "CaF2" else SUBSTRATE
OUT = f"EV_Acacia_vs_Nano_{SUBSTRATE}"
os.makedirs(OUT, exist_ok=True)
LO, HI = 400, 1800
MATCH_CM = 8.0
ZOOMS = [(780, 1150), (1150, 1500), (1500, 1800)]
C_A, C_N = "#1f4e79", "#1a9641"


def load(d):
    return {tag(f): dict(file=clean(f), wn_y=read_single(f)) for f in sorted(glob.glob(os.path.join(d, "*")))}


acacia, nano = load(A_DIR), load(N_DIR)
a_pk, a_thr = visible_peaks(acacia, A_NAME, LO, HI, substrate=SUB)
n_pk, n_thr = visible_peaks(nano, N_NAME, LO, HI, substrate=SUB)
wn_a, y_a = acacia[A_NAME]["wn_y"]
wn_n, y_n = nano[N_NAME]["wn_y"]

pd.set_option("display.width", 250)
pd.set_option("display.max_colwidth", 60)
cols = ["position_cm1", "raw_counts", "calib_score", "noise_threshold", "band_extent_cm1", "reproduced_in",
        "assignment", "flag", "confirmed"]
for lab, pk, f in [(f"EV Acacia on {SUBSTRATE} ({A_NAME})", a_pk, acacia[A_NAME]["file"]),
                   (f"EV nano Acacia on {SUBSTRATE} ({N_NAME})", n_pk, nano[N_NAME]["file"])]:
    print(f"\n=== {lab}  {f} ===\n{pk[cols].to_string(index=False)}")

# ------------------------------------------------------------------ band table
ac, nc = a_pk[a_pk.confirmed].reset_index(drop=True), n_pk[n_pk.confirmed].reset_index(drop=True)
rows, used = [], set()
for _, r in ac.iterrows():
    d = np.abs(nc.position_cm1.values - r.position_cm1) if not nc.empty else np.array([99.0])
    j = int(d.argmin())
    if d[j] <= MATCH_CM and j not in used:
        used.add(j)
        s = nc.iloc[j]
        rows.append(dict(Acacia_cm1=r.position_cm1, nano_cm1=s.position_cm1,
                         shift_cm1=round(s.position_cm1 - r.position_cm1, 1), score_Acacia=r.calib_score,
                         score_nano=s.calib_score, status="in both", band=r.assignment))
    else:
        near = f" (nearest nano {nc.position_cm1.values[j]:.1f}, Δ{d[j]:.1f})" if (not nc.empty and d[j] <= 15) else ""
        rows.append(dict(Acacia_cm1=r.position_cm1, nano_cm1=np.nan, shift_cm1=np.nan, score_Acacia=r.calib_score,
                         score_nano=np.nan, status="only EV Acacia" + near, band=r.assignment))
for j, s in nc.iterrows():
    if j not in used:
        rows.append(dict(Acacia_cm1=np.nan, nano_cm1=s.position_cm1, shift_cm1=np.nan, score_Acacia=np.nan,
                         score_nano=s.calib_score, status="only EV nano Acacia", band=s.assignment))
tab = pd.DataFrame(rows)
tab = tab.assign(_k=tab.Acacia_cm1.fillna(tab.nano_cm1)).sort_values("_k").drop(columns="_k")
tab.to_csv(f"{OUT}/band_table_400-1800.csv", index=False)
pd.concat([a_pk.assign(spectrum=f"EV Acacia {SUBSTRATE} {A_NAME}"),
           n_pk.assign(spectrum=f"EV nano Acacia {SUBSTRATE} {N_NAME}")]).to_csv(f"{OUT}/peaks_with_checks.csv", index=False)
print("\n##### Band table #####\n", tab.to_string(index=False))
shared = tab[tab.status == "in both"][["Acacia_cm1", "nano_cm1"]].values

# ------------------------------------------------------------------ figures (raw counts only)
plt.rcParams.update({"font.size": 13, "axes.linewidth": 1.1})
note = ("Raw data, no smoothing/baseline correction. Green dashed: band in both samples "
        f"(±{MATCH_CM:.0f} cm$^{{-1}}$). Labelled peaks exceed the noise-calibrated threshold.")

fig, axs = plt.subplots(2, 1, figsize=(13.33, 7.5), sharex=True)
draw(axs[0], wn_a, y_a, a_pk, LO, HI, C_A, shared[:, 0] if len(shared) else [], label_fs=11, headroom=1.3,
     title=f"EV Acacia on {SUB_TEX} ({A_NAME})")
draw(axs[1], wn_n, y_n, n_pk, LO, HI, C_N, shared[:, 1] if len(shared) else [], label_fs=11, headroom=1.3,
     title=f"EV nano Acacia on {SUB_TEX} ({N_NAME})")
for a in axs:
    a.set_ylabel("Intensity (counts)")
axs[1].set_xlabel("Raman shift (cm$^{-1}$)")
fig.text(0.99, 0.005, note, ha="right", fontsize=9, color="0.35")
fig.tight_layout(rect=(0, 0.02, 1, 1))
fig.savefig(f"{OUT}/EV_Acacia_vs_nano_{SUBSTRATE}_400-1800.png", dpi=300)
plt.close(fig)

fig, axs = plt.subplots(2, len(ZOOMS), figsize=(13.33, 7.5))
for c, (lo, hi) in enumerate(ZOOMS):
    draw(axs[0, c], wn_a, y_a, a_pk, lo, hi, C_A, shared[:, 0] if len(shared) else [], label_fs=11, headroom=1.6)
    draw(axs[1, c], wn_n, y_n, n_pk, lo, hi, C_N, shared[:, 1] if len(shared) else [], label_fs=11, headroom=1.6)
    axs[1, c].set_xlabel("Raman shift (cm$^{-1}$)")
    axs[0, c].set_title(f"{lo}–{hi} cm$^{{-1}}$", fontsize=13)
axs[0, 0].set_ylabel(f"EV Acacia ({A_NAME})\nIntensity (counts)", color=C_A)
axs[1, 0].set_ylabel(f"EV nano Acacia ({N_NAME})\nIntensity (counts)", color=C_N)
fig.text(0.99, 0.005, "Raw data. Labelled peaks exceed the noise-calibrated threshold.", ha="right", fontsize=9,
         color="0.35")
fig.tight_layout(rect=(0, 0.02, 1, 1))
fig.savefig(f"{OUT}/EV_Acacia_vs_nano_{SUBSTRATE}_zoom.png", dpi=300)
plt.close(fig)
print(f"\nOutputs written to {OUT}/")
