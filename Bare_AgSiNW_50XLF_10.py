"""EV spectra on AgSiNW vs bare AgSiNW at the SAME conditions (50XLF, 10 %, 532 nm, 1800 gr, BC200). Raw data.
Usage: python3 Bare_AgSiNW_50XLF_10.py <bare S4-AgSiNW_..._50XLF_1s_1a_10%.l6s> <EV Acacia S2 file> <EV nano s4 file>"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from raman_raw import calib_score, checked_peaks, noise_sigma, noise_threshold, read_single

BARE, EVA, EVN = sys.argv[1:4]
OUT = "Bare_AgSiNW_50XLF_10"
os.makedirs(OUT, exist_ok=True)
LO, HI = 400, 1800
wb, yb = read_single(BARE)
sb = noise_sigma(wb, yb)
tb = noise_threshold(wb, yb, sb)
bare_pk = checked_peaks(wb, yb, "SERS", LO, HI)[0]
bare_pk = bare_pk[bare_pk.confirmed]
print(f"bare AgSiNW 50XLF 10 %: noise sigma {sb:.1f}, confirmed peaks:", list(bare_pk.position_cm1))

rows = []
for name, csv, spec in [("EV Acacia S2", "EV_Acacia_PPT_Compare/peaks_with_checks.csv", "AgSiNW S2"),
                        ("EV nano Acacia s4", "EV_nano_Acacia_PPT_Compare/peaks_with_checks.csv", "AgSiNW s4")]:
    d = pd.read_csv(csv)
    for _, r in d[(d.spectrum == spec) & d.confirmed].iterrows():
        pos, sc = calib_score(wb, yb, r.position_cm1, sb)
        strict = bool(len(bare_pk) and np.min(np.abs(bare_pk.position_cm1.values - r.position_cm1)) <= 6)
        weak = bool(sc > tb and abs(pos - r.position_cm1) <= 12)  # broad weak humps on bare: ±12 cm-1
        rows.append(dict(sample=name, EV_band_cm1=r.position_cm1, assignment=r.assignment,
                         bare_max_cm1=pos, bare_score=round(sc, 1), bare_noise_threshold=round(tb, 1),
                         on_bare_AgSiNW="yes (clear peak)" if strict else "weak" if weak else "no"))
tab = pd.DataFrame(rows)
tab.to_csv(f"{OUT}/EV_bands_vs_bare_50XLF_10.csv", index=False)
pd.set_option("display.width", 200)
print(tab[["sample", "EV_band_cm1", "assignment", "bare_score", "bare_noise_threshold", "on_bare_AgSiNW"]].to_string(index=False))

plt.rcParams.update({"font.size": 12})
fig, axs = plt.subplots(3, 1, figsize=(13.33, 7.5), sharex=True)
col = {"yes (clear peak)": "#d95f02", "weak": "#e6a700", "no": "#2ca02c"}
for ax, (title, f, name, c) in zip(axs, [("EV Acacia on AgSiNW (S2)", EVA, "EV Acacia S2", "#1f4e79"),
                                         ("EV nano Acacia on AgSiNW (s4)", EVN, "EV nano Acacia s4", "#1f4e79"),
                                         ("bare AgSiNW (no sample)", BARE, None, "0.3")]):
    wn, y = read_single(f)
    m = (wn >= LO) & (wn <= HI)
    ax.plot(wn[m], y[m], lw=0.8, color=c)
    ax.set_title(title + " – 50X LF, 10 %", loc="left", fontsize=12)
    ax.set_ylabel("counts")
    if name:
        for _, r in tab[tab["sample"] == name].iterrows():
            k = col[r.on_bare_AgSiNW]
            ax.axvline(r.EV_band_cm1, color=k, ls="--", lw=0.9)
            ax.text(r.EV_band_cm1, y[m].max(), f"{r.EV_band_cm1:.0f}", rotation=90, va="top", ha="right", fontsize=9, color=k)
axs[-1].set_xlabel("Raman shift (cm$^{-1}$)")
axs[-1].set_xlim(LO, HI)
from matplotlib.lines import Line2D
fig.legend([Line2D([], [], color=v, ls="--") for v in col.values()],
           ["also clear on bare AgSiNW", "weak on bare AgSiNW", "not on bare AgSiNW"], loc="upper right", ncol=3, fontsize=10)
fig.text(0.99, 0.005, "Raw data. EV: 1 s × 4 accumulations; bare: 1 s × 1.", ha="right", fontsize=9, color="0.35")
fig.tight_layout(rect=(0, 0.02, 1, 0.95))
fig.savefig(f"{OUT}/EV_vs_bare_AgSiNW_50XLF_10.png", dpi=300)
