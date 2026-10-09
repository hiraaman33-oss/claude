"""
Bare substrate controls: laser-power series on bare AgSiNW and bare SiNW (no Ag), 532 nm, raw data.

Questions answered:
  1. Does the laser heat the substrate?  Si optical phonon (~520 cm-1) position and FWHM vs power
     (Lorentzian + linear background fitted to raw counts, 470-570 cm-1). Temperature rise estimated
     from the red-shift relative to the low-power (2.5-10 %) mean of the same series with
     d(omega)/dT = -0.022 cm-1/K (Balkanski et al., Phys Rev B 28, 1928 (1983)).
  2. Which bands does the bare substrate itself give?  raman_raw.checked_peaks (5 sigma + two re-checks)
     on every spectrum, 400-1800 cm-1.
  3. Which EV bands on AgSiNW can come from the substrate?  For every confirmed AgSiNW band of
     EV Acacia (S2) and EV nano Acacia (s4) the noise-calibrated score (raman_raw.calib_score) is
     evaluated at that position (raw maximum within ±6 cm-1) in each bare spectrum and compared with
     the 99th-percentile noise threshold of that spectrum (raman_raw.noise_threshold).

Usage:  python3 Bare_AgSiNW_PowerSeries.py <dir with bare *.l6s files, searched recursively>
"""
import glob
import os
import re
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

from raman_raw import calib_score, checked_peaks, noise_sigma, noise_threshold, read_single

ROOT = sys.argv[1]
OUT = "Bare_AgSiNW_PowerSeries"
os.makedirs(OUT, exist_ok=True)
LO, HI = 400, 1800
DWDT = -0.022  # cm-1 per K
EV_PEAKS = {"EV Acacia S2": ("EV_Acacia_PPT_Compare/peaks_with_checks.csv", "AgSiNW S2"),
            "EV nano Acacia s4": ("EV_nano_Acacia_PPT_Compare/peaks_with_checks.csv", "AgSiNW s4")}
EV_FILES = {"EV Acacia S2": "*S2-EV_ACACIA-AgSiNW*.l6s", "EV nano Acacia s4": "*s4-EV_nano_ACACIA*.txt"}


def meta(path):
    b = os.path.basename(path)
    m = re.match(r"(S\d+)-(AgSiNW|SiNW)_532nm_1800gr_BC(\d+)_(\w+?)_(\d+)s_(\d+)a_([\d-]+)%", b)
    sid, sub, bc, obj, t, acc, pw = m.groups()
    obj = "50XLF" if obj in ("50XLF", "5XLF") else obj  # "5XLF" in one SiNW file name: 50XLF series
    t = float(t) if not t.startswith("0") or t == "0" else float("0." + t[1:])  # 05 -> 0.5, 005 -> 0.05
    pwv = {"2-5": 2.5}.get(pw, None)
    if pwv is None:
        pwv = float(pw) if not pw.startswith("0") else float("0." + pw[1:])  # 01 -> 0.1, 001 -> 0.01
    return dict(id=sid, substrate=sub, objective=obj, BC=int(bc), time_s=t, acc=int(acc), power_pct=pwv,
                power_code=pw + "%", file=b)


def lor(x, a, x0, w, b0, b1):
    return a * (w / 2) ** 2 / ((x - x0) ** 2 + (w / 2) ** 2) + b0 + b1 * (x - 520)


files = sorted(f for f in glob.glob(os.path.join(ROOT, "**", "*.l6s"), recursive=True))
rows, spectra = [], {}
for f in files:
    md = meta(f)
    wn, y = read_single(f)
    key = f"{md['substrate']} {md['objective']} {md['id']} {md['power_code']} {md['time_s']:g}s×{md['acc']}"
    spectra[key] = (wn, y, md)
    sg = noise_sigma(wn, y)
    m = (wn >= 470) & (wn <= 570)
    x, yy = wn[m], y[m]
    i = int(np.argmax(yy))
    fit = dict(Si_center=np.nan, Si_FWHM=np.nan, Si_height=np.nan, Si_SNR=round(float((yy[i] - np.median(yy)) / sg), 1))
    if fit["Si_SNR"] >= 8:
        try:
            p, _ = curve_fit(lor, x, yy, p0=[yy[i] - np.median(yy), x[i], 8, np.median(yy), 0], maxfev=20000)
            if 505 < p[1] < 530 and 2 < abs(p[2]) < 60:
                fit.update(Si_center=round(p[1], 2), Si_FWHM=round(abs(p[2]), 1), Si_height=round(p[0], 1))
        except RuntimeError:
            pass
    pk, _, _ = checked_peaks(wn, y, "SERS", LO, HI)
    c = pk[pk.confirmed & (pk.position_cm1 > 540)]
    rows.append(dict(md, key=key, noise_sigma=round(sg, 1), background_p10=round(float(np.percentile(y[(wn > LO) & (wn < HI)], 10)), 1),
                     **fit, bands_540_1800=", ".join(f"{p:.0f} (SNR {s:.0f})" for p, s in zip(c.position_cm1, c.SNR))))
tab = pd.DataFrame(rows).sort_values(["substrate", "objective", "time_s", "acc", "power_pct"]).reset_index(drop=True)

# temperature rise from the Si red-shift (1 s × 1 series only, reference = mean of 2.5-10 %)
tab["dT_K"] = np.nan
for (sub, obj), g in tab[(tab.time_s == 1) & (tab.acc == 1)].groupby(["substrate", "objective"]):
    ref = g[(g.power_pct >= 2.5) & (g.power_pct <= 10)].Si_center.mean()
    tab.loc[g.index, "dT_K"] = ((g.Si_center - ref) / DWDT).round(0)
tab.drop(columns=["key"]).to_csv(f"{OUT}/bare_substrate_summary.csv", index=False)
pd.set_option("display.width", 250)
pd.set_option("display.max_colwidth", 90)
print(tab[["substrate", "objective", "power_code", "time_s", "acc", "Si_center", "Si_FWHM", "dT_K", "background_p10",
           "bands_540_1800"]].to_string(index=False))

# ------------------------------------------------------------------ EV bands vs bare AgSiNW
thr = {k: noise_threshold(v[0], v[1], noise_sigma(v[0], v[1])) for k, v in spectra.items()}
strict = {k: checked_peaks(v[0], v[1], "SERS", LO, HI)[0] for k, v in spectra.items()}
strict = {k: v[v.confirmed].position_cm1.values for k, v in strict.items()}
# The weak-band test (score above the 99th-percentile noise level, maximum within ±6 cm-1) gives false hits:
# 0/1480 random positions in 600-720 cm-1 and 79/1480 (5.3 %) in 1740-1755 cm-1 of these spectra. A
# conservative 5 % per spectrum is used, and a band counts as present on the bare substrate only if the number
# of hits is unlikely to be noise (binomial p < 0.01) or a strict peak (5 sigma + two re-checks) is found.
FALSE_RATE = 0.05
from scipy.stats import binom


def is_si(pos):
    return 505 <= pos <= 530 or 940 <= pos <= 990


ev_rows = []
for ev_name, (csv, spec) in EV_PEAKS.items():
    d = pd.read_csv(csv)
    d = d[(d.spectrum == spec) & d.confirmed]
    for _, r in d.iterrows():
        res = {"AgSiNW": dict(weak=[], strict=[], n=0), "SiNW": dict(weak=[], strict=[], n=0)}
        for k, (wn, y, md) in spectra.items():
            R = res[md["substrate"]]
            R["n"] += 1
            lab = f"{md['objective']} {md['power_code']} {md['time_s']:g}s×{md['acc']}"
            pos, s = calib_score(wn, y, r.position_cm1, noise_sigma(wn, y))
            if s > thr[k] and abs(pos - r.position_cm1) <= 6.0:
                R["weak"].append(f"{lab} ({pos:.0f})")
            sp = strict[k]
            if len(sp) and np.min(np.abs(sp - r.position_cm1)) <= 6.0:
                R["strict"].append(f"{lab} ({sp[np.argmin(np.abs(sp - r.position_cm1))]:.0f})")
        pv = {sub: float(binom.sf(len(R["weak"]) - 1, R["n"], FALSE_RATE)) if R["weak"] else 1.0 for sub, R in res.items()}
        ag, si = res["AgSiNW"], res["SiNW"]
        if is_si(r.position_cm1) or (si["strict"] and pv["SiNW"] < 0.01):
            verdict = "Si substrate"
        elif ag["strict"] or pv["AgSiNW"] < 0.01:
            verdict = "also on bare AgSiNW (not EV-specific)"
        elif pv["AgSiNW"] < 0.05:
            verdict = "possibly on bare AgSiNW (weak evidence)"
        else:
            verdict = "not on bare substrate (EV band)"
        ev_rows.append(dict(sample=ev_name, EV_band_cm1=r.position_cm1, assignment=r.assignment,
                            bare_AgSiNW_strict=f"{len(ag['strict'])}/{ag['n']}", bare_AgSiNW_weak=f"{len(ag['weak'])}/{ag['n']}",
                            p_noise_AgSiNW=round(pv["AgSiNW"], 4),
                            bare_SiNW_strict=f"{len(si['strict'])}/{si['n']}", bare_SiNW_weak=f"{len(si['weak'])}/{si['n']}",
                            p_noise_SiNW=round(pv["SiNW"], 4), verdict=verdict,
                            AgSiNW_strict_in="; ".join(ag["strict"]), AgSiNW_weak_in="; ".join(ag["weak"]),
                            SiNW_strict_in="; ".join(si["strict"]), SiNW_weak_in="; ".join(si["weak"])))
ev = pd.DataFrame(ev_rows)
ev.to_csv(f"{OUT}/EV_bands_vs_bare_substrate.csv", index=False)
print("\n##### EV bands on AgSiNW vs bare substrates #####")
print(ev[["sample", "EV_band_cm1", "assignment", "bare_AgSiNW_strict", "bare_AgSiNW_weak", "p_noise_AgSiNW",
          "bare_SiNW_strict", "bare_SiNW_weak", "verdict"]].to_string(index=False))

# ------------------------------------------------------------------ figures (raw counts)
plt.rcParams.update({"font.size": 12, "axes.linewidth": 1.0})
series = [("AgSiNW", "50XLF"), ("AgSiNW", "100X"), ("SiNW", "50XLF"), ("SiNW", "100X")]
pw_list = [1, 2.5, 5, 10, 25, 50, 100]
cmap = plt.get_cmap("Blues")
pcol = {p: cmap(0.3 + 0.7 * i / (len(pw_list) - 1)) for i, p in enumerate(pw_list)}

# (1) power series, 1 s × 1, raw counts, 2x2
fig, axs = plt.subplots(2, 2, figsize=(13.33, 7.5), sharex=True)
for ax, (sub, obj) in zip(axs.flat, series):
    g = tab[(tab.substrate == sub) & (tab.objective == obj) & (tab.time_s == 1) & (tab.acc == 1)]
    for _, r in g.sort_values("power_pct", ascending=False).iterrows():
        wn, y, _ = spectra[r.key]
        m = (wn >= LO) & (wn <= HI)
        ax.plot(wn[m], y[m], lw=0.8, color=pcol.get(r.power_pct, "0.5"), label=f"{r.power_code}")
    ax.set_title(f"bare {sub}, {obj}, 1 s × 1", fontsize=12, loc="left")
    for xv in (1384, 1584):
        ax.axvline(xv, color="0.6", ls=":", lw=0.8)
    ax.set_xlim(LO, HI)
for ax in axs[:, 0]:
    ax.set_ylabel("Intensity (counts)")
for ax in axs[1]:
    ax.set_xlabel("Raman shift (cm$^{-1}$)")
h, l = axs[0, 1].get_legend_handles_labels()
fig.legend(h[::-1], l[::-1], title="laser power", loc="upper right", bbox_to_anchor=(0.995, 0.93), fontsize=10)
fig.text(0.99, 0.005, "Raw data, no smoothing/baseline correction. Dotted: 1384 and 1584 cm$^{-1}$.",
         ha="right", fontsize=9, color="0.35")
fig.tight_layout(rect=(0, 0.02, 0.92, 1))
fig.savefig(f"{OUT}/1_power_series_raw.png", dpi=300)
plt.close(fig)

# (2) Si phonon vs power: centre, FWHM, background (three separate panels, one axis each)
mk = {("AgSiNW", "50XLF"): ("#1f4e79", "o", "-"), ("AgSiNW", "100X"): ("#1f4e79", "s", "--"),
      ("SiNW", "50XLF"): ("#b03a2e", "o", "-"), ("SiNW", "100X"): ("#b03a2e", "s", "--")}
fig, axs = plt.subplots(1, 3, figsize=(13.33, 4.6))
for (sub, obj), (c, mrk, ls) in mk.items():
    g = tab[(tab.substrate == sub) & (tab.objective == obj) & (tab.time_s == 1) & (tab.acc == 1)].sort_values("power_pct")
    lab = f"{sub} {obj}"
    gg = g.dropna(subset=["Si_center"])
    axs[0].plot(gg.power_pct, gg.Si_center, marker=mrk, color=c, ls=ls, lw=1.5, ms=7, label=lab)
    axs[1].plot(gg.power_pct, gg.Si_FWHM, marker=mrk, color=c, ls=ls, lw=1.5, ms=7, label=lab)
    axs[2].plot(g.power_pct, g.background_p10, marker=mrk, color=c, ls=ls, lw=1.5, ms=7, label=lab)
for ax, t in zip(axs, ["Si phonon centre (cm$^{-1}$)", "Si phonon FWHM (cm$^{-1}$)", "Background, 10th percentile (counts)"]):
    ax.set_xscale("log")
    ax.set_xlabel("Laser power (%)")
    ax.set_title(t, fontsize=12, loc="left")
    ax.grid(alpha=0.25)
    ax.set_xticks([1, 2.5, 5, 10, 25, 50, 100])
    ax.set_xticklabels(["1", "2.5", "5", "10", "25", "50", "100"])
axs[0].legend(fontsize=9)
fig.text(0.99, 0.005, "Lorentzian + linear background fitted to raw counts (470-570 cm$^{-1}$); 1 s × 1 accumulation series.",
         ha="right", fontsize=9, color="0.35")
fig.tight_layout(rect=(0, 0.03, 1, 1))
fig.savefig(f"{OUT}/2_Si_phonon_vs_power.png", dpi=300)
plt.close(fig)

# (3) EV spectra vs bare AgSiNW (same objective / power as the EV measurements: 50XLF, 10 %) + the bare
#     spectrum with the most bands at 10 % (100X)
upl = sys.argv[2] if len(sys.argv) > 2 else None
if upl:
    fig, axs = plt.subplots(4, 1, figsize=(13.33, 9.5), sharex=True)
    panels = []
    for ev_name, pat in EV_FILES.items():
        f = sorted(glob.glob(os.path.join(upl, pat)))[0]
        panels.append((ev_name + " (50XLF, 10 %, 1 s × 4)", read_single(f), ev_name))
    for k in ("AgSiNW 50XLF S4 10% 1s×1", "AgSiNW 100X S4 10% 1s×1"):
        wn, y, _ = spectra[k]
        panels.append(("bare " + k.replace("S4 ", ""), (wn, y), None))
    for ax, (title, (wn, y), ev_name) in zip(axs, panels):
        m = (wn >= LO) & (wn <= HI)
        ax.plot(wn[m], y[m], lw=0.8, color="#1f4e79" if ev_name else "0.35")
        ax.set_title(title, fontsize=11, loc="left")
        ax.set_ylabel("counts")
        if ev_name:
            top = y[m].max()
            for _, r in ev[ev["sample"] == ev_name].iterrows():
                col = {"not on bare substrate (EV band)": "#2ca02c", "Si substrate": "0.5",
                       "possibly on bare AgSiNW (weak evidence)": "#e6a700"}.get(r.verdict, "#d95f02")
                ax.axvline(r.EV_band_cm1, color=col, ls="--", lw=1.0)
                ax.text(r.EV_band_cm1, top, f"{r.EV_band_cm1:.0f}", rotation=90, va="top", ha="right", fontsize=9, color=col)
    axs[-1].set_xlabel("Raman shift (cm$^{-1}$)")
    from matplotlib.lines import Line2D
    fig.legend([Line2D([], [], color=c, ls="--") for c in ("#2ca02c", "#e6a700", "#d95f02", "0.5")],
               ["not on bare substrate (EV band)", "possibly on bare AgSiNW (weak)", "also on bare AgSiNW (no sample)",
                "Si substrate"], loc="upper right", fontsize=10, ncol=4)
    fig.text(0.99, 0.005, "Raw data. Bare-substrate check over all 37 bare spectra (all powers, both objectives): strict peak (5σ + 2 re-checks) "
             "or weak-band hits above the 5 % noise rate (binomial p < 0.01), ±6 cm$^{-1}$.", ha="right", fontsize=8.5, color="0.35")
    fig.tight_layout(rect=(0, 0.02, 1, 0.96))
    fig.savefig(f"{OUT}/3_EV_vs_bare_AgSiNW.png", dpi=300)
    plt.close(fig)
print(f"\nOutputs written to {OUT}/")
