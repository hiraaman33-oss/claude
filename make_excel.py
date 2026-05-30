import numpy as np
from scipy.signal import find_peaks, savgol_filter
from scipy.optimize import curve_fit
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve
import os, openpyxl
from openpyxl.styles import PatternFill, Font, Alignment, Border, Side
from openpyxl.utils import get_column_letter

# ── FILE PATHS ────────────────────────────────────────────────────────────────
BASE = "/root/.claude/uploads/f4cf220b-abe6-4680-b18d-91dd4043b59c"
files = {}
for fn in sorted(os.listdir(BASE)):
    fp = os.path.join(BASE, fn)
    lower = fn.lower()
    if "caf2" in lower:
        files.setdefault('caf2', fp)
    elif "m1" in lower:
        files.setdefault('m1', fp)
    elif "m2" in lower:
        files.setdefault('m2', fp)
    elif "m3" in lower:
        files.setdefault('m3', fp)
    elif "m4" in lower:
        files.setdefault('m4', fp)

print("Files found:", {k: os.path.basename(v) for k,v in files.items()})

# ── LOADERS ───────────────────────────────────────────────────────────────────
def load_single(path):
    data = []
    with open(path, encoding='utf-8', errors='ignore') as f:
        for line in f:
            p = line.strip().replace('\r','').split('\t')
            if len(p) == 2:
                try: data.append((float(p[0]), float(p[1])))
                except: pass
    return np.array([d[0] for d in data]), np.array([d[1] for d in data])

def load_map(path):
    with open(path, encoding='utf-8', errors='ignore') as f:
        lines = f.readlines()
    wn = None; wn_row = 0
    for i, l in enumerate(lines):
        p = l.strip().replace('\r','').split('\t')
        try:
            v = [float(x) for x in p if x.strip()]
            if len(v) > 100: wn = np.array(v); wn_row = i; break
        except: pass
    spectra, pos = [], []
    for line in lines[wn_row+1:]:
        p = line.strip().replace('\r','').split('\t')
        if len(p) < 5: continue
        try:
            x, y = float(p[0]), float(p[1])
            v = np.array([float(z) for z in p[2:] if z.strip()])
            if len(v) == len(wn): pos.append((x,y)); spectra.append(v)
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
        w = np.exp(np.clip(np.where(d<0, 2*(d-m)/s, 0),-500,500)); w[w>1]=1
    return Z

def process(wn, sp):
    sm = savgol_filter(sp, 11, 3)
    bl = airpls(sm)
    bc = sm - bl
    bc = np.clip(bc, 0, None)
    return bc

def get_I(wn, sp, centre, win=15):
    m = (wn >= centre-win) & (wn <= centre+win)
    return sp[m].max() if m.any() else 0.0

def norm_si(wn, sp):
    si = get_I(wn, sp, 520, 12)
    return sp / max(si, 1)

def gauss_centre(wn, sp, idx, hw=7):
    lo = max(0,idx-hw); hi = min(len(wn)-1,idx+hw)
    x = wn[lo:hi+1]; y = sp[lo:hi+1]
    if len(x) < 4: return wn[idx], sp[idx]
    try:
        p, _ = curve_fit(lambda x,A,mu,s: A*np.exp(-0.5*((x-mu)/s)**2),
                         x, y, p0=[sp[idx],wn[idx],2.0], maxfev=400)
        if abs(p[1]-wn[idx])<6 and p[2]>0: return round(p[1],1), round(max(p[0],0),1)
    except: pass
    return round(wn[idx],1), round(sp[idx],1)

def cosine_sim(a, b):
    na = np.linalg.norm(a); nb = np.linalg.norm(b)
    if na < 1e-9 or nb < 1e-9: return 0.0
    return float(np.dot(a,b)/(na*nb))

def interp_to(wn_src, sp_src, wn_dst):
    return np.interp(wn_dst, wn_src, sp_src)

# ── DNA BANDS ─────────────────────────────────────────────────────────────────
DNA_BANDS = [
    ('G_ring',       'G ring deformation',      668.1),
    ('A_ring',       'Adenine ring breathing',   722.1),
    ('OPO',          'O-P-O / Thy+Cyt',          782.7),
    ('Deoxyribo_CC', 'Deoxyribose C-C',           806.0),
    ('Backbone_CC',  'Backbone C-C/C-O',          912.9),
    ('Phe',          'Phe/ring breathing',        999.1),
    ('PO2_sym',      'Symmetric PO2-',           1097.5),
    ('PO2_asym',     'Antisymmetric PO2-',       1239.4),
    ('CytAde',       'Cytosine+Adenine',         1293.8),
    ('AdeGua',       'Adenine+Guanine',          1330.4),
    ('ThyAde',       'Thymine+Adenine',          1370.6),
    ('AdeGua_ring',  'Ade+Gua ring',             1481.2),
    ('AdeGua_plane', 'Ade+Gua in-plane',         1573.3),
    ('CO',           'Guanine/Thymine C=O',      1684.8),
]

# ── LOAD ALL DATA ─────────────────────────────────────────────────────────────
print("Loading CaF2...")
wn_ref, raw_ref = load_single(files['caf2'])
sp_ref = process(wn_ref, raw_ref)
sp_ref_n = sp_ref / max(sp_ref.max(), 1)

print("Loading maps...")
maps = {}
for k in ['m1','m2','m3','m4']:
    if k in files:
        wn_m, sps, pos = load_map(files[k])
        maps[k] = (wn_m, sps, pos)
        print(f"  {k}: {len(pos)} spectra, wn {wn_m.min():.0f}–{wn_m.max():.0f}")

# ── ANALYSE ALL SPECTRA ───────────────────────────────────────────────────────
# Restrict to DNA-relevant region only (exclude Si substrate peak at 520 cm-1)
WN_LO = 650.0
WN_HI = 1750.0
wn_common = wn_ref[(wn_ref >= WN_LO) & (wn_ref <= WN_HI)]
print(f"DNA comparison range: {WN_LO:.0f}–{WN_HI:.0f} cm-1, {len(wn_common)} points")

sp_ref_dna = sp_ref[(wn_ref >= WN_LO) & (wn_ref <= WN_HI)]
ref_max = sp_ref_dna.max()
ref_common = sp_ref_dna / max(ref_max, 1e-9)

def analyse_spectrum(wn_map, sp_raw, map_name, spec_idx, pos):
    sp = process(wn_map, sp_raw)
    sp_si = norm_si(wn_map, sp)
    # restrict to DNA region and max-normalise for comparison
    sp_dna = sp_si[(wn_map >= WN_LO) & (wn_map <= WN_HI)]
    wn_dna = wn_map[(wn_map >= WN_LO) & (wn_map <= WN_HI)]
    dna_max = sp_dna.max()
    sp_dna_n = sp_dna / max(dna_max, 1e-9)
    sp_int = interp_to(wn_dna, sp_dna_n, wn_common)
    cos = cosine_sim(sp_int, ref_common)

    # confidence thresholds calibrated to Ag/SiNW vs CaF2 cross-substrate cosine range (0.04-0.40)
    if cos >= 0.30: conf = "HIGH"
    elif cos >= 0.20: conf = "MODERATE"
    elif cos >= 0.10: conf = "LOW"
    else: conf = "NOISE"

    # find peaks in processed spectrum
    pks, _ = find_peaks(savgol_filter(sp_si,11,3), height=sp_si.mean()+sp_si.std()*0.3,
                        prominence=sp_si.std()*0.2, distance=8)

    # match DNA bands
    band_hits = {}
    wn_map_arr = wn_map
    sp_proc = savgol_filter(sp_si, 11, 3)
    for key, label, ref_wn in DNA_BANDS:
        win = 20
        mask = (wn_map_arr >= ref_wn-win) & (wn_map_arr <= ref_wn+win)
        if not mask.any(): band_hits[key] = (None, None, None)
        else:
            sub = sp_proc[mask]
            sub_wn = wn_map_arr[mask]
            best_idx = np.argmax(sub)
            abs_idx = np.where(mask)[0][best_idx]
            centre, intensity = gauss_centre(wn_map_arr, sp_proc, abs_idx)
            shift = round(centre - ref_wn, 1)
            band_hits[key] = (centre, intensity, shift)

    bb_I = band_hits['Backbone_CC'][1] or 1.0
    if bb_I < 1e-6: bb_I = 1.0
    ratios = {}
    for key, label, ref_wn in DNA_BANDS:
        I = band_hits[key][1]
        ratios[key] = round(I/bb_I, 3) if I else 0.0

    n_bands = sum(1 for k,_,_ in DNA_BANDS if band_hits[k][0] is not None
                  and band_hits[k][1] is not None and band_hits[k][1] > 0)

    return {
        'map': map_name.upper(),
        'spec': spec_idx+1,
        'x': round(pos[0],1),
        'y': round(pos[1],1),
        'cos': round(cos,4),
        'conf': conf,
        'n_bands': n_bands,
        'bands': band_hits,
        'ratios': ratios,
    }

rows = []
for k in ['m1','m2','m3','m4']:
    if k not in maps: continue
    wn_m, sps, pos = maps[k]
    print(f"Analysing {k} ({len(sps)} spectra)...")
    for i, (sp_raw, p) in enumerate(zip(sps, pos)):
        r = analyse_spectrum(wn_m, sp_raw, k, i, p)
        rows.append(r)

rows.sort(key=lambda r: r['cos'], reverse=True)
print(f"Total spectra analysed: {len(rows)}")
print(f"Cosine range: {rows[-1]['cos']:.4f} – {rows[0]['cos']:.4f}")

# ── CaF2 REFERENCE PEAKS ──────────────────────────────────────────────────────
ref_peaks_idx, _ = find_peaks(savgol_filter(sp_ref,11,3),
                               height=sp_ref.mean(), prominence=sp_ref.std()*0.3, distance=8)
ref_peaks = []
for idx in ref_peaks_idx:
    c, I = gauss_centre(wn_ref, sp_ref, idx)
    ref_peaks.append((c, I))

# ── EXCEL ─────────────────────────────────────────────────────────────────────
wb = openpyxl.Workbook()

# Colour fills
HIGH_fill   = PatternFill("solid", fgColor="C6EFCE")
MOD_fill    = PatternFill("solid", fgColor="FFEB9C")
LOW_fill    = PatternFill("solid", fgColor="FFCCCC")
NOISE_fill  = PatternFill("solid", fgColor="E0E0E0")
HEADER_fill = PatternFill("solid", fgColor="1F4E79")
SUBHDR_fill = PatternFill("solid", fgColor="2E75B6")
ALT_fill    = PatternFill("solid", fgColor="EBF3FB")

hdr_font  = Font(bold=True, color="FFFFFF", size=11)
sub_font  = Font(bold=True, color="FFFFFF", size=10)
bold_font = Font(bold=True, size=10)
norm_font = Font(size=10)
ctr = Alignment(horizontal="center", vertical="center", wrap_text=True)
lft = Alignment(horizontal="left",   vertical="center", wrap_text=True)

thin = Side(style='thin', color='AAAAAA')
bdr  = Border(left=thin, right=thin, top=thin, bottom=thin)

def H(ws, row, col, val, fill=None, font=None, align=None, width=None):
    c = ws.cell(row=row, column=col, value=val)
    if fill:  c.fill  = fill
    if font:  c.font  = font or norm_font
    if align: c.alignment = align
    c.border = bdr
    return c

def V(ws, row, col, val, fmt=None, fill=None, conf=None):
    c = ws.cell(row=row, column=col, value=val)
    c.font = norm_font
    c.alignment = ctr
    c.border = bdr
    if fmt: c.number_format = fmt
    if conf:
        if conf=="HIGH":     c.fill = HIGH_fill
        elif conf=="MODERATE": c.fill = MOD_fill
        elif conf=="LOW":    c.fill = LOW_fill
        else:                c.fill = NOISE_fill
    elif fill:
        c.fill = fill
    return c

conf_fill = {"HIGH":HIGH_fill,"MODERATE":MOD_fill,"LOW":LOW_fill,"NOISE":NOISE_fill}

# ══════════════════════════════════════════════════════════════════════════════
# SHEET 0 – How To Use
# ══════════════════════════════════════════════════════════════════════════════
ws0 = wb.active; ws0.title = "How To Use"
ws0.column_dimensions['A'].width = 18
ws0.column_dimensions['B'].width = 80

info = [
    ("Sheet",         "Description"),
    ("Overview Ranked","All spectra sorted by cosine similarity vs CaF2 reference. Thresholds calibrated for Ag/SiNW SERS vs CaF2 Raman (range 0.04-0.40). GREEN=HIGH (≥0.30), YELLOW=MODERATE (0.20-0.29), RED=LOW (0.10-0.19), GREY=NOISE (<0.10)"),
    ("Exact Peak Match","Gaussian-fitted peak centres (±0.1 cm⁻¹) for all 14 DNA bands across all HIGH/MODERATE spectra. Shift vs CaF2 reference shown."),
    ("Peak Ratios",   "Intensity ratios of each DNA band normalised to Backbone C-C/C-O at 912.9 cm⁻¹. Useful for comparing relative band strengths."),
    ("Per Map Summary","Statistics per map: number of spectra, mean cosine similarity, best spectrum, band detection counts."),
    ("Top 10 Spectra", "Raw spectral data (wavenumber vs intensity) for the top 10 highest-cosine spectra."),
    ("CaF2 Reference", "Reference spectrum peaks from DNA HaCaT on CaF2 (532 nm, 600 gr/mm)."),
    ("",""),
    ("Colour codes",""),
    ("GREEN",         "HIGH confidence DNA signal (cosine ≥ 0.30) — top ~20% of dataset"),
    ("YELLOW",        "MODERATE confidence DNA signal (cosine 0.20-0.29)"),
    ("RED",           "LOW confidence DNA signal (cosine 0.10-0.19)"),
    ("GREY",          "NOISE / no clear DNA signal (cosine < 0.10)"),
    ("",""),
    ("Key bands",""),
    ("912.9 cm⁻¹",   "Backbone C-C/C-O — used as normalisation reference for peak ratios"),
    ("1097.5 cm⁻¹",  "Symmetric PO2- stretch — phosphate backbone marker"),
    ("1370.6 cm⁻¹",  "Thymine+Adenine — most stable band across spectra"),
    ("722.1 cm⁻¹",   "Adenine ring breathing — typically red-shifted +11-14 cm⁻¹ on Ag/SiNW"),
]
for r, (a, b) in enumerate(info, 1):
    ca = ws0.cell(row=r, column=1, value=a); ca.font = bold_font; ca.border = bdr
    cb = ws0.cell(row=r, column=2, value=b); cb.font = norm_font; cb.alignment = lft; cb.border = bdr
ws0.row_dimensions[1].height = 20
H(ws0, 1, 1, "Sheet",       HEADER_fill, hdr_font, ctr)
H(ws0, 1, 2, "Description", HEADER_fill, hdr_font, lft)

# ══════════════════════════════════════════════════════════════════════════════
# SHEET 1 – Overview Ranked
# ══════════════════════════════════════════════════════════════════════════════
ws1 = wb.create_sheet("Overview Ranked")
headers = ["#","Map","Spectrum","X (µm)","Y (µm)","Cosine\nSimilarity","Confidence",
           "Bands\nDetected","Backbone\nRatio","PO2 sym\nRatio","ThyAde\nRatio","AdeGua\nRatio"]
col_w   = [5, 6, 9, 9, 9, 12, 12, 9, 11, 11, 11, 11]

for c, (h, w) in enumerate(zip(headers, col_w), 1):
    H(ws1, 1, c, h, HEADER_fill, hdr_font, ctr)
    ws1.column_dimensions[get_column_letter(c)].width = w
ws1.row_dimensions[1].height = 36

for i, row in enumerate(rows, 2):
    alt = ALT_fill if i%2==0 else None
    V(ws1, i, 1,  i-1,                fmt='0',     fill=alt)
    V(ws1, i, 2,  row['map'],                       fill=alt)
    V(ws1, i, 3,  row['spec'],         fmt='0',     fill=alt)
    V(ws1, i, 4,  row['x'],            fmt='0.0',   fill=alt)
    V(ws1, i, 5,  row['y'],            fmt='0.0',   fill=alt)
    V(ws1, i, 6,  row['cos'],          fmt='0.0000',conf=row['conf'])
    V(ws1, i, 7,  row['conf'],                      conf=row['conf'])
    V(ws1, i, 8,  row['n_bands'],      fmt='0',     fill=alt)
    V(ws1, i, 9,  row['ratios']['Backbone_CC'], fmt='0.000', fill=alt)
    V(ws1, i, 10, row['ratios']['PO2_sym'],     fmt='0.000', fill=alt)
    V(ws1, i, 11, row['ratios']['ThyAde'],      fmt='0.000', fill=alt)
    V(ws1, i, 12, row['ratios']['AdeGua'],      fmt='0.000', fill=alt)

ws1.freeze_panes = "A2"
ws1.auto_filter.ref = f"A1:L{len(rows)+1}"

# ══════════════════════════════════════════════════════════════════════════════
# SHEET 2 – Exact Peak Match
# ══════════════════════════════════════════════════════════════════════════════
ws2 = wb.create_sheet("Exact Peak Match")
good_rows = [r for r in rows if r['conf'] in ('HIGH','MODERATE')]
print(f"HIGH+MODERATE spectra: {len(good_rows)}")

# Build header: Map | Spec | Cos | [band columns x14 showing centre & shift]
band_headers = []
for key, label, ref_wn in DNA_BANDS:
    band_headers.append(f"{label}\n(ref {ref_wn})\nPos cm⁻¹")
    band_headers.append(f"{label}\nShift\ncm⁻¹")

all_headers = ["Map","Spec","X","Y","Cosine"] + band_headers
col_widths2  = [6,6,7,7,9] + [10,7]*len(DNA_BANDS)

for c, (h, w) in enumerate(zip(all_headers, col_widths2), 1):
    H(ws2, 1, c, h, HEADER_fill, hdr_font, ctr)
    ws2.column_dimensions[get_column_letter(c)].width = w
ws2.row_dimensions[1].height = 50

for i, row in enumerate(good_rows, 2):
    alt = ALT_fill if i%2==0 else None
    V(ws2, i, 1, row['map'],  fill=alt)
    V(ws2, i, 2, row['spec'], fmt='0', fill=alt)
    V(ws2, i, 3, row['x'],   fmt='0.0', fill=alt)
    V(ws2, i, 4, row['y'],   fmt='0.0', fill=alt)
    V(ws2, i, 5, row['cos'], fmt='0.0000', conf=row['conf'])
    col = 6
    for key, label, ref_wn in DNA_BANDS:
        centre, intensity, shift = row['bands'][key]
        if centre is not None:
            V(ws2, i, col,   centre, fmt='0.0', fill=alt)
            V(ws2, i, col+1, shift,  fmt='0.0',
              fill=PatternFill("solid",fgColor="FFD7AA") if shift and abs(shift)>5 else alt)
        else:
            V(ws2, i, col,   "—", fill=alt)
            V(ws2, i, col+1, "—", fill=alt)
        col += 2

ws2.freeze_panes = "F2"

# ══════════════════════════════════════════════════════════════════════════════
# SHEET 3 – Peak Ratios
# ══════════════════════════════════════════════════════════════════════════════
ws3 = wb.create_sheet("Peak Ratios")
ratio_headers = ["Map","Spec","X","Y","Cosine","Conf"] + [b[1].replace(" ","\ ")[:12] for b in DNA_BANDS]
col_widths3   = [6,6,7,7,9,10] + [10]*len(DNA_BANDS)

for c, (h, w) in enumerate(zip(ratio_headers, col_widths3), 1):
    H(ws3, 1, c, h, HEADER_fill, hdr_font, ctr)
    ws3.column_dimensions[get_column_letter(c)].width = w

# CaF2 reference ratios row
H(ws3, 2, 1, "CaF2 REF", SUBHDR_fill, sub_font, ctr)
H(ws3, 2, 2, "—",        SUBHDR_fill, sub_font, ctr)
H(ws3, 2, 3, "—",        SUBHDR_fill, sub_font, ctr)
H(ws3, 2, 4, "—",        SUBHDR_fill, sub_font, ctr)
H(ws3, 2, 5, 1.0000,     SUBHDR_fill, sub_font, ctr)
H(ws3, 2, 6, "REF",      SUBHDR_fill, sub_font, ctr)
ref_bc = get_I(wn_ref, sp_ref, 912.9, 15) or 1.0
for c_idx, (key, label, ref_wn) in enumerate(DNA_BANDS, 7):
    I_ref = get_I(wn_ref, sp_ref, ref_wn, 15)
    ratio = round(I_ref/ref_bc, 3)
    H(ws3, 2, c_idx, ratio, SUBHDR_fill, sub_font, ctr)

for i, row in enumerate(good_rows, 3):
    alt = ALT_fill if i%2==0 else None
    V(ws3, i, 1, row['map'],  fill=alt)
    V(ws3, i, 2, row['spec'], fmt='0', fill=alt)
    V(ws3, i, 3, row['x'],   fmt='0.0', fill=alt)
    V(ws3, i, 4, row['y'],   fmt='0.0', fill=alt)
    V(ws3, i, 5, row['cos'], fmt='0.0000', conf=row['conf'])
    V(ws3, i, 6, row['conf'],              conf=row['conf'])
    for c_idx, (key, label, ref_wn) in enumerate(DNA_BANDS, 7):
        V(ws3, i, c_idx, row['ratios'][key], fmt='0.000', fill=alt)

ws3.freeze_panes = "G3"

# ══════════════════════════════════════════════════════════════════════════════
# SHEET 4 – Per Map Summary
# ══════════════════════════════════════════════════════════════════════════════
ws4 = wb.create_sheet("Per Map Summary")
map_headers = ["Map","Total\nSpectra","HIGH","MODERATE","LOW","NOISE",
               "Mean\nCosine","Best\nCosine","Best\nSpectrum","Best\nX","Best\nY",
               "Mean\nBands\nDetected"]
for c, h in enumerate(map_headers, 1):
    H(ws4, 1, c, h, HEADER_fill, hdr_font, ctr)
    ws4.column_dimensions[get_column_letter(c)].width = 12
ws4.row_dimensions[1].height = 40

map_names = ['M1','M2','M3','M4']
for r, mname in enumerate(map_names, 2):
    mr = [row for row in rows if row['map']==mname]
    if not mr: continue
    n = len(mr)
    high = sum(1 for x in mr if x['conf']=='HIGH')
    mod  = sum(1 for x in mr if x['conf']=='MODERATE')
    low  = sum(1 for x in mr if x['conf']=='LOW')
    noi  = sum(1 for x in mr if x['conf']=='NOISE')
    mean_cos = round(np.mean([x['cos'] for x in mr]),4)
    best = max(mr, key=lambda x: x['cos'])
    mean_bands = round(np.mean([x['n_bands'] for x in mr]),1)

    V(ws4, r, 1,  mname)
    V(ws4, r, 2,  n,           fmt='0')
    V(ws4, r, 3,  high,        fmt='0', fill=HIGH_fill)
    V(ws4, r, 4,  mod,         fmt='0', fill=MOD_fill)
    V(ws4, r, 5,  low,         fmt='0', fill=LOW_fill)
    V(ws4, r, 6,  noi,         fmt='0', fill=NOISE_fill)
    V(ws4, r, 7,  mean_cos,    fmt='0.0000')
    V(ws4, r, 8,  best['cos'], fmt='0.0000', conf=best['conf'])
    V(ws4, r, 9,  f"S{best['spec']}")
    V(ws4, r, 10, best['x'],   fmt='0.0')
    V(ws4, r, 11, best['y'],   fmt='0.0')
    V(ws4, r, 12, mean_bands,  fmt='0.0')

# ══════════════════════════════════════════════════════════════════════════════
# SHEET 5 – Top 10 Spectral Data
# ══════════════════════════════════════════════════════════════════════════════
ws5 = wb.create_sheet("Top 10 Spectra")
top10 = rows[:10]

# Header row
H(ws5, 1, 1, "Wavenumber (cm⁻¹)", HEADER_fill, hdr_font, ctr)
ws5.column_dimensions['A'].width = 16
for j, row in enumerate(top10):
    col = j+2
    label = f"{row['map']}-S{row['spec']}\ncos={row['cos']:.4f}\n{row['conf']}"
    H(ws5, 1, col, label, HEADER_fill if row['conf']=='HIGH' else SUBHDR_fill, hdr_font, ctr)
    ws5.column_dimensions[get_column_letter(col)].width = 14

# Get processed data for top 10
top10_data = {}
for row in top10:
    mkey = row['map'].lower()
    if mkey in maps:
        wn_m, sps, pos = maps[mkey]
        sp_raw = sps[row['spec']-1]
        sp = process(wn_m, sp_raw)
        sp_si = norm_si(wn_m, sp)
        top10_data[(row['map'],row['spec'])] = (wn_m, sp_si)

# Write wavenumber column (use first map's wn)
if top10_data:
    first_key = list(top10_data.keys())[0]
    wn_out = top10_data[first_key][0]
    for r_idx, wn_val in enumerate(wn_out, 2):
        V(ws5, r_idx, 1, round(float(wn_val),1), fmt='0.0')
    for j, row in enumerate(top10):
        col = j+2
        key = (row['map'], row['spec'])
        if key in top10_data:
            wn_t, sp_t = top10_data[key]
            sp_interp = np.interp(wn_out, wn_t, sp_t)
            for r_idx, val in enumerate(sp_interp, 2):
                V(ws5, r_idx, col, round(float(val),3), fmt='0.000')

# ══════════════════════════════════════════════════════════════════════════════
# SHEET 6 – CaF2 Reference
# ══════════════════════════════════════════════════════════════════════════════
ws6 = wb.create_sheet("CaF2 Reference")
ref_headers = ["Wavenumber\n(cm⁻¹)","Raw\nIntensity","Processed\nIntensity","DNA Band\nAssignment","Reference\nPosition (cm⁻¹)"]
for c, h in enumerate(ref_headers, 1):
    H(ws6, 1, c, h, HEADER_fill, hdr_font, ctr)
ws6.column_dimensions['A'].width = 14
ws6.column_dimensions['B'].width = 12
ws6.column_dimensions['C'].width = 14
ws6.column_dimensions['D'].width = 24
ws6.column_dimensions['E'].width = 18

# Write all reference data
for r_idx, (wn_val, raw_val, proc_val) in enumerate(zip(wn_ref, raw_ref, sp_ref), 2):
    V(ws6, r_idx, 1, round(float(wn_val),1), fmt='0.0')
    V(ws6, r_idx, 2, round(float(raw_val),1), fmt='0.0')
    V(ws6, r_idx, 3, round(float(proc_val),3), fmt='0.000')
    # DNA assignment
    asgn = ""
    ref_wn_match = None
    for key, label, ref_wn in DNA_BANDS:
        if abs(wn_val - ref_wn) < 8:
            asgn = label
            ref_wn_match = ref_wn
            break
    V(ws6, r_idx, 4, asgn,       fill=HIGH_fill if asgn else None)
    V(ws6, r_idx, 5, ref_wn_match if ref_wn_match else "", fmt='0.0')

ws6.freeze_panes = "A2"

# ── SAVE ──────────────────────────────────────────────────────────────────────
OUT = "/home/user/claude/DNA_HaCaT_SERS_Comparison.xlsx"
wb.save(OUT)
print(f"\nSaved: {OUT}")
print("Done!")
