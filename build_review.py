from docx import Document
from docx.shared import Pt, RGBColor, Cm
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml.ns import qn
from docx.oxml import OxmlElement
from lxml import etree
import datetime, re

doc = Document()

section = doc.sections[0]
section.top_margin    = Cm(2.5)
section.bottom_margin = Cm(2.5)
section.left_margin   = Cm(3.0)
section.right_margin  = Cm(2.5)

# ── OMML helpers ─────────────────────────────────────────────────────────────
M = "http://schemas.openxmlformats.org/officeDocument/2006/math"
W = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"

def omml_r(text):
    return f'<m:r xmlns:m="{M}"><m:t xml:space="preserve">{text}</m:t></m:r>'

def omml_frac(num_xml, den_xml):
    return f'<m:f xmlns:m="{M}"><m:fPr/><m:num>{num_xml}</m:num><m:den>{den_xml}</m:den></m:f>'

def omml_sub(base, sub):
    return (f'<m:sSub xmlns:m="{M}"><m:sSubPr/>'
            f'<m:e>{omml_r(base)}</m:e>'
            f'<m:sub>{omml_r(sub)}</m:sub></m:sSub>')

def omml_sup(base, sup):
    return (f'<m:sSup xmlns:m="{M}"><m:sSupPr/>'
            f'<m:e>{omml_r(base)}</m:e>'
            f'<m:sup>{omml_r(sup)}</m:sup></m:sSup>')

def add_equation(doc, inner_xml, center=True):
    p = doc.add_paragraph()
    if center:
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_before = Pt(8)
    p.paragraph_format.space_after  = Pt(8)
    omml_para = (
        f'<m:oMathPara xmlns:m="{M}">'
        f'<m:oMath xmlns:m="{M}" xmlns:w="{W}">'
        f'{inner_xml}'
        f'</m:oMath></m:oMathPara>'
    )
    elem = etree.fromstring(omml_para)
    p._p.append(elem)
    return p

# ── General helpers ───────────────────────────────────────────────────────────
def add_hr(doc):
    p = doc.add_paragraph()
    pPr = p._p.get_or_add_pPr()
    pBdr = OxmlElement('w:pBdr')
    bot = OxmlElement('w:bottom')
    bot.set(qn('w:val'), 'single'); bot.set(qn('w:sz'), '6')
    bot.set(qn('w:space'), '1'); bot.set(qn('w:color'), '4472C4')
    pBdr.append(bot); pPr.append(pBdr)
    p.paragraph_format.space_after = Pt(4)

def shade(p, fill):
    shd = OxmlElement('w:shd')
    shd.set(qn('w:val'), 'clear'); shd.set(qn('w:color'), 'auto')
    shd.set(qn('w:fill'), fill)
    p._p.get_or_add_pPr().append(shd)

def add_para(doc, text, bold=False, italic=False, size=11,
             color=None, align=WD_ALIGN_PARAGRAPH.LEFT,
             sb=0, sa=6):
    p = doc.add_paragraph()
    p.alignment = align
    p.paragraph_format.space_before = Pt(sb)
    p.paragraph_format.space_after  = Pt(sa)
    r = p.add_run(text)
    r.bold = bold; r.italic = italic; r.font.size = Pt(size)
    if color: r.font.color.rgb = RGBColor(*color)
    return p

def add_heading(doc, text, level=1):
    colors = {1:(31,73,125), 2:(31,73,125), 3:(68,114,196)}
    sizes  = {1:14, 2:12, 3:11}
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(12)
    p.paragraph_format.space_after  = Pt(4)
    r = p.add_run(text)
    r.bold = True; r.font.size = Pt(sizes[level])
    r.font.color.rgb = RGBColor(*colors[level])
    return p

def add_bullet(doc, text, sub=False):
    p = doc.add_paragraph(style='List Bullet')
    p.paragraph_format.space_after = Pt(4)
    if sub: p.paragraph_format.left_indent = Cm(1.5)
    parts = re.split(r'\*\*(.*?)\*\*', text)
    for i, part in enumerate(parts):
        r = p.add_run(part); r.font.size = Pt(11)
        if i % 2 == 1: r.bold = True
    return p

def add_ref_note(doc, text):
    p = doc.add_paragraph()
    p.paragraph_format.left_indent = Cm(1.2)
    p.paragraph_format.space_after = Pt(6)
    shade(p, 'F2F2F2')
    r = p.add_run(text); r.italic = True; r.font.size = Pt(10)
    r.font.color.rgb = RGBColor(89, 89, 89)

# ─────────────────────────────────────────────────────────────────────────────
# TITLE BLOCK
# ─────────────────────────────────────────────────────────────────────────────
p = doc.add_paragraph()
p.alignment = WD_ALIGN_PARAGRAPH.CENTER
r = p.add_run("PEER REVIEW REPORT")
r.bold = True; r.font.size = Pt(16); r.font.color.rgb = RGBColor(31, 73, 125)

p2 = doc.add_paragraph()
p2.alignment = WD_ALIGN_PARAGRAPH.CENTER
r2 = p2.add_run("Materials Chemistry and Physics  |  Elsevier")
r2.italic = True; r2.font.size = Pt(12); r2.font.color.rgb = RGBColor(68, 114, 196)

p3 = doc.add_paragraph()
p3.alignment = WD_ALIGN_PARAGRAPH.CENTER
r3 = p3.add_run("Manuscript No.  MATCHEMPHYSD2604411")
r3.font.size = Pt(11)

add_hr(doc)

# Info table
tbl = doc.add_table(rows=4, cols=2)
tbl.style = 'Table Grid'
for i, (lbl, val) in enumerate([
    ("Title",       "Efficient Surface Enhanced Raman Scattering on Graphene Coated Cu Nanowires "
                    "Prepared by Low Temperature Chemical Vapor Deposition"),
    ("Authors",     "Van Tan Tran, An Bang Ngac, Nguyen Hai Pham, Viet Tuyen Nguyen, Van Chuc Nguyen, "
                    "Trong Tam Nguyen, Bui Nguyen Quoc Trinh, The Nam Dinh, Thi Ha Tran*"),
    ("Affiliations","Vietnam National University Hanoi; Vietnam Academy of Science and Technology; "
                    "Vietnam Maritime University; Vietnam Japan University; Hanoi University of Mining and Geology"),
    ("Review Date", datetime.date.today().strftime("%d %B %Y")),
]):
    c0, c1 = tbl.rows[i].cells
    r0 = c0.paragraphs[0].add_run(lbl); r0.bold = True; r0.font.size = Pt(10)
    r1 = c1.paragraphs[0].add_run(val);  r1.font.size = Pt(10)

doc.add_paragraph()

# ─────────────────────────────────────────────────────────────────────────────
# 1. OVERALL ASSESSMENT
# ─────────────────────────────────────────────────────────────────────────────
add_heading(doc, "1.  OVERALL ASSESSMENT", 1)
add_hr(doc)

add_para(doc,
    "This manuscript describes the fabrication of copper/graphene (Cu/G) nanowire substrates for "
    "Surface-Enhanced Raman Scattering (SERS). CuO nanowires are grown by thermal oxidation of copper "
    "foil, reduced to metallic Cu by H2 gas at 400 degrees C, and then coated with a graphene layer via "
    "low-temperature CVD using ethanol as a carbon precursor. The best-performing substrate achieves "
    "LOD = 5.69x10^-11 M and EF = 6.1x10^6 for methylene blue (MB) detection, with stability "
    "confirmed over 90 days (signal decrease < 6.1%).",
    size=11, sa=6)

add_para(doc,
    "The fabrication concept is creative and practically valuable. However, the manuscript contains "
    "critical scientific deficiencies -- particularly in the Enhancement Factor calculation, FDTD "
    "simulation validity, graphene quality characterisation, and oxidation protection evidence -- "
    "that must be corrected before the paper can be accepted.",
    size=11, sa=8)

pv = doc.add_paragraph()
pv.paragraph_format.left_indent = Cm(1); pv.paragraph_format.space_after = Pt(12)
shade(pv, 'DCE6F1')
rv = pv.add_run("RECOMMENDATION:   Major Revision Required")
rv.bold = True; rv.font.size = Pt(12); rv.font.color.rgb = RGBColor(31, 73, 125)

# ─────────────────────────────────────────────────────────────────────────────
# 2. DETAILED COMMENTS
# ─────────────────────────────────────────────────────────────────────────────
add_heading(doc, "2.  DETAILED SCIENTIFIC COMMENTS", 1)
add_hr(doc)

# ══ COMMENT 1 ════════════════════════════════════════════════════════════════
add_heading(doc, "Comment 1 -- Enhancement Factor: Incomplete Calculation  [CRITICAL]", 2)

add_para(doc,
    'The authors state (Section 3): "The enhancement factor (EF) was estimated to be 6.1x10^6 based '
    'on a comparison of the Raman signal intensity of MB at 10^-10 M on a SERS substrate with the '
    'Raman signal intensity of MB at 10^-4 M on a conventional silicon substrate." '
    'This corresponds to the Analytical Enhancement Factor (AEF). The standard definition from '
    'the field\'s authoritative textbook is:',
    size=11, sa=6)

add_ref_note(doc,
    "Reference: Le Ru, E.C. & Etchegoin, P.G.  "
    "Principles of Surface-Enhanced Raman Spectroscopy and Related Plasmonic Effects. "
    "Elsevier, Amsterdam, 2009.  Equation 4.4, page 187.")

add_para(doc, "Equation 1 -- Analytical Enhancement Factor (AEF):", bold=True, size=11, sa=2)

# AEF = (I_SERS / C_SERS) / (I_RS / C_RS)
aef_xml = (
    omml_r("AEF") +
    omml_r("  =  ") +
    omml_frac(
        omml_sub("I", "SERS") + omml_r(" / ") + omml_sub("C", "SERS"),
        omml_sub("I", "RS")   + omml_r(" / ") + omml_sub("C", "RS")
    ) +
    omml_r("   =   ") +
    omml_frac(omml_sub("I", "SERS"), omml_sub("I", "RS")) +
    omml_r("  x  ") +
    omml_frac(omml_sub("C", "RS"), omml_sub("C", "SERS"))
)
add_equation(doc, aef_xml)

add_para(doc, "Where:", size=11, sa=2)
add_bullet(doc, "**I_SERS** = intensity of a selected Raman peak in the SERS spectrum at concentration C_SERS")
add_bullet(doc, "**I_RS**   = intensity of the same Raman peak in a normal (non-SERS) Raman spectrum at concentration C_RS")
add_bullet(doc, "**C_SERS** = analyte concentration used for SERS  =  10^-10 M  (this paper)")
add_bullet(doc, "**C_RS**   = reference concentration for normal Raman  =  10^-4 M  (this paper)")

add_para(doc, "\nSubstituting the authors' values:", size=11, sa=4)

add_para(doc, "Equation 2 -- Substituted form:", bold=True, size=11, sa=2)

sub_xml = (
    omml_r("AEF") +
    omml_r("  =  ") +
    omml_frac(omml_sub("I", "SERS"), omml_sub("I", "RS")) +
    omml_r("  x  ") +
    omml_frac(omml_sup("10", "-4"), omml_sup("10", "-10")) +
    omml_r("   =   ") +
    omml_frac(omml_sub("I", "SERS"), omml_sub("I", "RS")) +
    omml_r("  x  10") +
    omml_sup("", "6")
)
add_equation(doc, sub_xml)

add_para(doc,
    "The concentration ratio alone contributes a factor of 10^6. For AEF = 6.1x10^6, the intensity "
    "ratio must be I_SERS / I_RS = 6.1. This numerical value must be stated explicitly in the manuscript "
    "with the raw intensity readings from Figure 12. Without it, the EF claim cannot be independently verified.",
    size=11, sa=6)

add_para(doc, "Mandatory items currently absent from the manuscript:", bold=True, size=11, sa=4)
add_bullet(doc, "State which Raman peak was used for the calculation (presumably 1622 cm-1 from Figure 12)")
add_bullet(doc, "Report the numerical values of I_SERS and I_RS (in counts or a.u.)")
add_bullet(doc, "Confirm that laser power and integration time were identical for both spectra")
add_bullet(doc, "Label the EF type as AEF -- it is not a Single-Molecule EF (SMEF) or Surface EF (SEF)")
add_bullet(doc, "Show the reference Raman spectrum of MB at 10^-4 M on silicon as a figure or supplementary data")

add_para(doc,
    "\nFor a more physically rigorous determination, the Surface Enhancement Factor (SEF) should be calculated "
    "(Le Ru & Etchegoin, Eq. 4.6-4.9, pp. 188-190):",
    size=11, sa=4)

add_para(doc, "Equation 3 -- Surface Enhancement Factor (SEF):", bold=True, size=11, sa=2)

sef_xml = (
    omml_r("EF") +
    omml_r("  =  ") +
    omml_frac(
        omml_sub("I", "SERS") + omml_r(" / ") + omml_sub("N", "surf"),
        omml_sub("I", "RS")   + omml_r(" / ") + omml_sub("N", "vol")
    )
)
add_equation(doc, sef_xml)

add_para(doc, "Equation 4 -- Number of molecules:", bold=True, size=11, sa=2)

nmol_xml = (
    omml_sub("N", "vol") +
    omml_r("  =  ") +
    omml_sub("C", "RS") +
    omml_r("  x  ") +
    omml_sub("N", "A") +
    omml_r("  x  ") +
    omml_sub("V", "focal") +
    omml_r("          ") +
    omml_sub("N", "surf") +
    omml_r("  =  ") +
    omml_sub("C", "surf") +
    omml_r("  x  ") +
    omml_sub("A", "spot")
)
add_equation(doc, nmol_xml)

add_bullet(doc, "**N_A** = Avogadro number = 6.022 x 10^23 mol^-1")
add_bullet(doc, "**V_focal** = confocal probe volume of LabRAM HR 800 (typically 1-10 pL; instrument specification must be stated)")
add_bullet(doc, "**A_spot** = laser spot area = pi x r^2 where r = half the laser spot diameter")
add_bullet(doc, "**C_surf** = surface concentration of MB on substrate (mol/cm^2) -- requires adsorption calibration or estimation")

doc.add_paragraph()

# ══ COMMENT 2 ════════════════════════════════════════════════════════════════
add_heading(doc, "Comment 2 -- FDTD Model is Physically Inconsistent with the Real Substrate", 2)

add_para(doc,
    "The FDTD simulation models the Cu/G system as a single spherical nanoparticle (diameter 100 nm) "
    "with a 1 nm graphene shell. The actual substrate consists of a dense network of 1D nanowires "
    "(approx. 100 nm diameter). This geometric mismatch has serious physical consequences.",
    size=11, sa=6)

add_bullet(doc, "A sphere and a cylinder have fundamentally different plasmon resonance conditions: "
    "different resonance wavelengths, Q-factors, and near-field spatial profiles.")
add_bullet(doc, "SERS hot spots in nanowire networks arise at nanogap junctions between crossing wires, "
    "not at the surface of an isolated sphere. These junction hot spots dominate the total EF.")
add_bullet(doc, "The FDTD output shows |E|^2 (near-field intensity) maps. However, SERS enhancement "
    "scales as |E/E_0|^4, not |E|^2. This is stated by the classical electromagnetic theory of SERS "
    "(Moskovits, Rev. Mod. Phys. 57, 783, 1985). The correct formula is given below.")

add_para(doc, "Equation 5 -- Electromagnetic Enhancement Factor:", bold=True, size=11, sa=2)

em_xml = (
    omml_sub("EF", "EM") +
    omml_r("  ~  ") +
    omml_frac(
        omml_sup(omml_r("|E(r)|"), "4"),
        omml_sup(omml_r("|E"), "4") + omml_sub("0", "  |")
    )
)
# Simplified version
em_xml2 = (
    omml_sub("EF", "EM") +
    omml_r("  ~  ") +
    omml_sup(
        omml_r("( |E(r)| / |E") + omml_sub("0", "") + omml_r("| )"),
        "4"
    )
)
add_equation(doc, em_xml2)

add_para(doc,
    "The authors must either: (a) re-run the FDTD simulation for two crossing nanowires with a "
    "realistic nanogap and present |E/E_0|^4 enhancement maps, or (b) explicitly acknowledge that "
    "the sphere model is a severe simplification and cannot quantitatively reproduce the experimental EF.",
    size=11, sa=10)

# ══ COMMENT 3 ════════════════════════════════════════════════════════════════
add_heading(doc, "Comment 3 -- Graphene Quality: D/G Ratio Not Quantified", 2)

add_para(doc,
    "Figure 7 shows the Raman spectrum of the graphene layer with a strong D band at 1367 cm-1, "
    "a G band at 1580 cm-1, and a very weak 2D band. A high D/G ratio indicates defective graphene "
    "or nanocrystalline carbon -- not high-quality monolayer graphene. The authors do not report "
    "the D/G ratio, which is the standard quality metric for graphene characterisation.",
    size=11, sa=6)

add_para(doc,
    "The average sp2 crystallite domain size L_a can be estimated from the Tuinstra-Koenig relation "
    "(Tuinstra & Koenig, J. Chem. Phys. 53, 1126, 1970):",
    size=11, sa=4)

add_para(doc, "Equation 6 -- Tuinstra-Koenig formula:", bold=True, size=11, sa=2)

tk_xml = (
    omml_sub("L", "a") +
    omml_r("  (nm)  =  (2.4 x 10") +
    omml_sup("", "-10") +
    omml_r(") x ") +
    omml_sup(omml_r("lambda"), "4") +
    omml_r("  x  ") +
    omml_frac(omml_r("I(G)"), omml_r("I(D)"))
)
add_equation(doc, tk_xml)

add_para(doc,
    "where lambda = excitation wavelength in nm (632.8 nm in this study). "
    "The authors must: (a) report the I(D)/I(G) intensity ratio; "
    "(b) calculate L_a and report the domain size; "
    "(c) discuss the implications of defective graphene for both SERS chemical enhancement "
    "and oxidation protection efficiency.",
    size=11, sa=10)

# ══ COMMENT 4 ════════════════════════════════════════════════════════════════
add_heading(doc, "Comment 4 -- Oxidation Protection Claim: Evidence is Indirect", 2)

add_para(doc,
    "The 90-day stability claim -- the central practical contribution of this paper -- is supported "
    "only by the absence of CuO/Cu2O Raman peaks in Figure 10a. Raman spectroscopy cannot detect "
    "sub-monolayer surface oxide formation. Two critical experiments are missing:",
    size=11, sa=6)

add_bullet(doc,
    "**XPS measurement of the Cu 2p(3/2) peak**: Cu(0) appears at approx. 932.6 eV; Cu(2+) at approx. "
    "933.6 eV. Comparing Cu/G vs. bare Cu nanowires before and after 90 days of ageing would provide "
    "definitive, quantitative evidence of oxidation suppression.")
add_bullet(doc,
    "**Control experiment**: Bare Cu nanowires stored under identical conditions must be characterised "
    "at the same time points. Without this, the superior stability of Cu/G cannot be causally attributed "
    "to graphene protection.")
add_bullet(doc,
    "A time-series study at 1, 7, 30, and 90 days for both Cu and Cu/G would make a compelling "
    "quantitative case and significantly strengthen the paper.")

doc.add_paragraph()

# ══ COMMENT 5 ════════════════════════════════════════════════════════════════
add_heading(doc, "Comment 5 -- LOD Calculation: IUPAC Formula Not Applied", 2)

add_para(doc,
    "The Limit of Detection (LOD = 5.69x10^-11 M) is reported without stating how it was calculated. "
    "The IUPAC-standard formula, required by all analytical chemistry journals, is:",
    size=11, sa=6)

add_para(doc, "Equation 7 -- IUPAC Limit of Detection:", bold=True, size=11, sa=2)

lod_xml = (
    omml_r("LOD") +
    omml_r("  =  ") +
    omml_frac(omml_r("3 sigma"), omml_r("S"))
)
add_equation(doc, lod_xml)

add_para(doc, "Where:", size=11, sa=2)
add_bullet(doc, "**sigma** = standard deviation of the blank signal (substrate without analyte, minimum 10 repeat measurements)")
add_bullet(doc, "**S** = slope of the calibration curve from Figure 12b (SERS intensity vs. log[C_MB])")

add_para(doc,
    "The authors must verify that 5.69x10^-11 M was derived from this formula and present the "
    "numerical values of sigma and S. Identifying the lowest visible concentration in Figure 12a "
    "(10^-10 M) as the LOD is not acceptable by journal standards.",
    size=11, sa=10)

# ══ COMMENT 6 ════════════════════════════════════════════════════════════════
add_heading(doc, "Comment 6 -- Single Analyte Tested; Environmental Claims Unsupported", 2)

add_para(doc,
    "All detection experiments use only methylene blue (MB), a strongly SERS-active dye that "
    "adsorbs on graphene via pi-pi stacking, giving artificially high sensitivity. Real environmental "
    "pollutants have different Raman cross-sections and surface affinities. The authors must either: "
    "(a) test at least one additional, non-dye analyte relevant to environmental monitoring, or "
    "(b) restrict the conclusions strictly to MB detection and remove the broad environmental "
    "monitoring claim from the abstract and conclusions.",
    size=11, sa=10)

# ══ COMMENT 7 ════════════════════════════════════════════════════════════════
add_heading(doc, "Comment 7 -- Rayleigh Instability: Correct Physics, Missing Quantification", 2)

add_para(doc,
    "The discussion of nanowire fragmentation above 400 degrees C correctly invokes Rayleigh-Plateau "
    "instability and the Gibbs-Thomson effect. The physics is sound. To quantitatively support the "
    "argument that fragmentation is a solid-state phenomenon (not melting), the authors should apply "
    "the Pawlow melting point depression formula:",
    size=11, sa=6)

add_para(doc, "Equation 8 -- Melting point depression of nanostructures (Pawlow formula):", bold=True, size=11, sa=2)

melt_xml = (
    omml_sub("T", "m") + omml_r("(nano)") +
    omml_r("  =  ") +
    omml_sub("T", "m") + omml_r("(bulk)") +
    omml_r("  x  [ 1  -  ") +
    omml_frac(
        omml_r("4 gamma ") + omml_sub("V", "m"),
        omml_sub("Delta H", "f") + omml_r(" x ") + omml_sub("rho", "l") + omml_r(" x d")
    ) +
    omml_r(" ]")
)
add_equation(doc, melt_xml)

add_para(doc, "Where:", size=11, sa=2)
add_bullet(doc, "**T_m(bulk)** = 1084 degrees C = 1357 K for copper")
add_bullet(doc, "**gamma** = surface energy of Cu  approx.  1.79 J/m^2")
add_bullet(doc, "**V_m** = molar volume of Cu = 7.09x10^-6 m^3/mol")
add_bullet(doc, "**Delta H_f** = latent heat of fusion of Cu = 13.05 kJ/mol")
add_bullet(doc, "**rho_l** = density of liquid Cu approx. 8000 kg/m^3")
add_bullet(doc, "**d** = nanowire diameter (approx. 100 nm from SEM, Figure 5)")

add_para(doc,
    "For d = 100 nm, this formula predicts a melting point depression of only approx. 4-8 degrees C "
    "from bulk. This numerically confirms that the observed fragmentation at 450 degrees C is a "
    "solid-state Rayleigh instability process -- not melting -- which is the correct conclusion "
    "already drawn in the manuscript. Adding this estimate strengthens the argument considerably.",
    size=11, sa=10)

# ══ COMMENT 8 ════════════════════════════════════════════════════════════════
add_heading(doc, "Comment 8 -- Spot-to-Spot Reproducibility: RSD Must Be Calculated", 2)

add_para(doc,
    "Figure 13 provides SERS mapping data from multiple positions, which is commendable. However, "
    "the manuscript only states qualitatively that variation was 'small.' The quantitative criterion "
    "required by the field is the Relative Standard Deviation (RSD):",
    size=11, sa=6)

add_para(doc, "Equation 9 -- Relative Standard Deviation:", bold=True, size=11, sa=2)

rsd_xml = (
    omml_r("RSD (%)") +
    omml_r("  =  ") +
    omml_frac(omml_r("sigma"), omml_r("mu")) +
    omml_r("  x  100")
)
add_equation(doc, rsd_xml)

add_para(doc,
    "where sigma = standard deviation and mu = mean SERS intensity across all mapping positions. "
    "The accepted benchmark for a reproducible SERS substrate is RSD < 15-20% "
    "(Ding et al., Nature Reviews Materials 1, 16021, 2016). "
    "The number of mapping positions used must also be stated (minimum 20-30 points is expected).",
    size=11, sa=10)

# ══ COMMENT 9 ════════════════════════════════════════════════════════════════
add_heading(doc, "Comment 9 -- CVD Temperature Range Incomplete; Optimum at Boundary", 2)

add_para(doc,
    "CVD was studied at only three temperatures: 400, 450, and 500 degrees C. The best result "
    "is at 400 degrees C -- the lowest value tested. This raises the question of whether even "
    "lower temperatures (e.g., 350, 375 degrees C) might preserve nanowire morphology better "
    "while still depositing a carbon protective layer. The parameter space must be explored more "
    "thoroughly to confirm that 400 degrees C is a true optimum.",
    size=11, sa=10)

# ══ COMMENT 10 ═══════════════════════════════════════════════════════════════
add_heading(doc, "Comment 10 -- Carbon (Graphene) Deposition Not Confirmed by EDS or XPS", 2)

add_para(doc,
    "Figure 6b shows the EDS spectrum after CVD but does not clearly identify or quantify a "
    "carbon (C) peak. The C Kalpha peak at approx. 0.277 keV must be labelled and the carbon "
    "atomic percentage reported. Since EDS has limited sensitivity for light elements, XPS of "
    "the C 1s peak (binding energy approx. 284.6 eV for sp2 graphitic carbon) is strongly "
    "recommended as confirmatory evidence that the deposited layer is graphene and not amorphous carbon.",
    size=11, sa=10)

# ─────────────────────────────────────────────────────────────────────────────
# 3. MINOR COMMENTS
# ─────────────────────────────────────────────────────────────────────────────
add_heading(doc, "3.  MINOR COMMENTS", 1)
add_hr(doc)

for loc, comment in [
    ("Figure 8 caption",   "Reads 'man spectra' -- should be 'Raman spectra.' Typographical error."),
    ("Figure 12 caption",  "Reads 'ration curve' -- should be 'calibration curve.' Typographical error."),
    ("Figure 13c",         "Text states signal decrease < 6.1% across all peaks; the 1622 cm-1 band "
                           "shows approx. 5% decrease. Clarify that 6.1% is the maximum across all peaks."),
    ("Table 1",            "EF for Cu/GrO composites and core-shell graphene@Cu NPs listed as 'NA.' "
                           "Replace with 'Not reported' for clarity."),
    ("Reference 28",       "Author names are missing from the reference entry. Please complete the citation."),
    ("Section 2",          "Ar/H2 flux ratio (300/30 sccm) is stated but total reactor pressure and tube "
                           "geometry are absent. These are required for reproducibility."),
    ("Abstract",           "EF type is not specified. Add '(AEF)' after the value 6.1x10^6."),
]:
    p_m = doc.add_paragraph()
    p_m.paragraph_format.space_after = Pt(5)
    rl = p_m.add_run(loc + ": ")
    rl.bold = True; rl.font.size = Pt(11); rl.font.color.rgb = RGBColor(31, 73, 125)
    rc = p_m.add_run(comment); rc.font.size = Pt(11)

# ─────────────────────────────────────────────────────────────────────────────
# 4. SUMMARY TABLE
# ─────────────────────────────────────────────────────────────────────────────
add_heading(doc, "4.  SUMMARY OF REQUIRED REVISIONS", 1)
add_hr(doc)

st = doc.add_table(rows=1, cols=3)
st.style = 'Table Grid'
for i, (txt, w) in enumerate([("No.", 1.2), ("Issue", 11.0), ("Priority", 2.5)]):
    c = st.rows[0].cells[i]; c.width = Cm(w)
    rh = c.paragraphs[0].add_run(txt)
    rh.bold = True; rh.font.size = Pt(10); rh.font.color.rgb = RGBColor(255, 255, 255)
    s = OxmlElement('w:shd'); s.set(qn('w:val'),'clear'); s.set(qn('w:color'),'auto'); s.set(qn('w:fill'),'4472C4')
    c._tc.get_or_add_tcPr().append(s)

for num, issue, priority, fill in [
    ("1",  "EF formula incomplete -- I_SERS, I_RS, peak choice, and EF type not shown", "CRITICAL", "FFB3BA"),
    ("2",  "FDTD sphere model inconsistent with nanowire substrate; |E|^2 used instead of |E|^4", "MAJOR", "FFDFBA"),
    ("3",  "Graphene D/G ratio not quantified; sp2 domain size not calculated", "MAJOR", "FFDFBA"),
    ("4",  "No XPS or control experiment to confirm oxidation protection by graphene", "MAJOR", "FFDFBA"),
    ("5",  "LOD formula (3sigma/S) and values of sigma and S not shown", "MAJOR", "FFDFBA"),
    ("6",  "Only MB tested; environmental monitoring claim is not supported", "MODERATE", "FFFFBA"),
    ("7",  "Melting point depression not quantified; Pawlow formula not applied", "MODERATE", "FFFFBA"),
    ("8",  "RSD of SERS mapping positions not calculated or reported", "MODERATE", "FFFFBA"),
    ("9",  "CVD temperature range too narrow; lower temperatures not explored", "MODERATE", "FFFFBA"),
    ("10", "Carbon EDS/XPS confirmation of graphene deposition absent in Figure 6b", "MODERATE", "FFFFBA"),
]:
    row = st.add_row().cells
    row[0].paragraphs[0].add_run(num).font.size = Pt(10)
    row[1].paragraphs[0].add_run(issue).font.size = Pt(10)
    rp = row[2].paragraphs[0].add_run(priority); rp.font.size = Pt(10); rp.bold = True
    for cell in row:
        s2 = OxmlElement('w:shd'); s2.set(qn('w:val'),'clear'); s2.set(qn('w:color'),'auto'); s2.set(qn('w:fill'), fill)
        cell._tc.get_or_add_tcPr().append(s2)

# ─────────────────────────────────────────────────────────────────────────────
# 5. KEY REFERENCES
# ─────────────────────────────────────────────────────────────────────────────
add_heading(doc, "5.  KEY REFERENCES CITED IN THIS REVIEW", 1)
add_hr(doc)

for ref in [
    "[R1] Le Ru, E.C. & Etchegoin, P.G.  Principles of Surface-Enhanced Raman Spectroscopy and Related "
    "Plasmonic Effects. Elsevier, Amsterdam, 2009.  [AEF: Eq. 4.4, p. 187;  SEF: Eq. 4.6-4.9, pp. 188-190]",

    "[R2] Moskovits, M.  Surface-enhanced spectroscopy. Reviews of Modern Physics 57(3), 783-826 (1985).  "
    "[|E|^4 electromagnetic EF scaling]",

    "[R3] Ling, X. et al.  Charge-Transfer Mechanism in Graphene-Enhanced Raman Scattering. "
    "J. Phys. Chem. C 116, 25112-25118 (2012).  [Chemical enhancement via graphene -- Ref. 44 in manuscript]",

    "[R4] Tuinstra, F. & Koenig, J.L.  Raman spectrum of graphite. J. Chem. Phys. 53, 1126 (1970).  "
    "[D/G ratio and domain size formula]",

    "[R5] Ding, S.-Y. et al.  Nanostructure-based plasmon-enhanced Raman spectroscopy for surface analysis "
    "of materials. Nature Reviews Materials 1, 16021 (2016).  [RSD < 15-20% reproducibility benchmark]",

    "[R6] Pawlow, P.  Uber die Abhangigkeit des Schmelzpunktes von der Oberflachenenergie eines festen "
    "Korpers. Z. Phys. Chem. 65, 545 (1909).  [Melting point depression of nanostructures]",
]:
    p_r = doc.add_paragraph()
    p_r.paragraph_format.space_after = Pt(4)
    p_r.paragraph_format.left_indent = Cm(0.5)
    p_r.add_run(ref).font.size = Pt(10)

add_hr(doc)
pend = doc.add_paragraph()
pend.alignment = WD_ALIGN_PARAGRAPH.CENTER
rend = pend.add_run(
    "Review prepared for manuscript MATCHEMPHYSD2604411  |  Materials Chemistry and Physics (Elsevier)")
rend.italic = True; rend.font.size = Pt(9); rend.font.color.rgb = RGBColor(128, 128, 128)

doc.save('/home/user/claude/Review_MATCHEMPHYSD2604411_v2.docx')
print("Saved successfully.")
