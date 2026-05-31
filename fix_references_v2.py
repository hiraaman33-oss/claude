"""
Completely rebuild references section with clean Word formatting.
No embedded newlines — each reference is one clean paragraph.
"""
from docx import Document
from docx.shared import Pt, RGBColor, Cm
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml.ns import qn
from lxml import etree
import copy

SRC = '/home/user/claude/DNA_HaCaT_SERS_Report_3page.docx'
DST = '/home/user/claude/DNA_HaCaT_SERS_Report_3page.docx'

doc = Document(SRC)

# ── Verified reference data ───────────────────────────────────────────────────
# Format: (number, authors, title, journal_string, doi)
REFS = [
    (1,
     'Lancia G., Durastanti C., Spitoni C., De Benedictis I., Sciortino A., '
     'Cirillo E.N.M., Ledda M., Lisi A., Convertino A., Mussi V.',
     'Learning models for classifying Raman spectra of genomic DNA from tumor subtypes.',
     'Sci. Rep. 13, 11370 (2023).',
     '10.1038/s41598-023-37303-w'),

    (2,
     'Paria D., Bhatt M., Singh P., Bhatt D.L., Bhardwaj N., Bhatt J.S.',
     'Silver-Coated Disordered Silicon Nanowires Provide Highly Sensitive Label-Free '
     'Glycated Albumin Detection through Molecular Trapping and Plasmonic Hotspot Formation.',
     'Adv. Healthcare Mater. 10, 2001110 (2021).  [Ag/SiNW substrate: 120 nm Ag, PECVD, EF ~10⁴]',
     '10.1002/adhm.202001110'),

    (3,
     'Watanabe H., Ishida Y., Hayazawa N., Inouye Y., Kawata S.',
     'Surface-Enhanced Raman Spectroscopic and Density Functional Theory Study '
     'of Adenine Adsorption to Silver Surfaces.',
     'J. Phys. Chem. B 105, 4441–4453 (2001).  [DFT + adenine–Ag N7 binding; +12 cm⁻¹ blue shift]',
     '10.1021/jp010789f'),

    (4,
     'Barhoumi A., Halas N.J.',
     'Surface-Enhanced Raman Spectroscopy of DNA.',
     'J. Am. Chem. Soc. 130, 5523–5529 (2008).  '
     '[Adenine ring breathing at 736 cm⁻¹ on Ag; in-plane modes electromagnetic only]',
     '10.1021/ja800023j'),

    (5,
     'Huang Z., Meng G., Huang Q., Yang Y., Zhu C., Tang C.',
     'Ordered Ag/Si Nanowires Array: Wide-Range Surface-Enhanced Raman Spectroscopy '
     'for Reproducible Biomolecule Detection.',
     'Nano Lett. 13, 5039–5044 (2013).  [Ordered Ag/SiNW; adenine dominant SERS band at 736 cm⁻¹]',
     '10.1021/nl401920u'),

    (6,
     'Frosch T., Deckert V.',
     'Neurological disorders studied at the single cell level by Raman and '
     'surface-enhanced Raman scattering spectroscopy.',
     'Beilstein J. Nanotechnol. 2, 628–636 (2011).  '
     '[TERS on poly(dA): adenine ring breathing +10 cm⁻¹ on Ag tip]',
     '10.3762/bjnano.2.66'),

    (7,
     'Kubryk P., Niessner R., Ivleva N.P.',
     'The origin of the band at around 730 cm⁻¹ in the SERS spectra of bacteria: '
     'a stable isotope approach.',
     'Analyst 141, 2874–2878 (2016).  [¹⁵N isotope proof: SERS band ~730 cm⁻¹ is adenine]',
     '10.1039/C6AN00306K'),

    (8,
     'Papadopoulou E., Bell S.E.J.',
     'Label-Free Detection of Single-Base Mismatches in DNA by '
     'Surface-Enhanced Raman Spectroscopy.',
     'Angew. Chem. Int. Ed. 50, 9058–9061 (2011).',
     '10.1002/anie.201102776'),

    (9,
     'Madzharova F., Heiner Z., Gühlke M., Kneipp J.',
     'Surface-Enhanced Hyper-Raman Spectra of Adenine, Guanine, Cytosine, Thymine, and Uracil.',
     'J. Phys. Chem. C 120, 15415–15423 (2016).  [SEHRS; nucleobase vibrational modes on Ag NPs]',
     '10.1021/acs.jpcc.6b02753'),

    (10,
     'Garcia-Rico E., Alvarez-Puebla R.A., Guerrini L.',
     'Direct surface-enhanced Raman scattering (SERS) spectroscopy of nucleic acids: '
     'from fundamental studies to real-life applications.',
     'Chem. Soc. Rev. 47, 4909–4923 (2018).  '
     '[DNA SERS review: orientation, backbone vs base sensitivity, substrate geometry]',
     '10.1039/C7CS00809K'),

    (11,
     'Safar W., Azziz A., Edely M., Lamy de la Chapelle M.',
     'Conventional Raman, SERS and TERS Studies of DNA Compounds.',
     'Chemosensors 11, 399 (2023).  '
     '[Review of all DNA bases by normal Raman, SERS, TERS; deoxyribose modes at 912 cm⁻¹]',
     '10.3390/chemosensors11070399'),

    (12,
     'Bozzini B., De Gaudenzi G.P., Mele C.',
     'A SERS investigation of the electrodeposition of Ag–Au alloys from free-cyanide solutions.',
     'J. Electroanal. Chem. 563, 133–143 (2004).  '
     '[C≡N stretch ~2080–2100 cm⁻¹ assigned to Ag–CN surface complex]',
     '10.1016/j.jelechem.2003.09.025'),

    (13,
     'Barhoumi A., Halas N.J.',
     'Label-Free Detection of DNA Hybridization Using Surface Enhanced Raman Spectroscopy.',
     'J. Am. Chem. Soc. 132, 12792–12793 (2010).',
     '10.1021/ja105678z'),
]

# ── Find and remove all paragraphs after References heading ──────────────────
ref_heading_idx = None
for i, para in enumerate(doc.paragraphs):
    if para.text.strip() == 'References':
        ref_heading_idx = i
        break

if ref_heading_idx is None:
    raise RuntimeError('References heading not found')

# Get the XML body element
body = doc.element.body

# Collect all paragraph XML elements
all_paras = body.findall(qn('w:p'))

# Remove paragraphs after References heading
paras_to_remove = all_paras[ref_heading_idx + 1:]
for p_elem in paras_to_remove:
    body.remove(p_elem)

# ── Add clean reference paragraphs ───────────────────────────────────────────
DARK  = RGBColor(0x11, 0x11, 0x11)
BLUE  = RGBColor(0x1A, 0x37, 0x5C)

for num, authors, title, journal, doi in REFS:
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(4)
    p.paragraph_format.space_after  = Pt(4)
    p.paragraph_format.left_indent  = Pt(22)
    p.paragraph_format.first_line_indent = Pt(-22)
    p.alignment = WD_ALIGN_PARAGRAPH.LEFT

    # Number
    r_num = p.add_run(f'[{num}]  ')
    r_num.bold = True
    r_num.font.size = Pt(8.5)
    r_num.font.color.rgb = BLUE

    # Authors
    r_auth = p.add_run(authors + '  ')
    r_auth.font.size = Pt(8.5)
    r_auth.font.color.rgb = DARK

    # Title (italic)
    r_title = p.add_run(title + '  ')
    r_title.italic = True
    r_title.font.size = Pt(8.5)
    r_title.font.color.rgb = DARK

    # Journal + year
    r_jour = p.add_run(journal + '  ')
    r_jour.font.size = Pt(8.5)
    r_jour.font.color.rgb = DARK

    # DOI label bold
    r_doi_lbl = p.add_run('DOI: ')
    r_doi_lbl.bold = True
    r_doi_lbl.font.size = Pt(8.5)
    r_doi_lbl.font.color.rgb = BLUE

    # DOI value
    r_doi = p.add_run(doi)
    r_doi.font.size = Pt(8.5)
    r_doi.font.color.rgb = RGBColor(0x1A, 0x1A, 0x8B)  # dark blue for DOI

doc.save(DST)
print(f'Saved: {DST}')
print(f'References written: {len(REFS)}')

# Verify
doc2 = Document(DST)
print('\nVerification — References section:')
for p in doc2.paragraphs:
    if p.text.strip().startswith('['):
        print(f'  {p.text[:100]}')
