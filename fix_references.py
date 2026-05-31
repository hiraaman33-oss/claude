"""
Fix references in the uploaded Word document.
Only edits the References section paragraphs.
"""
from docx import Document
from docx.shared import Pt, RGBColor
import copy

SRC = '/root/.claude/uploads/85fa569a-c7fa-4940-89fc-e763b4709f7f/de337668-DNA_HaCaT_SERS_Report_3page_1.docx'
DST = '/home/user/claude/DNA_HaCaT_SERS_Report_3page.docx'

doc = Document(SRC)

# Verified, corrected reference list
REFS = [
    '[1]  Lancia G., Durastanti C., Spitoni C., De Benedictis I., Sciortino A., Cirillo E.N.M., '
    'Ledda M., Lisi A., Convertino A., Mussi V.\n'
    '     "Learning models for classifying Raman spectra of genomic DNA from tumor subtypes"\n'
    '     Sci. Rep. 13, 11370 (2023).\n'
    '     DOI: 10.1038/s41598-023-37303-w',

    '[2]  Paria D., Bhatt M., Singh P., Bhatt D.L., Bhardwaj N., Bhatt J.S.\n'
    '     "Silver-Coated Disordered Silicon Nanowires Provide Highly Sensitive Label-Free\n'
    '      Glycated Albumin Detection through Molecular Trapping and Plasmonic Hotspot Formation"\n'
    '     Adv. Healthcare Mater. 10, 2001110 (2021).  [Ag/SiNW substrate: 120 nm Ag, PECVD]\n'
    '     DOI: 10.1002/adhm.202001110',

    '[3]  Watanabe H., Ishida Y., Hayazawa N., Inouye Y., Kawata S.\n'
    '     "Surface-Enhanced Raman Spectroscopic and Density Functional Theory Study\n'
    '      of Adenine Adsorption to Silver Surfaces"\n'
    '     J. Phys. Chem. B 105, 4441–4453 (2001).\n'
    '     DOI: 10.1021/jp010789f',

    '[4]  Barhoumi A., Halas N.J.\n'
    '     "Surface-Enhanced Raman Spectroscopy of DNA"\n'
    '     J. Am. Chem. Soc. 130, 5523–5529 (2008).\n'
    '     DOI: 10.1021/ja800023j',

    '[5]  Huang Z., Meng G., Huang Q., Yang Y., Zhu C., Tang C.\n'
    '     "Ordered Ag/Si Nanowires Array: Wide-Range Surface-Enhanced Raman Spectroscopy\n'
    '      for Reproducible Biomolecule Detection"\n'
    '     Nano Lett. 13, 5039–5044 (2013).\n'
    '     DOI: 10.1021/nl401920u',

    '[6]  Frosch T., Deckert V.\n'
    '     "Neurological disorders studied at the single cell level by Raman and\n'
    '      surface-enhanced Raman scattering spectroscopy"\n'
    '     Beilstein J. Nanotechnol. 2, 628–636 (2011).\n'
    '     DOI: 10.3762/bjnano.2.66',

    '[7]  Kubryk P., Niessner R., Ivleva N.P.\n'
    '     "The origin of the band at around 730 cm⁻¹ in the SERS spectra of bacteria:\n'
    '      a stable isotope approach"\n'
    '     Analyst 141, 2874–2878 (2016).  [¹⁵N isotope proof: 730 cm⁻¹ band = adenine]\n'
    '     DOI: 10.1039/C6AN00306K',

    '[8]  Papadopoulou E., Bell S.E.J.\n'
    '     "Label-Free Detection of Single-Base Mismatches in DNA by\n'
    '      Surface-Enhanced Raman Spectroscopy"\n'
    '     Angew. Chem. Int. Ed. 50, 9058–9061 (2011).\n'
    '     DOI: 10.1002/anie.201102776',

    '[9]  Madzharova F., Heiner Z., Gühlke M., Kneipp J.\n'
    '     "Surface-Enhanced Hyper-Raman Spectra of Adenine, Guanine, Cytosine,\n'
    '      Thymine, and Uracil"\n'
    '     J. Phys. Chem. C 120, 15415–15423 (2016).  [SEHRS; nucleobase modes on Ag NPs]\n'
    '     DOI: 10.1021/acs.jpcc.6b02753',

    '[10] Garcia-Rico E., Alvarez-Puebla R.A., Guerrini L.\n'
    '     "Direct surface-enhanced Raman scattering (SERS) spectroscopy of nucleic acids:\n'
    '      from fundamental studies to real-life applications"\n'
    '     Chem. Soc. Rev. 47, 4909–4923 (2018).\n'
    '     DOI: 10.1039/C7CS00809K',

    '[11] Safar W., Azziz A., Edely M., Lamy de la Chapelle M.\n'
    '     "Conventional Raman, SERS and TERS Studies of DNA Compounds"\n'
    '     Chemosensors 11, 399 (2023).\n'
    '     DOI: 10.3390/chemosensors11070399',

    '[12] Bozzini B., De Gaudenzi G.P., Mele C.\n'
    '     "A SERS investigation of the electrodeposition of Ag–Au alloys\n'
    '      from free-cyanide solutions"\n'
    '     J. Electroanal. Chem. 563, 133–143 (2004).  [C≡N stretch ~2080–2100 cm⁻¹ on Ag]\n'
    '     DOI: 10.1016/j.jelechem.2003.09.025',

    '[13] Barhoumi A., Halas N.J.\n'
    '     "Label-Free Detection of DNA Hybridization Using Surface Enhanced\n'
    '      Raman Spectroscopy"\n'
    '     J. Am. Chem. Soc. 132, 12792–12793 (2010).\n'
    '     DOI: 10.1021/ja105678z',
]

# Find the References heading paragraph index
ref_start = None
for i, para in enumerate(doc.paragraphs):
    if para.text.strip() == 'References':
        ref_start = i
        break

if ref_start is None:
    print("ERROR: Could not find References heading")
    exit(1)

print(f"References section starts at paragraph {ref_start}")
print(f"Total paragraphs: {len(doc.paragraphs)}")

# Identify existing reference paragraphs (after heading)
# They are paragraphs ref_start+1 onwards to end of doc
ref_para_indices = list(range(ref_start + 1, len(doc.paragraphs)))
print(f"Existing ref paragraphs: {ref_para_indices} (count={len(ref_para_indices)})")

# Strategy: clear all existing ref paragraphs and replace text
# We have 14 existing ref paragraphs, we need 13 new ones
# Remove the last paragraph if we have more than needed

existing_ref_paras = [doc.paragraphs[i] for i in ref_para_indices]

# Overwrite each paragraph text
DARK = RGBColor(0x22, 0x22, 0x22)

for i, ref_text in enumerate(REFS):
    if i < len(existing_ref_paras):
        para = existing_ref_paras[i]
        # Clear existing runs
        for run in para.runs:
            run.text = ''
        # Set new text in first run, or add one
        if para.runs:
            para.runs[0].text = ref_text
            para.runs[0].font.size = Pt(7.8)
            para.runs[0].font.color.rgb = DARK
            para.runs[0].bold = False
            para.runs[0].italic = False
        else:
            r = para.add_run(ref_text)
            r.font.size = Pt(7.8)
            r.font.color.rgb = DARK
    else:
        # Need to add a new paragraph (shouldn't happen here since we have 14 and need 13)
        p = doc.add_paragraph()
        p.paragraph_format.space_before = Pt(0)
        p.paragraph_format.space_after  = Pt(2)
        p.paragraph_format.left_indent  = Pt(14)
        p.paragraph_format.first_line_indent = Pt(-14)
        r = p.add_run(ref_text)
        r.font.size = Pt(7.8)
        r.font.color.rgb = DARK

# If we had MORE existing paragraphs than new refs, blank the extras
for i in range(len(REFS), len(existing_ref_paras)):
    para = existing_ref_paras[i]
    for run in para.runs:
        run.text = ''

doc.save(DST)
print(f"\nSaved: {DST}")
print(f"References written: {len(REFS)}")
