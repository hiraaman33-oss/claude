from docx import Document
from docx.shared import Pt, RGBColor, Inches, Cm
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml.ns import qn
from docx.oxml import OxmlElement

# Read the full Python code
with open('/home/user/claude/raman_analysis.py', 'r', encoding='utf-8') as f:
    code = f.read()

doc = Document()

# Page margins
sec = doc.sections[0]
sec.left_margin   = Cm(2.0)
sec.right_margin  = Cm(2.0)
sec.top_margin    = Cm(2.0)
sec.bottom_margin = Cm(2.0)

# Title
t = doc.add_heading('Raman Map Analysis Pipeline  v3', level=0)
t.alignment = WD_ALIGN_PARAGRAPH.CENTER

# Subtitle
sub = doc.add_paragraph('DNA + HACAT cells on AgSiNW  |  532 nm  |  Single-spectrum comparison added')
sub.alignment = WD_ALIGN_PARAGRAPH.CENTER
sub.runs[0].font.color.rgb = RGBColor(0x55, 0x55, 0x55)
sub.runs[0].font.size = Pt(11)

doc.add_paragraph()

# Quick start
doc.add_heading('Quick Start', level=2)
qs = [
    '1.  Install dependencies (once, in PyCharm terminal):',
    '       pip install numpy matplotlib PyWavelets scipy',
    '2.  Open raman_analysis.py in PyCharm.',
    '3.  Verify the two file paths at the top of the file:',
    '       FILE_PATH     -> your .l6m map file',
    '       FILE_PATH_L6S -> your .l6s single-spectrum file',
    '4.  Press  Run.  Four files are written to the same folder:',
    '       raman_pipeline_output.png   (6-panel preprocessing figure)',
    '       raman_final_zoomed.png      (fingerprint + C-H stretch zoomed)',
    '       raman_comparison.png        (map vs single spectrum, peak labels)',
    '       raman_processed_spectrum.csv',
]
for line in qs:
    p = doc.add_paragraph(line)
    p.runs[0].font.name = 'Courier New'
    p.runs[0].font.size = Pt(9)
    p.paragraph_format.space_after = Pt(2)

doc.add_paragraph()

# What is new in v3
doc.add_heading('New in v3 (vs v2)', level=2)
news = [
    '* load_l6s()  binary parser for LabSpec 6 .l6s single-spectrum files.',
    '  Handles non-standard (byte +1) float32 alignment via a 4-offset scan.',
    '* preprocess_single()  applies the identical pipeline to the single spectrum.',
    '* plot_comparison()  two-panel publication figure:',
    '  - Left panel : both SNV spectra with vertical dotted lines at every peak.',
    '    Green = peak in BOTH; orange = map only; blue = single-spectrum only.',
    '  - Right panel: difference spectrum (map minus single) with shaded fill.',
]
for line in news:
    p = doc.add_paragraph(line)
    p.runs[0].font.size = Pt(10)
    p.paragraph_format.space_after = Pt(3)

doc.add_paragraph()

# Processing steps table
doc.add_heading('Processing Steps', level=2)
table = doc.add_table(rows=1, cols=3)
table.style = 'Table Grid'
hdr = table.rows[0].cells
hdr[0].text = 'Step'
hdr[1].text = 'Method'
hdr[2].text = 'Purpose'
for cell in hdr:
    for para in cell.paragraphs:
        for run in para.runs:
            run.font.bold = True
            run.font.size = Pt(9)

steps = [
    ('1. Load .l6m', 'Custom binary parser (_find_wn_axis)', 'Extract N spectra x M wavenumber points'),
    ('1b. Load .l6s', 'load_l6s() 4-offset float32 scan', 'Extract single spectrum (400-1800 cm-1)'),
    ('2. Spike removal', 'Modified Z-score on 2nd derivative', 'Remove cosmic rays per spectrum'),
    ('3. Average', 'np.mean across cleaned spectra', 'Single representative spectrum'),
    ('4. ALS baseline', 'Eilers & Boelens 2005, lam=1e6, p=0.005', 'Subtract fluorescence background'),
    ('5. Wavelet denoise', 'PyWavelets db8, L=5, soft threshold', 'Reduce detector noise'),
    ('6. Min-Max norm.', '(x - min) / (max - min)', 'Scale to [0, 1]'),
    ('7. SNV', '(x - mean) / std', 'Remove baseline offsets between samples'),
    ('8-10. Figures', 'matplotlib 180 dpi', '6-panel pipeline + zoomed + comparison'),
]
for step, method, purpose in steps:
    row = table.add_row().cells
    row[0].text = step
    row[1].text = method
    row[2].text = purpose
    for cell in row:
        for para in cell.paragraphs:
            for run in para.runs:
                run.font.size = Pt(9)

doc.add_paragraph()

# Peak assignments table
doc.add_heading('Known Peak Assignments (SERS, 532 nm)', level=2)
ptable = doc.add_table(rows=1, cols=3)
ptable.style = 'Table Grid'
ph = ptable.rows[0].cells
ph[0].text = 'Wavenumber (cm-1)'
ph[1].text = 'Assignment'
ph[2].text = 'Origin'
for cell in ph:
    for para in cell.paragraphs:
        for run in para.runs:
            run.font.bold = True
            run.font.size = Pt(9)

peaks = [
    (382,  'Ag-N stretch',       'Ag-amine bond (SERS enhancement)'),
    (521,  'Si phonon',          'Si substrate / Si nanowire'),
    (662,  'G ring breathing',   'Guanine (DNA)'),
    (785,  'DNA backbone',       'O-P-O stretch, phosphodiester'),
    (1000, 'Phe ring',           'Phenylalanine (protein, HACAT)'),
    (1090, 'PO4- sym. stretch',  'Phosphate backbone (DNA)'),
    (1175, 'C-H in-plane bend',  'Thymine / cytosine (DNA)'),
    (1340, 'G C-N stretch',      'Guanine (DNA)'),
    (1484, 'A/C C=N stretch',    'Adenine / cytosine (DNA)'),
    (1575, 'G/A ring stretch',   'Guanine / adenine (DNA)'),
    (2850, 'CH2 sym. stretch',   'Lipids / proteins (HACAT)'),
    (2930, 'CH2 asym. stretch',  'Lipids / proteins (HACAT)'),
    (3060, 'ArC-H stretch',      'Aromatic residues (protein)'),
    (3130, 'C-H stretch',        'DNA/protein -NH, -OH'),
]
for wn, asgn, orig in peaks:
    row = ptable.add_row().cells
    row[0].text = str(wn)
    row[1].text = asgn
    row[2].text = orig
    for cell in row:
        for para in cell.paragraphs:
            for run in para.runs:
                run.font.size = Pt(9)

doc.add_paragraph()

# Full code section
doc.add_heading('Complete Python Code  (raman_analysis.py)', level=2)
note = doc.add_paragraph(
    'Copy the entire block below into a new file called raman_analysis.py '
    'in your PyCharm project folder.  Do not modify indentation.'
)
note.runs[0].font.size = Pt(10)
note.runs[0].font.italic = True

doc.add_paragraph()

# Code block - monospace, grey shaded
code_para = doc.add_paragraph()
code_para.paragraph_format.space_before = Pt(0)
code_para.paragraph_format.space_after = Pt(0)

pPr = code_para._p.get_or_add_pPr()
shd = OxmlElement('w:shd')
shd.set(qn('w:val'), 'clear')
shd.set(qn('w:color'), 'auto')
shd.set(qn('w:fill'), 'F2F2F2')
pPr.append(shd)

run = code_para.add_run(code)
run.font.name = 'Courier New'
run.font.size = Pt(7.5)

doc.add_paragraph()

# Output files
doc.add_heading('Output Files', level=2)
ot = doc.add_table(rows=1, cols=2)
ot.style = 'Table Grid'
oh = ot.rows[0].cells
oh[0].text = 'File'
oh[1].text = 'Contents'
for cell in oh:
    for para in cell.paragraphs:
        for run in para.runs:
            run.font.bold = True
            run.font.size = Pt(9)
outputs = [
    ('raman_pipeline_output.png',    '6-panel figure: raw -> spike-removed -> baseline -> denoised -> normalised -> SNV'),
    ('raman_final_zoomed.png',       'Two-panel zoom: fingerprint (400-1800) + C-H stretch (2700-3200 cm-1)'),
    ('raman_comparison.png',         'Comparison: map average vs single .l6s spectrum, colour-coded peak dotted lines'),
    ('raman_processed_spectrum.csv', 'Numeric table: wn, raw, spike-removed, baseline-corrected, denoised, normalised, SNV'),
]
for fname, desc in outputs:
    row = ot.add_row().cells
    row[0].text = fname
    row[1].text = desc
    for cell in row:
        for para in cell.paragraphs:
            for run in para.runs:
                run.font.size = Pt(9)

doc.save('/home/user/claude/Raman_Analysis_Code_and_Explanation.docx')
print("Document saved.")
