"""Generate PLSDA_SVM_4Class_Book2.docx from PLSDA_SVM_4Class_Book2.py"""
from docx import Document
from docx.shared import Pt, RGBColor, Inches
from docx.enum.text import WD_ALIGN_PARAGRAPH
import re

SRC = "/home/user/claude/PLSDA_SVM_4Class_Book2.py"
DST = "/home/user/claude/PLSDA_SVM_4Class_Book2.docx"

doc = Document()

# ── title block ─────────────────────────────────────────────────────────────
title = doc.add_heading("PLS-DA + SVM-RBF  |  4-Class  |  Book2.xlsx  |  68/32 Stratified Split", 0)
title.runs[0].font.color.rgb = RGBColor(0x1F, 0x49, 0x7D)

sub = doc.add_paragraph(
    "Raman Spectroscopy — 4-Class SSY Classification\n"
    "Train: 68 spectra (all 4 classes)  |  Test: graphene only, n=16\n"
    "PLS-DA + SVM-RBF  |  5-fold CV + LOO-CV  |  VIP loop {1000, 1500, 1800}\n"
    "File: Book2.xlsx  |  Sheet: 4 - Normalised  |  450-1800 cm⁻¹\n"
    "Path: C:/Users/Hira Aman/Desktop/PROF_DOMENICO'S"
)
sub.alignment = WD_ALIGN_PARAGRAPH.CENTER
for run in sub.runs:
    run.font.size = Pt(10)
    run.font.color.rgb = RGBColor(0x44, 0x44, 0x44)

doc.add_paragraph()

# ── how-to-run box ───────────────────────────────────────────────────────────
doc.add_heading("How to Run", 2)
htr = doc.add_paragraph()
htr.add_run("1. Install dependencies (one-time):\n").bold = True
htr.add_run("   pip install pandas openpyxl scikit-learn scipy matplotlib\n\n")
htr.add_run("2. Save the script as: PLSDA_SVM_4Class_Book2.py\n").bold = True
htr.add_run(
    "   Place it anywhere convenient (e.g. Desktop).\n\n"
)
htr.add_run("3. Run:\n").bold = True
htr.add_run("   python PLSDA_SVM_4Class_Book2.py\n\n")
htr.add_run("Outputs per VIP set (3 folders):\n").bold = True
htr.add_run(
    "   PLSDA_SVM_4Class_VIP1000\\   PLSDA_SVM_4Class_VIP1500\\   PLSDA_SVM_4Class_VIP1800\\\n"
    "   Each folder: 18 PNG figures + 1 Excel workbook (13 sheets)"
)
for run in htr.runs:
    run.font.size = Pt(10)

doc.add_paragraph()

# ── output file table ────────────────────────────────────────────────────────
doc.add_heading("Output Files (per VIP folder)", 2)
table = doc.add_table(rows=1, cols=2)
table.style = "Light List Accent 1"
hdr_cells = table.rows[0].cells
hdr_cells[0].text = "Filename"
hdr_cells[1].text = "Description"
for cell in hdr_cells:
    for run in cell.paragraphs[0].runs:
        run.bold = True

rows = [
    ("01_PLSDA_Scores_LV1_LV2.png",       "PLS-DA score plot LV1×LV2 (full model + 5-fold CV overlay)"),
    ("02_PLSDA_Scores_LV1_LV3.png",       "PLS-DA score plot LV1×LV3 (full model + LOO-CV overlay)"),
    ("03_PLSDA_Scores_3D.png",            "PLS-DA 3D score plot (LV1/LV2/LV3)"),
    ("04_PLSDA_Confusion_5fold_Train.png","PLS-DA confusion matrix — 5-fold CV (train)"),
    ("05_PLSDA_Confusion_LOO_Train.png",  "PLS-DA confusion matrix — LOO-CV (train)"),
    ("06_PLSDA_Confusion_Test_Graphene.png","PLS-DA confusion matrix — graphene test (n=16)"),
    ("07_PLSDA_VIP_Scores.png",           "VIP scores with VIP>1 threshold and top-N selection"),
    ("08_PLSDA_Loadings.png",             "PLS-DA loadings LV1/LV2/LV3"),
    ("09_PLSDA_Permutation.png",          "Permutation test (999 perm., train only)"),
    ("10_SVM_Cluster_5fold_Train.png",    "SVM-RBF cluster plot — 5-fold CV (PCA space, train)"),
    ("11_SVM_Cluster_LOO_Train.png",      "SVM-RBF cluster plot — LOO-CV (PCA space, train)"),
    ("12_SVM_Cluster_Test_Graphene.png",  "SVM-RBF cluster plot — graphene test (PCA space)"),
    ("13_SVM_Confusion_5fold_Train.png",  "SVM-RBF confusion matrix — 5-fold CV (train)"),
    ("14_SVM_Confusion_LOO_Train.png",    "SVM-RBF confusion matrix — LOO-CV (train)"),
    ("15_SVM_Confusion_Test_Graphene.png","SVM-RBF confusion matrix — graphene test (n=16)"),
    ("16_Accuracy_Comparison.png",        "Overall accuracy bar chart: PLS-DA vs SVM-RBF × 3 methods"),
    ("17_Recall_Comparison.png",          "Per-class recall (3-panel: 5-fold / LOO / Test-Graphene)"),
    ("18_Mean_Spectra.png",               "Mean Raman spectra ± 1 SD (all 100 spectra)"),
    ("PLSDA_SVM_4Class_VIP<N>_Results.xlsx", "Excel workbook — 13 sheets (summary, CMs, VIP, perm., ...)"),
]
for fn, desc in rows:
    row = table.add_row().cells
    row[0].text = fn
    row[1].text = desc
    for cell in row:
        for run in cell.paragraphs[0].runs:
            run.font.size = Pt(9)

doc.add_paragraph()

# ── Python source code ───────────────────────────────────────────────────────
doc.add_heading("Python Script: PLSDA_SVM_4Class_Book2.py", 1)

with open(SRC, encoding="utf-8") as f:
    code = f.read()

# split into lines, add as one paragraph with monospaced font
code_para = doc.add_paragraph()
code_para.paragraph_format.space_before = Pt(0)
code_para.paragraph_format.space_after  = Pt(0)

for line in code.split("\n"):
    run = code_para.add_run(line + "\n")
    run.font.name = "Courier New"
    run.font.size = Pt(8)

# ── save ─────────────────────────────────────────────────────────────────────
doc.save(DST)
print(f"Saved: {DST}")
