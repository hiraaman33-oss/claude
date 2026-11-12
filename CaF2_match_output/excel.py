exec(open('common.py').read())
import pandas as pd
from openpyxl import load_workbook
from openpyxl.drawing.image import Image
from openpyxl.styles import Font, PatternFill, Alignment
df=pd.read_pickle('final.pkl'); sel=list(np.load('sel.npy')); rest=[i for i in range(25) if i not in sel]
M=I[sel]; mean=M.mean(0); sd=M.std(0,ddof=1); sem=sd/np.sqrt(len(sel))
OUT='/home/user/claude/CaF2_match_output/CaF2_matched_spectra_1mgml_map.xlsx'
xaxis=np.frombuffer(d[542172:542192],'<f4')
readme=pd.DataFrame({'Item':[
 'Map file','Reference file','Map size','Spectral axis','Reference','Goal',
 'Step 1','Step 2','Step 3','Step 4','Step 5','Match criteria','Result','Note'],
 'Description':[
 'm1-1MGML_PS-AgSiNW_532nm_1800gr_BC200_50XLF_1s_2a_10_1ul_drop-center.l6m (LabSpec6 binary, decoded in Python)',
 's9-PolyS-1ml-dropCaF2-EDGE-532nm-1800gr-100-100x-BC50-5s-3a.txt',
 '5 x 5 = 25 spectra, X step ~4 µm (X = '+', '.join(f'{v:.2f}' for v in xaxis)+' µm); pixel index = row*5 + col',
 f'{len(x)} points, {x[0]:.1f}-{x[-1]:.1f} cm-1, gap {x[GAP-1]:.0f}-{x[GAP]:.0f} cm-1 (no data in either file)',
 'PS on CaF2: 1001.9 cm-1 ring-breathing (FWHM 4.9), 1032, 621, 1602, 2904/3054 cm-1',
 'Find map spectra that match the CaF2 reference and average them',
 'Reference interpolated onto map axis; Savitzky-Golay smoothing (9 pts, 2nd order) + asymmetric-least-squares baseline per spectral window',
 'Gaussian + quadratic-baseline fit of the 1001 cm-1 band (986-1020 cm-1) on raw counts; F-test vs baseline-only model',
 'Template matching: correlation of 985-1045 cm-1 window (1001 + 1031 shape) with reference; z-score vs null distribution of the same template slid over 1060-1550 cm-1 of the same spectrum',
 'Full-spectrum and fingerprint correlations also computed (excluding Si 520 and ~970 cm-1 bands) - low for all pixels because the map is dominated by Si/SERS bands',
 'Mean, SD and SEM computed from RAW counts of matched spectra',
 'center 1000.4-1003.4 cm-1 AND FWHM 3-9 cm-1 AND F-test p < 0.001 AND template z >= 2.0',
 f'{len(sel)} of 25 spectra matched: pixels {sel}',
 'Map spectra are 1 s SERS spectra on Ag-SiNW, so the PS bands are weak (~15-25 cnt vs ~5 cnt noise); the AgSiNW substrate bands (Si 520, ~970, broad 1100-1700) remain in the mean']})
scores=df.copy(); scores['X_um']=[xaxis[c] for c in scores.col]
scores=scores[['pixel','row','col','X_um','match','A_1001','center_1001','fwhm_1001','sigA_1001','F_1001','p_1001','r_template','z_template','r_fingerprint','r_full']].sort_values('pixel')
meanbc=proc(mean)
mean_df=pd.DataFrame({'Raman shift (cm-1)':x,'Mean (matched, raw cnt)':mean,'SD':sd,'SEM':sem,'Mean - SD':mean-sd,'Mean + SD':mean+sd,
    'Mean baseline-corrected':meanbc,'Mean of non-matched (raw)':I[rest].mean(0),'CaF2 reference (interp.)':yref,'CaF2 reference baseline-corrected':R})
matched=pd.DataFrame({'Raman shift (cm-1)':x,**{f'pixel_{i} (r{i//5}c{i%5})':I[i] for i in sel}})
allsp=pd.DataFrame({'Raman shift (cm-1)':x,**{f'pixel_{i} (r{i//5}c{i%5})':I[i] for i in range(25)}})
refdf=pd.DataFrame({'Raman shift (cm-1)':ref[:,0],'Intensity':ref[:,1]})
with pd.ExcelWriter(OUT,engine='openpyxl') as w:
    readme.to_excel(w,sheet_name='README',index=False); scores.to_excel(w,sheet_name='Match_scores',index=False)
    mean_df.to_excel(w,sheet_name='Mean_spectrum',index=False); matched.to_excel(w,sheet_name='Matched_spectra',index=False)
    allsp.to_excel(w,sheet_name='All_25_map_spectra',index=False); refdf.to_excel(w,sheet_name='CaF2_reference_raw',index=False)
wb=load_workbook(OUT)
for ws in wb:
    for c in ws[1]: c.font=Font(bold=True,color='FFFFFF'); c.fill=PatternFill('solid',fgColor='305496'); c.alignment=Alignment(wrap_text=True,vertical='center')
    ws.freeze_panes='B2'
    for col in ws.columns: ws.column_dimensions[col[0].column_letter].width=16
ws=wb['README']; ws.column_dimensions['A'].width=18; ws.column_dimensions['B'].width=130; ws.freeze_panes='A2'
for r in ws.iter_rows(min_row=2):
    r[1].alignment=Alignment(wrap_text=True,vertical='top')
ws=wb['Match_scores']; green=PatternFill('solid',fgColor='C6EFCE')
for r in ws.iter_rows(min_row=2):
    if r[4].value=='MATCH':
        for c in r: c.fill=green
    for c in r[5:]: c.number_format='0.000'
ws.column_dimensions['K'].width=12
for c in ws['K'][1:]: c.number_format='0.00E+00'
ws=wb.create_sheet('Plots'); ws.add_image(Image('/home/user/claude/CaF2_match_output/mean_spectrum.png'),'A1'); ws.add_image(Image('/home/user/claude/CaF2_match_output/match_map.png'),'A75')
wb['Plots']['A1'].value=None
wb.save(OUT); print('saved',OUT)
