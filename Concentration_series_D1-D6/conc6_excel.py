import pandas as pd, numpy as np
from openpyxl import load_workbook
from openpyxl.drawing.image import Image
from openpyxl.styles import Font, PatternFill, Alignment
traces,offs,rows,s1,s2,s3,s4,s5=pd.read_pickle('combined6.pkl')
D='/home/user/claude/Concentration_series_D1-D6/'; OUT=D+'PS_D1-D6_concentration_series.xlsx'
names=['CaF2 reference (1 mg/mL)','D1 1 mg/mL','D2 100 ug/mL','D3 10 ug/mL','D4 1 ug/mL','D5 0.5 ug/mL','D6 (conc. not given)']
labs=[t[0].replace('$_2$','2') for t in traces]; pk={n:[] for n in names}
for lab,p in rows: pk[names[labs.index(lab)]].append(round(float(p),1))
L=max(map(len,pk.values())); peaks=pd.DataFrame({n:v+['']*(L-len(v)) for n,v in pk.items()})
x=traces[2][1]; sp={'Raman shift (cm-1)':x}
ns=[len(s1),len(s2),len(s3),len(s4),len(s5)]
sp_dummy=0
for k in range(1,6): sp[f'{names[k]} mean of {ns[k-1]}']=np.interp(x,traces[k][1],traces[k][2])
if False: sp['_']=np.interp(x,traces[k][1],traces[k][2])
sp['D6 mean of 2 point spectra']=np.interp(x,traces[6][1],traces[6][2])
sp['CaF2 reference']=np.interp(x,traces[0][1],traces[0][2])
j=lambda s: ', '.join(str(i+1) for i in s)
info=pd.DataFrame({'Sample':names,'ug/mL':[1000,1000,100,10,1,0.5,'not given'],
 'Particles/mL (20 nm PS)':['2.3e14','2.3e14','2.3e13','2.3e12','2.3e11','1.1e11','-'],'Molar (particles)':['380 nM','380 nM','38 nM','3.8 nM','0.38 nM','0.19 nM','-'],
 'Spectra averaged':['1 (reference)',f'{ns[0]} of 25',f'{ns[1]} of 25',f'{ns[2]} of 25',f'{ns[3]} of 25',f'{ns[4]} of 20','2 point spectra (s1, s2)'],
 'Matched pixels (1-based)':['-',j(s1),j(s2),j(s3),j(s4),j(s5),'none - neither spectrum matches CaF2 (no PS detected)']})
method=pd.DataFrame({'Step':['Pre-processing','Match criteria','Mean','D4 concentration','Particle numbers'],'Description':[
 'Cosmic-ray spikes removed (median-filter test) before matching and before averaging; CaF2 reference interpolated onto each map axis.',
 'Each spectrum: 1001 cm-1 band centre 1000.4-1003.4 cm-1, FWHM 3-9 cm-1, F-test p<0.001, and 985-1045 cm-1 shape match to the CaF2 reference with z >= 2. Same criteria for all maps.',
 'Mean of all matching spectra per sample (counts, spike-removed). D1 = map1_d1.',
 'D4 assumed 1 ug/mL - please confirm. D5 = 0.5 ug/mL (20-point map; point spectra S1, S2 checked: no CaF2 match). D6: 2 point spectra, no CaF2 match; concentration not given. D6 s1 is identical to the earlier file s1-PolyS-D5.',
 '20 nm spheres, PS density 1.05 g/cm3.']})
d6=pd.DataFrame({'spectrum':['D6 s1','D6 s2'],'match':['no','no'],'A_1001':[7.98,4.62],'center_1001':[1002.12,1001.81],'fwhm_1001':[2.5,2.5],'p_1001':[0.005,0.219],'z_template':[1.12,0.65]})
scores={}
for k,f in [('D1','S5_final.pkl'),('D2','S2_final.pkl'),('D3','S3_final.pkl'),('D4','S4_final.pkl'),('D5','S6_final.pkl')]:
    d=pd.read_pickle(f).sort_values('pixel').copy(); d.insert(0,'pixel # (1-based)',d.pixel+1)
    scores[k]=d[['pixel # (1-based)','match','A_1001','center_1001','fwhm_1001','p_1001','z_template']]
with pd.ExcelWriter(OUT,engine='openpyxl') as w:
    info.to_excel(w,sheet_name='Samples',index=False); method.to_excel(w,sheet_name='Method',index=False)
    peaks.to_excel(w,sheet_name='Peak_positions',index=False); pd.DataFrame(sp).to_excel(w,sheet_name='Mean_spectra',index=False)
    for k,d in scores.items(): d.to_excel(w,sheet_name='Scores '+k,index=False)
    d6.to_excel(w,sheet_name='Scores D6',index=False)
wb=load_workbook(OUT)
for ws in wb:
    for c in ws[1]: c.font=Font(bold=True,color='FFFFFF'); c.fill=PatternFill('solid',fgColor='305496'); c.alignment=Alignment(wrap_text=True)
    for col in ws.columns: ws.column_dimensions[col[0].column_letter].width=20
wb['Method'].column_dimensions['B'].width=120; wb['Samples'].column_dimensions['F'].width=50
g=PatternFill('solid',fgColor='C6EFCE')
for n in wb.sheetnames:
    if n.startswith('Scores'):
        for r in wb[n].iter_rows(min_row=2):
            if r[1].value=='MATCH':
                for c in r: c.fill=g
ws=wb.create_sheet('Figures')
for f,cell in [('PS_D1-D6_concentration_series.png','A1'),('PS_D1-D6_paper_style.png','N1')]:
    im=Image(D+f); im.width,im.height=im.width*.33,im.height*.33; ws.add_image(im,cell)
wb.save(OUT); print(info.to_string()); print(peaks.to_string())
