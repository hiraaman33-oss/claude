import pandas as pd, numpy as np
from openpyxl import load_workbook
from openpyxl.drawing.image import Image
from openpyxl.styles import Font, PatternFill, Alignment
traces,offs,rows,s1,s2,s3=pd.read_pickle('combined.pkl')
ranks={k:pd.read_pickle(f) for k,f in [('1 mg/mL','S1_final.pkl'),('100 ug/mL','S2_final.pkl'),('10 ug/mL','S3_final.pkl')]}
D='/home/user/claude/Concentration_series/'; OUT=D+'PS_concentration_series.xlsx'
names=['CaF2 reference (1 mg/mL)','1 mg/mL AgSiNW','100 ug/mL AgSiNW','10 ug/mL AgSiNW']
labs=[t[0].replace('$_2$','2') for t in traces]; pk={n:[] for n in names}
for lab,p in rows: pk[names[labs.index(lab)]].append(round(float(p),1))
L=max(map(len,pk.values())); peaks=pd.DataFrame({n:v+['']*(L-len(v)) for n,v in pk.items()})
x=traces[2][1]; sp={'Raman shift (cm-1)':x}
for k,(nm,sel) in enumerate(zip(names[1:],[s1,s2,s3]),start=1):
    xx=traces[k][1]; _,_,I,_=None,None,None,None
    sp[nm+f' mean of {len(sel)}']=np.interp(x,xx,traces[k][2])
sp['CaF2 reference']=np.interp(x,traces[0][1],traces[0][2])
info=pd.DataFrame({'Sample':names,'ug/mL':[1000,1000,100,10],'Particles/mL (20 nm PS)':['2.3e14','2.3e14','2.3e13','2.3e12'],
 'Molar (particles)':['380 nM','380 nM','38 nM','3.8 nM'],
 'Spectra averaged':['1 (reference)',f'{len(s1)} of 25',f'{len(s2)} of 25',f'{len(s3)} of 25'],
 'Pixels used (1-based)':['-',', '.join(str(i+1) for i in s1),', '.join(str(i+1) for i in s2),', '.join(str(i+1) for i in s3)],
 'Strict matches in whole map':['-',f"{(ranks['1 mg/mL'].match=='MATCH').sum()}/25",f"{(ranks['100 ug/mL'].match=='MATCH').sum()}/25",f"{(ranks['10 ug/mL'].match=='MATCH').sum()}/25"]})
method=pd.DataFrame({'Step':['Pre-processing','Match criteria','Mean','Robustness check'],'Description':[
 'Cosmic-ray spikes removed (median-filter test); reference interpolated onto each map axis.',
 'For every one of the 25 spectra per map: 1001 cm-1 band centre 1000.4-1003.4 cm-1, FWHM 3-9 cm-1, F-test p<0.001, and 985-1045 cm-1 shape match to the CaF2 reference with z >= 2. Same criteria for all maps.',
 'Mean of all spectra that pass, per map (raw counts).',
 'Matches were identical with and without spike removal.']})
with pd.ExcelWriter(OUT,engine='openpyxl') as w:
    info.to_excel(w,sheet_name='Samples',index=False); method.to_excel(w,sheet_name='Method',index=False)
    peaks.to_excel(w,sheet_name='Peak_positions',index=False); pd.DataFrame(sp).to_excel(w,sheet_name='Mean_spectra',index=False)
    for k,d in ranks.items():
        d=d.sort_values('pixel').copy(); d.insert(0,'pixel # (1-based)',d.pixel+1)
        d[['pixel # (1-based)','match','A_1001','center_1001','fwhm_1001','p_1001','z_template']].to_excel(w,sheet_name=('Scores '+k).replace('/','per'),index=False)
wb=load_workbook(OUT)
for ws in wb:
    for c in ws[1]: c.font=Font(bold=True,color='FFFFFF'); c.fill=PatternFill('solid',fgColor='305496'); c.alignment=Alignment(wrap_text=True)
    for col in ws.columns: ws.column_dimensions[col[0].column_letter].width=20
wb['Method'].column_dimensions['B'].width=120
ws=wb.create_sheet('Figures')
for f,cell in [('PS_concentration_paper_style.png','A1'),('PS_concentration_series.png','A50')]:
    im=Image(D+f); im.width,im.height=im.width*.38,im.height*.38; ws.add_image(im,cell)
wb.save(OUT); print(info.to_string())
