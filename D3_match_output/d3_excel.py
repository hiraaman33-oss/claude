exec(open('common.py').read())
import pandas as pd, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from openpyxl import load_workbook
from openpyxl.drawing.image import Image
from openpyxl.styles import Font, PatternFill, Alignment
df,sel,t,mean=pd.read_pickle('D3_res.pkl'); rest=[i for i in range(25) if i not in sel]
OUTD='/home/user/claude/D3_match_output'
# match map
fig,ax=plt.subplots(1,2,figsize=(10,4.3))
ok=df.center_1001.between(1000.4,1003.4)&df.fwhm_1001.between(3,9)&(df.p_1001<1e-3)
for a,V,lab,cm in [(ax[0],np.where(ok,df.A_1001,0),'1001 cm$^{-1}$ amplitude (cnt)','viridis'),(ax[1],df.z_template.values,'template match z-score','magma')]:
    im=a.imshow(np.array(V,float).reshape(5,5),cmap=cm); plt.colorbar(im,ax=a,label=lab)
    for i in range(25):
        a.text(i%5,i//5,f'{i+1}',ha='center',va='center',color='w',fontsize=8,fontweight='bold' if i in sel else 'normal')
        if i in sel: a.add_patch(plt.Rectangle((i%5-.5,i//5-.5),1,1,fill=False,ec='red',lw=2))
    a.set_xlabel('column'); a.set_ylabel('row')
ax[0].set_title('Valid PS 1001 band amplitude'); ax[1].set_title('Match score vs CaF2 reference')
fig.suptitle('D3 map – matched pixels outlined (numbers = pixel #, 1-based)'); plt.tight_layout(); plt.savefig(f'{OUTD}/D3_match_map.png',dpi=130,bbox_inches='tight')
M=I[sel]; sd=M.std(0,ddof=1)
OUT=f'{OUTD}/D3_CaF2_matched_spectra.xlsx'
readme=pd.DataFrame({'Item':['Map file','Reference','Map size','Criteria','Result','Pixel numbering'],'Description':[
 'm1-D3_PS-AgSiNW_532nm_1800gr_BC200_50XLF_1s_2a_10_3ul_drop-center.l6m',
 's9-PolyS-1ml-dropCaF2-EDGE (1 mg/mL PS on CaF2)',
 f'5 x 5 = 25 spectra, {len(x)} points, {x[0]:.0f}-{x[-1]:.0f} cm-1 (no data 1800-2600 cm-1)',
 'Same as 1 mg/mL map: 1001 band centre 1000.4-1003.4 cm-1, FWHM 3-9 cm-1, F-test p<0.001, template (985-1045 cm-1) z>=2',
 f'{len(sel)} of 25 matched: pixel # (1-based) '+', '.join(str(i+1) for i in sel),
 '1-based, row-major (pixel # = row*5 + col + 1), same convention as your slide']})
sc=df.copy(); sc.insert(1,'pixel # (1-based)',sc.pixel+1)
sc=sc[['pixel # (1-based)','row','col','match','A_1001','center_1001','fwhm_1001','sigA_1001','F_1001','p_1001','r_template','z_template','r_fingerprint','r_full']]
meandf=pd.DataFrame({'Raman shift (cm-1)':x,f'Mean of {len(sel)} (raw cnt)':mean,'SD':sd,'SEM':sd/np.sqrt(len(sel)),
   'Mean baseline-corrected':proc(mean),'Mean of non-matched':I[rest].mean(0),'CaF2 reference (interp.)':yref})
matched=pd.DataFrame({'Raman shift (cm-1)':x,**{f'#{i+1} (r{i//5}c{i%5})':I[i] for i in sel}})
allsp=pd.DataFrame({'Raman shift (cm-1)':x,**{f'#{i+1}':I[i] for i in range(25)}})
with pd.ExcelWriter(OUT,engine='openpyxl') as w:
    readme.to_excel(w,sheet_name='README',index=False); t.to_excel(w,sheet_name='Peak_positions',index=False)
    sc.to_excel(w,sheet_name='Match_scores',index=False); meandf.to_excel(w,sheet_name='Mean_spectrum',index=False)
    matched.to_excel(w,sheet_name='Matched_spectra',index=False); allsp.to_excel(w,sheet_name='All_25_spectra',index=False)
wb=load_workbook(OUT)
for ws in wb:
    for c in ws[1]: c.font=Font(bold=True,color='FFFFFF'); c.fill=PatternFill('solid',fgColor='305496'); c.alignment=Alignment(wrap_text=True)
    for col in ws.columns: ws.column_dimensions[col[0].column_letter].width=16
wb['README'].column_dimensions['B'].width=120; wb['Peak_positions'].column_dimensions['A'].width=24
g=PatternFill('solid',fgColor='C6EFCE')
for r in wb['Match_scores'].iter_rows(min_row=2):
    if r[3].value=='MATCH':
        for c in r: c.fill=g
ws=wb.create_sheet('Plots')
for f,cell in [(f'D3_final_spectrum_{len(sel)}matched.png','A1'),('D3_match_map.png','A55')]:
    im=Image(f'{OUTD}/{f}'); im.width,im.height=im.width*0.45,im.height*0.45; ws.add_image(im,cell)
wb.save(OUT); print('saved')
