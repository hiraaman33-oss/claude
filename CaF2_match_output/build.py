exec(open('common.py').read())
import pandas as pd, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
df=pd.read_pickle('final.pkl'); sel=df[df.match=='MATCH'].pixel.tolist(); rest=[i for i in range(25) if i not in sel]
print(sel)
M=I[sel]; mean=M.mean(0); sd=M.std(0,ddof=1); sem=sd/np.sqrt(len(sel))
meanbc=proc(mean); restmean=I[rest].mean(0)
def G(v):
    v=np.array(v,float).copy(); v=np.insert(v,GAP,np.nan); return v
xp=np.insert(x,GAP,np.nan)
# verify PS bands in the mean
for pos in [621,1001,1031,1155,1583,1602,3054]:
    m=(x>pos-20)&(x<pos+20); print(pos, round(x[m][np.argmax(meanbc[m])],1), round(meanbc[m].max(),1))
OUT='/home/user/claude/CaF2_match_output'; import os; os.makedirs(OUT,exist_ok=True)
fig=plt.figure(figsize=(11,11)); gs=fig.add_gridspec(3,2,width_ratios=[1400,500],wspace=0.04,hspace=0.35)
L1,R1=(400,1800),(2700,3200)
def broken(row,draw,ylabel,title=None,legend=True):
    aL=fig.add_subplot(gs[row,0]); aR=fig.add_subplot(gs[row,1],sharey=aL)
    for a in (aL,aR): draw(a)
    aL.set_xlim(*L1); aR.set_xlim(*R1); aL.set_xticks(range(400,1800,200)); aR.set_xticks(range(2800,3201,200))
    aL.spines['right'].set_visible(False); aR.spines['left'].set_visible(False)
    aR.tick_params(axis='y',left=False,labelleft=False)
    k=.012; kw=dict(color='k',clip_on=False,lw=1)
    aL.plot((1-k,1+k),(-k*1.5,k*1.5),transform=aL.transAxes,**kw); aL.plot((1-k,1+k),(1-k*1.5,1+k*1.5),transform=aL.transAxes,**kw)
    kr=k*1400/500
    aR.plot((-kr,kr),(-k*1.5,k*1.5),transform=aR.transAxes,**kw); aR.plot((-kr,kr),(1-k*1.5,1+k*1.5),transform=aR.transAxes,**kw)
    aL.set_ylabel(ylabel); fig.text(0.5,aL.get_position().y0-0.035,'Raman shift (cm$^{-1}$)',ha='center')
    if title: fig.text(0.5,aL.get_position().y1+0.008,title,ha='center',fontsize=12)
    if legend: aL.legend(fontsize=8,loc='upper left')
    return aL,aR
def d1(a):
    a.fill_between(xp,G(mean-sd),G(mean+sd),color='C0',alpha=.2,label='±1 SD')
    a.plot(xp,G(mean),'C0',lw=.8,label=f'Mean of {len(sel)} matched map spectra')
    a.plot(xp,G(restmean),color='gray',lw=.6,alpha=.8,label=f'Mean of {len(rest)} non-matched spectra')
broken(0,d1,'Intensity (cnt)','1 mg/mL PS map (AgSiNW) – spectra matching the CaF2 reference')
nm=meanbc[(x>995)&(x<1008)].max(); nr=R[(x>995)&(x<1008)].max()
def d2(a):
    a.plot(xp,G(meanbc)/nm,'C0',lw=.8,label='Matched mean (baseline-corrected)')
    a.plot(xp,G(R)/nr+1.1,'C3',lw=.8,label='CaF2 reference s9 (baseline-corrected, offset)')
    for p in [621,1001,1031,1602,3054]: a.axvline(p,color='k',lw=.4,ls=':')
broken(1,d2,'Norm. intensity')
a=fig.add_subplot(gs[2,:]); m=(x>960)&(x<1060)
for i in sel: a.plot(x[m],I[i][m]-np.median(I[i][m]),lw=.6,alpha=.6,label=f'#{i}')
a.plot(x[m],mean[m]-np.median(mean[m]),'k',lw=1.6,label='mean'); a.axvline(1001.9,color='C3',ls='--',lw=.8)
a.set_xlabel('Raman shift (cm$^{-1}$)'); a.set_ylabel('Intensity – median'); a.legend(ncol=4,fontsize=8); a.set_title('Zoom: PS ring-breathing band (ref 1001.9 cm$^{-1}$)')
plt.savefig(f'{OUT}/mean_spectrum.png',bbox_inches='tight',dpi=130)
fig,ax=plt.subplots(1,2,figsize=(10,4.3))
dd=df.sort_values('pixel'); ok=dd.center_1001.between(1000.4,1003.4)&dd.fwhm_1001.between(3,9)&(dd.p_1001<1e-3)
A=np.where(ok,dd.A_1001,0).reshape(5,5); im=ax[0].imshow(A,cmap='viridis'); plt.colorbar(im,ax=ax[0],label='1001 cm$^{-1}$ amplitude (cnt)')
Z=df.sort_values('pixel').z_template.values.reshape(5,5); im=ax[1].imshow(Z,cmap='magma'); plt.colorbar(im,ax=ax[1],label='template match z-score')
for a,t in zip(ax,['Valid PS 1001 band amplitude (0 = not detected)','Match score vs CaF2 reference']):
    a.set_title(t)
    for i in range(25):
        a.text(i%5,i//5,str(i),ha='center',va='center',color='w',fontsize=8,fontweight='bold' if i in sel else 'normal')
        if i in sel: a.add_patch(plt.Rectangle((i%5-.5,i//5-.5),1,1,fill=False,ec='red',lw=2))
    a.set_xlabel('column'); a.set_ylabel('row')
plt.tight_layout(); plt.savefig(f'{OUT}/match_map.png',dpi=130)
np.save('sel.npy',np.array(sel))
