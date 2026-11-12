import numpy as np, pandas as pd, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
exec(open('common.py').read().split("yref=")[0])   # als + loaders
exec(open('load_generic.py').read())
traces,offs,rows,s1,s2,s3=pd.read_pickle('combined.pkl')
def proc_seg(xx,y):
    g=int(np.argmax(np.diff(xx)))+1; out=np.empty(len(y))
    for a,b in [(0,g),(g,len(y))]:
        s=savgol_filter(y[a:b],9,2); out[a:b]=s-als(s)
    return out,g
BLUE='#3b82c4'; RED='#c0262d'
bands=[(1,1002,8),(2,1190,25),(3,1600,15),(4,2930,30),(5,3060,20)]
panels=[('CaF$_2$\nreference','1 mg/mL PS',traces[0],[1,2,3,4,5]),
        ('1 mg/mL','2.3×10$^{14}$ NP/mL\n(1 matching spectrum)',traces[1],[1]),
        ('100 µg/mL','2.3×10$^{13}$ NP/mL\n(mean of 15 matching spectra)',traces[2],[1,3,4,5]),
        ('10 µg/mL','2.3×10$^{12}$ NP/mL\n(mean of 13 matching spectra)',traces[3],[1,3,4,5])]
fig,axs=plt.subplots(4,2,figsize=(12,11),gridspec_kw=dict(width_ratios=[1400,500],wspace=0.04,hspace=0.18))
for r,(tag,sub,(lab,xx,y,c,_,_),show) in enumerate(panels):
    yc,g=proc_seg(xx,y); xp=np.insert(xx,g,np.nan); yp=np.insert(yc,g,np.nan)
    aL,aR=axs[r]
    for a in (aL,aR): a.plot(xp,yp,color=BLUE,lw=.7)
    aL.set_xlim(400,1800); aR.set_xlim(2650,3180)
    m=((xx>420)&(xx<1790))|((xx>2660)&(xx<3170)); top=np.nanmax(yc[m])
    aL.set_ylim(min(-0.08*top,np.percentile(yc[m],0.5)),top*1.25); aR.set_ylim(aL.get_ylim())
    for n,pos,hw in bands:
        if n not in show: continue
        w=np.where(abs(xx-pos)<hw)[0]; i=w[np.argmax(savgol_filter(yc,15,2)[w])]
        a=aL if xx[i]<1800 else aR
        a.text(xx[i],yc[max(i-4,0):i+5].max()+0.04*top,str(n),color=RED,fontsize=12,fontweight='bold',ha='center',va='bottom')
    aL.spines['right'].set_visible(False); aR.spines['left'].set_visible(False)
    aR.tick_params(axis='y',left=False,labelleft=False)
    for a in (aL,aR): a.spines['top'].set_visible(False)
    aL.set_ylabel('Intensity',fontsize=10)
    aL.text(-0.13,0.5,tag,transform=aL.transAxes,rotation=90,ha='center',va='center',fontsize=13,fontweight='bold')
    aL.text(0.01,0.97,sub,transform=aL.transAxes,va='top',fontsize=9,color='#555')
    if r<3: aL.set_xticklabels([]); aR.set_xticklabels([])
    k=.012
    for a,kk,x0 in [(aL,k,1),(aR,k*2.8,0)]:
        a.plot((x0-kk,x0+kk),(-.03,.03),transform=a.transAxes,color='k',lw=1,clip_on=False)
axs[3][0].set_xticks(range(400,1801,200)); axs[3][1].set_xticks([2700,2900,3100])
fig.text(0.5,0.065,'Raman shift in wavenumber, cm$^{-1}$',ha='center',fontsize=12)
fig.savefig('/home/user/claude/Concentration_series/PS_concentration_paper_style.png',dpi=200,bbox_inches='tight',facecolor='white')
