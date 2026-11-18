import numpy as np, pandas as pd, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from scipy.optimize import curve_fit
traces,offs,rows,s1,s2,s3,s4,s5=pd.read_pickle('combined6.pkl')
LO,HI=985,1050
names=['CaF$_2$ reference · 1 mg/mL','D1 · 1 mg/mL','D2 · 100 µg/mL','D3 · 10 µg/mL','D4 · 1 µg/mL','D5 · 0.5 µg/mL','D6 · 0.1 µg/mL (not detected)']
ns=['',f'n = {len(s1)}',f'n = {len(s2)}',f'n = {len(s3)}',f'n = {len(s4)}',f'n = {len(s5)}','n = 2']
SCALE=[0.25,1,1,1,1,1,1]
def g(x,a,c,w,b0,b1): return a*np.exp(-4*np.log(2)*(x-c)**2/w**2)+b0+b1*(x-1002)
rows_out=[]; prep=[]
for k,(lab,xx,y,c,_,_) in enumerate(traces):
    m=(xx>LO)&(xx<HI); X=xx[m]; Y=savgol_filter(y,7,2)[m]
    bm=((X<993)|((X>1012)&(X<1024))|(X>1040))
    Y=(Y-np.polyval(np.polyfit(X[bm],Y[bm],1),X))*SCALE[k]
    w=(xx>988)&(xx<1016); Xf=xx[w]; Yf=y[w]
    try:
        p,_=curve_fit(g,Xf,Yf,p0=[max(Yf.max()-np.median(Yf),1),1002,5,np.median(Yf),0],bounds=([0,999,2.5,-np.inf,-np.inf],[np.inf,1005,12,np.inf,np.inf]),maxfev=20000)
        res=Yf-g(Xf,*p); snr=p[0]/np.std(res)
    except Exception: p=[0,np.nan,np.nan,0,0]; snr=0
    det=k==0 or (snr>=3 and 1000.4<=p[1]<=1003.4)
    rows_out.append(dict(sample=names[k].replace('$_2$','2'),spectra=ns[k] or '1',center_cm1=round(p[1],1) if det else None,height_counts=round(p[0],1) if det else None,fwhm_cm1=round(p[2],1) if det else None,snr=round(snr,1),detected='yes' if det else 'no'))
    prep.append((X,Y,c,det,p))
fig,ax=plt.subplots(figsize=(7,10),facecolor='white')
off=0; GAP=13
for k in range(len(traces)-1,-1,-1):
    X,Y,c,det,p=prep[k]
    base=off-np.percentile(Y,3)
    ax.plot(X,Y+base,color=c,lw=1.4)
    lab=names[k]+(f'  (×{SCALE[k]:g})' if SCALE[k]!=1 else '')
    full=lab
    ax.text(HI-0.5,base+Y[X>1012].max()+1.0,full,color=c,fontsize=10.5,fontweight='bold',ha='right',va='bottom')
    if det:
        i=np.argmin(abs(X-p[1])); top=Y[max(i-3,0):i+4].max()+base
        ax.text(p[1],top+1,f'{p[1]:.1f}',color=c,fontsize=9.5,ha='center',va='bottom',fontweight='bold')
    off=base+max(Y.max(),Y[X>1012].max()+6)+GAP
ax.axvline(1001.9,color='#888',lw=.8,ls='--',zorder=0)
ax.axvspan(997,1006,color='#e07030',alpha=.10,lw=0,zorder=0)
ax.set_xlim(LO,HI); ax.set_ylim(-3,off+2); ax.set_yticks([])
for s in ['top','right']: ax.spines[s].set_visible(False)
ax.set_xlabel('Raman shift (cm$^{-1}$)',fontsize=12); ax.set_ylabel('Intensity (counts, baseline-corrected, offset)',fontsize=11)
ax.set_title('PS ring-breathing band (~1002 cm$^{-1}$)',fontsize=12)
OUT='/home/user/claude/Concentration_series_D1-D6/'
fig.savefig(OUT+'PS_D1-D6_1001_zoom.png',dpi=220,bbox_inches='tight')
pd.DataFrame(rows_out).to_pickle('zoom_rows.pkl'); print(pd.DataFrame(rows_out).to_string())
np.save('zoom_prep.npy',np.array([(X,Y) for X,Y,_,_,_ in prep],dtype=object),allow_pickle=True)
