exec(open('common.py').read())
import sys, pandas as pd, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from scipy.signal import find_peaks
OUT='/home/user/claude/CaF2_match_output'
NAVY='#1f2a44'; ORG='#e07030'; BG='#fafaf7'
def peaks(xx,y,prom,dist=12,forced=()):
    s=savgol_filter(y,15,2); out=[]
    for a,b in [(0,GAP),(GAP,len(xx))] if len(xx)==len(x) else [(0,int(np.argmax(np.diff(xx)))+1),(int(np.argmax(np.diff(xx)))+1,len(xx))]:
        ss=s[a:b]; p,_=find_peaks(ss,prominence=prom,distance=dist); out+=[a+i for i in p]
    for f in forced:
        m=np.where(abs(xx-f)<4)[0]; out.append(m[np.argmax(s[m])])
    out=sorted(set(out)); return out,s
def make(sel,tag,label):
    mean=I[sel].mean(0)
    rx,ry=ref[:,0],ref[:,1]+OFFSET
    rp,rs=peaks(rx,ry,PROM_R)
    mp,ms=peaks(x,mean,PROM_M,forced=(1002,3060))
    fig=plt.figure(figsize=(14,6.6),facecolor=BG)
    gs=fig.add_gridspec(1,2,width_ratios=[1400,500],wspace=0.03)
    aL=fig.add_subplot(gs[0]); aR=fig.add_subplot(gs[1],sharey=aL)
    def ylim_lo(): return min(mean.min(),ry.min())-10
    for a in (aL,aR):
        a.set_facecolor(BG)
        for xx,yy,c in [(rx,ry,NAVY),(x,mean,ORG)]:
            g=int(np.argmax(np.diff(xx)))+1
            a.plot(np.insert(xx,g,np.nan),np.insert(yy,g,np.nan),color=c,lw=.8)
        a.axvspan(997,1006,color=ORG,alpha=.15,lw=0)
    for xx,s,pk,c in [(rx,savgol_filter(ry,15,2),rp,NAVY),(x,ms,mp,ORG)]:
        keep=[i for i in pk if (410<=xx[i]<=1790 or 2660<=xx[i]<=3170)]
        last=-1e9
        for i in keep:
            pos=xx[i]
            sp=16 if pos<=1800 else 34
            tx=max(pos,last+sp); last=tx
            a=aL if pos<=1800 else aR
            if pos>1800 and last<2000: pass
            raw=(ry if c==NAVY else mean)
            yv=raw[max(i-4,0):i+5].max()
            a.text(tx,yv+4,f'{pos:.0f}',rotation=90,ha='center',va='bottom',color=c,fontsize=9,fontweight='bold')
    aL.set_xlim(400,1800); aR.set_xlim(2650,3180)
    aL.set_xticks(range(400,1801,200)); aR.set_xticks(range(2700,3101,100))
    top=max(ry.max(),mean.max())+TOPPAD; aL.set_ylim(ylim_lo(),top)
    for a in (aL,aR):
        a.spines['top'].set_visible(False)
        a.set_xlabel('Raman shift (cm$^{-1}$)',fontsize=12)
    aL.spines['right'].set_visible(False); aR.spines['left'].set_visible(False); aR.spines['right'].set_visible(False)
    aR.tick_params(axis='y',left=False,labelleft=False)
    aL.set_ylabel(f'Counts (reference shifted +{OFFSET})',fontsize=12)
    from matplotlib.lines import Line2D
    aL.legend([Line2D([],[],color=NAVY),Line2D([],[],color=ORG)],
              ['1 mg/mL PS on CaF$_2$ (reference)',f'Mean of {len(sel)} matching AgSiNW spectra'],
              loc='upper left',frameon=False,fontsize=11)
    fig.savefig(f'{OUT}/final_spectrum_{tag}.png',dpi=200,bbox_inches='tight',facecolor=BG)
    # peak table
    rows=[('CaF2 reference',rx[i]) for i in rp if (410<=rx[i]<=1790 or 2660<=rx[i]<=3170)]+[(f'Mean of {len(sel)} AgSiNW',x[i]) for i in mp if (410<=x[i]<=1790 or 2660<=x[i]<=3170)]
    return pd.DataFrame(rows,columns=['Spectrum','Peak position (cm-1)']).round(1), mean
OFFSET=160; TOPPAD=45
PROM_R=float(sys.argv[1]) if len(sys.argv)>1 else 9; PROM_M=float(sys.argv[2]) if len(sys.argv)>2 else 7
sel7=list(np.load('sel.npy'))
slide=[1,4,5,9,10,12,13,14,16,17,18,22,23,24,25]; sel15=[i-1 for i in slide]
t7,m7=make(sel7,'7matched','7')
t15,m15=make(sel15,'15slide','15')
pd.to_pickle((t7,t15,m7,m15,sel7,sel15),'final_peaks.pkl')
print(t7.to_string()); print(t15.to_string())
