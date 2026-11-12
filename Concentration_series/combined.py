import os, numpy as np, pandas as pd, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from scipy.signal import savgol_filter, find_peaks
exec(open('load_generic.py').read())
U='/root/.claude/uploads/a2c2f359-63aa-5c06-a173-74003b657340/'
ref=np.loadtxt(U+'adb56f59-s9-PolyS-1ml-dropCaF2-EDGE-532nm-1800gr-100-100x-BC50-5s-3a.txt')
_,x1,I1,_=load_l6m(U+'05982966-m1-1MGML_PS-AgSiNW_532nm_1800gr_BC200_50XLF_1s_4a_10_1ul_drop-center.l6m')
_,x2,I2,_=load_l6m(U+'9a8bbf9a-m1-D2_PS-AgSiNW_532nm_1800gr_BC200_50XLF_1s_2a_10_3ul_drop-center.l6m')
_,x3,I3,_=load_l6m(U+'dc31b301-m1-D3_PS-AgSiNW_532nm_1800gr_BC200_50XLF_1s_2a_10_3ul_drop-center.l6m')
s1=pd.read_pickle('N1_final.pkl').query("match=='MATCH'").pixel.tolist(); s2=pd.read_pickle('D2_res.pkl')[1]; s3=pd.read_pickle('D3_res.pkl')[1]
traces=[  # label, x, y, color, prominence, forced peaks
 ('1 mg/mL PS on CaF$_2$ (reference)',ref[:,0],ref[:,1],'#1f2a44',9,()),
 (f'1 mg/mL PS on AgSiNW ({len(s1)} spectrum)' if len(s1)==1 else f'1 mg/mL PS on AgSiNW (mean of {len(s1)})',x1,I1[s1].mean(0),'#e07030',14,(1002,)),
 (f'100 µg/mL PS on AgSiNW – D2 (mean of {len(s2)})',x2,I2[s2].mean(0),'#2a9d8f',5,(1002,3060)),
 (f'10 µg/mL PS on AgSiNW – D3 (mean of {len(s3)})',x3,I3[s3].mean(0),'#8e44ad',5,(1002,3060)),
]
SHORT=['CaF$_2$ ref.','1 mg/mL','100 µg/mL','10 µg/mL']
def inwin(v): return 410<=v<=1790 or 2660<=v<=3170
def peaks(xx,y,prom,forced):
    s=savgol_filter(y,15,2); g=int(np.argmax(np.diff(xx)))+1; out=[]
    for a,b in [(0,g),(g,len(xx))]:
        p,_=find_peaks(s[a:b],prominence=prom,distance=12); out+=[a+i for i in p]
    for f in forced:
        m=np.where(abs(xx-f)<4)[0]; out.append(m[np.argmax(s[m])])
    return sorted(set(i for i in out if inwin(xx[i])))
# stack: bottom = lowest concentration
order=[3,2,1,0]; GAPC=40; cur=0; offs={}
for k in order:
    xx,y=traces[k][1],traces[k][2]; m=np.array([inwin(v) for v in xx])
    offs[k]=cur-np.percentile(y[m],2); cur=offs[k]+y[m].max()+GAPC+45
fig=plt.figure(figsize=(14,10),facecolor='#fafaf7'); gs=fig.add_gridspec(1,2,width_ratios=[1400,500],wspace=0.03)
aL=fig.add_subplot(gs[0]); aR=fig.add_subplot(gs[1],sharey=aL)
rows=[]
for k,(lab,xx,y,c,prom,forced) in enumerate(traces):
    yy=y+offs[k]; g=int(np.argmax(np.diff(xx)))+1
    for a in (aL,aR):
        a.set_facecolor('#fafaf7'); a.plot(np.insert(xx,g,np.nan),np.insert(yy,g,np.nan),color=c,lw=.8)
    pk=peaks(xx,y,prom,forced); last=-1e9
    for i in pk:
        pos=xx[i]; sp=16 if pos<=1800 else 34; tx=max(pos,last+sp); last=tx
        a=aL if pos<=1800 else aR
        a.text(tx,yy[max(i-4,0):i+5].max()+3,f'{pos:.0f}',rotation=90,ha='center',va='bottom',color=c,fontsize=8.5,fontweight='bold')
        rows.append((lab.replace('$_2$','2'),round(pos,1)))
    m=(xx>2660)&(xx<2800)
    aR.text(2665,np.median(yy[m])+14,SHORT[k],color=c,fontsize=12,fontweight='bold',va='bottom')
for a in (aL,aR): a.axvspan(997,1006,color='#e07030',alpha=.12,lw=0)
aL.set_xlim(400,1800); aR.set_xlim(2650,3180); aL.set_xticks(range(400,1801,200)); aR.set_xticks(range(2700,3101,100))
aL.set_ylim(None,cur-GAPC+150)
for a in (aL,aR): a.spines['top'].set_visible(False); a.set_xlabel('Raman shift (cm$^{-1}$)',fontsize=12)
aL.spines['right'].set_visible(False); aR.spines['left'].set_visible(False); aR.spines['right'].set_visible(False)
aR.tick_params(axis='y',left=False,labelleft=False); aL.set_yticks([])
from matplotlib.lines import Line2D
aL.legend([Line2D([],[],color=t[3],lw=2) for t in traces],[t[0] for t in traces],loc='upper left',frameon=False,fontsize=10.5)
aL.set_ylabel('Intensity (counts, offset for clarity)',fontsize=12)
OUT='/home/user/claude/Concentration_series'; os.makedirs(OUT,exist_ok=True)
fig.savefig(f'{OUT}/PS_concentration_series.png',dpi=200,bbox_inches='tight',facecolor='#fafaf7')
pd.to_pickle((traces,offs,rows,s1,s2,s3),'combined.pkl')
for r in rows: print(r)
