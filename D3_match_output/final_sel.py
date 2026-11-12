exec(open('common.py').read())
import pandas as pd
from scipy.stats import f as fdist
def gq(xx,a,c,w,b0,b1,b2): return a*np.exp(-4*np.log(2)*(xx-c)**2/w**2)+b0+b1*(xx-1001)+b2*(xx-1001)**2
def fit1001(y,lo=986,hi=1020):
    m=(x>lo)&(x<hi); X=x[m]; Y=y[m]
    q=np.polyfit(X-1001,Y,2); rss0=np.sum((Y-np.polyval(q,X-1001))**2)
    try:
        p,cov=curve_fit(gq,X,Y,p0=[max(Y.max()-np.median(Y),1),1001.5,5,q[2],q[1],q[0]],
            bounds=([0,998,2.5,-np.inf,-np.inf,-np.inf],[np.inf,1005,14,np.inf,np.inf,np.inf]),maxfev=50000)
    except Exception: return dict(A=0,center=np.nan,fwhm=np.nan,F=0,p=1,sigA=0)
    rss1=np.sum((Y-gq(X,*p))**2); n=len(X)
    F=((rss0-rss1)/3)/(rss1/(n-6)); e=np.sqrt(np.diag(cov))
    return dict(A=p[0],center=p[1],fwhm=p[2],F=F,p=fdist.sf(F,3,n-6),sigA=p[0]/e[0] if e[0]>0 else 0)
def mf_z(i):
    ys=savgol_filter(I[i],9,2); m=(x>985)&(x<1045); L=m.sum()
    tpl=R[m]-np.polyval(np.polyfit(np.arange(L),R[m],1),np.arange(L))
    def r(seg): seg=seg-np.polyval(np.polyfit(np.arange(L),seg,1),np.arange(L)); return np.corrcoef(seg,tpl)[0,1]
    i0=np.argmax(m); rr=r(ys[i0:i0+L])
    nulls=[r(ys[s:s+L]) for s in range(np.searchsorted(x,1060),np.searchsorted(x,1550)-L,5)]
    return rr,(rr-np.mean(nulls))/np.std(nulls)
rows=[]
for i in range(25):
    f=fit1001(I[i]); r,z=mf_z(i)
    rows.append(dict(pixel=i,row=i//5,col=i%5,**{k+'_1001':v for k,v in f.items()},r_template=r,z_template=z,
        r_fingerprint=corr(C[i],R,(580,1700)),r_full=corr(C[i],R,(580,3150))))
df=pd.DataFrame(rows)
fr=fit1001(yref); print('REF',fr)
c=(df.center_1001.between(1000.4,1003.4))&(df.fwhm_1001.between(3,9))&(df.p_1001<1e-3)&(df.z_template>=2.0)
df['match']=np.where(c,'MATCH','no')
pd.set_option('display.width',300); print(df.round(4).sort_values('p_1001').to_string())
import os; df.to_pickle(os.environ.get('TAG','')+'final.pkl')
