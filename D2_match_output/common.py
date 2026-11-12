import numpy as np, warnings; warnings.filterwarnings('ignore')
from scipy import sparse
from scipy.sparse.linalg import spsolve
from scipy.signal import savgol_filter
from scipy.optimize import curve_fit
exec(open('/tmp/claude-0/-home-user-claude/a2c2f359-63aa-5c06-a173-74003b657340/scratchpad/load.py').read())
import os
if os.environ.get('MAPFILE'):
    exec(open('/tmp/claude-0/-home-user-claude/a2c2f359-63aa-5c06-a173-74003b657340/scratchpad/load_generic.py').read())
    d,x,I,_ax=load_l6m(os.environ['MAPFILE'])
def als(y,lam=1e5,p=0.01,n=10):
    L=len(y); D=sparse.diags([1.,-2.,1.],[0,-1,-2],shape=(L,L-2)); D=lam*D.dot(D.T)
    w=np.ones(L)
    for _ in range(n):
        W=sparse.spdiags(w,0,L,L); z=spsolve((W+D).tocsc(),w*y); w=p*(y>z)+(1-p)*(y<z)
    return z
yref=np.interp(x,ref[:,0],ref[:,1])  # map gap region also gap in ref
GAP=int(np.argmax(np.diff(x)))+1
def proc(y):
    out=np.empty(len(y))
    for a,b in [(0,GAP),(GAP,len(y))]:
        s=savgol_filter(y[a:b],9,2); out[a:b]=s-als(s)
    return out
R=proc(yref); C=np.array([proc(v) for v in I])
def corr(a,b,rg):
    m=(x>rg[0])&(x<rg[1])&~((x>500)&(x<540))&~((x>930)&(x<990))
    return np.corrcoef(a[m],b[m])[0,1]
def g(xx,a,c,w,b0,b1): return a*np.exp(-4*np.log(2)*(xx-c)**2/w**2)+b0+b1*(xx-c)
def fitpk(xx,y,pos,win=25,cw=(4,6,16)):
    m=(xx>pos-win)&(xx<pos+win); X=xx[m]; Y=y[m]
    try:
        p,cov=curve_fit(g,X,Y,p0=[max(Y.max()-np.median(Y),1),pos,8,np.median(Y),0],
            bounds=([0,pos-cw[1],cw[0],-np.inf,-np.inf],[np.inf,pos+cw[1],cw[2],np.inf,np.inf]),maxfev=20000)
        return p[0],p[1],p[2],0
    except Exception: return 0,pos,0,0
