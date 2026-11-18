import sys,numpy as np,pandas as pd
from scipy.stats import f as fdist
exec(open('common.py').read())
exec(open('load_l6s.py').read()); exec(open('despike.py').read())
src=open('final_sel.py').read(); exec(src[src.index('def gq'):src.index('rows=[]')])
xs,ys=load_l6s(sys.argv[1]); print('points',len(xs),xs[0],xs[-1],'same axis as D-maps:',len(xs)==len(x) and np.allclose(xs,x,atol=0.05))
y,nsp=despike(ys); print('spikes removed',nsp)
x=xs; I=y[None,:]; R=proc(np.interp(x,ref[:,0],ref[:,1]))
f=fit1001(y); r,z=mf_z(0)
ok=(1000.4<=f['center']<=1003.4) and (3<=f['fwhm']<=9) and f['p']<1e-3 and z>=2
print({k:round(float(v),4) for k,v in f.items()}, 'template r',round(r,3),'z',round(z,2), 'MATCH' if ok else 'no')
np.save('d5.npy',np.vstack([xs,y]))
