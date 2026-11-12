import pandas as pd, numpy as np
N=4
def rank(f):
    d=pd.read_pickle(f).copy()
    valid=(d.center_1001.between(1000.4,1003.4)&d.fwhm_1001.between(3,9)&(d.p_1001<1e-3)).values
    d['score']=np.where(valid, np.clip(d.sigA_1001,0,15)/15+np.clip(d.z_template,0,4)/4, 0)
    return d.sort_values('score',ascending=False)
sel={}
for k,f in [('1','N1_final.pkl'),('2','D2_final.pkl'),('3','D3_final.pkl')]:
    d=rank(f); sel[k]=d.pixel.head(N).tolist(); print(k,[i+1 for i in sel[k]], d.head(N)[['match']].values.ravel())
pd.to_pickle(sel,'equal_sel.pkl')
