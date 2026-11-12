import numpy as np
from scipy.ndimage import median_filter
def despike(y,k=7,th=6):
    m=median_filter(y,k); r=y-m; s=1.4826*np.median(np.abs(r-np.median(r)))
    bad=r>th*s
    bad=bad|np.roll(bad,1)|np.roll(bad,-1)
    out=y.copy(); out[bad]=m[bad]; return out,bad.sum()
