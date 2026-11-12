import numpy as np,struct
F='/root/.claude/uploads/a2c2f359-63aa-5c06-a173-74003b657340/e2075235-m1-1MGML_PS-AgSiNW_532nm_1800gr_BC200_50XLF_1s_2a_10_1ul_drop-center.l6m'
R='/root/.claude/uploads/a2c2f359-63aa-5c06-a173-74003b657340/adb56f59-s9-PolyS-1ml-dropCaF2-EDGE-532nm-1800gr-100-100x-BC50-5s-3a.txt'
d=open(F,'rb').read()
N=4829
x=np.frombuffer(d[522568:522568+4*N],'<f4').astype(float)
st=483576-4*120725
I=np.frombuffer(d[st:483576],'<f4').astype(float).reshape(25,N)
ref=np.loadtxt(R)
