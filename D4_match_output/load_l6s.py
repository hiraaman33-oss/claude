import numpy as np,struct
def load_l6s(F):
    d=open(F,'rb').read()
    ii=d.find(b'Intens'); cnt=struct.unpack('<i',d[ii+52:ii+56])[0]; st=ii+80
    y=np.frombuffer(d[st:st+4*cnt],'<f4').astype(float)
    xi=d.find(b'1/cm'); xs=xi+80
    xa=np.frombuffer(d[xs:xs+4*cnt],'<f4').astype(float)
    return xa,y
