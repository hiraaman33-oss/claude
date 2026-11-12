import numpy as np,struct
def load_l6m(F):
    d=open(F,'rb').read()
    ii=d.find(b'Intens'); cnt=struct.unpack('<i',d[ii+52:ii+56])[0]
    mi=d.find(b'mean',ii); end=mi-32; st=end-4*cnt
    xi=d.find(b'1/cm'); xs=xi+80
    xa=np.frombuffer(d[xs:xs+4*8000],'<f4'); N=int(np.argmax(np.diff(xa)<=0))+1
    x=xa[:N].astype(float); I=np.frombuffer(d[st:end],'<f4').astype(float).reshape(cnt//N,N)
    # spatial axis after x-axis block (same layout as file 1: x-end + 288)
    ax=np.frombuffer(d[xs+4*N+288:xs+4*N+308],'<f4')
    return d,x,I,ax
