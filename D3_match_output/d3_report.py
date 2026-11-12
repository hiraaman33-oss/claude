import os
exec(open('common.py').read())
import pandas as pd
src=open('final_fig.py').read()
src=src[:src.index("PROM_R=float")]   # function defs only
src=src.replace("OUT='/home/user/claude/CaF2_match_output'","OUT='/home/user/claude/D3_match_output'")
src=src.replace("f'{OUT}/final_spectrum_{tag}.png'","f'{OUT}/D3_final_spectrum_{tag}.png'")
os.makedirs('/home/user/claude/D3_match_output',exist_ok=True)
exec(src)
OFFSET=200; TOPPAD=80; PROM_R=9; PROM_M=float(os.environ.get('PROM_M',7))
df=pd.read_pickle('D3_final.pkl').sort_values('pixel'); sel=df[df.match=='MATCH'].pixel.tolist()
print('matched',sel,len(sel))
t,mean=make(sel,f'{len(sel)}matched','')
print(t.to_string())
pd.to_pickle((df,sel,t,mean),'D3_res.pkl')
