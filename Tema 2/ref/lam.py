import numpy as np
from offc import run, eq, strains, MAT
for mat in ['S','D']:
  m=MAT[mat]
  for a in [0.2,0.3,0.4,0.5]:
    f=run(mat,8,a,0.0)
    k,w,e=f[0]
    if k!='out': print(mat,a,'first',k); continue
    h,_=eq(w,8,a,0.0,{'in':True,'out':True},m,np.array([0.,0.]))
    s0,_=strains(h,w,8,a,0.0,None)
    h2,_=eq(w,8,a,0.0,{'in':True,'out':False},m,h)
    s1,_=strains(h2,w,8,a,0.0,None)
    print(mat,a,'w_b=%.3f'%w,'pre: in/eb=%.3f rad/eb=%.3f'%(s0['in']/m['eb'],s0['rad'].max()/m['eb']),'post(same w): in/eb=%.3f rad/eb=%.3f'%(s1['in']/m['eb'],s1['rad'].max()/m['eb']),'w2=%.3f'%f[1][1])
