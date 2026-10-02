from offc import run
for N in [4,16]:
  for mat in ['S','D']:
    for a in [0.1,0.3,0.5]:
      f=run(mat,N,a,0.0); print(N,mat,a,[(k,round(w,3),round(float(e),3)) for k,w,e in f],flush=True)
