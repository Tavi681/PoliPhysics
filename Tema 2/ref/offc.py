import numpy as np
from scipy.optimize import minimize
MAT={'S':dict(E0=2e9,b=22.2e9,eb=0.30,rho=1300),'D':dict(E0=110e9,b=0.0,eb=0.032,rho=975)}
def Phi(e,m): e=np.maximum(e,0); return m['E0']*e**2/2+m['b']*e**4/4
def setup(N,a,ep,R=1.0):
    phi=2*np.pi*np.arange(1,N)/N
    anch=np.c_[R*np.cos(phi),R*np.sin(phi),np.zeros(N-1)]
    segs=[]  # (kind, endpointA, endpointB, rest)
    for k in range(N-1): segs.append(('rad',k))
    L0=R/(1+ep)
    rest={'rad':L0,'in':a/(1+ep),'out':(R-a)/(1+ep)}
    return anch,rest
def strains(h,w,N,a,ep,alive,R=1.0):
    anch,rest=setup(N,a,ep,R)
    P=np.array([a,0,w]); hub=np.array([h[0],0,h[1]])
    eps={}
    eps['rad']=np.linalg.norm(anch-hub,axis=1)/rest['rad']-1
    eps['in']=np.linalg.norm(P-hub)/rest['in']-1
    eps['out']=np.linalg.norm(P-np.array([R,0,0]))/rest['out']-1
    return eps,rest
def energy(h,w,N,a,ep,alive,m,R=1.0):
    eps,rest=strains(h,w,N,a,ep,alive,R)
    U=np.sum(Phi(eps['rad'],m))*rest['rad']
    if alive['in']: U+=Phi(eps['in'],m)*rest['in']
    if alive['out']: U+=Phi(eps['out'],m)*rest['out']
    return U  # per unit A
def eq(w,N,a,ep,alive,m,h0):
    f=lambda h: energy(h,w,N,a,ep,alive,m)/m['E0']
    r=minimize(f,h0,method='Nelder-Mead',options=dict(xatol=1e-10,fatol=1e-16,maxiter=4000))
    return r.x, energy(r.x,w,N,a,ep,alive,m)
def run(mat,N,a,ep_frac,dw=None):
    m=MAT[mat]; ep=ep_frac*m['eb']; R=1.0
    mass=N*R/(1+ep)  # per unit rho*A
    emat_vol=Phi(m['eb'],m)
    alive={'in':True,'out':True}; h=np.array([0.0,0.0]); w=0.0
    dw=dw or (0.002 if mat=='S' else 0.0005)
    Uabs=0.0; Uprev=0.0; fails=[]
    while True:
        w+=dw
        h,U=eq(w,N,a,ep,alive,m,h)
        eps,_=strains(h,w,N,a,ep,alive)
        cand={'rad':eps['rad'].max()}
        if alive['in']: cand['in']=eps['in']
        if alive['out']: cand['out']=eps['out']
        k=max(cand,key=cand.get)
        if cand[k]>=m['eb']:
            # bisect w for failure
            lo,hi=w-dw,w
            for _ in range(40):
                mid=(lo+hi)/2; hm,Um=eq(mid,N,a,ep,alive,m,h)
                e2,_=strains(hm,mid,N,a,ep,alive)
                c={'rad':e2['rad'].max()}
                if alive['in']: c['in']=e2['in']
                if alive['out']: c['out']=e2['out']
                if c[k]>=m['eb']: hi=mid
                else: lo=mid
            w=hi; h,U=eq(w,N,a,ep,alive,m,h)
            Uabs_tot=Uabs+U-Uprev  # energy absorbed up to failure
            fails.append((k,w,Uabs_tot/(mass*emat_vol)))
            if k=='rad' or len(fails)==2: break
            alive[k]=False
            h2,Urel=eq(w,N,a,ep,alive,m,h)
            Uabs=Uabs_tot; Uprev=Urel; h=h2
            continue
    return fails
if __name__=='__main__':
    import sys
    for mat in ['S','D']:
        for epf in [0.0,0.1]:
            for a in [0.05,0.1,0.2,0.3,0.4,0.5]:
                f=run(mat,8,a,epf)
                print(mat,epf,a,[(k,round(w,4),round(e,4)) for k,w,e in f],flush=True)
