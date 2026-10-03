"""Off-centre impact on a star net, drone free to move in the plane (no lateral force).
Same as offc.py, but the gripped point P=(px,0,w) has px as an extra equilibrium unknown.
Prestress energy is subtracted. Only criterion A (first failure) is computed."""
import numpy as np
from scipy.optimize import minimize
from offc import MAT, Phi
def strains(x,w,N,a,ep,R=1.0):
    hx,hz,px=x
    phi=2*np.pi*np.arange(1,N)/N
    anch=np.c_[R*np.cos(phi),R*np.sin(phi),np.zeros(N-1)]
    hub=np.array([hx,0,hz]); P=np.array([px,0,w])
    L0=R/(1+ep)
    e_rad=np.linalg.norm(anch-hub,axis=1)/L0-1
    e_in=np.linalg.norm(P-hub)/(a/(1+ep))-1
    e_out=np.linalg.norm(P-np.array([R,0,0]))/((R-a)/(1+ep))-1
    return e_rad,e_in,e_out,L0
def energy(x,w,N,a,ep,m,R=1.0):
    e_rad,e_in,e_out,L0=strains(x,w,N,a,ep,R)
    return np.sum(Phi(e_rad,m))*L0+Phi(e_in,m)*a/(1+ep)+Phi(e_out,m)*(R-a)/(1+ep)
def run(mat,N,a,ep_frac,dw=None):
    m=MAT[mat]; ep=ep_frac*m['eb']; R=1.0
    mass=N*R/(1+ep); ev=Phi(m['eb'],m)
    x=np.array([0.0,0.0,a]); U0=energy(x,0.0,N,a,ep,m)
    dw=dw or (0.01 if mat=='S' else 0.003); w=0.0
    def eq(w,x):
        r=minimize(lambda y: energy(y,w,N,a,ep,m)/m['E0'],x,method='BFGS',
                   options=dict(gtol=1e-14,maxiter=2000))
        return r.x
    def emax(x,w):
        e_rad,e_in,e_out,_=strains(x,w,N,a,ep); d={'rad':e_rad.max(),'in':e_in,'out':e_out}
        k=max(d,key=d.get); return k,d[k]
    while True:
        w+=dw; x=eq(w,x); k,e=emax(x,w)
        if e>=m['eb']:
            lo,hi=w-dw,w
            for _ in range(25):
                mid=(lo+hi)/2; xm=eq(mid,x)
                if emax(xm,mid)[1]>=m['eb']: hi=mid
                else: lo=mid
            x=eq(hi,x)
            return k,hi,x[2],(energy(x,hi,N,a,ep,m)-U0)/(mass*ev)
if __name__=='__main__':
    for mat in ['S','D']:
        for a in [0.1,0.2,0.3,0.4,0.5]:
            k,w,px,eta=run(mat,8,a,0.0)
            print(mat,a,k,round(w,4),'px=',round(px,4),'etaA=',round(eta,4),flush=True)
