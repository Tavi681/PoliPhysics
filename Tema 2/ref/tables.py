import numpy as np
from scipy.optimize import brentq
from riemann import S, D, junction, Mat

def SN(N): return 1.0 if N==2 else N/2-1
def neigh(m,N,r):
    c=np.cos(r['phi'][1:]); jo=np.argmin(c)
    return r['ej'][jo], np.max(r['ej'])
print("Table F1: hub failure of thread 0, eps_p=0.1 eb")
for m,name in [(S,'S'),(D,'D')]:
    ep=0.1*m.eb
    for N in [5,6,8,12,16]:
        g=lambda e1: junction(m,N,ep,ep,e1)['e0s']-m.eb
        e1c=brentq(g,ep*1.0001,m.eb)
        r=junction(m,N,ep,ep,e1c)
        eo,emax=neigh(m,N,r)
        # linear (Eq.15, tangent Z at ep, no slack): sigma0* - sigma_p = 2SN/(1+SN)(sigma1-sigma_p)
        s=SN(N); sig1=m.sig(ep)+(m.sig(m.eb)-m.sig(ep))*(1+s)/(2*s)
        e1lin=brentq(lambda e: m.sig(e)-sig1,0,1)
        nslack=sum(r['slack'])
        print(f"{name} N={N:2d}: e1c/eb={e1c/m.eb:.3f} (lin {e1lin/m.eb:.3f}); max neighbour strain/eb={emax/m.eb:.3f}; slack threads={nslack}; V={r['V']:.2f} m/s")
print("\nTable F2: slack threshold of nearest same-side neighbour (j=1)")
for m,name in [(S,'S'),(D,'D')]:
    ep=0.1*m.eb
    for N in [5,6,8,12,16]:
        g=lambda e1: junction(m,N,ep,ep,e1)['ej'][0]-1e-9
        try: e1s=brentq(g,ep*1.0001,1.5*m.eb)
        except ValueError: e1s=np.nan
        s=SN(N); cph=np.cos(2*np.pi/N)
        # linear: |dT_1| = 2cos/(1+SN) Tinc = sigma(ep)
        Tinc=m.sig(ep)*(1+s)/(2*cph)
        e1lin=brentq(lambda e: m.sig(e)-m.sig(ep)-Tinc,0,2)
        print(f"{name} N={N:2d}: e1s/eb={e1s/m.eb:.3f} (lin {e1lin/m.eb:.3f})")
print("\nTable F3: ratios vs incident amplitude, S, N=8 and 16")
for N in [8,16]:
  for f in [0.12,0.2,0.3,0.4,0.6,0.8]:
    m=S; ep=0.1*m.eb; e1=f*m.eb; r=junction(m,N,ep,ep,e1); Tinc=m.sig(e1)-m.sig(ep)
    eo,emax=neigh(m,N,r)
    print(f"N={N} e1/eb={f}: T0/Tinc={(m.sig(r['e0s'])-m.sig(ep))/Tinc:.3f} dTopp/Tinc={(m.sig(eo)-m.sig(ep))/Tinc:.3f} e0*/eb={r['e0s']/m.eb:.3f} eopp/eb={eo/m.eb:.3f} slack={sum(r['slack'])}")
print("\nLimit check vs Eq.(18): S, N=8, e0=0.5eb, ep=0.1eb, tiny jump")
m=S;ep=0.1*m.eb;e0=0.5*m.eb;e1=e0+1e-6
r=junction(m,8,e0,ep,e1);Tinc=m.sig(e1)-m.sig(e0)
rr=m.c(e0)/m.c(ep); print("T0/Tinc",(m.sig(r['e0s'])-m.sig(e0))/Tinc,"Eq18",2*3/(rr+3), "r",rr)
eo,_=neigh(m,8,r);print("dTopp/Tinc",(m.sig(eo)-m.sig(ep))/Tinc,"Eq18",2/(rr+3))
