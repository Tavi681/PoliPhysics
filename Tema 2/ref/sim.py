"""Direct check: N discretized threads (mass-spring chains) joined at a massless hub.
Thread 0's far end is pulled outward at constant speed v1 -> incident shock towards hub.
Small hub displacement: end of thread j follows X_h cos(phi_j)."""
import numpy as np, sys
from scipy.optimize import brentq
from riemann import S, D, junction

def run(m, N, ep, e1, L=1.0, n=1500, visc=0.6):
    h = L/n; mu = m.rho*h
    phi = 2*np.pi*np.arange(N)/N; cph = np.cos(phi)
    v1 = m.Phi(e1, ep)
    u = np.zeros((N, n+1)); v = np.zeros((N, n+1))   # node 0 = hub end, node n = far end
    cmax = m.c(max(e1, 1.3*m.eb)); dt = 0.25*h/cmax
    def sig(e): return np.where(e > 0, m.E0*e + m.b*e**3, 0.0)
    Xh = 0.0; t = 0.0
    tarr = L/m.c(ep) if e1 == ep else L/np.sqrt((m.sig(e1)-m.sig(ep))/(m.rho*(e1-ep)))
    tend = tarr + 0.6*L/cmax
    rec = []
    while t < tend:
        # hub: massless balance sum_j sig(eps_j,first) cos phi_j = 0 (increments; background balanced)
        def F(X):
            e = ep + (u[:,1] - X*cph)/h
            return np.sum(sig(e)*cph)
        a, b = Xh-1e-3, Xh+1e-3
        while F(a) < 0: a -= 1e-3
        while F(b) > 0: b += 1e-3
        Xold = Xh
        Xh = brentq(F, a, b, xtol=1e-15)
        u[:,0] = Xh*cph
        v[:,0] = (Xh-Xold)/dt*cph
        e = ep + np.diff(u, axis=1)/h
        de = np.diff(v, axis=1)/h
        s = sig(e) + visc*m.rho*m.c(max(e1,ep))*h*de*(e > 0)   # small artificial viscosity
        f = np.zeros_like(u); f[:,1:-1] = s[:,1:] - s[:,:-1]
        v[:,1:-1] += dt*f[:,1:-1]/mu
        v[0,-1] = v1; v[1:,-1] = 0.0                 # thread 0 pulled, others anchored
        u[:,1:] += dt*v[:,1:]
        t += dt
        if t > tarr + 0.3*L/cmax:
            rec.append((ep + (u[:,1+5]-u[:,5])/h))    # strain a few cells from hub
    rec = np.array(rec).mean(axis=0)
    return rec

if __name__ == "__main__":
    for m, name in [(S,'S'), (D,'D')]:
        ep = 0.1*m.eb
        for N in [3, 8, 16]:
            for f in [0.2, 0.5, 0.8]:
                e1 = f*m.eb
                sim = run(m, N, ep, e1)
                th = junction(m, N, ep, ep, e1)
                jo = 1+np.argmin(np.cos(th['phi'][1:])); js = 1  # opposite / nearest same-side
                print(f"{name} N={N:2d} e1/eb={f}: e0* sim {sim[0]/m.eb:.3f} th {th['e0s']/m.eb:.3f} | "
                      f"e_opp sim {sim[jo]/m.eb:.3f} th {th['ej'][jo-1]/m.eb:.3f} | e_j1 sim {sim[1]/m.eb:.3f} th {th['ej'][0]/m.eb:.3f}", flush=True)
