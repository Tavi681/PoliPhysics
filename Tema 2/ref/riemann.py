"""Finite-amplitude N-thread junction (massless hub), reference solution.
Lagrangian 1D waves in each thread, sigma(e)=E0 e + b e^3 for e>0, 0 for e<=0 (slack).
Units: stresses in Pa, velocities m/s. mu0 = rho*A, so everything per unit area."""
import numpy as np
from scipy.optimize import brentq
from scipy.integrate import quad

class Mat:
    def __init__(s, rho, E0, b, eb):
        s.rho, s.E0, s.b, s.eb = rho, E0, b, eb
    def sig(s, e):  return np.where(e > 0, s.E0*e + s.b*e**3, 0.0) if np.ndim(e) else (s.E0*e + s.b*e**3 if e > 0 else 0.0)
    def c(s, e):    return np.sqrt((s.E0 + 3*s.b*e**2)/s.rho) if e > 0 else 0.0
    def Phi(s, e, ea):
        """velocity jump v_behind - v_ahead for a LEFT-going wave (sign-flip for right-going),
        state ahead ea, state behind e. Shock if e>ea (Lax admissible, convex law), fan if e<ea."""
        if e >= ea:
            return np.sqrt(max((s.sig(e)-s.sig(ea))*(e-ea), 0.0)/s.rho)
        lo = max(e, 0.0)
        return -quad(s.c, lo, ea)[0]   # fan; below 0 the thread is slack (T=0)
    def Phi_inv(s, w, ea):
        """strain e behind the wave with Phi(e,ea)=w. Returns (e, slack_flag)."""
        wmin = s.Phi(0.0, ea)
        if w <= wmin: return 0.0, True
        hi = ea + 1.0
        while s.Phi(hi, ea) < w: hi *= 2
        return brentq(lambda e: s.Phi(e, ea) - w, 0.0 if ea > 0 else -1e-12, hi, xtol=1e-14), False

def junction(m, N, e0, ep, e1):
    """Thread 0: background e0 at the hub side, incident shock/wave raises it to e1 (v1>0 outward).
    Threads j=1..N-1 at ep, at rest. Hub massless, external in-plane force constant.
    Returns dict with hub velocity V and strains at hub after interaction."""
    phi = 2*np.pi*np.arange(N)/N
    v1 = m.Phi(e1, e0)                     # incident: v_behind - v_ahead = Phi(e1;e0), ahead at rest
    def states(V):
        E0s, _ = m.Phi_inv(v1 - V, e1)     # reflected right-going wave in thread 0: v0*-v1 = -Phi(e0*;e1), v0*=V
        ej = [m.Phi_inv(-V*np.cos(p), ep) for p in phi[1:]]   # outgoing: V cos phi = -Phi(ej*;ep)
        return E0s, ej
    def f(V):
        E0s, ej = states(V)
        return (m.sig(E0s)-m.sig(e0)) + sum((m.sig(e)-m.sig(ep))*np.cos(p) for (e,_),p in zip(ej,phi[1:]))
    hi = 1.0
    while f(hi) > 0: hi *= 2
    V = brentq(f, 0.0, hi, xtol=1e-13)
    E0s, ej = states(V)
    return dict(V=V, v1=v1, e0s=E0s, ej=np.array([e for e,_ in ej]), slack=[s for _,s in ej], phi=phi)

def linear_pred(m, N, e0, ep, e1):
    """Eq.(18) with tangent impedances at the background states, applied to finite Tinc."""
    SN = 1.0 if N == 2 else N/2-1
    Tinc = m.sig(e1)-m.sig(e0)
    r = m.c(e0)/m.c(ep)
    T0 = 2*SN/(r+SN)*Tinc
    dTmax = 2/(r+SN)*Tinc
    return T0, dTmax

S = Mat(1300.0, 2e9, (1.2e9-2e9*0.3)/0.3**3, 0.30)
D = Mat(975.0, 110e9, 0.0, 0.032)
