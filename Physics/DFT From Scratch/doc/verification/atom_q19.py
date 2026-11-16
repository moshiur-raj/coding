"""Self-consistent LDA pseudo-atom with the 19-electron parameters of Appendix A
(three s and p projectors), same radial method as atom.py. Exercises every number
of Table A.1 and compares the 5d and 6s levels with the 11-electron atom."""
import numpy as np
from math import pi, sqrt, gamma
from scipy import special
from pw import pw92, HA, f_xc, PSP
def p(i, l, r, rl): return sqrt(2)*r**(l+2*(i-1))*np.exp(-r*r/(2*rl*rl))/(rl**(l+(4*i-1)/2)*sqrt(gamma(l+(4*i-1)/2)))
def sym(u):
    n = {1: 1, 3: 2, 6: 3}[len(u)]; M = np.zeros((n, n)); k = 0
    for i in range(n):
        for j in range(i, n): M[i, j] = M[j, i] = u[k]; k += 1
    return M
Q19 = dict(Z=19, rloc=0.49, C=[9.07322639, -0.34013367], r=[0.28803780, 0.35586747, 0.33386175],
    h=[sym([-5.90182040, 26.42537412, -15.50837777, -59.33938963, 40.04245923, -31.78271670]),
       sym([-7.08858659, 8.19796427, 0.0, -9.69996213, 0.0, 0.0]),
       sym([-8.82572349, 9.56535285, -10.84609065])],
    k=[None, sym([0.62807493, 1.81668773, -1.65141098, -4.89888134, 3.90795165, -2.77740200]),
       sym([0.18880114, 0.03175114, -0.03600240])])
Q19PBE = dict(Z=19, rloc=0.49, C=[8.36945885], r=[0.29008402, 0.36150576, 0.34037313],
    h=[sym([-5.98869002, 26.10502137, -15.17621948, -58.38556050, 39.18483021, -31.10199477]),
       sym([-5.87695332, 4.73144606, 2.92710526, -0.72513863, -6.92679530, 4.92291022]),
       sym([-8.80458432, 9.49191532, -10.76282032])],
    k=[None, sym([0.18370022, 2.64213942, -2.01606276, -6.48265855, 4.77087526, -3.39068639]),
       sym([0.15609346, 0.06062427, -0.06874145])])
def vxc_radial(n, r, h, xc):
    nc = np.maximum(n, 1e-14)
    if xc == 'LDA':
        rs = (3/(4*pi*nc))**(1/3); ex = -0.75*(3/pi)**(1/3)*nc**(1/3); ec, dec = pw92(rs)
        return 4/3*ex+ec-rs/3*dec
    dn = np.gradient(nc, h, edge_order=2); g = dn*dn
    hn = 1e-5*nc; hg = 1e-5*g+1e-16
    dfdn = (f_xc(nc+hn, g, 'PBE')-f_xc(nc-hn, g, 'PBE'))/(2*hn)
    dfdg = (f_xc(nc, g+hg, 'PBE')-f_xc(nc, np.maximum(g-hg, 0), 'PBE'))/(hg+np.minimum(hg, g))
    return dfdn-np.gradient(r*r*2*dfdg*dn, h, edge_order=2)/(r*r)      # v = df/dn - div(2 df/dg grad n)
def q11(xc):
    p_ = PSP[xc]; return dict(Z=11, rloc=p_['rloc'], C=[p_['C1']], r=p_['r'], h=[np.array(x) for x in p_['h']],
                             k=[None]+[np.array(x) for x in p_['k'][1:]])
def atom(ps, occ, h=0.01, Rmax=24.0, scf=True, soc=True, xc='LDA'):
    N = int(Rmax/h); r = h*np.arange(1, N+1); t = r/ps['rloc']
    Vloc = -ps['Z']/r*special.erf(r/(sqrt(2)*ps['rloc']))+np.exp(-t*t/2)*sum(c*t**(2*i) for i, c in enumerate(ps['C']))
    T = (np.diag(np.full(N, 1/h**2))-np.diag(np.full(N-1, 0.5/h**2), 1)-np.diag(np.full(N-1, 0.5/h**2), -1))
    V = np.zeros(N); n = np.zeros(N); res = {}
    for it in range(300 if scf else 1):
        nnew = np.zeros(N)
        for (l, j) in [(0, .5), (1, .5), (1, 1.5), (2, 1.5), (2, 2.5)]:
            lam = 0 if l == 0 else (l/2 if j > l else -(l+1)/2)
            hm = ps['h'][l]+(lam*ps['k'][l] if (soc and l > 0) else 0)
            P = np.array([r*p(i+1, l, r, ps['r'][l]) for i in range(len(hm))])
            w, U = np.linalg.eigh(T+np.diag(l*(l+1)/(2*r*r)+Vloc+V)+h*P.T@hm@P); res[(l, j)] = w[:2]
            for nn, f in enumerate(occ.get((l, j), [])):
                u = U[:, nn]/sqrt(h); nnew += f*u*u/(4*pi*r*r)
        if not scf: break
        d = np.abs(nnew-n).max(); n = n+0.3*(nnew-n) if it else nnew
        q = 4*pi*np.cumsum(n*r*r)*h; VH = q/r+4*pi*(np.cumsum((n*r)[::-1])[::-1]*h-n*r*h)
        V = VH+vxc_radial(n, r, h, xc)
        if d < 1e-9: break
    return res, 4*pi*(n*r*r).sum()*h, it, d
if __name__ == '__main__':
    import sys
    occ19 = {(0, .5): [2, 1], (1, .5): [2], (1, 1.5): [4], (2, 1.5): [4], (2, 2.5): [6]}
    occ11 = {(0, .5): [1], (2, 1.5): [4], (2, 2.5): [6]}
    hs = [float(x) for x in sys.argv[1:]] or [0.02]
    for xc, p19 in (('LDA', Q19), ('PBE', Q19PBE)):
        for h in hs:
            for nm, ps, occ in (("q11", q11(xc), occ11), ("q19", p19, occ19)):
                res, ne, it, d = atom(ps, occ, h=h, Rmax=16.0, xc=xc)
                s_ = res[(0, .5)]; e6s = s_[0] if nm == "q11" else s_[1]; ip = 0 if nm == "q11" else 1
                line = "%s %s h=%.3f (N_el=%.4f, %d its, dn=%.0e): 6s %.5f | 5d3/2 %.5f 5d5/2 %.5f (split %.3f eV) | 6p1/2 %.5f 6p3/2 %.5f" % (
                    xc, nm, h, ne, it+1, d, e6s, res[(2, 1.5)][0], res[(2, 2.5)][0], (res[(2, 2.5)][0]-res[(2, 1.5)][0])*HA, res[(1, .5)][ip], res[(1, 1.5)][ip])
                if nm == "q19": line += " | semicore 5s %.4f 5p1/2 %.4f 5p3/2 %.4f (split %.2f eV)" % (s_[0], res[(1, .5)][0], res[(1, 1.5)][0], (res[(1, 1.5)][0]-res[(1, .5)][0])*HA)
                print(line, flush=True)
