"""Reference spinor plane-wave DFT for FCC Au, written to follow the document's
equations and pseudocode literally (verification tool, not optimised)."""
import numpy as np, os, sys, time
from math import pi, sqrt, gamma
from scipy import special, linalg

BOHR = 0.529177210903; HA = 27.211386
PSP = {
 'LDA': dict(Z=11, rloc=0.59, C1=11.60442790, r=[0.52117975, 0.63061306, 0.44070643],
   h=[[[2.53861359,-1.04613664],[-1.04613664,2.70111320]],
      [[0.39485270,-0.86959213],[-0.86959213,2.05783056]],
      [[-4.71906966,0.72777081],[0.72777081,-1.65042907]]],
   k=[None,[[-0.96005489,-0.54790119],[-0.54790119,1.29657086]],
           [[0.14848354,0.07473921],[0.07473921,-0.16949260]]]),
 'PBE': dict(Z=11, rloc=0.59, C1=10.51717881, r=[0.54384733, 0.60917298, 0.43730922],
   h=[[[5.78681103,-2.25067966],[-2.25067966,2.90561495]],
      [[4.28404873,-4.07128520],[-4.07128520,4.81720962]],
      [[-7.40328124,3.01019307],[3.01019307,-3.41323811]]],
   k=[None,[[1.21911104,-2.38187511],[-2.38187511,2.81827264]],
           [[-0.12113041,0.26246301],[0.26246301,-0.29760507]]]),
}

# ---------- angular parts ----------
def real_ylm(l, q):
    n = np.linalg.norm(q, axis=-1); n = np.where(n < 1e-12, 1.0, n)
    x, y, z = (q / n[..., None]).T
    if l == 0: return [np.full(x.shape, sqrt(1/(4*pi)))]
    if l == 1: c = sqrt(3/(4*pi)); return [c*y, c*z, c*x]
    c = sqrt(15/(4*pi))
    return [c*x*y, c*y*z, sqrt(5/(16*pi))*(3*z*z-1), c*x*z, sqrt(15/(16*pi))*(x*x-y*y)]

def ptilde(i, l, q, rl):
    x = q*rl; E = pi**0.25*np.exp(-x*x/2)
    T = {(1,0): sqrt(2*rl**3), (2,0): 2*sqrt(2*rl**3/15)*(3-x*x),
         (1,1): 2*sqrt(rl**5/3)*q, (2,1): 4*sqrt(rl**5/105)*q*(5-x*x),
         (1,2): 2*sqrt(2*rl**7/15)*q*q, (2,2): 4/3*sqrt(2*rl**7/105)*q*q*(7-x*x)}
    return E*T[(i, l)]

def LS_real(l):
    m = np.arange(-l, l+1); d = 2*l+1
    Lz = np.diag(m).astype(complex); Lp = np.zeros((d, d), complex)
    for a in range(d-1): Lp[a+1, a] = sqrt(l*(l+1)-m[a]*(m[a]+1))
    U = np.zeros((d, d), complex)
    for mu in range(-l, l+1):
        c = mu+l
        if mu > 0: U[-mu+l, c] = 1/sqrt(2); U[mu+l, c] = (-1)**mu/sqrt(2)
        elif mu == 0: U[l, c] = 1
        else:
            a = -mu; U[-a+l, c] = 1j/sqrt(2); U[a+l, c] = -1j*(-1)**a/sqrt(2)
    Lz, Lp = U.conj().T@Lz@U, U.conj().T@Lp@U
    Lm = Lp.conj().T
    return 0.5*np.block([[Lz, Lm], [Lp, -Lz]])     # index = s*(2l+1)+m

# ---------- XC ----------
def pw92(rs):
    A,a1,b1,b2,b3,b4 = 0.031091,0.21370,7.5957,3.5876,1.6382,0.49294
    q = 2*A*(b1*np.sqrt(rs)+b2*rs+b3*rs**1.5+b4*rs**2)
    dq = 2*A*(0.5*b1/np.sqrt(rs)+b2+1.5*b3*np.sqrt(rs)+2*b4*rs)
    lg = np.log1p(1/q)
    e = -2*A*(1+a1*rs)*lg
    de = -2*A*a1*lg + 2*A*(1+a1*rs)*dq/(q*q+q)
    return e, de

def f_xc(n, g, xc):
    """energy density per volume f(n, gamma=|grad n|^2), unpolarised."""
    n = np.maximum(n, 1e-12)
    rs = (3/(4*pi*n))**(1/3)
    ex = -0.75*(3/pi)**(1/3)*n**(1/3); ec, _ = pw92(rs)
    if xc == 'LDA': return n*(ex+ec)
    kF = (3*pi*pi*n)**(1/3); s2 = g/(4*kF*kF*n*n)
    kap, mu, beta, gam = 0.804, 0.2195149727645171, 0.06672455060314922, (1-np.log(2))/pi**2
    Fx = 1+kap-kap/(1+mu*s2/kap)
    ks2 = 4*kF/pi; t2 = g/(4*ks2*n*n)
    Aa = beta/gam/np.expm1(-ec/gam)
    H = gam*np.log1p(beta/gam*t2*(1+Aa*t2)/(1+Aa*t2+Aa*Aa*t2*t2))
    return n*(ex*Fx+ec+H)

def xc_potential(n, Gc, xc):
    """returns Exc density f, vxc on grid. Gc: cartesian G on FFT box [N1,N2,N3,3]."""
    n = np.maximum(n, 1e-12)
    if xc == 'LDA':
        rs = (3/(4*pi*n))**(1/3)
        ex = -0.75*(3/pi)**(1/3)*n**(1/3); ec, dec = pw92(rs)
        return n*(ex+ec), 4/3*ex+ec-rs/3*dec
    nG = np.fft.fftn(n)
    grad = np.stack([np.fft.ifftn(1j*Gc[..., i]*nG).real for i in range(3)], -1)
    g = (grad**2).sum(-1)
    f = f_xc(n, g, xc)
    hn = 1e-5*n; hg = 1e-5*g+1e-14
    dfdn = (f_xc(n+hn, g, xc)-f_xc(n-hn, g, xc))/(2*hn)
    dfdg = (f_xc(n, g+hg, xc)-f_xc(n, np.maximum(g-hg, 0), xc))/(hg+np.minimum(hg, g))
    div = sum(np.fft.ifftn(1j*Gc[..., i]*np.fft.fftn(2*dfdg*grad[..., i])).real for i in range(3))
    return f, dfdn-div

# ---------- smearing ----------
def occ(x, kind):
    x = np.clip(x, -40, 40)
    if kind == 'fd': return 1/(np.exp(x)+1)
    f = 0.5*special.erfc(x)
    if kind == 'mp1': f = f - x/(2*sqrt(pi))*np.exp(-x*x)
    return f
def entropy(x, kind):
    x = np.clip(x, -40, 40)
    if kind == 'fd':
        f = 1/(np.exp(x)+1); f = np.clip(f, 1e-300, 1-1e-16); return -(f*np.log(f)+(1-f)*np.log1p(-f))
    if kind == 'gauss': return np.exp(-x*x)/(2*sqrt(pi))
    return (1-2*x*x)*np.exp(-x*x)/(4*sqrt(pi))
def fermi(eps, w, Ne, sigma, kind):
    lo, hi = eps.min()-10*sigma, eps.max()+10*sigma
    for _ in range(100):
        mu = 0.5*(lo+hi)
        if (w[:, None]*occ((eps-mu)/sigma, kind)).sum() > Ne: hi = mu
        else: lo = mu
    return mu

# ---------- Ewald ----------
def ewald(A, Z, eta=0.5, nmax=10):
    Om = abs(np.linalg.det(A)); B = 2*pi*np.linalg.inv(A).T
    r = np.arange(-nmax, nmax+1); I = np.array(np.meshgrid(r, r, r, indexing='ij')).reshape(3, -1).T
    G2 = ((I@B)**2).sum(1); m = G2 > 1e-12
    d = np.linalg.norm(I@A, axis=1); md = d > 1e-10
    return (2*pi/Om*np.sum(np.exp(-G2[m]/(4*eta))/G2[m])*Z*Z + 0.5*Z*Z*np.sum(special.erfc(sqrt(eta)*d[md])/d[md])
            - sqrt(eta/pi)*Z*Z - pi*Z*Z/(2*eta*Om))

# ---------- the crystal ----------
class Crystal:
    def __init__(self, a_bohr, ecut, xc='LDA', soc=True, spinor=True, ngrid=None):
        self.a, self.ecut, self.xc, self.soc, self.spinor = a_bohr, ecut, xc, soc, spinor
        self.ns = 2 if spinor else 1
        self.psp = PSP[xc]
        self.A = a_bohr/2*np.array([[0, 1, 1], [1, 0, 1], [1, 1, 0.]])
        self.B = 2*pi*np.linalg.inv(self.A).T
        self.Om = abs(np.linalg.det(self.A))
        gmax = sqrt(2*ecut)
        nmin = int(np.ceil(2/pi*gmax*np.linalg.norm(self.A[0])+1))
        good = [n for n in range(8, 200) if self._smooth(n)]
        self.N = ngrid or next(n for n in good if n >= nmin)
        N = self.N
        fr = np.fft.fftfreq(N, 1/N).astype(int)
        self.mbox = np.stack(np.meshgrid(fr, fr, fr, indexing='ij'), -1)       # integer G on box
        self.Gc = self.mbox@self.B
        self.G2 = (self.Gc**2).sum(-1)
        p = self.psp; x2 = self.G2*p['rloc']**2
        with np.errstate(divide='ignore', invalid='ignore'):
            v = (-4*pi*p['Z']/self.G2*np.exp(-x2/2)+(2*pi)**1.5*p['rloc']**3*np.exp(-x2/2)*p['C1'])/self.Om
        self.alpha = 2*pi*p['Z']*p['rloc']**2+(2*pi)**1.5*p['rloc']**3*p['C1']
        v[0, 0, 0] = self.alpha/self.Om
        self.VlocG = v
        self.Vloc_r = (np.fft.ifftn(v)*N**3).real
        self.Eew = ewald(self.A, p['Z'])
        # coupling matrix D over (proj, spin)
        self.projlist = [(l, i, m) for l in range(3) for i in (1, 2) for m in range(2*l+1)]
        npj = len(self.projlist); ns = self.ns
        D = np.zeros((ns*npj, ns*npj), complex)
        for a, (l, i, m) in enumerate(self.projlist):
            for b, (l2, j, m2) in enumerate(self.projlist):
                if l != l2: continue
                for s in range(ns):
                    for s2 in range(ns):
                        val = p['h'][l][i-1][j-1]*(m == m2)*(s == s2)
                        if soc and spinor and l > 0:
                            val += p['k'][l][i-1][j-1]*LS_real(l)[s*(2*l+1)+m, s2*(2*l+1)+m2]
                        D[s*npj+a, s2*npj+b] = val
        self.D = D; self.npj = npj
        assert np.allclose(D, D.conj().T)

    @staticmethod
    def _smooth(n):
        for p in (2, 3, 5, 7):
            while n % p == 0: n //= p
        return n == 1

    def basis(self, kf):
        """plane-wave sphere at fractional k. returns dict."""
        kc = np.asarray(kf)@self.B
        nmax = int(np.ceil(sqrt(2*self.ecut)*np.linalg.norm(self.A, axis=1).max()/(2*pi)))+2
        r = np.arange(-nmax, nmax+1)
        M = np.array(np.meshgrid(r, r, r, indexing='ij')).reshape(3, -1).T
        q = kc+M@self.B; kin = 0.5*(q**2).sum(1)
        keep = kin <= self.ecut
        M, q, kin = M[keep], q[keep], kin[keep]
        N = self.N
        idx = np.ravel_multi_index((M % N).T, (N, N, N))
        qn = np.linalg.norm(q, axis=1)
        Bt = np.zeros((self.npj, len(M)), complex)
        for a, (l, i, m) in enumerate(self.projlist):
            Bt[a] = 4*pi/sqrt(self.Om)*(-1j)**l*ptilde(i, l, qn, self.psp['r'][l])*real_ylm(l, q)[m]
        return dict(k=np.asarray(kf, float), M=M, q=q, kin=kin, idx=idx, Bt=Bt, ng=len(M))

    def applyH(self, bs, C, Vr):
        """C: [nb, ns, ng] -> H C.  Vr: local effective potential on the grid."""
        nb, ns, ng = C.shape; N = self.N
        out = bs['kin']*C
        box = np.zeros((nb, ns, N**3), complex); box[:, :, bs['idx']] = C
        u = np.fft.ifftn(box.reshape(nb, ns, N, N, N), axes=(2, 3, 4))
        out += np.fft.fftn(u*Vr, axes=(2, 3, 4)).reshape(nb, ns, -1)[:, :, bs['idx']]
        P = np.einsum('pg,nsg->nsp', bs['Bt'].conj(), C).reshape(nb, ns*self.npj)
        DP = (P@self.D.T).reshape(nb, ns, self.npj)
        out += np.einsum('pg,nsp->nsg', bs['Bt'], DP)
        return out

    def Hdense(self, bs, Vr):
        ng = bs['ng']; ns = self.ns
        I = np.eye(ns*ng, dtype=complex).reshape(ns*ng, ns, ng)
        H = self.applyH(bs, I, Vr).reshape(ns*ng, ns*ng).T
        return 0.5*(H+H.conj().T)

    def density_k(self, bs, C, f):
        nb, ns, ng = C.shape; N = self.N
        box = np.zeros((nb, ns, N**3), complex); box[:, :, bs['idx']] = C
        u = np.fft.ifftn(box.reshape(nb, ns, N, N, N), axes=(2, 3, 4))*N**3/sqrt(self.Om)
        return np.einsum('n,nsxyz->xyz', f, np.abs(u)**2)

    def potential(self, n):
        N = self.N
        nG = np.fft.fftn(n)/N**3
        with np.errstate(divide='ignore', invalid='ignore'):
            vH = 4*pi*nG/self.G2
        vH[0, 0, 0] = 0
        EH = 0.5*self.Om*np.sum(4*pi*np.abs(nG[self.G2 > 0])**2/self.G2[self.G2 > 0])
        fxc, vxc = xc_potential(n, self.Gc, self.xc)
        dv = self.Om/N**3
        E = dict(EH=EH, Exc=fxc.sum()*dv, Evxc=(vxc*n).sum()*dv, Eloc=(self.Vloc_r*n).sum()*dv)
        return self.Vloc_r+(np.fft.ifftn(vH)*N**3).real+vxc, E

# ---------- eigensolver: the document's Algorithms (Rayleigh-Ritz + LOBPCG) ----------
def rayleigh_ritz(S, HS, nb):
    M = S.conj().T@S; Amat = S.conj().T@HS
    L = np.linalg.cholesky(M)
    At = linalg.solve_triangular(L, linalg.solve_triangular(L, Amat, lower=True).conj().T, lower=True).conj().T
    th, Yt = np.linalg.eigh(0.5*(At+At.conj().T))
    Y = linalg.solve_triangular(L.conj().T, Yt[:, :nb], lower=False)
    return th[:nb], Y

def teter(x):
    num = 27+18*x+12*x*x+8*x**3
    return num/(num+16*x**4)

def lobpcg(cr, bs, Vr, X, tmax, tol=0.0, nconv=None, stats=None):
    """X: [nb, ns, ng] rows are bands. Literal transcription of the document's algorithm
    (column convention -> matrices [dim, nb])."""
    nb, ns, ng = X.shape; dim = ns*ng
    H = lambda Y: cr.applyH(bs, Y.T.reshape(-1, ns, ng), Vr).reshape(-1, dim).T
    kin = np.tile(bs['kin'], ns)[:, None]
    X = X.reshape(nb, dim).T.copy()
    HX = H(X)
    lam, Y = rayleigh_ritz(X, HX, nb); X = X@Y; HX = HX@Y
    P = HP = None; restarts = 0
    for t in range(tmax):
        R = HX-X*lam
        rn = np.linalg.norm(R, axis=0)
        if tol and rn[:nconv].max() < tol: break
        ek = (kin*np.abs(X)**2).sum(0)
        W = teter(kin/ek)*R
        W = W-X@(X.conj().T@W); W /= np.linalg.norm(W, axis=0)
        HW = H(W)
        try:
            if P is None: raise np.linalg.LinAlgError
            S = np.hstack([X, W, P]); HS = np.hstack([HX, HW, HP])
            lam, Y = rayleigh_ritz(S, HS, nb)
            YX, YW, YP = Y[:nb], Y[nb:2*nb], Y[2*nb:]
            Pn = W@YW+P@YP; HPn = HW@YW+HP@YP
        except np.linalg.LinAlgError:
            if P is not None: restarts += 1
            S = np.hstack([X, W]); HS = np.hstack([HX, HW])
            lam, Y = rayleigh_ritz(S, HS, nb)
            YX, YW = Y[:nb], Y[nb:]
            Pn = W@YW; HPn = HW@YW
        P, HP = Pn, HPn
        X = X@YX+P; HX = HX@YX+HP
    if stats is not None: stats.update(iters=t+1, restarts=restarts, resid=np.linalg.norm(HX-X*lam, axis=0))
    return lam, X.T.reshape(nb, ns, ng)

def mp_mesh(n, gamma_centered=False):
    if gamma_centered: f = np.arange(n)/n
    else: f = (2*np.arange(1, n+1)-n-1)/(2*n)
    K = np.array(np.meshgrid(f, f, f, indexing='ij')).reshape(3, -1).T
    return K

def tr_reduce(K):
    """keep one of each (k,-k) pair; weights sum to 1."""
    seen = {}; ks = []; ws = []
    for k in K:
        key = tuple(np.round((k+0.5) % 1.0, 8)); mkey = tuple(np.round((-k+0.5) % 1.0, 8))
        if mkey in seen: ws[seen[mkey]] += 1; continue
        if key in seen: ws[seen[key]] += 1; continue
        seen[key] = len(ks); ks.append(k); ws.append(1)
    return np.array(ks), np.array(ws, float)/len(K)

# ---------- robust variant: Rayleigh-Ritz by canonical orthogonalisation + soft locking ----------
def rayleigh_ritz_safe(S, HS, nb, tau=1e-10):
    d = np.linalg.norm(S, axis=0); d = np.where(d > 0, d, 1.0)
    S = S/d; HS = HS/d
    M = S.conj().T@S; Amat = S.conj().T@HS
    sig, V = np.linalg.eigh(0.5*(M+M.conj().T))
    keep = sig > tau*sig.max()
    T = V*np.where(keep, 1/np.sqrt(np.where(keep, sig, 1.0)), 0.0)
    At = T.conj().T@Amat@T
    At = 0.5*(At+At.conj().T)+np.diag(np.where(keep, 0.0, 1e6))
    th, Yt = np.linalg.eigh(At)
    return th[:nb], (T@Yt[:, :nb])/d[:, None]

def lobpcg_safe(cr, bs, Vr, X, tmax, tol=0.0, nconv=None, stats=None):
    nb, ns, ng = X.shape; dim = ns*ng
    H = lambda Y: cr.applyH(bs, Y.T.reshape(-1, ns, ng), Vr).reshape(-1, dim).T
    kin = np.tile(bs['kin'], ns)[:, None]
    X = X.reshape(nb, dim).T.copy()
    HX = H(X)
    lam, Y = rayleigh_ritz_safe(X, HX, nb); X = X@Y; HX = HX@Y
    P = np.zeros_like(X); HP = np.zeros_like(X)
    for t in range(tmax):
        R = HX-X*lam
        rn = np.linalg.norm(R, axis=0)
        if tol and rn[:nconv].max() < tol: break
        active = rn > (tol if tol else 0.0)*0.1
        ek = (kin*np.abs(X)**2).sum(0)
        W = teter(kin/ek)*R*active
        W = W-X@(X.conj().T@W); W = W-X@(X.conj().T@W)
        HW = H(W)
        S = np.hstack([X, W, P]); HS = np.hstack([HX, HW, HP])
        lam, Y = rayleigh_ritz_safe(S, HS, nb)
        YX, YW, YP = Y[:nb], Y[nb:2*nb], Y[2*nb:]
        P = W@YW+P@YP; HP = HW@YW+HP@YP
        X = X@YX+P; HX = HX@YX+HP
    if stats is not None: stats.update(iters=t+1, restarts=0, resid=np.linalg.norm(HX-X*lam, axis=0))
    return lam, X.T.reshape(nb, ns, ng)

# ---------- robust LOBPCG: orthonormal basis [X, W, P] ----------
def orthonormalize(V, tau, HV=None):
    """canonical orthonormalisation of the columns of V with dropping; applies the same
    column transformation to HV if given. Dropped directions become zero columns."""
    d = np.linalg.norm(V, axis=0); d = np.where(d > 0, d, 1.0)
    G = (V/d).conj().T@(V/d)
    sig, U = np.linalg.eigh(0.5*(G+G.conj().T))
    keep = sig > tau
    T = (U*np.where(keep, 1/np.sqrt(np.where(keep, sig, 1.0)), 0.0))/d[:, None]
    return (V@T, None if HV is None else HV@T)

def lobpcg_orth(cr, bs, Vr, X, tmax, tol=0.0, nconv=None, stats=None, tauW=1e-10, tauP=1e-6, hist=None):
    nb, ns, ng = X.shape; dim = ns*ng
    H = lambda Y: cr.applyH(bs, Y.T.reshape(-1, ns, ng), Vr).reshape(-1, dim).T
    kin = np.tile(bs['kin'], ns)[:, None]
    X = X.reshape(nb, dim).T.copy()
    X, _ = orthonormalize(X, 1e-12); X, _ = orthonormalize(X, 1e-12)
    HX = H(X)
    A = X.conj().T@HX; lam, Y = np.linalg.eigh(0.5*(A+A.conj().T)); X = X@Y; HX = HX@Y
    P = np.zeros_like(X); HP = np.zeros_like(X)
    for t in range(tmax):
        R = HX-X*lam
        rn = np.linalg.norm(R, axis=0)
        if hist is not None: hist.append(rn.copy())
        if tol and rn[:nconv].max() < tol: break
        active = rn > (0.1*tol if tol else 0.0)
        ek = (kin*np.abs(X)**2).sum(0)
        W = teter(kin/ek)*R*active
        Q = np.hstack([X, P])
        for _ in range(2):
            W = W-Q@(Q.conj().T@W)
            W, _ = orthonormalize(W, tauW)
        HW = H(W)
        S = np.hstack([X, W, P]); HS = np.hstack([HX, HW, HP])
        A = S.conj().T@HS; A = 0.5*(A+A.conj().T)
        zero = np.linalg.norm(S, axis=0) < 0.5
        A = A+np.diag(np.where(zero, 1e6, 0.0))
        th, Y = np.linalg.eigh(A); lam = th[:nb]; Y = Y[:, :nb]
        YX, YW, YP = Y[:nb], Y[nb:2*nb], Y[2*nb:]
        P = W@YW+P@YP; HP = HW@YW+HP@YP
        X = X@YX+P; HX = HX@YX+HP
        for _ in range(2):      # two passes: the first can leave P orthonormal only to ~1e-10
            c = X.conj().T@P; P = P-X@c; HP = HP-HX@c
            P, HP = orthonormalize(P, tauP, HP)
    if stats is not None: stats.update(iters=t+1, restarts=0, resid=np.linalg.norm(HX-X*lam, axis=0))
    return lam, X.T.reshape(nb, ns, ng)
