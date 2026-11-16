"""Independent JAX implementation written ONLY from the text of the document
(equations, tables and Algorithms 1-7), to test that the document is
self-contained.  It shares no code with pw.py / scf.py.

Parameters are typed from Table 6.1 of the document; XC potentials come from
jax.grad of the energy density; all k-points are padded to a common size and
run through vmap, as Chapter 10 prescribes.

Usage:  python3 jax_from_text.py small      (three Ecut=20 regression cases)
        python3 jax_from_text.py caseA      (Ecut=25, shifted 4^3 mesh)
        python3 jax_from_text.py caseB      (Ecut=40, Gamma-centred 8^3 mesh; slow on CPU)
        python3 jax_from_text.py stage1     (empty lattice, cosine potential, parities)
        python3 jax_from_text.py jellium
Reading this file before writing your own code spoils the exercise.
"""
import sys, time
from math import pi, sqrt, gamma as Gamma
import numpy as np
import jax, jax.numpy as jnp
from jax.scipy.special import erfc
jax.config.update("jax_enable_x64", True)

BOHR = 0.529177211
HA_EV = 27.2114

# ---------------------------------------------------------------- Table 6.1
PSP = {
 'LDA': dict(Z=11, rloc=0.59, C1=11.60442790, r=[0.52117975, 0.63061306, 0.44070643],
             h=[[2.53861359, -1.04613664, 2.70111320],
                [0.39485270, -0.86959213, 2.05783056],
                [-4.71906966, 0.72777081, -1.65042907]],
             k=[[0, 0, 0], [-0.96005489, -0.54790119, 1.29657086],
                [0.14848354, 0.07473921, -0.16949260]]),
 'PBE': dict(Z=11, rloc=0.59, C1=10.51717881, r=[0.54384733, 0.60917298, 0.43730922],
             h=[[5.78681103, -2.25067966, 2.90561495],
                [4.28404873, -4.07128520, 4.81720962],
                [-7.40328124, 3.01019307, -3.41323811]],
             k=[[0, 0, 0], [1.21911104, -2.38187511, 2.81827264],
                [-0.12113041, 0.26246301, -0.29760507]]),
}
def sym2(v): return np.array([[v[0], v[1]], [v[1], v[2]]])

# ---------------------------------------------------------------- Chapter 4
def eps_c_pw92(rs):
    A, a1, b1, b2, b3, b4 = 0.031091, 0.21370, 7.5957, 3.5876, 1.6382, 0.49294
    Q = 2*A*(b1*jnp.sqrt(rs) + b2*rs + b3*rs**1.5 + b4*rs**2)
    return -2*A*(1 + a1*rs)*jnp.log1p(1/Q)

def f_xc(n, g2, kind):
    """XC energy per volume, unpolarised; g2 = |grad n|^2."""
    n = jnp.maximum(n, 1e-12)
    rs = (3/(4*pi*n))**(1/3)
    ex = -0.75*(3/pi)**(1/3)*n**(1/3)
    ec = eps_c_pw92(rs)
    if kind == 'LDA':
        return n*(ex + ec)
    if kind == 'none':
        return 0.0*n
    kF = (3*pi**2*n)**(1/3)
    beta = 0.06672455060314922; kappa, mu, gam = 0.804, pi**2*beta/3, (1 - np.log(2))/pi**2
    s2 = g2/(2*kF*n)**2
    Fx = 1 + kappa - kappa/(1 + mu*s2/kappa)
    ks = jnp.sqrt(4*kF/pi)
    t2 = g2/(2*ks*n)**2
    A = (beta/gam)/jnp.expm1(-ec/gam)
    H = gam*jnp.log1p(beta/gam*t2*(1 + A*t2)/(1 + A*t2 + A**2*t2**2))
    return n*(ex*Fx + ec + H)

# ---------------------------------------------------------------- Chapter 5: smearing
def occ_fn(x, kind):
    x = jnp.clip(x, -40, 40)
    if kind == 'fd':
        return 1/(jnp.exp(x) + 1)
    f = 0.5*erfc(x)
    if kind == 'mp1':
        f = f - x/(2*sqrt(pi))*jnp.exp(-x*x)
    return f

def ent_fn(x, kind):
    x = jnp.clip(x, -40, 40)
    if kind == 'fd':
        f = jnp.clip(1/(jnp.exp(x) + 1), 1e-300, 1 - 1e-16)
        return -(f*jnp.log(f) + (1 - f)*jnp.log1p(-f))
    if kind == 'gauss':
        return jnp.exp(-x*x)/(2*sqrt(pi))
    return (1 - 2*x*x)*jnp.exp(-x*x)/(4*sqrt(pi))

# ---------------------------------------------------------------- Chapter 6/7
def real_ylm(l, q):
    qn = np.linalg.norm(q, axis=-1)
    u = q/np.maximum(qn, 1e-12)[..., None]
    x, y, z = u[..., 0], u[..., 1], u[..., 2]
    if l == 0:
        return [np.full(x.shape, sqrt(1/(4*pi)))]
    if l == 1:
        c = sqrt(3/(4*pi)); return [c*y, c*z, c*x]
    c = sqrt(15/(4*pi))
    return [c*x*y, c*y*z, sqrt(5/(16*pi))*(3*z*z - 1), c*x*z, sqrt(15/(16*pi))*(x*x - y*y)]

def ptilde(i, l, q, rl):
    x = q*rl; E = pi**0.25*np.exp(-x*x/2)
    if (i, l) == (1, 0): return E*sqrt(2*rl**3)
    if (i, l) == (2, 0): return E*2*sqrt(2*rl**3/15)*(3 - x*x)
    if (i, l) == (1, 1): return E*2*sqrt(rl**5/3)*q
    if (i, l) == (2, 1): return E*4*sqrt(rl**5/105)*q*(5 - x*x)
    if (i, l) == (1, 2): return E*2*sqrt(2*rl**7/15)*q*q
    if (i, l) == (2, 2): return E*4/3*sqrt(2*rl**7/105)*q*q*(7 - x*x)

def LS_real(l):
    d = 2*l + 1; m = np.arange(-l, l + 1)
    Lz = np.diag(m).astype(complex); Lp = np.zeros((d, d), complex)
    for a in range(d - 1):                      # L+|l m> = sqrt(l(l+1)-m(m+1)) |l m+1>
        Lp[a + 1, a] = sqrt(l*(l + 1) - m[a]*(m[a] + 1))
    Lm = Lp.conj().T
    U = np.zeros((d, d), complex)               # column mu: real harmonic in complex ones
    for mu in range(-l, l + 1):
        c = mu + l
        if mu > 0:
            U[-mu + l, c] = 1/sqrt(2); U[mu + l, c] = (-1)**mu/sqrt(2)
        elif mu == 0:
            U[l, c] = 1
        else:
            a = -mu
            U[-a + l, c] = 1j/sqrt(2); U[a + l, c] = -1j*(-1)**a/sqrt(2)
    assert np.allclose(U.conj().T@U, np.eye(d))
    Lz, Lp, Lm = [U.conj().T@X@U for X in (Lz, Lp, Lm)]
    LS = 0.5*np.block([[Lz, Lm], [Lp, -Lz]])
    return LS

# ---------------------------------------------------------------- Chapter 8: Ewald
def ewald(A, Z, eta=0.5):
    from scipy.special import erfc as serfc
    Om = abs(np.linalg.det(A)); B = 2*pi*np.linalg.inv(A).T
    r = np.arange(-12, 13); I = np.array(np.meshgrid(r, r, r, indexing='ij')).reshape(3, -1).T
    G2 = ((I@B)**2).sum(1); mg = (G2 > 1e-12) & (G2 <= (12*sqrt(eta))**2)
    d = np.linalg.norm(I@A, axis=1); md = (d > 1e-10) & (d <= 6/sqrt(eta))
    return (2*pi/Om*np.sum(np.exp(-G2[mg]/(4*eta))/G2[mg])*Z*Z
            + 0.5*Z*Z*np.sum(serfc(sqrt(eta)*d[md])/d[md])
            - sqrt(eta/pi)*Z*Z - pi*Z*Z/(2*eta*Om))

# ---------------------------------------------------------------- setup
def smooth(n):
    for p in (2, 3, 5, 7):
        while n % p == 0: n //= p
    return n == 1

class System:
    def __init__(self, a, ecut, xc='LDA', soc=True, ions=True, ngrid=None, nelec=None):
        self.a, self.ecut, self.xc, self.soc, self.ions = a, ecut, xc, soc, ions
        ps = self.ps = PSP['PBE' if xc == 'PBE' else 'LDA']
        self.Ne = ps['Z'] if nelec is None else nelec
        A = self.A = a/2*np.array([[0, 1, 1], [1, 0, 1], [1, 1, 0.]])
        B = self.B = 2*pi*np.linalg.inv(A).T
        Om = self.Om = abs(np.linalg.det(A))
        gmax = sqrt(2*ecut)
        nmin = int(np.ceil(2/pi*gmax*np.linalg.norm(A[0]) + 1))
        N = self.N = ngrid or next(n for n in range(nmin, 400) if smooth(n))
        fr = np.fft.fftfreq(N, 1/N).astype(int)
        mbox = np.stack(np.meshgrid(fr, fr, fr, indexing='ij'), -1)
        self.Gc = mbox@B; self.G2 = (self.Gc**2).sum(-1)
        G2s = np.where(self.G2 > 0, self.G2, 1.0)
        # local pseudopotential, Eqs. (6.6)-(6.8), (8.4), rule 2
        if ions:
            x2 = self.G2*ps['rloc']**2
            v = (-4*pi*ps['Z']/G2s*np.exp(-x2/2) + (2*pi)**1.5*ps['rloc']**3*np.exp(-x2/2)*ps['C1'])/Om
            self.alpha = 2*pi*ps['Z']*ps['rloc']**2 + (2*pi)**1.5*ps['rloc']**3*ps['C1']
            v[0, 0, 0] = self.alpha/Om
            vr = np.fft.ifftn(v)*N**3
            assert np.abs(vr.imag).max() < 1e-10
            self.Vloc = vr.real
            self.Eew = ewald(A, ps['Z'])
        else:
            self.Vloc = np.zeros((N, N, N)); self.Eew = 0.0
        # coupling matrix D over (spin, projector); projector order (l, i, m)
        self.plist = [(l, i, m) for l in range(3) for i in (1, 2) for m in range(2*l + 1)]
        npj = self.npj = len(self.plist)
        D = np.zeros((2, npj, 2, npj), complex)
        for l in range(3):
            hl, kl = sym2(ps['h'][l]), sym2(ps['k'][l])
            LS = LS_real(l).reshape(2, 2*l + 1, 2, 2*l + 1) if l > 0 else None
            for pa, (la, i, m) in enumerate(self.plist):
                for pb, (lb, j, m2) in enumerate(self.plist):
                    if la != l or lb != l: continue
                    for s in range(2):
                        for s2 in range(2):
                            val = hl[i-1, j-1]*(m == m2)*(s == s2)
                            if soc and l > 0:
                                val = val + kl[i-1, j-1]*LS[s, m, s2, m2]
                            D[s, pa, s2, pb] = val
        if not ions: D[:] = 0
        self.D = D.reshape(2*npj, 2*npj)
        assert np.allclose(self.D, self.D.conj().T)

    # Algorithm 1 for a list of k-points, padded to a common size
    def build(self, kfracs):
        A, B, N, Om, ps = self.A, self.B, self.N, self.Om, self.ps
        M = int(np.ceil(sqrt(2*self.ecut)*np.linalg.norm(A, axis=1).max()/(2*pi))) + 1
        r = np.arange(-M, M + 1)
        trip = np.array(np.meshgrid(r, r, r, indexing='ij')).reshape(3, -1).T
        recs = []
        for kf in kfracs:
            q = (np.asarray(kf) + trip)@B
            T = 0.5*(q**2).sum(1); keep = T <= self.ecut
            recs.append((trip[keep], q[keep], T[keep]))
        ngmax = max(len(r[0]) for r in recs); nk = len(recs)
        kin = np.zeros((nk, ngmax)); idx = np.zeros((nk, ngmax), np.int32)
        mask = np.zeros((nk, ngmax)); Bt = np.zeros((nk, self.npj, ngmax), complex)
        mill = np.zeros((nk, ngmax, 3), int)
        for ik, (m, q, T) in enumerate(recs):
            ng = len(m)
            kin[ik, :ng] = T; mask[ik, :ng] = 1; mill[ik, :ng] = m
            idx[ik, :ng] = np.ravel_multi_index((m % N).T, (N, N, N))
            qn = np.linalg.norm(q, axis=1)
            for p, (l, i, mm) in enumerate(self.plist):
                Bt[ik, p, :ng] = 4*pi/sqrt(Om)*(-1j)**l*ptilde(i, l, qn, ps['r'][l])*real_ylm(l, q)[mm]
        return dict(kin=jnp.array(kin), idx=jnp.array(idx), mask=jnp.array(mask), Bt=jnp.array(Bt),
                    mill=mill, ng=np.array([len(r[0]) for r in recs]), kf=np.array(kfracs, float))

# ---------------------------------------------------------------- Algorithm 2
def apply_h(C, kin, idx, mask, Bt, Vr, D, N):
    """C [nb, 2, ng] at one k-point."""
    nb = C.shape[0]
    Y = kin*C
    box = jnp.zeros((nb, 2, N**3), C.dtype).at[:, :, idx].add(C)
    u = jnp.fft.ifftn(box.reshape(nb, 2, N, N, N), axes=(2, 3, 4))
    v = jnp.fft.fftn(Vr*u, axes=(2, 3, 4)).reshape(nb, 2, N**3)
    Y = Y + v[:, :, idx]
    P = jnp.einsum('pg,nsg->nsp', Bt.conj(), C).reshape(nb, -1)
    Q = (P@D.T).reshape(nb, 2, -1)
    Y = Y + jnp.einsum('pg,nsp->nsg', Bt, Q)
    return Y*mask

# ---------------------------------------------------------------- Algorithms 3, 4 (row convention)
def orthonormalize(V, tau, HV=None):
    d = jnp.linalg.norm(V, axis=1); d = jnp.where(d > 0, d, 1.0)
    Vs = V/d[:, None]
    G = Vs.conj()@Vs.T
    sig, U = jnp.linalg.eigh(0.5*(G + G.conj().T))
    keep = sig > tau
    T = (U*jnp.where(keep, 1/jnp.sqrt(jnp.where(keep, sig, 1.0)), 0.0))/d[:, None]
    if HV is None:
        return T.T@V
    return T.T@V, T.T@HV

def teter(x):
    num = 27 + 18*x + 12*x**2 + 8*x**3
    return num/(num + 16*x**4)

def make_solver(sysm, nb):
    N = sysm.N; D = jnp.array(sysm.D)

    def H(X, kin, idx, mask, Bt, Vr):
        return apply_h(X.reshape(X.shape[0], 2, -1), kin, idx, mask, Bt, Vr, D, N).reshape(X.shape[0], -1)

    def init(X, kin, idx, mask, Bt, Vr):
        X = X.reshape(nb, -1)
        X = orthonormalize(orthonormalize(X, 1e-12), 1e-12)
        HX = H(X, kin, idx, mask, Bt, Vr)
        A = X.conj()@HX.T
        lam, Y = jnp.linalg.eigh(0.5*(A + A.conj().T))
        return X_rot(Y, X), X_rot(Y, HX), jnp.zeros_like(X), jnp.zeros_like(X), lam

    def X_rot(Y, X): return Y.T@X

    def step(state, kin, idx, mask, Bt, Vr, eps):
        X, HX, P, HP, lam = state
        kin2 = jnp.concatenate([kin, kin])
        R = HX - lam[:, None]*X
        rn = jnp.linalg.norm(R, axis=1)
        ek = (kin2*jnp.abs(X)**2).sum(1)
        W = teter(kin2[None, :]/ek[:, None])*R
        W = jnp.where((rn < eps/10)[:, None], 0.0, W)
        Q = jnp.concatenate([X, P])
        for _ in range(2):
            W = W - (W@Q.conj().T)@Q
            W = orthonormalize(W, 1e-10)
        HW = H(W, kin, idx, mask, Bt, Vr)
        S = jnp.concatenate([X, W, P]); HS = jnp.concatenate([HX, HW, HP])
        A = S.conj()@HS.T; A = 0.5*(A + A.conj().T)
        zero = jnp.linalg.norm(S, axis=1) < 0.5
        A = A + jnp.diag(jnp.where(zero, 1e6, 0.0))
        th, Y = jnp.linalg.eigh(A)
        lam = th[:nb]; Y = Y[:, :nb]
        YX, YW, YP = Y[:nb], Y[nb:2*nb], Y[2*nb:]
        P = YW.T@W + YP.T@P; HP = YW.T@HW + YP.T@HP
        X = YX.T@X + P; HX = YX.T@HX + HP
        for _ in range(PPASS):
            C = X.conj()@P.T                  # C[i,j] = <x_i|p_j>
            P = P - C.T@X; HP = HP - C.T@HX
            P, HP = orthonormalize(P, 1e-6, HP)
        return (X, HX, P, HP, lam), rn

    vinit = jax.jit(jax.vmap(init, in_axes=(0, 0, 0, 0, 0, None)))
    vstep = jax.jit(jax.vmap(step, in_axes=(0, 0, 0, 0, 0, None, None)))
    return vinit, vstep

def solve_bands(sysm, bas, Vr, nb, nwant, eps=1e-8, tmax=400, seed=0, X0=None, niter=None):
    """Algorithm 7 (or, with niter given, a fixed number of LOBPCG steps)."""
    key = (id(sysm), nb)
    if key not in _cache: _cache[key] = make_solver(sysm, nb)
    vinit, vstep = _cache[key]
    nk, ng = bas['kin'].shape
    if X0 is None:
        rng = np.random.default_rng(seed)
        X0 = (rng.normal(size=(nk, nb, 2, ng)) + 1j*rng.normal(size=(nk, nb, 2, ng)))
        X0 = X0/(1 + np.asarray(bas['kin']))[:, None, None, :]*np.asarray(bas['mask'])[:, None, None, :]
        X0 = jnp.array(X0)
    args = (bas['kin'], bas['idx'], bas['mask'], bas['Bt'], Vr)
    state = vinit(X0, *args)
    it = 0
    for it in range(niter if niter is not None else tmax):
        state, rn = vstep(state, *args, 0.0 if niter is not None else eps)
        if niter is None and float(rn[:, :nwant].max()) < eps: break
    X, HX, P, HP, lam = state
    rn = jnp.linalg.norm(HX - lam[..., None]*X, axis=-1)
    return lam, X.reshape(nk, nb, 2, ng), rn, it + 1
_cache = {}
PPASS = 2

# ---------------------------------------------------------------- meshes
def mesh(n, gamma_centered):
    f = np.arange(n)/n if gamma_centered else (2*np.arange(1, n + 1) - n - 1)/(2*n)
    return np.array(np.meshgrid(f, f, f, indexing='ij')).reshape(3, -1).T

def tr_reduce(K):
    seen = {}; ks = []; ws = []
    for k in K:
        key = tuple(np.round(k % 1.0, 8) % 1.0); mkey = tuple(np.round((-k) % 1.0, 8) % 1.0)
        if mkey in seen: ws[seen[mkey]] += 1; continue
        if key in seen: ws[seen[key]] += 1; continue
        seen[key] = len(ks); ks.append(k); ws.append(1)
    return np.array(ks), np.array(ws, float)/len(K)

# ---------------------------------------------------------------- fields
def make_fields(sysm):
    N, Om = sysm.N, sysm.Om
    G2 = jnp.array(sysm.G2); Gc = jnp.array(sysm.Gc); Vloc = jnp.array(sysm.Vloc)
    G2s = jnp.where(G2 > 0, G2, 1.0)
    dfdn = jax.vmap(jax.grad(lambda n, g: f_xc(n, g, sysm.xc), 0))
    dfdg = jax.vmap(jax.grad(lambda n, g: f_xc(n, g, sysm.xc), 1))

    @jax.jit
    def potential(n):
        nG = jnp.fft.fftn(n)/N**3
        vH_G = jnp.where(G2 > 0, 4*pi*nG/G2s, 0.0)
        EH = 0.5*Om*jnp.sum(jnp.where(G2 > 0, 4*pi*jnp.abs(nG)**2/G2s, 0.0))
        grad = jnp.stack([(jnp.fft.ifftn(1j*Gc[..., i]*nG)*N**3).real for i in range(3)], -1)
        g2 = (grad**2).sum(-1)
        nc = jnp.maximum(n, 1e-12)
        f = f_xc(nc, g2, sysm.xc)
        vxc = dfdn(nc.ravel(), g2.ravel()).reshape(n.shape)
        if sysm.xc == 'PBE':
            w = 2*dfdg(nc.ravel(), g2.ravel()).reshape(n.shape)
            div = sum(jnp.fft.ifftn(1j*Gc[..., i]*jnp.fft.fftn(w*grad[..., i])).real for i in range(3))
            vxc = vxc - div
        dv = Om/N**3
        vH = (jnp.fft.ifftn(vH_G)*N**3).real
        E = dict(EH=EH, Exc=f.sum()*dv, Evxc=(vxc*n).sum()*dv, Eloc=(Vloc*n).sum()*dv)
        return Vloc + vH + vxc, E

    @jax.jit
    def density(C, idx, wf):
        """C [nk, nb, 2, ng]; wf [nk, nb] = w_k f_nk."""
        def one(Ck, idxk, wfk):
            nb = Ck.shape[0]
            box = jnp.zeros((nb, 2, N**3), Ck.dtype).at[:, :, idxk].add(Ck)
            u = jnp.fft.ifftn(box.reshape(nb, 2, N, N, N), axes=(2, 3, 4))*N**3/sqrt(Om)
            return jnp.einsum('n,nsxyz->xyz', wfk, jnp.abs(u)**2)
        return jax.lax.map(lambda a: one(*a), (C, idx, wf)).sum(0)
    return potential, density

def fermi(eps, w, Ne, sigma, kind):
    lo, hi = float(eps.min()) - 10*sigma, float(eps.max()) + 10*sigma
    for _ in range(80):
        mu = 0.5*(lo + hi)
        if float((w[:, None]*occ_fn((eps - mu)/sigma, kind)).sum()) > Ne: hi = mu
        else: lo = mu
    x = (eps - mu)/sigma
    return mu, occ_fn(x, kind), float((w[:, None]*ent_fn(x, kind)).sum())

# ---------------------------------------------------------------- Algorithms 5, 6 and the SCF driver
def scf(sysm, nmesh, gamma_centered, nb=20, sigma=0.01, kind='mp1', tol=1e-7, maxit=60,
        A=0.8, ktf=1.0, M=8, verbose=True):
    K, w = tr_reduce(mesh(nmesh, gamma_centered))
    bas = sysm.build(K); w = jnp.array(w)
    N, Om, Ne = sysm.N, sysm.Om, sysm.Ne
    potential, density = make_fields(sysm)
    G2 = sysm.G2
    n = (np.fft.ifftn(Ne/Om*np.exp(-G2*1.0/2))*N**3).real       # Gaussian blob, w = 1 Bohr
    n = jnp.array(n)
    kerker = A*G2/(G2 + ktf**2); metric = np.where(G2 > 0, 4*pi/np.where(G2 > 0, G2, 1), 0)
    hist_n, hist_R = [], []
    X = None; D = jnp.array(sysm.D); log = []
    for it in range(1, maxit + 1):
        t0 = time.time()
        Vr, Ein = potential(n)
        lam, X, rn, _ = solve_bands(sysm, bas, Vr, nb, nb, X0=X, niter=8 if it == 1 else 3)
        mu, f, S = fermi(lam, w, Ne, sigma, kind)
        nout = density(X, bas['idx'], w[:, None]*f)
        nel = float(nout.sum()*Om/N**3)
        # Eq. (8.10): eigenvalue-sum form, input density
        F_eig = float((w[:, None]*f*lam).sum()) - sigma*S - float(Ein['EH']) - float(Ein['Evxc']) + float(Ein['Exc']) + sysm.Eew
        # Eq. (8.9): direct form, output density
        Ts = float(jnp.einsum('k,kn,kg,knsg->', w, f, bas['kin'], jnp.abs(X)**2))
        P = jnp.einsum('kpg,knsg->knsp', bas['Bt'].conj(), X).reshape(X.shape[0], nb, -1)
        Enl = float(jnp.einsum('k,kn,kna,ab,knb->', w, f, P.conj(), D, P).real)
        _, Eo = potential(nout)
        F_dir = Ts + Enl + float(Eo['Eloc']) + float(Eo['EH']) + float(Eo['Exc']) + sysm.Eew - sigma*S
        R = np.asarray(nout - n)
        rnorm = sqrt((R**2).sum()*Om/N**3)
        log.append((it, rnorm, F_eig, F_dir - F_eig, mu))
        if verbose:
            print("it %2d |R|=%.2e F(8.10)=%.8f dir-eig=%+.1e mu=%.6f Nel=%.8f maxres=%.1e %.0fs"
                  % (it, rnorm, F_eig, F_dir - F_eig, mu, nel, float(rn[:, :Ne + 2].max()), time.time() - t0), flush=True)
        if rnorm < tol: break
        # Algorithm 6
        hist_n.append(np.fft.fftn(np.asarray(n))); hist_R.append(np.fft.fftn(R))
        hist_n, hist_R = hist_n[-M:], hist_R[-M:]
        m = len(hist_R)
        Bm = np.array([[np.sum(metric*(hist_R[i].conj()*hist_R[j])).real for j in range(m)] for i in range(m)])
        Bm = Bm/np.abs(Bm).max()
        Mx = np.zeros((m + 1, m + 1)); Mx[:m, :m] = Bm; Mx[m, :m] = Mx[:m, m] = 1
        rhs = np.zeros(m + 1); rhs[m] = 1
        g = np.linalg.lstsq(Mx, rhs, rcond=None)[0][:m]
        n = jnp.array(np.fft.ifftn(sum(g[i]*(hist_n[i] + kerker*hist_R[i]) for i in range(m))).real)
    return dict(n=n, Vr=Vr, mu=mu, F=F_eig, eps=lam, K=K, w=w, X=X, bas=bas, its=it, log=log)

def dense_levels(sysm, kf, Vr, nlev=14):
    bas = sysm.build([kf]); ng = int(bas['ng'][0])
    I = jnp.eye(2*ng, dtype=complex).reshape(2*ng, 2, ng)
    Hm = apply_h(I, bas['kin'][0], bas['idx'][0], bas['mask'][0], bas['Bt'][0], Vr, jnp.array(sysm.D), sysm.N)
    Hm = np.asarray(Hm.reshape(2*ng, 2*ng)).T
    assert np.abs(Hm - Hm.conj().T).max() < 1e-10
    return np.linalg.eigvalsh(Hm)[:nlev], ng

def group(levels, tol=2e-6):
    out = []; i = 0
    while i < len(levels):
        j = i
        while j + 1 < len(levels) and abs(levels[j + 1] - levels[i]) < tol: j += 1
        out.append("%.6f(%d)" % (np.mean(levels[i:j + 1]), j - i + 1)); i = j + 1
    return "  ".join(out)

def parity_matrix(bas, ik, C, nfrac):
    mill = bas['mill'][ik][:bas['ng'][ik]]
    look = {tuple(m): i for i, m in enumerate(mill)}
    perm = np.array([look[tuple(-m - nfrac)] for m in mill])
    C = np.asarray(C)[:, :, :len(mill)]
    Pi = np.einsum('msg,nsg->mn', C.conj(), C[:, :, perm])
    return Pi

if __name__ == '__main__':
    job = sys.argv[1] if len(sys.argv) > 1 else 'small'
    a = 4.0782/BOHR
    if job == 'small':
        for xc, soc in (('LDA', True), ('LDA', False), ('PBE', True)):
            s = System(a, 20.0, xc, soc)
            r = scf(s, 4, True, tol=1e-8, verbose=(xc == 'LDA' and soc))
            lev, ng = dense_levels(s, [0, 0, 0], r['Vr'])
            print("JAX %s SOC %s: grid %d ng(Gamma) %d nk %d its %d  F=%.6f  mu=%.6f"
                  % (xc, 'on' if soc else 'off', s.N, ng, len(r['K']), r['its'], r['F'], r['mu']))
            print("    Gamma levels (Ha):", group(lev), flush=True)
    elif job == 'caseA':
        s = System(a, 25.0, 'LDA', True)
        r = scf(s, 4, False)
        print("JAX case A: grid %d nk %d its %d F=%.6f mu=%.5f" % (s.N, len(r['K']), r['its'], r['F'], r['mu']))
    elif job == 'caseB':
        s = System(a, 40.0, 'LDA', True)
        r = scf(s, 8, True)
        print("JAX case B: grid %d nk %d its %d F=%.6f mu=%.5f" % (s.N, len(r['K']), r['its'], r['F'], r['mu']))
        np.save('n_jax_caseB.npy', np.asarray(r['n']))
        trim = [[0, 0, 0], [0, .5, .5], [.5, .5, .5]]
        bas = s.build(trim)
        lam, X, rn, its = solve_bands(s, bas, r['Vr'], 20, 14, eps=1e-9)
        for ik, kf in enumerate(trim):
            Pi = parity_matrix(bas, ik, X[ik, :12], np.round(2*np.array(kf)).astype(int))
            ev = np.linalg.eigvalsh(0.5*(Pi + Pi.conj().T))
            print(kf, "E-EF (eV), pairs:", np.round((np.asarray(lam[ik, :14:2]) - r['mu'])*HA_EV, 2),
                  " N-=%d  max||lambda|-1|=%.1e" % ((ev < 0).sum(), np.abs(np.abs(ev) - 1).max()))
    elif job == 'stage1':
        # empty lattice
        s = System(a, 10.0, 'none', False, ions=False)
        kf = [[0.1, 0.2, 0.3], [0, 0, 0], [0, .5, .5]]
        bas = s.build(kf); V0 = jnp.zeros((s.N,)*3)
        lam, X, rn, its = solve_bands(s, bas, V0, 12, 8, eps=1e-9)
        for ik in range(3):
            exact = np.sort(np.repeat(np.asarray(bas['kin'][ik])[:bas['ng'][ik]], 2))[:8]
            print("empty lattice k=%s  max|eps - k^2/2| = %.1e (its %d)" % (kf[ik], np.abs(np.asarray(lam[ik, :8]) - exact).max(), its))
        Pi = parity_matrix(bas, 2, X[2, :4], np.array([0, 1, 1]))
        print("empty lattice X: eig(Pi) on lowest 4 spinor states:", np.round(np.linalg.eigvalsh(0.5*(Pi + Pi.conj().T)), 8))
        # single cosine potential V = 2 V0 cos(b1.r), k = b1/2
        b1 = s.B[0]; N = s.N
        j = np.arange(N)/N
        frac = np.stack(np.meshgrid(j, j, j, indexing='ij'), -1)       # r = frac . A  ->  b1.r = 2 pi frac_1
        bas = s.build([[.5, 0, 0]])
        for V0c in (-0.05, 0.05):
            Vr = jnp.array(2*V0c*np.cos(2*pi*frac[..., 0]))
            lam, X, rn, its = solve_bands(s, bas, Vr, 12, 6, eps=1e-10)
            T = 0.125*(b1@b1)
            par = []
            for p in (0, 2):
                Pi = parity_matrix(bas, 0, X[0, p:p + 2], np.array([1, 0, 0]))
                par.append(np.round(np.linalg.eigvalsh(0.5*(Pi + Pi.conj().T)), 6).tolist())
            print("cosine V0=%+.2f: levels-T = %.5f, %.5f (expect -+|V0| + O(V0^2/|b1|^2)); gap=%.5f; parities low %s high %s"
                  % (V0c, lam[0, 0] - T, lam[0, 2] - T, lam[0, 2] - lam[0, 0], par[0], par[1]))
    elif job == 'jellium':
        for nm in (4, 8):
            s = System(a, 8.0, 'LDA', False, ions=False, nelec=11)
            r = scf(s, nm, True, nb=24, sigma=0.01, kind='fd', verbose=False)
            n0 = 11/s.Om; kF = (3*pi**2*n0)**(1/3)
            exc = float(f_xc(jnp.array(n0), 0.0, 'LDA'))/n0
            vxc = float(jax.grad(lambda n: f_xc(n, 0.0, 'LDA'))(n0))
            nn = np.asarray(r['n'])
            print("jellium %d^3: density uniform to %.1e; lowest eigenvalue - vxc = %.2e; F/N = %.6f; E/N (1/2(E+F)) vs analytic %.6f"
                  % (nm, np.abs(nn - n0).max(), float(r['eps'].min()) - vxc, r['F']/11, 0.3*kF**2 + exc))
