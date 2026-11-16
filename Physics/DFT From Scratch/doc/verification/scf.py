import os
os.environ.setdefault('OMP_NUM_THREADS', '1'); os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
import numpy as np, time, sys
from concurrent.futures import ThreadPoolExecutor
from pw import *

def run_scf(cr, nmesh, nb, sigma=0.01, kind='mp1', maxit=60, tol=1e-7, lob_it=3, A=0.8, ktf=1.0,
            hist=8, mixer='pulay', verbose=True, gamma_centered=False, n0=None, nthreads=4):
    K, w = tr_reduce(mp_mesh(nmesh, gamma_centered))
    bases = [cr.basis(k) for k in K]
    ns = cr.ns; Ne = cr.psp['Z']; fmax = 1.0 if ns == 2 else 2.0
    rng = np.random.default_rng(1)
    X = []
    for bs in bases:
        x = (rng.normal(size=(nb, ns, bs['ng']))+1j*rng.normal(size=(nb, ns, bs['ng'])))/(1+bs['kin'])
        X.append(x)
    N = cr.N
    if n0 is None:
        nG = Ne/cr.Om*np.exp(-cr.G2*1.0/2); n = (np.fft.ifftn(nG)*N**3).real
    else: n = n0.copy()
    nin_h, R_h = [], []
    G2 = cr.G2; kerker = A*G2/(G2+ktf**2)
    metric = np.where(G2 > 0, 4*pi/np.maximum(G2, 1e-12), 0)
    pool = ThreadPoolExecutor(nthreads)
    log = []
    for it in range(maxit):
        t0 = time.time()
        Vr, E = cr.potential(n)
        stats = [dict() for _ in K]
        def eig(ik): return lobpcg_orth(cr, bases[ik], Vr, X[ik], lob_it if it > 0 else 8, stats=stats[ik])
        res = list(pool.map(eig, range(len(K))))
        eps = np.array([r[0] for r in res]); X = [r[1] for r in res]
        mu = fermi(eps, w*fmax, Ne, sigma, kind)
        x = (eps-mu)/sigma; f = fmax*occ(x, kind)
        def dens(ik): return w[ik]*cr.density_k(bases[ik], X[ik], f[ik])
        nout = sum(pool.map(dens, range(len(K))))
        TS = sigma*fmax*(w[:, None]*entropy(x, kind)).sum()
        Eband = (w[:, None]*f*eps).sum()
        # eigenvalue-sum form (input density) and direct form (output density)
        F_eig = Eband-TS-E['EH']-E['Evxc']+E['Exc']+cr.Eew
        def tnl(ik):
            bs = bases[ik]; C = X[ik]
            T = (f[ik][:, None, None]*bs['kin']*np.abs(C)**2).sum()
            P = np.einsum('pg,nsg->nsp', bs['Bt'].conj(), C).reshape(nb, -1)
            Enl = np.einsum('n,na,ab,nb->', f[ik], P.conj(), cr.D, P).real
            return w[ik]*np.array([T, Enl])
        T, Enl = sum(pool.map(tnl, range(len(K))))
        _, Eo = cr.potential(nout)
        F_dir = T+Enl+Eo['Eloc']+Eo['EH']+Eo['Exc']+cr.Eew-TS
        R = nout-n
        rnorm = sqrt((R**2).sum()*cr.Om/N**3)
        maxres = max(s['resid'][:Ne+2 if ns == 2 else Ne//2+2].max() for s in stats)
        restarts = sum(s['restarts'] for s in stats)
        log.append((it, rnorm, F_eig, F_dir, mu, maxres, restarts))
        if verbose:
            print("it %2d |R|=%.2e F_eig=%.8f F_dir=%.8f diff=%.1e mu=%.5f maxres=%.1e restarts=%d Nel=%.6f t=%.0fs"
                  % (it, rnorm, F_eig, F_dir, F_dir-F_eig, mu, maxres, restarts, nout.sum()*cr.Om/N**3, time.time()-t0), flush=True)
        if rnorm < tol: break
        RG = np.fft.fftn(R); nGin = np.fft.fftn(n)
        if mixer == 'linear':
            n = np.fft.ifftn(nGin+A*RG).real
        elif mixer == 'kerker':
            n = np.fft.ifftn(nGin+kerker*RG).real
        else:
            nin_h.append(nGin); R_h.append(RG); nin_h, R_h = nin_h[-hist:], R_h[-hist:]
            m = len(R_h)
            Bm = np.array([[np.sum(metric*(R_h[i].conj()*R_h[j])).real for j in range(m)] for i in range(m)])
            Bm /= Bm.max()
            M = np.zeros((m+1, m+1)); M[:m, :m] = Bm; M[m, :m] = M[:m, m] = 1
            rhs = np.zeros(m+1); rhs[m] = 1
            g = np.linalg.lstsq(M, rhs, rcond=1e-12)[0][:m]
            n = np.fft.ifftn(sum(g[i]*(nin_h[i]+kerker*R_h[i]) for i in range(m))).real
    pool.shutdown()
    return dict(n=n, Vr=Vr, mu=mu, eps=eps, K=K, w=w, F=F_eig, Fdir=F_dir, log=log, X=X, bases=bases, f=f)

if __name__ == '__main__':
    a = 4.0782/BOHR
    # --- unit tests of the Hamiltonian ---
    cr = Crystal(a, 15.0, 'LDA', soc=True)
    print("grid", cr.N, "Ewald", cr.Eew, "alpha", cr.alpha)
    bs = cr.basis([0.1, 0.2, 0.3]); print("ng", bs['ng'])
    rng = np.random.default_rng(0)
    Vr = cr.Vloc_r+0.1*rng.normal(size=cr.Vloc_r.shape)
    c1 = rng.normal(size=(1, 2, bs['ng']))+1j*rng.normal(size=(1, 2, bs['ng'])); c2 = rng.normal(size=(1, 2, bs['ng']))+1j*rng.normal(size=(1, 2, bs['ng']))
    print("hermiticity:", abs(np.vdot(c1, cr.applyH(bs, c2, Vr))-np.conj(np.vdot(c2, cr.applyH(bs, c1, Vr)))))
    Hd = cr.Hdense(bs, cr.Vloc_r); wd = np.linalg.eigvalsh(Hd)
    print("dense lowest:", wd[:14])
    X0 = (rng.normal(size=(20, 2, bs['ng']))+1j*rng.normal(size=(20, 2, bs['ng'])))/(1+bs['kin'])
    for tm in (5, 10, 20, 40):
        st = {}; lam, X = lobpcg(cr, bs, cr.Vloc_r, X0, tm, stats=st)
        print("lobpcg its", tm, "max err lowest12 = %.2e" % np.abs(lam[:12]-wd[:12]).max(), "restarts", st['restarts'], "resid12 %.1e" % st['resid'][:12].max())
