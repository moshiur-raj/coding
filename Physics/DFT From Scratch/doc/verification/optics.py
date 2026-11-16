"""Velocity matrix elements, Drude weight and interband transitions of gold in
the fixed self-consistent potential (density n_lda40.npy).

  python3 optics.py test            Hellmann-Feynman test of the velocity operator
  python3 optics.py run NM [ECUT] [soc|nosoc]   bands + |v|^2 on the irreducible wedge of an
                                    NM^3 Gamma-centred mesh -> optics_<tag>.npz
Post-processing (dielectric function, reflectivity, colour): optics_post.py
"""
from scf import *
import itertools

def proj_table(cr, q):
    """projectors beta_p(q) for cartesian q [ng,3] -> [npj, ng] (same formula as Crystal.basis)."""
    qn = np.linalg.norm(q, axis=1)
    Bt = np.zeros((cr.npj, len(q)), complex)
    for a, (l, i, m) in enumerate(cr.projlist):
        Bt[a] = 4*pi/sqrt(cr.Om)*(-1j)**l*ptilde(i, l, qn, cr.psp['r'][l])*real_ylm(l, q)[m]
    return Bt

def velocity(cr, bs, C, delta=1e-4):
    """v^alpha_{mn} = <m| dH_k/dk_alpha |n> for the bands C [nb, ns, ng]; returns [3, nb, nb]."""
    nb = C.shape[0]
    P = np.einsum('pg,nsg->nsp', bs['Bt'].conj(), C).reshape(nb, -1)        # <beta|psi_n>
    DP = P@cr.D.T                                                          # (D P)_n
    V = np.zeros((3, nb, nb), complex)
    for al in range(3):
        e = np.zeros(3); e[al] = delta
        dB = (proj_table(cr, bs['q']+e)-proj_table(cr, bs['q']-e))/(2*delta)
        dP = np.einsum('pg,nsg->nsp', dB.conj(), C).reshape(nb, -1)        # <d beta|psi_n>
        kin = np.einsum('msg,g,nsg->mn', C.conj(), bs['q'][:, al], C)
        V[al] = kin + dP.conj()@DP.T + P.conj()@(dP@cr.D.T).T
    return V

def solve(cr, Vr, kf, nb, nconv, tol=1e-7, seed=11):
    rng = np.random.default_rng(seed); bs = cr.basis(kf)
    X0 = (rng.normal(size=(nb, cr.ns, bs['ng']))+1j*rng.normal(size=(nb, cr.ns, bs['ng'])))/(1+bs['kin'])
    lam, X = lobpcg_orth(cr, bs, Vr, X0, 400, tol=tol, nconv=nconv)
    return lam, X, bs

def ibz(cr, NM):
    ops = []
    for perm in itertools.permutations(range(3)):
        for sg in itertools.product([1, -1], repeat=3):
            M = np.zeros((3, 3))
            for r in range(3): M[r, perm[r]] = sg[r]
            ops.append(M)
    f = np.arange(NM)/NM; K = np.array(np.meshgrid(f, f, f, indexing='ij')).reshape(3, -1).T
    Binv = np.linalg.inv(cr.B); rep = {}; cnt = {}
    for k in K:
        kc = k@cr.B
        key = min(tuple(np.round(((kc@R.T)@Binv) % 1.0, 6) % 1.0) for R in ops)
        rep.setdefault(key, k); cnt[key] = cnt.get(key, 0)+1
    keys = list(rep)
    return np.array([rep[k] for k in keys]), np.array([cnt[k] for k in keys], float)/len(K)

if __name__ == '__main__':
    a = 4.0782/BOHR; n = np.load('n_lda40.npy'); job = sys.argv[1]
    if job == 'test':
        cr = Crystal(a, 20.0, 'LDA', soc=True, ngrid=n.shape[0]); Vr, _ = cr.potential(n)
        k0 = np.array([0.13, 0.27, 0.41]); h = 1e-4; nb = 20
        lam, X, bs = solve(cr, Vr, k0, nb, 16, 1e-9)
        V = velocity(cr, bs, X[:16])
        print("Hermitian:", max(np.abs(V[al]-V[al].conj().T).max() for al in range(3)))
        for al in range(3):
            e = np.zeros(3); e[al] = h; df = e@np.linalg.inv(cr.B)          # cartesian step -> fractional
            lp = solve(cr, Vr, k0+df, nb, 16, 1e-9)[0]; lm = solve(cr, Vr, k0-df, nb, 16, 1e-9)[0]
            fd = (lp-lm)[:16]/(2*h)
            vd = np.array([np.trace(V[al][i:i+2, i:i+2]).real/2 for i in range(0, 16, 2)])   # Kramers pairs
            off = max(abs(V[al][i, i+1]) for i in range(0, 16, 2))
            print("alpha=%d  max|<v> - d eps/dk| = %.1e   (|v| up to %.3f a.u.; pair off-diagonal %.1e)"
                  % (al, np.abs(vd-fd[::2]).max(), np.abs(vd).max(), off))
    else:
        NM = int(sys.argv[2]); EC = float(sys.argv[3]) if len(sys.argv) > 3 else 40.0
        soc = not (len(sys.argv) > 4 and sys.argv[4] == 'nosoc')
        cr40 = Crystal(a, 40.0, 'LDA', soc=True, ngrid=n.shape[0]); Vr, _ = cr40.potential(n)
        cr = Crystal(a, EC, 'LDA', soc=soc, ngrid=n.shape[0])
        K, w = ibz(cr, NM); print("irreducible points:", len(K), "weights sum", w.sum(), flush=True)
        NB, NC = 28, 22
        def one(k):
            lam, X, bs = solve(cr, Vr, k, NB, NC, 1e-6)
            V = velocity(cr, bs, X[:NC])
            return lam[:NC], (np.abs(V)**2).sum(0)/3
        pool = ThreadPoolExecutor(4); t0 = time.time()
        res = list(pool.map(one, K))
        tag = "%d_%d_%s" % (NM, int(EC), 'soc' if soc else 'nosoc')
        np.savez('optics_%s.npz' % tag, K=K, w=w, eps=np.array([r[0] for r in res]), v2=np.array([r[1] for r in res]), Om=cr.Om)
        print("done", tag, "%.0f s" % (time.time()-t0))
