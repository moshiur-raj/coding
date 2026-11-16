from scf import *
a=4.0782/BOHR
n=np.load('n_lda40.npy')
for spinor in (False,True):
    cr=Crystal(a,40.0,'LDA',soc=spinor,spinor=spinor); Vr,_=cr.potential(n)
    nb=12 if not spinor else 20; nconv=8 if not spinor else 14
    for kf in ([0,0,0],[0.1,0.2,0.3]):
        bs=cr.basis(kf); rng=np.random.default_rng(5)
        X0=(rng.normal(size=(nb,cr.ns,bs['ng']))+1j*rng.normal(size=(nb,cr.ns,bs['ng'])))/(1+bs['kin'])
        wd=np.linalg.eigvalsh(cr.Hdense(bs,Vr))[:nb] if not spinor else None
        for name,fn in (("literal",lobpcg),("safe",lobpcg_safe)):
            for tol,tm in ((1e-9,200),(0,30),(0,100)):
                st={}; t0=time.time()
                try:
                    lam,X=fn(cr,bs,Vr,X0,tm,tol=tol,nconv=nconv,stats=st)
                    err = "" if wd is None else "eigerr %.1e"%np.abs(lam[:nconv]-wd[:nconv]).max()
                    print(spinor,kf,name,"tol",tol,"its",st['iters'],"restarts",st['restarts'],"maxres(nconv) %.1e"%st['resid'][:nconv].max(),"lam0 %.6f"%lam[0],err,"%.1fs"%(time.time()-t0),flush=True)
                except Exception as e: print(spinor,kf,name,tol,"EXC",repr(e)[:80],flush=True)
