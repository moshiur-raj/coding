from scf import *
a=4.0782/BOHR; n=np.load('n_lda40.npy'); EC=float(sys.argv[1])
worst=0
for spinor in (False,True):
    cr=Crystal(a,EC,'LDA',soc=spinor,spinor=spinor,ngrid=n.shape[0]); Vr,_=cr.potential(n)
    nb=12 if not spinor else 20; nconv=8 if not spinor else 14
    for kf in ([0,0,0],[0,.5,.5],[.5,.5,.5],[.25,.5,.75],[.375,.375,.75],[.1,.2,.3],[0.25,0.25,0.5]):
        bs=cr.basis(kf); wd=np.linalg.eigvalsh(cr.Hdense(bs,Vr))[:nb]
        for seed in range(int(sys.argv[2]) if len(sys.argv) > 2 else 3):
            rng=np.random.default_rng(seed)
            X0=(rng.normal(size=(nb,cr.ns,bs['ng']))+1j*rng.normal(size=(nb,cr.ns,bs['ng'])))/(1+bs['kin'])
            row=[]
            for tol,tm in ((1e-9,300),(0,300)):
                st={}; h=[]; lam,X=lobpcg_orth(cr,bs,Vr,X0,tm,tol=tol,nconv=nconv,stats=st,hist=h)
                e=np.abs(lam[:nconv]-wd[:nconv]).max(); worst=max(worst,e)
                Xm=X.reshape(nb,-1); o=np.abs(Xm.conj()@Xm.T-np.eye(nb)).max()
                row.append("its %3d err %.0e res %.0e ortho %.0e"%(st['iters'],e,st['resid'][:nconv].max(),o))
            print(spinor,kf,seed," | ".join(row),flush=True)
print("WORST eigen error",worst)
