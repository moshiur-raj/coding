from scf import *
a=4.0782/BOHR
print("a (Bohr) =",a)
for xc,spinor in (('LDA',True),('LDA',False),('PBE',True)):
    cr=Crystal(a,20.0,xc,soc=spinor,spinor=spinor)
    r=run_scf(cr,4,20 if spinor else 12,maxit=40,gamma_centered=True,verbose=(xc=='LDA' and spinor),tol=1e-8)
    bs=cr.basis([0,0,0]); w=np.linalg.eigvalsh(cr.Hdense(bs,r['Vr']))
    print("REF",xc,"spinor" if spinor else "scalar","N=%d ng(Gamma)=%d nk=%d its=%d F=%.8f mu=%.6f Eew=%.8f"%(cr.N,bs['ng'],len(r['K']),len(r['log']),r['F'],r['mu'],cr.Eew))
    print("   Gamma eigenvalues (Ha):",np.round(w[:14 if spinor else 7],6),flush=True)
cr=Crystal(a,20.0,'LDA',soc=False,spinor=False)
for mixer,A in (('linear',0.5),('linear',0.8),('linear',1.0),('kerker',0.8),('pulay',0.8)):
    r=run_scf(cr,4,12,maxit=60,gamma_centered=True,verbose=False,tol=1e-7,mixer=mixer,A=A)
    print("MIX",mixer,A,"iterations:",len(r['log']),"final |R| %.1e"%r['log'][-1][1],flush=True)
for kind in ('mp1','gauss','fd'):
    r=run_scf(cr,4,12,maxit=60,gamma_centered=True,verbose=False,tol=1e-7,kind=kind)
    print("SMEAR",kind,"F=%.6f mu=%.5f its=%d"%(r['F'],r['mu'],len(r['log'])),flush=True)
