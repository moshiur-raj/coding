from scf import *
a=4.0782/BOHR
cr=Crystal(a,20.0,'LDA',soc=False,spinor=False)
for mixer,A in (('linear',0.8),('linear',1.0),('kerker',0.8),('pulay',0.8)):
    r=run_scf(cr,4,12,maxit=60,gamma_centered=True,verbose=False,tol=1e-7,mixer=mixer,A=A)
    print("MIX",mixer,A,"iterations:",len(r['log']),"final |R| %.1e"%r['log'][-1][1],flush=True)
