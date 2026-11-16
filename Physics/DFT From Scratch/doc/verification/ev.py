from scf import *
def bm(V,E0,V0,B0,Bp):
    x=(V0/V)**(2/3); return E0+9*V0*B0/16*((x-1)**3*Bp+(x-1)**2*(6-4*x))
from scipy.optimize import curve_fit
def scan(xc,spinor,ecut,mesh,alist,nb):
    Vs=[];Es=[]; n0=None
    for aA in alist:
        a=aA/BOHR; cr=Crystal(a,ecut,xc,soc=spinor,spinor=spinor)
        r=run_scf(cr,mesh,nb,maxit=40,gamma_centered=True,verbose=False,tol=1e-6)
        Vs.append(cr.Om); Es.append(r['F']); print(xc,"spinor" if spinor else "scalar","a=%.3f N=%d its=%d F=%.8f"%(aA,cr.N,len(r['log']),r['F']),flush=True)
    Vs=np.array(Vs);Es=np.array(Es); i=np.argmin(Es)
    p,_=curve_fit(bm,Vs,Es,p0=[Es[i],Vs[i],0.006,5.0])
    print("==> %s %s Ecut=%g mesh=%d: a0=%.4f A  B0=%.1f GPa  B0'=%.2f  rms fit err=%.1e Ha"%(xc,"SOC" if spinor else "SR",ecut,mesh,(4*p[1])**(1/3)*BOHR,p[2]*29421.02648,p[3],np.sqrt(np.mean((bm(Vs,*p)-Es)**2))),flush=True)
job=sys.argv[1]
if job=='main':
    scan('LDA',False,45.0,8,[3.90,3.96,4.02,4.08,4.14,4.20,4.26],12)
    scan('PBE',False,45.0,8,[4.00,4.06,4.12,4.18,4.24,4.30,4.36],12)
elif job=='soc':
    al=[3.96,4.02,4.08,4.14,4.20]
    scan('LDA',False,35.0,6,al,12); scan('LDA',True,35.0,6,al,20)
elif job=='cut':
    for ec in (30.0,40.0,50.0,60.0,80.0):
        cr=Crystal(4.0782/BOHR,ec,'LDA',soc=False,spinor=False); r=run_scf(cr,4,12,maxit=40,gamma_centered=True,verbose=False,tol=1e-6)
        print("Ecut",ec,"N",cr.N,"F=%.6f"%r['F'],"mu=%.5f"%r['mu'],flush=True)
