import os
os.environ['OMP_NUM_THREADS']='2'; os.environ['OPENBLAS_NUM_THREADS']='2'
import numpy as np, sys
from math import pi, sqrt
from scipy import special
from atom import p
from pw import *
def radial(xc='LDA', h=0.01, Rmax=30.0, w=1.0):
    ps=PSP[xc]; N=int(Rmax/h); r=h*np.arange(1,N+1)
    V=-ps['Z']/r*special.erf(r/(sqrt(2)*ps['rloc']))+np.exp(-0.5*(r/ps['rloc'])**2)*ps['C1']+ps['Z']/r*special.erf(r/(sqrt(2)*w))
    T=(np.diag(np.full(N,1/h**2))-np.diag(np.full(N-1,0.5/h**2),1)-np.diag(np.full(N-1,0.5/h**2),-1))
    out={}
    for (l,j) in [(0,.5),(1,.5),(1,1.5),(2,1.5),(2,2.5)]:
        lam=0 if l==0 else (l/2 if j>l else -(l+1)/2)
        hm=np.array(ps['h'][l])+(lam*np.array(ps['k'][l]) if l>0 else 0)
        P=np.array([r*p(i,l,r,ps['r'][l]) for i in (1,2)])
        H=T+np.diag(l*(l+1)/(2*r*r)+V)+h*P.T@hm@P
        out[(l,j)]=np.linalg.eigvalsh(H)[:2]
    return out
if sys.argv[1]=='radial':
    for w,h in ((1.5,0.02),(2.0,0.02),(3.0,0.02),(2.0,0.01)):
        o=radial(h=h,Rmax=30.0,w=w); print("w=",w,"h=",h," ".join("l=%d j=%.1f: %.6f"%(l,j,e[0]) for (l,j),e in o.items()),flush=True)
        print("     second:"," ".join("%.6f"%e[1] for e in o.values()))
else:
    from scf import *
    aL=float(sys.argv[2]); ec=float(sys.argv[3]); w=float(sys.argv[4])
    cr=Crystal(aL,ec,'LDA',soc=True); print("a=",aL,"Omega",cr.Om,"N",cr.N,flush=True)
    N=cr.N; nG=11/cr.Om*np.exp(-cr.G2*w*w/2)
    with np.errstate(divide='ignore',invalid='ignore'): vH=4*pi*nG/cr.G2
    vH[0,0,0]=0; Vr=cr.Vloc_r+(np.fft.ifftn(vH)*N**3).real
    bs=cr.basis([0,0,0]); print("ng",bs['ng'],flush=True)
    rng=np.random.default_rng(1); nb=22
    X0=(rng.normal(size=(nb,2,bs['ng']))+1j*rng.normal(size=(nb,2,bs['ng'])))/(1+bs['kin'])
    st={}; lam,X=lobpcg_orth(cr,bs,Vr,X0,400,tol=1e-7,nconv=18,stats=st)
    shift=2*pi*11*w*w/cr.Om
    print("its",st['iters'],"shift 2 pi Z w^2/Omega =",shift)
    print("PW eigenvalues minus shift:",np.round(lam[:20]-shift,6))
