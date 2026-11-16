from scf import *
from scipy.optimize import minimize
import itertools
a=4.0782/BOHR; n=np.load('n_lda40.npy')
cr=Crystal(a,40.0,'LDA',soc=True,ngrid=n.shape[0]); Vr,_=cr.potential(n)
cr0=Crystal(a,40.0,'LDA',soc=False,spinor=False,ngrid=n.shape[0])
def gap(c,k,i,tol=1e-6):
    rng=np.random.default_rng(11)
    bs=c.basis(k); nb=20 if c.ns==2 else 12
    X0=(rng.normal(size=(nb,c.ns,bs['ng']))+1j*rng.normal(size=(nb,c.ns,bs['ng'])))/(1+bs['kin'])
    lam,_=lobpcg_orth(c,bs,Vr,X0,300,tol=tol,nconv=i+3)
    return lam[i+1]-lam[i]
# irreducible wedge of a 20^3 Gamma-centred mesh
ops=[]
for perm in itertools.permutations(range(3)):
    for sg in itertools.product([1,-1],repeat=3):
        M=np.zeros((3,3)); 
        for r in range(3): M[r,perm[r]]=sg[r]
        ops.append(M)
NM=20; f=np.arange(NM)/NM; K=np.array(np.meshgrid(f,f,f,indexing='ij')).reshape(3,-1).T
Binv=np.linalg.inv(cr.B); seen={}
for k in K:
    kc=k@cr.B; key=min(tuple(np.round(((kc@R.T)@Binv)%1.0,6)%1.0) for R in ops)
    seen.setdefault(key,k)
reps=np.array(list(seen.values())); print("irreducible points:",len(reps),flush=True)
pool=ThreadPoolExecutor(4)
g1=np.array(list(pool.map(lambda k: gap(cr,k,11),reps))); g0=np.array(list(pool.map(lambda k: gap(cr0,k,5),reps)))
o=np.argsort(g1)[:6]; print("mesh20 IBZ SOC-on smallest gaps eV:",np.round(g1[o]*HA,4),"\n at",(reps[o]@cr.B*a/(2*pi)).round(4).tolist(),flush=True)
o0=np.argsort(g0)[:6]; print("mesh20 IBZ SOC-off smallest gaps eV:",np.round(g0[o0]*HA,4),"\n at",(reps[o0]@cr.B*a/(2*pi)).round(4).tolist(),flush=True)
starts=[reps[i] for i in o[:3]]+[np.array([0.35416667,0.5,0.64583333])]
def ref(c,i,k0):
    r=minimize(lambda k: gap(c,k,i,1e-7),k0,method='Nelder-Mead',options=dict(xatol=2e-4,fatol=1e-6,maxiter=120,initial_simplex=k0+0.01*np.vstack([np.zeros(3),np.eye(3)])))
    return r.fun*HA, r.x@c.B*a/(2*pi)
for r in pool.map(lambda k0: ref(cr,11,k0),starts): print("refined SOC-on : min gap %.4f eV at (2pi/a)"%r[0],np.round(r[1],4),flush=True)
starts0=[reps[i] for i in o0[:3]]+[np.array([0.36458333,0.5,0.63541667])]
for r in pool.map(lambda k0: ref(cr0,5,k0),starts0): print("refined SOC-off: min gap %.5f eV at (2pi/a)"%r[0],np.round(r[1],4),flush=True)
# fine scan along W-L
W=np.array([.25,.5,.75]); L=np.array([.5,.5,.5]); ts=np.linspace(0,1,81)
s1=np.array(list(pool.map(lambda t: gap(cr,W+(L-W)*t,11),ts))); s0=np.array(list(pool.map(lambda t: gap(cr0,W+(L-W)*t,5),ts)))
i=np.argmin(s1); j=np.argmin(s0); print("W-L line: SOC-on min %.4f eV at t=%.4f ; SOC-off min %.5f eV at t=%.4f"%(s1[i]*HA,ts[i],s0[j]*HA,ts[j]))
