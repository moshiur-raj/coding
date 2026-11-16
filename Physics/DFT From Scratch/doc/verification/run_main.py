from scf import *
import pickle
from scipy.optimize import minimize
a=4.0782/BOHR; EC=float(sys.argv[1]); NM=int(sys.argv[2]); XC=sys.argv[3]; tag=sys.argv[4]
cr=Crystal(a,EC,XC,soc=True); print("N",cr.N, flush=True)
r=run_scf(cr,NM,20,maxit=40,gamma_centered=True)
mu=r['mu']; Vr=r['Vr']; np.save(f'n_{tag}.npy',r['n'])
print("mu eV",mu*HA, "F",r['F'])
eps=r['eps']; print("SCF-mesh min direct gap 12-13 (eV):",(eps[:,12]-eps[:,11]).min()*HA, "at k",r['K'][np.argmin(eps[:,12]-eps[:,11])])
print("max Kramers splitting on mesh (Ha):",np.abs(eps[:,0:12:2]-eps[:,1:12:2]).max())
cr0=Crystal(a,EC,XC,soc=False,spinor=False)
rng=np.random.default_rng(5)
def solve(c,kf,nb=20,tol=1e-9):
    bs=c.basis(kf); X0=(rng.normal(size=(nb,c.ns,bs['ng']))+1j*rng.normal(size=(nb,c.ns,bs['ng'])))/(1+bs['kin'])
    st={}; lam,X=lobpcg_orth(c,bs,Vr,X0,200,tol=tol,nconv=14 if c.ns==2 else 8,stats=st)
    return lam,X,bs
def parities(kf,nb2=12):
    lam,X,bs=solve(cr,kf)
    n=np.round(2*np.asarray(kf)).astype(int)
    look={tuple(m):i for i,m in enumerate(bs['M'])}
    perm=np.array([look[tuple(-m-n)] for m in bs['M']])
    C=X[:nb2]; Pi=np.einsum('msg,nsg->mn',C.conj(),C[:,:,perm])
    ev=np.linalg.eigvalsh(0.5*(Pi+Pi.conj().T))
    # parity of each individual level (group degenerate)
    lev=[]; i=0
    while i<16:
        j=i
        while j+1<16 and abs(lam[j+1]-lam[i])<1e-6: j+=1
        Cs=X[i:j+1]; p=np.linalg.eigvalsh(np.einsum('msg,nsg->mn',Cs.conj(),Cs[:,:,perm]))
        lev.append(((lam[i]-mu)*HA,j-i+1,np.round(p,6))); i=j+1
    return lam,ev,lev
out={}
for name,kf in (("G",[0,0,0]),("X",[0,.5,.5]),("X",[.5,0,.5]),("X",[.5,.5,0]),("L",[.5,0,0]),("L",[0,.5,0]),("L",[0,0,.5]),("L",[.5,.5,.5])):
    lam,ev,lev=parities(kf)
    print(name,kf,"gap12-13 eV %.3f"%((lam[12]-lam[11])*HA),"Pi eig:",np.round(ev,5), "N-=",(ev<0).sum())
    for l in lev: print("    E-EF=%8.3f eV deg=%d parity=%s"%(l[0],l[1],l[2]))
    out[name+str(kf)]=(lam,ev,lev)
# scalar relativistic levels at the same potential (non-SCF comparison)
for name,kf in (("G",[0,0,0]),("X",[0,.5,.5]),("L",[.5,.5,.5]),("W",[.25,.5,.75]),("K",[.375,.375,.75])):
    lam0,_,_=solve(cr0,kf,12); lam1,_,_=solve(cr,kf)
    print(name,"SOC-off (E-EF eV):",np.round((lam0[:9]-mu)*HA,3)); print(name,"SOC-on  (E-EF eV):",np.round((lam1[:16:2]-mu)*HA,3))
# band structure
pts={'G':[0,0,0],'X':[0,.5,.5],'W':[.25,.5,.75],'L':[.5,.5,.5],'K':[.375,.375,.75]}
path=['G','X','W','L','G','K']; nseg=24; kl=[]; lab=[]
for s in range(len(path)-1):
    p0,p1=np.array(pts[path[s]]),np.array(pts[path[s+1]])
    for t in range(nseg): kl.append(p0+(p1-p0)*t/nseg)
kl.append(np.array(pts[path[-1]]))
pool=ThreadPoolExecutor(4)
b1=list(pool.map(lambda k: solve(cr,k,20,1e-7)[0],kl)); b0=list(pool.map(lambda k: solve(cr0,k,12,1e-7)[0],kl))
kc=np.array(kl)@cr.B; d=np.concatenate([[0],np.cumsum(np.linalg.norm(np.diff(kc,axis=0),axis=1))])
np.savez(f'bands_{tag}.npz',d=d,soc=np.array(b1),sr=np.array(b0),mu=mu,kl=np.array(kl))
g=np.array(b1)[:,12]-np.array(b1)[:,11]; print("path min gap12-13 eV:",g.min()*HA,"at",kl[np.argmin(g)])
g0=np.array(b0)[:,6]-np.array(b0)[:,5]; print("path min gap SOC-off (bands 6-7) eV:",g0.min()*HA,"at",kl[np.argmin(g0)])
# fine mesh scan of direct gap
K,w=tr_reduce(mp_mesh(10,True))
def gap(c,k,i): 
    lam=solve(c,k,20 if c.ns==2 else 12,1e-6)[0]; return lam[i+1]-lam[i]
g1=np.array(list(pool.map(lambda k: gap(cr,k,11),K))); g0=np.array(list(pool.map(lambda k: gap(cr0,k,5),K)))
o=np.argsort(g1)[:5]; print("mesh10 SOC-on smallest gaps eV:",g1[o]*HA,"at",K[o])
o0=np.argsort(g0)[:5]; print("mesh10 SOC-off smallest gaps eV:",g0[o0]*HA,"at",K[o0])
for k0 in K[o[:3]]:
    res=minimize(lambda k: gap(cr,k,11),k0,method='Nelder-Mead',options=dict(xatol=1e-4,fatol=1e-7,maxiter=150))
    print("refined SOC-on min gap eV %.4f at"%(res.fun*HA),res.x, "cart 2pi/a:",res.x@cr.B*a/(2*pi))
for k0 in K[o0[:3]]:
    res=minimize(lambda k: gap(cr0,k,5),k0,method='Nelder-Mead',options=dict(xatol=1e-4,fatol=1e-7,maxiter=150))
    print("refined SOC-off min gap eV %.5f at"%(res.fun*HA),res.x, "cart 2pi/a:",res.x@cr.B*a/(2*pi))
pickle.dump(out,open(f'trim_{tag}.pkl','wb'))
