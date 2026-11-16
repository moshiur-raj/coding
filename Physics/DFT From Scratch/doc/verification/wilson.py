from scf import *
a=4.0782/BOHR; EC=float(sys.argv[1]); N1=int(sys.argv[2]); N2=int(sys.argv[3]); NB=12
n=np.load('n_lda40.npy'); ng=n.shape[0]
cr40=Crystal(a,40.0,'LDA',soc=True,ngrid=ng); Vr,_=cr40.potential(n)
soc = (len(sys.argv)<5 or sys.argv[4]!='nosoc')
cr=Crystal(a,EC,'LDA',soc=soc,ngrid=ng)
def solve(kf):
    rng=np.random.default_rng(7)
    bs=cr.basis(kf); nb=NB+8
    X0=(rng.normal(size=(nb,2,bs['ng']))+1j*rng.normal(size=(nb,2,bs['ng'])))/(1+bs['kin'])
    lam,X=lobpcg_orth(cr,bs,Vr,X0,300,tol=1e-7,nconv=NB+1)
    return lam,X[:NB],bs
def box(X,bs,shift=(0,0,0)):
    N=cr.N; idx=np.ravel_multi_index(((bs['M']-np.array(shift))%N).T,(N,N,N))
    out=np.zeros((X.shape[0],2,N**3),complex); out[:,:,idx]=X; return out.reshape(X.shape[0],-1)
def wloop(k2,k3):
    ks=[np.array([j/N1,k2,k3]) for j in range(N1)]
    sols=[solve(k) for k in ks]
    gap=min(s[0][NB]-s[0][NB-1] for s in sols)
    Bx=[box(s[1],s[2]) for s in sols]; Bx.append(box(sols[0][1],sols[0][2],(1,0,0)))
    W=np.eye(NB,dtype=complex); smin=1
    for j in range(N1):
        M=Bx[j].conj()@Bx[j+1].T
        U,sg,Vh=np.linalg.svd(M); W=W@U@Vh; smin=min(smin,sg.min())
    return np.sort(np.angle(np.linalg.eigvals(W))), gap, smin
pool=ThreadPoolExecutor(4)
out={}
for k3 in (0.0,0.5):
    k2s=np.linspace(0,0.5,N2+1)
    res=list(pool.map(lambda k2: wloop(k2,k3),k2s))
    th=np.array([r[0] for r in res]); out[k3]=th
    print("k3=",k3,"min direct gap on loops eV %.3f"%(min(r[1] for r in res)*HA),"min singular value %.3f"%min(r[2] for r in res),flush=True)
    print(" theta/pi at k2=0  :",np.round(th[0]/pi,4)); print(" theta/pi at k2=1/2:",np.round(th[-1]/pi,4))
    # gap-following count (Soluyanov-Vanderbilt)
    def gapcenter(t):
        t=np.sort(t); d=np.diff(np.concatenate([t,[t[0]+2*pi]])); i=np.argmax(d); return (t[i]+d[i]/2+pi)%(2*pi)-pi, d[i]
    z=0; zc,_=gapcenter(th[0]); mingap=9
    for m in range(1,len(th)):
        zn,gw=gapcenter(th[m]); mingap=min(mingap,gw)
        lo,hi=zc,zn
        # count thetas of step m in the directed arc from zc to zn (shorter arc)
        dz=(zn-zc+pi)%(2*pi)-pi
        cnt=sum(1 for t in th[m] if 0 < ((t-zc+pi)%(2*pi)-pi)*np.sign(dz) < abs(dz)) if dz!=0 else 0
        z+=cnt; zc=zn
    print(" crossings counted:",z," -> z =",z%2," (smallest 'largest gap' = %.3f rad)"%mingap,flush=True)
np.savez('wilson_%s.npz'%('soc' if soc else 'nosoc'),k2=np.linspace(0,0.5,N2+1),th0=out[0.0],th1=out[0.5])
