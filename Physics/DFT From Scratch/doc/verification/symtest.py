from scf import *
a=4.0782/BOHR; cr=Crystal(a,12.0,'LDA',soc=True)
rng=np.random.default_rng(3)
n=np.load('n_lda40.npy') if False else None
Vr=cr.Vloc_r + 0.0
# make an inversion-symmetric, real local potential with some structure
nG=11/cr.Om*np.exp(-cr.G2/2); Vr,_=cr.potential((np.fft.ifftn(nG)*cr.N**3).real)
for kf in ([0,.5,.5],[.5,.5,.5],[0,0,0],[.5,0,0]):
    bs=cr.basis(kf); nn=np.round(2*np.array(kf)).astype(int)
    look={tuple(m):i for i,m in enumerate(bs['M'])}; perm=np.array([look[tuple(-m-nn)] for m in bs['M']])
    c=rng.normal(size=(1,2,bs['ng']))+1j*rng.normal(size=(1,2,bs['ng']))
    P=lambda c: c[:,:,perm]
    def T(c):
        o=np.empty_like(c); o[:,0]=np.conj(c[:,1][:,perm]); o[:,1]=-np.conj(c[:,0][:,perm]); return o
    H=lambda c: cr.applyH(bs,c,Vr)
    print(kf,"[P,H]=%.1e [T,H]=%.1e T^2+1=%.1e [P,T]=%.1e"%(np.abs(H(P(c))-P(H(c))).max(),np.abs(H(T(c))-T(H(c))).max(),np.abs(T(T(c))+c).max(),np.abs(P(T(c))-T(P(c))).max()))
# break things on purpose: swap two real harmonics -> does [P,H] detect? (P only sees l parity) ; T test detects LS errors?
D0=cr.D.copy(); 
cr.D=D0.copy(); cr.D[cr.npj:, :cr.npj]*=-1; cr.D[:cr.npj, cr.npj:]*=-1   # wrong sign of spin-flip blocks (still Hermitian)
bs=cr.basis([0,.5,.5]); nn=np.array([0,1,1]); look={tuple(m):i for i,m in enumerate(bs['M'])}; perm=np.array([look[tuple(-m-nn)] for m in bs['M']])
c=rng.normal(size=(1,2,bs['ng']))+1j*rng.normal(size=(1,2,bs['ng']))
def T(c):
    o=np.empty_like(c); o[:,0]=np.conj(c[:,1][:,perm]); o[:,1]=-np.conj(c[:,0][:,perm]); return o
H=lambda c: cr.applyH(bs,c,Vr)
print("spin-flip sign error: [T,H]=%.1e [P,H]=%.1e"%(np.abs(H(T(c))-T(H(c))).max(),np.abs(H(c[:,:,perm])-(H(c))[:,:,perm]).max()), "eig LS-part d:",np.round(np.linalg.eigvalsh(cr.Hdense(bs,Vr))[:12],4))
cr.D=D0.conj()   # complex-conjugated LS (i.e. L -> -L): 
print("conj(D): [T,H]=%.1e"%(np.abs(H(T(c))-T(H(c))).max()), np.round(np.linalg.eigvalsh(cr.Hdense(bs,Vr))[:12],4))
cr.D=D0; print("correct:", np.round(np.linalg.eigvalsh(cr.Hdense(bs,Vr))[:12],4))
