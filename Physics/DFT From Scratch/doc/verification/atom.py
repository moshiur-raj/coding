import numpy as np
from math import pi, sqrt, gamma
from scipy import special
from pw import PSP, pw92, HA
def p(i,l,r,rl): return sqrt(2)*r**(l+2*(i-1))*np.exp(-r*r/(2*rl*rl))/(rl**(l+(4*i-1)/2)*sqrt(gamma(l+(4*i-1)/2)))
def atom(xc="LDA", scf=True, soc=True, h=0.02, Rmax=24.0, occ=None, verbose=False):
    ps=PSP[xc]; N=int(Rmax/h); r=h*np.arange(1,N+1)
    Vloc=-ps['Z']/r*special.erf(r/(sqrt(2)*ps['rloc']))+np.exp(-0.5*(r/ps['rloc'])**2)*ps['C1']
    T=(np.diag(np.full(N,1/h**2))-np.diag(np.full(N-1,0.5/h**2),1)-np.diag(np.full(N-1,0.5/h**2),-1))
    chans=[(0,.5),(1,.5),(1,1.5),(2,1.5),(2,2.5)]
    occ=occ or {(0,.5):1.0,(2,1.5):4.0,(2,2.5):6.0}
    V=np.zeros(N); n=np.zeros(N); res={}
    for it in range(200 if scf else 1):
        nnew=np.zeros(N)
        for (l,j) in chans:
            lam=0 if l==0 else (l/2 if j>l else -(l+1)/2)
            hm=np.array(ps['h'][l])+ (lam*np.array(ps['k'][l]) if (soc and l>0) else 0)
            P=np.array([r*p(i,l,r,ps['r'][l]) for i in (1,2)])
            H=T+np.diag(l*(l+1)/(2*r*r)+Vloc+V)+h*P.T@hm@P
            w,U=np.linalg.eigh(H); res[(l,j)]=w[:2]
            if (l,j) in occ: u=U[:,0]/sqrt(h); nnew+=occ[(l,j)]*u*u/(4*pi*r*r)
        if not scf: break
        d=np.abs(nnew-n).max(); n=n+0.3*(nnew-n) if it else nnew
        q=4*pi*np.cumsum(n*r*r)*h; VH=q/r+4*pi*(np.cumsum((n*r)[::-1])[::-1]*h-n*r*h)
        nn=np.maximum(n,1e-14); rs=(3/(4*pi*nn))**(1/3); ex=-0.75*(3/pi)**(1/3)*nn**(1/3); ec,dec=pw92(rs)
        V=VH+4/3*ex+ec-rs/3*dec
        if d<1e-10: break
    return res
if __name__=='__main__':
    for h in (0.02,):
        b=atom(scf=False,h=h); print("bare ion LDA-psp, h=",h); [print("  l=%d j=%.1f  e1=%.6f e2=%.6f Ha"%(l,j,*w)) for (l,j),w in b.items()]
    b=atom(scf=False,soc=False); print("bare ion, SOC off"); [print("  l=%d  e1=%.6f e2=%.6f Ha"%(l,*w)) for (l,j),w in b.items() if j>l or l==0]
    s=atom(); print("neutral LDA atom d10 s1 (spherical, unpolarised)"); [print("  l=%d j=%.1f  e=%.5f Ha"%(l,j,w[0])) for (l,j),w in s.items()]
    print("5d SO splitting eV:",(s[(2,2.5)][0]-s[(2,1.5)][0])*HA, " 6p splitting eV:",(s[(1,1.5)][0]-s[(1,.5)][0])*HA)
