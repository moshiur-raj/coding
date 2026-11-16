"""Stand-alone numerical checks of formulas and constants quoted in the document."""
import numpy as np
from math import gamma, pi, sqrt, erfc, erf
from scipy import integrate, special

print("=== HEG / XC ===")
def eps_x(rs):
    n = 3/(4*pi*rs**3); return -0.75*(3/pi)**(1/3)*n**(1/3)
PW = {0:(0.031091,0.21370,7.5957,3.5876,1.6382,0.49294),
      1:(0.015545,0.20548,14.1189,6.1977,3.3662,0.62517),
      2:(0.016887,0.11125,10.357,3.6231,0.88026,0.49671)}
def G(rs,row):
    A,a1,b1,b2,b3,b4 = PW[row]
    return -2*A*(1+a1*rs)*np.log(1+1/(2*A*(b1*rs**.5+b2*rs+b3*rs**1.5+b4*rs**2)))
for rs in (1,2,3,4,5):
    print(rs, "ex=%.6f ec=%.6f ec(z=1)=%.6f  -alpha_c=%.6f" % (eps_x(rs), G(rs,0), G(rs,1), G(rs,2)))
print("0.4582 const:", 0.75*(3/pi)**(1/3)*(3/(4*pi))**(1/3))
f2 = 4/(9*(2**(1/3)-1)); print("f''(0)=", f2)
I,_ = integrate.quad(lambda x: x*x*(1+(1-x*x)/(2*x)*np.log(abs((1+x)/(1-x)))),0,1); print("int x^2 S =", I)
# inner integral check at k=0.5kF
k=0.3; val,_ = integrate.dblquad(lambda th, r: 2*pi*r*r*np.sin(th)/(k*k+r*r-2*k*r*np.cos(th)), 0,1, 0,pi)
print("inner integral", val, 2*pi*(1+(1-k*k)/(2*k)*np.log((1+k)/(1-k))))
print("kappa/mu/beta: mu=pi^2 beta/3 =", pi**2*0.066725/3, " gamma=", (1-np.log(2))/pi**2)
# vxc via rs derivative vs numerical derivative of n*eps
def exc_n(n):
    rs=(3/(4*pi*n))**(1/3); return n*(eps_x(rs)+G(rs,0))
n0=3/(4*pi*27); h=1e-6*n0
rs=3.0; d=1e-5
vx = (exc_n(n0+h)-exc_n(n0-h))/(2*h)
e = lambda r: eps_x(r)+G(r,0)
print("vxc num", vx, " via rs formula", e(rs)-rs/3*(e(rs+d)-e(rs-d))/(2*d), " vx=", -(3*n0/pi)**(1/3))

print("=== smearing ===")
x=np.linspace(-4,4,2001)
f0 = 0.5*special.erfc(x); s0=np.exp(-x*x)/(2*sqrt(pi))
f1 = f0 - x/(2*sqrt(pi))*np.exp(-x*x); s1=(1-2*x*x)*np.exp(-x*x)/(4*sqrt(pi))
fd = 1/(np.exp(x)+1); sd = -(fd*np.log(fd)+(1-fd)*np.log(1-fd))
for nm,f,s in (("gauss",f0,s0),("mp1",f1,s1),("fd",fd,sd)):
    print(nm, "max|ds/dx - x df/dx| =", np.abs(np.gradient(s,x)-x*np.gradient(f,x)).max(), " f range", f.min(), f.max())

print("=== HGH projectors ===")
def p(i,l,r,rl):
    return sqrt(2)*r**(l+2*(i-1))*np.exp(-r*r/(2*rl*rl))/(rl**(l+(4*i-1)/2)*sqrt(gamma(l+(4*i-1)/2)))
def pt(i,l,q,rl):
    x=q*rl; E=pi**0.25*np.exp(-x*x/2)
    T={(1,0):sqrt(2*rl**3),(2,0):2*sqrt(2*rl**3/15)*(3-x*x),(3,0):4/3*sqrt(2*rl**3/105)*(15-10*x*x+x**4),
       (1,1):2*sqrt(rl**5/3)*q,(2,1):4*sqrt(rl**5/105)*q*(5-x*x),(3,1):8/3*sqrt(rl**5/1155)*q*(35-14*x*x+x**4),
       (1,2):2*sqrt(2*rl**7/15)*q*q,(2,2):4/3*sqrt(2*rl**7/105)*q*q*(7-x*x),(3,2):8/3*sqrt(2*rl**7/15015)*q*q*(63-18*x*x+x**4)}
    return E*T[(i,l)]
rl=0.47; worst=0
for l in range(3):
    for i in (1,2,3):
        nrm,_=integrate.quad(lambda r:p(i,l,r,rl)**2*r*r,0,20)
        for q in (0.3,1.7,4.2):
            num,_=integrate.quad(lambda r:p(i,l,r,rl)*special.spherical_jn(l,q*r)*r*r,0,20,limit=200)
            worst=max(worst,abs(num-pt(i,l,q,rl)))
        print("l=%d i=%d norm=%.10f"%(l,i,nrm))
print("worst |bessel transform err| =", worst)
# overlap <p1|p2> (non-orthogonal projectors)
for l in range(3):
    o,_=integrate.quad(lambda r:p(1,l,r,rl)*p(2,l,r,rl)*r*r,0,20); print("l",l,"<p1|p2>=",o)

print("=== HGH local ===")
Z=11; rloc=0.59; C=[11.604428, 0.7, -0.3, 0.05]
def Vloc(r):
    t=r/rloc
    return -Z/r*erf(r/(sqrt(2)*rloc)) + np.exp(-t*t/2)*(C[0]+C[1]*t**2+C[2]*t**4+C[3]*t**6)
def VG(Gq):
    x=Gq*rloc
    return -4*pi*Z/Gq**2*np.exp(-x*x/2)+(2*pi)**1.5*rloc**3*np.exp(-x*x/2)*(C[0]+C[1]*(3-x*x)+C[2]*(15-10*x*x+x**4)+C[3]*(105-105*x*x+21*x**4-x**6))
for Gq in (0.5,2.0,5.0):
    # FT of short-range part: V + Z/r * exp(-eps r) trick avoided: transform V+Z erf/r analytically is the formula; do numerics on (V + Z/r) and add -4piZ/G^2
    num,_=integrate.quad(lambda r:4*pi*r*r*(Vloc(r)+Z/r)*np.sin(Gq*r)/(Gq*r),1e-12,40,limit=400)
    print(Gq, num-4*pi*Z/Gq**2, VG(Gq))
al,_=integrate.quad(lambda r:4*pi*r*r*(Vloc(r)+Z/r),1e-12,40)
print("alpha num", al, " formula", 2*pi*Z*rloc**2+(2*pi)**1.5*rloc**3*(C[0]+3*C[1]+15*C[2]+105*C[3]))

print("=== L.S ===")
def Lmats(l):
    m=np.arange(-l,l+1); d=2*l+1
    Lz=np.diag(m).astype(complex); Lp=np.zeros((d,d),complex)
    for a in range(d-1): Lp[a+1,a]=sqrt(l*(l+1)-m[a]*(m[a]+1))
    return Lz,Lp,Lp.conj().T
def Umat(l):
    d=2*l+1; U=np.zeros((d,d),complex)   # U[m, mu]
    for mu in range(-l,l+1):
        c=mu+l
        if mu>0: U[-mu+l,c]=1/sqrt(2); U[mu+l,c]=(-1)**mu/sqrt(2)
        elif mu==0: U[l,c]=1
        else:
            a=-mu; U[-a+l,c]=1j/sqrt(2); U[a+l,c]=-1j*(-1)**a/sqrt(2)
    return U
def LS_real(l):
    Lz,Lp,Lm=Lmats(l); U=Umat(l)
    assert np.allclose(U.conj().T@U,np.eye(2*l+1))
    Lz,Lp,Lm=[U.conj().T@X@U for X in (Lz,Lp,Lm)]
    return 0.5*np.block([[Lz,Lm],[Lp,-Lz]]), (Lz,(Lp+Lm)/2,(Lp-Lm)/(2j))
for l in (1,2):
    LS,(Lz,Lx,Ly)=LS_real(l)
    print("l=",l,"herm",np.allclose(LS,LS.conj().T),"eig",np.round(np.linalg.eigvalsh(LS),6))
    print("  L real parts zero:",np.abs(Lz.real).max(),np.abs(Lx.real).max(),np.abs(Ly.real).max())
LS,(Lz,Lx,Ly)=LS_real(1)
print("l=1 order (y,z,x): Lz=\n",np.round(Lz,3)); print("(Lz)_{x,y}=",Lz[2,0], "(Lx)_{y,z}=",Lx[0,1],"(Ly)_{z,x}=",Ly[1,2])
# verify U against explicit real harmonics numerically (l=2) on random directions
rng=np.random.default_rng(0); v=rng.normal(size=(50,3)); v/=np.linalg.norm(v,axis=1)[:,None]
th=np.arccos(v[:,2]); ph=np.arctan2(v[:,1],v[:,0])
for l in (1,2):
    Yc=np.array([special.sph_harm_y(l,m,th,ph) for m in range(-l,l+1)])   # [m, pts]
    Yr=(Umat(l).T@Yc)    # Y_real_mu = sum_m U[m,mu] Y_m
    x,y,z=v.T
    tab={1:[sqrt(3/(4*pi))*y,sqrt(3/(4*pi))*z,sqrt(3/(4*pi))*x],
         2:[sqrt(15/(4*pi))*x*y,sqrt(15/(4*pi))*y*z,sqrt(5/(16*pi))*(3*z*z-1),sqrt(15/(4*pi))*x*z,sqrt(15/(16*pi))*(x*x-y*y)]}[l]
    print("l",l,"real-harmonic table vs U:",np.abs(Yr-np.array(tab)).max())
# crystal-field + SOC model
LS2,_=LS_real(2); Peg=np.diag([0,0,1,0,1]*2).astype(complex)
for D in (2.0,):
    w=np.linalg.eigvalsh(D*Peg+LS2); print("model Delta=2xi eig/xi:",np.round(w,4))
w=np.linalg.eigvalsh(2.5*Peg+LS2); print("model Delta=2.5xi:",np.round(w,4))

print("=== Ewald ===")
def ewald(A,pos,Zs,eta,nmax=8):
    Om=abs(np.linalg.det(A)); B=2*pi*np.linalg.inv(A).T
    rng=np.arange(-nmax,nmax+1); I=np.array(np.meshgrid(rng,rng,rng,indexing='ij')).reshape(3,-1).T
    Gs=I@B; G2=(Gs**2).sum(1); msk=G2>1e-12
    S=sum(Zs[a]*np.exp(-1j*Gs@pos[a]) for a in range(len(Zs)))
    Er=2*pi/Om*np.sum(np.exp(-G2[msk]/(4*eta))/G2[msk]*np.abs(S[msk])**2)
    Rs=I@A; E=0
    for a in range(len(Zs)):
        for b in range(len(Zs)):
            d=np.linalg.norm(Rs+pos[a]-pos[b],axis=1); m=d>1e-10
            E+=0.5*Zs[a]*Zs[b]*np.sum(special.erfc(sqrt(eta)*d[m])/d[m])
    return Er+E-sqrt(eta/pi)*sum(z*z for z in Zs)-pi*sum(Zs)**2/(2*eta*Om)
a=4.0782/0.529177210903; A=a/2*np.array([[0,1,1],[1,0,1],[1,1,0.]])
for eta in (0.25,0.5,1.0,2.0):
    print(eta, "%.10f"%ewald(A,[np.zeros(3)],[11],eta))
Om=a**3/4; rws=(3*Om/(4*pi))**(1/3); print("a=",a,"Omega=",Om,"rws=",rws,"Madelung*:",ewald(A,[np.zeros(3)],[11],0.5)*rws/121)
print("|b|=",2*pi/a*sqrt(3)," kTF(rs=3)=",sqrt(4*(9*pi/4)**(1/3)/3/pi))
print("Npw 40,60:", Om/(6*pi**2)*80**1.5, Om/(6*pi**2)*120**1.5, " grid:", 2/pi*sqrt(80)*a/sqrt(2)+1, 2/pi*sqrt(120)*a/sqrt(2)+1)
