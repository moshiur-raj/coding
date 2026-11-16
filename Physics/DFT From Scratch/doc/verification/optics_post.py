"""Plasma frequency, dielectric function, reflectivity and colour of gold from
the output of optics.py.   python3 optics_post.py optics_24_40_soc.npz [mu_Ha]"""
import numpy as np, sys
from math import pi, sqrt
HA = 27.211386; AU_TIME_FS = 2.4188843e-2; AU_VEL = 2.18769126e6; AU_COND = 4.599848e6   # eV, fs, m/s, S/m

def wsl_cmf(lam):
    """CIE 1931 2-degree colour matching functions, analytic fit of Wyman, Sloan and Shirley (2013); lam in nm."""
    def g(mu, s1, s2): return np.exp(-0.5*((lam-mu)/np.where(lam < mu, s1, s2))**2)
    x = 1.056*g(599.8, 37.9, 31.0)+0.362*g(442.0, 16.0, 26.7)-0.065*g(501.1, 20.4, 26.2)
    y = 0.821*g(568.8, 46.9, 40.5)+0.286*g(530.9, 16.3, 31.1)
    z = 1.217*g(437.0, 11.8, 36.0)+0.681*g(459.0, 26.0, 13.8)
    return x, y, z

def planck(lam, T=6500.0):
    return lam**-5.0/np.expm1(1.438777e7/(lam*T))            # lam in nm, c2 = 1.438777e7 nm K

def srgb_of_reflectance(R_of_lam):
    lam = np.arange(380.0, 781.0, 1.0)
    x, y, z = wsl_cmf(lam); I = planck(lam); R = R_of_lam(lam)
    XYZ = np.array([(I*R*c).sum() for c in (x, y, z)])/np.array([(I*c).sum() for c in (x, y, z)])  # white -> (1,1,1)
    XYZ = XYZ*np.array([0.95047, 1.0, 1.08883])                # express relative to the sRGB (D65) white
    M = np.array([[3.2406, -1.5372, -0.4986], [-0.9689, 1.8758, 0.0415], [0.0557, -0.2040, 1.0570]])
    lin = M@XYZ
    c = np.clip(lin, 0, 1)
    srgb = np.where(c <= 0.0031308, 12.92*c, 1.055*c**(1/2.4)-0.055)
    return XYZ, lin, np.round(255*srgb).astype(int)

def analyse(fn, mu, sdrude=0.005, eta_eV=0.10, gam_eV=0.07, verbose=True, scissor_eV=0.0, inter=True):
    d = np.load(fn); w, eps, v2, Om = d['w'], d['eps'], d['v2'], float(d['Om'])
    nb = eps.shape[1]
    # --- intraband: Drude weight; delta function = Gaussian of width sdrude (Ha)
    dl = np.exp(-((eps-mu)/sdrude)**2)/(sdrude*sqrt(pi))
    v2d = np.zeros_like(eps)
    for i in range(0, nb, 2):                                # trace over each Kramers pair
        blk = v2[:, i:i+2, i:i+2].sum((1, 2))/2
        v2d[:, i] = v2d[:, i+1] = blk                        # = <|v|^2>/3 per state of the pair
    gF = (w[:, None]*dl).sum()                               # states / Ha / cell
    wp2 = 4*pi/Om*(w[:, None]*dl*v2d).sum()                  # (4 pi / Omega) sum delta(e-mu) |v|^2/3
    vF = sqrt(3*(w[:, None]*dl*v2d).sum()/gF)
    # --- interband: weights S(omega_mn) on a fine histogram
    f = (eps < mu).astype(float)
    de = eps[:, None, :]-eps[:, :, None]                     # de[k,n,m] = e_m - e_n
    df = f[:, :, None]-f[:, None, :]                         # f_n - f_m
    sel = (de > 1e-5) & (df > 0)
    wgt = (w[:, None, None]*df*v2)[sel]; om = de[sel]
    hw = np.arange(0.02, 8.0001, 0.01)/HA; eta = eta_eV/HA
    oms = om+scissor_eV/HA                                  # optional rigid shift of all interband transitions (illustration only)
    eps_inter = np.array([1+8*pi/Om*np.sum(wgt/om**2*oms/(oms**2-(x+1j*eta)**2)) for x in hw]) if inter else np.ones(len(hw), complex)
    gam = gam_eV/HA
    eps_tot = eps_inter-wp2/(hw*(hw+1j*gam))
    nk = np.sqrt(eps_tot); R = np.abs((nk-1)/(nk+1))**2
    out = dict(hw=hw*HA, e1=eps_tot.real, e2=eps_tot.imag, e2i=eps_inter.imag, e1i=eps_inter.real, R=R,
               wp=sqrt(wp2)*HA, gF=gF/HA, vF=vF*AU_VEL, wp2=wp2, Om=Om, nk=len(w), ntrans=len(om))
    Rl = lambda lam: np.interp(1239.84193/lam, hw*HA, R)
    out['XYZ'], out['lin'], out['rgb'] = srgb_of_reflectance(Rl)
    out['Rl'] = Rl
    if verbose:
        print("%s: nk=%d  hbar*omega_p = %.3f eV   g(E_F) = %.4f states/eV/cell   v_F(rms) = %.3e m/s" % (fn, len(w), out['wp'], out['gF'], out['vF']))
        tau = 27.0/AU_TIME_FS
        print("   sigma/tau = omega_p^2/(4 pi) = %.5f a.u. = %.3e S/m per fs;  with tau = 27 fs: sigma = %.2e S/m (rho = %.2f micro-ohm cm)"
              % (wp2/(4*pi), wp2/(4*pi)*AU_COND/AU_TIME_FS, wp2/(4*pi)*tau*AU_COND, 1e8/(wp2/(4*pi)*tau*AU_COND)))
        print("   eps_inter(0) = %.2f ;  R at 450/500/550/600/650/700 nm:" % eps_inter.real[0], np.round([Rl(l) for l in (450, 500, 550, 600, 650, 700)], 3))
        i = np.argmax(np.gradient(out['e2i'][(hw*HA > 1) & (hw*HA < 3.5)])); print("   steepest rise of interband eps2 at %.2f eV" % (hw*HA)[(hw*HA > 1) & (hw*HA < 3.5)][i])
        print("   XYZ =", np.round(out['XYZ'], 4), " linear RGB =", np.round(out['lin'], 4), " sRGB =", out['rgb'], "#%02x%02x%02x" % tuple(out['rgb']))
    return out

if __name__ == '__main__':
    # self-tests of the colour chain
    print("white (R=1):", srgb_of_reflectance(lambda l: np.ones_like(l))[2], " grey 50%:", srgb_of_reflectance(lambda l: 0.5*np.ones_like(l))[2],
          " mirror above 600 nm only:", srgb_of_reflectance(lambda l: (l > 600)*1.0)[2], " below 480 nm only:", srgb_of_reflectance(lambda l: (l < 480)*1.0)[2])
    try:
        import colour
        cm = colour.MSDS_CMFS['CIE 1931 2 Degree Standard Observer']
        lam = np.arange(380.0, 781.0, 5.0); ex = cm[lam]; fit = np.array(wsl_cmf(lam)).T
        print("analytic CMF fit vs CIE table: max abs deviation", np.abs(ex-fit).max(axis=0).round(4), "(peaks ~1.06, 1.00, 1.78)")
    except Exception as e: print("colour-science not available:", e)
    mu = float(sys.argv[2]) if len(sys.argv) > 2 else 0.5832153855388598
    o = analyse(sys.argv[1], mu)
    hw = o['hw']/HA
    # sum rule and Kramers-Kronig consistency of the interband part
    print("   N_eff: intraband %.3f, interband up to 8 eV %.3f" % (o['wp2']*o['Om']/(4*pi), o['Om']/(2*pi**2)*np.trapezoid(hw*o['e2i'], hw)))
    big = analyse(sys.argv[1], mu, verbose=False)   # same data; KK needs eps2 on the whole range -> recompute on a long grid
    d = np.load(sys.argv[1]); w_, e_, v2_ = d['w'], d['eps'], d['v2']; f_ = (e_ < mu).astype(float)
    de = e_[:, None, :]-e_[:, :, None]; df = f_[:, :, None]-f_[:, None, :]; sel = (de > 1e-5) & (df > 0)
    wg = (w_[:, None, None]*df*v2_)[sel]; om = de[sel]; eta = 0.1/HA
    xx = np.arange(0.0005, 4.0, 0.001)                 # Ha, far beyond the highest transition
    e2 = np.array([(8*pi/o['Om']*np.sum(wg/(om*(om**2-(x+1j*eta)**2)))).imag for x in xx[::4]]); x4 = xx[::4]
    for x0 in (1.0/HA, 2.0/HA, 3.0/HA):
        # principal value by subtraction of the singular point
        e20 = np.interp(x0, x4, e2); integrand = (x4*e2-x0*e20)/(x4**2-x0**2)
        kk = 1+2/pi*np.trapezoid(integrand, x4)+2/pi*x0*e20*0.5/x0*np.log(abs((x4[-1]-x0)/(x4[-1]+x0))/abs((x4[0]-x0)/(x4[0]+x0)))
        direct = (1+8*pi/o['Om']*np.sum(wg/(om*(om**2-(x0+1j*eta)**2)))).real
        print("   Kramers-Kronig at %.0f eV: eps1 direct %.3f, from eps2 %.3f" % (x0*HA, direct, kk))
    for s in (0.003, 0.005, 0.01):
        q = analyse(sys.argv[1], mu, sdrude=s, verbose=False); print("   Drude broadening %.3f Ha: omega_p = %.3f eV, g(E_F) = %.4f /eV, v_F = %.3e" % (s, q['wp'], q['gF'], q['vF']))
    for eta_, gam in ((0.05, 0.07), (0.2, 0.07), (0.1, 0.025)):
        q = analyse(sys.argv[1], mu, eta_eV=eta_, gam_eV=gam, verbose=False); print("   eta=%.2f gamma=%.3f eV: sRGB" % (eta_, gam), q['rgb'])
    for sc in (0.25, 0.5, 0.75):
        q = analyse(sys.argv[1], mu, scissor_eV=sc, verbose=False); print("   interband transitions shifted by +%.2f eV: sRGB" % sc, q['rgb'], " R(450,550,650 nm) =", np.round([q['Rl'](l) for l in (450, 550, 650)], 2))
    q = analyse(sys.argv[1], mu, inter=False, verbose=False); print("   Drude only: sRGB", q['rgb'], " R(450,550,650) =", np.round([q['Rl'](l) for l in (450, 550, 650)], 3))
