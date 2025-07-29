import particle.literals as particles
import numpy as np
from .biblio.biblio import citations, Constant
from .classes import LazyFloat
import warnings

import operator

class PreInitializedDict:
    _wrapped : dict | None = None
    _is_init = False

    def __init__(self, factory):
        self.__dict__['_factory'] = factory

    def _setup(self):
        self._wrapped = self._factory()
        self._is_init = True

    def new_method(func):
        def inner(self, *args, **kwargs):
            if not self._is_init:
                self._setup()
            return func(self._wrapped, *args, **kwargs)
        return inner
    
    __getitem__ = new_method(operator.getitem)
    def get(self, item, default):
        if not self._is_init:
            self._setup()
        return self._wrapped.get(item, default)
    
    __setitem__ = new_method(operator.setitem)
    def set(self, item, value):
        if not self._is_init:
            self._setup()
        self._wrapped[item] = value

def getpars():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        import flavio
    citations.register_inspire('Straub:2018kue')
    return flavio.default_parameters.get_central_all()

pars = PreInitializedDict(getpars)

mW = Constant(particles.W_minus.mass/1000, 'particle')
mt = Constant(particles.t.mass/1000, 'particle')
# EW sector
GF = LazyFloat(lambda: pars['GF']) #GeV-2
s2w = LazyFloat(lambda: pars['s2w'])
# GF = sqrt(2)/8 g^2/mW^2
g2 = LazyFloat(lambda: (GF*mW**2*8/2**0.5)**0.5)
vev = LazyFloat(lambda: (2**0.5*GF)**(-0.5))
yuk_t = LazyFloat(lambda: mt*2**0.5/vev)

def getC10():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        import flavio
    citations.register_inspire('Straub:201')
    return flavio.physics.bdecays.wilsoncoefficients.wcsm_nf5(4.18)[9]

C10 = LazyFloat(getC10)

def getC7():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        import flavio
    citations.register_inspire('Straub:201')
    return flavio.physics.bdecays.wilsoncoefficients.wcsm_nf5(4.18)[6]

C7 = LazyFloat(getC7)

def getC10sd():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        import flavio
    citations.register_inspire('Straub:201')
    citations.register_inspire('Bobeth:2013uxa')
    citations.register_inspire('Gorbahn:2006bm')
    wcdict = flavio.physics.kdecays.wilsoncoefficients.wilsoncoefficients_sm_sl(pars, 0)
    xi_t = flavio.physics.ckm.xi('t', 'sd')(pars)
    xi_c = flavio.physics.ckm.xi('c', 'sd')(pars)
    return wcdict['C10_t'] + xi_c / xi_t * wcdict['C10_c']

C10sdRe = LazyFloat(lambda: np.real(getC10sd()))
C10sdIm = LazyFloat(lambda: np.imag(getC10sd()))

# masses (in GeV)
me = Constant(particles.e_minus.mass/1000, 'particle')
mmu = Constant(particles.mu_minus.mass/1000, 'particle')
mtau = Constant(particles.tau_minus.mass/1000, 'particle')
mu = Constant(particles.u.mass/1000, 'particle')
md = Constant(particles.d.mass/1000, 'particle')
ms = Constant(particles.s.mass/1000, 'particle')
mc = Constant(particles.c.mass/1000, 'particle')
mb = Constant(particles.b.mass/1000, 'particle')
mpi_pm = Constant(particles.pi_minus.mass/1000, 'particle')
mpi0 = Constant(particles.pi_0.mass/1000, 'particle')
mZ = Constant(particles.Z_0.mass/1000, 'particle')
mB = Constant(particles.B_plus.mass/1000, 'particle')
mB0 = Constant(particles.B_0.mass/1000, 'particle')
mBs = Constant(particles.B_s_0.mass/1000, 'particle')
mK = Constant(particles.K_plus.mass/1000, 'particle')
mK0 = Constant(particles.K_0.mass/1000, 'particle')
mKL = Constant(particles.K_L_0.mass/1000, 'particle')
mKS = Constant(particles.K_S_0.mass/1000, 'particle')
mKst0 = Constant(particles.Kst_892_0.mass/1000, 'particle')
mKst_plus = Constant(particles.Kst_892_plus.mass/1000, 'particle')
meta = Constant(particles.eta.mass/1000, 'particle')
metap = Constant(particles.etap_958.mass/1000, 'particle')
mrho = Constant(particles.rho_770_0.mass/1000, 'particle')
mrho_pm = Constant(particles.rho_770_plus.mass/1000, 'particle')
mphi = Constant(particles.phi_1020.mass/1000, 'particle')
mJpsi = Constant(particles.Jpsi_1S.mass/1000, 'particle')
mUpsilon1S = Constant(particles.Upsilon_1S.mass/1000, 'particle')
mUpsilon2S = Constant(particles.Upsilon_2S.mass/1000, 'particle')
mUpsilon3S = Constant(particles.Upsilon_3S.mass/1000, 'particle')
mUpsilon4S = Constant(particles.Upsilon_4S.mass/1000, 'particle')
ma0 = Constant(particles.a_0_980_0.mass/1000, 'particle')
ma0_pm = Constant(particles.a_0_980_plus.mass/1000, 'particle')
msigma = Constant(particles.f_0_500.mass/1000, 'particle') # f0(500) used to be sigma
mf0 = Constant(particles.f_0_980.mass/1000, 'particle')
mf2 = Constant(particles.f_2_1270.mass/1000, 'particle')
momega = Constant(particles.omega_782.mass/1000, 'particle')
mD0 = Constant(particles.D_0.mass/1000, 'particle')
mDplus = Constant(particles.D_plus.mass/1000, 'particle')
mDs = Constant(particles.D_s_plus.mass/1000, 'particle')

# widths (in GeV)
GammaB = Constant(particles.B_plus.width/1000, 'particle')
GammaB0 = Constant(particles.B_0.width/1000, 'particle')
GammaBs = Constant(particles.B_s_0.width/1000, 'particle')
GammaK = Constant(particles.K_plus.width/1000, 'particle')
GammaKL = Constant(particles.K_L_0.width/1000, 'particle')
GammaKS = Constant(particles.K_S_0.width/1000, 'particle')
GammaJpsi = Constant(particles.Jpsi_1S.width/1000, 'particle')
GammaUpsilon1S = Constant(particles.Upsilon_1S.width/1000, 'particle')
GammaUpsilon2S = Constant(particles.Upsilon_2S.width/1000, 'particle')
GammaUpsilon3S = Constant(particles.Upsilon_3S.width/1000, 'particle')
GammaUpsilon4S = Constant(particles.Upsilon_4S.width/1000, 'particle')
Gammaa0 = Constant(particles.a_0_980_0.width/1000, 'particle')
Gammaa0_pm = Constant(particles.a_0_980_plus.width/1000, 'particle')
Gammasigma = Constant(particles.f_0_500.width/1000, 'particle') # f0(500) used to be sigma
Gammaf0 = Constant(particles.f_0_980.width/1000, 'particle')
Gammaf2 = Constant(particles.f_2_1270.width/1000, 'particle')
Gammarho = Constant(particles.rho_770_0.width/1000, 'particle')
Gammarho_pm = Constant(particles.rho_770_plus.width/1000, 'particle')
GammaD0 = Constant(particles.D_0.width/1000, 'particle')
GammaDplus = Constant(particles.D_plus.width/1000, 'particle')
GammaDs = Constant(particles.D_s_plus.width/1000, 'particle')
Gammatau = Constant(particles.tau_minus.width/1000, 'particle')
Gammamu = Constant(particles.mu_minus.width/1000, 'particle')

# Mixing angle
theta_eta_etap = Constant(-14.1/180*np.pi, 'Christ:2010dd')

# Form factors
fB = LazyFloat(lambda: pars['f_B+'])
fBs = LazyFloat(lambda: pars['f_Bs'])
fD0 = LazyFloat(lambda: pars['f_D0'])
fK = LazyFloat(lambda: pars['f_K+'])
fK0 = LazyFloat(lambda: pars['f_K0'])
fKst = LazyFloat(lambda: pars['f_K*0'])
fpi = LazyFloat(lambda: pars['f_pi+'])
fJpsi = Constant(0.4104, 'Hatton:2020qhk')
fUpsilon1S = Constant(0.6772, 'Hatton:2021dvg')
fUpsilon2S = Constant(0.481, 'Colquhoun:2014ica')
fUpsilon3S = Constant(0.395, 'Chung:2020zqc')

# Branching ratios
BeeJpsi = Constant(5.971e-2, 'ParticleDataGroup:2024cfk')
BeeUpsilon1S = Constant(2.39e-2, 'ParticleDataGroup:2024cfk')
BeeUpsilon3S = Constant(2.18e-2, 'ParticleDataGroup:2024cfk')
BeeUpsilon4S = Constant(1.57e-5, 'ParticleDataGroup:2024cfk')

# Units and conversion factors
h_Js = Constant(6.62607015e-34, 'Mohr:2024kco')
e_C = Constant(1.602176634e-19, 'Mohr:2024kco')
c_nm_per_ps = Constant(299792.458, 'Mohr:2024kco')
hbar_GeVps = LazyFloat(lambda: h_Js/(e_C*2*np.pi)*1e3)
hbarc_GeVnm = LazyFloat(lambda: hbar_GeVps*c_nm_per_ps)
hbarc2_GeV2pb = LazyFloat(lambda: hbarc_GeVnm**2*1e22)

# Collider parameters
sigmaW_BaBar = Constant(5.5e-3, 'Merlo:2019anv') # See footnote in page 5
sigmaW_Belle = Constant(5.24e-3, 'Merlo:2019anv') # Ibidem
sigmaW_BESIII = Constant(3.686*5e-4, 'Song:2022umk')

#Vckm = np.matrix(flavio.physics.ckm.get_ckm(pars))
#for i in range(3):
#    for j in range(3):
#        Vckm[i,j] = ComplexConstant(Vckm[i,j], 'flavio')

g8 = Constant(3.61, 'Cirigliano:2011ny')
g2732= Constant(0.165, 'Cirigliano:2011ny')
g2712= Constant(0.033, 'Cirigliano:2011ny') #Isospin limit assumed
epsisos = Constant(0.028,'Cornella:2023kjq')
epsilonKaon = Constant(2.228e-3, 'ParticleDataGroup:2024cfk')
phiepsilonKaon = Constant(43.52/180*np.pi, 'ParticleDataGroup:2024cfk')

# Mixing effects in Bs decays
DeltaGamma_Bs = Constant(pars['DeltaGamma/Gamma_Bs'], 'Straub:2018kue')
lambdaB0 = Constant(0.35, 'Bosch:2002bv')

# D0 -> gamma gamma
b_D0gammagamma_VMD = Constant(9.3e-10, 'Burdman:2001tf')
c_D0gammagamma_VMD = Constant(9.4e-10, 'Burdman:2001tf')

# KL,S->ll long-distance contributions
re_ae = Constant(31.68, 'Hoferichter:2023wiy')
re_ae_error = Constant(0.98, 'Hoferichter:2023wiy')
re_amu = Constant(-0.16, 'Hoferichter:2023wiy')
re_amu_error = Constant(0.38, 'Hoferichter:2023wiy')
br_KLgammagamma = Constant(5.47e-4, 'ParticleDataGroup:2024cfk')
ie_disp = Constant(1.4, 'Ecker:1991ru')
ie_abs = Constant(-35, 'Ecker:1991ru')
imu_disp = Constant(-2.82, 'Ecker:1991ru')
imu_abs = Constant(1.21, 'Ecker:1991ru')
br_ksphotons_LD = Constant(2.17e-6, 'Cirigianno:2011ny')

## Neutral meson mixing
# B0
B1_Bd = Constant(0.806, 'Dowdall:2019bea')
B2_Bd = Constant(0.769, 'Dowdall:2019bea')
B3_Bd = Constant(0.747, 'Dowdall:2019bea')
B4_Bd = Constant(1.077, 'Dowdall:2019bea')
B5_Bd = Constant(0.973, 'Dowdall:2019bea')
# Bs
B1_Bs = Constant(0.813, 'Dowdall:2019bea')
B2_Bs = Constant(0.817, 'Dowdall:2019bea')
B3_Bs = Constant(0.816, 'Dowdall:2019bea')
B4_Bs = Constant(1.033, 'Dowdall:2019bea')
B5_Bs = Constant(0.941, 'Dowdall:2019bea')
# K0
B1_K0 = Constant(0.5268, 'Boyle:2024gge')
B2_K0 = Constant(0.5596, 'Boyle:2024gge')
B3_K0 = Constant(0.856, 'Boyle:2024gge')
B4_K0 = Constant(0.9097, 'Boyle:2024gge')
B5_K0 = Constant(0.750, 'Boyle:2024gge')
# D0
O1_D0 = Constant(0.0805, 'Bazavov:2017weg') #GeV^4
O2_D0 = Constant(-0.1561, 'Bazavov:2017weg') #GeV^4
O3_D0 = Constant(0.0464, 'Bazavov:2017weg') #GeV^4
O4_D0 = Constant(0.2747, 'Bazavov:2017weg') #GeV^4
O5_D0 = Constant(0.1035, 'Bazavov:2017weg') #GeV^4