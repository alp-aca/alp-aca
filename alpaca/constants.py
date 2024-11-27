from particle import Particle
import numpy as np
from .citations import citations, Constant
from .classes import LazyFloat

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
    import flavio
    citations.register_inspire('Straub:2018kue')
    return flavio.default_parameters.get_central_all()

pars = PreInitializedDict(getpars)

mW = Particle.from_name('W-').mass/1000
mt = Particle.from_name('t').mass/1000
# EW sector
GF = LazyFloat(lambda: pars['GF']) #GeV-2
s2w = LazyFloat(lambda: pars['s2w'])
# GF = sqrt(2)/8 g^2/mW^2
g2 = LazyFloat(lambda: (GF*mW**2*8/2**0.5)**0.5)
vev = LazyFloat(lambda: (2**0.5*GF)**(-0.5))
yuk_t = LazyFloat(lambda: mt*2**0.5/vev)

def getC10():
    import flavio
    citations.register_inspire('Straub:201')
    return flavio.physics.bdecays.wilsoncoefficients.wcsm_nf5(4.18)[9]

C10 = LazyFloat(getC10)

# masses (in GeV)
me = Constant(Particle.from_name('e-').mass/1000, 'particle')
mmu = Constant(Particle.from_name('mu-').mass/1000, 'particle')
mtau = Constant(Particle.from_name('tau-').mass/1000, 'particle')
mu = Constant(Particle.from_name('u').mass/1000, 'particle')
md = Constant(Particle.from_name('d').mass/1000, 'particle')
ms = Constant(Particle.from_name('s').mass/1000, 'particle')
mc = Constant(Particle.from_name('c').mass/1000, 'particle')
mb = Constant(Particle.from_name('b').mass/1000, 'particle')
mt = Constant(mt, 'particle')
mpi_pm = Constant(Particle.from_name('pi-').mass/1000, 'particle')
mpi0 = Constant(Particle.from_name('pi0').mass/1000, 'particle')
mW = Constant(mW, 'particle')
mZ = Constant(Particle.from_name('Z0').mass/1000, 'particle')
mB = Constant(Particle.from_name('B+').mass/1000, 'particle')
mB0 = Constant(Particle.from_name('B0').mass/1000, 'particle')
mBs = Constant(Particle.from_name('B(s)0').mass/1000, 'particle')
mK = Constant(Particle.from_name('K+').mass/1000, 'particle')
mKL = Constant(Particle.from_name('K(L)0').mass/1000, 'particle')
mKS = Constant(Particle.from_name('K(S)0').mass/1000, 'particle')
mKst0 = Constant(Particle.from_name('K*(892)0').mass/1000, 'particle')
meta = Constant(Particle.from_name("eta").mass/1000, 'particle')
metap = Constant(Particle.from_name("eta'(958)").mass/1000, 'particle')
mrho = Constant(Particle.from_name("rho(770)0").mass/1000, 'particle')
mJpsi = Constant(Particle.from_name("J/psi(1S)").mass/1000, 'particle')
mUpsilon1S = Constant(Particle.from_name("Upsilon(1S)").mass/1000, 'particle')
mUpsilon2S = Constant(Particle.from_name("Upsilon(2S)").mass/1000, 'particle')
mUpsilon3S = Constant(Particle.from_name("Upsilon(3S)").mass/1000, 'particle')
mUpsilon4S = Constant(Particle.from_name("Upsilon(4S)").mass/1000, 'particle')
ma0 = Constant(Particle.from_name("a(0)(980)0").mass/1000, 'particle')
msigma = Constant(Particle.from_name("f(0)(500)").mass/1000, 'particle') # f0(500) used to be sigma
mf0 = Constant(Particle.from_name("f(0)(980)").mass/1000, 'particle')
mf2 = Constant(Particle.from_name("f(2)(1270)").mass/1000, 'particle')
momega = Constant(Particle.from_pdgid(223).mass/1000, 'particle')

mq_dict = {'u': mu, 'd': md, 's': ms, 'c': mc, 'b': mb, 't': mt}

# widths (in GeV)
GammaB = Constant(Particle.from_name('B+').width/1000, 'particle')
GammaB0 = Constant(Particle.from_name('B0').width/1000, 'particle')
GammaBs = Constant(Particle.from_name('B(s)0').width/1000, 'particle')
GammaK = Constant(Particle.from_name('K+').width/1000, 'particle')
GammaKL = Constant(Particle.from_name('K(L)0').width/1000, 'particle')
GammaKS = Constant(Particle.from_name('K(S)0').width/1000, 'particle')
GammaJpsi = Constant(Particle.from_name('J/psi(1S)').width/1000, 'particle')
GammaUpsilon1S = Constant(Particle.from_name('Upsilon(1S)').width/1000, 'particle')
GammaUpsilon2S = Constant(Particle.from_name('Upsilon(2S)').width/1000, 'particle')
GammaUpsilon3S = Constant(Particle.from_name('Upsilon(3S)').width/1000, 'particle')
GammaUpsilon4S = Constant(Particle.from_name('Upsilon(4S)').width/1000, 'particle')
Gammaa0 = Constant(Particle.from_name("a(0)(980)0").width/1000, 'particle')
Gammasigma = Constant(Particle.from_name("f(0)(500)").width/1000, 'particle') # f0(500) used to be sigma
Gammaf0 = Constant(Particle.from_name("f(0)(980)").width/1000, 'particle')
Gammaf2 = Constant(Particle.from_name("f(2)(1270)").width/1000, 'particle')
Gammarho = Constant(Particle.from_name("rho(770)0").width/1000, 'particle')

# Mixing angle
theta_eta_etap = Constant(-14.1/180*np.pi, 'Christ:2010dd')

# Form factors
fB = LazyFloat(lambda: pars['f_B+'])
fBs = LazyFloat(lambda: pars['f_Bs'])
fK = LazyFloat(lambda: pars['f_K+'])
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