from particle import Particle
import numpy as np
from .citations import citations, Constant, ComplexConstant

import operator

class PreInitializedConst(float):
    _wrapped = float('nan')
    _is_init = False

    def __new__(self, factory):
        return float.__new__(self, float('nan'))

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
    
    __add__ = new_method(operator.add)
    __sub__ = new_method(operator.sub)
    __mul__ = new_method(operator.mul)
    __truediv__ = new_method(operator.truediv)

class PreInitializedDict:
    _wrapped = None
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

def getpars():
    import flavio
    citations.register_inspire('Straub:2018kue')
    return flavio.default_parameters.get_central_all()

pars = PreInitializedDict(getpars)

mW = Particle.from_name('W-').mass/1000
mt = Particle.from_name('t').mass/1000
# EW sector
GF = PreInitializedConst(lambda: pars['GF']) #GeV-2
s2w = PreInitializedConst(lambda: pars['s2w'])
# GF = sqrt(2)/8 g^2/mW^2
g2 = PreInitializedConst(lambda: (GF*mW**2*8/2**0.5)**0.5)
vev = PreInitializedConst(lambda: (2**0.5*GF)**(-0.5))
yuk_t = PreInitializedConst(lambda: mt*2**0.5/vev)

def getC10():
    import flavio
    citations.register_inspire('Straub:201')
    return flavio.physics.bdecays.wilsoncoefficients.wcsm_nf5(4.18)[9]

C10 = PreInitializedConst(getC10)

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

mq_dict = {'u': mu, 'd': md, 's': ms, 'c': mc, 'b': mb, 't': mt}

# widths (in GeV)
GammaB = Constant(Particle.from_name('B+').width/1000, 'particle')
GammaB0 = Constant(Particle.from_name('B0').width/1000, 'particle')
GammaBs = Constant(Particle.from_name('B(s)0').width/1000, 'particle')
GammaK = Constant(Particle.from_name('K+').width/1000, 'particle')
GammaKL = Constant(Particle.from_name('K(L)0').width/1000, 'particle')
GammaKS = Constant(Particle.from_name('K(S)0').width/1000, 'particle')

# Form factors
fB = PreInitializedConst(lambda: pars['f_B+'])
fBs = PreInitializedConst(lambda: pars['f_Bs'])
fK = PreInitializedConst(lambda: pars['f_K+'])
fpi = PreInitializedConst(lambda: pars['f_pi+'])

hbar_GeVps = Constant(6.582119569e-13, 'ParticleDataGroup:2024cfk')


#Vckm = np.matrix(flavio.physics.ckm.get_ckm(pars))
#for i in range(3):
#    for j in range(3):
#        Vckm[i,j] = ComplexConstant(Vckm[i,j], 'flavio')