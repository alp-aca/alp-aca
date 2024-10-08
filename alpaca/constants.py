from particle import Particle
import flavio
import numpy as np
pars = flavio.default_parameters.get_central_all()

# masses (in GeV)
me = Particle.from_name('e-').mass/1000
mmu = Particle.from_name('mu-').mass/1000
mtau = Particle.from_name('tau-').mass/1000
mu = Particle.from_name('u').mass/1000
md = Particle.from_name('d').mass/1000
ms = Particle.from_name('s').mass/1000
mc = Particle.from_name('c').mass/1000
mb = Particle.from_name('b').mass/1000
mt = Particle.from_name('t').mass/1000
mpi_pm = Particle.from_name('pi-').mass/1000
mpi0 = Particle.from_name('pi0').mass/1000
mW = Particle.from_name('W-').mass/1000
mZ = Particle.from_name('Z0').mass/1000
mB = Particle.from_name('B+').mass/1000
mB0 = Particle.from_name('B0').mass/1000
mBs = Particle.from_name('B(s)0').mass/1000
mK = Particle.from_name('K+').mass/1000
mKL = Particle.from_name('K(L)0').mass/1000
mKS = Particle.from_name('K(S)0').mass/1000
mKst0 = Particle.from_name('K*(892)0').mass/1000
meta = Particle.from_name("eta").mass/1000
metap = Particle.from_name("eta'(958)").mass/1000
mrho = Particle.from_name("rho(770)0").mass/1000

mq_dict = {'u': mu, 'd': md, 's': ms, 'c': mc, 'b': mb, 't': mt}

# widths (in GeV)
GammaB = Particle.from_name('B+').width/1000
GammaB0 = Particle.from_name('B0').width/1000
GammaBs = Particle.from_name('B(s)0').width/1000
GammaK = Particle.from_name('K+').width/1000
GammaKL = Particle.from_name('K(L)0').width/1000
GammaKS = Particle.from_name('K(S)0').width/1000

# Form factors
fB = pars['f_B+']
fBs = pars['f_Bs']
fK = pars['f_K+']
fpi = pars['f_pi+']

# EW sector
GF = pars['GF'] #GeV-2
s2w = pars['s2w']
# GF = sqrt(2)/8 g^2/mW^2
g2 = (GF*mW**2*8/2**0.5)**0.5
vev = (2**0.5*GF)**(-0.5)
yuk_t = mt*2**0.5/vev
C10_SM = flavio.physics.bdecays.wilsoncoefficients.wcsm_nf5(4.18)[9]

hbar_GeVps = 6.582e-13

Vckm = np.matrix(flavio.physics.ckm.get_ckm(pars))