import numpy as np
import flavio
from .constants import pars



def kallen(a, b, c):
    return a**2+b**2+c**2-2*a*b-2*a*c-2*b*c


def floop(x):
    if x >= 1:
        return np.arcsin(x**(-0.5))
    else:
        return np.pi/2+0.5j*np.log((1+np.sqrt(1-x))/(1-np.sqrt(1-x)))

B1 = lambda x: 1-x*floop(x)**2
B2 = lambda x: 1-(x-1)*floop(x)**2


alpha_em = lambda q: flavio.physics.running.running.get_alpha_e(pars, q)
alpha_s = lambda q: flavio.physics.running.running.get_alpha_s(pars, q)

f0_BK = lambda q2: flavio.physics.bdecays.formfactors.b_p.bcl.ff('B->K', q2, pars)['f0']
f0_Kpi = lambda q2: flavio.physics.kdecays.formfactors.fp0_dispersive(q2, pars)['f0']
A0_BKst = lambda q2: flavio.physics.bdecays.formfactors.b_v.bsz.ff('B->K*', q2, pars)['A0']

################################################
############ Quarkonia Parameters ##############
################################################

Qb=-1/3
Qc=2/3

mJPSI=3.096
GammaJpsi=0.0929*10**-3
fJpsi=0.414
BeeJPSI=5.971*10**-2

muppsilon1s=9.46030
BeeUpsilon1s=2.48*10**-3
fUps1S=0.680
GammaUps1S=54.02*10**-6

muppsilon3s=10.3552
GammaUps3S=20.32*10**-6
BeeUppsilon3s=2.18*10**-2
fUps3S=0.405

s_BESIII=mJPSI**2
sigmaW_BESIII=(3.686)*5*10**-4

BabarCM=muppsilon3s**2
sigmaW_babar=5.5*10**-3

Belle1CM=muppsilon1s**2
sigmaW_belle1=5.5*10**-3

Belle2CM=10.58



