import numpy as np
import flavio

pars = flavio.default_parameters.get_central_all()

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