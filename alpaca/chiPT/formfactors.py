import numpy as np
from ..common import alpha_s
from ..biblio.biblio import citations

def ff_BrodskyLepage(ma, exponent):
    citations.register_inspire('Brodsky:1974vy')
    citations.register_inspire('Lepage:1980fj')
    mu_pQCD = 1.4 # GeV
    if ma < mu_pQCD:
        return 1.0
    else:
        return alpha_s(ma)**2/alpha_s(mu_pQCD)**2 *(mu_pQCD/ma)**exponent