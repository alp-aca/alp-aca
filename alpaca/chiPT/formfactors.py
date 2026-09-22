import numpy as np
from ..common import alpha_s
from ..biblio.biblio import citations

def ff_BrodskyLepage(ma, exp_alpha, exp_scale, mu_pQCD=1.9):
    citations.register_inspire('Brodsky:1974vy')
    citations.register_inspire('Lepage:1980fj')
    if ma <= mu_pQCD:
        return 1.0
    return (alpha_s(ma)/alpha_s(mu_pQCD))**exp_alpha *(mu_pQCD/ma)**(2*exp_scale)