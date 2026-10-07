import numpy as np
from scipy.interpolate import CubicSpline
from ..common import alpha_s
from ..biblio.biblio import citations

def ff_BrodskyLepage(ma, exp_alpha, exp_scale, mu_pQCD=1.9):
    citations.register_inspire('Brodsky:1974vy')
    citations.register_inspire('Lepage:1980fj')
    if ma <= mu_pQCD:
        return 1.0
    return (alpha_s(ma)/alpha_s(mu_pQCD))**exp_alpha *(mu_pQCD/ma)**(2*exp_scale)

class ff_BrodskyLepage_spline:
    def __init__(self, exp_alpha, exp_scale, mu_pQCD=1.9):
        def piecewise(ma):
            if ma < mu_pQCD:
                return 1.0
            else:
                return alpha_s(ma)**exp_alpha / alpha_s(mu_pQCD)**exp_alpha * mu_pQCD**(2*exp_scale) / ma**(2*exp_scale)
        ma_grid = np.hstack([np.linspace(0.0, 0.4*mu_pQCD, 200), np.linspace(1.6*mu_pQCD, 2.5*mu_pQCD, 200)])
        ff_grid = np.array([piecewise(ma) for ma in ma_grid])
        self.spline = CubicSpline(ma_grid, ff_grid)
        self.matching_scale = mu_pQCD
        self.exp_alpha = exp_alpha
        self.exp_scale = exp_scale

    def __call__(self, ma):
        if ma < 0.4 * self.matching_scale:
            return 1.0
        elif ma > 2.5 * self.matching_scale:
            return (alpha_s(ma) / alpha_s(self.matching_scale))**self.exp_alpha * (self.matching_scale / ma)**(2*self.exp_scale)
        else:
            return float(self.spline(ma))