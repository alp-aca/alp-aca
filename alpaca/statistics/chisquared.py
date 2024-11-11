import numpy as np
from ..decays.alp_decays.branching_ratios import total_decay_width
from ..decays.decays import branching_ratio
from ..constants import hbarc_GeVnm
from ..experimental_data.classes import MeasurementBase

def chi2_obs(meausrement: MeasurementBase, transition: str, ma, couplings, fa, **kwargs):
    kwargs_dw = {k: v for k, v in kwargs.items() if k != 'theta'}
    dw = np.vectorize(lambda ma, coupl, fa: total_decay_width(ma, coupl, fa, **kwargs_dw)['DW_tot'])(ma, couplings, fa)
    ctau = 1e-7*hbarc_GeVnm/dw
    prob_decay = meausrement.decay_probability(ctau, ma, theta=kwargs.get('theta', None))
    br = branching_ratio(transition, ma, couplings, fa, **kwargs_dw)
    return (meausrement.get_central(ma, ctau) - prob_decay*br)**2/(meausrement.get_sigma_left(ma, ctau)+meausrement.get_sigma_right(ma, ctau))**2

def combine_chi2(*chi2):
    ndof = np.sum(np.where(np.isnan(m), 0, 1) for m in chi2)
    return np.where(ndof == 0, np.nan, sum(np.nan_to_num(m) for m in chi2))/ndof