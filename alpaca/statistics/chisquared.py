import numpy as np
from .functions import tensor_meshgrid

def chi2_obs(observable, measurement, ma, obsargs):
    expv = np.vectorize(measurement, otypes=[float, float, float])(ma)
    ma_m, obsargs_m = tensor_meshgrid(ma, obsargs)
    expmean, _ = tensor_meshgrid(expv[0], obsargs)
    expuncert, _ = tensor_meshgrid(expv[2], obsargs)
    valid = np.where(expv[2] != 0)
    thpred = np.full(obsargs_m.shape, np.nan)
    thpred[valid,...] = observable(ma_m[valid,...], obsargs_m[valid,...])
    return (thpred - expmean)**2/expuncert**2

def combine_chi2(*chi2):
    ndof = np.sum(np.where(np.isnan(m), 0, 1) for m in chi2)
    return np.where(ndof == 0, np.nan, sum(np.nan_to_num(m) for m in chi2))/ndof