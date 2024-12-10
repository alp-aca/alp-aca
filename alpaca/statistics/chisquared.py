import numpy as np
from ..decays.alp_decays.branching_ratios import total_decay_width
from ..decays.decays import branching_ratio
from ..constants import hbarc_GeVnm
from ..experimental_data.classes import MeasurementBase
from ..experimental_data.measurements_exp import get_measurements
from ..rge import ALPcouplings

def chi2_obs(measurement: MeasurementBase, transition: str, ma, couplings, fa, br_dark = 0.0, sm_pred=0, sm_uncert=0, **kwargs):
    kwargs_dw = {k: v for k, v in kwargs.items() if k != 'theta'}
    ma = np.atleast_1d(ma).astype(float)
    couplings = np.atleast_1d(couplings)
    fa = np.atleast_1d(fa).astype(float)
    br_dark = np.atleast_1d(br_dark).astype(float)
    dw = np.vectorize(lambda ma, coupl, fa, br_dark: total_decay_width(ma, coupl, fa, br_dark, **kwargs_dw)['DW_SM'])(ma, couplings, fa, br_dark)
    ctau = np.where(br_dark == 1.0, np.inf, 1e-7*hbarc_GeVnm/dw)
    prob_decay = measurement.decay_probability(ctau, ma, theta=kwargs.get('theta', None), br_dark=br_dark)
    br = branching_ratio(transition, ma, couplings, fa, br_dark, **kwargs_dw)
    return (measurement.get_central(ma, ctau) - prob_decay*br - sm_pred)**2/((measurement.get_sigma_left(ma, ctau)+measurement.get_sigma_right(ma, ctau))**2+ sm_uncert**2)

def combine_chi2(*chi2):
    ndof = np.sum(np.where(np.isnan(m), 0, 1) for m in chi2)
    return np.where(ndof == 0, np.nan, sum(np.nan_to_num(m) for m in chi2))/ndof

def get_chi2(transitions: list[str], ma: np.ndarray[float], couplings: np.ndarray[ALPcouplings], fa: np.ndarray[float], sm_pred=0, sm_uncert=0, exclude_projections=True, **kwargs) -> dict[tuple[str, str], np.array]:
    """Calculate the chi-squared values for a set of transitions.

    Parameters
    ----------
    transitions (list[str])
        List of transition identifiers.

    ma : np.ndarray[float]
        Mass of the ALP.

    couplings : np.ndarray[ALPcouplings]
        Coupling constants.

    fa : np[float]
        Axion decay constant.

    sm_pred (float, optional):
        Standard Model prediction. Default is 0.

    sm_uncert (float, optional):
        Standard Model uncertainty. Default is 0.

    exclude_projections (bool, optional):
        Whether to exclude projections from measurements. Default is True.
        
    **kwargs:
        Additional keyword arguments passed to chi2_obs.

    Returns
    -------
    chi2_dict : dict[tuple[str, str], np.array]
        Dictionary with keys as tuples of transition and experiment identifiers, 
        and values as numpy arrays of chi-squared values. Includes a special key 
        ('', 'Global') for the combined chi-squared value.
    """
    dict_chi2 = {}
    for t in transitions:
        measurements = get_measurements(t, exclude_projections=exclude_projections)
        for experiment, measurement in measurements.items():
            dict_chi2[(t, experiment)] = chi2_obs(measurement, t, ma, couplings, fa, sm_pred=sm_pred, sm_uncert=sm_uncert, **kwargs)
    dict_chi2[('', 'Global')] = combine_chi2(*dict_chi2.values())
    return dict_chi2