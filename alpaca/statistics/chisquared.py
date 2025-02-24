import numpy as np
from ..decays.alp_decays.branching_ratios import total_decay_width
from ..decays.decays import branching_ratio, cross_section
from ..decays.mesons.mixing import mixing_observables, meson_mixing
from ..constants import hbarc_GeVnm
from ..experimental_data.classes import MeasurementBase
from ..experimental_data.measurements_exp import get_measurements
from ..experimental_data.theoretical_predictions import get_th_uncert, get_th_value
from ..rge import ALPcouplings

def chi2_obs(measurement: MeasurementBase, transition: str | tuple, ma, couplings, fa, min_probability=1e-3, br_dark = 0.0, sm_pred=0, sm_uncert=0, **kwargs):
    kwargs_dw = {k: v for k, v in kwargs.items() if k != 'theta'}
    ma = np.atleast_1d(ma).astype(float)
    couplings = np.atleast_1d(couplings)
    fa = np.atleast_1d(fa).astype(float)
    br_dark = np.atleast_1d(br_dark).astype(float)
    shape = np.broadcast_shapes(ma.shape, couplings.shape, fa.shape, br_dark.shape)
    ma = np.broadcast_to(ma, shape)
    couplings = np.broadcast_to(couplings, shape)
    fa = np.broadcast_to(fa, shape)
    br_dark = np.broadcast_to(br_dark, shape)
    if measurement.decay_type == 'flat':
        prob_decay = 1.0
        ctau = None # Arbitrary value
    else:
        dw = np.vectorize(lambda ma, coupl, fa, br_dark: total_decay_width(ma, coupl, fa, br_dark, **kwargs_dw)['DW_SM'])(ma, couplings, fa, br_dark)
        ctau = np.where(br_dark == 1.0, np.inf, 1e-7*hbarc_GeVnm/dw)
        prob_decay = measurement.decay_probability(ctau, ma, theta=kwargs.get('theta', None), br_dark=br_dark)
        prob_decay = np.where(prob_decay < min_probability, np.nan, prob_decay)
    if transition in mixing_observables:
        br = meson_mixing(transition, ma, couplings, fa, **kwargs_dw)
    elif isinstance(transition, str):
        br = branching_ratio(transition, ma, couplings, fa, br_dark, **kwargs_dw)
    else:
        br = cross_section(transition[0], ma, couplings, transition[1], fa, br_dark, **kwargs_dw)
    sigma_left = measurement.get_sigma_left(ma, ctau)
    sigma_right = measurement.get_sigma_right(ma, ctau)
    central = measurement.get_central(ma, ctau)
    value = prob_decay*br+sm_pred
    if measurement.conf_level is None:
        sigma = np.where(value > central, sigma_right, sigma_left)
        return (central - value)**2/(sigma**2 + sm_uncert**2), np.where(np.isnan(central), 0, 1)
    else:
        chi2 = np.where(value > central, (central - value)**2/sigma_right**2, 0)
        dofs = np.where(np.isnan(central), 0, np.where(value > central, 1, 0))
        return chi2, dofs

def combine_chi2(*chi2):
    chivalues = [c[0] for c in chi2]
    dofs = [c[1] for c in chi2]
    return np.nansum(chivalues, axis=0), np.sum(dofs, axis=0)

def get_chi2(transitions: list[str | tuple], ma: np.ndarray[float], couplings: np.ndarray[ALPcouplings], fa: np.ndarray[float], min_probability = 1e-3, exclude_projections=True, **kwargs) -> dict[tuple[str, str], np.array]:
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
            sm_pred = get_th_value(t)
            sm_uncert = get_th_uncert(t)
            dict_chi2[(t, experiment)] = chi2_obs(measurement, t, ma, couplings, fa, min_probability=min_probability, sm_pred=sm_pred, sm_uncert=sm_uncert, **kwargs)
    dict_chi2[('', 'Global')] = combine_chi2(*dict_chi2.values())
    return dict_chi2