import numpy as np
import scipy.stats
from ..decays.alp_decays.branching_ratios import total_decay_width
from ..decays.decays import branching_ratio, cross_section, decay_width, to_tex
from ..decays.mesons.mixing import mixing_observables, meson_mixing
from ..decays.mesons.decays import meson_widths
from ..decays.particles import particle_aliases
from ..constants import hbarc_GeVnm
from ..experimental_data.classes import MeasurementBase
from ..experimental_data.measurements_exp import get_measurements
from ..experimental_data.theoretical_predictions import get_th_uncert, get_th_value
from ..rge import ALPcouplings
from ..sectors import Sector, combine_sectors

class ChiSquared:
    def __init__(self, sector: Sector,
                 chi2_dict: dict[tuple[str, str], np.ndarray[float]],
                 dofs_dict: dict[tuple[str, str], np.ndarray[int]]):
        self.sector = sector
        self.chi2_dict = chi2_dict
        self.dofs_dict = dofs_dict
        self.name = self.sector.name

    def significance(self) -> np.ndarray[float]:
        chi2 = np.nansum([v for v in self.chi2_dict.values()], axis=0)
        ndof = np.sum([v for v in self.dofs_dict.values()], axis=0)
        p = 1 - scipy.stats.chi2.cdf(np.where(ndof == 0, np.nan, chi2), ndof)
        p = np.clip(p, 2e-16, 1)
        return scipy.stats.norm.ppf(1 - p/2)

    def __getitem__(self, meas: tuple[str, str]) -> 'ChiSquared':
        obs, experiment = meas
        if (obs, experiment) in self.chi2_dict.keys() and (obs, experiment) in self.dofs_dict.keys():
            s = Sector(obs + ' @ ' + experiment, to_tex(obs) + r'\ \mathrm{(' + experiment + ')}$', obs_measurements = {obs: set([experiment,])} , description=f'Measurement of {obs} at experiment {experiment}.')
            return ChiSquared(s, {(obs, experiment): self.chi2_dict[(obs, experiment)]}, {(obs, experiment): self.dofs_dict[(obs, experiment)]})
        else:
            raise KeyError(f'Unknown experiment {obs}, {meas}')
        
    def get_measurements(self) -> list[tuple[str, str]]:
        return list( set(self.chi2_dict.keys()) & set(self.dofs_dict.keys()) )

    def _ipython_key_completions_(self):
        return self.get_measurements()
    
    def split_measurements(self) -> list['ChiSquared']:
        results = []
        for m in self.get_measurements():
            obs, experiment = m
            s = Sector(str(obs) + ' @ ' + experiment, to_tex(obs)[:-1] + r'\ \mathrm{(' + experiment + ')}$', obs_measurements = {obs: set([experiment,])}, description=f'Measurement of {obs} at experiment {experiment}.')
            results.append(ChiSquared(s, {(obs, experiment): self.chi2_dict[m]}, {(obs, experiment): self.dofs_dict[m]}))
        return results
    
    def split_observables(self) -> list['ChiSquared']:
        results = []
        observables = set([obs for obs, _ in self.get_measurements()])
        for obs in observables:
            chi2_dict = {k: v for k, v in self.chi2_dict.items() if k[0] == obs}
            dofs_dict = {k: v for k, v in self.dofs_dict.items() if k[0] == obs}
            s = Sector(str(obs), to_tex(obs), obs_measurements={obs: set(k[1] for k in chi2_dict.keys())}, description=f'Measurements of {obs}.')
            results.append(ChiSquared(s, chi2_dict, dofs_dict))
        return results
    
    def set_plot_style(self, color: str | None = None, lw: float | None = None, ls: str | None = None):
        """Set the plot style of the sector.
        
        Parameters
        ----------
        color : str | None
            The color of the sector.
        lw : float | None
            The line width of the sector.
        ls : str | None
            The line style of the sector.
        """
        if color is not None:
            self.sector.color = color
        if lw is not None:
            self.sector.lw = lw
        if ls is not None:
            self.sector.ls = ls

    def _repr_markdown_(self) -> str:
        """Return a Markdown representation of the ChiSquared object."""
        return self.sector._repr_markdown_()

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
    elif particle_aliases.get(transition, '') in meson_widths.keys():
        br = decay_width(transition, ma, couplings, fa, br_dark, **kwargs_dw)
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

def combine_chi2(chi2: list[ChiSquared], name: str, tex: str, description: str = '') -> ChiSquared:
    """Combine chi-squared values from different measurements.

    Parameters
    ----------
    chi2 : list[ChiSquared]
        List of ChiSquared objects to be combined.
    name : str
        The name of the combined sector.
    tex : str
        The LaTeX representation of the combined sector name.
    description : str, optional
        A description of the combined sector (default is an empty string).
    """
    sector = combine_sectors([c.sector for c in chi2], name, tex, description)
    chi2_dict = {}
    dofs_dict = {}
    for c in chi2:
        chi2_dict |= c.chi2_dict
        dofs_dict |= c.dofs_dict
    return ChiSquared(sector, chi2_dict, dofs_dict)

def get_chi2(transitions: list[Sector | str | tuple] | Sector | str | tuple, ma: np.ndarray[float], couplings: np.ndarray[ALPcouplings], fa: np.ndarray[float], min_probability = 1e-3, exclude_projections=True, **kwargs) -> list[ChiSquared]:
    """Calculate the chi-squared values for a set of transitions.

    Parameters
    ----------
    transitions (list[str])
        List of transition identifiers.

    ma : np.ndarray[float]
        Mass of the ALP.

    couplings : np.ndarray[ALPcouplings]
        Coupling constants.

    fa : np.ndarray[float]
        Axion decay constant.

    min_probability (float, optional):
        Minimum probability for decay. Default is 1e-3.

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
    observables = set()
    obs_measurements = {}
    sectors: list[Sector] = []

    if isinstance(transitions, (Sector, str, tuple)):
        transitions = [transitions,]

    for t in transitions:
        if isinstance(t, Sector):
            if t.observables is not None:
                observables.update(t.observables)
            if t.obs_measurements is not None:
                for obs, measurements in t.obs_measurements.items():
                    if obs not in obs_measurements:
                        obs_measurements[obs] = set()
                    obs_measurements[obs].update(measurements)
            sectors.append(t)
        elif isinstance(t, (str, tuple)):
            observables.update([t])
            s = Sector(str(t), to_tex(t), observables=[t,], description=f'Observable {t}')
            sectors.append(s)

    dict_chi2 = {}
    for t in observables:
        measurements = get_measurements(t, exclude_projections=exclude_projections)
        for experiment, measurement in measurements.items():
            sm_pred = get_th_value(t)
            sm_uncert = get_th_uncert(t)
            dict_chi2[(t, experiment)] = chi2_obs(measurement, t, ma, couplings, fa, min_probability=min_probability, sm_pred=sm_pred, sm_uncert=sm_uncert, **kwargs)
    for t in obs_measurements.keys():
        if t not in dict_chi2:
            measurements = get_measurements(t, exclude_projections=exclude_projections)
            for experiment, measurement in measurements.items():
                if experiment in obs_measurements[t]:
                    sm_pred = get_th_value(t)
                    sm_uncert = get_th_uncert(t)
                    dict_chi2[(t, experiment)] = chi2_obs(measurement, t, ma, couplings, fa, min_probability=min_probability, sm_pred=sm_pred, sm_uncert=sm_uncert, **kwargs)
            
    results = []
    for s in sectors:
        chi2_dict = {}
        dofs_dict = {}
        for obs in dict_chi2.keys():
            if s.observables is not None and obs[0] in s.observables:
                chi2_dict |= {obs: dict_chi2[obs][0]}
                dofs_dict |= {obs: dict_chi2[obs][1]}
        for obs in obs_measurements.keys():
            if s.obs_measurements is not None and obs in s.obs_measurements:
                for experiment in s.obs_measurements[obs]:
                    if (obs, experiment) in dict_chi2:
                        chi2_dict[(obs, experiment)] = dict_chi2[(obs, experiment)][0]
                        dofs_dict[(obs, experiment)] = dict_chi2[(obs, experiment)][1]
        
        results.append(ChiSquared(s, chi2_dict, dofs_dict))

    return results