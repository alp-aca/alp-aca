import numpy as np
import scipy.stats
import contourpy
from ..decays.alp_decays.branching_ratios import total_decay_width
from ..decays.decays import branching_ratio, cross_section, decay_width, to_tex, canonical_transition
from ..decays.mesons.mixing import mixing_observables, meson_mixing
from ..decays.mesons.decays import meson_widths
from ..decays.particles import particle_aliases
from ..constants import hbarc_GeVnm
from ..experimental_data.classes import MeasurementBase
from ..experimental_data.measurements_exp import get_measurements
from ..experimental_data.theoretical_predictions import get_th_uncert, get_th_value
from ..rge import ALPcouplings
from ..sectors import Sector, combine_sectors
from ..biblio import citation_report
from typing import Optional
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
        return np.nan_to_num(scipy.stats.norm.ppf(1 - p/2))
        
    def get_measurements(self) -> list[tuple[str, str]]:
        return list( set(self.chi2_dict.keys()) & set(self.dofs_dict.keys()) )
    
    def get_observables(self) -> list[str]:
        """Get a set of all observables from the ChiSquared object."""
        return list(set(obs for obs, _ in self.get_measurements()))

    def _ipython_key_completions_(self):
        return self.get_measurements()
    
    def split_measurements(self) -> 'ChiSquaredList':
        results = []
        for m in self.get_measurements():
            obs, experiment = m
            meas_name = f'{obs} @ {experiment}'
            meas_tex = f'${to_tex(obs).replace("$", "")} \\ \\mathrm{{({experiment.replace(" ", "\\ ")})}}$'
            s = Sector(meas_name, meas_tex, obs_measurements = {obs: set([experiment,])}, description=f'Measurement of {obs} at experiment {experiment}.')
            results.append(ChiSquared(s, {(obs, experiment): self.chi2_dict[m]}, {(obs, experiment): self.dofs_dict[m]}))
        return ChiSquaredList(results)

    def split_observables(self) -> 'ChiSquaredList':
        results = []
        observables = set(self.get_observables())
        for obs in observables:
            chi2_dict = {k: v for k, v in self.chi2_dict.items() if k[0] == obs}
            dofs_dict = {k: v for k, v in self.dofs_dict.items() if k[0] == obs}
            s = Sector(str(obs), to_tex(obs), obs_measurements={obs: set(k[1] for k in chi2_dict.keys())}, description=f'Measurements of {obs}.')
            results.append(ChiSquared(s, chi2_dict, dofs_dict))
        return ChiSquaredList(results)

    def extract_measurements(self, measurements: list[tuple[str, str]]) -> 'ChiSquared':
        """Extract ChiSquared objects for specific measurements from the ChiSquared object."""
        measurements = [(canonical_transition(m[0]), m[1]) for m in measurements]
        not_found = set(measurements) - set(self.get_measurements())
        if not_found:
            raise KeyError(f'Measurements {not_found} not found in ChiSquared object.')
        results = []
        split_measurements = self.split_measurements()
        for m in measurements:
            for chi2 in split_measurements:
                if m in chi2.get_measurements():
                    results.append(chi2)
        s = ChiSquaredList(results).combine(
            self.sector.name, self.sector.tex, self.sector.description)
        s.set_plot_style(
            color=self.sector.color, lw=self.sector.lw, ls=self.sector.ls)
        return s
    
    def extract_observables(self, observables: list[str]) -> 'ChiSquared':
        """Extract ChiSquared objects for specific observable(s) from the ChiSquared object."""
        observables = [canonical_transition(obs) for obs in observables]
        not_found = set(observables) - set(self.get_observables())
        if not_found:
            raise KeyError(f'Observables {not_found} not found in ChiSquared object.')
        results = []
        split_observables = self.split_observables()
        for obs in observables:
            for chi2 in split_observables:
                if obs in [m[0] for m in chi2.get_measurements()]:
                    results.append(chi2)
        s = ChiSquaredList(results).combine(
            self.sector.name, self.sector.tex, self.sector.description)
        s.set_plot_style(
            color=self.sector.color, lw=self.sector.lw, ls=self.sector.ls)
        return s
    
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

    def __repr__(self) -> str:
        return f'ChiSquared(sector="{self.sector.name}")'

    def _repr_markdown_(self) -> str:
        """Return a Markdown representation of the ChiSquared object."""
        return self.sector._repr_markdown_()
    
    def contour(self, x: np.ndarray[float], y: np.ndarray[float], sigma: float = 2.0) -> tuple[np.ndarray[float], np.ndarray[float]]:
        """Generate contour lines for the chi-squared significance.

        Parameters
        ----------
        x : np.ndarray[float]
            The x-coordinates of the data points.
        y : np.ndarray[float]
            The y-coordinates of the data points.
        sigma : float, optional
            The significance level for the contour (default is 2.0).

        Returns
        -------
        tuple[np.ndarray[float], np.ndarray[float]]
            The x and y coordinates of the contour lines.
        """
        if len(self.significance().shape) != 2:
            raise ValueError("Significance must be a 2D array for contours.")
        lines = contourpy.contour_generator(x, y, np.nan_to_num(self.significance()), line_type=contourpy.LineType.ChunkCombinedNan).lines(sigma)[0][0]
        if lines is None:
            raise ValueError(f"No contour found for significance level {sigma}.")
        return lines[:, 0], lines[:, 1]

    def contour_to_csv(self, x: np.ndarray[float], y: np.ndarray[float], filename: str, sigma: float = 2.0, xlabel: str = 'x', ylabel: str = 'y'):
        """Export the contour data to a CSV file.

        Parameters
        ----------
        x : np.ndarray[float]
            The x-coordinates of the data points.
        y : np.ndarray[float]
            The y-coordinates of the data points.
        filename : str
            The name of the output CSV file.
        sigma : float, optional
            The significance level for the contour (default is 2.0).
        xlabel : str, optional
            The label for the x-axis (default is 'x').
        ylabel : str, optional
            The label for the y-axis (default is 'y').
        """
        points = self.contour(x, y, sigma)
        with open(filename, 'w') as f:
            f.write(f"{xlabel},{ylabel}\n")
            for xi, yi in zip(points[0], points[1]):
                f.write(f"{xi},{yi}\n")

    def exclude_observables(self, observables: list[str]) -> Optional['ChiSquared']:
        """Exclude specific observable(s) from the ChiSquared object."""
        all_observables = set(self.get_observables())
        observables = [canonical_transition(obs) for obs in observables]
        remaining_observables = list(all_observables - set(observables))
        if not remaining_observables:
            return None
        return self.extract_observables(remaining_observables)

    def exclude_measurements(self, measurements: list[tuple[str, str]]) -> Optional['ChiSquared']:
        """Exclude specific measurement(s) from the ChiSquared object."""
        all_measurements = set(self.get_measurements())
        measurements = [(canonical_transition(m[0]), m[1]) for m in measurements]
        remaining_measurements = list(all_measurements - set(measurements))
        if not remaining_measurements:
            return None
        return self.extract_measurements(remaining_measurements)

    def get_inspire_ids(self) -> tuple[dict[tuple[str, str], list[str]], dict[str, str]]:
        """Get the Inspire IDs of the measurements in the ChiSquared object."""
        ids = {}
        bibtex = {}
        for obs, experiment in self.get_measurements():
            measurements = get_measurements(obs, exclude_projections=False)
            if experiment in measurements:
                meas_ids = measurements[experiment].inspire_id
                if isinstance(meas_ids, str):
                    meas_ids = [meas_ids,]
                ids[(obs, experiment)] = meas_ids
                if measurements[experiment].bibtex is not None:
                    bibtex.update(measurements[experiment].bibtex)
                    ids[((obs, experiment))].extend(measurements[experiment].bibtex.keys())
        return ids, bibtex
    
    def citation_report(self, filename: str):
        """Generate a citation report for the measurements in the ChiSquared object."""
        ids = self.get_inspire_ids()
        ids_tex = {f'${to_tex(k[0])}$ at {k[1]} ': v for k, v in ids[0].items()}
        citation_report(ids_tex, ids[1], filename)

    def shape(self) -> tuple[int, ...]:
        """Get the shape of the chi-squared values."""
        return self.significance().shape

    def constraining_measurements(self, mode: str = 'y-inverted') -> 'ChiSquaredList':
        """Get the constraining measurements for the ChiSquared object.

        Parameters
        ----------
        mode : str, optional
            The mode for constraining measurements. Options are:
             - 'y-inverted': Selects the measurements that constrain high values of the y-axis.
             - 'y': Selects the measurements that constrain low values of the y-axis.
             - 'grid': Selects the measurements that constrain each grid point.

        Returns
        -------
        ChiSquaredList
            A list of ChiSquared objects representing the constraining measurements.
        """
        if mode not in ['y-inverted', 'y', 'grid']:
            raise ValueError("Mode must be either 'y-inverted', 'y' or 'grid'.")
        
        sectors_plot = set()
        if mode == 'y-inverted':
            start_y = self.shape()[0] - 1
            delta_y = -1
            end_y = -1
        elif mode == 'y':
            start_y = 0
            delta_y = 1
            end_y = self.shape()[0]
        splitted = self.split_measurements()
        sigmas = [c.significance() for c in splitted]
        sigmas_max = np.clip(np.max(sigmas, axis=0), 2.0, 10)
        if mode in ['y-inverted', 'y']:
            if len(self.shape()) != 2:
                raise ValueError("ChiSquared object must have a 2D shape for 'y-inverted' or 'y' mode.")
            for x in range(self.shape()[1]):
                for y in range(start_y, end_y, delta_y):
                    new_sectors = set()
                    for chi2 in self.split_measurements():
                        if chi2.significance()[y,x] == sigmas_max[y,x]:
                            new_sectors.update(set(chi2.get_measurements()))
                    if len(new_sectors) > 0:
                        sectors_plot.update(new_sectors)
                        break
        elif mode == 'grid':
            for i, chi2 in enumerate(splitted):
                if np.any(sigmas[i] == sigmas_max):
                    sectors_plot.update(set(chi2.get_measurements()))

        return self.extract_measurements(list(sectors_plot)).split_measurements()

    def constraining_observables(self, mode: str = 'y-inverted') -> 'ChiSquaredList':
        """Get the constraining observables for the ChiSquared object.

        Parameters
        ----------
        mode : str, optional
            The mode for constraining observables. Options are:
             - 'y-inverted': Selects the observables that constrain high values of the y-axis.
             - 'y': Selects the observables that constrain low values of the y-axis.
             - 'grid': Selects the observables that constrain each grid point.

        Returns
        -------
        ChiSquaredList
            A list of ChiSquared objects representing the constraining observables.
        """
        if mode not in ['y-inverted', 'y', 'grid']:
            raise ValueError("Mode must be either 'y-inverted', 'y' or 'grid'.")
        
        sectors_plot = set()
        if mode == 'y-inverted':
            start_y = self.shape()[0] - 1
            delta_y = -1
            end_y = -1
        elif mode == 'y':
            start_y = 0
            delta_y = 1
            end_y = self.shape()[0]
        splitted = self.split_observables()
        sigmas = [c.significance() for c in splitted]
        sigmas_max = np.clip(np.max(sigmas, axis=0), 2.0, 10)
        if mode in ['y-inverted', 'y']:
            if len(self.shape()) != 2:
                raise ValueError("ChiSquared object must have a 2D shape for 'y-inverted' or 'y' mode.")
            for x in range(self.shape()[1]):
                for y in range(start_y, end_y, delta_y):
                    new_sectors = set()
                    for chi2 in self.split_observables():
                        if chi2.significance()[y,x] == sigmas_max[y,x]:
                            new_sectors.update(set(chi2.get_observables()))
                    if len(new_sectors) > 0:
                        sectors_plot.update(set(new_sectors))
                        break
        elif mode == 'grid':
            for i, chi2 in enumerate(splitted):
                if np.any(sigmas[i] == sigmas_max):
                    sectors_plot.update(set(chi2.get_observables()))

        return self.extract_observables(list(sectors_plot)).split_observables()
    
    def __getitem__(self, idx) -> "ChiSquared":
        return ChiSquared(self.sector,
                          {k: v[idx].copy() for k, v in self.chi2_dict.items()},
                          {k: v[idx].copy() for k, v in self.dofs_dict.items()})
    
    def slicing(self, *idx: tuple[slice|int]) -> "ChiSquared":
        """Slice the ChiSquared object along the specified indices.

        Parameters
        ----------
        *idx : tuple[slice|int]
            The indices to slice the ChiSquared object.
            At each index, you can specify a slice or an integer.
            If a slice is provided, it will slice the data along that dimension.
            If an integer is provided, it will select that specific index along that dimension.
        
        Returns
        -------
        ChiSquared
            A new ChiSquared object with the sliced data.
        """
        return self[*idx]

class ChiSquaredList(list[ChiSquared]):
    """A list of ChiSquared objects with additional methods for combining and manipulating them."""
    
    def combine(self, name: str, tex: str, description: str = '') -> ChiSquared:
        """Combine the chi-squared values from the list into a single ChiSquared object."""
        return combine_chi2(self, name, tex, description)
    
    def split_measurements(self) -> 'ChiSquaredList':
        """Split each ChiSquared object in the list into individual measurements."""
        return self.combine('', '').split_measurements()
    
    def split_observables(self) -> 'ChiSquaredList':
        """Split each ChiSquared object in the list into individual observables."""
        return self.combine('', '').split_observables()

    def extract_observables(self, observables: list[str]) -> 'ChiSquaredList':
        """Extract ChiSquared objects for a specific observable(s) from the list."""
        results = []
        observables = [canonical_transition(obs) for obs in observables]
        for chi2 in self:
            obs_sector = set(chi2.get_observables())
            obs_common = list(set(observables) & obs_sector)
            if obs_common:
                results.append(chi2.extract_observables(list(obs_common)))
        if not results:
            raise KeyError(f'Observables {observables} not found in ChiSquaredList.')
        s = ChiSquaredList(results)
        not_found = set(observables) - set(s.get_observables())
        if not_found:
            raise KeyError(f'Observables {not_found} not found in ChiSquaredList.')
        return s
    
    def extract_measurements(self, measurements: list[tuple[str, str]]) -> 'ChiSquaredList':
        """Extract ChiSquared objects for specific measurement(s) from the list."""
        results = []
        measurements = [(canonical_transition(m[0]), m[1]) for m in measurements]
        for chi2 in self:
            meas_sector = set(chi2.get_measurements())
            meas_common = list(set(measurements) & meas_sector)
            if meas_common:
                results.append(chi2.extract_measurements(list(meas_common)))
        if not results:
            raise KeyError(f'Measurements {measurements} not found in ChiSquaredList.')
        s = ChiSquaredList(results)
        not_found = set(measurements) - set(s.get_measurements())
        if not_found:
            raise KeyError(f'Measurements {not_found} not found in ChiSquaredList.')
        return s
    
    def exclude_observables(self, observables: list[str]) -> Optional['ChiSquaredList']:
        """Exclude specific observable(s) from the ChiSquaredList."""
        results = []
        observables = [canonical_transition(obs) for obs in observables]
        for chi2 in self:
            common_observables = set(chi2.get_observables()) & set(observables)
            exc_chi2 = chi2.exclude_observables(list(common_observables))
            if exc_chi2 is not None:
                results.append(exc_chi2)
        if not results:
            return None
        return ChiSquaredList(results)
    
    def exclude_measurements(self, measurements: list[tuple[str, str]]) -> Optional['ChiSquaredList']:
        """Exclude specific measurement(s) from the ChiSquaredList."""
        results = []
        measurements = [(canonical_transition(m[0]), m[1]) for m in measurements]
        for chi2 in self:
            common_measurements = set(chi2.get_measurements()) & set(measurements)
            exc_chi2 = chi2.exclude_measurements(list(common_measurements))
            if exc_chi2 is not None:
                results.append(exc_chi2)
        if not results:
            return None
        return ChiSquaredList(results)

    def contains_observable(self, observable: str) -> bool:
        """Check if any ChiSquared object in the list contains a specific observable."""
        return any(chi2.sector.contains_observable(observable) for chi2 in self)
    
    def get_measurements(self) -> list[tuple[str, str]]:
        """Get a list of all measurements (observable, experiment) from the ChiSquared objects in the list."""
        measurements = set()
        for chi2 in self:
            measurements.update(chi2.get_measurements())
        return list(measurements)
    
    def get_observables(self) -> list[str]:
        """Get a set of all observables from the ChiSquared objects in the list."""
        observables = set()
        for chi2 in self:
            observables.update(chi2.get_observables())
        return list(observables)
    
    def __str__(self) -> str:
        lnum = len(str(len(self)-1))
        return '\n'.join(f'{i:>{lnum}}:\t{chi2.sector.name}' for i, chi2 in enumerate(self))
    
    def _repr_markdown_(self) -> str:
        """Return a Markdown representation of the ChiSquaredList."""
        return '|Index|Sector|\n| :-: | :- |\n' + '\n'.join(f'|{i}|${chi2.sector.tex.replace('|', r'\|')}$|' for i, chi2 in enumerate(self))
    
    def get_inspire_ids(self) -> tuple[dict[tuple[str, str], list[str]], dict[str, str]]:
        """Get the Inspire IDs of the measurements in the ChiSquaredList."""
        ids = {}
        bibtex = {}
        for chi2 in self:
            data = chi2.get_inspire_ids()
            ids.update(data[0])
            if data[1]:
                bibtex.update(data[1])
        return ids, bibtex

    def citation_report(self, filename: str):
        """Generate a citation report for the measurements in the ChiSquared object."""
        ids = self.get_inspire_ids()
        ids_tex = {f'{to_tex(k[0])} at {k[1]}': v for k, v in ids[0].items()}
        citation_report(ids_tex, ids[1], filename)

    def constraining_measurements(self, mode: str = 'y-inverted') -> 'ChiSquaredList':
        """Get the constraining measurements for the ChiSquaredList.

        Parameters
        ----------
        mode : str, optional
            The mode for constraining measurements. Options are:
             - 'y-inverted': Selects the measurements that constrain high values of the y-axis.
             - 'y': Selects the measurements that constrain low values of the y-axis.
             - 'grid': Selects the measurements that constrain each grid point.

        Returns
        -------
        ChiSquaredList
            A list of ChiSquared objects representing the constraining measurements.
        """
        return self.combine('', '').constraining_measurements(mode)
    
    def constraining_observables(self, mode: str = 'y-inverted') -> 'ChiSquaredList':
        """Get the constraining observables for the ChiSquaredList.

        Parameters
        ----------
        mode : str, optional
            The mode for constraining observables. Options are:
             - 'y-inverted': Selects the observables that constrain high values of the y-axis.
             - 'y': Selects the observables that constrain low values of the y-axis.
             - 'grid': Selects the observables that constrain each grid point.

        Returns
        -------
        ChiSquaredList
            A list of ChiSquared objects representing the constraining observables.
        """
        return self.combine('', '').constraining_observables(mode)

    def significance(self) -> np.ndarray[float]:
        """Calculate the significance for the combined."""
        return self.combine('', '').significance()
    
    def slicing(self, *idx: tuple[slice|int]) -> 'ChiSquaredList':
        """Slice the ChiSquaredList along the specified indices.

        Parameters
        ----------
        *idx : tuple[slice|int]
            The indices to slice the ChiSquaredList.
            At each index, you can specify a slice or an integer.
            If a slice is provided, it will slice the data along that dimension.
            If an integer is provided, it will select that specific index along that dimension.
        
        Returns
        -------
        ChiSquaredList
            A new ChiSquaredList with the sliced data.
        """
        return ChiSquaredList([chi2.slicing(*idx) for chi2 in self])
    
    def contour(self, x: np.ndarray[float], y: np.ndarray[float], sigma: float = 2.0) -> tuple[np.ndarray[float], np.ndarray[float]]:
        """Generate contour lines for the chi-squared significance across all ChiSquared objects in the list.

        Parameters
        ----------
        x : np.ndarray[float]
            The x-coordinates of the data points.
        y : np.ndarray[float]
            The y-coordinates of the data points.
        sigma : float, optional
            The significance level for the contour (default is 2.0).

        Returns
        -------
        tuple[np.ndarray[float], np.ndarray[float]]
            The x and y coordinates of the contour lines.
        """
        return self.combine('', '').contour(x, y, sigma)
    
    def contour_to_csv(self, x: np.ndarray[float], y: np.ndarray[float], filename: str, sigma: float = 2.0, xlabel: str = 'x', ylabel: str = 'y'):
        """Export the contour data to a CSV file for all ChiSquared objects in the list.

        Parameters
        ----------
        x : np.ndarray[float]
            The x-coordinates of the data points.
        y : np.ndarray[float]
            The y-coordinates of the data points.
        filename : str
            The name of the output CSV file.
        sigma : float, optional
            The significance level for the contour (default is 2.0).
        xlabel : str, optional
            The label for the x-axis (default is 'x').
        ylabel : str, optional
            The label for the y-axis (default is 'y').
        """
        self.combine('', '').contour_to_csv(x, y, filename, sigma, xlabel, ylabel)

def chi2_obs(measurement: MeasurementBase, transition: str | tuple, ma, couplings, fa, min_probability=0.0, br_dark = 0.0, sm_pred=0, sm_uncert=0, **kwargs):
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
    with np.errstate(divide='ignore', invalid='ignore'):
        if measurement.conf_level is None:
            sigma = np.where(value > central, sigma_right, sigma_left)
            chi2, dofs = (central - value)**2/(sigma**2 + sm_uncert**2), np.where(np.isnan(central), 0, 1)
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

def get_chi2(transitions: list[Sector | str | tuple] | Sector | str | tuple, ma: np.ndarray[float], couplings: np.ndarray[ALPcouplings], fa: np.ndarray[float], min_probability: float = 0.0, br_dark = 0.0, exclude_projections=True, **kwargs) -> ChiSquaredList:
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
        Minimum probability for decay. Default is 0.0.
        If greater than 0, the processes with an on-shell ALP will only be included
        if the probability of the ALP decaying in the designated region (prompt, long-lived, etc.)
        is greater than this value.

    br_dark (float, optional):
        Branching ratio for dark sector decays. Default is 0.0.

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
            s = Sector(str(t), to_tex(t), observables=[t,], description=f'Observable {t}')
            observables.update(s.observables)
            sectors.append(s)

    dict_chi2 = {}
    for t in observables:
        measurements = get_measurements(t, exclude_projections=exclude_projections)
        for experiment, measurement in measurements.items():
            sm_pred = get_th_value(t)
            sm_uncert = get_th_uncert(t)
            dict_chi2[(t, experiment)] = chi2_obs(measurement, t, ma, couplings, fa, min_probability=min_probability, br_dark=br_dark, sm_pred=sm_pred, sm_uncert=sm_uncert, **kwargs)
    for t in obs_measurements.keys():
        if t not in dict_chi2:
            measurements = get_measurements(t, exclude_projections=exclude_projections)
            for experiment, measurement in measurements.items():
                if experiment in obs_measurements[t]:
                    sm_pred = get_th_value(t)
                    sm_uncert = get_th_uncert(t)
                    dict_chi2[(t, experiment)] = chi2_obs(measurement, t, ma, couplings, fa, min_probability=min_probability, br_dark=br_dark, sm_pred=sm_pred, sm_uncert=sm_uncert, **kwargs)
            
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

    return ChiSquaredList(results)