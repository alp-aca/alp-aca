import numpy as np
from typing import Callable, Sequence
from ..rge import ALPcouplings
from ..uvmodels import ModelBase
from ..benchmarks import Benchmark
from ..sectors import Sector
from ..statistics import ChiSquaredList, get_chi2
from ..decays.decays import decay_width, branching_ratio, cross_section, alp_channels_decay_widths, alp_channels_branching_ratios
from ..decays.mesons.mixing import meson_mixing

try:
    import tqdm
    tqdm_installed = True
except ImportError:
    tqdm_installed = False

def my_range(ntot, verbose=False):
    if verbose:
        if tqdm_installed:
            return tqdm.tqdm(range(ntot), leave=False)
        else:
            return ((i, print(f'{i}/{ntot} ({i/ntot*100:.2f}%)'))[0] for i in range(ntot))
    else:
        return range(ntot)
    

class Axis:
    def __init__(self, values: Callable | Sequence | np.ndarray, axis: str, tex: str, name: str = '', units: str = ''):
        if axis in ['x', 'y', 'x_func', 'y_func', 'x_dep', 'y_dep']:
            self.axis = axis
        else:
            raise ValueError("Axis must be 'x', 'y', x_dep, y_dep, x_func, or y_func.")
        if '_func' in axis:
            self.func = values
            self.values = None
        else:
            self.values = np.array(values)
        self.tex = tex
        self.name = name
        self.units = units
    def __repr__(self):
        return f"Axis(axis='{self.axis}', values={self.values}, tex='{self.tex}', name='{self.name}', units='{self.units}')"
    

class Scan:
    def __init__(self,
                 model: ALPcouplings | ModelBase | Benchmark,
                 ma: float | Axis = 1.0,
                 fa: float | Axis = 1e3,
                 lambda_scale : float | Axis | None = None,
                 mu_scale : float | Axis | None = None,
                 brdark: float | Axis = 0.0,
                 model_pars: dict = {}):
        self.couplings = None
        self.model = model
        self.args = {
            'ma': ma,
            'fa': fa,
            'lambda_scale': lambda_scale,
            'mu_scale': mu_scale,
            'br_dark': brdark} | model_pars
        num_x = 0
        num_y = 0
        for key, val in self.args.items():
            if isinstance(val, Axis):
                if val.axis == 'x':
                    num_x += 1
                    self.x_axis_name = key
                    self.x_axis = val
                    self.x_dim = val.values.shape[0]
                elif val.axis == 'y':
                    num_y += 1
                    self.y_axis_name = key
                    self.y_axis = val
                    self.y_dim = val.values.shape[0]
        if num_x != 1 or num_y != 1:
            raise ValueError("Only one x-axis and one y-axis allowed in a scan.")
        for key, val in self.args.items():
            if isinstance(val, Axis):
                if val.axis == 'x_dep' and val.values.shape != self.x_axis.values.shape:
                    raise ValueError(f"x_dep axis '{key}' must have the same shape as the x-axis '{self.x_axis_name}'.")
                if val.axis == 'y_dep' and val.values.shape != self.y_axis.values.shape:
                    raise ValueError(f"y_dep axis '{key}' must have the same shape as the y-axis '{self.y_axis_name}'.")
                if val.axis == 'x_func':
                    self.args[key].values = np.apply_along_axis(val.func, 0, self.x_axis.values)
                if val.axis == 'y_func':
                    self.args[key].values = np.apply_along_axis(val.func, 0, self.y_axis.values)
        if isinstance(model, ModelBase):
            for par in model.model_parameters():
                if par not in model_pars:
                    raise ValueError(f"Model parameter '{par}' must be provided in model_pars.")
                if lambda_scale is None:
                    raise ValueError("lambda_scale must be provided for ModelBase objects.")
        else:
            if lambda_scale is not None:
                raise ValueError("lambda_scale is not a valid parameter for ALPcouplings or Benchmark objects.")
            
    def compute_grid(self, verbose = True, **kwargs):
        if self.couplings is not None:
            return self.couplings
        if isinstance(self.model, ModelBase):
            grid_pars = ['lambda_scale', 'mu_scale'] + self.model.model_parameters()
            pars_x = []
            pars_y = []
            for par in grid_pars:
                if isinstance(self.args[par], Axis):
                    if self.args[par].axis in ['x', 'x_func', 'x_dep']:
                        pars_x.append(par)
                    elif self.args[par].axis in ['y', 'y_func', 'y_dep']:
                        pars_y.append(par)
            if len(pars_x) == 0 and len(pars_y) == 0:
                model_args = {k: v for k, v in self.args.items() if k in self.model.model_parameters()}
                c1 = self.model.get_couplings(model_args, self.args['lambda_scale'])
                if self.args['mu_scale'] is not None:
                    if self.args['mu_scale'] < c1.ew_scale:
                        basis = 'VA_below'
                    else:
                        basis = 'derivative_above'
                    c2 = c1.match_run(self.args['mu_scale'], basis, **kwargs)
                else:
                    c2 = c1
                self.couplings = np.full((self.x_dim, self.y_dim), c2)
                return self.couplings
            if len(pars_x) == 0:
                y_couplings = []
                for i in my_range(self.y_dim, verbose):
                    model_args = {k: (self.args[k].values[i] if k in pars_y else self.args[k]) for k in self.model.model_parameters()}
                    lambda_val = self.args['lambda_scale'].values[i] if 'lambda_scale' in pars_y else self.args['lambda_scale']
                    c1 = self.model.get_couplings(model_args, lambda_val)
                    mu_val = self.args['mu_scale'].values[i] if 'mu_scale' in pars_y else self.args['mu_scale']
                    if mu_val is not None:
                        if mu_val < c1.ew_scale:
                            basis = 'VA_below'
                        else:
                            basis = 'derivative_above'
                        c2 = c1.match_run(mu_val, basis, **kwargs)
                    else:
                        c2 = c1
                    y_couplings.append(c2)
                _, self.couplings = np.meshgrid(np.zeros(self.x_dim), y_couplings)
                return self.couplings
            if len(pars_y) == 0:
                x_couplings = []
                for i in my_range(self.x_dim, verbose):
                    model_args = {k: (self.args[k].values[i] if k in pars_x else self.args[k]) for k in self.model.model_parameters()}
                    lambda_val = self.args['lambda_scale'].values[i] if 'lambda_scale' in pars_x else self.args['lambda_scale']
                    c1 = self.model.get_couplings(model_args, lambda_val)
                    mu_val = self.args['mu_scale'].values[i] if 'mu_scale' in pars_x else self.args['mu_scale']
                    if mu_val is not None:
                        if mu_val < c1.ew_scale:
                            basis = 'VA_below'
                        else:
                            basis = 'derivative_above'
                        c2 = c1.match_run(mu_val, basis, **kwargs)
                    else:
                        c2 = c1
                    x_couplings.append(c2)
                self.couplings, _ = np.meshgrid(x_couplings, np.zeros(self.y_dim))
                return self.couplings
            grids = {}
            for par in grid_pars:
                if par in pars_x:
                    grids[par] = np.meshgrid(self.args[par].values, np.zeros(self.y_dim))[0]
                elif par in pars_y:
                    grids[par] = np.meshgrid(np.zeros(self.x_dim), self.args[par].values)[1]
                else:
                    grids[par] = np.full((self.y_dim, self.x_dim), self.args[par])
            self.couplings = np.empty((self.y_dim, self.x_dim), dtype=object)
            for n in my_range(self.x_dim * self.y_dim, verbose):
                ix = n // self.y_dim
                iy = n % self.y_dim
                model_args = {k: grids[k][iy, ix] for k in self.model.model_parameters()}
                lambda_val = grids['lambda_scale'][iy, ix]
                c1 = self.model.get_couplings(model_args, lambda_val)
                mu_val = grids['mu_scale'][iy, ix]
                if mu_val is not None:
                    if mu_val < c1.ew_scale:
                        basis = 'VA_below'
                    else:
                        basis = 'derivative_above'
                    c2 = c1.match_run(mu_val, basis, **kwargs)
                else:
                    c2 = c1
                self.couplings[iy, ix] = c2
            return self.couplings
        elif isinstance(self.model, ALPcouplings):
            if not isinstance(self.args['mu_scale'], Axis):
                if self.args['mu_scale'] is not None:
                    if self.args['mu_scale'] < self.model.ew_scale:
                        basis = 'VA_below'
                    else:
                        basis = 'derivative_above'
                    c2 = self.model.match_run(self.args['mu_scale'], basis, **kwargs)
                else:
                    c2 = self.model
                self.couplings = np.full((self.x_dim, self.y_dim), c2)
                return self.couplings
            elif self.args['mu_scale'].axis in ['y', 'y_func', 'y_dep']:
                y_couplings = []
                for i in my_range(self.y_dim, verbose):
                    mu_val = self.args['mu_scale'].values[i]
                    if mu_val < self.model.ew_scale:
                        basis = 'VA_below'
                    else:
                        basis = 'derivative_above'
                    c2 = self.model.match_run(mu_val, basis, **kwargs)
                    y_couplings.append(c2)
                _, self.couplings = np.meshgrid(np.zeros(self.x_dim), y_couplings)
                return self.couplings
            elif self.args['mu_scale'].axis in ['x', 'x_func', 'x_dep']:
                x_couplings = []
                for i in my_range(self.x_dim, verbose):
                    mu_val = self.args['mu_scale'].values[i]
                    if mu_val < self.model.ew_scale:
                        basis = 'VA_below'
                    else:
                        basis = 'derivative_above'
                    c2 = self.model.match_run(mu_val, basis, **kwargs)
                    x_couplings.append(c2)
                self.couplings, _ = np.meshgrid(x_couplings, np.zeros(self.y_dim))
                return self.couplings
        elif isinstance(self.model, Benchmark):
            grid_pars = ['mu_scale'] + [self.model.model_parameter]
            pars_x = []
            pars_y = []
            for par in grid_pars:
                if isinstance(self.args[par], Axis):
                    if self.args[par].axis in ['x', 'x_func', 'x_dep']:
                        pars_x.append(par)
                    elif self.args[par].axis in ['y', 'y_func', 'y_dep']:
                        pars_y.append(par)
            if len(pars_x) == 0 and len(pars_y) == 0:
                model_args = self.args[self.model.model_parameter]
                c1 = self.model(model_args, 1000)
                if self.args['mu_scale'] is not None:
                    if self.args['mu_scale'] < c1.ew_scale:
                        basis = 'VA_below'
                    else:
                        basis = 'derivative_above'
                    c2 = c1.match_run(self.args['mu_scale'], basis, **kwargs)
                else:
                    c2 = c1
                self.couplings = np.full((self.x_dim, self.y_dim), c2)
                return self.couplings
            if len(pars_x) == 0:
                y_couplings = []
                for i in my_range(self.y_dim, verbose):
                    model_args = self.args[self.model.model_parameter].values[i] if self.model.model_parameter in pars_y else self.args[self.model.model_parameter]
                    c1 = self.model(model_args, 1000)
                    mu_val = self.args['mu_scale'].values[i] if 'mu_scale' in pars_y else self.args['mu_scale']
                    if mu_val is not None:
                        if mu_val < c1.ew_scale:
                            basis = 'VA_below'
                        else:
                            basis = 'derivative_above'
                        c2 = c1.match_run(mu_val, basis, **kwargs)
                    else:
                        c2 = c1
                    y_couplings.append(c2)
                _, self.couplings = np.meshgrid(np.zeros(self.x_dim), y_couplings)
                return self.couplings
            if len(pars_y) == 0:
                x_couplings = []
                for i in my_range(self.x_dim, verbose):
                    model_args = self.args[self.model.model_parameter].values[i] if self.model.model_parameter in pars_x else self.args[self.model.model_parameter]
                    c1 = self.model(model_args, 1000)
                    mu_val = self.args['mu_scale'].values[i] if 'mu_scale' in pars_x else self.args['mu_scale']
                    if mu_val is not None:
                        if mu_val < c1.ew_scale:
                            basis = 'VA_below'
                        else:
                            basis = 'derivative_above'
                        c2 = c1.match_run(mu_val, basis, **kwargs)
                    else:
                        c2 = c1
                    x_couplings.append(c2)
                self.couplings, _ = np.meshgrid(x_couplings, np.zeros(self.y_dim))
                return self.couplings
            grids = {}
            for par in grid_pars:
                if par in pars_x:
                    grids[par] = np.meshgrid(self.args[par].values, np.zeros(self.y_dim))[0]
                elif par in pars_y:
                    grids[par] = np.meshgrid(np.zeros(self.x_dim), self.args[par].values)[1]
                else:
                    grids[par] = np.full((self.y_dim, self.x_dim), self.args[par])
            self.couplings = np.empty((self.y_dim, self.x_dim), dtype=object)
            for n in my_range(self.x_dim * self.y_dim, verbose):
                ix = n // self.y_dim
                iy = n % self.y_dim
                model_args = grids[self.model.model_parameter][iy, ix]
                c1 = self.model(model_args, 1000)
                mu_val = grids['mu_scale'][iy, ix]
                if mu_val is not None:
                    if mu_val < c1.ew_scale:
                        basis = 'VA_below'
                    else:
                        basis = 'derivative_above'
                    c2 = c1.match_run(mu_val, basis, **kwargs)
                else:
                    c2 = c1
                self.couplings[iy, ix] = c2
            return self.couplings

            
    def _prepare_scan_params(self):
        if isinstance(self.args['ma'], Axis):
            if self.args['ma'].axis in ['x', 'x_func', 'x_dep']:
                ma = np.meshgrid(self.args['ma'].values, np.zeros(self.y_dim))[0]
            elif self.args['ma'].axis in ['y', 'y_func', 'y_dep']:
                ma = np.meshgrid(np.zeros(self.x_dim), self.args['ma'].values)[1]
        else:
            ma = self.args['ma']
        if isinstance(self.model, Benchmark):
            fa = 1000
        else:
            if isinstance(self.args['fa'], Axis):
                if self.args['fa'].axis in ['x', 'x_func', 'x_dep']:
                    fa = np.meshgrid(self.args['fa'].values, np.zeros(self.y_dim))[0]
                elif self.args['fa'].axis in ['y', 'y_func', 'y_dep']:
                    fa = np.meshgrid(np.zeros(self.x_dim), self.args['fa'].values)[1]
            else:
                fa = self.args['fa']
        if isinstance(self.args['br_dark'], Axis):
            if self.args['br_dark'].axis in ['x', 'x_func', 'x_dep']:
                br_dark = np.meshgrid(self.args['br_dark'].values, np.zeros(self.y_dim))[0]
            elif self.args['br_dark'].axis in ['y', 'y_func', 'y_dep']:
                br_dark = np.meshgrid(np.zeros(self.x_dim), self.args['br_dark'].values)[1]
        else:
            br_dark = self.args['br_dark']
        return ma, fa, br_dark

    def get_chi2(self, transitions: list[Sector | str | tuple] | Sector | str | tuple, exclude_projections: bool=True, **kwargs) -> ChiSquaredList:
        ma, fa, br_dark = self._prepare_scan_params()
        return get_chi2(transitions, ma, self.compute_grid(**kwargs), fa, br_dark=br_dark, exclude_projections=exclude_projections, **kwargs)
    
    def decay_width(self, transition: str, **kwargs):
        ma, fa, br_dark = self._prepare_scan_params()
        return decay_width(transition, ma, self.compute_grid(**kwargs), fa, br_dark=br_dark, **kwargs)
    
    def branching_ratio(self, transition: str, **kwargs):
        ma, fa, br_dark = self._prepare_scan_params()
        return branching_ratio(transition, ma, self.compute_grid(**kwargs), fa, br_dark=br_dark, **kwargs)
    
    def cross_section(self, transition: str, s: float, **kwargs):
        ma, fa, br_dark = self._prepare_scan_params()
        return cross_section(transition, ma, self.compute_grid(**kwargs), fa=fa, br_dark=br_dark, s=s, **kwargs)
    
    def alp_channels_decay_widths(self, **kwargs):
        ma, fa, br_dark = self._prepare_scan_params()
        return alp_channels_decay_widths(ma, self.compute_grid(**kwargs), fa, br_dark=br_dark, **kwargs)
    
    def alp_channels_branching_ratios(self, **kwargs):
        ma, fa, br_dark = self._prepare_scan_params()
        return alp_channels_branching_ratios(ma, self.compute_grid(**kwargs), fa, br_dark=br_dark, **kwargs)
    
    def meson_mixing(self, obs: str, **kwargs):
        ma, fa, br_dark = self._prepare_scan_params()
        return meson_mixing(obs, ma, self.compute_grid(**kwargs), fa, br_dark=br_dark, **kwargs)