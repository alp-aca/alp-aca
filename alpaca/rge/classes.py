"""Classes for RG evolution of the ALP couplings"""

import numpy as np
from .runSM import runSM
from ..citations import citations

from . import bases_above, bases_below
from functools import cache
class ALPcouplings:
    """Container for ALP couplings.

    Members
    -------
    values : dict
        dict containing the ALP couplings.

    scale : float
        Energy scale where the couplings are defined, in GeV.

    basis : str
        Basis in which the couplings are defined. The available bases are:

        * 'derivative_above':
            Basis with the explicitly shift-symmetric couplings of the fermion currents to the derivative of the ALP; above the EW scale.

    Raises
    ------
    ValueError
        If attempting to translate to an unrecognized basis.
    """
    def __init__(self, values: dict,
                 scale: float,
                 basis: str,
                 ew_scale: float = 100.0,
                 integrator: str = 'scipy',
                 beta: str = 'full',
                 match_2loops: bool = False,
                 min_scale: float|None = None,
                 ):
        """Constructor method

        Parameters
        -------
        values : dict
            dict containing the ALP couplings.

        scale : float
            Energy scale where the couplings are defined, in GeV.

        basis : str
            Basis in which the couplings are defined. The available bases are:

            - 'derivative_above':
                Basis with the explicitly shift-symmetric couplings of the fermion currents to the derivative of the ALP; above the EW scale.

        ew_scale : float, optional
            Energy scale of the electroweak symmetry breaking scale, in GeV. Defaults to 100 GeV

        integrator : str, optional
            Method to use for the RG evolution. The available integrators are:

            - 'scipy':
                Use the scipy.integrate.odeint function. (default)
            - 'leadinglog':
                Use the leading-log approximation.
            - 'no_rge':
                Return the couplings at the final scale without running them.

        beta : str, optional
            Beta function to use for the RG evolution. The available beta functions are:

            - 'full':
                Use the full beta function. (default)
            - 'ytop':
                Neglect the contributions of the Yukawa couplings other than the top Yukawa.
            
        match_2loops : bool, optional
            Whether to include some 2-loop matching corrections. Defaults to False.

        min_scale : float, optional
            Minimum scale at which the couplings are defined. Defaults to None.

        Raises
        ------
        ValueError
            If attempting to translate to an unrecognized basis.

        TypeError
            If attempting to assign a non-numeric value
        """
        citations.register_inspire('Bauer:2020jbp')
        self.ew_scale = ew_scale
        self.min_scale = min_scale
        if integrator not in ['scipy', 'leadinglog', 'no_rge']:
            raise ValueError(f'Unknown integrator {integrator}')
        self.integrator = integrator
        self.beta = beta
        if beta not in ['full', 'ytop']:
            raise ValueError(f'Unknown beta function {beta}')
        self.match_2loops = match_2loops
        if basis == 'derivative_above':
            if scale < ew_scale:
                raise ValueError('The scale must be above the electroweak scale when using the derivative_above basis')
            self.scale = scale
            self.basis = basis
            values = {'cg':0, 'cB': 0, 'cW':0, 'cqL': 0, 'cuR':0, 'cdR':0, 'clL':0, 'ceR':0} | values
            for c in ['cqL', 'cuR', 'cdR', 'clL', 'ceR']:
                if isinstance(values[c], (float, int)):
                    values[c] = np.matrix(values[c]*np.eye(3))
                elif isinstance(values[c], (np.ndarray, np.matrix, list)):
                    values[c] = np.matrix(values[c]).reshape([3,3])
                else:
                    raise TypeError(f'Invalid type for {c}')
            for c in ['cg', 'cW', 'cB']:
                if not isinstance(values[c], (int, float)):
                     raise TypeError(f'Invalid type for {c}')
            self.values = {c: values[c] for c in ['cg', 'cB', 'cW', 'cqL', 'cuR', 'cdR', 'clL', 'ceR']}
        elif basis == 'massbasis_above':
            if scale < ew_scale:
                raise ValueError('The scale must be above the electroweak scale when using the massbasis_above basis')
            self.scale = scale
            self.basis = basis
            values = {'cg': 0, 'cgamma':0, 'cgammaZ': 0, 'cW':0, 'cZ': 0, 'kU': 0, 'ku':0, 'kD':0, 'kd':0, 'kE':0, 'kNu': 0, 'ke': 0} | values
            for c in ['kU', 'ku', 'kD', 'kd', 'kE', 'kNu', 'ke']:
                if isinstance(values[c], (float, int)):
                    values[c] = np.matrix(values[c]*np.eye(3))
                elif isinstance(values[c], (np.ndarray, np.matrix, list)):
                    values[c] = np.matrix(values[c]).reshape([3,3])
                else:
                    raise TypeError(f'Invalid type for {c}')
            for c in ['cgamma', 'cgammaZ', 'cW', 'cZ', 'cg']:
                if not isinstance(values[c], (int, float)):
                     raise TypeError(f'Invalid type for {c}')
            self.values = {c: values[c] for c in ['kU', 'ku', 'kD', 'kd', 'kE', 'kNu', 'ke', 'cgamma', 'cgammaZ', 'cW', 'cZ', 'cg']}
        elif basis == 'kF_below':
            if scale > ew_scale:
                raise ValueError('The scale must be below the electroweak scale when using the kF_below basis')
            self.scale = scale
            self.basis = basis
            values = {'cg':0, 'cgamma': 0, 'kU': 0, 'kD': 0, 'kE': 0, 'kNu': 0, 'ku': 0, 'kd': 0, 'ke': 0} | values
            for c in ['kD', 'kE', 'kNu', 'kd', 'ke']:
                if isinstance(values[c], (float, int)):
                    values[c] = np.matrix(values[c]*np.eye(3))
                elif isinstance(values[c], (np.ndarray, np.matrix, list)):
                    values[c] = np.matrix(values[c]).reshape([3,3])
                else:
                    raise TypeError(f'Invalid type for {c}')
            for c in ['kU', 'ku']:
                if isinstance(values[c], (float, int)):
                    values[c] = np.matrix(values[c]*np.eye(2))
                elif isinstance(values[c], (np.ndarray, np.matrix, list)):
                    values[c] = np.matrix(values[c]).reshape([2,2])
                else:
                    raise TypeError(f'Invalid type for {c}')
            for c in ['cg', 'cgamma']:
                if not isinstance(values[c], (int, float)):
                     raise TypeError(f'Invalid type for {c}')
            self.values = {c: values[c] for c in ['kD', 'kE', 'kNu', 'kd', 'ke', 'kU', 'ku', 'cg', 'cgamma']}
        elif basis == 'VA_below':
            if scale > ew_scale:
                raise ValueError('The scale must be below the electroweak scale when using the VA_below basis')
            self.scale = scale
            self.basis = basis
            values = {'cg':0, 'cgamma': 0, 'cuV': 0, 'cuA': 0, 'cdV': 0, 'cdA': 0, 'ceV': 0, 'ceA': 0, 'cnu': 0} | values
            for c in ['cdV', 'cdA', 'ceV', 'ceA', 'cnu']:
                if isinstance(values[c], (float, int)):
                    values[c] = np.matrix(values[c]*np.eye(3))
                elif isinstance(values[c], (np.ndarray, np.matrix, list)):
                    values[c] = np.matrix(values[c]).reshape([3,3])
                else:
                    raise TypeError(f'Invalid type for {c}')
            for c in ['cuV', 'cuA']:
                if isinstance(values[c], (float, int)):
                    values[c] = np.matrix(values[c]*np.eye(2))
                elif isinstance(values[c], (np.ndarray, np.matrix, list)):
                    values[c] = np.matrix(values[c]).reshape([2,2])
                else:
                    raise TypeError(f'Invalid type for {c}')
            for c in ['cg', 'cgamma']:
                if not isinstance(values[c], (int, float)):
                     raise TypeError(f'Invalid type for {c}')
            self.values = {c: values[c] for c in ['cuV', 'cuA', 'cdV', 'cdA', 'ceV', 'ceA', 'cnu', 'cg', 'cgamma']}
        else:
            raise ValueError(f'Unknown basis {basis}')
    
    def __add__(self, other: 'ALPcouplings') -> 'ALPcouplings':
        if self.basis == other.basis:
            return ALPcouplings({k: self.values[k]+other.values[k] for k in self.values.keys()}, min(self.scale, other.scale), self.basis)
        
    def __mul__(self, a: float) -> 'ALPcouplings':
            return ALPcouplings({k: a*self.values[k] for k in self.values.keys()}, self.scale, self.basis)
                                
    def __rmul__(self, a: float) -> 'ALPcouplings':
            return ALPcouplings({k: a*self.values[k] for k in self.values.keys()}, self.scale, self.basis)
    
    def __getitem__(self, name: str):
         return self.values[name]
    
    def __setitem__(self, name: str, val):
        if self.basis == 'derivative_above':
            if name in ['cg', 'cW', 'cB']:
                if isinstance(val, (float, int)):
                    self.values[name] = val
                else:
                    raise TypeError(f'Invalid type for {name}')
            elif name in ['cqL', 'cuR', 'cdR', 'clL', 'ceR']:
                if isinstance(val, (float, int)):
                    self.values[name] = val * np.eye(3)
                elif isinstance(val, (np.ndarray, np.matrix, list)):
                    self.values[name] = np.matrix(val).reshape([3,3])
                else:
                    raise TypeError(f'Invalid type for {name}')
            else:
                raise KeyError
        elif self.basis == 'massbasis_above':
            if name in ['cg', 'cgamma', 'cgammaZ', 'cW', 'cZ']:
                if isinstance(val, (float, int)):
                    self.values[name] = val
                else:
                    raise TypeError(f'Invalid type for {name}')
            elif name in ['kU', 'ku', 'kD', 'kd', 'kE', 'kNu', 'ke']:
                if isinstance(val, (float, int)):
                    self.values[name] = val * np.eye(3)
                elif isinstance(val, (np.ndarray, np.matrix, list)):
                    self.values[name] = np.matrix(val).reshape([3,3])
                else:
                    raise TypeError(f'Invalid type for {name}')
            else:
                raise KeyError
        elif self.basis == 'kF_below':
            if name in ['cg', 'cgamma']:
                if isinstance(val, (float, int)):
                    self.values[name] = val
                else:
                    raise TypeError(f'Invalid type for {name}')
            elif name in ['kU', 'ku']:
                if isinstance(val, (float, int)):
                    self.values[name] = val * np.eye(2)
                elif isinstance(val, (np.ndarray, np.matrix, list)):
                    self.values[name] = np.matrix(val).reshape([2,2])
                else:
                    raise TypeError(f'Invalid type for {name}')
            elif name in ['kD', 'kE', 'kNu', 'kd', 'ke']:
                if isinstance(val, (float, int)):
                    self.values[name] = val * np.eye(3)
                elif isinstance(val, (np.ndarray, np.matrix, list)):
                    self.values[name] = np.matrix(val).reshape([3,3])
                else:
                    raise TypeError(f'Invalid type for {name}')
            else:
                raise KeyError
        elif self.basis == 'VA_below':
            if name in ['cg', 'cgamma']:
                if isinstance(val, (float, int)):
                    self.values[name] = val
                else:
                    raise TypeError(f'Invalid type for {name}')
            elif name in ['cuV', 'cuA']:
                if isinstance(val, (float, int)):
                    self.values[name] = val * np.eye(2)
                elif isinstance(val, (np.ndarray, np.matrix, list)):
                    self.values[name] = np.matrix(val).reshape([2,2])
                else:
                    raise TypeError(f'Invalid type for {name}')
            elif name in ['cdV', 'cdA', 'ceV', 'ceA', 'cnu']:
                if isinstance(val, (float, int)):
                    self.values[name] = val * np.eye(3)
                elif isinstance(val, (np.ndarray, np.matrix, list)):
                    self.values[name] = np.matrix(val).reshape([3,3])
                else:
                    raise TypeError(f'Invalid type for {name}')
            else:
                raise KeyError

    def translate(self, basis: str) -> 'ALPcouplings':
        """Translate the couplings to another basis at the same energy scale.
        
        Parameters
        ----------
        basis : str
            Target basis to translate.

        Returns
        -------
        a : ALPcouplings
            Translated couplings.

        Raises
        ------
        ValueError
            If attempting to translate to an unrecognized basis.
        """

        args = {'scale': self.scale, 'ew_scale': self.ew_scale, 'integrator': self.integrator, 'beta': self.beta, 'match_2loops': self.match_2loops, 'min_scale': self.min_scale}
        if basis == self.basis:
            return self
        if self.basis == 'derivative_above' and basis == 'massbasis_above':
            smpars = runSM(self.scale)
            s2w = smpars['s2w']
            c2w = 1-s2w
            Vckm = smpars['CKM']

            cgamma = self.values['cW'] + self.values['cB']
            cgammaZ = c2w * self.values['cW'] - s2w * self.values['cB']
            cZ = c2w**2 * self.values['cW'] + s2w**2 *self.values['cB']

            return ALPcouplings({'kU': self.values['cqL'], 'ku': self.values['cuR'], 'kD': Vckm.H @ self.values['cqL'] @ Vckm, 'kd': self.values['cdR'], 'kE': self.values['clL'], 'kNu': self.values['clL'], 'ke': self.values['ceR'], 'cgamma': cgamma, 'cW': self.values['cW'], 'cgammaZ': cgammaZ, 'cZ': cZ, 'cg': self.values['cg']}, basis='massbasis_above', **args)
        
        if self.basis == 'massbasis_above' and basis == 'derivative_above':
            smpars = runSM(self.scale)
            Vckm = smpars['CKM']

            cqL = self.values['kU']
            cuR = self.values['ku']
            cdR = self.values['kd']
            clL = self.values['kE']
            ceR = self.values['ke']

            cW = self.values['cW']
            cB = self.values['cgamma'] - self.values['cW']
            cg = self.values['cg']

            return ALPcouplings({'cg': cg, 'cB': cB, 'cW': cW, 'cqL': cqL, 'cuR': cuR, 'cdR': cdR, 'clL': clL, 'ceR': ceR}, basis='derivative_above', **args)
        
        if self.basis == 'kF_below' and basis == 'VA_below':
            return ALPcouplings({'cuV': self.values['ku'] + self.values['kU'],
                                 'cuA': self.values['ku'] - self.values['kU'],
                                 'cdV': self.values['kd'] + self.values['kD'],
                                 'cdA': self.values['kd'] - self.values['kD'],
                                 'ceV': self.values['ke'] + self.values['kE'],
                                 'ceA': self.values['ke'] - self.values['kE'],
                                 'cnu': self.values['kNu'], 'cg': self.values['cg'], 'cgamma': self.values['cgamma']}, basis='VA_below', **args)
        if self.basis == 'VA_below' and basis == 'kF_below':
            return ALPcouplings({'ku': (self.values['cuV'] + self.values['cuA'])/2,
                                 'kU': (self.values['cuV'] - self.values['cuA'])/2,
                                 'kd': (self.values['cdV'] + self.values['cdA'])/2,
                                 'kD': (self.values['cdV'] - self.values['cdA'])/2,
                                 'ke': (self.values['ceV'] + self.values['ceA'])/2,
                                 'kE': (self.values['ceV'] - self.values['ceA'])/2,
                                 'kNu': self.values['cnu'], 'cg': self.values['cg'], 'cgamma': self.values['cgamma']}, scale=self.scale, basis='kF_below', **args)
        else:
            if basis in bases_above and self.basis in bases_below:
                raise ValueError('Cannot translate from a below-EW scale basis to an above-EW scale basis')
            elif basis in bases_below and self.basis in bases_above:
                raise ValueError('Cannot translate from an above-EW scale basis to a below-EW scale basis')
            else:
                raise ValueError(f'Unknown basis: {basis}')
        
    def _toarray(self) -> np.ndarray:
        "Converts the object into a vector of coefficientes"
        if self.basis == 'derivative_above':
            return np.hstack([np.asarray(self.values[c]).ravel() for c in ['cqL', 'cuR', 'cdR', 'clL', 'ceR', 'cg', 'cB', 'cW']])
        if self.basis == 'massbasis_above':
            return self.translate('derivative_above')._toarray()
        if self.basis == 'kF_below':
            return np.hstack([np.asarray(self.values[c]).ravel() for c in ['kD', 'kE', 'kNu', 'kd', 'ke', 'kU', 'ku', 'cg', 'cgamma']])
        if self.basis == 'VA_below':
            return self.translate('kF_below')._toarray()

    
    @classmethod
    def _fromarray(cls, array: np.ndarray, scale: float, basis: str) -> 'ALPcouplings':
        if basis == 'derivative_above':
            vals = {}
            for i, c in enumerate(['cqL', 'cuR', 'cdR', 'clL', 'ceR']):
                vals |= {c: array[9*i:9*(i+1)].reshape([3,3])}
            vals |= {'cg': float(array[45]), "cB": float(array[46]), 'cW': float(array[47])}
            return ALPcouplings(vals, scale, basis)
        if basis == 'massbasis_above':
            return cls._fromarray(array, scale, 'derivative_above').translate('massbasis_above')
        if basis == 'kF_below':
            vals = {}
            for i, c in enumerate(['kD', 'kE', 'kNu', 'kd', 'ke']):
                vals |= {c: array[9*i:9*(i+1)].reshape([3,3])}
            for i, c in enumerate(['kU', 'ku']):
                vals |= {c: array[45+4*i:45+4*(i+1)].reshape([2,2])}
            vals |= {'cg': float(array[53]), "cgamma": float(array[54])}
            return ALPcouplings(vals, scale, basis)
        if basis == 'VA_below':
            return cls._fromarray(array, scale, 'kF_below').translate('VA_below')
    
    def match_run(self, scale_out: float, basis: str) -> 'ALPcouplings':
        """Match and run the couplings to another basis and energy scale.

        Parameters
        ----------
        scale_out : float
            Energy scale where the couplings are to be evolved, in GeV.

        basis : str
            Target basis to report the ALP couplings.

        Returns
        -------
        a : ALPcouplings
            Evolved couplings.

        Raises
        ------
        ValueError
            If the final scale is greater than the initial scale
        """
        return self._match_run(scale_out, basis)
    
    @cache
    def _match_run(self, scale_out: float, basis: str) -> 'ALPcouplings':
        from . import run_high, matching, run_low
        if scale_out > self.scale:
            raise ValueError("The final scale must be smaller than the initial scale.")
        if scale_out == self.scale:
            return self.translate(basis)
        if self.scale > self.ew_scale and scale_out < self.ew_scale: # Running, then matching and then running again
            couplings_ew = self.match_run(self.ew_scale, 'massbasis_above', self.integrator, self.beta, self.ew_scale)
            couplings_below = matching.match(couplings_ew, self.match_2loops)
            return couplings_below.match_run(scale_out, basis, self.integrator, self.beta, self.ew_scale)
        if scale_out < self.ew_scale and self.scale < self.ew_scale: # Only running below the EW scale
            if self.integrator == 'scipy':
                return run_low.run_scipy(self.translate('kF_below'), scale_out).translate(basis)
            elif self.integrator == 'leadinglog':
                return run_low.run_leadinglog(self.translate('kF_below'), scale_out).translate(basis)
            elif self.integrator == 'no_rge':
                return ALPcouplings(self.values, scale_out, self.basis).translate(basis)
        if scale_out > self.ew_scale and self.scale > self.ew_scale: # Only running above the EW scale
            if self.beta == 'ytop':
                betafunc = run_high.beta_ytop
            elif self.beta == 'full':
                betafunc = run_high.beta_full
            if self.integrator == 'scipy':
                return run_high.run_scipy(self, betafunc, scale_out).translate(basis)
            elif self.integrator == 'leadinglog':
                return run_high.run_leadinglog(self, betafunc, scale_out).translate(basis)
            elif self.integrator == 'no_rge':
                return ALPcouplings(self.values, scale_out, self.basis).translate(basis)


