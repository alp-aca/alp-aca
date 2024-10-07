"""Classes for RG evolution of the ALP couplings"""

import numpy as np
from .runSM import runSM

bases_above = ['derivative_above', 'massbasis_above']
bases_below = ['kF_below']
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
    def __init__(self, values: dict, scale:float, basis:str):
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

        Raises
        ------
        ValueError
            If attempting to translate to an unrecognized basis.

        TypeError
            If attempting to assign a non-numeric value
        """
        if basis == 'derivative_above':
            self.scale = scale
            self.basis = basis
            values = {'cg':0, 'cB': 0, 'cW':0, 'cqL': 0, 'cuR':0, 'cdR':0, 'clL':0, 'ceR':0} | values
            for c in ['cqL', 'cuR', 'cdR', 'clL', 'ceR']:
                if isinstance(values[c], (float, int)):
                    values[c] = np.matrix(values[c]*np.eye(3))
                elif isinstance(values[c], (np.ndarray, np.matrix, list)):
                    values[c] = np.matrix(values[c]).reshape([3,3])
                else:
                    raise TypeError
            for c in ['cg', 'cW', 'cB']:
                if not isinstance(values[c], (int, float)):
                     raise TypeError
            self.values = {c: values[c] for c in ['cg', 'cB', 'cW', 'cqL', 'cuR', 'cdR', 'clL', 'ceR']}
        elif basis == 'massbasis_above':
            self.scale = scale
            self.basis = basis
            values = {'cg': 0, 'cgamma':0, 'cgammaZ': 0, 'cW':0, 'cZ': 0, 'kU': 0, 'ku':0, 'kD':0, 'kd':0, 'kE':0, 'kNu': 0, 'ke': 0} | values
            for c in ['kU', 'ku', 'kD', 'kd', 'kE', 'kNu', 'ke']:
                if isinstance(values[c], (float, int)):
                    values[c] = np.matrix(values[c]*np.eye(3))
                elif isinstance(values[c], (np.ndarray, np.matrix, list)):
                    values[c] = np.matrix(values[c]).reshape([3,3])
                else:
                    raise TypeError
            for c in ['cgamma', 'cgammaZ', 'cW', 'cZ', 'cg']:
                if not isinstance(values[c], (int, float)):
                     raise TypeError
            self.values = {c: values[c] for c in ['kU', 'ku', 'kD', 'kd', 'kE', 'kNu', 'ke', 'cgamma', 'cgammaZ', 'cW', 'cZ', 'cg']}
        elif basis == 'kF_below':
            self.scale = scale
            self.basis = basis
            values = {'cg':0, 'cgamma': 0, 'kU': 0, 'kD': 0, 'kE': 0, 'kNu': 0, 'ku': 0, 'kd': 0, 'ke': 0} | values
            for c in ['kD', 'kE', 'kNu', 'kd', 'ke']:
                if isinstance(values[c], (float, int)):
                    values[c] = np.matrix(values[c]*np.eye(3))
                elif isinstance(values[c], (np.ndarray, np.matrix, list)):
                    values[c] = np.matrix(values[c]).reshape([3,3])
                else:
                    raise TypeError
            for c in ['kU', 'ku']:
                if isinstance(values[c], (float, int)):
                    values[c] = np.matrix(values[c]*np.eye(2))
                elif isinstance(values[c], (np.ndarray, np.matrix, list)):
                    values[c] = np.matrix(values[c]).reshape([2,2])
                else:
                    raise TypeError
            for c in ['cg', 'cgamma']:
                if not isinstance(values[c], (int, float)):
                     raise TypeError
            self.values = {c: values[c] for c in ['kD', 'kE', 'kNu', 'kd', 'ke', 'kU', 'ku', 'cg', 'cgamma']}
        else:
            raise ValueError('Unknown basis')
    
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
                    raise TypeError
            elif name in ['cqL', 'cuR', 'cdR', 'clL', 'ceR']:
                if isinstance(val, (float, int)):
                    self.values[name] = val * np.eye(3)
                elif isinstance(val, (np.ndarray, np.matrix, list)):
                    self.values[name] = np.matrix(val).reshape([3,3])
                else:
                    raise TypeError
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

            return ALPcouplings({'kU': self.values['cqL'], 'ku': self.values['cuR'], 'kD': Vckm.H @ self.values['cqL'] @ Vckm, 'kd': self.values['cdR'], 'kE': self.values['clL'], 'kNu': self.values['clL'], 'ke': self.values['ceR'], 'cgamma': cgamma, 'cW': self.values['cW'], 'cgammaZ': cgammaZ, 'cZ': cZ, 'cg': self.values['cg']}, scale=self.scale, basis='massbasis_above')
        else:
            raise ValueError('Unknown basis')
        
    def _toarray(self) -> np.ndarray:
        "Converts the object into a vector of coefficientes"
        if self.basis == 'derivative_above':
            return np.hstack([np.asarray(self.values[c]).ravel() for c in ['cqL', 'cuR', 'cdR', 'clL', 'ceR', 'cg', 'cB', 'cW']])
        if self.basis == 'kF_below':
            return np.hstack([np.asarray(self.values[c]).ravel() for c in ['kD', 'kE', 'kNu', 'kd', 'ke', 'kU', 'ku', 'cg', 'cgamma']])

    
    @classmethod
    def _fromarray(cls, array: np.ndarray, scale: float, basis: str) -> 'ALPcouplings':
        if basis == 'derivative_above':
            vals = {}
            for i, c in enumerate(['cqL', 'cuR', 'cdR', 'clL', 'ceR']):
                vals |= {c: array[9*i:9*(i+1)].reshape([3,3])}
            vals |= {'cg': float(array[45]), "cB": float(array[46]), 'cW': float(array[47])}
            return ALPcouplings(vals, scale, basis)
        if basis == 'kF_below':
            vals = {}
            for i, c in enumerate(['kD', 'kE', 'kNu', 'kd', 'ke']):
                vals |= {c: array[9*i:9*(i+1)].reshape([3,3])}
            for i, c in enumerate(['kU', 'ku']):
                vals |= {c: array[45+4*i:45+4*(i+1)].reshape([2,2])}
            vals |= {'cg': float(array[53]), "cgamma": float(array[54])}
            return ALPcouplings(vals, scale, basis)
        
    def match_run(self, scale_out: float, basis: str, integrator: str='scipy', beta: str='full', matching_scale: float = 100.0, match_2loops = False) -> 'ALPcouplings':
        from . import run_high, matching, run_low
        if scale_out > self.scale:
            raise ValueError("The final scale must be smaller than the initial scale.")
        if scale_out == self.scale:
            return self.translate(basis)
        if self.scale > matching_scale and scale_out < matching_scale:
            if self.basis in bases_above and basis in bases_below:
                couplings_ew = self.match_run(matching_scale, 'massbasis_above', integrator, beta)
                return matching.match(couplings_ew, match_2loops).translate(basis).match_run(scale_out, basis, integrator)
            else:
                raise KeyError(basis)
        if scale_out < matching_scale:
            if integrator == 'scipy':
                return run_low.run_scipy(self, scale_out).translate(basis)
            if integrator == 'leadinglog':
                return run_low.run_leadinglog(self, scale_out).translate(basis)
            else:
                raise KeyError(integrator)
        if basis in bases_above and self.basis in bases_above:
            if beta == 'ytop':
                betafunc = run_high.beta_ytop
            elif beta == 'full':
                betafunc = run_high.beta_full
            else:
                raise KeyError(beta)
            if integrator == 'scipy':
                return run_high.run_scipy(self, betafunc, scale_out).translate(basis)
            if integrator == 'leadinglog':
                return run_high.run_leadinglog(self, betafunc, scale_out).translate(basis)
            else:
                raise KeyError(integrator)
        else:
            raise KeyError(basis)


