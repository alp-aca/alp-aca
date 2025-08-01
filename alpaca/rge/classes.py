"""Classes for RG evolution of the ALP couplings"""

import numpy as np
from .runSM import runSM
from ..biblio.biblio import citations

from . import bases_above, bases_below
from functools import cache
from json import JSONEncoder, JSONDecoder
from sympy import Expr, Matrix
import sympy as sp
from os import PathLike
from io import TextIOBase
import wilson
from ..common import svd, diagonalise_yukawas

numeric = (int, float, complex, Expr)
matricial = (np.ndarray, np.matrix, Matrix, list)

def format_number(x):
    if isinstance(x, sp.Expr):
        return str(x)
    if isinstance(x, complex):
        if x.imag == 0:
            return format_number(x.real)
        if x.real == 0:
            return format_number(x.imag) + r'\,i'
        if x.imag < 0:
            return rf"{format_number(x.real)} - {format_number(-x.imag)}\,i"
        return rf"{format_number(x.real)} + {format_number(x.imag)}\,i"
    else:
        f = f"{x:.2e}"
        val = f[:-4]
        exp = f[-3:]
        times = r'\times'
        if exp == '+00':
            latex_exp = ''
            times = ''
        elif exp == '+01':
            latex_exp = '10'
        else:
            latex_exp = f'10^{{{int(exp)}}}'
        if val == '1.00' and latex_exp != '':
            val = ''
            times = ''
        return f"{val}{times}{latex_exp}"
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

        - 'derivative_above':
            Basis with the explicitly shift-symmetric couplings of the fermion currents to the derivative of the ALP; above the EW scale.
        - 'massbasis_ew':
            Basis with the couplings of the fermion currents to the ALP in the mass basis of the fermions, at the EW scale.
        - 'RL_below':
            Basis with the couplings of the fermion currents to the ALP in the mass basis of the fermions, below the EW scale.
        - 'VA_below':
            Basis with the couplings of the fermion currents to the ALP in the mass basis of the fermions, below the EW scale, with vector and axial-vector couplings.

    ew_scale : float
        Energy scale of the electroweak symmetry breaking scale, in GeV.

    """
    def __init__(self,
                 values: dict,
                 scale:float,
                 basis:str,
                 ew_scale: float = 100.0,
                 VuL: np.ndarray| None = None,
                 VdL: np.ndarray| None = None,
                 VuR: np.ndarray| None = None,
                 VdR: np.ndarray| None = None,
                 VeL: np.ndarray| None = None,
                 VeR: np.ndarray| None = None
                ):
        """Constructor method

        Parameters
        ------------
        values : dict
            dict containing the ALP couplings.

        scale : float
            Energy scale where the couplings are defined, in GeV.

        basis : str
            Basis in which the couplings are defined. The available bases are:

            - 'derivative_above':
                Basis with the explicitly shift-symmetric couplings of the fermion currents to the derivative of the ALP; above the EW scale.
            - 'massbasis_ew':
                Basis with the couplings of the fermion currents to the ALP in the mass basis of the fermions, at the EW scale.
            - 'RL_below':
                Basis with the couplings of the fermion currents to the ALP in the mass basis of the fermions, below the EW scale.
            - 'VA_below':
                Basis with the couplings of the fermion currents to the ALP in the mass basis of the fermions, below the EW scale, with vector and axial-vector couplings.

        ew_scale : float, optional
            Energy scale of the electroweak symmetry breaking scale, in GeV. Defaults to 100 GeV

        VuL : np.ndarray, optional
            Unitary rotation of the left-handed up-type quarks to diagonalize Yu. If None, it is set to the identity. (Only used in 'derivative_above' and 'massbasis_ew' bases)

        VdL : np.ndarray, optional
            Unitary rotation of the left-handed down-type quarks to diagonalize Yd. If None, it is set to the CKM matrix. (Only used in 'derivative_above' and 'massbasis_ew' bases)

        VuR : np.ndarray, optional
            Unitary rotation of the right-handed up-type quarks to diagonalize Yu. If None, it is set to the identity. (Only used in 'derivative_above' and 'massbasis_ew' bases)

        VdR : np.ndarray, optional
            Unitary rotation of the right-handed down-type quarks to diagonalize Yd. If None, it is set to the identity. (Only used in 'derivative_above' and 'massbasis_ew' bases)

        VeL : np.ndarray, optional
            Unitary rotation of the left-handed leptons to diagonalize Ye. If None, it is set to the identity. (Only used in 'derivative_above' and 'massbasis_ew' bases)

        VeR : np.ndarray, optional
            Unitary rotation of the right-handed leptons to diagonalize Ye. If None, it is set to the identity. (Only used in 'derivative_above' and 'massbasis_ew' bases)

        Raises
        ------
        ValueError
            If attempting to translate to an unrecognized basis.

        TypeError
            If attempting to assign a non-numeric value

        AttributeError
            If the matrices VuL and VdL are provided at the same time.
        """
        citations.register_inspire('Bauer:2020jbp')
        self.ew_scale = ew_scale
        if basis == 'derivative_above' or basis == 'sp_derivative_above':
            self.scale = scale
            self.basis = basis
            unknown_keys = set(values.keys()) - {'cG', 'cW', 'cB', 'cqL', 'cuR', 'cdR', 'clL', 'ceR'}
            if unknown_keys:
                raise KeyError(f'Unknown ALP couplings {unknown_keys} in basis {basis}')
            values = {'cG':0, 'cB': 0, 'cW':0, 'cqL': 0, 'cuR':0, 'cdR':0, 'clL':0, 'ceR':0} | values
            for c in ['cqL', 'cuR', 'cdR', 'clL', 'ceR']:
                if isinstance(values[c], numeric):
                    values[c] = np.matrix(values[c]*np.eye(3))
                elif isinstance(values[c], matricial):
                    values[c] = np.matrix(values[c]).reshape([3,3])
                else:
                    raise TypeError(f'The coupling {c} must be of a numeric or a matrix type, {type(values[c])} given')
            for c in ['cG', 'cW', 'cB']:
                if not isinstance(values[c], numeric):
                     raise TypeError(f'The coupling {c} must be of a numeric type, {type(values[c])} given')
            self.values = {c: values[c] for c in ['cG', 'cB', 'cW', 'cqL', 'cuR', 'cdR', 'clL', 'ceR']}
        elif basis == 'massbasis_ew' or basis == 'sp_massbasis_ew':
            if scale != ew_scale:
                raise ValueError(f'The scale must be equal to the electroweak scale ({ew_scale} GeV) for the {basis} basis')
            self.scale = scale
            self.basis = basis
            unknown_keys = set(values.keys()) - {'cG', 'cgamma', 'cgammaZ', 'cW', 'cZ', 'cG', 'cuL', 'cuR', 'cdL', 'cdR', 'ceL', 'cnuL', 'ceR'}
            if unknown_keys:
                raise KeyError(f'Unknown ALP couplings {unknown_keys} in basis {basis}')
            values = {'cG': 0, 'cgamma':0, 'cgammaZ': 0, 'cW':0, 'cZ': 0, 'cuL': 0, 'cuR':0, 'cdL':0, 'cdR':0, 'ceL':0, 'cnuL': 0, 'ceR': 0} | values
            for c in ['cuL', 'cuR', 'cdL', 'cdR', 'ceL', 'cnuL', 'ceR']:
                if isinstance(values[c], numeric):
                    values[c] = np.matrix(values[c]*np.eye(3))
                elif isinstance(values[c], matricial):
                    values[c] = np.matrix(values[c]).reshape([3,3])
                else:
                    raise TypeError(f'The coupling {c} must be of a numeric or a matrix type, {type(values[c])} given')
            for c in ['cgamma', 'cgammaZ', 'cW', 'cZ', 'cG']:
                if not isinstance(values[c], numeric):
                     raise TypeError(f'The coupling {c} must be of a numeric type, {type(values[c])} given')
            self.values = {c: values[c] for c in ['cuL', 'cuR', 'cdL', 'cdR', 'ceL', 'cnuL', 'ceR', 'cgamma', 'cgammaZ', 'cW', 'cZ', 'cG']}
        elif basis == 'RL_below' or basis == 'sp_RL_below':
            self.scale = scale
            self.basis = basis
            unknown_keys = set(values.keys()) - {'cG', 'cgamma', 'cuL', 'cdL', 'ceL', 'cnuL', 'cuR', 'cdR', 'ceR'}
            if unknown_keys:
                raise KeyError(f'Unknown ALP couplings {unknown_keys} in basis {basis}')
            values = {'cG':0, 'cgamma': 0, 'cuL': 0, 'cdL': 0, 'ceL': 0, 'cnuL': 0, 'cuR': 0, 'cdR': 0, 'ceR': 0} | values
            for c in ['cdL', 'ceL', 'cnuL', 'cdR', 'ceR']:
                if isinstance(values[c], numeric):
                    values[c] = np.matrix(values[c]*np.eye(3))
                elif isinstance(values[c], matricial):
                    values[c] = np.matrix(values[c]).reshape([3,3])
                else:
                    raise TypeError(f'The coupling {c} must be of a numeric or a matrix type, {type(values[c])} given')
            for c in ['cuL', 'cuR']:
                if isinstance(values[c], numeric):
                    values[c] = np.matrix(values[c]*np.eye(2))
                elif isinstance(values[c], matricial):
                    values[c] = np.matrix(values[c]).reshape([2,2])
                else:
                    raise TypeError(f'The coupling {c} must be of a numeric or a matrix type, {type(values[c])} given')
            for c in ['cG', 'cgamma']:
                if not isinstance(values[c], numeric):
                     raise TypeError(f'The coupling {c} must be of a numeric type, {type(values[c])} given')
            self.values = {c: values[c] for c in ['cdL', 'ceL', 'cnuL', 'cdR', 'ceR', 'cuL', 'cuR', 'cG', 'cgamma']}
        elif basis == 'VA_below' or basis == 'sp_VA_below':
            self.scale = scale
            self.basis = basis
            unknown_keys = set(values.keys()) - {'cG', 'cgamma', 'cuV', 'cuA', 'cdV', 'cdA', 'ceV', 'ceA', 'cnu'}
            if unknown_keys:
                raise KeyError(f'Unknown ALP couplings {unknown_keys} in basis {basis}')
            values = {'cG':0, 'cgamma': 0, 'cuV': 0, 'cuA': 0, 'cdV': 0, 'cdA': 0, 'ceV': 0, 'ceA': 0, 'cnu': 0} | values
            for c in ['cdV', 'cdA', 'ceV', 'ceA', 'cnu']:
                if isinstance(values[c], numeric):
                    values[c] = np.matrix(values[c]*np.eye(3))
                elif isinstance(values[c], matricial):
                    values[c] = np.matrix(values[c]).reshape([3,3])
                else:
                    raise TypeError(f'The coupling {c} must be of a numeric or a matrix type, {type(values[c])} given')
            for c in ['cuV', 'cuA']:
                if isinstance(values[c], numeric):
                    values[c] = np.matrix(values[c]*np.eye(2))
                elif isinstance(values[c], matricial):
                    values[c] = np.matrix(values[c]).reshape([2,2])
                else:
                    raise TypeError(f'The coupling {c} must be of a numeric or a matrix type, {type(values[c])} given')
            for c in ['cG', 'cgamma']:
                if not isinstance(values[c], numeric):
                     raise TypeError(f'The coupling {c} must be of a numeric type, {type(values[c])} given')
            self.values = {c: values[c] for c in ['cuV', 'cuA', 'cdV', 'cdA', 'ceV', 'ceA', 'cnu', 'cG', 'cgamma']}
        else:
            raise ValueError('Unknown basis')
        if self.basis.startswith('sp_'):
            for c in self.values.keys():
                if isinstance(self.values[c], numeric):
                    self.values[c] = sp.N(self.values[c])
                if isinstance(self.values[c], matricial):
                    self.values[c] = sp.Matrix(self.values[c])
        if self.basis in bases_above or self.basis[3:] in bases_above:
            def tuplize(V: np.ndarray | None) -> tuple[complex,...] | None:
                if V is not None:
                    return tuple(V.ravel())
                return None
            self.yu, self.yd, self.ye = _yukawa_matrices(
                scale,
                tuplize(VuL),
                tuplize(VuR),
                tuplize(VdL),
                tuplize(VdR),
                tuplize(VeL),
                tuplize(VeR)
            )

    def __add__(self, other: 'ALPcouplings') -> 'ALPcouplings':
        if self.basis == other.basis and self.ew_scale == other.ew_scale and self.scale == other.scale:
            a = ALPcouplings({k: self.values[k]+other.values[k] for k in self.values.keys()}, self.scale, self.basis, self.ew_scale)
            if 'yu' in self.__dict__.keys():
                a.yu = self.yu
                a.yd = self.yd
                a.ye = self.ye
            return a
        
    def __sub__(self, other: 'ALPcouplings') -> 'ALPcouplings':
        if self.basis == other.basis and self.ew_scale == other.ew_scale and self.scale == other.scale:
            a = ALPcouplings({k: self.values[k]-other.values[k] for k in self.values.keys()}, self.scale, self.basis, self.ew_scale)
            if 'yu' in self.__dict__.keys():
                a.yu = self.yu
                a.yd = self.yd
                a.ye = self.ye
            return a

    def __mul__(self, a: float) -> 'ALPcouplings':
            a1 = ALPcouplings({k: a*self.values[k] for k in self.values.keys()}, self.scale, self.basis, self.ew_scale)
            if 'yu' in self.__dict__.keys():
                a1.yu = self.yu 
                a1.yd = self.yd
                a1.ye = self.ye
            return a1

    def __rmul__(self, a: float) -> 'ALPcouplings':
            a1 =  ALPcouplings({k: a*self.values[k] for k in self.values.keys()}, self.scale, self.basis, self.ew_scale)
            if 'yu' in self.__dict__.keys():
                a1.yu = self.yu
                a1.yd = self.yd
                a1.ye = self.ye
            return a1
    
    def __truediv__(self, a: float) -> 'ALPcouplings':
            a1 = ALPcouplings({k: self.values[k]/a for k in self.values.keys()}, self.scale, self.basis, self.ew_scale)
            if 'yu' in self.__dict__.keys():
                a1.yu = self.yu
                a1.yd = self.yd
                a1.ye = self.ye
            return a1
    
    def __getitem__(self, name: str):
         return self.values[name]
    
    def __setitem__(self, name: str, val):
        if self.basis == 'derivative_above':
            if name in ['cG', 'cW', 'cB']:
                if isinstance(val, numeric):
                    self.values[name] = val
                else:
                    raise TypeError(f'The coupling {name} must be of a numeric type, {type(val)} given')
            elif name in ['cqL', 'cuR', 'cdR', 'clL', 'ceR']:
                if isinstance(val, numeric):
                    self.values[name] = val * np.eye(3)
                elif isinstance(val, matricial):
                    self.values[name] = np.matrix(val).reshape([3,3])
                else:
                    raise TypeError(f'The coupling {name} must be of a numeric or a matrix type, {type(val)} given')
            else:
                raise KeyError(f'Unknown ALP coupling {name} in basis {self.basis}')

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
        if self.basis.startswith('sp_') and basis.startswith('sp_'):
            separated = self.separate_expressions()
            a = {k: v.translate(basis[3:]) for k, v in separated.items()}
            return ALPcouplings.join_expressions(a)
        if self.basis == 'derivative_above' and basis == 'massbasis_ew':
            smpars = runSM(self.scale)
            s2w = smpars['s2w']
            c2w = 1-s2w
            
            d_y = diagonalise_yukawas(self.yu, self.yd, self.ye)
            UuL, mu, UuR = d_y['u']
            UdL, md, UdR = d_y['d']
            UeL, me, UeR = d_y['e']

            cgamma = self.values['cW'] + self.values['cB']
            cgammaZ = c2w * self.values['cW'] - s2w * self.values['cB']
            cZ = c2w**2 * self.values['cW'] + s2w**2 *self.values['cB']

            a = ALPcouplings({
                'cuL': np.matrix(UuL).H @ self.values['cqL'] @ UuL,
                'cuR': np.matrix(UuR).H @ self.values['cuR'] @ UuR,
                'cdL': np.matrix(UdL).H @ self.values['cqL'] @ UdL,
                'cdR': np.matrix(UdR).H @ self.values['cdR'] @ UdR,
                'ceL': np.matrix(UeL).H @ self.values['clL'] @ UeL,
                'cnuL': self.values['clL'],
                'ceR': np.matrix(UeR).H @ self.values['ceR'] @ UeR,
                'cgamma': cgamma, 'cW': self.values['cW'], 'cgammaZ': cgammaZ, 'cZ': cZ, 'cG': self.values['cG']
                }, scale=self.scale, basis='massbasis_ew', ew_scale=self.ew_scale)
            a.yu = self.yu
            a.yd = self.yd
            a.ye = self.ye
            return a
        
        if self.basis == 'RL_below' and basis == 'VA_below':
            return ALPcouplings({'cuV': self.values['cuR'] + self.values['cuL'],
                                 'cuA': self.values['cuR'] - self.values['cuL'],
                                 'cdV': self.values['cdR'] + self.values['cdL'],
                                 'cdA': self.values['cdR'] - self.values['cdL'],
                                 'ceV': self.values['ceR'] + self.values['ceL'],
                                 'ceA': self.values['ceR'] - self.values['ceL'],
                                 'cnu': self.values['cnuL'], 'cG': self.values['cG'], 'cgamma': self.values['cgamma']}, scale=self.scale, basis='VA_below', ew_scale=self.ew_scale)
        if self.basis == 'VA_below' and basis == 'RL_below':
            return ALPcouplings({'cuR': (self.values['cuV'] + self.values['cuA'])/2,
                                 'cuL': (self.values['cuV'] - self.values['cuA'])/2,
                                 'cdR': (self.values['cdV'] + self.values['cdA'])/2,
                                 'cdL': (self.values['cdV'] - self.values['cdA'])/2,
                                 'ceR': (self.values['ceV'] + self.values['ceA'])/2,
                                 'ceL': (self.values['ceV'] - self.values['ceA'])/2,
                                 'cnuL': self.values['cnu'], 'cG': self.values['cG'], 'cgamma': self.values['cgamma']}, scale=self.scale, basis='RL_below', ew_scale=self.ew_scale)
        else:
            raise ValueError(f'Unknown basis {basis}')
        
    def _toarray(self) -> np.ndarray:
        "Converts the object into a vector of coefficientes"
        if self.basis == 'derivative_above':
            return np.hstack([np.asarray(self.values[c]).ravel() for c in ['cqL', 'cuR', 'cdR', 'clL', 'ceR', 'cG', 'cB', 'cW']]+[np.asarray(self.yu).ravel()]+[np.asarray(self.yd).ravel()]+[np.asarray(self.ye).ravel()]).astype(dtype=complex)
        if self.basis == 'massbasis_ew':
            return np.hstack([np.asarray(self.values[c]).ravel() for c in ['cuL', 'cuR', 'cdL', 'cdR', 'ceL', 'cnuL', 'ceR', 'cgamma', 'cgammaZ', 'cW', 'cZ', 'cG']]+[np.asarray(self.yu).ravel()]+[np.asarray(self.yd).ravel()]+[np.asarray(self.ye).ravel()]).astype(dtype=complex)
        if self.basis == 'RL_below':
            return np.hstack([np.asarray(self.values[c]).ravel() for c in ['cdL', 'ceL', 'cnuL', 'cdR', 'ceR', 'cuL', 'cuR', 'cG', 'cgamma']]).astype(dtype=complex)

    
    @classmethod
    def _fromarray(cls, array: np.ndarray, scale: float, basis: str, ew_scale: float = 100.0) -> 'ALPcouplings':
        if basis == 'derivative_above':
            vals = {}
            for i, c in enumerate(['cqL', 'cuR', 'cdR', 'clL', 'ceR']):
                vals |= {c: array[9*i:9*(i+1)].reshape([3,3])}
            vals |= {'cG': array[45], "cB": array[46], 'cW': array[47]}
            a1 = ALPcouplings(vals, scale, basis, ew_scale)
            a1.yu = array[48:48+9].reshape([3,3])
            a1.yd = array[48+9: 48+18].reshape([3,3])
            a1.ye = array[48+18: 48+27].reshape([3,3])
            return a1
        if basis == 'massbasis_ew':
            vals = {}
            for i, c in enumerate(['cuL', 'cuR', 'cdL', 'cdR', 'ceL', 'cnuL', 'ceR']):
                vals |= {c: array[9*i:9*(i+1)].reshape([3,3])}
            for i, c in enumerate(['cgamma', 'cgammaZ', 'cW', 'cZ', 'cG']):
                vals |= {c: array[54+i]}
            a1 = ALPcouplings(vals, scale, basis, ew_scale)
            a1.yu = array[59:59+9].reshape([3,3])
            a1.yd = array[59+9:59+18].reshape([3,3])
            a1.ye = array[59+18:59+27].reshape([3,3])
        if basis == 'RL_below':
            vals = {}
            for i, c in enumerate(['cdL', 'ceL', 'cnuL', 'cdR', 'ceR']):
                vals |= {c: array[9*i:9*(i+1)].reshape([3,3])}
            for i, c in enumerate(['cuL', 'cuR']):
                vals |= {c: array[45+4*i:45+4*(i+1)].reshape([2,2])}
            vals |= {'cG': array[53], "cgamma": array[54]}
            return ALPcouplings(vals, scale, basis, ew_scale)
    
    def match_run(
            self,
            scale_out: float,
            basis: str,
            integrator: str='scipy',
            beta: str='full',
            match_tildecouplings = True,
            scipy_method: str = 'RK45',
            scipy_rtol: float = 1e-3,
            scipy_atol: float = 1e-6,
            **kwargs
            ) -> 'ALPcouplings':
        """Match and run the couplings to another basis and energy scale.

        Parameters
        ----------
        scale_out : float
            Energy scale where the couplings are to be evolved, in GeV.

        basis : str
            Target basis to translate.

        integrator : str, optional
            Method to use for the RG evolution. The available integrators are:

            - 'scipy':
                Use the scipy.integrate.odeint function.
            - 'leadinglog':
                Use the leading-log approximation.
            - 'symbolic':
                Use the leading-log approximation with symbolic expressions.
            - 'no_rge':
                Return the couplings at the final scale without running them.

        beta : str, optional
            Beta function to use for the RG evolution. The available beta functions are:

            - 'ytop':
                Use the beta function for the top Yukawa coupling.
            - 'full':
                Use the full beta function.

        match_tildecouplings : bool, optional
            Whether to implement the matching conditions with the 'tilde' version of the gauge couplings instead of the bare ones.
            The use of the 'tilde' coefficients partially captures 2-loop effects in the matching,
            and ensures RG invariance, as discussed in Bauer et al. (2020) arXiv:2012.12272.
            Defaults to True.

        scipy_method : str, optional
            Method to use for the scipy integrator. Defaults to 'RK45'. Other available options are 'RK23', 'DOP853', and 'BDF'. See the documentation of scipy.integrate.solve_ivp for more information.

        scipy_rtol : float, optional
            Relative tolerance for the scipy integrator. Defaults to 1e-3.

        scipy_atol : float, optional
            Absolute tolerance for the scipy integrator. Defaults to 1e-6.

        Returns
        -------
        a : ALPcouplings
            Evolved couplings.

        Raises
        ------
        KeyError
            If attempting to translate to an unrecognized basis.
        """
        return self._match_run(scale_out, basis, integrator, beta, match_tildecouplings, scipy_method, scipy_rtol, scipy_atol)
    
    @cache
    def _match_run(
            self,
            scale_out: float,
            basis: str,
            integrator: str='scipy',
            beta: str='full',
            match_tildecouplings = False,
            scipy_method: str = 'RK45',
            scipy_rtol: float = 1e-3,
            scipy_atol: float = 1e-6,
            ) -> 'ALPcouplings':
        from . import run_high, matching, run_low, symbolic
        if integrator == 'symbolic':
            if scale_out == self.scale:
                if self.basis == 'derivative_above' and basis == 'massbasis_ew':
                    return symbolic.derivative2massbasis(self)
                return self.translate(basis)
            if self.basis in bases_above and basis in bases_above:
                if beta == 'ytop':
                    betafunc = symbolic.beta_ytop
                elif beta == 'full':
                    betafunc = symbolic.beta_full
                else:
                    raise KeyError(f'beta function {beta} not recognized')
                return symbolic.run_leadinglog(self.translate('derivative_above'), betafunc, scale_out).match_run(scale_out, basis, integrator)
            if self.basis in bases_above and basis in bases_below:
                couplings_ew = self.match_run(self.ew_scale, 'massbasis_ew', integrator, beta)
                couplings_below = symbolic.match(couplings_ew, match_tildecouplings)
                return couplings_below.match_run(scale_out, basis, integrator)
            if self.basis in bases_below and basis in bases_below:
                return symbolic.run_leadinglog(self.translate('RL_below'), symbolic.beta_low, scale_out).translate(basis)
            if self.basis in bases_below and basis in bases_above:
                raise ValueError(f'Attempting to run from {self.basis} below the EW scale to {basis} above the EW scale')
            raise ValueError(f'basis {basis} not recognized')
        if scale_out > self.scale:
            raise ValueError("The final scale must be smaller than the initial scale.")
        if scale_out == self.scale:
            return self.translate(basis)
        if self.basis.startswith('sp_') and basis.startswith('sp_'):
            separated = self.separate_expressions()
            a = {k: v.match_run(scale_out, basis[3:], integrator, beta, match_tildecouplings, scipy_method=scipy_method, scipy_rtol=scipy_rtol, scipy_atol=scipy_atol) for k, v in separated.items()}
            return ALPcouplings.join_expressions(a)
        if self.scale > self.ew_scale and scale_out < self.ew_scale:
            if self.basis in bases_above and basis in bases_below:
                couplings_ew = self.match_run(self.ew_scale, 'massbasis_ew', integrator, beta, scipy_method=scipy_method, scipy_rtol=scipy_rtol, scipy_atol=scipy_atol)
                couplings_below = matching.match(couplings_ew, match_tildecouplings)
                return couplings_below.match_run(scale_out, basis, integrator, beta, scipy_method=scipy_method, scipy_rtol=scipy_rtol, scipy_atol=scipy_atol)
            else:
                raise KeyError(basis)
        if self.scale == self.ew_scale and self.basis in bases_above and basis in bases_below:
                couplings_below = matching.match(self, match_tildecouplings)
                return couplings_below.match_run(scale_out, basis, integrator, beta, scipy_method=scipy_method, scipy_rtol=scipy_rtol, scipy_atol=scipy_atol)
        if scale_out < self.ew_scale:
            if integrator == 'scipy':
                scipy_options = {'method': scipy_method, 'rtol': scipy_rtol, 'atol': scipy_atol}
                return run_low.run_scipy(self.translate('RL_below'), scale_out, scipy_options).translate(basis)
            elif integrator == 'leadinglog':
                return run_low.run_leadinglog(self.translate('RL_below'), scale_out).translate(basis)
            elif integrator == 'no_rge':
                return ALPcouplings(self.values, scale_out, self.basis).translate(basis)
            else:
                raise KeyError(f'Integrator {integrator} not recognized')
        if basis in bases_above and self.basis in bases_above:
            if beta == 'ytop':
                betafunc = run_high.beta_ytop
            elif beta == 'full':
                betafunc = run_high.beta_full
            else:
                raise KeyError(f'Option for beta function {beta} not recognized')
            if integrator == 'scipy':
                scipy_options = {'method': scipy_method, 'rtol': scipy_rtol, 'atol': scipy_atol}
                return run_high.run_scipy(self, betafunc, scale_out, scipy_options).translate(basis)
            elif integrator == 'leadinglog':
                return run_high.run_leadinglog(self, betafunc, scale_out).translate(basis)
            elif integrator == 'no_rge':
                return ALPcouplings(self.values, scale_out, self.basis).translate(basis)
            else:
                raise KeyError(f'Integrator {integrator} not recognized')
        else:
            raise KeyError(f'Unknown basis {basis} the ALP couplings')

    def to_dict(self) -> dict:
        """Convert the object into a dictionary.

        Returns
        -------
        a : dict
            Dictionary representation of the object.
        """
        def flatten(x):
            if isinstance(x, np.ndarray):
                return x.tolist()
            return x
        values = {f'{k}_Re': np.real(v) for k, v in self.values.items()} | {f'{k}_Im': np.imag(v) for k, v in self.values.items()}
        d = {'values': {k: flatten(v) for k, v in values.items()}, 'scale': self.scale, 'basis': self.basis, 'ew_scale': self.ew_scale}
        if self.basis in bases_above:
            yukawas = {f'{k}_Re': flatten(np.real(v)) for k, v in {'yu': self.yu, 'yd': self.yd, 'ye': self.ye}.items()} | {f'{k}_Im': flatten(np.imag(v)) for k, v in {'yu': self.yu, 'yd': self.yd, 'ye': self.ye}.items()}
            d |= {'yukawas': yukawas}
        return d
    
    @classmethod
    def from_dict(cls, data: dict) -> 'ALPcouplings':
        """Create an object from a dictionary.

        Parameters
        ----------
        data : dict
            Dictionary representation of the object.

        Returns
        -------
        a : ALPcouplings
            Object created from the dictionary.
        """
        def unflatten(x):
            if isinstance(x, list):
                return np.array(x)
            return x
        values = {k[:-3]: unflatten(np.array(data['values'][k]) + 1j*np.array(data['values'][k[:-3]+'_Im'])) for k in data['values'] if k[-3:] == '_Re'}
        a = ALPcouplings(values, data['scale'], data['basis'], data.get('ew_scale', 100.0))
        if 'yukawas' in data.keys():
            a.yu = unflatten(np.array(data['yukawas']['yu_Re']) + 1j*np.array(data['yukawas']['yu_Im']))
            a.yd = unflatten(np.array(data['yukawas']['yd_Re']) + 1j*np.array(data['yukawas']['yd_Im']))
            a.ye = unflatten(np.array(data['yukawas']['ye_Re']) + 1j*np.array(data['yukawas']['ye_Im']))
        return a
    
    def save(self, file: str | PathLike | TextIOBase) -> None:
        """Save the object to a JSON file.

        Parameters
        ----------
        file : str | PathLike | TextIOBase
            Name of the file, or object, where the object will be saved.
        """
        if isinstance(file, TextIOBase):
            file.write(ALPcouplingsEncoder().encode(self))
        else:
            with open(file, 'wt') as f:
                f.write(ALPcouplingsEncoder().encode(self))

    @classmethod
    def load(cls, file: str | PathLike | TextIOBase) -> 'ALPcouplings':
        """Load the object from a JSON file.

        Parameters
        ----------
        file : str | PathLike | TextIOBase
            Name of the file, or object, where the object is saved.

        Returns
        -------
        a : ALPcouplings
            Object loaded from the file.
        """
        if isinstance(file, TextIOBase):
            return ALPcouplingsDecoder().decode(file.read())
        with open(file, 'rt') as f:
            return ALPcouplingsDecoder().decode(f.read())
        
    def get_ckm(self):
        if self.scale < self.ew_scale:
            raise ValueError("The running of the CKM matrix is only computed above the EW scale")
        wSM = wilson.classes.SMEFT(wilson.wcxf.WC('SMEFT', 'Warsaw', self.scale, {})).C_in
        d_y = diagonalise_yukawas(self.yu, self.yd, self.ye)
        UuL, mu, UuR = d_y['u']
        UdL, md, UdR = d_y['d']
        return UuL.conj().T @ UdL
    
    def get_mup(self):
        if self.scale < self.ew_scale:
            raise ValueError("The running of the quark masses is only computed above the EW scale")
        wSM = wilson.classes.SMEFT(wilson.wcxf.WC('SMEFT', 'Warsaw', self.scale, {})).C_in
        UuL, mu, UuR = svd(self.yu)
        return mu * np.sqrt(wSM['m2']/wSM['Lambda'])
    
    def get_mdown(self):
        if self.scale < self.ew_scale:
            raise ValueError("The running of the quark masses is only computed above the EW scale")
        wSM = wilson.classes.SMEFT(wilson.wcxf.WC('SMEFT', 'Warsaw', self.scale, {})).C_in
        UdL, md, UdR = svd(self.yd)
        return md * np.sqrt(wSM['m2']/wSM['Lambda'])
    
    def get_mlept(self):
        if self.scale < self.ew_scale:
            raise ValueError("The running of the quark masses is only computed above the EW scale")
        wSM = wilson.classes.SMEFT(wilson.wcxf.WC('SMEFT', 'Warsaw', self.scale, {})).C_in
        UeL, me, UeR = svd(self.ye)
        return me * np.sqrt(wSM['m2']/wSM['Lambda'])
    
    def _ipython_key_completions_(self):
        return self.values.keys()
    
    def _repr_markdown_(self):
        if self.basis == 'VA_below' or self.basis == 'sp_VA_below':
            latex_couplings = {
                'cG': r'c_G',
                'cgamma': r'c_\gamma',
                'cuA': r'c_u^A',
                'cdA': r'c_d^A',
                'ceA': r'c_e^A',
                'cuV': r'c_u^V',
                'cdV': r'c_d^V',
                'ceV': r'c_e^V',
                'cnu': r'c_\nu',
            }
        elif self.basis == 'derivative_above' or self.basis == 'sp_derivative_above':
            latex_couplings = {
                'cG': r'c_G',
                'cW': r'c_W',
                'cB': r'c_B',
                'cuR': r'c_{u_R}',
                'cdR': r'c_{d_R}',
                'clL': r'c_{\ell_L}',
                'ceR': r'c_{e_R}',
                'cqL': r'c_{q_L}',
            }
        elif self.basis == 'massbasis_ew' or self.basis == 'sp_massbasis_ew':
            latex_couplings = {
                'cG': r'c_G',
                'cW': r'c_W',
                'cgamma': r'c_\gamma',
                'cZ': r'c_Z',
                'cgammaZ': r'c_{\gamma Z}',
                'cuL': r"c'_{u_L}",
                'cdL': r"c'_{d_L}",
                'ceL': r"c'_{e_L}",
                'cnuL': r"c'_{\nu_L}",
                'cuR': r"c'_{u_R}",
                'cdR': r"c'_{d_R}",
                'ceR': r"c'_{e_R}",
            }
        elif self.basis == 'RL_below' or self.basis == 'sp_RL_below':
            latex_couplings = {
                'cG': r'c_G',
                'cgamma': r'c_\gamma',
                'cuL': r"c_{u}^L",
                'cdL': r"c_{d}^L",
                'ceL': r"c_{e}^L",
                'cnuL': r"c_{\nu}",
                'cuR': r"c_{u}^R",
                'cdR': r"c_{d}^R",
                'ceR': r"c_{e}^R",
            }

        md = f"### ALP couplings\n"
        md += f"- Scale: ${format_number(self.scale)}$ GeV\n"
        md += f"- Basis: ```{self.basis}```\n"
        md += f"- EW scale: ${format_number(self.ew_scale)}$ GeV\n"
        md += f"<details><summary>Couplings:</summary>\n\n"
        for k, v in self.values.items():
            if isinstance(v, matricial):
                md += f"- ${latex_couplings[k]} = \\begin{{pmatrix}}"
                for i in range(v.shape[0]):
                    for j in range(v.shape[1]):
                        md += f"{format_number(v[i,j])} & "
                    md = md[:-2] + r"\\"
                md = md[:-2] + r"\end{pmatrix}$" + "\n"
            else:
                md += f"- ${latex_couplings[k]} = {format_number(v)}$\n"
        md += "</details>\n"

        if self.basis in bases_above:
            md += f"<details><summary>Yukawa matrices:</summary>\n\n"
            md += f"- $Y_u = \\begin{{pmatrix}}"
            for i in range(3):
                for j in range(3):
                    md += f"{format_number(self.yu[i,j])} & "
                md = md[:-2] + r"\\"
            md = md[:-2] + r"\end{pmatrix}$" + "\n"
            md += f"- $Y_d = \\begin{{pmatrix}}"
            for i in range(3):
                for j in range(3):
                    md += f"{format_number(self.yd[i,j])} & "
                md = md[:-2] + r"\\"
            md = md[:-2] + r"\end{pmatrix}$" + "\n"
            md += f"- $Y_e = \\begin{{pmatrix}}"
            for i in range(3):
                for j in range(3):
                    md += f"{format_number(self.ye[i,j])} & "
                md = md[:-2] + r"\\"
            md = md[:-2] + r"\end{pmatrix}$" + "\n"
            md += "</details>\n"
        return md
    
    def separate_expressions(self) -> dict[sp.Expr, 'ALPcouplings']:
        if not self.basis.startswith('sp_'):
            raise ValueError("This method is only available for sympy bases.")
        splitted = {}
        for k, v in self.values.items():
            if isinstance(v, sp.Expr):
                coeffs = v.factor().as_coefficients_dict()
                for c, val in coeffs.items():
                    if val == 0:
                        continue
                    if c not in splitted:
                        splitted[c] = ALPcouplings({k: float(val)}, self.scale, self.basis[3:], self.ew_scale)
                    else:
                        splitted[c] += ALPcouplings({k: float(val)}, self.scale, self.basis[3:], self.ew_scale)
            if isinstance(v, sp.Matrix):
                for i in range(v.shape[0]):
                    for j in range(v.shape[1]):
                        coeffs = v[i,j].factor().as_coefficients_dict()
                        for c, val in coeffs.items():
                            if val == 0:
                                continue
                            matr = np.zeros(v.shape, dtype=complex)
                            matr[i,j] = complex(val)
                            if c not in splitted:
                                splitted[c] = ALPcouplings({k: matr}, self.scale, self.basis[3:], self.ew_scale)
                            else:
                                splitted[c] += ALPcouplings({k: matr}, self.scale, self.basis[3:], self.ew_scale)
        if self.scale > self.ew_scale:
            for v in splitted.values():
                v.yu = self.yu
                v.yd = self.yd
                v.ye = self.ye
        return splitted
    
    @classmethod
    def join_expressions(cls, dd: dict[sp.Expr, 'ALPcouplings']) -> 'ALPcouplings':
        """Join a dictionary of ALPcouplings objects with sympy expressions as keys into a single ALPcouplings object."""
        if not all(isinstance(k, sp.Expr) for k in dd.keys()):
            raise ValueError("All keys must be sympy expressions.")
        if not all(isinstance(v, ALPcouplings) for v in dd.values()):
            raise ValueError("All values must be ALPcouplings objects.")
        v0 = next(iter(dd.values()))
        if not all(v.scale == v0.scale for v in dd.values()):
            raise ValueError("All ALPcouplings objects must have the same scale.")
        scale = v0.scale
        if not all(v.basis == v0.basis for v in dd.values()):
            raise ValueError("All ALPcouplings objects must have the same basis.")
        basis = 'sp_' + v0.basis
        if not all(v.ew_scale == v0.ew_scale for v in dd.values()):
            raise ValueError("All ALPcouplings objects must have the same EW scale.")
        ew_scale = v0.ew_scale
        a = ALPcouplings({}, scale, basis, ew_scale)
        for k, v in dd.items():
            v.basis = basis
            a += k*v
        if scale > ew_scale:
            a.yu = v0.yu
            a.yd = v0.yd
            a.ye = v0.ye
        return a
    
    def subs(self, subs_dict: dict[sp.Expr, complex]) -> 'ALPcouplings':
        num_values = {}
        for k, v in self.values.items():
            if isinstance(v, sp.Expr):
                num_values[k] = float(v.subs(subs_dict))
            elif isinstance(v, sp.Matrix):
                matr = np.zeros(v.shape, dtype=complex)
                for i in range(v.shape[0]):
                    for j in range(v.shape[1]):
                        matr[i,j] = complex(v[i,j].subs(subs_dict))
                num_values[k] = matr
        a = ALPcouplings(num_values, self.scale, self.basis[3:], self.ew_scale)
        if self.scale > self.ew_scale:
            a.yu = self.yu
            a.yd = self.yd
            a.ye = self.ye
        return a

class ALPcouplingsEncoder(JSONEncoder):
    """ JSON encoder for ALPcouplings objects and structures containing them.
     
    Usage
    -----
    >>> import json
    >>> from alpaca import ALPcouplings, ALPcouplingsEncoder

    >>> a = ALPcouplings({'cG': 1.0}, 1e3, 'derivative_above')
    >>> with open('file.json', 'wt') as f:
    ...     json.dump(a, f, cls=ALPcouplingsEncoder)
     """
    def default(self, o):
        if isinstance(o, ALPcouplings):
            return {'__class__': 'ALPcouplings'} | o.to_dict()
        return super().default(o)
    
class ALPcouplingsDecoder(JSONDecoder):
    """ JSON decoder for ALPcouplings objects and structures containing them.

    Usage
    -----
    >>> import json
    >>> from alpaca import ALPcouplingsDecoder

    >>> with open('file.json', 'rt') as f:
    ...     a = json.load(f, cls=ALPcouplingsDecoder)
    """
    def __init__(self, *args, **kwargs):
        super().__init__(object_hook=self.object_hook, *args, **kwargs)
    
    @staticmethod
    def object_hook(o):
        if o.get('__class__') == 'ALPcouplings':
            return ALPcouplings.from_dict(o)
        return o
    
@cache
def _yukawa_matrices(
    scale: float,
    VuL: tuple[complex,...],
    VuR: tuple[complex,...],
    VdL: tuple[complex,...],
    VdR: tuple[complex,...],
    VeL: tuple[complex,...],
    VeR: tuple[complex,...],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    wSM = wilson.classes.SMEFT(wilson.wcxf.WC('SMEFT', 'Warsaw', scale, {})).C_in
    d_y = diagonalise_yukawas(wSM['Gu'], wSM['Gd'], wSM['Ge'])
    UuL, mu, UuR = d_y['u']
    UdL, md, UdR = d_y['d']
    UeL, me, UeR = d_y['e']
    Vckm = UuL.conj().T @ UdL
    if VdL is not None:
        VuL_arr = np.array(VdL).reshape(3, 3) @ Vckm
        VdL_arr = np.eye(3, dtype=complex)
    elif VuL is not None:
        VdL_arr = np.array(VuL).reshape(3, 3) @ np.matrix(Vckm).H
        VuL_arr = np.eye(3, dtype=complex)
    else:
        VuL_arr = np.eye(3, dtype=complex)
        VdL_arr = Vckm
    if VdR is None:
        VdR_arr = np.eye(3, dtype=complex)
    else:
        VdR_arr = np.array(VdR).reshape(3, 3)
    if VuR is None:
        VuR_arr = np.eye(3, dtype=complex)
    else:
        VuR_arr = np.array(VuR).reshape(3, 3)
    if VeL is None:
        VeL_arr = np.eye(3, dtype=complex)
    else:
        VeL_arr = np.array(VeL).reshape(3, 3)
    if VeR is None:
        VeR_arr = np.eye(3, dtype=complex)
    else:
        VeR_arr = np.array(VeR).reshape(3, 3)
    yu = VuL_arr @ np.diag(mu) @ np.matrix(VuR_arr).H
    yd = VdL_arr @ np.diag(md) @ np.matrix(VdR_arr).H
    ye = VeL_arr @ np.diag(me) @ np.matrix(VeR_arr).H

    return yu, yd, ye
