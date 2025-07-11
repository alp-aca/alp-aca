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
import ckmutil
from cmath import phase

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

        * 'derivative_above':
            Basis with the explicitly shift-symmetric couplings of the fermion currents to the derivative of the ALP; above the EW scale.

    Raises
    ------
    ValueError
        If attempting to translate to an unrecognized basis.
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

        VuL : np.ndarray, optional
            Unitary rotation of the left-handed up-type quarks to diagonalize Yu. If None, it is set to the identity.

        VdL : np.ndarray, optional
            Unitary rotation of the left-handed down-type quarks to diagonalize Yd. If None, it is set to the CKM matrix.

        VuR : np.ndarray, optional
            Unitary rotation of the right-handed up-type quarks to diagonalize Yu. If None, it is set to the identity.

        VdR : np.ndarray, optional
            Unitary rotation of the right-handed down-type quarks to diagonalize Yd. If None, it is set to the identity.

        VeL : np.ndarray, optional
            Unitary rotation of the left-handed leptons to diagonalize Ye. If None, it is set to the identity.

        VeR : np.ndarray, optional
            Unitary rotation of the right-handed leptons to diagonalize Ye. If None, it is set to the identity.
            
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
            unknown_keys = set(values.keys()) - {'cg', 'cW', 'cB', 'cqL', 'cuR', 'cdR', 'clL', 'ceR'}
            if unknown_keys:
                raise KeyError(f'Unknown ALP couplings {unknown_keys} in basis {basis}')
            values = {'cg':0, 'cB': 0, 'cW':0, 'cqL': 0, 'cuR':0, 'cdR':0, 'clL':0, 'ceR':0} | values
            for c in ['cqL', 'cuR', 'cdR', 'clL', 'ceR']:
                if isinstance(values[c], numeric):
                    values[c] = np.matrix(values[c]*np.eye(3))
                elif isinstance(values[c], matricial):
                    values[c] = np.matrix(values[c]).reshape([3,3])
                else:
                    raise TypeError
            for c in ['cg', 'cW', 'cB']:
                if not isinstance(values[c], numeric):
                     raise TypeError
            self.values = {c: values[c] for c in ['cg', 'cB', 'cW', 'cqL', 'cuR', 'cdR', 'clL', 'ceR']}
        elif basis == 'massbasis_above' or basis == 'sp_massbasis_above':
            self.scale = scale
            self.basis = basis
            unknown_keys = set(values.keys()) - {'cg', 'cgamma', 'cgammaZ', 'cW', 'cZ', 'cg', 'kU', 'ku', 'kD', 'kd', 'kE', 'kNu', 'ke'}
            if unknown_keys:
                raise KeyError(f'Unknown ALP couplings {unknown_keys} in basis {basis}')
            values = {'cg': 0, 'cgamma':0, 'cgammaZ': 0, 'cW':0, 'cZ': 0, 'kU': 0, 'ku':0, 'kD':0, 'kd':0, 'kE':0, 'kNu': 0, 'ke': 0} | values
            for c in ['kU', 'ku', 'kD', 'kd', 'kE', 'kNu', 'ke']:
                if isinstance(values[c], numeric):
                    values[c] = np.matrix(values[c]*np.eye(3))
                elif isinstance(values[c], matricial):
                    values[c] = np.matrix(values[c]).reshape([3,3])
                else:
                    raise TypeError
            for c in ['cgamma', 'cgammaZ', 'cW', 'cZ', 'cg']:
                if not isinstance(values[c], numeric):
                     raise TypeError
            self.values = {c: values[c] for c in ['kU', 'ku', 'kD', 'kd', 'kE', 'kNu', 'ke', 'cgamma', 'cgammaZ', 'cW', 'cZ', 'cg']}
        elif basis == 'kF_below' or basis == 'sp_kF_below':
            self.scale = scale
            self.basis = basis
            unknown_keys = set(values.keys()) - {'cg', 'cgamma', 'kU', 'kD', 'kE', 'kNu', 'ku', 'kd', 'ke'}
            if unknown_keys:
                raise KeyError(f'Unknown ALP couplings {unknown_keys} in basis {basis}')
            values = {'cg':0, 'cgamma': 0, 'kU': 0, 'kD': 0, 'kE': 0, 'kNu': 0, 'ku': 0, 'kd': 0, 'ke': 0} | values
            for c in ['kD', 'kE', 'kNu', 'kd', 'ke']:
                if isinstance(values[c], numeric):
                    values[c] = np.matrix(values[c]*np.eye(3))
                elif isinstance(values[c], matricial):
                    values[c] = np.matrix(values[c]).reshape([3,3])
                else:
                    raise TypeError
            for c in ['kU', 'ku']:
                if isinstance(values[c], numeric):
                    values[c] = np.matrix(values[c]*np.eye(2))
                elif isinstance(values[c], matricial):
                    values[c] = np.matrix(values[c]).reshape([2,2])
                else:
                    raise TypeError
            for c in ['cg', 'cgamma']:
                if not isinstance(values[c], numeric):
                     raise TypeError
            self.values = {c: values[c] for c in ['kD', 'kE', 'kNu', 'kd', 'ke', 'kU', 'ku', 'cg', 'cgamma']}
        elif basis == 'VA_below' or basis == 'sp_VA_below':
            self.scale = scale
            self.basis = basis
            unknown_keys = set(values.keys()) - {'cg', 'cgamma', 'cuV', 'cuA', 'cdV', 'cdA', 'ceV', 'ceA', 'cnu'}
            if unknown_keys:
                raise KeyError(f'Unknown ALP couplings {unknown_keys} in basis {basis}')
            values = {'cg':0, 'cgamma': 0, 'cuV': 0, 'cuA': 0, 'cdV': 0, 'cdA': 0, 'ceV': 0, 'ceA': 0, 'cnu': 0} | values
            for c in ['cdV', 'cdA', 'ceV', 'ceA', 'cnu']:
                if isinstance(values[c], numeric):
                    values[c] = np.matrix(values[c]*np.eye(3))
                elif isinstance(values[c], matricial):
                    values[c] = np.matrix(values[c]).reshape([3,3])
                else:
                    raise TypeError
            for c in ['cuV', 'cuA']:
                if isinstance(values[c], numeric):
                    values[c] = np.matrix(values[c]*np.eye(2))
                elif isinstance(values[c], matricial):
                    values[c] = np.matrix(values[c]).reshape([2,2])
                else:
                    raise TypeError
            for c in ['cg', 'cgamma']:
                if not isinstance(values[c], numeric):
                     raise TypeError
            self.values = {c: values[c] for c in ['cuV', 'cuA', 'cdV', 'cdA', 'ceV', 'ceA', 'cnu', 'cg', 'cgamma']}
        else:
            raise ValueError('Unknown basis')
        if self.basis.startswith('sp_'):
            for c in self.values.keys():
                if isinstance(self.values[c], numeric):
                    self.values[c] = sp.N(self.values[c])
                if isinstance(self.values[c], matricial):
                    self.values[c] = sp.Matrix(self.values[c])
        if self.basis in bases_above or self.basis[3:] in bases_above:
            if VuL is not None and VdL is not None:
                raise AttributeError('It is not possible to provide VuL and VdL at the same time')
            wSM = wilson.classes.SMEFT(wilson.wcxf.WC('SMEFT', 'Warsaw', scale, {})).C_in
            UuL, mu, UuR = ckmutil.diag.msvd(wSM['Gu'])
            UdL, md, UdR = ckmutil.diag.msvd(wSM['Gd'])
            UeL, me, UeR = ckmutil.diag.msvd(wSM['Ge'])
            K = UuL.conj().T @ UdL
            Vub = abs(K[0,2])
            Vcb = abs(K[1,2])
            Vus = abs(K[0,1])
            gamma = phase(-K[0,0]*K[0,2].conj()/(K[1,0]*K[1,2].conj()))
            Vckm = ckmutil.ckm.ckm_tree(Vus, Vub, Vcb, gamma)
            if VdL is not None:
                VuL = VdL @ Vckm
            elif VuL is not None:
                VdL = VuL @ np.matrix(Vckm).H
            else:
                VuL = np.eye(3)
                VdL = Vckm
            if VdR is None:
                VdR = np.eye(3)
            if VuR is None:
                VuR = np.eye(3)
            if VeL is None:
                VeL = np.eye(3)
            if VeR is None:
                VeR = np.eye(3)
            self.yu = VuL @ np.diag(mu) @ np.matrix(VuR).H
            self.yd = VdL @ np.diag(md) @ np.matrix(VdR).H
            self.ye = VeL @ np.diag(me) @ np.matrix(VeR).H
    
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
            if name in ['cg', 'cW', 'cB']:
                if isinstance(val, numeric):
                    self.values[name] = val
                else:
                    raise TypeError
            elif name in ['cqL', 'cuR', 'cdR', 'clL', 'ceR']:
                if isinstance(val, numeric):
                    self.values[name] = val * np.eye(3)
                elif isinstance(val, matricial):
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
        if self.basis.startswith('sp_') and basis.startswith('sp_'):
            separated = self.separate_expressions()
            a = {k: v.translate(basis[3:]) for k, v in separated.items()}
            return ALPcouplings.join_expressions(a)
        if self.basis == 'derivative_above' and basis == 'massbasis_above':
            smpars = runSM(self.scale)
            s2w = smpars['s2w']
            c2w = 1-s2w
            UuL, mu, UuR = ckmutil.diag.msvd(self.yu)
            UdL, md, UdR = ckmutil.diag.msvd(self.yd)
            UeL, me, UeR = ckmutil.diag.msvd(self.ye)

            cgamma = self.values['cW'] + self.values['cB']
            cgammaZ = c2w * self.values['cW'] - s2w * self.values['cB']
            cZ = c2w**2 * self.values['cW'] + s2w**2 *self.values['cB']

            a = ALPcouplings({
                'kU': np.matrix(UuL).H @ self.values['cqL'] @ UuL,
                'ku': np.matrix(UuR).H @ self.values['cuR'] @ UuR,
                'kD': np.matrix(UdL).H @ self.values['cqL'] @ UdL,
                'kd': np.matrix(UdR).H @ self.values['cdR'] @ UdR,
                'kE': np.matrix(UeL).H @ self.values['clL'] @ UeL,
                'kNu': self.values['clL'],
                'ke': np.matrix(UeR).H @ self.values['ceR'] @ UeR,
                'cgamma': cgamma, 'cW': self.values['cW'], 'cgammaZ': cgammaZ, 'cZ': cZ, 'cg': self.values['cg']
                }, scale=self.scale, basis='massbasis_above', ew_scale=self.ew_scale)
            a.yu = self.yu
            a.yd = self.yd
            a.ye = self.ye
            return a
        
        if self.basis == 'massbasis_above' and basis == 'derivative_above':
            UuL, mu, UuR = ckmutil.diag.msvd(self.yu)
            UdL, md, UdR = ckmutil.diag.msvd(self.yd)
            UeL, me, UeR = ckmutil.diag.msvd(self.ye)
            a = ALPcouplings({'cg': self.values['cg'], 'cB': self.values['cgamma'] - self.values['cW'], 'cW': self.values['cW'],
                                 'cqL': UuL @ self.values['kU'] @ np.matrix(UuL).H/2 + UdL @ self.values['kD'] @ np.matrix(UdL).H/2,
                                 'cuR': UuR @ self.values['ku'] @ np.matrix(UuR),
                                 'cdR': UdR @ self.values['kD'] @ np.matrix(UdR),
                                 'clL': UeL @ self.values['kE'] @ np.matrix(UeL).H/2 + self.values['kNu'],
                                 'ceR': UeR @ self.values['ke'] @ np.matrix(UeR),
                                 }, scale=self.scale, basis='derivative_above', ew_scale=self.ew_scale)
            a.yu = self.yu
            a.yd = self.yd
            a.ye = self.ye
            return a
        
        if self.basis == 'kF_below' and basis == 'VA_below':
            return ALPcouplings({'cuV': self.values['ku'] + self.values['kU'],
                                 'cuA': self.values['ku'] - self.values['kU'],
                                 'cdV': self.values['kd'] + self.values['kD'],
                                 'cdA': self.values['kd'] - self.values['kD'],
                                 'ceV': self.values['ke'] + self.values['kE'],
                                 'ceA': self.values['ke'] - self.values['kE'],
                                 'cnu': self.values['kNu'], 'cg': self.values['cg'], 'cgamma': self.values['cgamma']}, scale=self.scale, basis='VA_below', ew_scale=self.ew_scale)
        if self.basis == 'VA_below' and basis == 'kF_below':
            return ALPcouplings({'ku': (self.values['cuV'] + self.values['cuA'])/2,
                                 'kU': (self.values['cuV'] - self.values['cuA'])/2,
                                 'kd': (self.values['cdV'] + self.values['cdA'])/2,
                                 'kD': (self.values['cdV'] - self.values['cdA'])/2,
                                 'ke': (self.values['ceV'] + self.values['ceA'])/2,
                                 'kE': (self.values['ceV'] - self.values['ceA'])/2,
                                 'kNu': self.values['cnu'], 'cg': self.values['cg'], 'cgamma': self.values['cgamma']}, scale=self.scale, basis='kF_below', ew_scale=self.ew_scale)
        else:
            raise ValueError('Unknown basis')
        
    def _toarray(self) -> np.ndarray:
        "Converts the object into a vector of coefficientes"
        if self.basis == 'derivative_above':
            return np.hstack([np.asarray(self.values[c]).ravel() for c in ['cqL', 'cuR', 'cdR', 'clL', 'ceR', 'cg', 'cB', 'cW']]+[np.asarray(self.yu).ravel()]+[np.asarray(self.yd).ravel()]+[np.asarray(self.ye).ravel()]).astype(dtype=complex)
        if self.basis == 'massbasis_above':
            return np.hstack([np.asarray(self.values[c]).ravel() for c in ['kU', 'ku', 'kD', 'kd', 'kE', 'kNu', 'ke', 'cgamma', 'cgammaZ', 'cW', 'cZ', 'cg']]+[np.asarray(self.yu).ravel()]+[np.asarray(self.yd).ravel()]+[np.asarray(self.ye).ravel()]).astype(dtype=complex)
        if self.basis == 'kF_below':
            return np.hstack([np.asarray(self.values[c]).ravel() for c in ['kD', 'kE', 'kNu', 'kd', 'ke', 'kU', 'ku', 'cg', 'cgamma']]).astype(dtype=complex)

    
    @classmethod
    def _fromarray(cls, array: np.ndarray, scale: float, basis: str, ew_scale: float = 100.0) -> 'ALPcouplings':
        if basis == 'derivative_above':
            vals = {}
            for i, c in enumerate(['cqL', 'cuR', 'cdR', 'clL', 'ceR']):
                vals |= {c: array[9*i:9*(i+1)].reshape([3,3])}
            vals |= {'cg': array[45], "cB": array[46], 'cW': array[47]}
            a1 = ALPcouplings(vals, scale, basis, ew_scale)
            a1.yu = array[48:48+9].reshape([3,3])
            a1.yd = array[48+9: 48+18].reshape([3,3])
            a1.ye = array[48+18: 48+27].reshape([3,3])
            return a1
        if basis == 'massbasis_above':
            vals = {}
            for i, c in enumerate(['kU', 'ku', 'kD', 'kd', 'kE', 'kNu', 'ke']):
                vals |= {c: array[9*i:9*(i+1)].reshape([3,3])}
            for i, c in enumerate(['cgamma', 'cgammaZ', 'cW', 'cZ', 'cg']):
                vals |= {c: array[54+i]}
            a1 = ALPcouplings(vals, scale, basis, ew_scale)
            a1.yu = array[59:59+9].reshape([3,3])
            a1.yd = array[59+9:59+18].reshape([3,3])
            a1.ye = array[59+18:59+27].reshape([3,3])
        if basis == 'kF_below':
            vals = {}
            for i, c in enumerate(['kD', 'kE', 'kNu', 'kd', 'ke']):
                vals |= {c: array[9*i:9*(i+1)].reshape([3,3])}
            for i, c in enumerate(['kU', 'ku']):
                vals |= {c: array[45+4*i:45+4*(i+1)].reshape([2,2])}
            vals |= {'cg': array[53], "cgamma": array[54]}
            return ALPcouplings(vals, scale, basis, ew_scale)
    
    def match_run(
            self,
            scale_out: float,
            basis: str,
            integrator: str='scipy',
            beta: str='full',
            match_2loops = False,
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

        match_2loops : bool, optional
            Whether to include 2-loop matching corrections.

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
        return self._match_run(scale_out, basis, integrator, beta, match_2loops, scipy_method, scipy_rtol, scipy_atol)
    
    @cache
    def _match_run(
            self,
            scale_out: float,
            basis: str,
            integrator: str='scipy',
            beta: str='full',
            match_2loops = False,
            scipy_method: str = 'RK45',
            scipy_rtol: float = 1e-3,
            scipy_atol: float = 1e-6,
            ) -> 'ALPcouplings':
        from . import run_high, matching, run_low, symbolic
        if integrator == 'symbolic':
            if scale_out == self.scale:
                if self.basis == 'derivative_above' and basis == 'massbasis_above':
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
                couplings_ew = self.match_run(self.ew_scale, 'massbasis_above', integrator, beta)
                couplings_below = symbolic.match(couplings_ew, match_2loops)
                return couplings_below.match_run(scale_out, basis, integrator)
            if self.basis in bases_below and basis in bases_below:
                return symbolic.run_leadinglog(self.translate('kF_below'), symbolic.beta_low, scale_out).translate(basis)
            if self.basis in bases_below and basis in bases_above:
                raise ValueError(f'Attempting to run from {self.basis} below the EW scale to {basis} above the EW scale')
            raise ValueError(f'basis {basis} not recognized')
        if scale_out > self.scale:
            raise ValueError("The final scale must be smaller than the initial scale.")
        if scale_out == self.scale:
            return self.translate(basis)
        if self.basis.startswith('sp_') and basis.startswith('sp_'):
            separated = self.separate_expressions()
            a = {k: v.match_run(scale_out, basis[3:], integrator, beta, match_2loops, scipy_method=scipy_method, scipy_rtol=scipy_rtol, scipy_atol=scipy_atol) for k, v in separated.items()}
            return ALPcouplings.join_expressions(a)
        if self.scale > self.ew_scale and scale_out < self.ew_scale:
            if self.basis in bases_above and basis in bases_below:
                couplings_ew = self.match_run(self.ew_scale, 'massbasis_above', integrator, beta, scipy_method=scipy_method, scipy_rtol=scipy_rtol, scipy_atol=scipy_atol)
                couplings_below = matching.match(couplings_ew, match_2loops)
                return couplings_below.match_run(scale_out, basis, integrator, beta, scipy_method=scipy_method, scipy_rtol=scipy_rtol, scipy_atol=scipy_atol)
            else:
                raise KeyError(basis)
        if self.scale == self.ew_scale and self.basis in bases_above and basis in bases_below:
                couplings_below = matching.match(self, match_2loops)
                return couplings_below.match_run(scale_out, basis, integrator, beta, scipy_method=scipy_method, scipy_rtol=scipy_rtol, scipy_atol=scipy_atol)
        if scale_out < self.ew_scale:
            if integrator == 'scipy':
                scipy_options = {'method': scipy_method, 'rtol': scipy_rtol, 'atol': scipy_atol}
                return run_low.run_scipy(self.translate('kF_below'), scale_out, scipy_options).translate(basis)
            elif integrator == 'leadinglog':
                return run_low.run_leadinglog(self.translate('kF_below'), scale_out).translate(basis)
            elif integrator == 'no_rge':
                return ALPcouplings(self.values, scale_out, self.basis).translate(basis)
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
                scipy_options = {'method': scipy_method, 'rtol': scipy_rtol, 'atol': scipy_atol}
                return run_high.run_scipy(self, betafunc, scale_out, scipy_options).translate(basis)
            elif integrator == 'leadinglog':
                return run_high.run_leadinglog(self, betafunc, scale_out).translate(basis)
            elif integrator == 'no_rge':
                return ALPcouplings(self.values, scale_out, self.basis).translate(basis)
            else:
                raise KeyError(integrator)
        else:
            raise KeyError(basis)

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
        UuL, mu, UuR = ckmutil.diag.msvd(self.yu)
        UdL, md, UdR = ckmutil.diag.msvd(self.yd)
        K = UuL.conj().T @ UdL
        Vub = abs(K[0,2])
        Vcb = abs(K[1,2])
        Vus = abs(K[0,1])
        gamma = phase(-K[0,0]*K[0,2].conj()/(K[1,0]*K[1,2].conj()))
        return ckmutil.ckm.ckm_tree(Vus, Vub, Vcb, gamma, 2)
    
    def get_mup(self):
        if self.scale < self.ew_scale:
            raise ValueError("The running of the quark masses is only computed above the EW scale")
        wSM = wilson.classes.SMEFT(wilson.wcxf.WC('SMEFT', 'Warsaw', self.scale, {})).C_in
        UuL, mu, UuR = ckmutil.diag.msvd(self.yu)
        return mu * np.sqrt(wSM['m2']/wSM['Lambda'])
    
    def get_mdown(self):
        if self.scale < self.ew_scale:
            raise ValueError("The running of the quark masses is only computed above the EW scale")
        wSM = wilson.classes.SMEFT(wilson.wcxf.WC('SMEFT', 'Warsaw', self.scale, {})).C_in
        UdL, md, UdR = ckmutil.diag.msvd(self.yd)
        return md * np.sqrt(wSM['m2']/wSM['Lambda'])
    
    def get_mlept(self):
        if self.scale < self.ew_scale:
            raise ValueError("The running of the quark masses is only computed above the EW scale")
        wSM = wilson.classes.SMEFT(wilson.wcxf.WC('SMEFT', 'Warsaw', self.scale, {})).C_in
        UeL, me, UeR = ckmutil.diag.msvd(self.ye)
        return me * np.sqrt(wSM['m2']/wSM['Lambda'])
    
    def _ipython_key_completions_(self):
        return self.values.keys()
    
    def _repr_markdown_(self):
        if self.basis == 'VA_below' or self.basis == 'sp_VA_below':
            latex_couplings = {
                'cg': r'c_G',
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
                'cg': r'c_G',
                'cW': r'c_W',
                'cB': r'c_B',
                'cuR': r'c_{u_R}',
                'cdR': r'c_{d_R}',
                'clL': r'c_{\ell_L}',
                'ceR': r'c_{e_R}',
                'cqL': r'c_{q_L}',
            }
        elif self.basis == 'massbasis_above' or self.basis == 'sp_massbasis_above':
            latex_couplings = {
                'cg': r'c_G',
                'cW': r'c_W',
                'cgamma': r'c_\gamma',
                'cZ': r'c_Z',
                'cgammaZ': r'c_{\gamma Z}',
                'kU': r"c'_{u_L}",
                'kD': r"c'_{d_L}",
                'kE': r"c'_{e_L}",
                'kNu': r"c'_{\nu_L}",
                'ku': r"c'_{u_R}",
                'kd': r"c'_{d_R}",
                'ke': r"c'_{e_R}",
            }
        elif self.basis == 'kF_below' or self.basis == 'sp_kF_below':
            latex_couplings = {
                'cg': r'c_G',
                'cgamma': r'c_\gamma',
                'kU': r"c_{u}^L",
                'kD': r"c_{d}^L",
                'kE': r"c_{e}^L",
                'kNu': r"c_{\nu}",
                'ku': r"c_{u}^R",
                'kd': r"c_{d}^R",
                'ke': r"c_{e}^R",
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

    >>> a = ALPcouplings({'cg': 1.0}, 1e3, 'derivative_above')
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