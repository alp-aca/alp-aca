########################
## Library for models ##
########################
import numpy as np
from ..rge import ALPcouplings
from . import su3
from ..biblio.biblio import citations

# It is enough for this program to give the charges for the model

import sympy as sp


couplings_latex = {'cG': r'c_G', 'cB': 'c_B', 'cW': 'c_W', 'cqL': r'c_q', 'cuR': r'c_u', 'cdR': r'c_d', 'clL': r'c_\ell', 'ceR': r'c_e'}
class ModelBase:
    """
    Base class representing a UV model with couplings to ALPs.
    Specific models should inherit from this class and implement the couplings.

    Attributes
    ----------
    model_name : str
        The name of the model.
    couplings : dict[str, sp.Expr]
        A dictionary with the couplings of the model.

    Methods
    -------
    get_couplings(substitutions: dict[sp.Expr, float | complex], scale: float) -> ALPcouplings
        Returns the couplings of the model with numerical values.
    couplings_latex(nonumber: bool = False) -> str
        Returns the couplings of the model in LaTeX format.
    E_over_N() -> sp.Rational
        Returns the ratio E/N for the model.
    """
    def __init__(self, model_name: str):
        """ Intialize an empty model with the given name."""
        self.model_name = model_name
        self.couplings: dict[str, sp.Expr] = {}

    def initialize(self):
        pass
    
    def get_couplings(self,
                      substitutions: dict[sp.Expr, float | complex],
                      scale: float,
                      ew_scale: float = 100.0,
                      VuL: np.ndarray| None = None,
                      VdL: np.ndarray| None = None,
                      VuR: np.ndarray| None = None,
                      VdR: np.ndarray| None = None,
                      VeL: np.ndarray| None = None,
                      VeR: np.ndarray| None = None,
                      ) -> ALPcouplings:
        """Substitute the symbolic variables with numerical values directly into the couplings

        Arguments
        ---------
        substitutions : dict[sp.Expr, float | complex]
            A dictionary with the values to substitute in the couplings.
        scale : float
            The scale at which the couplings are evaluated, in GeV.
        ew_scale : float
            The electroweak scale, in GeV. Default is 100.0 GeV.

        VuL : np.ndarray, optional
            Unitary rotation of the left-handed up-type quarks to diagonalize Yu. If None, it is set to the identity.

        VdL : np.ndarray, optional
            Unitary rotation of the left-handed down-type quarks to diagonalize Yd. If None, it is set to the CKM matrix.

        VuR : np.ndarray, optional
            Unitary rotation of the right-handed up-type quarks to diagonalize Yu. If None, it is set to the identity.

        VdR : np.ndarray, optional
            Unitary rotation of the right-handed down-type quarks to diagonalize Yd. If None, it is set to the identity.

        Returns
        -------
        ALPcouplings
            The couplings of the model with numerical values.
        """
        self.initialize()
        substituted_couplings = {key: np.array(value.subs(substitutions), dtype=float) for key, value in self.couplings.items()}
        for k, v in substituted_couplings.items():
            if np.array(v).shape == ():
                substituted_couplings[k] = float(v)
        return ALPcouplings(substituted_couplings, scale, 'derivative_above', ew_scale, VuL, VdL, VuR, VdR, VeL, VeR)
    
    def couplings_latex(self, eqnumber: bool = False) -> str:
        """Return the couplings of the model in LaTeX format.

        The couplings are returned inside an align environment,
        one coupling per line, aligned at the = sign.

        Arguments
        ---------
        eqnumber : bool
            If True, the align environment will number the lines. Default is False.

        Returns
        -------
        str
            The couplings of the model in LaTeX format.
        """
        self.initialize()
        if eqnumber:
            nn = ''
        else:
            nn = r' \nonumber '
        eqs = []
        for ck, cv in self.couplings.items():
            if not np.any(np.array(cv)):
                continue # Skip coefficients equal to zero
            if np.array(cv).shape == ():
                eqs.append(couplings_latex[ck] + ' &= ' + sp.latex(cv) + nn)
            else:
                eqs.append(couplings_latex[ck] + ' &= ' + sp.latex(sp.Matrix(cv), mat_delim='(') + nn)
        linebreak = r'\\' + '\n'
        return r'\begin{align}' + '\n' + linebreak.join(eqs) + '\n' + r'\end{align}'
    
    def symbolic_ALPcouplings(self,
                      scale: float,
                      ew_scale: float = 100.0,
                      VuL: np.ndarray| None = None,
                      VdL: np.ndarray| None = None,
                      VuR: np.ndarray| None = None,
                      VdR: np.ndarray| None = None,
                      VeL: np.ndarray| None = None,
                      VeR: np.ndarray| None = None,
                      ) -> ALPcouplings:
        """Return the couplings of the model as a symbolic ALPcouplings object.

        Arguments
        scale : float
            The scale at which the couplings are evaluated, in GeV.
        ew_scale : float
            The electroweak scale, in GeV. Default is 100.0 GeV.

        VuL : np.ndarray, optional
            Unitary rotation of the left-handed up-type quarks to diagonalize Yu. If None, it is set to the identity.

        VdL : np.ndarray, optional
            Unitary rotation of the left-handed down-type quarks to diagonalize Yd. If None, it is set to the CKM matrix.

        VuR : np.ndarray, optional
            Unitary rotation of the right-handed up-type quarks to diagonalize Yu. If None, it is set to the identity.

        VdR : np.ndarray, optional
            Unitary rotation of the right-handed down-type quarks to diagonalize Yd. If None, it is set to the identity.
        
        Returns
        -------
        ALPcouplings
            The couplings of the model as a symbolic ALPcouplings object.
        """
        self.initialize()
        return ALPcouplings(self.couplings, scale, 'sp_derivative_above', ew_scale, VuL, VdL, VuR, VdR, VeL, VeR)
    
    def E_over_N(self) -> sp.Rational:
        """Return the ratio E/N for the model relating the electromagnetic and QCD anomalies.

        Returns
        -------
        sp.Rational
            The ratio E/N for the model.
        
        Raises
        ------
        ZeroDivisionError
            If the coupling cG is zero.
        """
        self.initialize()
        if self.couplings.get('cG', 0) == 0:
            raise ZeroDivisionError('cG = 0')
        cgamma = self.couplings['cB'] + self.couplings['cW']
        return sp.Rational(sp.simplify(cgamma/self.couplings['cG'])).limit_denominator()
    
    def model_parameters(self) -> list[sp.Expr]:
        symbols = set()
        for value in self.couplings.values():
            symbols.update(value.free_symbols)
        return list(symbols)

    def _repr_markdown_(self):
        """Return a string representation of the model in Markdown format."""
        self.initialize()
        md = f"### UV-complete model\n\n**Name:** {self.model_name}\n\n**Model class:** {str(self.__class__).split('.')[-1][:-2]}\n\n"
        if len(self.model_parameters()) > 0:
            md += "<details><summary><b>Parameters:</b></summary>\n\n"
            for p in self.model_parameters():
                md += f"- ${sp.latex(p)}$\n"
            md += "\n\n</details>"
        md += "<details><summary><b>Couplings:</b></summary>\n\n" + self.couplings_latex(eqnumber=False) + "\n\n</details>"

        return md

class PQChargedModel(ModelBase):
    """A class to define a model given the PQ charges of the SM fermions.

    """
    def __init__(self, model_name: str, charges: dict[str, sp.Expr]):
        """Initialize the model with the given name and PQ charges.

        Arguments
        ---------
        model_name : str
            The name of the model.
        charges : dict[str, sp.Expr]
            A dictionary with the PQ charges of the SM fermions. The keys are the names of the fermions in the unbroken phase: 'cqL', 'cuR', 'cdR', 'clL', 'ceR'.

        Raises
        ------
        NotImplementedError
            If nonuniversal is True.
        """
        super().__init__(model_name)
        charges = {f: 0 for f in ['lL', 'eR', 'qL', 'uR', 'dR']} | charges # initialize to zero all missing charges
        self.charges = {key: sp.sympify(value) for key, value in charges.items()}  # Convert all values to sympy objects
        
        charges_np = {key: np.broadcast_to(value, 3) for key, value in charges.items()}  # Convert all values to numpy arrays
        for f in ['qL', 'uR', 'dR', 'lL', 'eR']:
            if np.array(self.charges[f]).shape == ():
                self.couplings[f'c{f}'] = -self.charges[f]
            else:
                self.couplings[f'c{f}'] = - sp.diag(charges_np[f].tolist(), unpack=True)
        self.couplings['cG'] = -sp.Rational(1,2) * sp.simplify(np.sum(
            2 * charges_np['qL'] - charges_np['dR'] - charges_np['uR']
        ))
        self.couplings['cW'] = -sp.Rational(1,2) * sp.simplify(np.sum(
            3 * charges_np['qL'] + charges_np['lL']
        ))
        self.couplings['cB'] = -sp.Rational(1,6) * sp.simplify(np.sum(
            charges_np['qL'] - 8 * charges_np['uR'] - 2 * charges_np['dR'] + 3 * charges_np['lL'] - 6 * charges_np['eR']
        ))
    def initialize(self):
        citations.register_inspire('DiLuzio:2020wdo')

    def _repr_markdown_(self):
        md = super()._repr_markdown_()
        md += "<details><summary><b>PQ charges:</b></summary>\n\n"
        for f, c in self.charges.items():
            md += f"- $\\mathcal{{X}}{couplings_latex['c'+f][1:]} = {sp.latex(c)}$\n"
        md += "\n\n</details>"
        return md


class HeavyFermion:
    """
    A class to represent a heavy fermion with specific group representations and charges.

    Attributes:
    -----------
    color_dim : int
        The dimension of the color representation.
    weak_isospin_dim : int
        The dimension of the weak isospin representation.
    dynkin_index_color : sympy.Rational
        The Dynkin index for the color representation.
    dynkin_index_weak : sympy.Rational
        The Dynkin index for the weak isospin representation.
    hypercharge : float
        The hypercharge of the fermion.
    PQ : float
        The Peccei-Quinn charge of the fermion.
    """
    def __init__(self,
                 SU3_rep: str | int | tuple[int, int] | list[int],
                 SU2_rep: str | int,
                 Y_hyper: float,
                 PQ: float
                ):
        """Initialize the heavy fermion with given representations and charges.

        Parameters:
        -----------
        SU3_rep : str | int | tuple[int, int] | list[int]
            The representation of the SU(3) group. It can be a string, an integer, 
            a tuple of two integers, or a list of integers.
        SU2_rep : str | int
            The representation of the SU(2) group. It can be a string or an integer.
        Y_hyper : float
            The hypercharge value.
        PQ : float
            The Peccei-Quinn charge value.
        """

        j = sp.Rational(int(SU2_rep)-1, 2)
        if isinstance(SU3_rep, (list, tuple)):
            self.label_su3 = SU3_rep
        else:
            self.label_su3 = su3.dynkinlabels_from_name(SU3_rep)
        self.color_dim = su3.dim_from_dynkinlabels(*self.label_su3)
        self.weak_isospin_dim = 2*j + 1
        self.dynkin_index_color = su3.index_from_dynkinlabels(*self.label_su3)
        self.dynkin_index_weak = sp.Rational(1,3) * (j*(j+1)*(2*j+1))
        self.hypercharge = Y_hyper
        self.PQ = PQ

    def _repr_markdown_(self):
        """Return a string representation of the heavy fermion in Markdown format."""
        md = '(' + su3.latex_from_dynkinlabels(*self.label_su3) + ', $\\mathbf{' + str(self.weak_isospin_dim) + f'}}, {self.hypercharge}, {self.PQ}$)'
        return md

class KSVZ_model(ModelBase):
    """A class to define the KSVZ-like models given the new heavy fermions."""
    def __init__(self, model_name: str, fermions: list[HeavyFermion]):
        """Initialize the KSVZ-like model with the given name and heavy fermions.
        
        Arguments
        ---------
        model_name : str
            The name of the model.
        fermions : list[HeavyFermion]
            A list with the heavy fermions of the model.
        """
        super().__init__(model_name)
        self.fermions = fermions
        self.couplings['cG']=-sum(f.PQ * f.weak_isospin_dim * f.dynkin_index_color for f in fermions)
        self.couplings['cB']=-sum(f.PQ * f.color_dim * f.weak_isospin_dim * f.hypercharge**2 for f in fermions)
        self.couplings['cW']=-sum(f.PQ * f.dynkin_index_weak * f.color_dim for f in fermions)
    def initialize(self):
        citations.register_inspire('Quevillon:2019zrd')

    def _repr_markdown_(self):
        """Return a string representation of the KSVZ-like model in Markdown format."""
        md = super()._repr_markdown_()
        md += "<details><summary><b>Heavy fermions:</b></summary>\n\n"
        for f in self.fermions:
            md += f"- {f._repr_markdown_()}\n"
        md += "\n\n</details>"
        return md

eps_flaxion = sp.symbols(r'\epsilon')
vev = sp.symbols(r'v')
class Flaxion(PQChargedModel):
    """A class to define the Flaxion model given the PQ charges of the SM fermions."""
    def __init__(self, model_name, charges):
        super().__init__(model_name, charges)
    def initialize(self):
        citations.register_inspire('Ema:2016ops')
    def masses_symbolic(self, fermion: str) -> list[sp.Expr]:
        """Return the mass of the SM fermions in the model."""
        if fermion == 'u':
            exponents = self.charges['qL'] - self.charges['uR']
        elif fermion == 'd':
            exponents = self.charges['qL'] - self.charges['dR']
        elif fermion == 'e':
            exponents = self.charges['lL'] - self.charges['eR']
        return [vev/sp.sqrt(2) * eps_flaxion**exponents[i] for i in range(3)]
    def masses(self, fermion: str, eps: float) -> list[float]:
        from ..constants import vev as vev_EW
        return np.array([float(m.subs({eps_flaxion: eps, vev: vev_EW})) for m in self.masses_symbolic(fermion)], dtype=float)
    def yukawas_symbolic(self, fermion: str) -> sp.Expr:
        if fermion == 'u':
            exponentsL = np.broadcast_to(self.charges['qL'], (3,3)).T
            exponentsR = np.broadcast_to(self.charges['uR'], (3,3))
        elif fermion == 'd':
            exponentsL = np.broadcast_to(self.charges['qL'], (3,3)).T
            exponentsR = np.broadcast_to(self.charges['dR'], (3,3))
        elif fermion == 'e':
            exponentsL = np.broadcast_to(self.charges['lL'], (3,3)).T
            exponentsR = np.broadcast_to(self.charges['eR'], (3,3))
        return sp.Matrix(eps_flaxion**(exponentsL-exponentsR))
    def yukawas(self, fermion: str, eps: float, coeffs: np.ndarray | None) -> np.matrix:
        if fermion == 'u':
            chargesL = np.array(self.charges['qL'], dtype=int)
            chargesR = np.array(self.charges['uR'], dtype=int)
            seed = 42
        elif fermion == 'd':
            chargesL = np.array(self.charges['qL'], dtype=int)
            chargesR = np.array(self.charges['dR'], dtype=int)
            seed = 43
        elif fermion == 'e':
            chargesL = np.array(self.charges['lL'], dtype=int)
            chargesR = np.array(self.charges['eR'], dtype=int)
            seed = 44
        Lf = eps**(np.abs(np.broadcast_to(chargesL, (3,3)) - np.broadcast_to(chargesL, (3,3)).T))
        mf = np.diag(eps**(chargesL - chargesR))
        Rfdagger = eps**(np.abs(np.broadcast_to(chargesR, (3,3)) - np.broadcast_to(chargesR, (3,3)).T))
        rng = np.random.default_rng(seed)
        if coeffs is None:
            coeffs = np.exp(rng.lognormal(sigma=0.5, size=(3,3))) * np.exp(2*np.pi*1j *rng.uniform(low=0, high=1, size=(3,3)))
        return np.matrix(coeffs*(Lf @ mf @ Rfdagger), dtype=complex)
    def get_couplings(self, eps: float, scale: float, ew_scale = 100,
                      coeffs_yu: np.ndarray | None = None,
                      coeffs_yd: np.ndarray | None = None,
                      coeffs_ye: np.ndarray | None = None) -> ALPcouplings:
        a =  super().get_couplings({eps_flaxion: eps}, scale, ew_scale)
        a.yu = self.yukawas('u', eps, coeffs_yu)
        a.yd = self.yukawas('d', eps, coeffs_yd)
        a.ye = self.yukawas('e', eps, coeffs_ye)
        return a
    def symbolic_ALPcouplings(self, scale: float, ew_scale: float = 100, VuL: np.ndarray | None = None, VdL: np.ndarray | None = None, VuR: np.ndarray | None = None, VdR: np.ndarray | None = None, VeL: np.ndarray | None = None, VeR: np.ndarray | None = None) -> ALPcouplings:
        raise NotImplementedError("The symbolic ALP couplings for the Flaxion model are not implemented. Use get_couplings() instead to get numerical values.")
    
    def model_parameters(self) -> list[sp.Expr]:
        symbols = super().model_parameters()
        symbols += [eps_flaxion,]
        return symbols

    def _repr_markdown_(self):
        md = super()._repr_markdown_()
        md += "<details><summary><b>Yukawa matrices:</b></summary>\n\n"
        md += f"- $Y_u \\sim {sp.latex(self.yukawas_symbolic('u'))}$\n"
        md += f"- $Y_d \\sim {sp.latex(self.yukawas_symbolic('d'))}$\n"
        md += f"- $Y_e \\sim {sp.latex(self.yukawas_symbolic('e'))}$\n"
        md += "\n\n</details>"
        return md

# Benchmark Models

beta = sp.symbols('beta')
"""Symbol representing the angle beta in the DFSZ-like models."""
KSVZ_charge = sp.symbols(r'\mathcal{X}')
"""Symbol representing the PQ charge of the heavy fermions in the KSVZ-like models."""

QED_DFSZ= PQChargedModel('QED-DFSZ', {'eR': -2*sp.cos(beta)**2, 'uR': -2*sp.sin(beta)**2, 'dR': 2*sp.sin(beta)**2})
"""QED-DFSZ: A DFSZ-like model with couplings to leptons and quarks that does not generate a QCD anomaly."""
u_DFSZ= PQChargedModel('u-DFSZ', {'eR': 1, 'dR':-2,'uR': 0})
"""u-DFSZ: A DFSZ-like model where the up-type quarks are decoupled."""
e_DFSZ= PQChargedModel('e-DFSZ', {'uR': 1, 'dR': 1})
"""e-DFSZ: A DFSZ-like model where the leptons are decoupled."""
Q_KSVZ=KSVZ_model('Q-KSVZ', [HeavyFermion(3,1,0,KSVZ_charge)])
"""Q-KSVZ: A KSVZ-like model with a heavy vector-like quark."""
L_KSVZ=KSVZ_model('L-KSVZ', [HeavyFermion(1,2,0,KSVZ_charge)])
"""L-KSVZ: A KSVZ-like model with a heavy vector-like lepton."""
Y_KSVZ=KSVZ_model('Y-KSVZ', [HeavyFermion(1,1,sp.Rational(1,2),KSVZ_charge)])
"""L-KSVZ: A KSVZ-like model with a heavy vector-like lepton."""
flaxion_benchmark = Flaxion('Flaxion', {'qL': np.array([3, 2, 0], dtype=int), 'uR': np.array([-5, -1, 0], dtype=int), 'dR': np.array([-4, -3, -3], dtype=int), 'lL': np.array([1, 0, 0], dtype=int), 'eR': np.array([-8, -5, -3], dtype=int)})
"""Flaxion: A model with a flaxion field."""
nonuniversal_DFSZ = PQChargedModel('2HDM_1', {'dR': -sp.cos(beta)**2, 'uR': -sp.sin(beta)**2, 'eR': -sp.cos(beta)**2, 'qL': [0, 0, -1], 'lL': [0,0,-1]})
"""Nonuniversal DFSZ: A DFSZ-like model with nonuniversal couplings to quarks and leptons."""