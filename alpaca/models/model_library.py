########################
## Library for models ##
########################
import numpy as np
from ..rge import ALPcouplings
from . import su3

# It is enough for this program to give the charges for the model

import sympy as sp


couplings_latex = {'cg': r'c_g', 'cB': 'c_B', 'cW': 'c_W', 'cqL': r'c_{q_L}', 'cuR': r'c_{u_R}', 'cdR': r'c_{d_R}', 'clL': r'c_{\ell_L}', 'ceR': r'c_{e_R}'}
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
    
    def get_couplings(self,
                      substitutions: dict[sp.Expr, float | complex],
                      scale: float,
                      ew_scale: float = 100.0
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

        Returns
        -------
        ALPcouplings
            The couplings of the model with numerical values.
        """
        substituted_couplings = {key: float(value.subs(substitutions)) for key, value in self.couplings.items()}
        return ALPcouplings(substituted_couplings, scale, 'derivative_above', ew_scale)
    
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
    
    def E_over_N(self) -> sp.Rational:
        """Return the ratio E/N for the model relating the electromagnetic and QCD anomalies.

        Returns
        -------
        sp.Rational
            The ratio E/N for the model.
        
        Raises
        ------
        ZeroDivisionError
            If the coupling cg is zero.
        """
        if self.couplings.get('cg', 0) == 0:
            raise ZeroDivisionError('cg = 0')
        cgamma = self.couplings['cB'] + self.couplings['cW']
        return sp.Rational(sp.simplify(cgamma/self.couplings['cg'])).limit_denominator()

class model(ModelBase):
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
                self.couplings[f'c{f}'] = - np.diag(charges_np[f])
        self.couplings['cg'] = sp.Rational(1,2) * sp.simplify(np.sum(
            2 * charges_np['qL'] - charges_np['dR'] - charges_np['uR']
        ))
        self.couplings['cW'] = sp.Rational(1,2) * sp.simplify(np.sum(
            3 * charges_np['qL'] + charges_np['lL']
        ))
        self.couplings['cB'] = sp.Rational(1,6) * sp.simplify(np.sum(
            charges_np['qL'] - 8 * charges_np['uR'] - 2 * charges_np['dR'] + 3 * charges_np['lL'] - 6 * charges_np['eR']
        ))


class fermion:
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
            label_su3 = SU3_rep
        else:
            label_su3 = su3.dynkinlabels_from_name(SU3_rep)
        self.color_dim = su3.dim_from_dynkinlabels(*label_su3)
        self.weak_isospin_dim = 2*j + 1
        self.dynkin_index_color = su3.index_from_dynkinlabels(*label_su3)
        self.dynkin_index_weak = sp.Rational(1,3) * (j*(j+1)*(2*j+1))
        self.hypercharge = Y_hyper
        self.PQ = PQ

class KSVZ_model(ModelBase):
    """A class to define the KSVZ-like models given the new heavy fermions."""
    def __init__(self, model_name: str, fermions: list[fermion]):
        """Initialize the KSVZ-like model with the given name and heavy fermions.
        
        Arguments
        ---------
        model_name : str
            The name of the model.
        fermions : list[fermion]
            A list with the heavy fermions of the model.
        """
        super().__init__(model_name)
        self.couplings['cg']=sum(f.PQ * f.weak_isospin_dim * f.dynkin_index_color for f in fermions)
        self.couplings['cB']=sum(f.PQ * f.color_dim * f.weak_isospin_dim * f.hypercharge**2 for f in fermions)
        self.couplings['cW']=sum(f.PQ * f.dynkin_index_weak * f.color_dim for f in fermions)


# Benchmark Models

beta = sp.symbols('beta')
"""Symbol representing the angle beta in the DFSZ-like models."""
KSVZ_charge = sp.symbols(r'\mathcal{X}')
"""Symbol representing the PQ charge of the heavy fermions in the KSVZ-like models."""

QED_DFSZ= model('QED-DFSZ', {'lL': 2*sp.cos(beta)**2, 'uR': -2*sp.sin(beta)**2, 'dR': 2*sp.sin(beta)**2})
"""QED-DFSZ: A DFSZ-like model with couplings to leptons and quarks that does not generate a QCD anomaly."""
u_DFSZ= model('u-DFSZ', {'lL': 2*sp.cos(beta)**2, 'uR': -2*sp.sin(beta)**2})
"""u-DFSZ: A DFSZ-like model with couplings to leptons and up-type quarks."""
d_DFSZ= model('d-DFSZ', {'lL': 2*sp.cos(beta)**2, 'dR': 2*sp.sin(beta)**2})
"""d-DFSZ: A DFSZ-like model with couplings to leptons and down-type quarks."""
Q_KSVZ=KSVZ_model('Q-KSVZ', [fermion(3,1,0,KSVZ_charge)])
"""Q-KSVZ: A KSVZ-like model with a heavy vector-like quark."""
L_KSVZ=KSVZ_model('L-KSVZ', [fermion(1,2,0,KSVZ_charge)])
"""L-KSVZ: A KSVZ-like model with a heavy vector-like lepton."""
