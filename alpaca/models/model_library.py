########################
## Library for models ##
########################
import numpy as np
from ..rge import ALPcouplings
from . import su3

# It is enough for this program to give the charges for the model

import sympy as sp

# Create a diagonal matrix
def family_universal(charge):
    return np.diag(np.full(3, charge))

Qleptons = family_universal(-1)
Ququarks = family_universal(2/3)
Qdquarks = family_universal(-1/3)

couplings_latex = {'cg': r'c_g', 'cB': 'c_B', 'cW': 'c_W', 'cqL': r'c_{q_L}', 'cuR': r'c_{u_R}', 'cdR': r'c_{d_R}', 'clL': r'c_{\ell_L}', 'ceR': r'c_{e_R}'}
class ModelBase:
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.couplings = {}
    
    def get_couplings(self, substitutions: dict, scale: float) -> ALPcouplings:
        # Substitute the symbolic variables with numerical values directly into the couplings
        substituted_couplings = {key: float(value.subs(substitutions)) for key, value in self.couplings.items()}
        return ALPcouplings(substituted_couplings, scale, 'derivative_above')
    
    def couplings_latex(self, nonumber: bool = False) -> str:
        latex = r'\begin{align}' + '\n'
        if nonumber:
            nn = ''
        else:
            nn = r' \nonumber '
        for ck, cv in self.couplings.items():
            latex += couplings_latex[ck] + ' &= ' + sp.latex(cv) + nn + r'\\'  + '\n'
        return latex + r'\end{align}'
    
    def E_over_N(self) -> sp.Rational:
        if self.couplings['cg'] == 0:
            raise ZeroDivisionError('cg = 0')
        cgamma = self.couplings['cB'] + self.couplings['cW']
        return sp.Rational(sp.simplify(cgamma/self.couplings['cg'])).limit_denominator()

class model(ModelBase): 
    def __init__(self, model_name: str, charges: dict, nonuniversal: bool = False):
        super().__init__(model_name)
        charges = {f: 0 for f in ['lL', 'eR', 'qL', 'uR', 'dR']} | charges # initialize to zero all missing charges
        self.charges = {key: sp.sympify(value) for key, value in charges.items()}  # Convert all values to sympy objects
        
        if nonuniversal:
            raise NotImplementedError('function not implemented... yet...')
        else:
            for f in ['qL', 'uR', 'dR', 'lL', 'eR']:
                self.couplings[f'c{f}'] = -self.charges[f]
            self.couplings['cg'] = sp.Rational(1,2) * sp.simplify(np.trace(
                2 * family_universal(self.charges['qL']) - family_universal(self.charges['dR']) - family_universal(self.charges['uR'])
            ))
            self.couplings['cW'] = sp.Rational(1,2) * sp.simplify(np.trace(3 * family_universal(self.charges['qL']) + family_universal(self.charges['lL'])))
            self.couplings['cB'] = sp.Rational(1,6) * sp.simplify(np.trace(family_universal(self.charges['qL']) - 8 * family_universal(self.charges['uR']) - 2 * family_universal(self.charges['dR']) + 3 * family_universal(self.charges['lL']) - 6 * family_universal(self.charges['eR'])))


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
    def __init__(self, model_name: str, fermions: list[fermion]):
        super().__init__(model_name)
        self.couplings['cg']=sum(f.PQ * f.weak_isospin_dim * f.dynkin_index_color for f in fermions)
        self.couplings['cB']=sum(f.PQ * f.color_dim * f.weak_isospin_dim * f.hypercharge**2 for f in fermions)
        self.couplings['cW']=sum(f.PQ * f.dynkin_index_weak * f.color_dim for f in fermions)


# Benchmark Models

beta = sp.symbols('beta')
KSVZ_charge = sp.symbols(r'\mathcal{X}')

QED_DFSZ= model('QED-DFSZ', {'lL': 2*sp.cos(beta)**2, 'uR': -2*sp.sin(beta)**2, 'dR': 2*sp.sin(beta)**2})
u_DFSZ= model('u-DFSZ', {'lL': 2*sp.cos(beta)**2, 'uR': -2*sp.sin(beta)**2})
d_DFSZ= model('d-DFSZ', {'lL': 2*sp.cos(beta)**2, 'dR': 2*sp.sin(beta)**2})
Q_KSVZ=KSVZ_model('Q-KSVZ', [fermion(3,1,0,KSVZ_charge)])
L_KSVZ=KSVZ_model('L-KSVZ', [fermion(1,2,0,KSVZ_charge)])
