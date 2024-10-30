########################
## Library for models ##
########################
import numpy as np
from ..rge import ALPcouplings

# It is enough for this program to give the charges for the model

import sympy as sp

# Create a diagonal matrix
def family_universal(charge):
    return np.diag(np.full(3, charge))

Qleptons = family_universal(-1)
Ququarks = family_universal(2/3)
Qdquarks = family_universal(-1/3)

# Dynkin index for common representations
def group_theory(group: str, representation: str) -> list[float]:
    """
    Returns the Dynkin index for common representations of SU(3) and SU(2) and its dimension
    Parameters:
    group (str): The group (e.g., 'SU(2)', 'SU(3)')
    representation (str): The representation (e.g., '2')
    Returns:
    float: The Dynkin index of the representation
    """
    indices = {
        'SU(2)': {
            '1': [0,1],
            '2': [1/2,2],
            '3': [2,3]
        },
        'SU(3)': {
            '1': [0,1],
            '3': [1/2,3],
            '6':[5/2,6],
            '6_bar': [5/2,6],
            '8': [3,8],
            '15': [10,15]
        }
        # Add more groups and representations as needed
    }
    result = indices.get(group, {}).get(representation)
    if result is None:
        raise KeyError(f"The representation {representation} for group {group} is not tabulated.")
    return result

class ModelBase:
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.couplings = {}
    
    def get_couplings(self, substitutions: dict, scale: float, basis: str) -> ALPcouplings:
        # Substitute the symbolic variables with numerical values directly into the couplings
        substituted_couplings = {key: float(value.subs(substitutions)) for key, value in self.couplings.items()}
        return ALPcouplings(substituted_couplings, scale, basis)
    
    def E_over_N(self) -> sp.Rational:
        return sp.Rational(sp.simplify(self.couplings['cgamma']/self.couplings['cg'])).limit_denominator()

class model(ModelBase): 
    def __init__(self, model_name: str, charges: dict, nonuniversal: bool = False):
        super().__init__(model_name)
        charges = {f: 0 for f in ['lL', 'eR', 'qL', 'uR', 'dR']} | charges # initialize to zero all missing charges
        self.charges = {key: sp.sympify(value) for key, value in charges.items()}  # Convert all values to sympy objects
        
        if nonuniversal:
            raise NotImplementedError('function not implemented... yet...')
        else:
            self.couplings['ceA'] = self.charges['lL'] - self.charges['eR']
            self.couplings['cuA'] = self.charges['qL'] - self.charges['uR']
            self.couplings['cdA'] = self.charges['qL'] - self.charges['dR']
            self.couplings['cgamma'] = sp.simplify(np.trace(
                (family_universal(self.charges['lL']) - family_universal(self.charges['eR'])) @ Qleptons @ Qleptons +
                3 * family_universal(self.charges['qL'] - self.charges['uR']) @ Ququarks @ Ququarks +
                3 * family_universal(self.charges['qL'] - self.charges['dR']) @ Qdquarks @ Qdquarks
            ))
            self.couplings['cg'] = 1/2 * sp.simplify(np.trace(
                2 * family_universal(self.charges['qL']) - family_universal(self.charges['dR']) - family_universal(self.charges['uR'])
            ))


class fermion:
    def __init__(self, SU3_rep: str, SU2_rep: str, Y_hyper: float, PQ: float):
        self.color_dim = group_theory('SU(3)', SU3_rep)[1]
        self.weak_isospin_dim = group_theory('SU(2)', SU2_rep)[1]
        self.dynkin_index_color = group_theory('SU(3)', SU3_rep)[0]
        self.dynkin_index_weak = group_theory('SU(2)', SU2_rep)[0]
        self.hypercharge = Y_hyper
        self.PQ = PQ

class KSVZ_model(ModelBase):
    def __init__(self, model_name: str, fermions: list[fermion]):
        super().__init__(model_name)
        self.couplings['cg']=sum(i.PQ*i.weak_isospin_dim*i.dynkin_index_color for i in fermions)
        self.couplings['cgamma']=sum(i.PQ*i.color_dim*i.weak_isospin_dim*((1/12)*(i.weak_isospin_dim**2-1)+i.hypercharge**2) for i in fermions)


# Benchmark Models

b = sp.symbols('b')
KSVZ_charge=sp.symbols('X')

QED_DFSZ= model('QED-DFSZ', {'lL': 2*sp.cos(b)**2, 'uR': -2*sp.sin(b)**2, 'dR': 2*sp.sin(b)**2})
u_DFSZ= model('u-DFSZ', {'lL': 2*sp.cos(b)**2, 'uR': -2*sp.sin(b)**2})
d_DFSZ= model('d-DFSZ', {'lL': 2*sp.cos(b)**2, 'dR': 2*sp.sin(b)**2})
Q_KSVZ=KSVZ_model('Q-KSVZ', [fermion('3','1',0,KSVZ_charge)])
L_KSVZ=KSVZ_model('L-KSVZ', [fermion('1','2',0,KSVZ_charge)])
