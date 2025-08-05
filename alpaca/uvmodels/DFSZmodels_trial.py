from alpaca.rge import ALPcouplings
from ..constants import mW, mu, mc, mt
from .model_library import PQChargedModel, couplings_latex, beta
from ..biblio import citations
from ..rge.matching import match_FCNC_d
from ..rge.runSM import runSM
import sympy as sp
import numpy as np


def X1(mq, mH):
    mHc = float(mH['mH+'])
    aux = 2 + (mHc**2)/(mHc**2-mq**2) - 3*mW**2/(mq**2-mW**2)
    aux += 3*mW**4*(mHc**2+mW**2-2*mq**2)/((mHc**2-mW**2)*(mq**2-mW**2)**2)*np.log(mq**2/mW**2)
    aux += mHc**2/(mHc**2-mq**2)*(mHc**2/(mHc**2-mq**2)-6*mW**2/(mHc**2-mW**2))*np.log(mq**2/mHc**2)
    return aux

def X2(mq, mH):
    mHc = float(mH['mH+'])
    aux = -2*mq**2/(mHc**2-mq**2)*(1+mHc**2/(mHc**2-mq**2)*np.log(mq**2/mHc**2))
    return aux


class DFSZSpecificModel(PQChargedModel):
    """A class to define a model of the type DFSZ given the PQ charges of the SM fermions.

    """
    
    def __init__(self, model_name: str, masses: dict[str, sp.Expr]):
        """Initialize the model with the given name and PQ charges.

        Arguments
        ---------
        model_name : str
            The name of the model.
        masses : dict[str, sp.Expr]
            Mass of the charged Higgses in 2HDM. The keys are the names of Higgses.
        """
        charges = {'uR' : 2*sp.sin(beta)**2, 'eR':2*sp.cos(beta)**2, 'dR': 2*sp.cos(beta)**2, 'qL': 0, 'lL': 0}
        super().__init__(model_name, charges)
        masses = {f: 0 for f in ['mH+']} | masses # initialize to zero all missing masses
        self.masses = {key: sp.sympify(value) for key, value in masses.items()}  # Convert all values to sympy objects
    

    def initialize(self):
        citations.register_inspire('DiLuzio:2020wdo')
        citations.register_inspire('Alonso-Alvarez:2021ett')

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
                      **kwargs
                      ) -> ALPcouplings:
        c1 = super().get_couplings(substitutions, scale, ew_scale, VuL, VdL, VuR, VdR, VeL, VeR)
        c2 = c1.match_run(c1.ew_scale, 'massbasis_ew', **kwargs)
        c3 = c2.match_run(c1.ew_scale*(1-1e-8), 'RL_below')
        c3['cdL'] -= match_FCNC_d(c2, two_loops=True)

        parsSM = runSM(c1.ew_scale)
        GF = parsSM['GF']
        VCKM = np.matrix(parsSM['CKM'])
        beta_val = substitutions[beta]

        mquark = [mu, mc, mt]
        for d1 in range(3):
            for d2 in range(3):
                if d1 == d2:
                    continue
                c3['cdL'][d1,d2] += GF/(16*np.pi**2) * np.cos(beta_val)**2/3*(np.sum([np.conjugate(VCKM[i,d1])*VCKM[i,d2]*mquark[i]**2*(X1(mquark[i], self.masses)+X2(mquark[i], self.masses)/np.tan(beta_val)**2) for i in range(3)]))
        return c3

    def _repr_markdown_(self):
        latex_higgses = {'mH+': r'H^\pm'}
        md = super()._repr_markdown_()
        md += "<details><summary><b>Masses:</b></summary>\n\n"
        for f, m in self.masses.items():
            md += f"- $m_{{{latex_higgses[f]}}} = {sp.latex(m)}$ GeV\n"
        md += "\n\n</details>"
        return md