from ..common import alpha_em, alpha_s
from ..rge.runSM import runSM
from ..rge import ALPcouplings
from ..biblio import citations
import numpy as np

class Benchmark:
    '''
    Base class for all benchmarks.
    '''
    def __init__(self):
        '''
        Initialize the Benchmark class.
        '''
        citations.register_inspire('Beacham:2019nyx')
        self.model_parameter = ''
    def __call__(self, c, fa) -> ALPcouplings:
        pass

class BC9(Benchmark):
    '''
    Benchmark class BC9 representing a photo-phillic ALP.
    '''
    def __init__(self):
        super().__init__()
        self.model_parameter = 'gagg'
    def __call__(self, g_agg, fa) -> ALPcouplings:
        '''
        Compute the ALP couplings for the BC9 benchmark.

        Parameters
        ----------
        g_agg : float
            The coupling of the ALP to gluons, in GeV^(-1).
        fa : float
            The ALP decay constant, in GeV.
        
        Returns
        -------
        ALPcouplings
            The ALP couplings in the derivative basis at scale 1000 GeV.
        '''
        cB = np.pi/alpha_em(1000) * fa * g_agg
        return ALPcouplings({'cB': cB}, scale=1000, basis='derivative_above')

class BC10(Benchmark):
    '''
    Benchmark class BC10 representing an ALP with universal couplings to fermions.
    '''
    def __init__(self):
        super().__init__()
        self.model_parameter = 'gY'
    def __call__(self, gY, fa) -> ALPcouplings:
        '''
        Compute the ALP couplings for the BC10 benchmark.

        Parameters
        ----------
        gY : float
            The coupling of the ALP to fermions, dimensionless.
        fa : float
            The ALP decay constant, in GeV.

        Returns
        -------
        ALPcouplings
            The ALP couplings in the derivative basis at scale 1000 GeV.
        '''
        vev = np.real(runSM(1000)['vev'])
        cY = 0.5 * gY * fa / vev
        return ALPcouplings({'cuR': cY, 'cdR': cY, 'ceR': cY, 'cqL': -cY, 'clL': -cY}, scale=1000, basis='derivative_above')

class BC11(Benchmark):
    '''
    Benchmark class BC11 representing a gluon-phillic ALP.
    '''
    def __init__(self):
        super().__init__()
        self.model_parameter = 'fG'
    def __call__(self, fG, fa) -> ALPcouplings:
        '''
        Compute the ALP couplings for the BC11 benchmark.
        
        Parameters
        ----------
        fG : float
            The ALP-gluon coupling scale, in GeV.
        fa : float
            The ALP decay constant, in GeV.

        Returns
        -------
        ALPcouplings
            The ALP couplings in the derivative basis at scale 1000 GeV.
        '''
        cG = np.pi/alpha_s(1000) * fa / fG
        return ALPcouplings({'cG': cG}, scale=1000, basis='derivative_above')