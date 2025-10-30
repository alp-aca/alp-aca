from ..common import alpha_em, alpha_s
from ..rge.runSM import runSM
from ..rge import ALPcouplings
from ..biblio import citations
import numpy as np

class Benchmark:
    def __init__(self):
        citations.register_inspire('Beacham:2019nyx')
        self.model_parameter = ''
    def __call__(self, c, fa) -> ALPcouplings:
        pass

class BC9(Benchmark):
    def __init__(self):
        super().__init__()
        self.model_parameter = 'gagg'
    def __call__(self, g_agg, fa) -> ALPcouplings:
        cB = np.pi/alpha_em(1000) * fa * g_agg
        return ALPcouplings({'cB': cB}, scale=1000, basis='derivative_above')

class BC10(Benchmark):
    def __init__(self):
        super().__init__()
        self.model_parameter = 'gY'
    def __call__(self, gY, fa) -> ALPcouplings:
        vev = np.real(runSM(1000)['vev'])
        cY = 0.5 * gY * fa / vev
        return ALPcouplings({'cuR': cY, 'cdR': cY, 'ceR': cY, 'cqL': -cY, 'clL': -cY}, scale=1000, basis='derivative_above')

class BC11(Benchmark):
    def __init__(self):
        super().__init__()
        self.model_parameter = 'fG'
    def __call__(self, fG, fa) -> ALPcouplings:
        cG = np.pi/alpha_s(1000) * fa / fG
        return ALPcouplings({'cG': cG}, scale=1000, basis='derivative_above')