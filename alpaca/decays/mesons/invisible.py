import numpy as np

from ...common import f0_BK, f0_Kpi, A0_BKst, kallen
from ...rge import ALPcouplings
from ...citations import citations

def Ktopia(ma: float, couplings: ALPcouplings, f_a: float=1000, **kwargs):
    citations.register_inspire('Izaguirre:2016dfi')
    from ...constants import mK, mpi_pm
    coup_low = couplings.match_run(ma, 'kF_below', **kwargs)
    gq_eff = coup_low['kD'][0,1]/f_a
    return mK**3*abs(gq_eff)**2/(64*np.pi) * f0_Kpi(ma**2)**2*np.sqrt(kallen(1, mpi_pm**2/mK**2, ma**2/mK**2))*(1-mpi_pm**2/mK**2)**2

def BtoKa(ma: float, couplings: ALPcouplings, f_a: float=1000, **kwargs):
    citations.register_inspire('Izaguirre:2016dfi')
    from ...constants import mK, mB
    coup_low = couplings.match_run(ma, 'kF_below', **kwargs)
    gq_eff = coup_low['kD'][1,2]/f_a
    return mB**3*abs(gq_eff)**2/(64*np.pi) * f0_BK(ma**2)**2*np.sqrt(kallen(1, mK**2/mB**2, ma**2/mB**2))*(1-mK**2/mB**2)**2

def B0toKsta(ma: float, couplings: ALPcouplings, f_a: float=1000, **kwargs):
    citations.register_inspire('Izaguirre:2016dfi')
    from ...constants import mKst0, mB0
    coup_low = couplings.match_run(ma, 'kF_below', **kwargs)
    gq_eff = coup_low['kD'][1,2]/f_a
    return mB0**3*abs(gq_eff)**2/(64*np.pi) * A0_BKst(ma**2)**2 * kallen(1, mKst0**2/mB0**2, ma**2/mB0**2)**1.5