import numpy as np
from ...biblio.biblio import citations
from ...rge.classes import ALPcouplings
from ...rge.runSM import runSM

def amp_PtoP(MI: float, MF: float, f0: float, fa: float, c: complex) -> complex:
    return 0.5 * c * (MI**2 - MF**2) * f0 / fa

def amp_PtoV(MV: float, A0: float, fa: float, c: complex) -> complex:
    return -1j * c * MV * A0 / fa

def amp_VtoP(MV: float, A0: float, fa: float, c: complex) -> complex:
    return 1j * c * MV * A0 / fa

def Kminustopia(ma: float, couplings: ALPcouplings, f_a: float, delta8=0, **kwargs):
    from ...constants import mK, mpi_pm, g8, fpi, GF
    if ma > mK-mpi_pm:
        return 0
    citations.register_inspire('Bauer:2021wjo')
    coupl_low = couplings.match_run(ma, 'RL_below', **kwargs)
    cG = coupl_low['cG']
    cuu = coupl_low['cuR'][0,0]-coupl_low['cuL'][0,0]
    cdd = coupl_low['cdR'][0,0]-coupl_low['cdL'][0,0]
    css = coupl_low['cdR'][1,1]-coupl_low['cdL'][1,1]
    kd = coupl_low['cdR'][0,0]
    kD = coupl_low['cdL'][0,0]
    ks = coupl_low['cdR'][1,1]
    kS = coupl_low['cdL'][1,1]
    parsSM = runSM(mK)
    Vckm = parsSM['CKM']
    N8 = -g8*GF/np.sqrt(2)*np.conj(Vckm[0,0])*Vckm[0,1]*fpi**2*(np.cos(delta8)+1j*np.sin(delta8))

    chiral_contrib = 16*cG*(mK**2-mpi_pm**2)*(mK**2-ma**2)/(4*mK**2-mpi_pm**2-3*ma**2)
    chiral_contrib += 6*(cuu+cdd-2*css)*ma**2*(mK**2-ma**2)/(4*mK**2-mpi_pm**2-3*ma**2)
    chiral_contrib += (2*cuu+cdd+css)*(mK**2-mpi_pm**2-ma**2) + 4*css*ma**2
    chiral_contrib += (kd+kD-ks-kS)*(mK**2+mpi_pm**2-ma**2)

    return -1j*N8*chiral_contrib/(4*f_a)+1j*(mK**2-mpi_pm**2)/(2*f_a)*(coupl_low['cdR'][0,1]+coupl_low['cdL'][0,1])

def K0bartopia(ma: float, couplings: ALPcouplings, f_a: float, delta8=0, **kwargs):
    from ...constants import mKL, mpi0, g8, fpi, GF, mu, md, ms
    if ma > mKL-mpi0:
        return (0, 0)
    B0 = mpi0**2/(mu+md)
    mK = np.sqrt(B0*(ms+mu/2+md/2))
    citations.register_inspire('Bauer:2021mvw')
    coupl_low = couplings.match_run(ma, 'RL_below', **kwargs)
    cG = coupl_low['cG']
    cuu = coupl_low['cuR'][0,0]-coupl_low['cuL'][0,0]
    cdd = coupl_low['cdR'][0,0]-coupl_low['cdL'][0,0]
    css = coupl_low['cdR'][1,1]-coupl_low['cdL'][1,1]
    kd = coupl_low['cdR'][0,0]
    kD = coupl_low['cdL'][0,0]
    ks = coupl_low['cdR'][1,1]
    kS = coupl_low['cdL'][1,1]
    parsSM = runSM(mK)
    Vckm = parsSM['CKM']
    N8 = -g8*GF/np.sqrt(2)*np.conj(Vckm[0,0])*Vckm[0,1]*fpi**2*(np.cos(delta8)+1j*np.sin(delta8))

    chiral_contrib = 16*cG*(mK**2-mpi0**2)*(mK**2-ma**2)/(4*mK**2-mpi0**2-3*ma**2)
    chiral_contrib -= 2*(cuu+cdd-2*css)*ma**2*(mK**2-ma**2)/(4*mK**2-mpi0**2-3*ma**2)
    chiral_contrib += (3*cdd+css)*(mK**2-mpi0**2)+(2*cuu-cdd-css)*ma**2
    chiral_contrib -= 2*(cuu-cdd)*ma**2*(mK**2-ma**2)/(mpi0**2-ma**2)
    chiral_contrib += (kd+kD-ks-kS)*(mK**2+mpi0**2-ma**2)

    return 1j*N8*chiral_contrib/(4*f_a*2**0.5)-1j*(mKL**2-mpi0**2)/(2*f_a*2**0.5)*(coupl_low['cdR'][0,1]+coupl_low['cdL'][0,1])