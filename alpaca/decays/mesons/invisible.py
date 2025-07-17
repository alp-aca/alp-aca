import numpy as np

from ...rge import ALPcouplings
from ...rge.runSM import runSM
from ...biblio.biblio import citations
from ..ee.cross_sections import sigmaNR_gammaALP
from ..effcouplings import effcouplings_cq1q2_W
from . import transition_fv, transition_tree_level
from ...common import ckm_xi

def Kminustopia(ma: float, couplings: ALPcouplings, f_a: float=1000, delta8=0, **kwargs):
    from ...constants import mK, mpi_pm, g8, g2712, g2732, fpi, GammaK, GF, epsisos
    from ... common import kallen
    if ma > mK-mpi_pm:
        return 0
    citations.register_inspire('Bauer:2021wjo')
    coupl_low = couplings.match_run(ma, 'RL_below', **kwargs)
    cG = coupl_low['cG']
    cAuu = coupl_low['cuR'][0,0]-coupl_low['cuL'][0,0]
    cVuu = coupl_low['cuR'][0,0]+coupl_low['cuL'][0,0]
    cAdd = coupl_low['cdR'][0,0]-coupl_low['cdL'][0,0]
    cVdd = coupl_low['cdR'][0,0]+coupl_low['cdL'][0,0]
    cAss = coupl_low['cdR'][1,1]-coupl_low['cdL'][1,1]
    cVss = coupl_low['cdR'][1,1]+coupl_low['cdL'][1,1]
    kd = coupl_low['cdR'][0,0]
    kD = coupl_low['cdL'][0,0]
    ks = coupl_low['cdR'][1,1]
    kS = coupl_low['cdL'][1,1]
    parsSM = runSM(mK)
    Vckm = parsSM['CKM']
    
    chiralG8 = -GF/np.sqrt(2)*np.conjugate(Vckm[0,0])*Vckm[0,1]*g8
    chiralG2712 = -GF/np.sqrt(2)*np.conjugate(Vckm[0,0])*Vckm[0,1]*g2712
    chiralG2732 = -GF/np.sqrt(2)*np.conjugate(Vckm[0,0])*Vckm[0,1]*g2732

    chiral_contrib = (fpi**2*chiralG8*(16*cG*(ma**2 - mK**2)*(mK**2 - mpi_pm**2) - cVdd*(ma**2 - mK**2 - mpi_pm**2)*(3*ma**2 - 4*mK**2 + mpi_pm**2) + cVss*(ma**2 - mK**2 - mpi_pm**2)*(3*ma**2 - 4*mK**2 + mpi_pm**2) +2*cAuu*(mK**2 - mpi_pm**2)*(4*ma**2 - 4*mK**2 + mpi_pm**2) - cAss*(3*ma**4 - 3*ma**2*mK**2 + 4*mK**4 - 5*mK**2*mpi_pm**2 + mpi_pm**4) + cAdd*(3*ma**4 - 4*mK**4 + 5*mK**2*mpi_pm**2 - mpi_pm**4 + ma**2*(mK**2 - 4*mpi_pm**2)) + (8*epsisos*(cAuu*ma**2*(5*ma**2 - 4*mK**2 - mpi_pm**2)*(mK**2 - mpi_pm**2)**2 + (mK**2 - mpi_pm**2)*(cAdd*ma**2*(ma**2 - mK**2)*(9*ma**2 - 4*mK**2 - 5*mpi_pm**2) - (ma**2 - mpi_pm**2)*(cAss*ma**2*(9*ma**2 - 8*mK**2 - mpi_pm**2) - 16*cG*(ma**2 - mK**2)*(mK**2 - mpi_pm**2)))))/(np.sqrt(3)*(ma**2 - mpi_pm**2)*(3*ma**2 - 4*mK**2 + mpi_pm**2))))/(4*f_a*(3*ma**2 - 4*mK**2 + mpi_pm**2)) #G8
    chiral_contrib += (fpi**2*chiralG2712*(4*cAuu*(mK**2 - mpi_pm**2)*(ma**2 - 4*mK**2 + mpi_pm**2) - cVdd*(ma**2 - mK**2 - mpi_pm**2)*(3*ma**2 - 4*mK**2 + mpi_pm**2) + cVss*(ma**2 - mK**2 - mpi_pm**2)*(3*ma**2 - 4*mK**2 + mpi_pm**2) + 8*cG*(mK**2 - mpi_pm**2)*(ma**2 - 4*mK**2 + 3*mpi_pm**2) + 3*cAdd*(ma**4 - ma**2*mK**2 - 4*mK**4 + 5*mK**2*mpi_pm**2 - mpi_pm**4) + cAss*(-3*ma**4 - ma**2*(mK**2 - 4*mpi_pm**2) + 7*(4*mK**4 - 5*mK**2*mpi_pm**2 + mpi_pm**4)) + (8*epsisos*(2*cAuu*ma**2*(mK**2 - mpi_pm**2)**2*(5*ma**2 - 12*mK**2 + 7*mpi_pm**2) + (mK**2 - mpi_pm**2)*(-((ma**2 - mpi_pm**2)*(4*cG*(ma**2 + 4*mK**2 - 5*mpi_pm**2)*(mK**2 - mpi_pm**2) + cAss*ma**2*(9*ma**2 - 28*mK**2 + 19*mpi_pm**2))) + cAdd*ma**2*(9*ma**4 + 24*mK**4 - 10*mK**2*mpi_pm**2 - 5*mpi_pm**4 + ma**2*(-38*mK**2 + 20*mpi_pm**2)))))/(np.sqrt(3)*(ma**2 - mpi_pm**2)*(3*ma**2 - 4*mK**2 + mpi_pm**2))))/(4*f_a*(3*ma**2 - 4*mK**2 + mpi_pm**2)) #G2712
    chiral_contrib += (fpi**2*chiralG2732*(8*cG*(ma**2 - mK**2)*(mK**2 - mpi_pm**2) - cVdd*(ma**2 - mK**2 - mpi_pm**2)*(3*ma**2 - 4*mK**2 + mpi_pm**2) + cVss*(ma**2 - mK**2 - mpi_pm**2)*(3*ma**2 - 4*mK**2 + mpi_pm**2) +cAss*(-3*ma**4 + 4*mK**4 - 5*mK**2*mpi_pm**2 + mpi_pm**4 - ma**2*(mK**2 - 4*mpi_pm**2)) + (4*cAuu*(mK**2 - mpi_pm**2)*(ma**4 + 4*mK**2*mpi_pm**2 - mpi_pm**4 - ma**2*(mK**2 + 3*mpi_pm**2)))/(ma**2 - mpi_pm**2) + (3*cAdd*(ma**6 - ma**4*(mK**2 + mpi_pm**2) + ma**2*(4*mK**2*mpi_pm**2 - 3*mpi_pm**4) - mpi_pm**2*(4*mK**4 - 5*mK**2*mpi_pm**2 + mpi_pm**4)))/(ma**2 - mpi_pm**2) + (8*epsisos*(2*cAuu*ma**2*(mK**2 - mpi_pm**2)**2*(5*ma**2 - 6*mK**2 + mpi_pm**2) + (mK**2 - mpi_pm**2)*(cAdd*ma**2*(ma**2 - mpi_pm**2)*(9*ma**2 - 11*mK**2 + 2*mpi_pm**2) - 4*cG*(mK**2 - mpi_pm**2)*(ma**4 + 12*mK**4 - 13*mK**2*mpi_pm**2 + 2*mpi_pm**4 + ma**2*(-11*mK**2 + 9*mpi_pm**2)) + cAss*ma**2*(-9*ma**4 + 12*mK**4 - 25*mK**2*mpi_pm**2 + 4*mpi_pm**4 + ma**2*(mK**2 + 17*mpi_pm**2)))))/(np.sqrt(3)*(ma**2 - mpi_pm**2)*(3*ma**2 - 4*mK**2 + mpi_pm**2))))/(4*f_a*(3*ma**2 - 4*mK**2 + mpi_pm**2)) #G2732
    
    amp = chiral_contrib-(mK**2-mpi_pm**2)/(2*f_a)*(coupl_low['cdR'][0,1]+coupl_low['cdL'][0,1])
    return np.abs(amp)**2/(16*np.pi*mK)*np.sqrt(kallen(1, mpi_pm**2/mK**2, ma**2/mK**2))/GammaK

def Kplustopia(ma: float, couplings: ALPcouplings, f_a: float=1000, delta8=0, **kwargs):
    from ...constants import mK, mpi_pm, g8, g2712, g2732, fpi, GammaK, GF, epsisos
    from ... common import kallen
    if ma > mK-mpi_pm:
        return 0
    citations.register_inspire('Bauer:2021wjo')
    coupl_low = couplings.match_run(ma, 'RL_below', **kwargs)
    cG = coupl_low['cG']
    cAuu = coupl_low['cuR'][0,0]-coupl_low['cuL'][0,0]
    cVuu = coupl_low['cuR'][0,0]+coupl_low['cuL'][0,0]
    cAdd = coupl_low['cdR'][0,0]-coupl_low['cdL'][0,0]
    cVdd = coupl_low['cdR'][0,0]+coupl_low['cdL'][0,0]
    cAss = coupl_low['cdR'][1,1]-coupl_low['cdL'][1,1]
    cVss = coupl_low['cdR'][1,1]+coupl_low['cdL'][1,1]
    kd = coupl_low['cdR'][0,0]
    kD = coupl_low['cdL'][0,0]
    ks = coupl_low['cdR'][1,1]
    kS = coupl_low['cdL'][1,1]
    parsSM = runSM(mK)
    Vckm = parsSM['CKM']

    chiralG8 = -GF/np.sqrt(2)*np.conjugate(Vckm[0,0])*Vckm[0,1]*g8
    chiralG2712 = -GF/np.sqrt(2)*np.conjugate(Vckm[0,0])*Vckm[0,1]*g2712
    chiralG2732 = -GF/np.sqrt(2)*np.conjugate(Vckm[0,0])*Vckm[0,1]*g2732

    chiral_contrib = (fpi**2*chiralG8*(16*cG*(ma**2 - mK**2)*(mK**2 - mpi_pm**2) - cVdd*(ma**2 - mK**2 - mpi_pm**2)*(3*ma**2 - 4*mK**2 + mpi_pm**2) + cVss*(ma**2 - mK**2 - mpi_pm**2)*(3*ma**2 - 4*mK**2 + mpi_pm**2) +2*cAuu*(mK**2 - mpi_pm**2)*(4*ma**2 - 4*mK**2 + mpi_pm**2) - cAss*(3*ma**4 - 3*ma**2*mK**2 + 4*mK**4 - 5*mK**2*mpi_pm**2 + mpi_pm**4) + cAdd*(3*ma**4 - 4*mK**4 + 5*mK**2*mpi_pm**2 - mpi_pm**4 + ma**2*(mK**2 - 4*mpi_pm**2)) + (8*epsisos*(cAuu*ma**2*(5*ma**2 - 4*mK**2 - mpi_pm**2)*(mK**2 - mpi_pm**2)**2 + (mK**2 - mpi_pm**2)*(cAdd*ma**2*(ma**2 - mK**2)*(9*ma**2 - 4*mK**2 - 5*mpi_pm**2) - (ma**2 - mpi_pm**2)*(cAss*ma**2*(9*ma**2 - 8*mK**2 - mpi_pm**2) - 16*cG*(ma**2 - mK**2)*(mK**2 - mpi_pm**2)))))/(np.sqrt(3)*(ma**2 - mpi_pm**2)*(3*ma**2 - 4*mK**2 + mpi_pm**2))))/(4*f_a*(3*ma**2 - 4*mK**2 + mpi_pm**2)) #G8
    chiral_contrib += (fpi**2*chiralG2712*(4*cAuu*(mK**2 - mpi_pm**2)*(ma**2 - 4*mK**2 + mpi_pm**2) - cVdd*(ma**2 - mK**2 - mpi_pm**2)*(3*ma**2 - 4*mK**2 + mpi_pm**2) + cVss*(ma**2 - mK**2 - mpi_pm**2)*(3*ma**2 - 4*mK**2 + mpi_pm**2) + 8*cG*(mK**2 - mpi_pm**2)*(ma**2 - 4*mK**2 + 3*mpi_pm**2) + 3*cAdd*(ma**4 - ma**2*mK**2 - 4*mK**4 + 5*mK**2*mpi_pm**2 - mpi_pm**4) + cAss*(-3*ma**4 - ma**2*(mK**2 - 4*mpi_pm**2) + 7*(4*mK**4 - 5*mK**2*mpi_pm**2 + mpi_pm**4)) + (8*epsisos*(2*cAuu*ma**2*(mK**2 - mpi_pm**2)**2*(5*ma**2 - 12*mK**2 + 7*mpi_pm**2) + (mK**2 - mpi_pm**2)*(-((ma**2 - mpi_pm**2)*(4*cG*(ma**2 + 4*mK**2 - 5*mpi_pm**2)*(mK**2 - mpi_pm**2) + cAss*ma**2*(9*ma**2 - 28*mK**2 + 19*mpi_pm**2))) + cAdd*ma**2*(9*ma**4 + 24*mK**4 - 10*mK**2*mpi_pm**2 - 5*mpi_pm**4 + ma**2*(-38*mK**2 + 20*mpi_pm**2)))))/(np.sqrt(3)*(ma**2 - mpi_pm**2)*(3*ma**2 - 4*mK**2 + mpi_pm**2))))/(4*f_a*(3*ma**2 - 4*mK**2 + mpi_pm**2)) #G2712
    chiral_contrib += (fpi**2*chiralG2732*(8*cG*(ma**2 - mK**2)*(mK**2 - mpi_pm**2) - cVdd*(ma**2 - mK**2 - mpi_pm**2)*(3*ma**2 - 4*mK**2 + mpi_pm**2) + cVss*(ma**2 - mK**2 - mpi_pm**2)*(3*ma**2 - 4*mK**2 + mpi_pm**2) +cAss*(-3*ma**4 + 4*mK**4 - 5*mK**2*mpi_pm**2 + mpi_pm**4 - ma**2*(mK**2 - 4*mpi_pm**2)) + (4*cAuu*(mK**2 - mpi_pm**2)*(ma**4 + 4*mK**2*mpi_pm**2 - mpi_pm**4 - ma**2*(mK**2 + 3*mpi_pm**2)))/(ma**2 - mpi_pm**2) + (3*cAdd*(ma**6 - ma**4*(mK**2 + mpi_pm**2) + ma**2*(4*mK**2*mpi_pm**2 - 3*mpi_pm**4) - mpi_pm**2*(4*mK**4 - 5*mK**2*mpi_pm**2 + mpi_pm**4)))/(ma**2 - mpi_pm**2) + (8*epsisos*(2*cAuu*ma**2*(mK**2 - mpi_pm**2)**2*(5*ma**2 - 6*mK**2 + mpi_pm**2) + (mK**2 - mpi_pm**2)*(cAdd*ma**2*(ma**2 - mpi_pm**2)*(9*ma**2 - 11*mK**2 + 2*mpi_pm**2) - 4*cG*(mK**2 - mpi_pm**2)*(ma**4 + 12*mK**4 - 13*mK**2*mpi_pm**2 + 2*mpi_pm**4 + ma**2*(-11*mK**2 + 9*mpi_pm**2)) + cAss*ma**2*(-9*ma**4 + 12*mK**4 - 25*mK**2*mpi_pm**2 + 4*mpi_pm**4 + ma**2*(mK**2 + 17*mpi_pm**2)))))/(np.sqrt(3)*(ma**2 - mpi_pm**2)*(3*ma**2 - 4*mK**2 + mpi_pm**2))))/(4*f_a*(3*ma**2 - 4*mK**2 + mpi_pm**2)) #G2732

    amp = chiral_contrib-(mK**2-mpi_pm**2)/(2*f_a)*(coupl_low['cdR'][1,0]+coupl_low['cdL'][1,0])
    return np.abs(amp)**2/(16*np.pi*mK)*np.sqrt(kallen(1, mpi_pm**2/mK**2, ma**2/mK**2))/GammaK

def ampK0topia(ma: float, couplings: ALPcouplings, f_a: float=1000, delta8=0, **kwargs):
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

    amp_K0bar = N8*chiral_contrib/(4*f_a*2**0.5)-(mKL**2-mpi0**2)/(2*f_a*2**0.5)*(coupl_low['cdR'][0,1]+coupl_low['cdL'][0,1])
    amp_K0 = -N8*chiral_contrib/(4*f_a*2**0.5)+(mKL**2-mpi0**2)/(2*f_a*2**0.5)*(coupl_low['cdR'][1,0]+coupl_low['cdL'][1,0])
    return (amp_K0, amp_K0bar)

def KLtopia(ma: float, couplings: ALPcouplings, f_a: float=1000, **kwargs):
    from ...constants import mKL, mpi0, epsilonKaon, phiepsilonKaon, GammaKL
    from ... common import kallen
    amp_K0, amp_K0bar = ampK0topia(ma, couplings, f_a, **kwargs)
    eps = epsilonKaon*(np.cos(phiepsilonKaon)+1j*np.sin(phiepsilonKaon))
    amp = ((1+eps)*amp_K0+(1-eps)*amp_K0bar)/np.sqrt(2*(1+np.abs(eps)**2))
    kallen_factor = kallen(1, mpi0**2/mKL**2, ma**2/mKL**2)
    kallen_factor = np.where(kallen_factor>0, kallen_factor, np.nan)
    return np.abs(amp)**2/(16*np.pi*mKL)*np.sqrt(kallen_factor)/GammaKL

def KStopia(ma: float, couplings: ALPcouplings, f_a: float=1000, **kwargs):
    from ...constants import mKS, mpi0, epsilonKaon, phiepsilonKaon, GammaKS
    from ... common import kallen
    amp_K0, amp_K0bar = ampK0topia(ma, couplings, f_a, **kwargs)
    eps = epsilonKaon*(np.cos(phiepsilonKaon)+1j*np.sin(phiepsilonKaon))
    amp = ((1+eps)*amp_K0-(1-eps)*amp_K0bar)/np.sqrt(2*(1+np.abs(eps)**2))
    kallen_factor = kallen(1, mpi0**2/mKS**2, ma**2/mKS**2)
    kallen_factor = np.where(kallen_factor>0, kallen_factor, np.nan)
    return np.abs(amp)**2/(16*np.pi*mKS)*np.sqrt(kallen_factor)/GammaKS

def B0toKsta(ma: float, couplings: ALPcouplings, f_a: float=1000, **kwargs):
    from ...constants import mKst0, mB0
    from ...common import A0_BKst, kallen
    if ma > mB0-mKst0:
        return 0
    citations.register_inspire('Izaguirre:2016dfi')
    coup_low = couplings.match_run(ma, 'VA_below', **kwargs)
    gq_eff = coup_low['cdA'][1,2]/f_a
    kallen_factor = kallen(1, mKst0**2/mB0**2, ma**2/mB0**2)
    kallen_factor = np.where(kallen_factor>0, kallen_factor, np.nan)
    return mB0**3*abs(gq_eff)**2/(64*np.pi) * A0_BKst(ma**2)**2 * kallen_factor**1.5

def BplustoKsta(ma: float, couplings: ALPcouplings, f_a: float=1000, **kwargs):
    from ...constants import mKst_plus, mB
    from ...common import A0_BKst, kallen
    if ma > mB-mKst_plus:
        return 0
    citations.register_inspire('Izaguirre:2016dfi')
    coup_low = couplings.match_run(ma, 'VA_below', **kwargs)
    gq_eff = coup_low['cdA'][1,2]/f_a
    kallen_factor = kallen(1, mKst_plus**2/mB**2, ma**2/mB**2)
    kallen_factor = np.where(kallen_factor>0, kallen_factor, np.nan)
    return mB**3*abs(gq_eff)**2/(64*np.pi) * A0_BKst(ma**2)**2 * kallen_factor**1.5

def Btopia(ma: float, couplings: ALPcouplings, f_a: float=1000, **kwargs):
    from ...constants import mpi_pm, mB
    from ...common import f0_Bpi, kallen
    if ma > mB-mpi_pm:
        return 0
    citations.register_inspire('Bauer:2021mvw')
    coup_low = couplings.match_run(ma, 'VA_below', **kwargs)
    gq_eff = coup_low['cdV'][0,2]/f_a
    kallen_factor = kallen(1, mpi_pm**2/mB**2, ma**2/mB**2)
    kallen_factor = np.where(kallen_factor>0, kallen_factor, np.nan)
    return mB**3*abs(gq_eff)**2/(64*np.pi) * f0_Bpi(ma**2)**2*np.sqrt(kallen_factor)*(1-mpi_pm**2/mB**2)**2

def B0topia(ma: float, couplings: ALPcouplings, f_a: float=1000, **kwargs):
    from ...constants import mpi0, mB0
    from ...common import f0_Bpi, kallen
    if ma > mB0-mpi0:
        return 0
    citations.register_inspire('Bauer:2021mvw')
    coup_low = couplings.match_run(ma, 'VA_below', **kwargs)
    gq_eff = coup_low['cdV'][0,2]/f_a
    kallen_factor = kallen(1, mpi0**2/mB0**2, ma**2/mB0**2)
    kallen_factor = np.where(kallen_factor>0, kallen_factor, np.nan)
    return mB0**3*abs(gq_eff)**2/(128*np.pi) * f0_Bpi(ma**2)**2*np.sqrt(kallen_factor)*(1-mpi0**2/mB0**2)**2

def B0torhoa(ma: float, couplings: ALPcouplings, f_a: float=1000, **kwargs):
    from ...constants import mrho, mB0
    from ...common import A0_Brho, kallen
    if ma > mB0-mrho:
        return 0
    citations.register_inspire('Izaguirre:2016dfi')
    coup_low = couplings.match_run(ma, 'VA_below', **kwargs)
    gq_eff = coup_low['cdA'][0,2]/f_a
    kallen_factor = kallen(1, mrho**2/mB0**2, ma**2/mB0**2)
    kallen_factor = np.where(kallen_factor>0, kallen_factor, np.nan)
    return mB0**3*abs(gq_eff)**2/(64*np.pi) * A0_Brho(ma**2)**2 * kallen_factor**1.5

def Bplustorhoa(ma: float, couplings: ALPcouplings, f_a: float=1000, **kwargs):
    from ...constants import mrho_pm, mB
    from ...common import A0_Brho, kallen
    if ma > mB-mrho_pm:
        return 0
    citations.register_inspire('Izaguirre:2016dfi')
    coup_low = couplings.match_run(ma, 'VA_below', **kwargs)
    gq_eff = coup_low['cdA'][0,2]/f_a
    kallen_factor = kallen(1, mrho_pm**2/mB**2, ma**2/mB**2)
    kallen_factor = np.where(kallen_factor>0, kallen_factor, np.nan)
    return mB**3*abs(gq_eff)**2/(64*np.pi) * A0_Brho(ma**2)**2 * kallen_factor**1.5

def Bstophia(ma: float, couplings: ALPcouplings, f_a: float=1000, **kwargs):
    from ...constants import mBs, mphi
    from ...common import A0_Bsphi, kallen
    if ma > mBs-mphi:
        return 0
    citations.register_inspire('Izaguirre:2016dfi')
    coup_low = couplings.match_run(ma, 'VA_below', **kwargs)
    gq_eff = coup_low['cdA'][1,2]/f_a
    kallen_factor = kallen(1, mphi**2/mBs**2, ma**2/mBs**2)
    kallen_factor = np.where(kallen_factor>0, kallen_factor, np.nan)
    return mBs**3*abs(gq_eff)**2/(64*np.pi) * A0_Bsphi(ma**2)**2 * kallen_factor**1.5

def D0topi0a(ma: float, couplings: ALPcouplings, f_a: float, **kwargs):
    from ...constants import mD0, mpi0
    from ...common import f0_Dpi, kallen
    if ma > mD0-mpi0:
        return 0
    coup_low = couplings.match_run(ma, 'RL_below', **kwargs)
    gq_eff = (effcouplings_cq1q2_W(coup_low, ma**2, 'c', 'u') + coup_low['cuR'][1,0])/f_a
    kallen_factor = kallen(1, mpi0**2/mD0**2, ma**2/mD0**2)
    kallen_factor = np.where(kallen_factor>0, kallen_factor, np.nan)
    return mD0**3*abs(gq_eff)**2/(128*np.pi) * f0_Dpi(ma**2)**2*np.sqrt(kallen_factor)*(1-mpi0**2/mD0**2)**2

def Dplustopiplusa(ma: float, couplings: ALPcouplings, f_a: float, **kwargs):
    from ...constants import mDplus, mpi_pm
    from ...common import f0_Dpi, kallen
    if ma > mDplus-mpi_pm:
        return 0
    coup_low = couplings.match_run(ma, 'RL_below', **kwargs)
    gq_eff = (effcouplings_cq1q2_W(coup_low, ma**2, 'c', 'u') + coup_low['cuR'][1,0])/f_a
    kallen_factor = kallen(1, mpi_pm**2/mDplus**2, ma**2/mDplus**2)
    kallen_factor = np.where(kallen_factor>0, kallen_factor, np.nan)
    return mDplus**3*abs(gq_eff)**2/(64*np.pi) * f0_Dpi(ma**2)**2*np.sqrt(kallen_factor)*(1-mpi_pm**2/mDplus**2)**2

def D0toetaa(ma: float, couplings: ALPcouplings, f_a: float, **kwargs):
    from ...constants import mD0, meta
    from ...common import f0_Deta, kallen
    if ma > mD0-meta:
        return 0
    coup_low = couplings.match_run(ma, 'RL_below', **kwargs)
    gq_eff = (effcouplings_cq1q2_W(coup_low, ma**2, 'c', 'u') + coup_low['cuR'][1,0])/f_a
    kallen_factor = kallen(1, meta**2/mD0**2, ma**2/mD0**2)
    kallen_factor = np.where(kallen_factor>0, kallen_factor, np.nan)
    return mD0**3*abs(gq_eff)**2/(128*np.pi) * f0_Deta(ma**2)**2*np.sqrt(kallen_factor)*(1-meta**2/mD0**2)**2

def D0toetapa(ma: float, couplings: ALPcouplings, f_a: float, **kwargs):
    from ...constants import mD0, metap
    from ...common import f0_Detap, kallen
    if ma > mD0-metap:
        return 0
    coup_low = couplings.match_run(ma, 'RL_below', **kwargs)
    gq_eff = (effcouplings_cq1q2_W(coup_low, ma**2, 'c', 'u') + coup_low['cuR'][1,0])/f_a
    kallen_factor = kallen(1, metap**2/mD0**2, ma**2/mD0**2)
    kallen_factor = np.where(kallen_factor>0, kallen_factor, np.nan)
    return mD0**3*abs(gq_eff)**2/(128*np.pi) * f0_Detap(ma**2)**2*np.sqrt(kallen_factor)*(1-metap**2/mD0**2)**2

def DstoKa(ma: float, couplings: ALPcouplings, f_a: float, **kwargs):
    from ...constants import mDs, mK
    from ...common import f0_DsK, kallen
    if ma > mDs-mK:
        return 0
    coup_low = couplings.match_run(ma, 'RL_below', **kwargs)
    gq_eff = (effcouplings_cq1q2_W(coup_low, ma**2, 'c', 'u') + coup_low['cuR'][1,0])/f_a
    kallen_factor = kallen(1, mK**2/mDs**2, ma**2/mDs**2)
    kallen_factor = np.where(kallen_factor>0, kallen_factor, np.nan)
    return mDs**3*abs(gq_eff)**2/(64*np.pi) * f0_DsK(ma**2)**2*np.sqrt(kallen_factor)*(1-mK**2/mDs**2)**2

def D0torhoa(ma: float, couplings: ALPcouplings, f_a: float, **kwargs):
    from ...constants import mD0, mrho
    from ...common import A0_Drho, kallen
    if ma > mD0-mrho:
        return 0
    coup_low = couplings.match_run(ma, 'RL_below', **kwargs)
    gq_eff = (effcouplings_cq1q2_W(coup_low, ma**2, 'c', 'u') - coup_low['cuR'][1,0])/f_a
    kallen_factor = kallen(1, mrho**2/mD0**2, ma**2/mD0**2)
    kallen_factor = np.where(kallen_factor>0, kallen_factor, np.nan)
    return mD0**3*abs(gq_eff)**2/(64*np.pi) * A0_Drho(ma**2)**2 * kallen_factor**1.5

def Dplustorhoa(ma: float, couplings: ALPcouplings, f_a: float, **kwargs):
    from ...constants import mDplus, mrho_pm
    from ...common import A0_Drho, kallen
    if ma > mDplus-mrho_pm:
        return 0
    coup_low = couplings.match_run(ma, 'RL_below', **kwargs)
    gq_eff = (effcouplings_cq1q2_W(coup_low, ma**2, 'c', 'u') - coup_low['cuR'][1,0])/f_a
    kallen_factor = kallen(1, mrho_pm**2/mDplus**2, ma**2/mDplus**2)
    kallen_factor = np.where(kallen_factor>0, kallen_factor, np.nan)
    return mDplus**3*abs(gq_eff)**2/(64*np.pi) * A0_Drho(ma**2)**2 * kallen_factor**1.5

def DstoKsta(ma: float, couplings: ALPcouplings, f_a: float, **kwargs):
    from ...constants import mDs, mKst_plus
    from ...common import A0_DsKst, kallen
    if ma > mDs-mKst_plus:
        return 0
    coup_low = couplings.match_run(ma, 'RL_below', **kwargs)
    gq_eff = (effcouplings_cq1q2_W(coup_low, ma**2, 'c', 'u') - coup_low['cuR'][1,0])/f_a
    kallen_factor = kallen(1, mKst_plus**2/mDs**2, ma**2/mDs**2)
    kallen_factor = np.where(kallen_factor>0, kallen_factor, np.nan)
    return mDs**3*abs(gq_eff)**2/(64*np.pi) * A0_DsKst(ma**2)**2 * kallen_factor**1.5

class ftilde_JpsiClass:
    def __init__(self):
        self.initialized = False
    def init(self):
        citations.register_inspire('Colquhoun:2025xlx')
        import pandas as pd
        import os
        from scipy.interpolate import interp1d
        current_dir = os.path.dirname(__file__)
        data = pd.read_csv(os.path.join(current_dir, 'HPQCD_Ftilde.txt'), sep=' ', names=['x', 'ftilde', 'sigma'], skiprows=6)
        self.interpolator = interp1d(data['x'], data['ftilde'], kind='cubic', bounds_error=False)
    def __call__(self, x):
        if not self.initialized:
            self.init()
            self.initialized = True
        return self.interpolator(x)
    
ftilde_Jpsi = ftilde_JpsiClass()

def BR_Vagamma(ma: float, couplings: ALPcouplings, mV: float, BeeV: float, GammaV: float, quark: str, f_a: float=1000, **kwargs):
    citations.register_inspire('Merlo:2019anv')
    citations.register_inspire('DiLuzio:2024jip')
    citations.register_inspire('Hwang:1997ie') # Eliminate fV in favour of BR(V->ee)
    from ...common import alpha_em
    coup_low = couplings.match_run(ma, 'VA_below', **kwargs)
    if quark == 'b':
        qq = -1/3
        gaff = coup_low['cdA'][2,2]/f_a
    elif quark=='c':
        qq = 2/3
        gaff = coup_low['cuA'][1,1]/f_a# * ftilde_Jpsi(ma**2/mV**2)
    else:
        raise ValueError("Q must be -1/3 or 2/3")
    aem = alpha_em(mV)
    gaphoton = coup_low['cgamma']*aem/(np.pi*f_a)

    fV = np.sqrt(3 * mV * BeeV * GammaV/(4*np.pi*aem**2 * qq**2))
    if quark == 'c':
        formfactor = ftilde_Jpsi(ma**2/mV**2)
    else:
        formfactor = 2*fV/mV

    return aem *qq**2 *mV**3/24/GammaV * (1-ma**2/mV**2) * np.abs(gaphoton * fV/mV*(1-ma**2/mV**2) - gaff*formfactor)**2

def sigmapeak(mV, BeeV):
     from ...constants import hbarc2_GeV2pb
     return (12*np.pi*BeeV/(mV**2))*hbarc2_GeV2pb

def Mixed_QuarkoniaSearches(ma: float, couplings: ALPcouplings, mV: float, quark: str, f_a: float=1000, **kwargs):
    from ...constants import mJpsi, mUpsilon3S, BeeJpsi, BeeUpsilon3S, GammaJpsi, GammaUpsilon3S
    if mV == mJpsi:
        Corr_Factor=0.03075942
        BeeV = BeeJpsi
        GammaV = GammaJpsi
    elif mV == mUpsilon3S:
        Corr_Factor=0.0023112
        BeeV = BeeUpsilon3S
        GammaV = GammaUpsilon3S
    else:
        raise ValueError("mV must be mJpsi or mUpsilon3S")
    
    sigma_peak=sigmapeak(mV, BeeV)
    return BR_Vagamma(ma, couplings, mV, BeeV, GammaV, quark, f_a, **kwargs)+sigmaNR_gammaALP(ma, couplings, mV**2, f_a, **kwargs)/(Corr_Factor*sigma_peak)
        
def dwPtoPa(ampsq: float, mP1: float, mP2: float, ma: float):
    from ...common import kallen
    return ampsq/(16*np.pi*mP1)*np.sqrt(kallen(1, mP2**2/mP1**2, ma**2/mP1**2))

def dwPtoVa(ampsq: float, mP: float, mV: float, ma: float):
    from ...common import kallen
    return ampsq/(64*np.pi)*np.sqrt(kallen(mP**2, mV**2, ma**2))**3/(mP**3*mV**2)

def dwVtoPa(ampsq: float, mV: float, mP: float, ma: float):
    from ...common import kallen
    return ampsq/(64*np.pi)*np.sqrt(kallen(mV**2, mP**2, ma**2))**3/mV**5

def brBplusKa(ma: float, couplings: ALPcouplings, f_a: float=1000, **kwargs):
    from ...constants import mB, mK, fB, fK, mb, ms, GammaB
    from ...common import f0_BK
    diags = kwargs.get('diagrams', 'fv')
    args_tree = kwargs.get('args_tree', {})
    kwargs = {k: v for k, v in kwargs.items() if k not in ['diagrams', 'args_tree']}
    if ma > mB-mK:
        return 0
    if ma < couplings.ew_scale:
        clow = couplings.match_run(ma, 'RL_below', **kwargs)
        csb = clow['cdL'][1,2]+clow['cdR'][1,2]
        cbb = clow['cdR'][2,2]-clow['cdL'][2,2]
        css = clow['cdR'][1,1]-clow['cdL'][1,1]
    else:
        clow = couplings.match_run(ma, 'massbasis_ew', **kwargs)
        csb = clow['cdL'][1,2]+clow['cdR'][1,2]+effcouplings_cq1q2_W(clow, ma**2, 's', 'b')
        cbb = clow['cdR'][2,2]-clow['cdL'][2,2]
        css = clow['cdR'][1,1]-clow['cdL'][1,1]
    amp = 0
    if diags != 'tree':
        amp += transition_fv.amp_PtoP(mB, mK, f0_BK(ma**2), f_a, csb)
    if diags != 'fv':
        amp += transition_tree_level.pseudo_to_pseudo_schannel_initial(mB, mK, ma, fB, fK, f_a, ms, mb, css, cbb, ckm_xi('u', 'sb'), **args_tree)
        amp += transition_tree_level.pseudo_to_pseudo_schannel_final(mB, mK, ma, fB, fK, f_a, ms, mb, css, cbb, ckm_xi('u', 'sb'), **args_tree)
    return dwPtoPa(np.abs(amp)**2, mB, mK, ma)/GammaB

def brB0Ka(ma: float, couplings: ALPcouplings, f_a: float=1000, **kwargs):
    from ...constants import mB0, mK0, GammaB0
    from ...common import f0_BK
    if ma > mB0-mK0:
        return 0
    if ma < couplings.ew_scale:
        clow = couplings.match_run(ma, 'RL_below', **kwargs)
        csb = clow['cdL'][1,2]+clow['cdR'][1,2]
    else:
        clow = couplings.match_run(ma, 'massbasis_ew', **kwargs)
        csb = clow['cdL'][1,2]+clow['cdR'][1,2]+effcouplings_cq1q2_W(clow, ma**2, 's', 'b')
    amp = transition_fv.amp_PtoP(mB0, mK0, f0_BK(ma**2), f_a, csb)/2
    return dwPtoPa(np.abs(amp)**2, mB0, mK0, ma)/GammaB0

def brBplusKsta(ma: float, couplings: ALPcouplings, f_a: float=1000, **kwargs):
    from ...constants import mB, mKst_plus, fB, fKst, mb, ms, GammaB
    from ...common import A0_BKst
    diags = kwargs.get('diagrams', 'fv')
    args_tree = kwargs.get('args_tree', {})
    kwargs = {k: v for k, v in kwargs.items() if k not in ['diagrams', 'args_tree']}
    if ma > mB-mKst_plus:
        return 0
    if ma < couplings.ew_scale:
        clow = couplings.match_run(ma, 'RL_below', **kwargs)
        csb = clow['cdL'][1,2]+clow['cdR'][1,2]
        cbb = clow['cdR'][2,2]-clow['cdL'][2,2]
        css = clow['cdR'][1,1]-clow['cdL'][1,1]
    else:
        clow = couplings.match_run(ma, 'massbasis_ew', **kwargs)
        csb = clow['cdL'][1,2]+clow['cdR'][1,2]+effcouplings_cq1q2_W(clow, ma**2, 's', 'b')
        cbb = clow['cdR'][2,2]-clow['cdL'][2,2]
        css = clow['cdR'][1,1]-clow['cdL'][1,1]
    amp = 0
    if diags != 'tree':
        amp += transition_fv.amp_PtoV(mKst_plus, A0_BKst(ma**2), f_a, csb)
    if diags != 'fv':
        amp += transition_tree_level.pseudo_to_vector_schannel_initial(mB, mKst_plus, ma, fB, fKst, f_a, ms, mb, css, cbb, ckm_xi('u', 'sb'), **args_tree)
        amp += transition_tree_level.pseudo_to_vector_schannel_final(mB, mKst_plus, ma, fB, fKst, f_a, ms, mb, css, cbb, ckm_xi('u', 'sb'), **args_tree)
    return dwPtoVa(np.abs(amp)**2, mB, mKst_plus, ma)/GammaB

def brB0Ksta(ma: float, couplings: ALPcouplings, f_a: float=1000, **kwargs):
    from ...constants import mB0, mKst0, GammaB0
    from ...common import A0_BKst
    kwargs = {k: v for k, v in kwargs.items() if k not in ['diagrams', 'args_tree']}
    if ma > mB0-mKst0:
        return 0
    if ma < couplings.ew_scale:
        clow = couplings.match_run(ma, 'RL_below', **kwargs)
        csb = clow['cdL'][1,2]+clow['cdR'][1,2]
    else:
        clow = couplings.match_run(ma, 'massbasis_ew', **kwargs)
        csb = clow['cdL'][1,2]+clow['cdR'][1,2]+effcouplings_cq1q2_W(clow, ma**2, 's', 'b')
    amp = transition_fv.amp_PtoV(mKst0, A0_BKst(ma**2), f_a, csb)/2
    return dwPtoVa(np.abs(amp)**2, mB0, mKst0, ma)/GammaB0