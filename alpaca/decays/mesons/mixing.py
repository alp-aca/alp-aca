import numpy as np
from ...biblio.biblio import citations
from ...common import alpha_s, pars
from ...rge.classes import ALPcouplings
from flavio.physics.mesonmixing.amplitude import M12_d_SM, M12_u_SM, G12_u_SM, G12_d_SM

tex_codes = {
    'delta_mK0': r'\Delta m_{K^0}',
    'epsK': r'|\epsilon_K|',
    'x_D0': r'x_{D^0}',
    'phi12_D0': r'\phi_{12,D^0}',
    'delta_mB0': r'\Delta m_{B^0}',
    'delta_mBs': r'\Delta m_{B_s^0}',
    'ASL_B0': r'\mathcal{A}_\mathrm{SL}(B^0)',
    'ASL_Bs': r'\mathcal{A}_\mathrm{SL}(B_s^0)',
}

def eta(meson) -> np.ndarray:
    citations.register_inspire('Bagger:1997gg')
    if meson == 'K0':
        from ...constants import mK0, md, ms
        factor = mK0**2/(md+ms)**2
    if meson == 'B0':
        from ...constants import mB0, md, mb
        factor = mB0**2/(md+mb)**2
    if meson == 'Bs':
        from ...constants import mBs, ms, mb
        factor = mBs**2/(ms+mb)**2
    if meson == 'D0':
        from ...constants import mD0, mu, mc
        factor = mD0**2/(mu+mc)**2
    Nc = 3
    return np.array([1, 1, -0.5*(1-0.5/Nc)*factor, -0.5*(1-0.5/Nc)*factor, -0.5*(1/Nc -0.5)*factor, -0.5*(1/Nc -0.5)*factor, 0.5*factor + 0.25/Nc, 0.5/Nc*factor+0.25])

def oeff(meson: str) -> np.ndarray:
    citations.register_inspire('Bagger:1997gg')
    if meson == 'K0':
        from ...constants import B1_K0, B2_K0, B3_K0, B4_K0, B5_K0, fK0, mK0
        bags = np.array([B1_K0, B1_K0, B2_K0, B2_K0, B3_K0, B3_K0, B4_K0, B5_K0])
        fM = fK0
        mM = mK0
    if meson == 'B0':
        from ...constants import B1_Bd, B2_Bd, B3_Bd, B4_Bd, B5_Bd, fB, mB0
        bags = np.array([B1_Bd, B1_Bd, B2_Bd, B2_Bd, B3_Bd, B3_Bd, B4_Bd, B5_Bd])
        fM = fB
        mM = mB0
    if meson == 'Bs':
        from ...constants import B1_Bs, B2_Bs, B3_Bs, B4_Bs, B5_Bs, fBs, mBs
        bags = np.array([B1_Bs, B1_Bs, B2_Bs, B2_Bs, B3_Bs, B3_Bs, B4_Bs, B5_Bs])
        fM = fBs
        mM = mBs
    if meson == 'D0':
        from ...constants import O1_D0, O2_D0, O3_D0, O4_D0, O5_D0
        oeff_mq = np.array([O1_D0, O1_D0, O2_D0, O2_D0, O3_D0, O3_D0, O4_D0, O5_D0])
    if meson != 'D0':
        oeff_mq = bags * fM**2 * mM**2 * eta(meson)
    return oeff_mq

def run_coeffs(coeffs: np.ndarray, mq1: float, scale: float) -> np.ndarray:
    if scale <= mq1:
        return coeffs
    citations.register_inspire('Bagger:1997gg')
    from ...constants import mc
    s_out = max(scale, mc)
    eta1 = (alpha_s(scale)/alpha_s(s_out))**(6/23)
    eta2 = eta1**(-2.42)
    eta3 = eta1**2.75
    eta4 = eta1**(-4)
    eta5 = eta1**0.5
    ####                  C1   C1tilde     C2                 C2tilde                 C3                     C3tilde               C4    C5
    eta_matrix = np.diag([eta1, eta1, 0.983*eta2+0.017*eta3, 0.983*eta2+0.017*eta3, 0.017*eta2+0.983*eta3, 0.017*eta2+0.983*eta3, eta4, eta5])
    eta_matrix[2,4] = -0.258*eta2+0.258*eta3
    eta_matrix[3,5] = -0.258*eta2+0.258*eta3
    eta_matrix[4,2] = -0.064*eta2+0.064*eta3
    eta_matrix[5,3] = -0.064*eta2+0.064*eta3
    eta_matrix[6,7] = (eta4-eta5)/3
    return eta_matrix @ coeffs

def coeffs_heavyALP(meson: str, couplings: ALPcouplings, ma, fa, **kwargs) -> np.ndarray:
    if ma < couplings.ew_scale:
        coup_low = couplings.match_run(ma, 'RL_below', **kwargs)
    else:
        coup_low = couplings.match_run(ma, 'massbasis_ew', **kwargs)
    if meson == 'K0':
        from ...constants import md, ms
        mq1 = ms
        mq2 = md
        cL = coup_low['cdL'][0,1]
        cR = coup_low['cdR'][0,1]
    if meson == 'B0':
        from ...constants import md, mb
        mq1 = mb
        mq2 = md
        cL = coup_low['cdL'][0,2]
        cR = coup_low['cdR'][0,2]
    if meson == 'Bs':
        from ...constants import ms, mb
        mq1 = mb
        mq2 = ms
        cL = coup_low['cdL'][1,2]
        cR = coup_low['cdR'][1,2]
    if meson == 'D0':
        from ...constants import mu, mc
        mq1 = mc
        mq2 = mu
        cL = coup_low['cuL'][0,1]
        cR = coup_low['cuR'][0,1]
    c2 = (cR*mq1-cL*mq2)**2/(2*ma**2*fa**2)
    c2tilde = (cL*mq1-cR*mq2)**2/(2*ma**2*fa**2)
    c4 = (cR*mq1-cL*mq2)*(cL*mq1-cR*mq2)/(ma**2*fa**2)
    return run_coeffs(np.array([0, 0, c2, c2tilde, 0, 0, c4, 0]), mq1, ma)

def coeffs_lightALP(meson: str, couplings: ALPcouplings, ma, fa, **kwargs) -> np.ndarray:
    citations.register_inspire('Bauer:2021mvw')
    coup_low = couplings.match_run(ma, 'RL_below', **kwargs)
    if meson == 'B0':
        from ...constants import md, mb, mB0
        mq1 = mb
        mq2 = md
        mM = mB0
        cL = coup_low['cdL'][0,2]
        cR = coup_low['cdR'][0,2]
    if meson == 'Bs':
        from ...constants import ms, mb, mBs
        mq1 = mb
        mq2 = ms
        mM = mBs
        cL = coup_low['cdL'][1,2]
        cR = coup_low['cdR'][1,2]
    Lam = mM - mq1
    prop_s = 1/(mM**2 - ma**2)
    prop_t = 1/((mq1-Lam)**2 - ma**2)
    Nc = 3
    cLcL = (cL*mq1-cR*mq2)**2
    cRcR = (cR*mq1-cL*mq2)**2
    cLcR = (cL*mq1-cR*mq2)*(cR*mq1-cL*mq2)
    c2 = - (Nc**2*prop_s - prop_t)/(Nc**2-1) * cRcR/(2*fa**2)
    c2tilde = - (Nc**2*prop_s - prop_t)/(Nc**2-1) * cLcL/(2*fa**2)
    c3 = - Nc*(prop_t - prop_s)/(Nc**2-1) * cRcR/(2*fa**2)
    c3tilde = - Nc*(prop_t - prop_s)/(Nc**2-1) * cLcL/(2*fa**2)
    c4 = - (Nc**2*prop_s - prop_t)/(Nc**2-1) * cLcR/(fa**2)
    c5 = - Nc*(prop_t - prop_s)/(Nc**2-1) * cLcR/(fa**2)
    return np.array([0, 0, c2, c2tilde, c3, c3tilde, c4, c5])

def effhamiltonian(meson: str, couplings: ALPcouplings, ma, fa, **kwargs) -> complex:
    from ...constants import mK0, mB0, mBs, mD0, mb
    mM = {'K0': mK0, 'B0': mB0, 'Bs': mBs, 'D0': mD0}[meson]
    if ma > 1.2*mM:
        effH = np.dot(oeff(meson), coeffs_heavyALP(meson, couplings, ma, fa, **kwargs))
    elif meson in ['B0', 'Bs'] and ma < 0.8*(2*mb - mM):
        effH = np.dot(oeff(meson), coeffs_lightALP(meson, couplings, ma, fa, **kwargs))
    else:
        effH = np.nan
    # Add SM contribution
    citations.register_inspire('Straub:2018kue')
    if meson == 'D0':
        effH += 2*M12_u_SM(pars)*mM
    else:
        effH += 2*M12_d_SM(pars, meson)*mM
    return effH

def delta_mK0(couplings: ALPcouplings, ma, fa, **kwargs) -> float:
    from ...constants import mK0, hbar_GeVps
    return np.real(effhamiltonian('K0', couplings, ma, fa, **kwargs)/mK0)/hbar_GeVps

def epsK(couplings: ALPcouplings, ma, fa, **kwargs) -> float:
    effH = effhamiltonian('K0', couplings, ma, fa, **kwargs)
    return 0.5*np.imag(effH)/np.real(effH)/np.sqrt(2)

def x_D0(couplings: ALPcouplings, ma, fa, **kwargs) -> float:
    from ...constants import mD0, GammaD0
    effH = effhamiltonian('D0', couplings, ma, fa, **kwargs)
    return np.abs(effH)/mD0/GammaD0

def phi12_D0(couplings: ALPcouplings, ma, fa, **kwargs) -> float:
    from ...constants import mD0
    effH = effhamiltonian('D0', couplings, ma, fa, **kwargs)
    return np.angle(effH/(2*mD0*G12_u_SM(pars)))

def delta_mB0(couplings: ALPcouplings, ma, fa, **kwargs) -> float:
    from ...constants import mB0, hbar_GeVps
    return np.abs(effhamiltonian('B0', couplings, ma, fa, **kwargs)/mB0/hbar_GeVps)

def delta_mBs(couplings: ALPcouplings, ma, fa, **kwargs) -> float:
    from ...constants import mBs, hbar_GeVps
    return np.abs(effhamiltonian('Bs', couplings, ma, fa, **kwargs)/mBs/hbar_GeVps)

def ASL_Bd(couplings: ALPcouplings, ma, fa, **kwargs) -> float:
    from ...constants import mB0
    heff = effhamiltonian('B0', couplings, ma, fa, **kwargs)
    gamma12 = G12_d_SM(pars, 'B0')
    phi12 = np.angle(-heff/(2*mB0*gamma12))
    return 2 * mB0 * np.abs(gamma12/heff) * np.sin(phi12)

def ASL_Bs(couplings: ALPcouplings, ma, fa, **kwargs) -> float:
    from ...constants import mBs
    heff = effhamiltonian('Bs', couplings, ma, fa, **kwargs)
    gamma12 = G12_d_SM(pars, 'Bs')
    phi12 = np.angle(-heff/(2*mBs*gamma12))
    return 2 * mBs * np.abs(gamma12/heff) * np.sin(phi12)

mixing_observables = {
    'delta_mK0': delta_mK0,
    'epsK': epsK,
    'x_D0': x_D0,
    'phi12_D0': phi12_D0,
    'delta_mB0': delta_mB0,
    'delta_mBs': delta_mBs,
    'ASL_B0': ASL_Bd,
    'ASL_Bs': ASL_Bs,
}

def meson_mixing(obs: str, ma: float, couplings: ALPcouplings, fa: float, **kwargs) -> float:
    '''Obtains the value of a meson mixing observable.

    Parameters
    ----------
    obs : str
        The observable to calculate. The available options are:
        - 'delta_mK0': The mass difference of the K0 meson, in ps^{-1}.
        - 'epsK': The epsilon parameter of the K0 meson.
        - 'x_D0': Normalized mass difference in D0 mixing.
        - 'phi12_D0': D0 mixing phase, in rad.
        - 'delta_mB0': The mass difference of the B0 meson, in ps^{-1}.
        - 'delta_mBs': The mass difference of the Bs meson, in ps^{-1}.

    ma : float
        The mass of the ALP, in GeV.

    couplings : ALPcouplings
        The couplings of the ALP to other particles.

    fa : float
        The decay constant of the ALP, in GeV.
    '''
    return np.vectorize(mixing_observables[obs], otypes=[float])(couplings, ma, fa, **kwargs)