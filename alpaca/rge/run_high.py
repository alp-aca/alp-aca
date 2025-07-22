"""Running of the RG for the ALP couplings above the EW scale

Auxiliary functions
-------------------

    gauge_tilde
    sm_params

Beta functions
--------------
    beta_ytop


Running
-------
    run_leadinglog
"""

import numpy as np
from . import ALPcouplings, runSM
from typing import Callable
from scipy.integrate import solve_ivp
from ..biblio.biblio import citations


def gauge_tilde(couplings: ALPcouplings) -> dict:
    """Calculate the gauge effective couplings invariant under filed redefinitions of the fermions
    
    Implements eq.(20) of 2012.12272

    Parameters
    ----------
    couplings : ALPCouplings
        Object containing the ALP couplings

    Returns
    -------
    cXtilde : dict
        Dictionary containing `cGtilde`, `cBtilde` and `cWtilde`
    """
    couplings = couplings.translate('derivative_above')
    cG = couplings['cG'] - 0.5 *np.trace(2*couplings['cqL']-couplings['cuR']-couplings['cdR'])
    cW = couplings['cW'] - 0.5*np.trace(3*couplings['cqL']+ couplings['clL'])
    cB = couplings['cB']+np.trace(4/3*couplings['cuR']+1/3*couplings['cdR']-1/6*couplings['cqL']+couplings['ceR']-1/2*couplings['clL'])
    return {'cGtilde': cG, 'cBtilde':cB, 'cWtilde': cW}


def beta_ytop(couplings: ALPcouplings) -> ALPcouplings:
    """beta function for the ALP couplings, neglecting all Yukawas except y_top
    
    Implements eq.(24) of 2012.12272

    Parameters
    ----------
    couplings : ALPCouplings
        An object containing the ALP couplings

    Returns
    -------
    beta : ALPCouplings
        An object containing the beta functions of the ALP couplings
    """

    tildes = gauge_tilde(couplings)
    pars = runSM(couplings.scale)
    ytop = couplings.yu[2,2]
    alpha_s = pars['alpha_s']
    alpha_1 = pars['alpha_1']
    alpha_2 = pars['alpha_2']
    yu = np.matrix(np.diag([0, 0, ytop]))
    g_s = np.sqrt(4*np.pi*alpha_s)
    g_1 = np.sqrt(4*np.pi*alpha_1)
    g_2 = np.sqrt(4*np.pi*alpha_2)

    # Field redefinitions of the fermionic fields that eliminate Ophi, see Eq.(5) of 2012.12272 and the discussion below
    bu = -1
    bd = 1
    be = 1
    bQ = 0
    bL = 0

    # eq(25)
    couplings = couplings.translate('derivative_above')
    ctt = couplings['cuR'][2,2]-couplings['cqL'][2,2]

    # eq(24)
    diag_betaqL = -2*ytop**2*np.matrix(np.diag([0,0,1])/2 + 3*bQ*np.eye(3))*ctt +np.eye(3)*(-16*alpha_s**2*tildes['cGtilde'] - 9*alpha_2**2*tildes['cWtilde'] - 1/3*alpha_1**2*tildes['cBtilde'])
    offdiag_betaqL = np.matrix(np.zeros([3,3]), dtype=complex)
    offdiag_betaqL[0,2] = couplings['cqL'][0,2]
    offdiag_betaqL[1,2] = couplings['cqL'][1,2]
    offdiag_betaqL[2,0] = couplings['cqL'][2,0]
    offdiag_betaqL[2,1] = couplings['cqL'][2,1]
    betaqL = diag_betaqL + ytop**2/2 * offdiag_betaqL

    diag_betauR = 2*ytop**2*np.matrix(np.diag([0,0,1])-3*bu*np.eye(3))*ctt + np.eye(3)*(16*alpha_s**2*tildes['cGtilde']+16/3*alpha_1**2*tildes['cBtilde'])
    offdiag_betauR = np.matrix(np.zeros([3,3]), dtype=complex)
    offdiag_betauR[0,2] = couplings['cuR'][0,2]
    offdiag_betauR[1,2] = couplings['cuR'][1,2]
    offdiag_betauR[2,0] = couplings['cuR'][2,0]
    offdiag_betauR[2,1] = couplings['cuR'][2,1]
    betauR = diag_betauR + ytop**2*offdiag_betauR

    betadR = np.eye(3)*(-6*ytop**2*bd*ctt + 16*alpha_s**2*tildes['cGtilde']+16/12*alpha_1**2*tildes['cBtilde'])

    betalL = np.eye(3)*(-6*ytop**2*bL*ctt-9*alpha_2**2*tildes['cWtilde']-3*alpha_1**2*tildes['cBtilde'])

    betaeR = np.eye(3)*(-6*ytop**2*be*ctt+12*alpha_1**2*tildes['cBtilde'])

    gammaH = np.trace(3* yu @ yu.H)
    beta_yu = 3/2*(yu @ yu.H @ yu) + (gammaH -9/4*g_2**2-17/12*g_1**2-8*g_s**2)*yu

    a = ALPcouplings({'cqL': betaqL, 'cuR': betauR, 'cdR': betadR, 'clL': betalL, 'ceR': betaeR}, couplings.scale, 'derivative_above', couplings.ew_scale)
    a.yu = beta_yu
    return a


def beta_full(couplings: ALPcouplings) -> ALPcouplings:
    """beta function for the ALP couplings, including the full Yukawa dependence
    
    Implements eq.(18) of 2012.12272

    Parameters
    ----------
    couplings : ALPCouplings
        An object containing the ALP couplings

    Returns
    -------
    beta : ALPCouplings
        An object containing the beta functions of the ALP couplings
    """

    tildes = gauge_tilde(couplings)
    pars = runSM(couplings.scale)
    yu = np.matrix(couplings.yu)
    yd = np.matrix(couplings.yd)
    ye = np.matrix(couplings.ye)
    alpha_s = pars['alpha_s']
    alpha_1 = pars['alpha_1']
    alpha_2 = pars['alpha_2']
    g_s = np.sqrt(4*np.pi*alpha_s)
    g_1 = np.sqrt(4*np.pi*alpha_1)
    g_2 = np.sqrt(4*np.pi*alpha_2)

    # Field redefinitions of the fermionic fields that eliminate Ophi, see Eq.(5) of 2012.12272 and the discussion below
    bu = -1
    bd = 1
    be = 1
    bQ = 0
    bL = 0

    # eq(19)
    couplings = couplings.translate('derivative_above')
    X = np.trace(3* couplings['cqL'] @ (yu @ yu.H- yd @ yd.H)-3*couplings['cuR'] @ yu.H @ yu + 3* couplings['cdR'] @ yd.H @ yd - couplings['clL'] @ ye @ ye.H + couplings['ceR'] @ ye.H @ ye)

    # Casimir
    CF = lambda N: 0.5*(N**2-1)/N

    # hypercharges
    hyp_qL = 1/6
    hyp_uR = 2/3
    hyp_dR = -1/3
    hyp_lL = -1/2
    hyp_eR = -1

    # eq(18)
    betaqL = 0.5*(yu @ yu.H @ couplings['cqL'] + couplings['cqL'] @ yu @ yu.H + yd @ yd.H @ couplings['cqL'] + couplings['cqL'] @ yd @ yd.H) - (yu @ couplings['cuR'] @ yu.H + yd @ couplings['cdR'] @ yd.H) + np.eye(3) * (2*bQ*X - 12*alpha_s**2*CF(3)*tildes['cGtilde']- 12*alpha_2**2*CF(2)*tildes['cWtilde'] - 12*alpha_1**2*hyp_qL**2*tildes['cBtilde'])

    betauR = yu.H @ yu @ couplings['cuR'] + couplings['cuR'] @ yu.H @ yu - 2* yu.H @ couplings['cqL'] @ yu + np.eye(3) * (2*bu*X + 12*alpha_s**2*CF(3)*tildes['cGtilde']+12*alpha_1**2*hyp_uR**2*tildes['cBtilde'])

    betadR = yd.H @ yd @ couplings['cdR'] + couplings['cdR'] @ yd.H @ yd - 2* yd.H @ couplings['cqL'] @ yd + np.eye(3) * (2*bd*X + 12*alpha_s**2*CF(3)*tildes['cGtilde']+12*alpha_1**2*hyp_dR**2*tildes['cBtilde'])

    betalL = 0.5*(ye @ ye.H @ couplings['clL'] + couplings['clL'] @ ye @ ye.H) - ye @ couplings['ceR'] @ ye.H + np.eye(3) * (2*bL*X - 12*alpha_2**2*CF(2)*tildes['cWtilde']-12*alpha_1**2*hyp_lL**2*tildes['cBtilde'])

    betaeR = ye.H @ ye @ couplings['ceR'] + couplings['ceR'] @ ye.H @ ye - 2*ye.H @ couplings['clL'] @ ye + np.eye(3) * (2*be*X+12*alpha_1**2*hyp_eR**2*tildes['cBtilde'])

    gammaH = np.trace(3* yu @ yu.H + 3* yd @ yd.H + ye @ ye.H)
    beta_yu = 3/2*(yu @ yu.H @ yu - yd @ yd.H @ yu) + (gammaH -9/4*g_2**2-17/12*g_1**2-8*g_s**2)*yu
    beta_yd = 3/2*(yd @ yd.H @ yd - yu @ yu.H @ yd) + (gammaH -9/4*g_2**2-5/12*g_1**2-8*g_s**2)*yd
    beta_ye = 3/2*(ye @ ye.H @ ye) + (gammaH -9/4*g_2**2-15/4*g_1**2)*ye

    a = ALPcouplings({'cqL': betaqL, 'cuR': betauR, 'cdR': betadR, 'clL': betalL, 'ceR': betaeR}, scale=couplings.scale, basis='derivative_above', ew_scale=couplings.ew_scale)
    a.yu = beta_yu
    a.yd = beta_yd
    a.ye = beta_ye
    return a

def run_leadinglog(couplings: ALPcouplings, beta: Callable[[ALPcouplings], ALPcouplings], scale_out: float) -> ALPcouplings:
    """Obtain the ALP couplings at a different scale using the leading log approximation
    
    Parameters
    ----------
    couplings : ALPcouplings
        Object containing the ALP couplings at the original scale

    beta : Callable[ALPcouplings, ALPcouplings]
        Function that return the beta function

    scale_out : float
        Final energy scale, in GeV
    """

    betac = beta(couplings)
    result = couplings + betac * (np.log(scale_out/couplings.scale)/(16*np.pi**2))
    result.scale = scale_out
    result.yu = couplings.yu + betac.yu * (np.log(scale_out/couplings.scale)/(16*np.pi**2))
    result.yd = couplings.yd + betac.yd * (np.log(scale_out/couplings.scale)/(16*np.pi**2))
    result.ye = couplings.ye + betac.ye * (np.log(scale_out/couplings.scale)/(16*np.pi**2))
    return result

def run_scipy(couplings: ALPcouplings, beta: Callable[[ALPcouplings], ALPcouplings], scale_out: float, scipy_options: dict) -> ALPcouplings:
    """Obtain the ALP couplings at a different scale using scipy's integration
    
    Parameters
    ----------
    couplings : ALPcouplings
        Object containing the ALP couplings at the original scale

    beta : Callable[ALPcouplings, ALPcouplings]
        Function that return the beta function

    scale_out : float
        Final energy scale, in GeV
    """
    citations.register_inspire('Virtanen:2019joe')
    def fun(t0, y):
        return beta(ALPcouplings._fromarray(y, np.exp(t0), 'derivative_above', couplings.ew_scale))._toarray()/(16*np.pi**2)
    
    y0 = couplings.translate('derivative_above')._toarray()
    sol = solve_ivp(fun=fun, t_span=(np.log(couplings.scale), np.log(scale_out)), y0=y0, **scipy_options)
    return ALPcouplings._fromarray(sol.y[:,-1], scale_out, 'derivative_above', couplings.ew_scale)
