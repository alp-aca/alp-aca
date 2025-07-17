from .classes import ALPcouplings
import numpy as np
import sympy as sp
from typing import Callable
from ..biblio.biblio import citations
import particle.literals


Vud = sp.symbols('V_{ud}')
Vus = sp.symbols('V_{us}')
Vub = sp.symbols('V_{ub}')
Vcd = sp.symbols('V_{cd}')
Vcs = sp.symbols('V_{cs}')
Vcb = sp.symbols('V_{cb}')
Vtd = sp.symbols('V_{td}')
Vts = sp.symbols('V_{ts}')
Vtb = sp.symbols('V_{tb}')

Vud_c = sp.symbols('V_{ud}^*')
Vus_c = sp.symbols('V_{us}^*')
Vub_c = sp.symbols('V_{ub}^*')
Vcd_c = sp.symbols('V_{cd}^*')
Vcs_c = sp.symbols('V_{cs}^*')
Vcb_c = sp.symbols('V_{cb}^*')
Vtd_c = sp.symbols('V_{td}^*')
Vts_c = sp.symbols('V_{ts}^*')
Vtb_c = sp.symbols('V_{tb}^*')
Vckm = np.array([[Vud, Vus, Vub], [Vcd, Vcs, Vcb], [Vtd, Vts, Vtb]]).reshape(3,3)
VckmH = np.array([[Vud_c, Vcd_c, Vtd_c], [Vus_c, Vcs_c, Vts_c], [Vub_c, Vcb_c, Vtb_c]]).reshape(3,3)

mW = sp.symbols('m_W')
mZ = sp.symbols('m_Z')

s2w = sp.symbols('s^2_w')
c2w = 1-s2w
yt = sp.symbols('y_t')
yc = sp.symbols('y_c')
y_u = sp.symbols('y_u')
yb = sp.symbols('y_b')
ys = sp.symbols('y_s')
y_d = sp.symbols('y_d')
y_e = sp.symbols('y_e')
ytau = sp.symbols(r'y_{\tau}')
ymu = sp.symbols(r'y_{\mu}')
mt = sp.symbols('m_t')
mc = sp.symbols('m_c')
mu = sp.symbols('m_u')
mb = sp.symbols('m_b')
ms = sp.symbols('m_s')
md = sp.symbols('m_d')
me = sp.symbols('m_e')
mmu = sp.symbols(r'm_{\mu}')
mtau = sp.symbols(r'm_{\tau}')
alpha_em = sp.symbols(r'\alpha_{em}')
alpha_s = sp.symbols('alpha_s')
alpha_1 = sp.symbols('alpha_1')
alpha_2 = sp.symbols('alpha_2')
pi = sp.symbols('pi')
loop = sp.symbols(r'\mathcal{O}_{loop}')


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
    cG = couplings['cG'] - np.trace(2*couplings['cqL']-couplings['cuR']-couplings['cdR'])*sp.Rational(1,2)
    cW = couplings['cW'] - np.trace(3*couplings['cqL']+ couplings['clL'])*sp.Rational(1,2)
    cB = couplings['cB']+np.trace(8*couplings['cuR']+2*couplings['cdR']-couplings['cqL']+6*couplings['ceR']-3*couplings['clL'])*sp.Rational(1,6)
    return {'cGtilde': sp.nsimplify(cG), 'cBtilde': sp.nsimplify(cB), 'cWtilde': sp.nsimplify(cW)}

def beta_ytop(couplings: ALPcouplings) -> ALPcouplings:
    """beta function for the ALP couplings, including only the top Yukawa dependence
    
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
    yu = np.matrix(np.diag([0, 0, yt]))
    yd = np.matrix(np.zeros((3,3)))
    ye = np.matrix(np.zeros((3,3)))

    # Field redefinitions of the fermionic fields that eliminate Ophi, see Eq.(5) of 2012.12272 and the discussion below
    bQ = sp.symbols('beta_qL')
    bL = sp.symbols('beta_lL')
    bu = bQ-1
    bd = bQ+1
    be = bL+1

    # eq(19)
    couplings = couplings.translate('derivative_above')
    X = sp.nsimplify(np.trace(3* couplings['cqL'] @ (yu @ yu.H- yd @ yd.H)-3*couplings['cuR'] @ yu.H @ yu + 3* couplings['cdR'] @ yd.H @ yd - couplings['clL'] @ ye @ ye.H + couplings['ceR'] @ ye.H @ ye))

    # Casimir
    CF = lambda N: sp.Rational(N**2-1, 2*N)

    # hypercharges
    hyp_qL = sp.Rational(1,6)
    hyp_uR = sp.Rational(2,3)
    hyp_dR = sp.Rational(-1,3)
    hyp_lL = sp.Rational(-1,2)
    hyp_eR = -1

    eye = sp.eye(3)

    # eq(18)
    betaqL = (yu @ yu.H @ couplings['cqL'] + couplings['cqL'] @ yu @ yu.H + yd @ yd.H @ couplings['cqL'] + couplings['cqL'] @ yd @ yd.H)/2 - (yu @ couplings['cuR'] @ yu.H + yd @ couplings['cdR'] @ yd.H) + eye * (2*bQ*X - 12*alpha_s**2*CF(3)*tildes['cGtilde']- 12*alpha_2**2*CF(2)*tildes['cWtilde'] - 12*alpha_1**2*hyp_qL**2*tildes['cBtilde'])

    betauR = yu.H @ yu @ couplings['cuR'] + couplings['cuR'] @ yu.H @ yu - 2* yu.H @ couplings['cqL'] @ yu + eye * (2*bu*X + 12*alpha_s**2*CF(3)*tildes['cGtilde']+12*alpha_1**2*hyp_uR**2*tildes['cBtilde'])

    betadR = yd.H @ yd @ couplings['cdR'] + couplings['cdR'] @ yd.H @ yd - 2* yd.H @ couplings['cqL'] @ yd + eye * (2*bd*X + 12*alpha_s**2*CF(3)*tildes['cGtilde']+12*alpha_1**2*hyp_dR**2*tildes['cBtilde'])

    betalL = (ye @ ye.H @ couplings['clL'] + couplings['clL'] @ ye @ ye.H)/2 - ye @ couplings['ceR'] @ ye.H + eye * (2*bL*X - 12*alpha_2**2*CF(2)*tildes['cWtilde']-12*alpha_1**2*hyp_lL**2*tildes['cBtilde'])

    betaeR = ye.H @ ye @ couplings['ceR'] + couplings['ceR'] @ ye.H @ ye - 2*ye.H @ couplings['clL'] @ ye + eye * (2*be*X+12*alpha_1**2*hyp_eR**2*tildes['cBtilde'])

    return ALPcouplings({'cqL': betaqL, 'cuR': betauR, 'cdR': betadR, 'clL': betalL, 'ceR': betaeR}, scale=couplings.scale, basis='derivative_above', ew_scale=couplings.ew_scale)

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
    yu = np.matrix(np.diag([y_u, yc, yt]))
    yd = Vckm @ np.matrix(np.diag([y_d, ys, yb]))
    ye = np.matrix(np.diag([y_e, ymu, ytau]))

    # Field redefinitions of the fermionic fields that eliminate Ophi, see Eq.(5) of 2012.12272 and the discussion below
    bQ = sp.symbols('beta_qL')
    bL = sp.symbols('beta_lL')
    bu = bQ-1
    bd = bQ+1
    be = bL+1

    # eq(19)
    couplings = couplings.translate('derivative_above')
    X = np.trace(3* couplings['cqL'] @ (yu @ yu.H- yd @ yd.H)-3*couplings['cuR'] @ yu.H @ yu + 3* couplings['cdR'] @ yd.H @ yd - couplings['clL'] @ ye @ ye.H + couplings['ceR'] @ ye.H @ ye)

    # Casimir
    CF = lambda N: sp.Rational(N**2-1, 2*N)

    # hypercharges
    hyp_qL = sp.Rational(1,6)
    hyp_uR = sp.Rational(2,3)
    hyp_dR = sp.Rational(-1,3)
    hyp_lL = sp.Rational(-1,2)
    hyp_eR = -1

    # eq(18)
    betaqL = (yu @ yu.H @ couplings['cqL'] + couplings['cqL'] @ yu @ yu.H + yd @ yd.H @ couplings['cqL'] + couplings['cqL'] @ yd @ yd.H)/2 - (yu @ couplings['cuR'] @ yu.H + yd @ couplings['cdR'] @ yd.H) + np.eye(3) * (2*bQ*X - 12*alpha_s**2*CF(3)*tildes['cGtilde']- 12*alpha_2**2*CF(2)*tildes['cWtilde'] - 12*alpha_1**2*hyp_qL**2*tildes['cBtilde'])

    betauR = yu.H @ yu @ couplings['cuR'] + couplings['cuR'] @ yu.H @ yu - 2* yu.H @ couplings['cqL'] @ yu + np.eye(3) * (2*bu*X + 12*alpha_s**2*CF(3)*tildes['cGtilde']+12*alpha_1**2*hyp_uR**2*tildes['cBtilde'])

    betadR = yd.H @ yd @ couplings['cdR'] + couplings['cdR'] @ yd.H @ yd - 2* yd.H @ couplings['cqL'] @ yd + np.eye(3) * (2*bd*X + 12*alpha_s**2*CF(3)*tildes['cGtilde']+12*alpha_1**2*hyp_dR**2*tildes['cBtilde'])

    betalL = (ye @ ye.H @ couplings['clL'] + couplings['clL'] @ ye @ ye.H)/2 - ye @ couplings['ceR'] @ ye.H + np.eye(3) * (2*bL*X - 12*alpha_2**2*CF(2)*tildes['cWtilde']-12*alpha_1**2*hyp_lL**2*tildes['cBtilde'])

    betaeR = ye.H @ ye @ couplings['ceR'] + couplings['ceR'] @ ye.H @ ye - 2*ye.H @ couplings['clL'] @ ye + np.eye(3) * (2*be*X+12*alpha_1**2*hyp_eR**2*tildes['cBtilde'])

    return ALPcouplings({'cqL': betaqL, 'cuR': betauR, 'cdR': betadR, 'clL': betalL, 'ceR': betaeR}, scale=couplings.scale, basis='derivative_above', ew_scale=couplings.ew_scale)

def cGtilde(couplings: ALPcouplings) -> complex:
    citations.register_particle()
    cG = couplings['cG']
    if isinstance(couplings.scale, sp.Expr):
        cG += sp.Rational(1,2)*(couplings['cuR'][0,0]-couplings['cuL'][0,0]) * sp.Heaviside(couplings.scale - mu)
        cG += sp.Rational(1,2)*(couplings['cdR'][0,0]-couplings['cdL'][0,0]) * sp.Heaviside(couplings.scale - md)
        cG += sp.Rational(1,2)*(couplings['cuR'][1,1]-couplings['cuL'][1,1]) * sp.Heaviside(couplings.scale - mc)
        cG += sp.Rational(1,2)*(couplings['cdR'][1,1]-couplings['cdL'][1,1]) * sp.Heaviside(couplings.scale - ms)
        cG += sp.Rational(1,2)*(couplings['cdR'][2,2]-couplings['cdL'][2,2]) * sp.Heaviside(couplings.scale - mb)
    else:
        if couplings.scale > particle.literals.u.mass/1000:
            cG += 0.5*(couplings['cuR'][0,0]-couplings['cuL'][0,0])
        if couplings.scale > particle.literals.d.mass/1000:
            cG += 0.5*(couplings['cdR'][0,0]-couplings['cdL'][0,0])
        if couplings.scale > particle.literals.c.mass/1000:
            cG += 0.5*(couplings['cuR'][1,1]-couplings['cuL'][1,1])
        if couplings.scale > particle.literals.s.mass/1000:
            cG += 0.5*(couplings['cdR'][1,1]-couplings['cdL'][1,1])
        if couplings.scale > particle.literals.b.mass/1000:
            cG += 0.5*(couplings['cdR'][2,2]-couplings['cdL'][2,2])
    return cG

def cgammatilde(couplings: ALPcouplings) -> complex:
    cG = couplings['cG']
    if isinstance(couplings.scale, sp.Expr):
        cG += 3*sp.Rational(2,3)**2*(couplings['cuR'][0,0]-couplings['cuL'][0,0]) * sp.Heaviside(couplings.scale - mu)
        cG += 3*sp.Rational(-1,3)**2*(couplings['cdR'][0,0]-couplings['cdL'][0,0]) * sp.Heaviside(couplings.scale - md)
        cG += 3*sp.Rational(2,3)**2*(couplings['cuR'][1,1]-couplings['cuL'][1,1]) * sp.Heaviside(couplings.scale - mc)
        cG += 3*sp.Rational(-1,3)**2*(couplings['cdR'][1,1]-couplings['cdL'][1,1]) * sp.Heaviside(couplings.scale - ms)
        cG += 3*sp.Rational(-1,3)**2*(couplings['cdR'][2,2]-couplings['cdL'][2,2]) * sp.Heaviside(couplings.scale - mb)
        cG += (couplings['ceR'][0,0]-couplings['ceL'][0,0]) * sp.Heaviside(couplings.scale - me)
        cG += (couplings['ceR'][1,1]-couplings['ceL'][1,1]) * sp.Heaviside(couplings.scale - mmu)
        cG += (couplings['ceR'][2,2]-couplings['ceL'][2,2]) * sp.Heaviside(couplings.scale - mtau)
    else:
        citations.register_particle()
        if couplings.scale > particle.literals.u.mass/1000:
            cG += 3*(2/3)**2*(couplings['cuR'][0,0]-couplings['cuL'][0,0])
        if couplings.scale > particle.literals.d.mass/1000:
            cG += 3*(-1/3)**2*(couplings['cdR'][0,0]-couplings['cdL'][0,0])
        if couplings.scale > particle.literals.c.mass/1000:
            cG += 3*(2/3)**2*(couplings['cuR'][1,1]-couplings['cuL'][1,1])
        if couplings.scale > particle.literals.s.mass/1000:
            cG += 3*(-1/3)**2*(couplings['cdR'][1,1]-couplings['cdL'][1,1])
        if couplings.scale > particle.literals.b.mass/1000:
            cG += 3*(-1/3)**2*(couplings['cdR'][2,2]-couplings['cdL'][2,2])
        if couplings.scale > particle.literals.e_minus.mass/1000:
            cG += (couplings['ceR'][0,0]-couplings['ceL'][0,0])
        if couplings.scale > particle.literals.mu_minus.mass/1000:
            cG += (couplings['ceR'][1,1]-couplings['ceL'][1,1])
        if couplings.scale > particle.literals.tau_minus.mass/1000:
            cG += (couplings['ceR'][2,2]-couplings['ceL'][2,2])
    return cG

def beta_low(couplings: ALPcouplings) -> ALPcouplings:

    beta_d = alpha_s**2/pi**2*cGtilde(couplings)+sp.Rational(3,4)*alpha_em**2/pi**2*sp.Rational(-1,3)**2*cgammatilde(couplings)
    beta_u = alpha_s**2/pi**2*cGtilde(couplings)+sp.Rational(3,4)*alpha_em**2/pi**2*sp.Rational(2,3)**2*cgammatilde(couplings)
    beta_e = sp.Rational(3,4)*alpha_em**2/pi**2*cgammatilde(couplings)

    return ALPcouplings({'cdR': beta_d*np.eye(3), 'cdL': -beta_d*np.eye(3), 'cuR': beta_u*np.eye(2), 'cuL': -beta_u*np.eye(2), 'ceR': beta_e * np.eye(3), 'ceL': beta_e*np.eye(3), 'cnuL': np.zeros((3,3)), 'cG': 0, 'cgamma': 0}, scale=couplings.scale, basis='RL_below', ew_scale=couplings.ew_scale)

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

    result = couplings + beta(couplings) * (sp.log(scale_out/couplings.scale)/(pi**2)*sp.Rational(1,16)*loop)
    result.scale = scale_out
    return result

def derivative2massbasis(couplings: ALPcouplings) -> ALPcouplings:
    cgamma = couplings.values['cW'] + couplings.values['cB']
    cgammaZ = c2w * couplings.values['cW'] - s2w * couplings.values['cB']
    cZ = c2w**2 * couplings.values['cW'] + s2w**2 *couplings.values['cB']

    kD = VckmH @ couplings.values['cqL'] @ Vckm
    return ALPcouplings({'cuL': couplings.values['cqL'], 'cuR': couplings.values['cuR'], 'cdL': kD, 'cdR': couplings.values['cdR'], 'ceL': couplings.values['clL'], 'cnuL': couplings.values['clL'], 'ceR': couplings.values['ceR'], 'cgamma': cgamma, 'cW': couplings.values['cW'], 'cgammaZ': cgammaZ, 'cZ': cZ, 'cG': couplings.values['cG']}, scale=couplings.scale, basis='massbasis_ew', ew_scale=couplings.ew_scale)
        
def gauge_tilde_match(couplings):
        dcW = - np.trace(3*couplings['cuL']+couplings['ceL'])/2
        dcB = np.trace(8*couplings['cuR']+2*couplings['cdR']-couplings['cuL']+6*couplings['ceR']-3*couplings['ceL'])/6
        cW = couplings['cW'] + dcW
        cZ = couplings['cZ'] + c2w**2 * dcW + s2w**2 * dcB
        cgammaZ = couplings['cgammaZ'] + c2w *dcW - s2w * dcB
        cgamma = couplings['cgamma'] + dcW + dcB
        return {'cWtilde': cW, 'cZtilde': cZ, 'cgammatilde': cgamma, 'cgammaZtilde': cgammaZ}

def match_FCNC_d(couplings: ALPcouplings, two_loops = False) -> np.matrix:
    

    if two_loops:
        tildes = gauge_tilde(couplings)
        cW = tildes['cWtilde']
    else:
        cW = couplings['cW']

    xt = sp.symbols('x_t')
    Vtop = np.einsum('ia,bj->ijab', VckmH, Vckm)[:,:,2,2]  # V_{ti}^* V_{tj}
    gx = (1-xt+sp.log(xt))/(1-xt)**2
    logm = sp.log(couplings.ew_scale**2/mt**2)
    kFCNC = 0
    kFCNC += (np.einsum('im,nj,mn->ijm', VckmH, Vckm, couplings['cuL'])[:,:,2] + np.einsum('im,nj,mn->ijn', VckmH, Vckm, couplings['cuL'])[:,:,2]) * (-sp.Rational(1,4)*logm-sp.Rational(3,8)+sp.Rational(3,4)*gx)
    kFCNC += Vtop*couplings['cuL'][2,2]
    kFCNC += Vtop*couplings['cuR'][2,2]*(sp.Rational(1,2)*logm-sp.Rational(1,4)-sp.Rational(3,2)*gx)
    kFCNC -= sp.Rational(3,2)*alpha_em/pi/s2w * cW * Vtop * (1-xt+xt*sp.log(xt))/(1-xt)**2
    return (yt**2/(16*pi**2)*loop) * kFCNC

def match(couplings: ALPcouplings, two_loops = False) -> ALPcouplings:
    T3f = {'U': sp.Rational(1,2), 'D': sp.Rational(-1,2), 'Nu': sp.Rational(1,2), 'E': sp.Rational(-1,2), 'u': 0, 'd': 0, 'e': 0}
    Qf = {'U': sp.Rational(2,3), 'D': sp.Rational(-1,3), 'Nu': 0, 'E': -1, 'u': sp.Rational(2,3), 'd': sp.Rational(-1,3), 'e': -1}

    delta1 = sp.Rational(-11,3)

    ctt = couplings['cuR'][2,2] - couplings['cuL'][2,2]
    if two_loops:
        tildes = gauge_tilde(couplings)
        cW = tildes['cWtilde']
        cZ = tildes['cZtilde']
        cgammaZ = tildes['cgammaZtilde']

    else:
        cW = couplings['cW']
        cZ = couplings['cZ']
        cgammaZ = couplings['cgammaZ']

    scale = couplings.ew_scale
    Delta_kF = {F: yt**2*ctt*(T3f[F]-Qf[F]*s2w)*sp.log(scale**2/mt**2) + \
        alpha_em**2*(cW/2/s2w**2*(sp.log(scale**2/mW**2) + 0.5 + delta1) + 2*cgammaZ/s2w/c2w*Qf[F]*(T3f[F]-Qf[F] * s2w)*(sp.log(scale**2/mZ**2) + 1.5 + delta1) + cZ/s2w**2/c2w**2 *(T3f[F]-Qf[F]*s2w)**2 *(sp.log(scale**2/mZ**2) + 0.5 + delta1) ) for F in ['U', 'D', 'Nu', 'E', 'u', 'd', 'e']}

    values = {f'k{F}': couplings[f'k{F}'] + (3/(8*pi**2)*Delta_kF[F]*loop)*sp.eye(3) for F in ['Nu', 'E', 'd', 'e']}
    values |= {f'k{F}': couplings[f'k{F}'][0:2,0:2] + (3/(8*pi**2)*Delta_kF[F]*loop)*sp.eye(2) for F in ['U', 'u']}
    values |= {'cdL': couplings['cdL'] + (3/(8*pi**2)*Delta_kF['D']*loop)*sp.eye(3) + match_FCNC_d(couplings, two_loops)}
    values |= {'cG': couplings['cG'], 'cgamma': couplings['cgamma']}
    return ALPcouplings(values, scale=couplings.scale, basis='RL_below', ew_scale=couplings.ew_scale)

def clean_expression(expr: sp.Expr, order_lam: int|None = None):
    """Simplifies a given symbolic expression and optionally substitutes the Wolfenstein parameterization of the CKM matrix.

    Parameters
    ----------
    expr : sympy expression
        The symbolic expression to be simplified.

    order_lam : int, optional
        The order of the series expansion in the Wolfenstein parameter lambda. If None, no substitution is performed.
    
    Returns
    -------
        sympy expression: The simplified expression, with optional Wolfenstein parameterization substitution.
    
    Notes:
    - The function first simplifies the expression by expanding and removing higher-order terms.
    - If `order_lam` is provided, the function substitutes the CKM matrix elements with their series expansions up to the specified order in lambda.
    - The CKM matrix elements are parameterized using the Wolfenstein parameters (A, lambda, eta, rho).
    """

    noloop = sp.nsimplify(sp.series(sp.expand(expr), loop, 0, 2).removeO().subs(loop, 1))
    if order_lam is None:
        return noloop
    A = sp.Symbol('A', real=True, nonnegative=True)
    lam = sp.Symbol('lambda', real=True, nonnegative=True)
    eta = sp.Symbol(r'\bar{\eta}', real=True, nonnegative=True)
    rho = sp.Symbol(r'\bar{\rho}', real=True, nonnegative=True)

    s12 = lam
    s23 = A*lam**2
    s13delta = A*lam**3*(rho+eta*sp.I)*sp.sqrt(1-A**2*lam**4)/sp.sqrt(1-lam**2)/(1-A**2*lam**4*(rho+eta*sp.I))
    s13deltac = A*lam**3*(rho-eta*sp.I)*sp.sqrt(1-A**2*lam**4)/sp.sqrt(1-lam**2)/(1-A**2*lam**4*(rho-eta*sp.I))

    c12 = sp.sqrt(1-s12**2)
    c23 = sp.sqrt(1-s23**2)
    c13 = sp.simplify(sp.expand(sp.sqrt(1-s13delta*s13deltac)))
    Vckm = np.array([[c12*c13, s12*c13, s13deltac], [-s12*c23-c12*s23*s13delta, c12*c23-s12*s23*s13delta, s23*c13], [s12*s23-c12*c23*s13delta, -c12*s23-s12*c23*s13delta, c23*c13]])
    VckmH = np.array([[c12*c13, -s12*c23-c12*s23*s13deltac, s12*s23-c12*c23*s13deltac], [s12*c13, c12*c23-s12*s23*s13deltac, -c12*s23-s12*c23*s13deltac], [s13delta, s23*c13, c23*c13]])
    wolfenstein = {
        Vud: sp.series(Vckm[0,0], lam, 0, order_lam+1),
        Vus: sp.series(Vckm[0,1], lam, 0, order_lam+1),
        Vub: sp.series(Vckm[0,2], lam, 0, order_lam+1),
        Vcd: sp.series(Vckm[1,0], lam, 0, order_lam+1),
        Vcs: sp.series(Vckm[1,1], lam, 0, order_lam+1),
        Vcb: sp.series(Vckm[1,2], lam, 0, order_lam+1),
        Vtd: sp.series(Vckm[2,0], lam, 0, order_lam+1),
        Vts: sp.series(Vckm[2,1], lam, 0, order_lam+1),
        Vtb: sp.series(Vckm[2,2], lam, 0, order_lam+1),
        Vud_c: sp.series(VckmH[0,0], lam, 0, order_lam+1),
        Vus_c: sp.series(VckmH[1,0], lam, 0, order_lam+1),
        Vub_c: sp.series(VckmH[2,0], lam, 0, order_lam+1),
        Vcd_c: sp.series(VckmH[0,1], lam, 0, order_lam+1),
        Vcs_c: sp.series(VckmH[1,1], lam, 0, order_lam+1),
        Vcb_c: sp.series(VckmH[2,1], lam, 0, order_lam+1),
        Vtd_c: sp.series(VckmH[0,2], lam, 0, order_lam+1),
        Vts_c: sp.series(VckmH[1,2], lam, 0, order_lam+1),
        Vtb_c: sp.series(VckmH[2,2], lam, 0, order_lam+1)
    }

    return sp.simplify(sp.expand(noloop.subs(wolfenstein)))