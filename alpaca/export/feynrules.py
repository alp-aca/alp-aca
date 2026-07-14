from ..rge import ALPcouplings
import os
import datetime
import numpy as np
from sympy.printing import mathematica_code # as mcode
import sympy as sp
from .utils import almost_real, is_zero_matrix_sp

par_base = '''    (*ParName*) == {
        ParameterType  -> External,
        BlockName      -> ALPINPUTS,
        Value          -> (*ParValue*),
        TeX            -> (*ParTeX*),
        InteractionOrder -> {ALP, 1},
        Description    -> "ALP coupling to (*ParDescription*)"
    }'''
parmatrix_base = '''    (*ParName*) == {
        ParameterType  -> Internal,
        Indices        -> {Index[Generation], Index[Generation]},
        Value          -> {(*ParValue*)},
        TeX            -> (*ParTeX*),
        InteractionOrder -> {ALP, 1},
        Description    -> "ALP coupling to (*ParDescription*)"
    }'''

def strM(x: float) -> str:
    """Convert a float to a string with Mathematica scientific notation."""
    return f'{np.real(x):.6e}'.replace('e', '*^')

def mcode(expr) -> str:
    coeffs_complex = {}
    for c, v in sp.expand(expr).as_coefficients_dict().items():
        if sp.I not in c.as_coeff_mul()[1]:
            coeffs_complex[c] = coeffs_complex.get(c, 0) + float(v)
        else:
            coeffs_complex[c/sp.I] = coeffs_complex.get(c/sp.I, 0) + float(v) * 1j
    result = []
    for c, v in coeffs_complex.items():
        if almost_real(v):
            num = f'({strM(v)})'
        else:
            num = f'({strM(np.real(v))} + I*{strM(np.imag(v))})'
        if c == 1:
            result.append(num)
        else:
            result.append(f'{num} * {mathematica_code(c)}')
    return ' + '.join(result)

def feynrules_export(name: str, couplings: ALPcouplings, ma: float, fa: float, file: str):
    '''
    Export ALP couplings to FeynRules format.

    Parameters
    ----------
    name : str
        Name of the model.
    couplings : ALPcouplings
        ALP couplings to be exported.
    ma : float
        ALP mass in GeV.
    fa : float
        ALP decay constant in GeV.
    file : str
        Path to the output file.
    '''
    if couplings.basis.startswith('sp_'):
        raise ValueError("Exporting couplings in the sp basis is not supported.")
    if couplings.scale > couplings.ew_scale:
        raise NotImplementedError("Exporting couplings above the electroweak scale is not supported yet.")
    else:
        n_up = 2
        couplings = couplings.translate('RL_below')
    with open(os.path.join(os.path.dirname(__file__), 'alp_base.fr'), 'r') as f:
        fr_base = f.read()
    fr_base = fr_base.replace('(*ModelName*)', f'"{name}"')
    fr_base = fr_base.replace('(*ALPmass*)', strM(ma))
    fr_base = fr_base.replace('(*fa*)', strM(fa))
    fr_base = fr_base.replace('(*date*)', f'"{datetime.datetime.now().strftime("%d. %m. %Y")}"')
    lagr = []
    pars = []
    if couplings['cG'] != 0.0:
        lagr.append('cG ALP/fa * aS/(4 * Pi) FS[G, mu, nu, aa] FS[G, rho, sigma, aa] * Eps[mu,nu,rho,sigma]/2')
        par_cg = par_base.replace('(*ParName*)', 'cG').replace('(*ParValue*)', strM(np.real(couplings['cG']))).replace('(*ParTeX*)', 'Subscript[c, G]').replace('(*ParDescription*)', 'gluon')
        pars.append(par_cg)
    if couplings['cgamma'] != 0.0:
        lagr.append('cgamma ALP/fa * aEW/(4 * Pi) FS[A, mu, nu] FS[A, rho, sigma] * Eps[mu,nu,rho,sigma]/2')
        par_cgamma = par_base.replace('(*ParName*)', 'cgamma').replace('(*ParValue*)', strM(np.real(couplings['cgamma']))).replace('(*ParTeX*)', r'Subscript[c, \[Gamma]]').replace('(*ParDescription*)', 'photon')
        pars.append(par_cgamma)
    if np.max(np.abs(couplings['cuL'])) != 0.0:
        lagr.append('del[ALP,mu]/fa * uqbar[sp1,ff1,cc].Ga[mu,sp1,sp2].ProjM[sp2,sp3].uq[sp3,ff2,cc] * cuL[ff1,ff2]')
        matrix_values = {}
        cuL = np.zeros((3, 3), dtype=complex)
        cuL[:n_up, :n_up] = couplings['cuL'][:n_up, :n_up]
        for i in range(3):
            for j in range(3):
                if cuL[i, j] != 0.0:
                    if almost_real(cuL[i, j]):
                        par_cuLij = par_base.replace('(*ParName*)', f'cuL{i+1}{j+1}').replace('(*ParValue*)', strM(cuL[i, j])).replace('(*ParTeX*)', f'Superscript[ Subscript[c, Subscript[u, L]], {i+1}{j+1}]').replace('(*ParDescription*)', f'left-handed up quarks {i+1} {j+1}')
                        pars.append(par_cuLij)
                        matrix_values[f'{i+1}{j+1}'] = f'cuL{i+1}{j+1}'
                    else:
                        par_cuLijRe = par_base.replace('(*ParName*)', f'cuL{i+1}{j+1}Re').replace('(*ParValue*)', strM(np.real(cuL[i, j]))).replace('(*ParTeX*)', f'Superscript[ Subscript[Re c, Subscript[u, L]], {i+1}{j+1}]').replace('(*ParDescription*)', f'left-handed up quarks {i+1} {j+1}, real part')
                        pars.append(par_cuLijRe)
                        par_cuLijIm = par_base.replace('(*ParName*)', f'cuL{i+1}{j+1}Im').replace('(*ParValue*)', strM(np.imag(cuL[i, j]))).replace('(*ParTeX*)', f'Superscript[ Subscript[Im c, Subscript[u, L]], {i+1}{j+1}]').replace('(*ParDescription*)', f'left-handed up quarks {i+1} {j+1}, imaginary part')
                        pars.append(par_cuLijIm)
                        matrix_values[f'{i+1}{j+1}'] = f'cuL{i+1}{j+1}Re + I*cuL{i+1}{j+1}Im'
                else:
                    matrix_values[f'{i+1}{j+1}'] = '0'
        values = ', '.join(f'cuL[{i},{j}] -> {matrix_values[f"{i}{j}"]}' for i in range(1, 4) for j in range(1, 4))
        par_cuL = parmatrix_base.replace('(*ParName*)', 'cuL').replace('(*ParValue*)', values).replace('(*ParTeX*)', 'Subscript[c, Subscript[u, L]]').replace('(*ParDescription*)', 'left-handed up up quarks')
        pars.append(par_cuL)
    if np.max(np.abs(couplings['cuR'])) != 0.0:
        lagr.append('del[ALP,mu]/fa * uqbar[sp1,ff1,cc].Ga[mu,sp1,sp2].ProjP[sp2,sp3].uq[sp3,ff2,cc] * cuR[ff1,ff2]')
        matrix_values = {}
        cuR = np.zeros((3, 3), dtype=complex)
        cuR[:n_up, :n_up] = couplings['cuR'][:n_up, :n_up]
        for i in range(3):
            for j in range(3):
                if cuR[i, j] != 0.0:
                    if almost_real(cuR[i, j]):
                        par_cuRij = par_base.replace('(*ParName*)', f'cuR{i+1}{j+1}').replace('(*ParValue*)', strM(cuR[i, j])).replace('(*ParTeX*)', f'Superscript[ Subscript[c, Subscript[u, R]], {i+1}{j+1}]').replace('(*ParDescription*)', f'right-handed up quarks {i+1} {j+1}')
                        pars.append(par_cuRij)
                        matrix_values[f'{i+1}{j+1}'] = f'cuR{i+1}{j+1}'
                    else:
                        par_cuRijRe = par_base.replace('(*ParName*)', f'cuR{i+1}{j+1}Re').replace('(*ParValue*)', strM(np.real(cuR[i, j]))).replace('(*ParTeX*)', f'Superscript[ Subscript[Re c, Subscript[u, R]], {i+1}{j+1}]').replace('(*ParDescription*)', f'right-handed up quarks {i+1} {j+1}, real part')
                        pars.append(par_cuRijRe)
                        par_cuRijIm = par_base.replace('(*ParName*)', f'cuR{i+1}{j+1}Im').replace('(*ParValue*)', strM(np.imag(cuR[i, j]))).replace('(*ParTeX*)', f'Superscript[ Subscript[Im c, Subscript[u, R]], {i+1}{j+1}]').replace('(*ParDescription*)', f'right-handed up quarks {i+1} {j+1}, imaginary part')
                        pars.append(par_cuRijIm)
                        matrix_values[f'{i+1}{j+1}'] = f'cuR{i+1}{j+1}Re + I*cuR{i+1}{j+1}Im'
                else:
                    matrix_values[f'{i+1}{j+1}'] = '0'
        values = ', '.join(f'cuR[{i},{j}] -> {matrix_values[f"{i}{j}"]}' for i in range(1, 4) for j in range(1, 4))
        par_cuR = parmatrix_base.replace('(*ParName*)', 'cuR').replace('(*ParValue*)', values).replace('(*ParTeX*)', 'Subscript[c, Subscript[u, R]]').replace('(*ParDescription*)', 'right-handed up quarks')
        pars.append(par_cuR)
    if np.max(np.abs(couplings['cdL'])) != 0.0:
        lagr.append('del[ALP,mu]/fa * dqbar[sp1,ff1,cc].Ga[mu,sp1,sp2].ProjM[sp2,sp3].dq[sp3,ff2,cc] * cdL[ff1,ff2]')
        matrix_values = {}
        cdL = couplings['cdL']
        for i in range(3):
            for j in range(3):
                if cdL[i, j] != 0.0:
                    if almost_real(cdL[i, j]):
                        par_cdLij = par_base.replace('(*ParName*)', f'cdL{i+1}{j+1}').replace('(*ParValue*)', strM(cdL[i, j])).replace('(*ParTeX*)', f'Superscript[ Subscript[c, Subscript[d, L]], {i+1}{j+1}]').replace('(*ParDescription*)', f'left-handed down quarks {i+1} {j+1}')
                        pars.append(par_cdLij)
                        matrix_values[f'{i+1}{j+1}'] = f'cdL{i+1}{j+1}'
                    else:
                        par_cdLijRe = par_base.replace('(*ParName*)', f'cdL{i+1}{j+1}Re').replace('(*ParValue*)', strM(np.real(cdL[i, j]))).replace('(*ParTeX*)', f'Superscript[ Subscript[Re c, Subscript[d, L]], {i+1}{j+1}]').replace('(*ParDescription*)', f'left-handed down quarks {i+1} {j+1}, real part')
                        pars.append(par_cdLijRe)
                        par_cdLijIm = par_base.replace('(*ParName*)', f'cdL{i+1}{j+1}Im').replace('(*ParValue*)', strM(np.imag(cdL[i, j]))).replace('(*ParTeX*)', f'Superscript[ Subscript[Im c, Subscript[d, L]], {i+1}{j+1}]').replace('(*ParDescription*)', f'left-handed down quarks {i+1} {j+1}, imaginary part')
                        pars.append(par_cdLijIm)
                        matrix_values[f'{i+1}{j+1}'] = f'cdL{i+1}{j+1}Re + I*cdL{i+1}{j+1}Im'
                else:
                    matrix_values[f'{i+1}{j+1}'] = '0'
        values = ', '.join(f'cdL[{i},{j}] -> {matrix_values[f"{i}{j}"]}' for i in range(1, 4) for j in range(1, 4))
        par_cdL = parmatrix_base.replace('(*ParName*)', 'cdL').replace('(*ParValue*)', values).replace('(*ParTeX*)', 'Subscript[c, Subscript[d, L]]').replace('(*ParDescription*)', 'left-handed down quarks')
        pars.append(par_cdL)
    if np.max(np.abs(couplings['cdR'])) != 0.0:
        lagr.append('del[ALP,mu]/fa * dqbar[sp1,ff1,cc].Ga[mu,sp1,sp2].ProjP[sp2,sp3].dq[sp3,ff2,cc] * cdR[ff1,ff2]')
        matrix_values = {}
        cdR = couplings['cdR']
        for i in range(3):
            for j in range(3):
                if cdR[i, j] != 0.0:
                    if almost_real(cdR[i, j]):
                        par_cdRij = par_base.replace('(*ParName*)', f'cdR{i+1}{j+1}').replace('(*ParValue*)', strM(cdR[i, j])).replace('(*ParTeX*)', f'Superscript[ Subscript[c, Subscript[d, R]], {i+1}{j+1}]').replace('(*ParDescription*)', f'right-handed down quarks {i+1} {j+1}')
                        pars.append(par_cdRij)
                        matrix_values[f'{i+1}{j+1}'] = f'cdR{i+1}{j+1}'
                    else:
                        par_cdRijRe = par_base.replace('(*ParName*)', f'cdR{i+1}{j+1}Re').replace('(*ParValue*)', strM(np.real(cdR[i, j]))).replace('(*ParTeX*)', f'Superscript[ Subscript[Re c, Subscript[d, R]], {i+1}{j+1}]').replace('(*ParDescription*)', f'right-handed down quarks {i+1} {j+1}, real part')
                        pars.append(par_cdRijRe)
                        par_cdRijIm = par_base.replace('(*ParName*)', f'cdR{i+1}{j+1}Im').replace('(*ParValue*)', strM(np.imag(cdR[i, j]))).replace('(*ParTeX*)', f'Superscript[ Subscript[Im c, Subscript[d, R]], {i+1}{j+1}]').replace('(*ParDescription*)', f'right-handed down quarks {i+1} {j+1}, imaginary part')
                        pars.append(par_cdRijIm)
                        matrix_values[f'{i+1}{j+1}'] = f'cdR{i+1}{j+1}Re + I*cdR{i+1}{j+1}Im'
                else:
                    matrix_values[f'{i+1}{j+1}'] = '0'
        values = ', '.join(f'cdR[{i},{j}] -> {matrix_values[f"{i}{j}"]}' for i in range(1, 4) for j in range(1, 4))
        par_cdR = parmatrix_base.replace('(*ParName*)', 'cdR').replace('(*ParValue*)', values).replace('(*ParTeX*)', 'Subscript[c, Subscript[d, R]]').replace('(*ParDescription*)', 'right-handed down quarks')
        pars.append(par_cdR)
    if np.max(np.abs(couplings['ceL'])) != 0.0:
        lagr.append('del[ALP,mu]/fa * lbar[sp1,ff1].Ga[mu,sp1,sp2].ProjM[sp2,sp3].l[sp3,ff2] * ceL[ff1,ff2]')
        matrix_values = {}
        ceL = couplings['ceL']
        for i in range(3):
            for j in range(3):
                if ceL[i, j] != 0.0:
                    if almost_real(ceL[i, j]):
                        par_ceLij = par_base.replace('(*ParName*)', f'ceL{i+1}{j+1}').replace('(*ParValue*)', strM(ceL[i, j])).replace('(*ParTeX*)', f'Superscript[ Subscript[c, Subscript[e, L]], {i+1}{j+1}]').replace('(*ParDescription*)', f'left-handed charged leptons {i+1} {j+1}')
                        pars.append(par_ceLij)
                        matrix_values[f'{i+1}{j+1}'] = f'ceL{i+1}{j+1}'
                    else:
                        par_ceLijRe = par_base.replace('(*ParName*)', f'ceL{i+1}{j+1}Re').replace('(*ParValue*)', strM(np.real(ceL[i, j]))).replace('(*ParTeX*)', f'Superscript[ Subscript[Re c, Subscript[e, L]], {i+1}{j+1}]').replace('(*ParDescription*)', f'left-handed charged leptons {i+1} {j+1}, real part')
                        pars.append(par_ceLijRe)
                        par_ceLijIm = par_base.replace('(*ParName*)', f'ceL{i+1}{j+1}Im').replace('(*ParValue*)', strM(np.imag(ceL[i, j]))).replace('(*ParTeX*)', f'Superscript[ Subscript[Im c, Subscript[e, L]], {i+1}{j+1}]').replace('(*ParDescription*)', f'left-handed charged leptons {i+1} {j+1}, imaginary part')
                        pars.append(par_ceLijIm)
                        matrix_values[f'{i+1}{j+1}'] = f'ceL{i+1}{j+1}Re + I*ceL{i+1}{j+1}Im'
                else:
                    matrix_values[f'{i+1}{j+1}'] = '0'
        values = ', '.join(f'ceL[{i},{j}] -> {matrix_values[f"{i}{j}"]}' for i in range(1, 4) for j in range(1, 4))
        par_ceL = parmatrix_base.replace('(*ParName*)', 'ceL').replace('(*ParValue*)', values).replace('(*ParTeX*)', 'Subscript[c, Subscript[e, L]]').replace('(*ParDescription*)', 'left-handed charged leptons')
        pars.append(par_ceL)
    if np.max(np.abs(couplings['ceR'])) != 0.0:
        lagr.append('del[ALP,mu]/fa * lbar[sp1,ff1].Ga[mu,sp1,sp2].ProjP[sp2,sp3].l[sp3,ff2] * ceR[ff1,ff2]')
        matrix_values = {}
        ceR = couplings['ceR']
        for i in range(3):
            for j in range(3):
                if ceR[i, j] != 0.0:
                    if almost_real(ceR[i, j]):
                        par_ceRij = par_base.replace('(*ParName*)', f'ceR{i+1}{j+1}').replace('(*ParValue*)', strM(ceR[i, j])).replace('(*ParTeX*)', f'Superscript[ Subscript[c, Subscript[e, R]], {i+1}{j+1}]').replace('(*ParDescription*)', f'right-handed charged leptons {i+1} {j+1}')
                        pars.append(par_ceRij)
                        matrix_values[f'{i+1}{j+1}'] = f'ceR{i+1}{j+1}'
                    else:
                        par_ceRijRe = par_base.replace('(*ParName*)', f'ceR{i+1}{j+1}Re').replace('(*ParValue*)', strM(np.real(ceR[i, j]))).replace('(*ParTeX*)', f'Superscript[ Subscript[Re c, Subscript[e, R]], {i+1}{j+1}]').replace('(*ParDescription*)', f'right-handed charged leptons {i+1} {j+1}, real part')
                        pars.append(par_ceRijRe)
                        par_ceRijIm = par_base.replace('(*ParName*)', f'ceR{i+1}{j+1}Im').replace('(*ParValue*)', strM(np.imag(ceR[i, j]))).replace('(*ParTeX*)', f'Superscript[ Subscript[Im c, Subscript[e, R]], {i+1}{j+1}]').replace('(*ParDescription*)', f'right-handed charged leptons {i+1} {j+1}, imaginary part')
                        pars.append(par_ceRijIm)
                        matrix_values[f'{i+1}{j+1}'] = f'ceR{i+1}{j+1}Re + I*ceR{i+1}{j+1}Im'
                else:
                    matrix_values[f'{i+1}{j+1}'] = '0'
        values = ', '.join(f'ceR[{i},{j}] -> {matrix_values[f"{i}{j}"]}' for i in range(1, 4) for j in range(1, 4))
        par_ceR = parmatrix_base.replace('(*ParName*)', 'ceR').replace('(*ParValue*)', values).replace('(*ParTeX*)', 'Subscript[c, Subscript[e, R]]').replace('(*ParDescription*)', 'right-handed charged leptons')
        pars.append(par_ceR)
    if len(pars) > 0:
        fr_base = fr_base.replace('(*ALPparameters*)', ',\n' + ',\n'.join(pars))
    fr_base = fr_base.replace('(*ALPlagrangian*)', '\n + '.join(lagr))
    with open(file, 'w') as f:
        f.write(fr_base)
