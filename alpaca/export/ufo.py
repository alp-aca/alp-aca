import os
from ..rge import ALPcouplings
import shutil
import datetime
from .utils import almost_real
import numpy as np

def a_ff(cL: str | None, cR: str | None, mf: str, ncols: bool) -> str:
    if cL is None and cR is None:
        return ''
    elif cL is None and cR is not None:
        result = f'abs({cR})**2'
    elif cL is not None and cR is None:
        result = f'abs({cL})**2'
    else:
        result = f'abs({cL}-{cR})**2'
    result += f' * {mf}**2 * mALP /(8. * fa**2 * cmath.pi) * cmath.sqrt(1 - 4 * {mf}**2 / mALP**2)'
    if ncols:
        result += ' * 3'
    return result

def sqkallen(a: str, b: str, c: str) -> str:
    return f'cmath.sqrt({a}**4 + {b}**4 + {c}**4 - 2*{a}**2*{b}**2 - 2*{b}**2*{c}**2 - 2*{c}**2*{a}**2)'

def a_f1f2(cL: str | None, cR: str | None, mf1: str, mf2: str, ncols: bool) -> str:
    if cL is None and cR is None:
        return ''
    elif cL is None and cR is not None:
        result = f'abs({cR})**2 * (mALP**2 * {mf1} * {mf2} - ({mf1}**2 - {mf2}**2)**2)'
    elif cL is not None and cR is None:
        result = f'abs({cL})**2 * (mALP**2 * {mf1} * {mf2} - ({mf1}**2 - {mf2}**2)**2)'
    else:
        result = f'((abs({cR})**2 + abs({cL})**2) * (mALP**2 * {mf1} * {mf2} - ({mf1}**2 - {mf2}**2)**2)'
        result += f' - 2 * {cL} * complexconjugate({cR}) - 2 * {cR} * complexconjugate({cL}))'
    result += '/(16. * fa**2 * cmath.pi * mALP**3) * ' + sqkallen(mf1, mf2, 'mALP')
    if ncols:
        result += ' * 3'
    return result

def f1_af2(cL: str | None, cR: str | None, mf1: str, mf2: str) -> str:
    if cL is None and cR is None:
        return ''
    elif cL is None and cR is not None:
        result = f'abs({cR})**2 * (({mf1}**2 - {mf2}**2)**2 - mALP**2 * {mf1} * {mf2})'
    elif cL is not None and cR is None:
        result = f'abs({cL})**2 * (({mf1}**2 - {mf2}**2)**2 - mALP**2 * {mf1} * {mf2})'
    else:
        result = f'((abs({cR})**2 + abs({cL})**2) * (({mf1}**2 - {mf2}**2)**2 - mALP**2 * {mf1} * {mf2})'
        result += f' + 2 * {cL} * complexconjugate({cR}) + 2 * {cR} * complexconjugate({cL}))'
    result += f'/(32. * fa**2 * cmath.pi * {mf1}**3) * ' + sqkallen(mf1, mf2, 'mALP')
    return result

def ufo_export(name: str, alp_couplings: ALPcouplings, ma: float, fa: float, filepath: str):
    '''
    Export ALP couplings to UFO format.

    Parameters
    ----------
    name : str
        Name of the model.
    alp_couplings : ALPcouplings
        ALP couplings to be exported.
    ma : float
        ALP mass in GeV.
    fa : float
        ALP decay constant in GeV.
    filepath : str
        Path to the output directory. The UFO model will be created in a subdirectory named f'{name}_UFO'.
    '''
    ufo_dir = os.path.join(os.path.dirname(filepath), f'{name}_UFO')
    shutil.copytree(os.path.join(os.path.dirname(__file__), 'template_UFO'), ufo_dir, dirs_exist_ok=True)
    with open(os.path.join(ufo_dir, 'parameters.py'), 'r') as f:
        parameter_data = f.read()
    with open(os.path.join(ufo_dir, 'vertices.py'), 'r') as f:
        vertices_data = f.read()
    with open(os.path.join(ufo_dir, 'couplings.py'), 'r') as f:
        couplings_data = f.read()
    with open(os.path.join(ufo_dir, 'decays.py'), 'r') as f:
        decays_data = f.read()
    parameter_data = parameter_data.replace('Date_running', datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
    vertices_data = vertices_data.replace('Date_running', datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
    couplings_data = couplings_data.replace('Date_running', datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
    decays_data = decays_data.replace('Date_running', datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
    parameter_data = parameter_data.replace("'massALPvalue'", f'{ma:.6e}').replace("'faALPvalue'", f'{fa:.6e}')
    
    if alp_couplings.scale > alp_couplings.ew_scale:
        raise NotImplementedError('Exporting ALP couplings at scales above the EW scale is not implemented yet.')
    else:
        alp_couplings = alp_couplings.translate('RL_below')
        n_up = 2

    lhacode = 2
    couplings_counter = 134
    vertices_counter = 164

    dquarks = ['d', 's', 'b']
    mdquarks = ['MD', 'MS', 'MB']
    uquarks = ['u', 'c', 't']
    muquarks = ['MU', 'MC', 'MT']
    leptons = ['e', 'mu', 'ta']
    mleptons = ['Me', 'MMU', 'MTA']

    if alp_couplings['cgamma'] != 0:
        parameter_data += f"\n\ncgamma = Parameter(name = 'cgamma',\n\tnature = 'external',\n\ttype = 'real',\n\tvalue = {np.real(alp_couplings['cgamma']):.6e},\n\ttexname = 'c_\\\\gamma',\n\tlhablock = 'ALPINPUTS',\n\tlhacode = [ {lhacode} ])"
        lhacode += 1
        couplings_data += f"\n\nGC_{couplings_counter} = Coupling(name = 'GC_{couplings_counter}',\n\tvalue = '-0.5 * complex(0,1) * aEW * cgamma/(fa * cmath.pi)',\n\torder = {{'ALP': 1, 'QED': 2}})"
        vertices_data += f"\n\nV_{vertices_counter} = Vertex(name = 'V_{vertices_counter}',\n\tparticles = [ P.a, P.a, P.alp ],\n\tcolor = [ '1' ],\n\tlorentz = [ L.VVS2 ],\n\tcouplings = {{(0,0): C.GC_{couplings_counter}}})"
        couplings_counter += 1
        vertices_counter += 1
        decays_data = decays_data.replace('# Decay_alp #', f"(P.a, P.a):'aEW**2 * abs(cgamma)**2 * mALP**3 / (64. * fa**2 * cmath.pi**3)',\n\t\t# Decay_alp #")

    if alp_couplings['cG'] != 0:
        parameter_data += f"\n\ncG = Parameter(name = 'cG',\n\tnature = 'external',\n\ttype = 'real',\n\tvalue = {np.real(alp_couplings['cG']):.6e},\n\ttexname = 'c_G',\n\tlhablock = 'ALPINPUTS',\n\tlhacode = [ {lhacode} ])"
        lhacode += 1
        couplings_data += f"\n\nGC_{couplings_counter} = Coupling(name = 'GC_{couplings_counter}',\n\tvalue = '-0.125 * complex(0,1) * aS * cG/(fa * cmath.pi)',\n\torder = {{'ALP': 1, 'QCD': 2}})"
        vertices_data += f"\n\nV_{vertices_counter} = Vertex(name = 'V_{vertices_counter}',\n\tparticles = [ P.g, P.g, P.alp ],\n\tcolor = [ 'Identity(1,2)' ],\n\tlorentz = [ L.VVS3 ],\n\tcouplings = {{(0,0): C.GC_{couplings_counter}}})"
        couplings_counter += 1
        vertices_counter += 1
        couplings_data += f"\n\nGC_{couplings_counter} = Coupling(name = 'GC_{couplings_counter}',\n\tvalue = '-0.25 * aS * cG * G/(fa * cmath.pi)',\n\torder = {{'ALP': 1, 'QCD': 3}})"
        vertices_data += f"\n\nV_{vertices_counter} = Vertex(name = 'V_{vertices_counter}',\n\tparticles = [ P.g, P.g, P.g, P.alp ],\n\tcolor = [ 'f(1,2,3)' ],\n\tlorentz = [ L.VVVS1 ],\n\tcouplings = {{(0,0): C.GC_{couplings_counter}}})"
        couplings_counter += 1
        vertices_counter += 1
        decays_data = decays_data.replace('# Decay_alp #', f"(P.g, P.g):'aS**2 * abs(cG)**2 * mALP**3 / (8. * fa**2 * cmath.pi**3)',\n\t\t# Decay_alp #")

    for i in range(3):
        for j in range(3):
            if alp_couplings['cdL'][i,j] != 0:
                cL = f'cdL{i+1}{j+1}'
                if almost_real(alp_couplings['cdL'][i,j]):
                    parameter_data += f"\n\n{cL} = Parameter(name = '{cL}',\n\tnature = 'external',\n\ttype = 'real',\n\tvalue = {np.real(alp_couplings['cdL'][i,j]):.6e},\n\ttexname = 'c_{{d_L}}^{{{i+1}{j+1}}}',\n\tlhablock = 'ALPINPUTS',\n\tlhacode = [ {lhacode} ])"
                    lhacode += 1
                else:
                    parameter_data += f"\n\n{cL}r = Parameter(name = '{cL}r',\n\tnature = 'external',\n\ttype = 'real',\n\tvalue = {np.real(alp_couplings['cdL'][i,j]):.6e},\n\ttexname = '\\\\text{{Re}}c_{{d_L}}^{{{i+1}{j+1}}}',\n\tlhablock = 'ALPINPUTS',\n\tlhacode = [ {lhacode} ])"
                    lhacode += 1
                    parameter_data += f"\n\n{cL}i = Parameter(name = '{cL}i',\n\tnature = 'external',\n\ttype = 'real',\n\tvalue = {np.imag(alp_couplings['cdL'][i,j]):.6e},\n\ttexname = '\\\\text{{Im}}c_{{d_L}}^{{{i+1}{j+1}}}',\n\tlhablock = 'ALPINPUTS',\n\tlhacode = [ {lhacode} ])"
                    lhacode += 1
                    parameter_data += f"\n\n{cL} = Parameter(name = '{cL}',\n\tnature = 'internal',\n\ttype = 'complex',\n\tvalue = '{cL}r + complex(0,1)*{cL}i',\n\ttexname = 'c_{{d_L}}^{{{i+1}{j+1}}}')"
            else:
                cL = None
            if alp_couplings['cdR'][i,j] != 0:
                cR = f'cdR{i+1}{j+1}'
                if almost_real(alp_couplings['cdR'][i,j]):
                    parameter_data += f"\n\n{cR} = Parameter(name = '{cR}',\n\tnature = 'external',\n\ttype = 'real',\n\tvalue = {np.real(alp_couplings['cdR'][i,j]):.6e},\n\ttexname = 'c_{{d_R}}^{{{i+1}{j+1}}}',\n\tlhablock = 'ALPINPUTS',\n\tlhacode = [ {lhacode} ])"
                    lhacode += 1
                else:
                    parameter_data += f"\n\n{cR}r = Parameter(name = '{cR}r',\n\tnature = 'external',\n\ttype = 'real',\n\tvalue = {np.real(alp_couplings['cdR'][i,j]):.6e},\n\ttexname = '\\\\text{{Re}}c_{{d_R}}^{{{i+1}{j+1}}}',\n\tlhablock = 'ALPINPUTS',\n\tlhacode = [ {lhacode} ])"
                    lhacode += 1
                    parameter_data += f"\n\n{cR}i = Parameter(name = '{cR}i',\n\tnature = 'external',\n\ttype = 'real',\n\tvalue = {np.imag(alp_couplings['cdR'][i,j]):.6e},\n\ttexname = '\\\\text{{Im}}c_{{d_R}}^{{{i+1}{j+1}}}',\n\tlhablock = 'ALPINPUTS',\n\tlhacode = [ {lhacode} ])"
                    lhacode += 1
                    parameter_data += f"\n\n{cR} = Parameter(name = '{cR}',\n\tnature = 'internal',\n\ttype = 'complex',\n\tvalue = '{cR}r + complex(0,1)*{cR}i',\n\ttexname = 'c_{{d_R}}^{{{i+1}{j+1}}}')"
            else:
                cR = None
            if cL is None and cR is None:
                continue
            elif cL is None and cR is not None:
                couplings_data += f"\n\nGC_{couplings_counter} = Coupling(name = 'GC_{couplings_counter}',\n\tvalue = '-{cR}/fa',\n\torder = {{'ALP': 1}})"
                vertices_data += f"\n\nV_{vertices_counter} = Vertex(name = 'V_{vertices_counter}',\n\tparticles = [ P.{dquarks[j]}__tilde__, P.{dquarks[i]}, P.alp ],\n\tcolor = [ 'Identity(1,2)' ],\n\tlorentz = [ L.FFS6 ],\n\tcouplings = {{(0,0): C.GC_{couplings_counter}}})"
                couplings_counter += 1
                vertices_counter += 1
            elif cL is not None and cR is None:
                couplings_data += f"\n\nGC_{couplings_counter} = Coupling(name = 'GC_{couplings_counter}',\n\tvalue = '-{cL}/fa',\n\torder = {{'ALP': 1}})"
                vertices_data += f"\n\nV_{vertices_counter} = Vertex(name = 'V_{vertices_counter}',\n\tparticles = [ P.{dquarks[j]}__tilde__, P.{dquarks[i]}, P.alp ],\n\tcolor = [ 'Identity(1,2)' ],\n\tlorentz = [ L.FFS5 ],\n\tcouplings = {{(0,0): C.GC_{couplings_counter}}})"
                couplings_counter += 1
                vertices_counter += 1
            else:
                couplings_data += f"\n\nGC_{couplings_counter} = Coupling(name = 'GC_{couplings_counter}',\n\tvalue = '-{cL}/fa',\n\torder = {{'ALP': 1}})"
                couplings_data += f"\n\nGC_{couplings_counter+1} = Coupling(name = 'GC_{couplings_counter+1}',\n\tvalue = '-{cR}/fa',\n\torder = {{'ALP': 1}})"
                vertices_data += f"\n\nV_{vertices_counter} = Vertex(name = 'V_{vertices_counter}',\n\tparticles = [ P.{dquarks[j]}__tilde__, P.{dquarks[i]}, P.alp ],\n\tcolor = [ 'Identity(1,2)' ],\n\tlorentz = [ L.FFS5, L.FFS6 ],\n\tcouplings = {{(0,0): C.GC_{couplings_counter}, (0,1): C.GC_{couplings_counter+1}}})"
                couplings_counter += 2
                vertices_counter += 1
            if i == j:
                decays_data = decays_data.replace('# Decay_alp #', f"(P.{dquarks[i]}, P.{dquarks[i]}__tilde__):'{a_ff(cL, cR, mdquarks[i], True)}',\n\t\t# Decay_alp #")
            else:
                decays_data = decays_data.replace('# Decay_alp #', f"(P.{dquarks[i]}, P.{dquarks[j]}__tilde__):'{a_f1f2(cL, cR, mdquarks[i], mdquarks[j], True)}',\n\t\t# Decay_alp #")
                decays_data = decays_data.replace(f'# Decay_{dquarks[j]} #', f"(P.{dquarks[i]},P.alp):'{f1_af2(cL, cR, mdquarks[j], mdquarks[i])}',\n\t\t# Decay_{dquarks[j]} #")

    

    for i in range(n_up):
        for j in range(n_up):
            if alp_couplings['cuL'][i,j] != 0:
                cL = f'cuL{i+1}{j+1}'
                if almost_real(alp_couplings['cuL'][i,j]):
                    parameter_data += f"\n\n{cL} = Parameter(name = '{cL}',\n\tnature = 'external',\n\ttype = 'real',\n\tvalue = {np.real(alp_couplings['cuL'][i,j]):.6e},\n\ttexname = 'c_{{u_L}}^{{{i+1}{j+1}}}',\n\tlhablock = 'ALPINPUTS',\n\tlhacode = [ {lhacode} ])"
                    lhacode += 1
                else:
                    parameter_data += f"\n\n{cL}r = Parameter(name = '{cL}r',\n\tnature = 'external',\n\ttype = 'real',\n\tvalue = {np.real(alp_couplings['cuL'][i,j]):.6e},\n\ttexname = '\\\\text{{Re}}c_{{u_L}}^{{{i+1}{j+1}}}',\n\tlhablock = 'ALPINPUTS',\n\tlhacode = [ {lhacode} ])"
                    lhacode += 1
                    parameter_data += f"\n\n{cL}i = Parameter(name = '{cL}i',\n\tnature = 'external',\n\ttype = 'real',\n\tvalue = {np.imag(alp_couplings['cuL'][i,j]):.6e},\n\ttexname = '\\\\text{{Im}}c_{{u_L}}^{{{i+1}{j+1}}}',\n\tlhablock = 'ALPINPUTS',\n\tlhacode = [ {lhacode} ])"
                    lhacode += 1
                    parameter_data += f"\n\n{cL} = Parameter(name = '{cL}',\n\tnature = 'internal',\n\ttype = 'complex',\n\tvalue = '{cL}r + complex(0,1)*{cL}i',\n\ttexname = 'c_{{u_L}}^{{{i+1}{j+1}}}')"
            else:
                cL = None
            if alp_couplings['cuR'][i,j] != 0:
                cR = f'cuR{i+1}{j+1}'
                if almost_real(alp_couplings['cuR'][i,j]):
                    parameter_data += f"\n\n{cR} = Parameter(name = '{cR}',\n\tnature = 'external',\n\ttype = 'real',\n\tvalue = {np.real(alp_couplings['cuR'][i,j]):.6e},\n\ttexname = 'c_{{u_R}}^{{{i+1}{j+1}}}',\n\tlhablock = 'ALPINPUTS',\n\tlhacode = [ {lhacode} ])"
                    lhacode += 1
                else:
                    parameter_data += f"\n\n{cR}r = Parameter(name = '{cR}r',\n\tnature = 'external',\n\ttype = 'real',\n\tvalue = {np.real(alp_couplings['cuR'][i,j]):.6e},\n\ttexname = '\\\\text{{Re}}c_{{u_R}}^{{{i+1}{j+1}}}',\n\tlhablock = 'ALPINPUTS',\n\tlhacode = [ {lhacode} ])"
                    lhacode += 1
                    parameter_data += f"\n\n{cR}i = Parameter(name = '{cR}i',\n\tnature = 'external',\n\ttype = 'real',\n\tvalue = {np.imag(alp_couplings['cuR'][i,j]):.6e},\n\ttexname = '\\\\text{{Im}}c_{{u_R}}^{{{i+1}{j+1}}}',\n\tlhablock = 'ALPINPUTS',\n\tlhacode = [ {lhacode} ])"
                    lhacode += 1
                    parameter_data += f"\n\n{cR} = Parameter(name = '{cR}',\n\tnature = 'internal',\n\ttype = 'complex',\n\tvalue = '{cR}r + complex(0,1)*{cR}i',\n\ttexname = 'c_{{u_R}}^{{{i+1}{j+1}}}')"
            else:
                cR = None
            if cL is None and cR is None:
                continue
            elif cL is None and cR is not None:
                couplings_data += f"\n\nGC_{couplings_counter} = Coupling(name = 'GC_{couplings_counter}',\n\tvalue = '-{cR}/fa',\n\torder = {{'ALP': 1}})"
                vertices_data += f"\n\nV_{vertices_counter} = Vertex(name = 'V_{vertices_counter}',\n\tparticles = [ P.{uquarks[j]}__tilde__, P.{uquarks[i]}, P.alp ],\n\tcolor = [ 'Identity(1,2)' ],\n\tlorentz = [ L.FFS6 ],\n\tcouplings = {{(0,0): C.GC_{couplings_counter}}})"
                couplings_counter += 1
                vertices_counter += 1
            elif cL is not None and cR is None:
                couplings_data += f"\n\nGC_{couplings_counter} = Coupling(name = 'GC_{couplings_counter}',\n\tvalue = '-{cL}/fa',\n\torder = {{'ALP': 1}})"
                vertices_data += f"\n\nV_{vertices_counter} = Vertex(name = 'V_{vertices_counter}',\n\tparticles = [ P.{uquarks[j]}__tilde__, P.{uquarks[i]}, P.alp ],\n\tcolor = [ 'Identity(1,2)' ],\n\tlorentz = [ L.FFS5 ],\n\tcouplings = {{(0,0): C.GC_{couplings_counter}}})"
                couplings_counter += 1
                vertices_counter += 1
            else:
                couplings_data += f"\n\nGC_{couplings_counter} = Coupling(name = 'GC_{couplings_counter}',\n\tvalue = '-{cL}/fa',\n\torder = {{'ALP': 1}})"
                couplings_data += f"\n\nGC_{couplings_counter+1} = Coupling(name = 'GC_{couplings_counter+1}',\n\tvalue = '-{cR}/fa',\n\torder = {{'ALP': 1}})"
                vertices_data += f"\n\nV_{vertices_counter} = Vertex(name = 'V_{vertices_counter}',\n\tparticles = [ P.{uquarks[j]}__tilde__, P.{uquarks[i]}, P.alp ],\n\tcolor = [ 'Identity(1,2)' ],\n\tlorentz = [ L.FFS5, L.FFS6 ],\n\tcouplings = {{(0,0): C.GC_{couplings_counter}, (0,1): C.GC_{couplings_counter+1}}})"
                couplings_counter += 2
                vertices_counter += 1
            if i == j:
                decays_data = decays_data.replace('# Decay_alp #', f"(P.{uquarks[i]}, P.{uquarks[i]}__tilde__):'{a_ff(cL, cR, muquarks[i], True)}',\n\t\t# Decay_alp #")
            else:
                decays_data = decays_data.replace('# Decay_alp #', f"(P.{uquarks[i]}, P.{uquarks[j]}__tilde__):'{a_f1f2(cL, cR, muquarks[i], muquarks[j], True)}',\n\t\t# Decay_alp #")
                decays_data = decays_data.replace(f'# Decay_{uquarks[j]} #', f"(P.{uquarks[i]},P.alp):'{f1_af2(cL, cR, muquarks[j], muquarks[i])}',\n\t\t# Decay_{uquarks[j]} #")

    for i in range(3):
        for j in range(3):
            if alp_couplings['ceL'][i,j] != 0:
                cL = f'ceL{i+1}{j+1}'
                if almost_real(alp_couplings['ceL'][i,j]):
                    parameter_data += f"\n\n{cL} = Parameter(name = '{cL}',\n\tnature = 'external',\n\ttype = 'real',\n\tvalue = {np.real(alp_couplings['ceL'][i,j]):.6e},\n\ttexname = 'c_{{e_L}}^{{{i+1}{j+1}}}',\n\tlhablock = 'ALPINPUTS',\n\tlhacode = [ {lhacode} ])"
                    lhacode += 1
                else:
                    parameter_data += f"\n\n{cL}r = Parameter(name = '{cL}r',\n\tnature = 'external',\n\ttype = 'real',\n\tvalue = {np.real(alp_couplings['ceL'][i,j]):.6e},\n\ttexname = '\\\\text{{Re}}c_{{e_L}}^{{{i+1}{j+1}}}',\n\tlhablock = 'ALPINPUTS',\n\tlhacode = [ {lhacode} ])"
                    lhacode += 1
                    parameter_data += f"\n\n{cL}i = Parameter(name = '{cL}i',\n\tnature = 'external',\n\ttype = 'real',\n\tvalue = {np.imag(alp_couplings['ceL'][i,j]):.6e},\n\ttexname = '\\\\text{{Im}}c_{{e_L}}^{{{i+1}{j+1}}}',\n\tlhablock = 'ALPINPUTS',\n\tlhacode = [ {lhacode} ])"
                    lhacode += 1
                    parameter_data += f"\n\n{cL} = Parameter(name = '{cL}',\n\tnature = 'internal',\n\ttype = 'complex',\n\tvalue = '{cL}r + complex(0,1)*{cL}i',\n\ttexname = 'c_{{e_L}}^{{{i+1}{j+1}}}')"
            else:
                cL = None
            if alp_couplings['ceR'][i,j] != 0:
                cR = f'ceR{i+1}{j+1}'
                if almost_real(alp_couplings['ceR'][i,j]):
                    parameter_data += f"\n\n{cR} = Parameter(name = '{cR}',\n\tnature = 'external',\n\ttype = 'real',\n\tvalue = {np.real(alp_couplings['ceR'][i,j]):.6e},\n\ttexname = 'c_{{e_R}}^{{{i+1}{j+1}}}',\n\tlhablock = 'ALPINPUTS',\n\tlhacode = [ {lhacode} ])"
                    lhacode += 1
                else:
                    parameter_data += f"\n\n{cR}r = Parameter(name = '{cR}r',\n\tnature = 'external',\n\ttype = 'real',\n\tvalue = {np.real(alp_couplings['ceR'][i,j]):.6e},\n\ttexname = '\\\\text{{Re}}c_{{e_R}}^{{{i+1}{j+1}}}',\n\tlhablock = 'ALPINPUTS',\n\tlhacode = [ {lhacode} ])"
                    lhacode += 1
                    parameter_data += f"\n\n{cR}i = Parameter(name = '{cR}i',\n\tnature = 'external',\n\ttype = 'real',\n\tvalue = {np.imag(alp_couplings['ceR'][i,j]):.6e},\n\ttexname = '\\\\text{{Im}}c_{{e_R}}^{{{i+1}{j+1}}}',\n\tlhablock = 'ALPINPUTS',\n\tlhacode = [ {lhacode} ])"
                    lhacode += 1
                    parameter_data += f"\n\n{cR} = Parameter(name = '{cR}',\n\tnature = 'internal',\n\ttype = 'complex',\n\tvalue = '{cR}r + complex(0,1)*{cR}i',\n\ttexname = 'c_{{e_R}}^{{{i+1}{j+1}}}')"
            else:
                cR = None
            if cL is None and cR is None:
                continue
            elif cL is None and cR is not None:
                couplings_data += f"\n\nGC_{couplings_counter} = Coupling(name = 'GC_{couplings_counter}',\n\tvalue = '-{cR}/fa',\n\torder = {{'ALP': 1}})"
                vertices_data += f"\n\nV_{vertices_counter} = Vertex(name = 'V_{vertices_counter}',\n\tparticles = [ P.{leptons[j]}__plus__, P.{leptons[i]}__minus__, P.alp ],\n\tcolor = [ '1' ],\n\tlorentz = [ L.FFS6 ],\n\tcouplings = {{(0,0): C.GC_{couplings_counter}}})"
                couplings_counter += 1
                vertices_counter += 1
            elif cL is not None and cR is None:
                couplings_data += f"\n\nGC_{couplings_counter} = Coupling(name = 'GC_{couplings_counter}',\n\tvalue = '-{cL}/fa',\n\torder = {{'ALP': 1}})"
                vertices_data += f"\n\nV_{vertices_counter} = Vertex(name = 'V_{vertices_counter}',\n\tparticles = [ P.{leptons[j]}__plus__, P.{leptons[i]}__minus__, P.alp ],\n\tcolor = [ '1' ],\n\tlorentz = [ L.FFS5 ],\n\tcouplings = {{(0,0): C.GC_{couplings_counter}}})"
                couplings_counter += 1
                vertices_counter += 1
            else:
                couplings_data += f"\n\nGC_{couplings_counter} = Coupling(name = 'GC_{couplings_counter}',\n\tvalue = '-{cL}/fa',\n\torder = {{'ALP': 1}})"
                couplings_data += f"\n\nGC_{couplings_counter+1} = Coupling(name = 'GC_{couplings_counter+1}',\n\tvalue = '-{cR}/fa',\n\torder = {{'ALP': 1}})"
                vertices_data += f"\n\nV_{vertices_counter} = Vertex(name = 'V_{vertices_counter}',\n\tparticles = [ P.{leptons[j]}__plus__, P.{leptons[i]}__minus__, P.alp ],\n\tcolor = [ '1' ],\n\tlorentz = [ L.FFS5, L.FFS6 ],\n\tcouplings = {{(0,0): C.GC_{couplings_counter}, (0,1): C.GC_{couplings_counter+1}}})"
                couplings_counter += 2
                vertices_counter += 1
            if i == j:
                decays_data = decays_data.replace('# Decay_alp #', f"(P.{leptons[i]}__minus__, P.{leptons[i]}__plus__):'{a_ff(cL, cR, mleptons[i], False)}',\n\t\t# Decay_alp #")
            else:
                decays_data = decays_data.replace('# Decay_alp #', f"(P.{leptons[i]}__minus__, P.{leptons[j]}__plus__):'{a_f1f2(cL, cR, mleptons[i], mleptons[j], False)}',\n\t\t# Decay_alp #")
                decays_data = decays_data.replace(f'# Decay_{leptons[j]}__minus__ #', f"(P.{leptons[i]}__minus__,P.alp):'{f1_af2(cL, cR, mleptons[j], mleptons[i])}',\n\t\t# Decay_{leptons[j]} #")


    with open(os.path.join(ufo_dir, 'parameters.py'), 'w') as f:
        f.write(parameter_data.replace('\t', '    '))
    with open(os.path.join(ufo_dir, 'vertices.py'), 'w') as f:
        f.write(vertices_data.replace('\t', '    '))
    with open(os.path.join(ufo_dir, 'couplings.py'), 'w') as f:
        f.write(couplings_data.replace('\t', '    '))
    with open(os.path.join(ufo_dir, 'decays.py'), 'w') as f:
        f.write(decays_data.replace('\t', '    '))