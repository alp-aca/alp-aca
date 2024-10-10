from .mesons import invisible
from ..rge import ALPcouplings

def decay(transition: str, ma: float, couplings: ALPcouplings, fa: float, **kwargs) -> float:
    initial, final = transition.split('->')
    initial = initial.strip()
    products = [f.strip() for f in final.split()]
    if initial == 'B':
        match products:
            case ['K', ('a' | 'ALP')]:
                decayrate = invisible.BtoKa
            case _:
                raise KeyError(f"Wrong decay {transition}")
    elif initial == 'B0':
        match products:
            case [('K*0' | 'K0*' | 'K*'), ('a' | 'ALP')]:
                decayrate = invisible.B0toKsta
            case _:
                raise KeyError(f"Wrong decay {transition}")
    elif initial == 'K+':
        match products:
            case [('pi' | 'pi+'), ('a' | 'ALP')]:
                decayrate = invisible.Ktopia
            case _:
                raise KeyError(f"Wrong decay {transition}")
    else:
                raise KeyError(f"Wrong decay {transition}")
    return decayrate(ma, couplings, fa, **kwargs)
    
def BR(transition: str, ma: float, couplings: ALPcouplings, fa: float, **kwargs) -> float:
    from ..constants import GammaB, GammaB0, GammaK
    initial, final = transition.split('->')
    initial = initial.strip()
    products = [f.strip() for f in final.split()]
    if initial == 'B':
        match products:
            case ['K', ('a' | 'ALP')]:
                decayrate = lambda ma, c, fa, **kwargs: invisible.BtoKa(ma, c, fa, **kwargs)/GammaB
            case _:
                raise KeyError(f"Wrong decay {transition}")
    elif initial == 'B0':
        match products:
            case [('K*0' | 'K0*' | 'K*'), ('a' | 'ALP')]:
                decayrate = lambda ma, c, fa, **kwargs: invisible.B0toKsta(ma, c, fa, **kwargs)/GammaB0
            case _:
                raise KeyError(f"Wrong decay {transition}")
    elif initial == 'K+':
        match products:
            case [('pi' | 'pi+'), ('a' | 'ALP')]:
                decayrate = lambda ma, c, fa, **kwargs: invisible.Ktopia(ma, c, fa, **kwargs)/GammaK
            case _:
                raise KeyError(f"Wrong decay {transition}")
    elif initial == 'J/psi':
        match products:
            case [('A' | 'gamma' | 'photon'), ('a' | 'ALP')] | [('a' | 'ALP'), ('A' | 'gamma' | 'photon')]:
                from ..constants import mJpsi, BeeJpsi
                decayrate = lambda ma, couplings, fa, **kwargs: invisible.BR_Vagamma(ma, couplings, mJpsi, BeeJpsi, 'c', fa, **kwargs)
            case _:
                raise KeyError(f"Wrong decay {transition}")
    elif initial in ['Upsilon(1S)', 'Y(1S)']:
        match products:
            case [('A' | 'gamma' | 'photon'), ('a' | 'ALP')] | [('a' | 'ALP'), ('A' | 'gamma' | 'photon')]:
                from ..constants import mUpsilon1S, BeeUpsilon1S
                decayrate = lambda ma, couplings, fa, **kwargs: invisible.BR_Vagamma(ma, couplings, mUpsilon1S, BeeUpsilon1S, 'b', fa, **kwargs)
            case _:
                raise KeyError(f"Wrong decay {transition}")
    elif initial in ['Upsilon(3S)', 'Y(3S)']:
        match products:
            case [('A' | 'gamma' | 'photon'), ('a' | 'ALP')] | [('a' | 'ALP'), ('A' | 'gamma' | 'photon')]:
                from ..constants import mUpsilon3S
                decayrate = lambda ma, couplings, fa, **kwargs: invisible.Mixed_QuarkoniaSearches(ma, couplings, mUpsilon3S, 'b', fa, **kwargs) 
            case _:
                raise KeyError(f"Wrong decay {transition}")
    else:
        raise KeyError(f"Wrong decay {transition}")
    return decayrate(ma, couplings, fa, **kwargs)

def cross_section(transition: str, ma: float, couplings: ALPcouplings, s:float, fa: float, **kwargs) -> float:
    initial, final = transition.split('->')
    initial = [f.strip() for f in initial.split()]
    products = [f.strip() for f in final.split()]
    match initial:
        case ['e', 'e'] | ['e+', 'e-'] | ['e-', 'e+'] | ['electron' | 'positron'] | ['positron' | 'electron']:
            match products:
                case [('A' | 'gamma' | 'photon'), ('a' | 'ALP')] | [('a' | 'ALP'), ('A' | 'gamma' | 'photon')]:
                    sigma = invisible.sigmaNR
        case _:
            raise KeyError(f"Wrong decay {transition}")
    return sigma(ma, couplings, s, fa, **kwargs)