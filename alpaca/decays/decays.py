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
    if initial == 'B0':
        match products:
            case [('K*0' | 'K0*' | 'K*'), ('a' | 'ALP')]:
                decayrate = invisible.B0toKsta
            case _:
                raise KeyError(f"Wrong decay {transition}")
    if initial == 'K+':
        match products:
            case [('pi' | 'pi+'), ('a' | 'ALP')]:
                decayrate = invisible.Ktopia
            case _:
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
    if initial == 'B0':
        match products:
            case [('K*0' | 'K0*' | 'K*'), ('a' | 'ALP')]:
                decayrate = lambda ma, c, fa, **kwargs: invisible.B0toKsta(ma, c, fa, **kwargs)/GammaB0
            case _:
                raise KeyError(f"Wrong decay {transition}")
    if initial == 'K+':
        match products:
            case [('pi' | 'pi+'), ('a' | 'ALP')]:
                decayrate = lambda ma, c, fa, **kwargs: invisible.Ktopia(ma, c, fa, **kwargs)/GammaK
            case _:
                raise KeyError(f"Wrong decay {transition}")
    return decayrate(ma, couplings, fa, **kwargs)