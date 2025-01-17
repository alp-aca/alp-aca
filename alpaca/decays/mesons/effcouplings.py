from ...rge import ALPcouplings
from ...common import B0disc_equalmass, ckm_xi
from ...constants import GF, mu, md, ms, mc, mb
import numpy as np

def effcouplings_cq1q2_W(couplings: ALPcouplings, pa2: float, q1: str, q2: str) -> complex:
    if couplings.scale > couplings.ew_scale:
        raise NotImplementedError(f"The effective couplings c_{q1}{q2} are implemented only below the EW scale.")
    couplings = couplings.translate('kF_below')
    mq = {'u': mu, 'd': md, 's': ms, 'c': mc, 'b': mb}
    ceff = 0
    if q1 == q2:
        return ceff
    if q1 in ['u', 'c'] and q2 in ['u', 'c']:
        gen = {'u': 0, 'c': 1}
        ceff = couplings['kU'][gen[q1], gen[q2]]
        for iq, qloop in enumerate(['d', 's', 'b']):
            cqloop = couplings['kD'][iq, iq] - couplings['kd'][iq, iq]
            ceff += GF/np.sqrt(2)/np.pi**2*ckm_xi(qloop, q1+q2)*cqloop * mq[qloop]**2 * (1 + B0disc_equalmass(pa2, mq[qloop]) + np.log(couplings.scale**2/mq[qloop]**2))
    elif q1 in ['d', 's', 'b'] and q2 in ['d', 's', 'b']:
        gen = {'d': 0, 's': 1, 'b': 2}
        ceff = couplings['kD'][gen[q1], gen[q2]]
        for iq, qloop in enumerate(['u', 'c']):
            cqloop = couplings['kU'][iq, iq] - couplings['ku'][iq, iq]
            ceff += GF/np.sqrt(2)/np.pi**2*ckm_xi(qloop, q1+q2) * cqloop * mq[qloop]**2 * (1 + B0disc_equalmass(pa2, mq[qloop]) + np.log(couplings.scale**2/mq[qloop]**2))
    return ceff