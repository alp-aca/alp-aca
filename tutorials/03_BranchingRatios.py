import numpy as np 
import matplotlib.pyplot as plt
# from alpaca.decays.alp_decays.fermion_decays import decay_width_electron, decay_width_muon, decay_width_tau, decay_width_charm, decay_width_bottom
# from alpaca.decays.alp_decays.hadronic_decays_def import decay_width_3pi000, decay_width_3pi0pm, decay_width_etapipi00, decay_width_etapipipm, decay_width_gammapipi, decay_width_gluongluon
# from alpaca.decays.alp_decays.gaugebosons import decay_width_2gamma
from alpaca.rge import ALPcouplings
from alpaca.decays.alp_decays.branching_ratios import BRsalp
import time

# Empieza a medir el tiempo
start_time = time.time()

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "sans-serif"
})
plt.rcParams.update({'font.size': 14})

#ma=np.logspace(-1,1,1000)
ma=np.logspace(-1,1,1000)
fa=1000

cphoton=1.0
clept=1.0
cu=1.0
cd=1.0 
cgluons=1,0
#low_scale=2.0
# couplings=ALPcouplings({'cgamma': cphoton, 'cuA': cfermi, 'cdA': cfermi, 'ceA': cfermi}, scale=, basis='VA_below')


BR_2gamma=[]
BRelec=[]
BRmu=[]
BRtau=[]
BRcharm=[]
BRbottom=[]
BR3pis=[]
BRetapipi=[]
BRgammapipi=[]
BRgluongluon=[]



for m in ma:
    couplings = ALPcouplings({'cgamma': cphoton, 'cuA': cu, 'cdA': cd, 'ceA': clept}, scale=m, basis='VA_below')
    BRs=BRsalp(m, couplings, fa, cores=6)
    BR_2gamma.append(BRs['2photons'])
    BRelec.append(BRs['e'])
    BRmu.append(BRs['mu'])
    BRtau.append(BRs['tau'])
    BRcharm.append(BRs['charm'])
    BRbottom.append(BRs['bottom'])
    BR3pis.append(BRs['3pis'])
    BRetapipi.append(BRs['etapipi'])
    BRgammapipi.append(BRs['gammapipi'])
    BRgluongluon.append(BRs['gluongluon'])



fig, ax = plt.subplots(figsize=(10, 6)) 

ax.plot(ma, BR_2gamma, label=r"$\textrm{BR}\left(a\rightarrow \gamma\gamma\right)$")
ax.plot(ma, BRelec, label=r"$\textrm{BR}\left(a\rightarrow e^+e^-\right)$")
ax.plot(ma, BRmu, label=r"$\textrm{BR}\left(a\rightarrow \mu^+\mu^-\right)$")
ax.plot(ma, BRtau, label=r"$\textrm{BR}\left(a\rightarrow \tau^+\tau^-\right)$")
ax.plot(ma, BRcharm, label=r"$\textrm{BR}\left(a\rightarrow c\bar{c}\right)$")
ax.plot(ma, BRbottom, label=r"$\textrm{BR}\left(a\rightarrow b\bar{b}\right)$")
ax.plot(ma, BR3pis, label=r"$\textrm{BR}\left(a\rightarrow 3\pi^0\right)$")
ax.plot(ma, BRetapipi, label=r"$\textrm{BR}\left(a\rightarrow \eta\pi\pi\right)$")
ax.plot(ma, BRgammapipi, label=r"$\textrm{BR}\left(a\rightarrow \gamma\pi\pi\right)$")
ax.plot(ma, BRgluongluon, label=r"$\textrm{BR}\left(a\rightarrow gg\right)$")

ax.set_ylim(1e-8,2)
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel(r"$m_a \, \left[\textrm{GeV}\right]$")
ax.set_ylabel(r"$\textrm{BR} \left(a\to \textrm{SM}\right)$")
ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), borderaxespad=0.1)

plt.tight_layout()  # Adjust the layout to make room for the legend

end_time = time.time()
print(end_time - start_time)
plt.show()
