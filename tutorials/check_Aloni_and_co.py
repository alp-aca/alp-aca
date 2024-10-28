import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.lines import Line2D
from scipy.interpolate import interp1d

from alpaca.decays.alp_decays import gaugebosons
from alpaca.decays.alp_decays import chiral
from alpaca.decays.alp_decays.hadronic_decays_def import decay_width_3pi000 as ato3pi000, decay_width_3pi0pm as ato3pi0pm, decay_width_etapipi00 as atoetapipi00, decay_width_etapipipm as atoetapipipm, decay_width_gammapipi as atogammapipi
from alpaca import ALPcouplings
from alpaca.constants import fpi, metap, mpi0, mpi_pm, meta
from alpaca.common import alpha_em, alpha_s

import numpy as np


plt.rcParams.update({
    "text.usetex": True,
    "font.family": "sans-serif"
})
plt.rcParams.update({'font.size': 14})

path_data = 'ficheros_datos/'

# Read the CSV files
a3pi = pd.read_csv(path_data + 'a3pi.csv')
aetapipi = pd.read_csv(path_data + 'aetapipi.csv')
apipigamma = pd.read_csv(path_data + 'apipigamma.csv')
agammagamma = pd.read_csv(path_data + 'agammagamma.csv')

# Sort the data by the first column
a3pi_sorted = a3pi.sort_values(by=a3pi.columns[0])
aetapipi_sorted = aetapipi.sort_values(by=aetapipi.columns[0])
apipigamma_sorted = apipigamma.sort_values(by=apipigamma.columns[0])
agammagamma_sorted = agammagamma.sort_values(by=agammagamma.columns[0])

# Create interpolation functions
a3pi_interp = interp1d(a3pi_sorted.iloc[:, 0], a3pi_sorted.iloc[:, 1], kind='linear', fill_value="extrapolate")
aetapipi_interp = interp1d(aetapipi_sorted.iloc[:, 0], aetapipi_sorted.iloc[:, 1], kind='linear', fill_value="extrapolate")
apipigamma_interp = interp1d(apipigamma_sorted.iloc[:, 0], apipigamma_sorted.iloc[:, 1], kind='linear', fill_value="extrapolate")
agammagamma_interp = interp1d(agammagamma_sorted.iloc[:, 0], agammagamma_sorted.iloc[:, 1], kind='linear', fill_value="extrapolate")


fig, (ax1, ax2)= plt.subplots(2, 1, sharex=True, gridspec_kw={'height_ratios': [3, 1]})

# a3pi_sorted.plot(x=0, y=1, ax=ax1, label='a3pi', color='goldenrod')
# aetapipi_sorted.plot(x=0, y=1, ax=ax1, label='aetapipi', color='red')
# apipigamma_sorted.plot(x=0, y=1, ax=ax1, label='apipigamma', color='orange')
# agammagamma_sorted.plot(x=0, y=1, ax=ax1, label='agammagamma', color='limegreen')

fa = 1000/32/np.pi**2
maalp2gamma=agammagamma_sorted.iloc[:, 0]
ma3pi=a3pi_sorted.iloc[:, 0]
maetapipi=aetapipi_sorted.iloc[:, 0]
magammapipi=apipigamma_sorted.iloc[:, 0]

ours_alp2gamma = [1e9*gaugebosons.decay_width_2gamma(ma, ALPcouplings({'cg': 1.0}, ma, 'VA_below'), fa) for ma in maalp2gamma]
ours_3pi = [1e9*ato3pi000(ma, ALPcouplings({'cg': 1.0}, ma, 'VA_below'), fa) + 1e9*ato3pi0pm(ma, ALPcouplings({'cg': 1.0}, ma, 'VA_below'), fa) for ma in ma3pi]
ours_etapipi = [1e9*atoetapipi00(ma, ALPcouplings({'cg': 1.0}, ma, 'VA_below'), fa) + 1e9*atoetapipipm(ma, ALPcouplings({'cg': 1.0}, ma, 'VA_below'), fa) for ma in maetapipi]
ours_gammapipi = [1e9*atogammapipi(ma, ALPcouplings({'cg': 1.0}, ma, 'VA_below'), fa) for ma in magammapipi]

# aloni_alp2gamma = [a3pi_interp(ma) for ma in maalp2gamma]
# aloni_3pi = [aetapipi_interp(ma) for ma in ma3pi]
# aloni_etapipi = [a3pi_interp(ma) for ma in maetapipi]
# aloni_gammapipi = [apipigamma_interp(ma) for ma in magammapipi]
print(meta, metap)
aloni_alp2gamma = agammagamma_sorted.iloc[:, 1]
aloni_3pi = a3pi_sorted.iloc[:, 1]
aloni_etapipi = aetapipi_sorted.iloc[:, 1]
aloni_gammapipi = apipigamma_sorted.iloc[:, 1]
dif_alp2gamma = [abs((ours_alp2gamma[i] - aloni_alp2gamma[i])) for i in range(len(maalp2gamma))]
dif_3pi = [abs((ours_3pi[i] - aloni_3pi[i])) for i in range(len(ma3pi))]
dif_etapipi = [abs(ours_etapipi[i] - aloni_etapipi[i]) for i in range(len(maetapipi))]
dif_gammapipi = [abs((ours_gammapipi[i] - aloni_gammapipi[i])) for i in range(len(magammapipi))]

ax1.plot(maalp2gamma, ours_alp2gamma, color='limegreen', lw=1)
ax1.plot(maalp2gamma, aloni_alp2gamma, color='limegreen', lw=1, linestyle='--')

ax1.plot(ma3pi, ours_3pi, color='goldenrod', lw=1)
ax1.plot(ma3pi, aloni_3pi, color='goldenrod', lw=1, linestyle='--')

ax1.plot(maetapipi, ours_etapipi, color='red', lw=1)
ax1.plot(maetapipi, aloni_etapipi, color='red', lw=1, linestyle='--')

ax1.plot(magammapipi, ours_gammapipi, color='orange', lw=1)
ax1.plot(magammapipi, aloni_gammapipi, color='orange', lw=1, linestyle='--')

ax2.plot(maalp2gamma, dif_alp2gamma, color='limegreen', lw=1)
ax2.plot(ma3pi, dif_3pi, color='goldenrod', lw=1)
ax2.plot(maetapipi, dif_etapipi, color='red', lw=1)
ax2.plot(magammapipi, dif_gammapipi, color='orange', lw=1)

ax1.set_ylabel(r'$\Gamma \, \left[\textrm{eV}\right]$')
ax2.set_ylabel(r'$|\Delta \Gamma| \, \left[\textrm{eV}\right]$')
ax2.set_xlabel(r'$m_{a} \, \left[\textrm{GeV}\right]$')
ax1.set_yscale('log')
ax1.set_xscale('linear')
ax2.set_yscale('log')
custom_lines = [
    Line2D([0], [0], color='black', lw=1, label='ours'),
    Line2D([0], [0], color='black', lw=1, linestyle='--', label=r'aloni \textit{et al}')
]

# Add the custom legend to the plot
ax1.legend(handles=custom_lines, loc=2)


plt.show()
