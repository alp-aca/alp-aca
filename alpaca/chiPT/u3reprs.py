import numpy as np
from ..constants import theta_eta_etap

###### Pseudoscalar mesons ###################
pi0 = np.diag([1, -1, 0])/2

eta0 = np.diag([1, 1, 1])/np.sqrt(6)

eta8 = np.diag([1/2, 1/2, -1])/np.sqrt(3)
# Change with the actual value of the eta-eta' mixing angle
#eta = np.diag([1, 1, -1])/np.sqrt(6)

#etap = np.diag([1, 1, 2])/2/np.sqrt(3)
eta = np.cos(theta_eta_etap) * eta8 - np.sin(theta_eta_etap) * eta0

etap = np.sin(theta_eta_etap) * eta8 + np.cos(theta_eta_etap) * eta0

K0 = np.array([[0,0,0], [0, 0, np.sqrt(2)], [0, 0, 0]])/2

K0bar = np.array([[0,0,0], [0, 0, 0], [0, np.sqrt(2), 0]])/2

sigma = np.diag([np.sqrt(5), np.sqrt(5), 1])/np.sqrt(22)

f0 = np.diag([1, 1, -2*np.sqrt(2)])/2/np.sqrt(5)

a0 = np.diag([1/2, -1/2, 0])

f2 = np.diag([1/2, 1/2, 0])

###### Vector mesons ###############

rho0 = np.diag([1/2, -1/2, 0])

omega = np.diag([1/2, 1/2, 0])

phi = np.diag([0, 0, 1])/np.sqrt(2)