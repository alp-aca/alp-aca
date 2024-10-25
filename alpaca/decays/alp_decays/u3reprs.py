import numpy as np

###### Pseudoscalar mesons ###################
pi0 = np.diag([1, -1, 0])/2

# Change with the actual value of the eta-eta' mixing angle
eta = np.diag([1, 1, -1])/np.sqrt(6)

etap = np.diag([1, 1, 2])/2/np.sqrt(3)

sigma = np.diag([np.sqrt(5), np.sqrt(5), 1])/np.sqrt(22)

f0 = np.diag([1, 1, -2*np.sqrt(2)])/2/np.sqrt(5)

a0 = np.diag([1/2, -1/2, 0])

f2 = np.diag([1/2, 1/2, 0])

###### Vector mesons ###############

rho0 = np.diag([1/2, -1/2, 0])

omega = np.diag([1/2, 1/2, 0])

phi = np.diag([0, 0, 1])/np.sqrt(2)