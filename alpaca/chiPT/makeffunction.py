# NOTE: This script prepares the interpolation of the Function F(ma) (eq. 17 in 1811.03474) between 1.4GeV and 2.1GeV
# It saves the result as a pickled function, so it is not necessary to make the interpolation each time

import pandas as pd
from scipy import interpolate
import pickle
import os
import numpy as np

path = os.path.dirname(__file__)


dataAloni = pd.read_csv(os.path.join(path, 'ffunction_Aloni.csv'))
# The file ffunction_Aloni.csv contains a digitized version of Fig. 2,
# but it is a bit noisy...

t, c, k = interpolate.splrep(dataAloni['ma_GeV'], dataAloni['F'], s=2)
# splrep makes a smooth interpolation, that doesn't pass through all the data points. The parameter s controls the smoothness

np.savez(os.path.join(path, 'ffunction.npz'), t=t, c=c, k=[k])