'''Parameters of the phenomenological Lagrangians describing scalar, vector and tensor mesons'''
import numpy as np
from ..biblio.biblio import Constant
from ..constants import theta_eta_etap
from ..classes import LazyFloat

gTf2 = Constant(13.1, 'Cheng:2021kjg')

CoefA = Constant(2.87, 'Fariborz:1999gr')
CoefB = Constant(-2.34, 'Fariborz:1999gr')
gS_sigmaetaetapr = Constant(1.71787, 'Fariborz:1999gr')
gS_f0etaetapr = Constant(9.54014, 'Fariborz:1999gr')
theta_S = Constant(-0.366519, 'Fariborz:1999gr')
CoefC = LazyFloat(lambda: (1/np.cos(2*theta_eta_etap)*((-(np.sqrt(2)*gS_f0etaetapr) + 2*gS_sigmaetaetapr)*np.cos(theta_S) + 2*CoefA*np.cos(2*theta_eta_etap) + 2*gS_f0etaetapr*np.sin(theta_S) + np.sqrt(2)*gS_sigmaetaetapr*np.sin(theta_S) + np.sqrt(2)*CoefA*np.sin(2*theta_eta_etap)))/3)
CoefD = LazyFloat(lambda: -1/72*(1/np.sin(theta_eta_etap)*1/np.cos(theta_eta_etap)*1/np.cos(2*theta_eta_etap)*(4*CoefA*np.cos(theta_S) - 12*CoefA*np.cos(theta_S - 4*theta_eta_etap) - 12*gS_sigmaetaetapr*np.cos(2*theta_eta_etap) - 8*np.sqrt(2)*gS_f0etaetapr*np.cos(2*(theta_S + theta_eta_etap)) + 
    4*gS_sigmaetaetapr*np.cos(2*(theta_S + theta_eta_etap)) + 8*CoefA*np.cos(theta_S + 4*theta_eta_etap) - 4*np.sqrt(2)*CoefA*np.sin(theta_S) - 3*np.sqrt(2)*CoefA*np.sin(theta_S - 4*theta_eta_etap) - 12*gS_f0etaetapr*np.sin(2*theta_eta_etap) + 
    4*gS_f0etaetapr*np.sin(2*(theta_S + theta_eta_etap)) + 8*np.sqrt(2)*gS_sigmaetaetapr*np.sin(2*(theta_S + theta_eta_etap)) + 7*np.sqrt(2)*CoefA*np.sin(theta_S + 4*theta_eta_etap)))/(np.sqrt(2)*np.cos(theta_S) - 2*np.sin(theta_S)))