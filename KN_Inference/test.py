'''
'''

import numpy as np
from utils import *
from model import *
from astropy.cosmology import Planck18
import matplotlib.pyplot as plt


'''
=========================================
Use of KilonovaGRB model in model.py
=========================================
'''
z=0.01
dL = Planck18.luminosity_distance(z).value*MPC_CGS
theta = 0.2
theta_core = np.deg2rad(5)
print(dL)
Z = {'jetType':     grb.jet.Gaussian,     # Gaussian jet
            'specType':    0,                  # Basic Synchrotron Emission Spectrum

            'thetaObs':    theta,   # Viewing angle in radians
            'E0':          10**(52.73), # Isotropic-equivalent energy in erg
            'thetaCore':   theta_core,    # Half-opening angle in radians
            'thetaWing':   2*theta_core,    # Outer truncation angle
            'n0':          10**(-2.8),    # circumburst density in cm^{-3}
            'p':           2.155,    # electron energy distribution index
            'epsilon_e':   10**(-0.51),    # epsilon_e
            'epsilon_B':   10**(-2.2),   # epsilon_B
            'xi_N':        1,    # Fraction of electrons accelerated
            'd_L':         dL, # Luminosity distance in cm
            'z':           z}   # redshift
phase = np.linspace(0.1,7,100)
svd_path = '/home/Aujust/data/Kilonova/GPR/NN/'
model_name = 'Bu2022Ye'
model_bns = KilonovaGRB(model_type='tensorflow',model_dir=svd_path,model_name=model_name,Z=Z)

band = 'ztfg'
p =np.array([np.log10(0.005),0.2,0.2,np.log10(0.05),0.05,np.rad2deg(theta)])
lc, lc_kn, lc_grb = model_bns.cal_lc(param_list=p,times=phase,bands=[band],dL=dL)
lc, lc_kn, lc_grb = lc[band], lc_kn[band], lc_grb[band]

plt.figure(dpi=100)
plt.plot(phase,lc,color='k',label='Hybrid')
plt.plot(phase,lc_kn,color='orange',label='KN')
plt.plot(phase,lc_grb,color='darkgreen',label='Afterglow')
plt.ylim([30,15.1])
plt.xlabel('Time')
plt.ylabel('Mag')
plt.legend()
plt.savefig('/home/Aujust/data/Kilonova/KN_Inference/KN_Inference/test.jpg',dpi=300)



print()