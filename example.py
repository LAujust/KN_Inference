'''
@copyright
aujust@mail.ustc.edu.cn
'''

import KN_Inference as knif
from KN_Inference import model
import numpy as np
import matplotlib.pyplot as plt
from astropy.cosmology import Planck18
import astropy.units as u
import afterglowpy as grb

print('Import!')


'''
=========================================
Use of KilonovaGRB model in model.py
=========================================
'''

MPC_CGS = u.Mpc.cgs.scale
z=0.01
dL = Planck18.luminosity_distance(z).value*MPC_CGS
theta = 0.2
theta_core = 0.08
print(dL)

#param from doi:10.1093/mnras/stz2248
Z = {'jetType':     grb.jet.Gaussian,     # Gaussian jet
            'specType':    0,                  # Basic Synchrotron Emission Spectrum

            'thetaObs':    theta,   # Viewing angle in radians
            'E0':          10**(54.), # Isotropic-equivalent energy in erg
            'thetaCore':   theta_core,    # Half-opening angle in radians
            'thetaWing':   0.77,    # Outer truncation angle
            'n0':          10**(-2.83),    # circumburst density in cm^{-3}
            'p':           2.2,    # electron energy distribution index
            'epsilon_e':   10**(-0.85),    # epsilon_e
            'epsilon_B':   10**(-2.18),   # epsilon_B
            'xi_N':        1,    # Fraction of electrons accelerated
            'd_L':         dL, # Luminosity distance in cm
            'z':           z}   # redshift
phase = np.linspace(0.1,20,500)
svd_path = '/home/Aujust/data/Kilonova/GPR/NN/'
model_name = 'Bu2022Ye'
model_bns = knif.model.KilonovaGRB(model_type='tensorflow',model_dir=svd_path,model_name=model_name,Z=Z)

band = 'ztfi'
p =np.array([np.log10(0.005),0.2,0.2,np.log10(0.05),0.05,np.rad2deg(theta)])
lc, lc_kn, lc_grb = model_bns.cal_lc(param_list=p,times=phase,bands=[band],dL=dL)
lc, lc_kn, lc_grb = lc[band], lc_kn[band], lc_grb[band]

plt.figure(dpi=100)

plt.plot(phase,lc_kn,color='orange',label='KN')
plt.plot(phase,lc_grb,color='darkgreen',label='Afterglow')
plt.plot(phase,lc,color='k',label='Hybrid')
plt.ylim([30,15.1])
plt.xlim([0.1,10])
plt.xlabel('Time')
#plt.xscale('log')
plt.ylabel('Mag')
plt.legend()
plt.savefig('/home/Aujust/data/Kilonova/KN_Inference/KN_Inference/test.jpg',dpi=300)
print()