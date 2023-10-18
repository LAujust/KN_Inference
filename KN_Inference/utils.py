'''
@copyright
aujust@mail.ustc.edu.cn
'''

'Kilonova Aimed Inference for WFST: KAIW'
'Kilonova Aimed Lightcurve Classifier for WFST: KALC'

import numpy as np
import pandas as pd
import sncosmo
from sncosmo import TimeSeriesSource
from scipy.interpolate import interp1d
import afterglowpy as grb
import astropy.units as u
import astropy.constants as c
import simsurvey
from limited_mag import getlim


def load_filter_list():
    all = True
    filt_list = ['sdssu','sdssg','sdssr','sdssi','sdssz','desg','desr','desi','desz','desy','f435w','f475w','f555w','f606w','f625w',
        'f775w','nicf110w','nicf160w','f098m','f105w','f110w','f125w','f127m','f139m','f140w','f153m','f160w','f218w','f225w',
        'f275w','f300x','f336w','f350lp','f390w','f689m','f763m','f845m','f438w','uvf555w','uvf475w',
        'uvf606w','uvf625w','uvf775w','uvf814w','uvf850lp','cspb','csphs','csphd','cspjs','cspjd','cspv3009',
        'cspv3014','cspv9844','cspys','cspyd','cspg','cspi','cspk','cspr','cspu','f070w','f090w','f115w','f150w',
        'f200w','f277w','f356w','f444w','f140m','f162m','f182m','f210m','f250m','f300m','f335m','f360m','f410m','f430m',
        'f460m','f480m','lsstu','lsstg','lsstr','lssti','lsstz','lssty','keplercam::us','keplercam::b','keplercam::v','keplercam::v',
        'keplercam::r','keplercam::i','4shooter2::us','4shooter2::b','4shooter2::v','4shooter2::r','4shooter2::i','f062','f087',
        'f106','f129','f158','f184','f213','f146','ztfg','ztfr','ztfi','uvot::b','uvot::u','uvot::uvm2','uvot::uvw1','uvot::uvw2',
        'uvot::v','uvot::white','ps1::open','ps1::g','ps1::r','ps1::i','ps1::z','ps1::y','ps1::w','atlasc','atlaso','2massJ',
        '2massH','2massKs','wfst_u','wfst_g','wfst_r','wfst_i','wfst_z','wfst_w'
    ]
    for filt in filt_list:
        try:
            _x = sncosmo.get_bandpass(filt)
        except:
            print('Fail for '+filt)
            if all:
                all = False
    if all:
        print('Load all filters successfully!')
    return filt_list

def load_wfst_bands():
    add_bands = ['u','g','r','i','w','z']
    wfst_bands = ['wfst_'+i for i in add_bands]
    try:
        for add_band in add_bands:
            data = np.loadtxt('/transmission/WFST_WFST.'+add_band+'_AB.dat')
            wavelength = data[:,0]
            trans = data[:,1]
            band = sncosmo.Bandpass(wavelength, trans, name='wfst_'+add_band)
            sncosmo.register(band, 'wfst_'+add_band)
    except:
        pass
    return wfst_bands

def mab2flux(mab):
    #erg s^-1 cm^-2
    return 10**(-(mab+48.6)/2.5)

def flux2mab(f):
    #erg s^-1 cm^-2
    return -2.5*np.log10(f)-48.6

def sumab(mab_list):
    _flux_all = 0
    for mab in mab_list:
        _flux_all += mab2flux(mab)
    return flux2mab(_flux_all)

def lim_mag(survey_file):
    default_maglim = {
        'g':23.35,
        'r':22.95,
        'i':22.59
    }
    bands_index = {'g':1,'r':2,'i':3}
    survey_file['maglim'] = [getlim(int(survey_file['exposure_time'].iloc[i]),bgsky=22.0,n_frame=1,airmass=survey_file['airmass'].iloc[i],sig=5)[0][bands_index[survey_file['filt'].iloc[i]]] for i in range(len(survey_file.index))]
    #survey_file['maglim'] = [default_maglim[survey_file.loc[i,'filt']]+1.25*np.log10(survey_file.loc[i,'exposure_time']/30) for i in range(len(survey_file.index))]
    return survey_file

def survey2plan(survey_dir,fields_dir,save_dir=None,GW_trigger=None):
    wfst_survey = pd.read_csv(survey_dir)
    fields_file = np.loadtxt(fields_dir)
    if GW_trigger:
        mjd_strat = GW_trigger
    else:
        mjd_start = wfst_survey['observ_time'].min()-0.1

    wfst_fields = dict()
    wfst_fields['field_id'] = fields_file[:,0].astype(int)
    wfst_fields['ra'] = fields_file[:,1]
    wfst_fields['dec'] = fields_file[:,2]

    wfst_survey['band'] = ['wfst_'+wfst_survey.loc[i,'filt'] for i in range(len(wfst_survey['filt'].index))]
    if 'maglim' in list(wfst_survey.columns):
        pass
    else:
        wfst_survey = lim_mag(wfst_survey)
    wfst_survey['time'] = wfst_survey['observ_time']
    wfst_survey['field'] = wfst_survey['field_id']
    wfst_survey['phase'] = wfst_survey['time']-mjd_start
    wfst_survey = wfst_survey.loc[:,['time','field','band','maglim','phase']]
    if save_dir:
        with open(save_dir,'wb') as handle:
            pickle.dump(wfst_survey,handle)
            handle.close()
    return wfst_survey


def photometry():
    pass

def extinction():
    pass

def log_likelihood():
    pass

def log_likelihood_upl():
    'log_likelihood with upper limit'
    pass

#==============================================================================#
bands_lam = {'ztfg':4783,'ztfr':6417,'ztfi':7867,
             'wfst_u':3641.35,
             'wfst_g':4691.74,
             'wfst_r':6158.74,
             'wfst_i':7435.86,
             'wfst_z':8562.26}





#--------------------------------------#
#               Constant               #
#--------------------------------------#

day2sec = u.day.cgs.scale
MPC_CGS = u.Mpc.cgs.scale
C_CGS = c.c.cgs.value
M_SUN_CGS = c.M_sun.cgs.value
G_CGS = c.G.cgs.value
Jy = u.Jy.cgs.scale
ANG_CGS = u.Angstrom.cgs.scale
pi = np.pi
pc10 = 10 * u.pc.cgs.scale
SB_CGS = c.sigma_sb.cgs.value
H_CGS = c.h.cgs.value
K_B_CGS = c.k_B.cgs.value
KM_CGS = u.km.cgs.scale