'''
@copyright
aujust@mail.ustc.edu.cn
'''

from .Knust import *
from .utils import *
import kilonovanet

'''
If you use kilonova only, just use Knust to genrate lightcurve.
Otherwise you should use following modulues to generate lighycurves, i.e. KN + GRB, KN + Cocoon...
'''

MODEL_PARAMETERS = {
    #KN
    'Bulla_3comp_full':['log_Mdyn','log_Mpm','Phi','cosTheta'],  #10.1126/science.abb4317
    'Bulla_3comp_spectra':['log_Mdyn','log_Mpm','Phi','cosTheta'],  
    'Bulla_bhns_full':['log_Mdyn','log_Mpm','cosTheta'],
    'Bulla_bhns_spectra':['log_Mdyn','log_Mpm','cosTheta'],  #10.1038/s41550-020-1183-3
    'Bu2022Ye':['log_Mdyn','vdyn','Ye_dyn','log_Mpm','vpm','Theta'],  #arXiv:2307.11080
    'Kasen_BNS_full':['log_Mej','vej','log_Xlan'],
    'Kasen_2comp_asym':['M_blue', 'v_blue', 'Xlan_blue', 'M_red', 'v_red', 'Xlan_red,' 'cos_theta_open', 'cos_theta'], #based on kilonovanet
    'Kasen_2comp_asym':['log_M_blue', 'v_blue', 'log_Xlan_blue', 'log_M_red', 'v_red', 'log_Xlan_red,' 'cos_theta_open', 'cos_theta'], #based on Knust
    #Afterglow
    'iso_cocoon':['M','v','s','R','kapps','t_day'], #dict
    'Aspherical_cocoon':['M','v','s','R','kapps','t_day','theta_open','theta'], #dict
    'agterglowpy.jet':['jetType','specType','E0','thetaObs','thetaCore','thetaWing','n0','p','epsilon_e','epsilon_B','xi_N','z','dL'],  #dict, Gaussian, Tophat
    
}



#=====================================================================================#
#===================================== MODELS ========================================#
#=====================================================================================#

class KilonovaGRB(object):
    '''
    model_type[str]:      regression method, i.e. gpr, tensorflow
    model_name[str]:      the name of surrogate model
    model_dir[dir]:       path dir of your stored model
    surrogator[modulue]:  the Surrogator you will use, i.e. Knust, kilonovanet
    '''
    def __init__(self,model_type,model_name,model_dir=None,Z=None,surrogator=Knust):
        self.model_dir = model_dir
        self.model_name = model_name
        self.model_type = model_type
        self.kn_model = surrogator(model_type=self.model_type,model_name=self.model_name,model_dir=self.model_dir)
        self.set_grb(Z)

    def set_grb(self,Z):
        'afterglow package'
        
        # Jet Parameters
        # Z = {'jetType':     grb.jet.TopHat,     # Gaussian jet
        #     'specType':    0,                  # Basic Synchrotron Emission Spectrum

        #     'thetaObs':    0.17,   # Viewing angle in radians
        #     'E0':          10**(50.4), # Isotropic-equivalent energy in erg
        #     'thetaCore':   0.14,    # Half-opening angle in radians
        #     'thetaWing':   0.4,    # Outer truncation angle
        #     'n0':          10**(-2.8),    # circumburst density in cm^{-3}
        #     'p':           2.31,    # electron energy distribution index
        #     'epsilon_e':   10**(-0.17),    # epsilon_e
        #     'epsilon_B':   10**(-2.1),   # epsilon_B
        #     'xi_N':        5e-2,    # Fraction of electrons accelerated
        #     'd_L':         Planck18.luminosity_distance(z).value*MPC_CGS, # Luminosity distance in cm
        #     'z':           z}   # redshift
        self.Z = Z
        
    def cal_lc(self,param_list,times,bands,dL):
        lcs = {}
        lcs_kn = {}
        lcs_grb = {}
        lc_kn_bands = self.kn_model.cal_lc(param_list=param_list,times=times,bands=bands,dL=dL)
        tsec = times * day2sec
        for band in bands:
            lam = bands_lam[band]
            nu = C_CGS/(lam*ANG_CGS)
            flux_grb = 1e-3 * grb.fluxDensity(tsec,nu,**self.Z)  #Jy

            lc_grb = -2.5*np.log10(flux_grb*1e-3/3631)   #AB mag
            lc_kn = lc_kn_bands[band]    #AB mag
            lc_ = sumab([lc_kn,lc_grb])

            lcs[band] = lc_
            lcs_kn[band] = lc_kn
            lcs_grb[band] = lc_grb

        return lcs, lcs_kn, lcs_grb
    



class Kasen_2comp_asym(object):
    def __init__(self,mode='kilonovanet'):
        if mode == 'kilonovanet':
            self.model = kilonovanet.Model('/home/Aujust/data/Kilonova/possis/data/metadata_kasen_bns.json',
                    '/home/Aujust/data/Kilonova/possis/models/kasen-bns-latent-10-hidden-500-CV-4-2021-04-17-epoch-200.pt',
                    filter_library_path='/home/Aujust/data/Kilonova/possis/data/filter_data')
            self.wave = np.loadtxt('/home/Aujust/data/Kilonova/GPR/Kasen_unprocessed_wavelength.txt')
        elif mode == 'Knust':
            self.model = Knust('gpr','Kasen_BNS_full','/home/Aujust/data/Kilonova/GPR/')
        else:
            raise KeyError('No Surrogator matched.')

        self.DIFF_CONST = 2.0 * M_SUN_CGS / (13.7 * C_CGS**2)
        self.mode = mode

    def get_A_proj(self,cos_theta,cos_theta_open):
        self._cos_theta = cos_theta
        # Opening angle
        self._cos_theta_open = cos_theta_open

        # Dharba and Kasen 2020 scalings for conical opening in opaque sphere

        ct = (1-self._cos_theta_open**2)**0.5

        if self._cos_theta > ct:
            Aproj_top = np.pi * ct * self._cos_theta
        else:
            theta_p = np.arccos(self._cos_theta_open /
                                (1 - self._cos_theta**2)**0.5)
            theta_d = np.arctan(np.sin(theta_p) / self._cos_theta_open *
                        (1 - self._cos_theta**2)**0.5 / np.abs(self._cos_theta))
            Aproj_top = (theta_p - np.sin(theta_p)*np.cos(theta_p)) - (ct *
                            self._cos_theta*(theta_d -
                            np.sin(theta_d)*np.cos(theta_d) - np.pi))

        minus_cos_theta = -1 * self._cos_theta

        if minus_cos_theta < -1 * ct:
            Aproj_bot = 0
        else:
            theta_p2 = np.arccos(self._cos_theta_open /
                                (1 - minus_cos_theta**2)**0.5)
            theta_d2 = np.arctan(np.sin(theta_p2) / self._cos_theta_open *
                        (1 - minus_cos_theta**2)**0.5 / np.abs(minus_cos_theta))

            Aproj_bot1 = (theta_p2 - np.sin(theta_p2)*np.cos(theta_p2)) + (ct *
            minus_cos_theta*(theta_d2 - np.sin(theta_d2)*np.cos(theta_d2)))
            Aproj_bot = np.max([Aproj_bot1, 0])

        Aproj = Aproj_top + Aproj_bot

        # Compute reference areas for this opening angle to scale luminosity

        cos_theta_ref = 0.5

        if cos_theta_ref > ct:
            Aref_top = np.pi * ct * cos_theta_ref
        else:
            theta_p_ref = np.arccos(self._cos_theta_open /
                                (1 - cos_theta_ref**2)**0.5)
            theta_d_ref = np.arctan(np.sin(theta_p_ref) / self._cos_theta_open *
                        (1 - cos_theta_ref**2)**0.5 / np.abs(cos_theta_ref))
            Aref_top = (theta_p_ref - np.sin(theta_p_ref) *
                        np.cos(theta_p_ref)) - (ct * cos_theta_ref *
                        (theta_d_ref - np.sin(theta_d_ref) *
                        np.cos(theta_d_ref) - np.pi))

        minus_cos_theta_ref = -1 * cos_theta_ref

        if minus_cos_theta_ref < -1 * ct:
            Aref_bot = 0
        else:
            theta_p2_ref = np.arccos(self._cos_theta_open /
                                (1 - minus_cos_theta_ref**2)**0.5)
            theta_d2_ref = np.arctan(np.sin(theta_p2_ref) /
                    self._cos_theta_open * (1 - minus_cos_theta_ref**2)**0.5 /
                        np.abs(minus_cos_theta_ref))

            Aref_bot = (theta_p2_ref - np.sin(theta_p2_ref) *
                        np.cos(theta_p2_ref)) + (ct * minus_cos_theta_ref *
                        (theta_d2_ref - np.sin(theta_d2_ref) *
                        np.cos(theta_d2_ref)))

        Aref = Aref_top + Aref_bot

        Ablue = Aproj
        Ablue_ref = Aref

        Ared = np.pi - Ablue
        Ared_ref = np.pi - Ablue_ref

        return Ablue,Ablue_ref,Ared,Ared_ref
    
    def get_kappa(self,Xlan):
        log_X = np.log10(Xlan)
        x_ = [-4,-2]
        y_ = [0,1]
        f = interp1d(x_,y_,fill_value='extrapolate')
        return 10**f(log_X)
    
    def _mab2flux(mab):
        #erg s^-1 cm^-2
        return 10**(-(mab+48.6)/2.5)

    def _sumab(self,mab_list,amplitude):
        """
        mab_list: (n,k) array of n M_ab's of length k
        amplitude: (n,k) array of n weighting factors on flux of length k
        """
        _flux_all = 0
        for i,mab in enumerate(mab_list):
            _flux_all += self._mab2flux(mab)*amplitude[i]
        return self._flux2mab(_flux_all)

    def calc_spectra(self,tt,param_list):
        'param: M_blue, v_blue, Xlan_blue, M_red, v_red, Xlan_red, cos_theta_open, cos_theta'
        uniq_times = tt

        Ablue,Ablue_ref,Ared,Ared_ref = self.get_A_proj(param_list[6],param_list[7])
        flux_blue = self.model.predict_spectra(param_list[:3],tt)[0].T/(4*np.pi*pc10**2)
        flux_red = self.model.predict_spectra(param_list[3:6],tt)[0].T/(4*np.pi*pc10**2)

        self.kappa_blue = self.get_kappa(param_list[2])
        self.kappa_red = self.get_kappa(param_list[5])
        
        self._tau_diff_blue = np.sqrt(self.DIFF_CONST * self.kappa_blue *
                                 param_list[0] / param_list[1]) / day2sec
        
        self._tau_diff_red = np.sqrt(self.DIFF_CONST * self.kappa_red *
                                 param_list[3] / param_list[4]) / day2sec
        
        
        flux_blue *= (1 + 1.4 * (2 + uniq_times/self._tau_diff_blue/0.59) / (1 +
                    np.exp(uniq_times/self._tau_diff_blue/0.59)) *
                    (Ablue/Ablue_ref - 1))

        flux_red *= (1 + 1.4 * (2 + uniq_times/self._tau_diff_red/0.59) / (1 +
                    np.exp(uniq_times/self._tau_diff_red/0.59)) *
                    (Ared/Ared_ref - 1))
        
        flux = flux_blue + flux_red

        return tt, self.wave, np.array(flux).T
    
    def cal_lc(self,param_list,times,bands,dL):
        'param: M_blue, v_blue, Xlan_blue, M_red, v_red, Xlan_red,cos_theta_open,cos_theta'
        if dL<1000:
            dL = dL * MPC_CGS

        if self.mode == 'Knust':
            return self._cal_lc_K(param_list,times,bands,dL)
        else:

            mags = {}
            times_min = np.min([np.min(times)-0.1,0])
            times_max = np.max(times)+1
            t_interp = np.arange(times_min,times_max,0.1)
            t_idxs = [int(np.median(np.where(np.isclose(t_interp,T,rtol=1e-2,atol=5e-2)))) for T in times]

            for band in bands:
                phase,wave,flux_data = self.calc_spectra(tt=t_interp,param_list=param_list)
                source = TimeSeriesSource(phase, wave, flux_data)
                model = sncosmo.Model(source=source)
                abmag_interp = model.bandmag(band,'ab',t_interp) + 5*np.log10(dL/pc10)
                abmag = abmag_interp[t_idxs]
                mags[band] = abmag
            return mags
    
    def _cal_lc_K(self,param_list,times,bands,dL):   #Knust lc
        uniq_times = times
        if dL<1000:
            dL = dL * MPC_CGS
        mags = {}
        param1 = param_list[:3]
        param2 = param_list[3:6]

        mags1 = self.model.cal_lc(param_list=param1,times=times,bands=bands,dL=dL)
        mags2 = self.model.cal_lc(param_list=param2,times=times,bands=bands,dL=dL)

        Ablue,Ablue_ref,Ared,Ared_ref = self.get_A_proj(param_list[6],param_list[7])

        self.kappa_blue = self.get_kappa(param_list[2])
        self.kappa_red = self.get_kappa(param_list[5])
        
        self._tau_diff_blue = np.sqrt(self.DIFF_CONST * self.kappa_blue *
                                 param_list[0] / param_list[1]) / day2sec
        
        self._tau_diff_red = np.sqrt(self.DIFF_CONST * self.kappa_red *
                                 param_list[3] / param_list[4]) / day2sec
        
        for band in bands:
            mag1,mag2 = mags1[band],mags2[band]
            Ms = [mag1,mag2]
            As = [
                (1 + 1.4 * (2 + uniq_times/self._tau_diff_blue/0.59) / (1 +
                    np.exp(uniq_times/self._tau_diff_blue/0.59)) *
                    (Ablue/Ablue_ref - 1)),
                (1 + 1.4 * (2 + uniq_times/self._tau_diff_red/0.59) / (1 +
                    np.exp(uniq_times/self._tau_diff_red/0.59)) *
                    (Ared/Ared_ref - 1))
            ]
            mags[band] = self._sumab(Ms,As)
        return mags





'Cocoon: https://doi.org/10.3847/1538-4357/aaaab3'

def iso_cocoon(param_dict):
    s = param_dict.get('s',3.)
    M = param_dict.get('M',0.01)  #M_sun
    R = param_dict.get('R',5e10)  #cm
    kappa = param_dict.get('kappa',0.2)  #g cm^-1
    v = param_dict.get('v',0.2)  ##c

    t_day = param_dict.get('t_day')

    t_diff = np.sqrt(kappa*M*M_SUN_CGS/(4*np.pi*C_CGS*v*C_CGS))/day2sec
    # t_diff2 = 0.5*np.sqrt(1e2*kappa*M/v)
    L = M*v*R*M_SUN_CGS*C_CGS*(t_day/t_diff)**(-4/(s+2))/(2*(t_diff*day2sec)**2)

    t_tau = np.sqrt(1./v)*t_diff
    r_ph = v*C_CGS*t_tau*day2sec*(t_day/t_tau)**((s+1)/(s+3))
    T_eff = (L/(4*np.pi*r_ph**2*SB_CGS))**0.25
    #print(t_diff,t_tau)
    return L, T_eff, r_ph, t_diff


class Aspherical_cocoon(object):
    def __init__(self):
        self.DIFF_CONST = 2.0 * M_SUN_CGS / (13.7 * C_CGS * KM_CGS)

    def theta_cocoon(self,param_dict):
        theta = param_dict.get('theta',0.)
        theta_open = param_dict.get('theta_open',0.5) #in Rad
        M = param_dict.get('M',0.01)
        kappa = param_dict.get('kappa',0.2)
        v = param_dict.get('v',0.2)
        t_day = param_dict.get('t_day')

        L,T_eff,r_ph, t_diff = iso_cocoon(param_dict)
        L_iso = L * (theta_open**2/2)**(1./3)

        Ablue,Ablue_ref,Ared,Ared_ref = self.get_A_proj(np.cos(theta),np.cos(theta_open))

        L_iso *= (1 + 1.4 * (2 + t_day/t_diff/0.59) / (1 +
                    np.exp(t_day/t_diff/0.59)) *
                    (Ablue/Ablue_ref - 1))

        T_eff = (L_iso/(4*np.pi*r_ph**2*SB_CGS))**0.25
        
        return L_iso,T_eff,r_ph

    def get_A_proj(self,cos_theta,cos_theta_open):
        self._cos_theta = cos_theta
        # Opening angle
        self._cos_theta_open = cos_theta_open

        # Dharba and Kasen 2020 scalings for conical opening in opaque sphere

        ct = (1-self._cos_theta_open**2)**0.5

        if self._cos_theta > ct:
            Aproj_top = np.pi * ct * self._cos_theta
        else:
            theta_p = np.arccos(self._cos_theta_open /
                                (1 - self._cos_theta**2)**0.5)
            theta_d = np.arctan(np.sin(theta_p) / self._cos_theta_open *
                        (1 - self._cos_theta**2)**0.5 / np.abs(self._cos_theta))
            Aproj_top = (theta_p - np.sin(theta_p)*np.cos(theta_p)) - (ct *
                            self._cos_theta*(theta_d -
                            np.sin(theta_d)*np.cos(theta_d) - np.pi))

        minus_cos_theta = -1 * self._cos_theta

        if minus_cos_theta < -1 * ct:
            Aproj_bot = 0
        else:
            theta_p2 = np.arccos(self._cos_theta_open /
                                (1 - minus_cos_theta**2)**0.5)
            theta_d2 = np.arctan(np.sin(theta_p2) / self._cos_theta_open *
                        (1 - minus_cos_theta**2)**0.5 / np.abs(minus_cos_theta))

            Aproj_bot1 = (theta_p2 - np.sin(theta_p2)*np.cos(theta_p2)) + (ct *
            minus_cos_theta*(theta_d2 - np.sin(theta_d2)*np.cos(theta_d2)))
            Aproj_bot = np.max([Aproj_bot1, 0])

        Aproj = Aproj_top + Aproj_bot


        # Compute reference areas for this opening angle to scale luminosity

        cos_theta_ref = 0.5

        if cos_theta_ref > ct:
            Aref_top = np.pi * ct * cos_theta_ref
        else:
            theta_p_ref = np.arccos(self._cos_theta_open /
                                (1 - cos_theta_ref**2)**0.5)
            theta_d_ref = np.arctan(np.sin(theta_p_ref) / self._cos_theta_open *
                        (1 - cos_theta_ref**2)**0.5 / np.abs(cos_theta_ref))
            Aref_top = (theta_p_ref - np.sin(theta_p_ref) *
                        np.cos(theta_p_ref)) - (ct * cos_theta_ref *
                        (theta_d_ref - np.sin(theta_d_ref) *
                        np.cos(theta_d_ref) - np.pi))

        minus_cos_theta_ref = -1 * cos_theta_ref

        if minus_cos_theta_ref < -1 * ct:
            Aref_bot = 0
        else:
            theta_p2_ref = np.arccos(self._cos_theta_open /
                                (1 - minus_cos_theta_ref**2)**0.5)
            theta_d2_ref = np.arctan(np.sin(theta_p2_ref) /
                    self._cos_theta_open * (1 - minus_cos_theta_ref**2)**0.5 /
                        np.abs(minus_cos_theta_ref))

            Aref_bot = (theta_p2_ref - np.sin(theta_p2_ref) *
                        np.cos(theta_p2_ref)) + (ct * minus_cos_theta_ref *
                        (theta_d2_ref - np.sin(theta_d2_ref) *
                        np.cos(theta_d2_ref)))

        Aref = Aref_top + Aref_bot

        Ablue = Aproj
        Ablue_ref = Aref

        Ared = np.pi - Ablue
        Ared_ref = np.pi - Ablue_ref

        return Ablue,Ablue_ref,Ared,Ared_ref
