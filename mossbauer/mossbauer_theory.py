from dataclasses import dataclass, field
import numpy as np
from scipy.integrate import quad_vec
from scipy.special import jv
import xraydb 
from datetime import datetime



# Fundamental constants
c = 3e8  # Speed of light in m/s
kB = 1.38e-23  # Boltzmann constant in J/K
e = 1.6e-19  # Elementary charge in C
Na = 6.022e23  # Avogadro's number


############################################     FUNCTIONS   #########################################################


def calculate_sigma0(E0, Ie, Ig, alpha):
    return 2.446e-15 / (E0 / 1e3) ** 2 * (2 * Ie + 1) / (2 * Ig + 1) * 1 / (1 + alpha)

def calculate_mu_e(element, energy):
    return xraydb.mu_elam(element, energy)

def calculate_recoilless_fraction(T, Td, E0, M):
    integral = quad(lambda x: x / (np.exp(x) - 1), 0, Td / T)[0]
    return np.exp(-(3 * (E0 * e) ** 2) / (kB * Td * M * c ** 2) * (1 / 4 + (T / Td) ** 2 * integral))

def get_current_activity(half_life_days, activity, date, nowdate=None):
    if nowdate is None:
        nowdate = datetime.now()
    else:
        nowdate = datetime.strptime(nowdate, '%Y%m%d')
    tdiff_seconds = (nowdate - datetime.strptime(date, '%Y%m%d')).total_seconds()
    return activity*((0.5)**(tdiff_seconds/(3600*24)/half_life_days))


def calculate_photon_rate(activity_Ci,relative_intensity):
    return 3.7e10 * activity_Ci * relative_intensity

def calculate_thickness_gcm2(thickness_m, rho, abundance):
    return thickness_m * rho/10 * abundance

def _lorentzian_s(x, x0, gamma):
    return gamma/(2*np.pi)/( ((x-x0))**2 + (gamma/2)**2)

def _lorentzian_a(x, x0, gamma):
    return (gamma/2)**2/( ((x-x0))**2 + (gamma/2)**2)
    




############################################     SOURCE CLASSES   #########################################################


@dataclass
class CobaltRhodium:
    T: float = 292
    Td: float = 510 #Rhodium
    E0: float = 14.4e3
    Gamma_ev = 4.55e-9 #eV
    Eres = [0]
    split_ratio = [1] 

    activity_Ci= 50e-3
    production_date = '20250112'
    date = None #means now
    half_life = 272
    mossbauer_relative_intensity = 0.0916
    M =  57e-3 / Na
    element = 'Fe'
    alpha = 8.17
    
    def update_params(self):
        
        self.Gamma = self.Gamma_ev*c/self.E0*1000
        self.current_activity_Ci = get_current_activity(self.half_life, self.activity_Ci, self.production_date,self.date)       
        self.mossbauer_photon_rate = calculate_photon_rate(self.current_activity_Ci, self.mossbauer_relative_intensity)       
        self.fs = calculate_recoilless_fraction(self.T, self.Td, self.E0, self.M)       
        self.transition_coefficients = np.asarray(self.split_ratio, dtype=float)/np.sum(self.split_ratio)
    


@dataclass
class CobaltFe:
    T = 292
    Td = 470 #Fe
    E0 = 14.4e3
    Gamma_ev = 4.55e-9 #eV
    Eres = [-5,-3,-1,1,3,5]
    split_ratio = [3,2,1,1,2,3] 
   
    activity_Ci= 50e-3
    production_date = '20250112'
    date = None #means now
    half_life = 272
    mossbauer_relative_intensity = 0.0916
    M =  57e-3 / Na
    element = 'Fe'
    alpha = 8.17
    
    def update_params(self):
        
        self.Gamma = self.Gamma_ev*c/self.E0*1000
        self.current_activity_Ci = get_current_activity(self.half_life, self.activity_Ci, self.production_date,self.date)       
        self.mossbauer_photon_rate = calculate_photon_rate(self.current_activity_Ci, self.mossbauer_relative_intensity)       
        self.fs = calculate_recoilless_fraction(self.T, self.Td, self.E0, self.M)       
        self.transition_coefficients = np.asarray(self.split_ratio, dtype=float)/np.sum(self.split_ratio)


    
########################################          ABSORBER CLASSES      ###############################################

@dataclass
class alphaFe:
    thickness_m = 25e-6 #m
    T = 292
    Td = 470
    E0 = 14.4e3
    Gamma_ev = 4.55e-9 #eV
    Eres = [-5,-3,-1,1,3,5]
    split_ratio = [3,2,1,1,2,3]
    abundance = 0.0212
    M = 57e-3 / Na
    nM = Na / 57
    rho = 7.87e3
    element = 'Fe'
    alpha = 8.17
    
    def update_params(self):
        
        self.Gamma = self.Gamma_ev*c/self.E0*1000
        self.mu_e = calculate_mu_e(self.element, self.E0)
        self.fa = calculate_recoilless_fraction(self.T, self.Td, self.E0, self.M)
        self.transition_coefficients = np.asarray(self.split_ratio, dtype=float)/np.sum(self.split_ratio)
        self.sigma0 = calculate_sigma0(self.E0, 3/2, 1/2, self.alpha)   
        self.thickness_gcm2 = calculate_thickness_gcm2(self.thickness_m, self.rho, 1)
        self.thickness_gcm2_Fe57 = calculate_thickness_gcm2(self.thickness_m, self.rho, self.abundance)
        self.thickness_normalized = self.fa * self.nM * self.sigma0 * self.thickness_gcm2_Fe57
    


class KFeCy:
    thickness_gcm2_Fe = 6e-3 #g/cm2
    E0 = 14.4e3 #eV
    Gamma_ev = 4.55e-9 #eV
    Eres = [0] #mm/s
    split_ratio = [1]
    abundance = 0.0212
    M =  57e-3 / Na
    nM = Na / 57
    rho = 7.87e3
    elements = {  # K4[Fe(CN)6]·3H2O
                'Fe': 1,
                'K': 4,
                'C': 6,
                'N': 6,
                'H': 6,
                'O': 3,
            }
    alpha = 8.17
    sigma0 = 2.1e-18 # https://journals.aps.org/prb/pdf/10.1103/PhysRevB.4.2915
    fa =0.311 # https://www.sciencedirect.com/science/article/abs/pii/0029554X81900987
    
    def update_params(self):

        self.Gamma = self.Gamma_ev*c/self.E0*1000       
        self.transition_coefficients = np.asarray(self.split_ratio, dtype=float)/np.sum(self.split_ratio)
        self.thickness_gcm2_Fe57 = self.thickness_gcm2_Fe * self.abundance
        self.thickness_normalized = self.fa * self.nM * self.sigma0 * self.thickness_gcm2_Fe57                       
        self.mass_sum = np.sum([val * xraydb.atomic_mass(element) for element, val in self.elements.items()])
        self.mu_e = np.sum([xraydb.mu_elam(element, self.E0) * multiplicity *xraydb.atomic_mass(element) / self.mass_sum \
                            for element, multiplicity in self.elements.items()])
        self.thickness_gcm2 = self.thickness_gcm2_Fe / (self.elements['Fe'] * xraydb.atomic_mass('Fe')) * self.mass_sum 
        
    
        
    
###########################################      MOSSBAUER    ######################################################
    


c=3e8

class Mossbauer:
    
    def __init__(self, source, absorber):
    
        self.source = source
        self.absorber = absorber

    def source_specrtrum(self, E, v):
        spectrum = 0
        for coef, Eres in zip(self.source.transition_coefficients, self.source.Eres):
                spectrum += coef *_lorentzian_s(
                    E,
                    Eres - v,
                    self.source.Gamma
                )
        
        return  self.source.mossbauer_photon_rate * self.source.fs  * spectrum
    
    def cross_section(self, E):
        spectrum = 0
        for coef, Eres in zip(self.absorber.transition_coefficients, self.absorber.Eres):
            spectrum += coef *_lorentzian_a(
                E,
                Eres,
                self.absorber.Gamma
            )
            
        return self.absorber.sigma0 * spectrum
    
    def non_resonant_attenuation(self):
        return np.exp(-self.absorber.mu_e * self.absorber.thickness_gcm2) 
    
    def resonant_transmission_fraction(self, E, v):
        
        res = ( self.source_specrtrum(E, v) 
                * np.exp(-self.cross_section(E) * self.absorber.fa * self.absorber.nM * self.absorber.thickness_gcm2_Fe57 )
                * self.non_resonant_attenuation()
              )
        return res
                         
    def reosnant_transmission_rate(self, v):
        res = quad_vec(lambda E: self.resonant_transmission_fraction(E,v), -np.inf, np.inf)[0]
        return res
    
    def nonresonant_transmission_rate(self):
        return self.source.mossbauer_photon_rate * (1 - self.source.fs) * self.non_resonant_attenuation()

    def total_transmission_rate1(self, v):
        return self.nonresonant_transmission_rate() + self.reosnant_transmission_rate(v)
 
    
    def epsilon(self,t):
        return 1-np.exp(-t/2)*jv(0,1j*t/2).real

