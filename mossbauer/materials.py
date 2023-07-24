import xraydb
from datetime import datetime

from mossbauer.utils import *
from mossbauer import MossbauerSource, MossbauerAbsorber

class PredefinedAbsorber(MossbauerAbsorber):
    """For specific absorbers so you don't have to construct on call

    Make a subclass of this for each absorber, and make one function get_all_pars()
    that takes a dict and returns a dict of all the required absorber params. 
    Examples below are the reference absorbers we have in hand.
    """
    def _setup_pars(self, **kwargs):
        self.__dict__.update(self.get_all_pars(**kwargs))
        return

class PredefinedSource(MossbauerSource):
    """For specific sources so you don't have to construct on call

    Make a subclass of this for each source, and make one function get_all_pars()
    that takes a dict and returns a dict of all the required source params. 
    Examples below are the reference sources we have in hand.
    """
    def _setup_pars(self, **kwargs):
        self.__dict__.update(self.get_all_pars(**kwargs))
        return


class PotassiumFerrocyanide(PredefinedAbsorber):
    """Two parameters (optional)

    abundance: default natural, as decimal fraction
    thickness_mgcm2_Fe57: column density Fe57
    """
    def get_all_pars(self, **kwargs):
        # properties of this particular absorber
        Fe57_abundance = kwargs.get('abundance', 0.02)
        thickness_mgcm2_Fe57 = kwargs.get('thickness_mgcm2_Fe57', 0.13)

        # properties of this compound
        recoilless_fraction = 0.6  # no idea!
        Eres = -0.038  # p/m 0.015
        #Eres = -0.159
        mixture = {  # K4[Fe(CN)6]·3H2O
            'Fe': 1,
            'K': 4,
            'C': 6,
            'N': 6,
            'H': 6,
            'O': 3,
        }

        # properties of Fe-57
        photon_energy = 14.4e3  # eV
        natural_linewidth = 4.55e-9  # eV
        resonant_absorption_coefficient = 25.0  # cm^2/mgFe57

        # derived quantities
        thickness_gcm2_Fetot = thickness_mgcm2_Fe57 / Fe57_abundance / 1e3

        mass_sum = np.sum([val * xraydb.atomic_mass(element) for element, val in mixture.items()])
        rho_eff = np.sum([xraydb.mu_elam(element, photon_energy) * val * xraydb.atomic_mass(element) / mass_sum for element, val in mixture.items()])
        thickness_gcm2 = thickness_gcm2_Fetot / (mixture['Fe'] * xraydb.atomic_mass('Fe') / mass_sum)
        optical_depth = rho_eff * thickness_gcm2

        linewidth = E_to_vel(natural_linewidth, photon_energy)
        thickness_normalized = thickness_mgcm2_Fe57 * resonant_absorption_coefficient * recoilless_fraction

        return dict(
            Eres=Eres,
            linewidth=linewidth,
            thickness_normalized=thickness_normalized,
            optical_depth=optical_depth,
        )


class AlphaIron(PredefinedAbsorber):
    """Two parameters (optional)

    abundance: default natural, as decimal fraction (default 0.02)
    thickness_microns: physical thickness in microns (default 0.4 mil)
    """
    def get_all_pars(self, **kwargs):
        # properties of Fe-57
        iron_density = 7.87  # g/cm3
        photon_energy = 14.4e3  # eV
        natural_linewidth = 4.55e-9  # eV
        resonant_absorption_coefficient = 25.0  # cm^2/mgFe57
        mass_absorption_coef = xraydb.mu_elam('Fe', photon_energy)  # cm^2/gFe

        # properties of this compound
        recoilless_fraction = 0.8

        split_ratio = (3, 2, 1, 1, 2, 3)
        transition_coefficients = np.asarray(split_ratio, dtype=float)/np.sum(split_ratio)
  
        # from Violet and Pipcorn 1971, using Palladium source
        # TODO: I guess we need to subtract palladium isomer shift and add rhodium?
        Eres = np.array([-5.48, -3.25, -1.01, 0.66, 2.90, 5.13])
        Eres -= Eres.mean()

        # properties of this particular absorber
        Fe57_abundance = kwargs.get('abundance', 0.02)
        thickness_microns = kwargs.get('thickness_microns', 0.4*25.4)
        thickness_gcm2_Fetot = thickness_microns / 1e4 * iron_density
        thickness_mgcm2_Fe57 = thickness_gcm2_Fetot * 1e3 * Fe57_abundance

        return dict(
            Eres=Eres,
            transition_coefficients=transition_coefficients,
            linewidth=E_to_vel(natural_linewidth, photon_energy),
            thickness_normalized=thickness_mgcm2_Fe57 * resonant_absorption_coefficient * recoilless_fraction,
            optical_depth=thickness_gcm2_Fetot * mass_absorption_coef
        )

class CobaltRhodiumMatrix(PredefinedSource):
    """One parameter (optional)

    activity: default to the original ~1 mCi source
    """
    def get_all_pars(self, **kwargs):
        Eres = 0.121
        photon_energy = 14.4e3  # eV
        natural_linewidth = 4.55e-9  # eV
        mossbauer_relative_intensity = 0.0916
        date = kwargs.get('date', datetime.now().strftime('%Y%m%d'))
        source_activity_Ci = get_current_activity(270., 2.6e-3, '20210830')
        source_activity = 3.7e10 * source_activity_Ci  # Hz

        ### source parameters
        return dict(
            Eres=Eres,
            linewidth=E_to_vel(natural_linewidth, photon_energy),
            total_activity=mossbauer_relative_intensity * source_activity
        )
