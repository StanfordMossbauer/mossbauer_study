from scipy.integrate import quad, quad_vec
from scipy.stats import cauchy, binom, poisson
from scipy.optimize import curve_fit, minimize
from scipy.special import jv

from scipy.interpolate import interp1d

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import pandas as pd
from os.path import join

from .utils import *
from mossbauer import physics

import pytest

class MossbauerMaterial:
    """Parent class for a Mossbauer isotope
    
    Mainly because source and absorber might share certain things
    or even be the same object eventually. So trying to be maximally
    general. For now, a child class just needs a function _additional_pars()
    that returns a list of required inputs.

    9/18/23 (Joey) - I maybe regret a bit how many classes I made, like I'm not sure
    this was ultimately necessary. The real motivation was that absorbers and
    sources share "Eres" and "linewidth" as properties... Anyway it works...
    """
    def __init__(self, **kwargs):
        # assert that all required pars were supplied
        # then update the dict with them (and no others)
        required_pars = [
            'Eres',
            'linewidth',
        ] 
        required_pars += self._additional_pars()
        self._setup_pars(**kwargs)
        self.is_split = False  # whether material has one line or several
        if hasattr(self.Eres, '__len__'):
            required_pars += ['transition_coefficients']
            self.is_split = True
        check_pars(self.__dict__, required_pars)
        if self.is_split:
            # coefficients must be unity up to 1 ppm
            assert 1==pytest.approx(np.sum(self.transition_coefficients), 1e-6), \
                "Coefficients must sum to unity!"
        return

    def _setup_pars(self, **kwargs):
        self.__dict__.update(kwargs)
        return

    def _additional_pars(self):
        """Placeholder - overwrite with subclass

        Should return a list of required kwargs to __init__
        """
        return


class MossbauerSource(MossbauerMaterial):
    """Mossbauer source class

    In addition to Eres and linewidth, a source requires a total activity in Hz.
    Its main attribute is spectrum() which tells you the rate of photons emitted at
    a given energy (in velocity units). This class is meant to ignore anything not
    emitted as a recoilless (Mossbauer) photon.
    """
    def _additional_pars(self):
        return ['total_activity']

    def spectrum(self, E):
        """Returns photons/sec at the given energy"""
        spectrum = 0
        if self.is_split:
            for coef, Eres in zip(self.transition_coefficients, self.Eres):
                spectrum += coef * lorentzian_norm(
                    E, Eres, self.linewidth/2
                )
        else:
            spectrum += lorentzian_norm(
                E, self.Eres, self.linewidth/2
            )
        spectrum *= self.total_activity
        return spectrum


class MossbauerAbsorber(MossbauerMaterial):
    """Mossbauer absorber class

    Besides Eres and linewidth, we need to specify the thickness in units of
    resonant absorption mean free paths (dimensionless) to determine which photons
    get absorbed. 

    There is also an optional kwarg `optical_depth` to account for non-resonant (mass)
    absorption in the absorber. In materials.py you can see examples of how to use NIST
    with this.

    The main attribute is transmission_fraction(), which determines
    how much absorption occurs.
    """
    def _additional_pars(self):
        """Require number of mean free paths thick"""
        return  ['thickness_normalized']

    def transmission_fraction(self, E, vel=0.0):
        """Returns surviving fraction of photons passing through"""
        T = np.exp(
            -1 
            * self.cross_section(E, vel) 
            * self.thickness_normalized
        )
        if hasattr(self, 'optical_depth'):
            T *= np.exp(-self.optical_depth)
        return T

    def cross_section(self, E, vel=0.0):
        """Returns resonant absorption cross-section at the given energy"""
        xsec = 0.0
        if self.is_split:
            for coef, Eres in zip(self.transition_coefficients, self.Eres):
                xsec += coef * lorentzian(
                    E,
                    Eres - vel,
                    self.linewidth/2,
                )
        else:
            xsec = lorentzian(
                E, 
                self.Eres - vel,
                self.linewidth/2,
            )
        return xsec


class MossbauerMeasurement:
    """Wrapper class for a Mossbauer scan
    
    Owns a MossbauerSource instance and a MossbauerAbsorber instance.
    For now this houses everything, maybe too much. The expected spectrum for 
    arbitrary velocity and the sensitivity of the experiment to arbitrary
    physics is here. But it also can load measured data, plot it against
    expectation, eventually do fitting etc.

    source_p: arguments to the source or MossbauerSource instance
    absorber_p: arguments to the absorber
    measurement_p: parameters for the detector, relative positions, 
                   measurement times, etc.
    """
    def __init__(self, source, absorber, measurement_p):
        # take in truth information of mossbauer setup
        # construct an expected pdf
        if isinstance(source, dict):
            self.source = MossbauerSource(**source)
        else:
            assert isinstance(source, MossbauerSource), "first arg must be dict or MossbauerSource instance"
            self.source = source
        if isinstance(absorber, dict):
            self.absorber = MossbauerAbsorber(**absorber)
        else:
            assert isinstance(absorber, MossbauerAbsorber), "second arg must be dict or MossbauerAbsorber instance"
            self.absorber = absorber
        self._setup_measurement(measurement_p)
        # init empty arrays for observed data
        self.clear_measurements()
        return

    def get_sensitivity(self, deltaE, model, **kwargs):
        """Median expected sensitivity 
        
        deltaE: minimum measureable energy shift for the experiment
                (assumed to be separation-independent)
        model: string identifier of the sensitivity model to use

        This will get more complicated when we think about separations.. really
        we should have some test statistic and the function should take in the
        radii being measured? Idk..
        """
        models = ['down_quark']
        assert model in models, "Model must be one of: " + str(models)
        rs = kwargs.get('separations', np.logspace(-9, -6, 100))
        sensi = getattr(self, f'_get_sensitivity_{model}')(deltaE, rs, **kwargs)
        return rs, sensi

    def set_velocity(self, vel):
        """Set the (relative?) velocity of the experiment
        For now this is a class attribute which can be
        an array or a number.
        """
        # TODO: generalize this to source and absorber velocity? so we can have both
        #       moving at once?
        self.velocity = vel
        return

    def transmitted_spectrum(self, vel):
        """Give the fraction of photons transmitted as a function of velocity

        Velocity can be an array or float. If array, you get a spectrum, and
        if float you get back a float. All returned values are in (0, 1).

        This is necessary because the integral doesnt have a closed-form solution
        and needs to be numerically evaluated. The result is Lorentzian only in the
        thin-absorber approximation (possibly valid for our eventual setup).
        """
        transmitted_mossbauer_photons = self._integrate_function_inf(
            vel, self._transmission_integrand
        )
        background_rate = self.__dict__.get('background_rate', 0.0)
        return background_rate + transmitted_mossbauer_photons

    def transmitted_spectrum_derivative(self, vel):
        return self._integrate_function_inf(vel, self._transmission_derivative_integrand)

    def get_deltaEmin_linear(self, **kwargs):
        """First order expansion about velocity
        Seems to be within 0.1% of full calculation, and much faster.
        """
        # idk if confusing to let kwargs override these
        vels = kwargs.get('vels', self.source.linewidth*np.logspace(-6, 2, 10000)/2)
        acquisition_time = kwargs.get('acquisition_time', self.acquisition_time)

        rates = self.transmitted_spectrum(vels)
        ders = self.transmitted_spectrum_derivative(vels)
        min_dE = rate_to_deltaEmin(acquisition_time, rates, ders)

        slope_vel = vels[min_dE.argmin()]
        slope_rate = rates[min_dE.argmin()]
        return (slope_vel, slope_rate, min_dE.min())

    def get_deltaEmin_full(self, **kwargs):
        """Recursive calculation, should be fully correct"""
        # idk if confusing to let kwargs override these
        vels = kwargs.get('vels', self.source.linewidth*np.logspace(-6, 2, 10000))
        acquisition_time = kwargs.get('acquisition_time', self.acquisition_time)
        
        rates = self.transmitted_spectrum(vels)
        f = interp1d(rates, vels, fill_value='extrapolate')

        nnew = (acquisition_time*rates) + np.sqrt(acquisition_time * rates)
        vnew = f(nnew/acquisition_time) - vels
        vnew_return = vnew.min()
        if kwargs.get('return_vels', False):
            vnew_return = vnew

        slope_vel = vels[vnew.argmin()]
        slope_rate = rates[vnew.argmin()]
        return (slope_vel, slope_rate, vnew_return)

    #### I/O Stuff ####
    # Maybe this makes the class too big and ugly... I have no idea.
    def load_from_file(self, filename):
        """Grab actual spectrometer data from file"""
        df = pd.read_csv(filename, sep=r"\s+")
        df = df[list(name_map.values())]
        df = pd.concat(
            [
                df,
                pd.DataFrame({
                    file_name: getattr(self, my_name) for my_name, file_name in name_map.items()
                })
            ], 
            axis=0
        )
        df = df.groupby(df[name_map['measured_velocities']]).sum().reset_index()
        for my_name, file_name in name_map.items():
            setattr(self, my_name, df[file_name].values.tolist())
        self.measured_rates = np.asarray(self.measured_counts)/np.asarray(self.measured_times)
        self.measured_rate_errs = np.sqrt(np.asarray(self.measured_counts))/np.asarray(self.measured_times)
        return

    def clear_measurements(self):
        self.measured_velocities = []
        self.measured_counts = []
        self.measured_times = []
        return

    def plot(self, ax=None):
        if ax is None:
            fig = plt.figure()
            ax = fig.add_subplot(111)
        plt.errorbar(
            self.measured_velocities,
            self.measured_rates,
            yerr=self.measured_rate_errs,
            fmt='.',
            capsize=3,
        )

        plt.xlabel('velocity [mm/s]')
        plt.ylabel('rate [Hz]')
        plt.grid(alpha=0.3)

        ax2 = ax.twiny()
        velticks = ax.get_xticks()
        xrange = ax.get_xlim()
        Eticks = vel_to_E(velticks)
        values = Eticks
        plt.xticks(velticks, labels=Eticks)
        plt.xlim(xrange)
        plt.xlabel(r'$E$ [eV] - 14400')
        return

    # Hidden functions
    def _get_sensitivity_down_quark(self, deltaE, rs, **kwargs):
        """Physics lives in separate module"""
        return physics.get_limits(rs, deltaE)

    def _setup_measurement(self, p):
        # duplicated code from MossbauerMaterial... idk.
        required_pars = [
            'solid_angle_fraction',
            'detection_efficiency',
        ]
        check_pars(p, required_pars)
        self.__dict__.update(p)
        return

    def _transmission_integrand(self, E):
        """The actual function integrated -- convolution of source and absorber

        From eq. 2.25 in Mossbauer Spectroscopy and Transition Metal Chemistry
        (Gutlich, Bill, Trautwein)
        """
        # photons/s between E and E + dE
        transmission_integrand = (
            self.source.spectrum(E) 
            * self.solid_angle_fraction
            * self.detection_efficiency
            * self.absorber.transmission_fraction(E, self.velocity)
        )
        if hasattr(self, 'background_rate'):
            # TODO this was buggy so I'm adding background elsewhere for now. Probably
            # should delete this part.
            pass
            #transmission_integrand += self.background_rate
        return transmission_integrand

    def _transmission_derivative_integrand(self, E):
        """Derivative of the above function"""
        transmission_integrand = self._transmission_integrand(E)
        Ediff = E - (self.absorber.Eres - self.velocity)
        half_linewidth = self.absorber.linewidth/2.0
        extra_factor = (
            (half_linewidth**2.0)*Ediff 
            / (Ediff**2 + half_linewidth**2)**2
        )
        ## what is this 2 from?
        return 2 * transmission_integrand * extra_factor * self.absorber.thickness_normalized

    def _integrate_function_inf(self, vel, func):
        """Integrate function over (-inf, inf) numerically

        Velocity can be an array or float. If array, you get a spectrum, and
        if float you get back a float. All returned values are in (0, 1).
        """
        self.set_velocity(vel)
        args = [
            func,
            -np.inf,
            np.inf
        ]
        if hasattr(vel, '__len__'):
            return quad_vec(*args)[0]
        else:
            return quad(*args)[0]



if __name__=='__main__':
    natural_linewidth = 4.55e-9  # eV

    ### source parameters
    source_parameters = dict(
        Eres=0.0,
        linewidth=E_to_vel(natural_linewidth),
        total_activity=3.7e10 * 0.001  # 1 mCi
    )

    ### absorber parameters
    recoilless_fraction_A = 1.
    t_mgcm2 = 0.13  # potassium ferrocyanide from ritverc
    absorption_coefficient = 25.0  # cm^2 / mgFe57  (doublecheck)
    absorber_parameters = dict(
        Eres=0.,
        linewidth = E_to_vel(natural_linewidth),
        thickness_normalized=t_mgcm2 * absorption_coefficient * recoilless_fraction_A
    )

    ### measurement parameters
    detector_face_OD = 2 * 25.4  # mm
    detector_distance = 400.0  # mm
    measurement_parameters = dict(
        acquisition_time=3600.*24*31,
        solid_angle_fraction=(detector_face_OD**2)/(16*detector_distance**2),
    )

    moss = MossbauerMeasurement(
        source_parameters,
        absorber_parameters,
        measurement_parameters
    )

