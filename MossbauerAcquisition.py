from scipy.stats import cauchy, binom, poisson
from scipy.optimize import curve_fit, minimize
from scipy.special import jv
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import pandas as pd
from os.path import join

name_map = {
    'velocities': 'vel',
    'times': 'seconds',
    'counts': 'count',
}

def vel_to_E(vel, E0=14.4e3):
    c = 3e11  # mm/s
    return E0*vel/c

def E_to_vel(E, E0=14.4e3):
    c = 3e11  # mm/s
    return E*c/E0

def lorentzian(x, x0, gamma, amplitude=1):
    """Lorentzian function with max default to 1

    Note: gamma is the HALF width at half max
    """
    return amplitude/( 1 + ((x-x0)/gamma)**2 )

def lorentzian_norm(x, x0, gamma):
    amplitude = 1.0 / np.pi / gamma
    return lorentzian(x, x0, gamma, amplitude)


def mossbauer_spectrum(vel, R0, velres, gamma, t_mgcm2, bkg=0.0, absorption_coeff=25.0):
    """Evaluate Mossbauer-reduced rate(s)

    vel : doppler velocity
    R0 : asymptotic rate
    velres : velocity of resonance
    gamma : natural linewidth
    t_mgcm2 : thickness in mgFe57/cm^2
    absorption_coeff : coefficient of absorption (cm^2/mgFe57)
    """
    t = t_mgcm2 * absorption_coeff  # t/tnat -- effective no. mfps
    contrast = 1 - (np.exp(-t/2) * jv(0, t/2*1j).real)
    width = 2 * gamma * (1 + (0.135*t))  # account for thickness broadening
    absorbed_fraction = lorentzian(vel, velres, width, contrast)
    return (R0 - bkg) * (1 - absorbed_fraction) + bkg

def composite_mossbauer_spectrum(vel, R0, offset, spacing, gamma, t_mgcm2, bkg=0.0, absorption_coeff=25.0):
    """Evaluate Mossbauer-reduced rate(s)

    vel : doppler velocity
    R0 : asymptotic rate
    velres : velocity of resonance
    gamma : natural linewidth
    t_mgcm2 : thickness in mgFe57/cm^2
    absorption_coeff : coefficient of absorption (cm^2/mgFe57)
    """
    t = t_mgcm2 * absorption_coeff  # t/tnat -- effective no. mfps
    contrast = 1 - (np.exp(-t/2) * jv(0, t/2*1j).real)
    width = 2 * gamma * (1 + (0.135*t))  # account for thickness broadening
    
    split_ratio = (3, 2, 1, 1, 2, 3)
    spacing = 2.0
    leftmost_res = -2.5*spacing + offset
    rate = np.zeros_like(vel)
    split_ratio = (3, 2, 1, 1, 2, 3)
    leftmost_res = -2.5*spacing + offset
    absorbed_fraction = np.zeros_like(vel)
    for i, r in enumerate(split_ratio):
        absorbed_fraction += lorentzian(vel, leftmost_res + (i*spacing), width, r/np.sum(split_ratio)*contrast)
    return (R0 - bkg) * (1 - absorbed_fraction) + bkg


def negative_lnl(pars, measured_velocities, measured_times, measured_counts):
    """Negative poisson log-likelihood to minimize"""
    mossbauer_rates = mossbauer_spectrum(measured_velocities, *pars)
    expected_counts = mossbauer_rates * measured_times
    lnl = measured_counts * np.log(expected_counts) - expected_counts
    return -lnl.sum()


def fit_mossbauer_spectrum(velocities, times, counts, p0, method=None):
    res = minimize(
        negative_lnl,
        p0,
        args=(
            velocities,
            times,
            counts,
        ),
        options={'maxiter': 1000000},
        #method=method,
        #method='Nelder-Mead',
    )
    return res


def chisqr(obs, exp, error):
    chisqr = 0
    for i in range(len(obs)):
        chisqr = chisqr + ((obs[i]-exp[i])**2)/(error[i]**2)
    return chisqr


class MossbauerMeasurement:
    def __init__(self, **kwargs):
        # take in truth information of mossbauer setup
        # construct an expected pdf
        default_kwargs = {
            'absorber_thickness_mgcm2': 0.13,  # thickness / mfp
            'source_detector_distance': 30.,  # cm
            'detector_diameter': 2.54*2,  # cm
            'detection_efficiency': 0.2, 
            'resonance_velocity': -0.159,  # mm/s
            'resonance_linewidth': 0.1,  # mm/s
            'source_rate': 75e3,  # count/s
            'line_intensity': 0.0916,  # 14 keV x-rays per decay (avg)
            'background': 0.0  # Hz
        }
        for key, val in default_kwargs.items():
            setattr(self, key, kwargs.get(key, val))
        self.solid_angle_fraction = (
            self.detector_diameter**2.0
            / (16.0 * self.source_detector_distance**2.0)
        )
        self.R0 = self.source_rate * self.line_intensity * self.solid_angle_fraction * self.detection_efficiency
        self.clear_measurements()
        return

    def clear_measurements(self):
        self.velocities = []
        self.counts = []
        self.times = []
        return

    def update(self):
        self.rates = np.asarray(self.counts)/np.asarray(self.times)
        self.rate_errs = np.sqrt(np.asarray(self.counts))/np.asarray(self.times)
        return

    def get_expected_rates(self, velocity):
        mossbauer_rate = mossbauer_spectrum(
            velocity,
            self.R0,
            self.resonance_velocity,
            self.resonance_linewidth,
            self.absorber_thickness_mgcm2,
            self.background,
        )
        return mossbauer_rate

    def get_expected_counts(self, velocity, time):
        mossbauer_rate = self.get_expected_rates(velocity)
        expected_count = mossbauer_rate * time
        return expected_count

    def fit_spectrum(self, p0=None, plot=False):
        if p0 is None:
            p0 = self.get_p0()
        res = fit_mossbauer_spectrum(
            np.asarray(self.velocities),
            np.asarray(self.times),
            np.asarray(self.counts),
            p0
        )
        if plot:
            fig = plt.figure()
            ax = fig.add_subplot(111)
            xx = np.linspace(np.min(self.velocities), np.max(self.velocities), 1000)
            yy = mossbauer_spectrum(xx, *res.x)
            plt.plot(xx, yy, 'k--')
            plt.xlabel('vel [mm/s]')
            plt.ylabel('rate [Hz]')
            self.plot(ax)
        return res

    def plot(self, ax=None):
        if ax is None:
            fig = plt.figure()
            ax = fig.add_subplot(111)
        plt.errorbar(
            self.velocities,
            self.rates,
            yerr=self.rate_errs,
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

    def add_measurements(
        self, measurement_velocities, measurement_times, measurement_counts
    ):
        """list of vels, list of times
        """
        if type(measurement_times) in (int, float):
            measurement_times = [measurement_times]*len(measurement_velocities)
        self.velocities.extend(measurement_velocities)
        self.times.extend(measurement_times)
        self.counts.extend(measurement_counts)
        self.update()
        return

    def load_from_file(self, filename):
        df = pd.read_csv(filename, sep=r"\s+")
        df = df[list(name_map.values())]
        df = pd.concat(
            [
                df,
                pd.DataFrame({file_name: getattr(self, my_name) for my_name, file_name in name_map.items()})
            ], 
            axis=0
        )
        df = df.groupby(df[name_map['velocities']]).sum().reset_index()
        for my_name, file_name in name_map.items():
            setattr(self, my_name, df[file_name].values.tolist())
        self.update()
        return

    def get_p0(self):
        return [
            self.R0,
            self.resonance_velocity,
            self.resonance_linewidth,
            self.absorber_thickness_mgcm2, 
        ]

class SimulatedMossbauerMeasurement(MossbauerMeasurement):
    def simulate_measurements(self, measurement_velocities, measurement_times):
        """list of vels, list of times
        """
        if type(measurement_times) in (int, float):
            measurement_times = [measurement_times]*len(measurement_velocities)
        counts = self.acquire(measurement_velocities, measurement_times)
        self.add_measurements(measurement_velocities, measurement_times, counts)
        return

    def acquire(self, velocities, times):
        # return number of events if we take data for [time] s at [velocity] mm/sec
        expected_count = self.get_expected_counts(np.asarray(velocities), np.asarray(times))
        return poisson.rvs(expected_count)



if __name__=='__main__':
    mossbauer_pars = dict(
        source_rate=0.001 * (3.7e10),  # Bq
        detector_diameter=2.54*2,  # cm
        source_detector_distance=2.54*32,  # cm
        detection_efficiency=0.099,
        line_intensity=0.0916,  # 14 keV x-rays per decay (avg)
        resonance_velocity=-0.159,
        resonance_linewidth=0.1,
        absorber_thickness_mgcm2=0.13*.8*.8,
        background=50.0,
    )
    acq = SimulatedMossbauerMeasurement(**mossbauer_pars)
    realmbm = MossbauerMeasurement(**mossbauer_pars)
    data_dir = '/Users/josephhowlett/research/mossbauer/analysis/co57-mossbauer-spectra/'
    filename = '20221026_1007.dat'
    realmbm.load_from_file(join(data_dir, filename))
    realmbm.plot()
    vels = np.asarray(realmbm.velocities) 
    times = np.asarray(realmbm.times)
    expected_count = acq.get_expected_counts(vels, times)
    expected_rate = expected_count/times
    plt.plot(vels, expected_rate)
    print(chisqr(realmbm.rates, expected_rate, realmbm.rate_errs))
    print(len(vels) - 1 - 2)
    plt.show()
