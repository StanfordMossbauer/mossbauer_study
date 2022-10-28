from scipy.stats import cauchy, binom, poisson
from scipy.optimize import curve_fit, minimize
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import pandas as pd

name_map = {
    'velocities': 'vel',
    'times': 'seconds',
    'counts': 'count',
}

def vel_to_E(vel, E=14.4e3):
    c = 3e11  # mm/s
    return E*vel/c

def lorentzian(x, x0, gamma):
    """Lorentzian function with max = 1"""
    return (1.0/( 1 + ((x-x0)/gamma)**2 ) )


def mossbauer_spectrum(vel, R0, velres, gamma, k):
    """Evaluate Mossbauer-reduced rate(s)

    vel : doppler velocity
    R0 : asymptotic rate
    velres : velocity of resonance
    gamma : linewidth
    k : coefficient of absorption
    """
    return R0 * np.exp(-1*k*lorentzian(vel, velres, gamma))


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

class MossbauerMeasurement:
    def __init__(self, **kwargs):
        # take in truth information of mossbauer setup
        # construct an expected pdf
        default_kwargs = {
            'absorber_thickness_mfp': 3.23,  # thickness / mfp
            'source_detector_distance': 30.,  # cm
            'detector_diameter': 2.54*2,  # cm
            'detection_efficiency': 0.2, 
            'resonance_velocity': -0.159,  # mm/s
            'resonance_linewidth': 0.1,  # mm/s
            'source_rate': 75e3,  # count/s
            'line_intensity': 0.0916  # 14 keV x-rays per decay (avg)
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

    def get_expected_counts(self, velocity, time):
        absorption_intensity = lorentzian(
            velocity,
            self.resonance_velocity,
            self.resonance_linewidth
        )
        mossbauer_rate = mossbauer_spectrum(
            velocity,
            self.R0,
            self.resonance_velocity,
            self.resonance_linewidth,
            self.absorber_thickness_mfp,
        )
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
            self.absorber_thickness_mfp, 
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
        detection_efficiency=0.1,
        line_intensity=0.0916,  # 14 keV x-rays per decay (avg)
        resonance_velocity=-0.159,
        resonance_linewidth=0.14,
        absorber_thickness_mfp=3.23/12,
    )
    acq = SimulatedMossbauerMeasurement(**mossbauer_pars)
    realmbm = MossbauerMeasurement(**mossbauer_pars)
