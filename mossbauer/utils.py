from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt

name_map = {
    'measured_velocities': 'vel',
    'measured_times': 'seconds',
    'measured_counts': 'count',
}

def get_current_activity(half_life_days, activity, date):
    tdiff_seconds = (datetime.now() - datetime.strptime(date, '%Y%m%d')).total_seconds()
    return activity*((0.5)**(tdiff_seconds/(3600*24)/half_life_days))


def check_pars(p, req):
    assert set(req).issubset(p.keys()), (
        'Not all required pars present:\n'
        + str(req)
        )
    return


def vel_to_E(vel, E0=14.4e3):
    c = 3e11  # mm/s
    return E0*vel/c


def E_to_vel(E, E0=14.4e3):
    c = 3e11  # mm/s
    return E*c/E0


def lorentzian(x, x0, gamma, amplitude=1):
    """Lorentzian function with max default to 1

    Note: gamma is the HALF width at half max, so
    if you have a "linewidth" then you want gamma = linewidth/2
    """
    return amplitude/( 1 + ((x-x0)/gamma)**2 )


def lorentzian_norm(x, x0, gamma):
    amplitude = 1.0 / np.pi / gamma
    return lorentzian(x, x0, gamma, amplitude)


def rate_to_deltaEmin(acquisition_time, rate, rate_derivative):
    required_dN = np.sqrt(rate*acquisition_time)
    min_dE = required_dN/(acquisition_time*rate_derivative)
    return min_dE


def add_energy_axis(ax):
    ax2 = ax.twiny()
    velticks = ax.get_xticks()
    xrange = ax.get_xlim()
    Eticks = vel_to_E(velticks)
    values = Eticks
    plt.xticks(velticks, labels=Eticks)
    plt.xlim(xrange)
    plt.xlabel(r'$E$ [eV] - 14400')
