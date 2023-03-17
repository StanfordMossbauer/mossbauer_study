import numpy as np

name_map = {
    'measured_velocities': 'vel',
    'measured_times': 'seconds',
    'measured_counts': 'count',
}


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


