import numpy as np

import matplotlib.pyplot as plt
import matplotlib.cm as cm

from scipy.optimize import curve_fit
from scipy.integrate import trapz, quad
from scipy.stats import norm, lognorm, uniform, loguniform
from scipy.interpolate import interp1d

tab20 = [cm.tab20(i) for i in np.linspace(0, 1, 20)]

def freezing_rate(A, J_het=200.24):
    return A * J_het

def g_t_lognorm(t, A, g, J_het=200.24):
    return np.exp(-J_het*A*t) * g.pdf(A)

def expectation_g(t):
    # First renormalise
    func = lambda A: g_t(t, A, g)
    norm, _ = quad(func, lower, upper)
    E = lambda A: A * g_t(t, A, g)
    expected, _ = quad(E, lower, upper)
    return expected/norm

def P_liq(t):
    func = lambda t: J_het * expectation_g(t)
    integral, _ = quad(func, 0, t)
    return np.exp(-integral)

def t_liq(p, w):
    return -np.log(p) / w

# == CONSTANTS ======================
A = 0.015

N = 1000
frac = np.linspace(1, 0, N)
w = 2
T_M = 273
SCALER = 100
MAX_TIME = 500
# ===================================

# == DISTRIBUTIONS ==================
rv_lognorm = lognorm(1)
g_lognorm = lambda A: rv_lognorm.pdf(A*SCALER)

LOWER, UPPER = rv_lognorm.ppf(0.01)/SCALER, rv_lognorm.ppf(0.99)/SCALER

rv_loguni = loguniform(LOWER*SCALER, UPPER*SCALER)
g_loguni = lambda A: rv_loguni.pdf(A*SCALER)
# ===================================

# == DROPLET SIMULATIONS ============

# All droplets the same
p = np.random.random(N)
uniform_ts = sorted(t_liq(p, A * w))

# log uniform
loguni_areas = rv_loguni.rvs(N) / SCALER
p = np.random.random(N)
loguni_ts = sorted(t_liq(p, loguni_areas*w))

# log normal
lognorm_areas = rv_lognorm.rvs(N) / SCALER
p = np.random.random(N)
lognorm_ts = sorted(t_liq(p, lognorm_areas*w))

# ===================================

# == FITS ===========================

# All droplets the same
t_fit = np.linspace(0, MAX_TIME, 250)
plt.plot(t_fit, np.exp(-w * A *t_fit), c=tab20[1])

def get_g_t_spline(rv, N=50):
    '''
    Find how the distribution rv changes with time based on N points up
    to MAX_TIME. Then return a spline of the points allowing the mean
    area as a function of t to be found efficiently.
    '''
    ts = np.linspace(0, MAX_TIME, N)
    mean_areas = []
    for t in ts:
        func = lambda A: np.exp(-w * A * t) * rv.pdf(A * SCALER)
        normalize, _ = quad(func, LOWER, UPPER)
        func2 = lambda A: A * func(A) / normalize
        mean_area, _ = quad(func2, LOWER, UPPER)
        mean_areas.append(mean_area)
    spline = interp1d(ts, mean_areas, kind='cubic')
    return spline

# log uniform
spline_loguni = get_g_t_spline(rv_loguni)
p_loguni = []
for t in t_fit:
    integral, _ = quad(spline_loguni, 0, t)
    p_loguni.append(np.exp(-integral * w))

plt.plot(t_fit, p_loguni, c=tab20[3])

# log normal
spline_lognorm = get_g_t_spline(rv_lognorm)
p_lognorm = []
for t in t_fit:
    integral, _ = quad(spline_lognorm, 0, t)
    p_lognorm.append(np.exp(-integral * w))

plt.plot(t_fit, p_lognorm, c=tab20[5])

# ===================================

plt.yscale('log')

plt.scatter(uniform_ts, frac, color=tab20[0], label='Uniform Droplets')
plt.scatter(loguni_ts, frac, color=tab20[2], label='Log-uniform Distribution')
plt.scatter(lognorm_ts, frac, color=tab20[4], label='Log-normal Distribution')



plt.legend()
plt.ylabel('Liquid Proportion')
plt.xlabel('Time (s)')

plt.show()
