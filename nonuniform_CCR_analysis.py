import numpy as np

import matplotlib.pyplot as plt
import matplotlib.cm as cm
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

from scipy.optimize import curve_fit
from scipy.integrate import trapz, quad
from scipy.stats import norm, lognorm, uniform, loguniform
from scipy.interpolate import interp1d

tab20 = [cm.tab20(i) for i in np.linspace(0, 1, 20)]

np.random.seed(9438528)

# == CONSTANTS ======================================
T_M = 273 # [K]

A = 1
B = 1
T_0 = 260 # [K]
ALPHA = 1/60 # [Ks^-1] abs(cooling rate)

N = 100 # Number of droplets

SCALER = 150
MAX_TIME = 1200
MIN_TEMP = 250
# ===================================================

# == FUNCTIONS ======================================
def J_T(T, B, T_0):
    return np.exp(-B * (T - T_0)) # [s^-1]

def P_liq_T(T, area, B, T_0):
    prefactor = area / ALPHA
    J_integral = np.exp(-B*(T_M-T_0))-np.exp(-B*(T-T_0))
    return np.exp(prefactor * J_integral)

def T_freeze(p, area, B, T_0):
    return (np.log(np.exp(-B*(T_M-T_0))-((ALPHA * B * np.log(p)) / area))/-B) + T_0

def K_to_degC(T):
    return T - T_M

def T_t(t):
    return T_M - (ALPHA * t)
# ===================================================

# == NON-UNIFORM SIMULATION =========================
'''
Simulate an array of N non-uniform droplets being linearly cooled.
steps:
1.  Generate N areas from a given distribution
2.  Generate N uniformly distributed random numbers in [0,1)
3.  Multiply each area by J_HET and generate the liquid probability curve
    as a function of temperature for that freezing rate.
4.  Map the corresponding random number to the liquid probability curve
    corresponding to the given droplet.
'''

fig, ax = plt.subplots(1, figsize=(7, 5))

rv = lognorm(1.5)
LOWER, UPPER = rv.ppf(0.001)/SCALER, rv.ppf(0.999)/SCALER
areas = rv.rvs(N) / SCALER # [cm^2]
ps = np.random.random(N) # probability for each droplet
mean_area = rv.mean() / SCALER
std_area = rv.std() / SCALER

Ts = T_freeze(ps, areas, B, T_0) # Simulated freezing temperatures

data = np.c_[areas, ps, Ts]

Ts.sort()

frac = [i/N for i in range(1, N+1)] # Liquid proportion


Ts_fit = np.linspace(Ts[0], Ts[-1], 250)
T_0_guess = Ts[int(N/2)]
func = lambda T, B, T_0: P_liq_T(T, mean_area, B, T_0)
popt, pcov = curve_fit(func, Ts, frac, [1, T_0_guess])

plt.plot(K_to_degC(Ts_fit), func(Ts_fit, *popt), color=tab20[2], ls='--',
         label='Uniform fit', zorder=10)
plt.plot(K_to_degC(Ts_fit), P_liq_T(Ts_fit, mean_area, B, T_0), color=tab20[6],
         ls='--', label='Uniform mean area', zorder=10)

plt.scatter(K_to_degC(Ts), frac, label='Simulated data',
            facecolors='none', edgecolors=tab20[0])

plt.xlabel('Temperature (\u00B0C)')
plt.ylabel('Liquid Proportion')

#plt.show()
# ===================================================

# == AREA HISTOGRAM =================================
axins = inset_axes(ax, width=2, height=1.5, loc=4,
                   bbox_to_anchor=(0.95, 0.1),
                   bbox_transform=ax.transAxes)
_ = axins.hist(np.log(areas), 20)
axins.set_ylabel('Count')
axins.set_xlabel('log[Area (cm$^2$)]')
# ===================================================

# == NON-UNIFORM NUMERICAL ==========================
def J_t(t, B, T_0):
    T = T_t(t)
    return np.exp(-B * (T - T_0))

def get_g_t_spline(rv, N=500):
    '''
    Find how the distribution rv changes with time based on N points
    between t_low and t_high. Then return a spline of the points allowing
    the mean area as a function of t to be found efficiently.
    '''
    ts = np.linspace(0, MAX_TIME, N)
    mean_areas = []
    for t in ts:
        T = T_t(t) # Temperature at time t
        func = lambda A: P_liq_T(T, A, B, T_0) * rv.pdf(A * SCALER)
        normalize, _ = quad(func, LOWER, UPPER)
        func2 = lambda A: A * func(A) / normalize
        mean_area, _ = quad(func2, LOWER, UPPER)
        mean_areas.append(mean_area)
    spline = interp1d(ts, mean_areas, kind='cubic')
    return spline

def get_g_trapz(rv):
    ts = np.linspace(0, MAX_TIME)
    T = T_t(t)
    As = np.linspace(LOWER, UPPER, 10000)
    g_a = rv.pdf(As * SCALER)
    mean_areas = []
    for t in ts:
        normalize = trapz(P_liq_T(T, As, B, T_0)*g_a)

def nonhom_P_liq_t(t, g_spline, B, T_0):
    func = lambda t: g_spline(t) * J_t(t, B, T_0)
    integral, _ = quad(func, 0, t)
    return np.exp(-integral)

g_spline = get_g_t_spline(rv)

ts = np.linspace(450, MAX_TIME)
Ps = [nonhom_P_liq_t(t, g_spline, B, T_0) for t in ts]

ax.plot(K_to_degC(T_t(ts)), Ps, color=tab20[4], ls='--',
        label='Non-uniform fit', zorder=10)

ax.legend()

plt.figure()
plt.plot(K_to_degC(Ts), J_T(Ts, B, T_0), color=tab20[6],
         label='True J$_\\mathrm{het}$')
plt.plot(K_to_degC(Ts), J_T(Ts, *popt), color=tab20[2],
         label='Assuming uniform sample J$_\\mathrm{het}$')

plt.yscale('log')
plt.ylabel('J$_\\mathrm{het}$ (cm$^{-2}$s$^{-1}$)')
plt.xlabel('Temperature (\u00B0C)')
plt.legend()
plt.show()
# ===================================================
