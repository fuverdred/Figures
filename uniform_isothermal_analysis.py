import numpy as np

import matplotlib.pyplot as plt
import matplotlib.cm as cm

from scipy.optimize import curve_fit

'''
A script for simulating isothermal experiments to examine how to analyse
them.
'''

# == CONSTANTS ============================================
tab20 = cm.tab20(np.linspace(0,1,20)) # Colour scheme

J_het = 0.02 # [cm^-2s^-1]
A = 1 # [cm^2]

freeze_rate = J_het * A # [s^-1]

REPEATS = 100000 # Number of times to generate N droplets
# =========================================================

# == EXAMPLE GRAPH ========================================
'''
Plot an example simulated experiment, along with lines of best fit
for varying amounts of the data
'''
N = 100 # Number of droplets

freeze_times = np.sort(np.random.exponential(1/freeze_rate, N))
frac = [(N-i)/N for i in range(N)]
log_frac = np.log(frac)

plt.scatter(freeze_times, log_frac, marker='x', c='k', label='Data',
            zorder=10)
plt.plot(freeze_times, -freeze_rate * freeze_times,
         'k--', label='True rate')

for i, cutoff in enumerate((1, 0.9, 0.8, 0.7)):
    keep = int(N*cutoff)
    rate = (np.dot(freeze_times[:keep], log_frac[:keep])
            / np.dot(freeze_times[:keep], freeze_times[:keep])) # fixed 0 intercept
    plt.plot(freeze_times, rate * freeze_times, color=tab20[2*i],
             label='{:.0f}%'.format(100*cutoff))

plt.ylabel('log(liquid proportion)')
plt.xlabel('Time (s)')
plt.legend()
plt.show()
# =========================================================

# == VARYING CUTOFF =======================================
'''
Simulate REPEATS number of experiments of N droplets, show how varying the
amount of data which is included in the log-linear fit changes the mean
absolute error to the true freeze_rate
'''

def quick_w_fit(times, log_frac):
    '''
    Perform a linear fit of log-linear data
    '''
    return -np.dot(times, log_frac) / np.dot(times, times)

def slow_w_fit(times, log_frac):
    '''
    Fit using curve_fit with weightings
    '''
    func = lambda x, w: -w * x
    popt, pcov = curve_fit(func,
                           times,
                           log_frac,
                           sigma=np.gradient(log_frac)**2)
    return popt

def get_cutoff(N, percent):
    '''
    Return the list index approximately corresponding to the percentage
    '''
    return round((percent/100) * N)
    

N = 100 # Number of droplets
log_frac = np.array([np.log((N-i)/N) for i in range(N)])

# REPEATS rows of N exponentially distributed freeze times
simulations = np.random.exponential(1/freeze_rate, (REPEATS, N))
simulations.sort(axis=1) # Sort the rows into freeze time order

percentages = range(25, 101)
errors = []
std = []
for percent in percentages:
    index = get_cutoff(N, percent)
    rates = np.apply_along_axis(lambda t: quick_w_fit(t[:index],
                                                      log_frac[:index]),
                                1,
                                simulations)
    errors.append(100*(np.mean(abs(rates - freeze_rate))/freeze_rate))
    std.append((np.std(rates)/np.mean(rates))*errors[-1])
    
    
plt.scatter(percentages, errors, label='Linear fit')
##plt.errorbar(percentages, errors, std, fmt='none', c='k',
##             capsize=3)

plt.ylabel('percentage mean absolute error')
plt.xlabel('Percentage of data fitted')
#plt.legend()

plt.show()
# =========================================================

# == VARYING N ============================================
'''
Plot how the percentage absolute mean error varies with N
'''
Ns = list(range(50, 501, 50))

errors = []
std = []
for N in Ns:
    simulations = np.random.exponential(1/freeze_rate, (REPEATS, N))
    simulations.sort(axis=1) # Sort the rows into freeze time order
    
    log_frac = np.array([np.log((N-i)/N) for i in range(N)])
    rates = np.apply_along_axis(lambda t:quick_w_fit(t[:N],
                                                     log_frac),
                                1,
                                simulations)
    errors.append(100*(np.mean(abs(rates - freeze_rate))/freeze_rate))
    std.append((np.std(rates)/np.mean(rates))*errors[-1])

plt.scatter(Ns, errors, zorder=10)
plt.errorbar(Ns, errors, std, c='k', fmt='none', capsize=3)

plt.ylabel('Percentage mean absolute error')
plt.xlabel('Number of droplets')

plt.show()
                                    
# =========================================================
