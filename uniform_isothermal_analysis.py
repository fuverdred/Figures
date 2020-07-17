import numpy as np
np.random.seed(2004) # Get the same graphs every time

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
fig, (ax1, ax2) = plt.subplots(1,2, figsize=(11,4.5))

N = 100 # Number of droplets

freeze_times = np.sort(np.random.exponential(1/freeze_rate, N))
frac = [(N-i)/N for i in range(N)]
log_frac = np.log(frac)

ax1.scatter(freeze_times, frac, marker='o', label='Data',
            facecolors='none', edgecolors='k', zorder=5)
ax1.plot(freeze_times, np.exp(-freeze_rate * freeze_times), 'k--',
         label='True rate')

exp_fit, _ = curve_fit(lambda t, w: np.exp(-w*t),
                       freeze_times,
                       frac,
                       1)
t = np.linspace(0, max(freeze_times), 1000)
ax1.plot(t, np.exp(-exp_fit* t),
         color=tab20[0], zorder=10,
         label='Exponential fit')
ax1.text(0.4,0.2,'Fitted rate={:.3f} s$^{{-1}}$'.format(exp_fit[0]),
         color=tab20[0],
         transform=ax1.transAxes)

ax1.set_ylabel('Liquid Proportion')
ax1.set_xlabel('Time (s)')
ax1.legend()


ax2.scatter(freeze_times, log_frac, marker='o', facecolors='none',
            edgecolors='k', label='Data', zorder=5)
ax2.plot(freeze_times, -freeze_rate * freeze_times,
         'k--', label='True rate')

lin_fit = -np.dot(freeze_times, log_frac) / np.dot(freeze_times, freeze_times)
free_fit = np.polyfit(freeze_times, log_frac, 1)
ax2.plot(freeze_times, -lin_fit * freeze_times, color=tab20[2],
         label='Constrained log-linear fit',
         zorder=10)
ax2.text(0.45,0.65,'Fitted rate={:.3f} s$^{{-1}}$'.format(lin_fit),
         color=tab20[2],
         transform=ax2.transAxes)

ax2.plot(freeze_times, np.polyval(free_fit, freeze_times), color=tab20[4],
         label='Free log-linear fit', zorder=10)
ax2.text(0.55,0.6,'Fitted rate={:.3f} s$^{{-1}}$'.format(-free_fit[0]),
         color=tab20[4],
         transform=ax2.transAxes)
                                  

ax1.text(0.1, 0.1, 'A', fontsize=16, fontweight='bold', transform=ax1.transAxes)
ax2.text(0.1, 0.1, 'B', fontsize=16, fontweight='bold', transform=ax2.transAxes)

ax2.set_ylabel('log(Liquid Proportion)')
ax2.set_xlabel('Time (s)')
ax2.legend()
plt.show()
# =========================================================

# == VARYING CUTOFF LOGLIN ================================
'''
Simulate REPEATS number of experiments of N droplets, show how varying the
amount of data which is included in the log-linear fit changes the mean
absolute error to the true freeze_rate
'''

def quick_w_fit(times, log_frac):
    '''
    Perform a linear fit of log-linear data forced through 0
    '''
    return -np.dot(times, log_frac) / np.dot(times, times)

def w_free_fit(times, log_frac):
    '''
    Perform a linear fit of log-linear data without forcing it through 0
    '''
    w, _ = np.polyfit(times, log_frac, 1)
    return -w
    

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
    

##N = 150 # Number of droplets
##log_frac = np.array([np.log((N-i)/N) for i in range(N)])
##
### REPEATS rows of N exponentially distributed freeze times
##simulations = np.random.exponential(1/freeze_rate, (REPEATS, N))
##simulations.sort(axis=1) # Sort the rows into freeze time order
##
##percentages = range(75, 101)
##loglin_errors = []
##loglin_std = []
##free_errors = []
##free_std = []
##for percent in percentages:
##    index = get_cutoff(N, percent)
##    rates = np.apply_along_axis(lambda t: quick_w_fit(t[:index],
##                                                      log_frac[:index]),
##                                1,
##                                simulations)
##    rates2 = np.apply_along_axis(lambda t: w_free_fit(t[:index],
##                                                       log_frac[:index]),
##                                  1,
##                                  simulations)
##    loglin_errors.append(100*(np.mean(abs(rates - freeze_rate))/freeze_rate))
##    loglin_std.append((np.std(rates)/np.mean(rates))*loglin_errors[-1])
##
##    free_errors.append(100*(np.mean(abs(rates2 - freeze_rate))/freeze_rate))
##    free_std.append((np.std(rates2)/np.mean(rates2))*free_errors[-1])
##    
##    
##plt.scatter(percentages, loglin_errors, label='log-linear fit', color=tab20[0])
##plt.errorbar(percentages, loglin_errors, loglin_std, fmt='none', color=tab20[1],
##             capsize=3)
##
##plt.scatter(percentages, free_errors, label='log-linear free fit',
##            color=tab20[2])
##plt.errorbar(percentages, free_errors, free_std, fmt='none',color=tab20[3],
##             capsize=3)
##
##plt.ylabel('Percentage Mean Absolute Error')
##plt.xlabel('Percentage of data fitted')
###plt.legend()
##
##plt.show()
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
    cutoff = get_cutoff(N, 95)
    
    log_frac = np.array([np.log((N-i)/N) for i in range(N)])
    rates = np.apply_along_axis(lambda t:quick_w_fit(t[:cutoff],
                                                     log_frac[:cutoff]),
                                1,
                                simulations)
    errors.append(100*(np.mean(abs(rates - freeze_rate))/freeze_rate))
    std.append((np.std(rates)/np.mean(rates))*errors[-1])

plt.scatter(Ns, errors, zorder=10)
plt.errorbar(Ns, errors, std, c='k', fmt='none', capsize=3)

plt.ylabel('Percentage Mean Absolute Error')
plt.xlabel('Number of Droplets')

plt.show()
                                    
# =========================================================

# == VARYING CUTOFF CDFEXP=================================
##def cdf_fit(times, cum_frac):
##    func = lambda t, w: 1 - np.exp(-w * t)
##    popt, pcov = curve_fit(func, times, cum_frac, 0.02)
##    return popt
##
##N = 250 # Number of droplets
##cum_frac = [i/N for i in range(1, N+1)] # cumulative frac
##
##### REPEATS rows of N exponentially distributed freeze times
####simulations = np.random.exponential(1/freeze_rate, (REPEATS, N))
####simulations.sort(axis=1) # Sort the rows into freeze time order
##
##cdf_percentages = range(80, 101, 2)
##cdf_errors = []
##cdf_std = []
##for percent in cdf_percentages:
##    print(percent)
##    index = get_cutoff(N, percent)
##    rates = np.apply_along_axis(lambda t: cdf_fit(t[:index],
##                                                  cum_frac[:index]),
##                                1,
##                                simulations)
##    cdf_errors.append(100*(np.mean(abs(rates - freeze_rate))/freeze_rate))
##    cdf_std.append((np.std(rates)/np.mean(rates))*cdf_errors[-1])
##    
##    
##plt.scatter(cdf_percentages, cdf_errors, label='cdf fit', color=tab20[2])
##plt.errorbar(cdf_percentages, cdf_errors, cdf_std, fmt='none', color=tab20[3],
##             capsize=3)
##
###plt.ylabel('percentage mean absolute error')
###plt.xlabel('Percentage of data fitted')
###plt.legend()
##
###plt.show()
# =========================================================

# == VARYING CUTOFF EXP ===================================
##def exp_fit(times, frac):
##    func = lambda t, w: np.exp(-w * t)
##    popt, pcov = curve_fit(func, times, frac, 0.02)
##    return popt
##
##N = 250 # Number of droplets
##frac = [(N-i)/N for i in range(1, N+1)] # cumulative frac
##
##### REPEATS rows of N exponentially distributed freeze times
####simulations = np.random.exponential(1/freeze_rate, (REPEATS, N))
####simulations.sort(axis=1) # Sort the rows into freeze time order
##
##exp_percentages = range(80, 101, 2)
##exp_errors = []
##exp_std = []
##for percent in exp_percentages:
##    print(percent)
##    index = get_cutoff(N, percent)
##    rates = np.apply_along_axis(lambda t: exp_fit(t[:index],
##                                                  frac[:index]),
##                                1,
##                                simulations)
##    exp_errors.append(100*(np.mean(abs(rates - freeze_rate))/freeze_rate))
##    exp_std.append((np.std(rates)/np.mean(rates))*exp_errors[-1])
##    
##    
##plt.scatter(exp_percentages, exp_errors, label='exponential fit', color=tab20[4])
##plt.errorbar(exp_percentages, exp_errors, exp_std, fmt='none', color=tab20[5],
##             capsize=3)
##
###plt.ylabel('percentage mean absolute error')
###plt.xlabel('Percentage of data fitted')
##plt.legend()
##
##plt.show()
# =========================================================

