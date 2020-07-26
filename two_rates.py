import numpy as np

import matplotlib.pyplot as plt

from scipy.integrate import quad

'''
Observe how two distinct populations freeze with time in an isothermal
experiment.
'''

# == CONSTANTS =====================

J_HET = 0.02 # [cm^-2s^-1]

N_1 = 500
N_2 = 500
N = N_1 + N_2

A_1 = 1 # [cm^2]
A_2 = 5 # [cm^2]

# ==================================

# == FUNCTIONS =====================
def p_liq_t(t, A):
    return np.exp(-J_HET * A * t)

def t_liq_p(p, A):
    return -np.log(p) / (J_HET * A) # [s]
# ==================================

# == SIMULATE ======================
ps = np.random.random(N) # generate uniform random numbers
ts_1 = sorted(t_liq_p(ps[:N_1], A_1))
ts_2 = sorted(t_liq_p(ps[N_1:], A_2))

combined_ts = sorted(ts_1 + ts_2)
liquid_proportion = [(N-i)/N for i in range(N)]

plt.yscale('log')
plt.ylabel('Liquid Proportion')
plt.xlabel('Time (s)')
plt.scatter(combined_ts, liquid_proportion)
# ==================================

# == ANALYTICAL ====================
def expectation_g(t):
    '''
    Return the expected value of the surface area distribution.

    For the case of two dirac delta functions it is a simple weighted
    average
    '''
    pop_1 = np.exp(-J_HET * A_1 * t)
    pop_2 = np.exp(-J_HET * A_2 * t)
    frac_1 = pop_1 / (pop_1 + pop_2)
    frac_2 = pop_2 / (pop_1 + pop_2)
    return frac_1 * A_1 + frac_2 * A_2 # Weighted average

def p_liq_integral(t):
    integral, _ = quad(expectation_g, 0, t)
    return np.exp(-integral * J_HET)

ts = np.linspace(0, max(combined_ts), 100)
p_liq = [p_liq_integral(t) for t in ts]

plt.plot(ts, p_liq, 'k--')
# ==================================

plt.subplots(3, figsize=(6,6))
times = (0,10,50)
for i in range(3):
    t = times[i]
    plot_no = int('31'+str(i+1))
    plt.subplot(plot_no)

    pop1 = np.exp(-J_HET * A_1 * t)
    pop2 = np.exp(-J_HET * A_2 * t)

    plt.plot([A_1, A_1], [0, pop1])
    plt.plot([A_2, A_2], [0, pop2])

    plt.text(1.4, pop1*0.9, 't={:d}s'.format(t))

    avg = expectation_g(t)
    plt.plot([avg, avg], [0, pop1], 'k--')
    plt.ylabel('Area populations')
    plt.xlabel('Area')

plt.show()
