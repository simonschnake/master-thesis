from src.binned_estimation import binned_estimation
import numpy as np
import matplotlib.pyplot as plt

def plot(x, y, name, variable, color, textpos_x=1, textpos_y=0):
    res = binned_estimation(x, y, bins=25)
    x, mu, sigma, std_mu, std_sigma = res[0], res[1], res[2], res[3], res[4]
    y = 0
    if variable == 'R':
        y = mu / x
        err_y = std_mu / x
    elif variable == 'deltaE':
        y = mu - x
    elif variable == 'res':
        y = sigma/np.sqrt(x)
        err_y = std_sigma/np.sqrt(x)
    else: raise Exception("not a valid key for variable")
    plt.plot(x, y, '-',  color=color)
    plt.errorbar(x, y, yerr=err_y, color=color)
    plt.text(x[-1] + textpos_x, y[-1]+textpos_y, name, ha='left',
             va='center', size=15, color=color, weight='bold')


