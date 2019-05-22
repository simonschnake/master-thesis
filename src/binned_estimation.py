import numpy as np
from scipy.stats import norm
from scipy.stats import t
import scipy.optimize as optimize
import matplotlib.pyplot as plt

def binned_estimation(x, y, bins=20):
    '''binned_estimation devides the data in bins defined by bins and
    estimates the mean and sigma of these components x, y.
    where x and y are numpy arrarrays. bins can be a integer, a list
    or a numpy array. if bins is an integer i , the the space between
    min(x) and max(x) is binned in equal distance with i as the number
    of bins. Else the values inside the list or numpy array are used
    as the edges of the bins. For each bin the mean and the std is
    calculated. Also the error of these estimations (sigma/n**1/2,
    sigma/)n**1/2 is calculated.
    
    returns len(bins-1) array with [bin_middle, mu, sigma, std_mu, std_sigma]

    ÜBERARBEITEN
    '''


    
    if len(x) != len(y):
        raise ValueError('xy[0] and xy[1] must be of the same length')

    if type(bins) is int:
        start = np.min(x)
        stop = np.max(x)
        diff = stop - start
        step = diff/bins
        print(step)
        bins = np.arange(start, stop, step)


    if type(bins) is list:
        bins = np.array(bins)

    if np.any(np.diff(bins) <= 0): # check if bins are monotonically increasing
        raise Exception('bins must be monotonically increasing')
              
    digit = np.digitize(x, bins=bins)

    res = []
    for i in range(len(bins)):
        bin_middle = (bins[i-1]+bins[i])/2
        arr = y[digit == i]
        n = len(arr)
        if len(arr) > 0:
            mu = np.mean(arr)
            sigma = np.std(arr)
            std_mu = sigma/np.sqrt(n)
            std_sigma = sigma/np.sqrt(2*n)
            res.append([bin_middle, mu, sigma, std_mu, std_sigma])
    res = np.array(res)
    res = res.T
    return res

