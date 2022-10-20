import numpy as np
from scipy.interpolate import splrep, splev
import random

dat = np.loadtxt('lakeshore.txt')

def lakeshore(V, data = dat):
    '''
    
    Parameters
    ----------
    V : float or array
        voltage (in V) for which temperature values needs to be estimated.
    data : np.ndarray
        each sub-element represents
        a (T, V(T), dV/dT) data point.

    Returns
    -------
    (est_temp, unc_temp): tuple of arrays
        values for estimated temperature(s) and 
        uncertainties.

    '''
    V = np.asarray(V) # turn possible float input into array
    T_values = [value[0] for value in data][::-1] # make a list of measured temperatures
    V_values = [value[1] for value in data][::-1] # and voltages
    fit = splrep(V_values, T_values)
    estimates = splev(V, fit)
    
    differences, est_unc = [], [] # create empty lists to store differences and final estimated uncertainties
    for i in range(1001):
        indices = random.sample(range(0, len(V_values)), 60) # get 60 indices with replacement
        indices.sort() # need to be sorted for splrep
        V_picked = [V_values[j] for j in indices] # get data points
        T_picked = [T_values[k] for k in indices]
        fit_alt = splrep(V_picked, T_picked) # fit bootstrapped population
        estimates_alt = splev(V, fit_alt) # evaluate fit for input V
        difference = abs(estimates - estimates_alt) 
        differences.append(difference) # store differences as array in list
    for index, voltage in enumerate(V):
        # for every voltage input, get all differences from differences list
        diffs = [differences[l][index] for l in range(len(differences))] 
        # compute standard deviation of differences
        est_unc.append(np.std(diffs))
    est_unc = np.array(est_unc)
    return (estimates, est_unc)


