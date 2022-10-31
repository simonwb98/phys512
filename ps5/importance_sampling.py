import numpy as np
# from matplotlib import pyplot as plt

mcmc_params = np.loadtxt('mcmc_params.txt')
tau_pol = 0.054
tau_unc = 0.0074

# first step: importance sampling
# selecting only last 5000 steps
mcmc_params = mcmc_params[5001:,:]
# get tau values
tau = mcmc_params[:,3]
# define delta chi squared
delta_chi = (tau - tau_pol)**2/tau_unc**2
# define weights
w = np.exp(-0.5*delta_chi)
# normalize weights
w = w/np.sum(w) 
# importance sample parameters
sampled_params = np.zeros(6)
for j in range(6):
    sampled_params[j] = sum(mcmc_params[:,j]*w)
# get covariance matrix
cov_mat = np.cov(mcmc_params, rowvar=False, aweights=w)

par_errs = np.sqrt(np.diag(cov_mat))