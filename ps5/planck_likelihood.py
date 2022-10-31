import numpy as np
import camb
from matplotlib import pyplot as plt
import time

pars=np.asarray([69, 0.022, 0.12, 0.06, 2.1e-9, 0.95])
# pars=np.asarray([50, 3, 2., 4., 1e-13, 1])
planck=np.loadtxt('COM_PowerSpect_CMB-TT-full_R3.01.txt',skiprows=1)
ell=planck[:,0]
spec=planck[:,1]
errs=0.5*(planck[:,2]+planck[:,3]);

def get_chs(model, data):
    return sum((model-data)**2/errs**2)

def get_spectrum(pars,lmax=3000):
    #print('pars are ',pars)
    H0=pars[0]
    ombh2=pars[1]
    omch2=pars[2]
    tau=pars[3]
    As=pars[4]
    ns=pars[5]
    pars=camb.CAMBparams()
    pars.set_cosmology(H0=H0,ombh2=ombh2,omch2=omch2,mnu=0.06,omk=0,tau=tau)
    pars.InitPower.set_params(As=As,ns=ns,r=0)
    pars.set_for_lmax(lmax,lens_potential_accuracy=0)
    results=camb.get_results(pars)
    powers=results.get_cmb_power_spectra(pars,CMB_unit='muK')
    cmb=powers['total']
    tt=cmb[:,0]    #you could return the full power spectrum here if you wanted to do say EE
    return tt[2:]

def gradient(m):
    fun = get_spectrum
    f = lambda params : fun(params)[:len(spec)]
    pred = f(m)
    # make empty array to store gradient values
    grad = np.empty([3049, m.size]) 
    grad = grad[:len(spec)]
    # numerical derivs, using forward difference 
    # for efficiency
    step = 1e-8
    f_m = f(m)
    for i, v in enumerate(m):
        dm = np.zeros(m.size)
        dm[i] = pars[i]*step
        grad[:,i] = (f(m + dm) - f_m)/dm[i]
    return pred, grad

iterator = 101
model_params = np.zeros([iterator, pars.size])
model_params[0] = pars
model=get_spectrum(pars)
model=model[:len(spec)]
resid=spec-model
# starting condition
chi = [4000]
d_chi = []
tol = 0.5
converged = False
Ninv = np.eye(len(spec))/(errs**2)

for i in range(iterator):
    pred, grad = gradient(model_params[i])
    r = spec - pred
    r = np.matrix(r).transpose()
    grad = np.matrix(grad)
    lhs = grad.T*Ninv*grad
    rhs = grad.T*Ninv*r
    u, s, vh = np.linalg.svd(lhs)
    dm = vh.T@np.linalg.inv(np.diag(s))@u.T@rhs
    dm = dm.reshape(6,)
    if i == iterator - 1:
        break
    else:
        model_params[i + 1] = model_params[i] + dm
        new_chi = get_chs(pred, spec)
        chi.append(new_chi)
        converged = np.abs(chi[-2] - chi[-1]) < tol
        # print(f"new chi = {new_chi}")
    if converged:
        print('Newton Method has converged.')
        break
    print(f"Params in current loop: {model_params[i]}\nWith chi sq: {new_chi}.")

plt.ion()


model=get_spectrum(pars)
model=model[:len(spec)]
resid=spec-pred

# calculate final chi square
chisq=np.sum((resid/errs)**2)
print("chisq is ",chisq," for ",len(resid)-len(pars)," degrees of freedom.")
# calculate parameter errors
model_errs = np.sqrt(np.diag(np.linalg.inv(lhs)))
to_save = np.vstack([model_params[i + 1], model_errs])
# curvature matrix
curv = lhs
#read in a binned version of the Planck PS for plotting purposes
planck_binned=np.loadtxt('COM_PowerSpect_CMB-TT-binned_R3.01.txt',skiprows=1)
errs_binned=0.5*(planck_binned[:,2]+planck_binned[:,3]);
plt.clf()
plt.plot(ell,model)
plt.errorbar(planck_binned[:,0],planck_binned[:,1],errs_binned,fmt='.')
plt.xlabel('ell')
plt.ylabel('spectrum (au)')
plt.savefig('newton_svd.jpg', dpi=300)
np.savetxt('planck fit params.txt', to_save)
plt.show()
