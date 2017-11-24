"""
Test RBFNetwork class with multiple kernels.
"""
import pickle
import numpy as np
import matplotlib.pyplot as plt
import functionobservers.mappers as fmap


# load data
d_krr = pickle.load(open("./data/KRR.pkl", "rb"))
X_tr = d_krr['x'].reshape((d_krr['x'].shape[1], 1))
y_tr = d_krr['y_n'].reshape((d_krr['y_n'].shape[1], 1))

# random state
rs = np.random.RandomState(seed=0)

# initialize RBFNetwork centers
centers_in = np.arange(-5, 5, 0.6)
ncent = centers_in.shape[0]
centers_in = np.reshape(centers_in, (ncent, 1))

# call negative likelihood with Gaussian kernel and parameters
k_name_gauss = "gaussian"
d_params_gauss = {"sigma": 0.1}
noise_i_gauss = 0.01
opt_options = {'maxiter': 350, 'disp': False}

# fit Gaussian RBFNetwork: same thing as squared exponential minus the nu parameter
rbfn_gauss = fmap.RBFNetwork(centers=centers_in, kernel_name=k_name_gauss, d_params=d_params_gauss, noise=noise_i_gauss,
                             optimizer="L-BFGS-B", d_opt=opt_options, random_state=rs, verbose=True)
rbfn_gauss.fit(X_tr, y_tr)
preds_gauss = rbfn_gauss.predict(X_tr)

# now call negative likelihood with squared exponential kernel and parameters
k_name_sqexp = "sqexp"
d_params_sqexp = {"ell1": 0.1, "nu": 0.2}
noise_i_sqexp = 0.01

# fit squared exponential RBFNetwork
rbfn_sqexp = fmap.RBFNetwork(centers=centers_in, kernel_name=k_name_sqexp, d_params=d_params_sqexp, noise=noise_i_sqexp,
                             optimizer="L-BFGS-B", d_opt=opt_options, random_state=rs, verbose=True)
rbfn_sqexp.fit(X_tr, y_tr)
preds_sqexp = rbfn_sqexp.predict(X_tr)
# make prediction using random weights
preds_r_sqexp = rbfn_sqexp.predict(X_tr, weights_in=rs.randn(1, rbfn_sqexp.ncent)/np.sqrt(rbfn_sqexp.ncent))


plt.figure(figsize=(10, 8))
plt.plot(X_tr, y_tr, 'ro', label='obs')
plt.plot(X_tr, preds_gauss, 'm-', label='estimate (gaussian)', linewidth=3.0)
plt.plot(X_tr, preds_sqexp, 'g-', label='estimate (sqexp)', linewidth=3.0)
plt.plot(X_tr, preds_r_sqexp, 'b-', label='random', linewidth=3.0)
plt.xlabel('Data')
plt.ylabel('Measurements')
plt.legend(loc="upper right")
plt.xlim([np.min(X_tr), np.max(X_tr)])
plt.title("Plot of RBFNetwork fit to observations as well as random weights")
plt.show()
