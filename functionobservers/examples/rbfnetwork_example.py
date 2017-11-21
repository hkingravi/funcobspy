"""
Test negative log-likelihood for RBFNetwork and RandomKitchenSinks.
"""
import cPickle as pickle
import numpy as np
import matplotlib.pyplot as plt
from functionobservers.mappers.mappers import RBFNetwork


# load data
d_krr = pickle.load(open("./data/KRR.pkl", "rb"))
X_tr = d_krr['x'].reshape((d_krr['x'].shape[1], 1))
y_tr = d_krr['y_n'].reshape((d_krr['y_n'].shape[1], 1))

# initialize RBFNetwork centers
centers_in = np.arange(-5, 5, 0.6)
ncent = centers_in.shape[0]
centers_in = np.reshape(centers_in, (ncent, 1))

# call negative likelihood with kernel and parameters
k_name = "sqexp"
d_params = {"ell1": 0.1, "nu": 0.2}
noise_i = 0.01
opt_options = {'maxiter': 350, 'disp': False}

# now fit RBFNetwork
rbfn = RBFNetwork(centers=centers_in, kernel_name=k_name, d_params=d_params, noise=noise_i,
                  optimizer="L-BFGS-B", d_opt=opt_options, verbose=True)
rbfn.fit(X_tr, y_tr)
preds = rbfn.predict(X_tr)

plt.figure()
plt.plot(X_tr, y_tr, 'ro', label='obs')
plt.plot(X_tr, preds, 'g-', label='estimate', linewidth=3.0)
plt.xlabel('Data')
plt.ylabel('Measurements')
plt.legend()
plt.xlim([np.min(X_tr), np.max(X_tr)])
plt.title("Plot of RBFNetwork fit to observations")
plt.show()
