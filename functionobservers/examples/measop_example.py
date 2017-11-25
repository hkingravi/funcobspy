"""
Example of measurement operators.
"""

import pickle
import numpy as np
import matplotlib.pyplot as plt
import functionobservers.mappers as fmap


# measurements
nmeas_small = 10
nmeas_large = 50

# random state
rs = np.random.RandomState(seed=20)

# initialize data
data = np.linspace(-6, 6, 500)
nsamp = data.shape[0]
data = np.reshape(data, (nsamp, 1))

# initialize RBFNetwork centers
centers_in = np.arange(-5, 5, 0.1)
ncent = centers_in.shape[0]
centers_in = np.reshape(centers_in, (ncent, 1))

# call negative likelihood with Gaussian kernel and parameters
k_name_gauss = "gaussian"
d_params_gauss = {"sigma": 0.9}
noise_i_gauss = 0.01
opt_options = {'maxiter': 350, 'disp': False}

# init Gaussian RBFNetworks: no need to fit in this instance, but in general, you will fit first, and then construct
# the map
rbfn_sort = fmap.RBFNetwork(centers=centers_in, kernel_name=k_name_gauss, d_params=d_params_gauss, noise=noise_i_gauss,
                            optimizer="L-BFGS-B", d_opt=opt_options, random_state=rs, verbose=True, sort_mat=True)
rbfn_nsort = fmap.RBFNetwork(centers=centers_in, kernel_name=k_name_gauss, d_params=d_params_gauss, noise=noise_i_gauss,
                            optimizer="L-BFGS-B", d_opt=opt_options, random_state=rs, verbose=True, sort_mat=False)

d_mopts = dict()
d_mopts['data'] = data
Kmat_small_rbfn_sorted = fmap.measurement_operator('random', nmeas_small, rbfn_sort, d_mopts, random_state=rs)
Kmat_large_rbfn_sorted = fmap.measurement_operator('random', nmeas_large, rbfn_sort, d_mopts, random_state=rs)
Kmat_small_rbfn_nsorted = fmap.measurement_operator('random', nmeas_small, rbfn_nsort, d_mopts, random_state=rs)
Kmat_large_rbfn_nsorted = fmap.measurement_operator('random', nmeas_large, rbfn_nsort, d_mopts, random_state=rs)

# plotting parameters
figsize = (12, 12)

fig1 = plt.figure(figsize=figsize)
ax1a = fig1.add_subplot(211)
im1a = ax1a.imshow(Kmat_small_rbfn_nsorted[0].T, cmap='jet')
plt.colorbar(im1a)
plt.title('RBFNetwork measurement operator (nmeas={})'.format(nmeas_small))
ax1b = fig1.add_subplot(212)
im1b = ax1b.imshow(Kmat_large_rbfn_nsorted[0].T, cmap='jet')
plt.colorbar(im1b)
plt.title('RBFNetwork measurement operator (nmeas={})'.format(nmeas_large))
plt.suptitle("Plot of non-sorted measurement maps for RBFNetwork")

fig2 = plt.figure(figsize=figsize)
ax2a = fig2.add_subplot(211)
im2a = ax2a.imshow(Kmat_small_rbfn_sorted[0].T, cmap='jet')
plt.colorbar(im1a)
plt.title('RBFNetwork measurement operator (nmeas={})'.format(nmeas_small))
ax2b = fig2.add_subplot(212)
im2b = ax2b.imshow(Kmat_large_rbfn_sorted[0].T, cmap='jet')
plt.colorbar(im2b)
plt.title('RBFNetwork measurement operator (nmeas={})'.format(nmeas_large))
plt.suptitle("Plot of sorted measurement maps for RBFNetwork")
plt.show()

