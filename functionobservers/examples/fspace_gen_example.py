"""
Test negative log-likelihood for RBFNetwork and RandomKitchenSinks.
"""
import time
import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import functionobservers.mappers as fmap
from functionobservers.log_utils import configure_logger
logger = configure_logger(level="INFO", name="funcobspy")

# load time series data
scheme = "switching"
data_dir = "./data"
f_prefix = "synthetic_time_series_generator_RBFNetwork_kernel_gaussian_scheme"
d_data = pickle.load(open(os.path.join(data_dir, f_prefix + "_" + scheme + ".pkl"), "rb"))

logger.info(
    "\nFunction data shape: {}\nFunction observations shape: {}\n".format(d_data['orig_func_data'].shape,
                                                                          d_data['orig_func_obs'].shape)
)

# truncate series for this test
nsteps_tr = 50
func_data = d_data['orig_func_data'][:, :, 0:nsteps_tr]
func_obs = d_data['orig_func_obs'][:, :, 0:nsteps_tr]

# create feature map
rs = np.random.RandomState(seed=0)
centers_in = np.arange(0, 6, 0.3)
ncent = centers_in.shape[0]
centers_in = np.reshape(centers_in, (ncent, 1))
k_name_sqexp = "sqexp"
d_params_sqexp = {"ell1": 1.0, "nu": 1.0}
param_bounds = ((-5, 2), (-5, 2), (-10, 2))  # bounds for each parameter plus noise: necessary to avoid blowup
opt_options = {'maxiter': 350, 'disp': False}
noise_i_sqexp = 0.01

rbfn = fmap.RBFNetwork(centers=centers_in, kernel_name=k_name_sqexp, d_params=d_params_sqexp, noise=noise_i_sqexp,
                       optimizer="L-BFGS-B", d_opt=opt_options, random_state=rs, verbose=False)

# construct feature space
s_t = time.time()
fgen = fmap.FeatureSpaceGenerator(rbfn, verbose=True)
fgen.fit(func_data, func_obs, reinit_params=True, bounds=param_bounds)
d_p, n_f = fgen.return_final_params()
logger.info(
    "Time taken to train feature space from data, observations of shapes "
    "({}, {}): {:.2f} seconds.".format(func_data.shape, func_obs.shape, time.time()-s_t)
)
logger.info(
    "Final global {} kernel parameters: {}, noise: {}.".format(k_name_sqexp, d_p, n_f)
)

plt.figure(figsize=(10, 8))
plt.plot(np.arange(nsteps_tr), fgen.param_stream[:, 0])
plt.xlabel('Time step')
plt.ylabel('ell1')
plt.title("sqexp kernel's ell1 parameter over time")

plt.figure(figsize=(10, 8))
plt.plot(np.arange(nsteps_tr), fgen.param_stream[:, 1])
plt.xlabel('Time step')
plt.ylabel('nu')
plt.title("sqexp kernel's nu parameter over time")

plt.figure(figsize=(10, 8))
plt.plot(np.arange(nsteps_tr), fgen.param_stream[:, 2])
plt.xlabel('Time step')
plt.ylabel('Noise')
plt.title("Noise parameter over time")

plt.show()
