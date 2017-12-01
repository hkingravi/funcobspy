"""
Test negative log-likelihood for RBFNetwork and RandomKitchenSinks.
"""
import time
import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import functionobservers.mappers as fmap
from functionobservers.funcobs.kernel_observer import KernelObserver
from functionobservers.log_utils import configure_logger
logger = configure_logger(level="INFO", name="funcobspy")

# load time series data
scheme = "smooth3"
data_dir = "./data"
f_prefix = "synthetic_time_series_generator_RBFNetwork_kernel_gaussian_scheme"
d_data = pickle.load(open(os.path.join(data_dir, f_prefix + "_" + scheme + ".pkl"), "rb"))

logger.info(
    "\nFunction data shape: {}\nFunction observations shape: {}\n".format(d_data['orig_func_data'].shape,
                                                                          d_data['orig_func_obs'].shape)
)


func_data = d_data['orig_func_data']
func_obs = d_data['orig_func_obs']


nsteps = func_data.shape[2]

# create series training and testing
nsteps_tr = 150
func_data_tr = func_data[:, :, 0:nsteps_tr]
func_obs_tr = func_obs[:, :, 0:nsteps_tr]
func_data_te = func_data[:, :, nsteps_tr:-1]
func_obs_te = func_obs[:, :, nsteps_tr:-1]
steps = np.arange(nsteps)

# create feature map
rs = np.random.RandomState(seed=0)
nbases = 300
centers_in = np.linspace(np.min(np.min(func_data[:, :, 0])),
                         np.max(np.max(func_data[:, :, 0])), nbases)
ncent = centers_in.shape[0]
centers_in = np.reshape(centers_in, (ncent, 1))
k_name = "gaussian"

# make data zero mean: important for dynamics
#for i in range(0, nsteps):
#    func_data[:, :, i] = func_data[:, :, i] - np.mean(func_data[:, :, i])

# bounds for each parameter plus noise: necessary to avoid blowup
if k_name == "gaussian":
    d_params = {"sigma": 1.0}
    param_bounds = ((-5, 1), (-6, 1))
elif k_name == "sqexp":
    d_params = {"nu": 1.0, "ell1": 0.9}
    param_bounds = ((-5, 2), (-5, 2), (-10, 2))
else:
    raise ValueError("Unsupported kernel. Halting execution.")
opt_options = {'maxiter': 350, 'disp': False}
noise_i = 0.01

rbfn = fmap.RBFNetwork(centers=centers_in, kernel_name=k_name, d_params=d_params, noise=noise_i,
                       optimizer="L-BFGS-B", d_opt=opt_options, random_state=rs, verbose=False)

# initialize other parameters
nmeas = 60
meas_op_type = 'random'
d_mopts = dict()
d_mopts['data'] = func_data[:, :, 0]  # use data locations from first time step
d_filter_params = {}  # pass in empty dict for parameters to use defaults

# train kernel observer
s_t = time.time()
kobs = KernelObserver(mapper=rbfn, nmeas=nmeas, meas_op_type=meas_op_type,
                      d_filter_params=d_filter_params, random_state=rs, verbose=True)
param_stream, est_weights, meas_inds = kobs.fit(data=func_data_tr, obs=func_obs_tr, d_mopts=d_mopts,
                                                bounds=param_bounds, reinit_params=False)

logger.info(
    "Time taken to train feature space from data, observations of shapes "
    "({}, {}): {:.2f} seconds.".format(func_data_tr.shape, func_obs_tr.shape, time.time()-s_t)
)
logger.info(
    "Final global {} kernel parameters: {}, noise: {}.".format(k_name, kobs.mapper.k_type.params,
                                                               kobs.mapper.noise)
)

nsamp_tr = func_data_tr.shape[2]
preds_ideal = np.zeros(func_obs_tr.shape)
rms_error_tr = np.zeros((nsamp_tr,))

# make 'ideal' predictions on training data
for i in range(0, nsteps_tr):
    f_ideal, _ = kobs.mapper.predict(func_data_te[:, :, i], weights_in=est_weights[i, :].reshape((1, nbases)))
    preds_ideal[:, :, i] = f_ideal
    rms_error_tr[i] = np.linalg.norm(preds_ideal[:, :, i] - func_obs_tr[:, :, i])/np.sqrt(preds_ideal.shape[0])

nsamp_te = func_data_te.shape[2]
rms_error_te = np.zeros((nsamp_te,))
preds = np.zeros(func_obs_te.shape)

for i in range(0, nsamp_te):
    f, _ = kobs.predict(func_data_te[:, :, i])
    preds[:, :, i] = f
    rms_error_te[i] = np.linalg.norm(preds[:, :, i]-func_obs_te[:, :, i])/np.sqrt(f.shape[0])
    # utilize measurements from certain locations to correct observer state
    current_meas = func_obs_te[meas_inds, :, i]  # + rs.randn(nmeas, 1)
    kobs.update(current_meas.T)

if k_name == "sqexp":
    figsize = (12, 16)
    fig1 = plt.figure(figsize=figsize)
    plt.suptitle("Kernel observer parameter training")
    ax1a = fig1.add_subplot(311)
    ax1a.plot(steps[0:nsteps_tr], param_stream[:, 0])
    plt.xlabel('Time step')
    plt.ylabel('ell1')
    plt.title("sqexp kernel's ell1 parameter over train time")

    ax1b = fig1.add_subplot(312)
    ax1b.plot(steps[0:nsteps_tr], param_stream[:, 1])
    plt.xlabel('Time step')
    plt.ylabel('nu')
    plt.title("sqexp kernel's nu parameter over train time")

    ax1c = fig1.add_subplot(313)
    ax1c.plot(steps[0:nsteps_tr], param_stream[:, 2])
    plt.xlabel('Time step')
    plt.ylabel('Noise')
    plt.title("Noise parameter over train time")
else:
    figsize = (12, 16)
    fig1 = plt.figure(figsize=figsize)
    plt.suptitle("Kernel observer parameter training")
    ax1a = fig1.add_subplot(211)
    ax1a.plot(steps[0:nsteps_tr], param_stream[:, 0])
    plt.xlabel('Time step')
    plt.ylabel('sigma')
    plt.title("Gaussian kernel's sigma parameter over train time")

    ax1b = fig1.add_subplot(212)
    ax1b.plot(steps[0:nsteps_tr], param_stream[:, 1])
    plt.xlabel('Time step')
    plt.ylabel('Noise')
    plt.title("Noise parameter over train time")

plt.figure()
plt.plot(steps[0:nsteps_tr], rms_error_tr)
plt.xlabel('Time step')
plt.ylabel('RMS Error')
plt.title("RMS prediction error over training time")

plt.figure()
plt.plot(steps[nsteps_tr:-1], rms_error_te)
plt.xlabel('Time step')
plt.ylabel('RMS Error')
plt.title("RMS prediction error over testing time")

fig2 = plt.figure(figsize=figsize)
nobs = preds_ideal.shape[0]
preds_ideal = preds_ideal.reshape((nobs, nsteps_tr))
func_obs_tr = func_obs_tr.reshape((nobs, nsteps_tr))
ax2a = fig2.add_subplot(211)
im2a = ax2a.imshow(func_obs_tr, cmap='jet')
plt.colorbar(im2a)
plt.title('Time series (train, clean)')
ax2b = fig2.add_subplot(212)
im2b = ax2b.imshow(preds_ideal, cmap='jet')
plt.colorbar(im2b)
plt.title('Time series ideal predictions (train)')
plt.suptitle("Plot of ideal predicted time series")

fig3 = plt.figure(figsize=figsize)
nobs = preds.shape[0]
nsteps_te = preds.shape[2]
preds = preds.reshape((nobs, nsteps_te))
func_obs_te = func_obs_te.reshape((nobs, nsteps_te))
ax3a = fig3.add_subplot(211)
im3a = ax3a.imshow(func_obs_te, cmap='jet')
plt.colorbar(im2a)
plt.title('Time series (test, clean)')
ax3b = fig3.add_subplot(212)
im3b = ax3b.imshow(preds, cmap='jet')
plt.colorbar(im3b)
plt.title('Time series observer predictions (test)')
plt.suptitle("Plot of actual vs predicted time series")

fig4 = plt.figure(figsize=(16, 16))
ax4a = fig4.add_subplot(111)
im4a = ax4a.imshow(np.absolute(kobs.filter.A), cmap='jet')
plt.colorbar(im2a)
plt.title('Dynamics operator (abs)')
plt.xlabel('Basis')
plt.ylabel('Basis')

fig5 = plt.figure(figsize=figsize)
ax5a = fig5.add_subplot(111)
im5a = ax5a.imshow(kobs.filter.C, cmap='jet')
plt.colorbar(im5a)
plt.xlabel('Basis')
plt.ylabel('Measurements')
plt.title('Measurement operator')

plt.show()

