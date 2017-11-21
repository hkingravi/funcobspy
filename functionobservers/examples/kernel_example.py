"""
Examples of kernel functions.
"""
import numpy as np
import matplotlib.pyplot as plt

from functionobservers.mappers.kernel import kernel, KernelType, dist_mat, map_data_rks
from functionobservers.utils.func_utils import pack_params_nll, unpack_params_nll
from functionobservers.log_utils import configure_logger
logger = configure_logger(level="INFO", name="funcobspy")


# create data
x = np.array(([1, 2, 3, 4], [2, 3, 4, 4])).T
y = np.array(([1, 3, 1], [2, 1, 4])).T

data1 = np.linspace(-3, 3, 100).T
data2 = np.array([0.2]).T

# initialize kernels
poly_deg = 3
poly_bias = 0.5
d_params_poly = {"degree": np.array(poly_deg), "bias": np.array(poly_bias)}
k_poly = KernelType("polynomial", params=d_params_poly)

gauss_band = 0.5
d_params_gauss = {"sigma": gauss_band}
k_gauss = KernelType("gaussian", params=d_params_gauss)

d_params_sqexp1 = {"ell1": 0.5, "nu": 0.7}
d_params_sqexp2 = {"ell1": 0.5, "ell2": 0.7, "nu": 2.0}
k_sqexp_1d = KernelType("sqexp", params=d_params_sqexp1)
k_sqexp_2d = KernelType("sqexp", params=d_params_sqexp2)

dist_vals = dist_mat(x.T, y.T)
K_gauss_2d = kernel(x, y, k_gauss)
try:
    K_poly_1d, grads_poly_1d = kernel(data1, data2, k_poly, return_grads=True)
except ValueError as e:
    logger.info("Caught exception: {}".format(e))
K_poly_1d = kernel(data1, data2, k_poly)
K_poly_2d = kernel(x, y, k_poly)
K_sqexp_1d, grads_sqexp_1d = kernel(data1, data2, k_sqexp_1d, return_grads=True)
K_sqexp_2d, grads_sqexp_2d = kernel(x, y, k_sqexp_2d, return_grads=True)

logger.info("Data shapes: {}, {}".format(data1.shape, data2.shape))
logger.info("x, y shapes: {}, {}".format(x.shape, y.shape))
logger.info("Gaussian, 2D shape: {}".format(K_gauss_2d.shape))

# now do everything for RandomKitchenSinks
nsamp2 = 50
random_state = np.random.RandomState(0)
data1 = data1.reshape((data1.shape[0], 1))
data2 = np.sort(random_state.randn(nsamp2, 1), axis=0)
K_gauss_1d, grads_gauss_1d = kernel(data1, data2, k_gauss, return_grads=True)

nbases = 1000
rks_basis = random_state.randn(nbases, data1.shape[1])
v1 = map_data_rks(centers=rks_basis, k_type=k_gauss, data=data1)
v2 = map_data_rks(centers=rks_basis, k_type=k_gauss, data=data2)
K_gauss_rks = np.dot(v1, v2.T)
logger.info("Data 1 bounds: ({}, {})".format(np.sort(data1, axis=None)[0], np.sort(data1, axis=None)[-1]))
logger.info("Data 2 bounds: ({}, {}).".format(np.sort(data2, axis=None)[0], np.sort(data2, axis=None)[-1]))

# plotting parameters
figsize = (12, 12)

fig = plt.figure(figsize=figsize)
ax2 = fig.add_subplot(311)
ax3 = fig.add_subplot(312)
ax4 = fig.add_subplot(313)

ax2.plot(data1, np.squeeze(K_sqexp_1d), linewidth=3.0)
sqexp_title = "Squared exponential kernel, (ell, nu) = (" + str(d_params_sqexp1["ell1"]) \
              + ", " + str(d_params_sqexp1["nu"]) + ")"
ax2.title.set_text(sqexp_title)

ax3.plot(data1, np.squeeze(grads_sqexp_1d["nu"]), linewidth=3.0, c="r", label="nu")
ax3.plot(data1, np.squeeze(grads_sqexp_1d["ell1"]), linewidth=3.0, c="g", label="ell")
sqexp_title = "Squared exponential kernel derivatives, (ell, nu) = (" + str(d_params_sqexp1["ell1"]) + ", " \
              + str(d_params_sqexp1["nu"]) + ")"
ax3.legend()
ax3.title.set_text(sqexp_title)

ax4.plot(data1, np.squeeze(K_poly_1d), linewidth=3.0)
poly_title = "Polynomial kernel (d, b) =(" + str(poly_deg) + ", " + str(poly_bias) + ")"
ax4.title.set_text(poly_title)
fig.suptitle("Plot of 1D kernels")

fig2 = plt.figure(figsize=figsize)
Kr = np.dot(v1, v2.T)
K = kernel(data1=data1, data2=data2, k_type=k_gauss)
ax2a = fig2.add_subplot(311)
im1 = ax2a.imshow(K.T)
plt.colorbar(im1)
plt.title("Full kernel matrix")
ax2b = fig2.add_subplot(312)
im2 = ax2b.imshow(Kr.T)
plt.title("RKS kernel matrix")
plt.colorbar(im2)
ax2c = fig2.add_subplot(313)
im3 = ax2c.imshow(K.T-Kr.T)
plt.title("Absolute Value Difference")
plt.colorbar(im3)
logger.info(
    "Percentage error in Gram matrix between RKS and " \
    "Gaussian kernel: {:.2f} percent.".format(100.0*np.linalg.norm(K-Kr)/np.linalg.norm(K))
)
fig2.suptitle("Plot of random kitchen sinks approximation")

# check parameter packing
noise = 0.2
try:
    params_sqexp2 = pack_params_nll(d_params_sqexp2, noise, k_name="etc")
except ValueError as e:
    logger.info(
        "Caught exception: {}".format(e)
    )
logger.info(
    "Original (sqexp, noise) parameters:\n ({}, {})".format(d_params_sqexp2, noise)
)
params_sqexp2 = pack_params_nll(d_params_sqexp2, noise, k_name="sqexp")
logger.info(
    "Packed sqexp parameters:\n {}".format(params_sqexp2)
)
d_params_sqexp2b, noiseb = unpack_params_nll(params_sqexp2, k_name="sqexp")
logger.info(
    "Unpacked (sqexp, noise) parameters:\n ({}, {})".format(d_params_sqexp2, noise)
)
plt.show()
