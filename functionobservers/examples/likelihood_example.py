"""
Test negative log-likelihood for RBFNetwork and RandomKitchenSinks.
"""
import numpy as np
from scipy.optimize import minimize, check_grad
import matplotlib.pyplot as plt
from functionobservers.utils.func_utils import pack_params_nll, unpack_params_nll
from functionobservers.optimizers.likelihood import negative_log_likelihood


# make data
seed = 0
rand_state = np.random.RandomState(seed)
nsamp = 1000
noise = 0.1
data_in_1D = np.reshape(np.linspace(-2, 2, num=nsamp), (nsamp, 1))
func = np.sin(data_in_1D) + 0.1*np.power(data_in_1D, 2)
func_obs = func + noise*rand_state.randn(nsamp, 1)

# construct training and validation sets
ntest = 300
rand_inds = rand_state.permutation(nsamp)
X_v = data_in_1D[rand_inds[0:ntest], :]
y_v = func_obs[rand_inds[0:ntest], :]
X_tr = data_in_1D[rand_inds[ntest:nsamp], :]
y_tr = func_obs[rand_inds[ntest:nsamp], :]

# initialize RBFNetwork centers
ncent = 30
centers_in = rand_state.randn(ncent, 1)

# call negative likelihood with kernel and parameters
k_name = "sqexp"
d_params = {"ell1": 0.01, "nu": 0.01}
noise_i = 0.01
params_i = np.log(pack_params_nll(d_params, noise_i, k_name))

# try running minimize function
rbfn_mname = "RBFNetwork"
arg_tup_rbfn = tuple([X_tr, y_tr, k_name, rbfn_mname, centers_in])


# include code to check gradient
def f_val(x):
    """

    :param x:
    :return:
    """
    arg_tup = tuple([X_tr, y_tr, k_name, rbfn_mname, centers_in])
    v, _ = negative_log_likelihood(x, *arg_tup)
    return v


def f_grad(x):
    """

    :param x:
    :return:
    """
    arg_tup = tuple([X_tr, y_tr, k_name, rbfn_mname, centers_in])
    _, g = negative_log_likelihood(x, *arg_tup)
    return g

print "Function call: {}\n\n".format(f_val(params_i))
print "Function grad: {}\n\n".format(f_grad(params_i))
print check_grad(func=f_val, grad=f_grad, x0=params_i)
print "\n\n"

#minimize(fun=negative_log_likelihood, x0=params_i, args=arg_tup_rbfn, jac=True, method="BFGS")


fig = plt.figure()
ax1 = fig.add_subplot(111)
ax1.plot(np.squeeze(data_in_1D), np.squeeze(func_obs), 'ro', linewidth=3.0, label="obs")
ax1.set_xlabel("x")
ax1.set_ylabel("f(x)")
ax1.title.set_text("Function with observations")
plt.show()
