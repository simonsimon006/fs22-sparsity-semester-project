import jax.numpy as np
from scipy.optimize import fsolve
from scipy.integrate import quad
from numba import jit
from typing import Iterable
from jax import jit as jax_jit


#@jit(nogil=True, parallel=True)
@jax_jit
def lambd(ß):
	return np.sqrt(2 * (ß + 1) + 8 * ß / ((ß + 1) + np.sqrt(ß**2 + 14 * ß + 1)))


#@jit(nogil=True, parallel=True)
@jax_jit
def intform(t, ß):
	return np.sqrt(
	    ((1 + np.sqrt(ß))**2 - t) * (t - (1 - np.sqrt(ß)**2))) / (2 * np.pi * t)


#@jit(nogil=True, parallel=True)
def intgl(u, ß):
	y = quad(intform, (1 - ß)**2, u, args=(ß, ))
	# The result is in y[0], y[1] holds e.g. the error.

	return y[0] - 0.5


#@jit(nogil=True, parallel=True)
def thresh(n: int, m: int, sigmas: Iterable[float]):
	ß = n / m
	u = fsolve(func=intgl, x0=(300, ), args=(ß, ), maxfev=400)
	w = lambd(ß) / u
	sig = np.median(sigmas)
	return (w * sig)[0]


#@jit(nogil=True, parallel=True)
def denoise(ten: np.ndarray):
	# No full_matrices so the matmul simply works without padding
	# as the 0 svalues are omitted by the function.
	U, E, V = np.linalg.svd(ten, full_matrices=False)

	t = thresh(ten.shape[0], ten.shape[1], E)
	Ec = np.zeros_like(E)
	indexs = E > t
	Ec[indexs] = E[indexs]

	X = (U * Ec) @ V
	return X, E, Ec, t