"""Does all the plotting."""

from pathlib import Path
from typing import List, Tuple
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from matplotlib.axes import Axes
from numpy import ndarray, absolute, subtract, arange, meshgrid
from numpy.fft import rfft
from torch import Tensor
from numba import jit


# From the previous project
def __make_axes(fig) -> List[Axes]:
	# Define the structure of the figure i.e. the position and number of
	# its subplots.
	gs0 = gridspec.GridSpec(3, 1, figure=fig)
	gs1 = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=gs0[0])
	gs2 = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=gs0[1])
	gs3 = gridspec.GridSpecFromSubplotSpec(1, 1, subplot_spec=gs0[2])

	pos = [
	    gs1[0],
	    gs1[1],  # Two subplots on the top.
	    gs2[0],
	    gs2[1],  # Two subplots in the middle.
	    gs3[0],  # One subplot on the bottom. Maybe the 3D thing needs to be added?
	]

	axs: List[Axes] = [fig.add_subplot(pos) for pos in pos]

	return axs


@jit(parallel=True, nogil=True)
def __spectral_transform(pred: ndarray,
                         true: ndarray) -> Tuple[ndarray, ndarray]:
	return absolute(rfft(pred)), absolute(rfft(true))


def __coeffs(axs: Axes, coeffs: ndarray):
	axs.set_ylabel("Strength")
	axs.set_xlabel("Coefficient")
	axs.set_yscale("log")
	axs.plot(coeffs)


def __diff(axs: Axes, original: ndarray, denoised: ndarray):
	diff = absolute(subtract(original, denoised))
	axs.set_title("original-difference logscaled")

	t, x = diff.shape
	x, t = arange(x), arange(t)

	X, T = meshgrid(x, t)
	
	axs.set_xlabel("Spatial axis")
	axs.set_ylabel("Temporal axis")
	axs.set_zlabel("Strain axis")

	axs.plot_surface(X, T, diff, color="blue")


def __diag(axs: Axes, original: ndarray, denoised: ndarray):
	pass


def __time_and_space(time_axs: Axes, space_axs: Axes, original: ndarray,
                     denoised: ndarray):
	pass


def plot(original: ndarray | Tensor, denoised: ndarray | Tensor,
         result: ndarray | Tensor, save: str | Path) -> None:
	"""Produes the relevant plots from the data."""
	'''# First we transform to ndarrays
	match data:
		case ndarray(data):
			pass
		case Tensor(data):
			data = data.numpy()
	'''
	fig = plt.figure(figsize=(12, 10), dpi=300)
	axs = __make_axes(fig)

	fig.savefig(save)
	plt.close(fig)