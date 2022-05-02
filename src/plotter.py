"""Does all the plotting."""

from pathlib import Path
from typing import List, Tuple

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from numba import jit
from numpy import absolute, arange, meshgrid, ndarray, subtract, square, NaN, log10
from numpy.fft import rfft
from scipy.stats import pearsonr


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
	]

	axs: List[Axes] = [fig.add_subplot(pos) for pos in pos]
	axs.append(
	    fig.add_subplot(gs3[0], projection="3d")
	),  # One subplot on the bottom. Maybe the 3D thing needs to be added?)
	return axs


@jit(parallel=True, nogil=True)
def __spectral_transform(pred: ndarray,
                         true: ndarray) -> Tuple[ndarray, ndarray]:
	return absolute(rfft(pred)), absolute(rfft(true))


def __coeffs(axs: Axes, coeffs: ndarray, orig_sv: ndarray, cutoff: float):
	axs.set_ylabel("Strength")
	axs.set_xlabel("Coefficient")
	axs.set_yscale("log")
	x = range(len(coeffs))
	q = sum(orig_sv > cutoff) - 1
	axs.scatter(x, coeffs, label="Culled", marker="1", linewidths=0.4)
	axs.scatter(x,
	            orig_sv,
	            label="Original",
	            marker="2",
	            linewidths=0.4,
	            alpha=0.3)
	axs.axvline(q, ls="--", color="red")
	axs.legend()


def __diff(axs: Axes, original: ndarray, denoised: ndarray):
	diff = absolute(subtract(original, denoised))
	axs.set_title("original-denoised logscaled")

	axs.set_xlabel("Spatial axis")
	axs.set_ylabel("Temporal axis")

	axs.matshow(diff, cmap="binary")


@jit(nogil=True, parallel=True)
def __variation(y_pred: ndarray, y_true: ndarray) -> float:
	'''Computes the variation coefficient.'''
	sigma = (square(y_pred - y_true)).mean()
	mean = y_pred.mean()
	if mean == 0:
		mean = NaN
	return sigma / mean


@jit(nogil=True, parallel=True)
def __var_and_corr(pred: ndarray, true: ndarray) -> Tuple[float, float]:
	var, (corr, _) = __variation(pred, true), pearsonr(pred, true)
	return var, corr


def __diag(axs: Axes, original: ndarray, denoised: ndarray):
	'''flat_original, flat_denoised = original.reshape(
	    denoised.shape[0] * denoised.shape[1]), denoised.reshape(
	        denoised.shape[0] * denoised.shape[1])'''

	flat_original, flat_denoised = original.flat, denoised.flat
	axs.set_title("45 deg correlation plot")
	axs.set_xscale("log")
	axs.set_yscale("log")
	axs.scatter(flat_original,
	            flat_denoised,
	            label="Signals",
	            marker=".",
	            s=0.3)
	axs.axline(xy1=(0, 0),
	           xy2=(1, 1),
	           linestyle="--",
	           color="yellow",
	           transform=axs.transAxes)
	axs.set_ylabel("Original")
	axs.set_xlabel("Denoised")
	axs.set_title("Signal comparision")

	axs.set_aspect("equal", 'box')
	'''var, corr = __var_and_corr(denoised, original)
	axs.text(
	    0.7,
	    0.9,
	    f"Var: {var:.3f}\nCorr: {corr:.3f}",
	    bbox={
	        'alpha': 0.2,
	        'pad': 10,
	        "boxstyle": "Round,pad=0.01",
	        "facecolor": "blue"
	    },
	    transform=axs.transAxes,
	)'''

	xll, xul = axs.get_xlim()
	yll, yul = axs.get_ylim()
	ll = min(xll, yll)
	up = max(xul, yul)
	axs.set_xlim(left=ll, right=up)
	axs.set_ylim(bottom=ll, top=up)
	axs.legend()


MSIZE = 0.3


def __time_and_space(time_axs: Axes, space_axs: Axes, original: ndarray,
                     denoised: ndarray):
	'''Simply plots the signals and their difference in a log scale onto the argument axes.'''

	# They should have the same shape
	assert (original.shape == denoised.shape)

	tmax, xmax = original.shape
	# Display at time t
	# 0.65mm stepsize
	t = tmax // 2  # Talked with Yasmin about that. She says the time is right.
	xax = arange(0, 0.65 * len(original[t, :]), 0.65)
	time_axs.set_title(f"Time fixed at timestep {t}.")
	time_axs.plot(xax, denoised[t, :], label="Denoised", linewidth=MSIZE)
	time_axs.plot(xax,
	              original[t, :],
	              label="Original",
	              linewidth=MSIZE,
	              alpha=0.3,
	              color="red")
	'''
	time_axs.plot(
		xax,
		filtered,
		label="Hand denoised",
		#linewidth=msize,
		ls="dotted",
		color="red")
	'''
	time_axs.set_ylabel("Strain")
	time_axs.set_xlabel("Location/mm")
	time_axs.legend()

	# Display at pos x
	# unknown time steps
	x = xmax // 4  # Talked with Yasmin about that. She says the time is right.
	xax = arange(0, len(original[:, x]))
	space_axs.set_title(f"Location fixed at pos {x * 0.65}mm.")
	space_axs.plot(xax, denoised[:, x], label="Denoised", linewidth=MSIZE)
	space_axs.plot(xax,
	               original[:, x],
	               label="Original",
	               linewidth=MSIZE,
	               alpha=0.3,
	               color="red")
	'''space_axs.plot(
		xax,
		filtered,
		label="Hand denoised",
		#linewidth=msize,
		ls="dotted",
		color="red")
	'''
	space_axs.set_ylabel("Strain")
	space_axs.set_xlabel("Timesteps")
	space_axs.legend()


def plot(original: ndarray, denoised: ndarray, orig_sv: ndarray,
         result: ndarray, cutoff: float, save: str | Path) -> None:
	"""Produes the relevant plots from the data."""

	fig = plt.figure(figsize=(14, 14), dpi=500)
	axs = __make_axes(fig)

	__time_and_space(axs[0], axs[1], original, denoised)
	print("TS done")
	__coeffs(axs[2], result, orig_sv, cutoff)
	print("cf done")

	__diag(axs[3], original, denoised)
	print("diag done")

	__diff(axs[4], original, denoised)
	print("diif done")

	fig.savefig(save)
	plt.close(fig)
