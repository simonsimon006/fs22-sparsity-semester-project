"""Does all the plotting."""

from pathlib import Path
from typing import List, Tuple

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.axes import Axes
from numba import jit
from numpy import (NaN, absolute, arange, array, ceil, log2, ndarray, square,
                   subtract, linspace)
from numpy.fft import rfft
from scipy.stats import pearsonr
from ssqueezepy import cwt

from util import vip


# From the previous project
def __make_axes(fig, upper_subs=2) -> List[Axes]:
	# Define the structure of the figure i.e. the position and number of
	# its subplots.
	gs0 = gridspec.GridSpec(3, 1, figure=fig)
	gs1 = gridspec.GridSpecFromSubplotSpec(1, upper_subs, subplot_spec=gs0[0])
	gs2 = gridspec.GridSpecFromSubplotSpec(1, 3, subplot_spec=gs0[1])
	gs3 = gridspec.GridSpecFromSubplotSpec(1, 1, subplot_spec=gs0[2])

	pos = [
	    *gs1[:upper_subs],  # upper_subs many subplots on the top.
	    gs2[0],
	    gs2[1],  # Three subplots in the middle.
	    gs2[2],
	]

	axs: List[Axes] = [fig.add_subplot(pos) for pos in pos]
	axs.append(
	    fig.add_subplot(gs3[0])
	),  # One subplot on the bottom. Maybe the 3D thing needs to be added?)
	return axs


@jit(parallel=True, nogil=True)
def __spectral_transform(pred: ndarray,
                         true: ndarray) -> Tuple[ndarray, ndarray]:
	return absolute(rfft(pred)), absolute(rfft(true))


def __coeffs(axs: Axes, coeffs: ndarray, orig_sv: ndarray, cutoff: float):
	axs.set_ylabel("Strength")
	axs.set_xlabel("Wav coefficient")
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
	maximum = absolute(original.max())
	diff /= maximum
	axs.set_title("original-denoised")

	axs.set_xlabel("Time")
	axs.set_ylabel("Space")
	axs.grid(False)
	axs.imshow(diff.T,
	           label="Difference between denoised and subtracted signal",
	           aspect="auto")


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
	axs.set_ylabel("True")
	axs.set_xlabel("Denoised")
	axs.set_title("Signal comparision")

	axs.set_aspect("auto", 'box')
	'''var, corr = __var_and_corr(denoised, original)
	axs.text(
	    0.7,
	    0.1,
	    f"Var: {var:.3f}\nCorr: {corr:.3f}",
	    bbox={
	        'alpha': 0.2,
	        'pad': 10,
	        "boxstyle": "Round,pad=0.01",
	        "facecolor": "blue"
	    },
	    transform=axs.transAxes,
	)
	'''
	xll, xul = axs.get_xlim()
	yll, yul = axs.get_ylim()
	ll = min(xll, yll)
	up = max(xul, yul)
	axs.set_xlim(left=ll, right=up)
	axs.set_ylim(bottom=ll, top=up)
	axs.legend()


MSIZE = 0.8


def __time_and_space(time_axs: Axes, space_axs: Axes, original: ndarray,
                     denoised: ndarray):
	'''Simply plots the signals and their difference in a log scale onto the argument axes.'''

	# They should have the same shape
	assert (original.shape == denoised.shape)

	tmax, xmax = original.shape
	# Display at time t
	# 0.65mm stepsize
	t = 1216  #tmax // 2  # Talked with Yasmin about that. She says the time is right.
	xax = arange(0, 0.65 * len(original[t, :]), 0.65)
	time_axs.set_title(f"Time fixed at timestep {t}.")
	time_axs.plot(xax, denoised[t, :], label="Denoised", linewidth=MSIZE)
	time_axs.plot(xax,
	              original[t, :],
	              label="Original",
	              linewidth=MSIZE,
	              alpha=0.3,
	              color="red")
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
	space_axs.set_ylabel("Strain")
	space_axs.set_xlabel("Timesteps")
	space_axs.legend()


def plot_wavedec(time: ndarray,
                 true: ndarray,
                 noisy: ndarray,
                 denoised: ndarray,
                 save: str | Path,
                 name_wavelet="db20") -> None:
	"""Produes the relevant plots from the data."""

	MODE = "periodization"
	fig = plt.figure(figsize=(16, 14), dpi=300)
	axs = __make_axes(fig)

	axs[0].plot(time, true, label="True signal", linewidth=MSIZE, alpha=0.7)
	axs[0].plot(time, noisy, label="Noisy signal", linewidth=MSIZE, alpha=0.3)
	axs[0].plot(time,
	            denoised,
	            label="Denoised signal",
	            linewidth=0.8 * MSIZE,
	            alpha=0.7)
	axs[0].legend()

	__diag(axs[1], true, denoised)
	'''
	dec_true = wavedec(true, name_wavelet, mode=MODE)
	dec_noisy = wavedec(noisy, name_wavelet, mode=MODE)
	dec_denoised = wavedec(denoised, name_wavelet, mode=MODE)
	'''
	# Compute the scales. I hope this makes sense.
	scales = 2**array(range(int(ceil(log2(true.shape[0])))))
	dec_true, _, _ = cwt(true)
	dec_noisy, _, _ = cwt(noisy)
	dec_denoised, _, _ = cwt(denoised)

	axs[2].set_title("Noisy scalogram")
	make_scalogram(axs[2], dec_noisy)

	axs[3].set_title("True scalogram")
	make_scalogram(axs[3], dec_true)

	axs[4].set_title("Denoised scalogram")
	make_scalogram(axs[4], dec_denoised)

	__diff(axs[5], true, denoised)
	fig.savefig(save)
	plt.close(fig)


def __make_meas_axs(fig, upper_subs):
	# Define the structure of the figure i.e. the position and number of
	# its subplots.
	gs0 = gridspec.GridSpec(4, 1, figure=fig)
	gs1 = gridspec.GridSpecFromSubplotSpec(1, upper_subs, subplot_spec=gs0[0])
	gs0_1 = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=gs0[1])
	gs2 = gridspec.GridSpecFromSubplotSpec(1, 3, subplot_spec=gs0[2])
	gs3 = gridspec.GridSpecFromSubplotSpec(1, 1, subplot_spec=gs0[3])

	pos = [
	    *gs1,  # upper_subs many subplots on the top.
	    *gs0_1,
	    gs2[0],
	    gs2[1],  # Three subplots in the middle.
	    gs2[2],
	]

	axs: List[Axes] = [fig.add_subplot(pos) for pos in pos]
	axs.append(
	    fig.add_subplot(gs3[0])
	),  # One subplot on the bottom. Maybe the 3D thing needs to be added?)
	return axs


def plot_measurement(true: ndarray,
                     noisy: ndarray,
                     denoised: ndarray,
                     save: str | Path,
                     name_measurement: str,
                     name_wavelet="morlet") -> None:
	"""Produes the relevant plots from the data."""

	relevant_timesteps = vip[name_measurement]
	plots = len(relevant_timesteps)
	fig = plt.figure(figsize=(22, 14), dpi=400)
	axs = __make_meas_axs(fig, plots)

	for ((time, name), axes) in zip(relevant_timesteps, axs[:plots]):

		axes.set_title(name)
		if name_measurement == "sine-test":
			tt = linspace(-1, 1, 1000)
			axes.plot(tt,
			          noisy[time, :],
			          label="Noisy signal",
			          linewidth=MSIZE,
			          alpha=0.6,
			          color="black")
			axes.plot(tt,
			          denoised[time, :],
			          label="Denoised signal",
			          linewidth=1,
			          color="green")
			axes.plot(tt,
			          true[time, :],
			          label="True signal",
			          linewidth=0.5,
			          color="red")
			axes.set_ylim(bottom=-10, top=10)
		else:
			axes.plot(noisy[time, :],
			          label="Noisy signal",
			          linewidth=MSIZE,
			          alpha=0.6,
			          color="black")
			axes.plot(denoised[time, :],
			          label="Denoised signal",
			          linewidth=1,
			          color="green")
			axes.plot(true[time, :],
			          label="True signal",
			          linewidth=0.5,
			          color="red")
		axes.sharey(axs[plots - 1])
		axes.legend()

	# Plot time
	axes = axs[-6]
	point = true.shape[1] // 2
	axes.set_title(f"Time plot at point {point}")

	axes.plot(noisy[:, point],
	          label="Noisy signal",
	          linewidth=MSIZE,
	          alpha=0.3,
	          color="black")
	axes.plot(denoised[:, point],
	          label="Denoised signal",
	          linewidth=1,
	          color="green")
	axes.plot(true[:, point], label="True signal", linewidth=0.5, color="red")
	axes.legend()

	__diag(axs[-5], true, denoised)

	# Just take the first relevant point.
	example_time = relevant_timesteps[-1][0]

	dec_true, _ = cwt(true[example_time], name_wavelet)
	dec_noisy, _ = cwt(noisy[example_time], name_wavelet)
	dec_denoised, _ = cwt(denoised[example_time], name_wavelet)

	# To normalize the data
	dec_true = absolute(dec_true)
	dec_noisy = absolute(dec_noisy)
	dec_denoised = absolute(dec_denoised)
	maximum = max(dec_noisy.max(), dec_true.max(), dec_denoised.max())
	axs[-4].set_title(f"Noisy scalogram at time {example_time}")
	make_scalogram(axs[-4], dec_noisy / maximum)

	axs[-3].set_title(f"True scalogram at time {example_time}")
	make_scalogram(axs[-3], dec_true / maximum)

	axs[-2].set_title(f"Denoised scalogram at time {example_time}")
	make_scalogram(axs[-2], dec_denoised / maximum)

	__diff(axs[-1], true, denoised)
	fig.colorbar(
	    cm.ScalarMappable(cmap="magma"),
	    ax=axs[-1],
	    location='bottom',
	)
	fig.tight_layout()
	fig.savefig(save)
	plt.close(fig)


def make_scalogram(ax, wavedec_coeffs, max=None):
	'''max_scale_size = len(wavedec_coeffs[-1])

	rows = []
	for row in wavedec_coeffs:
		row_len = len(row)
		multiples = max_scale_size // row_len
		kr = repeat(1, multiples)
		row = kron(row, kr)
		rows.append(row)
	'''
	ax.set_xlabel("Shifts")
	ax.set_ylabel("Scales")
	ax.grid(False)
	coeffs = array(wavedec_coeffs)
	if max:
		coeffs /= max
	ax.imshow(coeffs, aspect="auto", interpolation="nearest", cmap="magma")
