from datetime import datetime
from functools import reduce

import pandas as pd
from torch import device, manual_seed, has_cuda
from torch.nn import MSELoss as MainLoss
from torch.optim import AdamW as Opt
from tqdm import trange

from despawn2D import Despawn2D as Denoiser
from util import l0_counter, sqrt_reg


def make_model(max_levels,
               starting_filter,
               adapt_filters=True,
               actually_filter=True,
               padding=100,
               dev="cpu") -> Denoiser:
	if has_cuda:
		dev = device("cuda:0")
	return Denoiser(
	    max_levels,
	    starting_filter,
	    adapt_filters=adapt_filters,
	    filter=actually_filter,
	    padding=padding,
	).to(dev)


def make_optimizer(model, lr_wavelets, lr_filter) -> Opt:
	return Opt([{
	    'params': model.scaling
	}, {
	    'params': model.threshold.parameters(),
	    'lr': lr_filter
	}],
	           lr=lr_wavelets)


def train_loop(
    model,
    optimizer,
    dataset,
    wavelet_name,
    epoch_end_fun=lambda epoch, total, l0, model, optimizer: None,
    epochs=10,
):
	REG_FACT = 1e-1
	REC_FACT = 1e-7

	NOW = str(datetime.now().timestamp()).replace(".", "_")
	store_file_name = f"run_{NOW}.pkl"

	def loss_fn(denoised, target):
		rec_loss = MainLoss(reduction="mean")(denoised, target)
		return rec_loss

	def combine_loss(rec, reg):
		return rec * REC_FACT + REG_FACT * reg

	progress_store = pd.DataFrame(columns=(
	    'loss',
	    'reconstruction_mse',
	    'regularization_penalty',
	    "reconstruction_factor",
	    "regularization_factor",
	    "l0_norm",
	))
	progress_store.attrs["wavelet"] = wavelet_name
	progress_store.attrs["epochs"] = epochs
	progress_store.attrs["device"] = str(device)

	for epoch in range(epochs):

		def __evaluate_model(entry):
			input, target = entry
			denoised, coeffs = model(input)

			return (loss_fn(denoised, target.to(denoised.device)), coeffs)

		# I am sorry.
		processed = map(__evaluate_model, dataset)
		summed = reduce(
		    lambda acc, elem: (acc[0] + elem[0], sqrt_reg(elem[1]) + acc[1],
		                       l0_counter(elem[1]) + acc[2]), processed,
		    (0, 0, 0))
		rec_loss = summed[0]
		reg_loss = summed[1]
		l0_count = summed[2]

		total = combine_loss(rec_loss, reg_loss)

		# Store the progress
		progress_store.loc[epoch] = {
		    "loss": float(total),
		    "reconstruction_mse": float(rec_loss),
		    "regularization_penalty": float(reg_loss),
		    "reconstruction_factor": float(REC_FACT),
		    "regularization_factor": float(REG_FACT),
		    "l0_norm": l0_count,
		}

		total.backward()
		optimizer.step()

		# Rebalances the weighing of the components with a moving average.
		if (epoch + 1) % 1000 == 0:
			REG_FACT = 1 / (4 * float(reg_loss)) + (3 / 4) * REG_FACT
			REC_FACT = 1 / (4 * rec_loss.detach()) + (3 / 4) * REC_FACT

		# Store the progress.
		if (epoch + 1) % 300 == 0:
			progress_store.to_pickle(store_file_name)
		optimizer.zero_grad()

		epoch_end_fun(epoch, float(total), float(l0_count), model, optimizer)
	# Also store the computed filters.
	progress_store.attrs["filters"] = model.scaling.detach().cpu().numpy()
	progress_store.to_pickle(store_file_name)