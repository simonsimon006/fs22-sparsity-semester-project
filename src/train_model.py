from datetime import datetime

import pandas as pd
from ray.air import session, Checkpoint
from torch import device, has_cuda, tensor, inference_mode, load
from torch.nn import MSELoss as MainLoss
from torch.optim import AdamW as Opt
from torch.utils.data import DataLoader

from despawn2D import Despawn2D as Denoiser
from loader import MeasurementFolder
from util import l0_counter
from util import sqrt_reg as regularisation_fun


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


def make_optimizer(model, lr_wavelets) -> Opt:
	return Opt([{
	    'params': [model.alpha, model.b_minus, model.b_plus],
	    "lr": 1e-2,
	}, {
	    'params': model.bank.parameters(),
	    'lr': lr_wavelets
	}],
	           lr=1)


def train_loop(
    model,
    optimizer,
    wavelet_name,
    epochs=10,
):
	REG_FACT = 1e-1
	REC_FACT = 1e-4
	ORTH_FACT = 1e5
	VAL_REC_FACT = 1e-4  # 1e-6
	dataset = DataLoader(
	    MeasurementFolder("/home/simon/Code/semester-project-fs22/store/"),
	    batch_size=None,
	)
	validation = load("/home/simon/Code/semester-project-fs22/eps_yl_w3.pt")

	NOW = str(datetime.now().timestamp()).replace(".", "_")

	#store_file_name = f"run_{NOW}.pkl"

	def loss_fn(denoised, target):
		rec_loss = MainLoss(reduction="mean")(denoised, target)
		return rec_loss

	def combine_loss(rec, reg):
		return rec * REC_FACT + REG_FACT * reg

	"""progress_store = pd.DataFrame(columns=(
	    'loss',
	    'reconstruction_mse',
	    'regularization_penalty',
	    "reconstruction_factor",
	    "regularization_factor",
	    "l0_norm",
	    'val_loss',
	    'val_reconstruction_mse',
	    'val_regularization_penalty',
	    "val_l0_norm",
	))
	progress_store.attrs["wavelet"] = wavelet_name
	progress_store.attrs["epochs"] = epochs
	"""
	for epoch in range(epochs):
		for (input, target) in dataset:
			denoised, coeffs = model(input)
			target = target.to(denoised.device)

			rec_loss = loss_fn(denoised, target)
			reg_loss = regularisation_fun(coeffs)
			orth_loss = model.bank.filt_bank_orthogonality_loss()

			#with inference_mode():
			#	l0_count += l0_counter(coeffs)

			total = combine_loss(rec_loss, reg_loss) + ORTH_FACT * orth_loss

			total.backward()

			optimizer.step()
			optimizer.zero_grad()

		with inference_mode():
			denoised, coeffs = model(tensor(validation[0], device="cuda:0"))
			target = tensor(validation[1], device="cuda:0")

			val_rec_loss = loss_fn(denoised, target)
			val_reg_loss = regularisation_fun(coeffs)
			val_l0_count = l0_counter(coeffs)

			val_total = VAL_REC_FACT * val_rec_loss + val_l0_count
			'''if (epoch) == 2:
				#REG_FACT = 3 / 4 * REG_FACT + 1 / (4 * float(reg_loss))
				#REC_FACT = 3 / 4 * REC_FACT + 1 / (4 * float(rec_loss))

				REG_FACT = 1 / float(reg_loss)
				REC_FACT = 1 / float(rec_loss)
				ORTH_FACT = 1 / float(orth_loss)'''

		# Store the progress
		'''progress_store.loc[epoch] = {
		    "loss": float(total),
		    "reconstruction_mse": float(rec_loss),
		    "regularization_penalty": float(reg_loss),
		    "reconstruction_factor": float(REC_FACT),
		    "regularization_factor": float(REG_FACT),
		    "l0_norm": float(l0_count),
		    "val_loss": float(val_total),
		    "val_reconstruction_mse": float(val_rec_loss),
		    "val_regularization_penalty": float(val_reg_loss),
		    "val_l0_norm": float(val_l0_count),
		}'''

		# Store the progress.
		#if (epoch + 1) % 5 == 0:
		#		progress_store.to_pickle(store_file_name)

		session.report(
		    {
		        "total_loss": float(val_total),
		        "reconstruction_mse": float(val_rec_loss),
		        "l0_pnorm": float(val_l0_count),
		        "regularization_penalty": float(val_reg_loss),
		    },
		    checkpoint=Checkpoint.from_dict(
		        dict(epoch=epoch,
		             model=model.state_dict(),
		             opt=optimizer.state_dict())))
	# Also store the computed filters.
	#progress_store.attrs["filters"] = model.bank.filter_bank.detach().cpu(
	#).numpy()
	#progress_store.to_pickle(store_file_name)
