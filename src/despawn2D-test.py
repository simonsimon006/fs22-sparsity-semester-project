from datetime import datetime

import pandas as pd
from pywt import Wavelet
from torch import device, float32, load, manual_seed, norm, save, tensor, count_nonzero
from torch.nn import MSELoss as MainLoss
from torch.optim import LBFGS
from torch.optim import AdamW as Opt
from torch.optim.lr_scheduler import ReduceLROnPlateau as RLP
from tqdm import trange

from despawn2D import Despawn2D as Denoiser
from loader_old import load_train
from plotter import plot_measurement
from util import make_noisy_sine, sqrt_reg, l0_counter

# Set random seed for desterministic behaviour
manual_seed(0)

device = device("cuda:0")
REC_TEST = False
EPOCHS = 1 if REC_TEST else 3000
LEARNING_RATE = 1e-4
LR_HT = 3e-1
REG_FACT = 1e-1
REC_FACT = 1e-7
MEASUREMENT = "eps_yl_k3"
# More entries -> longer filters seem to be better. NN can adapt more I think.
WAVELET_NAME = "db20"
FILTER_LEN_FACTOR = 20
ADAPT_FILTERS = True
FILTER = True
LEVELS = 13

NOW = str(datetime.now().timestamp()).replace(".", "_")
store_file_name = f"start_{NOW}-{MEASUREMENT}-{WAVELET_NAME}.pkl"

#signal, data = make_noisy_sine(device=device)

signal = load("signal.pt").to(device=device, dtype=float32)
data = load("data.pt").to(device=device, dtype=float32)
target = data

losses = lambda den, reg_loss: REC_FACT * loss_fn(target, den
                                                  ) + REG_FACT * reg_loss

# Sets the scaling / wavelet function. If it is repeated so we have more weights
# we also scale it down so the total result of the convolution does tripple.
SCALING = tensor(Wavelet(WAVELET_NAME).dec_hi,
                 device=device).repeat(FILTER_LEN_FACTOR) / FILTER_LEN_FACTOR

loss_fn = MainLoss(reduction="mean")

model = Denoiser(
    LEVELS,
    SCALING,
    adapt_filters=ADAPT_FILTERS,
    filter=FILTER,
    reg_loss_fun=sqrt_reg,
    padding=10 * FILTER_LEN_FACTOR,
)

model = model.to(device)

optimizer = Opt([{
    'params': model.scaling
}, {
    'params': model.threshold.parameters(),
    'lr': LR_HT
}],
                lr=LEARNING_RATE)

scheduler = RLP(optimizer,
                'min',
                patience=100,
                factor=0.5,
                verbose=True,
                threshold=1e-4,
                threshold_mode="abs",
                min_lr=1e-7)

progress_store = pd.DataFrame(columns=(
    'loss',
    'reconstruction_mse',
    'regularization_penalty',
    "reconstruction_factor",
    "regularization_factor",
    "l0_norm",
))
progress_store.attrs["wavelet"] = WAVELET_NAME
progress_store.attrs["measurement"] = MEASUREMENT
progress_store.attrs["epochs"] = EPOCHS
progress_store.attrs["learning_rate"] = LEARNING_RATE
progress_store.attrs["device"] = str(device)

with trange(EPOCHS) as t:
	for epoch in t:
		denoised, coeffs = model(data)

		l0 = l0_counter(coeffs)

		# Do regularization
		reg_loss = sqrt_reg(coeffs)

		error = loss_fn(target, denoised)

		loss = losses(denoised, reg_loss)

		t.set_postfix(loss=float(loss),
		              reconstruction=float(error),
		              regularization=float(reg_loss),
		              alpha=float(model.threshold.alpha),
		              b_minus=float(model.threshold.b_minus),
		              b_plus=float(model.threshold.b_plus),
		              l0_avg=l0)

		# Store the progress
		progress_store.loc[epoch] = {
		    "loss": float(loss),
		    "reconstruction_mse": float(error),
		    "regularization_penalty": float(reg_loss),
		    "reconstruction_factor": float(REC_FACT),
		    "regularization_factor": float(REG_FACT),
		    "l0_norm": l0,
		}

		if not REC_TEST:
			loss.backward()
			optimizer.step()
			scheduler.step(loss)

		# Rebalances the weighing of the components with a moving average.
		if (epoch + 1) % 1000 == 0:
			REG_FACT = 1 / (4 * float(reg_loss)) + (3 / 4) * REG_FACT
			REC_FACT = 1 / (4 * error.detach()) + (3 / 4) * REC_FACT

		# Store the progress.
		if (epoch + 1) % 300 == 0:
			progress_store.to_pickle(store_file_name)
		optimizer.zero_grad()

# Use LBFGS for the last optim parts.
last_stage_optim = LBFGS(model.parameters())


def evaluator():
	denoised, coeffs = model(data)
	reg_loss = sqrt_reg(coeffs)
	return losses(denoised, reg_loss)


last_stage_optim.step(evaluator)

save(
    model,
    f"models/despawn-reg-{EPOCHS}-meas-{MEASUREMENT}-wavelet-{WAVELET_NAME}.pt")

denoised, coeffs = model(data)
reg_loss = sqrt_reg(coeffs)

print(f"Loss after LBFGS: {losses(denoised, reg_loss)}")

progress_store.loc[EPOCHS] = {
    "loss": float(losses(denoised, reg_loss)),
    "reconstruction_mse": float(loss_fn(denoised, signal)),
    "regularization_penalty": float(reg_loss),
    "reconstruction_factor": float(REC_FACT),
    "regularization_factor": float(REG_FACT),
    "l0_norm": l0_counter(coeffs),
}

# Also store the computed filters.
progress_store.attrs["filters"] = model.scaling.detach().cpu().numpy()

progress_store.to_pickle(store_file_name)

signal = signal.cpu().detach().numpy()
denoised = denoised.cpu().detach().numpy()
data = data.cpu().detach().numpy()

# The mexh wavelet is used because it has a continuous decomposition
plot_measurement(signal,
                 data,
                 denoised,
                 save=f"start_{NOW}_{MEASUREMENT}-{WAVELET_NAME}.png",
                 name_measurement=MEASUREMENT,
                 name_wavelet="morlet")
