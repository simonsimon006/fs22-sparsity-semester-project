import pandas as pd
from pywt import Wavelet
from torch import device, float32, load, manual_seed, norm, save, tensor
from torch.cuda.amp import GradScaler, autocast
from torch.nn import MSELoss as MainLoss
from torch.optim import LBFGS
from torch.optim import AdamW as Opt
from torch.optim.lr_scheduler import ReduceLROnPlateau as RLP
from tqdm import trange

from despawn2D import Despawn2D as Denoiser
from loader_old import load_train
from plotter import plot_measurement
from util import make_noisy_sine
from datetime import datetime
# Set random seed for desterministic behaviour
manual_seed(0)

device = device("cuda:0")
REC_TEST = False
USE_AMP = True
EPOCHS = 1 if REC_TEST else 100
LEARNING_RATE = 1e-3
REG_FACT = 1e-2
REC_FACT = 1e-5
MEASUREMENT = "eps_yl_k3"
# More entries -> longer filters seem to be better. NN can adapt more I think.
WAVELET_NAME = "dmey"
ADAPT_FILTERS = True

NOW = str(datetime.now().timestamp()).replace(".", "_")
store_file_name = f"start_{NOW}-{MEASUREMENT}-{WAVELET_NAME}.pkl"

#signal, data = make_noisy_sine(device=device)
'''DS_NAME = "eps_yl_k3.mat"
ds = load_train(f"/run/media/simon/midway/PA_Data/{DS_NAME}")
signal = tensor(ds.Filtered)
data = tensor(ds.Filled)

save(signal, "signal.pt")
save(data, "data.pt")
'''

signal = load("signal.pt").to(device=device, dtype=float32)[:4001, :]
data = load("data.pt").to(device=device, dtype=float32)[:4001, :]
losses = lambda den, reg_loss: REC_FACT * loss_fn(signal, denoised
                                                  ) + REG_FACT * reg_loss

SCALING = tensor(Wavelet(WAVELET_NAME).dec_hi, device=device)

loss_fn = MainLoss(reduction="mean")

model = Denoiser(11, SCALING, adapt_filters=ADAPT_FILTERS, filter=not REC_TEST)

model = model.to(device)

optimizer = Opt(model.parameters(), lr=LEARNING_RATE)
'''scheduler = RLP(optimizer,
                'min',
                patience=100,
                factor=0.5,
                verbose=False,
                threshold=5e-4,
                threshold_mode="rel",
                min_lr=1e-7)'''

progress_store = pd.DataFrame(columns=(
    'loss',
    'reconstruction_mse',
    'regularization_penalty',
    "reconstruction_factor",
    "regularization_factor",
))
progress_store.attrs["wavelet"] = WAVELET_NAME
progress_store.attrs["measurement"] = MEASUREMENT
progress_store.attrs["epochs"] = EPOCHS
progress_store.attrs["learning_rate"] = LEARNING_RATE
progress_store.attrs["device"] = str(device)

with trange(EPOCHS) as t:
	for epoch in t:
		denoised, reg_loss = model(data)

		# Do regularization
		error = loss_fn(signal, denoised)

		loss = losses(denoised, reg_loss)
		t.set_postfix(loss=float(loss),
		              reconstruction=float(error),
		              regularisation=float(reg_loss))

		# Store the progress
		progress_store.loc[epoch] = {
		    "loss": float(loss),
		    "reconstruction_mse": float(error),
		    "regularization_penalty": float(reg_loss),
		    "reconstruction_factor": float(REC_FACT),
		    "regularization_factor": float(REG_FACT)
		}

		if not REC_TEST:
			loss.backward()
			optimizer.step()
			#scheduler.step(loss)

		# Rebalances the weighing of the components with a moving average.
		if epoch % 300 == 0:
			REG_FACT = 1 / (4 * reg_loss.detach()) + (3 / 4) * REG_FACT
			REC_FACT = 1 / (4 * error.detach()) + (3 / 4) * REC_FACT

			progress_store.to_pickle(store_file_name)
		optimizer.zero_grad()

# Use LBFGS for the last optim parts.
last_stage_optim = LBFGS(model.parameters())


def evaluator():
	denoised, reg_loss = model(data)
	return losses(denoised, reg_loss)


last_stage_optim.step(evaluator)

save(
    model,
    f"models/despawn-reg-{EPOCHS}-meas-{MEASUREMENT}-wavelet-{WAVELET_NAME}.pt")

denoised, reg_loss = model(data)

print(f"Loss after LBFGS: {losses(denoised, reg_loss)}")

progress_store.loc[EPOCHS] = {
    "loss": float(losses(denoised, reg_loss)),
    "reconstruction_mse": float(loss_fn(denoised, signal)),
    "regularization_penalty": float(reg_loss),
    "reconstruction_factor": float(REC_FACT),
    "regularization_factor": float(REG_FACT)
}
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
