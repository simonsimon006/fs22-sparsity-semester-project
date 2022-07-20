from pywt import Wavelet
from torch import device, float32, load, manual_seed, save, tensor, qint8
from torch.cuda.amp import GradScaler, autocast
from torch.nn import MSELoss as MainLoss
from torch.optim import AdamW as Opt, LBFGS
from torch.optim.lr_scheduler import ReduceLROnPlateau as RLP
from tqdm import trange
from torch.quantization import quantize_dynamic
from despawn1D import HardThreshold
from despawn2D import Despawn2D as Denoiser
from loader_old import load_train
from plotter import plot_measurement
from util import make_noisy_sine
from math import log10
# Set random seed for desterministic behaviour
manual_seed(0)

device = device("cuda:0")
REC_TEST = False
USE_AMP = True
EPOCHS = 1 if REC_TEST else 3000
LEARNING_RATE = 3e-3
REG_FACT = 1e0

#signal, data = make_noisy_sine(device=device)
'''DS_NAME = "eps_yl_k3.mat"
ds = load_train(f"/run/media/simon/midway/PA_Data/{DS_NAME}")
signal = tensor(ds.Filtered)
data = tensor(ds.Filled)

save(signal, "signal.pt")
save(data, "data.pt")
'''

MEASUREMENT = "eps_yl_k3"
signal = load("signal.pt").to(device=device, dtype=float32)[:4001, :]
data = load("data.pt").to(device=device, dtype=float32)[:4001, :]
target = data
# Random choice
WAVELET_NAME = "db2"
ADAPT_FILTERS = True

SCALING = tensor(Wavelet(WAVELET_NAME).dec_lo, device=device)
SCALING_REC = tensor(Wavelet(WAVELET_NAME).rec_lo, device=device)

loss_fn = MainLoss(reduction="mean")

model = Denoiser(11,
                 SCALING,
                 SCALING_REC,
                 adapt_filters=ADAPT_FILTERS,
                 filter=not REC_TEST)

model = model.to(device)

optimizer = Opt(model.parameters(), lr=LEARNING_RATE, weight_decay=0)
scheduler = RLP(optimizer,
                'min',
                patience=5,
                factor=0.7,
                verbose=False,
                threshold=1e-4,
                threshold_mode="rel",
                min_lr=1e-7)
scaler = GradScaler(enabled=USE_AMP)
with trange(EPOCHS) as t:
	for epoch in t:
		with autocast(enabled=USE_AMP):
			denoised, reg_loss = model(data)

			# Do regularization
			error = loss_fn(target, denoised)

			loss = error + REG_FACT * reg_loss
			t.set_postfix(loss=float(loss),
			              reconstruction=float(error),
			              regularisation=float(reg_loss))

			if not REC_TEST:
				scaler.scale(loss).backward()
				scaler.step(optimizer)

				scaler.update()
				scheduler.step(loss)
			optimizer.zero_grad()
'''
# Use LBFGS for the last optim parts.
last_stage_optim = LBFGS(model.parameters(),
                         max_iter=200,
                         lr=1e-5,
                         line_search_fn="strong-wolfe")


def evaluator():
	denoised, reg_loss = model(data)
	error = loss_fn(target, denoised)
	loss = error + REG_FACT * reg_loss
	model.zero_grad()
	return loss


last_stage_optim.step(evaluator)

# Quantization step
quantize_dynamic(model, {HardThreshold}, dtype=qint8, inplace=True)

save(
    model,
    f"models/despawn-reg-{EPOCHS}-meas-{MEASUREMENT}-wavelet-{WAVELET_NAME}.pt")
'''
denoised, reg_loss = model(data)

print(
    f"Loss after quantization: {REG_FACT * reg_loss + loss_fn(denoised, target): e}"
)

signal = signal.cpu().detach().numpy()
denoised = denoised.cpu().detach().numpy()
data = data.cpu().detach().numpy()

# The mexh wavelet is used because it has a continuous decomposition
plot_measurement(signal,
                 data,
                 denoised,
                 save=f"{MEASUREMENT}-test-lift.png",
                 name_measurement=MEASUREMENT,
                 name_wavelet="morlet")
