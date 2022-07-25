from pywt import Wavelet
from torch import device, float32, load, manual_seed, qint8, save, tensor
from torch.nn import MSELoss as MainLoss
from torch.optim import LBFGS
from torch.optim import AdamW as Opt
from torch.optim.lr_scheduler import ReduceLROnPlateau as RLP
from tqdm import trange

from despawn2D import Despawn2D as Denoiser
from plotter import plot_measurement
from util import make_noisy_sine

# Set random seed for desterministic behaviour
manual_seed(0)

device = device("cuda:0")
REC_TEST = True
EPOCHS = 1 if REC_TEST else 1000
LEARNING_RATE = 1e-2
REG_FACT = 1e0
ADAPT_FILTERS = True
MEASUREMENT = "eps_yl_k3"

# Random choice
WAVELET_NAME = "db20"

#signal, data = make_noisy_sine(device=device)

signal = load("signal.pt").to(device=device, dtype=float32)
data = load("data.pt").to(device=device, dtype=float32)
target = data if REC_TEST else signal

SCALING = tensor(Wavelet(WAVELET_NAME).dec_lo, device=device)
SCALING_REC = tensor(Wavelet(WAVELET_NAME).rec_lo, device=device)

loss_fn = MainLoss(reduction="mean")

model = Denoiser(10,
                 SCALING,
                 SCALING_REC,
                 adapt_filters=ADAPT_FILTERS,
                 filter=not REC_TEST)

model = model.to(device)

optimizer = Opt(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-2)
scheduler = RLP(optimizer,
                'min',
                patience=5,
                factor=0.7,
                verbose=False,
                threshold=1e-4,
                threshold_mode="rel",
                min_lr=1e-7)

with trange(EPOCHS) as t:
	for epoch in t:
		denoised, reg_loss = model(data)
		# Do regularization
		error = loss_fn(target, denoised)
		loss = error + REG_FACT * reg_loss
		t.set_postfix(loss=float(loss),
		              reconstruction=float(error),
		              regularisation=float(reg_loss))
		if not REC_TEST:
			loss.backward()
			optimizer.step()
			scheduler.step(loss)
		optimizer.zero_grad()

# Use LBFGS for the last optim parts.
last_stage_optim = LBFGS(
    model.parameters(),
    #max_iter=200,
    #lr=1e-5,
    #line_search_fn="strong-wolfe",
)


def evaluator():
	denoised, reg_loss = model(data)
	error = loss_fn(target, denoised)
	loss = error + REG_FACT * reg_loss
	model.zero_grad()
	return loss


last_stage_optim.step(evaluator)

save(
    model,
    f"models/despawn-reg-{EPOCHS}-meas-{MEASUREMENT}-wavelet-{WAVELET_NAME}.pt")

denoised, reg_loss = model(data)

print(f"Loss after lbfgs: {REG_FACT * reg_loss + loss_fn(denoised, target): e}")

signal = signal.cpu().detach().numpy()
denoised = denoised.cpu().detach().numpy()
data = data.cpu().detach().numpy()

# The mexh wavelet is used because it has a continuous decomposition
plot_measurement(signal,
                 data,
                 denoised,
                 save=f"{MEASUREMENT}-test-{WAVELET_NAME}.png",
                 name_measurement=MEASUREMENT,
                 name_wavelet="morlet")
