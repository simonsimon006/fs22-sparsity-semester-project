from pywt import Wavelet
from torch import (
    device,
    linspace,
    manual_seed,
    norm,
    pi,
    rand_like,
    sin,
    tensor,
    load,
)
from torch.cuda.amp import GradScaler, autocast
from torch.nn import MSELoss as MainLoss

from torch.optim import Adadelta as Opt
from torch.optim.lr_scheduler import ReduceLROnPlateau as RLP

from despawn2D import Despawn2D as Denoiser
from plotter import plot_measurement

from loader_old import load_train
# Set random seed for desterministic behaviour
manual_seed(0)

device = device("cuda:0")
REC_TEST = False

TESTING_FREQ = 5 * 2 * pi  # Hz * rad
AMPLITUDE = 10
NOISE_AMP = 0 if REC_TEST else 4
SQUARE = 4000
time = linspace(0, 5, SQUARE, device=device)
signal = AMPLITUDE * sin(TESTING_FREQ * time).repeat(SQUARE, 1)
noise = NOISE_AMP * (rand_like(signal, device=device) - 0.5)

#noise = zeros_like(signal, device=device)
data = (signal + noise)
'''DS_NAME = "eps_yl_k3.mat"
ds = load_train(f"/run/media/simon/midway/PA_Data/{DS_NAME}")
signal = tensor(ds.Filtered)
data = tensor(ds.Filled)

save(signal, "signal.pt")
save(data, "data.pt")
'''
'''
signal = load("signal.pt").to(device=device, dtype=float32)
data = load("data.pt").to(device=device, dtype=float32)
'''
# Random choice
WAVELET_NAME = "db20"
ADAPT_FILTERS = True
EPOCHS = 1 if REC_TEST else 1000
LEARNING_RATE = 1e-1
SCALING = tensor(Wavelet(WAVELET_NAME).dec_hi, device=device)
REG_FACT = 1e-2
RELOAD = False
loss_fn = MainLoss(reduction="sum")
#regularization = L1Loss(reduction="sum")
model = load(
    f"models/despawn2d-{EPOCHS}-meas-eps_yl_k3-wavelet-{WAVELET_NAME}.pt"
) if RELOAD else Denoiser(
    2, SCALING, adapt_filters=ADAPT_FILTERS, filter=not REC_TEST)

model = model.to(device)

optimizer = load("optimizer.pt") if RELOAD else Opt(model.parameters(),
                                                    lr=LEARNING_RATE)
scheduler = load("scheduler.pt") if RELOAD else RLP(optimizer,
                                                    'min',
                                                    patience=3,
                                                    factor=0.8,
                                                    verbose=True,
                                                    threshold=1e-4,
                                                    threshold_mode="rel")
scaler = GradScaler()

for epoch in range(EPOCHS):
	optimizer.zero_grad()
	with autocast():
		denoised, wav_coeffs = model(data)

		# Do regularization
		reg_loss = norm(wav_coeffs, p=1)
		error = loss_fn(signal, denoised)

		loss = error + REG_FACT * reg_loss

		if not REC_TEST:

			scaler.scale(loss).backward()
			scaler.step(optimizer)

			scaler.update()
			scheduler.step(error)

	if epoch % 5 == 0:
		print(
		    f"Epoch {epoch+1} of {EPOCHS}: Reconstruction error: {float(error)}; Regularization penalty: {float(reg_loss)}; total loss: {float(loss)}"
		)

#save(model,
#     f"models/despawn-reg-{EPOCHS}-meas-eps_yl_k3-wavelet-{WAVELET_NAME}.pt")

denoised, _ = model(data)

signal = signal.cpu().detach().numpy()
denoised = denoised.cpu().detach().numpy()
data = data.cpu().detach().numpy()

# The mexh wavelet is used because it has a continuous decomposition
plot_measurement(signal,
                 data,
                 denoised,
                 save=f"sine-test.png",
                 name_measurement="sine-test",
                 name_wavelet="morlet")
