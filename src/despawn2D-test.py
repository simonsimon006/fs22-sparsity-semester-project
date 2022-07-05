import matplotlib.pyplot as plt
from pywt import Wavelet
from torch import (device, float64, linspace, manual_seed, norm, pi, rand_like,
                   save, set_default_dtype, sin, tensor, zeros_like, load,
                   float32)

from torch.nn import MSELoss as MainLoss

from torch.optim import Adadelta as Opt
from torch.optim.lr_scheduler import ReduceLROnPlateau as RLP

from despawn2D import Despawn2D
from plotter import plot_measurement, plot_wavedec

from loader_old import load_train
# Set random seed for desterministic behaviour
manual_seed(0)

device = device("cuda:0")
'''
TESTING_FREQ = 5 * 2 * pi  # Hz * rad
AMPLITUDE = 10





print(SCALING.shape)


time = linspace(0, 5, 4096, device=device)
signal = AMPLITUDE * sin(TESTING_FREQ * time).repeat(500, 1)
noise = rand_like(signal, device=device)
#noise = zeros_like(signal, device=device)
data = (signal + noise)
'''
'''DS_NAME = "eps_yl_k3.mat"
ds = load_train(f"/run/media/simon/midway/PA_Data/{DS_NAME}")
signal = tensor(ds.Filtered)
data = tensor(ds.Filled)

save(signal, "signal.pt")
save(data, "data.pt")
'''

signal = load("signal.pt").to(device=device, dtype=float32)
data = load("data.pt").to(device=device, dtype=float32)
# Random choice
WAVELET_NAME = "db20"
ADAPT_FILTERS = True
EPOCHS = 4_000
LEARNING_RATE = 4e-2
SCALING = tensor(Wavelet(WAVELET_NAME).dec_hi, device=device)
REG_FACT = 1e0
RELOAD = False
loss_fn = MainLoss(reduction="sum")
#regularization = L1Loss(reduction="sum")
model = load(
    f"models/despawn2d-{EPOCHS}-meas-eps_yl_k3-wavelet-{WAVELET_NAME}.pt"
) if RELOAD else Despawn2D(12, SCALING, adapt_filters=ADAPT_FILTERS)

model = model.to(device)

optimizer = load("optimizer.pt") if RELOAD else Opt(model.parameters(),
                                                    lr=LEARNING_RATE)
scheduler = load("scheduler.pt") if RELOAD else RLP(optimizer,
                                                    'min',
                                                    patience=3,
                                                    factor=0.8,
                                                    verbose=True,
                                                    threshold=1,
                                                    threshold_mode="abs")

for epoch in range(EPOCHS):
	optimizer.zero_grad()

	denoised, wav_coeffs = model(data)

	# Do regularization
	reg_loss = norm(wav_coeffs, p=1)
	error = loss_fn(signal, denoised)

	loss = error + REG_FACT * reg_loss

	loss.backward()

	scheduler.step(error)

	optimizer.step()

	if epoch % 5 == 0:
		print(
		    f"Epoch {epoch+1} of {EPOCHS}: Reconstruction error: {float(error)}; Regularization penalty: {float(reg_loss)}; total loss: {float(loss)}"
		)

save(model,
     f"models/despawn-reg-{EPOCHS}-meas-eps_yl_k3-wavelet-{WAVELET_NAME}.pt")

denoised, _ = model(data)

signal = signal.cpu().detach().numpy()
denoised = denoised.cpu().detach().numpy()
data = data.cpu().detach().numpy()
'''plt.plot(time.cpu(), denoised[0, :], label="Denoised")
plt.plot(time.cpu(), signal[0, :], label="Signal", alpha=0.2)
plt.plot(time.cpu(), data[0, :], label="Noisy signal", alpha=0.5)
plt.legend()'''

# The mexh wavelet is used because it has a continuous decomposition
plot_measurement(signal,
                 data,
                 denoised,
                 save=f"denoised-reg-epochs-{EPOCHS}-eps_yl_k3.png",
                 name_measurement="eps_yl_k3",
                 name_wavelet="morlet")
