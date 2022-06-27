import matplotlib.pyplot as plt
from pywt import Wavelet
from torch import (device, float64, linspace, manual_seed, norm, pi, rand_like,
                   save, set_default_dtype, sin, tensor, zeros_like)

from torch.nn import MSELoss as MainLoss

from torch.optim import Adadelta as Opt
from torch.optim.lr_scheduler import ReduceLROnPlateau as RLP

from despawn2D import Despawn2D
from plotter import plot_wavedec
# Set random seed for desterministic behaviour
manual_seed(0)

device = device("cuda:0")

TESTING_FREQ = 5 * 2 * pi  # Hz * rad
AMPLITUDE = 10
ADAPT_FILTERS = True
EPOCHS = 11_000
LEARNING_RATE = 2e-2

# Random choice
WAVELET_NAME = "coif5"

SCALING = tensor(Wavelet(WAVELET_NAME).dec_hi, device=device)
print(SCALING.shape)
REG_FACT = 1e-1

time = linspace(0, 5, 4096, device=device)
signal = AMPLITUDE * sin(TESTING_FREQ * time).repeat(500, 1)
noise = rand_like(signal, device=device)
#noise = zeros_like(signal, device=device)
data = (signal + noise)

loss_fn = MainLoss(reduction="sum")
#regularization = L1Loss(reduction="sum")
model = Despawn2D(12, SCALING, adapt_filters=ADAPT_FILTERS)
model = model.to(device)

optimizer = Opt(model.parameters(), lr=LEARNING_RATE)
scheduler = RLP(optimizer,
                'min',
                patience=4,
                factor=0.8,
                verbose=True,
                threshold=1e-4,
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

save(
    model,
    f"models/despawn2d-{EPOCHS}-sine-{TESTING_FREQ}_Hz-amp-{AMPLITUDE}-wavelet-{WAVELET_NAME}.pt"
)

denoised, _ = model(data)

signal = signal.cpu().detach().numpy()[0]
denoised = denoised.cpu().detach().numpy()[0]
data = data.cpu().detach().numpy()[0]
'''plt.plot(time.cpu(), denoised[0, :], label="Denoised")
plt.plot(time.cpu(), signal[0, :], label="Signal", alpha=0.2)
plt.plot(time.cpu(), data[0, :], label="Noisy signal", alpha=0.5)
plt.legend()'''

# The mexh wavelet is used because it has a continuous decomposition
plot_wavedec(
    time.cpu().detach().numpy(),
    signal,
    data,
    denoised,
    save=
    f"sine-figures/despawn2d-{EPOCHS}-sine-{TESTING_FREQ}_Hz-amp-{AMPLITUDE}-adapt_filters-{ADAPT_FILTERS}-wavelet-{WAVELET_NAME}-reg-{REG_FACT}.svg",
    name_wavelet="mexh")
