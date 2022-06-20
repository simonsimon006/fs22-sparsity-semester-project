from re import S
from despawn2D import Despawn2D

from torch.optim import Adadelta as Opt
from torch.nn import MSELoss as MainLoss, L1Loss
from torch import tensor, norm, save, sin, linspace, pi, manual_seed, zeros_like, device, rand_like, set_default_dtype, float64
import matplotlib.pyplot as plt
from pywt import Wavelet
from torch.nn.utils.clip_grad import clip_grad_norm_

# Set random seed for desterministic behaviour
manual_seed(0)
#set_default_dtype(float64)
device = device("cpu")

TESTING_FREQ = 5 * 2 * pi  # Hz * rad
AMPLITUDE = 10
ADAPT_FILTERS = False
EPOCHS = 55
LEARNING_RATE = 2 * 1e1

# Random choice
WAVELET_NAME = "bior1.1"

SCALING = tensor(Wavelet(WAVELET_NAME).dec_hi, device=device)
print(SCALING.shape)
REG_FACT = 0

time = linspace(0, 5, 3000, device=device)
signal = AMPLITUDE * sin(TESTING_FREQ * time).repeat(250, 1)
noise = rand_like(signal, device=device)
#noise = zeros_like(signal, device=device)
data = (signal + noise)

loss_fn = MainLoss(reduction="sum")
#regularization = L1Loss(reduction="sum")
model = Despawn2D(12, SCALING, adapt_filters=ADAPT_FILTERS)
model = model.to(device)

optimizer = Opt(model.parameters(), lr=LEARNING_RATE)

for epoch in range(EPOCHS):
	optimizer.zero_grad()

	denoised, wav_coeffs = model(signal)

	# Do regularization
	reg_loss = norm(wav_coeffs, p=1)
	error = loss_fn(signal, denoised)

	loss = error + REG_FACT * reg_loss
	#clip_grad_norm_(model.parameters(), 1)
	#loss.backward()

	#optimizer.step()

	if epoch % 2 == 0:
		print(
		    f"Epoch {epoch+1} of {EPOCHS}: Reconstruction error: {float(error)}; Regularization penalty: {float(reg_loss)}; total loss: {float(loss)}"
		)
print(list(model.parameters()))
plt.plot(time.cpu(), denoised.cpu().detach().numpy()[0, :], label="Denoised")

plt.plot(time.cpu(),
         signal.cpu().detach().numpy()[0, :],
         label="Signal",
         alpha=0.2)
plt.legend()
plt.show()
exit()
plt.savefig(
    f"sine-figures/despawn2d-{EPOCHS}-sine-{TESTING_FREQ}_Hz-amp-{AMPLITUDE}-adapt_filters-{ADAPT_FILTERS}-wavelet-{WAVELET_NAME}.svg"
)
save(
    model,
    f"models/despawn2d-{EPOCHS}-sine-{TESTING_FREQ}_Hz-amp-{AMPLITUDE}-wavelet-{WAVELET_NAME}.pt"
)
