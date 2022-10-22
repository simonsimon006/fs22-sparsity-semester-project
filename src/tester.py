from plotter import plot_measurement
from despawn2D import Despawn2D
import torch as t

#model = Despawn2D(14, [1, 2, 3])
model = t.load("/home/simon/Code/semester-project-fs22/synth-run/test-high-lr/cos-test-500-wavelet-dmey.pt", map_location="cpu")
model.device = "cpu"
#model.load_state_dict(model_state)
#validation = t.load("/home/simon/Code/semester-project-fs22/synth-run/New folder/cos_val.pt")

#signal = t.load("/home/simon/Code/semester-project-fs22/synth-run/New folder/val_sig.pt", map_location="cpu")

validation, signal = t.load("/home/simon/Code/semester-project-fs22/synth-run/small-one/eps_S2-ZG_04.pt", map_location = "cpu")
denoised, coeffs = model(t.tensor(validation, dtype=t.float32))

plot_measurement(signal, validation, denoised.cpu().detach().numpy(), "synth-test-k3.png", "eps_S2-ZG_04")
