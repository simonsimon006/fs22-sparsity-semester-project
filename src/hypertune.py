from ray import tune
from ray.air import session, RunConfig
from ray.tune import CLIReporter
from ray.tune.search.hyperopt import HyperOptSearch as SearchAlgo
from ray.tune.schedulers import ASHAScheduler
import pathlib as pl
from train_model import *
from loader import MeasurementFolder
import torch
from pywt import Wavelet
from torch.utils.data import DataLoader
from functools import partial
from datetime import datetime

dev = "cuda:0"

NOW = str(datetime.now().timestamp()).replace(".", "_")

run_config = RunConfig(name=NOW)
wavelist = {
    name: torch.tensor(Wavelet(name).dec_hi).to(dev)
    for name in ["db20", "dmey", "db2", "db10", "bior6.8", "sym20", "sym10"]
}

factors = [5, 10, 20, 30]
for fac in factors:
	wavelist[f"{fac}xdmey"] = (torch.tensor(Wavelet("dmey").dec_hi) /
	                           fac).repeat(fac).to(dev)


def end_ep(epoch, total, l0, model, optimizer):
	'''with tune.checkpoint_dir(epoch) as checkpoint_dir:
		path = pl.Path(checkpoint_dir) / pl.Path("checkpoint")
		torch.save((model.state_dict(), optimizer.state_dict()), path)'''

	session.report({"loss": total, "l0": l0})


def tune_loop(config, dataloader):
	model = make_model(config["levels"], config["wav"][1], device="cuda:0")
	opt = make_optimizer(model, 1e-4, 3e-1)
	train_loop(model=model,
	           optimizer=opt,
	           dataset=dataloader,
	           epoch_end_fun=end_ep,
	           wavelet_name=config["wav"][0],
	           epochs=10000)


config = {
    "levels": tune.randint(2, 17),
    "wav": tune.choice(wavelist.items()),
}
scheduler = ASHAScheduler(time_attr="training_iteration",
                          max_t=10,
                          grace_period=5,
                          reduction_factor=2)
reporter = CLIReporter(parameter_columns=[
    "levels",
],
                       metric_columns=["loss", "l0"])
datasets = DataLoader(
    MeasurementFolder("/home/simon/Code/semester-project-fs22/store/"),
    pin_memory=True,
    batch_size=None,
)

algo = SearchAlgo(random_state_seed=0)

tune_config = tune.TuneConfig(
    metric="loss",
    mode="min",
    num_samples=1,
    scheduler=scheduler,
    search_alg=algo,
)
tuner = tune.Tuner(tune.with_resources(partial(tune_loop, dataloader=datasets),
                                       resources={
                                           "cpu": 2,
                                           "gpu": 1,
                                       }),
                   tune_config=tune_config,
                   param_space=config,
                   run_config=run_config)
results = tuner.fit()

best_result = results.get_best_result()
best_config = best_result.config

torch.save(best_result, "best_model.pt")

print(best_config)
best_result.metrics_dataframe.to_json("best_result_metrics.json")