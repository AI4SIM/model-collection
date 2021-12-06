import names
import os

from torch.utils.tensorboard import SummaryWriter

name = names.get_last_name().lower()

root_path = os.path.dirname(os.path.realpath(__file__))

data_path = os.path.join(root_path, 'data')
experiment_path = os.path.join(root_path, 'experiments', name)
logs_path = os.path.join(experiment_path, 'logs')
artifacts_path = os.path.join(experiment_path, 'artifacts')
plots_path = os.path.join(experiment_path, 'plots')

paths = [experiment_path, logs_path, artifacts_path, plots_path]
for path in paths:
    os.makedirs(path, exist_ok=True)
