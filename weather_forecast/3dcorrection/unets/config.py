import os
import randomname

root_path = os.path.dirname(os.path.realpath(__file__))

data_path = os.path.join(root_path, 'data')

# Create all path for the current experiment.
experiments_path = os.path.join(root_path, 'experiments')
os.makedirs(experiments_path, exist_ok=True)
_existing_xps = os.listdir(experiments_path)

# Generate experiment name
_randomize_name = True
while _randomize_name:
    _experiment_name = randomname.get_name()
    if _experiment_name not in _existing_xps:
        break

experiment_path = os.path.join(experiments_path, _experiment_name)

if os.getenv("AI4SIM_EXPERIMENT_PATH") is None:
    os.environ["AI4SIM_EXPERIMENT_PATH"] = experiment_path
else:
    experiment_path = os.getenv("AI4SIM_EXPERIMENT_PATH")

logs_path = os.path.join(experiment_path, 'logs')
artifacts_path = os.path.join(experiment_path, 'artifacts')
plots_path = os.path.join(experiment_path, 'plots')

for path in [experiment_path, logs_path, artifacts_path, plots_path]:
    os.makedirs(path, exist_ok=True)
