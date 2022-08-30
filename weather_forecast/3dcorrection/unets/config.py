import os
import randomname

root_path = os.path.dirname(os.path.realpath(__file__))

data_path = os.path.join(root_path, 'data')

# Create all path for the current experiment.
experiment_path = os.path.join(root_path, 'experiments', randomname.get_name())
logs_path = os.path.join(experiment_path, 'logs')
artifacts_path = os.path.join(experiment_path, 'artifacts')
plots_path = os.path.join(experiment_path, 'plots')

for path in [experiment_path, logs_path, artifacts_path, plots_path]:
    os.makedirs(path, exist_ok=True)
