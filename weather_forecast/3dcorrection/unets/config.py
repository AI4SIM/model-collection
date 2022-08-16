import os
import randomname
import shutil

root_path = os.path.dirname(os.path.realpath(__file__))

data_path = os.path.join(root_path, 'data')

# Create all path for the current experiment.
experiment_path = os.path.join(root_path, 'experiments', randomname.get_name())
logs_path = os.path.join(experiment_path, 'logs')
artifacts_path = os.path.join(experiment_path, 'artifacts')
plots_path = os.path.join(experiment_path, 'plots')

for path in [experiment_path, logs_path, artifacts_path, plots_path]:
    os.makedirs(path, exist_ok=True)

if os.getenv("AI4SIM_EXPERIMENT_PATH") is None:
    os.environ["AI4SIM_EXPERIMENT_PATH"] = experiment_path
    os.environ["AI4SIM_LOGS_PATH"] = logs_path
    os.environ["AI4SIM_ARTIFACTS_PATH"] = artifacts_path
    os.environ["AI4SIM_PLOTS_PATH"] = plots_path
elif os.getenv("AI4SIM_EXPERIMENT_PATH") != experiment_path:
    shutil.rmtree(experiment_path)
    experiment_path = os.getenv("AI4SIM_EXPERIMENT_PATH")
    logs_path = os.getenv("AI4SIM_LOGS_PATH")
    artifacts_path = os.getenv("AI4SIM_ARTIFACTS_PATH")
    plots_path = os.getenv("AI4SIM_PLOTS_PATH")
