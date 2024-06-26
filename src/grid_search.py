import yaml
import itertools
import subprocess

def generate_param_combinations(param_grid):
    keys, values = zip(*param_grid.items())
    experiments = [dict(zip(keys, v)) for v in itertools.product(*values)]
    return experiments

# Load the YAML file
with open('parameters.yml', 'r') as file:
    params = yaml.safe_load(file)

param_grid = params['param_grid']

# Generate parameter combinations
param_combinations = generate_param_combinations(param_grid)

best_score = 0
best_params = None

for param_set in param_combinations:
    # Construct the command to run the ML script with the current parameters
    cmd = [
        'python', 'train_model_assi_org.py',
        '--num_participants', int(param_set['num_participants']),
        '--num_attackers', int(param_set['num_attackers']),
        '--assist_mode', str(param_set['assist_mode']),
        '--min_samples_leaf', str(param_set['min_samples_leaf'])
    ]

    # Remove empty arguments
    cmd = [arg for arg in cmd if arg]

    # Execute the command and capture the output
    subprocess.run(cmd, text=True)