import sys
import yaml
import itertools
import subprocess

done = 0

def generate_param_combinations(param_grid):
    keys, values = zip(*param_grid.items())
    experiments = [dict(zip(keys, v)) for v in itertools.product(*values)]
    filtered_experiments = [
        exp for exp in experiments 
        if not ((exp['attack'] is None and exp['num_attackers'] != 0) or 
                (exp['num_attackers'] == 0 and exp['attack'] is not None) or
                (exp['attack'] is None and exp['poison_percentage'] !=0.02))
    ]
    return filtered_experiments

# Load the YAML file
with open('parameters.yml', 'r') as file:
    params = yaml.safe_load(file)
    
with open('config.yml', 'r') as file:
    config = yaml.safe_load(file)

param_grid = params['param_grid']

# Generate parameter combinations
param_combinations = generate_param_combinations(param_grid)
for line in param_combinations:
    print(line)

best_score = 0
best_params = None

python_executable = sys.executable

for idx, param_set in enumerate(param_combinations):
    if idx < done:
        continue
    
    print(f"Experiment [{idx+1}/{len(param_combinations)}]")
    # Construct the command to run the ML script with the current parameters
    cmd = [
        python_executable, 'train_model_assi_org.py',
        '--control_name', '{}_{}_{}_{}_search_0'.format(param_set['num_participants'], 
                                                     param_set['assist_mode'], 
                                                     config['control']['global_epoch'], 
                                                     config['control']['local_epoch']),
        '--num_attackers', str(param_set['num_attackers']),
        '--attack', str(param_set['attack']),
        '--poison_percentage', str(param_set['poison_percentage']),
        '--defense', str(param_set['defense']),
    ]

    # Remove empty arguments
    cmd = [arg for arg in cmd if arg]

    # Execute the command and capture the output
    try:
        # Execute the command and capture the output
        print(f"Executing: {cmd}")
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        for line in result.stdout.splitlines():
            print(line)

    except subprocess.CalledProcessError as e:
        print(f"Command '{cmd}' failed with error: {e.stderr}")
        break