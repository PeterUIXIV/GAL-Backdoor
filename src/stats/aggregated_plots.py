import itertools
from matplotlib import pyplot as plt
from tensorboard.backend.event_processing import event_accumulator
import pandas as pd
import os

def extract_scalar_data(event_file):
    """
    Extract scalar data from a TensorBoard event file.
    
    Parameters:
        event_file (str): Path to the event file.
        
    Returns:
        pd.DataFrame: DataFrame containing scalar data.
    """
    # Initialize the EventAccumulator
    ea = event_accumulator.EventAccumulator(event_file)
    ea.Reload()

    # Extract tags
    tags = ea.Tags()['scalars']

    # Prepare a list to store all the scalar data
    scalar_data = []

    # Iterate over all tags (e.g., loss, accuracy)
    for tag in tags:
        events = ea.Scalars(tag)
        for event in events:
            scalar_data.append({
                'wall_time': event.wall_time,
                'step': event.step,
                'tag': tag,
                'value': event.value
            })

    return pd.DataFrame(scalar_data)

def process_folder_for_tags(folder_path, tag_data, root_dir, params):
    """
    Process all TensorBoard event files in a given folder and organize data by tag.
    
    Parameters:
        folder_path (str): Path to the folder containing TensorBoard event files.
        tag_data (dict): Dictionary to store tag data, organized by tag.
    """
    event_files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.startswith('events.out.tfevents')]
    
    relative_path = os.path.relpath(folder_path, root_dir)
    column_name = relative_path.replace("/", " ").replace("\\", " ").replace("_", " ")

    parts = column_name.split()
        
    # Add the extracted parameters to the respective sets
    params['attack'].add(parts[0])
    params['defense'].add(parts[1] if parts[1] != "None" else None)
    params['num_participants'].add(int(parts[4]))
    params['assist_mode'].add(parts[5])
    params['poison_percentage'].add(float(parts[-2]))
    params['num_attacker'].add(int(parts[-1]))
    
    if event_files:  # Only process if there are event files in the folder
        for event_file in event_files:
            scalar_data = extract_scalar_data(event_file)
            
            # Group data by tag
            for tag in scalar_data['tag'].unique():
                tag_specific_data = scalar_data[scalar_data['tag'] == tag][['step', 'value']]
                tag_specific_data = tag_specific_data.set_index('step')
                tag_specific_data = tag_specific_data.rename(columns={'value': column_name})

                if tag in tag_data:
                    if column_name in tag_data[tag]:
                        max_step_tag_specific = tag_specific_data.index.max()
                        if max_step_tag_specific > len(tag_data[tag].index):
                            tag_data[tag] = tag_data[tag].reindex(range(1, max_step_tag_specific + 1))
                            tag_data[tag].update(tag_specific_data)
                        else:
                            tag_data[tag].update(tag_specific_data)
                    else:
                        tag_data[tag] = pd.concat([tag_data[tag], tag_specific_data], axis=1)
                else:
                    tag_data[tag] = tag_specific_data

def create_combinations(params, exclude_param=None):
    # Filter out the parameter to be excluded
    if exclude_param:
        params = {key: value for key, value in params.items() if key != exclude_param}

    # Use itertools.product on the remaining parameters
    keys = list(params.keys())
    combinations = list(itertools.product(*(params[key] for key in keys)))
    
    # Convert the combinations to a list of dictionaries
    combinations_dicts = [dict(zip(keys, combo)) for combo in combinations]
    
    return combinations_dicts

def plot_combinations(params, tag_data, output_dir):

    for param, values in params.items():
        combs = create_combinations(params, exclude_param=param)
        # file_name = '{} {}'.format(param, '-'.join(values))

        for comb in combs:
            column_names = []
            for value in values:
                comb[param] = value
                column_name = '{} {} train 0 {} {} 10 10 search 0 {} {}'.format(comb['attack'], 
                                                                                comb['defense'], 
                                                                                comb['num_participants'], 
                                                                                comb['assist_mode'], 
                                                                                comb['poison_percentage'], 
                                                                                comb['num_attacker'])
                column_names.append(column_name)
            
            for tag, df in tag_data.items():
                plot_columns(tag, df, column_names, output_dir, param, comb)

def plot_columns(tag, df, column_names, output_dir, param, comb):

    for column in column_names:
        if column not in df.columns:
            print(f"Column '{column}' does not exist in the DataFrame.")
            return
    
    plt.figure(figsize=(10, 6))

    for column in column_names:
        plt.plot(df.index, df[column], marker='o', label=column)

    plt.xlabel('Step')
    plt.title(f'{tag} {param} {comb} over Steps')
    plt.legend()

    # Save the plot
    print(tag)
    print(output_dir)
    output_path = os.path.join(output_dir, f'{tag}_{param}_{comb} plot.png')
    print(output_path)
    plt.show()
    plt.savefig(output_path)
    plt.close()
    print(f"Plot saved to {output_path}")

def walk_through_folders_and_collect_data(root_dir):
    """
    Walk through all folders starting from root_dir and collect data by tag.
    
    Parameters:
        root_dir (str): Root directory to start the search.
    """
    tag_data = {}

    params = {
        'attack': set(),
        'defense': set(),
        'num_participants': set(),
        'assist_mode': set(),
        'poison_percentage': set(),
        'num_attacker': set()
    }

    for dirpath, _, filenames in os.walk(root_dir):
        # Only process if there are event files in the current directory
        if any(f.startswith('events.out.tfevents') for f in filenames):
            process_folder_for_tags(dirpath, tag_data, root_dir, params)
    
    plot_combinations(params, tag_data, root_dir)
    # plot_data(tag_data, root_dir)

if __name__ == "__main__":
    root_directory = "output/runs"  # replace with the path to your root directory
    
    walk_through_folders_and_collect_data(root_directory)
