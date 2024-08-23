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

def ensure_columns(df, required_columns):
    for col in required_columns:
        if col not in df.columns:
            df[col] = pd.NA
    return df

def process_folder_for_tags(folder_path, tag_data, root_dir):
    """
    Process all TensorBoard event files in a given folder and organize data by tag.
    
    Parameters:
        folder_path (str): Path to the folder containing TensorBoard event files.
        tag_data (dict): Dictionary to store tag data, organized by tag.
    """
    event_files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.startswith('events.out.tfevents')]
    
    relative_path = os.path.relpath(folder_path, root_dir)
    if event_files:  # Only process if there are event files in the folder
        for event_file in event_files:
            scalar_data = extract_scalar_data(event_file)
            
            # Group data by tag
            for tag in scalar_data['tag'].unique():
                tag_specific_data = scalar_data[scalar_data['tag'] == tag][['step', 'value']]
                tag_specific_data = tag_specific_data.set_index('step')
                tag_specific_data = tag_specific_data.rename(columns={'value': relative_path})

                if tag in tag_data:
                    if relative_path in tag_data[tag]:
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


def write_tag_data_to_csv(tag_data, parent_folder):
    """
    Write the organized tag data to CSV files, one for each tag.
    
    Parameters:
        tag_data (dict): Dictionary containing tag data, organized by tag.
        parent_folder (str): The root directory where CSV files will be saved.
    """
    for tag, df in tag_data.items():
        output_csv_path = os.path.join(parent_folder, f"{tag.replace('/', '_')}.csv")
        df.to_csv(output_csv_path)
        print(f"Data for tag '{tag}' has been written to {output_csv_path}")

def walk_through_folders_and_collect_data(root_dir):
    """
    Walk through all folders starting from root_dir and collect data by tag.
    
    Parameters:
        root_dir (str): Root directory to start the search.
    """
    tag_data = {}

    for dirpath, _, filenames in os.walk(root_dir):
        # Only process if there are event files in the current directory
        if any(f.startswith('events.out.tfevents') for f in filenames):
            process_folder_for_tags(dirpath, tag_data, root_dir)
    
    write_tag_data_to_csv(tag_data, root_dir)

if __name__ == "__main__":
    root_directory = "output/runs"  # replace with the path to your root directory
    
    walk_through_folders_and_collect_data(root_directory)
