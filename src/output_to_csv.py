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

def reshape_data(df):
    """
    Reshape the DataFrame such that each row corresponds to a step and each column to a tag.
    
    Parameters:
        df (pd.DataFrame): Input DataFrame with 'step', 'tag', and 'value' columns.
        
    Returns:
        pd.DataFrame: Reshaped DataFrame with one row per step and one column per tag, including wall_time.
    """
    # Remove rows with the tag "train/Loss"
    df = df[df['tag'] != 'train/Loss']

    # Pivot the DataFrame to have tags as columns and steps as rows
    reshaped_df = df.pivot_table(index='step', columns='tag', values='value')

    # Reset the index so that 'step' becomes a column again
    reshaped_df = reshaped_df.reset_index()

    # Add wall_time by taking the first occurrence of wall_time for each step
    wall_time_series = df.groupby('step')['wall_time'].first()
    reshaped_df['wall_time'] = reshaped_df['step'].map(wall_time_series)

    return reshaped_df

def process_folder(folder_path):
    """
    Process all TensorBoard event files in a given folder and create an output.csv file.
    
    Parameters:
        folder_path (str): Path to the folder containing TensorBoard event files.
    """
    event_files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.startswith('events.out.tfevents')]
    
    if event_files:  # Only process if there are event files in the folder
        all_scalar_data = pd.DataFrame()

        for event_file in event_files:
            scalar_data = extract_scalar_data(event_file)
            all_scalar_data = pd.concat([all_scalar_data, scalar_data], ignore_index=True)

        # Reshape the data to have one row per step and one column per tag, including wall_time
        reshaped_data = reshape_data(all_scalar_data)

        # Write to CSV in the same folder
        output_csv_path = os.path.join(folder_path, 'output.csv')
        reshaped_data.to_csv(output_csv_path, index=False)
        print(f"Data has been written to {output_csv_path}")

def walk_through_folders(root_dir):
    """
    Walk through all folders starting from root_dir and process each one.
    
    Parameters:
        root_dir (str): Root directory to start the search.
    """
    for dirpath, _, filenames in os.walk(root_dir):
        # Only process if there are event files in the current directory
        if any(f.startswith('events.out.tfevents') for f in filenames):
            process_folder(dirpath)

if __name__ == "__main__":
    root_directory = "output"  # replace with the path to your root directory
    
    walk_through_folders(root_directory)
