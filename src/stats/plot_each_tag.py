from tensorboard.backend.event_processing import event_accumulator
import pandas as pd
import os
import matplotlib.pyplot as plt

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

def plot_data(df, output_dir):
    """
    Plot the data for each tag.
    
    Parameters:
        df (pd.DataFrame): Input DataFrame with 'step', 'tag', and 'value' columns.
    """
    unique_tags = df['tag'].unique()

    for tag in unique_tags:
        tag_df = df[df['tag'] == tag]
        plt.figure()
        plt.plot(tag_df['step'], tag_df['value'], label=tag)
        plt.xlabel('Step')
        plt.ylabel('Value')
        plt.title(f'Plot for {tag}')
        plt.legend()
        plt.grid(True)

        # Save the plot
        plot_filename = os.path.join(output_dir, f"{tag.replace('/', '_')}.png")
        plt.savefig(plot_filename)
        plt.close()
        print(f"Plot saved as {plot_filename}")

def process_folder(folder_path):
    """
    Load TensorBoard event data and create plots for each tag.
    
    Parameters:
        event_dir (str): Directory containing the TensorBoard event files.
    """
    event_files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.startswith('events.out.tfevents')]
    
    if event_files:  # Only process if there are event files in the folder
        all_scalar_data = pd.DataFrame()

        for event_file in event_files:
            scalar_data = extract_scalar_data(event_file)
            all_scalar_data = pd.concat([all_scalar_data, scalar_data], ignore_index=True)

        # Plot the data
        plot_data(all_scalar_data, folder_path)

    
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
    root_directory = "src/output/runs"  # replace with the path to your event files
    
    walk_through_folders(root_directory)
