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
        pd.DataFrame: Reshaped DataFrame with one row per step and one column per tag.
    """
    df = df[df['tag'] != 'train/Loss']
    # Pivot the DataFrame to have tags as columns and steps as rows
    reshaped_df = df.pivot_table(index='step', columns='tag', values='value')

    # Reset the index so that 'step' becomes a column again
    reshaped_df = reshaped_df.reset_index()

    wall_time_series = df.groupby('step')['wall_time'].first()
    reshaped_df['wall_time'] = reshaped_df['step'].map(wall_time_series)

    return reshaped_df

def main(event_dir, output_csv):
    """
    Load TensorBoard event data and write it to a CSV file with one row per step and one column per tag.
    
    Parameters:
        event_dir (str): Directory containing the TensorBoard event files.
        output_csv (str): Path to the output CSV file.
    """
    event_files = [os.path.join(event_dir, f) for f in os.listdir(event_dir) if f.startswith('events.out.tfevents')]
    
    # If there are multiple event files, process them all
    all_scalar_data = pd.DataFrame()

    for event_file in event_files:
        scalar_data = extract_scalar_data(event_file)
        all_scalar_data = pd.concat([all_scalar_data, scalar_data], ignore_index=True)

    # Reshape the data to have one row per step and one column per tag
    reshaped_data = reshape_data(all_scalar_data)

    # Write to CSV
    reshaped_data.to_csv(output_csv, index=False)
    print(f"Data has been written to {output_csv}")

if __name__ == "__main__":
    event_directory = "output/runs/badnet/IF/train_0_2_bag_10_10_search_0_0.2_1"  # replace with the path to your event files
    output_csv_file = "output/output.csv"              # replace with your desired output CSV filename
    
    main(event_directory, output_csv_file)
