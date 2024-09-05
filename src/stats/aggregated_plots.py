import itertools
from pathlib import Path
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
    tags = ea.Tags()["scalars"]

    # Prepare a list to store all the scalar data
    scalar_data = []

    # Iterate over all tags (e.g., loss, accuracy)
    for tag in tags:
        events = ea.Scalars(tag)
        for event in events:
            scalar_data.append(
                {
                    "wall_time": event.wall_time,
                    "step": event.step,
                    "tag": tag,
                    "value": event.value,
                }
            )

    return pd.DataFrame(scalar_data)


def process_folder_for_tags(folder_path, tag_data, root_dir, params):
    """
    Process all TensorBoard event files in a given folder and organize data by tag.

    Parameters:
        folder_path (str): Path to the folder containing TensorBoard event files.
        tag_data (dict): Dictionary to store tag data, organized by tag.
    """
    event_files = [
        os.path.join(folder_path, f)
        for f in os.listdir(folder_path)
        if f.startswith("events.out.tfevents")
    ]

    relative_path = os.path.relpath(folder_path, root_dir)
    column_name = relative_path.replace("/", " ").replace("\\", " ").replace("_", " ")

    parts = column_name.split()

    # Add the extracted parameters to the respective sets
    params["attack"].add(parts[0])
    params["defense"].add(parts[1] if parts[1] != "None" else None)
    num_participants = int(parts[4])
    params["num_participants"].add(num_participants)
    params["assist_mode"].add(parts[5])
    params["poison_percentage"].add(float(parts[-2]))
    num_attacker = int(parts[-1])
    params["num_attacker"].add(num_attacker)

    if event_files:  # Only process if there are event files in the folder
        for event_file in event_files:
            scalar_data = extract_scalar_data(event_file)

            if num_attacker == 1:
                pattern = f"org{num_participants-1}/"
                replacement = "mal_org1"
                scalar_data["tag"] = scalar_data["tag"].str.replace(
                    pattern, replacement, regex=True
                )
            elif num_attacker == 2:
                pattern = f"org{num_participants-1}/"
                replacement = "mal_org1"
                scalar_data["tag"] = scalar_data["tag"].str.replace(
                    pattern, replacement, regex=True
                )
                pattern = f"org{num_participants-2}/"
                replacement = "mal_org2"
                scalar_data["tag"] = scalar_data["tag"].str.replace(
                    pattern, replacement, regex=True
                )
            scalar_data = scalar_data[~scalar_data["tag"].str.startswith("org")]

            # Group data by tag
            for tag in scalar_data["tag"].unique():
                tag_specific_data = scalar_data[scalar_data["tag"] == tag][
                    ["step", "value"]
                ]
                tag_specific_data = tag_specific_data.set_index("step")
                tag_specific_data = tag_specific_data.rename(
                    columns={"value": column_name}
                )

                if tag in tag_data:
                    if column_name in tag_data[tag]:
                        max_step_tag_specific = tag_specific_data.index.max()
                        if max_step_tag_specific > len(tag_data[tag].index):
                            tag_data[tag] = tag_data[tag].reindex(
                                range(1, max_step_tag_specific + 1)
                            )
                            tag_data[tag].update(tag_specific_data)
                        else:
                            tag_data[tag].update(tag_specific_data)
                    else:
                        tag_data[tag] = pd.concat(
                            [tag_data[tag], tag_specific_data], axis=1
                        )
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
            ordered_values = []
            for value in values:
                ordered_values.append(value)
                comb[param] = value
                column_name = "{} {} train 0 {} {} 10 10 search 0 {} {}".format(
                    comb["attack"],
                    comb["defense"],
                    comb["num_participants"],
                    comb["assist_mode"],
                    comb["poison_percentage"],
                    comb["num_attacker"],
                )
                column_names.append(column_name)
            del comb[param]

            for tag, df in tag_data.items():
                plot_columns(
                    tag, df, column_names, output_dir, param, comb, ordered_values
                )


def dict_to_string_no_special_chars(d):
    return " ".join([f"{key} {value}" for key, value in d.items()])


def plot_columns(tag, df, column_names, output_dir, param, comb, values):

    for column in column_names:
        if column not in df.columns:
            print(f"Column '{column}' does not exist in the DataFrame.")
            return

    plt.figure(figsize=(10, 6))
    if tag in ("test/Accuracy", "test/ASR"):
        plt.ylim(0, 100)
    elif tag in ("test/Loss", "train/Loss"):
        pass
    elif "anomaly" in tag:
        plt.ylim(0, 1)
    else:
        print(f"ylim not set, tag: {tag}")

    for idx, column in enumerate(column_names):
        label = f"{param}: {values[idx]}"
        plt.plot(df.index - 1, df[column], marker="o", label=label)

    tag_clean = tag.replace("/", "_")
    plt.ylabel(tag_clean.replace("_", " "))
    plt.xlabel("Assistance round")
    if comb["attack"] == "None":
        comb["poison_percentage"] = 0

    # plt.title(f'{tag_clean} {param}\n{comb} over Steps')
    if tag_clean in ("test_Accuracy", "ASR"):
        plt.legend(loc="lower right")
    else:
        plt.legend()

    comb = dict_to_string_no_special_chars(comb)
    output_path = os.path.join(output_dir, f"{tag_clean}_{param} {comb} plot.png")
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
        "attack": set(),
        "defense": set(),
        "num_participants": set(),
        "assist_mode": set(),
        "poison_percentage": set(),
        "num_attacker": set(),
    }

    for dirpath, _, filenames in os.walk(root_dir):
        # Only process if there are event files in the current directory
        if any(f.startswith("events.out.tfevents") for f in filenames):
            process_folder_for_tags(dirpath, tag_data, root_dir, params)

    output_dir = os.path.join(root_directory, "plots")
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    plot_combinations(params, tag_data, output_dir)
    # plot_data(tag_data, root_dir)


if __name__ == "__main__":
    root_directory = "output/runs"  # replace with the path to your root directory

    walk_through_folders_and_collect_data(root_directory)
