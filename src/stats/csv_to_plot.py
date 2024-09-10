import os
from matplotlib import pyplot as plt
import pandas as pd


def dict_to_string_no_special_chars(d):
    return " ".join([f"{key} {value}" for key, value in d.items()])


def plot_columns(tag, df, column_names, output_dir, param, comb, experiments):

    for column in column_names:
        if column not in df.columns:
            print(f"Column '{column}' does not exist in the DataFrame.")
            return

    plt.figure(figsize=(10, 6))
    if tag in ("test/Accuracy", "test/ASR"):
        y_min = 0
        y_max = 100
        margin = (y_max - y_min) * 0.05
        plt.ylim(y_min - margin, y_max + margin)
    elif tag in ("test/Loss", "train/Loss"):
        pass
    elif "anomaly" in tag:
        y_min = 0
        y_max = 1
        margin = (y_max - y_min) * 0.05
    else:
        print(f"ylim not set, tag: {tag}")

    for idx, column in enumerate(column_names):
        label = f"{param}: {experiments[idx]['label']}"
        plt.plot(df.index, df[column], marker="o", label=label)

    tag_clean = tag.replace("/", "_")
    plt.ylabel(tag_clean.replace("_", " "))
    plt.xlabel("Assistance round")

    # plt.title(f'{tag_clean} {param}\n{comb} over Steps')
    if tag_clean in ("test_Accuracy", "test_ASR"):
        plt.legend(loc="lower right")
    else:
        plt.legend()

    comb = dict_to_string_no_special_chars(comb)
    output_path = os.path.join(output_dir, f"{tag_clean}_{param} {comb} plot.png")
    plt.savefig(output_path)
    plt.close()
    print(f"Plot saved to {output_path}")


if __name__ == "__main__":
    root_directory = "output/runs"
    output_dir = os.path.join(root_directory, "specific_plots")
    file_name = "test_Accuracy.csv"
    # file_name = "test_ASR.csv"
    # file_name = "mal_org1anomaly recall.csv"

    param = "attack"
    tag = "test/Accuracy"
    # tag = "test/ASR"

    comb = {
        # "attack": "badnet",
        "defense": "None",
        "num_participants": "4",
        "assist_mode": "stack",
        "poison_percentage": "0.2",
        "num_attacker": 1,
    }

    # experiment1 = {
    #     "attack": "badnet",
    #     "defense": "None",
    #     "num_participants": "2",
    #     "assist_mode": "bag",
    #     "poison_percentage": "0.2",
    #     "num_attacker": 1,
    # }

    # experiment2 = {
    #     "attack": "badnet",
    #     "defense": "None",
    #     "num_participants": "4",
    #     "assist_mode": "bag",
    #     "poison_percentage": "0.2",
    #     "num_attacker": 1,
    # }

    # experiment3 = {
    #     "attack": "badnet",
    #     "defense": "None",
    #     "num_participants": "8",
    #     "assist_mode": "bag",
    #     "poison_percentage": "0.2",
    #     "num_attacker": 1,
    # }

    experiment4 = {
        "attack": "None",
        "defense": "None",
        "num_participants": "4",
        "assist_mode": "stack",
        "poison_percentage": "0.02",
        "num_attacker": 0,
        "label": "baseline",
    }

    experiment5 = {
        "attack": "badnet",
        "defense": "None",
        "num_participants": "4",
        "assist_mode": "stack",
        "poison_percentage": "0.2",
        "num_attacker": 1,
        "label": "badnet",
    }

    experiment6 = {
        "attack": "ftrojan",
        "defense": "None",
        "num_participants": "4",
        "assist_mode": "stack",
        "poison_percentage": "0.2",
        "num_attacker": 1,
        "label": "ftrojan",
    }
    experiments = [
        # experiment1,
        # experiment2,
        # experiment3,
        experiment4,
        experiment5,
        experiment6,
    ]

    df = pd.read_csv(os.path.join(root_directory, file_name))

    column_names = []
    for exp in experiments:
        column_name = column_name = "{} {} train 0 {} {} 10 10 search 0 {} {}".format(
            exp["attack"],
            exp["defense"],
            exp["num_participants"],
            exp["assist_mode"],
            exp["poison_percentage"],
            exp["num_attacker"],
        )
        column_names.append(column_name)

    plot_columns(
        tag=tag,
        df=df,
        column_names=column_names,
        output_dir=output_dir,
        param=param,
        comb=comb,
        experiments=experiments,
    )
