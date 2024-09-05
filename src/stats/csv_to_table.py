import os
import pandas as pd


if __name__ == "__main__":
    root_directory = "output/runs"
    file_name = "test_Accuracy.csv"

    experiment1 = {
        "attack": "None",
        "defense": "None",
        "num_participants": "2",
        "assist_mode": "stack",
        "poison_percentage": "0.02",
        "num_attacker": 0,
    }

    experiment2 = {
        "attack": "None",
        "defense": "None",
        "num_participants": "4",
        "assist_mode": "stack",
        "poison_percentage": "0.02",
        "num_attacker": 0,
    }

    experiment3 = {
        "attack": "None",
        "defense": "None",
        "num_participants": "8",
        "assist_mode": "stack",
        "poison_percentage": "0.02",
        "num_attacker": 0,
    }
    experiments = [experiment1, experiment2, experiment3]

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

    print(
        df[column_names].to_latex(
            index=False,
            float_format="{:.2f}".format,
        )
    )
