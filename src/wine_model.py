import argparse

from sklearn import datasets

import run_model


PREFIX = "wine"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and Evaluate QuClassi with wine dataset.")
    parser.add_argument("-c", "--config_yaml_path", required=False, type=str, default="./config_wine.yaml")
    args = parser.parse_args()

    # Load the wine dataset
    wine_data = datasets.load_wine(return_X_y=True)
    data, labels = wine_data
    labels = [f"{label}" for label in labels]

    run_model.run_with_config(config_yaml_path=args.config_yaml_path, data=data, labels=labels, prefix=PREFIX)
