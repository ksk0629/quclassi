import argparse

from sklearn import datasets

import run_model


PREFIX = "breast_cancer"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and Evaluate QuClassi breast_cancer dataset.")
    parser.add_argument("-c", "--config_yaml_path", required=False, type=str, default="./config_breast_cancer.yaml")
    args = parser.parse_args()

    # breast_cancerデータセットを読み込む
    breast_cancer_data = datasets.load_breast_cancer(return_X_y=True)
    data, labels = breast_cancer_data
    labels = [f"{label}" for label in labels]

    run_model.run_with_config(config_yaml_path=args.config_yaml_path, data=data, labels=labels, prefix=PREFIX)
