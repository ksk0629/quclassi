import argparse

from sklearn import datasets

import run_model


PREFIX = "iris"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and Evaluate QuClassi with iris dataset.")
    parser.add_argument("-c", "--config_yaml_path", required=False, type=str, default="./config_iris.yaml")
    args = parser.parse_args()

    # irisデータセットを読み込む
    iris = datasets.load_iris()
    data = iris.data
    labels = list(iris.target)
    for index in range(len(labels)):
        labels[index] = iris.target_names[labels[index]]

    run_model.run_with_config(config_yaml_path=args.config_yaml_path, data=data, labels=labels, prefix=PREFIX)
