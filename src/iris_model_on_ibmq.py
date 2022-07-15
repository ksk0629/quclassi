import argparse
import time
from typing import Dict, Tuple

import mlflow
import numpy as np
import qiskit
from sklearn import datasets
import yaml

import iris_model
import run_model
from quclassi import QuClassi


LATEST_SETOSA_PATH = "latest_setosa.json"
LATEST_VERSICOLOR_PATH = "latest_versicolor.json"
LATEST_VIRGINICA_PATH = "latest_virginica.json"


def load_latest_circuits(quclassi_object: QuClassi) -> QuClassi:
    """最直近のirisデータ用の量子回路情報を読込む

    :param QuClassi quclassi_object: irisデータ用のQuClassi
    :return QuClassi quclassi_object: 最直近の量子回路情報を読込んだQuClassi
    """
    try:
        quclassi_object.quantum_circuits["setosa"].load_parameters_from_json(LATEST_SETOSA_PATH)
    except:
        msg = f"There is no {LATEST_SETOSA_PATH}"
        print(msg)

    try:
        quclassi_object.quantum_circuits["versicolor"].load_parameters_from_json(LATEST_VERSICOLOR_PATH)
    except:
        msg = f"There is no {LATEST_VERSICOLOR_PATH}"
        print(msg)

    try:
        quclassi_object.quantum_circuits["virginica"].load_parameters_from_json(LATEST_VIRGINICA_PATH)
    except:
        msg = f"There is no {LATEST_VIRGINICA_PATH}"
        print(msg)

    return quclassi_object


def set_epochs(quclassi_object: QuClassi, objective_epochs: int) -> Dict[str, int]:
    """目標学習回数に向けて、現在の学習回数を参考に細かく分けた学習回数を各量子回路毎に設定する

    :param QuClassi quclassi_object: irisデータ用のQuClassi
    :param int objective_epochs: 目標学習回数
    :return Dict[str, int] epochs_dict: 目標学習回数に応じて細かく分けた学習回数
    """
    small_epoch = 5 if objective_epochs > 5 else objective_epochs

    if quclassi_object.quantum_circuits["setosa"].epochs >= objective_epochs:
        setosa_epochs = 0
    elif quclassi_object.quantum_circuits["setosa"].epochs >= objective_epochs - small_epoch:
        setosa_epochs = objective_epochs - quclassi_object.quantum_circuits["setosa"].epochs
    else:
        setosa_epochs = small_epoch

    if quclassi_object.quantum_circuits["versicolor"].epochs >= objective_epochs:
        versicolor_epochs = 0
    elif quclassi_object.quantum_circuits["versicolor"].epochs >= objective_epochs - small_epoch:
        versicolor_epochs = objective_epochs - quclassi_object.quantum_circuits["versicolor"].epochs
    else:
        versicolor_epochs = small_epoch

    if quclassi_object.quantum_circuits["virginica"].epochs >= objective_epochs:
        virginica_epochs = 0
    elif quclassi_object.quantum_circuits["virginica"].epochs >= objective_epochs - small_epoch:
        virginica_epochs = objective_epochs - quclassi_object.quantum_circuits["virginica"].epochs
    else:
        virginica_epochs = small_epoch

    epochs_dict = {
        "setosa": setosa_epochs,
        "versicolor": versicolor_epochs,
        "virginica": virginica_epochs
    }

    return epochs_dict


def train_and_evaluate_iris_on_ibmq(random_state: int, shuffle: bool, train_size: float, should_scale: bool,
                                    structure: str, epochs: int, learning_rate: float, back_end: str, shots: int,
                                    should_normalize: bool, should_show: bool, should_save_each_epoch: bool) -> Tuple[QuClassi, float, Tuple[time.time, time.time]]:
    """irisデータを使い実機上でQuClassiの学習と評価を行う

    :param int random_state: 乱数シード
    :param bool shuffle: 分割時にデータをシャッフルするかどうか
    :param float train_size: 分割時の学習データの比率
    :param bool should_scale: スケーリングするかどうか
    :param str structure: QuClassiの構造を定める文字列
    :param int epochs: 学習回数
    :param float learning_rate: 学習率
    :param str back_end: バックエンド名
    :param int shots: 量子回路の実行回数(期待値を取る回数)
    :param bool should_normalize: 各データを正規化するかどうか
    :param bool should_show: 学習過程を標準出力に出すかどうか
    :param bool should_save_each_epoch: 1エポック毎に量子回路の情報を出力するかどうか
    :return QuClassi quclassi: 学習済みのQuClassiオブジェクト
    :return float accuracy: 評価結果(正確度)
    :return Tuple[time.time, time.time] (train_time, eval_time): それぞれ学習と評価にかかった時間
    """
    # 前処理を行い、学習用と評価用に分割したirisデータを取得する
    iris = datasets.load_iris()
    data = iris.data
    labels = list(iris.target)
    for index in range(len(labels)):
        labels[index] = iris.target_names[labels[index]]
    x_train, x_test, y_train, y_test = run_model.preprocess_dataset(data=data, labels=labels, random_state=random_state, shuffle=shuffle, train_size=train_size, should_scale=should_scale)
    print("Dataset preparation is done.")

    # irisデータ用のQuClassiを作成する
    input_size = len(data[0])
    unique_labels = np.unique(labels)
    quclassi = QuClassi(input_size, unique_labels)
    quclassi.build_quantum_circuits(structure)
    print("Quclassi was built.")

    # 学習回数が目標学習回数に到達するまで分割して細かく学習を進める
    start_time = time.time()
    count = 0
    while True:
        print(f"====================================== {count} ================================================")
        quclassi = load_latest_circuits(quclassi)

        epochs_dict = set_epochs(quclassi, epochs)

        total_epochs = sum([e for e in epochs_dict.values()])
        if total_epochs == 0:
            break

        quclassi.train(x_train, y_train,
                       epochs=epochs_dict, learning_rate=learning_rate,
                       back_end=back_end, shots=shots, should_normalize=should_normalize,
                       should_show=should_show, should_save_each_epoch=should_save_each_epoch, on_ibmq=True)
    train_time = time.time() - start_time
    print("Training is done.")

    # 学習済みのQuClassiを保存する
    quclassi.save_model_as_json(iris_model.PREFIX)

    # 学習済みのQuClassiを評価する
    start_time = time.time()
    accuracy = quclassi.evaluate(x_test, y_test, back_end=back_end, shots=shots, should_normalize=should_normalize, on_ibmq=True)
    eval_time = time.time() - start_time
    print("Evaluating is done.")

    return quclassi, accuracy, (train_time, eval_time)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and Evaluate QuClassi with Iris dataset on IBMQ.")
    parser.add_argument("-c", "--config_yaml_path", required=False, type=str, default="./config_iris.yaml")
    args = parser.parse_args()

    # 設定ファイルを読み込む
    with open(args.config_yaml_path, "r") as yaml_f:
        config = yaml.safe_load(yaml_f)
    config_mlflow = config["mlflow"]
    config_dataset = config["dataset"]
    config_train = config["train"]

    # on_ibmqがFalseである場合にはエラーを発生させる
    if not config_train["on_ibmq"]:
        raise ValueError(f"on_ibmq in {args.config_yaml_path} must be True.")

    del config_train["on_ibmq"]

    # 標準入力からIBMQに接続するためのapiトークンを取得する
    print("Input api_token: ", end="")
    api_token = input()

    # IBMQに関するアカウント情報を取得する
    qiskit.IBMQ.save_account(api_token, overwrite=True)

    # 使用可能な実機を標準出力に出力する
    provider = qiskit.IBMQ.load_account()
    print()
    qiskit.tools.monitor.backend_overview()
    for backend in provider.backends():
        print(backend)

    # どの実機を使うかを標準入力から取得する
    print("Which backend should be used? Tell me the name as a position (integer).: ", end="")
    back_end_position = int(input())
    config_train["back_end"] = provider.backends()[back_end_position]

    # mlflowの設定を行い、学習と評価を実行する
    mlflow.set_experiment(config_mlflow["experiment_name"])
    with mlflow.start_run(run_name=config_mlflow["run_name"]):
        # 現在アクティブなrunのidを取得する
        run_id = mlflow.active_run().info.run_id

        try:
            # データセットと学習に関連するパラメータをmlflowに登録する
            mlflow.log_params(config_dataset)
            mlflow.log_params(config_train)

            # 学習と訓練を実行する
            quclassi, accuracy, (train_time, eval_time) = train_and_evaluate_iris_on_ibmq(**config_dataset, **config_train)

            # 学習時の損失値を量子回路毎にmlflowに登録する
            mlflow.log_metric("train_time", train_time)
            for label in quclassi.unique_labels:
                loss_history = quclassi.quantum_circuits[label].loss_history
                for epoch, loss in enumerate(loss_history):
                    mlflow.log_metric(f"{label}_loss", loss, step=epoch+1)

            # 評価結果をmlflowに登録する
            mlflow.log_metric("eval_time", eval_time)
            mlflow.log_metric("accuracy", accuracy)

            # 使用した設定ファイルそのものとモデルファイルをmlflowに登録する
            mlflow.log_artifact(args.config_yaml_path)
            paths = [f"{iris_model.PREFIX}_{label}.json" for label in quclassi.unique_labels]
            paths.append(f"{iris_model.PREFIX}_config.json")
            for path in paths:
                mlflow.log_artifact(path)

        # 中断されたrunのidを標準出力へ出力する
        except:
            print(f"Interrupted run id is {run_id}")
