import time
from typing import List, Tuple

import mlflow
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import yaml

from quclassi import QuClassi


def preprocess_dataset(data: List[List[float]], labels: List[str], random_state: int, shuffle: bool, train_size: float, should_scale: bool) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """データセットに前処理を施した上で、学習用と評価用に分ける

    :param List[List[float]] data: データ
    :param List[str] labels: データのラベル
    :param int random_state: 乱数シード
    :param bool shuffle: 分割時にデータをシャッフルするかどうか
    :param float train_size: 分割時の学習データの比率
    :param bool should_scale: スケーリングするかどうか
    :return np.ndarray x_train: 学習データ
    :return np.ndarray x_test: 評価データ
    :return np.ndarray y_train: 学習データのラベル
    :return np.ndarray y_test: 評価データのラベル
    """
    if should_scale:
        # スケーリングを行う
        scaler = MinMaxScaler(feature_range=(0, 1))
        data = scaler.fit_transform(data)

    # 学習用と評価用にデータを分ける
    x_train, x_test, y_train, y_test = train_test_split(data, labels, train_size=train_size, random_state=random_state, shuffle=shuffle, stratify=labels)

    return x_train, x_test, y_train, y_test


def train_and_evaluate(data: List[List[float]], labels: List[str], prefix: str,
                       random_state: int, shuffle: bool, train_size: float, should_scale: bool,
                       structure: str, epochs: int, learning_rate: float, back_end: str, shots: int,
                       should_normalize: bool, should_show: bool, should_save_each_epoch: bool,
                       on_ibmq: bool) -> Tuple[object, time.time]:
    """QuClassiの学習と評価を行う

    :param List[List[float]] data: データ
    :param List[str] labels: データのラベル
    :param str prefix: 学習済みモデルを保存するときの接頭語
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
    :param bool on_ibmq: 実機を使うかどうか
    :return QuClassi quclassi: 学習済みのQuClassiオブジェクト
    :return time.time process_time: 学習と評価にかかった時間
    """
    # 前処理を行い、学習用と評価用に分割したデータを取得する
    x_train, x_test, y_train, y_test = preprocess_dataset(data=data, labels=labels, random_state=random_state, shuffle=shuffle, train_size=train_size, should_scale=should_scale)
    print("Dataset preparation is done.")

    # QuClassiを作成する
    input_size = len(data[0])
    unique_labels = np.unique(labels).tolist()
    quclassi = QuClassi(input_size, unique_labels)
    quclassi.build_quantum_circuits(structure)
    print("Quclassi was built.")

    # QuClassiを学習させる
    start_time = time.time()
    quclassi.train_and_eval(x_train, y_train, x_test, y_test,
                            epochs=epochs, learning_rate=learning_rate,
                            back_end=back_end, shots=shots, should_normalize=should_normalize,
                            should_show=should_show, should_save_each_epoch=should_save_each_epoch, on_ibmq=on_ibmq)
    process_time = time.time() - start_time
    print("Training and evaluating is done.")

    # 学習済みのQuClassiを保存する
    quclassi.save_model_as_json(prefix)

    return quclassi, process_time


def run_with_config(config_yaml_path: str, data: List[List[float]], labels: List[str], prefix: str):
    # 設定ファイルを読み込む
    with open(config_yaml_path, "r") as yaml_f:
        config = yaml.safe_load(yaml_f)
    config_mlflow = config["mlflow"]
    config_dataset = config["dataset"]
    config_train = config["train"]

    # mlflowの設定を行い、学習と評価を実行する
    mlflow.set_experiment(config_mlflow["experiment_name"])
    with mlflow.start_run(run_name=config_mlflow["run_name"]):
        # データセットと学習に関連するパラメータをmlflowに登録する
        mlflow.log_params(config_dataset)
        mlflow.log_params(config_train)

        # 学習と訓練を実行する
        quclassi, process_time = train_and_evaluate(data=data, labels=labels, prefix=prefix, **config_dataset, **config_train)

        # 学習時の損失値をmlflowに登録する
        mlflow.log_metric("process_time", process_time)
        for label in quclassi.unique_labels:
            loss_history = quclassi.quantum_circuits[label].loss_history
            for epoch, loss in enumerate(loss_history):
                mlflow.log_metric(f"train_{label}_loss", loss, step=epoch+1)
        for epoch, loss in enumerate(quclassi.loss_history):
            mlflow.log_metric(f"train_crros_entropy_loss", loss, step=epoch+1)

        # 評価結果をmlflowに登録する
        for epoch, accuracy in enumerate(quclassi.accuracy_history):
            mlflow.log_metric(f"dev_accuracy", accuracy, step=epoch+1)

        # 使用した設定ファイルそのものとモデルファイルをmlflowに登録する
        mlflow.log_artifact(config_yaml_path)
        paths = [f"{prefix}_{label}.json" for label in quclassi.unique_labels]
        paths.append(f"{prefix}_config.json")
        for path in paths:
            mlflow.log_artifact(path)
