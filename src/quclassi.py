import copy
import json
import os
from typing import Dict, List, Optional, Tuple, Union

import numpy as np

from quclassi_circuit import QuClassiCircuit


class QuClassi():
    """QuClassiクラス"""

    def __init__(self, input_size: int, labels: List[str]) -> None:
        """
        :param int input_size: 入力ベクトルの次元
        :param List[str] labels: 分類ラベル名
        """
        self.input_size = input_size
        self.unique_labels = np.unique(labels)

        self.loss_history = []
        self.accuracy_history = []

        # ラベル毎に量子回路を生成する
        self.quantum_circuits = dict()
        for label in self.unique_labels:
            self.quantum_circuits[label] = QuClassiCircuit(self.input_size, label)

    @property
    def best_loss(self) -> float:
        """loss_historyの中で最も良いものを返す

        :return float: 最もよい損失値
        """
        return np.min(self.loss_history) if len(self.loss_history) != 0 else None

    @property
    def best_accuracy(self) -> float:
        """accuracy_historyの中で最も良いものを返す

        :return float: 最もよい正確度
        """
        return np.max(self.accuracy_history) if len(self.accuracy_history) != 0 else None

    def build_quantum_circuits(self, structure: List[str], thetas_lists: Optional[List[List[List[float]]]] = None, seed: int = 0) -> None:
        """ラベル毎に量子回路を構成する

        :param List[str] structure: 構造を定める文字列 (s,d,cのみからなる文字列)
        :param Optional[List[List[List[float]]]] thetas_lists: structureで定めた構造における初期回転角 (Noneの場合は固定したシードでランダム生成), defaults to None
        :param int seed: ランダムに初期値を作成する場合の乱数シード, defaults to None
        """
        # 初期回転角が与えられなかった場合は固定されたシードの下でランダムに作成する
        if thetas_lists is None:
            np.random.seed(seed)
            thetas_list = []
            num_trained_qubits = self.quantum_circuits[self.unique_labels[0]].num_trained_qubits
            for letter in structure:
                if letter == 's' or letter == 'c':
                    thetas = np.random.rand(num_trained_qubits, 2)
                elif letter == 'd':
                    thetas = np.random.rand(np.arange(1, num_trained_qubits).sum() * 2).reshape(-1, 2)
                thetas_list.append(thetas)

            thetas_lists = [thetas_list] * len(self.unique_labels)

        # ラベル毎に量子回路を構成する
        for index, label in enumerate(self.unique_labels):
            print(f"=== label {label}: ", end="")
            thetas_list = thetas_lists[index]
            self.quantum_circuits[label].build_quantum_circuit(structure=structure, thetas_list=copy.deepcopy(thetas_list), is_in_train=False)

    def train(self, train_data: List[List[float]], train_labels: List[str],
              epochs: Union[Dict[str, int], int], learning_rate: float, back_end: str,
              shots: int, should_normalize: bool, should_show: bool, should_save_each_epoch: bool, on_ibmq: bool) -> None:
        """ラベル毎に量子回路の学習を行う

        :param List[List[float]] train_data: 学習データ
        :param List[str] train_labels: 学習データのラベル
        :param Union[Dict[str, int], int] epochs: 学習回数
        :param float learning_rate: 学習率
        :param str back_end: バックエンド名
        :param int shots: 量子回路の実行回数(期待値を取る回数)
        :param bool should_normalize: 各データを正規化するかどうか
        :param bool should_show: 学習過程を標準出力に出すかどうか
        :param bool should_save_each_epoch: 1エポック毎に量子回路の情報を出力するかどうか
        :param bool on_ibmq: 実機を使うかどうか
        :raises ValueError: train_dataとtrain_labelsの長さが異なる場合
        :raises ValueError: epochsが辞書型でありながら、その鍵が量子回路のラベルと一致しない場合
        """
        # パラメータの妥当性を確認し、問題があれば、エラーを発生させる
        if len(train_data) != len(train_labels):
            msg = f"len(train_data) must be same as len(train_labels), but {len(train_data)} != {len(train_labels)}"
            raise ValueError(msg)
        if type(epochs) is dict:
            msg = "epochs must be int or dict whose the keys are same as labels of quantum circuits."
            if set(epochs.keys()) != set(self.unique_labels):
                raise ValueError(msg)
        elif type(epochs) is int:
            epochs_tmp = epochs
            epochs = {label: epochs_tmp for label in self.unique_labels}

        # ラベル毎に学習を実行する
        for label in self.unique_labels:
            current_epochs = epochs[label]
            focused_indices = np.where(np.array(train_labels) == label)[0]
            focused_train_data = np.array(train_data)[focused_indices]
            self.quantum_circuits[label].train(focused_train_data, label=label,
                                               epochs=current_epochs, learning_rate=learning_rate,
                                               back_end=back_end, shots=shots, should_normalize=should_normalize, should_show=should_show,
                                               should_save_each_epoch=should_save_each_epoch, on_ibmq=on_ibmq)

    def train_and_eval(self, train_data: List[List[float]], train_labels: List[str],
                       dev_data: List[List[float]], dev_label: List[str],
                       epochs: int, learning_rate: float, back_end: str,
                       shots: int, should_normalize: bool, should_show: bool, should_save_each_epoch: bool, on_ibmq: bool) -> None:
        """全ての量子回路を1エポックずつ学習・評価する

        :param List[List[float]] train_data: 学習データ
        :param List[str] train_labels: 学習データのラベル
        :param int epochs: 学習回数
        :param float learning_rate: 学習率
        :param str back_end: バックエンド名
        :param int shots: 量子回路の実行回数(期待値を取る回数)
        :param bool should_normalize: 各データを正規化するかどうか
        :param bool should_show: 学習過程を標準出力に出すかどうか
        :param bool should_save_each_epoch: 1エポック毎に量子回路の情報を出力するかどうか
        :param bool on_ibmq: 実機を使うかどうか
        :raises ValueError: train_dataとtrain_labelsの長さが異なる場合
        """
        # パラメータの妥当性を確認し、問題があれば、エラーを発生させる
        if len(train_data) != len(train_labels):
            msg = f"len(train_data) must be same as len(train_labels), but {len(train_data)} != {len(train_labels)}"
            raise ValueError(msg)

        for epoch in range(1, epochs+1):

            # ラベル毎に学習を実行する
            for label in self.unique_labels:
                focused_indices = np.where(np.array(train_labels) == label)[0]
                focused_train_data = np.array(train_data)[focused_indices]
                self.quantum_circuits[label].train(focused_train_data, label=label,
                                                   epochs=1, learning_rate=learning_rate,
                                                   back_end=back_end, shots=shots, should_normalize=should_normalize, should_show=should_show,
                                                   should_save_each_epoch=should_save_each_epoch, on_ibmq=on_ibmq)

            print(f"============ epoch {epoch} evaluation ============")
            # 学習結果で分類を行い、予想分布を取得する
            _, probabilities_list = self.classify(data=train_data, back_end=back_end, shots=shots,
                                                  should_normalize=should_normalize, on_ibmq=on_ibmq)

            # 予測結果と真の正解とのクロスエントロピー誤差を計算する
            current_loss = self.calculate_cross_entropy_error(probabilities_list=probabilities_list, true_labels=train_labels)

            # 標準出力に全体のロスに関する情報を出力する
            if self.best_loss is None or current_loss < self.best_loss:
                print(f"\tloss = {current_loss} <- the best loss ever")
            else:
                print(f"\tloss = {current_loss}")

            self.loss_history.append(current_loss)

            current_accuracy = self.evaluate(data=dev_data, true_labels=dev_label,
                                             back_end=back_end, shots=shots,
                                             should_normalize=should_normalize, on_ibmq=on_ibmq)

            if self.best_accuracy is None or current_accuracy < self.best_accuracy:
                print(f"\taccuracy = {current_accuracy} <- the best accuracy ever")
            else:
                print(f"\taccuracy = {current_accuracy}")

            self.accuracy_history.append(current_accuracy)

    def calculate_cross_entropy_error(self, probabilities_list: List[List[float]], true_labels: List[str]) -> float:
        """正解ラベルから正解分布を生成し、クロスエントロピーを計算する

        :param List[List[float]] probabilities_list: QuClassiの出力確率分布
        :param List[str] true_labels: 正解ラベル
        :return float loss: クロスエントロピーの値
        """
        true_distribution = [np.where(self.unique_labels == true_label, 1, 0).tolist() for true_label in true_labels]
        loss = self.cross_entropy_error(probabilities_list=probabilities_list, true_distribution=true_distribution)

        return loss

    def cross_entropy_error(self, probabilities_list: List[List[float]], true_distribution: List[List[float]]) -> float:
        """クロスエントロピーを計算する

        :param List[List[float]] probabilities_list: QuClassiの出力確率分布
        :param List[List[float]] true_distribution: 正解分布
        :return float loss: クロスエントロピーの値
        """
        loss = - np.sum(true_distribution * np.log(probabilities_list)) / len(probabilities_list)
        return loss

    def classify(self, data: List[List[float]], back_end: str, shots: int,
                 should_normalize: bool, on_ibmq: bool) -> Tuple[List[str], List[List[float]]]:
        """分類する

        :param List[List[float]] data: 分類したいデータ
        :param str back_end: バックエンド名
        :param int shots: 量子回路の実行回数(期待値を取る回数)
        :param bool should_normalize: 各データを正規化するかどうか
        :param bool on_ibmq: 実機を使うかどうか
        :return List[str] classified_labels: 分類結果
        :return List[List[float]]] probabilities_list: 各ラベルの確率
        """
        # 各ラベルの尤度を求める
        likelihoods_list = []
        for label in self.unique_labels:
            likelihoods_list.append(self.quantum_circuits[label].calculate_likelihood(data,
                                                                                      back_end=back_end,
                                                                                      shots=shots,
                                                                                      should_normalize=should_normalize,
                                                                                      on_ibmq=on_ibmq))

        # 尤度から確率を求め、最も確率が高いものに割り振る
        probabilities_list = []
        classified_labels = []
        for index in range(len(data)):
            likelihoods = [likelihoods[index] for likelihoods in likelihoods_list]
            probabilities = self.softmax(likelihoods)

            probabilities_list.append(probabilities)
            classified_labels.append(self.unique_labels[np.argmax(probabilities)])

        return classified_labels, probabilities_list

    def evaluate(self, data: List[List[float]], true_labels: List[str],
                 back_end: str, shots: int, should_normalize: bool, on_ibmq: bool) -> float:
        """評価する

        :param List[List[float]] data: 評価データ
        :param List[str] true_labels: 評価データのラベル
        :param str back_end: バックエンド名
        :param int shots: 量子回路の実行回数(期待値を取る回数)
        :param bool should_normalize: 各データを正規化するかどうか
        :param bool on_ibmq: 実機を使うかどうか
        :raises ValueError: 評価データと評価データのラベルの長さが異なる場合
        :return float: 正解率
        """
        # 評価データとそのラベルの長さが一致していなければ、エラーを発生させる
        if len(data) != len(true_labels):
            raise ValueError(f"len(data) must be same as len(true_labels), but {len(data)} != {len(true_labels)}")

        # ラベル毎に評価する
        total_correct = 0
        total_wrong = 0
        for label in self.unique_labels:
            print(f"label {label}: Start evaluating")
            focused_indices = np.where(np.array(true_labels) == label)[0]
            focused_data = data[focused_indices]
            focused_true_labels = np.array(true_labels)[focused_indices]

            classified_labels, _ = self.classify(data=focused_data, back_end=back_end, shots=shots, should_normalize=should_normalize, on_ibmq=on_ibmq)

            correct = 0
            wrong = 0
            for true_label, classified_label in zip(focused_true_labels, classified_labels):
                if true_label == classified_label:
                    correct += 1
                else:
                    wrong += 1
            accuracy = correct / (correct + wrong) * 100

            # ラベル毎の結果を標準出力に出力する
            print(f"the number of correct classified_labels is {correct}")
            print(f"the number of wrong classified_labels is {wrong}")
            print(f"the accuracy is {accuracy} [%]")

            total_correct += correct
            total_wrong += wrong

        # 全体での結果を標準出力に出力する
        total_accuracy = total_correct / (total_correct + total_wrong) * 100
        print("=== FINAL RESULT ===")
        print(f"the number of correct classified_labels is {total_correct}")
        print(f"the number of wrong classified_labels is {total_wrong}")
        print(f"the accuracy is {total_accuracy} [%]")

        return total_accuracy

    def softmax(self, x: List[float]) -> np.array:
        """ソフトマックス関数

        :param List[float] x: データ
        :return numpy.ndarray y: ソフトマックス関数作用後のデータ
        """
        u = np.sum(np.exp(x))
        y = np.exp(x) / u
        return y

    def save_model_as_json(self, path_prefix: str) -> None:
        """QuClassiの情報をjsonで保存する

        :param str path_prefix: 出力パスの接頭語
        """
        # QuClassi全体としての情報を取得して保存する
        config_path = path_prefix + "_config.json"
        config = dict()
        config["input_size"] = self.input_size
        config["labels"] = list(self.unique_labels)
        config["loss"] = self.loss_history
        with open(config_path, 'w') as output:
            json.dump(config, output, indent=4)

        # ラベル毎に量子回路の情報をjsonで保存する
        for label in self.unique_labels:
            circuit_path = path_prefix + "_" + label + ".json"
            self.quantum_circuits[label].save_parameters_as_json(circuit_path)

    def load_latest_circuits(self, dir_path: str) -> None:
        """最直近の量子回路情報を読込む

        :param str dir_path: 量子回路情報ファイルが配置されているフォルダパス
        """
        for label in self.unique_labels:
            latest_label = os.path.join(dir_path, f"latest_{label}.json")
            try:
                self.quantum_circuits[label] = QuClassiCircuit.load_parameters_from_json(latest_label)
            except:
                msg = f"There is no {latest_label}."
                print(msg)

    @classmethod
    def load_model_from_json(cls, path_prefix: str) -> object:
        """QuClassiをjsonファイルから生成する

        :param str path_prefix: 出力パスの接頭語
        :return QuClassi loaded_quclassi: 読込んだQuClassiオブジェクト
        """
        # QuClassi全体のの情報を読込む
        config_path = path_prefix + "_config.json"
        with open(config_path) as config_file:
            config = json.load(config_file)

        input_size = config["input_size"]
        unique_labels = config["labels"]

        # QuClassiのラベル毎の量子回路の情報を読込む
        loaded_quclassi = cls(input_size, unique_labels)
        loaded_quclassi.quantum_circuits = dict()
        loaded_quclassi.loss_history = config["loss"]
        for label in loaded_quclassi.unique_labels:
            loaded_quclassi.quantum_circuits[label] = QuClassiCircuit(loaded_quclassi.input_size, label)

            print(f"label {label}: ", end="")
            circuit_path = path_prefix + "_" + label + ".json"
            loaded_quclassi.quantum_circuits[label].load_parameters_from_json(circuit_path)

        return loaded_quclassi
