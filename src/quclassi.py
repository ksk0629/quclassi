import copy
import json
import os
from typing import Dict, List, Optional, Tuple, Union

import numpy as np

from quclassi_circuit import QuClassiCircuit


class QuClassi():
    """QuClassi"""

    def __init__(self, input_size: int, labels: List[str]) -> None:
        """
        :param int input_size: the input size
        :param List[str] labels: training labels
        """
        self.__input_size = input_size
        self.__unique_labels = np.unique(labels)

        self.__loss_history = []
        self.__accuracy_history = []

        # Generate quantum circuits of each label
        self.__quantum_circuits = dict()
        for label in self.unique_labels:
            self.__quantum_circuits[label] = QuClassiCircuit(self.input_size, label)

    @property
    def input_size(self) -> int:
        """Return the input size.

        :return int: the input size
        """
        return self.__input_size

    @property
    def unique_labels(self) -> np.ndarray:
        """Return the unique labels.

        :return np.ndarray: the unique labels
        """
        return self.__unique_labels

    @property
    def loss_history(self) -> List[float]:
        """Return the history of loss values.

        :return List[float]: the history of loss values
        """
        return self.__loss_history

    @property
    def accuracy_history(self) -> List[float]:
        """Return the hisotry of the accuracy values.

        :return List[float]: the hisotry of the accuracy values
        """
        return self.__accuracy_history

    @property
    def quantum_circuits(self) -> Dict[str, QuClassiCircuit]:
        """Return the quantum circuits in dict.

        :return Dict[str, QuClassiCircuit]: the quantum circuits
        """
        return self.__quantum_circuits

    @property
    def best_loss(self) -> float:
        """Return the best loss value.

        :return float: best loss value
        """
        return np.min(self.loss_history) if len(self.loss_history) != 0 else None

    @property
    def best_accuracy(self) -> float:
        """Return the best accuracy value.

        :return float: the best accuracy value
        """
        return np.max(self.accuracy_history) if len(self.accuracy_history) != 0 else None

    def build_quantum_circuits(self, structure: List[str], thetas_lists: Optional[List[List[List[float]]]] = None, seed: int = 0) -> None:
        """Build quantum circuits for each label

        :param List[str] structure: string, which has only s, d and c, to decide the structure
        :param Optional[List[List[List[float]]]] thetas_lists: initial rotation angles, defaults to None
        :param int seed: random seed, defaults to None
        """
        # Initialise random thetas if seed is not given
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

        # Build quantum circuits for each label
        for index, label in enumerate(self.unique_labels):
            print(f"=== label {label}: ", end="")
            thetas_list = thetas_lists[index]
            self.quantum_circuits[label].build_quantum_circuit(structure=structure, thetas_list=copy.deepcopy(thetas_list), is_in_train=False)

    def train(self, train_data: List[List[float]], train_labels: List[str],
              epochs: Union[Dict[str, int], int], learning_rate: float, backend: str,
              shots: int, should_normalise: bool, should_save_each_epoch: bool, on_ibmq: bool) -> None:
        """Train the quantum circuits

        :param List[List[float]] train_data: learning data
        :param List[str] train_labels: training labels
        :param Union[Dict[str, int], int] epochs: number of epochs
        :param float learning_rate: learning rate
        :param str backend: backend
        :param int shots: number of executions
        :param bool should_normalise: whether or not normalise each data
        :param bool should_save_each_epoch: whether or not print the information of the quantum curcuit per one epoch
        :param bool on_ibmq: whether or not ibmq is used
        :raises ValueError: if the lengths of the given lables the data are not the same
        :raises ValueError: if the type of the given epochs is dict and the key is not the same the label
        """
        # Check whether the given parameters are valid or not
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

        # Train each quantum circuit
        for label in self.unique_labels:
            current_epochs = epochs[label]
            focused_indices = np.where(np.array(train_labels) == label)[0]
            focused_train_data = np.array(train_data)[focused_indices]
            self.quantum_circuits[label].train(focused_train_data, label=label,
                                               epochs=current_epochs, learning_rate=learning_rate,
                                               backend=backend, shots=shots, should_normalise=should_normalise,
                                               should_save_each_epoch=should_save_each_epoch, on_ibmq=on_ibmq)

    def train_and_eval(self, train_data: List[List[float]], train_labels: List[str],
                       dev_data: List[List[float]], dev_label: List[str],
                       epochs: int, learning_rate: float, backend: str,
                       shots: int, should_normalise: bool, should_save_each_epoch: bool, on_ibmq: bool) -> None:
        """Train and evaluate all quantum circuits

        :param List[List[float]] train_data: learning data
        :param List[str] train_labels: training labels
        :param Union[Dict[str, int], int] epochs: number of epochs
        :param float learning_rate: learning rate
        :param str backend: backend
        :param int shots: number of executions
        :param bool should_normalise: whether or not normalise each data
        :param bool should_save_each_epoch: whether or not print the information of the quantum curcuit per one epoch
        :param bool on_ibmq: whether or not ibmq is used
        :raises ValueError: if the lengths of the given lables the data are not the same
        """
        # Check whether the given data is valid or not
        if len(train_data) != len(train_labels):
            msg = f"len(train_data) must be same as len(train_labels), but {len(train_data)} != {len(train_labels)}"
            raise ValueError(msg)

        for epoch in range(1, epochs+1):
            print(f"================== epoch {epoch} ==================")

            # Train each quantum circuits
            for label in self.unique_labels:
                focused_indices = np.where(np.array(train_labels) == label)[0]
                focused_train_data = np.array(train_data)[focused_indices]
                self.quantum_circuits[label].train(focused_train_data, label=label,
                                                   epochs=1, learning_rate=learning_rate,
                                                   backend=backend, shots=shots, should_normalise=should_normalise,
                                                   should_save_each_epoch=should_save_each_epoch, on_ibmq=on_ibmq)

            # Classify data
            _, probabilities_list = self.classify(data=train_data, backend=backend, shots=shots,
                                                  should_normalise=should_normalise, on_ibmq=on_ibmq)

            # Calculate the crossentropy between predicions and truths
            current_loss = self.calculate_cross_entropy_error(probabilities_list=probabilities_list, true_labels=train_labels)

            # Print loss information
            if self.best_loss is None or current_loss < self.best_loss:
                print(f"\tloss = {current_loss} <- the best loss ever")
            else:
                print(f"\tloss = {current_loss}")

            self.loss_history.append(current_loss)

            current_accuracy = self.evaluate(data=dev_data, true_labels=dev_label,
                                             backend=backend, shots=shots,
                                             should_normalise=should_normalise, on_ibmq=on_ibmq)

            if self.best_accuracy is None or current_accuracy < self.best_accuracy:
                print(f"\taccuracy = {current_accuracy} <- the best accuracy ever")
            else:
                print(f"\taccuracy = {current_accuracy}")

            self.accuracy_history.append(current_accuracy)

    def calculate_cross_entropy_error(self, probabilities_list: List[List[float]], true_labels: List[str]) -> float:
        """Calculate the cross entropy from true labels

        :param List[List[float]] probabilities_list: probability distribution
        :param List[str] true_labels: true labels
        :return float loss: cross entropy value
        """
        true_distribution = [np.where(self.unique_labels == true_label, 1, 0).tolist() for true_label in true_labels]
        loss = self.cross_entropy_error(probabilities_list=probabilities_list, true_distribution=true_distribution)

        return loss

    def cross_entropy_error(self, probabilities_list: List[List[float]], true_distribution: List[List[float]]) -> float:
        """Calculate the cross entropy

        :param List[List[float]] probabilities_list: probability distribution
        :param List[List[float]] true_distribution: true probability distribution
        :return float loss: cross entropy value
        """
        loss = - np.sum(true_distribution * np.log(probabilities_list)) / len(probabilities_list)
        return loss

    def classify(self, data: List[List[float]], backend: str, shots: int,
                 should_normalise: bool, on_ibmq: bool) -> Tuple[List[str], List[List[float]]]:
        """Classify data

        :param List[List[float]] data: data
        :param str backend: backend
        :param int shots: number of executions
        :param bool should_normalise: whether or not normalise each data
        :param bool on_ibmq: whether or not ibmq is used
        :return List[str] classified_labels: classified labels
        :return List[List[float]]] probabilities_list: probabilities
        """
        # Calculate likelihoods for each label
        likelihoods_list = []
        for label in self.unique_labels:
            likelihoods_list.append(self.quantum_circuits[label].calculate_likelihood(data,
                                                                                      backend=backend,
                                                                                      shots=shots,
                                                                                      should_normalise=should_normalise,
                                                                                      on_ibmq=on_ibmq))

        # Get probabilities from the likelihoods and classify data
        probabilities_list = []
        classified_labels = []
        for index in range(len(data)):
            likelihoods = [likelihoods[index] for likelihoods in likelihoods_list]
            probabilities = self.softmax(likelihoods)

            probabilities_list.append(probabilities)
            classified_labels.append(self.unique_labels[np.argmax(probabilities)])

        return classified_labels, probabilities_list

    def evaluate(self, data: List[List[float]], true_labels: List[str],
                 backend: str, shots: int, should_normalise: bool, on_ibmq: bool) -> float:
        """Evaluate the quantum circuits

        :param List[List[float]] data: data
        :param List[str] true_labels: true labels
        :param str backend: backend
        :param int shots: number of executions
        :param bool should_normalise: whether or not normalise each data
        :param bool on_ibmq: whether or not ibmq is used
        :raises ValueError: if the lengths of the given lables the data are not the same
        :return float: accuracy
        """
        # Check whether the given data is valid or not
        if len(data) != len(true_labels):
            raise ValueError(f"len(data) must be same as len(true_labels), but {len(data)} != {len(true_labels)}")

        # Evaluate each circuit
        total_correct = 0
        total_wrong = 0
        for label in self.unique_labels:
            print(f"label {label}: Start evaluating")
            focused_indices = np.where(np.array(true_labels) == label)[0]
            focused_data = data[focused_indices]
            focused_true_labels = np.array(true_labels)[focused_indices]

            classified_labels, _ = self.classify(data=focused_data, backend=backend, shots=shots, should_normalise=should_normalise, on_ibmq=on_ibmq)

            correct = 0
            wrong = 0
            for true_label, classified_label in zip(focused_true_labels, classified_labels):
                if true_label == classified_label:
                    correct += 1
                else:
                    wrong += 1
            accuracy = correct / (correct + wrong) * 100

            # Print the resutls
            print(f"the number of correct classified_labels is {correct}")
            print(f"the number of wrong classified_labels is {wrong}")
            print(f"the accuracy is {accuracy} [%]")

            total_correct += correct
            total_wrong += wrong

        # Print the final results
        total_accuracy = total_correct / (total_correct + total_wrong) * 100
        print("=== FINAL RESULT ===")
        print(f"the number of correct classified_labels is {total_correct}")
        print(f"the number of wrong classified_labels is {total_wrong}")
        print(f"the accuracy is {total_accuracy} [%]")

        return total_accuracy

    def softmax(self, x: List[float]) -> np.array:
        """Calculate the sofmax function

        :param List[float] x: data
        :return numpy.ndarray y: value of softmax function
        """
        u = np.sum(np.exp(x))
        y = np.exp(x) / u
        return y

    def save_model_as_json(self, path_prefix: str) -> None:
        """Save the QuClassi information in JSON

        :param str path_prefix: prefix of output path
        """
        config_path = path_prefix + "_config.json"
        config = dict()
        config["input_size"] = self.input_size
        config["labels"] = list(self.unique_labels)
        config["loss"] = self.loss_history
        with open(config_path, 'w') as output:
            json.dump(config, output, indent=4)

        for label in self.unique_labels:
            circuit_path = path_prefix + "_" + label + ".json"
            self.quantum_circuits[label].save_parameters_as_json(circuit_path)

    def load_latest_circuits(self, dir_path: str) -> None:
        """Load the latest quantum circuits

        :param str dir_path: path to the latest quantum circuits
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
        """Generate the QuClassi from JSON

        :param str path_prefix: prefix of output path
        :return QuClassi loaded_quclassi: Loaded QuClassi object
        """
        config_path = path_prefix + "_config.json"
        with open(config_path) as config_file:
            config = json.load(config_file)

        input_size = config["input_size"]
        unique_labels = config["labels"]

        loaded_quclassi = cls(input_size, unique_labels)
        loaded_quclassi.quantum_circuits = dict()
        loaded_quclassi.loss_history = config["loss"]
        for label in loaded_quclassi.unique_labels:
            loaded_quclassi.quantum_circuits[label] = QuClassiCircuit(loaded_quclassi.input_size, label)

            print(f"label {label}: ", end="")
            circuit_path = path_prefix + "_" + label + ".json"
            loaded_quclassi.quantum_circuits[label].load_parameters_from_json(circuit_path)

        return loaded_quclassi
