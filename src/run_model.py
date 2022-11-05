from typing import List, Optional, Tuple

import mlflow
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import yaml

from quclassi import QuClassi


def preprocess_dataset(data: List[List[float]], labels: List[str], random_state: int, shuffle: bool, train_size: float, should_scale: bool) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Preprocess dataset and separate it into datasets for training and evaluating

    :param List[List[float]] data: data
    :param List[str] labels: data labels
    :param int random_state: random seed
    :param bool shuffle: whether or not data is shuffled
    :param float train_size: ratio of training data and evaluating data
    :param bool should_scale: whether or not each data is scaled
    :return np.ndarray x_train: training data
    :return np.ndarray x_test: evaluating data
    :return np.ndarray y_train: training label
    :return np.ndarray y_test: evaluating label
    """
    if should_scale:
        # Scale data
        scaler = MinMaxScaler(feature_range=(0, 1))
        data = scaler.fit_transform(data)

    # Separate data into data for training and evaluating
    x_train, x_test, y_train, y_test = train_test_split(data, labels, train_size=train_size, random_state=random_state, shuffle=shuffle, stratify=labels)

    return x_train, x_test, y_train, y_test


def train_and_evaluate(data: List[List[float]], labels: List[str], prefix: str,
                       random_state: int, shuffle: bool, train_size: float, should_scale: bool,
                       structure: str, epochs: int, learning_rate: float, backend: str, shots: int,
                       patience: Optional[int], objective_value: Optional[float],
                       should_normalise: bool, should_save_each_epoch: bool,
                       on_ibmq: bool) -> object:
    """Train and evaluate QuClassi.

    :param List[List[float]] data: data
    :param List[str] labels: data labels
    :param str prefix: prefix of output path of trained model
    :param int random_state: random seed
    :param bool shuffle: whether or not data is shuffled
    :param float train_size: ratio of training data and evaluating data
    :param bool should_scale: whether or not each data is scaled
    :param List[str] structure: string, which has only s, d and c, to decide the structure
    :param int epochs: number of epochs
    :param float learning_rate: learning rate
    :param str backend: backend
    :param int shots: number of executions
    :param Optional[int] patience: patience for early stopping
    :param Optional[float] objective_value: objective validation value
    :param bool should_normalise: whether or not normalise each data
    :param bool should_save_each_epoch: whether or not print the information of the quantum curcuit per one epoch
    :param bool on_ibmq: whether or not ibmq is used
    :return QuClassi quclassi: trained quclassi object
    """
    # Preprocess dataset and separate it into datasets for training and evaluating
    x_train, x_test, y_train, y_test = preprocess_dataset(data=data, labels=labels, random_state=random_state, shuffle=shuffle, train_size=train_size, should_scale=should_scale)
    print("Dataset preparation is done.")

    # Generate QuClassi
    input_size = len(data[0])
    unique_labels = np.unique(labels).tolist()
    quclassi = QuClassi(input_size, unique_labels)
    quclassi.build_quantum_circuits(structure)
    print("Quclassi was built.")

    # Train and evaluate QuClassi
    quclassi.train_and_eval(x_train, y_train, x_test, y_test,
                            epochs=epochs, learning_rate=learning_rate,
                            backend=backend, shots=shots, patience=patience,
                            objective_value=objective_value, should_normalise=should_normalise,
                            should_save_each_epoch=should_save_each_epoch, on_ibmq=on_ibmq)

    # Save the QuClassi model
    quclassi.save_model_as_json(prefix)

    return quclassi


def run_with_config(config_yaml_path: str, data: List[List[float]], labels: List[str], prefix: str):
    # Load the setting file
    with open(config_yaml_path, "r") as yaml_f:
        config = yaml.safe_load(yaml_f)
    config_mlflow = config["mlflow"]
    config_dataset = config["dataset"]
    config_train = config["train"]

    # Run mlflow and train and evaluate a QuClassi model
    mlflow.set_experiment(config_mlflow["experiment_name"])
    with mlflow.start_run(run_name=config_mlflow["run_name"]):
        # Register information of training parameters to mlflow
        mlflow.log_params(config_dataset)
        mlflow.log_params(config_train)

        # Train and Evaluate a QuClassi model
        quclassi = train_and_evaluate(data=data, labels=labels, prefix=prefix, **config_dataset, **config_train)

        # Register the setting file and the model file to mlflow
        mlflow.log_artifact(config_yaml_path)
        paths = [f"{prefix}_{label}.json" for label in quclassi.unique_labels]
        paths.append(f"{prefix}_config.json")
        for path in paths:
            mlflow.log_artifact(path)
