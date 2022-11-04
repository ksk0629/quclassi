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
    """Load the latest quantum circuits for the iris dataset

    :param QuClassi quclassi_object: QuClassi for iris dataset
    :return QuClassi quclassi_object: Loaded the latest QuClassi for iris dataset
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
    """Get epochs that are adjusted with objective epochs for each quantum circuit

    :param QuClassi quclassi_object: QuClassi for iris dataset
    :param int objective_epochs: objective epochs
    :return Dict[str, int] epochs_dict: adjusted epochs
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
                                    structure: str, epochs: int, learning_rate: float, backend: str, shots: int,
                                    should_normalise: bool, should_save_each_epoch: bool) -> Tuple[QuClassi, float, Tuple[time.time, time.time]]:
    """Train and evaluate QuClassi with the iris dataset on IBMQ

    :param int random_state: random seed
    :param bool shuffle: whether or not data is shuffled
    :param float train_size: ratio of training data and evaluating data
    :param bool should_scale: whether or not each data is scaled
    :param List[str] structure: string, which has only s, d and c, to decide the structure
    :param int epochs: number of epochs
    :param float learning_rate: learning rate
    :param str backend: backend
    :param int shots: number of executions
    :param bool should_normalise: whether or not normalise each data
    :param bool should_save_each_epoch: whether or not print the information of the quantum curcuit per one epoch
    :return QuClassi quclassi: Trained QuClassi
    :return float accuracy: accuracy
    :return Tuple[time.time, time.time] (train_time, eval_time): process times
    """
    # Preprocess the iris dataset and separate it into ones for training and evaluating
    iris = datasets.load_iris()
    data = iris.data
    labels = list(iris.target)
    for index in range(len(labels)):
        labels[index] = iris.target_names[labels[index]]
    x_train, x_test, y_train, y_test = run_model.preprocess_dataset(data=data, labels=labels, random_state=random_state, shuffle=shuffle, train_size=train_size, should_scale=should_scale)
    print("Dataset preparation is done.")

    # Generate QuClassi for the iris dataset
    input_size = len(data[0])
    unique_labels = np.unique(labels)
    quclassi = QuClassi(input_size, unique_labels)
    quclassi.build_quantum_circuits(structure)
    print("Quclassi was built.")

    # Train the QuClassi repeatedly with small epochs
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
                       backend=backend, shots=shots, should_normalise=should_normalise,
                       should_save_each_epoch=should_save_each_epoch, on_ibmq=True)
    train_time = time.time() - start_time
    print("Training is done.")

    # Save the QuClassi
    quclassi.save_model_as_json(iris_model.PREFIX)

    # Evaluate the QuClassi
    start_time = time.time()
    accuracy = quclassi.evaluate(x_test, y_test, backend=backend, shots=shots, should_normalise=should_normalise, on_ibmq=True)
    eval_time = time.time() - start_time
    print("Evaluating is done.")

    return quclassi, accuracy, (train_time, eval_time)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and Evaluate QuClassi with Iris dataset on IBMQ.")
    parser.add_argument("-c", "--config_yaml_path", required=False, type=str, default="./config_iris.yaml")
    args = parser.parse_args()

    # Load the setting file
    with open(args.config_yaml_path, "r") as yaml_f:
        config = yaml.safe_load(yaml_f)
    config_mlflow = config["mlflow"]
    config_dataset = config["dataset"]
    config_train = config["train"]

    # Raise the error if the given on_ibmq is False
    if not config_train["on_ibmq"]:
        raise ValueError(f"on_ibmq in {args.config_yaml_path} must be True.")

    del config_train["on_ibmq"]

    # Get an API token of IBMQ from the standard input
    print("Input api_token: ", end="")
    api_token = input()

    # Set the account information
    qiskit.IBMQ.save_account(api_token, overwrite=True)

    # Print all available quantum computers
    provider = qiskit.IBMQ.load_account()
    print()
    qiskit.tools.monitor.backend_overview()
    for backend in provider.backends():
        print(backend)

    # Get which a quantum computer is used from the standard input
    print("Which backend should be used? Tell me the name as a position (integer).: ", end="")
    backend_position = int(input())
    config_train["backend"] = provider.backends()[backend_position]

    # Run mlflow and train and evaluate a QuClassi model
    mlflow.set_experiment(config_mlflow["experiment_name"])
    with mlflow.start_run(run_name=config_mlflow["run_name"]):
        # Get an activate run id
        run_id = mlflow.active_run().info.run_id

        try:
            # Register information of training parameters to mlflow
            mlflow.log_params(config_dataset)
            mlflow.log_params(config_train)

            # Train and evaluate a QuClassi
            quclassi, accuracy, (train_time, eval_time) = train_and_evaluate_iris_on_ibmq(**config_dataset, **config_train)

            # Register loss values to mlflow
            mlflow.log_metric("train_time", train_time)
            for label in quclassi.unique_labels:
                loss_history = quclassi.quantum_circuits[label].loss_history
                for epoch, loss in enumerate(loss_history):
                    mlflow.log_metric(f"{label}_loss", loss, step=epoch+1)

            # Register evaluation result to mlflow
            mlflow.log_metric("eval_time", eval_time)
            mlflow.log_metric("accuracy", accuracy)

            # Register the setting file and the model file to mlflow
            mlflow.log_artifact(args.config_yaml_path)
            paths = [f"{iris_model.PREFIX}_{label}.json" for label in quclassi.unique_labels]
            paths.append(f"{iris_model.PREFIX}_config.json")
            for path in paths:
                mlflow.log_artifact(path)

        except:
            # Print the run id that was interrupted
            print(f"Interrupted run id is {run_id}")
