import copy
import json
from pathlib import Path
from typing import List, Optional, Union

import numpy as np
from tqdm import tqdm
import qiskit


class QuClassiCircuit():
    """One quantum circuit belonging to a QuClassi, which corresponds to one class"""

    def __init__(self, input_size: int, name: Optional[str] = None) -> None:
        """Initialise a quantum circuit

        :param int input_size: input size
        :param Optional[str] name: name of quantum circuit, defaults to None
        """
        # Let the given input_size be an even number
        if input_size % 2 != 0:
            modified_input_size = input_size + 1
        else:
            modified_input_size = input_size

        # Initialise class variables
        self.__modified_input_size = modified_input_size  # an input size
        self.__num_trained_qubits = modified_input_size // 2  # the number of qubits for generating a representative quantum state
        self.__num_cbits = 1  # the number of classical bits
        self.__loss_history = []
        self.__epochs = 0
        self.__label = None
        self.__structure = None
        self.__thetas_list = []

        # Generate a quantum circuit
        self.__control_quantum_register = qiskit.QuantumRegister(1, name="control_qubit")
        self.__trained_quantum_register = qiskit.QuantumRegister(self.num_trained_qubits, name="trained_qubit")
        self.__loaded_quantum_register = qiskit.QuantumRegister(self.num_trained_qubits, name="loaded_qubit")
        self.__classical_register = qiskit.ClassicalRegister(self.num_cbits, name="classical_bit")
        self.__quantum_circuit = qiskit.QuantumCircuit(self.control_quantum_register,
                                                       self.trained_quantum_register,
                                                       self.loaded_quantum_register,
                                                       self.classical_register,
                                                       name=name)

    @property
    def modified_input_size(self) -> int:
        """Return the modified input size

        :return int: modified input size
        """
        return self.__modified_input_size

    @property
    def num_trained_qubits(self) -> int:
        """Return the number of trained qubits

        :return int: number of trained qubits
        """
        return self.__num_trained_qubits

    @property
    def num_cbits(self) -> int:
        """Return the number of classical bits

        :return int: number of classical bits
        """
        return self.__num_cbits

    @property
    def loss_history(self) -> List[float]:
        """Return the loss values

        :return List[float]: loss values
        """
        return self.__loss_history

    @property
    def epochs(self) -> int:
        """Return the number of epochs that have been done

        :return int: number of epochs
        """
        return self.__epochs

    @property
    def label(self) -> str:
        """Return the label

        :return str: label
        """
        return self.__label

    @property
    def control_quantum_register(self) -> qiskit.QuantumRegister:
        return self.__control_quantum_register

    @property
    def trained_quantum_register(self) -> qiskit.QuantumRegister:
        return self.__trained_quantum_register

    @property
    def loaded_quantum_register(self) -> qiskit.QuantumRegister:
        return self.__loaded_quantum_register

    @property
    def classical_register(self) -> qiskit.ClassicalRegister:
        return self.__classical_register

    @property
    def quantum_circuit(self) -> qiskit.QuantumCircuit:
        return self.__quantum_circuit

    @property
    def structure(self) -> str:
        """Return the structure of the quantum circiut.

        :return str: the structure of the quantum circuit
        """
        return self.__structure

    @property
    def thetas_list(self) -> List[float]:
        """Return the rotation angles.

        :return List[float]: the rotation angles
        """
        return self.__thetas_list

    @property
    def num_thetas(self) -> int:
        """Return the number of the rotation angles.

        :return int: the number of the rotation angles
        """
        return sum([len(np.array(thetas).reshape(-1)) for thetas in self.thetas_list])

    @property
    def best_loss(self) -> float:
        """Return the best loss value

        :return float: best loss value
        """
        return np.min(self.loss_history) if len(self.loss_history) != 0 else None

    @property
    def best_epochs(self) -> int:
        """Return the epoch with the best loss value

        :return int: best epoch
        """
        return np.argmin(self.loss_history) + 1 if len(self.loss_history) != 0 else None

    def build_quantum_circuit(self, structure: List[str], thetas_list: List[List[float]], is_in_train: bool = False) -> None:
        """Build a quantum circuit

        :param List[str] structure: string, which has only s, d and c, to decide the structure
        :param List[List[float]] thetas_list: initial rotation angles
        :param bool is_in_train: whether or not this function is called whilst learning, defaults to False
        :raises ValueError: if a given structure has other than s, d and c
        :raises ValueError: if the lengths of structure and this_list are not the same
        """
        # Check whether the given structure is valid or not
        num_s = structure.count('s')
        num_d = structure.count('d')
        num_c = structure.count('c')
        if num_s + num_d + num_c != len(structure):
            msg = f"A given structure must have only 's', 'd' and 'c'," \
                  f"but this one is '{structure}'."
            raise ValueError(msg)
        if len(structure) != len(thetas_list):
            msg = f"The length of a given structure must equal to" \
                  f"the length of a given thetas_list,\n" \
                  f"but len(structure) = {len(structure)} and" \
                  f"len(thetas_list) = {len(thetas_list)}."
            raise ValueError(msg)

        # Prepare the Hadamard gate in order to perform the SWAP test
        self.quantum_circuit.h(0)
        self.quantum_circuit.barrier()

        # Prepare quantum gates in order to generate the representative quantum state
        for letter, thetas in zip(structure, thetas_list):
            if letter == 's':
                self.__add_single_qubit_unitary_layer(thetas)
            elif letter == 'd':
                self.__add_dual_qubit_unitary_layer(thetas)
            elif letter == 'c':
                self.__add_controlled_qubit_unitary_layer(thetas)

            self.quantum_circuit.barrier()

        # Store the information into class variables
        self.__structure = structure
        self.__thetas_list = thetas_list

        # Prepare quantum gates in order to load data
        self.__add_load_structure()

        # Prepare the cswap gates, the Hadamard gate and the measurement in order to perform the SWAP test
        self.__add_cswap()
        self.quantum_circuit.h(0)
        self.quantum_circuit.measure(0, 0)

        if not is_in_train:
            print("Successfully built.")

    def __add_single_qubit_unitary_layer(self, thetas: List[float]) -> None:
        """Add the single qubit unitary layer into the quantum circuit.

        :param List[float] thetas: the rotation angles
        :raises ValueError: if the length of thetas is not the same as the number of qubits for the representative quantum state
        """
        # Check whether the given thetas is valid or not
        if self.num_trained_qubits != len(thetas):
            msg = f"The shape of given thetas must be ({self.num_trained_qubits}, 2)" \
                  f"but this is ({len(thetas)}, 2)."
            raise ValueError(msg)

        # Prepare ry and rz gates
        for qubit, theta in enumerate(thetas, 1):
            theta_y, theta_z = theta
            self.quantum_circuit.ry(qubit=qubit, theta=theta_y)
            self.quantum_circuit.rz(qubit=qubit, phi=theta_z)

    def __add_dual_qubit_unitary_layer(self, thetas: List[float]) -> None:
        """Add the dual qubit unitary layer into the quantum circuit.

        :param List[float] thetas: rotation angles
        :raises ValueError: if the length of thetas is not the same the combinations of the qubits for the representative quantum state
        """
        # Check whether the given thetas is valid or not
        num_combinations = np.arange(1, self.num_trained_qubits).sum()
        if num_combinations != len(thetas):
            msg = f"The shape of thetas must be ({num_combinations}, 2)" \
                  f"but this is ({len(thetas)}, 2)."
            raise ValueError(msg)

        # Prepare ryy and rzz gates
        for basis_qubit, theta in enumerate(thetas, 1):
            theta_y, theta_z = theta

            partner_qubits = np.arange(basis_qubit+1, self.num_trained_qubits+1)
            for partner_qubit in partner_qubits:
                self.quantum_circuit.ryy(theta=theta_y, qubit1=basis_qubit, qubit2=partner_qubit)
                self.quantum_circuit.rzz(theta=theta_z, qubit1=basis_qubit, qubit2=partner_qubit)

    def __add_controlled_qubit_unitary_layer(self, thetas: List[float]) -> None:
        """Add the controlled qubit unitary layer into the quantum circuit.

        :param List[float] thetas: rotation angles
        :raises ValueError: if the length of thetas is not the same the number of the qubits for the representative quantum state
        """
        # Check whether the given thetas is valid or not
        if self.num_trained_qubits != len(thetas):
            msg = f"The shape of thetas must be ({self.num_trained_qubits}, 2)" \
                  f"but this is ({len(thetas)}, 2)."
            raise ValueError(msg)

        # Prepare cry and crz gates
        for qubit, theta in enumerate(thetas, 1):
            theta_y, theta_z = theta
            self.quantum_circuit.cry(theta=theta_y, control_qubit=0, target_qubit=qubit)
            self.quantum_circuit.crz(theta=theta_z, control_qubit=0, target_qubit=qubit)

    def __add_load_structure(self) -> None:
        """Add gates for loading data into the quantum cirucit.
        """
        thetas = np.array([0, 0] * self.num_trained_qubits).reshape(-1, 2)

        # Prepare ry and rz gates
        for qubit, theta in enumerate(thetas, 1):
            qubit += self.num_trained_qubits
            theta_y, theta_z = theta
            self.quantum_circuit.ry(qubit=qubit, theta=theta_y)
            self.quantum_circuit.rz(qubit=qubit, phi=theta_z)

    def __add_cswap(self) -> None:
        """Add the controlled swap into the quantum circuit.
        """
        # Add the cswap gate (Fredkin gate) into the quantum circuit
        for trained_qubit in range(1, self.num_trained_qubits+1):
            loaded_qubit = trained_qubit + self.num_trained_qubits
            self.quantum_circuit.fredkin(0, trained_qubit, loaded_qubit)

    def run(self, backend: str, shots: int, on_ibmq: bool) -> float:
        """Run the quantum circuit and obtain the quantum state fidelity.

        :param str backend: backend
        :param int shots: number of executions
        :param bool on_ibmq: whether or not ibmq is used
        :raises ValueError: if "0" is not observed
        :return float fidelity_like: fidelity-like value
        """
        # Execute the quantum circuit
        if on_ibmq:
            job = qiskit.execute(qiskit.transpile(self.quantum_circuit, backend), backend=backend, shots=shots)
        else:
            simulator = qiskit.Aer.get_backend(backend)
            job = qiskit.execute(self.quantum_circuit, simulator, shots=shots)

        # Get the result
        result = job.result()
        counts = result.get_counts(self.quantum_circuit)

        try:
            num_of_zeros = counts["0"]
        except KeyError:
            # if there is no observatino '0' in the result
            msg = "There is no observation '0' in the result of this execution."
            raise ValueError(msg)

        fidelity = self.get_quantum_state_fidelity(num_of_zeros=num_of_zeros, shots=shots)

        return fidelity

    def get_fidelity_like_value(self, num_of_zeros: int, shots: int) -> float:
        """Get the fidelity like value.

        Note that, it is not the exact quantum state fidelity |<x|y>| but (1 + |<x|y>|^2)/2.
        This output is for comparing two numbers by their size and (1 + k^2)/2 < (1 + l^2)/2 holds for any k < l.
        Therefore it is not a big problem if either the fidelity-like value or the exact quantum state fidelity is used.

        :param int num_of_zero: the number of zeros
        :param int shots: the number of executions
        :return float: the fidelity-like value
        """
        return num_of_zeros / shots

    def get_quantum_state_fidelity(self, num_of_zeros: int, shots: int) -> float:
        """Get the quantum state fidelity.

        :param int num_of_zeros: the number of zeros
        :param int shots: the number of executions
        :raises ValueError: if the fidelity value is negative
        :return float: the quantum state fidelity
        """
        fidelity_like = self.get_fidelity_like_value(num_of_zeros=num_of_zeros, shots=shots)  # (1 + |<x|y>|^2)/2
        squared_fidelity = fidelity_like * 2 - 1  # |<x|y>|^2
        fidelity = np.sqrt(squared_fidelity)  # |<x|y>|

        if fidelity < 0:
            msg = f"The quantum state fidelity must be non-negative," \
                  f"but this is {fidelity}."
            raise ValueError(msg)

        return fidelity

    def draw(self) -> Optional[object]:
        """Visualise the quantum circuit.

        This function returns figure if matplotlib is available otherwise prints the quantum circuit to the standard output.

        :return Optional[object]: matplotlib.figure.Figure if matplotlib is available
        """
        try:
            return self.quantum_circuit.draw("mpl")
        except:
            print(self.quantum_circuit.draw())

    def train(self, data: List[List[float]], label: str, epochs: int, learning_rate: float, backend: str, shots: int,
              should_normalise: bool, should_save_each_epoch: bool, on_ibmq: bool) -> None:
        """Train the quantum circuit.

        :param List[List[float]] data: training data
        :param str label: training label
        :param int epochs: number of epochs
        :param float learning_rate: learning rate
        :param str backend: backend
        :param int shots: number of executions
        :param bool should_normalise: whether or not normalise each data
        :param bool should_save_each_epoch: whether or not print the information of the quantum curcuit per one epoch
        :param bool on_ibmq: whether or not ibmq is used
        """
        # Prepare the data
        prepared_data = self.normalise_data(data) if should_normalise else data.copy()

        # Store the given label into a class variable
        self.__label = label

        # Print basic information
        print(f"label {self.label}: Start training.")

        # Train
        for epoch in range(1, epochs+1):
            print(f"epoch {epoch}: ", end="")

            total_loss_over_epochs = 0
            for vector in tqdm(prepared_data):

                total_loss = 0
                for first_theta_index, thetas in enumerate(self.thetas_list):  # for each layer
                    for second_theta_index, theta_yz in enumerate(thetas):  # for each quantum gate for the representative quantum state
                        for third_theta_index in range(len(theta_yz)):  # for each rotation angle
                            # Calculate the quantum fidelity-like value of the foward state
                            forward_thetas_list = copy.deepcopy(self.thetas_list)
                            forward_thetas_list[first_theta_index][second_theta_index][third_theta_index] += np.pi / (2 * np.sqrt(epoch))
                            forward_fidelity = self.__run_with_building_another_circuit(thetas_list=forward_thetas_list, data=vector,
                                                                                        backend=backend, shots=shots, on_ibmq=on_ibmq)

                            # Calculate the quantum fidelity-like value of the backward state
                            backward_thetas_list = copy.deepcopy(self.thetas_list)
                            backward_thetas_list[first_theta_index][second_theta_index][third_theta_index] -= np.pi / (2 * np.sqrt(epoch))
                            backward_fidelity = self.__run_with_building_another_circuit(thetas_list=backward_thetas_list, data=vector,
                                                                                         backend=backend, shots=shots, on_ibmq=on_ibmq)

                            # Calculate the loss value
                            self.__load_into_qubits(vector)
                            loss = self.run(backend=backend, shots=shots, on_ibmq=on_ibmq)
                            total_loss += loss

                            # Update the parameter
                            update_term = -0.5 * (np.log(forward_fidelity) - np.log(backward_fidelity))
                            self.thetas_list[first_theta_index][second_theta_index][third_theta_index] -= learning_rate * update_term

                            # Reconstruce the quantum circuit with updated parameters
                            # This step is not needed if the loss value is not needed
                            self.quantum_circuit.data = []
                            self.build_quantum_circuit(self.structure, self.thetas_list, is_in_train=True)

                total_loss_over_epochs += total_loss / self.num_thetas

            # Update learning information in class variables
            total_loss_over_epochs = len(data) - total_loss_over_epochs
            self.__loss_history.append(total_loss_over_epochs)
            self.__epochs += 1
            if len(self.loss_history) == 1 or total_loss_over_epochs < self.best_loss:
                print(f"\tloss = {total_loss_over_epochs} <- the best loss ever")
            else:
                print(f"\tloss = {total_loss_over_epochs}")

            if should_save_each_epoch:
                self.save_parameters_as_json(f"latest_{label}.json")

        print(f"The best loss is {self.best_loss} on {self.best_epochs} epochs.")

    def __run_with_building_another_circuit(self, thetas_list, data, backend, shots, on_ibmq) -> float:
        another_quclassi = QuClassiCircuit(self.modified_input_size)
        another_quclassi.build_quantum_circuit(self.structure, thetas_list, is_in_train=True)                        
        another_quclassi.__load_into_qubits(data)
        fidelity = another_quclassi.run(backend=backend, shots=shots, on_ibmq=on_ibmq)

        return fidelity

    def __load_into_qubits(self, data: List[float]) -> None:
        """Load classical data on the qubits.

        :param List[float] data: classical data
        :raises ValueError: if the dimension of the given classical data is not the same as the input size of the quantum circuit
        """
        # Convert the given classical data into rotation angles
        thetas = 2 * np.arcsin(np.sqrt(data))

        # Let the length of the given thetas be an even number
        if len(thetas) % 2 != 0:
            thetas = np.append(thetas, 0)

        # Check whether the given thetas is valid or not
        if len(thetas) != self.modified_input_size:
            msg = f"The length of the given data and self.modified_input_size must be same\n" \
                  f"but the length is {len(thetas)} and self.modified_input_size is {self.modified_input_size}"
            raise ValueError(msg)

        theta_count = 0
        for gate_information in self.quantum_circuit.data:
            # Set rotation angles on the quantum gate that satisfies the following conditions.
            #    - The quantum gate is operated to loading qubits.
            #    - The quantum gate is ry or rz.
            if (gate_information[1][0].register.name == "loaded_qubit") and (gate_information[0].name in ["ry", "rz"]):
                gate_information[0]._params = [thetas[theta_count]]
                theta_count += 1

    def normalise_data(self, data: List[List[float]]) -> np.array:
        """Normalise data.

        :param List[List[float]] data: classical data
        :return numpy.array data_normalised: normalised classical data
        """
        data_normalised = []
        for d in data:
            data_normalised.append(np.array(d) / np.linalg.norm(d))

        data_normalised = np.array(data_normalised)
        return data_normalised

    def save_parameters_as_json(self, output_path: str) -> None:
        """Save the quantum circuit information in json.

        :param str output_path: output path
        :raises ValueError: if the extension of the given output_path is not .json
        """
        if ".json" != Path(output_path).suffix:
            msg = f"The suffix of output_path must be .json" \
                  f"but this output_path is {output_path}."
            raise ValueError(msg)

        model_information = dict()

        model_information["modified_input_size"] = self.modified_input_size
        model_information["structure"] = self.structure

        thetas_list = []
        for ts in self.thetas_list:
            thetas = []
            for theta_yz in ts:
                thetas.append(list(theta_yz))
            thetas_list.append(thetas)
        model_information["thetas_list"] = thetas_list

        model_information["label"] = self.label
        model_information["epochs"] = self.epochs
        model_information["loss_history"] = self.loss_history
        model_information["best_loss"] = self.best_loss

        # Output the json file
        with open(output_path, 'w', encoding="utf-8") as output:
            json.dump(model_information, output, indent=4)

    @classmethod
    def load_parameters_from_json(cls, model_path: str) -> object:
        """Load the quantum circuit by the given json

        :param str model_path: model path
        :return QuClassiCircuit loaded_quclassi_circuit: Loaded QuClassiCircuit object
        :raises ValueError: if the extension of the model_path is not .json
        """
        if ".json" != Path(model_path).suffix:
            msg = f"The suffix of model_path must be .json" \
                  f"but this model_path is {model_path}."
            raise ValueError(msg)

        with open(model_path) as model_file:
            model_information = json.load(model_file)

        loaded_quclassi_circuit = cls(model_information["modified_input_size"])
        loaded_quclassi_circuit.build_quantum_circuit(model_information["structure"], model_information["thetas_list"])
        loaded_quclassi_circuit.label = model_information["label"]
        loaded_quclassi_circuit.epochs = model_information["epochs"]
        loaded_quclassi_circuit.loss_history = model_information["loss_history"]
        loaded_quclassi_circuit.best_loss = model_information["best_loss"]

        return loaded_quclassi_circuit

    def calculate_likelihood(self, data: Union[List[List[float]], List[float]], backend: str, shots: int, should_normalise: bool, on_ibmq: bool) -> List[float]:
        """Calculate likeliohoods

        :param Union[List[List[float]], List[float]] data: classical data
        :param str backend: backend
        :param int shots: number of executions
        :param bool should_normalise: whether or not normalise each data
        :param bool on_ibmq: whether or not ibmq is used
        :return List[float] likelihoods: likelihoods
        """
        data_np = np.array(data)
        if should_normalise:
            prepared_data = self.normalise_data(data) if len(data_np.shape) != 1 else self.normalise_data([data])
        else:
            prepared_data = data if len(data_np.shape) != 1 else [data]

        likelihoods = []
        for vector in prepared_data:
            self.__load_into_qubits(vector)
            fidelity_like = self.run(backend=backend, shots=shots, on_ibmq=on_ibmq)

            likelihoods.append(fidelity_like)

        return likelihoods
