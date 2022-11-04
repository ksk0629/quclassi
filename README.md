# Quclassi
This repository is forked from [arkmohba/ARK_study_Quclassi](https://github.com/arkmohba/ARK_study_Quclassi).

## Easy-peasy introduction
QuClassi is a novel hybrid quantum-classic deep neural network architecture introduced in [[QuClassi: a hybrid deep neural network architecture based on quantum state fidelity](https://arxiv.org/abs/2103.11307)]. This can be used for binary and multi-class classification.

QuClassi has $n$ quantum circuits for $n$-classes. One quantum circuit has three types of quantum registers: the first quantum register for encoding data, the second one for expressing a representation state and the third for obtaining quantum state fidelity. The first and second parts have the same numbers of qubits and the third part has only one qubit.

Two scalars are encoded into one qubit by the original encode method. Namely, an $x$-dimensional classical data is encoded $x/2$-qubits. You need $(x+1)$-qubits to construct one quantum circuit since the first and second quantum registers need $x/2$ qubits and the third one needs one qubit: $x/2 \times 2 + 1 = x + 1$.

Note that, it is not necessary to run all quantum circuits at the same time in order to classify an input data. It means that you do not need to prepare $(x+1) * n$ qubits.

## Environment
This scripts are only tested by the following environment.

- Python 3.10

## Training
Run `python ./src/[objective_script]`, whose `[objective_script]` is one of the following one.

| script name | #classes | dimension |
| --- | --- | --- |
| `breast_cancer_model.py` | 2 | 30 |
| `iris_model.py` | 3 | 4 |
| `wine_model.py` | 3 | 13 |
