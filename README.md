# KAN vs MLP Comparison on MNIST Dataset

-----
## Introduction
This repository contains a comparative study between Kernel Adaptive Networks (KAN) and traditional Multi-Layer Perceptrons (MLP) using the MNIST dataset. The goal is to assess how KAN performs relative to a standard MLP in terms of learning capabilities and performance metrics such as training convergence, model parameters, and test set accuracy.

KAN is known for its adaptability through the use of B-splines for nonlinear transformation of input features, offering a potential advantage over the simpler linear transformations used in MLPs.

## Repository Structure

- KAN.py: Implementation of the KAN.
- train.py: Training code for KAN and MLP using MNIST.

## Installation
To set up the environment to run the code, follow these steps:

```
# Clone the repository
git clone git@github.com:MaybeRichard/KAN-Layer.git

cd KAN-Layer
```

## Usage
To run the comparison between KAN and MLP models, execute the training code provided. This script will train both models on the MNIST dataset and then compare their performance.

## Run the comparison script
`python train.py` This command will train both the KAN and MLP models and output the training losses and test accuracies. Plots for these metrics will also be generated and displayed, providing a visual comparison of the performance of both models.

## Code Overview
KAN Implementation
The KAN model is implemented in KAN.py, which defines a custom PyTorch module KANLinear. This module uses B-splines for transforming input features, enabling more flexible and adaptive feature representation compared to traditional linear transformations.

## MLP Implementation
For comparison, a simple MLP model is defined in the main script train.py. This model consists of two linear layers with ReLU activation, providing a baseline for evaluating the KAN's performance.

## Training and Evaluation
Both models are trained on the MNIST dataset, using a modified ResNet as a feature extractor to adapt to the input size of MNIST images. The training process records metrics such as loss and accuracy, which are used to compare the models' performances.

## Contributing
Contributions to improve the code or extend the functionality are welcome. Please feel free to fork the repository and submit a pull request.

