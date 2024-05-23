# Implementation for the paper "On the Minimal Degree Bias in OOD Generalization for non-Boolean Functions" (ICML 2024)

This is uses the PyTorch framework.

## Content
- <mark>datasets.py</mark>: contains the code to generate data. Supports Gaussian distribution, and uniform distribution (on the Boolean domain or the set of bounded integers);
- helpers.py: all heplers functions, including calculating the (Fourier / Hermite / simple polynomials) basis coefficient of the model;
- main.py: main function to run the experiments;
- models.py: implementation of Vision Transformer, Random Features and MLP models;
- tasks_icml2024.py: includes the definition of the tasks to train the models on;
- training.py; implementation of the training procedure. Includes standard training with stochastic optimizer and Gradient Descent with line search method. 
