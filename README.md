# MLE vs MAP: A Comparative Study

This project implements and compares **Maximum Likelihood Estimation (MLE)** and **Maximum a Posteriori (MAP)** solutions for a polynomial regression problem. It provides tools for experimentation with different model complexities, regularization parameters, and optimization algorithms.

## Problem Definition
We aim to solve the regression problem:

$$
y_i = \theta_1 + \theta_2 x_i + \cdots + \theta_K (x_i)^{K-1} + e_i, \quad \text{where } e_i \sim \mathcal{N}(0, \sigma^2)
$$

Given:
- \( \mathbf{X} \): Input data points.
- \( \mathbf{Y} \): Observed outputs with noise.
- \( \theta_{true} \): Ground-truth parameters of degree \( K-1 \).

The objectives:
1. Compute \( \theta_{MLE} \) and \( \theta_{MAP} \).
2. Compare their performance across different scenarios.

## Usage
### Compute MLE
Use different methods to compute \( \theta_{MLE} \):
```python
theta_mle = compute_mle(X, Y, K=4, method="normal_equations")
