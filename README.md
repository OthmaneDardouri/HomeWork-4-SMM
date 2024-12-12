# MLE vs MAP: A Comparative Study

This project implements and compares **Maximum Likelihood Estimation (MLE)** and **Maximum a Posteriori (MAP)** solutions for a polynomial regression problem. It provides tools for experimentation with different model complexities, regularization parameters, and optimization algorithms.

---

## **Problem Definition**
We aim to solve the regression problem:
\[
y_i = \theta_1 + \theta_2 x_i + \cdots + \theta_K (x_i)^{K-1} + e_i, \quad \text{where } e_i \sim \mathcal{N}(0, \sigma^2)
\]

Given:
- \( \mathbf{X} \): Input data points.
- \( \mathbf{Y} \): Observed outputs with noise.
- \( \theta_{true} \): Ground-truth parameters of degree \( K-1 \).

The objectives:
1. Compute \( \theta_{MLE} \) and \( \theta_{MAP} \).
2. Compare their performance across different scenarios.

---

## **Features**
- Generate synthetic datasets with:
  - Polynomial data and Gaussian noise.
  - Customizable \( K \), \( N \), and \( \sigma^2 \).
- Solve for \( \theta_{MLE} \) using:
  - **Gradient Descent (GD)**.
  - **Stochastic Gradient Descent (SGD)**.
  - **Normal Equations** (closed-form solution).
- Solve for \( \theta_{MAP} \) with a regularization parameter \( \lambda \).
- Evaluate model performance:
  - Train and test error.
  - Relative error with respect to \( \theta_{true} \).
- Visualize results:
  - Fitted regression curves.
  - Error vs. model complexity plots.
  - Impact of regularization and dataset size.

---

## **Requirements**
- Python 3.8+
- Libraries:
  - `numpy`
  - `matplotlib`

Install the dependencies:
```bash
pip install numpy matplotlib
