# Maximum Likelihood Estimation (MLE) and Maximum a Posteriori (MAP) Comparison

This repository implements and compares **Maximum Likelihood Estimation (MLE)** and **Maximum a Posteriori (MAP)** for a polynomial regression problem. The task is to estimate parameters of a polynomial model using the MLE and MAP approaches, with Gaussian noise added to the observations.

## Problem Definition

Given a dataset \( D = \{ X, Y \} \), where

\[
X = \begin{bmatrix} x_1 & x_2 & \dots & x_N \end{bmatrix} \in \mathbb{R}^N
\]
\[
Y = \begin{bmatrix} y_1 & y_2 & \dots & y_N \end{bmatrix} \in \mathbb{R}^N
\]

and for each \( i = 1, \dots, N \), the following model is assumed:

\[
y_i = \theta_1 + \theta_2 x_i + \dots + \theta_K (x_i)^{K-1} + e_i \quad \forall i = 1, \dots, N
\]

where \( e_i \sim \mathcal{N}(0, \sigma^2) \) is random Gaussian noise, and \( \theta = (\theta_1, \dots, \theta_K)^T \) are the parameters of the polynomial model.

### Maximum Likelihood Estimation (MLE)

The MLE approach works by defining the conditional probability of \( y \) given \( x \), \( p_\theta(y|x) \), and optimizing the parameters \( \theta \) to maximize this probability distribution over the dataset \( D \). Assuming Gaussian noise \( e_i \), we have:

\[
p_\theta(y|x) = \mathcal{N}(f_\theta(x_i), \sigma^2 I) = \frac{1}{\sqrt{2\pi \sigma^2}} e^{-\frac{1}{2\sigma^2}(f_\theta(x) - y)^2}
\]

Thus, the log-likelihood function is:

\[
\log p_\theta(Y|X) = \sum_{i=1}^{N} \log p_\theta(y_i|x_i)
\]

To maximize the log-likelihood, we minimize the negative log-likelihood:

\[
\theta_{MLE} = \arg \min_\theta \sum_{i=1}^{N} \frac{1}{2\sigma^2} (f_\theta(x_i) - y_i)^2
\]

where \( f_\theta(x) = \sum_{j=1}^{K} \varphi_j(x_i) \theta_j \) with \( \varphi_j(x) = x^{j-1} \).

This is equivalent to the function minimized in the Exercise 3 of Lab 3 using Gradient Descent (GD) or Normal Equations.

### Maximum a Posteriori (MAP)

When the correct degree \( K \) is unclear, the MAP approach is used. Suppose the parameters \( \theta \) are normally distributed:

\[
\theta \sim \mathcal{N}(0, \sigma_\theta^2 I)
\]

Using Bayes' Theorem, we can express the posterior probability:

\[
p(\theta|X, Y) = \frac{p(Y|X, \theta) p(\theta)}{p(Y|X)}
\]

The MAP solution maximizes the posterior probability:

\[
\theta_{MAP} = \arg \min_\theta \sum_{i=1}^{N} -\log p(\theta|x_i, y_i) = \arg \min_\theta \sum_{i=1}^{N} -\log p(y_i|x_i, \theta) - \log p(\theta)
\]

### Comparison of MLE and MAP

1. **Define the test problem**:
   - Let the user define a positive integer \( K \) and \( \theta_{\text{true}} = (1, 1, \dots, 1)^T \).
   - Define \( X = [x_1, x_2, \dots, x_N] \in \mathbb{R}^N \), where \( x_i \) are uniformly distributed in the interval \([a, b]\).
   - Define the Generalized Vandermonde matrix \( \Phi(X) \in \mathbb{R}^{N \times K} \), where the element \( \varphi_j(x) = x^{j-1} \).

2. **Dataset Generation**:
   - Compute \( Y = \Phi(X) \theta_{\text{true}} + e \), where \( e \sim \mathcal{N}(0, \sigma^2 I) \) is Gaussian noise.

3. **MLE and MAP Implementation**:
   - Implement functions to compute the **MLE** and **MAP** solutions:
     - For MLE, use Gradient Descent (GD), Stochastic Gradient Descent (SGD), or Normal Equations.
     - For MAP, incorporate a regularization term based on a Gaussian prior for the parameters.

4. **Performance Evaluation**:
   - Evaluate the training and test errors for different values of \( K \) and \( \lambda \).
   - Compare MLE and MAP for increasing values of \( K \) and analyze the errors for different values of \( N \) and \( \lambda \).

## Key Equations

- **MLE Loss Function**:

\[
\theta_{MLE} = \arg \min_\theta \frac{1}{2\sigma^2} \sum_{i=1}^{N} \left( f_\theta(x_i) - y_i \right)^2
\]

- **MAP Loss Function**:

\[
\theta_{MAP} = \arg \min_\theta \sum_{i=1}^{N} \left( -\log p(y_i|x_i, \theta) - \log p(\theta) \right)
\]

## Installation

Clone the repository and install the dependencies:

```bash
git clone https://github.com/your-repository.git
cd your-repository
pip install -r requirements.txt
