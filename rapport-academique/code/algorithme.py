import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple


class AdaptiveOptimizer:
    """
    Implementation of an adaptive optimization algorithm.
    """

    def __init__(self, learning_rate: float = 0.01, max_iter: int = 1000,
                 tolerance: float = 1e-6, reduction_factor: float = 0.5):
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.tolerance = tolerance
        self.reduction_factor = reduction_factor
        self.loss_history = []

    def objective_function(self, theta: np.ndarray, X: np.ndarray, y: np.ndarray) -> float:
        """
        Objective function to minimize.
        """
        predictions = X @ theta
        error = predictions - y
        loss = np.mean(error ** 2)
        return loss

    def gradient(self, theta: np.ndarray, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Compute the gradient of the objective function.
        """
        predictions = X @ theta
        error = predictions - y
        grad = (2 / len(y)) * X.T @ error
        return grad

    def optimize(self, X: np.ndarray, y: np.ndarray, theta_init: np.ndarray = None) -> Tuple[np.ndarray, List[float]]:
        """
        Main optimization algorithm.
        """
        if theta_init is None:
            theta = np.random.randn(X.shape[1])
        else:
            theta = theta_init.copy()

        current_lr = self.learning_rate

        for iteration in range(self.max_iter):
            grad = self.gradient(theta, X, y)
            theta = theta - current_lr * grad
            current_loss = self.objective_function(theta, X, y)
            self.loss_history.append(current_loss)

            # Adapt learning rate
            if iteration > 10 and self.loss_history[iteration] > self.loss_history[iteration - 5]:
                current_lr *= self.reduction_factor
                print(f"Iteration {iteration}: learning rate reduced to {current_lr}")

            # Stop if gradient is small
            if np.linalg.norm(grad) < self.tolerance:
                print(f"Convergence reached at iteration {iteration}")
                break

        return theta, self.loss_history


if __name__ == "__main__":
    # Generate synthetic data
    np.random.seed(42)
    n_samples, n_features = 1000, 20
    X = np.random.randn(n_samples, n_features)
    true_theta = np.random.randn(n_features)
    y = X @ true_theta + np.random.normal(0, 0.1, n_samples)

    # Run optimizer
    optimizer = AdaptiveOptimizer(learning_rate=0.01, max_iter=1000)
    theta_opt, losses = optimizer.optimize(X, y)

    print(f"Optimal parameters: {theta_opt}")
    print(f"Final loss: {losses[-1]:.6f}")

