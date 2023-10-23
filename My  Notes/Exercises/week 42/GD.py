import jax
import jax.numpy as jnp
import numpy as np

from sklearn.preprocessing import PolynomialFeatures

class GradientDescent:


    def __init__(self, x, y, poly_degree):

        self.poly_degree = poly_degree
        self.x = x
        self.X = PolynomialFeatures(degree=self.poly_degree).fit_transform(x.reshape(-1, 1))
        self.y = y

        self.analytical_gradient = analytical_gradient
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.tol = tol


    def gradient_descent(cost_function, cost_gradient,
                         method_learning_schedule, learning_rate,
                         max_iter=1000, tol=10e-4,
                         delta_momentum=None):

        # Initialize the first step
        beta = np.random.randn(2, 1)  # random starting point

        # Initialize lists to store values
        beta_values_GD = []
        MSE_scores = []

        num_iter = 0
        for _ in range(max_iter):
            # Calculate gradient
            gradient = cost_gradient(X, y, beta)

            # Update learning rate
            learning_rate = learning_schedule(method_learning_schedule, learning_rate)

            # Calculate new x and y  values and append to storage
            if delta_momentum is not None:
                change = learning_rate * f_grad_k + delta_momentum * change
                beta = beta + change
            else:
                beta = beta + learning_rate * f_grad_k

            beta_values_GD.append(beta)
            MSE_score = MSE(y, X @ beta)
            MSE_scores.append(MSE_score)

            # Check if converged by checking gradient
            if np.linalg.norm(f_grad_k) <= tol:
                print(f'Converged after {num_iter} iterations.')
                break
            num_iter += 1

        # print results
        if num_iter == max_iter:
            print(f'Gradient descent does not converge when max_iter={max_iter} and tol={tol}.')
        else:
            print('Solution found:\n  MSE = {:.4f}\n  beta = {:.4f}'.format(MSE_score, beta))

        return beta_values_GD, MSE_scores