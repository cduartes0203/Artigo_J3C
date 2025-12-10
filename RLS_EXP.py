import numpy as np

class RLS_ExponentialRegressor:
    def __init__(self, lambda_=0.99, delta=1000):
        """
        Regressão exponencial recursiva:
        y = beta_0 * exp(beta_1 * x)

        Após log-transformação: ln(y) = ln(beta_0) + beta_1 * x
        """
        self.theta = np.zeros((2, 1))  # [theta_0 = ln(beta_0), theta_1 = beta_1]
        self.P = delta * np.eye(2)
        self.lambda_ = lambda_

    def _phi(self, x):
        return np.array([[1, x]]).T

    def update(self, x, y):
        if y <= 0:
            raise ValueError("y deve ser positivo para aplicar log em regressão exponencial.")
        phi_x = self._phi(x)
        y_trans = np.log(y)
        K = self.P @ phi_x / (self.lambda_ + phi_x.T @ self.P @ phi_x)
        error = y_trans - phi_x.T @ self.theta
        self.theta += K * error
        self.P = (self.P - K @ phi_x.T @ self.P) / self.lambda_

    def predict(self, x):
        phi_x = self._phi(x)
        ln_y = float(phi_x.T @ self.theta)
        #print('ln_y:',ln_y)
        return np.exp(ln_y)

    def get_params(self):
        """Retorna beta_0 e beta_1"""
        theta_0, theta_1 = self.theta.flatten()
        beta_0 = np.exp(theta_0)
        beta_1 = theta_1
        return beta_0, beta_1