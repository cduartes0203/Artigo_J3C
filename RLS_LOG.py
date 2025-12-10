import numpy as np

class RLS_LogarithmicRegressor:
    def __init__(self, lambda_=0.9, delta=10000):
        """
        Regressão logarítmica recursiva via RLS.
        y = beta_0 + beta_1 * ln(x)

        :param lambda_: Fator de esquecimento
        :param delta: Valor inicial para P (confiança inicial baixa)
        """
        self.theta = np.zeros((3, 1))              # [beta_0, beta_1]
        self.P = delta * np.eye(3)
        self.lambda_ = lambda_

    def _phi(self, x):
        """Transforma x em vetor de regressão [1, ln(x)]"""
        if x <= 0:
            raise ValueError("x deve ser positivo para regressão logarítmica.")
        return np.array([[1, np.log(x), np.log(x)**2]]).T

    def update(self, x, y):
        phi_x = self._phi(x)
        #print(phi_x)
        K = self.P @ phi_x / (self.lambda_ + phi_x.T @ self.P @ phi_x)
        error = y - phi_x.T @ self.theta
        self.theta += K * error
        self.P = (self.P - K @ phi_x.T @ self.P) / self.lambda_

    def predict(self, x):
        phi_x = self._phi(x)
        return float(phi_x.T @ self.theta)

    def get_params(self):
        return self.theta.flatten()