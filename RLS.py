import numpy as np

class RLS:
    def __init__(self,n, lambda_=0.9, delta=10000):
        """
        Regressão logarítmica recursiva via RLS.
        y = beta_0 + beta_1 * ln(x)

        :param lambda_: Fator de esquecimento
        :param delta: Valor inicial para P (confiança inicial baixa)
        """
        self.n = n+1
        self.theta = np.zeros((self.n, 1))              # [beta_0, beta_1]
        self.P = delta * np.eye(self.n)
        self.lambda_ = lambda_

    def _phi(self, x):
        return np.array([[x**i for i in range(self.n)]]).T

    def adapt(self, x, y):
        phi_x = self._phi(x)
        K = self.P @ phi_x / (self.lambda_ + phi_x.T @ self.P @ phi_x)
        error = y - phi_x.T @ self.theta
        self.theta += K * error
        self.P = (self.P - K @ phi_x.T @ self.P) / self.lambda_

    def predict(self, x):
        phi_x = self._phi(x)
        return float(phi_x.T @ self.theta)

    def get_params(self):
        return self.theta.flatten()